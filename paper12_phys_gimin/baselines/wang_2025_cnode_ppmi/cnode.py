"""Clean-room Conditional Neural ODE for PD progression forecasting.

Independent re-implementation from:
    Wang et al. 2025. "Conditional Neural ODE for Longitudinal Parkinson's Disease
    Progression Forecasting." arXiv:2511.04789.

The model predicts patient-specific morphometry trajectories from a baseline
observation + a conditioning covariate vector. A single scalar time is integrated
through a learned vector field to produce the trajectory. See CLEAN_ROOM_NOTES.md
for the list of assumptions made where the paper under-specifies architectural
or training details.

Core equations (clean-room interpretation)
------------------------------------------

Given:
    x_0 ∈ R^D                baseline morphometry (FreeSurfer subcortical volumes
                             + vertex-wise cortical thickness), D features
    c   ∈ R^C                conditioning covariates (age, sex, MDS-UPDRS,
                             disease duration at baseline, etc.)
    {t_k}_{k=1..K}           observed follow-up times (years from baseline)

Encode to a latent disease-state trajectory:

    h(0) = ENC_θ([x_0 ; c]) ∈ R^H           (encoder MLP)
    dh/dt = f_θ(h(t), c, t) ∈ R^H            (ODE vector field, NN)
    h(t_k) = h(0) + ∫_0^{t_k} f_θ(h, c, τ) dτ   (via torchdiffeq.odeint)
    x̂(t_k) = DEC_θ(h(t_k)) ∈ R^D             (decoder MLP)

Training loss (MSE over observed timepoints):

    L = (1 / Σ_i K_i) · Σ_i Σ_k ‖x̂_i(t_{i,k}) - x_{i,k}‖²_2

Clean-room-room simplifications (documented in CLEAN_ROOM_NOTES.md):

  1. The paper's "learnable patient-specific initial time and progression speed"
     is implemented as conditioning on c (a simple, well-specified alternative
     to a second optimization loop over per-patient scalars).
  2. `dopri5` with default rtol/atol (torchdiffeq default); adjoint NOT used —
     the integration horizon is short (<=10 years), batches fit in memory.
  3. Hidden dim H=64, encoder/decoder 2-layer MLPs with GELU, vector field 2-layer
     MLP with GELU — typical values when the paper does not report them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torchdiffeq import odeint


@dataclass(frozen=True)
class CNODEConfig:
    """Hyperparameter container for CNODE.

    Defaults reflect clean-room assumptions documented in CLEAN_ROOM_NOTES.md.
    """

    feature_dim: int = 216            # 68 subcortical vols + 148 vertex-thickness
    covariate_dim: int = 8            # age, sex, MDS-UPDRS-I/II/III, MoCA, dur, MDS-total
    hidden_dim: int = 64              # latent disease-state dimensionality (H)
    encoder_hidden: int = 128         # encoder MLP width
    decoder_hidden: int = 128         # decoder MLP width
    vf_hidden: int = 128              # vector field MLP width
    n_vf_layers: int = 2              # vector field depth (clean-room default)
    activation: str = "gelu"          # {"gelu", "relu", "tanh"}
    ode_solver: str = "dopri5"        # torchdiffeq solver
    ode_rtol: float = 1e-5
    ode_atol: float = 1e-7
    dropout: float = 0.0              # no dropout in the ODE field (standard NODE)


def _build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation '{name}'")


def _mlp(in_dim: int, hidden: int, out_dim: int, act: str, dropout: float = 0.0) -> nn.Sequential:
    """2-hidden-layer MLP: Linear -> Act -> (Dropout) -> Linear -> Act -> Linear."""
    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden),
        _build_activation(act),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers += [
        nn.Linear(hidden, hidden),
        _build_activation(act),
        nn.Linear(hidden, out_dim),
    ]
    return nn.Sequential(*layers)


class _ConditionalVectorField(nn.Module):
    """Neural ODE vector field f_θ(h, c, t): R^H x R^C x R -> R^H.

    The conditioning vector c is concatenated with h at every ODE step. Time t is
    encoded as a scalar feature. The paper describes a continuous-time process
    so we use a continuous time embedding (raw t, bounded).
    """

    def __init__(self, config: CNODEConfig) -> None:
        super().__init__()
        self.config = config
        in_dim = config.hidden_dim + config.covariate_dim + 1  # +1 for time
        self.net = _mlp(
            in_dim=in_dim,
            hidden=config.vf_hidden,
            out_dim=config.hidden_dim,
            act=config.activation,
            dropout=config.dropout,
        )
        self._c: Optional[torch.Tensor] = None  # cached conditioning (batch, C)

    def set_condition(self, c: torch.Tensor) -> None:
        """Cache the per-batch conditioning vector for the upcoming ODE solve."""
        if c.dim() != 2:
            raise ValueError(f"Expected 2-D conditioning tensor, got shape {tuple(c.shape)}")
        self._c = c

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:  # torchdiffeq API
        if self._c is None:
            raise RuntimeError("set_condition(c) must be called before odeint.")
        # t is a 0-d scalar tensor in torchdiffeq; broadcast to (batch, 1)
        batch = h.shape[0]
        t_col = t.reshape(1, 1).expand(batch, 1).to(h.dtype)
        inp = torch.cat([h, self._c, t_col], dim=-1)
        return self.net(inp)


class CNODE(nn.Module):
    """Conditional Neural ODE for longitudinal morphometry forecasting.

    Parameters
    ----------
    config : CNODEConfig
        Dimensions + solver parameters. See class docstring for the exact math.

    Forward interface
    -----------------
    >>> model = CNODE(CNODEConfig())
    >>> x_hat = model(x_0, c, t_eval)   # x_hat : (batch, T, feature_dim)

    where x_0 is (batch, feature_dim), c is (batch, covariate_dim), and
    t_eval is (T,) monotonically increasing with t_eval[0] == 0.
    """

    def __init__(self, config: CNODEConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder: [x_0 ; c] -> h_0
        self.encoder = _mlp(
            in_dim=config.feature_dim + config.covariate_dim,
            hidden=config.encoder_hidden,
            out_dim=config.hidden_dim,
            act=config.activation,
            dropout=config.dropout,
        )

        # Vector field
        self.vector_field = _ConditionalVectorField(config)

        # Decoder: h -> x̂
        self.decoder = _mlp(
            in_dim=config.hidden_dim,
            hidden=config.decoder_hidden,
            out_dim=config.feature_dim,
            act=config.activation,
            dropout=config.dropout,
        )

    def encode(self, x_0: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Encode baseline + covariates into initial latent state h_0."""
        if x_0.shape[0] != c.shape[0]:
            raise ValueError(
                f"Batch mismatch: x_0 has {x_0.shape[0]} rows but c has {c.shape[0]}."
            )
        return self.encoder(torch.cat([x_0, c], dim=-1))

    def integrate(
        self, h_0: torch.Tensor, c: torch.Tensor, t_eval: torch.Tensor
    ) -> torch.Tensor:
        """Solve dh/dt = f(h, c, t) on t_eval. Returns (T, batch, H)."""
        if t_eval.dim() != 1:
            raise ValueError(f"t_eval must be 1-D, got shape {tuple(t_eval.shape)}")
        if t_eval.numel() < 2:
            raise ValueError("t_eval must contain at least 2 timepoints (t=0 and t>=0).")

        self.vector_field.set_condition(c)
        return odeint(
            self.vector_field,
            h_0,
            t_eval,
            method=self.config.ode_solver,
            rtol=self.config.ode_rtol,
            atol=self.config.ode_atol,
        )

    def forward(
        self, x_0: torch.Tensor, c: torch.Tensor, t_eval: torch.Tensor
    ) -> torch.Tensor:
        """Predict the trajectory x̂(t_eval) from baseline x_0 conditioned on c.

        Parameters
        ----------
        x_0 : (batch, feature_dim) float tensor, baseline morphometry.
        c   : (batch, covariate_dim) float tensor, clinical covariates.
        t_eval : (T,) float tensor, time grid (years), ascending, t_eval[0] == 0.

        Returns
        -------
        (batch, T, feature_dim) predicted morphometry trajectory.
        """
        h_0 = self.encode(x_0, c)                           # (batch, H)
        h_t = self.integrate(h_0, c, t_eval)                # (T, batch, H)
        x_hat = self.decoder(h_t)                           # (T, batch, feature_dim)
        return x_hat.transpose(0, 1).contiguous()           # (batch, T, feature_dim)

    # -----------------------------------------------------------------------
    # Loss (observed-timepoint-masked MSE)
    # -----------------------------------------------------------------------
    @staticmethod
    def trajectory_mse(
        pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Masked MSE over observed timepoints.

        Parameters
        ----------
        pred   : (batch, T, D) model output.
        target : (batch, T, D) observed morphometry (any value at masked entries).
        mask   : (batch, T) 1=observed, 0=missing.

        Returns
        -------
        Scalar MSE averaged over observed (sample, timepoint, feature) entries.
        """
        if pred.shape != target.shape:
            raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")
        if mask.shape != pred.shape[:2]:
            raise ValueError(
                f"mask shape must be {pred.shape[:2]}, got {tuple(mask.shape)}"
            )
        # mask: (batch, T) -> (batch, T, 1) for broadcast over feature dim
        m = mask.unsqueeze(-1).to(pred.dtype)
        sq_err = (pred - target) ** 2 * m
        denom = (m.sum() * pred.shape[-1]) + 1e-8
        return sq_err.sum() / denom
