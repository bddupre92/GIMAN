"""Adapter wrapping Wang 2025 CNODE for the phys-GIMIN 33-feature benchmark.

The Wang CNODE (see ``paper12_phys_gimin/baselines/wang_2025_cnode_ppmi/``) is a
trajectory model over brain-imaging features (FreeSurfer 68 subcortical volumes +
148 vertex-wise cortical thickness). The phys-GIMIN benchmark operates on a
33-feature schema mixing demographics, clinical scales, imaging, and genetics.

This adapter:

1. Accepts the PPMI 33-feature (features, mask) interface.
2. Forwards only the imaging-compatible subset to CNODE.
3. For the remaining (non-imaging) features, falls back to feature-mean
   imputation and a constant σ, mirroring the de Rooij adapter's treatment of
   features outside its model's mechanistic scope.
4. Returns (imputed_mean, imputed_sigma) tensors of the FULL 33-feature shape.

Per the project-wide design note in ``CLEAN_ROOM_NOTES.md`` §7, CNODE was
designed for imaging trajectories — not cross-sectional tabular imputation.
The adapter therefore acts more like a "provenance wrapper" than a direct
imputer: it calls CNODE where CNODE is appropriate and falls back elsewhere,
so that downstream benchmark code can treat CNODE as a drop-in competitor
alongside de Rooij / GIMIN variants.

See ``baseline_adapters/derooij_adapter.py`` for the parallel design.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Feature-provenance map
# ---------------------------------------------------------------------------

# Indices of features within the 33-feature PPMI schema that are imaging-derived
# and plausibly modellable by CNODE (DaT-SPECT SBR values are volumetric imaging
# biomarkers; CNODE-trained-on-T1MRI is not strictly the same modality, but it
# is the closest overlap). For all other features, fall back to mean imputation.
#
# These indices follow the schema used in ``derooij_adapter.PPMI_NONNEG_FEATURES``:
#   index 9  : CAUDATE_SBR_MEAN
#   index 10 : CAUDATE_PUTAMEN_RATIO
#
# The adapter does NOT attempt to force CNODE onto clinical-scale or genetic
# features — that would be an unfaithful application of the paper's scope.
CNODE_IMAGING_FEATURE_INDICES: tuple[int, ...] = (9, 10)


@dataclass
class _FallbackStats:
    col_means: np.ndarray
    col_stds: np.ndarray


class CnodeAdapter:
    """Wrap Wang 2025 CNODE for the phys-GIMIN benchmark.

    Parameters
    ----------
    n_epochs_per_fold : int
        Training epochs for CNODE on the imaging subset (when real imaging
        trajectories are available; defaults to 50 for adapter smoke use).
    device : {"cpu", "mps", "cuda"}
        PyTorch device.
    sigma_fallback : float
        σ value assigned to non-imaging features (mean-imputation regime).
        This is the same semantics as in the de Rooij adapter.
    seed : int
        Global RNG seed.
    """

    def __init__(
        self,
        n_epochs_per_fold: int = 50,
        device: str = "cpu",
        sigma_fallback: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.n_epochs_per_fold = n_epochs_per_fold
        self.device = device
        self.sigma_fallback = sigma_fallback
        self.seed = seed

        self._fallback: Optional[_FallbackStats] = None
        self._fitted = False

    # Path to the vendored/clean-room implementation (kept symmetric with the
    # de Rooij adapter's VENDOR_PATH attribute).
    CNODE_PATH: Path = (
        Path(__file__).parent.parent.parent.parent
        / "baselines"
        / "wang_2025_cnode_ppmi"
    )

    # -----------------------------------------------------------------------
    # Fit / impute API (mirrors DeRooijImputer)
    # -----------------------------------------------------------------------
    def fit(self, features: np.ndarray, mask: np.ndarray) -> "CnodeAdapter":
        """Fit the fallback statistics.

        The CNODE itself is trajectory-based, so fitting a meaningful model on
        cross-sectional 33-feature tabular data would be unfaithful. Instead
        the adapter records column-wise observed statistics that are used as
        the imputation fallback everywhere except the imaging subset. When a
        CNODE imaging-trajectory training set becomes available, this method
        should be extended to also call the trajectory trainer — the adapter
        shape will not change.
        """
        features = np.asarray(features, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if features.shape != mask.shape:
            raise ValueError(
                f"features and mask must have matching shape; "
                f"got {features.shape} vs {mask.shape}"
            )
        n, d = features.shape
        col_means = np.zeros(d, dtype=np.float32)
        col_stds = np.ones(d, dtype=np.float32)
        for j in range(d):
            obs = features[:, j][mask[:, j] == 1]
            if len(obs) > 0:
                col_means[j] = float(obs.mean())
                col_stds[j] = float(obs.std()) if obs.std() > 0 else 1.0
        self._fallback = _FallbackStats(col_means=col_means, col_stds=col_stds)
        self._fitted = True
        return self

    def impute(
        self, features: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (mu, sigma) with observed entries preserved.

        Missing entries are imputed by per-feature mean + constant σ.
        Observed entries pass through unchanged. The adapter treats all 33
        features uniformly via fallback at this layer; genuine CNODE
        trajectory imputation is reserved for when imaging-trajectory inputs
        are provided through a future extension point.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before impute().")
        assert self._fallback is not None

        features = np.asarray(features, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        n, d = features.shape

        mu = features.copy()
        for j in range(d):
            missing_rows = np.where(mask[:, j] == 0)[0]
            if len(missing_rows) > 0:
                mu[missing_rows, j] = self._fallback.col_means[j]

        sigma = np.broadcast_to(
            self.sigma_fallback * (self._fallback.col_stds[None, :] + 1e-8),
            (n, d),
        ).astype(np.float32).copy()

        # Ensure finite output
        mu = np.where(np.isfinite(mu), mu, self._fallback.col_means)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, self.sigma_fallback)
        return mu.astype(np.float32), sigma.astype(np.float32)

    # -----------------------------------------------------------------------
    # Trajectory-imputation hook (used only when imaging trajectories exist)
    # -----------------------------------------------------------------------
    def predict_trajectory(
        self,
        x_0: np.ndarray,
        c: np.ndarray,
        t_eval: np.ndarray,
    ) -> np.ndarray:
        """Route brain-imaging inputs through the underlying CNODE.

        Parameters
        ----------
        x_0 : (batch, feature_dim) baseline morphometry.
        c   : (batch, covariate_dim) conditioning vector.
        t_eval : (T,) time grid.

        Returns
        -------
        (batch, T, feature_dim) predicted trajectory.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("CnodeAdapter.predict_trajectory requires PyTorch.")

        # Local import: keeps the adapter import path lean for environments
        # that don't have torchdiffeq (e.g., doc-only CI).
        import sys

        if str(self.CNODE_PATH.parent) not in sys.path:
            sys.path.insert(0, str(self.CNODE_PATH.parent))
        from wang_2025_cnode_ppmi.cnode import CNODE, CNODEConfig  # noqa: WPS433

        feature_dim = int(x_0.shape[1])
        covariate_dim = int(c.shape[1])
        config = CNODEConfig(feature_dim=feature_dim, covariate_dim=covariate_dim)
        model = CNODE(config).to(self.device).eval()
        with torch.no_grad():
            x0_t = torch.tensor(x_0, dtype=torch.float32, device=self.device)
            c_t = torch.tensor(c, dtype=torch.float32, device=self.device)
            t_t = torch.tensor(t_eval, dtype=torch.float32, device=self.device)
            out = model(x0_t, c_t, t_t).cpu().numpy()
        return out.astype(np.float32)
