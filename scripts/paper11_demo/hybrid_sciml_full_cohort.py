#!/usr/bin/env python3
"""Paper 11 Hybrid SciML full-cohort extension (Task 1).

Scales the demo at ``scripts/paper11_demo/hybrid_sciml_neural_ode.py`` from
150 train / 50 test to the full ≥3-scan cohort (~428 patients), with the
following methodological upgrades:

1. Full cohort, patient-level 70/15/15 split (seed 42) — no caps.
2. Fair pure-mechanistic baseline that forecasts from ``s[0]`` at ``t[0]``
   (matching the deep-model horizon), with ``pure_mech_anchor_last`` kept
   as a secondary oracle-anchored reference.
3. Train/val split with early stopping on val-last-scan MAE (patience=10,
   max 100 epochs). Best-val-MAE state is restored before test eval.
4. CLI grid-sweep hyperparameters: --lambda-physics, --lambda-monotone,
   --use-gru (GRU vs. pointwise MLP residual), --seed, --config-id,
   --epochs, --patience, --bootstrap-resamples, --models.
5. Patient-bootstrap 95% CIs on Δ(hybrid − pure_mech_fair) and
   Δ(hybrid − pure_nn). Resamples test patients with replacement, NOT
   marginal MAEs.
6. Outputs under ``outputs/paper11_demo/full_cohort/<config_id>/`` only.
   The existing demo script + outputs are NOT touched.

Usage (smoke test)::

    .venv/bin/python scripts/paper11_demo/hybrid_sciml_full_cohort.py \\
        --config-id smoke --epochs 3 --patience 2 \\
        --bootstrap-resamples 50 --models both

Usage (baseline grid-point)::

    .venv/bin/python scripts/paper11_demo/hybrid_sciml_full_cohort.py \\
        --config-id baseline --epochs 100 --patience 10 \\
        --lambda-physics 0.01 --lambda-monotone 0.01 --models both
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchdiffeq import odeint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LONGITUDINAL = PROJECT_ROOT / "data/07_paper3_features/longitudinal_features.csv"
OUT_ROOT = PROJECT_ROOT / "outputs/paper11_demo/full_cohort"

DEVICE = "cpu"  # torchdiffeq adjoint complications on MPS/CUDA — CPU is the demo standard.

# Fearnley-Lees literature-prior baseline rate used by both the pure-mech
# fair baseline and the hybrid initialisation (tautology-free per
# paper12_phys_gimin_scope).
K_AGE_LIT_RATE = 0.025  # /yr
K_AGE_LIT_LOG = np.log(1.0 - K_AGE_LIT_RATE)  # kept for provenance; not used in the fair baseline formula

BASELINE_FEATS = [
    "age_at_visit", "sex",
    "updrs1_total", "updrs3_total",
    "ess_total", "rbd_total", "scopa_aut_total", "upsit_total",
    "moca_total",
    "lrrk2_carrier", "gba_carrier",
]


# ─────────────────────────────────────────────────────────────────────
# Data loading (identical filter + normalisation to the demo, no caps)
# ─────────────────────────────────────────────────────────────────────

def load_split(seed: int = 42) -> dict:
    """Load longitudinal features, filter to ≥3-scan patients, patient-level 70/15/15 split."""
    df = pd.read_csv(LONGITUDINAL, low_memory=False)

    sbr_col = "putamen_mean_sbr"
    if sbr_col not in df.columns:
        left = "sbr_putamen_l" if "sbr_putamen_l" in df.columns else "putamen_left_sbr"
        right = "sbr_putamen_r" if "sbr_putamen_r" in df.columns else "putamen_right_sbr"
        df[sbr_col] = (df[left] + df[right]) / 2

    df = df.dropna(subset=[sbr_col, "months_from_baseline", "PATNO"])
    df["t_years"] = df["months_from_baseline"] / 12.0

    present_feats = [f for f in BASELINE_FEATS if f in df.columns]
    baseline_df = df.loc[df.groupby("PATNO")["t_years"].idxmin()][["PATNO"] + present_feats].copy()
    for f in present_feats:
        baseline_df[f] = baseline_df[f].fillna(baseline_df[f].median())

    visit_counts = df.groupby("PATNO").size()
    keep_patnos = visit_counts[visit_counts >= 3].index.tolist()
    df = df[df["PATNO"].isin(keep_patnos)].copy()
    baseline_df = baseline_df[baseline_df["PATNO"].isin(keep_patnos)].copy()

    patnos = sorted(keep_patnos)
    rng = np.random.default_rng(seed)
    rng.shuffle(patnos)
    n = len(patnos)
    train_pats = set(patnos[: int(0.70 * n)])
    val_pats = set(patnos[int(0.70 * n): int(0.85 * n)])
    test_pats = set(patnos[int(0.85 * n):])

    def subset(pats: set) -> dict:
        sub = df[df["PATNO"].isin(pats)].sort_values(["PATNO", "t_years"])
        base = baseline_df[baseline_df["PATNO"].isin(pats)].set_index("PATNO")
        return {"trajectories": sub, "baseline": base, "feats": present_feats}

    print(f"[data] Kept {len(keep_patnos)} patients with ≥3 scans (from {len(visit_counts)} total)")
    print(f"[data] Split: train={len(train_pats)}  val={len(val_pats)}  test={len(test_pats)}")
    return {"train": subset(train_pats), "val": subset(val_pats), "test": subset(test_pats)}


def prepare_batch(split: dict, *, feat_norms: dict | None = None) -> dict:
    """Return list of (patno, t_tensor, sbr_tensor, feat_tensor) per patient.

    If ``feat_norms`` is provided, use those means/stds (from train split)
    to z-score normalise. Otherwise compute them fresh from the current
    split and return them. This avoids val/test leakage.
    """
    traj = split["trajectories"]
    base = split["baseline"].copy()
    feats = split["feats"]

    # NaN fill with per-feature cohort-level fallback
    for f in feats:
        col = pd.to_numeric(base[f], errors="coerce")
        m = col.mean()
        if pd.isna(m):
            m = 0.0
        base[f] = col.fillna(m)

    # Z-score normalise
    if feat_norms is None:
        feat_norms = {}
        for f in feats:
            mu = float(base[f].mean())
            sd = float(base[f].std() or 1.0)
            feat_norms[f] = (mu, sd)
    for f in feats:
        mu, sd = feat_norms[f]
        base[f] = (base[f] - mu) / (sd if sd else 1.0)

    records = []
    for patno, g in traj.groupby("PATNO"):
        g = g.sort_values("t_years")
        t = torch.tensor(g["t_years"].values, dtype=torch.float32)
        s = torch.tensor(g["putamen_mean_sbr"].values, dtype=torch.float32)
        if patno not in base.index:
            continue
        f_vec = base.loc[patno, feats].values.astype(np.float32)
        if np.any(np.isnan(f_vec)):
            continue
        f = torch.tensor(f_vec, dtype=torch.float32)
        records.append((int(patno), t, s, f))
    return {"records": records, "n_features": len(feats), "feat_norms": feat_norms}


# ─────────────────────────────────────────────────────────────────────
# Residual modules: pointwise MLP vs. GRU-over-trajectory
# ─────────────────────────────────────────────────────────────────────

class PointwiseMLPResidual(nn.Module):
    """Original demo residual: MLP(S_t, features) → scalar."""

    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + n_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    layer.weight.mul_(0.1)
                    if layer.bias is not None:
                        layer.bias.zero_()
        self._features = None
        self._history = None  # unused, for API parity with GRU

    def set_context(self, features: torch.Tensor, t_obs: torch.Tensor, s_obs: torch.Tensor) -> None:
        self._features = features

    def residual_at(self, y: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        inp = torch.cat([y.view(-1), self._features]).unsqueeze(0)
        return self.net(inp).squeeze(0)


class GRUTrajectoryResidual(nn.Module):
    """GRU-based residual: consumes trajectory-so-far ``{(s_k, feat_k) : t_k ≤ t}`` at each call.

    The hidden state is recomputed from scratch on every forward pass
    (simple, correct, slow — acceptable for the grid we're running,
    per the Task 1 architectural spec).
    """

    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.hidden = hidden
        self.n_features = n_features
        self.gru = nn.GRU(input_size=1 + n_features, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        with torch.no_grad():
            # Small init on the output head so early training ≈ pure mechanistic.
            self.head.weight.mul_(0.1)
            if self.head.bias is not None:
                self.head.bias.zero_()
        self._features = None
        self._t_obs = None
        self._s_obs = None

    def set_context(self, features: torch.Tensor, t_obs: torch.Tensor, s_obs: torch.Tensor) -> None:
        self._features = features
        self._t_obs = t_obs
        self._s_obs = s_obs

    def residual_at(self, y: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        # Select the prefix of observed visits with t_k ≤ current t.
        t_val = float(t) if isinstance(t, (int, float)) else float(t.detach().item())
        mask = self._t_obs <= t_val
        if mask.sum().item() == 0:
            # Before any observation — use just the first observed visit as anchor.
            mask = torch.zeros_like(self._t_obs, dtype=torch.bool)
            mask[0] = True
        s_prefix = self._s_obs[mask]  # shape (k,)
        feat_expanded = self._features.unsqueeze(0).expand(int(mask.sum().item()), -1)  # (k, n_feat)
        seq = torch.cat([s_prefix.unsqueeze(-1), feat_expanded], dim=-1).unsqueeze(0)  # (1, k, 1+n_feat)
        _, h = self.gru(seq)  # h: (1, 1, hidden)
        out = self.head(h.squeeze(0).squeeze(0))  # (1,)
        return out


# ─────────────────────────────────────────────────────────────────────
# Model classes (share a common residual-based API for training loop)
# ─────────────────────────────────────────────────────────────────────

class PureNeuralODE(nn.Module):
    """dS/dt = NN(S, features) — fully data-driven baseline."""

    def __init__(self, n_features: int, hidden: int = 32, use_gru: bool = False):
        super().__init__()
        if use_gru:
            self.residual_net = GRUTrajectoryResidual(n_features, hidden)
        else:
            self.residual_net = PointwiseMLPResidual(n_features, hidden)
        self.use_gru = use_gru

    def set_context(self, features: torch.Tensor, t_obs: torch.Tensor, s_obs: torch.Tensor) -> None:
        self.residual_net.set_context(features, t_obs, s_obs)

    def residual(self, y: torch.Tensor, t: torch.Tensor | float = 0.0) -> torch.Tensor:
        return self.residual_net.residual_at(y, t)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.residual_net.residual_at(y, t)


class PhysicsInformedNeuralODE(nn.Module):
    """dS/dt = -k_age·S + NN_residual(S, features).

    ``log_k_age`` is initialised to log(0.025) (Fearnley-Lees lit prior).
    The residual is small-initialised so early training ≈ pure mechanistic.
    """

    def __init__(self, n_features: int, hidden: int = 32, use_gru: bool = False):
        super().__init__()
        self.log_k_age = nn.Parameter(torch.tensor(np.log(K_AGE_LIT_RATE), dtype=torch.float32))
        if use_gru:
            self.residual_net = GRUTrajectoryResidual(n_features, hidden)
        else:
            self.residual_net = PointwiseMLPResidual(n_features, hidden)
        self.use_gru = use_gru

    def set_context(self, features: torch.Tensor, t_obs: torch.Tensor, s_obs: torch.Tensor) -> None:
        self.residual_net.set_context(features, t_obs, s_obs)

    def residual(self, y: torch.Tensor, t: torch.Tensor | float = 0.0) -> torch.Tensor:
        return self.residual_net.residual_at(y, t)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k_age = torch.exp(self.log_k_age)
        return -k_age * y + self.residual_net.residual_at(y, t)


# ─────────────────────────────────────────────────────────────────────
# Evaluation primitives
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_records(
    model: nn.Module | None,
    records: list,
    *,
    mode: str,
    k_rate: float = K_AGE_LIT_RATE,
) -> list[dict]:
    """Per-patient last-scan eval.

    ``mode`` ∈ {"pure_mech_fair", "pure_mech_anchor_last", "deep"}.

    Returns a list of per-patient dicts (one entry per patient that yielded
    a valid prediction) with fields: patno, s_obs_last, s_pred_last,
    abs_err, t_horizon_yrs, t_baseline_yrs, n_observed_visits.
    """
    out = []
    eval_failures = 0
    first_failure_logged = False
    n_records = len(records)
    for patno, t, s, f in records:
        if len(t) < 3:
            continue
        t_np = t.numpy()
        s_np = s.numpy()
        t_0 = float(t_np[0])
        t_last = float(t_np[-1])
        s_0 = float(s_np[0])
        s_prev_last = float(s_np[-2])
        s_last = float(s_np[-1])
        horizon_fair = t_last - t_0
        horizon_anchor = t_last - float(t_np[-2])

        if mode == "pure_mech_fair":
            s_pred = s_0 * float(np.exp(-k_rate * horizon_fair))
        elif mode == "pure_mech_anchor_last":
            s_pred = s_prev_last * float(np.exp(-k_rate * horizon_anchor))
        elif mode == "deep":
            if model is None:
                raise ValueError("deep mode requires a model")
            model.eval()
            # Feed the full trajectory as observed context (last-scan-out);
            # the GRU/MLP residual sees (t[:-1], s[:-1]) during ODE integration
            # from t[0] to t[-1]. This matches the Task 1 spec of forecasting
            # the full trajectory from s[0].
            model.set_context(f, t[:-1], s[:-1])
            try:
                pred = odeint(model, s[0:1], t, method="dopri5", atol=1e-4, rtol=1e-3)
            except Exception as exc:
                eval_failures += 1
                if not first_failure_logged:
                    print(f"  [warn] odeint fail (eval, patno {int(patno)}): "
                          f"{type(exc).__name__}: {exc}")
                    first_failure_logged = True
                continue
            pred = pred.squeeze(-1) if pred.ndim > 1 else pred
            s_pred = float(pred[-1].item() if pred[-1].ndim == 0 else pred[-1, 0].item())
        else:
            raise ValueError(f"unknown mode: {mode}")

        abs_err = abs(s_pred - s_last)
        # For anchor-last the prediction horizon is 1-step (t[-2] -> t[-1]);
        # for fair/deep modes it's the full trajectory (t[0] -> t[-1]).
        t_horizon = horizon_anchor if mode == "pure_mech_anchor_last" else horizon_fair
        out.append({
            "patno": int(patno),
            "s_obs_last": float(s_last),
            "s_pred_last": float(s_pred),
            "abs_err": float(abs_err),
            "t_horizon_yrs": float(t_horizon),
            "t_baseline_yrs": float(t_0),
            "n_observed_visits": int(len(t)),
        })
    if mode == "deep" and eval_failures > 0:
        print(f"  [eval-{mode}] odeint_failures={eval_failures}/{n_records}")
    return out


def aggregate_metrics(per_patient: list[dict]) -> dict:
    """Compute MAE, RMSE, median AE, and per-horizon MAE from per-patient records."""
    if not per_patient:
        return {
            "test_mae": float("nan"),
            "test_rmse": float("nan"),
            "test_median_abs_err": float("nan"),
            "per_horizon_mae": {},
            "n": 0,
        }
    errs = np.array([r["abs_err"] for r in per_patient], dtype=float)
    horizons = np.array([r["t_horizon_yrs"] for r in per_patient], dtype=float)

    per_horizon: dict = {}
    bins = [("0-1yr", 0, 1), ("1-3yr", 1, 3), ("3-5yr", 3, 5), ("5yr+", 5, float("inf"))]
    for label, lo, hi in bins:
        mask = (horizons > lo if lo > 0 else horizons >= lo) & (horizons <= hi)
        if mask.sum() > 0:
            per_horizon[label] = {"mae": float(errs[mask].mean()), "n": int(mask.sum())}
        else:
            per_horizon[label] = {"mae": None, "n": 0}

    return {
        "test_mae": float(errs.mean()),
        "test_rmse": float(np.sqrt((errs ** 2).mean())),
        "test_median_abs_err": float(np.median(errs)),
        "per_horizon_mae": per_horizon,
        "n": int(len(errs)),
    }


def horizon_bin(h: float) -> str:
    if h <= 1:
        return "0-1yr"
    if h <= 3:
        return "1-3yr"
    if h <= 5:
        return "3-5yr"
    return "5yr+"


# ─────────────────────────────────────────────────────────────────────
# Training with early stopping on val-last-scan MAE
# ─────────────────────────────────────────────────────────────────────

def train_with_early_stopping(
    model: nn.Module,
    train_records: list,
    val_records: list,
    *,
    epochs: int,
    patience: int,
    lambda_physics: float,
    lambda_monotone: float,
    lr: float = 1e-3,
) -> dict:
    """Train with val-last-scan MAE early stopping. Returns history dict and best state dict."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses: list[float] = []
    val_maes: list[float] = []
    best_val_mae = float("inf")
    best_state: dict | None = None
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(epochs):
        model.train()
        total = 0.0
        n = 0
        epoch_failures = 0
        for patno, t, s, f in train_records:
            if len(t) < 3:
                continue
            t_train = t[:-1]
            s_train = s[:-1]

            if len(t_train) < 2:
                continue

            # The deep model sees the observed trajectory prefix (excluding the
            # held-out last scan) as context for residual computations.
            model.set_context(f, t_train, s_train)
            try:
                pred_train = odeint(model, s_train[0:1], t_train, method="dopri5", atol=1e-4, rtol=1e-3)
            except Exception as exc:
                epoch_failures += 1
                if epoch == 0 and epoch_failures == 1:
                    print(f"  [warn] odeint fail (epoch {epoch}, patno {int(patno)}): "
                          f"{type(exc).__name__}: {exc}")
                continue
            pred_train = pred_train.squeeze(-1) if pred_train.ndim > 1 else pred_train

            mse = ((pred_train.squeeze() - s_train) ** 2).mean()

            phys_pen = torch.tensor(0.0)
            if isinstance(model, PhysicsInformedNeuralODE) and lambda_physics > 0:
                resid_samples = torch.stack([
                    model.residual(s_train[i:i + 1], t_train[i]) for i in range(len(s_train))
                ])
                phys_pen = lambda_physics * (resid_samples ** 2).mean()

            # NOTE: The demo at scripts/paper11_demo/hybrid_sciml_neural_ode.py wraps
            # `derivs` in `torch.no_grad()`, which accidentally disables the gradient
            # through the monotonicity penalty. We intentionally do NOT wrap here — the
            # penalty should be a real differentiable constraint. Consequence: results at
            # lambda_monotone > 0 are not bit-for-bit comparable to the demo.
            mono_pen = torch.tensor(0.0)
            if lambda_monotone > 0:
                derivs = torch.stack([
                    model(t_train[i], s_train[i:i + 1]) for i in range(len(s_train))
                ])
                mono_pen = lambda_monotone * torch.relu(derivs + 1e-4).mean()

            loss = mse + phys_pen + mono_pen
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            n += 1
        if epoch_failures > 0:
            print(f"  epoch {epoch + 1}/{epochs} odeint_failures={epoch_failures}/{len(train_records)}")
        epoch_train_loss = total / max(n, 1)
        train_losses.append(epoch_train_loss)

        # --- val-last-scan MAE ---
        val_eval = evaluate_records(model, val_records, mode="deep")
        val_agg = aggregate_metrics(val_eval)
        val_mae = val_agg["test_mae"]
        val_maes.append(val_mae)

        improved = val_mae < best_val_mae - 1e-6
        if improved:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        print(f"  epoch {epoch + 1:3d}/{epochs}  train_loss={epoch_train_loss:.4f}  "
              f"val_mae={val_mae:.4f}  best={best_val_mae:.4f}@{best_epoch + 1}  "
              f"stale={epochs_since_improve}")

        if epochs_since_improve >= patience:
            print(f"  [early stop] no val-MAE improvement for {patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        # Degenerate: training produced no valid val eval; keep current state.
        best_state = copy.deepcopy(model.state_dict())
        best_val_mae = val_maes[-1] if val_maes else float("nan")
        best_epoch = len(train_losses) - 1

    return {
        "train_losses": train_losses,
        "val_maes": val_maes,
        "best_val_mae": float(best_val_mae),
        "best_epoch": int(best_epoch),
        "epochs_trained": int(len(train_losses)),
        "best_state_dict": best_state,
    }


# ─────────────────────────────────────────────────────────────────────
# Bootstrap CI on Δ metrics
# ─────────────────────────────────────────────────────────────────────

def bootstrap_delta_ci(
    errs_a: np.ndarray,
    errs_b: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> dict:
    """Patient-bootstrap CI on Δ = mean(errs_a) − mean(errs_b).

    Per Task 1 spec: resample patient indices with replacement each time,
    compute mean per model on the resample, then take Δ. The 2.5/97.5
    percentiles of the Δ distribution are the 95% CI.
    """
    assert len(errs_a) == len(errs_b), "bootstrap requires per-patient paired errors"
    n = len(errs_a)
    deltas = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[b] = errs_a[idx].mean() - errs_b[idx].mean()
    point = float(errs_a.mean() - errs_b.mean())
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return {"point": point, "ci_lo": lo, "ci_hi": hi}


def maybe_bootstrap(
    errs_a: np.ndarray,
    errs_b: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
    label: str = "",
) -> dict:
    """Wrap bootstrap_delta_ci with a guard against <2 paired samples.

    Aggressive smoke caps (e.g., --train-cap 30 --test-cap 10) or pathological
    odeint failures can leave fewer than 2 paired patients, in which case
    percentile CIs are meaningless. Return {nan, None, None} with a warning.
    """
    n_paired = min(len(errs_a), len(errs_b))
    if len(errs_a) < 2 or len(errs_b) < 2:
        print(f"  [warn] bootstrap skipped{' (' + label + ')' if label else ''}: "
              f"paired n={n_paired} < 2")
        return {"point": float("nan"), "ci_lo": None, "ci_hi": None}
    return bootstrap_delta_ci(errs_a, errs_b, n_resamples=n_resamples,
                              rng=rng, alpha=alpha)


# ─────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────

def per_patient_csv_rows(per_patient: list[dict], split_name: str, model_name: str) -> list[dict]:
    rows = []
    for r in per_patient:
        rows.append({
            "patno": r["patno"],
            "split": split_name,
            "model": model_name,
            "s_obs_last": r["s_obs_last"],
            "s_pred_last": r["s_pred_last"],
            "abs_err": r["abs_err"],
            "t_horizon_yrs": r["t_horizon_yrs"],
            "horizon_bin": horizon_bin(r["t_horizon_yrs"]),
            "t_baseline_yrs": r["t_baseline_yrs"],
            "n_observed_visits": r["n_observed_visits"],
        })
    return rows


def get_git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper 11 hybrid SciML full-cohort extension (Task 1).")
    p.add_argument("--lambda-physics", type=float, default=0.01,
                   help="Weight on ‖NN_residual‖² penalty (default 0.01).")
    p.add_argument("--lambda-monotone", type=float, default=0.01,
                   help="Weight on ReLU(dS/dt+1e-4) penalty (default 0.01).")
    p.add_argument("--use-gru", action="store_true",
                   help="Use GRU-over-trajectory residual instead of pointwise MLP.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config-id", type=str, required=True,
                   help="Subdirectory name under outputs/paper11_demo/full_cohort/.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--bootstrap-resamples", type=int, default=1000)
    p.add_argument("--models", choices=["pure_nn", "hybrid", "both"], default="both")
    p.add_argument("--train-cap", type=int, default=None,
                   help="Optional smoke-test cap on #train patients (default: no cap).")
    p.add_argument("--val-cap", type=int, default=None,
                   help="Optional smoke-test cap on #val patients (default: no cap).")
    p.add_argument("--test-cap", type=int, default=None,
                   help="Optional smoke-test cap on #test patients (default: no cap).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = OUT_ROOT / args.config_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f"Paper 11 hybrid SciML (full cohort) — config={args.config_id}")
    print("=" * 75)
    print(f"[cfg] lambda_physics={args.lambda_physics}  lambda_monotone={args.lambda_monotone}  "
          f"use_gru={args.use_gru}  seed={args.seed}")
    print(f"[cfg] epochs_max={args.epochs}  patience={args.patience}  bootstrap={args.bootstrap_resamples}")
    print(f"[cfg] models={args.models}")

    splits = load_split(seed=args.seed)
    train_batch = prepare_batch(splits["train"])
    feat_norms = train_batch["feat_norms"]
    val_batch = prepare_batch(splits["val"], feat_norms=feat_norms)
    test_batch = prepare_batch(splits["test"], feat_norms=feat_norms)
    n_features = train_batch["n_features"]

    train_records = train_batch["records"]
    val_records = val_batch["records"]
    test_records = test_batch["records"]

    if args.train_cap is not None:
        train_records = train_records[: args.train_cap]
    if args.val_cap is not None:
        val_records = val_records[: args.val_cap]
    if args.test_cap is not None:
        test_records = test_records[: args.test_cap]

    print(f"[data] n_features={n_features}  n_train={len(train_records)}  "
          f"n_val={len(val_records)}  n_test={len(test_records)}")

    rng = np.random.default_rng(args.seed)

    all_per_patient_rows: list[dict] = []

    # ── Pure-mech fair (forecast s[0] → t[-1]) ──
    print("\n--- Pure mechanistic (FAIR baseline: s[0] → t[-1] at 2.5%/yr) ---")
    pm_fair_train = evaluate_records(None, train_records, mode="pure_mech_fair")
    pm_fair_val = evaluate_records(None, val_records, mode="pure_mech_fair")
    pm_fair_test = evaluate_records(None, test_records, mode="pure_mech_fair")
    pm_fair_test_agg = aggregate_metrics(pm_fair_test)
    print(f"  test MAE={pm_fair_test_agg['test_mae']:.4f}  RMSE={pm_fair_test_agg['test_rmse']:.4f}")
    all_per_patient_rows.extend(per_patient_csv_rows(pm_fair_train, "train", "pure_mech"))
    all_per_patient_rows.extend(per_patient_csv_rows(pm_fair_val, "val", "pure_mech"))
    all_per_patient_rows.extend(per_patient_csv_rows(pm_fair_test, "test", "pure_mech"))

    # ── Pure-mech anchor-last (oracle-ish, secondary reference) ──
    print("\n--- Pure mechanistic (anchor-last reference: s[-2] → t[-1]) ---")
    pm_al_train = evaluate_records(None, train_records, mode="pure_mech_anchor_last")
    pm_al_val = evaluate_records(None, val_records, mode="pure_mech_anchor_last")
    pm_al_test = evaluate_records(None, test_records, mode="pure_mech_anchor_last")
    pm_al_test_agg = aggregate_metrics(pm_al_test)
    print(f"  test MAE={pm_al_test_agg['test_mae']:.4f}  RMSE={pm_al_test_agg['test_rmse']:.4f}")
    all_per_patient_rows.extend(per_patient_csv_rows(pm_al_train, "train", "pure_mech_anchor_last"))
    all_per_patient_rows.extend(per_patient_csv_rows(pm_al_val, "val", "pure_mech_anchor_last"))
    all_per_patient_rows.extend(per_patient_csv_rows(pm_al_test, "test", "pure_mech_anchor_last"))

    # ── Deep models ──
    train_hist: dict = {}

    pure_nn_summary: dict | None = None
    pure_nn_test_errs: np.ndarray | None = None
    hybrid_summary: dict | None = None
    hybrid_test_errs: np.ndarray | None = None

    if args.models in ("pure_nn", "both"):
        print("\n--- Pure Neural ODE (data-driven; no mechanistic prior) ---")
        torch.manual_seed(args.seed)
        model_a = PureNeuralODE(n_features=n_features, use_gru=args.use_gru).to(DEVICE)
        hist_a = train_with_early_stopping(
            model_a, train_records, val_records,
            epochs=args.epochs, patience=args.patience,
            lambda_physics=0.0, lambda_monotone=args.lambda_monotone,  # no phys pen on pure NN
            lr=1e-3,
        )
        eval_a_train = evaluate_records(model_a, train_records, mode="deep")
        eval_a_val = evaluate_records(model_a, val_records, mode="deep")
        eval_a_test = evaluate_records(model_a, test_records, mode="deep")
        agg_a_test = aggregate_metrics(eval_a_test)
        print(f"  [pure_nn] test MAE={agg_a_test['test_mae']:.4f}  best_val={hist_a['best_val_mae']:.4f}"
              f"  @epoch {hist_a['best_epoch'] + 1}")
        torch.save(hist_a["best_state_dict"], ckpt_dir / "pure_nn.pt")
        train_hist["pure_nn"] = {
            "train_losses": hist_a["train_losses"],
            "val_maes": hist_a["val_maes"],
            "best_val_mae": hist_a["best_val_mae"],
            "best_epoch": hist_a["best_epoch"],
            "epochs_trained": hist_a["epochs_trained"],
        }
        all_per_patient_rows.extend(per_patient_csv_rows(eval_a_train, "train", "pure_nn"))
        all_per_patient_rows.extend(per_patient_csv_rows(eval_a_val, "val", "pure_nn"))
        all_per_patient_rows.extend(per_patient_csv_rows(eval_a_test, "test", "pure_nn"))
        pure_nn_summary = {
            **agg_a_test,
            "best_val_mae": float(hist_a["best_val_mae"]),
            "best_epoch": int(hist_a["best_epoch"]),
            "epochs_trained": int(hist_a["epochs_trained"]),
        }
        pure_nn_test_errs = np.array([r["abs_err"] for r in eval_a_test], dtype=float)
        pure_nn_test_patnos = [r["patno"] for r in eval_a_test]
    else:
        pure_nn_test_patnos = None

    if args.models in ("hybrid", "both"):
        print("\n--- Physics-Informed Neural ODE (lit prior + learned residual) ---")
        torch.manual_seed(args.seed)
        model_b = PhysicsInformedNeuralODE(n_features=n_features, use_gru=args.use_gru).to(DEVICE)
        hist_b = train_with_early_stopping(
            model_b, train_records, val_records,
            epochs=args.epochs, patience=args.patience,
            lambda_physics=args.lambda_physics, lambda_monotone=args.lambda_monotone,
            lr=1e-3,
        )
        eval_b_train = evaluate_records(model_b, train_records, mode="deep")
        eval_b_val = evaluate_records(model_b, val_records, mode="deep")
        eval_b_test = evaluate_records(model_b, test_records, mode="deep")
        agg_b_test = aggregate_metrics(eval_b_test)
        learned_k = float(np.exp(model_b.log_k_age.item()))
        print(f"  [hybrid] test MAE={agg_b_test['test_mae']:.4f}  best_val={hist_b['best_val_mae']:.4f}"
              f"  @epoch {hist_b['best_epoch'] + 1}  learned_k={learned_k:.4f}/yr")
        torch.save(hist_b["best_state_dict"], ckpt_dir / "hybrid.pt")
        train_hist["hybrid"] = {
            "train_losses": hist_b["train_losses"],
            "val_maes": hist_b["val_maes"],
            "best_val_mae": hist_b["best_val_mae"],
            "best_epoch": hist_b["best_epoch"],
            "epochs_trained": hist_b["epochs_trained"],
        }
        all_per_patient_rows.extend(per_patient_csv_rows(eval_b_train, "train", "hybrid"))
        all_per_patient_rows.extend(per_patient_csv_rows(eval_b_val, "val", "hybrid"))
        all_per_patient_rows.extend(per_patient_csv_rows(eval_b_test, "test", "hybrid"))
        hybrid_summary = {
            **agg_b_test,
            "best_val_mae": float(hist_b["best_val_mae"]),
            "best_epoch": int(hist_b["best_epoch"]),
            "epochs_trained": int(hist_b["epochs_trained"]),
            "learned_k_age_per_yr": learned_k,
        }
        hybrid_test_errs = np.array([r["abs_err"] for r in eval_b_test], dtype=float)
        hybrid_test_patnos = [r["patno"] for r in eval_b_test]
    else:
        hybrid_test_patnos = None

    # ── Bootstrap Δ CIs on paired test set ──
    print("\n--- Bootstrap 95% CI on Δ metrics ---")
    deltas: dict = {}

    if hybrid_test_errs is not None:
        # Build per-patient paired error arrays: patient index → (hybrid_err, pure_mech_fair_err)
        pm_fair_by_patno = {r["patno"]: r["abs_err"] for r in pm_fair_test}
        errs_hybrid = []
        errs_pm_fair = []
        for i, patno in enumerate(hybrid_test_patnos):
            if patno in pm_fair_by_patno:
                errs_hybrid.append(hybrid_test_errs[i])
                errs_pm_fair.append(pm_fair_by_patno[patno])
        errs_hybrid_np = np.array(errs_hybrid, dtype=float)
        errs_pm_fair_np = np.array(errs_pm_fair, dtype=float)
        ci = maybe_bootstrap(
            errs_hybrid_np, errs_pm_fair_np,
            n_resamples=args.bootstrap_resamples, rng=rng,
            label="hybrid vs pure_mech_fair",
        )
        deltas["hybrid_minus_puremech_fair"] = ci
        if ci["ci_lo"] is None:
            print(f"  Δ(hybrid − pure_mech_fair): point={ci['point']}  [CI skipped]")
        else:
            print(f"  Δ(hybrid − pure_mech_fair): {ci['point']:+.4f}  "
                  f"[{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}]")

        if pure_nn_test_errs is not None:
            pure_nn_by_patno = {pure_nn_test_patnos[i]: pure_nn_test_errs[i]
                                for i in range(len(pure_nn_test_patnos))}
            errs_hybrid_pn = []
            errs_pn = []
            for i, patno in enumerate(hybrid_test_patnos):
                if patno in pure_nn_by_patno:
                    errs_hybrid_pn.append(hybrid_test_errs[i])
                    errs_pn.append(pure_nn_by_patno[patno])
            ci = maybe_bootstrap(
                np.array(errs_hybrid_pn, dtype=float),
                np.array(errs_pn, dtype=float),
                n_resamples=args.bootstrap_resamples, rng=rng,
                label="hybrid vs pure_nn",
            )
            deltas["hybrid_minus_purenn"] = ci
            if ci["ci_lo"] is None:
                print(f"  Δ(hybrid − pure_nn):       point={ci['point']}  [CI skipped]")
            else:
                print(f"  Δ(hybrid − pure_nn):       {ci['point']:+.4f}  "
                      f"[{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}]")

    # ── Write summary.json ──
    summary = {
        "config_id": args.config_id,
        "lambda_physics": float(args.lambda_physics),
        "lambda_monotone": float(args.lambda_monotone),
        "use_gru": bool(args.use_gru),
        "seed": int(args.seed),
        "epochs_max": int(args.epochs),
        "patience": int(args.patience),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "n_train": int(len(train_records)),
        "n_val": int(len(val_records)),
        "n_test": int(len(test_records)),
        "n_features": int(n_features),
        "models_trained": [m for m in ("pure_nn", "hybrid")
                           if (m == "pure_nn" and pure_nn_summary is not None)
                           or (m == "hybrid" and hybrid_summary is not None)],
        "pure_mech_fair": pm_fair_test_agg,
        "pure_mech_anchor_last": pm_al_test_agg,
        "pure_nn": pure_nn_summary if pure_nn_summary is not None else {},
        "hybrid": hybrid_summary if hybrid_summary is not None else {},
        "deltas": deltas,
        "git_sha": get_git_sha(),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command_line": " ".join(sys.argv),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved: {out_dir / 'summary.json'}")

    # ── Write per_patient_results.csv ──
    pd.DataFrame(all_per_patient_rows).to_csv(out_dir / "per_patient_results.csv", index=False)
    print(f"Saved: {out_dir / 'per_patient_results.csv'}")

    # ── Write training_history.json ──
    (out_dir / "training_history.json").write_text(json.dumps(train_hist, indent=2, default=str))
    print(f"Saved: {out_dir / 'training_history.json'}")

    # ── 3-panel figure (same skeleton as the demo) ──
    try:
        models_present = ["Pure\nMech\n(fair)", "Pure\nMech\n(anchor-last)"]
        maes = [pm_fair_test_agg["test_mae"], pm_al_test_agg["test_mae"]]
        colors = ["#009E73", "#66C2A5"]
        if pure_nn_summary is not None:
            models_present.append("Pure\nNeural ODE")
            maes.append(pure_nn_summary["test_mae"])
            colors.append("#CC79A7")
        if hybrid_summary is not None:
            models_present.append("Hybrid\nPINN")
            maes.append(hybrid_summary["test_mae"])
            colors.append("#0072B2")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4.2))

        bars = ax1.bar(models_present, maes, color=colors, edgecolor="black", linewidth=0.5)
        for b, m in zip(bars, maes):
            ax1.text(b.get_x() + b.get_width() / 2, m, f"{m:.3f}",
                     ha="center", va="bottom", fontsize=9)
        ax1.set_ylabel("Held-out last-scan MAE (SBR)")
        ax1.set_title(f"(a) Test MAE (n={len(test_records)})")
        ax1.grid(axis="y", alpha=0.3)

        if "pure_nn" in train_hist:
            ax2.plot(train_hist["pure_nn"]["val_maes"], "o-", color="#CC79A7",
                     label="Pure Neural ODE (val)", linewidth=2, markersize=3)
        if "hybrid" in train_hist:
            ax2.plot(train_hist["hybrid"]["val_maes"], "s-", color="#0072B2",
                     label="Physics-Informed (val)", linewidth=2, markersize=3)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Val-last-scan MAE")
        ax2.set_title("(b) Val MAE per epoch")
        ax2.legend(fontsize=9, frameon=False)
        ax2.grid(alpha=0.3)

        horizons_labels = ["0-1yr", "1-3yr", "3-5yr", "5yr+"]
        def get_horizon_values(agg: dict) -> list[float]:
            vals = []
            for h in horizons_labels:
                rec = agg.get("per_horizon_mae", {}).get(h, {"mae": 0})
                vals.append(rec["mae"] if rec.get("mae") is not None else 0.0)
            return vals

        per_horizon_vals = {"Pure Mech (fair)": get_horizon_values(pm_fair_test_agg)}
        colors_ph = ["#009E73"]
        if pure_nn_summary is not None:
            per_horizon_vals["Pure NN"] = get_horizon_values(pure_nn_summary)
            colors_ph.append("#CC79A7")
        if hybrid_summary is not None:
            per_horizon_vals["Hybrid"] = get_horizon_values(hybrid_summary)
            colors_ph.append("#0072B2")
        x = np.arange(len(horizons_labels))
        w = 0.8 / len(per_horizon_vals)
        for i, (label, vals) in enumerate(per_horizon_vals.items()):
            ax3.bar(x + (i - (len(per_horizon_vals) - 1) / 2) * w, vals, w,
                    label=label, color=colors_ph[i], edgecolor="black", linewidth=0.3)
        ax3.set_xticks(x)
        ax3.set_xticklabels(horizons_labels, fontsize=9)
        ax3.set_ylabel("MAE by horizon")
        ax3.set_title("(c) Horizon-stratified error")
        ax3.legend(fontsize=8, frameon=False)
        ax3.grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"Paper 11 Hybrid SciML — full cohort ({args.config_id}, "
            f"n_test={len(test_records)})",
            y=1.02, fontsize=11,
        )
        fig.tight_layout()
        fig_pdf = out_dir / "hybrid_sciml_full.pdf"
        fig_png = out_dir / "hybrid_sciml_full.png"
        fig.savefig(fig_pdf, bbox_inches="tight")
        fig.savefig(fig_png, bbox_inches="tight", dpi=150)
        print(f"Saved: {fig_pdf}")
        print(f"Saved: {fig_png}")
    except Exception as e:
        print(f"[warn] figure generation failed: {e!r}")

    # Summary line
    print("\n" + "=" * 75)
    print(f"CONFIG {args.config_id} — test MAE summary")
    print("=" * 75)
    print(f"  pure_mech_fair        : {pm_fair_test_agg['test_mae']:.4f}")
    print(f"  pure_mech_anchor_last : {pm_al_test_agg['test_mae']:.4f}")
    if pure_nn_summary is not None:
        print(f"  pure_nn               : {pure_nn_summary['test_mae']:.4f}")
    if hybrid_summary is not None:
        print(f"  hybrid                : {hybrid_summary['test_mae']:.4f}")
    for k, v in deltas.items():
        if v["ci_lo"] is None or v["ci_hi"] is None:
            print(f"  Δ[{k}] = point={v['point']}  [CI skipped]")
        else:
            print(f"  Δ[{k}] = {v['point']:+.4f}  [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]")


if __name__ == "__main__":
    main()
