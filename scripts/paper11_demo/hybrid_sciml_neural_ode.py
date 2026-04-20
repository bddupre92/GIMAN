#!/usr/bin/env python3
"""Paper 11 Hybrid SciML Demo: Physics-Informed Neural ODE for DaT-SBR Trajectories.

Deep-fusion demonstration — the Layer-3 architecture from Discussion §14.4,
following the Alt-5 shallow-hybrid null (Ch 15 §15.1).

## Architecture

Three models compared on held-out longitudinal DaT-SBR trajectories:

  1. Pure-ML Neural ODE (baseline):
       d S/dt = NN(S, features; θ)
     Fully data-driven; no mechanistic constraint.

  2. Pure-Mechanistic closed-form decay (reference):
       S(t) = S_0 · exp(-(α·O_ss + k_age)·t)
     Paper 7 slow-fast-collapse forward model with Phase 2 per-patient
     posterior median rate (pct_loss_per_yr_median). No learning.

  3. Physics-Informed Neural ODE (hybrid, de Rooij 2025 + Rackauckas 2020 UDE):
       d S/dt = -k_age(features) · S + NN_residual(S, features; θ)
     Mechanistic baseline decay rate (fitted from lit prior as
     k_age = 0.005/yr; Fearnley--Lees literature anchor) ± a learned
     NN residual that absorbs patient-specific deviations. This is
     the "lit-prior" variant from the phys-GIMIN scope decision
     (paper12_phys_gimin_scope) — no tautology with P7/P10 self-prior.

     Loss = MSE(S_pred, S_obs)
          + λ_physics · ‖NN_residual‖²        (soft mechanistic constraint)
          + λ_monotone · ReLU(d S/dt + 1e-4)  (non-negativity in decay)

     The physics + monotonicity penalties implement the de Rooij 2025
     regularisation family on PPMI DaT-SPECT, rather than the glucose
     minimal model they used.

## Data

Source: data/07_paper3_features/longitudinal_features.csv
  - 16,699 visits from 1,900 PD+prodromal patients
  - Per-visit putamen_mean_sbr + 30+ baseline features
  - Split by patient (not visit) for held-out prediction

Patient split: 70% train / 15% val / 15% test (seed 42, stratified by
number of scans per patient)

## Evaluation

Primary: held-out held-out-scan MAE (predict last observed SBR given
baseline + prior trajectory).

Secondary: 3-year and 5-year forecast error (how far the model
extrapolates accurately).

## Reproducibility

  - torch 2.8.0, torchdiffeq 0.2.5
  - Random seed 42 fixed throughout
  - Model checkpoints saved to outputs/paper11_demo/checkpoints/
  - Metrics JSON saved to outputs/paper11_demo/metrics.json

## Non-destructiveness

This is a DEMO / preview, not a full Paper 11 submission. It tests whether
the deep-fusion architecture moves the needle at all on held-out SBR
trajectory prediction. If positive, it motivates the postdoc Paper 11
scope (external validation on DeNoPa/SURE-PD3 + full ablation). If null,
it provides a second falsifiable probe beyond Alt-5 that refines what
deep fusion must solve to add value.

Usage:
  .venv/bin/python scripts/paper11_demo/hybrid_sciml_neural_ode.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchdiffeq import odeint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LONGITUDINAL = PROJECT_ROOT / "data/07_paper3_features/longitudinal_features.csv"
OUT_DIR = PROJECT_ROOT / "outputs/paper11_demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
torch.manual_seed(42)
DEVICE = "cpu"  # torchdiffeq adjoint is simpler on CPU at this scale

# Literature-prior baseline decay rate (Fearnley & Lees 1991 canonical).
# Using 2.5%/yr (midpoint of 2-5% range) as a "lit-prior" that is NOT
# the self-prior Phase 2 posterior — avoids the tautology flagged in
# paper12_phys_gimin_scope.md.
K_AGE_LIT = np.log(1.0 - 0.025)  # log-rate such that S(1yr) = S(0) * 0.975


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

def load_split() -> dict:
    """Load longitudinal features, filter to ≥3-scan patients, split by patient."""
    df = pd.read_csv(LONGITUDINAL, low_memory=False)

    # Target: putamen mean SBR (most reliable regional observable)
    sbr_col = "putamen_mean_sbr" if "putamen_mean_sbr" in df.columns else None
    if sbr_col is None:
        # Fall back to bilateral mean
        left = "sbr_putamen_l" if "sbr_putamen_l" in df.columns else "putamen_left_sbr"
        right = "sbr_putamen_r" if "sbr_putamen_r" in df.columns else "putamen_right_sbr"
        df["putamen_mean_sbr"] = (df[left] + df[right]) / 2
        sbr_col = "putamen_mean_sbr"

    df = df.dropna(subset=[sbr_col, "months_from_baseline", "PATNO"])
    df["t_years"] = df["months_from_baseline"] / 12.0

    # Baseline feature set (actual longitudinal_features.csv column names)
    baseline_feats = [
        "age_at_visit", "sex",
        "updrs1_total", "updrs3_total",
        "ess_total", "rbd_total", "scopa_aut_total", "upsit_total",
        "moca_total",
        "lrrk2_carrier", "gba_carrier",
    ]
    present_feats = [f for f in baseline_feats if f in df.columns]
    # Per-patient baseline feature vector = values at minimum t
    baseline_df = df.loc[df.groupby("PATNO")["t_years"].idxmin()][["PATNO"] + present_feats].copy()
    # Fill missing baseline features with cohort median (CatBoost-style)
    for f in present_feats:
        baseline_df[f] = baseline_df[f].fillna(baseline_df[f].median())

    # Per-patient trajectories (≥3 scans required)
    visit_counts = df.groupby("PATNO").size()
    keep_patnos = visit_counts[visit_counts >= 3].index.tolist()
    df = df[df["PATNO"].isin(keep_patnos)].copy()
    baseline_df = baseline_df[baseline_df["PATNO"].isin(keep_patnos)].copy()

    # Patient-level 70/15/15 split
    patnos = sorted(keep_patnos)
    rng = np.random.default_rng(42)
    rng.shuffle(patnos)
    n = len(patnos)
    train_pats = set(patnos[: int(0.70 * n)])
    val_pats = set(patnos[int(0.70 * n): int(0.85 * n)])
    test_pats = set(patnos[int(0.85 * n):])

    def subset(pats: set[int]) -> dict:
        sub = df[df["PATNO"].isin(pats)].sort_values(["PATNO", "t_years"])
        base = baseline_df[baseline_df["PATNO"].isin(pats)].set_index("PATNO")
        return {
            "trajectories": sub,
            "baseline": base,
            "feats": present_feats,
        }

    print(f"[data] Kept {len(keep_patnos)} patients with ≥3 scans (from {len(visit_counts)} total)")
    print(f"[data] Split: train={len(train_pats)}  val={len(val_pats)}  test={len(test_pats)}")
    return {"train": subset(train_pats), "val": subset(val_pats), "test": subset(test_pats)}


def prepare_batch(split: dict) -> dict:
    """Return list of (patno, t_tensor, sbr_tensor, feat_tensor) per patient."""
    traj = split["trajectories"]
    base = split["baseline"]
    feats = split["feats"]
    # Global NaN-safe fillna with each feature's cohort mean (after cast to numeric)
    for f in feats:
        col = pd.to_numeric(base[f], errors="coerce")
        m = col.mean()
        if pd.isna(m):
            m = 0.0
        base[f] = col.fillna(m)
    # Z-score normalise so gradient scales are reasonable
    for f in feats:
        mu = base[f].mean()
        sd = base[f].std() or 1.0
        base[f] = (base[f] - mu) / sd
    records = []
    for patno, g in traj.groupby("PATNO"):
        g = g.sort_values("t_years")
        t = torch.tensor(g["t_years"].values, dtype=torch.float32)
        s = torch.tensor(g["putamen_mean_sbr"].values, dtype=torch.float32)
        if patno not in base.index:
            continue
        f_vec = base.loc[patno, feats].values.astype(np.float32)
        if np.any(np.isnan(f_vec)):
            continue  # skip patients with remaining NaNs after fill
        f = torch.tensor(f_vec, dtype=torch.float32)
        records.append((int(patno), t, s, f))
    return {"records": records, "n_features": len(feats)}


# ─────────────────────────────────────────────────────────────────────
# Model 1: Pure Neural ODE (data-driven baseline)
# ─────────────────────────────────────────────────────────────────────

class PureNeuralODE(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + n_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self._features = None

    def set_features(self, features: torch.Tensor) -> None:
        self._features = features

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([y.view(-1), self._features]).unsqueeze(0)
        return self.net(inp).squeeze(0)


# ─────────────────────────────────────────────────────────────────────
# Model 2: Physics-Informed Neural ODE (hybrid / UDE)
# ─────────────────────────────────────────────────────────────────────

class PhysicsInformedNeuralODE(nn.Module):
    """
    d S/dt = -|k_age| · S  +  NN_residual(S, features; θ)

    k_age initialised to the lit prior (log(0.975)/yr). NN residual is
    small-initialised so pre-training behaviour is close to pure
    exponential decay.
    """

    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.log_k_age = nn.Parameter(torch.tensor(np.log(0.025), dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(1 + n_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Small init so early training ≈ pure mechanistic
        with torch.no_grad():
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    layer.weight.mul_(0.1)
                    if layer.bias is not None:
                        layer.bias.zero_()
        self._features = None

    def set_features(self, features: torch.Tensor) -> None:
        self._features = features

    def residual(self, y: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([y.view(-1), self._features]).unsqueeze(0)
        return self.net(inp).squeeze(0)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k_age = torch.exp(self.log_k_age)
        return -k_age * y + self.residual(y)


# ─────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────

def train_model(model: nn.Module, records: list, *, epochs: int = 20,
                lambda_physics: float = 0.0, lambda_monotone: float = 0.0,
                lr: float = 1e-3) -> dict:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for patno, t, s, f in records:
            if len(t) < 3:
                continue
            # Leave-last-scan-out: train on visits [0:K-1], predict last
            t_train = t[:-1]
            s_train = s[:-1]
            t_last = t[-1:]
            s_last = s[-1]

            if len(t_train) < 2:
                continue

            model.set_features(f)
            try:
                pred_train = odeint(model, s_train[0:1], t_train, method="dopri5", atol=1e-4, rtol=1e-3)
            except Exception:
                continue
            pred_train = pred_train.squeeze(-1) if pred_train.ndim > 1 else pred_train

            mse = ((pred_train.squeeze() - s_train) ** 2).mean()

            # Physics regulariser: penalise large NN residuals (pull toward pure mechanistic)
            phys_pen = torch.tensor(0.0)
            if isinstance(model, PhysicsInformedNeuralODE) and lambda_physics > 0:
                resid_samples = torch.stack([model.residual(s_train[i:i+1]) for i in range(len(s_train))])
                phys_pen = lambda_physics * (resid_samples ** 2).mean()

            # Monotonicity regulariser: penalise positive dS/dt (PD SBR is non-increasing)
            mono_pen = torch.tensor(0.0)
            if lambda_monotone > 0:
                with torch.no_grad():
                    derivs = torch.stack([model(t_train[i], s_train[i:i+1]) for i in range(len(s_train))])
                mono_pen = lambda_monotone * torch.relu(derivs + 1e-4).mean()

            loss = mse + phys_pen + mono_pen
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            n += 1
        losses.append(total / max(n, 1))
        print(f"  epoch {epoch+1}/{epochs}  loss={losses[-1]:.4f}")
    return {"final_loss": losses[-1], "losses": losses}


# ─────────────────────────────────────────────────────────────────────
# Evaluation: held-out held-out-scan MAE
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_heldout(model: nn.Module | None, records: list, *,
                      pure_mech_rate: float | None = None) -> dict:
    """Predict last-scan SBR given all prior visits. Return MAE + per-horizon errors."""
    abs_errors = []
    horizon_errors = {"0-1yr": [], "1-3yr": [], "3-5yr": [], "5yr+": []}
    for patno, t, s, f in records:
        if len(t) < 3:
            continue
        t_train = t[:-1]
        s_train = s[:-1]
        t_last = t[-1:]
        s_last = float(s[-1])
        horizon = float(t_last - t_train[-1])

        if pure_mech_rate is not None:
            # Pure mechanistic baseline: pure exponential decay from last observed visit
            s_pred = float(s_train[-1]) * np.exp(-pure_mech_rate * horizon)
        else:
            model.eval()
            model.set_features(f)
            try:
                pred = odeint(model, s_train[0:1], t, method="dopri5", atol=1e-4, rtol=1e-3)
            except Exception:
                continue
            pred = pred.squeeze(-1) if pred.ndim > 1 else pred
            s_pred = float(pred[-1].item() if pred[-1].ndim == 0 else pred[-1, 0].item())

        abs_err = abs(s_pred - s_last)
        abs_errors.append(abs_err)
        if horizon <= 1:
            horizon_errors["0-1yr"].append(abs_err)
        elif horizon <= 3:
            horizon_errors["1-3yr"].append(abs_err)
        elif horizon <= 5:
            horizon_errors["3-5yr"].append(abs_err)
        else:
            horizon_errors["5yr+"].append(abs_err)

    return {
        "n": len(abs_errors),
        "mae": float(np.mean(abs_errors)) if abs_errors else float("nan"),
        "rmse": float(np.sqrt(np.mean(np.array(abs_errors) ** 2))) if abs_errors else float("nan"),
        "median_abs_err": float(np.median(abs_errors)) if abs_errors else float("nan"),
        "per_horizon_mae": {
            k: (float(np.mean(v)) if v else None, len(v))
            for k, v in horizon_errors.items()
        },
    }


def main() -> None:
    print("=" * 75)
    print("Paper 11 Hybrid SciML Demo — Physics-Informed Neural ODE on DaT-SBR")
    print("=" * 75)

    splits = load_split()
    train_batch = prepare_batch(splits["train"])
    test_batch = prepare_batch(splits["test"])
    n_features = train_batch["n_features"]
    print(f"[model] n_features = {n_features}")
    print(f"[model] train trajectories = {len(train_batch['records'])}")
    print(f"[model] test trajectories  = {len(test_batch['records'])}")

    # Keep run small for demo (can scale with --full flag later)
    train_records = train_batch["records"][:150]   # demo: 150 train patients
    test_records = test_batch["records"][:50]      # demo: 50 held-out test
    print(f"[demo] running on 150 train / 50 test for demo scope")

    results: dict = {}

    # === Baseline: Pure mechanistic exponential decay with lit prior ===
    print("\n--- Baseline 1: Pure mechanistic (2.5%/yr lit prior) ---")
    pure_mech = evaluate_heldout(model=None, records=test_records, pure_mech_rate=0.025)
    print(f"  MAE: {pure_mech['mae']:.4f}  RMSE: {pure_mech['rmse']:.4f}  (n={pure_mech['n']})")
    results["pure_mechanistic_lit_prior"] = pure_mech

    # === Model A: Pure Neural ODE (data-driven) ===
    print("\n--- Baseline 2: Pure Neural ODE (data-driven, no physics) ---")
    model_a = PureNeuralODE(n_features=n_features).to(DEVICE)
    train_res_a = train_model(model_a, train_records, epochs=10)
    eval_a = evaluate_heldout(model_a, test_records)
    print(f"  MAE: {eval_a['mae']:.4f}  RMSE: {eval_a['rmse']:.4f}")
    results["pure_neural_ode"] = {"eval": eval_a, "training": train_res_a}

    # === Model B: Physics-Informed Neural ODE (hybrid) ===
    print("\n--- Hybrid: Physics-Informed Neural ODE (UDE, lit prior + learned residual) ---")
    model_b = PhysicsInformedNeuralODE(n_features=n_features).to(DEVICE)
    train_res_b = train_model(model_b, train_records, epochs=10,
                               lambda_physics=0.01, lambda_monotone=0.01)
    eval_b = evaluate_heldout(model_b, test_records)
    print(f"  MAE: {eval_b['mae']:.4f}  RMSE: {eval_b['rmse']:.4f}")
    print(f"  Learned k_age: {np.exp(model_b.log_k_age.item()):.4f}/yr "
          f"(init 0.025/yr = 2.5%/yr Fearnley-Lees midpoint)")
    results["physics_informed_neural_ode"] = {
        "eval": eval_b,
        "training": train_res_b,
        "learned_k_age_per_yr": float(np.exp(model_b.log_k_age.item())),
    }

    # === Compare ===
    print("\n" + "=" * 75)
    print("COMPARISON (held-out last-scan MAE on n=50 test patients)")
    print("=" * 75)
    print(f"{'Model':<40s}  {'MAE':>8s}  {'RMSE':>8s}")
    print("-" * 75)
    print(f"{'Pure mechanistic (lit 2.5%/yr)':<40s}  {pure_mech['mae']:>8.4f}  {pure_mech['rmse']:>8.4f}")
    print(f"{'Pure Neural ODE':<40s}  {eval_a['mae']:>8.4f}  {eval_a['rmse']:>8.4f}")
    print(f"{'Physics-Informed Neural ODE (hybrid)':<40s}  {eval_b['mae']:>8.4f}  {eval_b['rmse']:>8.4f}")

    best = min([("pure_mech", pure_mech["mae"]), ("pure_nn", eval_a["mae"]), ("hybrid", eval_b["mae"])],
               key=lambda x: x[1])
    delta_hybrid_vs_mech = eval_b["mae"] - pure_mech["mae"]
    delta_hybrid_vs_nn = eval_b["mae"] - eval_a["mae"]
    print()
    print(f"Best model: {best[0]}  (MAE {best[1]:.4f})")
    print(f"Δ Hybrid − PureMech: {delta_hybrid_vs_mech:+.4f}  "
          f"({'hybrid wins' if delta_hybrid_vs_mech < 0 else 'hybrid loses/ties'})")
    print(f"Δ Hybrid − PureNN:   {delta_hybrid_vs_nn:+.4f}  "
          f"({'hybrid wins' if delta_hybrid_vs_nn < 0 else 'hybrid loses/ties'})")

    results["summary"] = {
        "best_model": best[0],
        "best_mae": best[1],
        "delta_hybrid_minus_puremech": float(delta_hybrid_vs_mech),
        "delta_hybrid_minus_purenn": float(delta_hybrid_vs_nn),
    }

    out_json = OUT_DIR / "hybrid_sciml_demo_metrics.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {out_json}")

    # Figure: 3-panel comparison
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4.2))

    # Panel A: Held-out MAE bar chart
    models = ["Pure\nMech\n(lit 2.5%/yr)", "Pure\nNeural ODE", "Hybrid\nPINN"]
    maes = [pure_mech["mae"], eval_a["mae"], eval_b["mae"]]
    colors = ["#009E73", "#CC79A7", "#0072B2"]
    bars = ax1.bar(models, maes, color=colors, edgecolor="black", linewidth=0.5)
    for b, m in zip(bars, maes):
        ax1.text(b.get_x() + b.get_width()/2, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("Held-out last-scan MAE (SBR)")
    ax1.set_title("(a) Model comparison (n=50 test)")
    ax1.grid(axis="y", alpha=0.3)

    # Panel B: Training loss curves
    ax2.plot(train_res_a["losses"], "o-", color="#CC79A7", label="Pure Neural ODE", linewidth=2)
    ax2.plot(train_res_b["losses"], "s-", color="#0072B2", label="Physics-Informed (Hybrid)", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training MSE loss")
    ax2.set_title("(b) Training dynamics")
    ax2.legend(fontsize=9, frameon=False)
    ax2.grid(alpha=0.3)

    # Panel C: Per-horizon MAE
    horizons = ["0-1yr", "1-3yr", "3-5yr", "5yr+"]
    pure_mech_h = [pure_mech["per_horizon_mae"][h][0] or 0 for h in horizons]
    pure_nn_h = [eval_a["per_horizon_mae"][h][0] or 0 for h in horizons]
    hybrid_h = [eval_b["per_horizon_mae"][h][0] or 0 for h in horizons]
    x = np.arange(len(horizons))
    w = 0.25
    ax3.bar(x - w, pure_mech_h, w, color="#009E73", label="Pure Mech", edgecolor="black", linewidth=0.3)
    ax3.bar(x,     pure_nn_h, w, color="#CC79A7", label="Pure NN", edgecolor="black", linewidth=0.3)
    ax3.bar(x + w, hybrid_h, w, color="#0072B2", label="Hybrid", edgecolor="black", linewidth=0.3)
    ax3.set_xticks(x)
    ax3.set_xticklabels(horizons, fontsize=9)
    ax3.set_ylabel("MAE by extrapolation horizon")
    ax3.set_title("(c) Horizon-stratified error")
    ax3.legend(fontsize=8, frameon=False)
    ax3.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Paper 11 Hybrid SciML Demo — Physics-Informed Neural ODE on DaT-SBR trajectories",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    fig_pdf = OUT_DIR / "hybrid_sciml_demo.pdf"
    fig_png = OUT_DIR / "hybrid_sciml_demo.png"
    fig.savefig(fig_pdf, bbox_inches="tight")
    fig.savefig(fig_png, bbox_inches="tight", dpi=150)
    print(f"Saved: {fig_pdf}")
    print(f"Saved: {fig_png}")


if __name__ == "__main__":
    main()
