#!/usr/bin/env python3
"""MCAR held-out validation of per-visit GIMIN σ.

Pillar 6 of the L1 integration. Takes the per-visit GIMIN pipeline and
artificially masks 50% of OBSERVED DaT-SBR visits as held-out, then
compares the imputed (μ, σ) against the held-out ground truth. This
directly answers two open questions from Pillar 5:

  1. Is the CAUDATE imputed-vs-observed 1.05-unit divergence selection
     bias (missing-visit cohort is systematically later/more progressed)
     or feature-specific miscalibration? If imputed mean on held-out
     OBSERVED CAUDATE matches observed, it's selection bias. If the
     per-visit GIMIN CAUDATE σ is similarly mis-aligned, it's feature
     miscalibration.

  2. Is the per-visit σ calibrated (95% coverage) under MCAR? Paper 2
     §V.E fit temperature scaler targets γ=0.90 on baseline masking.
     Per-visit inference uses the SAME temperature scalers; if they
     transfer, the 95% CI should cover ~95% of held-out truths.

Protocol:
  1. Read P3 longitudinal features (16,699 visits).
  2. Identify visits where ALL 4 DaT-SBR values are observed (2,681).
  3. Randomly mask 50% of these (seed=42) → 1,340 held-out visits.
  4. Run per-visit GIMIN inference on the modified feature matrix.
  5. For each held-out visit, compare imputed (μ, σ) vs ground truth.
  6. Report per-feature bias, MAE, 95% coverage, MAE/median(σ) ratio.

Outputs:
  outputs/mechanistic_twin/l1_gimin_bridge/per_visit_mcar_holdout.json
    (full per-feature metrics)
  outputs/mechanistic_twin/l1_gimin_bridge/per_visit_mcar_holdout.parquet
    (per-heldout-visit records with observed + imputed + residuals)
  outputs/mechanistic_twin/l1_gimin_bridge/fig_per_visit_mcar.{pdf,png}
    (scatter + calibration diagnostic)

Usage:
  .venv/bin/python scripts/mechanistic_twin/run_gimin_per_visit_mcar_holdout.py
  .venv/bin/python scripts/mechanistic_twin/run_gimin_per_visit_mcar_holdout.py --holdout-frac 0.5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "GIMImpN_imputation"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "paper6"))

P3_FEATURES = PROJECT_ROOT / "data" / "07_paper3_features" / "longitudinal_features.csv"
GIMIN_COHORT = PROJECT_ROOT / "GIMImpN_imputation" / "outputs" / "ppmi_full_cohort.parquet"
GIMIN_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/paper2_benchmark/runs/cal_retune_lambda0.1_warmup0/checkpoints/frac0.1_run0_GIMIN_StageDecoderOnly.pt"
)
OUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "l1_gimin_bridge"

P3_DAT_COLS = ["caudate_l_sbr", "caudate_r_sbr", "putamen_l_sbr", "putamen_r_sbr"]
GIMIN_DAT_COLS = ["CAUDATE_L_SBR", "CAUDATE_R_SBR", "PUTAMEN_L_SBR", "PUTAMEN_R_SBR"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mc-samples", type=int, default=20)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse the per-visit pipeline's feature assembly
    from run_gimin_per_visit import (  # type: ignore[import]
        build_per_visit_matrix,
        P3_TO_GIMIN,
    )

    print("Loading inputs...")
    p3 = pd.read_csv(P3_FEATURES, low_memory=False)
    baseline = pd.read_parquet(GIMIN_COHORT)
    print(f"  P3 longitudinal: {p3.shape}")

    # ────────────────────────────────────────────────────────────────────
    # Step 1: Identify visits with ALL 4 DaT-SBR observed → candidate holdouts
    # ────────────────────────────────────────────────────────────────────
    obs_mask = p3[P3_DAT_COLS].notna().all(axis=1)
    candidate_indices = np.where(obs_mask.values)[0]
    print(f"  Visits with ALL DaT observed: {len(candidate_indices)}")

    # Randomly sample holdout_frac of these
    rng = np.random.default_rng(args.seed)
    n_holdout = int(len(candidate_indices) * args.holdout_frac)
    holdout_idx_set = set(rng.choice(candidate_indices, size=n_holdout, replace=False).tolist())
    print(f"  Holdout count ({args.holdout_frac*100:.0f}% MCAR): {len(holdout_idx_set)}")

    # Preserve ground truth
    gt = p3.loc[sorted(holdout_idx_set), ["PATNO", "EVENT_ID", "months_from_baseline"] + P3_DAT_COLS].copy()

    # Mask held-out DaT in a copy of p3 for inference
    p3_masked = p3.copy()
    for idx in holdout_idx_set:
        for col in P3_DAT_COLS:
            p3_masked.loc[idx, col] = np.nan

    # ────────────────────────────────────────────────────────────────────
    # Step 2: Build per-visit feature matrix on the masked data
    # ────────────────────────────────────────────────────────────────────
    print("\nSetting up GIMIN stack + building masked per-visit matrix...")
    from argparse import Namespace
    from unified_pipeline_demo_v2 import setup_gimin_stack  # noqa: E402

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    stack = setup_gimin_stack(Namespace(gimin_checkpoint=GIMIN_CHECKPOINT), device)
    gimin_features = list(stack["feature_names"])

    X, mask = build_per_visit_matrix(p3_masked, baseline, gimin_features)
    print(f"  Matrix: {X.shape}, observed rate: {100*mask.mean():.1f}%")

    # Stages per visit
    stage_map = {0.0: 0, 1.0: 1, 2.5: 2, 3.0: 3, 4.0: 4}
    per_visit_stages = p3["nsd_stage_numeric"].map(stage_map).fillna(5).astype(int).values

    # ────────────────────────────────────────────────────────────────────
    # Step 3: Inference — clone of per-visit runner logic, but with masked input
    # ────────────────────────────────────────────────────────────────────
    import torch as T
    from gimin.config import GIMINConfig
    from gimin.data.scaler import build_scaler_from_config
    from giman_pipeline.imputation.stage_conditioned_gimin import StageConditionedGIMIN
    from giman_pipeline.imputation.stage_graph_builder import StageAwareGraphBuilder
    from run_paper2_experiments import MODALITY_DIMS, load_data
    from unified_pipeline_demo_v2 import _reload_training_cohort_patnos

    cfg = GIMINConfig()
    scaler = build_scaler_from_config(cfg)

    train_X, train_mask, train_stages, _ = load_data()
    train_X_t = T.tensor(train_X, dtype=T.float32)
    train_mask_t = T.tensor(train_mask, dtype=T.float32)
    scaler.fit(train_X_t, train_mask_t)
    train_X_norm_t = scaler.transform(train_X_t, train_mask_t)
    builder = StageAwareGraphBuilder(k_neighbors=15, min_overlap=3, stage_affinity_beta=0.3)
    graph = builder.build_full_graph(
        train_X_norm_t.numpy() * train_mask.astype(np.float32),
        train_mask,
        stages=train_stages,
    )

    train_patnos = _reload_training_cohort_patnos()
    patno_to_train_idx = {int(p): i for i, p in enumerate(train_patnos)}

    model = StageConditionedGIMIN(
        modality_dims=MODALITY_DIMS,
        embed_dim=64, num_gnn_layers=3, num_heads=4,
        mc_dropout=0.1, num_stages=6, stage_embed_dim=16,
    )
    state = T.load(str(GIMIN_CHECKPOINT), map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()
    model.train()  # activate dropout for MC sampling

    edge_index = graph["edge_index"].to(device)
    edge_weight = graph["edge_weight"].to(device)
    overlap_frac = graph["overlap_frac"].to(device)
    stages_t = T.tensor(train_stages, dtype=T.long, device=device)
    train_X_norm_d = train_X_norm_t.to(device)
    train_mask_d = train_mask_t.to(device)

    n_visits = len(p3_masked)
    imputed_mean = np.zeros((n_visits, len(gimin_features)), dtype=np.float32)
    imputed_std = np.zeros((n_visits, len(gimin_features)), dtype=np.float32)

    patient_idxs = []
    for v in range(n_visits):
        patno = int(p3.iloc[v]["PATNO"])
        patient_idxs.append(patno_to_train_idx.get(patno, -1))

    from collections import defaultdict
    pat_visit_map = defaultdict(list)
    for v in range(n_visits):
        if patient_idxs[v] >= 0:
            pat_visit_map[int(p3.iloc[v]["PATNO"])].append(v)
    max_visits_per_pat = max(len(lst) for lst in pat_visit_map.values())

    passes = [[] for _ in range(max_visits_per_pat)]
    for vlist in pat_visit_map.values():
        for k, v in enumerate(vlist):
            passes[k].append(v)

    import time
    t0 = time.time()
    total_calls = 0
    for pass_idx, pass_visits in enumerate(passes):
        B = min(args.batch_size, len(pass_visits))
        if B == 0:
            continue
        for sub_start in range(0, len(pass_visits), B):
            batch = pass_visits[sub_start : sub_start + B]
            total_calls += 1

            X_cur = train_X_norm_d.clone()
            mask_cur = train_mask_d.clone()
            stages_cur = stages_t.clone()

            valid_visits = []
            train_idxs = []
            for v in batch:
                t_idx = patient_idxs[v]
                if t_idx < 0:
                    continue
                vx = T.tensor(X[v:v+1], dtype=T.float32)
                vm = T.tensor(mask[v:v+1], dtype=T.float32)
                vx_norm = scaler.transform(vx, vm).to(device)
                X_cur[t_idx] = vx_norm[0]
                mask_cur[t_idx] = T.tensor(mask[v], dtype=T.float32, device=device)
                stages_cur[t_idx] = int(per_visit_stages[v])
                valid_visits.append(v)
                train_idxs.append(t_idx)

            if not valid_visits:
                continue

            mc_outs = []
            with T.no_grad():
                for _ in range(args.mc_samples):
                    out = model(X_cur, mask_cur, edge_index, edge_weight,
                                overlap_frac, stage_ids=stages_cur)
                    mc_outs.append(out["imputed_mean"].unsqueeze(0))
            mc_stack = T.cat(mc_outs, dim=0)
            mu_norm = mc_stack.mean(dim=0)
            total_var_norm = mc_stack.var(dim=0) + T.exp(out["imputed_log_var"]).clamp(max=10.0)

            mu_orig = scaler.inverse_transform(mu_norm.cpu(), mask=None).numpy()
            var_orig = scaler.inverse_transform_variance(
                total_var_norm.cpu(), mean_normalized=mu_norm.cpu(), mask=None,
            ).numpy()
            sigma_orig = np.sqrt(np.clip(var_orig, 0.0, None))

            temps = stack["temperatures"]
            sigma_calibrated = sigma_orig * temps[None, :]

            for local_i, v in enumerate(valid_visits):
                t_idx = train_idxs[local_i]
                imputed_mean[v] = mu_orig[t_idx]
                imputed_std[v] = sigma_calibrated[t_idx]

    print(f"  Inference done: {total_calls} forward calls in {time.time()-t0:.0f}s")

    # ────────────────────────────────────────────────────────────────────
    # Step 4: Compare imputed vs ground truth on the 1,340 held-out visits
    # ────────────────────────────────────────────────────────────────────
    rows = []
    for holdout_v in sorted(holdout_idx_set):
        if patient_idxs[holdout_v] < 0:
            continue  # patient not in training cohort
        gt_row = p3.iloc[holdout_v]
        rec = {
            "PATNO": int(gt_row["PATNO"]),
            "EVENT_ID": gt_row.get("EVENT_ID", ""),
            "months_from_baseline": float(gt_row.get("months_from_baseline", np.nan)),
        }
        for p3_col, gimin_col in zip(P3_DAT_COLS, GIMIN_DAT_COLS):
            j = gimin_features.index(gimin_col)
            truth = float(gt_row[p3_col])
            mu = float(imputed_mean[holdout_v, j])
            sig = float(imputed_std[holdout_v, j])
            lo = mu - 1.96 * sig
            hi = mu + 1.96 * sig
            rec[f"{gimin_col}_truth"] = truth
            rec[f"{gimin_col}_imputed_mean"] = mu
            rec[f"{gimin_col}_imputed_std"] = sig
            rec[f"{gimin_col}_residual"] = mu - truth
            rec[f"{gimin_col}_covered"] = bool(lo <= truth <= hi)
        rows.append(rec)

    val_df = pd.DataFrame(rows)
    out_parquet = OUT_DIR / "per_visit_mcar_holdout.parquet"
    val_df.to_parquet(out_parquet, index=False)
    print(f"\n  Saved per-heldout-visit records: {out_parquet}  n={len(val_df)}")

    # ────────────────────────────────────────────────────────────────────
    # Step 5: Per-feature calibration summary
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"MCAR HELD-OUT VALIDATION — n={len(val_df)} visits, {args.holdout_frac*100:.0f}% masked")
    print("=" * 78)

    summary = {"config": vars(args), "n_holdout": len(val_df), "per_feature": {}}
    for feat in GIMIN_DAT_COLS:
        truth_col = f"{feat}_truth"
        mu_col = f"{feat}_imputed_mean"
        sig_col = f"{feat}_imputed_std"
        cov_col = f"{feat}_covered"
        resid = val_df[mu_col].values - val_df[truth_col].values
        sig = val_df[sig_col].values
        mae = float(np.mean(np.abs(resid)))
        bias = float(np.mean(resid))
        rmse = float(np.sqrt(np.mean(resid**2)))
        pearson = float(np.corrcoef(val_df[mu_col], val_df[truth_col])[0, 1])
        coverage = float(val_df[cov_col].mean())
        sig_median = float(np.median(sig))
        ratio = mae / sig_median if sig_median > 0 else float("nan")
        summary["per_feature"][feat] = {
            "bias_mean_minus_truth": bias,
            "mae": mae,
            "rmse": rmse,
            "pearson_r": pearson,
            "coverage_95ci": coverage,
            "median_sigma": sig_median,
            "mae_over_median_sigma": ratio,
        }
        print(f"\n{feat}:")
        print(f"  Bias (imputed − truth): {bias:+.4f}")
        print(f"  MAE:                    {mae:.4f}")
        print(f"  RMSE:                   {rmse:.4f}")
        print(f"  Pearson r:              {pearson:.3f}")
        print(f"  Median σ:               {sig_median:.4f}")
        print(f"  95% CI coverage:        {coverage:.3f}  "
              f"({'under' if coverage < 0.90 else 'well-calibrated' if coverage < 0.98 else 'over'})")
        print(f"  MAE / median(σ):        {ratio:.2f}  "
              f"({'under-confident' if ratio > 1.5 else 'over-confident' if ratio < 0.6 else 'well-calibrated'})")

    out_json = OUT_DIR / "per_visit_mcar_holdout.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {out_json}")

    # ────────────────────────────────────────────────────────────────────
    # Step 6: Diagnostic figure — scatter + residuals
    # ────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for col_idx, feat in enumerate(GIMIN_DAT_COLS):
        truth = val_df[f"{feat}_truth"].values
        mu = val_df[f"{feat}_imputed_mean"].values
        sig = val_df[f"{feat}_imputed_std"].values

        ax = axes[0, col_idx]
        ax.errorbar(truth, mu, yerr=1.96 * sig, fmt="o", ms=3, alpha=0.3,
                    elinewidth=0.4, color="#4b6cb7")
        mn = min(truth.min(), mu.min())
        mx = max(truth.max(), mu.max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel(f"Observed {feat}")
        ax.set_ylabel("Imputed μ ± 95% CI")
        ax.set_title(f"{feat}\nr={np.corrcoef(truth, mu)[0,1]:.3f}")

        ax = axes[1, col_idx]
        resid = mu - truth
        ax.hist(resid, bins=50, color="#d62728", alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.axvline(resid.mean(), color="blue", ls="-", lw=0.8,
                   label=f"bias={resid.mean():+.3f}")
        ax.set_xlabel("Residual (imputed − truth)")
        ax.set_ylabel("n visits")
        ax.set_title(f"MAE={np.mean(np.abs(resid)):.3f}")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"MCAR held-out validation: per-visit GIMIN on n={len(val_df)} DaT-SBR visits",
        y=1.02,
    )
    fig.tight_layout()
    fig_path = OUT_DIR / "fig_per_visit_mcar.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
