#!/usr/bin/env python3
"""Per-visit GIMIN inference on the P3 1,900-patient longitudinal cohort.

For each of the 16,699 visits, build a 33-feature vector and run GIMIN
StageDecoderOnly inference. Features are drawn from three sources:

  1. Per-visit values from data/07_paper3_features/longitudinal_features.csv
     (available for ~18 of 33: SEX, AGE_AT_VISIT, NP3TOT (updrs3_total),
     NHY (hy_stage), MCATOT (moca_total), LRRK2/GBA/SNCA/APOE4 carriers,
     CAUDATE_L/R_SBR, PUTAMEN_L/R_SBR, CAUDATE_ASYMMETRY, UPSIT/RBD/
     SCOPA/ESS totals)

  2. Baseline-held values from GIMImpN_imputation/outputs/ppmi_full_cohort.parquet
     for features that P3 doesn't track per visit (CAUDATE/PUTAMEN/
     HIPPOCAMPUS_L/R_VOL, CSF_biomarkers, CTH, GENETIC_RISK_SCORE,
     PIGD_SCORE, TREMOR_SCORE). Held constant across visits under the
     "slow-changing" approximation.

  3. Computed: PUTAMEN_ASYMMETRY derived from bilateral SBR when observed.

Output (long format, one row per visit):
  outputs/mechanistic_twin/l1_gimin_bridge/gimin_per_visit_dat_sbr.parquet
    columns:
      PATNO, EVENT_ID, months_from_baseline,
      CAUDATE_L/R_SBR_{obs, imputed_mean, imputed_std, is_observed}
      PUTAMEN_L/R_SBR_{obs, imputed_mean, imputed_std, is_observed}
      CAUDATE/PUTAMEN_MEAN_SBR_{obs, imputed_mean, imputed_std, is_observed}

MNAR caveat: σ calibration was validated on MCAR masking (Paper 2 §V.E).
Per-visit visits where DaT is randomly missed (scheduling/quality-control)
are MCAR-like. Patients whose WHOLE TRAJECTORY has no DaT (667 of 1,900)
are MNAR and produce less-reliable σ (external validation showed 4-14×
under-confidence on baseline MNAR patients). We flag these with a
column `mnar_flag` so downstream consumers can apply a σ floor.

Usage:
  .venv/bin/python scripts/mechanistic_twin/run_gimin_per_visit.py
  .venv/bin/python scripts/mechanistic_twin/run_gimin_per_visit.py --n-visits 500  # smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

SBR_SENSOR_SIGMA = 0.08

DAT_FEATURES = [
    "CAUDATE_L_SBR", "CAUDATE_R_SBR",
    "PUTAMEN_L_SBR", "PUTAMEN_R_SBR",
]


# P3 column name → GIMIN feature name
P3_TO_GIMIN = {
    "sex": "SEX",
    "age_at_visit": "AGE_AT_VISIT",
    "updrs3_total": "NP3TOT",
    "hy_stage": "NHY",
    "moca_total": "MCATOT",
    "lrrk2_carrier": "LRRK2",
    "gba_carrier": "GBA",
    "apoe_e4": "APOE_E4",
    "snca_carrier": "SNCA",
    "upsit_total": "UPSIT_TOTAL",
    "rbd_total": "RBD_TOTAL",
    "scopa_aut_total": "SCOPA_AUT_TOTAL",
    "ess_total": "ESS_TOTAL",
    "caudate_l_sbr": "CAUDATE_L_SBR",
    "caudate_r_sbr": "CAUDATE_R_SBR",
    "putamen_l_sbr": "PUTAMEN_L_SBR",
    "putamen_r_sbr": "PUTAMEN_R_SBR",
    "caudate_asymmetry": "CAUDATE_ASYMMETRY",
}


def build_per_visit_matrix(p3: pd.DataFrame, baseline: pd.DataFrame, gimin_features: list[str]):
    """Build (n_visits, 33) per-visit feature matrix + mask + stages."""
    n_visits = len(p3)
    n_feat = len(gimin_features)
    X = np.full((n_visits, n_feat), np.nan, dtype=np.float32)

    # Step 1: fill per-visit values from P3
    for p3_col, gimin_col in P3_TO_GIMIN.items():
        if p3_col not in p3.columns or gimin_col not in gimin_features:
            continue
        j = gimin_features.index(gimin_col)
        vals = p3[p3_col].values.astype(np.float32)
        X[:, j] = vals

    # Step 2: compute PUTAMEN_ASYMMETRY from bilateral SBR where observed
    if "PUTAMEN_ASYMMETRY" in gimin_features:
        j = gimin_features.index("PUTAMEN_ASYMMETRY")
        pl = X[:, gimin_features.index("PUTAMEN_L_SBR")]
        pr = X[:, gimin_features.index("PUTAMEN_R_SBR")]
        with np.errstate(divide="ignore", invalid="ignore"):
            asy = np.abs(pl - pr) / (0.5 * (pl + pr))
        X[:, j] = asy  # NaN where inputs NaN

    # Step 3: baseline-held fill for features NOT in P3
    # baseline is indexed by PATNO
    p3_patnos = p3["PATNO"].values
    # Features we've already filled from P3 or computed
    p3_gimin = set(P3_TO_GIMIN.values()) | {"PUTAMEN_ASYMMETRY"}
    static_features = [f for f in gimin_features if f not in p3_gimin]

    # Build a lookup: for each visit, pull baseline-held values
    baseline_lookup = baseline.reindex(p3_patnos)  # (n_visits, n_baseline_cols)
    for fname in static_features:
        if fname not in baseline.columns:
            continue
        j = gimin_features.index(fname)
        X[:, j] = baseline_lookup[fname].values.astype(np.float32)

    # Mask: 1 where observed, 0 where NaN
    mask = ~np.isnan(X)
    X = np.where(mask, X, 0.0)  # 0-fill NaN so tensor ops don't propagate NaN

    return X.astype(np.float32), mask.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-visits", type=int, default=None,
                        help="Cap for smoke test; default = all 16,699")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for GIMIN forward pass")
    parser.add_argument("--mc-samples", type=int, default=20,
                        help="MC-dropout samples (default 20)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading inputs...")
    p3 = pd.read_csv(P3_FEATURES, low_memory=False)
    baseline = pd.read_parquet(GIMIN_COHORT)
    print(f"  P3 longitudinal: {p3.shape}")
    print(f"  GIMIN baseline cohort: {baseline.shape}")

    if args.n_visits:
        p3 = p3.head(args.n_visits).copy()
        print(f"  Smoke test: using first {args.n_visits} visits")

    # Set up GIMIN stack
    print("\nSetting up GIMIN stack via P6 v2 setup_gimin_stack()...")
    from argparse import Namespace
    from unified_pipeline_demo_v2 import setup_gimin_stack  # noqa: E402
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    stack = setup_gimin_stack(Namespace(gimin_checkpoint=GIMIN_CHECKPOINT), device)
    gimin_features = list(stack["feature_names"])
    print(f"  GIMIN features: {len(gimin_features)}")

    # Build per-visit feature matrix
    print("\nBuilding per-visit feature matrix (11 P3 per-visit + 22 baseline-held)...")
    X, mask = build_per_visit_matrix(p3, baseline, gimin_features)
    print(f"  Matrix shape: {X.shape}")
    print(f"  Overall observed rate: {100 * mask.mean():.1f}%")

    # Per-visit cohort-identity stages (needed for StageConditionedGIMIN)
    # Use the P3 nsd_stage_numeric mapped to {0,1,2,3,4,5} for graph affinity
    stage_map = {0.0: 0, 1.0: 1, 2.5: 2, 3.0: 3, 4.0: 4}  # 2.5 = Stage 2B
    per_visit_stages = p3["nsd_stage_numeric"].map(stage_map).fillna(5).astype(int).values

    # For per-visit inference, we need to run GIMIN on each visit. GIMIN's graph
    # is built on the TRAINING cohort (2,197 baseline rows). For per-visit inference,
    # we REPLACE one baseline row at a time with the visit's features.
    # This uses the patient's baseline kNN neighbors to inform per-visit imputation.

    import torch as T
    from gimin.config import GIMINConfig
    from gimin.data.scaler import build_scaler_from_config
    from giman_pipeline.imputation.stage_conditioned_gimin import StageConditionedGIMIN
    from giman_pipeline.imputation.stage_graph_builder import StageAwareGraphBuilder

    cfg = GIMINConfig()
    scaler = build_scaler_from_config(cfg)

    # Reload the staged training cohort + graph (from setup_gimin_stack)
    from run_paper2_experiments import MODALITY_DIMS, load_data  # noqa: E402

    train_X, train_mask, train_stages, _ = load_data()
    train_X_t = T.tensor(train_X, dtype=T.float32)
    train_mask_t = T.tensor(train_mask, dtype=T.float32)
    scaler.fit(train_X_t, train_mask_t)

    # Graph built on normalized observed training features
    train_X_norm_t = scaler.transform(train_X_t, train_mask_t)
    builder = StageAwareGraphBuilder(k_neighbors=15, min_overlap=3, stage_affinity_beta=0.3)
    graph = builder.build_full_graph(
        train_X_norm_t.numpy() * train_mask.astype(np.float32),
        train_mask,
        stages=train_stages,
    )

    # Build PATNO → training-row-index lookup
    from unified_pipeline_demo_v2 import _reload_training_cohort_patnos  # noqa: E402
    train_patnos = _reload_training_cohort_patnos()
    patno_to_train_idx = {int(p): i for i, p in enumerate(train_patnos)}

    # Load GIMIN model
    model = StageConditionedGIMIN(
        modality_dims=MODALITY_DIMS,
        embed_dim=64, num_gnn_layers=3, num_heads=4,
        mc_dropout=0.1, num_stages=6, stage_embed_dim=16,
    )
    state = T.load(str(GIMIN_CHECKPOINT), map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"  Loaded GIMIN checkpoint.")
    model.train()  # Keep dropout active for MC-dropout inference

    # Transfer graph to device
    edge_index = graph["edge_index"].to(device)
    edge_weight = graph["edge_weight"].to(device)
    overlap_frac = graph["overlap_frac"].to(device)
    stages_t = T.tensor(train_stages, dtype=T.long, device=device)

    # Pre-compute training-cohort scaler-transformed values on device
    train_X_norm_d = train_X_norm_t.to(device)
    train_mask_d = train_mask_t.to(device)

    # Per-visit inference: batch multiple visits by swapping multiple rows
    n_visits = len(p3)
    print(f"\nRunning per-visit inference on {n_visits} visits (batch={args.batch_size}, MC={args.mc_samples})...")

    # Storage for per-visit results
    imputed_mean = np.zeros((n_visits, len(gimin_features)), dtype=np.float32)
    imputed_std = np.zeros((n_visits, len(gimin_features)), dtype=np.float32)
    patient_idxs_for_visits = []
    missing_in_graph = 0

    # For each visit, find which training-row to swap. If patient has no
    # baseline training row (not in the 2,197 staged cohort), skip.
    for v in range(n_visits):
        patno = int(p3.iloc[v]["PATNO"])
        if patno in patno_to_train_idx:
            patient_idxs_for_visits.append(patno_to_train_idx[patno])
        else:
            patient_idxs_for_visits.append(-1)
            missing_in_graph += 1

    print(f"  Visits with no training-graph node (skipped): {missing_in_graph}")

    import time
    t0 = time.time()

    # CORRECTNESS: ensure each batch has AT MOST ONE visit per PATNO.
    # Same-patient multi-visits would overwrite each other's swap slot in the
    # training matrix. Grouping per-patient visits into SEPARATE batches
    # guarantees each forward pass produces independent imputations.
    #
    # Strategy: sort visits by PATNO, split into "passes" where pass k gets
    # each patient's k-th visit. Within each pass, batch_size visits per
    # forward call.
    from collections import defaultdict
    pat_visit_map: dict[int, list[int]] = defaultdict(list)
    for v in range(n_visits):
        patno = int(p3.iloc[v]["PATNO"])
        if patno in patno_to_train_idx:
            pat_visit_map[patno].append(v)
    max_visits_per_pat = max(len(lst) for lst in pat_visit_map.values())
    print(f"  Max visits per patient: {max_visits_per_pat}")
    print(f"  Organizing {n_visits} visits into {max_visits_per_pat} passes "
          f"(one visit per patient per pass) to ensure correctness...")

    # Build passes: pass k contains the k-th visit of each patient
    passes: list[list[int]] = [[] for _ in range(max_visits_per_pat)]
    for patno, vlist in pat_visit_map.items():
        for k, v in enumerate(vlist):
            passes[k].append(v)

    total_forward_calls = 0
    for pass_idx, pass_visits in enumerate(passes):
        # Sub-batch for memory
        B = min(args.batch_size, len(pass_visits))
        if B == 0:
            continue
        for sub_start in range(0, len(pass_visits), B):
            sub_end = min(sub_start + B, len(pass_visits))
            batch = pass_visits[sub_start:sub_end]
            total_forward_calls += 1

            # Build swap-modified copy of the training matrix
            X_cur = train_X_norm_d.clone()
            mask_cur = train_mask_d.clone()
            stages_cur = stages_t.clone()

            valid_visits = []
            train_idxs = []
            for v in batch:
                t_idx = patient_idxs_for_visits[v]
                if t_idx < 0:
                    continue
                # Normalize visit's feature vector
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

            # MC-dropout inference
            mc_outs = []
            with T.no_grad():
                for _ in range(args.mc_samples):
                    out = model(
                        X_cur, mask_cur,
                        edge_index, edge_weight, overlap_frac,
                        stage_ids=stages_cur,
                    )
                    mc_outs.append(out["imputed_mean"].unsqueeze(0))
            mc_stack = T.cat(mc_outs, dim=0)  # (T, N, 33)
            mu_norm = mc_stack.mean(dim=0)  # (N, 33)
            total_var_norm = mc_stack.var(dim=0) + T.exp(out["imputed_log_var"]).clamp(max=10.0)

            # Inverse-transform mean + variance back to original scale
            mu_orig = scaler.inverse_transform(mu_norm.cpu(), mask=None).numpy()
            var_orig = scaler.inverse_transform_variance(
                total_var_norm.cpu(),
                mean_normalized=mu_norm.cpu(),
                mask=None,
            ).numpy()
            sigma_orig = np.sqrt(np.clip(var_orig, 0.0, None))

            # Apply temperature scaling (per-feature; from the P6 stack)
            temps = stack["temperatures"]
            sigma_calibrated = sigma_orig * temps[None, :]

            # Extract visit-specific rows from the swap
            for local_i, v in enumerate(valid_visits):
                t_idx = train_idxs[local_i]
                imputed_mean[v] = mu_orig[t_idx]
                imputed_std[v] = sigma_calibrated[t_idx]

        elapsed = time.time() - t0
        print(f"  Pass {pass_idx + 1}/{max_visits_per_pat}: "
              f"{len(pass_visits)} visits, elapsed={elapsed:.0f}s")

    print(f"\nTotal inference time: {time.time() - t0:.0f}s")
    print(f"Total forward calls: {total_forward_calls}")

    # Build the long-format per-visit bridge parquet
    print("\nBuilding per-visit bridge parquet...")
    rows = []
    patient_total_dat = p3.groupby("PATNO")["putamen_l_sbr"].apply(lambda s: s.notna().sum()).to_dict()

    for v in range(n_visits):
        if patient_idxs_for_visits[v] < 0:
            continue
        record = {
            "PATNO": int(p3.iloc[v]["PATNO"]),
            "EVENT_ID": p3.iloc[v].get("EVENT_ID", ""),
            "months_from_baseline": float(p3.iloc[v].get("months_from_baseline", np.nan)),
            "mnar_flag": bool(patient_total_dat.get(int(p3.iloc[v]["PATNO"]), 0) == 0),
        }
        for feat in DAT_FEATURES:
            j = gimin_features.index(feat)
            is_obs = bool(mask[v, j])
            obs_val = float(X[v, j]) if is_obs else np.nan
            record[f"{feat}_obs"] = obs_val
            record[f"{feat}_is_observed"] = is_obs
            if is_obs:
                record[f"{feat}_imputed_mean"] = obs_val
                record[f"{feat}_imputed_std"] = SBR_SENSOR_SIGMA
            else:
                record[f"{feat}_imputed_mean"] = float(imputed_mean[v, j])
                record[f"{feat}_imputed_std"] = float(imputed_std[v, j])

        # Bilateral mean with error propagation
        for side, lf, rf in [("CAUDATE", "CAUDATE_L_SBR", "CAUDATE_R_SBR"),
                             ("PUTAMEN", "PUTAMEN_L_SBR", "PUTAMEN_R_SBR")]:
            ml = record[f"{lf}_imputed_mean"]
            mr = record[f"{rf}_imputed_mean"]
            sl = record[f"{lf}_imputed_std"]
            sr = record[f"{rf}_imputed_std"]
            both_obs = record[f"{lf}_is_observed"] and record[f"{rf}_is_observed"]
            record[f"{side}_MEAN_SBR_obs"] = 0.5 * (ml + mr) if both_obs else np.nan
            record[f"{side}_MEAN_SBR_is_observed"] = both_obs
            record[f"{side}_MEAN_SBR_imputed_mean"] = 0.5 * (ml + mr)
            record[f"{side}_MEAN_SBR_imputed_std"] = 0.5 * np.sqrt(sl**2 + sr**2)
        rows.append(record)

    bridge = pd.DataFrame(rows)
    out_path = OUT_DIR / "gimin_per_visit_dat_sbr.parquet"
    bridge.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}  shape={bridge.shape}")

    # Summary
    print("\n" + "=" * 70)
    print("PER-VISIT GIMIN SUMMARY")
    print("=" * 70)
    for feat in ["CAUDATE_MEAN_SBR", "PUTAMEN_MEAN_SBR"]:
        n_obs = bridge[f"{feat}_is_observed"].sum()
        n_total = len(bridge)
        imp_rows = bridge[~bridge[f"{feat}_is_observed"]]
        if len(imp_rows):
            sig_med = imp_rows[f"{feat}_imputed_std"].median()
            print(f"  {feat}: {n_obs}/{n_total} observed, {n_total-n_obs} imputed (σ median={sig_med:.4f})")
    print(f"\n  MNAR-flagged visits: {bridge['mnar_flag'].sum()}/{len(bridge)} "
          f"(patients with 0 observed DaT in entire P3 trajectory)")


if __name__ == "__main__":
    main()
