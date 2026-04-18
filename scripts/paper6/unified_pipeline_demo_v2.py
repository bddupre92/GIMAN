#!/usr/bin/env python3
"""Paper 6: Unified Clinical Decision Support Pipeline — v2 with GIMIN integration.

Supersedes scripts/paper6/unified_pipeline_demo.py, which declared GIMIN_CKPT
but never used it (feeding column-median imputation into CatBoost instead).
This v2 actually loads GIMIN, imputes the 33-feature multimodal vector for each
patient, and pipes GIMIN-imputed values + calibrated uncertainty into CatBoost.

Key changes from v1:
  1. Load Paper 2 StageConditionedGIMIN checkpoint + fit ModalityAwareScaler
     (checkpoint does not embed scaler; fit at runtime matching the benchmark).
  2. Build stage-aware patient-similarity graph (β=0.3, k=15) matching the
     StageConditioned variant's training configuration.
  3. Fit per-feature temperature scaler (PerFeatureTemperatureScaler) on a
     held-out artificial-mask split, for calibrated parametric uncertainty.
  4. Replace median-imputation fallback with GIMIN-imputed values for the 5
     CatBoost-12 features GIMIN shares with its 33-feature schema (AGE,
     SEX, MCATOT↔MOCA_TOTAL, ESS_TOTAL, RBD_TOTAL). The remaining 7 CatBoost
     features (UPDRS1/2/4_TOTAL + 4 UPDRS-III subscales) are pulled raw from
     paper1_features_with_targets.csv with a final col_median fallback.
  5. New CLI flags: --cohort {full,pd_only}, --checkpoint-dir, --vignette-ids.
  6. Expanded output JSON records GIMIN provenance per CatBoost feature and
     both raw + calibrated per-feature std.

Usage:
    # Full 1,900 cohort (matches P3/P4/P5):
    python scripts/paper6/unified_pipeline_demo_v2.py --cohort full

    # Just vignettes:
    python scripts/paper6/unified_pipeline_demo_v2.py --vignette-ids 3203 4059 5009

    # With retuned calibration checkpoint (P2-Cal.A sensitivity):
    python scripts/paper6/unified_pipeline_demo_v2.py \\
        --gimin-checkpoint outputs/paper2_benchmark/runs/cal_retune_lambda0.1_warmup0/checkpoints/frac0.1_run0_GIMIN_StageDecoderOnly.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "GIMImpN_imputation"))
sys.path.insert(0, str(ROOT / "scripts"))

# ── Paths ────────────────────────────────────────────────────────────
FEATURES_PATH = ROOT / "data" / "07_paper3_features" / "longitudinal_features.csv"
PAPER1_PATH = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
GIMIN_COHORT_PATH = (
    ROOT / "GIMImpN_imputation" / "outputs" / "ppmi_full_cohort.parquet"
)
GIMIN_MASK_PATH = (
    ROOT / "GIMImpN_imputation" / "outputs" / "missingness_mask.parquet"
)
OUTPUT_DIR = ROOT / "outputs" / "paper6" / "pipeline_results"

DEEPHIT_CKPT = ROOT / "outputs" / "paper3_checkpoints" / "deephit" / "fold0_deephit.pt"
GRAPHDT_CKPT = (
    ROOT / "outputs" / "paper3_checkpoints" / "graph_dt" / "fold0_graph_dt.pt"
)
# StageDecoderOnly at frac=0.1 was the lowest-RMSE Paper 2 variant (107.1).
DEFAULT_GIMIN_CKPT = (
    ROOT
    / "outputs"
    / "paper2_benchmark"
    / "runs"
    / "full_benchmark_20260222_160247"
    / "checkpoints"
    / "frac0.1_run0_GIMIN_StageDecoderOnly.pt"
)

# CatBoost-12 features (cross-cohort clinical)
CATBOOST_12_FEATURES = [
    "AGE_AT_BASELINE",
    "SEX",
    "UPDRS1_TOTAL",
    "UPDRS2_TOTAL",
    "UPDRS3_TREMOR",
    "UPDRS3_RIGIDITY",
    "UPDRS3_BRADYKINESIA",
    "UPDRS3_AXIAL",
    "UPDRS4_TOTAL",
    "MOCA_TOTAL",
    "ESS_TOTAL",
    "RBD_TOTAL",
]

# Features GIMIN shares with CatBoost-12 (after name harmonization).
# Map GIMIN feature name → CatBoost-12 feature name.
GIMIN_TO_CATBOOST_12 = {
    "AGE_AT_VISIT": "AGE_AT_BASELINE",  # approximate: baseline visit only
    "SEX": "SEX",
    "MCATOT": "MOCA_TOTAL",
    "ESS_TOTAL": "ESS_TOTAL",
    "RBD_TOTAL": "RBD_TOTAL",
}

STAGE_LABELS = {0: "0", 1: "1", 2: "2B", 3: "3", 4: "4", 5: "5", 6: "6"}
NSD_POSITIVE_LABELS = {0: "1", 1: "2B", 2: "3", 3: "4"}


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Paper 6 unified pipeline (v2 with GIMIN)")
    parser.add_argument(
        "--cohort",
        choices=["full", "pd_only"],
        default="full",
        help="Use full P3 cohort (1,900) or PD+Prodromal only (Paper 1 Espay fix).",
    )
    parser.add_argument(
        "--vignette-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of PATNOs (for vignette-only runs).",
    )
    parser.add_argument(
        "--gimin-checkpoint",
        type=Path,
        default=DEFAULT_GIMIN_CKPT,
        help="GIMIN state_dict to load for imputation.",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Cap total patients for debugging (e.g., --max-patients 5).",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help="Subdirectory under outputs/paper6/pipeline_results/ (avoids overwrite).",
    )
    return parser.parse_args()


# ── Step 1: Load GIMIN + build graph + fit temperature scaler ─────────
def setup_gimin_stack(args, device):
    """Load GIMIN imputation stack: config + scaler + graph + temperature.

    Returns dict with keys: model (loaded), scaler, graph, temperature, config,
    feature_names (33), patnos (cohort order), patno_to_idx.
    """
    from gimin.config import GIMINConfig
    from gimin.data.scaler import build_scaler_from_config
    from giman_pipeline.imputation.stage_conditioned_gimin import (
        StageConditionedGIMIN,
    )
    from giman_pipeline.imputation.stage_graph_builder import (
        StageAwareGraphBuilder,
    )
    from giman_pipeline.imputation.temperature_scaling import (
        PerFeatureTemperatureScaler,
    )

    # Load data that GIMIN was trained on (2,197 staged baseline visits).
    from run_paper2_experiments import (
        MODALITY_DIMS,
        build_gimin_config,
        create_artificial_mask,
        evaluate_gimin_model,
        load_data,
    )

    print("  Loading GIMIN training cohort (2,197 staged patients)...")
    features_bl, mask_bl, stages_bl, feature_names = load_data()

    cfg = build_gimin_config()
    scaler = build_scaler_from_config(cfg)

    features_t = torch.tensor(features_bl, dtype=torch.float32)
    mask_t = torch.tensor(mask_bl, dtype=torch.float32)
    stages_t = torch.tensor(stages_bl, dtype=torch.long)
    scaler.fit(features_t, mask_t)
    features_norm_t = scaler.transform(features_t, mask_t)

    # Build stage-aware graph (β=0.3 matches StageConditioned / StageDecoderOnly training)
    print("  Building stage-aware patient similarity graph (β=0.3, k=15)...")
    graph_builder = StageAwareGraphBuilder(
        k_neighbors=15,
        min_overlap=3,
        stage_affinity_beta=0.3,
    )
    graph = graph_builder.build_full_graph(
        features_norm_t.numpy() * mask_bl.astype(np.float32),
        mask_bl,
        stages=stages_bl,
    )

    # Construct model matching the checkpoint variant.
    ckpt_name = str(args.gimin_checkpoint)
    if "StageDecoderOnly" in ckpt_name or "StageConditioned" in ckpt_name:
        model = StageConditionedGIMIN(
            modality_dims=MODALITY_DIMS,
            embed_dim=64,
            num_gnn_layers=3,
            num_heads=4,
            mc_dropout=0.1,
            num_stages=6,
            stage_embed_dim=16,
        )
        is_stage_conditioned = True
    else:
        from giman_pipeline.imputation.stage_conditioned_gimin import VanillaGIMIN
        model = VanillaGIMIN(
            modality_dims=MODALITY_DIMS,
            embed_dim=64,
            num_gnn_layers=3,
            num_heads=4,
            mc_dropout=0.1,
        )
        is_stage_conditioned = False

    state = torch.load(str(args.gimin_checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"  Loaded GIMIN checkpoint: {args.gimin_checkpoint.name}")

    # Run GIMIN inference on the training cohort with an artificial mask
    # so we can fit the temperature scaler on residuals we can measure.
    print("  Running GIMIN inference for temperature-scaler calibration...")
    corrupted_mask, eval_mask = create_artificial_mask(mask_bl, 0.1, seed=42)
    corrupted_mask_t = torch.tensor(corrupted_mask, dtype=torch.float32)

    _, _, mean_pred, total_std = evaluate_gimin_model(
        model,
        features_norm_t.to(device),
        mask_t.to(device),
        corrupted_mask_t.to(device),
        eval_mask,
        graph["edge_index"].to(device),
        graph["edge_weight"].to(device),
        graph["overlap_frac"].to(device),
        stages_bl,
        features_original=features_bl,
        scaler=scaler,
        stages_t=stages_t.to(device) if is_stage_conditioned else None,
        is_stage_conditioned=is_stage_conditioned,
        mc_samples=20,
    )

    print("  Fitting per-feature temperature scaler at γ=0.90 target...")
    temp_scaler = PerFeatureTemperatureScaler(target_coverage=0.90)
    temp_scaler.fit(mean_pred, total_std, features_bl, eval_mask)
    print(
        f"    Temperature range: [{temp_scaler.temperatures_.min():.3f}, "
        f"{temp_scaler.temperatures_.max():.3f}], median={np.median(temp_scaler.temperatures_):.3f}"
    )

    # Keep the imputed mean + calibrated std for the training cohort so we can
    # serve per-patient imputations without re-running inference.
    calibrated_std = temp_scaler.transform(total_std)

    # Need to know which rows correspond to which PATNO. The benchmark pipeline
    # reads features from ppmi_full_cohort.parquet indexed by PATNO, then
    # inner-joins with staging → features_staged is in staging order. We'll
    # reconstruct the PATNO list by re-running the same join.
    cohort_patnos = _reload_training_cohort_patnos()
    assert len(cohort_patnos) == len(features_bl), (
        f"PATNO count ({len(cohort_patnos)}) != feature rows ({len(features_bl)})"
    )

    patno_to_idx = {p: i for i, p in enumerate(cohort_patnos)}

    return {
        "feature_names": feature_names,
        "imputed_means": mean_pred,  # (N, 33) original scale
        "raw_std": total_std,  # (N, 33)
        "calibrated_std": calibrated_std,  # (N, 33)
        "temperatures": temp_scaler.temperatures_,  # (33,)
        "patnos": cohort_patnos,
        "patno_to_idx": patno_to_idx,
    }


def _reload_training_cohort_patnos():
    """Re-derive the PATNO order the GIMIN benchmark used."""
    from run_paper2_experiments import STAGE_MAP

    features_df = pd.read_parquet(GIMIN_COHORT_PATH)
    staging_df = pd.read_csv(
        ROOT / "data" / "04_staging" / "nsd_iss_staging_results.csv"
    )
    staging_df["stage_encoded"] = staging_df["nsd_iss_stage"].astype(str).map(STAGE_MAP)
    staging_df["stage_encoded"] = staging_df["stage_encoded"].fillna(5).astype(int)
    staging_valid = staging_df[staging_df["nsd_iss_stage"] != "unclassified"].copy()
    staged_patnos = set(staging_valid["PATNO"].values)
    staged_mask = features_df.index.isin(staged_patnos)
    features_staged = features_df[staged_mask].reset_index().merge(
        staging_valid[["PATNO", "stage_encoded"]],
        on="PATNO",
        how="inner",
    )
    return list(features_staged["PATNO"].values)


# ── Step 2: Train CatBoost on --cohort-filtered training data ─────────
def train_catboost_staging(cohort_choice: str):
    """Train Paper 1 CatBoost NSD+ model; PD-only filter optional (Espay fix)."""
    from catboost import CatBoostClassifier

    df = pd.read_csv(PAPER1_PATH)
    df_nsd = df[df["target_nsd_positive"] >= 0].copy()

    if cohort_choice == "pd_only" and "COHORT_DEFINITION" in df_nsd.columns:
        before = len(df_nsd)
        df_nsd = df_nsd[
            df_nsd["COHORT_DEFINITION"].isin(["PD", "Prodromal"])
        ].copy()
        print(f"  PD-only cohort filter: {before} → {len(df_nsd)}")

    y = df_nsd["target_nsd_positive"].values.astype(int)
    X = df_nsd[CATBOOST_12_FEATURES].values.astype(float)
    col_medians = np.nanmedian(X, axis=0)
    X_filled = np.where(np.isnan(X), col_medians, X)

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        auto_class_weights="Balanced",
        verbose=0,
        random_state=42,
    )
    model.fit(X_filled, y)
    y_pred = np.asarray(model.predict(X_filled)).ravel().astype(int)
    print(f"  CatBoost train accuracy: {(y_pred == y).mean():.3f}")
    return model, col_medians


# ── Step 3: Per-patient pipeline (as in v1, with GIMIN integration) ───
def run_patient_pipeline(
    patno, features_df, gimin_stack, catboost_model, col_medians,
    dh_model, dh_ckpt, gdt_model, gdt_ckpt, device,
    p1_df,
):
    """Full pipeline for one patient with GIMIN-imputed CatBoost inputs."""
    from giman_pipeline.paper3.dynamic_deephit import (
        FEATURES_WITH_MISSING,
        N_STATES,
        N_TIME_BINS,
        STATIC_FEATURES,
        TIME_BIN_ENDS,
        TIME_VARYING_FEATURES,
    )
    from giman_pipeline.paper3.multistate_markov import STAGE_TO_IDX

    pat_visits = features_df[features_df["PATNO"] == patno].sort_values(
        "months_from_baseline"
    )
    n_visits = len(pat_visits)
    if n_visits == 0:
        return {"patno": int(patno), "error": "no_longitudinal_visits"}

    stages = pat_visits["nsd_stage"].tolist()
    times = pat_visits["months_from_baseline"].tolist()
    latest = pat_visits.iloc[-1]

    # ── CatBoost staging with GIMIN-imputed features ──
    # Build feature vector; track provenance per feature.
    x_staging = np.zeros(len(CATBOOST_12_FEATURES))
    feat_provenance = {}
    gimin_imputed_means = {}
    gimin_calibrated_stds = {}

    gimin_idx = gimin_stack["patno_to_idx"].get(patno)
    gimin_means = gimin_stack["imputed_means"][gimin_idx] if gimin_idx is not None else None
    gimin_std = gimin_stack["calibrated_std"][gimin_idx] if gimin_idx is not None else None
    gimin_feature_names = gimin_stack["feature_names"]
    gimin_feat_to_idx = {f: i for i, f in enumerate(gimin_feature_names)}

    p1_row = p1_df[p1_df["PATNO"] == patno]

    for j, feat in enumerate(CATBOOST_12_FEATURES):
        # Try GIMIN imputed first (5 features overlap)
        gimin_src_name = None
        for gname, cname in GIMIN_TO_CATBOOST_12.items():
            if cname == feat and gname in gimin_feat_to_idx:
                gimin_src_name = gname
                break

        used_val = None
        prov = None
        if gimin_means is not None and gimin_src_name is not None:
            gidx = gimin_feat_to_idx[gimin_src_name]
            used_val = float(gimin_means[gidx])
            prov = "gimin_imputed"
            gimin_imputed_means[feat] = used_val
            gimin_calibrated_stds[feat] = float(gimin_std[gidx])

        # Fall back to Paper 1 raw value
        if used_val is None or not np.isfinite(used_val):
            if len(p1_row) > 0 and pd.notna(p1_row.iloc[0].get(feat)):
                used_val = float(p1_row.iloc[0][feat])
                prov = "paper1_raw"

        # Final fallback: col_median
        if used_val is None or not np.isfinite(used_val):
            used_val = float(col_medians[j])
            prov = "median_fallback"

        x_staging[j] = used_val
        feat_provenance[feat] = prov

    x_staging_2d = x_staging.reshape(1, -1)
    stage_probs = catboost_model.predict_proba(x_staging_2d)[0]
    stage_pred = int(np.asarray(catboost_model.predict(x_staging_2d)).ravel()[0])

    staging_result = {
        "predicted_class": stage_pred,
        "predicted_stage": NSD_POSITIVE_LABELS.get(stage_pred, str(stage_pred)),
        "probabilities": {
            NSD_POSITIVE_LABELS.get(k, str(k)): round(float(p), 4)
            for k, p in enumerate(stage_probs)
        },
        "actual_stage": str(latest["nsd_stage"]),
        "features_used": {
            feat: round(float(x_staging[j]), 3)
            for j, feat in enumerate(CATBOOST_12_FEATURES)
        },
        "feature_provenance": feat_provenance,
    }

    # ── DeepHit + Graph-DT (unchanged from v1) ──
    input_dim = dh_ckpt["input_dim"]
    means = dh_ckpt["means"]
    stds = dh_ckpt["stds"]

    all_features = TIME_VARYING_FEATURES + STATIC_FEATURES
    seq_data = []
    for _, row in pat_visits.iterrows():
        visit_vec = []
        for f in all_features:
            v = row.get(f)
            visit_vec.append(float(v) if pd.notna(v) else 0.0)
        for f in FEATURES_WITH_MISSING:
            visit_vec.append(0.0 if pd.notna(row.get(f)) else 1.0)
        seq_data.append(visit_vec)
    seq_array = np.array(seq_data, dtype=np.float32)
    if len(means) == seq_array.shape[1]:
        seq_normed = (seq_array - means) / (stds + 1e-8)
    else:
        seq_normed = seq_array
    seq_tensor = torch.tensor(seq_normed, dtype=torch.float32).unsqueeze(0).to(device)
    seq_len = torch.tensor([n_visits], dtype=torch.long).to(device)
    stage_idx = STAGE_TO_IDX.get(str(latest["nsd_stage"]), 3)
    stage_tensor = torch.tensor([stage_idx], dtype=torch.long).to(device)

    dh_model.eval()
    with torch.no_grad():
        dh_pmf = dh_model(seq_tensor, seq_len, stage_tensor)
    pmf_np = dh_pmf.cpu().numpy()[0]
    n_causes = N_STATES
    n_tbins = N_TIME_BINS
    dh_cif = np.cumsum(pmf_np[: n_causes * n_tbins].reshape(n_causes, n_tbins), axis=1)

    gdt_model.eval()
    pat_to_gidx = gdt_ckpt["pat_to_gidx"]
    node_baseline = gdt_ckpt["node_baseline"].to(device)
    edge_index = gdt_ckpt["edge_index"].to(device)
    with torch.no_grad():
        node_enc = gdt_model.node_encoder(node_baseline)
        for gat_layer in gdt_model.gat_layers_list:
            node_enc = gat_layer(node_enc, edge_index)
        node_enc = gdt_model.gat_norm(gdt_model.gat_proj(node_enc))
    graph_idx = pat_to_gidx.get(patno, 0)
    graph_idx_tensor = torch.tensor([graph_idx], dtype=torch.long).to(device)
    with torch.no_grad():
        gdt_pmf = gdt_model(seq_tensor, seq_len, stage_tensor, graph_idx_tensor, node_enc)
    gdt_pmf_np = gdt_pmf.cpu().numpy()[0]
    gdt_cif = np.cumsum(gdt_pmf_np[: n_causes * n_tbins].reshape(n_causes, n_tbins), axis=1)

    # Conformal band from Paper 4 aggregate
    band_width = 0.037
    agg_path = ROOT / "outputs" / "paper4" / "conformal" / "aggregate_summary.json"
    if agg_path.exists():
        agg = json.load(open(agg_path))
        for entry in agg.get("per_model", []):
            if (
                entry.get("model") == "deephit"
                and abs(entry.get("confidence_level", 0) - 0.90) < 0.01
            ):
                band_width = entry.get("mean_band_width", 0.037)
                break
    dh_bands = np.stack(
        [np.clip(dh_cif - band_width, 0, 1), np.clip(dh_cif + band_width, 0, 1)],
        axis=-1,
    )
    gdt_bands = np.stack(
        [np.clip(gdt_cif - band_width, 0, 1), np.clip(gdt_cif + band_width, 0, 1)],
        axis=-1,
    )

    # Top transitions
    top_transitions = []
    for k in range(n_causes):
        dh_max = float(dh_cif[k, -1])
        gdt_max = float(gdt_cif[k, -1])
        max_cif = max(dh_max, gdt_max)
        if max_cif > 0.005:
            top_transitions.append({
                "destination_stage": STAGE_LABELS.get(k, str(k)),
                "cause_idx": k,
                "max_cif_deephit": round(dh_max, 4),
                "max_cif_graphdt": round(gdt_max, 4),
                "max_cif": round(max_cif, 4),
                "cif_at_12mo": round(float(dh_cif[k, 2]), 4),
                "cif_at_36mo": round(float(dh_cif[k, 5]), 4),
                "cif_at_60mo": round(float(dh_cif[k, 7]), 4),
                "gdt_cif_at_12mo": round(float(gdt_cif[k, 2]), 4),
                "gdt_cif_at_36mo": round(float(gdt_cif[k, 5]), 4),
                "gdt_cif_at_60mo": round(float(gdt_cif[k, 7]), 4),
            })
    top_transitions.sort(key=lambda d: d["max_cif"], reverse=True)

    return {
        "patno": int(patno),
        "n_visits": n_visits,
        "follow_up_months": float(times[-1]),
        "stage_trajectory": [str(s) for s in stages],
        "visit_times_months": [round(float(t), 1) for t in times],
        "current_stage": str(latest["nsd_stage"]),
        "staging": staging_result,
        "gimin_imputed_means": gimin_imputed_means,
        "gimin_calibrated_stds": gimin_calibrated_stds,
        "gimin_used": gimin_idx is not None,
        "deephit_cif": dh_cif.tolist(),
        "graph_dt_cif": gdt_cif.tolist(),
        "deephit_cif_bands": dh_bands.tolist(),
        "graph_dt_cif_bands": gdt_bands.tolist(),
        "top_transitions": top_transitions[:5],
        "time_bin_months": TIME_BIN_ENDS,
        "stage_labels": STAGE_LABELS,
        "conformal_band_width": band_width,
    }


# ── Main ──────────────────────────────────────────────────────────────
def run_unified_pipeline():
    args = parse_args()
    device = _get_device()

    out_dir = OUTPUT_DIR
    if args.output_subdir:
        out_dir = out_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"  PAPER 6 UNIFIED PIPELINE v2 (GIMIN integrated)")
    print(f"  Device: {device}")
    print(f"  Cohort: {args.cohort}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}")

    # Step 1: GIMIN stack
    print("\n[1/4] GIMIN imputation stack...")
    gimin_stack = setup_gimin_stack(args, device)

    # Step 2: CatBoost
    print("\n[2/4] Training CatBoost staging model...")
    catboost_model, col_medians = train_catboost_staging(args.cohort)
    catboost_model.save_model(str(out_dir / "catboost_nsd_positive.cbm"))

    # Step 3: Survival models
    print("\n[3/4] Loading DeepHit + Graph-DT checkpoints...")
    from giman_pipeline.paper3.dynamic_deephit import load_deephit_checkpoint
    from giman_pipeline.paper3.graph_digital_twin import load_graph_dt_checkpoint

    dh_model, dh_ckpt = load_deephit_checkpoint(DEEPHIT_CKPT, device=device)
    gdt_model, gdt_ckpt = load_graph_dt_checkpoint(GRAPHDT_CKPT, device=device)

    # Step 4: Per-patient pipeline
    features_df = pd.read_csv(FEATURES_PATH, low_memory=False)
    p1_df = pd.read_csv(PAPER1_PATH)

    if args.vignette_ids:
        patnos = args.vignette_ids
    elif args.cohort == "pd_only" and "COHORT_DEFINITION" in p1_df.columns:
        pd_patnos = set(p1_df[p1_df["COHORT_DEFINITION"].isin(["PD", "Prodromal"])]["PATNO"])
        long_patnos = set(features_df["PATNO"].unique())
        patnos = sorted(pd_patnos & long_patnos)
    else:
        patnos = sorted(features_df["PATNO"].unique())

    if args.max_patients:
        patnos = patnos[: args.max_patients]

    print(f"\n[4/4] Running pipeline on {len(patnos)} patients...")

    all_results = {}
    total_start = time.time()
    for i, patno in enumerate(patnos):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - total_start
            rate = i / elapsed
            eta = (len(patnos) - i) / rate
            print(f"  [{i}/{len(patnos)}] {rate:.2f} pts/s, ETA {eta / 60:.1f} min")
        try:
            r = run_patient_pipeline(
                patno, features_df, gimin_stack, catboost_model, col_medians,
                dh_model, dh_ckpt, gdt_model, gdt_ckpt, device, p1_df,
            )
            all_results[str(patno)] = r
        except Exception as e:
            print(f"  ERROR for {patno}: {e}")
            all_results[str(patno)] = {"patno": int(patno), "error": str(e)}

    total_elapsed = time.time() - total_start
    print(f"\nPipeline complete: {len(patnos)} patients in {total_elapsed:.1f}s")

    # Aggregate summary
    successful = [r for r in all_results.values() if "error" not in r]
    summary = {
        "cohort": args.cohort,
        "n_requested": len(patnos),
        "n_successful": len(successful),
        "n_errors": len(patnos) - len(successful),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "per_patient_seconds": round(total_elapsed / max(len(patnos), 1), 3),
        "gimin_checkpoint": str(args.gimin_checkpoint),
        "temperature_median": float(np.median(gimin_stack["temperatures"])),
        "temperature_mean": float(np.mean(gimin_stack["temperatures"])),
        "conformal_band_width_from_paper4": 0.037,
    }

    # Save
    all_results["summary"] = summary
    with open(out_dir / "pipeline_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=_convert)
    # Also save each patient's JSON (only if successful + cohort is small enough)
    if args.cohort == "pd_only" or args.max_patients or args.vignette_ids:
        for patno, r in all_results.items():
            if patno == "summary":
                continue
            with open(out_dir / f"patient_{patno}_pipeline.json", "w") as f:
                json.dump(r, f, indent=2, default=_convert)

    print(f"\nResults saved: {out_dir / 'pipeline_summary.json'}")
    print(f"Summary: {json.dumps(summary, indent=2)}")


def _convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return list(obj)
    return obj


if __name__ == "__main__":
    run_unified_pipeline()
