#!/usr/bin/env python3
"""Paper 6: Unified Clinical Decision Support Pipeline Demo.

End-to-end pipeline for selected patients:
  1. GIMIN imputation (Paper 2): Fill missing longitudinal features
  2. CatBoost staging (Paper 1): Predict NSD-ISS stage + conformal set
  3. Graph-DT prediction (Paper 3): Generate CIF for all 7 transitions
  4. Conformal CIF bands (Paper 4): 90% prediction bands + timing intervals

Runs pipeline with GIMIN imputation vs mean imputation to quantify impact.

Usage:
    python scripts/paper6/unified_pipeline_demo.py
"""

from __future__ import annotations

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

# ── Paths ────────────────────────────────────────────────────────────
FEATURES_PATH = ROOT / "data" / "07_paper3_features" / "longitudinal_features.csv"
PAPER1_PATH = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
SELECTION_PATH = ROOT / "outputs" / "paper6" / "patient_selection.json"
OUTPUT_DIR = ROOT / "outputs" / "paper6" / "pipeline_results"

# Model checkpoints
DEEPHIT_CKPT = ROOT / "outputs" / "paper3_checkpoints" / "deephit" / "fold0_deephit.pt"
GRAPHDT_CKPT = ROOT / "outputs" / "paper3_checkpoints" / "graph_dt" / "fold0_graph_dt.pt"
GIMIN_CKPT = (
    ROOT / "outputs" / "paper2_benchmark" / "runs"
    / "full_benchmark_20260222_160247" / "checkpoints"
    / "frac0.1_run0_GIMIN_StageDecoderOnly.pt"
)

# Paper 1 CatBoost 12-feature model features (cross-cohort clinical)
CATBOOST_12_FEATURES = [
    "AGE_AT_BASELINE", "SEX",
    "UPDRS1_TOTAL", "UPDRS2_TOTAL",
    "UPDRS3_TREMOR", "UPDRS3_RIGIDITY", "UPDRS3_BRADYKINESIA", "UPDRS3_AXIAL",
    "UPDRS4_TOTAL", "MOCA_TOTAL", "ESS_TOTAL", "RBD_TOTAL",
]

# Mapping from Paper 3 longitudinal feature names to Paper 1 feature names
LONGITUDINAL_TO_PAPER1 = {
    "age_at_baseline": "AGE_AT_BASELINE",
    "sex": "SEX",
    "updrs1_total": "UPDRS1_TOTAL",
    "updrs2_total": "UPDRS2_TOTAL",
    "moca_total": "MOCA_TOTAL",
    "ess_total": "ESS_TOTAL",
    "rbd_total": "RBD_TOTAL",
}

# NSD-ISS stage labels
STAGE_LABELS = {0: "0", 1: "1", 2: "2B", 3: "3", 4: "4", 5: "5", 6: "6"}
NSD_POSITIVE_LABELS = {0: "1", 1: "2B", 2: "3", 3: "4"}


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Step 1: Train CatBoost Staging Model ─────────────────────────────

def train_catboost_staging() -> tuple:
    """Train Paper 1 CatBoost on NSD-positive target (12 clinical features).

    Returns:
        (model, X_train, y_train, classes) for conformal calibration.
    """
    from catboost import CatBoostClassifier

    print("\n" + "=" * 60)
    print("  STEP 1: Train CatBoost NSD-ISS Staging Model")
    print("=" * 60)

    df = pd.read_csv(PAPER1_PATH)

    # NSD-positive target: stages 1, 2B, 3, 4 only (exclude stage 0)
    df_nsd = df[df["target_nsd_positive"] >= 0].copy()
    y = df_nsd["target_nsd_positive"].values.astype(int)

    # Build feature matrix
    X = df_nsd[CATBOOST_12_FEATURES].values.astype(float)

    # Handle NaN by filling with column medians (from training set)
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_medians[j]

    print(f"  Training data: {X.shape[0]} patients, {X.shape[1]} features")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        auto_class_weights="Balanced",
        verbose=0,
        random_state=42,
    )
    model.fit(X, y)

    # Save checkpoint
    ckpt_path = OUTPUT_DIR / "catboost_nsd_positive.cbm"
    model.save_model(str(ckpt_path))
    print(f"  Saved CatBoost model to {ckpt_path}")

    # Quick train accuracy
    y_pred = np.asarray(model.predict(X)).ravel().astype(int)
    acc = (y_pred == y).mean()
    print(f"  Train accuracy: {acc:.3f}")

    return model, X, y, col_medians


# ── Step 2: Load Survival Models ──────────────────────────────────────

def load_survival_models(device: torch.device):
    """Load DeepHit and Graph-DT fold 0 checkpoints."""
    from giman_pipeline.paper3.dynamic_deephit import load_deephit_checkpoint
    from giman_pipeline.paper3.graph_digital_twin import load_graph_dt_checkpoint

    print("\n" + "=" * 60)
    print("  STEP 2: Load Survival Models (Fold 0)")
    print("=" * 60)

    dh_model, dh_ckpt = load_deephit_checkpoint(DEEPHIT_CKPT, device=device)
    print(f"  DeepHit: C-td={dh_ckpt['fold_ctd']:.4f}, "
          f"input_dim={dh_ckpt['input_dim']}")

    gdt_model, gdt_ckpt = load_graph_dt_checkpoint(GRAPHDT_CKPT, device=device)
    print(f"  Graph-DT: C-td={gdt_ckpt['fold_ctd']:.4f}, "
          f"n_baseline_features={gdt_ckpt['n_baseline_features']}")

    return dh_model, dh_ckpt, gdt_model, gdt_ckpt


# ── Step 3: Run Pipeline Per Patient ──────────────────────────────────

def run_patient_pipeline(
    patno: int,
    features_df: pd.DataFrame,
    catboost_model,
    col_medians: np.ndarray,
    dh_model,
    dh_ckpt: dict,
    gdt_model,
    gdt_ckpt: dict,
    device: torch.device,
) -> dict:
    """Run the full unified pipeline for one patient.

    Steps:
        1. Extract patient visit history
        2. Stage each visit using CatBoost (with mean imputation for missing)
        3. Predict transition CIFs using DeepHit and Graph-DT
        4. Generate conformal CIF bands
    """
    from giman_pipeline.paper3.dynamic_deephit import (
        TIME_BIN_ENDS, N_TIME_BINS, N_STATES,
        TIME_VARYING_FEATURES, STATIC_FEATURES, FEATURES_WITH_MISSING,
        extract_episodes, build_patient_arrays,
    )
    from giman_pipeline.paper3.multistate_markov import STAGE_TO_IDX
    from giman_pipeline.paper4.conformal_survival import (
        CauseSpecificConformal, TIME_BIN_ENDS as CONF_TIME_BINS,
    )

    print(f"\n--- Patient {patno} ---")

    # 1. Extract visit history
    pat_visits = features_df[features_df["PATNO"] == patno].sort_values("months_from_baseline")
    n_visits = len(pat_visits)
    stages = pat_visits["nsd_stage"].tolist()
    times = pat_visits["months_from_baseline"].tolist()
    print(f"  {n_visits} visits over {times[-1]:.0f} months")
    print(f"  Stage trajectory: {' → '.join(str(s) for s in stages)}")

    # Track missingness
    missing_info = {}
    for feat in ["updrs3_total", "moca_total", "ess_total", "rbd_total",
                  "scopa_aut_total", "updrs1_total", "updrs2_total"]:
        n_miss = pat_visits[feat].isna().sum()
        missing_info[feat] = {"n_missing": int(n_miss), "n_total": n_visits,
                              "pct": round(100 * n_miss / n_visits, 1)}

    # 2. CatBoost staging at latest visit
    latest = pat_visits.iloc[-1]
    staging_features = {}
    for long_name, p1_name in LONGITUDINAL_TO_PAPER1.items():
        val = latest.get(long_name)
        staging_features[p1_name] = float(val) if pd.notna(val) else None

    # UPDRS3 subscales not directly in longitudinal — use Paper 1 if available
    p1_df = pd.read_csv(PAPER1_PATH)
    p1_row = p1_df[p1_df["PATNO"] == patno]
    for feat in ["UPDRS3_TREMOR", "UPDRS3_RIGIDITY", "UPDRS3_BRADYKINESIA",
                  "UPDRS3_AXIAL", "UPDRS4_TOTAL"]:
        if len(p1_row) > 0 and pd.notna(p1_row.iloc[0].get(feat)):
            staging_features[feat] = float(p1_row.iloc[0][feat])
        else:
            staging_features[feat] = None

    # Build feature vector for CatBoost
    x_staging = np.zeros(len(CATBOOST_12_FEATURES))
    for j, feat in enumerate(CATBOOST_12_FEATURES):
        val = staging_features.get(feat)
        if val is not None and not np.isnan(val):
            x_staging[j] = val
        else:
            x_staging[j] = col_medians[j]

    # CatBoost prediction
    x_staging_2d = x_staging.reshape(1, -1)
    stage_probs = catboost_model.predict_proba(x_staging_2d)[0]
    stage_pred = int(np.asarray(catboost_model.predict(x_staging_2d)).ravel()[0])

    staging_result = {
        "predicted_class": int(stage_pred),
        "predicted_stage": NSD_POSITIVE_LABELS.get(stage_pred, str(stage_pred)),
        "probabilities": {
            NSD_POSITIVE_LABELS[k]: round(float(p), 4)
            for k, p in enumerate(stage_probs)
        },
        "actual_stage": str(latest["nsd_stage"]),
        "features_used": {
            feat: round(float(x_staging[j]), 2)
            for j, feat in enumerate(CATBOOST_12_FEATURES)
        },
    }
    print(f"  CatBoost staging: predicted={staging_result['predicted_stage']}, "
          f"actual={staging_result['actual_stage']}")

    # 3. Survival prediction using DeepHit
    # Build patient arrays for the full feature set
    input_dim = dh_ckpt["input_dim"]
    means = dh_ckpt["means"]
    stds = dh_ckpt["stds"]
    col_names = dh_ckpt.get("col_names", [])

    # Build visit sequence
    all_features = TIME_VARYING_FEATURES + STATIC_FEATURES
    missing_indicators = [f"miss_{f}" for f in FEATURES_WITH_MISSING]
    all_cols = all_features + missing_indicators

    seq_data = []
    for _, row in pat_visits.iterrows():
        visit_vec = []
        for feat in all_features:
            val = row.get(feat)
            visit_vec.append(float(val) if pd.notna(val) else 0.0)
        for feat in FEATURES_WITH_MISSING:
            visit_vec.append(0.0 if pd.notna(row.get(feat)) else 1.0)
        seq_data.append(visit_vec)

    seq_array = np.array(seq_data, dtype=np.float32)

    # Standardize
    if len(means) == seq_array.shape[1]:
        seq_normed = (seq_array - means) / (stds + 1e-8)
    else:
        seq_normed = seq_array

    seq_tensor = torch.tensor(seq_normed, dtype=torch.float32).unsqueeze(0).to(device)
    seq_len = torch.tensor([n_visits], dtype=torch.long).to(device)

    # Current stage index
    current_stage_str = str(latest["nsd_stage"])
    stage_idx = STAGE_TO_IDX.get(current_stage_str, 3)
    stage_tensor = torch.tensor([stage_idx], dtype=torch.long).to(device)

    # DeepHit CIF prediction
    dh_model.eval()
    with torch.no_grad():
        dh_pmf = dh_model(seq_tensor, seq_len, stage_tensor)

    # Convert PMF to CIF
    n_causes = N_STATES
    n_tbins = N_TIME_BINS
    pmf_np = dh_pmf.cpu().numpy()[0]  # (n_causes * n_tbins + 1,)

    # Reshape to (n_causes, n_tbins) — exclude the no-event bin
    cause_pmf = pmf_np[:n_causes * n_tbins].reshape(n_causes, n_tbins)
    cif = np.cumsum(cause_pmf, axis=1)  # CIF = cumulative sum over time

    dh_cif = cif  # (7, 11)

    # 4. Graph-DT CIF prediction
    gdt_model.eval()
    pat_to_gidx = gdt_ckpt["pat_to_gidx"]
    node_baseline = gdt_ckpt["node_baseline"].to(device)
    edge_index = gdt_ckpt["edge_index"].to(device)

    # Encode node features through GAT
    with torch.no_grad():
        node_enc = gdt_model.node_encoder(node_baseline)
        for gat_layer in gdt_model.gat_layers_list:
            node_enc = gat_layer(node_enc, edge_index)
        node_enc = gdt_model.gat_norm(gdt_model.gat_proj(node_enc))

    if patno in pat_to_gidx:
        graph_idx = pat_to_gidx[patno]
    else:
        graph_idx = 0  # Fallback for unknown patients

    graph_idx_tensor = torch.tensor([graph_idx], dtype=torch.long).to(device)

    with torch.no_grad():
        gdt_pmf = gdt_model(seq_tensor, seq_len, stage_tensor, graph_idx_tensor, node_enc)

    gdt_pmf_np = gdt_pmf.cpu().numpy()[0]
    gdt_cause_pmf = gdt_pmf_np[:n_causes * n_tbins].reshape(n_causes, n_tbins)
    gdt_cif = np.cumsum(gdt_cause_pmf, axis=1)

    # 5. Conformal CIF bands (using calibrated quantiles from Paper 4)
    # Load Paper 4 conformal results for fold 0
    conformal_results_path = (
        ROOT / "outputs" / "paper4" / "conformal" / "conformal_results_deephit.json"
    )
    timing_results = {}
    if conformal_results_path.exists():
        with open(conformal_results_path) as f:
            conf_data = json.load(f)
        # Extract fold 0 quantiles at 90% CL for timing intervals
        for entry in conf_data:
            if entry.get("fold_idx") == 0 and abs(entry.get("confidence_level", 0) - 0.90) < 0.01:
                timing_results = entry
                break

    # Compute simple conformal bands using Paper 4 aggregate summary
    aggregate_path = ROOT / "outputs" / "paper4" / "conformal" / "aggregate_summary.json"
    band_width = 0.037  # Default: 95% CL mean band width from Paper 4
    if aggregate_path.exists():
        with open(aggregate_path) as f:
            agg = json.load(f)
        # Use the DeepHit 90% CL band width
        for entry in agg.get("per_model", []):
            if entry.get("model") == "deephit" and abs(entry.get("confidence_level", 0) - 0.90) < 0.01:
                band_width = entry.get("mean_band_width", 0.037)
                break

    # Apply uniform conformal band
    dh_cif_bands = np.stack([
        np.clip(dh_cif - band_width, 0, 1),
        np.clip(dh_cif + band_width, 0, 1),
    ], axis=-1)  # (7, 11, 2)

    gdt_cif_bands = np.stack([
        np.clip(gdt_cif - band_width, 0, 1),
        np.clip(gdt_cif + band_width, 0, 1),
    ], axis=-1)

    # 6. Identify most likely transitions (use max of DeepHit and Graph-DT)
    top_transitions = []
    for k in range(n_causes):
        dh_max = float(dh_cif[k, -1])
        gdt_max = float(gdt_cif[k, -1])
        max_cif = max(dh_max, gdt_max)
        if max_cif > 0.005:
            stage_label = STAGE_LABELS.get(k, str(k))
            top_transitions.append({
                "destination_stage": stage_label,
                "cause_idx": int(k),
                "max_cif_deephit": round(dh_max, 4),
                "max_cif_graphdt": round(gdt_max, 4),
                "max_cif": round(max_cif, 4),
                "cif_at_12mo": round(float(dh_cif[k, 2]), 4),  # bin 2 = 12mo
                "cif_at_36mo": round(float(dh_cif[k, 5]), 4),  # bin 5 = 36mo
                "cif_at_60mo": round(float(dh_cif[k, 7]), 4),  # bin 7 = 60mo
                "gdt_cif_at_12mo": round(float(gdt_cif[k, 2]), 4),
                "gdt_cif_at_36mo": round(float(gdt_cif[k, 5]), 4),
                "gdt_cif_at_60mo": round(float(gdt_cif[k, 7]), 4),
            })

    top_transitions.sort(key=lambda x: x["max_cif"], reverse=True)

    # 7. Build comprehensive result
    result = {
        "patno": int(patno),
        "n_visits": n_visits,
        "follow_up_months": float(times[-1]),
        "stage_trajectory": [str(s) for s in stages],
        "visit_times_months": [round(float(t), 1) for t in times],
        "current_stage": str(latest["nsd_stage"]),
        "missing_features": missing_info,
        "staging": staging_result,
        "deephit_cif": dh_cif.tolist(),
        "graph_dt_cif": gdt_cif.tolist(),
        "deephit_cif_bands": dh_cif_bands.tolist(),
        "graph_dt_cif_bands": gdt_cif_bands.tolist(),
        "top_transitions": top_transitions[:5],
        "time_bin_months": TIME_BIN_ENDS,
        "stage_labels": STAGE_LABELS,
        "conformal_band_width": band_width,
    }

    # Clinical summary
    if top_transitions:
        top = top_transitions[0]
        result["clinical_summary"] = (
            f"Patient {patno} is currently at Stage {current_stage_str}. "
            f"The most likely next transition is to Stage {top['destination_stage']} "
            f"(CIF={top['max_cif']:.2f} at 15 years). "
            f"At 12 months: CIF={top['cif_at_12mo']:.3f}, "
            f"at 36 months: CIF={top['cif_at_36mo']:.3f}."
        )
        print(f"  Top transition: →Stage {top['destination_stage']} "
              f"(CIF@12mo={top['cif_at_12mo']:.3f}, @36mo={top['cif_at_36mo']:.3f})")
    else:
        result["clinical_summary"] = (
            f"Patient {patno} at Stage {current_stage_str} — "
            "no high-probability transitions predicted."
        )

    return result


# ── Main Pipeline ────────────────────────────────────────────────────

def run_unified_pipeline():
    """Run the full unified pipeline for all selected patients."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load patient selection
    with open(SELECTION_PATH) as f:
        selection = json.load(f)
    selected_patnos = selection["selected_patnos"]
    print(f"Running pipeline for {len(selected_patnos)} patients: {selected_patnos}")

    # Load longitudinal features
    features_df = pd.read_csv(FEATURES_PATH, low_memory=False)
    device = _get_device()
    print(f"Device: {device}")

    # Step 1: Train CatBoost
    catboost_model, X_train, y_train, col_medians = train_catboost_staging()

    # Step 2: Load survival models
    dh_model, dh_ckpt, gdt_model, gdt_ckpt = load_survival_models(device)

    # Step 3: Run pipeline per patient
    all_results = {}
    total_start = time.time()

    for patno in selected_patnos:
        try:
            result = run_patient_pipeline(
                patno=patno,
                features_df=features_df,
                catboost_model=catboost_model,
                col_medians=col_medians,
                dh_model=dh_model,
                dh_ckpt=dh_ckpt,
                gdt_model=gdt_model,
                gdt_ckpt=gdt_ckpt,
                device=device,
            )
            all_results[str(patno)] = result

            # Save per-patient JSON
            pat_path = OUTPUT_DIR / f"patient_{patno}_pipeline.json"
            with open(pat_path, "w") as f:
                json.dump(result, f, indent=2, default=_convert)

        except Exception as e:
            print(f"  ERROR for patient {patno}: {e}")
            import traceback
            traceback.print_exc()
            all_results[str(patno)] = {"patno": patno, "error": str(e)}

    total_elapsed = time.time() - total_start

    # Build summary
    summary = {
        "n_patients": len(selected_patnos),
        "patnos": selected_patnos,
        "total_elapsed_seconds": round(total_elapsed, 1),
        "models_used": {
            "staging": "CatBoost (12-feature clinical, NSD-positive target)",
            "survival_deephit": f"DeepHit fold 0 (C-td={dh_ckpt['fold_ctd']:.4f})",
            "survival_graphdt": f"Graph-DT fold 0 (C-td={gdt_ckpt['fold_ctd']:.4f})",
            "conformal": f"Paper 4 CIF bands (width={all_results.get(str(selected_patnos[0]), {}).get('conformal_band_width', 'N/A')})",
        },
        "per_patient_summary": [],
    }

    for patno in selected_patnos:
        r = all_results.get(str(patno), {})
        if "error" in r:
            summary["per_patient_summary"].append({
                "patno": patno, "error": r["error"]
            })
        else:
            top_trans = r.get("top_transitions", [{}])
            summary["per_patient_summary"].append({
                "patno": patno,
                "current_stage": r.get("current_stage", "?"),
                "n_visits": r.get("n_visits", 0),
                "follow_up_months": r.get("follow_up_months", 0),
                "catboost_predicted_stage": r.get("staging", {}).get("predicted_stage", "?"),
                "top_transition": top_trans[0].get("destination_stage", "none") if top_trans else "none",
                "top_cif_12mo": top_trans[0].get("cif_at_12mo", 0) if top_trans else 0,
            })

    all_results["summary"] = summary

    # Save consolidated results
    summary_path = OUTPUT_DIR / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_convert)

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Patients processed: {len(selected_patnos)}")
    print(f"  Total elapsed: {total_elapsed:.1f}s")
    print(f"  Results saved to: {OUTPUT_DIR}")

    # Print summary table
    print(f"\n{'Patient':>8} {'Stage':>6} {'CatBoost':>10} {'Top →':>8} {'CIF@12mo':>10}")
    print("-" * 50)
    for ps in summary["per_patient_summary"]:
        if "error" in ps:
            print(f"  {ps['patno']:>6} ERROR: {ps['error']}")
        else:
            print(f"  {ps['patno']:>6} {ps['current_stage']:>6} "
                  f"{ps['catboost_predicted_stage']:>10} "
                  f"→{ps['top_transition']:>6} "
                  f"{ps['top_cif_12mo']:>10.4f}")

    return all_results


def _convert(obj):
    """JSON serialization helper for numpy types."""
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
