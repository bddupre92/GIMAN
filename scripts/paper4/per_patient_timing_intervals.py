#!/usr/bin/env python3
"""Paper 3+4 Workstream 3 — Per-patient conformal timing intervals at scale.

Extends the existing fold-aggregated conformal timing analysis
(run_conformal_survival.py) to emit per-patient records for all
patients with observed transitions. The 5 existing vignettes at
outputs/paper4/expanded/patient_case_studies.json become "featured
illustrations" within a stratified 4-phenotype-class analysis that
matches the Hu et al. 2025 npj Digital Medicine precedent (22,000-
MS-patient conformal case study).

Steps:
  1. Load 10 fold checkpoints (5 DeepHit + 5 Graph-DT).
  2. For each fold, re-run the conformal calibration/evaluation split
     (same 50/50 as run_conformal_survival.py).
  3. Record per-patient timing intervals (lo, hi, observed_time, covered).
  4. Cluster patients into 4 phenotype classes:
       - rapid:    first forward transition <= 6 months
       - stable:   no forward transition in >= 24 months
       - skip:     non-adjacent stage transition (2B->4, 3->5, etc.)
       - regressor: backward transition as first event
  5. Emit per-class aggregate stats (coverage rate at 90% CL, median
     width, IQR) + a JSON with every patient's record.

Outputs:
  outputs/paper4/conformal/per_patient_intervals_all.json
    -- Master per-patient record (phenotype class + per-model timing
       intervals for every observed transition)
  outputs/paper4/conformal/per_class_aggregate.json
    -- 4 phenotype classes x 2 models x (coverage, median_width, IQR)
  outputs/paper4/conformal/phenotype_class_labels.csv
    -- PATNO, phenotype_class, first_transition_dir, first_event_months

Usage:
  .venv/bin/python scripts/paper4/per_patient_timing_intervals.py
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.paper3.dynamic_deephit import (  # noqa: E402
    DeepHitDataset,
    _get_time_bin,
    build_patient_arrays,
    extract_episodes,
    load_deephit_checkpoint,
    predict_all,
)
from giman_pipeline.paper3.graph_digital_twin import (  # noqa: E402
    GraphDeepHitDataset,
    load_graph_dt_checkpoint,
    predict_all_graph,
)
from giman_pipeline.paper4.conformal_survival import (  # noqa: E402
    ConformalTransitionTiming,
)

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "07_paper3_features" / "longitudinal_features.csv"
TRANSITIONS_PATH = DATA_DIR / "06_longitudinal_staging" / "transition_events.csv"
LONG_NSD_PATH = DATA_DIR / "06_longitudinal_staging" / "longitudinal_nsd_iss.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "paper3_checkpoints"
OUT_DIR = PROJECT_ROOT / "outputs" / "paper4" / "conformal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
CONFIDENCE_LEVEL = 0.90
SEED = 42

# Stage string -> integer mapping (matches Paper 3 conventions)
STAGE_STR_TO_IDX = {"0": 0, "1": 1, "2B": 2, "3": 3, "4": 4, "5": 5, "6": 6}
STAGE_IDX_TO_STR = {v: k for k, v in STAGE_STR_TO_IDX.items()}


def classify_phenotype(
    patno: int,
    trans_df: pd.DataFrame,
) -> tuple[str, float, str]:
    """Assign phenotype class based on patient's transition history.

    Returns (class, first_event_months, first_transition_direction).
    """
    pt = trans_df[trans_df["PATNO"] == patno].sort_values("months_from_baseline_src")
    if pt.empty:
        return "stable", float("inf"), "none"

    first = pt.iloc[0]
    dt_months = float(first["time_interval_months"])
    src_idx = STAGE_STR_TO_IDX.get(str(first["source_stage"]), -1)
    dst_idx = STAGE_STR_TO_IDX.get(str(first["dest_stage"]), -1)

    if dst_idx < src_idx:
        return "regressor", dt_months, "backward"

    # Forward or skip
    step = dst_idx - src_idx
    if step >= 2:
        direction = "skip"
    else:
        direction = "forward"

    if direction == "skip":
        return "skip", dt_months, "skip"
    if dt_months <= 6.0:
        return "rapid", dt_months, "forward"
    if dt_months >= 24.0:
        return "stable", dt_months, "forward_late"
    # Default forward progressor bucket
    return "rapid" if dt_months <= 12.0 else "stable", dt_months, "forward"


def _load_dh_fold(fold_idx: int, episodes, patient_arrays):
    cpath = CHECKPOINT_DIR / "deephit" / f"fold{fold_idx}_deephit.pt"
    model, cp = load_deephit_checkpoint(cpath, device=DEVICE)
    test_pats = set(cp["test_pats"])
    val_pats = set(cp.get("val_pats", []))
    # Use all fold-test episodes as the "test" set; split half for calibration.
    test_eps = [e for e in episodes if e.patno in test_pats]
    ds = DeepHitDataset(test_eps, patient_arrays, cp["means"], cp["stds"])
    preds = predict_all(model, ds, DEVICE)
    return test_eps, preds, cp


def _load_gdt_fold(fold_idx: int, episodes, patient_arrays):
    cpath = CHECKPOINT_DIR / "graph_dt" / f"fold{fold_idx}_graph_dt.pt"
    model, cp = load_graph_dt_checkpoint(cpath, device=DEVICE)
    test_pats = set(cp["test_pats"])
    pat_to_gidx = cp["pat_to_gidx"]
    test_eps = [
        e for e in episodes if e.patno in test_pats and e.patno in pat_to_gidx
    ]
    ds = GraphDeepHitDataset(
        test_eps, patient_arrays, cp["means"], cp["stds"], pat_to_gidx
    )
    preds = predict_all_graph(
        model, ds, DEVICE,
        cp["node_baseline"], cp["edge_index"], cp["edge_weight"],
    )
    return test_eps, preds, cp


def _fold_per_patient_intervals(
    test_eps, preds, confidence_level, seed,
) -> list[dict]:
    """Split fold-test into calibration (50%) + evaluation (50%), fit
    conformal, emit per-episode intervals for evaluation set."""
    cif = preds["cif"].numpy()
    # Build per-episode features: durations (months), event_idx, censored
    durations = np.array([e.duration_months for e in test_eps])
    event_idxs = np.array([
        e.event_stage_idx if not e.censored else 0 for e in test_eps
    ])
    censored = np.array([e.censored for e in test_eps])
    patnos = np.array([e.patno for e in test_eps])
    src_stages = np.array([e.current_stage_idx for e in test_eps])

    n = len(test_eps)
    if n < 10:
        return []

    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_cal = n // 2
    cal_idx = idx[:n_cal]
    eval_idx = idx[n_cal:]

    ct = ConformalTransitionTiming(confidence_level=confidence_level)
    ct.calibrate(
        cif_pred=cif[cal_idx],
        durations=durations[cal_idx],
        event_idxs=event_idxs[cal_idx],
        censored=censored[cal_idx],
    )
    eval_intervals = ct.predict_intervals(cif[eval_idx])

    records = []
    for local_i, global_i in enumerate(eval_idx):
        if censored[global_i]:
            continue
        cause = int(event_idxs[global_i])
        patient_intervals = eval_intervals[local_i]
        iv = patient_intervals.get(cause)
        if iv is None:
            continue
        lo, hi = iv
        covered = (lo <= float(durations[global_i]) <= hi)
        records.append({
            "patno": int(patnos[global_i]),
            "source_stage_idx": int(src_stages[global_i]),
            "dest_stage_idx": cause,
            "observed_time_months": float(durations[global_i]),
            "timing_lo_months": float(lo),
            "timing_hi_months": float(hi),
            "interval_width_months": float(hi - lo),
            "covered": bool(covered),
        })
    return records


def main():
    print("Loading episodes...")
    features = pd.read_csv(FEATURES_PATH, low_memory=False)
    transitions = pd.read_csv(TRANSITIONS_PATH, low_memory=False)
    patient_arrays, _ = build_patient_arrays(features)
    episodes = extract_episodes(features, verbose=False)
    print(f"  n_episodes={len(episodes)}  n_patients={features['PATNO'].nunique()}")
    print(f"  n_transitions_observed={len(transitions)}  n_patients_with_transitions={transitions['PATNO'].nunique()}")

    # Phenotype classification (all patients with >=1 transition)
    print("\nClassifying phenotype classes...")
    transitioning = transitions["PATNO"].unique()
    pheno_records = []
    for patno in transitioning:
        cls, first_mo, direction = classify_phenotype(int(patno), transitions)
        pheno_records.append({
            "PATNO": int(patno),
            "phenotype_class": cls,
            "first_event_months": first_mo,
            "first_transition_direction": direction,
        })
    pheno_df = pd.DataFrame(pheno_records)
    pheno_df.to_csv(OUT_DIR / "phenotype_class_labels.csv", index=False)
    class_counts = pheno_df["phenotype_class"].value_counts().to_dict()
    print(f"  Class counts: {class_counts}")
    print(f"  Saved: {OUT_DIR / 'phenotype_class_labels.csv'}")

    # Run per-fold per-patient conformal
    print("\nComputing per-patient timing intervals for DeepHit...")
    all_dh_records = []
    for fi in range(5):
        print(f"  Fold {fi}...")
        test_eps, preds, cp = _load_dh_fold(fi, episodes, patient_arrays)
        recs = _fold_per_patient_intervals(test_eps, preds, CONFIDENCE_LEVEL, SEED + fi)
        for r in recs:
            r["fold"] = fi
            r["model"] = "deephit"
        all_dh_records.extend(recs)
        print(f"    n_evaluated_uncensored={len(recs)}")

    print("\nComputing per-patient timing intervals for Graph-DT...")
    all_gdt_records = []
    for fi in range(5):
        print(f"  Fold {fi}...")
        test_eps, preds, cp = _load_gdt_fold(fi, episodes, patient_arrays)
        recs = _fold_per_patient_intervals(test_eps, preds, CONFIDENCE_LEVEL, SEED + fi + 100)
        for r in recs:
            r["fold"] = fi
            r["model"] = "graph_dt"
        all_gdt_records.extend(recs)
        print(f"    n_evaluated_uncensored={len(recs)}")

    all_records = all_dh_records + all_gdt_records
    print(f"\nTotal per-patient timing records: {len(all_records)}")
    print(f"  DeepHit: {len(all_dh_records)}")
    print(f"  Graph-DT: {len(all_gdt_records)}")

    # Attach phenotype class and save master JSON
    patno_to_class = dict(zip(pheno_df["PATNO"], pheno_df["phenotype_class"]))
    for r in all_records:
        r["phenotype_class"] = patno_to_class.get(r["patno"], "stable")

    master_path = OUT_DIR / "per_patient_intervals_all.json"
    with open(master_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"  Saved: {master_path}")

    # Per-class aggregate analysis
    print("\nPer-class aggregate analysis...")
    agg = {}
    records_df = pd.DataFrame(all_records)
    for cls in sorted(records_df["phenotype_class"].unique()):
        agg[cls] = {}
        for model in ["deephit", "graph_dt"]:
            subset = records_df[(records_df["phenotype_class"] == cls) & (records_df["model"] == model)]
            if len(subset) < 2:
                agg[cls][model] = {
                    "n_records": int(len(subset)),
                    "coverage": None,
                    "median_width_months": None,
                    "iqr_width_months": None,
                }
                continue
            widths = subset["interval_width_months"].values
            agg[cls][model] = {
                "n_records": int(len(subset)),
                "n_unique_patients": int(subset["patno"].nunique()),
                "coverage": float(subset["covered"].mean()),
                "median_width_months": float(np.median(widths)),
                "iqr_width_months": float(np.quantile(widths, 0.75) - np.quantile(widths, 0.25)),
                "mean_width_months": float(np.mean(widths)),
            }
        print(f"  {cls}: DH {agg[cls]['deephit']}  GDT {agg[cls]['graph_dt']}")

    agg_path = OUT_DIR / "per_class_aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"  Saved: {agg_path}")

    # Headline summary
    print("\n" + "=" * 70)
    print("HEADLINE SUMMARY")
    print("=" * 70)
    print(f"Total transitioning patients: {len(transitioning)}")
    print(f"Total timing-interval records (covered): {records_df['covered'].sum()}/{len(records_df)} = {100 * records_df['covered'].mean():.1f}%")
    print(f"\nBy phenotype class (DeepHit):")
    for cls in sorted(records_df["phenotype_class"].unique()):
        dh_agg = agg[cls]["deephit"]
        if dh_agg.get("coverage") is not None:
            print(f"  {cls:10s} n={dh_agg['n_records']:4d}  cov={dh_agg['coverage']:.3f}  median_width={dh_agg['median_width_months']:.1f}mo")
    print(f"\nBy phenotype class (Graph-DT):")
    for cls in sorted(records_df["phenotype_class"].unique()):
        gdt_agg = agg[cls]["graph_dt"]
        if gdt_agg.get("coverage") is not None:
            print(f"  {cls:10s} n={gdt_agg['n_records']:4d}  cov={gdt_agg['coverage']:.3f}  median_width={gdt_agg['median_width_months']:.1f}mo")

    print(f"\nArtifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
