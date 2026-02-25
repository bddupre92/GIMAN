#!/usr/bin/env python3
"""Integration test: Run conformal + calibration on DeepHit fold 0.

Loads the fold 0 DeepHit checkpoint, predicts CIF for test patients,
splits 50/50 into calibration and evaluation, and runs conformal bands
+ calibration analysis.

This verifies the Paper 4 modules work end-to-end before scaling to
all 10 checkpoints.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.paper3.dynamic_deephit import (
    extract_episodes,
    build_patient_arrays,
    DeepHitDataset,
    predict_all,
    load_deephit_checkpoint,
    TIME_BIN_ENDS,
)
from giman_pipeline.paper3.multistate_markov import STAGE_LABELS
from giman_pipeline.paper4.conformal_survival import (
    evaluate_conformal_on_fold,
    conformal_result_to_dict,
    timing_result_to_dict,
)
from giman_pipeline.paper4.calibration import (
    evaluate_calibration,
    calibration_result_to_dict,
)

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "07_paper3_features" / "longitudinal_features.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "paper3_checkpoints" / "deephit" / "fold0_deephit.pt"


def main():
    print("=" * 60)
    print("PAPER 4 INTEGRATION TEST: Fold 0 DeepHit")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    features_df = pd.read_csv(FEATURES_PATH, low_memory=False)
    patient_arrays, _ = build_patient_arrays(features_df)
    episodes = extract_episodes(features_df, verbose=False)
    print(f"  {len(episodes)} episodes from {features_df['PATNO'].nunique()} patients")

    # Load checkpoint
    print(f"\nLoading checkpoint: {CHECKPOINT_PATH.name}")
    model, cp = load_deephit_checkpoint(CHECKPOINT_PATH)
    test_pats = set(cp["test_pats"])
    means = cp["means"]
    stds = cp["stds"]
    print(f"  Test patients: {len(test_pats)}")

    # Build test dataset and predict
    test_eps = [e for e in episodes if e.patno in test_pats]
    test_ds = DeepHitDataset(test_eps, patient_arrays, means, stds)

    device = next(model.parameters()).device
    preds = predict_all(model, test_ds, device)

    cif = preds["cif"].numpy()
    durations = np.array([ep.duration_months for ep in test_eps])
    event_idxs = preds["event_idxs"].numpy()
    censored = preds["censored"].numpy()

    print(f"  Predictions shape: {cif.shape}")
    print(f"  Events: {(~censored).sum()}, Censored: {censored.sum()}")

    # --- Conformal evaluation ---
    print("\n" + "-" * 40)
    print("CONFORMAL PREDICTION BANDS")
    print("-" * 40)

    cif_result, timing_result = evaluate_conformal_on_fold(
        cif_pred=cif,
        durations=durations,
        event_idxs=event_idxs,
        censored=censored,
        model_name="DeepHit",
        fold_idx=0,
        confidence_level=0.90,
    )

    print(f"  Marginal coverage: {cif_result.marginal_coverage:.4f} (target: 0.90)")
    print(f"  Mean band width:   {cif_result.mean_band_width:.4f}")
    print(f"  N calibration:     {cif_result.n_calibration}")
    print(f"  N evaluation:      {cif_result.n_evaluation}")

    print("\n  Per-cause coverage:")
    for k, cov in sorted(cif_result.per_cause_coverage.items()):
        label = STAGE_LABELS[k] if k < len(STAGE_LABELS) else f"Cause {k}"
        width = cif_result.per_cause_band_width.get(k, 0.0)
        print(f"    {label}: coverage={cov:.4f}, band_width={width:.4f}")

    print("\n  Per-horizon coverage:")
    for h, cov in sorted(cif_result.per_horizon_coverage.items()):
        print(f"    {h} months: {cov:.4f}")

    # Timing intervals
    print("\n  Transition timing intervals:")
    for k, info in sorted(timing_result.timing_coverage.items()):
        label = STAGE_LABELS[k] if k < len(STAGE_LABELS) else f"Cause {k}"
        width = timing_result.median_interval_width_months.get(k, float("inf"))
        n = timing_result.n_uncensored_per_cause.get(k, 0)
        print(f"    {label}: coverage={info:.4f}, median_width={width:.1f}mo, n={n}")

    # --- Calibration ---
    print("\n" + "-" * 40)
    print("CALIBRATION ANALYSIS")
    print("-" * 40)

    cal_result = evaluate_calibration(
        cif_pred=cif,
        durations=durations,
        event_idxs=event_idxs,
        censored=censored,
        model_name="DeepHit",
    )

    print("  Aggregate ECE by horizon:")
    for h, ece in cal_result.aggregate_ece_by_horizon.items():
        print(f"    {h}: ECE={ece:.4f}")

    print("\n  Per-cause ECE (1yr horizon):")
    if "1yr" in cal_result.ece_by_cause_horizon:
        for k, ece in sorted(cal_result.ece_by_cause_horizon["1yr"].items()):
            label = STAGE_LABELS[k] if k < len(STAGE_LABELS) else f"Cause {k}"
            print(f"    {label}: ECE={ece:.4f}" if not np.isnan(ece) else f"    {label}: ECE=N/A (insufficient data)")

    # Hosmer-Lemeshow for major causes at 3yr
    print("\n  Hosmer-Lemeshow test (3yr horizon, major causes):")
    if "3yr" in cal_result.hosmer_lemeshow:
        for k in [2, 3, 4]:  # Stage 2B, 3, 4
            hl = cal_result.hosmer_lemeshow["3yr"].get(k, {})
            label = STAGE_LABELS[k]
            chi2 = hl.get("chi2_stat", float("nan"))
            pval = hl.get("p_value", float("nan"))
            if not np.isnan(chi2):
                print(f"    {label}: chi2={chi2:.2f}, p={pval:.4f}")
            else:
                print(f"    {label}: insufficient data")

    # Summary
    print("\n" + "=" * 60)
    coverage_ok = cif_result.marginal_coverage >= 0.85  # Allow 5% slack on fold 0
    print(f"INTEGRATION TEST: {'PASS' if coverage_ok else 'WARN'}")
    print(f"  Coverage {cif_result.marginal_coverage:.4f} {'>=0.85' if coverage_ok else '<0.85 (below target)'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
