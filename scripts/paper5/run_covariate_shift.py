#!/usr/bin/env python3
"""Paper 5: Covariate Shift Detection across Temporal Windows.

For each temporal window, compares train vs test feature distributions:
    1. Per-feature KS test (20 features, baseline values)
    2. Population Stability Index (PSI) per feature
    3. Multivariate Maximum Mean Discrepancy (MMD) with RBF kernel
    4. Shift severity classification (mild/moderate/severe)

Usage:
    python scripts/paper5/run_covariate_shift.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.paper5.temporal_validation import (
    TemporalSplitter,
    compute_covariate_shift,
)

# ── Paths ────────────────────────────────────────────────────────────

FEATURES_PATH = ROOT / "data" / "07_paper3_features" / "longitudinal_features.csv"
DEMOGRAPHICS_PATH = ROOT / "data" / "00_raw" / "Demographics_08Feb2026.csv"

OUTPUT_DIR = ROOT / "outputs" / "paper5" / "covariate_shift"


def run_covariate_shift_analysis(verbose: bool = True):
    """Run full covariate shift analysis across all 4 temporal windows."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("Loading longitudinal features...")
    features_df = pd.read_csv(FEATURES_PATH)

    # 2. Build temporal splitter
    print("\nBuilding temporal splitter...")
    splitter = TemporalSplitter(
        features_df=features_df,
        demographics_path=DEMOGRAPHICS_PATH,
        verbose=verbose,
    )

    # 3. Run shift analysis for each window
    all_ks = {}
    all_psi = {}
    all_mmd = {}
    all_summaries = {}

    for wi in range(splitter.n_windows):
        ws = splitter.get_window(wi)
        wname = f"W{wi + 1}"
        print(f"\n{'=' * 60}")
        print(f"  Covariate Shift Analysis — {wname}")
        print(
            f"  Train: {ws.n_train} ({ws.train_enrollment_range[0]} to "
            f"{ws.train_enrollment_range[1]})"
        )
        print(
            f"  Test: {ws.n_test} ({ws.test_enrollment_range[0]} to "
            f"{ws.test_enrollment_range[1]})"
        )
        print(f"{'=' * 60}")

        per_feature, summary = compute_covariate_shift(
            features_df=features_df,
            train_patnos=ws.train_patnos,
            test_patnos=ws.test_patnos,
            verbose=verbose,
        )

        # Store per-feature results
        ks_results = []
        psi_results = []
        for sr in per_feature:
            sr_dict = {
                "feature": sr.feature,
                "ks_statistic": sr.ks_statistic,
                "ks_pvalue": sr.ks_pvalue,
                "psi": sr.psi,
                "shifted": sr.shifted,
            }
            ks_results.append(sr_dict)
            psi_results.append(
                {
                    "feature": sr.feature,
                    "psi": sr.psi,
                    "shifted_psi": sr.psi > 0.25 if not np.isnan(sr.psi) else False,
                }
            )

        all_ks[wname] = ks_results
        all_psi[wname] = psi_results
        all_mmd[wname] = {
            "mmd_statistic": summary["mmd_statistic"],
            "mmd_pvalue": summary["mmd_pvalue"],
        }
        all_summaries[wname] = {
            "n_features_tested": summary["n_features_tested"],
            "n_shifted": summary["n_shifted"],
            "fraction_shifted": summary["fraction_shifted"],
            "severity": summary["severity"],
            "mmd_statistic": summary["mmd_statistic"],
            "mmd_pvalue": summary["mmd_pvalue"],
            "n_train": ws.n_train,
            "n_test": ws.n_test,
            "train_range": list(ws.train_enrollment_range),
            "test_range": list(ws.test_enrollment_range),
        }

        # Print shifted features
        shifted = [sr for sr in per_feature if sr.shifted]
        if shifted:
            print(f"\n  Shifted features ({len(shifted)}):")
            for sr in shifted:
                print(
                    f"    {sr.feature}: KS={sr.ks_statistic:.3f} "
                    f"(p={sr.ks_pvalue:.2e}), PSI={sr.psi:.3f}"
                )

    # 4. Save results
    def _save_json(data, filename):
        path = OUTPUT_DIR / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_json_convert)
        print(f"  Saved: {path}")

    _save_json(all_ks, "ks_tests_per_window.json")
    _save_json(all_psi, "psi_per_window.json")
    _save_json(all_mmd, "mmd_per_window.json")
    _save_json(all_summaries, "shift_summary.json")

    # 5. Print summary table
    _print_summary(all_summaries)

    return all_summaries


def _json_convert(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _print_summary(summaries: dict):
    """Print formatted summary table."""
    print(f"\n{'=' * 70}")
    print("  COVARIATE SHIFT SUMMARY")
    print(f"{'=' * 70}")

    header = f"{'Window':<8} {'N_train':>8} {'N_test':>7} {'Shifted':>8} {'Frac':>7} {'Severity':>10} {'MMD':>10} {'MMD p':>8}"
    print(header)
    print("-" * len(header))

    for wname in sorted(summaries):
        s = summaries[wname]
        mmd_str = (
            f"{s['mmd_statistic']:.6f}" if s["mmd_statistic"] is not None else "N/A"
        )
        mmd_p_str = f"{s['mmd_pvalue']:.4f}" if s["mmd_pvalue"] is not None else "N/A"
        print(
            f"{wname:<8} {s['n_train']:>8} {s['n_test']:>7} "
            f"{s['n_shifted']:>4}/{s['n_features_tested']:<3} "
            f"{s['fraction_shifted']:>6.1%} {s['severity']:>10} "
            f"{mmd_str:>10} {mmd_p_str:>8}"
        )

    print("\n  Shift thresholds: KS p<0.001 OR PSI>0.25 = shifted feature")
    print("  Severity: <10% mild, 10-30% moderate, >30% severe")


def main():
    run_covariate_shift_analysis(verbose=True)


if __name__ == "__main__":
    main()
