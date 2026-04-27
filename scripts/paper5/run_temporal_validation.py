#!/usr/bin/env python3
"""Paper 5: Temporal Validation — Expanding-Window Training & Evaluation.

For each of 4 temporal windows:
    1. Build temporal train/test split via TemporalSplitter
    2. Validate no temporal leakage
    3. Train DeepHit from scratch (same hyperparams as Paper 3)
    4. Train Graph-DT from scratch with inductive graph extension
    5. Compute C-td, IBS, per-transition C-td for both models
    6. Save per-window results + checkpoints

Usage:
    python scripts/paper5/run_temporal_validation.py [--windows 0 1 2 3] [--skip-graphdt]

Runtime estimate: ~2-4 hours total (4 windows x 2 models, no 5-fold CV).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.paper5.temporal_validation import TemporalSplitter
from giman_pipeline.paper5.train_per_window import (
    train_deephit_on_window,
    train_graph_dt_on_window,
)

# ── Paths ────────────────────────────────────────────────────────────

FEATURES_PATH = ROOT / "data" / "07_paper3_features" / "longitudinal_features.csv"
DEMOGRAPHICS_PATH = ROOT / "data" / "00_raw" / "Demographics_08Feb2026.csv"

OUTPUT_DIR = ROOT / "outputs" / "paper5" / "temporal_validation"
CHECKPOINT_DIR = ROOT / "outputs" / "paper5" / "checkpoints"

# Paper 3 random CV reference results
PAPER3_DEEPHIT_CTD = 0.924
PAPER3_GRAPHDT_CTD = 0.904


def run_temporal_validation(
    windows: list[int] | None = None,
    skip_graphdt: bool = False,
    verbose: bool = True,
):
    """Run full temporal validation across expanding windows.

    Args:
        windows: Which windows to run (0-3). None = all 4.
        skip_graphdt: If True, only run DeepHit (faster).
        verbose: Print progress.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load features
    print("Loading longitudinal features...")
    features_df = pd.read_csv(FEATURES_PATH)
    n_patients = features_df["PATNO"].nunique()
    n_visits = len(features_df)
    print(f"  {n_patients} patients, {n_visits} visits")

    # 2. Build temporal splitter
    print("\nBuilding temporal splitter...")
    splitter = TemporalSplitter(
        features_df=features_df,
        demographics_path=DEMOGRAPHICS_PATH,
        verbose=verbose,
    )

    # 3. Determine which windows to run
    if windows is None:
        windows = list(range(splitter.n_windows))

    # 4. Validate all splits
    print("\nValidating temporal splits...")
    for wi in windows:
        validation = splitter.validate_split(wi)
        leakage_ok = "PASS" if validation["no_temporal_leakage"] else "FAIL"
        overlap_ok = "PASS" if validation["patient_overlap"] == 0 else "FAIL"
        print(
            f"  W{wi + 1}: {validation['n_train']} train, {validation['n_test']} test | "
            f"Leakage: {leakage_ok} | Overlap: {overlap_ok} | "
            f"Train: {validation['train_date_range'][0]}-{validation['train_date_range'][1]} | "
            f"Test: {validation['test_date_range'][0]}-{validation['test_date_range'][1]}"
        )

    # 5. Run per-window training (load existing results to merge)
    results_path = OUTPUT_DIR / "temporal_validation_results.json"
    all_results = {}
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
        # Preserve previously computed window results (skip summary key)
        for k, v in existing.items():
            if k.startswith("W"):
                all_results[k] = v
        print(f"  Loaded existing results for: {sorted(k for k in all_results)}")
    total_start = time.time()

    for wi in windows:
        window_start = time.time()
        ws = splitter.get_window(wi)
        print(f"\n{'#' * 60}")
        print(f"  WINDOW {wi + 1} (W{wi + 1})")
        print(
            f"  Train: {ws.n_train} patients ({ws.train_enrollment_range[0]} to "
            f"{ws.train_enrollment_range[1]})"
        )
        print(
            f"  Test: {ws.n_test} patients ({ws.test_enrollment_range[0]} to "
            f"{ws.test_enrollment_range[1]})"
        )
        print(f"{'#' * 60}")

        window_results = {
            "window_idx": wi,
            "window_name": f"W{wi + 1}",
            "n_train": ws.n_train,
            "n_test": ws.n_test,
            "train_enrollment_range": list(ws.train_enrollment_range),
            "test_enrollment_range": list(ws.test_enrollment_range),
        }

        # --- DeepHit ---
        print(f"\n--- DeepHit (W{wi + 1}) ---")
        dh_ckpt = CHECKPOINT_DIR / f"window{wi + 1}_deephit.pt"
        dh_results = train_deephit_on_window(
            features_df=features_df,
            train_patnos=ws.train_patnos,
            test_patnos=ws.test_patnos,
            checkpoint_path=dh_ckpt,
            seed=42 + wi,
            verbose=verbose,
        )
        window_results["deephit"] = {
            "c_td": dh_results["c_td"],
            "ibs": dh_results["ibs"],
            "per_transition_ctd": dh_results["per_transition_ctd"],
            "n_train_episodes": dh_results["n_train_episodes"],
            "n_test_episodes": dh_results["n_test_episodes"],
            "best_val_loss": dh_results["best_val_loss"],
            "degradation_from_cv": PAPER3_DEEPHIT_CTD - dh_results["c_td"],
        }

        # --- Graph-DT ---
        if not skip_graphdt:
            print(f"\n--- Graph-DT (W{wi + 1}) ---")
            gdt_ckpt = CHECKPOINT_DIR / f"window{wi + 1}_graph_dt.pt"
            gdt_results = train_graph_dt_on_window(
                features_df=features_df,
                train_patnos=ws.train_patnos,
                test_patnos=ws.test_patnos,
                checkpoint_path=gdt_ckpt,
                seed=42 + wi,
                verbose=verbose,
            )
            window_results["graph_dt"] = {
                "c_td": gdt_results["c_td"],
                "ibs": gdt_results["ibs"],
                "per_transition_ctd": gdt_results["per_transition_ctd"],
                "graph_stats": gdt_results["graph_stats"],
                "n_train_episodes": gdt_results["n_train_episodes"],
                "n_test_episodes": gdt_results["n_test_episodes"],
                "best_val_loss": gdt_results["best_val_loss"],
                "degradation_from_cv": PAPER3_GRAPHDT_CTD - gdt_results["c_td"],
            }

        window_elapsed = time.time() - window_start
        window_results["elapsed_seconds"] = round(window_elapsed, 1)
        all_results[f"W{wi + 1}"] = window_results

        # Save incrementally after each window
        _save_results(all_results)

        print(f"\n  Window {wi + 1} complete in {window_elapsed / 60:.1f} minutes")

    total_elapsed = time.time() - total_start

    # 6. Build summary
    summary = _build_summary(all_results, skip_graphdt)
    summary["total_elapsed_seconds"] = round(total_elapsed, 1)

    all_results["summary"] = summary
    _save_results(all_results)

    # 7. Print summary table
    _print_summary(all_results, skip_graphdt)

    return all_results


def _save_results(results: dict):
    """Save results JSON incrementally."""
    out_path = OUTPUT_DIR / "temporal_validation_results.json"

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)


def _build_summary(results: dict, skip_graphdt: bool) -> dict:
    """Build aggregate summary across all windows."""
    dh_ctds = []
    gdt_ctds = []

    for key, wr in results.items():
        if not key.startswith("W"):
            continue
        if "deephit" in wr:
            dh_ctds.append(wr["deephit"]["c_td"])
        if "graph_dt" in wr:
            gdt_ctds.append(wr["graph_dt"]["c_td"])

    summary = {
        "deephit": {
            "mean_ctd": float(sum(dh_ctds) / len(dh_ctds)) if dh_ctds else None,
            "min_ctd": float(min(dh_ctds)) if dh_ctds else None,
            "max_ctd": float(max(dh_ctds)) if dh_ctds else None,
            "paper3_cv_ctd": PAPER3_DEEPHIT_CTD,
            "mean_degradation": float(PAPER3_DEEPHIT_CTD - sum(dh_ctds) / len(dh_ctds))
            if dh_ctds
            else None,
        },
    }

    if not skip_graphdt and gdt_ctds:
        summary["graph_dt"] = {
            "mean_ctd": float(sum(gdt_ctds) / len(gdt_ctds)),
            "min_ctd": float(min(gdt_ctds)),
            "max_ctd": float(max(gdt_ctds)),
            "paper3_cv_ctd": PAPER3_GRAPHDT_CTD,
            "mean_degradation": float(
                PAPER3_GRAPHDT_CTD - sum(gdt_ctds) / len(gdt_ctds)
            ),
        }

    return summary


def _print_summary(results: dict, skip_graphdt: bool):
    """Print formatted summary table."""
    print(f"\n{'=' * 70}")
    print("  TEMPORAL VALIDATION SUMMARY")
    print(f"{'=' * 70}")

    header = f"{'Window':<8} {'N_train':>8} {'N_test':>7} {'DH C-td':>9}"
    if not skip_graphdt:
        header += f" {'GDT C-td':>9} {'DH Δ':>7} {'GDT Δ':>7}"
    else:
        header += f" {'DH Δ':>7}"
    print(header)
    print("-" * len(header))

    for key in sorted(k for k in results if k.startswith("W")):
        wr = results[key]
        dh_ctd = wr["deephit"]["c_td"]
        dh_deg = wr["deephit"]["degradation_from_cv"]
        row = f"{key:<8} {wr['n_train']:>8} {wr['n_test']:>7} {dh_ctd:>9.4f}"

        if not skip_graphdt and "graph_dt" in wr:
            gdt_ctd = wr["graph_dt"]["c_td"]
            gdt_deg = wr["graph_dt"]["degradation_from_cv"]
            row += f" {gdt_ctd:>9.4f} {dh_deg:>+7.4f} {gdt_deg:>+7.4f}"
        else:
            row += f" {dh_deg:>+7.4f}"

        print(row)

    print("-" * len(header))

    # Reference line
    ref = f"{'Paper3':>8} {'(5-CV)':>8} {'':>7} {PAPER3_DEEPHIT_CTD:>9.3f}"
    if not skip_graphdt:
        ref += f" {PAPER3_GRAPHDT_CTD:>9.3f}"
    print(ref)

    if "summary" in results:
        s = results["summary"]
        mean_dh = s["deephit"]["mean_ctd"]
        if mean_dh is not None:
            print(
                f"\n  DeepHit mean temporal C-td: {mean_dh:.4f} "
                f"(Δ = {s['deephit']['mean_degradation']:+.4f} from random CV)"
            )
        if (
            not skip_graphdt
            and "graph_dt" in s
            and s["graph_dt"]["mean_ctd"] is not None
        ):
            mean_gdt = s["graph_dt"]["mean_ctd"]
            print(
                f"  Graph-DT mean temporal C-td: {mean_gdt:.4f} "
                f"(Δ = {s['graph_dt']['mean_degradation']:+.4f} from random CV)"
            )

    print(
        f"\n  Total elapsed: {results.get('summary', {}).get('total_elapsed_seconds', 0) / 60:.1f} minutes"
    )


def main():
    parser = argparse.ArgumentParser(description="Paper 5: Temporal Validation")
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=None,
        help="Which windows to run (0-3). Default: all.",
    )
    parser.add_argument(
        "--skip-graphdt",
        action="store_true",
        help="Skip Graph-DT training (faster, DeepHit only).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity.",
    )
    args = parser.parse_args()

    run_temporal_validation(
        windows=args.windows,
        skip_graphdt=args.skip_graphdt,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
