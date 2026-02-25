#!/usr/bin/env python3
"""Run conformal survival analysis across all 10 checkpoints.

Loads 5 DeepHit + 5 Graph-DT checkpoints, splits each fold's test
patients 50/50 into calibration and evaluation, runs conformal CIF bands
and timing intervals, aggregates across folds.

Outputs:
    outputs/paper4/conformal/
        conformal_results_deephit.json
        conformal_results_graph_dt.json
        timing_intervals_deephit.json
        timing_intervals_graph_dt.json
        aggregate_summary.json
"""

import json
import sys
import time
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
)
from giman_pipeline.paper3.graph_digital_twin import (
    GraphDeepHitDataset,
    predict_all_graph,
    load_graph_dt_checkpoint,
)
from giman_pipeline.paper3.multistate_markov import STAGE_LABELS
from giman_pipeline.paper4.conformal_survival import (
    evaluate_conformal_on_fold,
    conformal_result_to_dict,
    timing_result_to_dict,
)

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "07_paper3_features" / "longitudinal_features.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "paper3_checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper4" / "conformal"

CONFIDENCE_LEVELS = [0.90, 0.95, 0.80]


def run_model_conformal(
    model_type: str,
    episodes: list,
    patient_arrays: dict,
    n_folds: int = 5,
) -> dict:
    """Run conformal evaluation for one model type across all folds."""

    all_cif_results = []
    all_timing_results = []

    for fi in range(n_folds):
        if model_type == "deephit":
            path = CHECKPOINT_DIR / "deephit" / f"fold{fi}_deephit.pt"
            model, cp = load_deephit_checkpoint(path)
            test_pats = set(cp["test_pats"])
            means, stds = cp["means"], cp["stds"]

            test_eps = [e for e in episodes if e.patno in test_pats]
            test_ds = DeepHitDataset(test_eps, patient_arrays, means, stds)
            device = next(model.parameters()).device
            preds = predict_all(model, test_ds, device)

        elif model_type == "graph_dt":
            path = CHECKPOINT_DIR / "graph_dt" / f"fold{fi}_graph_dt.pt"
            model, cp = load_graph_dt_checkpoint(path)
            test_pats = set(cp["test_pats"])
            means, stds = cp["means"], cp["stds"]
            pat_to_gidx = cp["pat_to_gidx"]
            edge_index = cp["edge_index"]
            edge_weight = cp["edge_weight"]
            node_baseline = cp["node_baseline"]

            test_eps = [e for e in episodes if e.patno in test_pats]
            test_ds = GraphDeepHitDataset(test_eps, patient_arrays, means, stds, pat_to_gidx)
            device = next(model.parameters()).device
            preds = predict_all_graph(
                model, test_ds, device, node_baseline, edge_index, edge_weight,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        cif = preds["cif"].numpy()
        durations = np.array([ep.duration_months for ep in test_eps])
        event_idxs = preds["event_idxs"].numpy()
        censored = preds["censored"].numpy()

        model_name = "DeepHit" if model_type == "deephit" else "Graph-DT"

        for cl in CONFIDENCE_LEVELS:
            cif_result, timing_result = evaluate_conformal_on_fold(
                cif_pred=cif,
                durations=durations,
                event_idxs=event_idxs,
                censored=censored,
                model_name=model_name,
                fold_idx=fi,
                confidence_level=cl,
            )
            all_cif_results.append(cif_result)
            all_timing_results.append(timing_result)

        print(f"  Fold {fi}: {len(test_eps)} test episodes, "
              f"marginal coverage (90%)={[r for r in all_cif_results if r.fold_idx == fi and abs(r.confidence_level - 0.90) < 0.01][0].marginal_coverage:.4f}")

    return {
        "cif_results": all_cif_results,
        "timing_results": all_timing_results,
    }


def aggregate_results(results: list, key: str = "marginal_coverage") -> dict:
    """Aggregate results across folds at each confidence level."""
    agg = {}
    for cl in CONFIDENCE_LEVELS:
        fold_values = [
            getattr(r, key) for r in results
            if abs(r.confidence_level - cl) < 0.01
        ]
        if fold_values:
            agg[f"{cl:.2f}"] = {
                "mean": float(np.mean(fold_values)),
                "std": float(np.std(fold_values)),
                "per_fold": fold_values,
            }
    return agg


def main():
    t0 = time.time()

    print("=" * 60)
    print("PAPER 4: CONFORMAL SURVIVAL ANALYSIS")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    features_df = pd.read_csv(FEATURES_PATH, low_memory=False)
    patient_arrays, _ = build_patient_arrays(features_df)
    episodes = extract_episodes(features_df, verbose=False)
    print(f"  {len(episodes)} episodes from {features_df['PATNO'].nunique()} patients")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- DeepHit ---
    print("\n" + "-" * 40)
    print("DEEPHIT CONFORMAL")
    print("-" * 40)
    dh_results = run_model_conformal("deephit", episodes, patient_arrays)

    # --- Graph-DT ---
    print("\n" + "-" * 40)
    print("GRAPH-DT CONFORMAL")
    print("-" * 40)
    gdt_results = run_model_conformal("graph_dt", episodes, patient_arrays)

    # Save results
    for model_type, results in [("deephit", dh_results), ("graph_dt", gdt_results)]:
        with open(OUTPUT_DIR / f"conformal_results_{model_type}.json", "w") as f:
            json.dump(
                [conformal_result_to_dict(r) for r in results["cif_results"]],
                f, indent=2, default=str,
            )
        with open(OUTPUT_DIR / f"timing_intervals_{model_type}.json", "w") as f:
            json.dump(
                [timing_result_to_dict(r) for r in results["timing_results"]],
                f, indent=2, default=str,
            )

    # Aggregate summary
    summary = {}
    for model_type, results in [("DeepHit", dh_results), ("Graph-DT", gdt_results)]:
        summary[model_type] = {
            "marginal_coverage": aggregate_results(results["cif_results"], "marginal_coverage"),
            "mean_band_width": aggregate_results(results["cif_results"], "mean_band_width"),
        }

    with open(OUTPUT_DIR / "aggregate_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    for model_name in ["DeepHit", "Graph-DT"]:
        print(f"\n  {model_name}:")
        for cl_key, vals in summary[model_name]["marginal_coverage"].items():
            print(f"    Coverage (CL={cl_key}): {vals['mean']:.4f} ± {vals['std']:.4f}")
        for cl_key, vals in summary[model_name]["mean_band_width"].items():
            print(f"    Band width (CL={cl_key}): {vals['mean']:.4f} ± {vals['std']:.4f}")

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
