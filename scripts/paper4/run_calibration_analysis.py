#!/usr/bin/env python3
"""Run calibration analysis across all 10 checkpoints.

Computes cause-specific ECE, reliability diagrams, and Hosmer-Lemeshow
tests at 1, 3, 5 year horizons for both DeepHit and Graph-DT.

Outputs:
    outputs/paper4/calibration/
        calibration_results_deephit.json
        calibration_results_graph_dt.json
        aggregate_ece.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.paper3.dynamic_deephit import (
    DeepHitDataset,
    build_patient_arrays,
    extract_episodes,
    load_deephit_checkpoint,
    predict_all,
)
from giman_pipeline.paper3.graph_digital_twin import (
    GraphDeepHitDataset,
    load_graph_dt_checkpoint,
    predict_all_graph,
)
from giman_pipeline.paper4.calibration import (
    calibration_result_to_dict,
    evaluate_calibration,
)

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "07_paper3_features" / "longitudinal_features.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "paper3_checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper4" / "calibration"


def run_model_calibration(model_type, episodes, patient_arrays, n_folds=5):
    """Run calibration for one model type across all folds."""
    all_results = []

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
        else:
            path = CHECKPOINT_DIR / "graph_dt" / f"fold{fi}_graph_dt.pt"
            model, cp = load_graph_dt_checkpoint(path)
            test_pats = set(cp["test_pats"])
            means, stds = cp["means"], cp["stds"]
            pat_to_gidx = cp["pat_to_gidx"]
            test_eps = [e for e in episodes if e.patno in test_pats]
            test_ds = GraphDeepHitDataset(
                test_eps, patient_arrays, means, stds, pat_to_gidx
            )
            device = next(model.parameters()).device
            preds = predict_all_graph(
                model,
                test_ds,
                device,
                cp["node_baseline"],
                cp["edge_index"],
                cp["edge_weight"],
            )

        cif = preds["cif"].numpy()
        durations = np.array([ep.duration_months for ep in test_eps])
        event_idxs = preds["event_idxs"].numpy()
        censored = preds["censored"].numpy()

        model_name = "DeepHit" if model_type == "deephit" else "Graph-DT"
        cal_result = evaluate_calibration(
            cif,
            durations,
            event_idxs,
            censored,
            f"{model_name}_fold{fi}",
        )
        all_results.append(cal_result)

        print(
            f"  Fold {fi}: "
            + ", ".join(
                f"{h}={v:.4f}" for h, v in cal_result.aggregate_ece_by_horizon.items()
            )
        )

    return all_results


def main():
    t0 = time.time()
    print("=" * 60)
    print("PAPER 4: CALIBRATION ANALYSIS")
    print("=" * 60)

    print("\nLoading data...")
    features_df = pd.read_csv(FEATURES_PATH, low_memory=False)
    patient_arrays, _ = build_patient_arrays(features_df)
    episodes = extract_episodes(features_df, verbose=False)
    print(f"  {len(episodes)} episodes")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- DEEPHIT CALIBRATION ---")
    dh_results = run_model_calibration("deephit", episodes, patient_arrays)

    print("\n--- GRAPH-DT CALIBRATION ---")
    gdt_results = run_model_calibration("graph_dt", episodes, patient_arrays)

    # Save per-model results
    for model_type, results in [("deephit", dh_results), ("graph_dt", gdt_results)]:
        with open(OUTPUT_DIR / f"calibration_results_{model_type}.json", "w") as f:
            json.dump(
                [calibration_result_to_dict(r) for r in results],
                f,
                indent=2,
                default=str,
            )

    # Aggregate ECE across folds
    aggregate = {}
    for model_name, results in [("DeepHit", dh_results), ("Graph-DT", gdt_results)]:
        agg = {}
        for h in ["1yr", "3yr", "5yr"]:
            fold_eces = [
                r.aggregate_ece_by_horizon.get(h, float("nan")) for r in results
            ]
            valid = [v for v in fold_eces if not np.isnan(v)]
            agg[h] = {
                "mean": float(np.mean(valid)) if valid else float("nan"),
                "std": float(np.std(valid)) if valid else float("nan"),
                "per_fold": fold_eces,
            }
        aggregate[model_name] = agg

    with open(OUTPUT_DIR / "aggregate_ece.json", "w") as f:
        json.dump(aggregate, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("AGGREGATE ECE")
    print("=" * 60)
    for model_name, agg in aggregate.items():
        print(f"\n  {model_name}:")
        for h, vals in agg.items():
            print(f"    {h}: ECE={vals['mean']:.4f} ± {vals['std']:.4f}")

    print(f"\n  Total time: {time.time() - t0:.1f}s")
    print(f"  Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
