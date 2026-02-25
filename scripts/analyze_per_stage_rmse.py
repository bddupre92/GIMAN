#!/usr/bin/env python3
"""Per-Stage RMSE Analysis for Paper 2.

Extracts and analyzes per_stage_rmse from benchmark results to show that
stage-conditioned imputation improves on minority stages (1, 2B, 4)
even when aggregate RMSE is comparable or worse.

This script supports the "imputation-utility paradox" narrative:
StageConditioned wins downstream because it allocates imputation capacity
to clinically important minority stages rather than optimizing aggregate
RMSE (dominated by Stage 0 at 64.4%).

Usage:
    python scripts/analyze_per_stage_rmse.py
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = (
    PROJECT_ROOT / "outputs" / "paper2_benchmark"
    / "imputation_benchmark_results_combined.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "outputs" / "paper2_benchmark" / "per_stage_analysis.json"
)

STAGE_NAMES = {
    "0": "Stage 0 (64.4%)",
    "1": "Stage 1 (3.0%)",
    "2": "Stage 2B (9.5%)",
    "3": "Stage 3 (22.1%)",
    "4": "Stage 4 (0.8%)",
}
MINORITY_STAGES = ["1", "2", "4"]  # Key stages for clinical utility
MAJORITY_STAGES = ["0", "3"]


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def extract_per_stage(raw_data: dict, frac_key: str) -> dict:
    """Extract per-stage RMSE for each model, averaged across runs."""
    result = {}
    frac = raw_data.get(frac_key, {})
    for model, runs in frac.items():
        if not isinstance(runs, list):
            continue
        stage_rmses = {}
        for stage in ["0", "1", "2", "3", "4"]:
            values = [r.get("per_stage_rmse", {}).get(stage) for r in runs]
            values = [v for v in values if v is not None]
            if values:
                stage_rmses[stage] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
        if stage_rmses:
            result[model] = stage_rmses
    return result


def compute_advantage(per_stage: dict) -> dict:
    """Compute StageConditioned advantage over Vanilla per stage."""
    sc = per_stage.get("GIMIN_StageConditioned", {})
    vanilla = per_stage.get("GIMIN_Vanilla", {})
    if not sc or not vanilla:
        return {}

    advantage = {}
    for stage in ["0", "1", "2", "3", "4"]:
        if stage in sc and stage in vanilla:
            sc_rmse = sc[stage]["mean"]
            v_rmse = vanilla[stage]["mean"]
            delta = v_rmse - sc_rmse  # Positive = SC wins
            pct = 100 * delta / v_rmse if v_rmse > 0 else 0
            advantage[stage] = {
                "vanilla_rmse": v_rmse,
                "stage_cond_rmse": sc_rmse,
                "delta": delta,
                "pct_improvement": pct,
                "winner": "StageConditioned" if delta > 0 else "Vanilla",
            }
    return advantage


def main():
    data = load_results()
    raw = data["raw"]

    all_analysis = {}

    for frac_key in sorted(raw.keys()):
        print(f"\n{'=' * 80}")
        print(f"  {frac_key}")
        print(f"{'=' * 80}")

        per_stage = extract_per_stage(raw, frac_key)
        advantage = compute_advantage(per_stage)

        # Print table
        models_order = [
            "GIMIN_Vanilla", "GIMIN_StageConditioned",
            "GIMIN_StageGraphOnly", "GIMIN_StageDecoderOnly",
            "MissForest", "MICE", "KNN", "GAIN", "SAITS", "MIWAE",
            "Mean", "Median",
        ]
        available_models = [m for m in models_order if m in per_stage]

        # Header
        header = f"{'Model':<30s}"
        for stage_id in ["0", "1", "2", "3", "4"]:
            header += f" {STAGE_NAMES.get(stage_id, stage_id):>18s}"
        header += f" {'Minority Avg':>14s}"
        print(header)
        print("─" * 140)

        for model in available_models:
            stages = per_stage[model]
            row = f"{model:<30s}"
            minority_vals = []
            for stage_id in ["0", "1", "2", "3", "4"]:
                if stage_id in stages:
                    rmse = stages[stage_id]["mean"]
                    row += f" {rmse:>18.1f}"
                    if stage_id in MINORITY_STAGES:
                        minority_vals.append(rmse)
                else:
                    row += f" {'N/A':>18s}"

            if minority_vals:
                row += f" {np.mean(minority_vals):>14.1f}"
            print(row)

        # Print advantage analysis
        if advantage:
            print(f"\n  StageConditioned vs Vanilla advantage:")
            for stage_id in ["0", "1", "2", "3", "4"]:
                if stage_id in advantage:
                    a = advantage[stage_id]
                    marker = "**" if a["delta"] > 0 else ""
                    print(
                        f"    {STAGE_NAMES.get(stage_id, stage_id)}: "
                        f"Δ = {a['delta']:+.1f} ({a['pct_improvement']:+.1f}%) "
                        f"[{a['winner']}] {marker}"
                    )

            # Minority vs majority
            minority_delta = np.mean([
                advantage[s]["delta"] for s in MINORITY_STAGES if s in advantage
            ])
            majority_delta = np.mean([
                advantage[s]["delta"] for s in MAJORITY_STAGES if s in advantage
            ])
            print(
                f"\n    Minority stages (1, 2B, 4) avg Δ: {minority_delta:+.1f}"
            )
            print(
                f"    Majority stages (0, 3) avg Δ:     {majority_delta:+.1f}"
            )
            if minority_delta > 0 and majority_delta < 0:
                print(
                    "    → CONFIRMED: StageConditioned allocates capacity to "
                    "minority stages at the cost of majority-stage RMSE"
                )

        all_analysis[frac_key] = {
            "per_stage": {
                model: {s: v["mean"] for s, v in stages.items()}
                for model, stages in per_stage.items()
            },
            "advantage": advantage,
        }

    # Save analysis
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_analysis, f, indent=2)
    print(f"\n\nAnalysis saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
