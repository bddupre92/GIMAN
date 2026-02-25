#!/usr/bin/env python3
"""Run expanded Paper 4 analyses: conformal baselines, directional analysis, patient cases.

Three additional analyses to strengthen Paper 4:
  1. Conformal baselines comparison (Marginal, Naive, Bonferroni vs IPCW)
  2. Forward vs backward transition directional analysis
  3. Individual patient case studies (3-5 vignettes)

Outputs:
    outputs/paper4/expanded/
        conformal_baselines.json
        directional_analysis.json
        patient_case_studies.json
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
    TIME_BIN_ENDS,
)
from giman_pipeline.paper3.graph_digital_twin import (
    GraphDeepHitDataset,
    predict_all_graph,
    load_graph_dt_checkpoint,
)
from giman_pipeline.paper3.multistate_markov import STAGE_LABELS, N_STATES
from giman_pipeline.paper4.conformal_survival import (
    CauseSpecificConformal,
    MarginalConformal,
    NaiveConformal,
    BonferroniConformal,
    ConformalTransitionTiming,
    evaluate_directional_conformal,
)

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "07_paper3_features" / "longitudinal_features.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "paper3_checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper4" / "expanded"


def get_test_predictions(model_type, fi, episodes, patient_arrays):
    """Load checkpoint and get test predictions for a fold."""
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
        test_ds = GraphDeepHitDataset(test_eps, patient_arrays, means, stds, pat_to_gidx)
        device = next(model.parameters()).device
        preds = predict_all_graph(
            model, test_ds, device,
            cp["node_baseline"], cp["edge_index"], cp["edge_weight"],
        )

    return preds, test_eps


# =========================================================================
# 1. Conformal Baselines Comparison
# =========================================================================

def run_conformal_baselines(episodes, patient_arrays):
    """Compare 4 conformal methods across all folds."""
    print("\n" + "=" * 60)
    print("1. CONFORMAL BASELINES COMPARISON")
    print("=" * 60)

    methods = {
        "IPCW (proposed)": CauseSpecificConformal,
        "Marginal (pooled)": MarginalConformal,
        "Naive (no IPCW)": NaiveConformal,
        "Bonferroni": BonferroniConformal,
    }

    results = {}
    for cl in [0.90, 0.95]:
        results[f"CL={cl}"] = {}

        for method_name, MethodClass in methods.items():
            fold_results = []

            for fi in range(5):
                for model_type, model_name in [("deephit", "DeepHit"), ("graph_dt", "Graph-DT")]:
                    preds, test_eps = get_test_predictions(
                        model_type, fi, episodes, patient_arrays,
                    )

                    cif = preds["cif"].numpy()
                    durations = np.array([ep.duration_months for ep in test_eps])
                    event_idxs = preds["event_idxs"].numpy()
                    censored = preds["censored"].numpy()

                    # Split 50/50
                    rng = np.random.RandomState(42 + fi)
                    n = len(durations)
                    idx = rng.permutation(n)
                    n_cal = n // 2

                    method = MethodClass(confidence_level=cl)
                    method.calibrate(
                        cif[idx[:n_cal]], durations[idx[:n_cal]],
                        event_idxs[idx[:n_cal]], censored[idx[:n_cal]],
                    )
                    report = method.coverage_report(
                        cif[idx[n_cal:]], durations[idx[n_cal:]],
                        event_idxs[idx[n_cal:]], censored[idx[n_cal:]],
                    )

                    fold_results.append({
                        "model": model_name,
                        "fold": fi,
                        "marginal_coverage": report["marginal_coverage"],
                        "mean_band_width": report["mean_band_width"],
                    })

            # Aggregate
            coverages = [r["marginal_coverage"] for r in fold_results]
            widths = [r["mean_band_width"] for r in fold_results]
            results[f"CL={cl}"][method_name] = {
                "mean_coverage": float(np.mean(coverages)),
                "std_coverage": float(np.std(coverages)),
                "mean_width": float(np.mean(widths)),
                "std_width": float(np.std(widths)),
                "per_fold": fold_results,
            }

            print(f"  {method_name} (CL={cl}): "
                  f"coverage={np.mean(coverages):.4f}±{np.std(coverages):.4f}, "
                  f"width={np.mean(widths):.6f}±{np.std(widths):.6f}")

    return results


# =========================================================================
# 2. Forward vs Backward Directional Analysis
# =========================================================================

def run_directional_analysis(episodes, patient_arrays, features_df):
    """Evaluate conformal coverage separately for forward/backward transitions."""
    print("\n" + "=" * 60)
    print("2. FORWARD VS BACKWARD TRANSITION ANALYSIS")
    print("=" * 60)

    results = []
    for fi in range(5):
        for model_type, model_name in [("deephit", "DeepHit"), ("graph_dt", "Graph-DT")]:
            preds, test_eps = get_test_predictions(
                model_type, fi, episodes, patient_arrays,
            )

            cif = preds["cif"].numpy()
            durations = np.array([ep.duration_months for ep in test_eps])
            event_idxs = preds["event_idxs"].numpy()
            censored = preds["censored"].numpy()

            # Extract source stages from episodes
            source_stages = np.array([ep.current_stage_idx for ep in test_eps])

            dir_result = evaluate_directional_conformal(
                cif, durations, event_idxs, censored, source_stages,
                model_name, fi, confidence_level=0.90,
            )

            for direction, info in dir_result.items():
                results.append({
                    "model": model_name,
                    "fold": fi,
                    "direction": direction,
                    "coverage": info["coverage"],
                    "mean_band_width": info["mean_band_width"],
                    "n_patients": info["n_patients"],
                })

        print(f"  Fold {fi} done")

    # Aggregate
    summary = {}
    for direction in ["forward", "backward"]:
        dir_results = [r for r in results if r["direction"] == direction]
        covs = [r["coverage"] for r in dir_results]
        widths = [r["mean_band_width"] for r in dir_results]
        summary[direction] = {
            "mean_coverage": float(np.mean(covs)),
            "std_coverage": float(np.std(covs)),
            "mean_width": float(np.mean(widths)),
            "std_width": float(np.std(widths)),
            "total_patients": sum(r["n_patients"] for r in dir_results),
        }
        print(f"\n  {direction.upper()}: coverage={np.mean(covs):.4f}±{np.std(covs):.4f}, "
              f"n={summary[direction]['total_patients']}")

    return {"per_fold": results, "summary": summary}


# =========================================================================
# 3. Individual Patient Case Studies
# =========================================================================

def run_patient_case_studies(episodes, patient_arrays, features_df):
    """Generate 3-5 patient case vignettes with conformal bands."""
    print("\n" + "=" * 60)
    print("3. INDIVIDUAL PATIENT CASE STUDIES")
    print("=" * 60)

    # Use fold 0 DeepHit for case studies
    preds, test_eps = get_test_predictions("deephit", 0, episodes, patient_arrays)

    cif = preds["cif"].numpy()
    durations = np.array([ep.duration_months for ep in test_eps])
    event_idxs = preds["event_idxs"].numpy()
    censored = preds["censored"].numpy()

    # Calibrate conformal at 90%
    rng = np.random.RandomState(42)
    n = len(durations)
    idx = rng.permutation(n)
    n_cal = n // 2
    cal_idx, eval_idx = idx[:n_cal], idx[n_cal:]

    csc = CauseSpecificConformal(confidence_level=0.90)
    csc.calibrate(cif[cal_idx], durations[cal_idx], event_idxs[cal_idx], censored[cal_idx])
    bands = csc.predict_bands(cif[eval_idx])

    ctt = ConformalTransitionTiming(confidence_level=0.90)
    ctt.calibrate(cif[cal_idx], durations[cal_idx], event_idxs[cal_idx], censored[cal_idx])
    timing_intervals = ctt.predict_intervals(cif[eval_idx])

    # Select interesting patients from evaluation set
    time_bins = np.array(TIME_BIN_ENDS, dtype=float)

    # Criteria for case study selection:
    # 1. Uncensored (so we have ground truth)
    # 2. Transition to Stage 3 or 4 (most clinically relevant)
    # 3. Diverse source stages
    # 4. Diverse transition times

    candidates = []
    for i_local, i_global in enumerate(eval_idx):
        ep = test_eps[i_global]
        if censored[i_global]:
            continue
        if event_idxs[i_global] not in [3, 4]:  # Stage 3 or 4 destinations
            continue
        candidates.append({
            "local_idx": i_local,
            "global_idx": i_global,
            "patno": ep.patno,
            "source_stage": ep.current_stage_idx,
            "dest_stage": int(event_idxs[i_global]),
            "duration_months": float(durations[i_global]),
        })

    # Select up to 5 diverse cases
    selected = []

    # Try to get: forward to 3, forward to 4, backward from 4 to 3, different durations
    for dest in [3, 4]:
        dest_cands = [c for c in candidates if c["dest_stage"] == dest]
        if not dest_cands:
            continue

        # Sort by duration to get short, medium, long
        dest_cands.sort(key=lambda c: c["duration_months"])
        if len(dest_cands) >= 3:
            selected.append(dest_cands[0])  # Short
            selected.append(dest_cands[len(dest_cands) // 2])  # Medium
            selected.append(dest_cands[-1])  # Long
        else:
            selected.extend(dest_cands)

    # Limit to 5
    selected = selected[:5]
    if not selected:
        print("  WARNING: No suitable case study patients found")
        return {"cases": []}

    # Build case study data
    cases = []
    for case in selected:
        i_local = case["local_idx"]
        dest_k = case["dest_stage"]

        # Get patient demographics
        baseline = features_df[features_df["PATNO"] == case["patno"]].sort_values("visit_number").iloc[0]

        # CIF curves for destination stage
        cif_curve = cif[eval_idx[i_local], dest_k, :].tolist()
        band_lo = bands[i_local, dest_k, :, 0].tolist()
        band_hi = bands[i_local, dest_k, :, 1].tolist()

        # Timing interval
        timing = timing_intervals[i_local].get(dest_k)

        case_data = {
            "patno": int(case["patno"]),
            "source_stage": STAGE_LABELS[case["source_stage"]],
            "dest_stage": STAGE_LABELS[dest_k],
            "actual_duration_months": case["duration_months"],
            "age_at_baseline": float(baseline.get("age_at_baseline", -1)),
            "sex": "Female" if baseline.get("sex", 0) == 1 else "Male",
            "cif_curve": cif_curve,
            "band_lower": band_lo,
            "band_upper": band_hi,
            "time_bins_months": time_bins.tolist(),
            "timing_interval": {
                "lower_months": timing[0] if timing else None,
                "upper_months": timing[1] if timing else None,
            } if timing else None,
            "clinical_interpretation": _generate_interpretation(
                case, cif_curve, timing, time_bins,
            ),
        }
        cases.append(case_data)

        print(f"  Patient {case['patno']}: "
              f"{STAGE_LABELS[case['source_stage']]} → {STAGE_LABELS[dest_k]} "
              f"at {case['duration_months']:.0f} months"
              + (f", timing CI: [{timing[0]:.0f}, {timing[1]:.0f}] months" if timing else ""))

    return {"cases": cases, "n_cases": len(cases)}


def _generate_interpretation(case, cif_curve, timing, time_bins):
    """Generate a clinical interpretation string for a case study."""
    dest = STAGE_LABELS[case["dest_stage"]]
    src = STAGE_LABELS[case["source_stage"]]
    dur = case["duration_months"]

    lines = [
        f"Patient transitioned from Stage {src} to Stage {dest} "
        f"at {dur:.0f} months.",
    ]

    if timing:
        if timing[0] <= dur <= timing[1]:
            lines.append(
                f"The 90% conformal timing interval [{timing[0]:.0f}, {timing[1]:.0f}] months "
                f"correctly covers the actual transition time."
            )
        else:
            lines.append(
                f"The 90% conformal timing interval [{timing[0]:.0f}, {timing[1]:.0f}] months "
                f"does NOT cover the actual transition time — this is expected ~10% of the time."
            )

    # Find time bin closest to actual event
    t_idx = np.searchsorted(time_bins, dur)
    t_idx = min(t_idx, len(cif_curve) - 1)
    pred_cif = cif_curve[t_idx]
    lines.append(
        f"At the transition time, the predicted CIF was {pred_cif:.3f}."
    )

    return " ".join(lines)


# =========================================================================
# Main
# =========================================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("PAPER 4: EXPANDED ANALYSIS")
    print("=" * 60)

    print("\nLoading data...")
    features_df = pd.read_csv(FEATURES_PATH, low_memory=False)
    patient_arrays, _ = build_patient_arrays(features_df)
    episodes = extract_episodes(features_df, verbose=False)
    print(f"  {len(episodes)} episodes")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Conformal baselines
    baseline_results = run_conformal_baselines(episodes, patient_arrays)
    with open(OUTPUT_DIR / "conformal_baselines.json", "w") as f:
        json.dump(baseline_results, f, indent=2, default=str)

    # 2. Directional analysis
    directional_results = run_directional_analysis(episodes, patient_arrays, features_df)
    with open(OUTPUT_DIR / "directional_analysis.json", "w") as f:
        json.dump(directional_results, f, indent=2, default=str)

    # 3. Patient case studies
    case_results = run_patient_case_studies(episodes, patient_arrays, features_df)
    with open(OUTPUT_DIR / "patient_case_studies.json", "w") as f:
        json.dump(case_results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
