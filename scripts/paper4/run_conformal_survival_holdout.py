#!/usr/bin/env python3
"""Paper 4 holdout rerun: conformal survival analysis on seed=2026 holdout.

Forks scripts/paper4/run_conformal_survival.py. The Paper 3 holdout checkpoints
were trained ONCE on the 1,520-patient dev set; this runner now calibrates the
conformal bands on a 50/50 split of that dev set (seed=2026) and evaluates on
the 380-patient holdout that was never seen during training.

Key discipline: the calibration split uses dev-set patients, so the 380
holdout patients are never touched during training OR calibration — a genuine
single-use generalization evaluation.

Outputs
-------
outputs/paper4_holdout_v1/
    conformal_results.json           — marginal + per-cause + per-horizon coverage
    timing_intervals.json            — per-cause timing coverage
    subgroup_coverage.json           — conditional coverage by sex and age bin
    holdout_report.md                — human-readable summary
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.paper3.dynamic_deephit import (  # noqa: E402
    DeepHitDataset,
    build_patient_arrays,
    extract_episodes,
    predict_all,
)
from giman_pipeline.paper3.graph_digital_twin import (  # noqa: E402
    GraphDeepHitDataset,
    predict_all_graph,
)
from giman_pipeline.paper3.multistate_markov import STAGE_LABELS  # noqa: E402
from giman_pipeline.paper4.conformal_survival import (  # noqa: E402
    CauseSpecificConformal,
    ConformalTransitionTiming,
    conformal_result_to_dict,
    timing_result_to_dict,
    ConformalSurvivalResult,
    TimingIntervalResult,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "07_paper3_features" / "longitudinal_features.csv"
HOLDOUT_JSON = DATA_DIR / "06_longitudinal_staging" / "holdout_v1_patnos.json"

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "paper3_checkpoints" / "holdout_v1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper4_holdout_v1"

# Reuse Paper-3 holdout seed for calibration split — distinct from MODEL seed
CALIB_SEED = 2026
CONFIDENCE_LEVELS = [0.80, 0.90, 0.95]


def _load_deephit(device: torch.device):
    from giman_pipeline.paper3.dynamic_deephit import DynamicDeepHit, N_STATES, N_TIME_BINS

    ckpt = torch.load(
        CHECKPOINT_DIR / "deephit.pt", weights_only=False, map_location=device
    )
    model = DynamicDeepHit(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        n_gru_layers=ckpt["n_gru_layers"],
        n_causes=N_STATES,
        n_time_bins=N_TIME_BINS,
        dropout=ckpt.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def _load_graph_dt(device: torch.device):
    from giman_pipeline.paper3.graph_digital_twin import GraphDigitalTwin
    from giman_pipeline.paper3.dynamic_deephit import N_STATES, N_TIME_BINS

    ckpt = torch.load(
        CHECKPOINT_DIR / "graph_dt.pt", weights_only=False, map_location=device
    )
    model = GraphDigitalTwin(
        input_dim=ckpt["input_dim"],
        n_baseline_features=ckpt["n_baseline_features"],
        hidden_dim=ckpt["hidden_dim"],
        n_gru_layers=ckpt["n_gru_layers"],
        gat_heads=ckpt.get("gat_heads", 4),
        gat_layers=ckpt.get("gat_layers", 2),
        n_causes=N_STATES,
        n_time_bins=N_TIME_BINS,
        dropout=ckpt.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def _predict_cif_bulk(model_type: str, episodes: list, patient_arrays, means, stds,
                     device, ckpt):
    """Run predict_all / predict_all_graph on the provided episodes list."""
    if model_type == "deephit":
        ds = DeepHitDataset(episodes, patient_arrays, means, stds)
        preds = predict_all(model_ref[0], ds, device)
    elif model_type == "graph_dt":
        pat_to_gidx = ckpt["pat_to_gidx"]
        ds = GraphDeepHitDataset(episodes, patient_arrays, means, stds, pat_to_gidx)
        preds = predict_all_graph(
            model_ref[0],
            ds,
            device,
            ckpt["node_baseline"],
            ckpt["edge_index"],
            ckpt["edge_weight"],
        )
    else:
        raise ValueError(model_type)
    return preds


def run_conformal(
    model_type: str,
    episodes: list,
    patient_arrays: dict,
    means: np.ndarray,
    stds: np.ndarray,
    ckpt: dict,
    device: torch.device,
    model_name: str,
) -> dict:
    """Calibrate on dev-eval split, evaluate on 380-pt holdout.

    Pipeline:
    1. Split dev set 50/50 into calibration and eval-on-dev (seed=2026)
    2. Run model inference on calibration dev episodes → CIFs
    3. Calibrate CauseSpecificConformal + ConformalTransitionTiming
    4. Run model inference on holdout episodes → CIFs
    5. Apply bands / intervals, report coverage
    """
    # --- Split dev patients ---
    dev_pats = ckpt["dev_pats"]
    rng = np.random.RandomState(CALIB_SEED)
    dev_list = sorted(dev_pats)
    rng.shuffle(dev_list)
    n_calib = len(dev_list) // 2
    calib_pats = set(dev_list[:n_calib])
    # Dev-eval patients are unused here (not holdout) — we reserve them for
    # potential future ablation but the primary evaluation is on the 380 holdout

    calib_eps = [e for e in episodes if e.patno in calib_pats]
    holdout_pats = set(ckpt["holdout_pats"])
    holdout_eps = [e for e in episodes if e.patno in holdout_pats]

    print(f"  Calibration episodes: {len(calib_eps)}")
    print(f"  Holdout episodes:     {len(holdout_eps)}")

    # --- Run inference ---
    if model_type == "deephit":
        calib_ds = DeepHitDataset(calib_eps, patient_arrays, means, stds)
        holdout_ds = DeepHitDataset(holdout_eps, patient_arrays, means, stds)
        calib_preds = predict_all(model_ref[0], calib_ds, device)
        holdout_preds = predict_all(model_ref[0], holdout_ds, device)
    elif model_type == "graph_dt":
        pat_to_gidx = ckpt["pat_to_gidx"]
        calib_ds = GraphDeepHitDataset(
            calib_eps, patient_arrays, means, stds, pat_to_gidx
        )
        holdout_ds = GraphDeepHitDataset(
            holdout_eps, patient_arrays, means, stds, pat_to_gidx
        )
        calib_preds = predict_all_graph(
            model_ref[0],
            calib_ds,
            device,
            ckpt["node_baseline"],
            ckpt["edge_index"],
            ckpt["edge_weight"],
        )
        holdout_preds = predict_all_graph(
            model_ref[0],
            holdout_ds,
            device,
            ckpt["node_baseline"],
            ckpt["edge_index"],
            ckpt["edge_weight"],
        )
    else:
        raise ValueError(model_type)

    # --- Prepare arrays ---
    cif_cal = calib_preds["cif"].numpy()
    dur_cal = np.array([e.duration_months for e in calib_eps])
    ev_cal = calib_preds["event_idxs"].numpy()
    cens_cal = calib_preds["censored"].numpy()

    cif_eval = holdout_preds["cif"].numpy()
    dur_eval = np.array([e.duration_months for e in holdout_eps])
    ev_eval = holdout_preds["event_idxs"].numpy()
    cens_eval = holdout_preds["censored"].numpy()

    # --- Per-confidence-level conformal evaluation ---
    results = []
    timing_results = []
    bands_per_cl = {}
    for cl in CONFIDENCE_LEVELS:
        csc = CauseSpecificConformal(confidence_level=cl)
        csc.calibrate(cif_cal, dur_cal, ev_cal, cens_cal)
        bands_per_cl[cl] = csc.predict_bands(cif_eval)
        report = csc.coverage_report(cif_eval, dur_eval, ev_eval, cens_eval)

        result = ConformalSurvivalResult(
            model_name=model_name,
            fold_idx=-1,  # "holdout" — no fold
            confidence_level=cl,
            n_calibration=len(calib_eps),
            n_evaluation=len(holdout_eps),
            marginal_coverage=report["marginal_coverage"],
            mean_band_width=report["mean_band_width"],
            per_cause_coverage=report["per_cause_coverage"],
            per_cause_band_width=report["per_cause_band_width"],
            per_horizon_coverage=report["per_horizon_coverage"],
        )
        results.append(result)

        ctt = ConformalTransitionTiming(confidence_level=cl)
        ctt.calibrate(cif_cal, dur_cal, ev_cal, cens_cal)
        trep = ctt.timing_coverage(cif_eval, dur_eval, ev_eval, cens_eval)
        tr = TimingIntervalResult(
            model_name=model_name,
            fold_idx=-1,
            confidence_level=cl,
        )
        for k, info in trep.items():
            tr.timing_coverage[k] = info["coverage"]
            tr.median_interval_width_months[k] = info["median_width_months"]
            tr.n_uncensored_per_cause[k] = info["n_evaluated"]
        timing_results.append(tr)

        print(
            f"    CL={cl:.2f}: marginal={report['marginal_coverage']:.4f} "
            f"mean_width={report['mean_band_width']:.4f}"
        )

    return {
        "cif_results": results,
        "timing_results": timing_results,
        "cif_eval": cif_eval,
        "dur_eval": dur_eval,
        "ev_eval": ev_eval,
        "cens_eval": cens_eval,
        "holdout_eps": holdout_eps,
        "bands_per_cl": bands_per_cl,
    }


def conditional_coverage_by_subgroup(
    conformal_bands: np.ndarray,
    holdout_eps: list,
    cif_eval: np.ndarray,
    dur_eval: np.ndarray,
    ev_eval: np.ndarray,
    cens_eval: np.ndarray,
    features_df: pd.DataFrame,
) -> dict:
    """Compute per-subgroup (sex, age bin) conformal coverage on holdout."""
    from giman_pipeline.paper4.conformal_survival import TIME_BIN_ENDS
    from giman_pipeline.paper3.multistate_markov import N_STATES

    n_eval = len(holdout_eps)
    patnos = np.array([e.patno for e in holdout_eps])

    pat_baseline = features_df[features_df["months_from_baseline"] == 0.0]
    pat_baseline = pat_baseline.drop_duplicates("PATNO").set_index("PATNO")

    def _subgroup_coverage(mask: np.ndarray, cl_idx: int):
        if mask.sum() < 10:
            return None
        covered = 0
        total = 0
        time_bins_months = np.array(TIME_BIN_ENDS, dtype=float)
        for i in np.where(mask)[0]:
            for k in range(N_STATES):
                for t_idx in range(len(TIME_BIN_ENDS)):
                    t_months = time_bins_months[t_idx]
                    if cens_eval[i] and dur_eval[i] < t_months:
                        continue
                    if not cens_eval[i] and ev_eval[i] == k and dur_eval[i] <= t_months:
                        cif_obs = 1.0
                    elif not cens_eval[i] and ev_eval[i] != k and dur_eval[i] <= t_months:
                        cif_obs = 0.0
                    else:
                        cif_obs = 0.0
                    total += 1
                    lo = conformal_bands[cl_idx][i, k, t_idx, 0]
                    hi = conformal_bands[cl_idx][i, k, t_idx, 1]
                    if lo <= cif_obs <= hi:
                        covered += 1
        return (covered / total) if total > 0 else None

    # Extract subgroup info
    sex_vals = []
    age_vals = []
    for pat in patnos:
        if pat in pat_baseline.index:
            sex_vals.append(pat_baseline.loc[pat].get("sex", np.nan))
            age_vals.append(pat_baseline.loc[pat].get("age_at_baseline", np.nan))
        else:
            sex_vals.append(np.nan)
            age_vals.append(np.nan)
    sex_arr = np.array(sex_vals)
    age_arr = np.array(age_vals)

    out = {}
    for cl_idx, cl in enumerate(CONFIDENCE_LEVELS):
        cl_key = f"{cl:.2f}"
        out[cl_key] = {
            "male": _subgroup_coverage(sex_arr == 1, cl_idx),
            "female": _subgroup_coverage(sex_arr == 0, cl_idx),
            "age_lt60": _subgroup_coverage(age_arr < 60, cl_idx),
            "age_60_70": _subgroup_coverage((age_arr >= 60) & (age_arr < 70), cl_idx),
            "age_gte70": _subgroup_coverage(age_arr >= 70, cl_idx),
        }
    return out


def main():
    t0 = time.time()

    print("=" * 60)
    print("PAPER 4 HOLDOUT: CONFORMAL SURVIVAL (seed=2026)")
    print("=" * 60)

    # Device — CPU is safest for external cross-process evaluation
    device = torch.device("cpu")
    print(f"Device: {device}")

    print("\nLoading data + checkpoints...")
    features_df = pd.read_csv(FEATURES_PATH, low_memory=False)
    with open(HOLDOUT_JSON) as f:
        split = json.load(f)

    patient_arrays, _ = build_patient_arrays(features_df)
    episodes = extract_episodes(features_df, verbose=False)
    holdout_pats = set(split["holdout"])
    print(f"  Holdout patients: {len(holdout_pats)}")
    print(f"  Episodes total:   {len(episodes)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Shared model reference container (reused by helpers above)
    global model_ref
    model_ref = [None]

    final_results = {}

    for model_type, model_name in [("deephit", "DeepHit"), ("graph_dt", "Graph-DT")]:
        print(f"\n--- {model_name} ---")
        if model_type == "deephit":
            model, ckpt = _load_deephit(device)
        else:
            model, ckpt = _load_graph_dt(device)
        model_ref[0] = model

        means = ckpt["means"]
        stds = ckpt["stds"]

        results = run_conformal(
            model_type,
            episodes,
            patient_arrays,
            means,
            stds,
            ckpt,
            device,
            model_name,
        )

        # --- Bands at each CL for subgroup analysis ---
        # Re-use the bands computed inside run_conformal to avoid redundant
        # model inference + calibration.
        conformal_bands = {
            cl_idx: results["bands_per_cl"][cl]
            for cl_idx, cl in enumerate(CONFIDENCE_LEVELS)
        }

        subgroup_cov = conditional_coverage_by_subgroup(
            conformal_bands,
            results["holdout_eps"],
            results["cif_eval"],
            results["dur_eval"],
            results["ev_eval"],
            results["cens_eval"],
            features_df,
        )

        final_results[model_name] = {
            "cif_results": [conformal_result_to_dict(r) for r in results["cif_results"]],
            "timing_results": [timing_result_to_dict(r) for r in results["timing_results"]],
            "subgroup_coverage": subgroup_cov,
            "n_holdout_episodes": len(results["holdout_eps"]),
            "n_holdout_patients": len(holdout_pats),
        }

    # --- Save JSON ---
    with open(OUTPUT_DIR / "conformal_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\nWrote {OUTPUT_DIR / 'conformal_results.json'}")

    subgroup_only = {
        k: v["subgroup_coverage"] for k, v in final_results.items()
    }
    with open(OUTPUT_DIR / "subgroup_coverage.json", "w") as f:
        json.dump(subgroup_only, f, indent=2, default=str)
    print(f"Wrote {OUTPUT_DIR / 'subgroup_coverage.json'}")

    timing_only = {
        k: v["timing_results"] for k, v in final_results.items()
    }
    with open(OUTPUT_DIR / "timing_intervals.json", "w") as f:
        json.dump(timing_only, f, indent=2, default=str)
    print(f"Wrote {OUTPUT_DIR / 'timing_intervals.json'}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("HOLDOUT CONFORMAL SUMMARY")
    print("=" * 60)
    for model_name, r in final_results.items():
        print(f"\n{model_name}:")
        for entry in r["cif_results"]:
            cl = entry["confidence_level"]
            print(
                f"  CL={cl:.2f}: marginal={entry['marginal_coverage']:.4f}  "
                f"mean_width={entry['mean_band_width']:.4f}"
            )
        print("  Subgroup coverage @ 90% CL:")
        for grp, cov in r["subgroup_coverage"]["0.90"].items():
            print(f"    {grp}: {cov if cov is None else f'{cov:.4f}'}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
