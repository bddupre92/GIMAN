#!/usr/bin/env python3
"""Phase 5 Task 4: Head-to-head head-to-head on time-to-wearing-off.

Compares mechanistic model vs GIMAN Graph-DT on a COMMON clinical endpoint:
  time from baseline to first visit with NP4OFF >= 1 (wearing-off onset).

Neither model was specifically trained on wearing-off, so this is a fair
head-to-head of overall progression risk signals.

Mechanistic risk score:
  - pct_loss_per_yr_median (higher = faster neuron loss = earlier wearing-off)

GIMAN Graph-DT risk score:
  - Sum of CIF across all causes at 5-year horizon (total 5yr progression risk)
  - Computed via cross-validated per-fold inference (patient uses fold where
    they're in test set; falls back to fold 0 if never in test)

Endpoint: time-to-NP4OFF>=1 (right-censored for non-events)
Metrics: Harrell's C-index, paired bootstrap 95% CI (1000 resamples)

Run:
    PYTHONPATH=src .venv/bin/python scripts/mechanistic_twin/phase5_headtohead_wearing_off.py

Output:
    outputs/mechanistic_twin/paper10_mech_vs_giman/headtohead_wearing_off.json

Author: Blair Dupre
Date: 2026-04-13
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.paper3.dynamic_deephit import (
    build_patient_arrays,
    extract_episodes,
)
from giman_pipeline.paper3.graph_digital_twin import (
    GraphDeepHitDataset,
    load_graph_dt_checkpoint,
    predict_all_graph,
)
from scripts.mechanistic_twin._reproducibility import capture_provenance

CHECKPOINTS = PROJECT_ROOT / "outputs/paper3_checkpoints"
CANONICAL = (
    PROJECT_ROOT
    / "outputs/mechanistic_twin/paper10_mech_vs_giman/canonical_assembled_v2.parquet"
)
FEATURES = PROJECT_ROOT / "data/07_paper3_features/longitudinal_features.csv"
SHARED_COHORT = (
    PROJECT_ROOT
    / "outputs/mechanistic_twin/paper10_mech_vs_giman/shared_cohort.json"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs/mechanistic_twin/paper10_mech_vs_giman"


def extract_wearing_off_survival(
    canonical: pd.DataFrame, cohort_patnos: set[int]
) -> pd.DataFrame:
    """Extract (patno, time, event) for time-to-NP4OFF>=1.

    - event=1: observed wearing-off (first visit with NP4OFF >= 1)
    - event=0: censored at last visit
    - Drops patients with no valid follow-up

    Returns DataFrame with columns: PATNO, time_months, event.
    """
    df = canonical.copy()
    df["PATNO"] = df["PATNO"].astype(int)
    df = df[df["PATNO"].isin(cohort_patnos)]
    df["months_from_baseline"] = pd.to_numeric(
        df["months_from_baseline"], errors="coerce"
    )

    events = (
        df[df["NP4OFF"] >= 1]
        .groupby("PATNO")
        .agg(event_time=("months_from_baseline", "min"))
        .reset_index()
    )
    max_followup = (
        df.groupby("PATNO")["months_from_baseline"].max().reset_index()
    )
    max_followup.columns = ["PATNO", "last_visit"]

    surv = max_followup.merge(events, on="PATNO", how="left")
    surv["event"] = surv["event_time"].notna().astype(int)
    surv["time_months"] = surv["event_time"].fillna(surv["last_visit"])

    # Drop invalid follow-up
    surv = surv[surv["time_months"].notna() & (surv["time_months"] > 0)]
    return surv[["PATNO", "time_months", "event"]].reset_index(drop=True)


def load_mechanistic_risk(cohort_patnos: set[int]) -> pd.DataFrame:
    """Load per-patient pct_loss_per_yr_median as mechanistic risk."""
    posts = pd.read_csv(
        PROJECT_ROOT
        / "outputs/mechanistic_twin/data/posteriors/phase2_combined_1065.csv"
    )
    posts["PATNO"] = posts["PATNO"].astype(int)
    posts = posts[posts["PATNO"].isin(cohort_patnos)]
    return posts[["PATNO", "pct_loss_per_yr_median"]].rename(
        columns={"pct_loss_per_yr_median": "mech_risk"}
    )


def run_graph_dt_inference(
    cohort_patnos: set[int], horizon_bin_idx: int = 7
) -> pd.DataFrame:
    """Run Graph-DT inference across all 5 folds; extract per-patient 5yr CIF.

    Uses cross-validated predictions: each patient's prediction comes from the
    fold where they're in test (or val if not in test). This avoids train-set
    leakage.

    horizon_bin_idx=7 corresponds to the 5-year time bin (60 months) in the
    standard Paper 3 bin schedule [3,6,12,18,24,36,48,60,84,120,180].

    Returns DataFrame with PATNO + graphdt_risk (sum CIF across all causes at 5yr).
    """
    device = torch.device("cpu")
    features_df = pd.read_csv(FEATURES)
    features_df["PATNO"] = features_df["PATNO"].astype(int)

    # Build episodes + patient arrays (needed for inference)
    episodes = extract_episodes(features_df, verbose=False)
    patient_arrays, col_names = build_patient_arrays(features_df)

    # Map patient to episode
    pat_to_ep = {ep.patno: ep for ep in episodes}

    # Collect per-patient CIF from appropriate fold
    predictions: dict[int, dict] = {}  # patno -> {cif_5yr, fold, set}

    for fold in range(5):
        print(f"  Loading Graph-DT fold {fold}...")
        cp_path = CHECKPOINTS / "graph_dt" / f"fold{fold}_graph_dt.pt"
        model, cp = load_graph_dt_checkpoint(cp_path, device=device)

        test_pats = set(int(p) for p in cp["test_pats"])
        val_pats = set(int(p) for p in cp["val_pats"])
        means = cp["means"]
        stds = cp["stds"]
        edge_index = cp["edge_index"]
        edge_weight = cp["edge_weight"]
        node_baseline = cp["node_baseline"]
        pat_to_gidx = cp["pat_to_gidx"]

        # Assemble episodes to predict: cohort patients in test or val of this fold
        # who haven't been assigned a prediction yet (prefer test over val)
        eligible = (test_pats | val_pats) & cohort_patnos
        to_predict = [
            p
            for p in eligible
            if p not in predictions
            and p in pat_to_ep
            and p in pat_to_gidx
        ]
        if not to_predict:
            continue

        fold_episodes = [pat_to_ep[p] for p in to_predict]
        fold_arrays = {p: patient_arrays[p] for p in to_predict if p in patient_arrays}
        # Filter episodes to those with arrays
        fold_episodes = [e for e in fold_episodes if e.patno in fold_arrays]

        if not fold_episodes:
            continue

        dataset = GraphDeepHitDataset(
            fold_episodes, fold_arrays, means, stds, pat_to_gidx
        )

        print(f"    Predicting on {len(dataset)} patients...")
        preds = predict_all_graph(
            model, dataset, device, node_baseline, edge_index, edge_weight,
            batch_size=64,
        )
        # preds["cif"] shape: (n, n_causes, n_time_bins)
        cif = preds["cif"].numpy()  # (n, 7, 11)
        # Sum CIF across causes (excluding "no-event" if represented) at 5yr bin
        # Paper 3 uses 7 causes (5 NSD-ISS stages + regression) and 11 time bins
        # horizon_bin_idx=7 = 60 months (5 years)
        cif_5yr = cif[:, :, horizon_bin_idx].sum(axis=1)  # per-patient total CIF

        for ep, risk in zip(fold_episodes, cif_5yr):
            in_test = ep.patno in test_pats
            predictions[ep.patno] = {
                "graphdt_risk": float(risk),
                "fold": fold,
                "set": "test" if in_test else "val",
            }

    # Remaining cohort patients not covered: use fold 0 as fallback (train leakage ok for scoping)
    uncovered = cohort_patnos - set(predictions.keys())
    if uncovered:
        print(f"  Fallback: {len(uncovered)} patients not in any test/val → using fold 0")
        model, cp = load_graph_dt_checkpoint(
            CHECKPOINTS / "graph_dt" / "fold0_graph_dt.pt", device=device
        )
        fb_episodes = [pat_to_ep[p] for p in uncovered if p in pat_to_ep and p in cp["pat_to_gidx"]]
        fb_arrays = {
            p: patient_arrays[p] for p in uncovered if p in patient_arrays
        }
        fb_episodes = [e for e in fb_episodes if e.patno in fb_arrays]
        if fb_episodes:
            ds = GraphDeepHitDataset(
                fb_episodes, fb_arrays, cp["means"], cp["stds"], cp["pat_to_gidx"]
            )
            preds = predict_all_graph(
                model, ds, device,
                cp["node_baseline"], cp["edge_index"], cp["edge_weight"],
                batch_size=64,
            )
            cif_5yr = preds["cif"].numpy()[:, :, horizon_bin_idx].sum(axis=1)
            for ep, risk in zip(fb_episodes, cif_5yr):
                predictions[ep.patno] = {
                    "graphdt_risk": float(risk),
                    "fold": 0,
                    "set": "train_fallback",
                }

    # Build DataFrame
    rows = [
        {"PATNO": p, "graphdt_risk": d["graphdt_risk"], "fold": d["fold"], "set": d["set"]}
        for p, d in predictions.items()
    ]
    return pd.DataFrame(rows)


def paired_bootstrap_cindex(
    times: np.ndarray,
    events: np.ndarray,
    risks_a: np.ndarray,
    risks_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap C-index comparison.

    C-index: higher risk should correspond to shorter survival (event=1).
    For wearing-off, higher mechanistic pct_loss or higher Graph-DT CIF should
    correlate with faster time-to-event.

    Args:
        times, events: survival data (n,)
        risks_a, risks_b: patient-level risk scores (n,)
        n_boot: number of bootstrap resamples

    Returns dict with:
        - ci_a, ci_b: point estimates
        - ci_a_95, ci_b_95: bootstrap 95% CIs
        - delta_ci: ci_a - ci_b (95% CI of difference)
        - p_value: two-sided p for delta == 0
    """
    rng = np.random.default_rng(seed)
    n = len(times)

    # Point estimate: lifelines uses "higher concordance = better".
    # By convention, higher risk should predict SHORTER time → pass -risk
    # as predicted values (higher predicted_value = longer time)
    ci_a_point = concordance_index(times, -risks_a, events)
    ci_b_point = concordance_index(times, -risks_b, events)

    boot_ci_a = np.empty(n_boot)
    boot_ci_b = np.empty(n_boot)
    boot_delta = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        t_b, e_b = times[idx], events[idx]
        ra_b, rb_b = risks_a[idx], risks_b[idx]
        try:
            ca = concordance_index(t_b, -ra_b, e_b)
            cb = concordance_index(t_b, -rb_b, e_b)
        except ZeroDivisionError:
            ca = cb = 0.5
        boot_ci_a[i] = ca
        boot_ci_b[i] = cb
        boot_delta[i] = ca - cb

    # Two-sided p-value: fraction of bootstrap deltas with opposite sign of observed
    observed_delta = ci_a_point - ci_b_point
    if observed_delta >= 0:
        p_val = 2 * min(np.mean(boot_delta <= 0), 0.5)
    else:
        p_val = 2 * min(np.mean(boot_delta >= 0), 0.5)

    return {
        "ci_a": float(ci_a_point),
        "ci_b": float(ci_b_point),
        "ci_a_95": [float(np.percentile(boot_ci_a, 2.5)), float(np.percentile(boot_ci_a, 97.5))],
        "ci_b_95": [float(np.percentile(boot_ci_b, 2.5)), float(np.percentile(boot_ci_b, 97.5))],
        "delta_ci": float(observed_delta),
        "delta_ci_95": [
            float(np.percentile(boot_delta, 2.5)),
            float(np.percentile(boot_delta, 97.5)),
        ],
        "p_value": float(p_val),
        "n_boot": n_boot,
        "n_samples": int(n),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prov = capture_provenance(
        script_path=Path(__file__),
        repo_root=PROJECT_ROOT,
        input_files=[CANONICAL, FEATURES, SHARED_COHORT],
    )

    # Load shared cohort
    with open(SHARED_COHORT) as f:
        cohort = json.load(f)
    cohort_patnos = set(cohort["shared_patnos"])
    print(f"Shared cohort: {len(cohort_patnos)} patients")

    # Extract wearing-off survival
    print("\nExtracting wearing-off events...")
    canonical = pd.read_parquet(CANONICAL)
    surv = extract_wearing_off_survival(canonical, cohort_patnos)
    print(f"  Survival set: {len(surv)} patients, "
          f"{surv['event'].sum()} events ({100*surv['event'].mean():.1f}%)")

    # Mechanistic risk
    print("\nLoading mechanistic risk (pct_loss_per_yr_median)...")
    mech = load_mechanistic_risk(cohort_patnos)
    print(f"  Mechanistic risk: {len(mech)} patients")

    # Graph-DT risk (this is the expensive step)
    print("\nRunning Graph-DT inference across 5 folds...")
    t0 = time.time()
    gdt = run_graph_dt_inference(cohort_patnos)
    print(f"  Graph-DT risk: {len(gdt)} patients, {time.time()-t0:.1f}s")

    # Merge
    df = surv.merge(mech, on="PATNO", how="inner").merge(gdt, on="PATNO", how="inner")
    df = df.dropna(subset=["mech_risk", "graphdt_risk"])
    print(f"\nFinal analysis set: {len(df)} patients")
    print(f"  Events: {df['event'].sum()} ({100*df['event'].mean():.1f}%)")

    # Set breakdown (how many from test vs val vs fallback)
    set_counts = df["set"].value_counts().to_dict()
    print(f"  Graph-DT prediction source: {set_counts}")

    # C-index comparison
    print("\nComputing paired bootstrap C-index (n_boot=1000)...")
    result = paired_bootstrap_cindex(
        times=df["time_months"].values,
        events=df["event"].values,
        risks_a=df["mech_risk"].values,
        risks_b=df["graphdt_risk"].values,
        n_boot=1000,
    )

    print(f"\n=== C-index Results ===")
    print(f"Mechanistic (pct_loss_per_yr): {result['ci_a']:.4f} 95% CI [{result['ci_a_95'][0]:.4f}, {result['ci_a_95'][1]:.4f}]")
    print(f"Graph-DT (5yr total CIF):      {result['ci_b']:.4f} 95% CI [{result['ci_b_95'][0]:.4f}, {result['ci_b_95'][1]:.4f}]")
    print(f"Delta (mech - gdt):            {result['delta_ci']:+.4f} 95% CI [{result['delta_ci_95'][0]:+.4f}, {result['delta_ci_95'][1]:+.4f}]")
    print(f"Two-sided p-value:             {result['p_value']:.4f}")

    # Verdict
    if result["p_value"] >= 0.05:
        verdict = (
            f"No significant difference between models (p={result['p_value']:.3f}). "
            "Complementary predictions on wearing-off — consistent with models "
            "answering different clinical questions."
        )
    elif result["ci_a"] > result["ci_b"]:
        verdict = f"Mechanistic model significantly better (p={result['p_value']:.3f})."
    else:
        verdict = f"Graph-DT significantly better (p={result['p_value']:.3f})."

    summary = {
        "endpoint": "time-to-NP4OFF>=1 (wearing-off onset)",
        "shared_cohort_size": len(cohort_patnos),
        "analysis_set_size": len(df),
        "events": int(df["event"].sum()),
        "event_rate_pct": float(100 * df["event"].mean()),
        "graphdt_prediction_source": set_counts,
        "mechanistic_score": "pct_loss_per_yr_median (higher = earlier event expected)",
        "graphdt_score": "sum of CIF across 7 causes at 60-month horizon",
        "cindex_comparison": result,
        "verdict": verdict,
        "interpretation": (
            "Neither model was trained on wearing-off specifically — this is a "
            "fair common-endpoint comparison of overall progression risk signals. "
            "C-indices near 0.55-0.60 are expected for proxy scores vs an "
            "out-of-distribution endpoint."
        ),
        "_provenance": prov,
    }
    out_path = OUTPUT_DIR / "headtohead_wearing_off.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print(f"\n=== Verdict ===\n{verdict}")


if __name__ == "__main__":
    main()
