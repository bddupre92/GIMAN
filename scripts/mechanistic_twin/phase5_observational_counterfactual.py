#!/usr/bin/env python3
"""Phase 5 Task 6 — Observational Counterfactual Calibration.

Replaces the v1 plan's synthetic counterfactual with an OBSERVATIONAL one
(Bica et al. ICLR 2020 standard). For each PPMI visit pair where LEDD escalated
by >= 200mg (the Tomlinson 2010 canonical threshold), compute:

    predicted_delta_gap = beta_ledd * delta_ledd_scaled
                        + beta_interaction * n_frac_c_baseline * delta_ledd_scaled
                        + beta_updrs3 * delta_updrs3_c

using the Phase 4 Path B severity-controlled interaction model coefficients
(model2_severity_controlled from phase4_confounding_control.json):

    Intercept            = 9.6949
    beta_nfrac_c         = -2.8371   (p=0.0009)
    beta_ledd_c          = 0.2579    (p=0.112)
    beta_interaction     = 1.4096    (p=0.044)   <-- the headline interaction
    beta_updrs3_off_c    = 0.3714    (p<<0.001)

Compare to observed_delta_gap = gap_t - gap_{t-1}. Report calibration slope
(predicted vs observed), intercept, R^2, and paired bootstrap CIs. A
calibration slope near 1.0 indicates the Phase 4 model predicts actual drug
responses in real PPMI patients, not just synthetic extrapolations.

Centering follows Phase 4 (grand-mean): ledd_scaled = ledd_total / 500.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

CANONICAL = PROJECT_ROOT / "outputs/mechanistic_twin/paper10_mech_vs_giman/canonical_assembled_v2.parquet"
PATH_B_RESULTS = PROJECT_ROOT / "outputs/mechanistic_twin/phase4/phase4_confounding_control.json"
OUTPUT = PROJECT_ROOT / "outputs/mechanistic_twin/paper10_mech_vs_giman/observational_counterfactual.json"

# Phase 4 severity-controlled interaction coefficients (model2_severity_controlled)
COEFS = {
    "intercept": 9.6949,
    "beta_nfrac_c": -2.8371,
    "beta_ledd_c": 0.2579,
    "beta_interaction": 1.4096,
    "beta_updrs3_off_c": 0.3714,
}

LEDD_ESCALATION_THRESHOLD_MG = 200.0
LEDD_SCALE = 500.0  # Phase 4 used ledd_scaled = ledd / 500
RNG_SEED = 202604131
N_BOOTSTRAP = 1000


def load_data() -> pd.DataFrame:
    """Load canonical parquet, filter to rows with all Phase 4 Path B variables."""
    df = pd.read_parquet(CANONICAL)
    df = df.dropna(subset=["gap", "ledd_total", "n_frac", "updrs3_off", "visit_dt"])
    df = df.sort_values(["PATNO", "visit_dt"]).reset_index(drop=True)
    df["ledd_scaled"] = df["ledd_total"] / LEDD_SCALE
    return df


def centering_means(df: pd.DataFrame) -> dict[str, float]:
    """Grand-mean centering to match Phase 4 Path B cohort statistics."""
    return {
        "n_frac": float(df["n_frac"].mean()),
        "ledd_scaled": float(df["ledd_scaled"].mean()),
        "updrs3_off": float(df["updrs3_off"].mean()),
    }


def extract_escalation_events(
    df: pd.DataFrame, threshold_mg: float
) -> pd.DataFrame:
    """Identify consecutive visit pairs where LEDD increased by >= threshold_mg.

    Returns a DataFrame with baseline + follow-up values per event. Only pairs
    within the SAME patient are considered; first-difference enforces
    patient-anchoring (absorbs fixed effects per Bica 2020 requirement).
    """
    events = []
    for patno, grp in df.groupby("PATNO"):
        grp = grp.sort_values("visit_dt").reset_index(drop=True)
        for i in range(1, len(grp)):
            delta_ledd = grp.at[i, "ledd_total"] - grp.at[i - 1, "ledd_total"]
            if delta_ledd < threshold_mg:
                continue
            events.append(
                {
                    "PATNO": int(patno),
                    "visit_t0": grp.at[i - 1, "visit_dt"],
                    "visit_t1": grp.at[i, "visit_dt"],
                    "n_frac_t0": float(grp.at[i - 1, "n_frac"]),
                    "updrs3_off_t0": float(grp.at[i - 1, "updrs3_off"]),
                    "updrs3_off_t1": float(grp.at[i, "updrs3_off"]),
                    "ledd_t0": float(grp.at[i - 1, "ledd_total"]),
                    "ledd_t1": float(grp.at[i, "ledd_total"]),
                    "gap_t0": float(grp.at[i - 1, "gap"]),
                    "gap_t1": float(grp.at[i, "gap"]),
                    "delta_ledd_mg": float(delta_ledd),
                    "delta_updrs3_off": float(
                        grp.at[i, "updrs3_off"] - grp.at[i - 1, "updrs3_off"]
                    ),
                    "delta_gap_observed": float(
                        grp.at[i, "gap"] - grp.at[i - 1, "gap"]
                    ),
                }
            )
    return pd.DataFrame(events)


def predict_delta_gap(
    events: pd.DataFrame, means: dict, coefs: dict
) -> pd.DataFrame:
    """Apply Phase 4 Path B interaction model to predict delta_gap per event.

    Model: gap = intercept + beta_nfrac * n_frac_c + beta_ledd * ledd_c
                 + beta_interaction * n_frac_c * ledd_c
                 + beta_updrs3 * updrs3_off_c

    Holding patient's n_frac (approximately constant over a short follow-up
    interval) and the severity trajectory, the predicted change in gap is:

      delta_gap_pred = beta_ledd * delta_ledd_scaled
                     + beta_interaction * n_frac_c_baseline * delta_ledd_scaled
                     + beta_updrs3 * delta_updrs3_off

    n_frac_c_baseline is the patient's baseline n_frac after grand-mean
    centering; we use the baseline visit's value because the Phase 4 interaction
    was fit on visit-level data and the plausible interpretation is that the
    patient's current neuron loss moderates the drug response.
    """
    ev = events.copy()
    ev["n_frac_c_t0"] = ev["n_frac_t0"] - means["n_frac"]
    ev["delta_ledd_scaled"] = (ev["ledd_t1"] - ev["ledd_t0"]) / LEDD_SCALE

    ev["pred_delta_gap"] = (
        coefs["beta_ledd_c"] * ev["delta_ledd_scaled"]
        + coefs["beta_interaction"] * ev["n_frac_c_t0"] * ev["delta_ledd_scaled"]
        + coefs["beta_updrs3_off_c"] * ev["delta_updrs3_off"]
    )
    ev["residual"] = ev["delta_gap_observed"] - ev["pred_delta_gap"]
    return ev


def calibration_stats(
    ev: pd.DataFrame, rng: np.random.Generator, n_boot: int
) -> dict:
    """Regress observed on predicted; bootstrap CI on slope, intercept, R^2, MAE."""
    x = ev["pred_delta_gap"].to_numpy()
    y = ev["delta_gap_observed"].to_numpy()

    def _fit(x_, y_):
        slope, intercept = np.polyfit(x_, y_, deg=1)
        yhat = slope * x_ + intercept
        ss_res = np.sum((y_ - yhat) ** 2)
        ss_tot = np.sum((y_ - y_.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = float(np.mean(np.abs(y_ - x_)))  # direct MAE of prediction vs obs
        rmse = float(np.sqrt(np.mean((y_ - x_) ** 2)))
        return slope, intercept, r2, mae, rmse

    slope, intercept, r2, mae, rmse = _fit(x, y)

    slopes, intercepts, r2s, maes = [], [], [], []
    n = len(ev)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s, i_, r_, m_, _ = _fit(x[idx], y[idx])
        slopes.append(s)
        intercepts.append(i_)
        r2s.append(r_)
        maes.append(m_)
    return {
        "n_events": int(n),
        "slope": float(slope),
        "slope_ci95": [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))],
        "intercept": float(intercept),
        "intercept_ci95": [float(np.percentile(intercepts, 2.5)), float(np.percentile(intercepts, 97.5))],
        "r2": float(r2),
        "r2_ci95": [float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5))],
        "mae": float(mae),
        "mae_ci95": [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))],
        "rmse": float(rmse),
    }


def stratify_by_nfrac(ev: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Split events into advanced (low n_frac) vs early (high n_frac) strata.

    If the Phase 4 interaction is predictive, advanced patients (more neuron
    loss) should show a smaller LEDD-induced gap increase than early patients.
    """
    median_nfrac = ev["n_frac_t0"].median()
    early = ev[ev["n_frac_t0"] >= median_nfrac]
    advanced = ev[ev["n_frac_t0"] < median_nfrac]
    return {
        "median_n_frac_cutoff": float(median_nfrac),
        "early_stratum": calibration_stats(early, rng, N_BOOTSTRAP // 2),
        "advanced_stratum": calibration_stats(advanced, rng, N_BOOTSTRAP // 2),
        "observed_delta_gap_early_mean": float(early["delta_gap_observed"].mean()),
        "observed_delta_gap_advanced_mean": float(
            advanced["delta_gap_observed"].mean()
        ),
    }


def main() -> None:
    print("[Task 6] Loading canonical parquet...")
    df = load_data()
    print(f"  rows with Path B vars: {len(df)} / patients: {df['PATNO'].nunique()}")

    means = centering_means(df)
    print(f"[Task 6] Grand-mean centering:")
    for k, v in means.items():
        print(f"  {k}: {v:.4f}")

    print(f"[Task 6] Extracting LEDD escalations >= {LEDD_ESCALATION_THRESHOLD_MG}mg...")
    events = extract_escalation_events(df, LEDD_ESCALATION_THRESHOLD_MG)
    print(f"  events: {len(events)}, patients: {events['PATNO'].nunique()}")
    if len(events) < 20:
        print("  WARNING: too few events for stable calibration.")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(
            json.dumps(
                {"status": "insufficient_events", "n_events": len(events)}, indent=2
            )
        )
        return

    ev = predict_delta_gap(events, means, COEFS)
    print(
        f"[Task 6] Observed delta_gap: mean={ev['delta_gap_observed'].mean():.3f}, "
        f"median={ev['delta_gap_observed'].median():.3f}"
    )
    print(
        f"[Task 6] Predicted delta_gap: mean={ev['pred_delta_gap'].mean():.3f}, "
        f"median={ev['pred_delta_gap'].median():.3f}"
    )

    rng = np.random.default_rng(RNG_SEED)
    overall = calibration_stats(ev, rng, N_BOOTSTRAP)
    strat = stratify_by_nfrac(ev, rng)

    summary = {
        "endpoint": "Observational counterfactual calibration (LEDD escalation >= 200mg)",
        "source_model": "Phase 4 Path B severity-controlled interaction (phase4_confounding_control.json model2)",
        "coefficients_used": COEFS,
        "centering_means": means,
        "ledd_escalation_threshold_mg": LEDD_ESCALATION_THRESHOLD_MG,
        "ledd_scale": LEDD_SCALE,
        "analysis_set": {
            "n_events": int(len(ev)),
            "n_patients": int(ev["PATNO"].nunique()),
            "mean_delta_ledd_mg": float(ev["delta_ledd_mg"].mean()),
            "median_delta_ledd_mg": float(ev["delta_ledd_mg"].median()),
            "mean_observed_delta_gap": float(ev["delta_gap_observed"].mean()),
            "mean_predicted_delta_gap": float(ev["pred_delta_gap"].mean()),
        },
        "calibration_overall": overall,
        "stratified_by_nfrac": strat,
        "verdict": (
            "Calibration slope 95% CI contains 1.0" if overall["slope_ci95"][0] <= 1.0 <= overall["slope_ci95"][1]
            else f"Calibration slope 95% CI {overall['slope_ci95']} does NOT contain 1.0 (miscalibrated)"
        ),
        "seed": RNG_SEED,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[Task 6] Wrote {OUTPUT}")

    print("\n=== Calibration (overall) ===")
    print(
        f"slope = {overall['slope']:.3f} [95% CI {overall['slope_ci95'][0]:.3f}, {overall['slope_ci95'][1]:.3f}]"
    )
    print(
        f"intercept = {overall['intercept']:.3f} [95% CI {overall['intercept_ci95'][0]:.3f}, {overall['intercept_ci95'][1]:.3f}]"
    )
    print(f"R^2 = {overall['r2']:.3f}, MAE = {overall['mae']:.3f}, RMSE = {overall['rmse']:.3f}")
    print(f"\nVerdict: {summary['verdict']}")


if __name__ == "__main__":
    main()
