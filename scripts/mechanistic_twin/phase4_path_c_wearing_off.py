#!/usr/bin/env python
"""Phase 4 Path C: Survival model predicting time to wearing-off from N(t)/N₀.

Scientific rationale:
    Wearing-off occurs when levodopa's effect doesn't last the full dosing
    interval. As N(t) declines, fewer neurons convert levodopa to dopamine,
    so the DA trough between doses drops below the symptom threshold sooner.
    Patients with faster N(t) decline should experience wearing-off earlier.

Models:
    C1 — Kaplan-Meier by neuron-loss-rate tertiles + log-rank test
    C2 — Cox PH: time_to_wearingoff ~ pct_loss_per_yr_median + age
    C3 — Cox PH with N(t) trajectory features
    C4 — Spearman(pct_loss_per_yr_median, time_to_wearingoff)

Sensitivity:
    - NP4OFF >= 2 threshold (more definite wearing-off)
    - NP4WDYSK >= 1 (dyskinesia onset) as secondary endpoint

Inputs:
    - MDS-UPDRS Part IV (NP4OFF wearing-off, NP4WDYSK dyskinesia)
    - Phase 2 posteriors (pct_loss_per_yr_median)
    - Longitudinal staging (months_from_baseline per EVENT_ID)
    - Use of PD Medication (PDMEDYN to filter treated patients)

Outputs:
    - outputs/mechanistic_twin/phase4/phase4_path_c_results.json
    - outputs/mechanistic_twin/phase4/phase4_path_c_km_curves.png
    - RUN_MANIFEST.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── project paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "mechanistic_twin"))
from _reproducibility import capture_provenance, write_run_manifest

# Input paths
PART4_CSV = PROJECT_ROOT / "data" / "00_raw" / "MDS-UPDRS Part IV" / "MDS-UPDRS_Part_IV__Motor_Complications_12Apr2026.csv"
POSTERIORS_CSV = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "data" / "posteriors" / "phase2_coupled_is_step26v4.csv"
LONGITUDINAL_CSV = PROJECT_ROOT / "data" / "06_longitudinal_staging" / "longitudinal_nsd_iss.csv"
PDMED_CSV = PROJECT_ROOT / "data" / "00_raw" / "Use_of_PD_Medication-Archived_07Feb2026.csv"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "phase4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_part4() -> pd.DataFrame:
    """Load MDS-UPDRS Part IV, clean NP4OFF and NP4WDYSK."""
    df = pd.read_csv(PART4_CSV, dtype={"PATNO": str})
    # Clean NP4OFF: 101 = "not applicable" → NaN
    df["NP4OFF"] = pd.to_numeric(df["NP4OFF"], errors="coerce")
    df.loc[df["NP4OFF"] == 101, "NP4OFF"] = np.nan
    # Clean NP4WDYSK similarly
    df["NP4WDYSK"] = pd.to_numeric(df["NP4WDYSK"], errors="coerce")
    df.loc[df["NP4WDYSK"] == 101, "NP4WDYSK"] = np.nan
    return df


def load_visit_months() -> pd.DataFrame:
    """Load longitudinal staging to get months_from_baseline per (PATNO, EVENT_ID)."""
    df = pd.read_csv(LONGITUDINAL_CSV, dtype={"PATNO": str})
    return df[["PATNO", "EVENT_ID", "months_from_baseline"]].drop_duplicates()


def load_posteriors() -> pd.DataFrame:
    """Load Phase 2 IS posteriors."""
    df = pd.read_csv(POSTERIORS_CSV, dtype={"PATNO": str})
    return df[["PATNO", "pct_loss_per_yr_median"]].copy()


def load_treated_patients() -> set[str]:
    """Return set of PATNOs who were ever on PD medication (PDMEDYN == 1)."""
    df = pd.read_csv(PDMED_CSV, dtype={"PATNO": str})
    df["PDMEDYN"] = pd.to_numeric(df["PDMEDYN"], errors="coerce")
    return set(df.loc[df["PDMEDYN"] == 1, "PATNO"].unique())


def compute_time_to_event(
    part4: pd.DataFrame,
    visit_months: pd.DataFrame,
    threshold: int,
    col: str = "NP4OFF",
) -> pd.DataFrame:
    """Compute time-to-event for first visit where col >= threshold.

    Returns DataFrame with columns: PATNO, time_to_event, event_observed.
    """
    # Merge Part IV with visit months
    merged = part4.merge(visit_months, on=["PATNO", "EVENT_ID"], how="inner")
    merged = merged.dropna(subset=[col, "months_from_baseline"])
    merged = merged.sort_values(["PATNO", "months_from_baseline"])

    results = []
    for patno, grp in merged.groupby("PATNO"):
        event_rows = grp[grp[col] >= threshold]
        if len(event_rows) > 0:
            first_event = event_rows.iloc[0]
            results.append({
                "PATNO": patno,
                "time_to_event": first_event["months_from_baseline"],
                "event_observed": 1,
            })
        else:
            # Censored at last visit
            last_visit = grp.iloc[-1]
            results.append({
                "PATNO": patno,
                "time_to_event": last_visit["months_from_baseline"],
                "event_observed": 0,
            })

    return pd.DataFrame(results)


def run_km_analysis(surv_df: pd.DataFrame, label: str) -> dict:
    """Run Kaplan-Meier by pct_loss tertiles + log-rank test."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    # Split into tertiles of pct_loss_per_yr_median
    surv_df = surv_df.copy()
    surv_df["prog_group"] = pd.qcut(
        surv_df["pct_loss_per_yr_median"], 3,
        labels=["slow", "medium", "fast"],
    )

    km_results = {}
    km_fitters = {}
    for group_name in ["slow", "medium", "fast"]:
        subset = surv_df[surv_df["prog_group"] == group_name]
        kmf = KaplanMeierFitter()
        kmf.fit(
            subset["time_to_event"],
            event_observed=subset["event_observed"],
            label=f"{group_name} ({len(subset)} pts)",
        )
        km_fitters[group_name] = kmf
        median_surv = kmf.median_survival_time_
        km_results[group_name] = {
            "n": int(len(subset)),
            "n_events": int(subset["event_observed"].sum()),
            "median_survival_months": float(median_surv) if np.isfinite(median_surv) else None,
            "pct_loss_range": [
                float(subset["pct_loss_per_yr_median"].min()),
                float(subset["pct_loss_per_yr_median"].max()),
            ],
        }

    # Log-rank test: fast vs slow
    fast = surv_df[surv_df["prog_group"] == "fast"]
    slow = surv_df[surv_df["prog_group"] == "slow"]
    lr = logrank_test(
        fast["time_to_event"], slow["time_to_event"],
        event_observed_A=fast["event_observed"],
        event_observed_B=slow["event_observed"],
    )

    # Also overall log-rank across all 3 groups
    medium = surv_df[surv_df["prog_group"] == "medium"]
    from lifelines.statistics import multivariate_logrank_test
    mlr = multivariate_logrank_test(
        surv_df["time_to_event"],
        surv_df["prog_group"],
        surv_df["event_observed"],
    )

    return {
        "label": label,
        "km_by_group": km_results,
        "logrank_fast_vs_slow": {
            "test_statistic": float(lr.test_statistic),
            "p_value": float(lr.p_value),
        },
        "multivariate_logrank": {
            "test_statistic": float(mlr.test_statistic),
            "p_value": float(mlr.p_value),
        },
        "_km_fitters": km_fitters,  # For plotting
    }


def run_cox_analysis(surv_df: pd.DataFrame, label: str, model_name: str) -> dict:
    """Run Cox PH regression."""
    from lifelines import CoxPHFitter

    surv_df = surv_df.copy()
    # Ensure no zero durations (lifelines doesn't handle well)
    surv_df["time_to_event"] = surv_df["time_to_event"].clip(lower=0.1)

    if model_name == "C2":
        covariates = ["pct_loss_per_yr_median", "age_at_baseline"]
    elif model_name == "C3":
        # Use n_frac_3yr (predicted neuron reserve at 3yr) instead of raw pct_loss
        # This is a nonlinear reparameterization testing whether the exponential
        # decay model's predicted reserve fraction is more predictive than the
        # linear rate
        covariates = ["n_frac_at_baseline", "age_at_baseline"]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Check we have all covariates
    for c in covariates:
        if c not in surv_df.columns:
            return {"label": label, "model": model_name, "error": f"Missing covariate: {c}"}

    fit_df = surv_df[["time_to_event", "event_observed"] + covariates].dropna()

    if len(fit_df) < 10 or fit_df["event_observed"].sum() < 5:
        return {
            "label": label, "model": model_name,
            "error": f"Insufficient data: {len(fit_df)} rows, {fit_df['event_observed'].sum()} events",
        }

    cph = CoxPHFitter()
    try:
        cph.fit(fit_df, duration_col="time_to_event", event_col="event_observed")
    except Exception as e:
        return {"label": label, "model": model_name, "error": str(e)}

    # Extract results
    summary = cph.summary
    hr_results = {}
    for covar in covariates:
        if covar in summary.index:
            row = summary.loc[covar]
            hr_results[covar] = {
                "hazard_ratio": float(row["exp(coef)"]),
                "coef": float(row["coef"]),
                "se": float(row["se(coef)"]),
                "z": float(row["z"]),
                "p": float(row["p"]),
                "ci_lower": float(row["exp(coef) lower 95%"]),
                "ci_upper": float(row["exp(coef) upper 95%"]),
            }

    return {
        "label": label,
        "model": model_name,
        "n": int(len(fit_df)),
        "n_events": int(fit_df["event_observed"].sum()),
        "concordance_index": float(cph.concordance_index_),
        "log_likelihood_ratio_test_p": float(cph.log_likelihood_ratio_test().p_value),
        "hazard_ratios": hr_results,
    }


def run_spearman(surv_df: pd.DataFrame, label: str) -> dict:
    """H2: Spearman(pct_loss_per_yr_median, time_to_event) among event patients."""
    from scipy.stats import spearmanr

    # Among those who had the event
    event_df = surv_df[surv_df["event_observed"] == 1].dropna(
        subset=["pct_loss_per_yr_median", "time_to_event"]
    )

    if len(event_df) < 5:
        return {"label": label, "error": f"Too few events: {len(event_df)}"}

    rho, p = spearmanr(event_df["pct_loss_per_yr_median"], event_df["time_to_event"])

    # Also among all patients (censored + events)
    all_df = surv_df.dropna(subset=["pct_loss_per_yr_median", "time_to_event"])
    rho_all, p_all = spearmanr(all_df["pct_loss_per_yr_median"], all_df["time_to_event"])

    return {
        "label": label,
        "events_only": {
            "n": int(len(event_df)),
            "spearman_rho": float(rho),
            "p_value": float(p),
        },
        "all_patients": {
            "n": int(len(all_df)),
            "spearman_rho": float(rho_all),
            "p_value": float(p_all),
        },
    }


def plot_km_curves(km_results_list: list[dict], output_path: Path):
    """Plot KM curves for primary and sensitivity analyses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = len(km_results_list)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    colors = {"slow": "#2196F3", "medium": "#FF9800", "fast": "#F44336"}

    for ax, result in zip(axes, km_results_list):
        fitters = result.get("_km_fitters", {})
        for group_name in ["slow", "medium", "fast"]:
            if group_name in fitters:
                fitters[group_name].plot_survival_function(
                    ax=ax, color=colors[group_name], ci_show=True, ci_alpha=0.15
                )

        p_val = result.get("multivariate_logrank", {}).get("p_value", None)
        p_str = f"p = {p_val:.4f}" if p_val is not None else ""
        ax.set_title(f"{result['label']}\nLog-rank {p_str}", fontsize=11)
        ax.set_xlabel("Months from baseline", fontsize=10)
        ax.set_ylabel("Survival probability (no wearing-off)", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved KM curves: {output_path}")


def main():
    # ── provenance ─────────────────────────────────────────────────────
    provenance = capture_provenance(
        script_path=Path(__file__).resolve(),
        repo_root=PROJECT_ROOT,
        input_files=[PART4_CSV, POSTERIORS_CSV, LONGITUDINAL_CSV, PDMED_CSV],
        extra={"seed": 42, "primary_threshold": 1, "sensitivity_threshold": 2},
    )

    # ── load data ──────────────────────────────────────────────────────
    print("Loading data...")
    part4 = load_part4()
    visit_months = load_visit_months()
    posteriors = load_posteriors()
    treated_patients = load_treated_patients()

    print(f"  Part IV rows: {len(part4)}")
    print(f"  Unique Part IV patients: {part4['PATNO'].nunique()}")
    print(f"  Visit-months rows: {len(visit_months)}")
    print(f"  Phase 2 posteriors: {len(posteriors)} patients")
    print(f"  Treated patients (PDMEDYN=1): {len(treated_patients)}")

    # ── get baseline age ───────────────────────────────────────────────
    long_df = pd.read_csv(LONGITUDINAL_CSV, dtype={"PATNO": str})
    baseline_age = (
        long_df.sort_values("months_from_baseline")
        .groupby("PATNO")
        .first()
        .reset_index()[["PATNO", "age_at_visit"]]
        .rename(columns={"age_at_visit": "age_at_baseline"})
    )

    # ── compute N(t)/N₀ at baseline from posteriors ────────────────────
    # n_frac_at_baseline placeholder — will be replaced per-patient after
    # merging with time-to-event data (n_frac at event/censor time)
    posteriors["n_frac_at_baseline"] = np.nan  # Computed below after merge

    # ── build analysis cohort ──────────────────────────────────────────
    # Only include patients who:
    # 1. Have Phase 2 posteriors
    # 2. Were ever on PD medication (wearing-off requires treatment)
    # 3. Have Part IV data

    part4_patients = set(part4["PATNO"].unique())
    posterior_patients = set(posteriors["PATNO"].unique())
    patients_with_part4_and_posteriors = part4_patients & posterior_patients & treated_patients

    print(f"\n  Patients with Part IV + posteriors + treated: {len(patients_with_part4_and_posteriors)}")

    # ── run analyses for multiple endpoints ────────────────────────────
    analyses = [
        {"label": "NP4OFF >= 1 (primary)", "col": "NP4OFF", "threshold": 1},
        {"label": "NP4OFF >= 2 (sensitivity)", "col": "NP4OFF", "threshold": 2},
        {"label": "NP4WDYSK >= 1 (dyskinesia)", "col": "NP4WDYSK", "threshold": 1},
    ]

    all_results = {}
    km_for_plot = []

    for analysis in analyses:
        label = analysis["label"]
        col = analysis["col"]
        threshold = analysis["threshold"]
        key = label.split("(")[1].rstrip(")")  # e.g. "primary", "sensitivity", "dyskinesia"
        print(f"\n{'='*60}")
        print(f"Analysis: {label}")
        print(f"{'='*60}")

        # Compute time-to-event
        tte = compute_time_to_event(part4, visit_months, threshold, col)

        # Filter to analysis cohort
        tte = tte[tte["PATNO"].isin(patients_with_part4_and_posteriors)]

        # Merge with posteriors and baseline age
        surv_df = (
            tte
            .merge(posteriors, on="PATNO", how="inner")
            .merge(baseline_age, on="PATNO", how="left")
        )

        # Remove patients with time_to_event == 0 and no event (no information)
        surv_df = surv_df[~((surv_df["time_to_event"] == 0) & (surv_df["event_observed"] == 0))]

        # Compute predicted N(t)/N₀ at 3-year landmark for C3 model
        # This is a patient-specific trajectory feature that varies independently
        # from pct_loss (nonlinear transform) and captures predicted neuron reserve
        surv_df["n_frac_at_baseline"] = (
            (1 - surv_df["pct_loss_per_yr_median"] / 100) ** 3
        ).clip(lower=0.01)

        n_total = len(surv_df)
        n_events = int(surv_df["event_observed"].sum())
        n_censored = n_total - n_events
        median_tte = surv_df.loc[surv_df["event_observed"] == 1, "time_to_event"].median()

        print(f"  N patients: {n_total}")
        print(f"  N events: {n_events} ({100*n_events/n_total:.1f}%)")
        print(f"  N censored: {n_censored}")
        if n_events > 0:
            print(f"  Median time to event (among events): {median_tte:.1f} months")

        analysis_result = {
            "n_patients": n_total,
            "n_events": n_events,
            "n_censored": n_censored,
            "event_rate_pct": round(100 * n_events / n_total, 1) if n_total > 0 else 0,
            "median_time_to_event_months": round(float(median_tte), 1) if pd.notna(median_tte) else None,
        }

        # ── C1: KM by tertiles ────────────────────────────────────────
        print("\n  C1: Kaplan-Meier by neuron-loss-rate tertiles...")
        if n_events >= 10 and n_total >= 30:
            km_res = run_km_analysis(surv_df, label)
            km_for_plot.append(km_res)
            # Remove non-serializable fitters
            km_serializable = {k: v for k, v in km_res.items() if k != "_km_fitters"}
            analysis_result["C1_km"] = km_serializable
            print(f"    Log-rank (fast vs slow): p = {km_res['logrank_fast_vs_slow']['p_value']:.4f}")
            print(f"    Multivariate log-rank: p = {km_res['multivariate_logrank']['p_value']:.4f}")
            for g in ["slow", "medium", "fast"]:
                info = km_res["km_by_group"][g]
                med_str = f"{info['median_survival_months']:.1f}" if info["median_survival_months"] else "NR"
                print(f"    {g}: n={info['n']}, events={info['n_events']}, median={med_str} months")
        else:
            analysis_result["C1_km"] = {"skipped": f"Too few events ({n_events}) or patients ({n_total})"}
            print(f"    Skipped (too few events: {n_events})")

        # ── C2: Cox PH basic ──────────────────────────────────────────
        print("\n  C2: Cox PH (pct_loss + age)...")
        if n_events >= 10:
            cox_c2 = run_cox_analysis(surv_df, label, "C2")
            analysis_result["C2_cox"] = cox_c2
            if "concordance_index" in cox_c2:
                print(f"    C-index: {cox_c2['concordance_index']:.3f}")
                print(f"    LR test p: {cox_c2['log_likelihood_ratio_test_p']:.4f}")
                for cov, hr_info in cox_c2.get("hazard_ratios", {}).items():
                    print(f"    {cov}: HR={hr_info['hazard_ratio']:.3f} "
                          f"[{hr_info['ci_lower']:.3f}-{hr_info['ci_upper']:.3f}], p={hr_info['p']:.4f}")
            else:
                print(f"    Error: {cox_c2.get('error', 'unknown')}")
        else:
            analysis_result["C2_cox"] = {"skipped": f"Too few events ({n_events})"}
            print(f"    Skipped (too few events: {n_events})")

        # ── C3: Cox PH with trajectory features ───────────────────────
        print("\n  C3: Cox PH with N(t) trajectory features...")
        if n_events >= 10:
            cox_c3 = run_cox_analysis(surv_df, label, "C3")
            analysis_result["C3_cox_trajectory"] = cox_c3
            if "concordance_index" in cox_c3:
                print(f"    C-index: {cox_c3['concordance_index']:.3f}")
                for cov, hr_info in cox_c3.get("hazard_ratios", {}).items():
                    print(f"    {cov}: HR={hr_info['hazard_ratio']:.3f} "
                          f"[{hr_info['ci_lower']:.3f}-{hr_info['ci_upper']:.3f}], p={hr_info['p']:.4f}")
            else:
                print(f"    Error: {cox_c3.get('error', 'unknown')}")
        else:
            analysis_result["C3_cox_trajectory"] = {"skipped": f"Too few events ({n_events})"}
            print(f"    Skipped (too few events: {n_events})")

        # ── C4: Spearman correlation ──────────────────────────────────
        print("\n  C4: Spearman(pct_loss, time_to_event)...")
        spearman_res = run_spearman(surv_df, label)
        analysis_result["C4_spearman"] = spearman_res
        if "events_only" in spearman_res:
            ev = spearman_res["events_only"]
            print(f"    Events only (n={ev['n']}): rho={ev['spearman_rho']:.3f}, p={ev['p_value']:.4f}")
            al = spearman_res["all_patients"]
            print(f"    All patients (n={al['n']}): rho={al['spearman_rho']:.3f}, p={al['p_value']:.4f}")
        else:
            print(f"    Error: {spearman_res.get('error', 'unknown')}")

        all_results[key] = analysis_result

    # ── Plot KM curves ─────────────────────────────────────────────────
    if km_for_plot:
        plot_km_curves(km_for_plot, OUTPUT_DIR / "phase4_path_c_km_curves.png")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    primary = all_results.get("primary", {})
    print(f"Primary analysis (NP4OFF >= 1):")
    print(f"  N = {primary.get('n_patients', 0)}, events = {primary.get('n_events', 0)}")

    c2 = primary.get("C2_cox", {})
    if "concordance_index" in c2:
        print(f"  Cox C-index: {c2['concordance_index']:.3f}")
        pct_hr = c2.get("hazard_ratios", {}).get("pct_loss_per_yr_median", {})
        if pct_hr:
            print(f"  pct_loss HR: {pct_hr['hazard_ratio']:.3f} (p={pct_hr['p']:.4f})")

    c4 = primary.get("C4_spearman", {})
    if "events_only" in c4:
        print(f"  H2 Spearman rho: {c4['events_only']['spearman_rho']:.3f} (p={c4['events_only']['p_value']:.4f})")

    # ── Save results ───────────────────────────────────────────────────
    output = {
        "_provenance": provenance,
        "analyses": all_results,
    }

    results_path = OUTPUT_DIR / "phase4_path_c_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results: {results_path}")

    # ── Run manifest ───────────────────────────────────────────────────
    gate_results = {}
    if "primary" in all_results:
        p = all_results["primary"]
        gate_results["sufficient_events"] = p.get("n_events", 0) >= 10
        c2 = p.get("C2_cox", {})
        if "concordance_index" in c2:
            gate_results["cox_c_index_above_0.5"] = c2["concordance_index"] > 0.5

    write_run_manifest(
        manifest_path=OUTPUT_DIR / "phase4_path_c_RUN_MANIFEST.md",
        step_name="Phase 4 Path C: Wearing-Off Survival Analysis",
        provenance=provenance,
        gate_results=gate_results,
        summary_metrics={
            "n_patients": primary.get("n_patients", 0),
            "n_events": primary.get("n_events", 0),
            "event_rate": f"{primary.get('event_rate_pct', 0)}%",
        },
    )

    print("Done.")


if __name__ == "__main__":
    main()
