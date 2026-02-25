#!/usr/bin/env python3
"""
Paper 3, Step 2: Extract NSD-ISS stage transition events from longitudinal staging.

Identifies stage changes between consecutive visits, computes time-to-event,
handles right-censoring, and validates against Simuni et al. (2025) transition times.

Outputs:
  - data/06_longitudinal_staging/transition_events.csv
  - data/06_longitudinal_staging/censored_patients.csv
  - data/06_longitudinal_staging/cohort_summary.json

Reference transition times (Simuni et al. 2025, Movement Disorders):
  - 2B -> 3: median 1.19 years (1.1-2.0 CI)
  - 3 -> 4:  median 4.98 years (4.1-5.4 CI)
  - 4 -> 5:  median 9.77 years (7.0-NA CI)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Paths
LONGITUDINAL_CSV = PROJECT_ROOT / "data" / "06_longitudinal_staging" / "longitudinal_nsd_iss.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "06_longitudinal_staging"

# NSD-ISS stage ordering (numeric values for comparison)
STAGE_ORDER = {"0": 0, "1": 1, "2B": 2.5, "3": 3, "4": 4, "5": 5, "6": 6}

# Define which transitions are forward (progression) vs backward (regression)
# Forward: higher numeric stage. Backward: lower numeric stage.


def load_longitudinal_data() -> pd.DataFrame:
    """Load, deduplicate same-timepoint observations, and sort longitudinal staging data.

    When multiple observations exist at the same months_from_baseline for a patient
    (e.g., SC and BL both at month 0), we keep only one per timepoint. Priority:
    BL > V** > SC (BL is the official baseline assessment). This prevents
    same-timepoint staging differences from being counted as transitions.
    """
    df = pd.read_csv(LONGITUDINAL_CSV)
    # Ensure stage is string
    df["nsd_stage"] = df["nsd_stage"].astype(str)

    n_before = len(df)

    # Event priority for dedup: lower = preferred
    EVENT_PRIORITY = {"BL": 0}
    # V01-V22, R01-R20 get priority 1-50
    for i in range(1, 23):
        EVENT_PRIORITY[f"V{i:02d}"] = i
    for i in range(1, 21):
        EVENT_PRIORITY[f"R{i:02d}"] = 50 + i
    EVENT_PRIORITY["SC"] = 100  # SC is lowest priority (screening, less reliable)
    EVENT_PRIORITY["SC99"] = 101
    EVENT_PRIORITY["ST"] = 102
    EVENT_PRIORITY["PW"] = 103
    EVENT_PRIORITY["U01"] = 104
    EVENT_PRIORITY["U02"] = 105
    EVENT_PRIORITY["RS1"] = 106

    df["_event_priority"] = df["EVENT_ID"].map(EVENT_PRIORITY).fillna(200)

    # Sort by patient, time, then priority (keep best per timepoint)
    df = df.sort_values(["PATNO", "months_from_baseline", "_event_priority"])
    df = df.drop_duplicates(subset=["PATNO", "months_from_baseline"], keep="first")
    df = df.drop(columns=["_event_priority"])
    df = df.sort_values(["PATNO", "months_from_baseline"]).reset_index(drop=True)

    n_after = len(df)
    print(f"Loaded {n_before} observations, deduplicated to {n_after} "
          f"({n_before - n_after} same-timepoint duplicates removed)")
    print(f"  {df['PATNO'].nunique()} patients")
    return df


def extract_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract stage transitions between consecutive visits for each patient.

    Returns DataFrame with one row per transition event.
    """
    transitions = []

    for patno, group in df.groupby("PATNO"):
        group = group.sort_values("months_from_baseline")
        stages = group["nsd_stage"].values
        months = group["months_from_baseline"].values
        event_ids = group["EVENT_ID"].values
        ages = group["age_at_visit"].values
        cohort = group["cohort"].iloc[0]

        for i in range(len(stages) - 1):
            src_stage = stages[i]
            dst_stage = stages[i + 1]

            if src_stage != dst_stage:
                src_num = STAGE_ORDER.get(src_stage, np.nan)
                dst_num = STAGE_ORDER.get(dst_stage, np.nan)

                if np.isnan(src_num) or np.isnan(dst_num):
                    continue

                direction = "forward" if dst_num > src_num else "backward"
                skip_distance = abs(dst_num - src_num)
                # Skip transitions are those jumping more than one stage level
                is_skip = skip_distance > 1.0 and not (
                    (src_stage == "2B" and dst_stage == "3") or
                    (src_stage == "3" and dst_stage == "2B")
                )

                time_interval = months[i + 1] - months[i]

                transitions.append({
                    "PATNO": patno,
                    "cohort": cohort,
                    "source_stage": src_stage,
                    "dest_stage": dst_stage,
                    "source_stage_numeric": src_num,
                    "dest_stage_numeric": dst_num,
                    "direction": direction,
                    "is_skip_transition": is_skip,
                    "time_interval_months": time_interval,
                    "time_interval_years": time_interval / 12.0,
                    "months_from_baseline_src": months[i],
                    "months_from_baseline_dst": months[i + 1],
                    "event_id_from": event_ids[i],
                    "event_id_to": event_ids[i + 1],
                    "age_at_transition": ages[i + 1],
                })

    trans_df = pd.DataFrame(transitions)
    print(f"\nExtracted {len(trans_df)} transition events from "
          f"{trans_df['PATNO'].nunique()} patients")
    return trans_df


def extract_censored_patients(df: pd.DataFrame, trans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify right-censored patients: those with no observed transition.
    Also creates per-stage censoring records for KM analysis.

    For KM analysis of specific transitions (e.g., 2B->3), a patient is censored
    if they were observed in stage 2B but never transitioned to 3 (or higher).
    """
    censored = []

    for patno, group in df.groupby("PATNO"):
        group = group.sort_values("months_from_baseline")
        first_stage = group["nsd_stage"].iloc[0]
        last_stage = group["nsd_stage"].iloc[-1]
        total_follow_up = group["months_from_baseline"].max() - group["months_from_baseline"].min()
        n_visits = len(group)
        cohort = group["cohort"].iloc[0]

        # Check if patient had ANY transition
        patient_trans = trans_df[trans_df["PATNO"] == patno] if len(trans_df) > 0 else pd.DataFrame()
        had_transition = len(patient_trans) > 0
        had_forward = len(patient_trans[patient_trans["direction"] == "forward"]) > 0 if had_transition else False
        had_backward = len(patient_trans[patient_trans["direction"] == "backward"]) > 0 if had_transition else False

        censored.append({
            "PATNO": patno,
            "cohort": cohort,
            "baseline_stage": first_stage,
            "last_observed_stage": last_stage,
            "total_follow_up_months": total_follow_up,
            "total_follow_up_years": total_follow_up / 12.0,
            "n_visits": n_visits,
            "had_any_transition": had_transition,
            "had_forward_transition": had_forward,
            "had_backward_transition": had_backward,
            "n_forward_transitions": len(patient_trans[patient_trans["direction"] == "forward"]) if had_transition else 0,
            "n_backward_transitions": len(patient_trans[patient_trans["direction"] == "backward"]) if had_transition else 0,
        })

    cens_df = pd.DataFrame(censored)
    print(f"\nPatient-level summary: {len(cens_df)} patients")
    print(f"  Had any transition: {cens_df['had_any_transition'].sum()} ({cens_df['had_any_transition'].mean():.1%})")
    print(f"  Had forward transition: {cens_df['had_forward_transition'].sum()} ({cens_df['had_forward_transition'].mean():.1%})")
    print(f"  Had backward transition: {cens_df['had_backward_transition'].sum()} ({cens_df['had_backward_transition'].mean():.1%})")
    return cens_df


def compute_transition_matrix(trans_df: pd.DataFrame) -> pd.DataFrame:
    """Compute transition frequency matrix (from x to)."""
    stages = ["0", "1", "2B", "3", "4", "5", "6"]
    matrix = pd.DataFrame(0, index=stages, columns=stages)

    for _, row in trans_df.iterrows():
        src = row["source_stage"]
        dst = row["dest_stage"]
        if src in stages and dst in stages:
            matrix.loc[src, dst] += 1

    return matrix


def compute_first_transition_km(df: pd.DataFrame, trans_df: pd.DataFrame,
                                 source_stage: str, dest_stages: list) -> dict:
    """
    Compute Kaplan-Meier estimate for time from first observation in source_stage
    to first transition to any of dest_stages.

    This mirrors Simuni et al.'s methodology: time from baseline stage assignment
    to first forward transition.
    """
    km_data = []

    for patno, group in df.groupby("PATNO"):
        group = group.sort_values("months_from_baseline")

        # Find first observation at source stage
        src_obs = group[group["nsd_stage"] == source_stage]
        if len(src_obs) == 0:
            continue

        first_src_time = src_obs["months_from_baseline"].iloc[0]

        # Find first transition from source to any dest stage
        pat_trans = trans_df[
            (trans_df["PATNO"] == patno) &
            (trans_df["source_stage"] == source_stage) &
            (trans_df["dest_stage"].isin(dest_stages))
        ]

        if len(pat_trans) > 0:
            first_trans = pat_trans.sort_values("months_from_baseline_dst").iloc[0]
            # Use destination time: transition is observed at the DESTINATION visit
            time_to_event = first_trans["months_from_baseline_dst"] - first_src_time
            event = True
        else:
            # Censored: use time from first source observation to last observation
            last_obs_time = group["months_from_baseline"].max()
            time_to_event = last_obs_time - first_src_time
            event = False

        # Only include if some follow-up time (>0 months)
        if time_to_event > 0:
            km_data.append({
                "PATNO": patno,
                "time_months": time_to_event,
                "time_years": time_to_event / 12.0,
                "event": event,
            })

    if not km_data:
        return {"n_patients": 0, "n_events": 0, "median_years": None}

    km_df = pd.DataFrame(km_data)

    # Fit KM
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=km_df["time_years"],
        event_observed=km_df["event"],
        label=f"{source_stage} → {'/'.join(dest_stages)}"
    )

    median = kmf.median_survival_time_

    # Compute CI for median via survival function CI
    # lifelines 0.30+ doesn't have confidence_interval_median_survival_time_
    # Instead, find where the CI bounds cross 0.5
    ci_sf = kmf.confidence_interval_survival_function_
    try:
        lower_col = ci_sf.columns[0]  # lower bound of S(t)
        upper_col = ci_sf.columns[1]  # upper bound of S(t)
        # Median CI lower bound: time when UPPER S(t) CI first drops to 0.5
        # (optimistic survival → later median → this is the CI lower bound)
        crosses_upper = ci_sf[upper_col] <= 0.5
        ci_lower = float(crosses_upper.idxmax()) if crosses_upper.any() else None
        # Median CI upper bound: time when LOWER S(t) CI first drops to 0.5
        # (pessimistic survival → earlier median... but actually lower S(t)
        #  means WORSE prognosis → median is EARLIER, so this is CI lower)
        # Correction: upper CI of S(t) crossing 0.5 gives LOWER CI of median
        #             lower CI of S(t) crossing 0.5 gives UPPER CI of median
        crosses_lower = ci_sf[lower_col] <= 0.5
        ci_upper = float(crosses_lower.idxmax()) if crosses_lower.any() else None
    except Exception:
        ci_lower = None
        ci_upper = None

    # Ensure CI ordering is correct (lower <= median <= upper)
    if ci_lower is not None and ci_upper is not None and ci_lower > ci_upper:
        ci_lower, ci_upper = ci_upper, ci_lower

    result = {
        "n_patients": len(km_df),
        "n_events": int(km_df["event"].sum()),
        "n_censored": int((~km_df["event"]).sum()),
        "event_rate": float(km_df["event"].mean()),
        "median_years": float(median) if not np.isinf(median) else None,
        "median_ci_lower": ci_lower,
        "median_ci_upper": ci_upper,
        "survival_table": kmf.survival_function_.to_dict(),
    }

    return result


def compute_regression_analysis(trans_df: pd.DataFrame, cens_df: pd.DataFrame) -> dict:
    """
    Analyze stage regression patterns.

    Espay et al. (2025) argued ~50% of stage changes are treatment-driven regression.
    We compute actual regression rates from our data.
    """
    # Overall regression rate among patients with any transition
    patients_with_trans = cens_df[cens_df["had_any_transition"]]

    # Among NSD+ patients who transitioned
    nsd_stages = ["1", "2B", "3", "4", "5", "6"]
    nsd_patients = cens_df[cens_df["baseline_stage"].isin(nsd_stages)]
    nsd_with_trans = nsd_patients[nsd_patients["had_any_transition"]]

    forward_trans = trans_df[trans_df["direction"] == "forward"]
    backward_trans = trans_df[trans_df["direction"] == "backward"]

    # Regression rate per source stage
    regression_by_stage = {}
    for stage in nsd_stages:
        stage_trans = trans_df[trans_df["source_stage"] == stage]
        if len(stage_trans) > 0:
            n_back = len(stage_trans[stage_trans["direction"] == "backward"])
            regression_by_stage[stage] = {
                "n_transitions": len(stage_trans),
                "n_backward": n_back,
                "regression_rate": n_back / len(stage_trans),
            }

    # Medication status at regression transitions
    # pdmedyn: 0 = off meds, 1 = on meds, NaN = unknown
    if "pdmedyn" in trans_df.columns:
        # Need to join back to get medication status
        pass

    result = {
        "total_transitions": len(trans_df),
        "forward_transitions": len(forward_trans),
        "backward_transitions": len(backward_trans),
        "overall_regression_rate": len(backward_trans) / len(trans_df) if len(trans_df) > 0 else 0,
        "nsd_positive_patients_total": len(nsd_patients),
        "nsd_positive_with_transition": len(nsd_with_trans),
        "nsd_forward_rate": nsd_with_trans["had_forward_transition"].mean() if len(nsd_with_trans) > 0 else 0,
        "nsd_backward_rate": nsd_with_trans["had_backward_transition"].mean() if len(nsd_with_trans) > 0 else 0,
        "regression_by_source_stage": regression_by_stage,
    }

    return result


def analyze_medication_confound(df: pd.DataFrame, trans_df: pd.DataFrame) -> dict:
    """
    Analyze the medication confound on stage transitions.

    Espay et al. (2025) argued that levodopa masks functional impairment,
    causing apparent stage regression. We check medication status at transitions.
    """
    # Join medication status from longitudinal data to transitions
    med_status = df[["PATNO", "EVENT_ID", "pdmedyn"]].copy()
    med_status = med_status.rename(columns={"EVENT_ID": "event_id_to", "pdmedyn": "pdmedyn_at_transition"})

    trans_with_meds = trans_df.merge(med_status, on=["PATNO", "event_id_to"], how="left")

    forward = trans_with_meds[trans_with_meds["direction"] == "forward"]
    backward = trans_with_meds[trans_with_meds["direction"] == "backward"]

    result = {
        "forward_on_meds": int(forward["pdmedyn_at_transition"].eq(1.0).sum()),
        "forward_off_meds": int(forward["pdmedyn_at_transition"].eq(0.0).sum()),
        "forward_unknown_meds": int(forward["pdmedyn_at_transition"].isna().sum()),
        "backward_on_meds": int(backward["pdmedyn_at_transition"].eq(1.0).sum()),
        "backward_off_meds": int(backward["pdmedyn_at_transition"].eq(0.0).sum()),
        "backward_unknown_meds": int(backward["pdmedyn_at_transition"].isna().sum()),
    }

    # Compute rates
    for direction, dir_df in [("forward", forward), ("backward", backward)]:
        known = dir_df["pdmedyn_at_transition"].notna()
        if known.sum() > 0:
            result[f"{direction}_on_meds_rate"] = float(
                dir_df.loc[known, "pdmedyn_at_transition"].eq(1.0).mean()
            )
        else:
            result[f"{direction}_on_meds_rate"] = None

    return result


def print_summary(trans_df: pd.DataFrame, cens_df: pd.DataFrame,
                  trans_matrix: pd.DataFrame, km_results: dict,
                  regression_analysis: dict, med_analysis: dict):
    """Print comprehensive summary to console."""
    print("\n" + "=" * 80)
    print("PAPER 3 STEP 2: TRANSITION EVENT EXTRACTION SUMMARY")
    print("=" * 80)

    # Transition counts
    print("\n--- Transition Event Summary ---")
    print(f"Total transitions: {len(trans_df)}")
    print(f"  Forward (progression):  {regression_analysis['forward_transitions']}")
    print(f"  Backward (regression):  {regression_analysis['backward_transitions']}")
    print(f"  Overall regression rate: {regression_analysis['overall_regression_rate']:.1%}")

    # Skip transitions
    skip = trans_df[trans_df["is_skip_transition"]]
    print(f"  Skip transitions: {len(skip)}")
    if len(skip) > 0:
        for _, row in skip.groupby(["source_stage", "dest_stage"]).size().reset_index(name="count").iterrows():
            print(f"    {row['source_stage']} → {row['dest_stage']}: {row['count']}")

    # Transition matrix
    print("\n--- Transition Frequency Matrix ---")
    print(trans_matrix.to_string())

    # Regression by stage
    print("\n--- Regression Rate by Source Stage ---")
    for stage, data in regression_analysis["regression_by_source_stage"].items():
        print(f"  Stage {stage}: {data['n_backward']}/{data['n_transitions']} "
              f"= {data['regression_rate']:.1%} regression")

    # KM results vs Simuni
    print("\n--- Kaplan-Meier Transition Times (vs Simuni 2025) ---")
    simuni_ref = {
        "2B_to_3": {"median": 1.19, "ci": "1.1-2.0"},
        "3_to_4": {"median": 4.98, "ci": "4.1-5.4"},
        "4_to_5": {"median": 9.77, "ci": "7.0-NA"},
    }

    for key, result in km_results.items():
        ref = simuni_ref.get(key, {})
        ref_str = f" (Simuni: {ref['median']:.2f}yr [{ref['ci']}])" if ref else ""
        median_str = f"{result['median_years']:.2f}" if result['median_years'] is not None else "not reached"
        ci_str = ""
        if result.get("median_ci_lower") is not None or result.get("median_ci_upper") is not None:
            lo = f"{result['median_ci_lower']:.2f}" if result['median_ci_lower'] is not None else "NA"
            hi = f"{result['median_ci_upper']:.2f}" if result['median_ci_upper'] is not None else "NA"
            ci_str = f" [{lo}-{hi}]"

        print(f"  {key}: n={result['n_patients']}, events={result['n_events']}, "
              f"median={median_str}yr{ci_str}{ref_str}")

    # Medication confound
    print("\n--- Medication Confound Analysis ---")
    print(f"  Forward transitions on meds: {med_analysis['forward_on_meds']} "
          f"(rate: {med_analysis.get('forward_on_meds_rate', 'N/A')})")
    print(f"  Backward transitions on meds: {med_analysis['backward_on_meds']} "
          f"(rate: {med_analysis.get('backward_on_meds_rate', 'N/A')})")

    # Patient-level
    print("\n--- Patient-Level Summary ---")
    nsd_stages = ["1", "2B", "3", "4", "5", "6"]
    nsd = cens_df[cens_df["baseline_stage"].isin(nsd_stages)]
    print(f"  NSD+ patients (baseline): {len(nsd)}")
    print(f"    With any transition: {nsd['had_any_transition'].sum()} ({nsd['had_any_transition'].mean():.1%})")
    print(f"    With forward transition: {nsd['had_forward_transition'].sum()} ({nsd['had_forward_transition'].mean():.1%})")
    print(f"    With backward transition: {nsd['had_backward_transition'].sum()} ({nsd['had_backward_transition'].mean():.1%})")

    # By baseline stage
    print("\n  Transition rates by baseline stage:")
    for stage in nsd_stages:
        stage_pats = nsd[nsd["baseline_stage"] == stage]
        if len(stage_pats) > 0:
            fwd_rate = stage_pats["had_forward_transition"].mean()
            bwd_rate = stage_pats["had_backward_transition"].mean()
            median_fu = stage_pats["total_follow_up_months"].median() / 12
            print(f"    Stage {stage:>2s}: n={len(stage_pats):>4d}, "
                  f"fwd={fwd_rate:.1%}, bwd={bwd_rate:.1%}, "
                  f"median FU={median_fu:.1f}yr")

    # Follow-up times
    print("\n--- Follow-up Duration ---")
    print(f"  Mean:   {cens_df['total_follow_up_months'].mean() / 12:.1f} years")
    print(f"  Median: {cens_df['total_follow_up_months'].median() / 12:.1f} years")
    print(f"  Max:    {cens_df['total_follow_up_months'].max() / 12:.1f} years")


def build_km_survival_data(df: pd.DataFrame, trans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-patient per-stage KM-ready dataset.

    For each patient and each stage they occupy, compute:
    - Time spent in that stage
    - Whether they exited (event=1) or were censored (event=0)
    - If exited: which direction (forward/backward)

    This is the input format needed for multi-state survival models (Step 4).
    """
    records = []

    for patno, group in df.groupby("PATNO"):
        group = group.sort_values("months_from_baseline")
        cohort = group["cohort"].iloc[0]

        # Identify contiguous stage episodes
        stages = group["nsd_stage"].values
        months = group["months_from_baseline"].values

        episode_start_idx = 0
        for i in range(1, len(stages)):
            if stages[i] != stages[episode_start_idx]:
                # Stage changed: episode ended at visit i-1, transition at visit i
                entry_time = months[episode_start_idx]
                exit_time = months[i]
                duration = exit_time - entry_time

                src_num = STAGE_ORDER.get(stages[episode_start_idx], np.nan)
                dst_num = STAGE_ORDER.get(stages[i], np.nan)
                direction = "forward" if dst_num > src_num else "backward"

                records.append({
                    "PATNO": patno,
                    "cohort": cohort,
                    "stage": stages[episode_start_idx],
                    "entry_time_months": entry_time,
                    "exit_time_months": exit_time,
                    "duration_months": duration,
                    "duration_years": duration / 12.0,
                    "event": 1,  # Transition observed
                    "exit_to": stages[i],
                    "exit_direction": direction,
                })

                episode_start_idx = i

        # Last episode: censored (no transition observed after last stage change)
        entry_time = months[episode_start_idx]
        exit_time = months[-1]
        duration = exit_time - entry_time

        records.append({
            "PATNO": patno,
            "cohort": cohort,
            "stage": stages[episode_start_idx],
            "entry_time_months": entry_time,
            "exit_time_months": exit_time,
            "duration_months": duration,
            "duration_years": duration / 12.0,
            "event": 0,  # Censored
            "exit_to": None,
            "exit_direction": None,
        })

    surv_df = pd.DataFrame(records)
    print(f"\nBuilt KM survival dataset: {len(surv_df)} stage episodes "
          f"from {surv_df['PATNO'].nunique()} patients")
    print(f"  Events: {surv_df['event'].sum()}, Censored: {(surv_df['event'] == 0).sum()}")
    return surv_df


def compute_simuni_aligned_km(df: pd.DataFrame, trans_df: pd.DataFrame,
                               baseline_stage: str, dest_stages: list) -> dict:
    """
    Compute KM estimate matching Simuni et al. (2025) methodology exactly:
    - Time origin = enrollment (baseline), NOT first observation at stage
    - Only include patients whose BASELINE stage matches baseline_stage
    - Only include NSD-positive patients (Stage 1+)
    - Only include PD and Prodromal cohorts (matching Simuni's 494 PD + 74 prodromal)

    This differs from compute_first_transition_km which uses first-at-stage origin
    and includes patients who ENTER the stage during follow-up.
    """
    km_data = []

    # Simuni included PD + Prodromal (not HC, not SWEDD)
    simuni_cohorts = {"Parkinson's Disease", "Prodromal"}

    for patno, group in df.groupby("PATNO"):
        group = group.sort_values("months_from_baseline")

        # Must be in Simuni-compatible cohort
        cohort = group["cohort"].iloc[0]
        if cohort not in simuni_cohorts:
            continue

        # Must have baseline stage matching target
        baseline_obs = group[group["months_from_baseline"] == 0]
        if len(baseline_obs) == 0:
            # Use first observation as proxy
            baseline_obs = group.iloc[:1]
        bl_stage = baseline_obs["nsd_stage"].iloc[0]

        if bl_stage != baseline_stage:
            continue

        # Must be NSD-positive (D+ or S+)
        d_pos = baseline_obs["d_positive"].iloc[0]
        s_pos = baseline_obs["s_positive"].iloc[0]
        if not (d_pos is True or s_pos is True or str(d_pos) == "True" or str(s_pos) == "True"):
            continue

        # Time origin = baseline (month 0)
        # Look for first forward transition to any dest stage
        # (using ANY stage transition, not just from baseline_stage,
        #  matching Simuni's "first increase in stage from baseline")
        first_stage_increase = None
        for _, obs in group.iterrows():
            obs_stage = obs["nsd_stage"]
            obs_num = STAGE_ORDER.get(obs_stage, -1)
            bl_num = STAGE_ORDER.get(baseline_stage, -1)
            if obs_num > bl_num and obs_stage in dest_stages:
                first_stage_increase = obs
                break

        if first_stage_increase is not None:
            time_to_event = first_stage_increase["months_from_baseline"]
            event = True
        else:
            # Censored at last follow-up
            time_to_event = group["months_from_baseline"].max()
            event = False

        if time_to_event > 0:
            km_data.append({
                "PATNO": patno,
                "time_months": time_to_event,
                "time_years": time_to_event / 12.0,
                "event": event,
            })

    if not km_data:
        return {"n_patients": 0, "n_events": 0, "median_years": None}

    km_df = pd.DataFrame(km_data)

    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=km_df["time_years"],
        event_observed=km_df["event"],
        label=f"Simuni-aligned: {baseline_stage} → {'/'.join(dest_stages)}"
    )

    median = kmf.median_survival_time_

    ci_sf = kmf.confidence_interval_survival_function_
    try:
        lower_col = ci_sf.columns[0]
        upper_col = ci_sf.columns[1]
        crosses_upper = ci_sf[upper_col] <= 0.5
        ci_lower = float(crosses_upper.idxmax()) if crosses_upper.any() else None
        crosses_lower = ci_sf[lower_col] <= 0.5
        ci_upper = float(crosses_lower.idxmax()) if crosses_lower.any() else None
    except Exception:
        ci_lower = None
        ci_upper = None

    if ci_lower is not None and ci_upper is not None and ci_lower > ci_upper:
        ci_lower, ci_upper = ci_upper, ci_lower

    return {
        "n_patients": len(km_df),
        "n_events": int(km_df["event"].sum()),
        "n_censored": int((~km_df["event"]).sum()),
        "event_rate": float(km_df["event"].mean()),
        "median_years": float(median) if not np.isinf(median) else None,
        "median_ci_lower": ci_lower,
        "median_ci_upper": ci_upper,
    }


def compute_updrs2_staging_comparison(df: pd.DataFrame) -> dict:
    """
    Compare our H&Y-based staging with Simuni/Dam UPDRS Part II-based staging.

    Dam et al. (2024) thresholds for functional impairment:
      Stage 2B: UPDRS-II < 3 (no functional impairment)
      Stage 3:  UPDRS-II 3-13 (slight)
      Stage 4:  UPDRS-II 14-26 (mild)
      Stage 5:  UPDRS-II 27-39 (moderate)
      Stage 6:  UPDRS-II >= 40 (severe)

    Our code uses H&Y: <2 = none, 2-3 = mild, 3-4 = moderate, 4-5 = severe, 5 = complete

    This function loads UPDRS Part II data and checks staging agreement.
    """
    updrs2_path = PROJECT_ROOT / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv"

    # Try to find UPDRS Part II file
    updrs2_files = list(updrs2_path.glob("MDS_UPDRS_Part_II*"))
    if not updrs2_files:
        updrs2_files = list(updrs2_path.glob("*Part_II*Patient*"))
    if not updrs2_files:
        return {"error": "UPDRS Part II file not found"}

    updrs2_csv = updrs2_files[0]
    print(f"\n  Loading UPDRS Part II from: {updrs2_csv.name}")
    u2_df = pd.read_csv(updrs2_csv)

    # Compute UPDRS-II total if not present
    if "NP2TOT" in u2_df.columns:
        total_col = "NP2TOT"
    else:
        # Sum NP2 items
        np2_cols = [c for c in u2_df.columns if c.startswith("NP2") and c != "NP2TOT"]
        if np2_cols:
            for c in np2_cols:
                u2_df[c] = pd.to_numeric(u2_df[c], errors="coerce")
            u2_df["NP2TOT"] = u2_df[np2_cols].sum(axis=1, min_count=1)
            total_col = "NP2TOT"
        else:
            return {"error": "Cannot compute UPDRS-II total"}

    # Merge with longitudinal data on PATNO + EVENT_ID
    u2_df["PATNO"] = u2_df["PATNO"].astype(int)
    u2_slim = u2_df[["PATNO", "EVENT_ID", total_col]].dropna(subset=[total_col])
    u2_slim = u2_slim.drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="last")

    merged = df.merge(u2_slim, on=["PATNO", "EVENT_ID"], how="inner")
    print(f"  Merged UPDRS-II data: {len(merged)} observations (of {len(df)} total)")

    # Apply Dam et al. thresholds
    def dam_stage(updrs2_total, has_bio, has_clinical):
        """Stage using Dam et al. UPDRS-II thresholds."""
        if not has_bio:
            return "0"
        if not has_clinical:
            return "1"
        if pd.isna(updrs2_total):
            return None
        u2 = float(updrs2_total)
        if u2 < 3:
            return "2B"
        elif u2 <= 13:
            return "3"
        elif u2 <= 26:
            return "4"
        elif u2 <= 39:
            return "5"
        else:
            return "6"

    merged["dam_stage"] = merged.apply(
        lambda r: dam_stage(
            r[total_col],
            str(r["d_positive"]) == "True" or str(r["s_positive"]) == "True",
            r["has_clinical_signs"]
        ), axis=1
    )

    # Compare with our H&Y-based staging
    valid = merged.dropna(subset=["dam_stage"])
    agreement = (valid["nsd_stage"] == valid["dam_stage"]).mean()

    # Stage-by-stage confusion
    confusion = pd.crosstab(
        valid["nsd_stage"], valid["dam_stage"],
        rownames=["Our_HY_stage"], colnames=["Dam_UPDRS2_stage"]
    )

    # Count key disagreements
    # How many we call 2B that Dam would call 3?
    hy_2b_dam_3 = len(valid[(valid["nsd_stage"] == "2B") & (valid["dam_stage"] == "3")])
    hy_3_dam_2b = len(valid[(valid["nsd_stage"] == "3") & (valid["dam_stage"] == "2B")])
    hy_3_dam_4 = len(valid[(valid["nsd_stage"] == "3") & (valid["dam_stage"] == "4")])
    hy_4_dam_3 = len(valid[(valid["nsd_stage"] == "4") & (valid["dam_stage"] == "3")])

    result = {
        "n_observations_compared": len(valid),
        "overall_agreement": float(agreement),
        "confusion_matrix": confusion.to_dict(),
        "key_disagreements": {
            "our_2B_dam_3": hy_2b_dam_3,
            "our_3_dam_2B": hy_3_dam_2b,
            "our_3_dam_4": hy_3_dam_4,
            "our_4_dam_3": hy_4_dam_3,
        },
    }

    print(f"  Overall staging agreement (H&Y vs UPDRS-II): {agreement:.1%}")
    print(f"  Our 2B → Dam 3: {hy_2b_dam_3} (we call 2B, Dam says 3)")
    print(f"  Our 3 → Dam 2B: {hy_3_dam_2b} (we call 3, Dam says 2B)")
    print(f"  Our 3 → Dam 4:  {hy_3_dam_4} (we call 3, Dam says 4)")
    print(f"  Our 4 → Dam 3:  {hy_4_dam_3} (we call 4, Dam says 3)")

    return result


def main():
    print("=" * 80)
    print("Paper 3, Step 2: Extract NSD-ISS Stage Transitions")
    print("=" * 80)

    # Load data
    df = load_longitudinal_data()

    # Extract transition events
    trans_df = extract_transitions(df)

    # Extract censored patient records
    cens_df = extract_censored_patients(df, trans_df)

    # Compute transition matrix
    trans_matrix = compute_transition_matrix(trans_df)

    # Compute KM estimates for key transitions (our method: first-at-stage origin)
    km_results = {}

    # 2B -> 3 (or higher): forward from Stage 2B
    km_results["2B_to_3"] = compute_first_transition_km(
        df, trans_df, source_stage="2B", dest_stages=["3", "4", "5", "6"]
    )

    # 3 -> 4 (or higher): forward from Stage 3
    km_results["3_to_4"] = compute_first_transition_km(
        df, trans_df, source_stage="3", dest_stages=["4", "5", "6"]
    )

    # 4 -> 5 (or higher): forward from Stage 4
    km_results["4_to_5"] = compute_first_transition_km(
        df, trans_df, source_stage="4", dest_stages=["5", "6"]
    )

    # Compute Simuni-aligned KM (baseline origin, NSD+ PD/Prodromal only)
    print("\n--- Simuni-Aligned KM Analysis ---")
    print("  (Baseline origin, NSD+ PD/Prodromal only)")
    simuni_km = {}
    simuni_km["2B_to_3"] = compute_simuni_aligned_km(
        df, trans_df, baseline_stage="2B", dest_stages=["3", "4", "5", "6"]
    )
    simuni_km["3_to_4"] = compute_simuni_aligned_km(
        df, trans_df, baseline_stage="3", dest_stages=["4", "5", "6"]
    )
    simuni_km["4_to_5"] = compute_simuni_aligned_km(
        df, trans_df, baseline_stage="4", dest_stages=["5", "6"]
    )

    # Print Simuni comparison
    simuni_ref = {
        "2B_to_3": {"median": 1.19, "ci": "1.1-2.0"},
        "3_to_4": {"median": 4.98, "ci": "4.1-5.4"},
        "4_to_5": {"median": 9.77, "ci": "7.0-NA"},
    }
    for key, result in simuni_km.items():
        ref = simuni_ref[key]
        median_str = f"{result['median_years']:.2f}" if result['median_years'] is not None else "not reached"
        ci_lo = f"{result['median_ci_lower']:.2f}" if result.get('median_ci_lower') is not None else "NA"
        ci_hi = f"{result['median_ci_upper']:.2f}" if result.get('median_ci_upper') is not None else "NA"
        print(f"  {key}: n={result['n_patients']}, events={result['n_events']}, "
              f"median={median_str}yr [{ci_lo}-{ci_hi}] "
              f"(Simuni: {ref['median']:.2f}yr [{ref['ci']}])")

    # UPDRS Part II staging comparison
    print("\n--- UPDRS Part II Staging Comparison (Dam et al. 2024) ---")
    updrs2_comparison = compute_updrs2_staging_comparison(df)

    # Regression analysis
    regression_analysis = compute_regression_analysis(trans_df, cens_df)

    # Medication confound analysis
    med_analysis = analyze_medication_confound(df, trans_df)

    # Build KM survival dataset (for Step 4 multi-state model)
    surv_df = build_km_survival_data(df, trans_df)

    # Print full summary
    print_summary(trans_df, cens_df, trans_matrix, km_results,
                  regression_analysis, med_analysis)

    # Save outputs
    print("\n--- Saving Outputs ---")

    # Transition events
    trans_path = OUTPUT_DIR / "transition_events.csv"
    trans_df.to_csv(trans_path, index=False)
    print(f"  Transition events: {trans_path} ({len(trans_df)} rows)")

    # Censored patients
    cens_path = OUTPUT_DIR / "censored_patients.csv"
    cens_df.to_csv(cens_path, index=False)
    print(f"  Censored patients: {cens_path} ({len(cens_df)} rows)")

    # KM survival dataset
    surv_path = OUTPUT_DIR / "stage_episodes.csv"
    surv_df.to_csv(surv_path, index=False)
    print(f"  Stage episodes: {surv_path} ({len(surv_df)} rows)")

    # Cohort summary JSON
    # Strip survival_table from KM results for JSON (too large)
    km_json = {}
    for key, result in km_results.items():
        km_json[key] = {k: v for k, v in result.items() if k != "survival_table"}

    summary = {
        "total_observations": len(df),
        "unique_patients": int(df["PATNO"].nunique()),
        "transition_events": {
            "total": len(trans_df),
            "forward": int(regression_analysis["forward_transitions"]),
            "backward": int(regression_analysis["backward_transitions"]),
            "regression_rate": float(regression_analysis["overall_regression_rate"]),
            "unique_patients_with_transitions": int(trans_df["PATNO"].nunique()) if len(trans_df) > 0 else 0,
        },
        "transition_matrix": trans_matrix.to_dict(),
        "kaplan_meier": km_json,
        "regression_analysis": {
            k: v for k, v in regression_analysis.items()
            if k != "regression_by_source_stage"
        },
        "regression_by_stage": regression_analysis["regression_by_source_stage"],
        "medication_confound": med_analysis,
        "stage_episodes": {
            "total_episodes": len(surv_df),
            "events": int(surv_df["event"].sum()),
            "censored": int((surv_df["event"] == 0).sum()),
        },
        "follow_up": {
            "mean_years": float(cens_df["total_follow_up_months"].mean() / 12),
            "median_years": float(cens_df["total_follow_up_months"].median() / 12),
            "max_years": float(cens_df["total_follow_up_months"].max() / 12),
        },
        "simuni_2025_reference": {
            "2B_to_3_median_years": 1.19,
            "3_to_4_median_years": 4.98,
            "4_to_5_median_years": 9.77,
        },
        "simuni_aligned_km": simuni_km,
        "updrs2_staging_comparison": {
            k: v for k, v in updrs2_comparison.items()
            if k != "confusion_matrix"
        } if isinstance(updrs2_comparison, dict) else {},
        "staging_methodology_note": (
            "Our staging uses H&Y stage thresholds for functional impairment. "
            "Simuni/Dam et al. use MDS-UPDRS Part II thresholds "
            "(3-13=slight, 14-26=mild, 27-39=moderate, >=40=severe). "
            "This difference affects the 2B/3 boundary most significantly."
        ),
    }

    summary_path = OUTPUT_DIR / "cohort_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Cohort summary: {summary_path}")

    print("\nStep 2 complete!")


if __name__ == "__main__":
    main()
