"""Extract real phenoconversion endpoints from PPMI Primary Clinical Diagnosis table.

This script derives survival endpoints exclusively from PPMI-observed data:
- Phenoconversion: Prodromal patient receiving a PD diagnosis (PRIMDIAG=1) at follow-up
- Time-to-event: Months from baseline to first PD diagnosis visit
- Censoring: Last diagnosis assessment visit for non-converters

NO simulated, synthetic, or hybrid endpoints are generated.
NO UPDRS-III threshold proxies are used — only actual clinical PD diagnosis.

Data sources (all from PPMI):
- Participant_Status: Cohort membership (Prodromal)
- Primary_Clinical_Diagnosis: Longitudinal diagnosis at each visit
- Demographics: Age, sex
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[1]

# PPMI data locations (Google Drive)
GDRIVE = (
    Path.home()
    / "Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025"
)
PPMI_07FEB = GDRIVE / "PPMI_Data_07FEB"
PPMI_CSV = GDRIVE / "data/00_raw/GIMAN/ppmi_data_csv"

# PPMI visit-to-month mapping
EVENT_TO_MONTH: dict[str, float] = {
    "SC": 0,
    "BL": 0,
    "V01": 3,
    "V02": 6,
    "V03": 9,
    "V04": 12,
    "V05": 18,
    "V06": 24,
    "V07": 30,
    "V08": 36,
    "V09": 42,
    "V10": 48,
    "V11": 54,
    "V12": 60,
    "V13": 66,
    "V14": 72,
    "V15": 78,
    "V16": 84,
    "V17": 90,
    "V18": 96,
    "V19": 102,
    "V20": 108,
    # Return visits (mapped to approximate equivalent)
    "R01": 3,
    "R04": 12,
    "R06": 24,
    "R08": 36,
    "R13": 66,
    "R17": 90,
}

# PPMI diagnosis codes
PRIMDIAG_PD = 1  # Parkinson's Disease


def load_prodromal_patnos() -> set[int]:
    """Load prodromal patient IDs from Participant Status."""
    ps = pd.read_csv(PPMI_07FEB / "Participant_Status_07Feb2026.csv")
    patnos = set(ps.loc[ps["COHORT_DEFINITION"] == "Prodromal", "PATNO"].values)
    print(f"Prodromal patients in PPMI: {len(patnos)}")
    return patnos


def extract_phenoconversion(prodromal_patnos: set[int]) -> pd.DataFrame:
    """Extract real phenoconversion endpoints from Primary Clinical Diagnosis.

    Returns DataFrame with columns:
        PATNO, phenoconverted (0/1), time_to_event (months), endpoint_source
    """
    diag = pd.read_csv(PPMI_07FEB / "Primary_Clinical_Diagnosis_07Feb2026.csv")
    diag = diag[diag["PATNO"].isin(prodromal_patnos)].copy()
    diag["visit_month"] = diag["EVENT_ID"].map(EVENT_TO_MONTH)
    diag = diag.dropna(subset=["visit_month"])
    diag = diag.sort_values(["PATNO", "visit_month"])

    print(f"Diagnosis records for prodromal patients: {len(diag)}")
    print(f"Unique patients with diagnosis data: {diag['PATNO'].nunique()}")

    # Identify patients with PD at baseline (exclude — already PD at enrollment)
    baseline = diag[diag["EVENT_ID"].isin(["SC", "BL"])]
    baseline_pd = set(baseline.loc[baseline["PRIMDIAG"] == PRIMDIAG_PD, "PATNO"].values)
    print(f"Excluded: {len(baseline_pd)} patients with PD at baseline/screening")

    # True phenoconverters: NOT PD at baseline, PD at follow-up
    followup = diag[~diag["EVENT_ID"].isin(["SC", "BL"])]
    followup_pd = followup[followup["PRIMDIAG"] == PRIMDIAG_PD]
    true_converters = set(followup_pd["PATNO"].unique()) - baseline_pd

    # For converters: first PD diagnosis visit
    converter_first_pd = (
        followup_pd[followup_pd["PATNO"].isin(true_converters)]
        .groupby("PATNO")["visit_month"]
        .min()
        .reset_index()
    )
    converter_first_pd.columns = ["PATNO", "time_to_event"]
    converter_first_pd["phenoconverted"] = 1
    converter_first_pd["endpoint_source"] = "ppmi_primary_clinical_diagnosis"

    # For non-converters: last diagnosis assessment visit (censoring time)
    eligible = prodromal_patnos - baseline_pd
    non_converters = eligible - true_converters
    non_converter_last = (
        diag[diag["PATNO"].isin(non_converters)]
        .groupby("PATNO")["visit_month"]
        .max()
        .reset_index()
    )
    non_converter_last.columns = ["PATNO", "time_to_event"]
    non_converter_last["phenoconverted"] = 0
    non_converter_last["endpoint_source"] = "ppmi_censored"

    # Combine
    endpoints = pd.concat([converter_first_pd, non_converter_last], ignore_index=True)

    # Remove patients with zero follow-up time (only baseline data)
    endpoints = endpoints[endpoints["time_to_event"] > 0]

    print("\nEndpoint extraction results:")
    print(f"  True converters: {len(converter_first_pd)}")
    print(f"  Censored (non-converters with follow-up): {len(non_converter_last)}")
    print(f"  After removing zero follow-up: {len(endpoints)}")
    print(f"  Event rate: {endpoints['phenoconverted'].mean():.1%}")
    print(
        f"  Median time-to-event (converters): {endpoints.loc[endpoints['phenoconverted'] == 1, 'time_to_event'].median():.0f} months"
    )
    print(
        f"  Median follow-up (censored): {endpoints.loc[endpoints['phenoconverted'] == 0, 'time_to_event'].median():.0f} months"
    )

    return endpoints


def add_demographics(endpoints: pd.DataFrame) -> pd.DataFrame:
    """Add age and sex from Demographics table."""
    demo_path = GDRIVE / "data/00_raw/Demographics_08Feb2026.csv"
    demo = pd.read_csv(demo_path)
    demo = demo[["PATNO", "SEX"]].drop_duplicates(subset="PATNO")

    ps = pd.read_csv(PPMI_07FEB / "Participant_Status_07Feb2026.csv")
    age = ps[["PATNO", "ENROLL_AGE"]].copy()
    age = age.rename(columns={"ENROLL_AGE": "age_at_enrollment"})

    endpoints = endpoints.merge(demo, on="PATNO", how="left")
    endpoints = endpoints.merge(age, on="PATNO", how="left")

    print("\nDemographics coverage:")
    print(f"  SEX: {endpoints['SEX'].notna().sum()}/{len(endpoints)}")
    print(f"  Age: {endpoints['age_at_enrollment'].notna().sum()}/{len(endpoints)}")

    return endpoints


def validate_endpoints(endpoints: pd.DataFrame) -> bool:
    """Run quality checks on extracted endpoints."""
    print(f"\n{'=' * 60}")
    print("ENDPOINT VALIDATION")
    print(f"{'=' * 60}")

    passed = True

    # Check 1: No early PD contamination
    contaminated = (endpoints["time_to_event"] == 0) & (
        endpoints["phenoconverted"] == 1
    )
    n_contaminated = contaminated.sum()
    status = "PASS" if n_contaminated == 0 else "FAIL"
    print(f"  {status}: Early PD contamination check ({n_contaminated} contaminated)")
    if n_contaminated > 0:
        passed = False

    # Check 2: All times positive
    neg_times = (endpoints["time_to_event"] <= 0).sum()
    status = "PASS" if neg_times == 0 else "FAIL"
    print(f"  {status}: Positive times check ({neg_times} non-positive)")
    if neg_times > 0:
        passed = False

    # Check 3: Event rate in realistic range
    event_rate = endpoints["phenoconverted"].mean()
    status = "PASS" if 0.01 <= event_rate <= 0.20 else "WARN"
    print(f"  {status}: Event rate {event_rate:.1%} (expected 1-20% for prodromal)")

    # Check 4: No duplicate patients
    dups = endpoints["PATNO"].duplicated().sum()
    status = "PASS" if dups == 0 else "FAIL"
    print(f"  {status}: Unique patients ({dups} duplicates)")
    if dups > 0:
        passed = False

    # Check 5: 100% real endpoints
    sources = endpoints["endpoint_source"].value_counts()
    all_real = all(
        s in ("ppmi_primary_clinical_diagnosis", "ppmi_censored") for s in sources.index
    )
    status = "PASS" if all_real else "FAIL"
    print(f"  {status}: Endpoint provenance (all PPMI real)")
    for src, n in sources.items():
        print(f"    {src}: {n}")
    if not all_real:
        passed = False

    # Check 6: Sufficient events for Cox modeling
    n_events = endpoints["phenoconverted"].sum()
    status = "PASS" if n_events >= 20 else "WARN"
    print(f"  {status}: Sufficient events ({n_events}, need >=20 for Cox)")

    return passed


def main() -> int:
    print("=" * 60)
    print("EXTRACT REAL PPMI PHENOCONVERSION ENDPOINTS")
    print("=" * 60 + "\n")

    # Step 1: Get prodromal patient list
    prodromal_patnos = load_prodromal_patnos()

    # Step 2: Extract real endpoints
    endpoints = extract_phenoconversion(prodromal_patnos)

    # Step 3: Add demographics
    endpoints = add_demographics(endpoints)

    # Step 4: Validate
    valid = validate_endpoints(endpoints)

    # Step 5: Save
    output_dir = project_root / "data" / "prodromal_cohort"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "prodromal_survival_data.csv"
    endpoints.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  {len(endpoints)} patients, {endpoints['phenoconverted'].sum()} events")

    # Summary statistics
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total patients with follow-up: {len(endpoints)}")
    print(f"Phenoconverters: {endpoints['phenoconverted'].sum()}")
    print(f"Event rate: {endpoints['phenoconverted'].mean():.1%}")
    print(
        f"Follow-up range: {endpoints['time_to_event'].min():.0f} - {endpoints['time_to_event'].max():.0f} months"
    )
    print("Endpoint source: 100% PPMI Primary Clinical Diagnosis (no simulation)")

    if not valid:
        print("\nWARNING: Some validation checks failed!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
