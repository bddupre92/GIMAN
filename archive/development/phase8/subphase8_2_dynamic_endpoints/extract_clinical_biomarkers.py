"""Phase 8.2 Week 1: Clinical Biomarkers Extraction

Purpose:
    Extract non-motor clinical biomarkers from PPMI questionnaires for the
    prodromal prognostic model. These capture early non-motor symptoms that
    often precede motor symptoms in PD.

Features Extracted:
    1. UPSIT (University of Pennsylvania Smell Identification Test) - Olfactory dysfunction
    2. RBD (REM sleep behavior disorder severity) - Sleep disorder common in prodromal PD
    3. SCOPA-AUT (Scales for Outcomes in PD - Autonomic) - Autonomic dysfunction
    4. ESS (Epworth Sleepiness Scale) - Daytime sleepiness

Data Sources:
    - University_of_Pennsylvania_Smell_ID_Test_18Sep2025.csv (UPSIT)
    - REM_Sleep_Disorder_Questionnaire_18Sep2025.csv (RBD)
    - SCOPA-AUT_18Sep2025.csv (SCOPA-AUT)
    - Epworth_Sleepiness_Scale_18Sep2025.csv (ESS)

Output:
    - data/03_prodromal/enhanced/clinical_biomarkers.csv
      Columns: PATNO, UPSIT_TOTAL, RBD_TOTAL, SCOPA_AUT_TOTAL, ESS_TOTAL

Expected Coverage:
    - 75% average (per PHASE8_2_ALIGNED_STRATEGY.md)

Author: GIMAN Research Team
Date: October 12, 2025
Phase: 8.2 Week 1
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from raw_file_resolver import RawFileResolver, default_raw_roots


def _sum_numeric_items(frame: pd.DataFrame, item_cols: list[str], out_col: str) -> None:
    vals = frame[item_cols].apply(pd.to_numeric, errors="coerce")
    summed = vals.sum(axis=1, skipna=True)
    summed[vals.notna().sum(axis=1) == 0] = np.nan
    frame[out_col] = summed


def _select_one_visit_per_patient(
    df: pd.DataFrame,
    *,
    preferred_events: tuple[str, ...] = ("BL", "SC", "BLTOT"),
) -> pd.DataFrame:
    """Select one baseline-like visit per patient, with deterministic fallback."""
    if df.empty or "PATNO" not in df.columns:
        return df

    out = df.copy()

    # Prefer baseline/screening style events when present.
    if "EVENT_ID" in out.columns:
        event_series = out["EVENT_ID"].astype(str)
        out["_event_rank"] = np.where(event_series.isin(preferred_events), 0, 1)
    else:
        out["_event_rank"] = 1

    date_col = None
    for candidate in ("INFODT", "CREATED_AT", "MODIFIED_AT", "RUNDATE"):
        if candidate in out.columns:
            date_col = candidate
            break

    if date_col is not None:
        out["_visit_date"] = pd.to_datetime(out[date_col], errors="coerce")
    else:
        out["_visit_date"] = pd.NaT

    out = out.sort_values(["PATNO", "_event_rank", "_visit_date"])
    out = out.drop_duplicates(subset=["PATNO"], keep="first")
    return out.drop(columns=["_event_rank", "_visit_date"], errors="ignore")


def load_upsit(
    data_dir: Path, resolver: RawFileResolver
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load University of Pennsylvania Smell Identification Test (UPSIT) data.

    UPSIT is a 40-item smell identification test. Olfactory dysfunction
    is one of the earliest prodromal symptoms of PD.

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with PATNO and UPSIT_TOTAL
    """
    resolved = resolver.resolve_latest(
        "upsit",
        [
            "University_of_Pennsylvania_Smell_Identification_Test_UPSIT_*.csv",
            "University_of_Pennsylvania_Smell_ID_Test_*.csv",
        ],
        required=True,
        allow_empty=False,
        required_columns=["PATNO"],
    )
    if resolved is None:
        raise RuntimeError("Failed to resolve UPSIT source.")
    upsit_file = Path(resolved.path)

    print(f"Loading UPSIT from: {upsit_file}")
    df = pd.read_csv(upsit_file)
    print(f"✓ Loaded {len(df)} records")

    # Look for total score column
    total_cols = [
        col for col in df.columns if "TOTAL" in col.upper() or "UPSIT" in col.upper()
    ]

    if total_cols:
        score_col = total_cols[0]
        print(f"  Using column: {score_col}")
    else:
        print("  ⚠ No total score column found, computing from items")
        # UPSIT items typically: UPSITBK1 through UPSITBK4 (4 books, 10 items each)
        item_cols = [col for col in df.columns if col.startswith("UPSITBK")]
        if item_cols:
            _sum_numeric_items(df, item_cols, "UPSIT_TOTAL")
            score_col = "UPSIT_TOTAL"
        else:
            print("  ⚠ No UPSIT items found, returning empty")
            return pd.DataFrame({"PATNO": [], "UPSIT_TOTAL": []}), resolved.as_dict()
    if score_col in df.columns:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    # Keep one baseline-like observation per patient.
    df = _select_one_visit_per_patient(df)

    upsit_df = df[["PATNO", score_col]].copy()
    upsit_df.columns = ["PATNO", "UPSIT_TOTAL"]

    # Remove duplicates
    upsit_df = upsit_df.drop_duplicates(subset=["PATNO"], keep="first")

    print(
        f"✓ UPSIT: {upsit_df['UPSIT_TOTAL'].notna().sum()}/{len(upsit_df)} non-null scores"
    )
    return upsit_df, resolved.as_dict()


def load_rbd(
    data_dir: Path, resolver: RawFileResolver
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load REM Sleep Behavior Disorder Questionnaire data.

    RBD is a strong prodromal marker for PD and related synucleinopathies.

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with PATNO and RBD_TOTAL
    """
    resolved = resolver.resolve_latest(
        "rbd_questionnaire",
        [
            "REM_Sleep_Behavior_Disorder_Questionnaire_*.csv",
            "REM_Sleep_Disorder_Questionnaire_*.csv",
        ],
        required=True,
        allow_empty=False,
        required_columns=["PATNO"],
    )
    if resolved is None:
        raise RuntimeError("Failed to resolve RBD source.")
    rbd_file = Path(resolved.path)

    print(f"\nLoading RBD from: {rbd_file}")
    df = pd.read_csv(rbd_file)
    print(f"✓ Loaded {len(df)} records")

    # Look for total score or severity column
    score_cols = [
        col
        for col in df.columns
        if any(x in col.upper() for x in ["TOTAL", "SCORE", "SEVERITY"])
    ]

    if score_cols:
        score_col = score_cols[0]
        print(f"  Using column: {score_col}")
    else:
        print("  ⚠ No score column found, computing from items")
        # RBD questionnaire items typically: DRMVIVID, DRMVIOL, etc.
        item_cols = [col for col in df.columns if col.startswith("DRM")]
        if item_cols:
            _sum_numeric_items(df, item_cols, "RBD_TOTAL")
            score_col = "RBD_TOTAL"
        else:
            print("  ⚠ No RBD items found, returning empty")
            return pd.DataFrame({"PATNO": [], "RBD_TOTAL": []}), resolved.as_dict()
    if score_col in df.columns:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    df = _select_one_visit_per_patient(df)

    rbd_df = df[["PATNO", score_col]].copy()
    rbd_df.columns = ["PATNO", "RBD_TOTAL"]
    rbd_df = rbd_df.drop_duplicates(subset=["PATNO"], keep="first")

    print(f"✓ RBD: {rbd_df['RBD_TOTAL'].notna().sum()}/{len(rbd_df)} non-null scores")
    return rbd_df, resolved.as_dict()


def load_scopa_aut(
    data_dir: Path, resolver: RawFileResolver
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load SCOPA-AUT (Scales for Outcomes in PD - Autonomic) data.

    Measures autonomic dysfunction (GI, urinary, cardiovascular, etc.).

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with PATNO and SCOPA_AUT_TOTAL
    """
    resolved = resolver.resolve_latest(
        "scopa_aut",
        ["SCOPA-AUT_*.csv"],
        required=True,
        allow_empty=False,
        required_columns=["PATNO"],
    )
    if resolved is None:
        raise RuntimeError("Failed to resolve SCOPA-AUT source.")
    scopa_file = Path(resolved.path)

    print(f"\nLoading SCOPA-AUT from: {scopa_file}")
    df = pd.read_csv(scopa_file)
    print(f"✓ Loaded {len(df)} records")

    # Look for total score
    total_cols = [col for col in df.columns if "TOTAL" in col.upper()]

    if total_cols:
        score_col = total_cols[0]
        print(f"  Using column: {score_col}")
    else:
        print("  ⚠ No total score column found, computing from items")
        # SCOPA-AUT items typically: SCAU1 through SCAU25
        item_cols = [col for col in df.columns if col.startswith("SCAU")]
        if item_cols:
            _sum_numeric_items(df, item_cols, "SCOPA_AUT_TOTAL")
            score_col = "SCOPA_AUT_TOTAL"
        else:
            print("  ⚠ No SCOPA-AUT items found, returning empty")
            return pd.DataFrame(
                {"PATNO": [], "SCOPA_AUT_TOTAL": []}
            ), resolved.as_dict()
    if score_col in df.columns:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    df = _select_one_visit_per_patient(df)

    scopa_df = df[["PATNO", score_col]].copy()
    scopa_df.columns = ["PATNO", "SCOPA_AUT_TOTAL"]
    scopa_df = scopa_df.drop_duplicates(subset=["PATNO"], keep="first")

    print(
        f"✓ SCOPA-AUT: {scopa_df['SCOPA_AUT_TOTAL'].notna().sum()}/{len(scopa_df)} non-null scores"
    )
    return scopa_df, resolved.as_dict()


def load_ess(
    data_dir: Path, resolver: RawFileResolver
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load Epworth Sleepiness Scale (ESS) data.

    Measures daytime sleepiness, common in PD.

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with PATNO and ESS_TOTAL
    """
    resolved = resolver.resolve_latest(
        "epworth_sleepiness",
        ["Epworth_Sleepiness_Scale_*.csv", "Epworth_Sleepiness_Scale__Online__*.csv"],
        required=True,
        allow_empty=False,
        required_columns=["PATNO"],
    )
    if resolved is None:
        raise RuntimeError("Failed to resolve ESS source.")
    ess_file = Path(resolved.path)

    print(f"\nLoading ESS from: {ess_file}")
    df = pd.read_csv(ess_file)
    print(f"✓ Loaded {len(df)} records")

    # Look for an explicit total score column only.
    total_cols = [
        col
        for col in df.columns
        if "TOTAL" in col.upper() or col.upper() in {"ESS_TOTAL", "EPWORTH_TOTAL"}
    ]

    if total_cols:
        score_col = total_cols[0]
        print(f"  Using column: {score_col}")
    else:
        print("  ⚠ No total score column found, computing from items")
        # ESS items typically: ESS1 through ESS8
        item_cols = [
            col
            for col in df.columns
            if col.startswith("ESS")
            and (col[3:].isdigit() or (col.endswith("_OL") and col[3:-3].isdigit()))
        ]
        if item_cols:
            _sum_numeric_items(df, item_cols, "ESS_TOTAL")
            score_col = "ESS_TOTAL"
        else:
            print("  ⚠ No ESS items found, returning empty")
            return pd.DataFrame({"PATNO": [], "ESS_TOTAL": []}), resolved.as_dict()
    if score_col in df.columns:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    # ESS online pulls can use OLxx event codes; prefer baseline-like events,
    # but fall back deterministically to earliest available observation.
    df = _select_one_visit_per_patient(df)

    ess_df = df[["PATNO", score_col]].copy()
    ess_df.columns = ["PATNO", "ESS_TOTAL"]
    ess_df = ess_df.drop_duplicates(subset=["PATNO"], keep="first")

    print(f"✓ ESS: {ess_df['ESS_TOTAL'].notna().sum()}/{len(ess_df)} non-null scores")
    return ess_df, resolved.as_dict()


def merge_clinical_biomarkers(
    upsit_df: pd.DataFrame,
    rbd_df: pd.DataFrame,
    scopa_df: pd.DataFrame,
    ess_df: pd.DataFrame,
    data_dir: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Merge all clinical biomarkers with prodromal cohort.

    Args:
        upsit_df: UPSIT DataFrame
        rbd_df: RBD DataFrame
        scopa_df: SCOPA-AUT DataFrame
        ess_df: ESS DataFrame
        data_dir: Base data directory

    Returns:
        Tuple of (merged_df, coverage_stats)
    """
    cohort_override = os.getenv("GIMAN_COHORT_CSV", "").strip()
    prodromal_file = (
        Path(cohort_override)
        if cohort_override
        else data_dir / "prodromal_cohort" / "prodromal_survival_data.csv"
    )
    print(f"\nLoading prodromal cohort: {prodromal_file}")
    prodromal_df = pd.read_csv(prodromal_file)
    print(f"✓ Loaded {len(prodromal_df)} prodromal patients")

    # Start with PATNO only
    merged_df = prodromal_df[["PATNO"]].copy()

    # Merge each biomarker
    for df_feat, feat_name in [
        (upsit_df, "UPSIT_TOTAL"),
        (rbd_df, "RBD_TOTAL"),
        (scopa_df, "SCOPA_AUT_TOTAL"),
        (ess_df, "ESS_TOTAL"),
    ]:
        if len(df_feat) > 0:
            merged_df = merged_df.merge(df_feat, on="PATNO", how="left")
        else:
            merged_df[feat_name] = np.nan

    # Compute coverage
    feature_cols = ["UPSIT_TOTAL", "RBD_TOTAL", "SCOPA_AUT_TOTAL", "ESS_TOTAL"]
    coverage_stats = {}

    print("\nClinical biomarker coverage in prodromal cohort:")
    for col in feature_cols:
        n_available = merged_df[col].notna().sum()
        coverage_pct = 100 * n_available / len(merged_df)
        coverage_stats[col] = coverage_pct
        print(f"  {col}: {n_available}/{len(merged_df)} ({coverage_pct:.1f}%)")

    avg_coverage = np.mean(list(coverage_stats.values()))
    print(f"\n✓ Average clinical biomarker coverage: {avg_coverage:.1f}%")

    if avg_coverage < 75:
        print(f"⚠ WARNING: Coverage {avg_coverage:.1f}% below target 75%")
    else:
        print("✓ Coverage exceeds target (75%)")

    return merged_df, coverage_stats


def save_clinical_biomarkers(
    clinical_df: pd.DataFrame,
    coverage_stats: dict[str, float],
    source_files: list[dict[str, object]],
    output_dir: Path,
) -> None:
    """Save clinical biomarker features and metadata.

    Args:
        clinical_df: DataFrame with PATNO and clinical biomarker features
        coverage_stats: Dict of feature_name -> coverage percentage
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    output_file = output_dir / "clinical_biomarkers.csv"
    clinical_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved clinical biomarkers: {output_file}")
    print(f"  Shape: {clinical_df.shape}")

    # Save metadata
    metadata = {
        "extraction_date_utc": datetime.now(timezone.utc).isoformat(),
        "n_patients": len(clinical_df),
        "n_features": len(clinical_df.columns) - 1,
        "features": list(clinical_df.columns.drop("PATNO")),
        "coverage": coverage_stats,
        "average_coverage": np.mean(list(coverage_stats.values())),
        "target_coverage": 75.0,
        "source_files": source_files,
        "clinical_relevance": {
            "UPSIT": "Olfactory dysfunction, earliest prodromal symptom",
            "RBD": "Sleep disorder, strong predictor of synucleinopathy",
            "SCOPA_AUT": "Autonomic dysfunction, non-motor burden",
            "ESS": "Daytime sleepiness, quality of life impact",
        },
    }

    import json

    metadata_file = output_dir / "clinical_biomarkers_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata: {metadata_file}")


def main() -> None:
    """Main execution function for clinical biomarker extraction."""
    print("=" * 70)
    print("PHASE 8.2 WEEK 1: CLINICAL BIOMARKERS EXTRACTION")
    print("=" * 70)

    # Setup paths
    base_dir = Path(__file__).resolve().parents[4]
    data_dir = base_dir / "data"
    output_dir = data_dir / "03_prodromal" / "enhanced"
    resolver = RawFileResolver(default_raw_roots(base_dir))

    print(f"\nBase directory: {base_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Load biomarkers
    print("\n" + "-" * 70)
    print("STEP 1: Load Clinical Biomarkers")
    print("-" * 70)

    upsit_df, upsit_src = load_upsit(data_dir, resolver)
    rbd_df, rbd_src = load_rbd(data_dir, resolver)
    scopa_df, scopa_src = load_scopa_aut(data_dir, resolver)
    ess_df, ess_src = load_ess(data_dir, resolver)

    # Merge with prodromal cohort
    print("\n" + "-" * 70)
    print("STEP 2: Merge with Prodromal Cohort")
    print("-" * 70)
    merged_df, coverage_stats = merge_clinical_biomarkers(
        upsit_df, rbd_df, scopa_df, ess_df, data_dir
    )

    # Save results
    print("\n" + "-" * 70)
    print("STEP 3: Save Results")
    print("-" * 70)
    save_clinical_biomarkers(
        merged_df,
        coverage_stats,
        [upsit_src, rbd_src, scopa_src, ess_src],
        output_dir,
    )

    # Summary
    print("\n" + "=" * 70)
    print("CLINICAL BIOMARKER EXTRACTION COMPLETE")
    print("=" * 70)
    print("✓ Extracted 4 clinical biomarker features")
    print(f"✓ Cohort size: {len(merged_df)} patients")
    print(f"✓ Average coverage: {np.mean(list(coverage_stats.values())):.1f}%")
    print(f"✓ Output: {output_dir / 'clinical_biomarkers.csv'}")
    print("\nNext step: scripts/phase8_2/extract_cortical_thickness.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
