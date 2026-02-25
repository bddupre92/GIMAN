"""NSD-ISS Biological Staging System for Parkinson's Disease.

Implements the Neuronal alpha-Synuclein Disease Integrated Staging System
(NSD-ISS) as defined by Simuni et al., Lancet Neurology (2024).

The NSD-ISS defines Parkinson's Disease biologically rather than clinically:
- S anchor: Pathological neuronal alpha-synuclein detected in vivo (SAA+)
- D anchor: Dopaminergic dysfunction documented (DaT-SPECT deficit)

Staging:
- Stage 0: Genetic risk only (e.g., SNCA mutation), no S+, no D+
- Stage 1: S+ (alpha-synuclein positive) OR D+ (dopaminergic deficit),
           no clinical signs
- Stage 2A: S+ AND/OR D+ with subtle signs, no functional impairment
- Stage 2B: S+ AND/OR D+ with clinical signs sufficient for clinical diagnosis,
            no functional impairment
- Stage 3: S+ AND/OR D+ with mild functional impairment
- Stage 4: S+ AND/OR D+ with moderate functional impairment
- Stage 5: S+ AND/OR D+ with severe functional impairment
- Stage 6: S+ AND/OR D+ with complete dependency

This module computes NSD-ISS stages from PPMI data using:
- SAA results (S anchor) from biospecimen analysis
- DaT-SPECT SBR values (D anchor) from imaging
- MDS-UPDRS scores + Hoehn & Yahr (functional impairment)

References:
- Simuni et al. (2024) Lancet Neurology, DOI: 10.1016/S1474-4422(23)00405-2
- NSD-ISS validation in PPMI (2024) npj Parkinson's Disease
- NSD-ISS in BioFIND (2025) npj Parkinson's Disease

Author: GIMAN Research Team
Date: February 2026
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants based on NSD-ISS definition (Lancet Neurology 2024)
# ---------------------------------------------------------------------------

# DaT-SPECT SBR threshold for dopaminergic deficit (D+)
# Based on PPMI protocol: deficit = lowest putamen SBR < 65% of age-expected mean
# Simplified threshold used in PPMI studies
PUTAMEN_SBR_DEFICIT_THRESHOLD = 0.80  # Below this = dopaminergic deficit (D+)

# Hoehn & Yahr stage thresholds for functional impairment levels
HY_STAGE_THRESHOLDS = {
    "no_impairment": (0, 2.0),    # H&Y 0-2: Stages 1, 2A, 2B
    "mild_impairment": (2.0, 3.0),  # H&Y 2-3: Stage 3
    "moderate_impairment": (3.0, 4.0),  # H&Y 3-4: Stage 4
    "severe_impairment": (4.0, 5.0),    # H&Y 4-5: Stage 5
    "complete_dependency": (5.0, float("inf")),  # H&Y 5: Stage 6
}

# MDS-UPDRS Part III threshold for clinical parkinsonism
UPDRS3_CLINICAL_THRESHOLD = 10  # ≥10 suggests clinical parkinsonism

# Genetic risk markers for Stage 0
GENETIC_RISK_GENES = ["LRRK2", "GBA", "SNCA"]


@dataclass
class NSDISSResult:
    """Result of NSD-ISS staging for a single patient."""

    patno: int
    stage: str  # "0", "1", "2A", "2B", "3", "4", "5", "6", or "unclassified"
    stage_numeric: float  # 0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0
    s_positive: Optional[bool]  # SAA result
    d_positive: Optional[bool]  # DaT-SPECT deficit
    has_clinical_signs: bool
    has_functional_impairment: bool
    functional_impairment_level: str  # "none", "mild", "moderate", "severe", "complete"
    confidence: str  # "high", "medium", "low" based on data completeness
    missing_anchors: list[str] = field(default_factory=list)
    raw_values: dict[str, Any] = field(default_factory=dict)


# Map stage labels to numeric values for ML
STAGE_NUMERIC_MAP = {
    "0": 0.0,
    "1": 1.0,
    "2A": 2.0,
    "2B": 2.5,
    "3": 3.0,
    "4": 4.0,
    "5": 5.0,
    "6": 6.0,
    "unclassified": float("nan"),
}

# Ordinal encoding for classification tasks
STAGE_ORDINAL_MAP = {
    "0": 0,
    "1": 1,
    "2A": 2,
    "2B": 3,
    "3": 4,
    "4": 5,
    "5": 6,
    "6": 7,
    "unclassified": -1,
}


def compute_s_anchor(
    saa_label: Optional[float],
    saa_positive_rate: Optional[float] = None,
) -> Optional[bool]:
    """Determine S anchor (alpha-synuclein pathology) from SAA results.

    Args:
        saa_label: Binary SAA result (1=positive, 0=negative)
        saa_positive_rate: Fraction of positive SAA observations for this patient

    Returns:
        True if S+, False if S-, None if data missing
    """
    if saa_label is None or pd.isna(saa_label):
        return None
    return bool(int(saa_label) == 1)


def compute_d_anchor(
    putamen_mean_sbr: Optional[float] = None,
    caudate_mean_sbr: Optional[float] = None,
    putamen_l_sbr: Optional[float] = None,
    putamen_r_sbr: Optional[float] = None,
    threshold: float = PUTAMEN_SBR_DEFICIT_THRESHOLD,
) -> Optional[bool]:
    """Determine D anchor (dopaminergic dysfunction) from DaT-SPECT.

    Uses the lowest putamen SBR (left or right) as the primary indicator.
    Falls back to mean putamen or mean caudate if lateralized values unavailable.

    Args:
        putamen_mean_sbr: Mean putamen SBR
        caudate_mean_sbr: Mean caudate SBR
        putamen_l_sbr: Left putamen SBR
        putamen_r_sbr: Right putamen SBR
        threshold: SBR threshold below which = deficit

    Returns:
        True if D+, False if D-, None if data missing
    """
    # Prefer lateralized putamen values (most informative)
    sbr_values = []
    if putamen_l_sbr is not None and not pd.isna(putamen_l_sbr):
        sbr_values.append(putamen_l_sbr)
    if putamen_r_sbr is not None and not pd.isna(putamen_r_sbr):
        sbr_values.append(putamen_r_sbr)

    if sbr_values:
        # D+ if LOWEST putamen SBR is below threshold
        return min(sbr_values) < threshold

    # Fallback to mean putamen
    if putamen_mean_sbr is not None and not pd.isna(putamen_mean_sbr):
        return putamen_mean_sbr < threshold

    # Last resort: caudate (less sensitive but still indicative)
    if caudate_mean_sbr is not None and not pd.isna(caudate_mean_sbr):
        return caudate_mean_sbr < threshold

    return None


def compute_functional_impairment(
    hy_stage: Optional[float] = None,
    updrs3_total: Optional[float] = None,
    updrs2_total: Optional[float] = None,
) -> tuple[str, bool]:
    """Determine functional impairment level from clinical assessments.

    Uses Hoehn & Yahr stage as primary indicator, with UPDRS scores
    as supporting evidence.

    Args:
        hy_stage: Hoehn & Yahr stage (0-5)
        updrs3_total: MDS-UPDRS Part III total score
        updrs2_total: MDS-UPDRS Part II total score

    Returns:
        Tuple of (impairment_level, has_functional_impairment)
        impairment_level: "none", "mild", "moderate", "severe", "complete"
    """
    if hy_stage is not None and not pd.isna(hy_stage):
        hy = float(hy_stage)
        if hy < 2.0:
            return "none", False
        elif hy < 3.0:
            return "mild", True
        elif hy < 4.0:
            return "moderate", True
        elif hy < 5.0:
            return "severe", True
        else:
            return "complete", True

    # Fallback: Use UPDRS-III as proxy
    if updrs3_total is not None and not pd.isna(updrs3_total):
        u3 = float(updrs3_total)
        if u3 < 20:
            return "none", False
        elif u3 < 40:
            return "mild", True
        elif u3 < 60:
            return "moderate", True
        else:
            return "severe", True

    return "none", False  # Assume no impairment if no data


def has_clinical_parkinsonism(
    updrs3_total: Optional[float] = None,
    hy_stage: Optional[float] = None,
    primary_diagnosis: Optional[int] = None,
) -> bool:
    """Determine if patient shows clinical signs of parkinsonism.

    Args:
        updrs3_total: MDS-UPDRS Part III total
        hy_stage: Hoehn & Yahr stage
        primary_diagnosis: PPMI primary diagnosis code (1=PD)

    Returns:
        True if clinical parkinsonism present
    """
    # Clinical diagnosis is strongest evidence
    if primary_diagnosis is not None and not pd.isna(primary_diagnosis):
        if int(primary_diagnosis) == 1:  # Idiopathic PD
            return True

    # UPDRS-III threshold
    if updrs3_total is not None and not pd.isna(updrs3_total):
        if float(updrs3_total) >= UPDRS3_CLINICAL_THRESHOLD:
            return True

    # H&Y > 0 implies clinical parkinsonism
    if hy_stage is not None and not pd.isna(hy_stage):
        if float(hy_stage) > 0:
            return True

    return False


def compute_nsd_iss_stage(
    s_positive: Optional[bool],
    d_positive: Optional[bool],
    has_clinical: bool,
    impairment_level: str,
    has_impairment: bool,
    has_genetic_risk: bool = False,
) -> tuple[str, float]:
    """Compute NSD-ISS stage from biological and clinical anchors.

    Implements the staging algorithm from Simuni et al. (2024).

    Args:
        s_positive: SAA positive (S anchor)
        d_positive: DaT deficit (D anchor)
        has_clinical: Clinical parkinsonism present
        impairment_level: Functional impairment level
        has_impairment: Whether functional impairment exists
        has_genetic_risk: Whether patient has PD genetic risk

    Returns:
        Tuple of (stage_label, stage_numeric)
    """
    # Stage 0: Genetic risk only, no biological markers
    if has_genetic_risk and not s_positive and not d_positive:
        return "0", 0.0

    # Must have at least one biological anchor for stages 1+
    has_biological = (s_positive is True) or (d_positive is True)
    if not has_biological:
        if s_positive is None and d_positive is None:
            return "unclassified", float("nan")
        return "0", 0.0

    # Stages 1-6 require S+ and/or D+
    if has_impairment:
        # Stages 3-6: Functional impairment present
        if impairment_level == "complete":
            return "6", 6.0
        elif impairment_level == "severe":
            return "5", 5.0
        elif impairment_level == "moderate":
            return "4", 4.0
        else:  # "mild"
            return "3", 3.0
    elif has_clinical:
        # Stage 2B: Clinical signs without functional impairment
        return "2B", 2.5
    else:
        # Stage 1 vs 2A depends on subtle signs
        # In practice, Stage 2A is hard to distinguish from 1 without detailed assessment
        # For PPMI prodromal cohort: if enrolled as prodromal/at-risk = Stage 1
        # If enrolled with subtle motor signs = Stage 2A
        # Default: Stage 1 if no clinical signs, as this is the prodromal stage
        return "1", 1.0


def stage_single_patient(
    patno: int,
    saa_label: Optional[float] = None,
    saa_positive_rate: Optional[float] = None,
    putamen_mean_sbr: Optional[float] = None,
    caudate_mean_sbr: Optional[float] = None,
    putamen_l_sbr: Optional[float] = None,
    putamen_r_sbr: Optional[float] = None,
    hy_stage: Optional[float] = None,
    updrs3_total: Optional[float] = None,
    updrs2_total: Optional[float] = None,
    primary_diagnosis: Optional[int] = None,
    has_lrrk2: bool = False,
    has_gba: bool = False,
    has_snca: bool = False,
) -> NSDISSResult:
    """Compute NSD-ISS stage for a single patient.

    Args:
        patno: Patient number
        saa_label: Binary SAA result
        saa_positive_rate: Rate of SAA positivity across observations
        putamen_mean_sbr: Mean putamen SBR
        caudate_mean_sbr: Mean caudate SBR
        putamen_l_sbr: Left putamen SBR
        putamen_r_sbr: Right putamen SBR
        hy_stage: Hoehn & Yahr stage
        updrs3_total: MDS-UPDRS Part III total
        updrs2_total: MDS-UPDRS Part II total
        primary_diagnosis: PPMI primary diagnosis code
        has_lrrk2: LRRK2 mutation carrier
        has_gba: GBA mutation carrier
        has_snca: SNCA variant carrier

    Returns:
        NSDISSResult with stage and supporting data
    """
    # Compute anchors
    s_positive = compute_s_anchor(saa_label, saa_positive_rate)
    d_positive = compute_d_anchor(
        putamen_mean_sbr, caudate_mean_sbr, putamen_l_sbr, putamen_r_sbr
    )

    # Compute functional impairment
    impairment_level, has_impairment = compute_functional_impairment(
        hy_stage, updrs3_total, updrs2_total
    )

    # Check clinical parkinsonism
    has_clinical = has_clinical_parkinsonism(updrs3_total, hy_stage, primary_diagnosis)

    # Genetic risk
    has_genetic_risk = has_lrrk2 or has_gba or has_snca

    # Compute stage
    stage, stage_numeric = compute_nsd_iss_stage(
        s_positive, d_positive, has_clinical, impairment_level,
        has_impairment, has_genetic_risk,
    )

    # Determine confidence based on data completeness
    missing = []
    if s_positive is None:
        missing.append("SAA")
    if d_positive is None:
        missing.append("DaT-SPECT")
    if hy_stage is None and updrs3_total is None:
        missing.append("clinical_severity")

    if len(missing) == 0:
        confidence = "high"
    elif len(missing) == 1:
        confidence = "medium"
    else:
        confidence = "low"

    return NSDISSResult(
        patno=patno,
        stage=stage,
        stage_numeric=stage_numeric,
        s_positive=s_positive,
        d_positive=d_positive,
        has_clinical_signs=has_clinical,
        has_functional_impairment=has_impairment,
        functional_impairment_level=impairment_level,
        confidence=confidence,
        missing_anchors=missing,
        raw_values={
            "saa_label": saa_label,
            "saa_positive_rate": saa_positive_rate,
            "putamen_mean_sbr": putamen_mean_sbr,
            "caudate_mean_sbr": caudate_mean_sbr,
            "putamen_l_sbr": putamen_l_sbr,
            "putamen_r_sbr": putamen_r_sbr,
            "hy_stage": hy_stage,
            "updrs3_total": updrs3_total,
            "primary_diagnosis": primary_diagnosis,
        },
    )


def stage_cohort(
    cohort_df: pd.DataFrame,
    saa_df: Optional[pd.DataFrame] = None,
    dat_df: Optional[pd.DataFrame] = None,
    clinical_df: Optional[pd.DataFrame] = None,
    genetic_df: Optional[pd.DataFrame] = None,
    diagnosis_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute NSD-ISS stages for an entire cohort.

    Merges data from multiple PPMI tables and computes staging for each patient.

    Args:
        cohort_df: Base cohort DataFrame with PATNO column
        saa_df: SAA results (PATNO, saa_label, saa_positive_rate)
        dat_df: DaT-SPECT data (PATNO, PUTAMEN_MEAN, CAUDATE_MEAN, etc.)
        clinical_df: Clinical data (PATNO, NHY, NP3TOT, etc.)
        genetic_df: Genetic data (PATNO, LRRK2, GBA, etc.)
        diagnosis_df: Diagnosis data (PATNO, PRIMDIAG)

    Returns:
        DataFrame with NSD-ISS stages and metadata for all patients
    """
    patnos = cohort_df["PATNO"].unique()
    logger.info(f"Computing NSD-ISS stages for {len(patnos)} patients")

    # Build lookup dictionaries for efficient access
    saa_lookup = {}
    if saa_df is not None:
        for _, row in saa_df.iterrows():
            saa_lookup[int(row["PATNO"])] = row

    dat_lookup = {}
    if dat_df is not None:
        for _, row in dat_df.iterrows():
            dat_lookup[int(row["PATNO"])] = row

    clinical_lookup = {}
    if clinical_df is not None:
        for _, row in clinical_df.iterrows():
            clinical_lookup[int(row["PATNO"])] = row

    genetic_lookup = {}
    if genetic_df is not None:
        for _, row in genetic_df.iterrows():
            genetic_lookup[int(row["PATNO"])] = row

    diagnosis_lookup = {}
    if diagnosis_df is not None:
        for _, row in diagnosis_df.iterrows():
            diagnosis_lookup[int(row["PATNO"])] = row

    # Stage each patient
    results = []
    for patno in patnos:
        patno = int(patno)

        # Get SAA data
        saa_row = saa_lookup.get(patno)
        saa_label = saa_row.get("saa_label") if saa_row is not None else None
        saa_rate = saa_row.get("saa_positive_rate") if saa_row is not None else None

        # Get DaT-SPECT data
        dat_row = dat_lookup.get(patno)
        putamen_mean = None
        caudate_mean = None
        putamen_l = None
        putamen_r = None
        if dat_row is not None:
            for col in ["PUTAMEN_MEAN", "PUTAMEN_MEAN_SBR", "putamen_mean_sbr"]:
                if col in dat_row.index:
                    putamen_mean = dat_row.get(col)
                    break
            for col in ["CAUDATE_MEAN", "CAUDATE_MEAN_SBR", "caudate_mean_sbr"]:
                if col in dat_row.index:
                    caudate_mean = dat_row.get(col)
                    break
            for col in ["PUTAMEN_L_SBR", "putamen_l_sbr"]:
                if col in dat_row.index:
                    putamen_l = dat_row.get(col)
                    break
            for col in ["PUTAMEN_R_SBR", "putamen_r_sbr"]:
                if col in dat_row.index:
                    putamen_r = dat_row.get(col)
                    break

        # Get clinical data
        clin_row = clinical_lookup.get(patno)
        hy = None
        u3 = None
        u2 = None
        if clin_row is not None:
            for col in ["NHY", "nhy", "HY_STAGE", "HOEHN_YAHR"]:
                if col in clin_row.index:
                    hy = clin_row.get(col)
                    break
            for col in ["NP3TOT", "np3tot", "UPDRS3_TOTAL", "MDS_UPDRS_III"]:
                if col in clin_row.index:
                    u3 = clin_row.get(col)
                    break
            for col in ["NP2TOT", "np2tot", "UPDRS2_TOTAL"]:
                if col in clin_row.index:
                    u2 = clin_row.get(col)
                    break

        # Get genetic data
        gen_row = genetic_lookup.get(patno)
        has_lrrk2 = False
        has_gba = False
        has_snca = False
        if gen_row is not None:
            for col in ["LRRK2", "lrrk2"]:
                if col in gen_row.index:
                    val = gen_row.get(col)
                    if val is not None and not pd.isna(val):
                        has_lrrk2 = bool(int(val) == 1)
                    break
            for col in ["GBA", "gba"]:
                if col in gen_row.index:
                    val = gen_row.get(col)
                    if val is not None and not pd.isna(val):
                        has_gba = bool(int(val) == 1)
                    break

        # Get diagnosis
        diag_row = diagnosis_lookup.get(patno)
        primary_diag = None
        if diag_row is not None:
            for col in ["PRIMDIAG", "primdiag", "PRIMARY_DIAGNOSIS"]:
                if col in diag_row.index:
                    primary_diag = diag_row.get(col)
                    break

        result = stage_single_patient(
            patno=patno,
            saa_label=saa_label,
            saa_positive_rate=saa_rate,
            putamen_mean_sbr=putamen_mean,
            caudate_mean_sbr=caudate_mean,
            putamen_l_sbr=putamen_l,
            putamen_r_sbr=putamen_r,
            hy_stage=hy,
            updrs3_total=u3,
            updrs2_total=u2,
            primary_diagnosis=primary_diag,
            has_lrrk2=has_lrrk2,
            has_gba=has_gba,
            has_snca=has_snca,
        )
        results.append(result)

    # Convert to DataFrame
    stage_df = pd.DataFrame([
        {
            "PATNO": r.patno,
            "nsd_iss_stage": r.stage,
            "nsd_iss_stage_numeric": r.stage_numeric,
            "nsd_iss_stage_ordinal": STAGE_ORDINAL_MAP.get(r.stage, -1),
            "s_positive": r.s_positive,
            "d_positive": r.d_positive,
            "has_clinical_signs": r.has_clinical_signs,
            "has_functional_impairment": r.has_functional_impairment,
            "functional_impairment_level": r.functional_impairment_level,
            "staging_confidence": r.confidence,
            "n_missing_anchors": len(r.missing_anchors),
            "missing_anchors": ",".join(r.missing_anchors) if r.missing_anchors else "",
        }
        for r in results
    ])

    # Summary statistics
    n_staged = stage_df["nsd_iss_stage"].ne("unclassified").sum()
    n_total = len(stage_df)
    stage_dist = stage_df["nsd_iss_stage"].value_counts().to_dict()

    logger.info(f"NSD-ISS staging complete: {n_staged}/{n_total} patients staged")
    logger.info(f"Stage distribution: {stage_dist}")
    logger.info(
        f"Confidence: high={stage_df['staging_confidence'].eq('high').sum()}, "
        f"medium={stage_df['staging_confidence'].eq('medium').sum()}, "
        f"low={stage_df['staging_confidence'].eq('low').sum()}"
    )

    return stage_df


def save_staging_results(
    stage_df: pd.DataFrame,
    output_dir: Path,
    prefix: str = "nsd_iss_staging",
) -> dict[str, Path]:
    """Save staging results and metadata.

    Args:
        stage_df: Staging results DataFrame
        output_dir: Output directory
        prefix: File name prefix

    Returns:
        Dictionary of output file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main staging CSV
    csv_path = output_dir / f"{prefix}_results.csv"
    stage_df.to_csv(csv_path, index=False)

    # Save metadata
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_patients": int(len(stage_df)),
        "n_staged": int(stage_df["nsd_iss_stage"].ne("unclassified").sum()),
        "n_unclassified": int(stage_df["nsd_iss_stage"].eq("unclassified").sum()),
        "stage_distribution": stage_df["nsd_iss_stage"].value_counts().to_dict(),
        "confidence_distribution": stage_df["staging_confidence"].value_counts().to_dict(),
        "s_positive_rate": float(
            stage_df["s_positive"].eq(True).sum() / stage_df["s_positive"].notna().sum()
        ) if stage_df["s_positive"].notna().any() else None,
        "d_positive_rate": float(
            stage_df["d_positive"].eq(True).sum() / stage_df["d_positive"].notna().sum()
        ) if stage_df["d_positive"].notna().any() else None,
        "thresholds": {
            "putamen_sbr_deficit": PUTAMEN_SBR_DEFICIT_THRESHOLD,
            "updrs3_clinical": UPDRS3_CLINICAL_THRESHOLD,
        },
        "reference": "Simuni et al. (2024) Lancet Neurology. NSD-ISS framework.",
    }

    meta_path = output_dir / f"{prefix}_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    logger.info(f"Saved staging results to {csv_path}")
    logger.info(f"Saved metadata to {meta_path}")

    return {"csv": csv_path, "metadata": meta_path}
