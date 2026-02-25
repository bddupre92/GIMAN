"""ML Target Encoding for NSD-ISS Biological Stages.

Transforms NSD-ISS staging results into ML-ready prediction targets.
Addresses the severe class imbalance (64.4% Stage 0) by providing
multiple target formulations at different granularities.

Target Formulations:
1. Binary: NSD-positive (Stages 1+) vs NSD-negative (Stage 0)
2. Ordinal 3-class: Early (0-1), Mild clinical (2B), Impaired (3+)
3. Full ordinal: All observed stages (0, 1, 2B, 3, 4) as 5-class

Each formulation includes class weights for focal loss / weighted CE.

Author: GIMAN Research Team
Date: February 2026
Phase: PhD Paper 1 - NSD-ISS Stage Prediction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetSpec:
    """Specification for a single ML target formulation."""

    name: str
    n_classes: int
    class_names: list[str]
    class_weights: np.ndarray  # Balanced class weights for loss function
    label_column: str  # Column name in the enriched DataFrame
    is_ordinal: bool


# ---------------------------------------------------------------------------
# Binary target: NSD-positive vs NSD-negative
# ---------------------------------------------------------------------------

def encode_binary(stage_df: pd.DataFrame) -> pd.Series:
    """Encode NSD-ISS stages as binary: NSD-positive (1) vs NSD-negative (0).

    NSD-positive = any patient with at least one biological anchor positive
    (Stage 1 or higher). Stage 0 and unclassified → 0.

    This is the most balanced formulation (783 positive vs 1418 negative
    in our PPMI staging).
    """
    binary = stage_df["nsd_iss_stage"].map(
        lambda s: 1 if s not in ("0", "unclassified") else 0
    ).astype(int)
    return binary


# ---------------------------------------------------------------------------
# 3-class ordinal: Early / Mild Clinical / Functionally Impaired
# ---------------------------------------------------------------------------

_THREE_CLASS_MAP = {
    "0": 0,           # Early / no biological markers
    "1": 0,           # Early / biological markers, no clinical signs
    "2A": 1,          # Mild clinical
    "2B": 1,          # Mild clinical
    "3": 2,           # Functionally impaired
    "4": 2,           # Functionally impaired
    "5": 2,           # Functionally impaired
    "6": 2,           # Functionally impaired
    "unclassified": -1,
}

THREE_CLASS_NAMES = ["Early (0-1)", "Mild Clinical (2A-2B)", "Impaired (3+)"]


def encode_three_class(stage_df: pd.DataFrame) -> pd.Series:
    """Encode NSD-ISS stages as 3-class ordinal.

    Class 0: Early (Stages 0-1) — no/minimal clinical involvement
    Class 1: Mild clinical (Stages 2A-2B) — clinical signs, no impairment
    Class 2: Functionally impaired (Stages 3+) — functional impairment

    Unclassified patients get -1 (should be filtered before training).
    """
    return stage_df["nsd_iss_stage"].map(_THREE_CLASS_MAP).astype(int)


# ---------------------------------------------------------------------------
# Full ordinal: All observed stages
# ---------------------------------------------------------------------------

_FULL_ORDINAL_MAP = {
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

# For our PPMI data, only stages 0, 1, 2B, 3, 4 are observed
OBSERVED_STAGE_NAMES = ["Stage 0", "Stage 1", "Stage 2B", "Stage 3", "Stage 4"]
_OBSERVED_ORDINAL_MAP = {
    "0": 0,
    "1": 1,
    "2B": 2,
    "3": 3,
    "4": 4,
    "unclassified": -1,
}


def encode_full_ordinal(stage_df: pd.DataFrame) -> pd.Series:
    """Encode NSD-ISS stages as full ordinal (observed stages only).

    Maps to 0-4 for stages (0, 1, 2B, 3, 4).
    Unclassified → -1 (filter before training).
    """
    return stage_df["nsd_iss_stage"].map(_OBSERVED_ORDINAL_MAP).fillna(-1).astype(int)


# ---------------------------------------------------------------------------
# NSD-positive subgroup ordinal (Stages 1+ only)
# ---------------------------------------------------------------------------

_NSD_POSITIVE_MAP = {
    "1": 0,
    "2A": 1,
    "2B": 1,  # Collapse 2A/2B since 2A is unobserved
    "3": 2,
    "4": 3,
    "5": 4,
    "6": 5,
}

NSD_POSITIVE_NAMES = ["Stage 1", "Stage 2B", "Stage 3", "Stage 4"]


def encode_nsd_positive_ordinal(stage_df: pd.DataFrame) -> pd.Series:
    """Encode stages for NSD-positive subgroup only (Stages 1+).

    Excludes Stage 0 and unclassified. Maps to 0-3 for stages (1, 2B, 3, 4).
    Returns -1 for Stage 0 / unclassified patients (filter before training).
    """
    result = stage_df["nsd_iss_stage"].map(_NSD_POSITIVE_MAP).fillna(-1).astype(int)
    return result


# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------

def compute_balanced_weights(labels: np.ndarray) -> np.ndarray:
    """Compute balanced class weights for a label array.

    Uses sklearn's compute_class_weight with 'balanced' strategy:
    weight_i = n_samples / (n_classes * n_samples_i)

    Excludes -1 (unclassified) from computation.
    """
    valid = labels[labels >= 0]
    classes = np.unique(valid)
    weights = compute_class_weight("balanced", classes=classes, y=valid)
    return weights


# ---------------------------------------------------------------------------
# Main enrichment function
# ---------------------------------------------------------------------------

def enrich_staging_with_targets(
    stage_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, TargetSpec]]:
    """Add all ML target columns to staging DataFrame.

    Args:
        stage_df: NSD-ISS staging results (from compute_nsd_iss_stages.py)

    Returns:
        Tuple of:
        - Enriched DataFrame with target columns added
        - Dictionary of TargetSpec objects keyed by target name
    """
    df = stage_df.copy()

    # 1. Binary target
    df["target_binary"] = encode_binary(df)
    binary_weights = compute_balanced_weights(df["target_binary"].values)

    # 2. Three-class ordinal
    df["target_3class"] = encode_three_class(df)
    valid_3class = df["target_3class"].values
    weights_3class = compute_balanced_weights(valid_3class[valid_3class >= 0])

    # 3. Full ordinal (observed stages)
    df["target_full_ordinal"] = encode_full_ordinal(df)
    valid_full = df["target_full_ordinal"].values
    weights_full = compute_balanced_weights(valid_full[valid_full >= 0])

    # 4. NSD-positive subgroup ordinal
    df["target_nsd_positive"] = encode_nsd_positive_ordinal(df)
    valid_nsd = df["target_nsd_positive"].values
    weights_nsd = compute_balanced_weights(valid_nsd[valid_nsd >= 0])

    # Build TargetSpec objects
    specs = {
        "binary": TargetSpec(
            name="binary",
            n_classes=2,
            class_names=["NSD-negative", "NSD-positive"],
            class_weights=binary_weights,
            label_column="target_binary",
            is_ordinal=False,
        ),
        "three_class": TargetSpec(
            name="three_class",
            n_classes=3,
            class_names=THREE_CLASS_NAMES,
            class_weights=weights_3class,
            label_column="target_3class",
            is_ordinal=True,
        ),
        "full_ordinal": TargetSpec(
            name="full_ordinal",
            n_classes=len(OBSERVED_STAGE_NAMES),
            class_names=OBSERVED_STAGE_NAMES,
            class_weights=weights_full,
            label_column="target_full_ordinal",
            is_ordinal=True,
        ),
        "nsd_positive": TargetSpec(
            name="nsd_positive",
            n_classes=len(NSD_POSITIVE_NAMES),
            class_names=NSD_POSITIVE_NAMES,
            class_weights=weights_nsd,
            label_column="target_nsd_positive",
            is_ordinal=True,
        ),
    }

    # Log summary
    for name, spec in specs.items():
        col = spec.label_column
        valid = df[col][df[col] >= 0]
        dist = valid.value_counts().sort_index()
        logger.info(
            f"Target '{name}' ({spec.n_classes} classes): "
            f"n={len(valid)}, distribution={dist.to_dict()}, "
            f"weights={np.round(spec.class_weights, 3).tolist()}"
        )

    return df, specs


def save_enriched_targets(
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Save enriched staging DataFrame with ML targets.

    Args:
        df: Enriched DataFrame from enrich_staging_with_targets()
        output_path: Output CSV path

    Returns:
        Path to saved file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved enriched targets to {output_path}")
    return output_path


def load_enriched_targets(path: Path) -> pd.DataFrame:
    """Load enriched staging DataFrame with ML targets."""
    return pd.read_csv(path)
