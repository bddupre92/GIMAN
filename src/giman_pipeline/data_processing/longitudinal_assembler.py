"""Longitudinal data assembly for Temporal GIMAN.

Provides reusable classes for extracting multi-visit PPMI features and
assembling them into temporal sequences suitable for recurrent models.

Classes:
    LongitudinalAssembler: Extracts features at multiple visits per patient.
    MedicationTracker: Extracts PD medication status across visits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


# Default PPMI visit-to-month mapping
VISIT_MONTH_MAP: dict[str, int] = {
    "BL": 0,
    "V01": 3,
    "V02": 6,
    "V04": 12,
    "V06": 18,
    "V08": 24,
    "V10": 30,
    "V12": 36,
}

# Ordered visit codes for consistent sequencing
VISIT_ORDER: list[str] = ["BL", "V01", "V02", "V04", "V06", "V08", "V10", "V12"]


@dataclass
class PatientSequence:
    """Temporal sequence for a single patient."""

    patno: int
    features: np.ndarray  # [T, F] feature matrix
    time_months: np.ndarray  # [T] months from baseline
    obs_mask: np.ndarray  # [T, F] binary observation mask (1=observed, 0=missing)
    visit_ids: list[str]  # [T] EVENT_ID labels
    n_visits: int


@dataclass
class LongitudinalDataset:
    """Collection of patient temporal sequences with metadata."""

    sequences: dict[int, PatientSequence]  # patno -> sequence
    feature_names: list[str]
    modality_map: dict[str, list[str]]  # modality_name -> [feature_names]
    visit_schedule: dict[str, int]
    coverage_matrix: pd.DataFrame  # [N_patients x N_visits] boolean
    summary: dict[str, Any]


class LongitudinalAssembler:
    """Assembles multi-visit PPMI features into temporal sequences.

    Generalises the baseline-only extraction in build_training_dataset.py
    to extract features at every available visit per patient.
    """

    def __init__(
        self,
        config_path: str | Path,
        visit_schedule: dict[str, int] | None = None,
    ) -> None:
        self.config = yaml.safe_load(open(config_path))
        self.modalities = self.config["modalities"]
        self.visit_schedule = visit_schedule or dict(VISIT_MONTH_MAP)
        self.valid_events = set(self.visit_schedule.keys())

        # Build ordered feature list from config
        self.feature_names: list[str] = []
        self.modality_map: dict[str, list[str]] = {}
        for mod_name, mod_cfg in self.modalities.items():
            feats = mod_cfg["features"]
            self.modality_map[mod_name] = feats
            self.feature_names.extend(feats)

        self.n_features = len(self.feature_names)

    def extract_multi_visit(
        self,
        raw_df: pd.DataFrame,
        patnos: set[int],
        feature_col: str,
        rename: str | None = None,
        event_col: str = "EVENT_ID",
        patno_col: str = "PATNO",
    ) -> pd.DataFrame:
        """Extract a single feature across multiple visits.

        Instead of filtering EVENT_ID=="BL", retains all visits in
        the valid schedule.

        Returns DataFrame with columns [PATNO, EVENT_ID, <feature>].
        """
        df = raw_df[
            (raw_df[patno_col].isin(patnos))
            & (raw_df[event_col].isin(self.valid_events))
        ].copy()
        # Take first record per patient-visit (handles duplicates)
        df = df.groupby([patno_col, event_col]).first().reset_index()
        cols = [patno_col, event_col]
        if feature_col in df.columns:
            cols.append(feature_col)
        result = df[cols].copy()
        if rename and feature_col in result.columns:
            result = result.rename(columns={feature_col: rename})
        return result

    def extract_multi_visit_multi_col(
        self,
        raw_df: pd.DataFrame,
        patnos: set[int],
        col_map: dict[str, str],
        event_col: str = "EVENT_ID",
        patno_col: str = "PATNO",
    ) -> pd.DataFrame:
        """Extract multiple features across multiple visits.

        Args:
            col_map: {source_column: target_feature_name}

        Returns DataFrame with columns [PATNO, EVENT_ID, target1, target2, ...].
        """
        df = raw_df[
            (raw_df[patno_col].isin(patnos))
            & (raw_df[event_col].isin(self.valid_events))
        ].copy()
        df = df.groupby([patno_col, event_col]).first().reset_index()

        result = df[[patno_col, event_col]].copy()
        for src, dst in col_map.items():
            if src in df.columns:
                result[dst] = df[src].values
            else:
                result[dst] = np.nan
        return result

    def merge_visit_features(
        self,
        feature_dfs: list[pd.DataFrame],
    ) -> pd.DataFrame:
        """Merge multiple per-visit feature DataFrames into a single long-form table.

        Each DataFrame must have [PATNO, EVENT_ID, ...features].
        Result is outer-joined on (PATNO, EVENT_ID).
        """
        if not feature_dfs:
            return pd.DataFrame(columns=["PATNO", "EVENT_ID"])

        merged = feature_dfs[0]
        for df in feature_dfs[1:]:
            merged = merged.merge(df, on=["PATNO", "EVENT_ID"], how="outer")
        return merged

    def long_to_sequences(
        self,
        long_df: pd.DataFrame,
        endpoints: pd.DataFrame,
    ) -> LongitudinalDataset:
        """Convert long-form visit data to PatientSequence objects.

        Args:
            long_df: Long-form DataFrame with [PATNO, EVENT_ID, feature1, ...]
            endpoints: Endpoint DataFrame with [PATNO, time_to_event, phenoconverted, ...]
                       (used to restrict to cohort patients)

        Returns:
            LongitudinalDataset with per-patient temporal sequences.
        """
        cohort_patnos = set(endpoints["PATNO"].values)
        long_df = long_df[long_df["PATNO"].isin(cohort_patnos)].copy()

        # Add month column
        long_df["t_month"] = long_df["EVENT_ID"].map(self.visit_schedule)
        long_df = long_df.dropna(subset=["t_month"])
        long_df["t_month"] = long_df["t_month"].astype(int)

        # Sort by patient then time
        long_df = long_df.sort_values(["PATNO", "t_month"])

        # Ensure all feature columns exist
        for feat in self.feature_names:
            if feat not in long_df.columns:
                long_df[feat] = np.nan

        # Build coverage matrix
        coverage_rows = []
        sequences: dict[int, PatientSequence] = {}

        for patno, group in long_df.groupby("PATNO"):
            group = group.sort_values("t_month")
            T = len(group)
            if T == 0:
                continue

            features = group[self.feature_names].values.astype(np.float32)  # [T, F]
            obs_mask = (~np.isnan(features)).astype(np.float32)  # [T, F]
            # Replace NaN with 0 in feature matrix (will be masked)
            features = np.nan_to_num(features, nan=0.0)
            time_months = group["t_month"].values.astype(np.float32)
            visit_ids = group["EVENT_ID"].tolist()

            sequences[int(patno)] = PatientSequence(
                patno=int(patno),
                features=features,
                time_months=time_months,
                obs_mask=obs_mask,
                visit_ids=visit_ids,
                n_visits=T,
            )

            # Coverage row
            row = {"PATNO": int(patno)}
            for vid in VISIT_ORDER:
                row[vid] = vid in visit_ids
            coverage_rows.append(row)

        coverage_df = pd.DataFrame(coverage_rows)

        # Summary statistics
        n_visits_list = [s.n_visits for s in sequences.values()]
        summary = {
            "n_patients": len(sequences),
            "n_features": self.n_features,
            "n_visits_median": float(np.median(n_visits_list)) if n_visits_list else 0,
            "n_visits_mean": float(np.mean(n_visits_list)) if n_visits_list else 0,
            "n_visits_min": int(np.min(n_visits_list)) if n_visits_list else 0,
            "n_visits_max": int(np.max(n_visits_list)) if n_visits_list else 0,
            "coverage_per_visit": {
                vid: int(coverage_df[vid].sum()) if vid in coverage_df.columns else 0
                for vid in VISIT_ORDER
            },
        }

        return LongitudinalDataset(
            sequences=sequences,
            feature_names=self.feature_names,
            modality_map=self.modality_map,
            visit_schedule=self.visit_schedule,
            coverage_matrix=coverage_df,
            summary=summary,
        )


class MedicationTracker:
    """Tracks PD medication status across visits.

    Extracts levodopa, dopamine agonist, MAO-B inhibitor usage
    from PPMI PD Medical History and CONMED tables.
    """

    MEDICATION_FEATURES = [
        "LEVODOPA",
        "DOPAMINE_AGONIST",
        "MAO_B_INHIBITOR",
        "MEDICATION_ACTIVE",
    ]

    def __init__(self, valid_events: set[str] | None = None) -> None:
        self.valid_events = valid_events or set(VISIT_MONTH_MAP.keys())

    def extract_medication_features(
        self,
        med_history_path: Path,
        conmed_path: Path | None,
        patnos: set[int],
    ) -> pd.DataFrame:
        """Extract binary medication status per patient-visit.

        Returns DataFrame with [PATNO, EVENT_ID, LEVODOPA, DOPAMINE_AGONIST,
        MAO_B_INHIBITOR, MEDICATION_ACTIVE].
        """
        rows: list[dict] = []

        # PD Medical History
        try:
            med = pd.read_csv(med_history_path, low_memory=False)
            med = med[
                (med["PATNO"].isin(patnos))
                & (med["EVENT_ID"].isin(self.valid_events))
            ]

            for (patno, event_id), group in med.groupby(["PATNO", "EVENT_ID"]):
                row: dict[str, Any] = {
                    "PATNO": int(patno),
                    "EVENT_ID": str(event_id),
                    "LEVODOPA": 0,
                    "DOPAMINE_AGONIST": 0,
                    "MAO_B_INHIBITOR": 0,
                    "MEDICATION_ACTIVE": 0,
                }

                # Check medication columns (varies by PPMI version)
                text_cols = [c for c in group.columns if "PDMEDYN" in c or "PDMED" in c]
                if text_cols:
                    any_med = group[text_cols].notna().any().any()
                    row["MEDICATION_ACTIVE"] = int(any_med)

                # Check for specific medication class indicators
                for col in group.columns:
                    col_upper = col.upper()
                    vals = group[col].astype(str).str.upper()
                    if "LEVODOPA" in col_upper or "SINEMET" in col_upper:
                        if vals.str.contains("1|YES|Y", na=False).any():
                            row["LEVODOPA"] = 1
                            row["MEDICATION_ACTIVE"] = 1
                    if "AGONIST" in col_upper or "PRAMIPEXOLE" in col_upper:
                        if vals.str.contains("1|YES|Y", na=False).any():
                            row["DOPAMINE_AGONIST"] = 1
                            row["MEDICATION_ACTIVE"] = 1
                    if "MAOB" in col_upper or "RASAGILINE" in col_upper or "SELEGILINE" in col_upper:
                        if vals.str.contains("1|YES|Y", na=False).any():
                            row["MAO_B_INHIBITOR"] = 1
                            row["MEDICATION_ACTIVE"] = 1

                rows.append(row)

            print(f"Medication history: {len(rows)} patient-visit records")

        except FileNotFoundError:
            print(f"Medication history file not found: {med_history_path}")
        except Exception as e:
            print(f"Medication extraction error: {e}")

        # CONMED (concomitant medications) as supplementary source
        if conmed_path and conmed_path.exists():
            try:
                conmed = pd.read_csv(conmed_path, low_memory=False)
                conmed = conmed[
                    (conmed["PATNO"].isin(patnos))
                    & (conmed["EVENT_ID"].isin(self.valid_events))
                ]
                # Merge CONMED signals into existing rows (additive)
                existing = {(r["PATNO"], r["EVENT_ID"]) for r in rows}

                for (patno, event_id), group in conmed.groupby(["PATNO", "EVENT_ID"]):
                    if (int(patno), str(event_id)) in existing:
                        continue
                    row = {
                        "PATNO": int(patno),
                        "EVENT_ID": str(event_id),
                        "LEVODOPA": 0,
                        "DOPAMINE_AGONIST": 0,
                        "MAO_B_INHIBITOR": 0,
                        "MEDICATION_ACTIVE": 0,
                    }
                    # Check LEDD or medication name columns
                    for col in group.columns:
                        vals = group[col].astype(str).str.upper()
                        if vals.str.contains("LEVODOPA|SINEMET|CARBIDOPA", na=False).any():
                            row["LEVODOPA"] = 1
                            row["MEDICATION_ACTIVE"] = 1
                    rows.append(row)

                print(f"CONMED supplementary: {len(conmed)} records checked")
            except Exception as e:
                print(f"CONMED extraction: {e}")

        if not rows:
            return pd.DataFrame(
                columns=["PATNO", "EVENT_ID"] + self.MEDICATION_FEATURES
            )

        result = pd.DataFrame(rows)
        # Deduplicate: keep max medication signal per patient-visit
        result = (
            result.groupby(["PATNO", "EVENT_ID"])
            .max()
            .reset_index()
        )
        return result

    def get_medication_summary(self, med_df: pd.DataFrame) -> dict[str, Any]:
        """Summarise medication coverage."""
        if med_df.empty:
            return {"n_records": 0, "n_patients": 0}

        return {
            "n_records": len(med_df),
            "n_patients": med_df["PATNO"].nunique(),
            "levodopa_any": int(med_df["LEVODOPA"].sum()),
            "dopamine_agonist_any": int(med_df["DOPAMINE_AGONIST"].sum()),
            "mao_b_any": int(med_df["MAO_B_INHIBITOR"].sum()),
            "any_medication": int(med_df["MEDICATION_ACTIVE"].sum()),
        }
