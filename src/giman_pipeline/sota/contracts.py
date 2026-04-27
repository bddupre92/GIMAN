from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModalityRegistry:
    modality_id: str
    source_path: str
    key_columns: list[str]
    time_columns: list[str]
    quality_status: str
    readiness_status: str


@dataclass(frozen=True)
class FeatureLineageRecord:
    feature_name: str
    raw_source: str
    transform_steps: list[str]
    imputation_rule: str
    current_usage: str


@dataclass(frozen=True)
class DatasetContract:
    patno_key: str = "PATNO"
    time_key: str = "time"
    event_key: str = "event"
    classification_key: str = "saa_label"
    split_hash: str = ""
    schema_version: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "patno_key": self.patno_key,
            "time_key": self.time_key,
            "event_key": self.event_key,
            "classification_key": self.classification_key,
            "split_hash": self.split_hash,
            "schema_version": self.schema_version,
        }


PROXY_CLASSIFICATION_KEYS = {"event", "event_observed", "phenoconverted"}


def assert_classification_contract(
    classification_key: str,
    survival_event_key: str,
) -> None:
    if classification_key in PROXY_CLASSIFICATION_KEYS:
        raise ValueError(
            f"classification label key cannot be a survival proxy: {classification_key}"
        )
    if classification_key == survival_event_key:
        raise ValueError("classification label key must differ from survival event key")


def load_contract_from_metadata(metadata_path: Path) -> DatasetContract:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    path_contract = payload.get("path_contract", {})
    return DatasetContract(
        patno_key=path_contract.get("patient_key", "PATNO"),
        time_key=path_contract.get("survival_time_key", "time"),
        event_key=path_contract.get("survival_event_key", "event"),
        classification_key=path_contract.get("classification_label_key", "saa_label"),
        split_hash=payload.get("split_hash", ""),
        schema_version=payload.get("schema_version", ""),
    )


def validate_required_columns(columns: list[str], contract: DatasetContract) -> None:
    required = {
        contract.patno_key,
        contract.time_key,
        contract.event_key,
        contract.classification_key,
    }
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
