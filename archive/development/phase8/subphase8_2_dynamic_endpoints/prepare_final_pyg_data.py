"""Prepare deterministic, leakage-safe PyTorch Geometric datasets for Phase 8/9.

Canonical internal contract:
- Required columns after schema normalization: PATNO, time, event, saa_label
- train_data.pt/test_data.pt fields: x, edge_index, time, event, saa_label, patno, source_index, split_mask, split_id
- Metadata includes schema, seed, split method, split hash, and cohort statistics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

SEED_DEFAULT = 42
SCHEMA_VERSION = "sota_sprint2_v1"

PATNO_ALIASES = ("PATNO", "patno", "patient_id", "subject_id")
TIME_ALIASES = ("time", "time_to_event", "event_time")
EVENT_ALIASES = ("event", "phenoconverted", "event_observed")
SAA_ALIASES = (
    "saa_label",
    "SAA_POSITIVE",
    "SAA_LABEL",
    "saa_status",
    "SAA_STATUS",
    "saa",
    "SAA",
    "subacute_anxiety",
    "anxiety_label",
)
SAA_PROXY_ALIASES = ("event", "event_observed", "phenoconverted")

IMAGING_HINTS = ("_VOL", "_CTH", "_SBR", "ASYMMETRY", "DATSCAN", "MRI")
GENETIC_FEATURES = {"LRRK2", "GBA", "APOE_E4", "SNCA", "GENETIC_RISK_SCORE"}
CSF_FEATURES = {"ALPHA_SYNUCLEIN", "TOTAL_TAU", "ABETA42", "PTAU181"}
CLINICAL_FEATURES = {
    "UPDRS_I",
    "UPDRS_II",
    "SCHWAB_ENGLAND",
    "PIGD_SCORE",
    "TREMOR_SCORE",
    "UPSIT_SCORE",
    "RBD_SCORE",
    "SCOPA_AUT_SCORE",
    "ESS_SCORE",
}


class SchemaError(ValueError):
    """Raised when required schema fields are missing or invalid."""


def resolve_column(df: pd.DataFrame, aliases: tuple[str, ...], canonical: str) -> str:
    """Resolve first matching alias for canonical column name."""
    for name in aliases:
        if name in df.columns:
            return name
    raise SchemaError(
        f"Missing required column for '{canonical}'. Tried aliases: {list(aliases)}"
    )


def load_unified_dataset(input_csv: Path) -> pd.DataFrame:
    """Load final unified training dataset."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    df = pd.read_csv(input_csv)
    print(f"✓ Loaded unified dataset: {df.shape}")
    print(f"  Observations: {len(df)}")
    return df


def _build_pat_key(series: pd.Series) -> pd.Series:
    """Build stable PATNO join keys that tolerate numeric/string representations."""
    txt = series.astype(str).str.strip()
    numeric = pd.to_numeric(series, errors="coerce")
    mask = numeric.notna()
    if mask.any():
        txt.loc[mask] = numeric.loc[mask].astype(np.int64).astype(str)
    return txt


def inject_saa_labels(
    df: pd.DataFrame,
    saa_label_csv: Path,
    *,
    drop_unlabeled_saa: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inject real SAA labels (PATNO,saa_label) from an external source CSV."""
    if not saa_label_csv.exists():
        raise FileNotFoundError(f"SAA label CSV not found: {saa_label_csv}")

    labels = pd.read_csv(saa_label_csv)
    pat_source = resolve_column(df, PATNO_ALIASES, "PATNO(base)")
    label_pat_source = resolve_column(labels, PATNO_ALIASES, "PATNO(label_source)")
    label_col = resolve_column(
        labels,
        SAA_ALIASES + ("saa_positive", "saa_final", "SAA_POSITIVE_FINAL"),
        "saa_label(label_source)",
    )

    labels = labels[[label_pat_source, label_col]].copy()
    labels.columns = ["PATNO_LABEL_SRC", "saa_label_external"]
    labels["saa_label_external"] = pd.to_numeric(
        labels["saa_label_external"], errors="coerce"
    )
    labels = labels.dropna(subset=["saa_label_external"])
    labels["saa_label_external"] = labels["saa_label_external"].astype(int)
    unique_labels = sorted(labels["saa_label_external"].unique().tolist())
    if any(v not in (0, 1) for v in unique_labels):
        raise SchemaError(
            f"External saa_label must be binary 0/1. Observed values: {unique_labels}"
        )

    labels["_pat_key"] = _build_pat_key(labels["PATNO_LABEL_SRC"])
    labels = (
        labels.groupby("_pat_key", as_index=False)["saa_label_external"]
        .max()
        .rename(columns={"saa_label_external": "saa_label"})
    )

    out = df.copy()
    out["_pat_key"] = _build_pat_key(out[pat_source])
    out = out.merge(labels, on="_pat_key", how="left")

    total_rows = int(len(out))
    labeled_rows = int(out["saa_label"].notna().sum())
    dropped_rows = 0
    if drop_unlabeled_saa:
        before = len(out)
        out = out[out["saa_label"].notna()].copy()
        dropped_rows = int(before - len(out))

    out = out.drop(columns=["_pat_key"])
    summary = {
        "label_source_csv": str(saa_label_csv),
        "rows_total": total_rows,
        "rows_with_saa_label": labeled_rows,
        "rows_dropped_unlabeled": dropped_rows,
        "label_coverage": float(labeled_rows / total_rows) if total_rows else 0.0,
        "drop_unlabeled_saa": bool(drop_unlabeled_saa),
    }
    print(
        "✓ Injected external SAA labels "
        f"(coverage={summary['label_coverage']:.1%}, dropped={dropped_rows})"
    )
    return out, summary


def normalize_schema(
    df: pd.DataFrame,
    *,
    strict_real_saa_label: bool = True,
    allow_identical_event_saa: bool = False,
) -> pd.DataFrame:
    """Normalize endpoint and label columns to canonical internal schema."""
    df = df.copy()
    event_source = resolve_column(df, EVENT_ALIASES, "event")
    try:
        saa_source = resolve_column(df, SAA_ALIASES, "saa_label")
    except SchemaError:
        if strict_real_saa_label:
            raise
        saa_source = resolve_column(df, SAA_PROXY_ALIASES, "saa_label(proxy_override)")

    if strict_real_saa_label and saa_source in SAA_PROXY_ALIASES:
        raise SchemaError(
            "Invalid SAA source: classification label resolves to survival/proxy column "
            f"'{saa_source}'. Provide a true SAA label column (e.g., 'saa_label')."
        )

    patno_source = resolve_column(df, PATNO_ALIASES, "PATNO")
    time_source = resolve_column(df, TIME_ALIASES, "time")

    # Use explicit column assignment so event/saa can safely share a source name
    # in proxy-override mode without rename collisions.
    df["PATNO"] = df[patno_source]
    df["time"] = df[time_source]
    df["event"] = df[event_source]
    df["saa_label"] = df[saa_source]

    # Drop non-canonical aliases to avoid leakage (e.g., phenoconverted/time_to_event).
    alias_cols = set(PATNO_ALIASES + TIME_ALIASES + EVENT_ALIASES + SAA_ALIASES)
    alias_cols -= {"PATNO", "time", "event", "saa_label"}
    drop_aliases = [c for c in alias_cols if c in df.columns]
    if drop_aliases:
        df = df.drop(columns=drop_aliases)

    # Validate + coerce numeric targets
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["event"] = pd.to_numeric(df["event"], errors="coerce")
    df["saa_label"] = pd.to_numeric(df["saa_label"], errors="coerce")

    required = ["PATNO", "time", "event", "saa_label"]
    missing_after = [c for c in required if c not in df.columns]
    if missing_after:
        raise SchemaError(f"Missing required canonical columns: {missing_after}")

    if df[required].isna().any().any():
        nan_counts = df[required].isna().sum().to_dict()
        raise SchemaError(
            "Canonical schema contains nulls after coercion. "
            f"Fix label/source data first. Null counts: {nan_counts}"
        )

    # Enforce binary classification/event labels
    for col in ("event", "saa_label"):
        unique_vals = sorted(pd.unique(df[col]).tolist())
        if any(v not in (0, 1) for v in unique_vals):
            raise SchemaError(
                f"Column '{col}' must be binary (0/1). Observed values: {unique_vals}"
            )

    if strict_real_saa_label and not allow_identical_event_saa:
        if np.array_equal(
            df["event"].to_numpy(dtype=int), df["saa_label"].to_numpy(dtype=int)
        ):
            raise SchemaError(
                "Proxy-label violation: 'saa_label' is identical to 'event'. "
                "Provide a true SAA endpoint label."
            )

    print("✓ Normalized canonical schema: PATNO, time, event, saa_label")
    print(f"  event source: {event_source}")
    print(f"  saa_label source: {saa_source}")
    return df


def _modality_cols(df: pd.DataFrame, mode: str) -> list[str]:
    """Return candidate columns for a modality."""
    cols = list(df.columns)
    if mode == "imaging":
        return [
            c
            for c in cols
            if c not in {"PATNO", "time", "event", "saa_label"}
            and any(h in c for h in IMAGING_HINTS)
        ]
    if mode == "genetic":
        return [c for c in cols if c in GENETIC_FEATURES]
    if mode == "csf":
        return [c for c in cols if c in CSF_FEATURES]
    if mode == "clinical":
        return [c for c in cols if c in CLINICAL_FEATURES]
    return []


def add_modality_presence_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    """Add binary modality-presence masks used for missing-modality handling."""
    out = df.copy()
    summary: dict[str, dict[str, float | int]] = {}
    for mode in ("imaging", "genetic", "csf", "clinical"):
        cols = _modality_cols(out, mode)
        mask_col = f"modality_present_{mode}"
        if cols:
            present = out[cols].notna().any(axis=1).astype(int)
        else:
            present = pd.Series(np.zeros(len(out), dtype=int), index=out.index)
        out[mask_col] = present
        summary[mode] = {
            "n_features": int(len(cols)),
            "present_rate": float(present.mean()) if len(out) else 0.0,
        }
    print("✓ Added modality presence masks: imaging/genetic/csf/clinical")
    return out, summary


def ensure_finite(name: str, array: np.ndarray) -> None:
    """Fail fast when arrays contain NaN/Inf values."""
    nan_count = int(np.isnan(array).sum())
    inf_count = int(np.isinf(array).sum())
    if nan_count or inf_count:
        raise SchemaError(
            f"{name} contains non-finite values (nan={nan_count}, inf={inf_count}). "
            "Fix preprocessing/imputation before model preparation."
        )


def encode_patno(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    """Encode PATNO as deterministic integer ids for tensor storage."""
    patno_series = df["PATNO"]
    numeric_patno = pd.to_numeric(patno_series, errors="coerce")

    if numeric_patno.notna().all():
        patno_ids = numeric_patno.astype(np.int64).to_numpy()
        mapping = {str(int(v)): int(v) for v in sorted(pd.unique(patno_ids))}
        return patno_ids, mapping

    # Fall back to deterministic factorized ids
    codes, uniques = pd.factorize(patno_series.astype(str), sort=True)
    patno_ids = codes.astype(np.int64)
    mapping = {u: int(i) for i, u in enumerate(uniques)}
    return patno_ids, mapping


def split_patient_level(
    patno_ids: np.ndarray,
    event: np.ndarray,
    saa_label: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Patient-level deterministic split with leakage protection."""
    patient_df = (
        pd.DataFrame({"PATNO": patno_ids, "event": event, "saa_label": saa_label})
        .groupby("PATNO", as_index=False)
        .max()
    )

    # Prefer joint stratification over (event, saa_label) when feasible
    joint_strata = patient_df["event"].astype(int) * 2 + patient_df["saa_label"].astype(
        int
    )
    strata_counts = joint_strata.value_counts().to_dict()

    stratify = None
    split_method = "patient_random_no_strat"
    if len(strata_counts) > 1 and min(strata_counts.values()) >= 2:
        stratify = joint_strata
        split_method = "patient_stratified_event_saa"
    else:
        event_strata = patient_df["event"].astype(int)
        event_counts = event_strata.value_counts().to_dict()
        if len(event_counts) > 1 and min(event_counts.values()) >= 2:
            stratify = event_strata
            split_method = "patient_stratified_event_only"

    train_patnos, test_patnos = train_test_split(
        patient_df["PATNO"].to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    train_mask = np.isin(patno_ids, train_patnos)
    test_mask = np.isin(patno_ids, test_patnos)

    if np.any(np.isin(train_patnos, test_patnos)):
        raise RuntimeError(
            "Patient leakage detected: overlapping PATNO between train/test"
        )

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    split_blob = {
        "seed": int(random_state),
        "split_method": split_method,
        "n_patients_train": int(len(np.unique(train_patnos))),
        "n_patients_test": int(len(np.unique(test_patnos))),
        "patient_disjoint": True,
        "train_patnos": sorted(map(int, np.unique(train_patnos).tolist())),
        "test_patnos": sorted(map(int, np.unique(test_patnos).tolist())),
    }

    hash_payload = json.dumps(split_blob, sort_keys=True).encode("utf-8")
    split_blob["split_hash"] = hashlib.sha256(hash_payload).hexdigest()

    print(
        f"✓ Patient-level split: train={len(train_idx)} rows, test={len(test_idx)} rows"
    )
    print(
        f"  Patients train/test: {split_blob['n_patients_train']}/{split_blob['n_patients_test']}"
    )
    print(f"  Method: {split_method}")

    return train_idx, test_idx, split_blob


def prepare_feature_blocks(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    drop_feature_names: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Prepare feature matrices with train-only imputer/scaler fitting."""
    exclude_cols = {
        "PATNO",
        "time",
        "event",
        "saa_label",
        "landmark_month",
        "original_time",
        "original_event",
        "cohort",
    }
    exclude_cols |= set(TIME_ALIASES)
    exclude_cols |= set(EVENT_ALIASES)
    exclude_cols |= set(SAA_ALIASES)
    exclude_cols |= set(SAA_PROXY_ALIASES)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    if drop_feature_names:
        feature_cols = [c for c in feature_cols if c not in drop_feature_names]
        print(
            f"✓ Feature drop list applied: removed {len(drop_feature_names)} requested columns"
        )
    if not feature_cols:
        raise SchemaError("No feature columns found after exclusions")

    # Coerce any non-numeric feature columns (e.g., EVENT_ID) and drop columns
    # that remain fully missing after coercion.
    feature_df = df[feature_cols].copy()
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(feature_df[col]):
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

    dropped_cols = [c for c in feature_df.columns if feature_df[c].isna().all()]
    if dropped_cols:
        feature_df = feature_df.drop(columns=dropped_cols)
        feature_cols = [c for c in feature_cols if c not in dropped_cols]
        print(f"✓ Dropped non-numeric/all-NaN feature columns: {dropped_cols}")

    if not feature_cols:
        raise SchemaError("No usable numeric feature columns remain after coercion")

    x_all = feature_df[feature_cols].to_numpy(dtype=float)

    x_train_raw = x_all[train_idx]
    x_test_raw = x_all[test_idx]

    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=SEED_DEFAULT,
            n_jobs=-1,
        ),
        max_iter=15,
        random_state=SEED_DEFAULT,
        initial_strategy="median",
        skip_complete=True,
    )
    x_train_imp = imputer.fit_transform(x_train_raw)
    x_test_imp = imputer.transform(x_test_raw)

    # Keep biologic marker values in observed train ranges.
    csf_clip_ranges: dict[str, dict[str, float]] = {}
    feature_to_idx = {name: i for i, name in enumerate(feature_cols)}
    for feature in sorted(CSF_FEATURES):
        if feature not in feature_to_idx:
            continue
        idx = feature_to_idx[feature]
        observed = x_train_raw[:, idx]
        observed = observed[np.isfinite(observed)]
        if observed.size < 5:
            continue
        clip_min = float(np.min(observed))
        clip_max = float(np.max(observed))
        if clip_min >= clip_max:
            continue
        x_train_imp[:, idx] = np.clip(x_train_imp[:, idx], clip_min, clip_max)
        x_test_imp[:, idx] = np.clip(x_test_imp[:, idx], clip_min, clip_max)
        csf_clip_ranges[feature] = {"min": clip_min, "max": clip_max}

    ensure_finite("x_train_imputed", x_train_imp)
    ensure_finite("x_test_imputed", x_test_imp)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_imp)
    x_test = scaler.transform(x_test_imp)
    ensure_finite("x_train_scaled", x_train)
    ensure_finite("x_test_scaled", x_test)

    missing_fraction = feature_df[feature_cols].isna().mean()
    variance = np.var(x_train, axis=0)
    constant_mask = variance <= 1e-8
    dominant_fraction = []
    for i in range(x_train.shape[1]):
        _, counts = np.unique(np.round(x_train[:, i], 8), return_counts=True)
        dominant_fraction.append(float(np.max(counts) / max(len(x_train), 1)))

    records = []
    for i, name in enumerate(feature_cols):
        action = "drop_constant" if constant_mask[i] else "keep"
        records.append(
            {
                "feature_name": name,
                "variance": float(variance[i]),
                "is_constant": bool(constant_mask[i]),
                "missing_fraction": float(missing_fraction[name]),
                "lineage_ok": True,
                "dominant_value_fraction": dominant_fraction[i],
                "action": action,
            }
        )

    feature_quality = {
        "n_features": int(len(feature_cols)),
        "n_constant_train_scaled": int(np.sum(constant_mask)),
        "constant_features": [
            feature_cols[i] for i, flag in enumerate(constant_mask) if flag
        ],
        "missing_fraction_mean": float(missing_fraction.mean()),
        "missing_fraction_max": float(missing_fraction.max()),
        "imputation": {
            "method": "IterativeImputer",
            "estimator": "RandomForestRegressor",
            "max_iter": 15,
            "fit_scope": "train_only_then_transform_test",
        },
        "csf_clip_ranges": csf_clip_ranges,
        "records": records,
    }

    print(f"✓ Prepared feature blocks: train={x_train.shape}, test={x_test.shape}")
    return x_train, x_test, feature_cols, feature_quality


def construct_knn_graph(x: np.ndarray, k: int = 10) -> torch.Tensor:
    """Construct deterministic kNN graph on feature space."""
    adj = kneighbors_graph(x, n_neighbors=k, mode="connectivity", include_self=False)
    adj_coo = adj.tocoo()
    edge_index = torch.tensor(np.vstack([adj_coo.row, adj_coo.col]), dtype=torch.long)
    return edge_index


def create_pyg_data(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    saa_label: np.ndarray,
    patno: np.ndarray,
    source_index: np.ndarray,
    edge_index: torch.Tensor,
    split_id: int,
) -> Data:
    """Create PyTorch Geometric Data object with canonical fields."""
    n_rows = len(x)
    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        time=torch.tensor(time, dtype=torch.float32),
        event=torch.tensor(event, dtype=torch.long),
        saa_label=torch.tensor(saa_label, dtype=torch.long),
        patno=torch.tensor(patno, dtype=torch.long),
        source_index=torch.tensor(source_index, dtype=torch.long),
        split_mask=torch.ones(n_rows, dtype=torch.bool),
        split_id=torch.full((n_rows,), split_id, dtype=torch.long),
    )


def save_outputs(
    train_data: Data,
    test_data: Data,
    feature_names: list[str],
    feature_quality: dict[str, Any],
    modality_summary: dict[str, dict[str, float | int]],
    split_blob: dict[str, object],
    patno_mapping: dict[str, int],
    output_dir: Path,
    seed: int,
    label_injection_summary: dict[str, Any] | None = None,
) -> None:
    """Persist datasets + metadata + split manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(train_data, output_dir / "train_data.pt")
    torch.save(test_data, output_dir / "test_data.pt")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "split_method": split_blob["split_method"],
        "split_hash": split_blob["split_hash"],
        "patient_disjoint": True,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "train_size": int(train_data.num_nodes),
        "test_size": int(test_data.num_nodes),
        "n_patients_train": int(split_blob["n_patients_train"]),
        "n_patients_test": int(split_blob["n_patients_test"]),
        "train_events": int(train_data.event.sum().item()),
        "test_events": int(test_data.event.sum().item()),
        "train_event_rate": float(train_data.event.float().mean().item()),
        "test_event_rate": float(test_data.event.float().mean().item()),
        "train_saa_positive": int(train_data.saa_label.sum().item()),
        "test_saa_positive": int(test_data.saa_label.sum().item()),
        "train_saa_rate": float(train_data.saa_label.float().mean().item()),
        "test_saa_rate": float(test_data.saa_label.float().mean().item()),
        "path_contract": {
            "survival_time_key": "time",
            "survival_event_key": "event",
            "classification_label_key": "saa_label",
            "patient_key": "patno",
        },
        "feature_quality_summary": {
            "n_constant_train_scaled": feature_quality.get(
                "n_constant_train_scaled", 0
            ),
            "missing_fraction_mean": feature_quality.get("missing_fraction_mean", 0.0),
            "missing_fraction_max": feature_quality.get("missing_fraction_max", 0.0),
            "imputation": feature_quality.get("imputation", {}),
        },
        "modality_presence_summary": modality_summary,
        "label_injection_summary": label_injection_summary or {},
    }

    (output_dir / "pyg_data_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_blob, indent=2), encoding="utf-8"
    )
    (output_dir / "patno_mapping.json").write_text(
        json.dumps(patno_mapping, indent=2), encoding="utf-8"
    )
    (output_dir / "feature_quality_report.json").write_text(
        json.dumps(feature_quality, indent=2), encoding="utf-8"
    )

    print(f"✓ Saved artifacts to: {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Prepare canonical final PyG data")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=project_root
        / "data"
        / "03_prodromal"
        / "final_training_dataset"
        / "unified_longitudinal_early_pd.csv",
        help="Input unified longitudinal CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "03_prodromal" / "final_pyg_data",
        help="Output directory for train/test .pt and metadata",
    )
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Random seed")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Patient-level test split fraction",
    )
    parser.add_argument("--knn-k", type=int, default=10, help="k for kNN graph")
    parser.add_argument(
        "--allow-proxy-saa-label",
        action="store_true",
        help="Allow saa_label to resolve from proxy/event columns (not recommended)",
    )
    parser.add_argument(
        "--allow-identical-event-saa",
        action="store_true",
        help="Allow saa_label to be identical to event (not recommended)",
    )
    parser.add_argument(
        "--drop-feature-names",
        type=str,
        default="",
        help="Comma-separated feature names to exclude from model inputs",
    )
    parser.add_argument(
        "--drop-feature-file",
        type=Path,
        default=None,
        help="Optional file with one feature name per line to exclude",
    )
    parser.add_argument(
        "--saa-label-csv",
        type=Path,
        default=None,
        help="Optional CSV containing PATNO + real saa_label values",
    )
    parser.add_argument(
        "--drop-unlabeled-saa",
        action="store_true",
        help="Drop rows without real saa_label after --saa-label-csv merge",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare deterministic, leakage-safe canonical PyG train/test datasets."""
    args = parse_args()

    print("=" * 72)
    print("PHASE 8.2: CANONICAL PYG DATA PREPARATION (SOTA HARDENED)")
    print("=" * 72)

    df_raw = load_unified_dataset(args.input_csv)
    label_injection_summary: dict[str, Any] | None = None
    if args.saa_label_csv is not None:
        df_raw, label_injection_summary = inject_saa_labels(
            df_raw,
            args.saa_label_csv,
            drop_unlabeled_saa=args.drop_unlabeled_saa,
        )
    df = normalize_schema(
        df_raw,
        strict_real_saa_label=not args.allow_proxy_saa_label,
        allow_identical_event_saa=args.allow_identical_event_saa,
    )
    df, modality_summary = add_modality_presence_features(df)

    patno_ids, patno_mapping = encode_patno(df)
    time = df["time"].to_numpy(dtype=float)
    event = df["event"].to_numpy(dtype=int)
    saa_label = df["saa_label"].to_numpy(dtype=int)

    train_idx, test_idx, split_blob = split_patient_level(
        patno_ids=patno_ids,
        event=event,
        saa_label=saa_label,
        test_size=args.test_size,
        random_state=args.seed,
    )

    drop_features: set[str] = set()
    if args.drop_feature_names.strip():
        drop_features |= {
            x.strip() for x in args.drop_feature_names.split(",") if x.strip()
        }
    if args.drop_feature_file is not None and args.drop_feature_file.exists():
        drop_features |= {
            line.strip()
            for line in args.drop_feature_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    x_train, x_test, feature_names, feature_quality = prepare_feature_blocks(
        df,
        train_idx,
        test_idx,
        drop_feature_names=drop_features if drop_features else None,
    )

    edge_index_train = construct_knn_graph(x_train, k=args.knn_k)
    edge_index_test = construct_knn_graph(x_test, k=args.knn_k)

    train_data = create_pyg_data(
        x=x_train,
        time=time[train_idx],
        event=event[train_idx],
        saa_label=saa_label[train_idx],
        patno=patno_ids[train_idx],
        source_index=train_idx,
        edge_index=edge_index_train,
        split_id=0,
    )
    test_data = create_pyg_data(
        x=x_test,
        time=time[test_idx],
        event=event[test_idx],
        saa_label=saa_label[test_idx],
        patno=patno_ids[test_idx],
        source_index=test_idx,
        edge_index=edge_index_test,
        split_id=1,
    )

    save_outputs(
        train_data=train_data,
        test_data=test_data,
        feature_names=feature_names,
        feature_quality=feature_quality,
        modality_summary=modality_summary,
        split_blob=split_blob,
        patno_mapping=patno_mapping,
        output_dir=args.output_dir,
        seed=args.seed,
        label_injection_summary=label_injection_summary,
    )

    print("\n" + "=" * 72)
    print("DONE: canonical train/test PyG datasets with split manifest and metadata")
    print("=" * 72)


if __name__ == "__main__":
    main()
