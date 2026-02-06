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

import numpy as np
import pandas as pd
import torch
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

SEED_DEFAULT = 42
SCHEMA_VERSION = "sota_sprint1_v1"

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


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize endpoint and label columns to canonical internal schema."""
    df = df.copy()

    rename_map = {
        resolve_column(df, PATNO_ALIASES, "PATNO"): "PATNO",
        resolve_column(df, TIME_ALIASES, "time"): "time",
        resolve_column(df, EVENT_ALIASES, "event"): "event",
        resolve_column(df, SAA_ALIASES, "saa_label"): "saa_label",
    }
    df = df.rename(columns=rename_map)

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

    print("✓ Normalized canonical schema: PATNO, time, event, saa_label")
    return df


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
    patient_df = pd.DataFrame(
        {"PATNO": patno_ids, "event": event, "saa_label": saa_label}
    ).groupby("PATNO", as_index=False).max()

    # Prefer joint stratification over (event, saa_label) when feasible
    joint_strata = patient_df["event"].astype(int) * 2 + patient_df["saa_label"].astype(int)
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
        raise RuntimeError("Patient leakage detected: overlapping PATNO between train/test")

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
) -> tuple[np.ndarray, np.ndarray, list[str]]:
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
    feature_cols = [c for c in df.columns if c not in exclude_cols]
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

    imputer = KNNImputer(n_neighbors=5)
    x_train_imp = imputer.fit_transform(x_train_raw)
    x_test_imp = imputer.transform(x_test_raw)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_imp)
    x_test = scaler.transform(x_test_imp)

    print(f"✓ Prepared feature blocks: train={x_train.shape}, test={x_test.shape}")
    return x_train, x_test, feature_cols


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
    split_blob: dict[str, object],
    patno_mapping: dict[str, int],
    output_dir: Path,
    seed: int,
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
        "--test-size", type=float, default=0.15, help="Patient-level test split fraction"
    )
    parser.add_argument("--knn-k", type=int, default=10, help="k for kNN graph")
    return parser.parse_args()


def main() -> None:
    """Prepare deterministic, leakage-safe canonical PyG train/test datasets."""
    args = parse_args()

    print("=" * 72)
    print("PHASE 8.2: CANONICAL PYG DATA PREPARATION (SOTA HARDENED)")
    print("=" * 72)

    df_raw = load_unified_dataset(args.input_csv)
    df = normalize_schema(df_raw)

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

    x_train, x_test, feature_names = prepare_feature_blocks(df, train_idx, test_idx)

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
        split_blob=split_blob,
        patno_mapping=patno_mapping,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print("\n" + "=" * 72)
    print("DONE: canonical train/test PyG datasets with split manifest and metadata")
    print("=" * 72)


if __name__ == "__main__":
    main()
