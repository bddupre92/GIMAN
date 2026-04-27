"""
Phase 8.2 Week 2: Prepare Enhanced Training Data

Purpose:
    Prepare PyTorch Geometric Data objects for GIMAN-Prognostic model
    using the 23 high-coverage multimodal features extracted in Week 1.

Input:
    - data/03_prodromal/enhanced/prodromal_multimodal_features.csv
    - data/prodromal_cohort/prodromal_survival_data.csv (time/event labels)

Output:
    - data/03_prodromal/enhanced_training_ready/train_data.pt
    - data/03_prodromal/enhanced_training_ready/val_data.pt
    - data/03_prodromal/enhanced_training_ready/test_data.pt

Author: GIMAN Research Team
Date: October 12, 2025
Phase: 8.2 Week 2
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# List of high-coverage features (≥75%)
HIGH_COVERAGE_FEATURES = [
    "LRRK2", "GBA", "APOE_E4", "SNCA", "GENETIC_RISK_SCORE",
    "UPDRS_I", "UPDRS_II", "PIGD_SCORE", "TREMOR_SCORE",
    "CAUDATE_L_VOL", "CAUDATE_R_VOL", "PUTAMEN_L_VOL", "PUTAMEN_R_VOL", "HIPPOCAMPUS_L_VOL", "HIPPOCAMPUS_R_VOL",
    "ENTORHINAL_L_CTH", "ENTORHINAL_R_CTH", "CINGULATE_L_CTH", "CINGULATE_R_CTH", "PRECENTRAL_L_CTH", "PRECENTRAL_R_CTH",
    "SCOPA_AUT_SCORE", "ESS_SCORE"
]


def load_data(base_dir: Path):
    features_file = base_dir / "data" / "03_prodromal" / "enhanced" / "prodromal_multimodal_features.csv"
    survival_file = base_dir / "data" / "prodromal_cohort" / "prodromal_survival_data.csv"
    print(f"Loading features: {features_file}")
    df = pd.read_csv(features_file)
    print(f"✓ Loaded {df.shape[0]} patients, {df.shape[1]-1} features")
    print(f"Loading survival labels: {survival_file}")
    surv = pd.read_csv(survival_file)
    print(f"✓ Loaded survival labels: {surv.shape[0]} patients")
    # Merge time/event
    if "time_to_event" not in surv.columns and "event_time" in surv.columns:
        surv = surv.rename(columns={"event_time": "time_to_event"})
    if "phenoconverted" not in surv.columns and "event_observed" in surv.columns:
        surv = surv.rename(columns={"event_observed": "phenoconverted"})
    required = {"PATNO", "time_to_event", "phenoconverted"}
    missing = required - set(surv.columns)
    if missing:
        raise ValueError(f"Missing required survival columns: {sorted(missing)}")

    df = df.merge(surv[["PATNO", "time_to_event", "phenoconverted"]], on="PATNO", how="left")
    print(f"✓ Merged shape: {df.shape}")
    return df


def preprocess_features(df: pd.DataFrame):
    # Select high-coverage features
    X = df[HIGH_COVERAGE_FEATURES].copy()
    # Impute missing values (KNN)
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    print(f"✓ Features imputed and standardized: {X_scaled.shape}")
    return X_scaled


def create_graph(df: pd.DataFrame, X_scaled: np.ndarray):
    # Simple kNN graph (k=10)
    from sklearn.neighbors import NearestNeighbors
    k = 10
    nbrs = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)
    edge_index = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # skip self
            edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    print(f"✓ Graph constructed: {edge_index.shape[1]} edges")
    return edge_index


def split_data(df: pd.DataFrame, X_scaled: np.ndarray, edge_index: torch.Tensor):
    # Use same split as Phase 8.1 if available
    # Otherwise, stratified split (event)
    y = df["phenoconverted"].values
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.15, stratify=y, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, stratify=y[test_idx], random_state=42)
    print(f"✓ Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx


def save_pyg_data(df: pd.DataFrame, X_scaled: np.ndarray, edge_index: torch.Tensor, idx, out_file: Path):
    x = torch.tensor(X_scaled[idx], dtype=torch.float32)
    time = torch.tensor(df.iloc[idx]["time_to_event"].values, dtype=torch.float32)
    event = torch.tensor(df.iloc[idx]["phenoconverted"].values, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, time=time, event=event)
    torch.save(data, out_file)
    print(f"✓ Saved: {out_file}")


def main():
    base_dir = Path(__file__).resolve().parents[4]
    df = load_data(base_dir)
    X_scaled = preprocess_features(df)
    edge_index = create_graph(df, X_scaled)
    train_idx, val_idx, test_idx = split_data(df, X_scaled, edge_index)
    out_dir = base_dir / "data" / "03_prodromal" / "enhanced_training_ready"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pyg_data(df, X_scaled, edge_index, train_idx, out_dir / "train_data.pt")
    save_pyg_data(df, X_scaled, edge_index, val_idx, out_dir / "val_data.pt")
    save_pyg_data(df, X_scaled, edge_index, test_idx, out_dir / "test_data.pt")
    print("✓ All training data prepared!")

if __name__ == "__main__":
    main()
