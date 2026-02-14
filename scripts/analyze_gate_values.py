"""Analyze adaptive fusion gate values.

Trains the full adaptive model once and extracts gate values per patient,
stratified by the number of observed modalities.

Usage:
    python scripts/analyze_gate_values.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.giman_pipeline.modeling.true_giman import TrueGIMAN
from src.giman_pipeline.training.multi_task_trainer import (
    MultiTaskConfig,
    MultiTaskTrainer,
    build_knn_graph,
)


def load_data():
    data_path = (
        project_root
        / "data/03_prodromal/final_training_dataset/prodromal_full_clinical.csv"
    )
    config_path = project_root / "config/feature_configs/full_clinical.yaml"
    df = pd.read_csv(data_path)
    config = yaml.safe_load(open(config_path))
    feature_cols = []
    modality_dims = {}
    for mod_name, mod_config in config["modalities"].items():
        modality_dims[mod_name] = mod_config["dim"]
        feature_cols.extend(mod_config["features"])
    features = df[feature_cols].values.astype(np.float32)
    times = df["time_to_event"].values.astype(np.float32)
    events = df["phenoconverted"].values.astype(np.int64)
    return df, features, times, events, feature_cols, modality_dims


def tiered_impute_fold(features_train, features_val, modality_dims):
    obs_mask_train = (~np.isnan(features_train)).astype(np.float32)
    obs_mask_val = (~np.isnan(features_val)).astype(np.float32)
    offset = 0
    tier_3_cols, tier_12_cols = [], []
    for mod_name, dim in modality_dims.items():
        miss_rate = np.isnan(features_train[:, offset : offset + dim]).mean()
        cols = list(range(offset, offset + dim))
        (tier_3_cols if miss_rate > 0.50 else tier_12_cols).extend(cols)
        offset += dim
    for col in tier_3_cols:
        observed = features_train[:, col][~np.isnan(features_train[:, col])]
        fill_val = observed.mean() if len(observed) > 0 else 0.0
        features_train[:, col] = np.where(
            np.isnan(features_train[:, col]), fill_val, features_train[:, col]
        )
        features_val[:, col] = np.where(
            np.isnan(features_val[:, col]), fill_val, features_val[:, col]
        )
    if tier_12_cols and np.isnan(features_train[:, tier_12_cols]).any():
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=10, max_depth=5, random_state=42, n_jobs=-1
            ),
            max_iter=10,
            random_state=42,
            sample_posterior=False,
        )
        features_train[:, tier_12_cols] = imputer.fit_transform(
            features_train[:, tier_12_cols]
        )
        features_val[:, tier_12_cols] = imputer.transform(features_val[:, tier_12_cols])
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_val = scaler.transform(features_val)
    return features_train, features_val, obs_mask_train, obs_mask_val


def main():
    print("\n" + "=" * 60)
    print("GATE VALUE ANALYSIS — Adaptive Cross-Modal Attention")
    print("=" * 60 + "\n")

    df, features, times, events, feature_cols, modality_dims = load_data()
    print(f"Dataset: {len(df)} patients, {events.sum()} events")

    train_config = yaml.safe_load(
        open(project_root / "config/training/true_giman.yaml")
    )
    n_modalities = len(modality_dims)
    embed_dim = train_config["model"]["modality_embed_dim"]

    config = MultiTaskConfig(
        modality_dims=modality_dims,
        modality_embed_dim=embed_dim,
        cross_modal_heads=train_config["model"]["cross_modal_heads"],
        fused_dim=train_config["model"]["fused_dim"],
        gat_hidden_dim=train_config["model"]["gat_hidden_dim"],
        gat_output_dim=train_config["model"]["gat_output_dim"],
        gat_heads=train_config["model"]["gat_heads"],
        gat_layers=train_config["model"]["gat_layers"],
        dropout=train_config["model"]["dropout"],
        num_subtypes=train_config["model"]["num_subtypes"],
        num_diagnostic_classes=train_config["model"]["num_diagnostic_classes"],
        lr=train_config["training"]["lr"],
        weight_decay=float(train_config["training"]["weight_decay"]),
        max_epochs=train_config["training"]["max_epochs"],
        patience=train_config["training"]["patience"],
        grad_clip=train_config["training"]["grad_clip"],
        graph_k=train_config["data"]["graph_k"],
        warmup_survival=0,
        warmup_subtype=0,
        warmup_diagnostic=0,
        adaptive_fusion=True,
        observed_threshold=0.3,
    )

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_gate_data = []

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(np.arange(len(features)), events)
    ):
        print(f"\nFold {fold + 1}/5...")
        feat_train, feat_val = features[train_idx].copy(), features[val_idx].copy()
        feat_train, feat_val, mask_train, mask_val = tiered_impute_fold(
            feat_train, feat_val, modality_dims
        )
        k = min(config.graph_k, len(val_idx) - 1)

        train_data = Data(
            x=torch.tensor(feat_train, dtype=torch.float32),
            edge_index=build_knn_graph(
                feat_train, k=config.graph_k, obs_mask=mask_train
            ),
            time=torch.tensor(times[train_idx], dtype=torch.float32),
            event=torch.tensor(events[train_idx], dtype=torch.long),
            obs_mask=torch.tensor(mask_train, dtype=torch.float32),
        )
        val_data = Data(
            x=torch.tensor(feat_val, dtype=torch.float32),
            edge_index=build_knn_graph(feat_val, k=k, obs_mask=mask_val),
            time=torch.tensor(times[val_idx], dtype=torch.float32),
            event=torch.tensor(events[val_idx], dtype=torch.long),
            obs_mask=torch.tensor(mask_val, dtype=torch.float32),
        )

        trainer = MultiTaskTrainer(config)
        result = trainer.train_fold(train_data, val_data, fold=fold, verbose=False)
        print(f"  C-index: {result['final_metrics'].get('c_index', -1):.4f}")

        # Reconstruct model with best weights to extract gate values
        model = TrueGIMAN(
            modality_dims=modality_dims,
            modality_embed_dim=embed_dim,
            cross_modal_heads=config.cross_modal_heads,
            fused_dim=config.fused_dim,
            gat_hidden_dim=config.gat_hidden_dim,
            gat_output_dim=config.gat_output_dim,
            gat_heads=config.gat_heads,
            gat_layers=config.gat_layers,
            num_subtypes=config.num_subtypes,
            num_diagnostic_classes=config.num_diagnostic_classes,
            dropout=config.dropout,
            adaptive_fusion=True,
            observed_threshold=0.3,
        )
        model.load_state_dict(result["model_state"])
        model.eval()

        with torch.no_grad():
            out = model(
                val_data.x,
                val_data.edge_index,
                obs_mask=val_data.obs_mask,
                return_attention=True,
            )

        gate_vals = out.fusion_gate_values.squeeze().cpu().numpy()

        # Count observed modalities per patient
        for patient_i in range(len(val_idx)):
            n_obs = 0
            col = 0
            for mod_name, dim in modality_dims.items():
                mod_avail = mask_val[patient_i, col : col + dim].mean()
                if mod_avail > 0.3:
                    n_obs += 1
                col += dim
            gate_val = (
                float(gate_vals[patient_i]) if gate_vals.ndim > 0 else float(gate_vals)
            )
            all_gate_data.append(
                {
                    "gate_value": gate_val,
                    "n_modalities": n_obs,
                    "fold": fold + 1,
                    "event": int(events[val_idx[patient_i]]),
                }
            )

    # Analyze
    by_count = defaultdict(list)
    for entry in all_gate_data:
        by_count[entry["n_modalities"]].append(entry["gate_value"])

    print(f"\n{'=' * 60}")
    print("GATE ANALYSIS BY MODALITY COUNT")
    print(f"{'=' * 60}")
    print(
        f"\n{'# Modalities':>14s} {'N':>6s} {'Mean Gate':>11s} {'Std':>8s} {'Min':>8s} {'Max':>8s}"
    )
    print("-" * 60)

    analysis = {}
    for n_mod in sorted(by_count.keys()):
        vals = by_count[n_mod]
        analysis[f"{n_mod}_modalities"] = {
            "n_patients": len(vals),
            "mean_gate": float(np.mean(vals)),
            "std_gate": float(np.std(vals)),
            "min_gate": float(np.min(vals)),
            "max_gate": float(np.max(vals)),
        }
        print(
            f"  {n_mod:>12d} {len(vals):>6d} {np.mean(vals):>11.4f} {np.std(vals):>8.4f} "
            f"{np.min(vals):>8.4f} {np.max(vals):>8.4f}"
        )

    # Event-specific analysis
    by_event = defaultdict(list)
    for entry in all_gate_data:
        by_event[entry["event"]].append(entry["gate_value"])

    print(f"\n{'Gate by Event Status':}")
    print(f"{'Event':>8s} {'N':>6s} {'Mean Gate':>11s} {'Std':>8s}")
    print("-" * 40)
    for event in sorted(by_event.keys()):
        vals = by_event[event]
        label = "Converted" if event == 1 else "Censored"
        print(
            f"  {label:>6s} {len(vals):>6d} {np.mean(vals):>11.4f} {np.std(vals):>8.4f}"
        )

    # Save
    output = {"gate_analysis_by_modality": analysis, "all_gate_data": all_gate_data}
    output_path = project_root / "results/true_giman/gate_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    main()
