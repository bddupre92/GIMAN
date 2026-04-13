"""Ablation studies for True GIMAN — Full Clinical config (34 features, 7 modalities).

Same ablation suite as baseline but on the full_clinical feature set which
includes motor_cognitive modality (NP3TOT, NP1RTOT, NHY, MCATOT).

Usage:
    python scripts/run_ablation_full_clinical.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.giman_pipeline.training.multi_task_trainer import (
    MultiTaskConfig,
    MultiTaskTrainer,
    build_knn_graph,
    concordance_index,
    cox_partial_likelihood_loss,
)


def load_data():
    """Load full_clinical dataset."""
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


def make_config(modality_dims, cross_modal_heads=None):
    train_config = yaml.safe_load(
        open(project_root / "config/training/true_giman.yaml")
    )
    n_modalities = len(modality_dims)
    embed_dim = train_config["model"]["modality_embed_dim"]
    if cross_modal_heads is None:
        cross_modal_heads = train_config["model"]["cross_modal_heads"]
    fused_dim = train_config["model"]["fused_dim"]
    if cross_modal_heads == 0:
        fused_dim = n_modalities * embed_dim
    return MultiTaskConfig(
        modality_dims=modality_dims,
        modality_embed_dim=embed_dim,
        cross_modal_heads=cross_modal_heads,
        fused_dim=fused_dim,
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
    )


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95):
    rng = np.random.RandomState(42)
    bootstrapped = [
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ]
    bootstrapped = np.array(bootstrapped)
    alpha = (1 - ci) / 2
    return (
        np.mean(values),
        np.percentile(bootstrapped, 100 * alpha),
        np.percentile(bootstrapped, 100 * (1 - alpha)),
    )


def run_giman_cv(features, times, events, modality_dims, config, label, k_folds=5):
    """Run GIMAN CV with given config."""
    trainer = MultiTaskTrainer(config)
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    c_indices = []
    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(np.arange(len(features)), events)
    ):
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
        result = trainer.train_fold(train_data, val_data, fold=fold, verbose=False)
        c_indices.append(result["final_metrics"].get("c_index", -1))
    mean, ci_low, ci_high = bootstrap_ci(np.array(c_indices))
    return {
        "name": label,
        "c_indices": c_indices,
        "mean": float(mean),
        "std": float(np.std(c_indices)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def run_mlp_cv(features, times, events, modality_dims, k_folds=5):
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    c_indices = []
    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(np.arange(len(features)), events)
    ):
        feat_train, feat_val = features[train_idx].copy(), features[val_idx].copy()
        feat_train, feat_val, _, _ = tiered_impute_fold(
            feat_train, feat_val, modality_dims
        )
        model = SimpleMLP(input_dim=feat_train.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        x_train = torch.tensor(feat_train, dtype=torch.float32)
        t_train = torch.tensor(times[train_idx], dtype=torch.float32)
        e_train = torch.tensor(events[train_idx], dtype=torch.long)
        x_val = torch.tensor(feat_val, dtype=torch.float32)
        best_c, patience_counter = -1.0, 0
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            loss = cox_partial_likelihood_loss(model(x_train), t_train, e_train)
            if loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_risk = model(x_val).squeeze(-1).numpy()
            c = concordance_index(val_risk, times[val_idx], events[val_idx])
            if c > best_c:
                best_c = c
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break
        c_indices.append(best_c)
    mean, ci_low, ci_high = bootstrap_ci(np.array(c_indices))
    return {
        "name": "Simple MLP (no graph, no attention)",
        "c_indices": c_indices,
        "mean": float(mean),
        "std": float(np.std(c_indices)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def main():
    start_time = time.time()
    print("\n" + "=" * 70)
    print("ABLATION STUDIES — FULL CLINICAL (34 features, 7 modalities)")
    print("=" * 70 + "\n")

    df, features, times, events, feature_cols, modality_dims = load_data()
    print(f"Dataset: {len(df)} patients, {events.sum()} events ({events.mean():.1%})")
    print(f"Features: {len(feature_cols)} across {len(modality_dims)} modalities")
    print(f"Modalities: {list(modality_dims.keys())}\n")

    results = {}

    # 1. Full model
    print("-" * 50)
    print("1. Full True GIMAN (missingness-aware)")
    print("-" * 50)
    config_full = make_config(modality_dims)
    results["full_model"] = run_giman_cv(
        features,
        times,
        events,
        modality_dims,
        config_full,
        "Full True GIMAN (missingness-aware)",
    )
    print(
        f"  C-index: {results['full_model']['mean']:.4f} ± {results['full_model']['std']:.4f}\n"
    )

    # 2. Simple MLP
    print("-" * 50)
    print("2. Simple MLP (no graph, no attention)")
    print("-" * 50)
    results["simple_mlp"] = run_mlp_cv(features, times, events, modality_dims)
    print(
        f"  C-index: {results['simple_mlp']['mean']:.4f} ± {results['simple_mlp']['std']:.4f}\n"
    )

    # 3. No cross-modal attention
    print("-" * 50)
    print("3. No cross-modal attention")
    print("-" * 50)
    config_no_cm = make_config(modality_dims, cross_modal_heads=0)
    results["no_cross_modal"] = run_giman_cv(
        features,
        times,
        events,
        modality_dims,
        config_no_cm,
        "No cross-modal attention (concat only)",
    )
    print(
        f"  C-index: {results['no_cross_modal']['mean']:.4f} ± {results['no_cross_modal']['std']:.4f}\n"
    )

    # 4. Modality dropout
    print("-" * 50)
    print("4. Modality dropout (mask-based)")
    print("-" * 50)
    col_offset = 0
    modality_col_indices = {}
    for mod_name, dim in modality_dims.items():
        modality_col_indices[mod_name] = list(range(col_offset, col_offset + dim))
        col_offset += dim

    modality_results = []
    config_mod = make_config(modality_dims)
    for dropout_mod in modality_dims:
        print(f"  Dropping: {dropout_mod} ({modality_dims[dropout_mod]} features)")
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        c_indices = []
        trainer = MultiTaskTrainer(config_mod)
        for fold, (train_idx, val_idx) in enumerate(
            kfold.split(np.arange(len(features)), events)
        ):
            feat_train, feat_val = features[train_idx].copy(), features[val_idx].copy()
            feat_train, feat_val, mask_train, mask_val = tiered_impute_fold(
                feat_train, feat_val, modality_dims
            )
            for idx in modality_col_indices[dropout_mod]:
                mask_train[:, idx] = 0.0
                mask_val[:, idx] = 0.0
            k = min(config_mod.graph_k, len(val_idx) - 1)
            train_data = Data(
                x=torch.tensor(feat_train, dtype=torch.float32),
                edge_index=build_knn_graph(
                    feat_train, k=config_mod.graph_k, obs_mask=mask_train
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
            result = trainer.train_fold(train_data, val_data, fold=fold, verbose=False)
            c_indices.append(result["final_metrics"].get("c_index", -1))
        mean, ci_low, ci_high = bootstrap_ci(np.array(c_indices))
        modality_results.append(
            {
                "dropped_modality": dropout_mod,
                "c_indices": c_indices,
                "mean": float(mean),
                "std": float(np.std(c_indices)),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        )
        print(
            f"    → Without {dropout_mod}: C-index = {mean:.4f} ± {np.std(c_indices):.4f}"
        )

    results["modality_dropout"] = modality_results

    # Summary
    elapsed = time.time() - start_time
    full_c = results["full_model"]["mean"]
    print(f"\n{'=' * 70}")
    print("FULL CLINICAL ABLATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Architecture':}")
    print(f"{'Model':<50s} {'C-index':>8s} {'95% CI':>18s} {'Δ':>8s}")
    print("-" * 90)
    for key in ["full_model", "simple_mlp", "no_cross_modal"]:
        r = results[key]
        delta = "" if key == "full_model" else f"{r['mean'] - full_c:>+.4f}"
        ci_str = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        print(f"  {r['name']:<48s} {r['mean']:>8.4f} {ci_str:>18s} {delta:>8s}")
    print(f"\n{'Modality Dropout':}")
    print(f"{'Dropped':<50s} {'C-index':>8s} {'95% CI':>18s} {'Δ':>8s}")
    print("-" * 90)
    for r in modality_results:
        delta = r["mean"] - full_c
        ci_str = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        print(
            f"  Without {r['dropped_modality']:<43s} {r['mean']:>8.4f} {ci_str:>18s} {delta:>+8.4f}"
        )
    if modality_results:
        most_imp = min(modality_results, key=lambda r: r["mean"])
        print(
            f"\n  Most important modality: {most_imp['dropped_modality']} (Δ = {most_imp['mean'] - full_c:+.4f})"
        )
    print(f"  Total time: {elapsed:.0f}s ({elapsed / 60:.1f}min)")

    output_dir = project_root / "results" / "true_giman"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "ablation_results_full_clinical.json"
    with open(results_path, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
            default=lambda x: x.tolist() if hasattr(x, "tolist") else x,
        )
    print(f"  Results saved: {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
