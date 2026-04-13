"""Ablation studies for True GIMAN (missingness-aware version).

Tests contribution of key architectural components:
1. Full model (missingness-aware, tiered imputation)
2. No cross-modal attention (replace with simple concatenation)
3. No graph attention (replace GATConv with simple MLP)
4. Modality dropout via observation mask zeroing

Modality dropout now works cleanly via the missingness-aware architecture:
- Set obs_mask to 0 for the dropped modality → learned missing embedding is used
- No imputation artifacts, no tensor shape issues
- The model naturally handles "missing" modalities via availability gating

Usage:
    python scripts/run_ablation_studies.py
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
    """Load baseline dataset with modality feature mapping."""
    data_path = (
        project_root
        / "data/03_prodromal/final_training_dataset/prodromal_only_clean.csv"
    )
    config_path = project_root / "config/feature_configs/baseline_36.yaml"

    df = pd.read_csv(data_path)
    config = yaml.safe_load(open(config_path))

    feature_cols = []
    modality_dims = {}
    modality_features = {}
    for mod_name, mod_config in config["modalities"].items():
        modality_dims[mod_name] = mod_config["dim"]
        modality_features[mod_name] = mod_config["features"]
        feature_cols.extend(mod_config["features"])

    features = df[feature_cols].values.astype(np.float32)
    times = df["time_to_event"].values.astype(np.float32)
    events = df["phenoconverted"].values.astype(np.int64)

    return df, features, times, events, feature_cols, modality_dims, modality_features


def tiered_impute_fold(features_train, features_val, modality_dims):
    """Tiered imputation for a single fold. Returns features + obs_masks."""
    # Record observation masks BEFORE imputation
    obs_mask_train = (~np.isnan(features_train)).astype(np.float32)
    obs_mask_val = (~np.isnan(features_val)).astype(np.float32)

    # Classify tiers
    offset = 0
    tier_3_cols = []
    tier_12_cols = []
    for mod_name, dim in modality_dims.items():
        miss_rate = np.isnan(features_train[:, offset : offset + dim]).mean()
        cols = list(range(offset, offset + dim))
        if miss_rate > 0.50:
            tier_3_cols.extend(cols)
        else:
            tier_12_cols.extend(cols)
        offset += dim

    # Tier 3: placeholder fill
    for col in tier_3_cols:
        observed = features_train[:, col][~np.isnan(features_train[:, col])]
        fill_val = observed.mean() if len(observed) > 0 else 0.0
        features_train[:, col] = np.where(
            np.isnan(features_train[:, col]), fill_val, features_train[:, col]
        )
        features_val[:, col] = np.where(
            np.isnan(features_val[:, col]), fill_val, features_val[:, col]
        )

    # Tier 1-2: MICE
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
    """Create trainer config from YAML with optional overrides."""
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


# -----------------------------------------------------------------------
# Ablation: Full model (missingness-aware, reference)
# -----------------------------------------------------------------------


def train_full_model(features, times, events, modality_dims, k_folds=5) -> dict:
    """Train full True GIMAN with missingness-aware tiered imputation."""
    config = make_config(modality_dims)
    trainer = MultiTaskTrainer(config)
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    c_indices = []

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(np.arange(len(features)), events)
    ):
        feat_train = features[train_idx].copy()
        feat_val = features[val_idx].copy()
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
        "name": "Full True GIMAN (missingness-aware)",
        "c_indices": c_indices,
        "mean": float(mean),
        "std": float(np.std(c_indices)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


# -----------------------------------------------------------------------
# Ablation: Simple MLP (no graph, no attention)
# -----------------------------------------------------------------------


class SimpleMLP(nn.Module):
    """Flat MLP baseline — no graph structure, no cross-modal attention."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
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


def train_simple_mlp(features, times, events, modality_dims, k_folds=5) -> dict:
    """Train simple MLP baseline with tiered imputation."""
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    c_indices = []

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(np.arange(len(features)), events)
    ):
        feat_train = features[train_idx].copy()
        feat_val = features[val_idx].copy()
        feat_train, feat_val, _, _ = tiered_impute_fold(
            feat_train, feat_val, modality_dims
        )

        model = SimpleMLP(input_dim=feat_train.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

        x_train = torch.tensor(feat_train, dtype=torch.float32)
        t_train = torch.tensor(times[train_idx], dtype=torch.float32)
        e_train = torch.tensor(events[train_idx], dtype=torch.long)
        x_val = torch.tensor(feat_val, dtype=torch.float32)

        best_c = -1.0
        patience_counter = 0
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            risk = model(x_train)
            loss = cox_partial_likelihood_loss(risk, t_train, e_train)
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


# -----------------------------------------------------------------------
# Ablation: No cross-modal attention
# -----------------------------------------------------------------------


def train_no_cross_modal(features, times, events, modality_dims, k_folds=5) -> dict:
    """Train True GIMAN with cross-modal attention disabled."""
    config = make_config(modality_dims, cross_modal_heads=0)
    trainer = MultiTaskTrainer(config)
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    c_indices = []

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(np.arange(len(features)), events)
    ):
        feat_train = features[train_idx].copy()
        feat_val = features[val_idx].copy()
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
        "name": "No cross-modal attention (concat only)",
        "c_indices": c_indices,
        "mean": float(mean),
        "std": float(np.std(c_indices)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


# -----------------------------------------------------------------------
# Modality dropout via observation mask zeroing
# -----------------------------------------------------------------------


def train_modality_dropout(
    features, times, events, modality_dims, k_folds=5
) -> list[dict]:
    """Train with each modality masked out via observation mask.

    Instead of zeroing features (which creates artifacts), we set the
    observation mask to 0 for the dropped modality. This triggers:
    1. Learned missing embedding is used instead of encoder output
    2. Cross-modal attention gates this modality to near-zero weight
    3. Graph construction ignores these features for similarity

    This is the correct way to test modality importance in a
    missingness-aware architecture.
    """
    results = []

    # Build modality-to-column index mapping
    col_offset = 0
    modality_col_indices = {}
    for mod_name, dim in modality_dims.items():
        modality_col_indices[mod_name] = list(range(col_offset, col_offset + dim))
        col_offset += dim

    config = make_config(modality_dims)

    for dropout_mod in modality_dims:
        print(
            f"  Dropping modality: {dropout_mod} ({modality_dims[dropout_mod]} features)"
        )
        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        c_indices = []
        trainer = MultiTaskTrainer(config)

        for fold, (train_idx, val_idx) in enumerate(
            kfold.split(np.arange(len(features)), events)
        ):
            feat_train = features[train_idx].copy()
            feat_val = features[val_idx].copy()
            feat_train, feat_val, mask_train, mask_val = tiered_impute_fold(
                feat_train, feat_val, modality_dims
            )

            # Zero out observation mask for the dropped modality
            # This triggers learned missing embeddings + availability gating
            for idx in modality_col_indices[dropout_mod]:
                mask_train[:, idx] = 0.0
                mask_val[:, idx] = 0.0

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
        results.append(
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

    return results


def main() -> int:
    start_time = time.time()

    print("\n" + "=" * 70)
    print("TRUE GIMAN ABLATION STUDIES (Missingness-Aware)")
    print("=" * 70 + "\n")

    df, features, times, events, feature_cols, modality_dims, modality_features = (
        load_data()
    )

    print(f"Dataset: {len(df)} patients, {events.sum()} events ({events.mean():.1%})")
    print(f"Features: {len(feature_cols)} across {len(modality_dims)} modalities\n")

    results = {}

    # 1. Full model (missingness-aware)
    print("-" * 50)
    print("Ablation 1: Full True GIMAN (missingness-aware)")
    print("-" * 50)
    results["full_model"] = train_full_model(features, times, events, modality_dims)
    print(
        f"  C-index: {results['full_model']['mean']:.4f} ± {results['full_model']['std']:.4f}"
        f" (95% CI: [{results['full_model']['ci_low']:.4f}, {results['full_model']['ci_high']:.4f}])\n"
    )

    # 2. Simple MLP (no graph, no attention)
    print("-" * 50)
    print("Ablation 2: Simple MLP (no graph, no attention)")
    print("-" * 50)
    results["simple_mlp"] = train_simple_mlp(features, times, events, modality_dims)
    print(
        f"  C-index: {results['simple_mlp']['mean']:.4f} ± {results['simple_mlp']['std']:.4f}"
        f" (95% CI: [{results['simple_mlp']['ci_low']:.4f}, {results['simple_mlp']['ci_high']:.4f}])\n"
    )

    # 3. No cross-modal attention
    print("-" * 50)
    print("Ablation 3: No cross-modal attention")
    print("-" * 50)
    results["no_cross_modal"] = train_no_cross_modal(
        features, times, events, modality_dims
    )
    print(
        f"  C-index: {results['no_cross_modal']['mean']:.4f} ± {results['no_cross_modal']['std']:.4f}"
        f" (95% CI: [{results['no_cross_modal']['ci_low']:.4f}, {results['no_cross_modal']['ci_high']:.4f}])\n"
    )

    # 4. Modality dropout (mask-based)
    print("-" * 50)
    print("Ablation 4: Modality dropout (observation mask zeroing)")
    print("-" * 50)
    modality_results = train_modality_dropout(features, times, events, modality_dims)
    results["modality_dropout"] = modality_results

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print("ABLATION SUMMARY")
    print(f"{'=' * 70}")

    full_c = results["full_model"]["mean"]
    print(f"\n{'Architecture Ablations':}")
    print(f"{'Model':<50s} {'C-index':>8s} {'95% CI':>18s} {'Δ':>8s}")
    print("-" * 90)

    for key in ["full_model", "simple_mlp", "no_cross_modal"]:
        r = results[key]
        delta = "" if key == "full_model" else f"{r['mean'] - full_c:>+.4f}"
        ci_str = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        print(f"  {r['name']:<48s} {r['mean']:>8.4f} {ci_str:>18s} {delta:>8s}")

    print(f"\n{'Modality Dropout (mask-based)':}")
    print(f"{'Dropped Modality':<50s} {'C-index':>8s} {'95% CI':>18s} {'Δ':>8s}")
    print("-" * 90)
    for r in modality_results:
        delta = r["mean"] - full_c
        ci_str = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        print(
            f"  Without {r['dropped_modality']:<43s} {r['mean']:>8.4f} {ci_str:>18s} {delta:>+8.4f}"
        )

    print(f"\n  Total time: {elapsed:.0f}s ({elapsed / 60:.1f}min)")

    # Most important modality
    if modality_results:
        most_important = min(modality_results, key=lambda r: r["mean"])
        least_important = max(modality_results, key=lambda r: r["mean"])
        print(
            f"\n  Most important modality: {most_important['dropped_modality']} "
            f"(dropping it reduces C-index by {full_c - most_important['mean']:.4f})"
        )
        print(
            f"  Least important modality: {least_important['dropped_modality']} "
            f"(dropping it changes C-index by {least_important['mean'] - full_c:+.4f})"
        )

    # Save
    output_dir = project_root / "results" / "true_giman"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
            default=lambda x: x.tolist() if hasattr(x, "tolist") else x,
        )
    print(f"\n  Results saved: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
