"""Investigate why graph attention and cross-modal attention don't outperform MLP.

Hypotheses tested:
1. OVERPARAMETERIZATION: 331K params for 188 events (1760:1 ratio) — the complex
   model may be overfitting where MLP generalizes better. Test reduced capacity.
2. GRAPH QUALITY: k-NN on 30 features with 41% missing may produce poor graphs.
   Test different k values and graph construction strategies.
3. GAT DEPTH: 3-layer GAT may over-smooth embeddings. Test 1-layer GAT.
4. CROSS-MODAL OVERHEAD: 6 modalities with 3 having ~70% missing means attention
   operates on mostly "missing" embeddings. Test with clinical-only modalities.
5. EARLY STOPPING: Models stop at 22-39 epochs — complex models may need longer.

Usage:
    python scripts/investigate_architecture_gap.py
"""

from __future__ import annotations

import json
import sys
import time
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

from src.giman_pipeline.training.multi_task_trainer import (
    MultiTaskConfig,
    MultiTaskTrainer,
    build_knn_graph,
)


def load_data():
    data_path = (
        project_root
        / "data/03_prodromal/final_training_dataset/prodromal_only_clean.csv"
    )
    config_path = project_root / "config/feature_configs/baseline_36.yaml"
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


def run_experiment(
    features, times, events, modality_dims, config_overrides, label, k_folds=5
):
    """Run a single experiment with config overrides."""
    train_config = yaml.safe_load(
        open(project_root / "config/training/true_giman.yaml")
    )

    n_modalities = len(modality_dims)
    embed_dim = config_overrides.get(
        "modality_embed_dim", train_config["model"]["modality_embed_dim"]
    )
    cross_modal_heads = config_overrides.get(
        "cross_modal_heads", train_config["model"]["cross_modal_heads"]
    )
    fused_dim = config_overrides.get("fused_dim", train_config["model"]["fused_dim"])
    if cross_modal_heads == 0:
        fused_dim = n_modalities * embed_dim

    config = MultiTaskConfig(
        modality_dims=modality_dims,
        modality_embed_dim=embed_dim,
        cross_modal_heads=cross_modal_heads,
        fused_dim=fused_dim,
        gat_hidden_dim=config_overrides.get(
            "gat_hidden_dim", train_config["model"]["gat_hidden_dim"]
        ),
        gat_output_dim=config_overrides.get(
            "gat_output_dim", train_config["model"]["gat_output_dim"]
        ),
        gat_heads=config_overrides.get("gat_heads", train_config["model"]["gat_heads"]),
        gat_layers=config_overrides.get(
            "gat_layers", train_config["model"]["gat_layers"]
        ),
        dropout=config_overrides.get("dropout", train_config["model"]["dropout"]),
        num_subtypes=train_config["model"]["num_subtypes"],
        num_diagnostic_classes=train_config["model"]["num_diagnostic_classes"],
        lr=config_overrides.get("lr", train_config["training"]["lr"]),
        weight_decay=float(
            config_overrides.get(
                "weight_decay", train_config["training"]["weight_decay"]
            )
        ),
        max_epochs=config_overrides.get(
            "max_epochs", train_config["training"]["max_epochs"]
        ),
        patience=config_overrides.get("patience", train_config["training"]["patience"]),
        grad_clip=train_config["training"]["grad_clip"],
        graph_k=config_overrides.get("graph_k", train_config["data"]["graph_k"]),
        warmup_survival=0,
        warmup_subtype=0,
        warmup_diagnostic=0,
    )

    trainer = MultiTaskTrainer(config)
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    c_indices = []
    epochs_list = []

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
        epochs_list.append(result["epochs_trained"])

    mean, ci_low, ci_high = bootstrap_ci(np.array(c_indices))

    # Count params
    model = trainer._create_model()
    total_params = sum(p.numel() for p in model.parameters())

    return {
        "label": label,
        "c_indices": c_indices,
        "mean": float(mean),
        "std": float(np.std(c_indices)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "avg_epochs": float(np.mean(epochs_list)),
        "total_params": total_params,
    }


def main():
    start_time = time.time()

    print("\n" + "=" * 70)
    print("ARCHITECTURE INVESTIGATION: Why doesn't GAT outperform MLP?")
    print("=" * 70 + "\n")

    df, features, times, events, feature_cols, modality_dims = load_data()
    print(f"Dataset: {len(df)} patients, {events.sum()} events ({events.mean():.1%})")
    print(f"Features: {len(feature_cols)} across {len(modality_dims)} modalities")
    print(
        f"Events-to-features ratio: {events.sum()}/{len(feature_cols)} = {events.sum() / len(feature_cols):.1f}"
    )
    print()

    results = []

    # ---- Reference: current full model (64 embed, 4 heads, 3 layers) ----
    print("-" * 60)
    print("H0: Reference — Full model (current config)")
    print("-" * 60)
    r = run_experiment(
        features, times, events, modality_dims, {}, "Reference: Full model"
    )
    results.append(r)
    print(
        f"  C-index: {r['mean']:.4f} ± {r['std']:.4f}, params: {r['total_params']:,}, avg epochs: {r['avg_epochs']:.0f}"
    )

    # ---- H1: OVERPARAMETERIZATION — reduce model capacity ----
    print("\n" + "-" * 60)
    print("H1: Reduced capacity (32 embed, 2 heads, 2 GAT layers)")
    print("-" * 60)
    r = run_experiment(
        features,
        times,
        events,
        modality_dims,
        {
            "modality_embed_dim": 32,
            "cross_modal_heads": 2,
            "fused_dim": 64,
            "gat_hidden_dim": 32,
            "gat_output_dim": 32,
            "gat_heads": 2,
            "gat_layers": 2,
        },
        "H1: Reduced capacity GIMAN",
    )
    results.append(r)
    print(
        f"  C-index: {r['mean']:.4f} ± {r['std']:.4f}, params: {r['total_params']:,}, avg epochs: {r['avg_epochs']:.0f}"
    )

    # ---- H1b: Minimal capacity (16 embed, 2 heads, 1 GAT layer) ----
    print("\n" + "-" * 60)
    print("H1b: Minimal capacity (16 embed, 2 heads, 1 GAT layer)")
    print("-" * 60)
    r = run_experiment(
        features,
        times,
        events,
        modality_dims,
        {
            "modality_embed_dim": 16,
            "cross_modal_heads": 2,
            "fused_dim": 32,
            "gat_hidden_dim": 16,
            "gat_output_dim": 16,
            "gat_heads": 2,
            "gat_layers": 1,
        },
        "H1b: Minimal capacity GIMAN",
    )
    results.append(r)
    print(
        f"  C-index: {r['mean']:.4f} ± {r['std']:.4f}, params: {r['total_params']:,}, avg epochs: {r['avg_epochs']:.0f}"
    )

    # ---- H2: GRAPH QUALITY — test k=5 and k=20 ----
    print("\n" + "-" * 60)
    print("H2a: Sparser graph (k=5)")
    print("-" * 60)
    r = run_experiment(
        features, times, events, modality_dims, {"graph_k": 5}, "H2a: k=5 graph"
    )
    results.append(r)
    print(
        f"  C-index: {r['mean']:.4f} ± {r['std']:.4f}, avg epochs: {r['avg_epochs']:.0f}"
    )

    print("\n" + "-" * 60)
    print("H2b: Denser graph (k=20)")
    print("-" * 60)
    r = run_experiment(
        features, times, events, modality_dims, {"graph_k": 20}, "H2b: k=20 graph"
    )
    results.append(r)
    print(
        f"  C-index: {r['mean']:.4f} ± {r['std']:.4f}, avg epochs: {r['avg_epochs']:.0f}"
    )

    # ---- H3: GAT DEPTH — 1-layer GAT (less over-smoothing) ----
    print("\n" + "-" * 60)
    print("H3: Shallow GAT (1 layer instead of 3)")
    print("-" * 60)
    r = run_experiment(
        features, times, events, modality_dims, {"gat_layers": 1}, "H3: 1-layer GAT"
    )
    results.append(r)
    print(
        f"  C-index: {r['mean']:.4f} ± {r['std']:.4f}, params: {r['total_params']:,}, avg epochs: {r['avg_epochs']:.0f}"
    )

    # ---- H4: Higher regularization (more dropout) ----
    print("\n" + "-" * 60)
    print("H4: Higher dropout (0.5 instead of 0.3)")
    print("-" * 60)
    r = run_experiment(
        features, times, events, modality_dims, {"dropout": 0.5}, "H4: Dropout 0.5"
    )
    results.append(r)
    print(
        f"  C-index: {r['mean']:.4f} ± {r['std']:.4f}, avg epochs: {r['avg_epochs']:.0f}"
    )

    # ---- H5: More patience / lower LR ----
    print("\n" + "-" * 60)
    print("H5: Lower LR (0.0003) + more patience (40)")
    print("-" * 60)
    r = run_experiment(
        features,
        times,
        events,
        modality_dims,
        {
            "lr": 0.0003,
            "patience": 40,
            "max_epochs": 400,
        },
        "H5: Lower LR + patience",
    )
    results.append(r)
    print(
        f"  C-index: {r['mean']:.4f} ± {r['std']:.4f}, avg epochs: {r['avg_epochs']:.0f}"
    )

    # ---- H6: "Sweet spot" — reduced capacity + 1-layer GAT + more dropout ----
    print("\n" + "-" * 60)
    print("H6: Sweet spot (32 embed, 2 heads, 1-layer GAT, dropout 0.4)")
    print("-" * 60)
    r = run_experiment(
        features,
        times,
        events,
        modality_dims,
        {
            "modality_embed_dim": 32,
            "cross_modal_heads": 2,
            "fused_dim": 64,
            "gat_hidden_dim": 32,
            "gat_output_dim": 32,
            "gat_heads": 2,
            "gat_layers": 1,
            "dropout": 0.4,
        },
        "H6: Sweet spot config",
    )
    results.append(r)
    print(
        f"  C-index: {r['mean']:.4f} ± {r['std']:.4f}, params: {r['total_params']:,}, avg epochs: {r['avg_epochs']:.0f}"
    )

    # ---- Summary ----
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print("INVESTIGATION SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"\n{'Experiment':<45s} {'C-index':>8s} {'± std':>8s} {'Params':>10s} {'Epochs':>8s}"
    )
    print("-" * 85)

    # Sort by mean C-index descending
    results_sorted = sorted(results, key=lambda r: r["mean"], reverse=True)
    ref_c = results[0]["mean"]
    for r in results_sorted:
        delta = r["mean"] - ref_c
        delta_str = f"({delta:+.4f})" if r["label"] != "Reference: Full model" else ""
        params_str = f"{r['total_params']:,}" if "total_params" in r else "N/A"
        print(
            f"  {r['label']:<43s} {r['mean']:>8.4f} {r['std']:>7.4f} {params_str:>10s} {r['avg_epochs']:>7.0f}  {delta_str}"
        )

    print(f"\n  Total time: {elapsed:.0f}s ({elapsed / 60:.1f}min)")

    # Key insight
    best = results_sorted[0]
    print(f"\n  Best config: {best['label']}")
    print(
        f"  C-index: {best['mean']:.4f} (95% CI: [{best['ci_low']:.4f}, {best['ci_high']:.4f}])"
    )

    if best["label"] != "Reference: Full model":
        improvement = best["mean"] - ref_c
        print(f"  Improvement over reference: {improvement:+.4f}")

    # Save
    output_dir = project_root / "results" / "true_giman"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "architecture_investigation.json"
    with open(results_path, "w") as f:
        json.dump({"experiments": results, "total_time_seconds": elapsed}, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
