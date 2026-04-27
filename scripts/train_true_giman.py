"""Train True GIMAN on clean prodromal PPMI data.

End-to-end pipeline:
1. Load clean prodromal dataset (real PPMI endpoints)
2. MICE imputation (fit on train split only)
3. k-fold cross-validation with per-fold graph construction
4. Report metrics with bootstrap confidence intervals

Usage:
    python scripts/train_true_giman.py [--config baseline|full_clinical]
"""

from __future__ import annotations

import argparse
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
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.giman_pipeline.training.multi_task_trainer import (
    MultiTaskConfig,
    MultiTaskTrainer,
    build_knn_graph,
)


def load_dataset(config_name: str) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Load dataset and feature config.

    Args:
        config_name: "baseline" or "full_clinical"

    Returns:
        (dataframe, feature_columns, modality_dims)
    """
    if config_name == "baseline":
        data_path = (
            project_root
            / "data/03_prodromal/final_training_dataset/prodromal_only_clean.csv"
        )
        config_path = project_root / "config/feature_configs/baseline_36.yaml"
    elif config_name == "full_clinical":
        data_path = (
            project_root
            / "data/03_prodromal/final_training_dataset/prodromal_full_clinical.csv"
        )
        config_path = project_root / "config/feature_configs/full_clinical.yaml"
    else:
        raise ValueError(f"Unknown config: {config_name}")

    df = pd.read_csv(data_path)
    config = yaml.safe_load(open(config_path))

    feature_cols = []
    modality_dims = {}
    for mod_name, mod_config in config["modalities"].items():
        modality_dims[mod_name] = mod_config["dim"]
        feature_cols.extend(mod_config["features"])

    print(f"Dataset: {data_path.name}")
    print(f"Config: {config_path.name}")
    print(
        f"Patients: {len(df)}, Events: {df['phenoconverted'].sum()} ({df['phenoconverted'].mean():.1%})"
    )
    print(f"Features: {len(feature_cols)} across {len(modality_dims)} modalities")
    print(f"Modalities: {modality_dims}")

    return df, feature_cols, modality_dims


def classify_modality_tiers(
    feature_names: list[str],
    modality_dims: dict[str, int],
    features: np.ndarray,
) -> dict[str, int]:
    """Classify each modality into an imputation tier based on missingness rate.

    Tier 1 (<15% missing): MICE imputation — MAR assumption is reasonable.
    Tier 2 (15-50% missing): MICE imputation + missing indicators.
    Tier 3 (>50% missing): No imputation — placeholder fill + learned embeddings.

    Returns:
        Dict mapping modality name -> tier (1, 2, or 3).
    """
    tiers = {}
    offset = 0
    for mod_name, dim in modality_dims.items():
        mod_features = features[:, offset : offset + dim]
        miss_rate = np.isnan(mod_features).mean()
        if miss_rate < 0.15:
            tiers[mod_name] = 1
        elif miss_rate < 0.50:
            tiers[mod_name] = 2
        else:
            tiers[mod_name] = 3
        offset += dim

    return tiers


def impute_features(
    features_train: np.ndarray,
    features_val: np.ndarray,
    feature_names: list[str],
    modality_dims: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tiered imputation with observation mask generation.

    Strategy:
    - Tier 1 (<15% missing): MICE imputation — MAR assumption is reasonable.
    - Tier 2 (15-50% missing): MICE imputation (still works, missingness moderate).
    - Tier 3 (>50% missing): No imputation — fill with column mean as placeholder.
      The model uses learned missing embeddings for these modalities, gated by
      the observation mask. Imputing 69-78% of values would fabricate data.

    Returns:
        (features_train_imp, features_val_imp, obs_mask_train, obs_mask_val)
        - Features are imputed (Tier 1-2) or placeholder-filled (Tier 3), then scaled.
        - obs_mask is binary [N, F]: 1=originally observed, 0=was missing.
    """
    # Record observation masks BEFORE any imputation
    obs_mask_train = (~np.isnan(features_train)).astype(np.float32)
    obs_mask_val = (~np.isnan(features_val)).astype(np.float32)

    # Classify modalities into tiers
    tiers = classify_modality_tiers(feature_names, modality_dims, features_train)

    # Report
    total_missing = np.isnan(features_train).mean()
    print(f"  Overall missingness: {total_missing:.1%}")
    print("  Imputation tiers:")
    offset = 0
    tier_3_cols = []
    tier_12_cols = []
    for mod_name, dim in modality_dims.items():
        mod_miss = np.isnan(features_train[:, offset : offset + dim]).mean()
        tier = tiers[mod_name]
        strategy = {1: "MICE", 2: "MICE", 3: "placeholder (learned embedding)"}[tier]
        print(f"    {mod_name}: {mod_miss:.0%} missing → Tier {tier} ({strategy})")
        cols = list(range(offset, offset + dim))
        if tier == 3:
            tier_3_cols.extend(cols)
        else:
            tier_12_cols.extend(cols)
        offset += dim

    # --- Tier 3: Fill with column mean (placeholder) ---
    # These values won't actually be used by the model because availability ≈ 0
    # triggers the learned missing embedding. The placeholder just prevents NaN errors.
    if tier_3_cols:
        for col in tier_3_cols:
            train_col = features_train[:, col]
            observed = train_col[~np.isnan(train_col)]
            fill_val = observed.mean() if len(observed) > 0 else 0.0
            features_train[:, col] = np.where(np.isnan(train_col), fill_val, train_col)
            features_val[:, col] = np.where(
                np.isnan(features_val[:, col]), fill_val, features_val[:, col]
            )

    # --- Tier 1-2: MICE imputation on remaining features ---
    if tier_12_cols:
        mice_train = features_train[:, tier_12_cols]
        mice_val = features_val[:, tier_12_cols]
        mice_miss = np.isnan(mice_train).mean()
        print(
            f"  MICE imputation on {len(tier_12_cols)} Tier 1-2 features ({mice_miss:.1%} missing)"
        )

        imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=10, max_depth=5, random_state=42, n_jobs=-1
            ),
            max_iter=10,
            random_state=42,
            sample_posterior=False,
        )
        features_train[:, tier_12_cols] = imputer.fit_transform(mice_train)
        features_val[:, tier_12_cols] = imputer.transform(mice_val)

    # Standardize all features after imputation/placeholder fill
    scaler = StandardScaler()
    features_train_imp = scaler.fit_transform(features_train)
    features_val_imp = scaler.transform(features_val)

    return features_train_imp, features_val_imp, obs_mask_train, obs_mask_val


def bootstrap_ci(
    values: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval."""
    rng = np.random.RandomState(42)
    bootstrapped = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrapped.append(np.mean(sample))
    bootstrapped = np.array(bootstrapped)
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrapped, 100 * alpha)
    upper = np.percentile(bootstrapped, 100 * (1 - alpha))
    return np.mean(values), lower, upper


def train_model(config_name: str, k_folds: int = 5) -> dict:
    """Train True GIMAN with k-fold cross-validation."""
    start_time = time.time()

    # Load data
    df, feature_cols, modality_dims = load_dataset(config_name)

    # Extract arrays
    features = df[feature_cols].values.astype(np.float32)
    times = df["time_to_event"].values.astype(np.float32)
    events = df["phenoconverted"].values.astype(np.int64)

    # Load training config
    train_config = yaml.safe_load(
        open(project_root / "config/training/true_giman.yaml")
    )

    # Create trainer config
    trainer_config = MultiTaskConfig(
        modality_dims=modality_dims,
        modality_embed_dim=train_config["model"]["modality_embed_dim"],
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
        warmup_diagnostic=train_config["multi_task"]["warmup_diagnostic"],
        warmup_subtype=train_config["multi_task"]["warmup_subtype"],
        warmup_survival=train_config["multi_task"]["warmup_survival"],
        survival_weight=train_config["multi_task"]["survival_weight"],
        subtype_weight=train_config["multi_task"]["subtype_weight"],
        diagnostic_weight=train_config["multi_task"]["diagnostic_weight"],
    )

    # Run cross-validation with per-fold imputation
    from sklearn.model_selection import StratifiedKFold
    from torch_geometric.data import Data

    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    trainer = MultiTaskTrainer(trainer_config)

    fold_results = []
    all_c_indices = []

    print(f"\n{'=' * 60}")
    print(f"TRUE GIMAN: {k_folds}-FOLD CROSS-VALIDATION")
    print(f"Config: {config_name}")
    print(f"{'=' * 60}\n")

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(np.arange(len(df)), events)
    ):
        print(f"\nFold {fold + 1}/{k_folds}")
        print("-" * 40)

        # Per-fold tiered imputation (fit on train only)
        # Returns features + observation masks for missingness-aware model
        features_train_fold = features[train_idx].copy()
        features_val_fold = features[val_idx].copy()
        features_train_imp, features_val_imp, obs_mask_train, obs_mask_val = (
            impute_features(
                features_train_fold, features_val_fold, feature_cols, modality_dims
            )
        )

        # Per-fold graph construction (availability-aware when masks available)
        k = min(trainer_config.graph_k, len(val_idx) - 1)
        train_edge_index = build_knn_graph(
            features_train_imp, k=trainer_config.graph_k, obs_mask=obs_mask_train
        )
        val_edge_index = build_knn_graph(features_val_imp, k=k, obs_mask=obs_mask_val)

        # Create PyG Data objects with observation masks
        train_data = Data(
            x=torch.tensor(features_train_imp, dtype=torch.float32),
            edge_index=train_edge_index,
            time=torch.tensor(times[train_idx], dtype=torch.float32),
            event=torch.tensor(events[train_idx], dtype=torch.long),
            obs_mask=torch.tensor(obs_mask_train, dtype=torch.float32),
        )
        val_data = Data(
            x=torch.tensor(features_val_imp, dtype=torch.float32),
            edge_index=val_edge_index,
            time=torch.tensor(times[val_idx], dtype=torch.float32),
            event=torch.tensor(events[val_idx], dtype=torch.long),
            obs_mask=torch.tensor(obs_mask_val, dtype=torch.float32),
        )

        print(f"  Train: {len(train_idx)} samples, {events[train_idx].sum()} events")
        print(f"  Val: {len(val_idx)} samples, {events[val_idx].sum()} events")

        result = trainer.train_fold(train_data, val_data, fold=fold)
        fold_results.append(result)

        c_idx = result["final_metrics"].get("c_index", -1)
        all_c_indices.append(c_idx)

        metrics_str = ", ".join(
            f"{k}={v:.4f}" for k, v in result["final_metrics"].items()
        )
        print(f"  Fold {fold + 1} final: {metrics_str}")

    # Summary with CIs
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'=' * 60}")

    all_metrics: dict[str, list[float]] = {}
    for r in fold_results:
        for k, v in r["final_metrics"].items():
            all_metrics.setdefault(k, []).append(v)

    summary = {}
    for metric, values in all_metrics.items():
        arr = np.array(values)
        mean, ci_low, ci_high = bootstrap_ci(arr)
        summary[metric] = {
            "mean": mean,
            "std": arr.std(),
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
        print(
            f"  {metric}: {mean:.4f} +/- {arr.std():.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}])"
        )

    print(f"\n  Training time: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"  Config: {config_name}")
    print(f"  Patients: {len(df)}")
    print(f"  Events: {events.sum()} ({events.mean():.1%})")
    print(f"  Features: {len(feature_cols)}")
    print("  Endpoint source: 100% PPMI Primary Clinical Diagnosis")

    # Save results
    output_dir = project_root / "results" / "true_giman"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "config": config_name,
        "n_patients": len(df),
        "n_events": int(events.sum()),
        "event_rate": float(events.mean()),
        "n_features": len(feature_cols),
        "modalities": modality_dims,
        "k_folds": k_folds,
        "summary_metrics": {
            k: {kk: float(vv) for kk, vv in v.items()} for k, v in summary.items()
        },
        "per_fold": [
            {
                "fold": r["fold"],
                "epochs_trained": r["epochs_trained"],
                "best_primary_metric": float(r["best_primary_metric"]),
                "final_metrics": {k: float(v) for k, v in r["final_metrics"].items()},
            }
            for r in fold_results
        ],
        "training_time_seconds": elapsed,
        "endpoint_source": "ppmi_primary_clinical_diagnosis",
        "imputation_strategy": "tiered (MICE <50% miss, learned embeddings >50% miss)",
        "missingness_aware": True,
    }

    results_path = output_dir / f"cv_results_{config_name}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Train True GIMAN")
    parser.add_argument(
        "--config",
        choices=["baseline", "full_clinical", "both"],
        default="both",
        help="Feature configuration to use",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    configs = ["baseline", "full_clinical"] if args.config == "both" else [args.config]

    for config_name in configs:
        print(f"\n\n{'#' * 60}")
        print(f"# TRAINING: {config_name.upper()}")
        print(f"{'#' * 60}\n")
        train_model(config_name, k_folds=args.folds)

    return 0


if __name__ == "__main__":
    sys.exit(main())
