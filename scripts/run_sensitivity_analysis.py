"""Sensitivity analysis: Quantify imputation's contribution to performance.

Compares three training configurations to understand how much of the model's
performance comes from real signal vs. imputation artifacts:

1. **Full model (missingness-aware)**: All 1,985 patients, all 30 features,
   tiered imputation + observation masks + learned missing embeddings.
   This is the primary model.

2. **Clinical-only model**: Only clinical + genetic + clinical_biomarkers
   modalities (~14 features, >85% complete). Drops imaging, CSF, cortical
   thickness entirely. Tests: how much do the high-missingness modalities
   actually contribute?

3. **Complete-case model**: Only patients who have ALL modalities observed
   (imaging + CSF + cortical thickness present). Much smaller N but no
   imputation at all. Tests: is the full signal there in complete cases?

If clinical-only ≈ full model, imaging/CSF aren't contributing meaningful
signal beyond what the model learns from their missingness pattern.

Usage:
    python scripts/run_sensitivity_analysis.py
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


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95):
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


def load_baseline_data():
    """Load baseline dataset and feature config."""
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

    return df, feature_cols, modality_dims, modality_features


def make_trainer_config(modality_dims: dict, train_config: dict) -> MultiTaskConfig:
    """Create trainer config from modality dims and training YAML."""
    return MultiTaskConfig(
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


def run_cv(
    features: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    trainer_config: MultiTaskConfig,
    obs_mask: np.ndarray | None = None,
    k_folds: int = 5,
    label: str = "",
) -> dict:
    """Run k-fold CV and return summary."""
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    trainer = MultiTaskTrainer(trainer_config)
    all_c_indices = []

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(np.arange(len(features)), events)
    ):
        # Copy to avoid mutation
        feat_train = features[train_idx].copy()
        feat_val = features[val_idx].copy()

        # Record observation masks before imputation
        if obs_mask is not None:
            mask_train = obs_mask[train_idx].copy()
            mask_val = obs_mask[val_idx].copy()
        else:
            mask_train = (~np.isnan(feat_train)).astype(np.float32)
            mask_val = (~np.isnan(feat_val)).astype(np.float32)

        # MICE imputation on any remaining NaN
        if np.isnan(feat_train).any():
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=10, max_depth=5, random_state=42, n_jobs=-1
                ),
                max_iter=10,
                random_state=42,
                sample_posterior=False,
            )
            feat_train = imputer.fit_transform(feat_train)
            feat_val = imputer.transform(feat_val)

        scaler = StandardScaler()
        feat_train = scaler.fit_transform(feat_train)
        feat_val = scaler.transform(feat_val)

        k = min(trainer_config.graph_k, len(val_idx) - 1)
        train_edge = build_knn_graph(
            feat_train, k=trainer_config.graph_k, obs_mask=mask_train
        )
        val_edge = build_knn_graph(feat_val, k=k, obs_mask=mask_val)

        train_data = Data(
            x=torch.tensor(feat_train, dtype=torch.float32),
            edge_index=train_edge,
            time=torch.tensor(times[train_idx], dtype=torch.float32),
            event=torch.tensor(events[train_idx], dtype=torch.long),
            obs_mask=torch.tensor(mask_train, dtype=torch.float32),
        )
        val_data = Data(
            x=torch.tensor(feat_val, dtype=torch.float32),
            edge_index=val_edge,
            time=torch.tensor(times[val_idx], dtype=torch.float32),
            event=torch.tensor(events[val_idx], dtype=torch.long),
            obs_mask=torch.tensor(mask_val, dtype=torch.float32),
        )

        result = trainer.train_fold(train_data, val_data, fold=fold, verbose=False)
        c_idx = result["final_metrics"].get("c_index", -1)
        all_c_indices.append(c_idx)
        print(
            f"    Fold {fold + 1}: C-index = {c_idx:.4f} (trained {result['epochs_trained']} epochs)"
        )

    arr = np.array(all_c_indices)
    mean, ci_low, ci_high = bootstrap_ci(arr)
    return {
        "label": label,
        "c_indices": all_c_indices,
        "mean": float(mean),
        "std": float(arr.std()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def main():
    start_time = time.time()

    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: Imputation Contribution Assessment")
    print("=" * 70 + "\n")

    df, feature_cols, modality_dims, modality_features = load_baseline_data()
    train_config = yaml.safe_load(
        open(project_root / "config/training/true_giman.yaml")
    )

    features = df[feature_cols].values.astype(np.float32)
    times = df["time_to_event"].values.astype(np.float32)
    events = df["phenoconverted"].values.astype(np.int64)

    results = []

    # ---- Experiment 1: Full model (missingness-aware, tiered imputation) ----
    print("\n" + "-" * 70)
    print("Experiment 1: FULL MODEL (missingness-aware, 30 features, 6 modalities)")
    print(f"  Patients: {len(df)}, Events: {events.sum()}")
    print("-" * 70)

    # Record obs mask, then do tiered imputation
    obs_mask_full = (~np.isnan(features)).astype(np.float32)

    # For Tier 3 modalities, fill with column mean as placeholder
    features_full = features.copy()
    offset = 0
    for mod_name, dim in modality_dims.items():
        mod_miss = np.isnan(features_full[:, offset : offset + dim]).mean()
        if mod_miss > 0.50:  # Tier 3
            for col in range(offset, offset + dim):
                observed = features_full[:, col][~np.isnan(features_full[:, col])]
                fill_val = observed.mean() if len(observed) > 0 else 0.0
                features_full[:, col] = np.where(
                    np.isnan(features_full[:, col]), fill_val, features_full[:, col]
                )
        offset += dim

    config_full = make_trainer_config(modality_dims, train_config)
    r1 = run_cv(
        features_full,
        times,
        events,
        config_full,
        obs_mask=obs_mask_full,
        label="Full model (missingness-aware)",
    )
    results.append(r1)

    # ---- Experiment 2: Clinical-only model ----
    # Only use modalities with <50% missing: genetic, expanded_clinical, clinical_biomarkers
    clinical_mods = ["genetic", "expanded_clinical", "clinical_biomarkers"]
    clinical_cols = []
    clinical_dims = {}
    for mod in clinical_mods:
        clinical_dims[mod] = modality_dims[mod]
        clinical_cols.extend(modality_features[mod])

    clinical_col_indices = [feature_cols.index(c) for c in clinical_cols]
    features_clinical = df[clinical_cols].values.astype(np.float32)

    print("\n" + "-" * 70)
    n_feats = len(clinical_cols)
    miss = np.isnan(features_clinical).mean()
    print(
        f"Experiment 2: CLINICAL-ONLY ({n_feats} features, {len(clinical_mods)} modalities)"
    )
    print(f"  Modalities: {list(clinical_dims.keys())}")
    print(f"  Patients: {len(df)}, Events: {events.sum()}, Missing: {miss:.1%}")
    print("-" * 70)

    config_clinical = make_trainer_config(clinical_dims, train_config)
    r2 = run_cv(
        features_clinical,
        times,
        events,
        config_clinical,
        label="Clinical-only (no imaging/CSF/cortical)",
    )
    results.append(r2)

    # ---- Experiment 3: Complete-case model ----
    # Only patients with ALL modalities observed (no NaN in any feature)
    complete_mask = ~np.isnan(features).any(axis=1)
    n_complete = complete_mask.sum()

    print("\n" + "-" * 70)
    print(f"Experiment 3: COMPLETE-CASE ({n_complete} patients, all 30 features)")
    print(
        f"  Events: {events[complete_mask].sum()} ({events[complete_mask].mean():.1%} event rate)"
    )
    print("-" * 70)

    if n_complete >= 50 and events[complete_mask].sum() >= 5:
        features_complete = features[complete_mask].copy()
        times_complete = times[complete_mask]
        events_complete = events[complete_mask]
        obs_mask_complete = np.ones_like(features_complete)  # All observed

        config_complete = make_trainer_config(modality_dims, train_config)
        # Adjust k for smaller dataset
        config_complete.graph_k = min(10, n_complete // 5 - 1)

        r3 = run_cv(
            features_complete,
            times_complete,
            events_complete,
            config_complete,
            obs_mask=obs_mask_complete,
            label="Complete-case (no imputation)",
        )
        results.append(r3)
    else:
        print(
            f"  ⚠ Not enough complete cases ({n_complete}) or events for reliable CV."
        )
        r3 = {
            "label": "Complete-case (insufficient data)",
            "c_indices": [],
            "mean": -1,
            "std": 0,
            "ci_low": -1,
            "ci_high": -1,
        }
        results.append(r3)

    # ---- Summary ----
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 70)
    for r in results:
        if r["mean"] > 0:
            print(
                f"  {r['label']:55s} C-index: {r['mean']:.4f} ± {r['std']:.4f} "
                f"(95% CI: [{r['ci_low']:.4f}, {r['ci_high']:.4f}])"
            )
        else:
            print(f"  {r['label']:55s} (insufficient data)")

    print(f"\n  Total time: {elapsed:.0f}s ({elapsed / 60:.1f}min)")

    # Interpretation
    print("\n  Interpretation:")
    if len(results) >= 2 and results[0]["mean"] > 0 and results[1]["mean"] > 0:
        diff = results[0]["mean"] - results[1]["mean"]
        if abs(diff) < 0.02:
            print(
                "  → Clinical-only ≈ Full model: imaging/CSF modalities contribute minimal signal."
            )
            print(
                "    The model may be learning mostly from clinical/genetic features."
            )
        elif diff > 0.02:
            print(
                f"  → Full model > Clinical-only by {diff:.4f}: imaging/CSF modalities contribute real signal."
            )
            print(
                "    The missingness-aware architecture is extracting value from partial observations."
            )
        else:
            print(
                f"  → Clinical-only > Full model by {-diff:.4f}: high-missingness modalities may add noise."
            )

    # Save results
    output_dir = project_root / "results" / "true_giman"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "sensitivity_analysis.json"
    with open(results_path, "w") as f:
        json.dump({"experiments": results, "total_time_seconds": elapsed}, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
