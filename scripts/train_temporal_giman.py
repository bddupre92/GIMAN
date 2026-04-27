#!/usr/bin/env python3
"""Train Temporal GIMAN with 5-fold cross-validation.

Loads longitudinal sequences from Phase 1 output, builds the TemporalGIMAN
model, and runs k-fold CV with dynamic C-index evaluation at landmark times.

Usage:
    python scripts/train_temporal_giman.py
    python scripts/train_temporal_giman.py --max_epochs 5 --k_folds 2  # smoke test
    python scripts/train_temporal_giman.py --batch_size 32 --lr 0.001
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.giman_pipeline.training.temporal_data_loaders import (
    load_longitudinal_dataset,
)
from src.giman_pipeline.training.temporal_trainer import (
    TemporalMultiTaskTrainer,
    TemporalTrainingConfig,
)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_modality_dims(feature_config_path: str) -> dict[str, int]:
    """Extract modality_dims dict from feature config YAML.

    Returns dict like: {'genetic': 5, 'expanded_clinical': 5, ...}
    """
    with open(feature_config_path) as f:
        config = yaml.safe_load(f)

    modality_dims = {}
    for name, info in config["modalities"].items():
        modality_dims[name] = info["dim"]

    return modality_dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Temporal GIMAN with k-fold CV"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training/temporal_giman.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Override max epochs (for smoke testing)",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=None,
        help="Override number of CV folds",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--min_visits",
        type=int,
        default=None,
        help="Override minimum visits per patient",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/temporal_giman",
        help="Directory to save results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("TEMPORAL GIMAN TRAINING PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Load configuration ──
    config_path = PROJECT_ROOT / args.config
    config = load_config(str(config_path))
    print(f"\nConfig: {config_path}")

    # Load modality dims from feature config
    feature_config_path = PROJECT_ROOT / config["data"]["feature_config"]
    modality_dims = load_modality_dims(str(feature_config_path))
    total_features = sum(modality_dims.values())
    print(f"Feature config: {feature_config_path}")
    print(f"Modalities: {len(modality_dims)} ({total_features} features)")
    for name, dim in modality_dims.items():
        print(f"  {name}: {dim}")

    # ── Load longitudinal dataset ──
    data_dir = PROJECT_ROOT / config["data"]["data_dir"]
    endpoints_path = PROJECT_ROOT / config["data"]["endpoints"]
    min_visits = args.min_visits or config["data"].get("min_visits", 2)

    print(f"\nData: {data_dir}")
    print(f"Endpoints: {endpoints_path}")
    print(f"Min visits: {min_visits}")

    dataset = load_longitudinal_dataset(
        data_dir=str(data_dir),
        endpoints_path=str(endpoints_path),
        min_visits=min_visits,
    )

    # ── Build training config ──
    model_cfg = config["model"]
    train_cfg = config["training"]
    loss_cfg = config["loss"]
    eval_cfg = config["evaluation"]
    temporal_cfg = model_cfg.get("temporal", {})

    trainer_config = TemporalTrainingConfig(
        modality_dims=modality_dims,
        modality_embed_dim=model_cfg.get("modality_embed_dim", 64),
        cross_modal_heads=model_cfg.get("cross_modal_heads", 4),
        fused_dim=model_cfg.get("fused_dim", 128),
        gat_hidden_dim=model_cfg.get("gat_hidden_dim", 64),
        gat_output_dim=model_cfg.get("gat_output_dim", 64),
        gat_heads=model_cfg.get("gat_heads", 4),
        gat_layers=model_cfg.get("gat_layers", 3),
        dropout=model_cfg.get("dropout", 0.3),
        adaptive_fusion=model_cfg.get("adaptive_fusion", True),
        observed_threshold=model_cfg.get("observed_threshold", 0.3),
        temporal_hidden_dim=temporal_cfg.get("hidden_dim", 64),
        temporal_num_layers=temporal_cfg.get("num_layers", 2),
        lr=args.lr or float(train_cfg.get("lr", 0.0005)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        max_epochs=args.max_epochs or int(train_cfg.get("max_epochs", 200)),
        patience=int(train_cfg.get("patience", 25)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        batch_size=args.batch_size or train_cfg.get("batch_size", 64),
        graph_k=config["data"].get("graph_k", 10),
        cox_loss_weight=loss_cfg.get("cox_weight", 1.0),
        trajectory_loss_weight=loss_cfg.get("trajectory_weight", 0.5),
        landmark_times=eval_cfg.get("landmark_times", [12, 24, 36, 48]),
    )

    k_folds = args.k_folds or config["data"].get("k_folds", 5)

    # Print model summary
    from src.giman_pipeline.modeling.temporal_giman import TemporalGIMAN

    summary_model = TemporalGIMAN(
        modality_dims=modality_dims,
        modality_embed_dim=trainer_config.modality_embed_dim,
        cross_modal_heads=trainer_config.cross_modal_heads,
        fused_dim=trainer_config.fused_dim,
        temporal_hidden_dim=trainer_config.temporal_hidden_dim,
        temporal_num_layers=trainer_config.temporal_num_layers,
        gat_hidden_dim=trainer_config.gat_hidden_dim,
        gat_output_dim=trainer_config.gat_output_dim,
        gat_heads=trainer_config.gat_heads,
        gat_layers=trainer_config.gat_layers,
        dropout=trainer_config.dropout,
        adaptive_fusion=trainer_config.adaptive_fusion,
        observed_threshold=trainer_config.observed_threshold,
        graph_k=trainer_config.graph_k,
    )
    param_counts = summary_model.count_parameters()
    print(f"\nModel parameters:")
    for component, count in param_counts.items():
        print(f"  {component}: {count:,}")
    del summary_model

    # ── Run cross-validation ──
    trainer = TemporalMultiTaskTrainer(trainer_config)
    results = trainer.cross_validate(dataset, k_folds=k_folds)

    # ── Save results ──
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary (without model states — too large)
    summary = {
        "timestamp": timestamp,
        "config": {
            "model": model_cfg,
            "training": train_cfg,
            "loss": loss_cfg,
            "evaluation": eval_cfg,
            "k_folds": k_folds,
            "min_visits": min_visits,
            "seed": args.seed,
        },
        "modality_dims": modality_dims,
        "total_features": total_features,
        "dataset_size": len(dataset),
        "parameter_counts": param_counts,
        "folds": [],
    }

    for r in results:
        fold_summary = {
            "fold": r["fold"],
            "best_primary_metric": r["best_primary_metric"],
            "final_metrics": r["final_metrics"],
            "epochs_trained": r["epochs_trained"],
        }
        summary["folds"].append(fold_summary)

    # Aggregate metrics
    all_metrics: dict[str, list[float]] = {}
    for r in results:
        for k, v in r["final_metrics"].items():
            all_metrics.setdefault(k, []).append(v)

    summary["aggregate"] = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in all_metrics.items()
    }

    summary_path = output_dir / f"cv_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    # Save training histories
    histories_path = output_dir / f"training_histories_{timestamp}.json"
    histories = {
        f"fold_{r['fold']}": r["training_history"] for r in results
    }
    with open(histories_path, "w") as f:
        json.dump(histories, f, indent=2)
    print(f"Histories saved: {histories_path}")

    # Save best model states
    for r in results:
        if r["model_state"] is not None:
            model_path = (
                output_dir / f"model_fold{r['fold']}_{timestamp}.pt"
            )
            torch.save(r["model_state"], model_path)
    print(f"Model states saved to: {output_dir}")

    # ── Final report ──
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    for metric, stats in summary["aggregate"].items():
        print(f"  {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
