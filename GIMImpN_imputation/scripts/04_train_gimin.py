#!/usr/bin/env python3
"""Step 4: Train the GIMIN imputation model.

Trains the Graph-based Iterative Multimodal Imputation Network using
self-supervised learning with artificial missingness masks. The model
learns to reconstruct masked values using graph message-passing and
cross-modal attention.

Prerequisites:
    - outputs/ppmi_full_cohort.parquet (from Step 1/2)
    - outputs/missingness_mask.parquet
    - outputs/patient_graph.pt (from Step 3)

Outputs:
    outputs/checkpoints/gimin_best.pt      -- best model checkpoint
    outputs/checkpoints/gimin_final.pt     -- final epoch checkpoint
    outputs/logs/training_metrics.json     -- per-epoch loss curves
    outputs/logs/tensorboard/              -- TensorBoard logs (optional)

Usage:
    python scripts/04_train_gimin.py [--config configs/default.yaml]
    python scripts/04_train_gimin.py --epochs 300 --lr 0.0005
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GIMIN imputation model.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-mask-fraction", type=float, default=None)
    parser.add_argument("--lambda-dist", type=float, default=None)
    parser.add_argument("--lambda-cross", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("gimin.train")

    import numpy as np
    import pandas as pd

    from gimin.config import GIMINConfig
    from gimin.data.scaler import build_scaler_from_config

    # Load config
    if args.config:
        config = GIMINConfig.from_yaml(args.config)
    else:
        config = GIMINConfig()

    # Apply CLI overrides
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.lr is not None:
        config.training.lr = args.lr
    if args.batch_mask_fraction is not None:
        config.training.batch_mask_fraction = args.batch_mask_fraction
    if args.lambda_dist is not None:
        config.training.lambda_dist = args.lambda_dist
    if args.lambda_cross is not None:
        config.training.lambda_cross = args.lambda_cross
    if args.seed is not None:
        config.random_seed = args.seed

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GIMIN: Model Training (Step 4)")
    print("=" * 70)
    print(f"  Epochs:          {config.training.num_epochs}")
    print(f"  Learning rate:   {config.training.lr}")
    print(f"  Mask fraction:   {config.training.batch_mask_fraction}")
    print(f"  Lambda dist:     {config.training.lambda_dist}")
    print(f"  Lambda cross:    {config.training.lambda_cross}")
    print(f"  Random seed:     {config.random_seed}")

    # Load data
    feat_path = output_dir / "ppmi_full_cohort.parquet"
    mask_path = output_dir / "missingness_mask.parquet"
    graph_path = output_dir / "patient_graph.pt"

    logger.info("Loading features from: %s", feat_path)
    features_df = pd.read_parquet(feat_path)
    logger.info("Loading mask from: %s", mask_path)
    mask_df = pd.read_parquet(mask_path)

    # Filter to only the features defined in config (drops zero-variance cols)
    keep_cols = config.all_feature_names
    missing_cols = [c for c in keep_cols if c not in features_df.columns]
    if missing_cols:
        logger.error("Config features not found in parquet: %s", missing_cols)
        sys.exit(1)
    dropped = [c for c in features_df.columns if c not in keep_cols]
    if dropped:
        logger.info(
            "Dropping %d zero-variance/unused columns: %s", len(dropped), dropped
        )
    features_df = features_df[keep_cols]
    mask_df = mask_df[keep_cols]

    logger.info(
        "  %d patients, %d features",
        features_df.shape[0],
        features_df.shape[1],
    )

    # Set random seed
    np.random.seed(config.random_seed)

    try:
        import torch

        from gimin.graph.partial_similarity import PartialObservationGraphBuilder
        from gimin.model.gimin_core import GIMIN
        from gimin.training.trainer import GIMINTrainer

        # ----------------------------------------------------------
        # Set reproducibility seeds
        # ----------------------------------------------------------
        torch.manual_seed(config.random_seed)

        # ----------------------------------------------------------
        # Load or build the patient similarity graph
        # ----------------------------------------------------------
        builder = PartialObservationGraphBuilder(
            k_neighbors=config.graph.k_neighbors,
            min_overlap=config.graph.min_overlap,
        )

        if graph_path.exists():
            logger.info("Loading precomputed graph from: %s", graph_path)
            graph_data = torch.load(graph_path, weights_only=False)
            edge_index = graph_data["edge_index"]
            edge_weight = graph_data["edge_weight"]
            overlap_frac = graph_data["overlap_frac"]

            # Filter to eligible patients if the graph was built on a subset
            eligible_idx = graph_data.get("eligible_indices")
            if eligible_idx is not None:
                logger.info(
                    "  Graph covers %d eligible patients (of %d total)",
                    len(eligible_idx),
                    len(features_df),
                )
                features_df = features_df.iloc[eligible_idx].reset_index(drop=True)
                mask_df = mask_df.iloc[eligible_idx].reset_index(drop=True)
            logger.info("  Graph loaded: %d directed edges", edge_index.shape[1])
        else:
            logger.info("Graph not found at %s; building from scratch...", graph_path)
            graph_result = builder.build_full_graph(
                features_df.fillna(0).values,
                mask_df.values.astype(np.float32),
            )
            edge_index = graph_result["edge_index"]
            edge_weight = graph_result["edge_weight"]
            overlap_frac = graph_result["overlap_frac"]
            logger.info("  Graph built: %d directed edges", edge_index.shape[1])

        # ----------------------------------------------------------
        # Convert DataFrames to tensors
        # ----------------------------------------------------------
        features_t = torch.tensor(features_df.fillna(0).values, dtype=torch.float32)
        mask_t = torch.tensor(mask_df.values.astype(np.float32), dtype=torch.float32)
        logger.info(
            "Tensors created: features %s, mask %s",
            tuple(features_t.shape),
            tuple(mask_t.shape),
        )

        # ----------------------------------------------------------
        # Fit modality-aware scaler and normalize features
        # ----------------------------------------------------------
        scaler = build_scaler_from_config(config)
        scaler.fit(features_t, mask_t)
        features_t = scaler.transform(features_t, mask_t)
        logger.info("Features normalized with modality-aware scaler.")

        # ----------------------------------------------------------
        # Instantiate the GIMIN model
        # ----------------------------------------------------------
        model = GIMIN(
            modality_dims=config.modality_dims,
            embed_dim=config.model.embed_dim,
            num_gnn_layers=config.model.num_gnn_layers,
            num_heads=config.model.num_heads,
            mc_dropout=config.model.mc_dropout_rate,
            binary_feature_indices=getattr(config, "binary_feature_indices", None),
        )
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "GIMIN model created: %s parameters, "
            "embed_dim=%d, num_gnn_layers=%d, num_heads=%d",
            f"{total_params:,}",
            config.model.embed_dim,
            config.model.num_gnn_layers,
            config.model.num_heads,
        )

        # ----------------------------------------------------------
        # Create the trainer
        # ----------------------------------------------------------
        cross_modal_pairs = (
            [tuple(pair) for pair in config.cross_modal_pairs]
            if config.cross_modal_pairs
            else None
        )
        trainer = GIMINTrainer(
            model,
            config,
            graph_builder=builder,
            cross_modal_pairs=cross_modal_pairs,
        )
        trainer.scaler = scaler
        logger.info("Trainer initialised on device: %s", trainer.device)
        if cross_modal_pairs:
            logger.info(
                "Cross-modal pairs configured: %d pairs", len(cross_modal_pairs)
            )

        # Resume from checkpoint if requested
        if args.resume:
            logger.info("Resuming from checkpoint: %s", args.resume)
            trainer.load_checkpoint(args.resume)

        # ----------------------------------------------------------
        # Assemble the dataset dict
        # ----------------------------------------------------------
        dataset = {
            "features": features_t,
            "mask": mask_t,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "overlap_frac": overlap_frac,
        }

        # ----------------------------------------------------------
        # Train
        # ----------------------------------------------------------
        print("\n  Starting training...")
        t_start = time.time()
        history = trainer.fit(dataset)
        elapsed_total = time.time() - t_start

        logger.info(
            "Training complete: %d epochs in %.1fs (%.2fs/epoch)",
            len(history),
            elapsed_total,
            elapsed_total / max(len(history), 1),
        )

        # ----------------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------------
        best_ckpt_path = str(ckpt_dir / "gimin_best.pt")
        trainer.save_checkpoint(best_ckpt_path)
        logger.info("Best checkpoint saved to: %s", best_ckpt_path)

        # ----------------------------------------------------------
        # Save training history to JSON
        # ----------------------------------------------------------
        history_record = {
            "status": "completed",
            "total_epochs": len(history),
            "elapsed_seconds": round(elapsed_total, 2),
            "best_loss": round(trainer.best_loss, 6),
            "config": {
                "epochs": config.training.num_epochs,
                "lr": config.training.lr,
                "batch_mask_fraction": config.training.batch_mask_fraction,
                "lambda_dist": config.training.lambda_dist,
                "lambda_cross": config.training.lambda_cross,
                "embed_dim": config.model.embed_dim,
                "num_gnn_layers": config.model.num_gnn_layers,
                "num_heads": config.model.num_heads,
            },
            "data": {
                "num_patients": len(features_df),
                "num_features": features_df.shape[1],
                "overall_missingness": float(1 - mask_df.values.mean()),
            },
            "per_epoch": history,
        }

        metrics_path = log_dir / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(history_record, f, indent=2)
        logger.info("Training history saved to: %s", metrics_path)

        print(f"\n{'=' * 70}")
        print("Training Complete")
        print(f"{'=' * 70}")
        print(f"  Best loss:      {trainer.best_loss:.6f}")
        print(f"  Epochs trained: {len(history)}")
        print(f"  Wall time:      {elapsed_total:.1f}s")
        print(f"  Checkpoint dir: {ckpt_dir}")
        print(f"  Log dir:        {log_dir}")
        print("=" * 70)

    except ImportError as e:
        logger.warning("PyTorch or model components not available: %s", e)
        logger.info("Training requires: pip install torch torch-geometric")


if __name__ == "__main__":
    main()
