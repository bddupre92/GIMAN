#!/usr/bin/env python3
"""Step 5: Evaluate the trained GIMIN model.

Runs evaluation experiments on the trained model, including:
    - Reconstruction error under artificial masking at various fractions
    - Per-modality imputation accuracy (RMSE, MAE, R-squared)
    - Uncertainty calibration (MC-dropout reliability diagrams)
    - Comparison against baseline methods (mean, kNN, MICE)

Prerequisites:
    - outputs/checkpoints/gimin_best.pt  (from Step 4)
    - outputs/ppmi_full_cohort.parquet
    - outputs/missingness_mask.parquet
    - outputs/patient_graph.pt

Outputs:
    outputs/evaluation/eval_results.json       -- quantitative metrics
    outputs/evaluation/per_modality_rmse.csv   -- per-modality breakdown
    outputs/evaluation/calibration_plot.png     -- uncertainty calibration
    outputs/evaluation/baseline_comparison.csv  -- vs. baselines

Usage:
    python scripts/05_evaluate.py [--config configs/default.yaml]
    python scripts/05_evaluate.py --mask-fractions 0.1 0.2 0.3 --mc-samples 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained GIMIN imputation model."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--mask-fractions",
        type=float,
        nargs="+",
        default=None,
        help="Masking fractions to evaluate at. Default: [0.1, 0.2, 0.3, 0.5]",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=None,
        help="Number of MC-dropout samples for uncertainty estimation.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help="Number of random runs per masking fraction.",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("gimin.eval")

    import numpy as np
    import pandas as pd

    from gimin.config import GIMINConfig
    from gimin.data.missingness import MissingnessAnalyzer
    from gimin.data.modality_registry import ModalityRegistry

    # Load config
    if args.config:
        config = GIMINConfig.from_yaml(args.config)
    else:
        config = GIMINConfig()

    mask_fractions = args.mask_fractions or config.evaluation.eval_mask_fractions
    mc_samples = args.mc_samples or config.evaluation.mc_samples
    num_runs = args.num_runs or config.evaluation.eval_num_runs

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GIMIN: Model Evaluation (Step 5)")
    print("=" * 70)
    print(f"  Mask fractions:  {mask_fractions}")
    print(f"  MC samples:      {mc_samples}")
    print(f"  Runs per frac:   {num_runs}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    feat_path = output_dir / "ppmi_full_cohort.parquet"
    mask_path = output_dir / "missingness_mask.parquet"

    logger.info("Loading features from: %s", feat_path)
    features_df = pd.read_parquet(feat_path)
    logger.info("Loading mask from: %s", mask_path)
    mask_df = pd.read_parquet(mask_path)

    # Analyze existing missingness
    registry = ModalityRegistry()
    analyzer = MissingnessAnalyzer(registry)
    summary = analyzer.summary(features_df)

    logger.info(
        "Dataset: %d patients, %d features", features_df.shape[0], features_df.shape[1]
    )
    logger.info("Overall missingness: %.3f", summary["total_fraction"])

    # ------------------------------------------------------------------
    # Load graph to get eligible patient indices
    # ------------------------------------------------------------------
    import torch

    graph_path = output_dir / "patient_graph.pt"
    graph_data = None
    eligible_idx = None
    if graph_path.exists():
        graph_data = torch.load(graph_path, weights_only=False)
        eligible_idx = graph_data.get("eligible_indices")

    if eligible_idx is not None:
        logger.info(
            "Filtering to %d eligible patients (of %d total)",
            len(eligible_idx),
            len(features_df),
        )
        features_df = features_df.iloc[eligible_idx].reset_index(drop=True)
        mask_df = mask_df.iloc[eligible_idx].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Prepare features and mask as numpy arrays
    # ------------------------------------------------------------------
    features_np = features_df.values.astype(np.float32)
    mask_np = mask_df.values.astype(np.float32)

    # Replace NaN in features with 0 (the model expects zeros at missing positions)
    features_np = np.nan_to_num(features_np, nan=0.0)

    # ------------------------------------------------------------------
    # Load trained GIMIN model from checkpoint
    # ------------------------------------------------------------------
    from gimin.model.gimin_core import GIMIN

    model = GIMIN(
        modality_dims=config.modality_dims,
        embed_dim=config.model.embed_dim,
        num_gnn_layers=config.model.num_gnn_layers,
        num_heads=config.model.num_heads,
        mc_dropout=config.model.mc_dropout_rate,
    )

    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else output_dir / "checkpoints" / "gimin_best.pt"
    )
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        logger.info("Loaded model from %s", ckpt_path)
    else:
        logger.warning("No checkpoint at %s, running baselines only", ckpt_path)
        model = None

    # ------------------------------------------------------------------
    # Reuse graph data already loaded above for patient filtering
    # ------------------------------------------------------------------
    edge_index, edge_weight = None, None
    if graph_data is not None:
        edge_index = graph_data.get("edge_index")
        edge_weight = graph_data.get("edge_weight")
        if edge_index is not None:
            logger.info("Loaded patient graph: %d edges", edge_index.shape[1])
    else:
        logger.warning("No patient graph available")

    # ------------------------------------------------------------------
    # Create baseline imputers
    # ------------------------------------------------------------------
    from gimin.evaluation.baselines import (
        KNNBaseline,
        MeanBaseline,
        MedianBaseline,
        MICEBaseline,
    )

    baselines = {
        "MICE": MICEBaseline(),
        "KNN": KNNBaseline(),
        "Mean": MeanBaseline(),
        "Median": MedianBaseline(),
    }

    # ------------------------------------------------------------------
    # Run MaskedValueExperiment
    # ------------------------------------------------------------------
    from gimin.evaluation.masked_experiment import MaskedValueExperiment

    experiment = MaskedValueExperiment(config)

    if model is not None:
        logger.info("Running full evaluation (GIMIN + baselines)...")
        results = experiment.run(
            model=model,
            features=features_np,
            mask=mask_np,
            mask_fractions=mask_fractions,
            num_runs=num_runs,
            baselines=baselines,
            edge_index=edge_index,
            edge_weight=edge_weight,
            random_seed=args.seed,
        )
    else:
        # Baselines-only mode: run each baseline through the experiment
        # by creating a simple mean-imputation "model" stand-in
        logger.info("No model checkpoint — running baselines-only evaluation...")
        results = {
            "baselines_only": True,
            "summary": {},
        }
        # Run each baseline individually through the masking protocol
        for bname, bmodel in baselines.items():
            results[bname] = {}
            for frac in mask_fractions:
                frac_key = f"{frac:.2f}"
                run_metrics_list = []
                for run_idx in range(num_runs):
                    rng = np.random.default_rng(args.seed + run_idx)
                    corrupted_mask, target_mask = (
                        MaskedValueExperiment._create_evaluation_mask(
                            mask_np, frac, rng
                        )
                    )
                    corrupted_features = features_np.copy()
                    corrupted_features[~corrupted_mask.astype(bool)] = 0.0
                    b_imputed = bmodel.fit_transform(corrupted_features, corrupted_mask)

                    from gimin.evaluation import metrics as M

                    run_metrics = {
                        "rmse": M.rmse(b_imputed, features_np, target_mask),
                        "mae": M.mae(b_imputed, features_np, target_mask),
                        "r_squared": M.r_squared(b_imputed, features_np, target_mask),
                    }
                    run_metrics_list.append(run_metrics)

                # Aggregate
                mean_dict = {
                    k: float(np.mean([r[k] for r in run_metrics_list]))
                    for k in run_metrics_list[0]
                }
                std_dict = {
                    k: float(np.std([r[k] for r in run_metrics_list]))
                    for k in run_metrics_list[0]
                }
                results[bname][frac_key] = {
                    "runs": run_metrics_list,
                    "mean": mean_dict,
                    "std": std_dict,
                }

        # Build baselines-only summary
        for bname in baselines:
            bsummary = {}
            rmses = []
            for frac in mask_fractions:
                frac_key = f"{frac:.2f}"
                frac_data = results[bname].get(frac_key, {})
                mean_metrics = frac_data.get("mean", {})
                if "rmse" in mean_metrics:
                    rmses.append(mean_metrics["rmse"])
                    bsummary[f"rmse@{frac_key}"] = mean_metrics["rmse"]
            if rmses:
                bsummary["avg_rmse"] = float(np.mean(rmses))
            results["summary"][bname] = bsummary

    # ------------------------------------------------------------------
    # Enrich results with dataset metadata
    # ------------------------------------------------------------------
    results["metadata"] = {
        "mask_fractions": mask_fractions,
        "num_runs": num_runs,
        "mc_samples": mc_samples,
        "dataset_stats": {
            "num_patients": len(features_df),
            "num_features": features_df.shape[1],
            "overall_missingness": float(summary["total_fraction"]),
            "per_patient_mean": float(summary["per_patient_mean"]),
            "num_complete_rows": int(summary["num_complete_rows"]),
        },
        "per_modality_missingness": {
            k: float(v) for k, v in summary["per_modality"].items()
        },
        "model_checkpoint": str(ckpt_path) if ckpt_path.exists() else None,
    }

    # Classify missingness mechanisms
    try:
        mechanisms = analyzer.classify_missingness(features_df)
        results["metadata"]["missingness_mechanisms"] = {
            k: v.value for k, v in mechanisms.items()
        }
        logger.info("Missingness mechanism classification complete")
    except Exception as e:
        logger.warning("Missingness classification failed: %s", e)

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    results_path = eval_dir / "eval_results.json"

    def _make_serializable(obj):
        """Recursively convert numpy types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, "w") as f:
        json.dump(_make_serializable(results), f, indent=2)
    logger.info("Saved evaluation results to: %s", results_path)

    # ------------------------------------------------------------------
    # Create per-modality CSV from results
    # ------------------------------------------------------------------
    modality_rows = []
    summary_data = results.get("summary", {})
    methods = [k for k in results if k not in ("summary", "metadata", "baselines_only")]

    # Try to extract per-modality breakdown from GIMIN results
    if "gimin" in results and isinstance(results["gimin"], dict):
        for frac_key, frac_data in results["gimin"].items():
            if isinstance(frac_data, dict) and "modality_breakdown" in frac_data:
                for mod_name, mod_metrics in frac_data["modality_breakdown"].items():
                    modality_rows.append(
                        {
                            "method": "GIMIN",
                            "mask_fraction": frac_key,
                            "modality": mod_name,
                            "rmse": mod_metrics.get("rmse", np.nan),
                            "mae": mod_metrics.get("mae", np.nan),
                            "r_squared": mod_metrics.get("r_squared", np.nan),
                        }
                    )

    # Add overall metrics for all methods
    for method in methods:
        method_data = results[method]
        if isinstance(method_data, dict):
            for frac_key, frac_data in method_data.items():
                if isinstance(frac_data, dict) and "mean" in frac_data:
                    mean_m = frac_data["mean"]
                    modality_rows.append(
                        {
                            "method": method,
                            "mask_fraction": frac_key,
                            "modality": "OVERALL",
                            "rmse": mean_m.get("rmse", np.nan),
                            "mae": mean_m.get("mae", np.nan),
                            "r_squared": mean_m.get("r_squared", np.nan),
                        }
                    )

    mod_path = eval_dir / "per_modality_rmse.csv"
    if modality_rows:
        mod_df = pd.DataFrame(modality_rows)
        mod_df.to_csv(mod_path, index=False)
        logger.info("Saved per-modality breakdown to: %s", mod_path)
    else:
        logger.warning("No modality breakdown data to write.")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("Evaluation Complete -- RMSE Summary")
    print(f"{'=' * 70}")

    # Build header row
    frac_headers = [f"{f:.2f}" for f in mask_fractions]
    header = (
        f"  {'Method':<12s}"
        + "".join(f"  frac={h:>5s}" for h in frac_headers)
        + "   avg"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for method in methods:
        method_data = results.get(method, {})
        if not isinstance(method_data, dict):
            continue
        row = f"  {method:<12s}"
        rmses = []
        for frac in mask_fractions:
            frac_key = f"{frac:.2f}"
            frac_data = method_data.get(frac_key, {})
            mean_metrics = (
                frac_data.get("mean", {}) if isinstance(frac_data, dict) else {}
            )
            rmse_val = mean_metrics.get("rmse", float("nan"))
            row += f"  {rmse_val:>10.4f}"
            if not np.isnan(rmse_val):
                rmses.append(rmse_val)
        avg_rmse = float(np.mean(rmses)) if rmses else float("nan")
        row += f"  {avg_rmse:>6.4f}"
        print(row)

    print(f"{'=' * 70}")
    print(f"  Results JSON:    {results_path}")
    print(f"  Modality CSV:    {mod_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
