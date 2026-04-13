#!/usr/bin/env python3
"""Step 6: Impute the full PPMI cohort using the trained GIMIN model.

Runs the trained model in inference mode to fill in ALL missing values
across the full cohort, producing a complete (imputed) feature matrix
with uncertainty estimates for every imputed value.

Prerequisites:
    - outputs/checkpoints/gimin_best.pt (from Step 4)
    - outputs/ppmi_full_cohort.parquet
    - outputs/missingness_mask.parquet
    - outputs/patient_graph.pt

Outputs:
    outputs/imputed/ppmi_imputed.parquet         -- complete matrix
    outputs/imputed/uncertainty_estimates.parquet -- per-value uncertainty
    outputs/imputed/imputation_summary.json      -- summary statistics
    outputs/imputed/ppmi_imputed.csv             -- CSV for inspection

Usage:
    python scripts/06_impute_full_cohort.py [--config configs/default.yaml]
    python scripts/06_impute_full_cohort.py --mc-samples 100 --clip-to-range
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
        description="Impute missing values for the full PPMI cohort."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=50,
        help="Number of MC-dropout forward passes for uncertainty.",
    )
    parser.add_argument(
        "--clip-to-range",
        action="store_true",
        help="Clip imputed values to clinically valid ranges.",
    )
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
    logger = logging.getLogger("gimin.impute")

    import numpy as np
    import pandas as pd

    from gimin.config import GIMINConfig
    from gimin.data.modality_registry import ModalityRegistry

    # Load config
    if args.config:
        config = GIMINConfig.from_yaml(args.config)
    else:
        config = GIMINConfig()

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    impute_dir = output_dir / "imputed"
    impute_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GIMIN: Full Cohort Imputation (Step 6)")
    print("=" * 70)
    print(f"  MC samples:      {args.mc_samples}")
    print(f"  Clip to range:   {args.clip_to_range}")

    # Load data
    feat_path = output_dir / "ppmi_full_cohort.parquet"
    mask_path = output_dir / "missingness_mask.parquet"
    graph_path = output_dir / "patient_graph.pt"
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else output_dir / "checkpoints" / "gimin_best.pt"
    )

    logger.info("Loading features from: %s", feat_path)
    features_df = pd.read_parquet(feat_path)
    mask_df = pd.read_parquet(mask_path)

    logger.info(
        "  %d patients, %d features", features_df.shape[0], features_df.shape[1]
    )

    # ── Missingness statistics ──────────────────────────────────────────
    n_observed = mask_df.values.sum()  # mask==1 means observed
    n_total = mask_df.values.size
    n_missing = n_total - n_observed
    pct_missing = n_missing / n_total * 100 if n_total > 0 else 0.0

    per_feature_miss = (mask_df == 0).sum().sort_values(ascending=False)
    top_missing = per_feature_miss.head(10)

    logger.info("  Total cells:     %d", n_total)
    logger.info("  Observed cells:  %d", n_observed)
    logger.info("  Missing cells:   %d  (%.2f%%)", n_missing, pct_missing)
    logger.info("  Top-10 features by missingness:")
    for feat_name, cnt in top_missing.items():
        logger.info(
            "    %s: %d missing (%.1f%% of patients)",
            feat_name,
            cnt,
            cnt / len(mask_df) * 100,
        )

    print(f"  Missing values:  {n_missing:,} / {n_total:,} ({pct_missing:.1f}%)")
    print()

    # ── Load model and run imputation ────────────────────────────────────
    import torch

    from gimin.inference.impute import GIMINImputer

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    uncertainty_df = None

    if ckpt_path.exists():
        logger.info("Loading GIMIN checkpoint from: %s", ckpt_path)

        imputer = GIMINImputer(
            model_path=str(ckpt_path),
            config=config,
            mc_samples=args.mc_samples,
        )

        # Load graph state if available
        if graph_path.exists():
            logger.info("Loading patient graph from: %s", graph_path)
            imputer.load_graph_state(str(graph_path))
        else:
            logger.warning(
                "Patient graph not found at %s; "
                "imputer will proceed without graph context.",
                graph_path,
            )

        # Determine which feature columns are actually present in the data
        feature_columns = config.all_feature_names
        available_cols = [c for c in feature_columns if c in features_df.columns]
        logger.info(
            "  Config defines %d features; %d found in dataframe",
            len(feature_columns),
            len(available_cols),
        )

        if not available_cols:
            logger.warning(
                "No config feature columns matched the dataframe -- "
                "falling back to all numeric columns."
            )
            available_cols = features_df.select_dtypes(
                include="number"
            ).columns.tolist()

        # Run GIMIN MC-dropout imputation
        logger.info("Running GIMIN imputation with %d MC samples...", args.mc_samples)
        imputed_df = imputer.impute_to_dataframe(
            df=features_df,
            feature_columns=available_cols,
            return_uncertainty=True,
        )
        imputation_method = "gimin_mc_dropout"
        logger.info("GIMIN imputation complete.")

    else:
        # Fallback to mean imputation
        logger.warning("No checkpoint at %s, using mean imputation fallback", ckpt_path)
        imputed_df = features_df.copy()
        for col in features_df.columns:
            if features_df[col].isna().any():
                imputed_df[col] = imputed_df[col].fillna(imputed_df[col].mean())
        imputation_method = "mean_fallback"

    # ── Separate uncertainty columns (_std suffix) ───────────────────────
    std_cols = [c for c in imputed_df.columns if c.endswith("_std")]
    if std_cols:
        uncertainty_df = imputed_df[std_cols].copy()
        imputed_df = imputed_df.drop(columns=std_cols)
        logger.info(
            "  Extracted %d uncertainty columns (mean std = %.6f)",
            len(std_cols),
            uncertainty_df.values.mean(),
        )
    else:
        logger.info("  No uncertainty columns (*_std) returned by imputer.")

    # ── Optionally clip imputed values to clinical ranges ────────────────
    if args.clip_to_range:
        registry = ModalityRegistry()
        n_clipped_features = 0
        for mod in registry:
            for feat in mod.features:
                if feat in imputed_df.columns and feat in mod.clinical_ranges:
                    lo, hi = mod.clinical_ranges[feat]
                    if lo is not None:
                        imputed_df[feat] = imputed_df[feat].clip(lower=lo)
                    if hi is not None:
                        imputed_df[feat] = imputed_df[feat].clip(upper=hi)
                    n_clipped_features += 1
        logger.info("Clipped %d features to clinical ranges", n_clipped_features)

    # ── Save outputs ─────────────────────────────────────────────────────
    logger.info("Saving imputed data to: %s", impute_dir)

    imputed_df.to_parquet(impute_dir / "ppmi_imputed.parquet")
    imputed_df.to_csv(impute_dir / "ppmi_imputed.csv")
    logger.info("  Saved ppmi_imputed.parquet and ppmi_imputed.csv")

    if uncertainty_df is not None and not uncertainty_df.empty:
        uncertainty_df.to_parquet(impute_dir / "uncertainty_estimates.parquet")
        logger.info("  Saved uncertainty_estimates.parquet")
        mean_uncertainty = float(uncertainty_df.values.mean())
    else:
        mean_uncertainty = 0.0

    # Summary JSON
    remaining_nans = int(imputed_df.isna().sum().sum())
    summary = {
        "num_patients": len(imputed_df),
        "num_features": imputed_df.shape[1],
        "total_cells": int(n_total),
        "values_missing_before": int(n_missing),
        "pct_missing_before": round(pct_missing, 4),
        "values_imputed": int(n_missing),
        "imputation_method": imputation_method,
        "mc_samples": args.mc_samples if imputation_method == "gimin_mc_dropout" else 0,
        "clipped_to_range": args.clip_to_range,
        "remaining_nans": remaining_nans,
        "mean_uncertainty": mean_uncertainty,
        "checkpoint_used": str(ckpt_path) if ckpt_path.exists() else None,
        "seed": args.seed,
    }

    summary_path = impute_dir / "imputation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("  Saved imputation_summary.json")

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Imputation Complete")
    print(f"{'=' * 70}")
    print(f"  Patients:          {summary['num_patients']:,}")
    print(f"  Features:          {summary['num_features']:,}")
    print(f"  Values imputed:    {summary['values_imputed']:,}")
    print(f"  Method:            {summary['imputation_method']}")
    print(f"  MC samples:        {summary['mc_samples']}")
    print(f"  Clipped:           {summary['clipped_to_range']}")
    print(f"  Remaining NaN:     {summary['remaining_nans']}")
    print(f"  Mean uncertainty:  {summary['mean_uncertainty']:.6f}")
    print(f"  Output dir:        {impute_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
