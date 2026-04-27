#!/usr/bin/env python3
"""Step 7: Export imputed data for the parent GIMAN project.

Converts the GIMIN-imputed feature matrix back into the format expected
by the GIMAN prodromal prognostic model, including:
    - Splitting modalities into individual CSV files matching GIMAN's schema
    - Generating a unified feature matrix with imputation flags
    - Producing a confidence-weighted version for downstream models
    - Writing compatibility metadata

Prerequisites:
    - outputs/imputed/ppmi_imputed.parquet    (from Step 6)
    - outputs/imputed/uncertainty_estimates.parquet
    - outputs/missingness_mask.parquet

Outputs:
    exports/giman_features_imputed.csv        -- unified GIMAN-compatible matrix
    exports/giman_features_original.csv       -- original (with NaN) for comparison
    exports/imputation_flags.csv              -- 1=imputed, 0=observed
    exports/confidence_weights.csv            -- 1/(1+uncertainty) per value
    exports/per_modality/                     -- individual modality CSVs
    exports/export_manifest.json              -- metadata and checksums

Usage:
    python scripts/07_export_for_giman.py [--output-dir exports/]
    python scripts/07_export_for_giman.py --format both  # CSV + parquet
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

GIMAN_ROOT = PROJECT_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GIMIN-imputed data for the GIMAN project."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing imputed outputs. Default: outputs/imputed/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Export directory. Default: exports/",
    )
    parser.add_argument(
        "--giman-data-dir",
        type=str,
        default=None,
        help=(
            "Also write to GIMAN's data directory. "
            f"Default: {GIMAN_ROOT / 'data' / '03_prodromal' / 'gimin_imputed'}"
        ),
    )
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["csv", "parquet", "both"],
        help="Output format. Default: both",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("gimin.export")

    import pandas as pd

    from gimin.config import GIMINConfig
    from gimin.data.modality_registry import ModalityRegistry

    config = GIMINConfig()
    registry = ModalityRegistry()

    input_dir = (
        Path(args.input_dir) if args.input_dir else PROJECT_ROOT / "outputs" / "imputed"
    )
    export_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    per_modality_dir = export_dir / "per_modality"
    per_modality_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GIMIN: Export for GIMAN (Step 7)")
    print("=" * 70)

    # Load imputed data
    imputed_path = input_dir / "ppmi_imputed.parquet"
    uncertainty_path = input_dir / "uncertainty_estimates.parquet"
    mask_path = PROJECT_ROOT / "outputs" / "missingness_mask.parquet"

    if not imputed_path.exists():
        logger.error(
            f"Imputed data not found: {imputed_path}\n"
            "Please run scripts/06_impute_full_cohort.py first."
        )
        sys.exit(1)

    logger.info("Loading imputed features from: %s", imputed_path)
    imputed_df = pd.read_parquet(imputed_path)

    # Load original mask
    if mask_path.exists():
        mask_df = pd.read_parquet(mask_path)
        # Align indices
        common_idx = imputed_df.index.intersection(mask_df.index)
        common_cols = [c for c in imputed_df.columns if c in mask_df.columns]
        mask_aligned = mask_df.loc[common_idx, common_cols]
    else:
        logger.warning("Original mask not found; generating from imputed data")
        mask_aligned = (~imputed_df.isna()).astype(int)

    # Load uncertainty
    if uncertainty_path.exists():
        uncertainty_df = pd.read_parquet(uncertainty_path)
    else:
        logger.warning("Uncertainty estimates not found; using zeros")
        uncertainty_df = pd.DataFrame(
            0.0, index=imputed_df.index, columns=imputed_df.columns
        )

    logger.info("  %d patients, %d features", imputed_df.shape[0], imputed_df.shape[1])

    # --- Export unified imputed matrix ---
    export_files = {}

    if args.format in ("csv", "both"):
        csv_path = export_dir / "giman_features_imputed.csv"
        imputed_df.to_csv(csv_path)
        export_files["giman_features_imputed.csv"] = csv_path
        logger.info("Saved: %s", csv_path)

    if args.format in ("parquet", "both"):
        pq_path = export_dir / "giman_features_imputed.parquet"
        imputed_df.to_parquet(pq_path)
        export_files["giman_features_imputed.parquet"] = pq_path
        logger.info("Saved: %s", pq_path)

    # --- Export original (with NaN) for comparison ---
    orig_path = PROJECT_ROOT / "outputs" / "ppmi_full_cohort.parquet"
    if orig_path.exists():
        original_df = pd.read_parquet(orig_path)
        orig_csv = export_dir / "giman_features_original.csv"
        original_df.to_csv(orig_csv)
        export_files["giman_features_original.csv"] = orig_csv
        logger.info("Saved original: %s", orig_csv)

    # --- Export imputation flags ---
    # 1 = imputed (was missing), 0 = observed (original data)
    imputation_flags = (1 - mask_aligned).astype(int)
    flags_path = export_dir / "imputation_flags.csv"
    imputation_flags.to_csv(flags_path)
    export_files["imputation_flags.csv"] = flags_path
    logger.info("Saved imputation flags: %s", flags_path)

    # --- Export confidence weights ---
    # weight = 1 / (1 + uncertainty), capped at [0, 1]
    common_cols_unc = [c for c in imputed_df.columns if c in uncertainty_df.columns]
    if common_cols_unc:
        unc_aligned = uncertainty_df.reindex(
            index=imputed_df.index, columns=common_cols_unc
        ).fillna(0)
        weights = 1.0 / (1.0 + unc_aligned)
        weights = weights.clip(0, 1)
    else:
        weights = pd.DataFrame(1.0, index=imputed_df.index, columns=imputed_df.columns)

    weights_path = export_dir / "confidence_weights.csv"
    weights.to_csv(weights_path)
    export_files["confidence_weights.csv"] = weights_path
    logger.info("Saved confidence weights: %s", weights_path)

    # --- Per-modality CSVs ---
    for mod in registry:
        mod_cols = [f for f in mod.features if f in imputed_df.columns]
        if not mod_cols:
            continue
        mod_df = imputed_df[mod_cols].copy()
        mod_path = per_modality_dir / f"{mod.name}.csv"
        mod_df.to_csv(mod_path)
        export_files[f"per_modality/{mod.name}.csv"] = mod_path
        logger.info("  %s: %s", mod.name, mod_path)

    # --- Optionally copy to GIMAN data directory ---
    giman_data_dir = (
        Path(args.giman_data_dir)
        if args.giman_data_dir
        else GIMAN_ROOT / "data" / "03_prodromal" / "gimin_imputed"
    )
    if giman_data_dir.parent.exists():
        giman_data_dir.mkdir(parents=True, exist_ok=True)
        giman_csv = giman_data_dir / "giman_features_imputed.csv"
        imputed_df.to_csv(giman_csv)
        logger.info("Copied to GIMAN: %s", giman_csv)

    # --- Export manifest ---
    manifest = {
        "export_date_utc": datetime.now(timezone.utc).isoformat(),
        "source": "GIMIN (Graph-based Iterative Multimodal Imputation Network)",
        "version": "0.1.0",
        "num_patients": len(imputed_df),
        "num_features": imputed_df.shape[1],
        "modalities": registry.modality_names,
        "files": {},
    }

    for name, path in export_files.items():
        manifest["files"][name] = {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }

    manifest_path = export_dir / "export_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Export Complete")
    print(f"{'=' * 70}")
    print(f"  Patients:  {manifest['num_patients']}")
    print(f"  Features:  {manifest['num_features']}")
    print(f"  Files exported: {len(export_files)}")
    print(f"  Export dir: {export_dir}")
    if giman_data_dir.exists():
        print(f"  GIMAN dir: {giman_data_dir}")
    print(f"  Manifest:  {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
