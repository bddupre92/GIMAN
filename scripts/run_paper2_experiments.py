#!/usr/bin/env python3
"""Paper 2: Stage-Conditioned Graph-Informed Multimodal Imputation Benchmark.

Runs the full Paper 2 experimental pipeline:

1. Load GIMIN data (33 features, 7 modalities) with NSD-ISS stages
2. Train/evaluate baseline imputation methods (Mean, Median, KNN, MICE)
3. Train/evaluate vanilla GIMIN (no stage conditioning)
4. Train/evaluate Stage-Conditioned GIMIN (stage graph + stage decoder)
5. Ablation: Stage graph only, Stage decoder only
6. Per-stage conformal calibration
7. Downstream experiment: Does stage-aware imputation improve stage prediction?

Outputs:
    outputs/paper2_benchmark/
        imputation_benchmark_results.json
        per_stage_calibration.json
        downstream_comparison.json
        ablation_results.json

Usage:
    python scripts/run_paper2_experiments.py
    python scripts/run_paper2_experiments.py --mask-fractions 0.1 0.2 0.3 0.5
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "GIMImpN_imputation"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("paper2")

# ── Normalization / loss imports (from GIMIN codebase) ──────────────
from gimin.config import GIMINConfig
from gimin.data.scaler import build_scaler_from_config
from gimin.training.losses import GIMINLoss

# ── NSD-ISS stage encoding ──────────────────────────────────────────
STAGE_MAP = {"0": 0, "1": 1, "2B": 2, "3": 3, "4": 4, "unclassified": 5}
STAGE_NAMES = {
    0: "Stage 0",
    1: "Stage 1",
    2: "Stage 2B",
    3: "Stage 3",
    4: "Stage 4",
    5: "Unknown",
}

# GIMIN config: 33 features after dropping zero-variance columns
KEEP_FEATURES = [
    "SEX",
    "AGE_AT_VISIT",
    "NP3TOT",
    "NHY",
    "PIGD_SCORE",
    "TREMOR_SCORE",
    "MCATOT",
    "CAUDATE_L_VOL",
    "CAUDATE_R_VOL",
    "PUTAMEN_L_VOL",
    "PUTAMEN_R_VOL",
    "HIPPOCAMPUS_L_VOL",
    "HIPPOCAMPUS_R_VOL",
    "CAUDATE_L_SBR",
    "CAUDATE_R_SBR",
    "PUTAMEN_L_SBR",
    "PUTAMEN_R_SBR",
    "CAUDATE_ASYMMETRY",
    "PUTAMEN_ASYMMETRY",
    "ALPHA_SYNUCLEIN",
    "TOTAL_TAU",
    "ABETA42",
    "PTAU181",
    "UPSIT_TOTAL",
    "RBD_TOTAL",
    "SCOPA_AUT_TOTAL",
    "ESS_TOTAL",
    "ENTORHINAL_L_CTH",
    "ENTORHINAL_R_CTH",
    "CINGULATE_L_CTH",
    "CINGULATE_R_CTH",
    "PRECENTRAL_L_CTH",
    "PRECENTRAL_R_CTH",
]

MODALITY_DIMS = [2, 5, 6, 6, 4, 4, 6]  # 33 total


def parse_args():
    parser = argparse.ArgumentParser(description="Paper 2 imputation benchmark")
    parser.add_argument(
        "--mask-fractions",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.5],
        help="Artificial missingness fractions for evaluation",
    )
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip classical baselines (Mean, Median, KNN, MICE, MissForest)",
    )
    parser.add_argument(
        "--skip-gimin",
        action="store_true",
        help="Skip GIMIN models (for fast baseline-only runs)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        default=True,
        help="Save model checkpoints after training (default: True)",
    )
    parser.add_argument(
        "--no-save-checkpoints",
        dest="save_checkpoints",
        action="store_false",
        help="Disable model checkpoint saving",
    )
    # P2-Cal.A: calibration-loss retuning for the calibration comparison
    parser.add_argument(
        "--lambda-cal",
        type=float,
        default=0.01,
        help="Weight on the differentiable calibration loss term (default 0.01)",
    )
    parser.add_argument(
        "--cal-warmup-epochs",
        type=int,
        default=50,
        help="Epochs before calibration loss is activated (default 50)",
    )
    parser.add_argument(
        "--lambda-dist",
        type=float,
        default=0.1,
        help="Weight on the distribution-matching loss term (default 0.1)",
    )
    parser.add_argument(
        "--lambda-cross",
        type=float,
        default=0.10,
        help="Weight on the cross-modal consistency loss term (default 0.10)",
    )
    # W1 non-leaky ablation: JSON config overriding default 33-feature schema
    parser.add_argument(
        "--keep-features-json",
        type=str,
        default=None,
        help=(
            "Optional path to a JSON config overriding the default 33-feature "
            "KEEP_FEATURES + MODALITY_DIMS + cross-modal pairs. Used for the "
            "Paper 2 Workstream 1 non-leaky ablation (29-feature schema)."
        ),
    )
    return parser.parse_args()


def apply_feature_override(json_path: str):
    """Override module-level KEEP_FEATURES and MODALITY_DIMS from a JSON config.

    Returns the set of old-index cross-modal pairs that should be dropped
    (as reported in the JSON), so the caller can filter the GIMIN config
    accordingly. Any pair referencing a dropped feature index is removed;
    surviving pairs are re-indexed to match the new 29-feature order.
    """
    global KEEP_FEATURES, MODALITY_DIMS
    import json as _json
    from pathlib import Path as _Path

    cfg_path = _Path(json_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"keep-features JSON not found: {cfg_path}")
    with open(cfg_path) as f:
        override = _json.load(f)

    new_features = override["features"]
    new_dims = override["modality_dims"]
    assert sum(new_dims) == len(new_features), (
        f"modality_dims sum ({sum(new_dims)}) != len(features) ({len(new_features)})"
    )

    old_to_new = {
        old_idx: new_idx
        for new_idx, feat in enumerate(new_features)
        for old_idx, old_feat in enumerate(KEEP_FEATURES)
        if old_feat == feat
    }

    KEEP_FEATURES = list(new_features)
    MODALITY_DIMS = list(new_dims)
    print(
        f"[apply_feature_override] KEEP_FEATURES override active: "
        f"{len(new_features)} features, MODALITY_DIMS={new_dims}"
    )
    return old_to_new


def load_data():
    """Load GIMIN cohort filtered to patients with NSD-ISS stages.

    The full GIMIN parquet has 35,687 patients but only 2,201 have
    NSD-ISS staging. We filter to staged patients for the Paper 2
    experiments, as stage conditioning requires known stages.
    """
    data_dir = PROJECT_ROOT / "GIMImpN_imputation" / "outputs"
    staging_path = PROJECT_ROOT / "data" / "04_staging" / "nsd_iss_staging_results.csv"

    logger.info("Loading GIMIN cohort from %s", data_dir)
    features_df = pd.read_parquet(data_dir / "ppmi_full_cohort.parquet")
    mask_df = pd.read_parquet(data_dir / "missingness_mask.parquet")

    logger.info("Loading NSD-ISS stages from %s", staging_path)
    staging_df = pd.read_csv(staging_path)

    # Encode stages
    staging_df["stage_encoded"] = staging_df["nsd_iss_stage"].astype(str).map(STAGE_MAP)
    staging_df["stage_encoded"] = staging_df["stage_encoded"].fillna(5).astype(int)

    # Filter to patients with known NSD-ISS stages (exclude unclassified)
    staging_valid = staging_df[staging_df["nsd_iss_stage"] != "unclassified"].copy()
    staged_patnos = set(staging_valid["PATNO"].values)

    logger.info("Full cohort: %d patients", len(features_df))
    logger.info(
        "Staged patients: %d (excluding %d unclassified)",
        len(staging_valid),
        len(staging_df) - len(staging_valid),
    )

    # Filter features and mask to staged patients
    staged_mask = features_df.index.isin(staged_patnos)
    features_staged = features_df[staged_mask].copy()
    mask_staged = mask_df[staged_mask].copy()

    # Join stages
    features_staged = features_staged.reset_index()
    features_staged = features_staged.merge(
        staging_valid[["PATNO", "stage_encoded", "nsd_iss_stage"]],
        on="PATNO",
        how="inner",
    )

    logger.info("Matched: %d patients with stages", len(features_staged))
    logger.info(
        "Stage distribution:\n%s",
        features_staged["stage_encoded"].value_counts().sort_index(),
    )

    # Filter to KEEP_FEATURES only
    available = [f for f in KEEP_FEATURES if f in features_staged.columns]
    missing_feats = [f for f in KEEP_FEATURES if f not in features_staged.columns]
    if missing_feats:
        logger.warning("Missing features: %s", missing_feats)

    features_np = features_staged[available].fillna(0).values.astype(np.float32)

    # Re-index mask to match filtered patients
    mask_staged_reindexed = mask_staged.reset_index()
    # Merge to align mask with features_staged order
    mask_merged = features_staged[["PATNO"]].merge(
        mask_staged_reindexed,
        on="PATNO",
        how="inner",
    )
    mask_np = mask_merged[available].values.astype(np.float32)

    stages_np = features_staged["stage_encoded"].values.astype(int)

    logger.info(
        "Data loaded: %d patients, %d features, %.1f%% missing",
        features_np.shape[0],
        features_np.shape[1],
        100 * (1 - mask_np.mean()),
    )

    return features_np, mask_np, stages_np, available


def evaluate_imputation(
    imputed: np.ndarray,
    true_values: np.ndarray,
    eval_mask: np.ndarray,
    stages: np.ndarray,
) -> dict:
    """Compute imputation metrics on artificially masked positions.

    Args:
        imputed: Imputed feature matrix (N, F).
        true_values: Ground-truth values (N, F).
        eval_mask: Binary mask (1 = position was artificially masked) (N, F).
        stages: Stage indices (N,).

    Returns:
        Dict with RMSE, MAE, R², and per-stage metrics.
    """
    eval_positions = eval_mask > 0

    if eval_positions.sum() == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    errors = imputed[eval_positions] - true_values[eval_positions]
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))

    ss_res = np.sum(errors**2)
    ss_tot = np.sum(
        (true_values[eval_positions] - true_values[eval_positions].mean()) ** 2
    )
    r2 = float(1 - ss_res / max(ss_tot, 1e-10))

    # Per-feature RMSE
    per_feature_rmse = []
    F = imputed.shape[1]
    for f in range(F):
        f_mask = eval_mask[:, f] > 0
        if f_mask.sum() > 0:
            f_err = imputed[f_mask, f] - true_values[f_mask, f]
            per_feature_rmse.append(float(np.sqrt(np.mean(f_err**2))))
        else:
            per_feature_rmse.append(float("nan"))

    # Per-stage RMSE
    per_stage_rmse = {}
    for stage_id in np.unique(stages):
        s_mask = stages == stage_id
        s_eval = eval_mask[s_mask]
        if s_eval.sum() > 0:
            s_err = imputed[s_mask][s_eval > 0] - true_values[s_mask][s_eval > 0]
            per_stage_rmse[int(stage_id)] = float(np.sqrt(np.mean(s_err**2)))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "per_feature_rmse": per_feature_rmse,
        "per_stage_rmse": per_stage_rmse,
        "n_eval_positions": int(eval_positions.sum()),
    }


def create_artificial_mask(
    mask: np.ndarray,
    fraction: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create artificial missingness by hiding a fraction of observed values.

    Args:
        mask: Original observation mask (N, F). 1 = observed.
        fraction: Fraction of observed values to hide.
        seed: Random seed.

    Returns:
        Tuple of (corrupted_mask, eval_mask).
        corrupted_mask: New mask with additional artificial missingness.
        eval_mask: Binary mask marking artificially hidden positions.
    """
    rng = np.random.RandomState(seed)
    observed = mask > 0
    n_observed = observed.sum()
    n_to_hide = int(n_observed * fraction)

    # Get indices of observed values
    obs_indices = np.argwhere(observed)
    selected = rng.choice(len(obs_indices), size=n_to_hide, replace=False)
    hidden_indices = obs_indices[selected]

    eval_mask = np.zeros_like(mask)
    eval_mask[hidden_indices[:, 0], hidden_indices[:, 1]] = 1.0

    corrupted_mask = mask.copy()
    corrupted_mask[eval_mask > 0] = 0.0

    return corrupted_mask, eval_mask


# ── Baseline imputation methods ────────────────────────────────────


def run_baselines(
    features: np.ndarray,
    mask: np.ndarray,
    corrupted_mask: np.ndarray,
    eval_mask: np.ndarray,
    stages: np.ndarray,
) -> dict:
    """Run classical and deep learning imputation baselines."""
    from gimin.evaluation.baselines import (
        GAINBaseline,
        KNNBaseline,
        MeanBaseline,
        MedianBaseline,
        MICEBaseline,
        MissForestBaseline,
        MIWAEBaseline,
        SAITSBaseline,
    )

    results = {}
    true_values = features.copy()

    # Prepare corrupted features (zero out hidden values)
    corrupted_features = features * corrupted_mask

    for name, baseline in [
        # Classical baselines
        ("Mean", MeanBaseline()),
        ("Median", MedianBaseline()),
        ("KNN", KNNBaseline(k=10, weights="distance")),
        ("MICE", MICEBaseline(max_iter=10, n_estimators=50)),
        ("MissForest", MissForestBaseline(max_iter=10, n_estimators=100)),
        # Deep learning baselines
        ("GAIN", GAINBaseline(n_epochs=100, batch_size=128, hint_rate=0.9)),
        (
            "SAITS",
            SAITSBaseline(
                n_layers=2, d_model=64, n_heads=4, d_ffn=64, epochs=50, patience=5
            ),
        ),
        ("MIWAE", MIWAEBaseline(n_epochs=100, batch_size=128)),
    ]:
        logger.info("Running baseline: %s", name)
        t0 = time.time()
        try:
            imputed = baseline.fit_transform(corrupted_features, corrupted_mask)
            elapsed = time.time() - t0
            metrics = evaluate_imputation(imputed, true_values, eval_mask, stages)
            metrics["time_seconds"] = round(elapsed, 2)
            results[name] = metrics
            logger.info(
                "  %s: RMSE=%.4f, MAE=%.4f, R²=%.4f (%.1fs)",
                name,
                metrics["rmse"],
                metrics["mae"],
                metrics["r2"],
                elapsed,
            )
        except Exception as e:
            logger.error("  %s failed: %s", name, e)
            results[name] = {"error": str(e)}

    return results


# ── GIMIN training helpers ──────────────────────────────────────────


def build_gimin_config(old_to_new_index=None):
    """Build a GIMINConfig matching the Paper 2 feature set.

    If old_to_new_index is provided (W1 non-leaky ablation), filter the
    default 33-feature cross-modal pairs: drop any pair referencing a feature
    that is not in old_to_new_index, and re-index surviving pair endpoints
    to the new (e.g., 29-feature) index space. Also rebuild binary-feature
    indices and modality definitions on the reduced schema.
    """
    cfg = GIMINConfig()
    if old_to_new_index is None:
        return cfg

    # Filter + re-index cross-modal pairs
    filtered_pairs = []
    for pair in cfg.cross_modal_pairs:
        if pair[0] in old_to_new_index and pair[1] in old_to_new_index:
            filtered_pairs.append(
                [old_to_new_index[pair[0]], old_to_new_index[pair[1]]]
            )
    dropped = len(cfg.cross_modal_pairs) - len(filtered_pairs)
    print(
        f"[build_gimin_config] Cross-modal pairs: "
        f"{len(cfg.cross_modal_pairs)} -> {len(filtered_pairs)} ({dropped} dropped)"
    )
    cfg.cross_modal_pairs = filtered_pairs

    # Rebuild binary_feature_indices (SEX is the only binary in the 33 schema
    # at old index 0; stays at new index 0 since SEX is first in non-leaky too)
    if cfg.binary_feature_indices:
        cfg.binary_feature_indices = [
            old_to_new_index[i]
            for i in cfg.binary_feature_indices
            if i in old_to_new_index
        ]

    return cfg


def train_gimin_model(
    model,
    features_t,
    mask_t,
    corrupted_mask_t,
    edge_index,
    edge_weight,
    overlap_frac,
    stages_t=None,
    epochs=200,
    lr=1e-3,
    modality_dims=None,
    is_stage_conditioned=False,
    cross_modal_pairs=None,
    binary_feature_indices=None,
    lambda_dist: float = 0.1,
    lambda_cross: float = 0.10,
    lambda_cal: float = 0.01,
    cal_warmup_epochs: int = 50,
):
    """Train a GIMIN model with self-supervised masking and composite loss.

    Uses the full GIMINLoss (reconstruction + distribution + cross-modal +
    calibration) from the GIMIN codebase. Features MUST be pre-normalized
    via ModalityAwareScaler before calling this function.

    Returns trained model and training history.
    """
    import torch

    if modality_dims is None:
        modality_dims = MODALITY_DIMS

    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
    )

    # Build composite loss with all 4 terms
    loss_fn = GIMINLoss(
        lambda_dist=lambda_dist,
        lambda_cross=lambda_cross,
        lambda_cal=lambda_cal,
        cal_warmup_epochs=cal_warmup_epochs,
        cross_modal_pairs=cross_modal_pairs,
        binary_feature_indices=binary_feature_indices or [0],
    ).to(device)

    features_t = features_t.to(device)
    mask_t = mask_t.to(device)
    corrupted_mask_t = corrupted_mask_t.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    overlap_frac = overlap_frac.to(device)
    if stages_t is not None:
        stages_t = stages_t.to(device)

    # Target mask: positions that were observed but artificially masked
    target_mask = mask_t - corrupted_mask_t
    target_mask = target_mask.clamp(min=0)

    best_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        if is_stage_conditioned:
            output = model(
                features=features_t * corrupted_mask_t,
                mask=corrupted_mask_t,
                edge_index=edge_index,
                edge_weight=edge_weight,
                overlap_frac=overlap_frac,
                stage_ids=stages_t,
            )
        else:
            output = model(
                features=features_t * corrupted_mask_t,
                mask=corrupted_mask_t,
                edge_index=edge_index,
                edge_weight=edge_weight,
                overlap_frac=overlap_frac,
            )

        # Compute composite loss (recon + dist + cross-modal + calibration)
        losses = loss_fn(
            model_output=output,
            true_values=features_t,
            target_mask=target_mask,
            observed_mask=corrupted_mask_t,
            epoch=epoch,
        )

        loss = losses["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss.item())

        history.append(
            {
                "epoch": epoch,
                "loss": round(loss.item(), 6),
                "recon": round(losses["reconstruction"].item(), 6),
                "dist": round(losses["distribution"].item(), 6),
                "cross": round(losses["cross_modal"].item(), 6),
                "cal": round(losses["calibration"].item(), 6),
            }
        )

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 40:
            logger.info(
                "Early stopping at epoch %d (best loss: %.6f)", epoch, best_loss
            )
            break

        if (epoch + 1) % 50 == 0:
            logger.info(
                "  Epoch %d/%d: loss=%.4f (recon=%.4f, dist=%.4f, cross=%.4f, cal=%.4f)",
                epoch + 1,
                epochs,
                loss.item(),
                losses["reconstruction"].item(),
                losses["distribution"].item(),
                losses["cross_modal"].item(),
                losses["calibration"].item(),
            )

    return model, history


def evaluate_gimin_model(
    model,
    features_norm_t,
    mask_t,
    corrupted_mask_t,
    eval_mask,
    edge_index,
    edge_weight,
    overlap_frac,
    stages,
    features_original,
    scaler=None,
    stages_t=None,
    is_stage_conditioned=False,
    mc_samples=20,
):
    """Evaluate a trained GIMIN model with MC dropout uncertainty.

    Predictions are made in normalized space and then inverse-transformed
    back to the original clinical scale for fair comparison with baselines.

    Args:
        features_norm_t: Normalized features tensor (N, F).
        mask_t: Observation mask tensor (N, F).
        corrupted_mask_t: Corrupted mask tensor (N, F).
        eval_mask: Numpy eval mask (N, F) — positions to evaluate.
        edge_index, edge_weight, overlap_frac: Graph tensors.
        stages: Numpy stage array (N,).
        features_original: ORIGINAL (unnormalized) features numpy (N, F).
        scaler: Fitted ModalityAwareScaler for inverse transform.
        stages_t: Stage tensor (N,) for stage-conditioned models.
        is_stage_conditioned: Whether model expects stage_ids.
        mc_samples: Number of MC dropout forward passes.

    Returns:
        Tuple of (metrics, imputed_original, mean_pred_original, total_std_original).
    """
    import torch

    model.eval()
    device = next(model.parameters()).device

    # Enable MC dropout
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
            module.train()

    means_list = []
    logvars_list = []

    with torch.no_grad():
        for _ in range(mc_samples):
            if is_stage_conditioned:
                output = model(
                    features=features_norm_t * corrupted_mask_t,
                    mask=corrupted_mask_t,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    overlap_frac=overlap_frac,
                    stage_ids=stages_t,
                )
            else:
                output = model(
                    features=features_norm_t * corrupted_mask_t,
                    mask=corrupted_mask_t,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    overlap_frac=overlap_frac,
                )
            means_list.append(output["pred_mean"].cpu().numpy())
            logvars_list.append(output["pred_log_var"].cpu().numpy())

    model.eval()

    # Aggregate MC samples (in normalized space)
    means_stack = np.stack(means_list, axis=0)
    logvars_stack = np.stack(logvars_list, axis=0)

    mean_pred_norm = means_stack.mean(axis=0)
    epistemic_var_norm = means_stack.var(axis=0)
    aleatoric_var_norm = np.exp(logvars_stack).mean(axis=0)

    # Apply sigmoid for binary features (SEX, index 0) in normalized space
    mean_for_blend = mean_pred_norm.copy()
    mean_for_blend[:, 0] = 1.0 / (1.0 + np.exp(-mean_for_blend[:, 0]))

    # Blend in normalized space: keep observed values, fill missing
    corrupted_np = corrupted_mask_t.cpu().numpy()
    features_norm_np = features_norm_t.cpu().numpy()
    imputed_norm = features_norm_np * corrupted_np + mean_for_blend * (
        1.0 - corrupted_np
    )

    # ── Inverse-transform predictions to original scale ──────────
    if scaler is not None:
        mask_np = mask_t.cpu().numpy()
        # Inverse-transform the imputed values back to original scale
        imputed_original = scaler.inverse_transform(
            imputed_norm,
            mask=None,  # transform ALL positions
        ).numpy()
        # Inverse-transform the mean predictions for conformal calibration
        mean_pred_original = scaler.inverse_transform(mean_pred_norm, mask=None).numpy()
        # Inverse-transform variance to original scale
        total_var_norm = epistemic_var_norm + aleatoric_var_norm
        total_var_original = scaler.inverse_transform_variance(
            total_var_norm,
            mean_normalized=mean_pred_norm,
            mask=None,
        ).numpy()
        total_std_original = np.sqrt(np.maximum(total_var_original, 1e-12))

        # For epistemic/aleatoric breakdown
        epistemic_var_orig = scaler.inverse_transform_variance(
            epistemic_var_norm, mean_normalized=mean_pred_norm, mask=None
        ).numpy()
        aleatoric_var_orig = scaler.inverse_transform_variance(
            aleatoric_var_norm, mean_normalized=mean_pred_norm, mask=None
        ).numpy()
    else:
        # No scaler — predictions already in original scale
        imputed_original = imputed_norm
        mean_pred_original = mean_pred_norm
        total_std_original = np.sqrt(epistemic_var_norm + aleatoric_var_norm)
        epistemic_var_orig = epistemic_var_norm
        aleatoric_var_orig = aleatoric_var_norm

    # Compute metrics against ORIGINAL-scale features
    metrics = evaluate_imputation(
        imputed_original, features_original, eval_mask, stages
    )
    metrics["epistemic_std_mean"] = float(
        np.sqrt(np.maximum(epistemic_var_orig, 0)).mean()
    )
    metrics["aleatoric_std_mean"] = float(
        np.sqrt(np.maximum(aleatoric_var_orig, 0)).mean()
    )
    metrics["total_std_mean"] = float(total_std_original.mean())

    return metrics, imputed_original, mean_pred_original, total_std_original


def _save_checkpoint(model, run_dir, frac, run_idx, model_name):
    """Save a model checkpoint to the run directory.

    Creates: run_dir/checkpoints/frac{frac}_run{run_idx}_{model_name}.pt
    """
    import torch

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fname = f"frac{frac:.1f}_run{run_idx}_{model_name}.pt"
    path = ckpt_dir / fname
    torch.save(model.state_dict(), path)
    logger.info("  Checkpoint saved: %s", path)
    return str(path)


def _save_training_history(history, run_dir, frac, run_idx, model_name):
    """Save training loss history for a model run.

    Creates: run_dir/training_history/frac{frac}_run{run_idx}_{model_name}.json
    """
    hist_dir = run_dir / "training_history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    fname = f"frac{frac:.1f}_run{run_idx}_{model_name}.json"
    path = hist_dir / fname
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("  History saved: %s", path)
    return str(path)


def _save_incremental_results(all_results, summary, run_dir, output_dir):
    """Save results incrementally — both to timestamped run dir AND legacy path.

    This means partial results survive if the process crashes mid-run.
    The legacy path (output_dir/imputation_benchmark_results.json) is
    written for backward compatibility but the authoritative copy lives
    in the timestamped run directory.
    """
    payload = {"summary": summary, "raw": all_results}
    serializer = lambda x: (  # noqa: E731
        float(x) if isinstance(x, (np.floating, np.integer)) else str(x)
    )

    # Authoritative copy in timestamped run dir (NEVER overwritten by another run)
    run_path = run_dir / "imputation_benchmark_results.json"
    with open(run_path, "w") as f:
        json.dump(payload, f, indent=2, default=serializer)

    # Legacy copy (latest results — overwritten each run but timestamped dir is safe)
    legacy_path = output_dir / "imputation_benchmark_results.json"
    with open(legacy_path, "w") as f:
        json.dump(payload, f, indent=2, default=serializer)

    logger.info("  Results saved → %s", run_path)


def main():
    import datetime

    args = parse_args()
    np.random.seed(args.seed)

    # W1 non-leaky ablation: override feature schema before load_data()
    old_to_new_index = None
    if args.keep_features_json:
        old_to_new_index = apply_feature_override(args.keep_features_json)

    # ── Timestamped output directory ───────────────────────────────
    # Each run gets its own directory that is NEVER overwritten.
    output_dir = PROJECT_ROOT / "outputs" / "paper2_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save run configuration for reproducibility
    run_config = {
        "timestamp": datetime.datetime.now().isoformat(),
        "run_name": run_name,
        "mask_fractions": args.mask_fractions,
        "num_runs": args.num_runs,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "skip_baselines": args.skip_baselines,
        "skip_gimin": args.skip_gimin,
        "save_checkpoints": args.save_checkpoints,
        "keep_features_json": args.keep_features_json,
        "keep_features_count": len(KEEP_FEATURES),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    logger.info("Run directory: %s", run_dir)

    print("=" * 70)
    print("Paper 2: Stage-Conditioned Imputation Benchmark")
    print(f"  Run: {run_name}")
    print(f"  Output: {run_dir}")
    print("=" * 70)

    # ── 1. Load data ────────────────────────────────────────────────
    features, mask, stages, feature_names = load_data()
    N, F = features.shape

    print(f"\n  Patients (visits): {N}")
    print(f"  Features: {F}")
    print(f"  Missing: {100 * (1 - mask.mean()):.1f}%")
    print(
        f"  Stages: {dict(zip(*np.unique(stages, return_counts=True), strict=False))}"
    )

    # ── 2. Data is already filtered to staged patients ──────────────
    # load_data() filtered to 2,197 patients with known NSD-ISS stages.
    # This is a manageable size for O(N²) graph construction.
    features_bl = features
    mask_bl = mask
    stages_bl = stages
    N_bl = N

    print("\n  Stage distribution:")
    for s, name in STAGE_NAMES.items():
        count = (stages_bl == s).sum()
        if count > 0:
            print(f"    {name}: {count} ({100 * count / N_bl:.1f}%)")

    # ── 3. Run experiments per mask fraction ───────────────────────
    all_results = {}

    for frac in args.mask_fractions:
        print(f"\n{'─' * 70}")
        print(f"Mask fraction: {frac:.0%}")
        print(f"{'─' * 70}")

        frac_results = {}

        for run in range(args.num_runs):
            seed = args.seed + run
            corrupted_mask, eval_mask = create_artificial_mask(
                mask_bl,
                frac,
                seed=seed,
            )

            run_results = {}

            # ── 3a. Classical baselines ────────────────────────────
            if not args.skip_baselines:
                logger.info("Run %d/%d: Classical baselines", run + 1, args.num_runs)
                baseline_results = run_baselines(
                    features_bl,
                    mask_bl,
                    corrupted_mask,
                    eval_mask,
                    stages_bl,
                )
                for name, metrics in baseline_results.items():
                    run_results[name] = metrics

            # ── 3b. GIMIN models ──────────────────────────────────
            if not args.skip_gimin:
                import torch

                from giman_pipeline.imputation.stage_conditioned_gimin import (
                    StageConditionedGIMIN,
                    VanillaGIMIN,
                )
                from giman_pipeline.imputation.stage_graph_builder import (
                    StageAwareGraphBuilder,
                )

                torch.manual_seed(seed)

                # ── Build modality-aware scaler and normalize ─────
                cfg = build_gimin_config(old_to_new_index=old_to_new_index)
                scaler = build_scaler_from_config(cfg)

                # Fit scaler on CORRUPTED observation (what the model sees)
                features_t_raw = torch.tensor(features_bl, dtype=torch.float32)
                mask_t = torch.tensor(mask_bl, dtype=torch.float32)
                corrupted_mask_t = torch.tensor(corrupted_mask, dtype=torch.float32)

                # Fit scaler on ALL observed values (mask_bl), NOT corrupted
                # mask, so we learn the true distribution.
                scaler.fit(features_t_raw, mask_t)

                # Transform features to normalized space
                features_norm_t = scaler.transform(features_t_raw, mask_t)
                # features_norm_t has normalized values where observed, 0 where missing

                stages_t = torch.tensor(stages_bl, dtype=torch.long)

                logger.info(
                    "Features normalized: mean=%.4f, std=%.4f (should be ~0, ~1)",
                    features_norm_t[mask_t.bool()].mean().item(),
                    features_norm_t[mask_t.bool()].std().item(),
                )

                # ── Build graphs on NORMALIZED features ──────────
                features_norm_np = features_norm_t.numpy()

                logger.info("Building vanilla graph (on normalized features)...")
                vanilla_builder = StageAwareGraphBuilder(
                    k_neighbors=15,
                    min_overlap=3,
                    stage_affinity_beta=0.0,
                )
                vanilla_graph = vanilla_builder.build_full_graph(
                    features_norm_np * corrupted_mask.astype(np.float32),
                    corrupted_mask,
                    stages=None,
                )

                logger.info("Building stage-aware graph (on normalized features)...")
                stage_builder = StageAwareGraphBuilder(
                    k_neighbors=15,
                    min_overlap=3,
                    stage_affinity_beta=0.3,
                )
                stage_graph = stage_builder.build_full_graph(
                    features_norm_np * corrupted_mask.astype(np.float32),
                    corrupted_mask,
                    stages=stages_bl,
                )

                # Cross-modal pairs from config
                cross_modal_pairs = cfg.cross_modal_pairs
                binary_indices = cfg.binary_feature_indices

                # Normalized mask tensors for training
                # Mask for NORMALIZED features: same as original mask
                mask_norm_t = mask_t.clone()

                # ── Model A: Vanilla GIMIN (no stage) ─────────────
                logger.info("Training Vanilla GIMIN (with normalization)...")
                t0 = time.time()
                vanilla_model = VanillaGIMIN(
                    modality_dims=MODALITY_DIMS,
                    embed_dim=64,
                    num_gnn_layers=3,
                    num_heads=4,
                    mc_dropout=0.1,
                )
                vanilla_model, v_history = train_gimin_model(
                    vanilla_model,
                    features_norm_t,
                    mask_norm_t,
                    corrupted_mask_t,
                    vanilla_graph["edge_index"],
                    vanilla_graph["edge_weight"],
                    vanilla_graph["overlap_frac"],
                    epochs=args.epochs,
                    lr=args.lr,
                    is_stage_conditioned=False,
                    cross_modal_pairs=cross_modal_pairs,
                    binary_feature_indices=binary_indices,
                    lambda_dist=args.lambda_dist,
                    lambda_cross=args.lambda_cross,
                    lambda_cal=args.lambda_cal,
                    cal_warmup_epochs=args.cal_warmup_epochs,
                )
                v_metrics, v_imputed, v_mean, v_std = evaluate_gimin_model(
                    vanilla_model,
                    features_norm_t,
                    mask_norm_t,
                    corrupted_mask_t,
                    eval_mask,
                    vanilla_graph["edge_index"],
                    vanilla_graph["edge_weight"],
                    vanilla_graph["overlap_frac"],
                    stages_bl,
                    features_original=features_bl,
                    scaler=scaler,
                    is_stage_conditioned=False,
                )
                v_metrics["time_seconds"] = round(time.time() - t0, 2)
                v_metrics["n_epochs"] = len(v_history)
                run_results["GIMIN_Vanilla"] = v_metrics
                logger.info(
                    "  GIMIN_Vanilla: RMSE=%.4f, R²=%.4f (%.1fs, %d epochs)",
                    v_metrics["rmse"],
                    v_metrics["r2"],
                    v_metrics["time_seconds"],
                    len(v_history),
                )
                if args.save_checkpoints:
                    _save_checkpoint(vanilla_model, run_dir, frac, run, "GIMIN_Vanilla")
                    _save_training_history(
                        v_history, run_dir, frac, run, "GIMIN_Vanilla"
                    )

                # ── Model B: Stage-Conditioned GIMIN (full) ───────
                logger.info("Training Stage-Conditioned GIMIN (with normalization)...")
                t0 = time.time()
                stage_model = StageConditionedGIMIN(
                    modality_dims=MODALITY_DIMS,
                    embed_dim=64,
                    num_gnn_layers=3,
                    num_heads=4,
                    mc_dropout=0.1,
                    num_stages=6,
                    stage_embed_dim=16,
                    use_stage_attention_bias=True,
                )
                stage_model, s_history = train_gimin_model(
                    stage_model,
                    features_norm_t,
                    mask_norm_t,
                    corrupted_mask_t,
                    stage_graph["edge_index"],
                    stage_graph["edge_weight"],
                    stage_graph["overlap_frac"],
                    stages_t=stages_t,
                    epochs=args.epochs,
                    lr=args.lr,
                    is_stage_conditioned=True,
                    cross_modal_pairs=cross_modal_pairs,
                    binary_feature_indices=binary_indices,
                    lambda_dist=args.lambda_dist,
                    lambda_cross=args.lambda_cross,
                    lambda_cal=args.lambda_cal,
                    cal_warmup_epochs=args.cal_warmup_epochs,
                )
                s_metrics, s_imputed, s_mean, s_std = evaluate_gimin_model(
                    stage_model,
                    features_norm_t,
                    mask_norm_t,
                    corrupted_mask_t,
                    eval_mask,
                    stage_graph["edge_index"],
                    stage_graph["edge_weight"],
                    stage_graph["overlap_frac"],
                    stages_bl,
                    features_original=features_bl,
                    scaler=scaler,
                    stages_t=stages_t,
                    is_stage_conditioned=True,
                )
                s_metrics["time_seconds"] = round(time.time() - t0, 2)
                s_metrics["n_epochs"] = len(s_history)
                s_metrics["stage_graph_stats"] = stage_graph.get("stage_stats", {})
                run_results["GIMIN_StageConditioned"] = s_metrics
                logger.info(
                    "  GIMIN_StageConditioned: RMSE=%.4f, R²=%.4f (%.1fs, %d epochs)",
                    s_metrics["rmse"],
                    s_metrics["r2"],
                    s_metrics["time_seconds"],
                    len(s_history),
                )
                if args.save_checkpoints:
                    _save_checkpoint(
                        stage_model, run_dir, frac, run, "GIMIN_StageConditioned"
                    )
                    _save_training_history(
                        s_history, run_dir, frac, run, "GIMIN_StageConditioned"
                    )

                # ── Model C: Stage Graph Only (ablation) ──────────
                logger.info(
                    "Training GIMIN + Stage Graph only (ablation, normalized)..."
                )
                t0 = time.time()
                ablation_graph_model = VanillaGIMIN(
                    modality_dims=MODALITY_DIMS,
                    embed_dim=64,
                    num_gnn_layers=3,
                    num_heads=4,
                    mc_dropout=0.1,
                )
                ablation_graph_model, ag_history = train_gimin_model(
                    ablation_graph_model,
                    features_norm_t,
                    mask_norm_t,
                    corrupted_mask_t,
                    stage_graph["edge_index"],  # Use STAGE graph
                    stage_graph["edge_weight"],
                    stage_graph["overlap_frac"],
                    epochs=args.epochs,
                    lr=args.lr,
                    is_stage_conditioned=False,
                    cross_modal_pairs=cross_modal_pairs,
                    binary_feature_indices=binary_indices,
                    lambda_dist=args.lambda_dist,
                    lambda_cross=args.lambda_cross,
                    lambda_cal=args.lambda_cal,
                    cal_warmup_epochs=args.cal_warmup_epochs,
                )
                ag_metrics, _, _, _ = evaluate_gimin_model(
                    ablation_graph_model,
                    features_norm_t,
                    mask_norm_t,
                    corrupted_mask_t,
                    eval_mask,
                    stage_graph["edge_index"],
                    stage_graph["edge_weight"],
                    stage_graph["overlap_frac"],
                    stages_bl,
                    features_original=features_bl,
                    scaler=scaler,
                    is_stage_conditioned=False,
                )
                ag_metrics["time_seconds"] = round(time.time() - t0, 2)
                run_results["GIMIN_StageGraphOnly"] = ag_metrics
                logger.info(
                    "  GIMIN_StageGraphOnly: RMSE=%.4f, R²=%.4f",
                    ag_metrics["rmse"],
                    ag_metrics["r2"],
                )
                if args.save_checkpoints:
                    _save_checkpoint(
                        ablation_graph_model, run_dir, frac, run, "GIMIN_StageGraphOnly"
                    )
                    _save_training_history(
                        ag_history, run_dir, frac, run, "GIMIN_StageGraphOnly"
                    )

                # ── Model D: Stage Decoder Only (ablation) ────────
                logger.info(
                    "Training GIMIN + Stage Decoder only (ablation, normalized)..."
                )
                t0 = time.time()
                ablation_decoder_model = StageConditionedGIMIN(
                    modality_dims=MODALITY_DIMS,
                    embed_dim=64,
                    num_gnn_layers=3,
                    num_heads=4,
                    mc_dropout=0.1,
                    num_stages=6,
                    stage_embed_dim=16,
                    use_stage_attention_bias=True,
                )
                ablation_decoder_model, ad_history = train_gimin_model(
                    ablation_decoder_model,
                    features_norm_t,
                    mask_norm_t,
                    corrupted_mask_t,
                    vanilla_graph["edge_index"],  # Use VANILLA graph
                    vanilla_graph["edge_weight"],
                    vanilla_graph["overlap_frac"],
                    stages_t=stages_t,
                    epochs=args.epochs,
                    lr=args.lr,
                    is_stage_conditioned=True,
                    cross_modal_pairs=cross_modal_pairs,
                    binary_feature_indices=binary_indices,
                    lambda_dist=args.lambda_dist,
                    lambda_cross=args.lambda_cross,
                    lambda_cal=args.lambda_cal,
                    cal_warmup_epochs=args.cal_warmup_epochs,
                )
                ad_metrics, _, _, _ = evaluate_gimin_model(
                    ablation_decoder_model,
                    features_norm_t,
                    mask_norm_t,
                    corrupted_mask_t,
                    eval_mask,
                    vanilla_graph["edge_index"],
                    vanilla_graph["edge_weight"],
                    vanilla_graph["overlap_frac"],
                    stages_bl,
                    features_original=features_bl,
                    scaler=scaler,
                    stages_t=stages_t,
                    is_stage_conditioned=True,
                )
                ad_metrics["time_seconds"] = round(time.time() - t0, 2)
                run_results["GIMIN_StageDecoderOnly"] = ad_metrics
                logger.info(
                    "  GIMIN_StageDecoderOnly: RMSE=%.4f, R²=%.4f",
                    ad_metrics["rmse"],
                    ad_metrics["r2"],
                )
                if args.save_checkpoints:
                    _save_checkpoint(
                        ablation_decoder_model,
                        run_dir,
                        frac,
                        run,
                        "GIMIN_StageDecoderOnly",
                    )
                    _save_training_history(
                        ad_history, run_dir, frac, run, "GIMIN_StageDecoderOnly"
                    )

                # ── Conformal calibration ─────────────────────────
                # Per-feature conformal: each feature gets its own quantile
                # based on absolute residuals. This avoids the Jacobian
                # blow-up issue with log-normal inverse variance and produces
                # feature-appropriate interval widths.
                if run == 0:
                    from giman_pipeline.imputation.conformal_imputation import (
                        ConformalImputation,
                    )

                    logger.info(
                        "Running per-feature stage-conditioned conformal calibration..."
                    )

                    # --- Per-feature conformal (recommended) ---
                    conformal_pf = ConformalImputation(
                        coverage_target=0.90,
                        mode="per_feature",
                    )
                    conformal_result_pf = conformal_pf.calibrate_per_stage(
                        predicted_means=s_mean,
                        true_values=features_bl,
                        mask=corrupted_mask,
                        stages=stages_bl,
                        predicted_stds=None,
                        stage_names=STAGE_NAMES,
                    )

                    # --- Also try normalized conformal in original space ---
                    conformal_norm = ConformalImputation(
                        coverage_target=0.90,
                        mode="normalized",
                    )
                    conformal_result_norm = conformal_norm.calibrate_per_stage(
                        predicted_means=s_mean,
                        true_values=features_bl,
                        mask=corrupted_mask,
                        stages=stages_bl,
                        predicted_stds=s_std,
                        stage_names=STAGE_NAMES,
                    )

                    # --- Calibration curves (coverage at 5 levels) ---
                    cal_curves = conformal_pf.evaluate_coverage(
                        predicted_means=s_mean,
                        true_values=features_bl,
                        mask=corrupted_mask,
                        stages=stages_bl,
                        coverages=[0.50, 0.70, 0.80, 0.90, 0.95],
                    )

                    # Save per-feature conformal results
                    def _serialize_conformal(result, label="per_feature"):
                        d = {
                            "mode": label,
                            "marginal": {
                                "coverage": result.marginal.observed_coverage,
                                "nmpiw": result.marginal.nmpiw,
                                "mean_width": result.marginal.mean_interval_width,
                                "median_width": result.marginal.median_interval_width,
                                "n_cal": result.marginal.n_calibration,
                            },
                            "per_stage": {},
                        }
                        if result.marginal.per_feature_coverage is not None:
                            d["marginal"]["per_feature_coverage"] = (
                                result.marginal.per_feature_coverage.tolist()
                            )
                        if result.marginal.per_feature_width is not None:
                            d["marginal"]["per_feature_width"] = (
                                result.marginal.per_feature_width.tolist()
                            )
                        for s_id, s_res in result.per_stage.items():
                            s_dict = {
                                "coverage": s_res.observed_coverage,
                                "nmpiw": s_res.nmpiw,
                                "mean_width": s_res.mean_interval_width,
                                "median_width": s_res.median_interval_width,
                                "n_cal": s_res.n_calibration,
                            }
                            if s_res.per_feature_coverage is not None:
                                s_dict["per_feature_coverage"] = (
                                    s_res.per_feature_coverage.tolist()
                                )
                            d["per_stage"][STAGE_NAMES.get(s_id, str(s_id))] = s_dict
                        return d

                    conformal_dict = {
                        "per_feature": _serialize_conformal(
                            conformal_result_pf, "per_feature"
                        ),
                        "normalized": _serialize_conformal(
                            conformal_result_norm, "normalized"
                        ),
                        "calibration_curves": {
                            str(k): {str(kk): vv for kk, vv in v.items()}
                            for k, v in cal_curves.items()
                        },
                    }

                    # Save to BOTH timestamped run dir AND legacy location
                    conformal_dir = run_dir / "conformal"
                    conformal_dir.mkdir(parents=True, exist_ok=True)
                    conformal_serializer = (
                        lambda x: float(x)
                        if isinstance(x, (np.floating, np.integer))
                        else str(x)
                    )  # noqa: E731

                    # Authoritative copy in timestamped run dir
                    conformal_run_path = (
                        conformal_dir / f"conformal_frac{frac:.1f}.json"
                    )
                    with open(conformal_run_path, "w") as f:
                        json.dump(
                            conformal_dict, f, indent=2, default=conformal_serializer
                        )

                    # Legacy copy for backward compatibility
                    conformal_legacy_path = (
                        output_dir / f"conformal_frac{frac:.1f}.json"
                    )
                    with open(conformal_legacy_path, "w") as f:
                        json.dump(
                            conformal_dict, f, indent=2, default=conformal_serializer
                        )

                    logger.info(
                        "Conformal results saved to %s (+ legacy)", conformal_run_path
                    )

            # Store run results
            frac_key = f"frac_{frac:.1f}"
            if frac_key not in frac_results:
                frac_results[frac_key] = {}

            for model_name, metrics in run_results.items():
                if model_name not in frac_results[frac_key]:
                    frac_results[frac_key][model_name] = []
                frac_results[frac_key][model_name].append(metrics)

        all_results[f"frac_{frac:.1f}"] = frac_results[f"frac_{frac:.1f}"]

        # ── INCREMENTAL SAVE after each fraction ──────────────────
        # This protects against data loss if the process crashes mid-run
        _incremental_summary = {}
        for fk, fd in all_results.items():
            _incremental_summary[fk] = {}
            for mn, rns in fd.items():
                rmses = [
                    r["rmse"]
                    for r in rns
                    if "rmse" in r and not np.isnan(r.get("rmse", float("nan")))
                ]
                r2s = [
                    r["r2"]
                    for r in rns
                    if "r2" in r and not np.isnan(r.get("r2", float("nan")))
                ]
                if rmses:
                    _incremental_summary[fk][mn] = {
                        "rmse_mean": round(float(np.mean(rmses)), 4),
                        "rmse_std": round(float(np.std(rmses)), 4),
                        "r2_mean": round(float(np.mean(r2s)), 4),
                        "r2_std": round(float(np.std(r2s)), 4),
                        "n_runs": len(rmses),
                    }
        _save_incremental_results(
            all_results, _incremental_summary, run_dir, output_dir
        )
        logger.info("Incremental save after frac=%.1f complete", frac)

    # ── 4. Final aggregation & summary ─────────────────────────────
    print(f"\n{'=' * 70}")
    print("Results Summary")
    print(f"{'=' * 70}")

    summary = {}
    for frac_key, frac_data in all_results.items():
        print(f"\n{frac_key}:")
        summary[frac_key] = {}
        for model_name, runs in frac_data.items():
            rmses = [
                r["rmse"]
                for r in runs
                if "rmse" in r and not np.isnan(r.get("rmse", float("nan")))
            ]
            r2s = [
                r["r2"]
                for r in runs
                if "r2" in r and not np.isnan(r.get("r2", float("nan")))
            ]
            if rmses:
                mean_rmse = np.mean(rmses)
                std_rmse = np.std(rmses)
                mean_r2 = np.mean(r2s)
                std_r2 = np.std(r2s)
                print(
                    f"  {model_name:30s}: RMSE={mean_rmse:.4f}±{std_rmse:.4f}, R²={mean_r2:.4f}±{std_r2:.4f}"
                )
                summary[frac_key][model_name] = {
                    "rmse_mean": round(float(mean_rmse), 4),
                    "rmse_std": round(float(std_rmse), 4),
                    "r2_mean": round(float(mean_r2), 4),
                    "r2_std": round(float(std_r2), 4),
                    "n_runs": len(rmses),
                }

    # ── 5. Final save (authoritative + legacy) ─────────────────────
    _save_incremental_results(all_results, summary, run_dir, output_dir)

    # Also save a copy of the run config alongside final results
    import datetime as _dt

    run_config_final = run_config.copy()
    run_config_final["completed_at"] = _dt.datetime.now().isoformat()
    run_config_final["status"] = "completed"
    with open(run_dir / "config.json", "w") as f:
        json.dump(run_config_final, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Paper 2 Benchmark Complete")
    print(f"{'=' * 70}")
    print(f"  Authoritative output: {run_dir}")
    print(f"  Legacy output:        {output_dir}")
    print(f"  Checkpoints:          {run_dir / 'checkpoints'}")
    print(f"  Training histories:   {run_dir / 'training_history'}")
    print(f"  Conformal results:    {run_dir / 'conformal'}")


if __name__ == "__main__":
    main()
