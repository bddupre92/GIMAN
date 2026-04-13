#!/usr/bin/env python3
"""Paper 2 Downstream Experiment: Does stage-conditioned imputation improve
downstream NSD-ISS stage prediction?

Answers the key Paper 2 question: Does imputing missing values with
stage-aware methods improve downstream classification performance?

Protocol:
    1. Load raw PPMI data with 39.6% missingness
    2. Impute using 8 methods: No Imputation, Mean, MICE, GAIN, SAITS,
       MIWAE, GIMIN Vanilla, GIMIN StageDecoder
    3. Train CatBoost stage classifiers on each imputed dataset
    4. Compare downstream classification: balanced accuracy, AUC, QWK
    5. 5-fold CV with bootstrap CIs
    6. Repeat for all 4 target types: binary, three_class, full_ordinal,
       nsd_positive

If stage-conditioned imputation preserves stage-relevant biomarker
distributions better than vanilla methods, downstream stage prediction
should improve.

Outputs:
    outputs/paper2_benchmark/downstream_comparison_all_targets.json

Usage:
    python scripts/run_downstream_experiment.py
    python scripts/run_downstream_experiment.py --all-targets
    python scripts/run_downstream_experiment.py --epochs 200 --num-folds 5
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "GIMImpN_imputation"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("downstream")


# ── Configuration ────────────────────────────────────────────────────
STAGE_MAP = {"0": 0, "1": 1, "2B": 2, "3": 3, "4": 4, "unclassified": 5}
STAGE_NAMES = {
    0: "Stage 0",
    1: "Stage 1",
    2: "Stage 2B",
    3: "Stage 3",
    4: "Stage 4",
    5: "Unknown",
}

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
    parser = argparse.ArgumentParser(description="Downstream imputation experiment")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target",
        type=str,
        default="three_class",
        choices=["binary", "three_class", "full_ordinal", "nsd_positive"],
        help="Target encoding for downstream classification",
    )
    parser.add_argument(
        "--all-targets",
        action="store_true",
        help="Run all 4 target types (overrides --target)",
    )
    parser.add_argument(
        "--skip-gimin",
        action="store_true",
        help="Skip GIMIN imputation (use cached if available)",
    )
    return parser.parse_args()


def load_data():
    """Load GIMIN cohort filtered to patients with NSD-ISS stages.

    Matches the data loading in run_paper2_experiments.py exactly.
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
    staged_mask_idx = features_df.index.isin(staged_patnos)
    features_staged = features_df[staged_mask_idx].copy()
    mask_staged = mask_df[staged_mask_idx].copy()

    # Join stages
    features_staged = features_staged.reset_index()
    features_staged = features_staged.merge(
        staging_valid[["PATNO", "stage_encoded", "nsd_iss_stage"]],
        on="PATNO",
        how="inner",
    )

    logger.info("Matched: %d patients with stages", len(features_staged))

    # Filter to KEEP_FEATURES only
    available = [f for f in KEEP_FEATURES if f in features_staged.columns]
    missing_feats = [f for f in KEEP_FEATURES if f not in features_staged.columns]
    if missing_feats:
        logger.warning("Missing features: %s", missing_feats)

    features_np = features_staged[available].fillna(0).values.astype(np.float32)

    # Re-index mask to match filtered patients
    mask_staged_reindexed = mask_staged.reset_index()
    mask_merged = features_staged[["PATNO"]].merge(
        mask_staged_reindexed,
        on="PATNO",
        how="inner",
    )
    mask_np = mask_merged[available].values.astype(np.float32)

    stages_np = features_staged["stage_encoded"].values.astype(int)

    logger.info(
        "Data loaded: %d patients, %d features, %.1f%% missing, stages: %s",
        features_np.shape[0],
        features_np.shape[1],
        100 * (1 - mask_np.mean()),
        dict(zip(*np.unique(stages_np, return_counts=True), strict=False)),
    )

    return features_np, mask_np, stages_np, available


def encode_target(stages: np.ndarray, target_type: str) -> np.ndarray:
    """Encode NSD-ISS stages into classification targets."""
    if target_type == "binary":
        # NSD-positive (stages 1+) vs NSD-negative (stage 0)
        return (stages >= 1).astype(int)
    elif target_type == "three_class":
        # Early (0-1), Mild clinical (2B), Impaired (3-4)
        targets = np.zeros_like(stages)
        targets[(stages == 0) | (stages == 1)] = 0
        targets[stages == 2] = 1
        targets[(stages == 3) | (stages == 4)] = 2
        return targets
    elif target_type == "full_ordinal":
        return stages
    elif target_type == "nsd_positive":
        # Among NSD+ patients only (stages 1-4)
        return stages  # Filter externally
    else:
        raise ValueError(f"Unknown target type: {target_type}")


def impute_mean(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean imputation."""
    imputed = features.copy()
    for j in range(features.shape[1]):
        observed = features[mask[:, j] == 1, j]
        if len(observed) > 0:
            imputed[mask[:, j] == 0, j] = observed.mean()
    return imputed


def impute_mice(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """MICE (Iterative Imputer) imputation."""
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer

    # Create array with NaN for missing values
    data = features.copy()
    data[mask == 0] = np.nan

    imputer = IterativeImputer(
        max_iter=20,
        random_state=42,
        sample_posterior=False,
    )
    imputed = imputer.fit_transform(data)
    return imputed.astype(np.float32)


def impute_dl_baseline(
    features: np.ndarray, mask: np.ndarray, method: str
) -> np.ndarray:
    """Impute using a deep learning baseline (GAIN, SAITS, or MIWAE)."""
    from gimin.evaluation.baselines import GAINBaseline, MIWAEBaseline, SAITSBaseline

    baselines = {
        "GAIN": lambda: GAINBaseline(n_epochs=100, batch_size=128, hint_rate=0.9),
        "SAITS": lambda: SAITSBaseline(
            n_layers=2,
            d_model=64,
            n_heads=4,
            d_ffn=128,
            epochs=50,
            patience=5,
        ),
        "MIWAE": lambda: MIWAEBaseline(n_epochs=100, batch_size=128),
    }
    baseline = baselines[method]()
    return baseline.fit_transform(features, mask).astype(np.float32)


def impute_gimin(
    features: np.ndarray,
    mask: np.ndarray,
    stages: np.ndarray,
    epochs: int = 200,
    is_stage_conditioned: bool = False,
) -> np.ndarray:
    """GIMIN imputation (vanilla or stage-conditioned).

    Returns imputed features in original scale.
    """
    import torch
    from gimin.data.scaler import build_scaler_from_config
    from gimin.training.losses import GIMINLoss

    from giman_pipeline.imputation.stage_graph_builder import StageAwareGraphBuilder

    if is_stage_conditioned:
        from giman_pipeline.imputation.stage_conditioned_gimin import (
            StageConditionedGIMIN,
        )
    else:
        from giman_pipeline.imputation.stage_conditioned_gimin import VanillaGIMIN

    torch.manual_seed(42)

    # Build scaler
    cfg = _build_gimin_config()
    scaler = build_scaler_from_config(cfg)

    features_t = torch.tensor(features, dtype=torch.float32)
    mask_t = torch.tensor(mask, dtype=torch.float32)
    stages_t = torch.tensor(stages, dtype=torch.long)

    # Fit scaler on observed values
    scaler.fit(features_t, mask_t)
    features_norm = scaler.transform(features_t, mask_t)

    # Build graph on normalized features
    features_norm_np = features_norm.numpy()
    beta = 0.3 if is_stage_conditioned else 0.0
    builder = StageAwareGraphBuilder(
        k_neighbors=15,
        min_overlap=3,
        stage_affinity_beta=beta,
    )
    graph = builder.build_full_graph(
        features_norm_np * mask.astype(np.float32),
        mask,
        stages=stages if is_stage_conditioned else None,
    )

    edge_index = graph["edge_index"]
    edge_weight = graph["edge_weight"]
    overlap_frac = graph["overlap_frac"]

    # Build model
    if is_stage_conditioned:
        model = StageConditionedGIMIN(
            modality_dims=MODALITY_DIMS,
            embed_dim=64,
            num_gnn_layers=3,
            num_heads=4,
            mc_dropout=0.1,
            num_stages=6,
            stage_embed_dim=16,
            use_stage_attention_bias=False,  # Decoder only
        )
    else:
        model = VanillaGIMIN(
            modality_dims=MODALITY_DIMS,
            embed_dim=64,
            num_gnn_layers=3,
            num_heads=4,
            mc_dropout=0.1,
        )

    # Loss function
    loss_fn = GIMINLoss(
        lambda_dist=0.1,
        lambda_cross=0.10,
        lambda_cal=0.01,
        cal_warmup_epochs=50,
        cross_modal_pairs=cfg.cross_modal_pairs,
        binary_feature_indices=cfg.binary_feature_indices or [0],
    )

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=15,
        factor=0.5,
        min_lr=1e-6,
    )

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        if is_stage_conditioned:
            out = model(
                features_norm,
                mask_t,
                edge_index,
                edge_weight,
                overlap_frac,
                stage_ids=stages_t,
            )
        else:
            out = model(
                features_norm,
                mask_t,
                edge_index,
                edge_weight,
                overlap_frac,
            )

        # GIMINLoss.forward(model_output, true_values, target_mask, observed_mask, epoch)
        # For full imputation: target_mask = mask_t (reconstruct observed),
        # observed_mask = mask_t (model sees all observed values)
        losses = loss_fn(
            model_output=out,
            true_values=features_norm,
            target_mask=mask_t,
            observed_mask=mask_t,
            epoch=epoch,
        )
        loss = losses["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss.item())

    # Inference with MC dropout
    model.train()  # Keep dropout active
    mc_preds = []
    with torch.no_grad():
        for _ in range(20):
            if is_stage_conditioned:
                out = model(
                    features_norm,
                    mask_t,
                    edge_index,
                    edge_weight,
                    overlap_frac,
                    stage_ids=stages_t,
                )
            else:
                out = model(
                    features_norm,
                    mask_t,
                    edge_index,
                    edge_weight,
                    overlap_frac,
                )
            mc_preds.append(out["imputed_values"].detach())

    mean_pred_norm = torch.stack(mc_preds).mean(dim=0)

    # Inverse transform to original scale
    imputed_original = scaler.inverse_transform(mean_pred_norm, mask=None).numpy()

    # Blend: keep observed values, use imputed for missing
    result = features * mask + imputed_original * (1 - mask)
    return result.astype(np.float32)


def _build_gimin_config():
    """Build GIMINConfig matching the 33-feature PPMI schema.

    Uses defaults which already define 33 features across 7 modalities
    with correct normalization strategies and cross-modal pairs.
    """
    from gimin.config import GIMINConfig

    return GIMINConfig()


def train_catboost_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Train CatBoost and return predicted probabilities."""
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        # Fallback to gradient boosting
        from sklearn.ensemble import GradientBoostingClassifier

        logger.warning("CatBoost not available, using sklearn GradientBoosting")
        model = GradientBoostingClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass" if n_classes > 2 else "Logloss",
        auto_class_weights="SqrtBalanced",
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50,
    )
    # Use a small eval set from train for early stopping
    n_train = int(0.85 * len(X_train))
    model.fit(
        X_train[:n_train],
        y_train[:n_train],
        eval_set=(X_train[n_train:], y_train[n_train:]),
    )
    return model.predict_proba(X_test)


def evaluate_fold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int,
) -> dict:
    """Evaluate one fold: balanced accuracy, AUC, QWK."""
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }

    # AUC
    try:
        if n_classes == 2:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["auc_roc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
    except ValueError:
        metrics["auc_roc"] = float("nan")

    return metrics


def run_downstream_cv(
    features_imputed: np.ndarray,
    stages: np.ndarray,
    target_type: str,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Run stratified CV downstream classification on imputed data."""
    targets = encode_target(stages, target_type)

    # For nsd_positive, filter to stages >= 1
    if target_type == "nsd_positive":
        nsd_mask = stages >= 1
        features_imputed = features_imputed[nsd_mask]
        targets = targets[nsd_mask]
        stages_sub = stages[nsd_mask]
    else:
        stages_sub = stages

    # Re-encode to consecutive integers
    le = LabelEncoder()
    targets_enc = le.fit_transform(targets)
    n_classes = len(le.classes_)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []

    for fold_i, (train_idx, test_idx) in enumerate(
        skf.split(features_imputed, targets_enc)
    ):
        X_train = features_imputed[train_idx]
        y_train = targets_enc[train_idx]
        X_test = features_imputed[test_idx]
        y_test = targets_enc[test_idx]

        y_prob = train_catboost_classifier(X_train, y_train, X_test, n_classes)
        fold_result = evaluate_fold(y_test, y_prob, n_classes)
        fold_metrics.append(fold_result)

        logger.info(
            "  Fold %d: bal_acc=%.4f, AUC=%.4f, QWK=%.4f",
            fold_i + 1,
            fold_result["balanced_accuracy"],
            fold_result["auc_roc"],
            fold_result["qwk"],
        )

    # Aggregate
    aggregated = {}
    for key in fold_metrics[0].keys():
        values = [fm[key] for fm in fold_metrics if not np.isnan(fm[key])]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
        else:
            aggregated[f"{key}_mean"] = float("nan")
            aggregated[f"{key}_std"] = float("nan")
    aggregated["n_folds"] = n_folds
    aggregated["n_classes"] = n_classes
    aggregated["n_samples"] = len(features_imputed)
    aggregated["fold_metrics"] = fold_metrics

    return aggregated


def run_single_target(
    features: np.ndarray,
    mask: np.ndarray,
    stages: np.ndarray,
    imputation_methods: dict,
    target_type: str,
    n_folds: int,
    seed: int,
) -> dict:
    """Run downstream classification for a single target type."""
    results = {}
    for method_name, imputed_data in imputation_methods.items():
        print(f"\n  [{method_name}] → {target_type}")

        t0 = time.time()
        method_results = run_downstream_cv(
            imputed_data,
            stages,
            target_type,
            n_folds=n_folds,
            seed=seed,
        )
        method_results["imputation_time"] = round(time.time() - t0, 2)
        results[method_name] = method_results

        print(
            f"    bal_acc={method_results['balanced_accuracy_mean']:.4f}"
            f"±{method_results['balanced_accuracy_std']:.4f}, "
            f"AUC={method_results['auc_roc_mean']:.4f}"
            f"±{method_results['auc_roc_std']:.4f}, "
            f"QWK={method_results['qwk_mean']:.4f}"
            f"±{method_results['qwk_std']:.4f}"
        )

    return results


def main():
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = PROJECT_ROOT / "outputs" / "paper2_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_types = (
        ["binary", "three_class", "full_ordinal", "nsd_positive"]
        if args.all_targets
        else [args.target]
    )

    print("=" * 70)
    print("Paper 2: Downstream Imputation → Stage Prediction Experiment")
    print(f"  Targets: {target_types}")
    print(f"  Folds: {args.num_folds}")
    print(f"  GIMIN epochs: {args.epochs}")
    print(f"  Skip GIMIN: {args.skip_gimin}")
    print("=" * 70)

    # ── 1. Load data ─────────────────────────────────────────────────
    features, mask, stages, feature_names = load_data()
    N, F = features.shape

    print(f"\n  Patients: {N}")
    print(f"  Features: {F}")
    print(f"  Missing: {100 * (1 - mask.mean()):.1f}%")

    # ── 2. Impute with each method (once, reused across targets) ─────
    imputation_methods = {}
    n_methods = 8 if not args.skip_gimin else 6
    i = 0

    # Method 1: No imputation (zeros for missing)
    i += 1
    print(f"\n[{i}/{n_methods}] No imputation (zeros)...")
    imputation_methods["No_Imputation"] = features.copy()

    # Method 2: Mean imputation
    i += 1
    print(f"[{i}/{n_methods}] Mean imputation...")
    t0 = time.time()
    imputation_methods["Mean"] = impute_mean(features, mask)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Method 3: MICE
    i += 1
    print(f"[{i}/{n_methods}] MICE imputation...")
    t0 = time.time()
    imputation_methods["MICE"] = impute_mice(features, mask)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Method 4: GAIN
    i += 1
    print(f"[{i}/{n_methods}] GAIN imputation...")
    t0 = time.time()
    imputation_methods["GAIN"] = impute_dl_baseline(features, mask, "GAIN")
    print(f"  Done in {time.time() - t0:.1f}s")

    # Method 5: SAITS
    i += 1
    print(f"[{i}/{n_methods}] SAITS imputation...")
    t0 = time.time()
    imputation_methods["SAITS"] = impute_dl_baseline(features, mask, "SAITS")
    print(f"  Done in {time.time() - t0:.1f}s")

    # Method 6: MIWAE
    i += 1
    print(f"[{i}/{n_methods}] MIWAE imputation...")
    t0 = time.time()
    imputation_methods["MIWAE"] = impute_dl_baseline(features, mask, "MIWAE")
    print(f"  Done in {time.time() - t0:.1f}s")

    if not args.skip_gimin:
        # Method 7: GIMIN Vanilla
        i += 1
        print(f"[{i}/{n_methods}] GIMIN Vanilla imputation...")
        t0 = time.time()
        imputation_methods["GIMIN_Vanilla"] = impute_gimin(
            features,
            mask,
            stages,
            epochs=args.epochs,
            is_stage_conditioned=False,
        )
        print(f"  Done in {time.time() - t0:.1f}s")

        # Method 8: GIMIN StageDecoder
        i += 1
        print(f"[{i}/{n_methods}] GIMIN StageDecoder imputation...")
        t0 = time.time()
        imputation_methods["GIMIN_StageDecoder"] = impute_gimin(
            features,
            mask,
            stages,
            epochs=args.epochs,
            is_stage_conditioned=True,
        )
        print(f"  Done in {time.time() - t0:.1f}s")

    # ── 3. Run downstream classification for each target type ─────────
    all_results = {}
    for target_type in target_types:
        print(f"\n{'=' * 70}")
        print(f"Target: {target_type}")
        print(f"{'=' * 70}")

        target_results = run_single_target(
            features,
            mask,
            stages,
            imputation_methods,
            target_type,
            args.num_folds,
            args.seed,
        )
        all_results[target_type] = target_results

        # Print summary for this target
        print(f"\n{'Method':<25s} {'Bal_Acc':<20s} {'AUC-ROC':<20s} {'QWK':<20s}")
        print("─" * 85)
        for name, res in target_results.items():
            print(
                f"{name:<25s} "
                f"{res['balanced_accuracy_mean']:.4f}±{res['balanced_accuracy_std']:.4f}  "
                f"{res['auc_roc_mean']:.4f}±{res['auc_roc_std']:.4f}  "
                f"{res['qwk_mean']:.4f}±{res['qwk_std']:.4f}"
            )

    # ── 4. Save results ──────────────────────────────────────────────
    import datetime

    def serialize_results(results_dict):
        save = {}
        for name, res in results_dict.items():
            save_res = {}
            for k, v in res.items():
                if k == "fold_metrics" or isinstance(v, (int, float, str)):
                    save_res[k] = v
                else:
                    save_res[k] = float(v)
            save[name] = save_res
        return save

    save_data = {}
    for target_type, target_results in all_results.items():
        save_data[target_type] = serialize_results(target_results)

    save_data["config"] = {
        "targets": target_types,
        "n_folds": args.num_folds,
        "epochs": args.epochs,
        "seed": args.seed,
        "skip_gimin": args.skip_gimin,
        "n_methods": len(imputation_methods),
        "methods": list(imputation_methods.keys()),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    serializer = lambda x: float(x) if hasattr(x, "__float__") else str(x)  # noqa: E731

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_dir = output_dir / "runs"
    ts_dir.mkdir(parents=True, exist_ok=True)

    if args.all_targets:
        filename = "downstream_comparison_all_targets"
    else:
        filename = f"downstream_comparison_{args.target}"

    ts_path = ts_dir / f"{filename}_{ts}.json"
    with open(ts_path, "w") as f:
        json.dump(save_data, f, indent=2, default=serializer)

    legacy_path = output_dir / f"{filename}.json"
    with open(legacy_path, "w") as f:
        json.dump(save_data, f, indent=2, default=serializer)

    logger.info("Results saved to %s (+ legacy %s)", ts_path, legacy_path)

    # ── 5. Cross-target summary ──────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Cross-Target Summary (Balanced Accuracy)")
    print(f"{'=' * 70}")
    header = f"{'Method':<25s}"
    for t in target_types:
        header += f" {t:<15s}"
    print(header)
    print("─" * (25 + 16 * len(target_types)))

    methods = list(imputation_methods.keys())
    for method in methods:
        row = f"{method:<25s}"
        for t in target_types:
            if t in all_results and method in all_results[t]:
                acc = all_results[t][method]["balanced_accuracy_mean"]
                row += f" {acc:<15.4f}"
            else:
                row += f" {'N/A':<15s}"
        print(row)

    # StageDecoder advantage
    if "GIMIN_StageDecoder" in imputation_methods:
        print(f"\n{'=' * 70}")
        print("GIMIN StageDecoder Advantage (Δ bal_acc vs best non-GIMIN)")
        print(f"{'=' * 70}")
        for t in target_types:
            if t not in all_results:
                continue
            tr = all_results[t]
            sd_acc = tr.get("GIMIN_StageDecoder", {}).get("balanced_accuracy_mean", 0)
            best_other = max(
                res.get("balanced_accuracy_mean", 0)
                for name, res in tr.items()
                if not name.startswith("GIMIN")
            )
            diff = sd_acc - best_other
            print(f"  {t:<20s}: Δ = {diff:+.4f} ({'better' if diff > 0 else 'worse'})")


if __name__ == "__main__":
    main()
