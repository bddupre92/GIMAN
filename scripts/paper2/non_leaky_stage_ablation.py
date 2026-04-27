#!/usr/bin/env python3
"""Paper 2 non-leaky stage-label ablation.

Defends against peer-reviewer concern that NSD-ISS labels leak into GIMIN
imputation via the stage-conditioned decoder. Re-derives NSD-ISS stage
labels from features NOT in the GIMIN 33-feature schema (dropping
PUTAMEN_L_SBR, PUTAMEN_R_SBR, NP3TOT, NHY -- the 4 staging variables that
GIMIN sees during training) and then re-evaluates downstream CatBoost
balanced accuracy on binary + 3-class + NSD+ targets.

Protocol:
  1. Identify 4 GIMIN features that drive NSD-ISS stage (staging variables
     that are ALSO GIMIN features): PUTAMEN_L_SBR, PUTAMEN_R_SBR, NP3TOT, NHY
  2. Derive a "held-out" stage label using:
       - s_positive  (from SAA, NOT in GIMIN)
       - has_clinical_signs_held_out from PRIMDIAG + UPDRS3 subscale SUM
         (subscales are in Paper 1 feature file but NOT in GIMIN schema)
       - has_functional_impairment_held_out from UPDRS2_TOTAL (Paper 1 only)
         and UPDRS3 subscale SUM fallback
     The held-out stage is a NON-LEAKY proxy: ground truth the model
     literally could not have learned during imputation training.
  3. For each of 48 checkpoints (4 variants x 4 fractions x 3 runs):
     - Load state_dict, reconstruct model, re-fit scaler on training data
     - Run MC-dropout inference (20 samples) with ORIGINAL stage_ids
       (preserving the training-time conditioning)
     - Emit imputed feature matrix in original scale
  4. Run CatBoost downstream CV (5-fold stratified) on binary + 3_class
     with TWO target encodings:
       (a) original NSD-ISS stage (leaky baseline: matches Paper 2 Table V)
       (b) held-out NSD-ISS stage (non-leaky ablation)
  5. Compute delta balanced accuracy (StageDecoderOnly - Vanilla), bootstrap
     1000x per configuration.

Budget: ~1h for StageDecoderOnly+Vanilla x 4 fracs x 3 runs = 24 checkpoints.
Full 48-checkpoint run can complete in ~2h but we prioritize the variant
that makes the +2.9%/+3.1% claim.

Outputs:
  - outputs/mechanistic_twin/paper2_submission/ieee-jbhi/revision_analyses/
    non_leaky_ablation.json
  - outputs/mechanistic_twin/paper2_submission/ieee-jbhi/revision_analyses/
    stage_label_circularity.md

Author: GIMAN Research Team
Date: 2026-04-18
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "GIMImpN_imputation"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_paper2_experiments import (  # noqa: E402
    MODALITY_DIMS,
    build_gimin_config,
    build_scaler_from_config,
    create_artificial_mask,
    evaluate_gimin_model,
    load_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CKPT_DIR = (
    ROOT
    / "outputs"
    / "paper2_benchmark"
    / "runs"
    / "full_benchmark_20260222_160247"
    / "checkpoints"
)
OUT_DIR = (
    ROOT
    / "outputs"
    / "mechanistic_twin"
    / "paper2_submission"
    / "ieee-jbhi"
    / "revision_analyses"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MASK_FRACS = [0.1, 0.2, 0.3, 0.5]
NUM_RUNS = 3
SEED = 42
MC_SAMPLES = 20

# Staging variables that appear in the GIMIN 33-feature schema.
# These are the features whose values COULD leak into stage labels.
# (Mapped to feature names used in the KEEP_FEATURES / ppmi_full_cohort parquet.)
LEAKY_GIMIN_STAGING_FEATURES = {
    "PUTAMEN_L_SBR",   # drives D anchor (d_positive)
    "PUTAMEN_R_SBR",   # drives D anchor (d_positive)
    "NP3TOT",          # drives has_clinical_signs (>=10) + has_impairment (>=20)
    "NHY",             # drives has_clinical_signs (>0) + has_functional_impairment (>=2)
}

# Features that can act as non-leaky substitutes (all in Paper 1, NONE in GIMIN).
NON_LEAKY_CLINICAL = [
    "UPDRS1_TOTAL",      # non-motor
    "UPDRS2_TOTAL",      # motor ADL (substitute for functional impairment)
    "UPDRS3_TREMOR",     # UPDRS-III subscale
    "UPDRS3_RIGIDITY",
    "UPDRS3_BRADYKINESIA",
    "UPDRS3_AXIAL",
    "UPDRS4_TOTAL",
    "MOCA_TOTAL",
]


# ─── Held-out stage derivation ──────────────────────────────────────────

def _held_out_clinical_sign(row: pd.Series) -> bool:
    """has_clinical_parkinsonism using PRIMDIAG + UPDRS3 subscales (NOT NP3TOT, NHY).

    Logic (parallel to compute_nsd_iss_stage's `has_clinical`):
      - PRIMDIAG == 1 (Idiopathic PD) → True
      - UPDRS3 subscale SUM >= 10 (proxy for NP3TOT >= 10) → True
        (subscales are NOT in GIMIN schema, so non-leaky)
    """
    primdiag = row.get("PRIMDIAG", np.nan)
    if not pd.isna(primdiag) and int(primdiag) == 1:
        return True

    # Substitute NP3TOT threshold with UPDRS3 subscale sum
    subscale_sum = 0.0
    n_present = 0
    for col in ("UPDRS3_TREMOR", "UPDRS3_RIGIDITY",
                "UPDRS3_BRADYKINESIA", "UPDRS3_AXIAL"):
        v = row.get(col, np.nan)
        if not pd.isna(v):
            subscale_sum += float(v)
            n_present += 1

    if n_present >= 3 and subscale_sum >= 10:
        return True
    return False


def _held_out_functional_impairment(row: pd.Series) -> tuple[str, bool]:
    """has_functional_impairment using UPDRS2 (ADL) + UPDRS3 subscale sum (NOT NHY, NP3TOT).

    Logic (parallel to compute_functional_impairment):
      - UPDRS2_TOTAL (motor ADL) is the primary functional-impairment signal
        *independent* of NHY/NP3TOT. Thresholds calibrated against Russo 2025.
      - UPDRS3 subscale SUM is fallback.
    Returns (impairment_level, has_impairment).
    """
    u2 = row.get("UPDRS2_TOTAL", np.nan)
    if not pd.isna(u2):
        u2 = float(u2)
        if u2 < 3:
            return "none", False
        if u2 < 14:
            return "mild", True
        if u2 < 27:
            return "moderate", True
        return "severe", True

    # Fallback: UPDRS3 subscale sum (non-leaky proxy for NP3TOT)
    subscale_sum = 0.0
    n_present = 0
    for col in ("UPDRS3_TREMOR", "UPDRS3_RIGIDITY",
                "UPDRS3_BRADYKINESIA", "UPDRS3_AXIAL"):
        v = row.get(col, np.nan)
        if not pd.isna(v):
            subscale_sum += float(v)
            n_present += 1
    if n_present >= 3:
        if subscale_sum < 20:
            return "none", False
        if subscale_sum < 40:
            return "mild", True
        if subscale_sum < 60:
            return "moderate", True
        return "severe", True

    return "none", False  # Default when no data


def _compute_held_out_stage(
    s_pos: bool | None,
    has_clinical_alt: bool,
    has_impairment_alt: bool,
    impairment_level_alt: str,
    has_genetic_risk: bool = False,
) -> int:
    """Re-run NSD-ISS staging with NON-LEAKY clinical/functional signals.

    Only `s_pos` is from canonical staging (SAA not in GIMIN).
    `has_clinical_alt` and `has_impairment_alt` replace the PUTAMEN_SBR/NP3TOT/NHY
    driven signals with non-leaky substitutes.

    Returns integer stage (0=Stage 0, 1=Stage 1, 2=Stage 2B, 3=Stage 3, 4=Stage 4,
    -1 = unclassified).

    Logic is a simplified-but-faithful NSD-ISS assignment, matching the
    canonical mapping used in target_binary/target_3class:
      - No S+ and no clinical = 0
      - S+ and no clinical   = 1 (early NSD+)
      - Clinical, not impaired = 2B
      - Mild impairment = 3
      - Moderate/severe impairment = 4
    """
    if s_pos is None and not has_clinical_alt:
        # No bio signal, no clinical signal → Stage 0
        return 0

    if not has_clinical_alt:
        if s_pos:
            return 1  # S+ but asymptomatic
        return 0

    # Clinical parkinsonism present
    if not has_impairment_alt:
        return 2  # Stage 2B (clinical signs, no functional impairment)

    if impairment_level_alt == "mild":
        return 3
    return 4  # moderate/severe/complete


def derive_held_out_stages(
    patnos: np.ndarray,
    project_root: Path,
) -> tuple[dict[int, int], pd.DataFrame]:
    """Build held-out stage labels keyed by PATNO.

    Returns:
      - dict PATNO -> held_out_stage (integer 0,1,2,3,4 or -1 unclassified)
      - debug DataFrame with per-patient fields used in derivation
    """
    # Original staging (for s_positive only; we re-derive clinical/functional)
    staging = pd.read_csv(project_root / "data" / "04_staging" / "nsd_iss_staging_results.csv")
    p1 = pd.read_csv(project_root / "data" / "05_features" / "paper1_features_with_targets.csv")

    # Primary diagnosis
    primdiag_path = sorted(
        (project_root / "data" / "00_raw").glob("Primary_Clinical_Diagnosis_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[0]
    diag = pd.read_csv(primdiag_path, low_memory=False)
    if "EVENT_ID" in diag.columns:
        bl = diag["EVENT_ID"].isin(["BL", "SC", "V01"])
        diag_bl = diag[bl] if bl.any() else diag
    else:
        diag_bl = diag
    diag_bl = diag_bl[["PATNO", "PRIMDIAG"]].drop_duplicates("PATNO", keep="first")
    diag_bl["PATNO"] = pd.to_numeric(diag_bl["PATNO"], errors="coerce").astype("Int64")
    diag_bl["PRIMDIAG"] = pd.to_numeric(diag_bl["PRIMDIAG"], errors="coerce")

    # Build working df
    work = p1[["PATNO"] + NON_LEAKY_CLINICAL].copy()
    work = work.merge(staging[["PATNO", "s_positive", "nsd_iss_stage_numeric"]], on="PATNO", how="left")
    work = work.merge(diag_bl, on="PATNO", how="left")

    # Filter to patnos requested
    work = work[work["PATNO"].isin(patnos)].copy()

    # Derive held-out fields
    held_out_stages = {}
    debug_rows = []
    for _, row in work.iterrows():
        patno = int(row["PATNO"])
        s_pos_val = row.get("s_positive", None)
        if pd.isna(s_pos_val):
            s_pos = None
        else:
            # s_positive stored as bool or 0/1
            s_pos = bool(int(s_pos_val)) if isinstance(s_pos_val, (int, float)) else bool(s_pos_val)

        has_clinical_alt = _held_out_clinical_sign(row)
        level_alt, has_imp_alt = _held_out_functional_impairment(row)
        stage = _compute_held_out_stage(s_pos, has_clinical_alt, has_imp_alt, level_alt)

        held_out_stages[patno] = stage
        debug_rows.append({
            "PATNO": patno,
            "s_positive": s_pos,
            "PRIMDIAG": row.get("PRIMDIAG"),
            "UPDRS2_TOTAL": row.get("UPDRS2_TOTAL"),
            "UPDRS3_subscale_sum": (
                (row.get("UPDRS3_TREMOR", 0) or 0)
                + (row.get("UPDRS3_RIGIDITY", 0) or 0)
                + (row.get("UPDRS3_BRADYKINESIA", 0) or 0)
                + (row.get("UPDRS3_AXIAL", 0) or 0)
            ),
            "held_out_clinical": has_clinical_alt,
            "held_out_impairment": has_imp_alt,
            "held_out_impairment_level": level_alt,
            "held_out_stage": stage,
            "original_stage": row.get("nsd_iss_stage_numeric", np.nan),
        })

    debug_df = pd.DataFrame(debug_rows)
    return held_out_stages, debug_df


# ─── Model inference ───────────────────────────────────────────────────

def _build_model(variant: str):
    """Instantiate the GIMIN variant matching the saved checkpoint."""
    from giman_pipeline.imputation.stage_conditioned_gimin import (
        StageConditionedGIMIN,
        VanillaGIMIN,
    )
    if variant in ("Vanilla", "StageGraphOnly"):
        return VanillaGIMIN(
            modality_dims=MODALITY_DIMS,
            embed_dim=64,
            num_gnn_layers=3,
            num_heads=4,
            mc_dropout=0.1,
        ), False
    return StageConditionedGIMIN(
        modality_dims=MODALITY_DIMS,
        embed_dim=64,
        num_gnn_layers=3,
        num_heads=4,
        mc_dropout=0.1,
        num_stages=6,
        stage_embed_dim=16,
    ), True


def impute_with_checkpoint(
    features: np.ndarray,
    mask: np.ndarray,
    stages: np.ndarray,
    corrupted_mask: np.ndarray,
    scaler,
    graph,
    variant: str,
    ckpt_path: Path,
    device: torch.device,
) -> np.ndarray:
    """Load a GIMIN checkpoint and run MC-dropout inference, returning imputed values.

    Uses the SAME stage_ids that the model was trained with (the original
    NSD-ISS stages). The non-leaky test is done at the DOWNSTREAM evaluation
    stage by using held-out stage labels as targets, not by changing the
    stage_ids at inference.
    """
    model, is_stage = _build_model(variant)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load {ckpt_path.name}: {e}")
    model.to(device).eval()

    features_t = torch.tensor(features, dtype=torch.float32).to(device)
    mask_t = torch.tensor(mask, dtype=torch.float32).to(device)
    corrupted_t = torch.tensor(corrupted_mask, dtype=torch.float32).to(device)
    stages_t = torch.tensor(stages, dtype=torch.long).to(device)

    # Normalize features
    features_norm_t = scaler.transform(features_t.cpu(), mask_t.cpu()).to(device)

    eval_mask = (mask - corrupted_mask).clip(min=0)  # positions that were "held out"

    _, imputed_original, _, _ = evaluate_gimin_model(
        model,
        features_norm_t,
        mask_t,
        corrupted_t,
        eval_mask,
        graph["edge_index"].to(device),
        graph["edge_weight"].to(device),
        graph["overlap_frac"].to(device),
        stages,
        features_original=features,
        scaler=scaler,
        stages_t=stages_t if is_stage else None,
        is_stage_conditioned=is_stage,
        mc_samples=MC_SAMPLES,
    )
    return imputed_original


# ─── Downstream CatBoost evaluation ─────────────────────────────────────

def _encode_target(stages: np.ndarray, target_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_subset, keep_mask) for each target formulation.

    stages are integer codes: 0, 1, 2 (=2B), 3, 4, or -1 (unclassified).
    """
    valid = stages >= 0
    y_sub = stages[valid]
    if target_type == "binary":
        return (y_sub >= 1).astype(int), valid
    if target_type == "three_class":
        y = np.zeros_like(y_sub)
        y[(y_sub == 0) | (y_sub == 1)] = 0
        y[y_sub == 2] = 1
        y[(y_sub == 3) | (y_sub == 4)] = 2
        return y, valid
    if target_type == "nsd_positive":
        nsd_mask = y_sub >= 1
        # Among NSD+ patients only (1,2B,3,4) - remap to 0..3
        y_nsd = y_sub[nsd_mask].copy()
        remap = {1: 0, 2: 1, 3: 2, 4: 3}
        y_encoded = np.array([remap[v] for v in y_nsd])
        # Compose final mask
        final_mask = valid.copy()
        final_idx = np.where(valid)[0]
        keep_idx = final_idx[nsd_mask]
        new_mask = np.zeros_like(valid)
        new_mask[keep_idx] = True
        return y_encoded, new_mask
    raise ValueError(target_type)


def downstream_ctboost(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """5-fold stratified CV with CatBoost. Returns fold metrics."""
    from catboost import CatBoostClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold

    n_classes = int(y.max() + 1) if y.max() > 0 else 2
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_bal_accs = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

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
        n_train = int(0.85 * len(X_tr))
        model.fit(
            X_tr[:n_train], y_tr[:n_train],
            eval_set=(X_tr[n_train:], y_tr[n_train:]),
        )
        y_prob = model.predict_proba(X_te)
        y_pred = np.argmax(y_prob, axis=1)
        fold_bal_accs.append(float(balanced_accuracy_score(y_te, y_pred)))

    return {
        "fold_bal_acc": fold_bal_accs,
        "mean": float(np.mean(fold_bal_accs)),
        "std": float(np.std(fold_bal_accs)),
        "n_classes": n_classes,
        "n_samples": len(X),
    }


def bootstrap_delta_ci(
    x1: np.ndarray, x2: np.ndarray, n_boot: int = 1000, seed: int = 42,
) -> dict:
    """Bootstrap delta (x2 - x1) with 95% CI. Inputs are per-fold bal_accs."""
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(n_boot):
        idx1 = rng.randint(0, len(x1), len(x1))
        idx2 = rng.randint(0, len(x2), len(x2))
        deltas.append(float(np.mean(x2[idx2]) - np.mean(x1[idx1])))
    deltas = np.array(deltas)
    return {
        "delta_mean": float(deltas.mean()),
        "delta_ci_low": float(np.quantile(deltas, 0.025)),
        "delta_ci_high": float(np.quantile(deltas, 0.975)),
        "delta_obs": float(x2.mean() - x1.mean()),
    }


# ─── Main ablation pipeline ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--variants", nargs="+",
                        default=["Vanilla", "StageDecoderOnly"],
                        help="GIMIN variants to evaluate (default: Vanilla + "
                             "StageDecoderOnly; StageDecoderOnly is the one "
                             "making the +2.9%%/+3.1%% claim)")
    parser.add_argument("--fracs", nargs="+", type=float, default=MASK_FRACS)
    parser.add_argument("--runs", nargs="+", type=int, default=list(range(NUM_RUNS)))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--output-name", type=str, default="non_leaky_ablation.json")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.manual_seed(SEED)

    logger.info("=" * 70)
    logger.info("Paper 2 non-leaky stage-label ablation")
    logger.info("  Variants: %s", args.variants)
    logger.info("  Fracs:    %s", args.fracs)
    logger.info("  Runs:     %s", args.runs)
    logger.info("  Checkpoint dir: %s", args.checkpoint_dir)
    logger.info("  Output: %s", OUT_DIR / args.output_name)
    logger.info("=" * 70)

    # ── 1. Load benchmark data (matches run_paper2_experiments.py) ────
    features, mask, stages, feature_names = load_data()
    n, f = features.shape
    logger.info("Data: n=%d, f=%d, %.1f%% missing", n, f, 100 * (1 - mask.mean()))

    # Overlap analysis: which GIMIN features ARE staging drivers?
    gimin_staging_overlap = [
        (i, fn) for i, fn in enumerate(feature_names)
        if fn in LEAKY_GIMIN_STAGING_FEATURES
    ]
    logger.info(
        "GIMIN features that drive NSD-ISS stage assignment "
        "(leaky candidates):\n  %s",
        gimin_staging_overlap,
    )

    # ── 2. Build held-out stage labels ────────────────────────────────
    # We need PATNOs in the same row order as features[].
    # load_data returns features filtered to staged patients; let's recover PATNOs.
    staging_path = ROOT / "data" / "04_staging" / "nsd_iss_staging_results.csv"
    staging_df = pd.read_csv(staging_path)
    valid_stage = staging_df["nsd_iss_stage"] != "unclassified"
    features_df = pd.read_parquet(
        ROOT / "GIMImpN_imputation" / "outputs" / "ppmi_full_cohort.parquet"
    )
    staged_patnos = set(staging_df.loc[valid_stage, "PATNO"].values)
    mask_bool = features_df.index.isin(staged_patnos)
    # The order returned by load_data() matches features_df[staged].merge(staging, inner).
    patnos_ordered = (
        features_df[mask_bool].reset_index()[["PATNO"]]
        .merge(staging_df.loc[valid_stage, ["PATNO"]], on="PATNO", how="inner")
        ["PATNO"].to_numpy()
    )
    assert len(patnos_ordered) == n, (
        f"PATNO ordering mismatch: {len(patnos_ordered)} != {n}"
    )

    held_out_dict, debug_df = derive_held_out_stages(patnos_ordered, ROOT)
    debug_df.to_csv(OUT_DIR / "held_out_stage_debug.csv", index=False)

    held_out_stages = np.array(
        [held_out_dict.get(int(p), -1) for p in patnos_ordered]
    )
    # Agreement with original
    agreement = (held_out_stages == stages).mean()
    overlap_on_valid = (
        (held_out_stages[held_out_stages >= 0] == stages[held_out_stages >= 0]).mean()
    )
    logger.info(
        "Held-out stage distribution: %s",
        dict(zip(*np.unique(held_out_stages, return_counts=True), strict=False)),
    )
    logger.info(
        "Original stage distribution : %s",
        dict(zip(*np.unique(stages, return_counts=True), strict=False)),
    )
    logger.info(
        "Agreement (non-leaky vs leaky): overall=%.3f, on-valid=%.3f",
        agreement, overlap_on_valid,
    )

    # ── 3. Setup scaler + config ──────────────────────────────────────
    cfg = build_gimin_config()
    scaler = build_scaler_from_config(cfg)
    features_t_raw = torch.tensor(features, dtype=torch.float32)
    mask_t = torch.tensor(mask, dtype=torch.float32)
    scaler.fit(features_t_raw, mask_t)
    features_norm_np = scaler.transform(features_t_raw, mask_t).numpy()

    # ── 4. Build graphs + run inference per checkpoint ───────────────
    from giman_pipeline.imputation.stage_graph_builder import StageAwareGraphBuilder

    all_graphs: dict[str, dict] = {}

    results = {
        "metadata": {
            "variants": args.variants,
            "mask_fracs": args.fracs,
            "runs": args.runs,
            "mc_samples": MC_SAMPLES,
            "n_folds": args.n_folds,
            "n_boot": args.n_boot,
            "seed": SEED,
            "n_patients": int(n),
            "n_features": int(f),
            "feature_names": list(feature_names),
            "leaky_gimin_staging_overlap": gimin_staging_overlap,
            "held_out_vs_original_agreement": {
                "overall": float(agreement),
                "on_valid_held_out": float(overlap_on_valid),
            },
            "held_out_distribution": {
                int(k): int(v)
                for k, v in zip(*np.unique(held_out_stages, return_counts=True), strict=False)
            },
            "original_distribution": {
                int(k): int(v)
                for k, v in zip(*np.unique(stages, return_counts=True), strict=False)
            },
        },
        "per_checkpoint_downstream": [],
        "delta_stagedecoderonly_vs_vanilla": {},
    }

    target_types = ["binary", "three_class", "nsd_positive"]

    for frac in args.fracs:
        for run in args.runs:
            seed = SEED + run
            corrupted_mask, eval_mask = create_artificial_mask(mask, frac, seed=seed)

            # Build graphs once per frac/run (vanilla + stage-aware)
            for key_prefix, beta in [("vanilla", 0.0), ("stage", 0.3)]:
                gkey = f"{key_prefix}_frac{frac}_run{run}"
                if gkey in all_graphs:
                    continue
                builder = StageAwareGraphBuilder(
                    k_neighbors=15,
                    min_overlap=3,
                    stage_affinity_beta=beta,
                )
                all_graphs[gkey] = builder.build_full_graph(
                    features_norm_np * corrupted_mask.astype(np.float32),
                    corrupted_mask,
                    stages=stages if beta > 0 else None,
                )

            for variant in args.variants:
                ckpt_fname = f"frac{frac:.1f}_run{run}_GIMIN_{variant}.pt"
                ckpt_path = args.checkpoint_dir / ckpt_fname
                if not ckpt_path.exists():
                    logger.warning("[MISS] %s", ckpt_fname)
                    continue

                t0 = time.time()
                use_stage_graph = variant in ("StageGraphOnly", "StageConditioned")
                gkey = ("stage" if use_stage_graph else "vanilla") + f"_frac{frac}_run{run}"

                try:
                    imputed = impute_with_checkpoint(
                        features=features,
                        mask=mask,
                        stages=stages,
                        corrupted_mask=corrupted_mask,
                        scaler=scaler,
                        graph=all_graphs[gkey],
                        variant=variant,
                        ckpt_path=ckpt_path,
                        device=device,
                    )
                except Exception as e:
                    logger.error("[INF-ERR] %s: %s", ckpt_fname, e)
                    continue

                # Downstream CatBoost on BOTH targets (original + held-out stages)
                ds_entry = {
                    "variant": variant,
                    "frac": float(frac),
                    "run": int(run),
                    "checkpoint": ckpt_fname,
                    "elapsed_infer_sec": round(time.time() - t0, 2),
                    "downstream": {},
                }

                for tgt in target_types:
                    for label_kind, y_source in [("leaky", stages), ("held_out", held_out_stages)]:
                        y, keep = _encode_target(y_source, tgt)
                        X_sub = imputed[keep]
                        try:
                            res = downstream_ctboost(X_sub, y, n_folds=args.n_folds, seed=SEED)
                        except Exception as e:
                            logger.error("[DS-ERR] %s %s %s: %s",
                                         ckpt_fname, tgt, label_kind, e)
                            res = {"error": str(e), "fold_bal_acc": []}
                        ds_entry["downstream"][f"{tgt}__{label_kind}"] = res

                logger.info(
                    "[%s frac=%.1f run=%d] infer=%.1fs  "
                    "binary leaky=%.3f held_out=%.3f  "
                    "3cls leaky=%.3f held_out=%.3f",
                    variant, frac, run, ds_entry["elapsed_infer_sec"],
                    ds_entry["downstream"].get("binary__leaky", {}).get("mean", float("nan")),
                    ds_entry["downstream"].get("binary__held_out", {}).get("mean", float("nan")),
                    ds_entry["downstream"].get("three_class__leaky", {}).get("mean", float("nan")),
                    ds_entry["downstream"].get("three_class__held_out", {}).get("mean", float("nan")),
                )

                results["per_checkpoint_downstream"].append(ds_entry)

                # Incremental save
                with open(OUT_DIR / args.output_name, "w") as fp:
                    json.dump(results, fp, indent=2)

    # ── 5. Compute delta (StageDecoderOnly - Vanilla) aggregates ─────
    # Group by (frac), average across runs, bootstrap.
    entries = results["per_checkpoint_downstream"]
    for tgt in target_types:
        for label_kind in ("leaky", "held_out"):
            key = f"{tgt}__{label_kind}"
            per_frac = {}
            for frac in args.fracs:
                # Collect per-fold bal_accs across all runs
                for variant in ("Vanilla", "StageDecoderOnly"):
                    folds_concat = []
                    for e in entries:
                        if e["variant"] != variant or abs(e["frac"] - frac) > 1e-6:
                            continue
                        ds = e["downstream"].get(key)
                        if ds and "fold_bal_acc" in ds:
                            folds_concat.extend(ds["fold_bal_acc"])
                    per_frac.setdefault(frac, {})[variant] = folds_concat

                v = per_frac[frac].get("Vanilla", [])
                s = per_frac[frac].get("StageDecoderOnly", [])
                if not v or not s:
                    continue
                v_arr = np.array(v)
                s_arr = np.array(s)
                boot = bootstrap_delta_ci(v_arr, s_arr, n_boot=args.n_boot, seed=SEED)
                per_frac[frac]["Vanilla_mean"] = float(v_arr.mean())
                per_frac[frac]["Vanilla_std"] = float(v_arr.std())
                per_frac[frac]["StageDecoderOnly_mean"] = float(s_arr.mean())
                per_frac[frac]["StageDecoderOnly_std"] = float(s_arr.std())
                per_frac[frac]["bootstrap"] = boot

            results["delta_stagedecoderonly_vs_vanilla"][key] = {
                str(k): v for k, v in per_frac.items()
            }

    # Save final
    with open(OUT_DIR / args.output_name, "w") as fp:
        json.dump(results, fp, indent=2)
    logger.info("Saved: %s", OUT_DIR / args.output_name)

    # ── 6. Print summary ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("NON-LEAKY STAGE ABLATION — SUMMARY")
    print("=" * 80)
    print(f"GIMIN features overlapping with staging variables: {len(gimin_staging_overlap)}/33")
    for idx, fn in gimin_staging_overlap:
        print(f"  [idx {idx}] {fn}")
    print(f"\nHeld-out vs original stage agreement: {agreement:.3f}")
    print(f"\n{'Target':<14}{'Frac':>6} {'Vanilla':>10} {'StageDec':>10} "
          f"{'Δ_obs':>9} {'[CI]':>20} {'Label':>10}")
    print("-" * 90)
    for tgt in target_types:
        for label_kind in ("leaky", "held_out"):
            key = f"{tgt}__{label_kind}"
            per_frac = results["delta_stagedecoderonly_vs_vanilla"].get(key, {})
            for frac_s in sorted(per_frac.keys(), key=float):
                row = per_frac[frac_s]
                if "bootstrap" not in row:
                    continue
                b = row["bootstrap"]
                print(
                    f"{tgt:<14}{float(frac_s):>6.2f} "
                    f"{row['Vanilla_mean']:>10.3f} "
                    f"{row['StageDecoderOnly_mean']:>10.3f} "
                    f"{b['delta_obs']:>+9.4f} "
                    f"[{b['delta_ci_low']:+.3f},{b['delta_ci_high']:+.3f}] "
                    f"{label_kind:>10}"
                )
    print("=" * 80)


if __name__ == "__main__":
    main()
