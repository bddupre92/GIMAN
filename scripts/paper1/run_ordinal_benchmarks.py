"""Paper 1 WS1.4 — Ordinal-Aware Modeling Benchmark.

Benchmark rank-aware classifiers against the WS1.1 fold-local multiclass
CatBoost baseline on the ordinal NSD-ISS targets. Addresses reviewer
W3/W8/Q6 for the IEEE JBHI revision.

Methods
-------
1. CORAL  (Cao, Mirjalili, Raschka 2020, Pattern Recognition Letters 140:325)
2. CORN   (Shi, Cao, Raschka 2023, Pattern Analysis and Applications 26:941)
3. Ordinal CatBoost
   - Primary:  CatBoostClassifier(loss_function="YetiRank")
   - Fallback: CatBoostRegressor(loss_function="RMSE") + argmin-distance decoding

All three share the WS1.1 fold-local preprocessing protocol: the imputer
(and scaler, for NN methods) is fit ONLY on the training fold.

Targets
-------
- ``target_full_ordinal`` (5 classes: 0, 1, 2B, 3, 4)
- ``target_nsd_positive`` (4 classes: 1, 2B, 3, 4)
- Optional ``--merge-stage-3-4`` (pre-registered sensitivity collapse of
  stages 3+4 into a single "Stage 3+" class).

Outputs
-------
``outputs/paper1_ordinal/results/<method>_<target>.json`` with per-fold
metrics (QWK, macro_auc, MAOE, per_class_acc), aggregate, and bootstrap CIs
(1000 resamples), matching the WS1.1 schema.

Pre-registration
----------------
Locked at ``outputs/paper1_ordinal/PRE_REGISTRATION.md`` (commit fcc81d7).
Decision rule on QWK delta vs. WS1.1 baseline QWK = 0.8632:
    >=+0.05 -> PROMOTE-TO-HEADLINE
    +0.02..+0.05 -> CO-REPORT
    -0.02..+0.02 -> FAIR-ORDINAL-BASELINE (expected per Bonnier 2022)
    <-0.02 -> DOWNGRADE-ORDINAL

Requirements
------------
- ``coral-pytorch`` (pip install coral-pytorch; MIT; Raschka Research Group)
- ``torch`` >= 2.0
- ``catboost`` >= 1.2
- Existing WS1.1 helpers from ``giman_pipeline.sota.nsd_iss_benchmark``.

Author: Blair Dupre (UND BME)
Date:   2026-04-23
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path (matches run_fold_local_imputation.py pattern)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.data.db import read_sql  # noqa: E402
from giman_pipeline.sota.nsd_iss_benchmark import (  # noqa: E402
    _bootstrap_aggregate_ci,
    _compute_metrics,
    _serialize_metric_set,
)

# NOTE: Do not import torch / coral-pytorch at module top level — they are
# only required for --method coral/corn, and importing torch at module
# load can collide with other scripts invoked from the same venv.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---- Paths (align with WS1.1 refit) ----
OUTPUT_DIR = ROOT / "outputs" / "paper1_ordinal" / "results"

# ---- Feature exclusion list (identical to run_fold_local_imputation.py) ----
STAGING_COLS = {
    "PATNO",
    "nsd_iss_stage",
    "nsd_iss_stage_numeric",
    "nsd_iss_stage_ordinal",
    "s_positive",
    "d_positive",
    "has_clinical_signs",
    "has_functional_impairment",
    "functional_impairment_level",
    "staging_confidence",
    "n_missing_anchors",
    "missing_anchors",
    "target_binary",
    "target_3class",
    "target_full_ordinal",
    "target_nsd_positive",
}
HIGH_MISS_COLS = {"UPDRS4_TOTAL", "MOCA_TOTAL"}

# ---- Hyperparameters (locked in pre-registration) ----
MLP_HIDDEN_1 = 128
MLP_HIDDEN_2 = 64
MLP_DROPOUT = 0.3
MLP_LR = 1e-3
MLP_WD = 1e-4
MLP_BATCH = 64
MLP_MAX_EPOCHS = 200
MLP_EARLY_STOP_PATIENCE = 20
INNER_VAL_FRAC = 0.2

CATBOOST_ITERATIONS = 1000
CATBOOST_DEPTH = 6
CATBOOST_LR = 0.05

SEED = 42
N_FOLDS = 5
N_BOOTSTRAP = 1000


# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------


def load_features_sql() -> pd.DataFrame:
    """Load the Paper 1 feature table directly from local Postgres.

    Preferred path per CLAUDE.md (features schema) rather than re-reading the
    CSV. Falls back to the CSV if the SQL call fails (e.g. on a machine
    without the local DB wired up).
    """
    try:
        df = read_sql("SELECT * FROM features.paper1_features_with_targets")
        logger.info(f"Loaded {len(df)} patients from features.paper1_features_with_targets")
        return df
    except Exception as exc:  # pragma: no cover - environmental fallback
        logger.warning(
            f"SQL load failed ({exc!s}); falling back to CSV at "
            f"data/05_features/paper1_features_with_targets.csv"
        )
        csv_path = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} patients from CSV fallback")
        return df


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    exclude_stage0: bool = False,
    merge_stage_3_4: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Prepare fold-local X_raw (may contain NaN) and integer-encoded y.

    Returns
    -------
    X_raw : np.ndarray, shape (n, p) with NaN entries
    y     : np.ndarray, shape (n,) with labels in {0, ..., K-1}
    feature_cols : list[str]
    n_classes : int (K after any merging / remapping)
    """
    feature_cols = [
        c for c in df.columns if c not in STAGING_COLS and c not in HIGH_MISS_COLS
    ]

    mask = df[target_col] >= 0
    if exclude_stage0:
        mask = mask & (df["nsd_iss_stage"] != "0")

    sub = df[mask].copy()
    y_raw = sub[target_col].values.astype(int)

    # Stage 3+4 merge: collapse the top two labels into a single rank.
    if merge_stage_3_4:
        max_label = int(y_raw.max())
        top_label = max_label
        second_label = max_label - 1
        y_raw = np.where(y_raw == top_label, second_label, y_raw)
        logger.info(
            f"  --merge-stage-3-4: collapsed labels {{ {second_label}, {top_label} }} -> {second_label} "
            f"(n now at collapsed class = {(y_raw == second_label).sum()})"
        )

    # Re-map to 0..K-1 contiguous ranks (important for CORAL/CORN + regression decode)
    unique_sorted = np.sort(np.unique(y_raw))
    remap = {int(v): i for i, v in enumerate(unique_sorted)}
    y = np.array([remap[int(v)] for v in y_raw], dtype=int)
    n_classes = int(len(unique_sorted))

    X_raw = sub[feature_cols].copy()
    for col in feature_cols:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")
    X_raw_arr = X_raw.values

    n_nan = int(np.isnan(X_raw_arr).sum())
    logger.info(
        f"Prepared data: target={target_col}, merge_3_4={merge_stage_3_4}, "
        f"n={X_raw_arr.shape[0]}, p={X_raw_arr.shape[1]}, K={n_classes}, "
        f"class_counts={np.bincount(y, minlength=n_classes).tolist()}, "
        f"nan_cells={n_nan} (will be imputed per-fold)"
    )

    return X_raw_arr, y, feature_cols, n_classes


# ---------------------------------------------------------------------------
# Method 1: CORAL
# ---------------------------------------------------------------------------


def _build_mlp_trunk(input_dim: int):
    """Shared 2-layer MLP trunk for CORAL and CORN."""
    import torch.nn as nn  # noqa: WPS433 (delayed import)

    return nn.Sequential(
        nn.Linear(input_dim, MLP_HIDDEN_1),
        nn.ReLU(),
        nn.Dropout(MLP_DROPOUT),
        nn.Linear(MLP_HIDDEN_1, MLP_HIDDEN_2),
        nn.ReLU(),
        nn.Dropout(MLP_DROPOUT),
    )


def _pick_torch_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _train_ordinal_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    head_type: str,  # 'coral' or 'corn'
) -> tuple[np.ndarray, np.ndarray]:
    """Train a CORAL or CORN head with early stopping on inner-val QWK.

    Returns
    -------
    y_pred : (n_test,) int labels
    y_prob : (n_test, K) float class probabilities (from the ordinal head)
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import cohen_kappa_score

    from coral_pytorch.dataset import (
        corn_label_from_logits,
        levels_from_labelbatch,
        proba_to_label,
    )
    from coral_pytorch.losses import coral_loss, corn_loss

    device = _pick_torch_device()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Inner split for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=INNER_VAL_FRAC,
        random_state=SEED, stratify=y_train,
    )

    trunk = _build_mlp_trunk(input_dim=X_tr.shape[1]).to(device)

    if head_type == "coral":
        # CORAL: one shared weight vector + K-1 bias terms.
        # The canonical construction: a Linear(64, 1) feeding raw logits
        # that we then broadcast-subtract K-1 learned thresholds from.
        # To stay in the canonical coral-pytorch API, we implement the head
        # as Linear(64, 1) + Parameter(K-1 thresholds), and produce K-1
        # logits by subtracting.
        class CORALHead(nn.Module):
            def __init__(self, feat_dim: int, n_cls: int):
                super().__init__()
                self.fc = nn.Linear(feat_dim, 1, bias=False)
                # Ordered thresholds (initialised ascending). The biases are
                # NOT constrained to remain ordered; the rank-consistency of
                # CORAL comes from weight-sharing (single fc), not from
                # threshold ordering.
                self.bias = nn.Parameter(torch.zeros(n_cls - 1))

            def forward(self, x):
                z = self.fc(x)  # (B, 1)
                # broadcast to (B, K-1) logits
                return z + self.bias

        head = CORALHead(MLP_HIDDEN_2, n_classes).to(device)
    elif head_type == "corn":
        head = nn.Linear(MLP_HIDDEN_2, n_classes - 1).to(device)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    optim = torch.optim.AdamW(
        list(trunk.parameters()) + list(head.parameters()),
        lr=MLP_LR, weight_decay=MLP_WD,
    )

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_np = y_val.astype(int)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    best_qwk = -np.inf
    best_state: dict[str, Any] | None = None
    patience_ctr = 0

    n_tr = X_tr_t.shape[0]
    for epoch in range(MLP_MAX_EPOCHS):
        trunk.train()
        head.train()
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, MLP_BATCH):
            idx = perm[i : i + MLP_BATCH]
            xb = X_tr_t[idx]
            yb = y_tr_t[idx]

            feats = trunk(xb)
            logits = head(feats)  # (B, K-1)

            if head_type == "coral":
                levels = levels_from_labelbatch(yb, num_classes=n_classes).to(device)
                loss = coral_loss(logits, levels)
            else:  # corn
                loss = corn_loss(logits, yb, num_classes=n_classes)

            optim.zero_grad()
            loss.backward()
            optim.step()

        # Inner validation
        trunk.eval()
        head.eval()
        with torch.no_grad():
            feats_val = trunk(X_val_t)
            logits_val = head(feats_val)
            if head_type == "coral":
                probs_val = torch.sigmoid(logits_val)
                y_pred_val = proba_to_label(probs_val).cpu().numpy()
            else:
                y_pred_val = corn_label_from_logits(logits_val).cpu().numpy()
        val_qwk = cohen_kappa_score(y_val_np, y_pred_val, weights="quadratic")

        if val_qwk > best_qwk + 1e-6:
            best_qwk = float(val_qwk)
            best_state = {
                "trunk": {k: v.detach().clone() for k, v in trunk.state_dict().items()},
                "head": {k: v.detach().clone() for k, v in head.state_dict().items()},
                "epoch": epoch,
            }
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= MLP_EARLY_STOP_PATIENCE:
                break

    if best_state is not None:
        trunk.load_state_dict(best_state["trunk"])
        head.load_state_dict(best_state["head"])

    # Test-set prediction
    trunk.eval()
    head.eval()
    with torch.no_grad():
        feats_test = trunk(X_test_t)
        logits_test = head(feats_test)
        if head_type == "coral":
            probs_cum = torch.sigmoid(logits_test).cpu().numpy()  # (N, K-1)
            y_pred = proba_to_label(torch.tensor(probs_cum)).numpy()
        else:
            probs_cum = torch.sigmoid(logits_test).cpu().numpy()
            y_pred = corn_label_from_logits(logits_test).cpu().numpy()

    # Convert cumulative probabilities (>= class k) into per-class
    # probabilities for macro-AUC computation. p(c=0) = 1 - p(>=1);
    # p(c=K-1) = p(>=K-1); p(c=k) = p(>=k) - p(>=k+1) for 0<k<K-1.
    n_test = probs_cum.shape[0]
    y_prob = np.zeros((n_test, n_classes), dtype=float)
    y_prob[:, 0] = 1.0 - probs_cum[:, 0]
    for k in range(1, n_classes - 1):
        y_prob[:, k] = probs_cum[:, k - 1] - probs_cum[:, k]
    y_prob[:, n_classes - 1] = probs_cum[:, n_classes - 2]
    # Clip tiny negatives from numerical error; renormalise rows.
    y_prob = np.clip(y_prob, 1e-8, 1.0)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    return y_pred.astype(int), y_prob


# ---------------------------------------------------------------------------
# Method 3: Ordinal CatBoost
# ---------------------------------------------------------------------------


def _train_ordinal_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Primary: YetiRank classifier. Fallback: RMSE regressor + argmin decode.

    Returns
    -------
    y_pred, y_prob, branch_used (str)
    """
    import catboost as cb

    # ---- Primary attempt: YetiRank classifier ----
    try:
        clf = cb.CatBoostClassifier(
            loss_function="YetiRank",
            iterations=CATBOOST_ITERATIONS,
            depth=CATBOOST_DEPTH,
            learning_rate=CATBOOST_LR,
            random_seed=SEED,
            verbose=False,
        )
        clf.fit(X_train, y_train)
        y_pred = np.asarray(clf.predict(X_test)).ravel().astype(int)
        try:
            y_prob = clf.predict_proba(X_test)
        except Exception:
            y_prob = None
        logger.info("  ord_catboost branch: YetiRank classifier (primary)")
        return y_pred, y_prob, "yetirank_classifier"
    except Exception as e_primary:
        logger.info(
            f"  ord_catboost branch: YetiRank classifier failed ({e_primary!s}); "
            f"falling back to RMSE regressor"
        )

    # ---- Fallback: RMSE regressor + nearest-rank decode ----
    reg = cb.CatBoostRegressor(
        loss_function="RMSE",
        iterations=CATBOOST_ITERATIONS,
        depth=CATBOOST_DEPTH,
        learning_rate=CATBOOST_LR,
        random_seed=SEED,
        verbose=False,
    )
    reg.fit(X_train, y_train.astype(float))
    y_pred_cont = np.asarray(reg.predict(X_test)).ravel()
    y_pred = np.clip(np.round(y_pred_cont), 0, n_classes - 1).astype(int)

    # Heuristic pseudo-probabilities for AUC: Gaussian smoothing around the
    # continuous regression output, with sigma ~ residual_std on the train
    # set. This is not a calibrated probability (true calibration is a
    # separate conformal task) — it's only used for macro-AUC computation
    # where CatBoost multiclass usually provides its own predict_proba.
    train_resid = reg.predict(X_train) - y_train.astype(float)
    sigma = float(np.std(train_resid)) + 1e-3
    ranks = np.arange(n_classes, dtype=float)
    y_prob = np.exp(
        -((y_pred_cont[:, None] - ranks[None, :]) ** 2) / (2.0 * sigma**2)
    )
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    return y_pred, y_prob, "rmse_regressor_fallback"


# ---------------------------------------------------------------------------
# Shared CV loop
# ---------------------------------------------------------------------------


def run_cv_for_method(
    method: str,
    X_raw: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    target_name: str,
    n_folds: int = N_FOLDS,
    random_state: int = SEED,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict[str, Any]:
    """5-fold stratified CV with fold-local preprocessing, for a single method."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_metrics: list[Any] = []
    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    all_y_prob: list[np.ndarray] = []
    total_train_time = 0.0
    total_predict_time = 0.0
    branch_log: list[str] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_raw, y)):
        X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ---- Fold-local imputation (WS1.1 protocol) ----
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_test_imp = imputer.transform(X_test_raw)

        # Scale only for NN methods; CatBoost is scale-invariant
        if method in {"coral", "corn"}:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_imp)
            X_test_s = scaler.transform(X_test_imp)
        else:
            X_train_s = X_train_imp
            X_test_s = X_test_imp

        t0 = time.time()
        if method == "coral":
            y_pred, y_prob = _train_ordinal_nn(
                X_train_s, y_train, X_test_s, n_classes, head_type="coral"
            )
            branch_log.append("coral")
        elif method == "corn":
            y_pred, y_prob = _train_ordinal_nn(
                X_train_s, y_train, X_test_s, n_classes, head_type="corn"
            )
            branch_log.append("corn")
        elif method == "ord_catboost":
            y_pred, y_prob, branch = _train_ordinal_catboost(
                X_train_s, y_train, X_test_s, n_classes
            )
            branch_log.append(branch)
        else:
            raise ValueError(f"Unknown method: {method}")
        total_train_time += time.time() - t0

        t0 = time.time()
        # (no separate predict timing — train/predict are fused for NN and
        # fast for CatBoost; log zero to preserve schema)
        total_predict_time += time.time() - t0

        fm = _compute_metrics(y_test, y_pred, y_prob, n_classes, is_ordinal=True)
        fold_metrics.append(fm)

        all_y_true.append(y_test)
        all_y_pred.append(y_pred)
        if y_prob is not None:
            all_y_prob.append(y_prob)

        logger.info(
            f"  fold {fold_idx}: QWK={fm.quadratic_weighted_kappa:.4f}, "
            f"bal_acc={fm.balanced_accuracy:.4f}, "
            f"macro_auc={(fm.macro_auc_ovr or float('nan')):.4f}, "
            f"MAOE={(fm.mean_absolute_error or float('nan')):.4f}"
        )

    cat_y_true = np.concatenate(all_y_true)
    cat_y_pred = np.concatenate(all_y_pred)
    cat_y_prob = np.concatenate(all_y_prob) if all_y_prob else None

    aggregate = _compute_metrics(
        cat_y_true, cat_y_pred, cat_y_prob, n_classes, is_ordinal=True
    )
    bootstrap_cis = _bootstrap_aggregate_ci(
        cat_y_true, cat_y_pred, cat_y_prob, n_classes,
        is_ordinal=True, n_bootstrap=n_bootstrap, seed=random_state,
    )

    logger.info(
        f"  AGGREGATE: QWK={aggregate.quadratic_weighted_kappa:.4f}, "
        f"bal_acc={aggregate.balanced_accuracy:.4f}, "
        f"macro_auc={(aggregate.macro_auc_ovr or float('nan')):.4f}, "
        f"MAOE={(aggregate.mean_absolute_error or float('nan')):.4f}"
    )

    # Stage-4 (top-rank) exact binomial CI. When stages 3+4 have been
    # merged, this targets the collapsed "Stage 3+" rank; otherwise it
    # targets the canonical Stage 4.
    top_rank = n_classes - 1
    stage_top_mask = cat_y_true == top_rank
    stage_top_n = int(stage_top_mask.sum())
    stage_top_correct = int((cat_y_pred[stage_top_mask] == top_rank).sum())
    try:
        from scipy.stats import binomtest  # type: ignore

        ci = binomtest(stage_top_correct, stage_top_n).proportion_ci(
            confidence_level=0.95, method="exact"
        )
        stage_top_ci = {"low": float(ci.low), "high": float(ci.high)}
    except Exception:  # pragma: no cover
        stage_top_ci = {"low": float("nan"), "high": float("nan")}

    stage_top_summary = {
        "rank": top_rank,
        "n": stage_top_n,
        "correct": stage_top_correct,
        "proportion": float(stage_top_correct / max(stage_top_n, 1)),
        "exact_binomial_ci_95": stage_top_ci,
    }

    result = {
        "method": method,
        "target_name": target_name,
        "n_classes": n_classes,
        "n_samples": int(len(y)),
        "train_time_seconds": total_train_time,
        "predict_time_seconds": total_predict_time,
        "branch_log": branch_log,
        "aggregate": _serialize_metric_set(aggregate),
        "fold_metrics": [_serialize_metric_set(fm) for fm in fold_metrics],
        "bootstrap_cis": {
            k: {"value": ci.value, "ci_low": ci.ci_low, "ci_high": ci.ci_high}
            for k, ci in bootstrap_cis.items()
        },
        "stage_top_summary": stage_top_summary,
    }
    return result


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _target_config(target: str) -> tuple[str, bool]:
    """Return (target_col, exclude_stage0) for a --target string."""
    if target == "full_ordinal":
        return "target_full_ordinal", False
    if target == "nsd_positive":
        return "target_nsd_positive", True
    if target == "full_ordinal_merged":
        # alias for full_ordinal + --merge-stage-3-4 convenience
        return "target_full_ordinal", False
    raise ValueError(f"Unknown target: {target}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paper 1 WS1.4 ordinal-aware benchmarks (CORAL + CORN + ord-CatBoost). "
            "Pre-registered at outputs/paper1_ordinal/PRE_REGISTRATION.md."
        )
    )
    parser.add_argument(
        "--method", required=True, choices=["coral", "corn", "ord_catboost"],
        help="Ordinal method to run.",
    )
    parser.add_argument(
        "--target", required=True,
        choices=["full_ordinal", "nsd_positive", "full_ordinal_merged"],
        help=(
            "Target label. 'full_ordinal_merged' is an alias for full_ordinal "
            "+ --merge-stage-3-4."
        ),
    )
    parser.add_argument(
        "--merge-stage-3-4", action="store_true",
        help=(
            "Pre-registered sensitivity: collapse the top two ranks into one. "
            "Automatically enabled when --target full_ordinal_merged."
        ),
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=N_BOOTSTRAP,
        help=f"Bootstrap resamples for CIs (default {N_BOOTSTRAP}).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default {OUTPUT_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger.info("=" * 72)
    logger.info(f"Paper 1 WS1.4: Ordinal benchmark  method={args.method}  target={args.target}")
    logger.info("=" * 72)

    target_col, exclude_stage0 = _target_config(args.target)
    merge = args.merge_stage_3_4 or args.target == "full_ordinal_merged"

    df = load_features_sql()
    X_raw, y, feat_names, n_classes = prepare_data(
        df, target_col, exclude_stage0=exclude_stage0, merge_stage_3_4=merge
    )

    # Run CV
    result = run_cv_for_method(
        method=args.method,
        X_raw=X_raw,
        y=y,
        n_classes=n_classes,
        target_name=args.target,
        n_folds=N_FOLDS,
        random_state=SEED,
        n_bootstrap=args.n_bootstrap,
    )
    result["feature_names"] = feat_names
    result["merge_stage_3_4"] = bool(merge)
    result["seed"] = SEED
    result["n_folds"] = N_FOLDS
    result["pre_registration"] = "outputs/paper1_ordinal/PRE_REGISTRATION.md"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.method}_{args.target}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(f"\nSaved results to {out_path}")

    # Quick summary line for the log
    agg = result["aggregate"]
    macro_auc_val = agg.get("macro_auc_ovr")
    maoe_val = agg.get("mean_absolute_error")
    macro_auc_f = macro_auc_val if macro_auc_val is not None else float("nan")
    maoe_f = maoe_val if maoe_val is not None else float("nan")
    print(
        f"\n[WS1.4] {args.method} / {args.target}: "
        f"QWK={agg.get('quadratic_weighted_kappa', float('nan')):.4f}  "
        f"bal_acc={agg.get('balanced_accuracy', float('nan')):.4f}  "
        f"macro_auc={macro_auc_f:.4f}  "
        f"MAOE={maoe_f:.4f}"
    )


if __name__ == "__main__":
    # Keep MPS / CUDA quiet from parallel invocations
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
