"""Paper 1 WS1.6 — Medication-Status Confounder Sensitivity.

Three pre-registered sensitivity arms addressing reviewer W4/Q4 on whether
baseline PD medication status (PDMEDYN) confounds the headline 22-feature
CatBoost binary-NSD AUC:

  Arm 1 — STRATIFY: 5-fold CV CatBoost on PDMEDYN=0 and PDMEDYN=1 separately,
    with a 1000-bootstrap sex/stratum interaction test on binary AUC.
  Arm 2 — MEDICATION-LOCO: train on PDMEDYN=0, test on PDMEDYN=1 (and
    vice versa). Reports both directions to make the asymmetry explicit.
  Arm 3 — COVARIATE: add PDMEDYN as a 23rd feature, re-run 5-fold CV
    CatBoost, compare to WS1.1 fold-local 22-feature baseline.

The decision rule is LOCKED in
``outputs/paper1_medication_sensitivity/PRE_REGISTRATION.md`` before this
script was written. See §V.C of that document for the promote-vs-null rule.

Data source: PDMEDYN baseline flag is the per-PATNO MAX(pdmedyn::int) from
``ppmi_raw.use_of_pd_medication`` (or ``ledd.use_of_pd_medication`` if that
schema variant is populated) restricted to EVENT_ID IN ('BL', 'SC'). The
script resolves the correct schema at runtime via ``information_schema``
lookup and falls back gracefully if one schema is empty.

Hyperparameters exactly match Table I / WS1.1: CatBoost iterations=1000
depth=6 seed=42, 5-fold stratified CV, 1000-bootstrap 95% CIs on AUC.

Outputs go to ``outputs/paper1_medication_sensitivity/results/`` as per-arm
per-target JSONs. Run ``summarize_medication_sensitivity.py`` after to
compute the decision-rule verdict.

Author: Blair Dupre (UND BME)
Date: April 2026

References:
  Fahn S et al. NEJM 2004;351:2498 (ELLDOPA — levodopa alters β-CIT SPECT)
  Khosousi S et al. Mov Disord 2024;39:1881 (Biopark+PPMI DDC-DaT-SPECT)
  Espay AJ et al. Mov Disord 2025;40:601 (NSD-ISS critique)
  Simuni T et al. Mov Disord 2025;40:1746 (reply)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.data.db import get_engine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUT_DIR = ROOT / "outputs" / "paper1_medication_sensitivity"
RESULTS_DIR = OUT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOTSTRAP_CI = 1000
N_BOOTSTRAP_INTERACTION = 1000

# WS1.1 baseline AUCs on the same 22-feature schema (from
# outputs/paper1_benchmark/fold_local_refit/*_results.json) — used as the
# reference comparator for Arm 3 delta.
WS1_1_BASELINE_AUC = {
    "binary": 0.979,
    "3class": 0.942,
    "full_ordinal": 0.946,
    "nsd_positive": 0.904,
}

# Mirror of STAGING_COLS / HIGH_MISS_COLS from run_fold_local_imputation.py
STAGING_COLS = {
    "patno", "PATNO",
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
HIGH_MISS_COLS = {"updrs4_total", "moca_total", "UPDRS4_TOTAL", "MOCA_TOTAL"}
# Columns added by our JOIN that are not features
JOIN_HELPER_COLS = {"pdmedyn"}


TARGETS: dict[str, dict[str, Any]] = {
    "binary": {
        "col": "target_binary",
        "n_classes": 2,
        "is_ordinal": False,
        "exclude_stage0": False,
    },
    "3class": {
        "col": "target_3class",
        "n_classes": 3,
        "is_ordinal": True,
        "exclude_stage0": False,
    },
    "full_ordinal": {
        "col": "target_full_ordinal",
        "n_classes": 5,
        "is_ordinal": True,
        "exclude_stage0": False,
    },
    "nsd_positive": {
        "col": "target_nsd_positive",
        "n_classes": 4,
        "is_ordinal": True,
        "exclude_stage0": True,
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def resolve_pdmed_table() -> tuple[str, str]:
    """Return (schema, table) for use_of_pd_medication, preferring `ledd` then
    `ppmi_raw`. Raises RuntimeError if neither schema has a non-empty table.
    """
    eng = get_engine()
    candidates = [("ledd", "use_of_pd_medication"),
                  ("ppmi_raw", "use_of_pd_medication")]
    with eng.connect() as c:
        for schema, table in candidates:
            exists = c.execute(
                text(
                    "SELECT EXISTS ("
                    "  SELECT 1 FROM information_schema.tables "
                    "  WHERE table_schema = :s AND table_name = :t"
                    ")"
                ),
                {"s": schema, "t": table},
            ).scalar()
            if not exists:
                continue
            # Ensure it has at least one row with pdmedyn populated
            n = c.execute(
                text(
                    f"SELECT COUNT(*) FROM {schema}.{table} "
                    "WHERE pdmedyn IS NOT NULL"
                )
            ).scalar()
            if n and int(n) > 0:
                logger.info(
                    f"PDMEDYN source: {schema}.{table} ({int(n)} non-null rows)"
                )
                return schema, table
    raise RuntimeError(
        "No populated use_of_pd_medication table found in `ledd` or "
        "`ppmi_raw`. Load PPMI Use_of_PD_Medication_*.csv via "
        "scripts/load_csvs_to_local_pg.py first."
    )


def load_joined_frame() -> pd.DataFrame:
    """Load paper1 features + per-PATNO baseline PDMEDYN flag.

    Baseline flag = MAX(pdmedyn::int) over EVENT_ID IN ('BL', 'SC'). Patients
    absent from use_of_pd_medication default to pdmedyn=0 via COALESCE.
    """
    schema, table = resolve_pdmed_table()

    # Detect column case: loader lower-cases, but be defensive.
    eng = get_engine()
    with eng.connect() as c:
        cols = [
            r[0] for r in c.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema=:s AND table_name=:t"
                ),
                {"s": schema, "t": table},
            ).fetchall()
        ]
    col_lookup = {c.lower(): c for c in cols}
    patno_c = col_lookup.get("patno", "patno")
    event_c = col_lookup.get("event_id", "event_id")
    pdmedyn_c = col_lookup.get("pdmedyn", "pdmedyn")

    q = f"""
    SELECT p.*,
           COALESCE(m.pdmedyn, 0) AS pdmedyn
    FROM features.paper1_features_with_targets p
    LEFT JOIN (
        SELECT {patno_c} AS patno,
               MAX(CAST(NULLIF({pdmedyn_c}::text, '') AS INTEGER)) AS pdmedyn
        FROM {schema}.{table}
        WHERE {event_c} IN ('BL', 'SC')
        GROUP BY {patno_c}
    ) m ON LOWER(CAST(p.patno AS text)) = LOWER(CAST(m.patno AS text))
    WHERE p.target_binary >= 0
    """
    with eng.connect() as c:
        df = pd.read_sql_query(text(q), c)

    n_on = int((df["pdmedyn"] == 1).sum())
    n_off = int((df["pdmedyn"] == 0).sum())
    logger.info(
        f"Joined cohort: n={len(df)}, PDMEDYN=1 (on): {n_on} "
        f"({100.0 * n_on / len(df):.1f}%), PDMEDYN=0 (off): {n_off} "
        f"({100.0 * n_off / len(df):.1f}%)"
    )
    return df


def prepare_xy(
    df: pd.DataFrame,
    target_col: str,
    exclude_stage0: bool = False,
    include_pdmedyn: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare (X_raw, y, feature_names) for fold-local downstream imputation.

    Mirrors run_fold_local_imputation.prepare_data_fold_local but (a) operates
    on a Postgres-joined DataFrame with lowercase column names and (b)
    optionally retains pdmedyn as the 23rd feature (Arm 3).
    """
    exclude = STAGING_COLS | HIGH_MISS_COLS
    if not include_pdmedyn:
        exclude = exclude | JOIN_HELPER_COLS
    feature_cols = [c for c in df.columns if c not in exclude]

    mask = df[target_col] >= 0
    if exclude_stage0:
        mask = mask & (df["nsd_iss_stage"].astype(str) != "0")
    sub = df[mask].copy()

    y = sub[target_col].values.astype(int)
    X_raw = sub[feature_cols].copy()
    for c in feature_cols:
        X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")

    return X_raw.values, y, feature_cols


# ---------------------------------------------------------------------------
# CatBoost helpers
# ---------------------------------------------------------------------------


def fit_predict_catboost(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    n_classes: int,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold-local impute + scale + CatBoost fit, returning (y_pred, y_prob)."""
    import catboost as cb

    imp = SimpleImputer(strategy="median")
    X_tr_i = imp.fit_transform(X_tr)
    X_te_i = imp.transform(X_te)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_i)
    X_te_s = sc.transform(X_te_i)

    model = cb.CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        random_seed=seed,
        auto_class_weights="Balanced",
        verbose=0,
    )
    model.fit(X_tr_s, y_tr)
    y_pred = np.asarray(model.predict(X_te_s)).ravel().astype(int)
    y_prob = model.predict_proba(X_te_s)
    return y_pred, y_prob


def cv_catboost(
    X_raw: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    is_ordinal: bool,
    n_folds: int = 5,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    """5-fold stratified CV CatBoost with fold-local preprocessing.

    Returns (y_true_concat, y_pred_concat, y_prob_concat, per_fold_metrics).
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []
    fold_metrics: list[dict[str, float]] = []
    for fold_idx, (tr, te) in enumerate(skf.split(X_raw, y)):
        y_pred, y_prob = fit_predict_catboost(
            X_raw[tr], y[tr], X_raw[te], n_classes, seed=seed
        )
        y_true_all.append(y[te])
        y_pred_all.append(y_pred)
        y_prob_all.append(y_prob)
        bal = balanced_accuracy_score(y[te], y_pred)
        if n_classes == 2:
            try:
                a = roc_auc_score(y[te], y_prob[:, 1])
            except ValueError:
                a = float("nan")
        else:
            try:
                a = roc_auc_score(
                    y[te], y_prob, multi_class="ovr", average="macro"
                )
            except ValueError:
                a = float("nan")
        fold_metrics.append(
            {"fold": fold_idx, "balanced_accuracy": float(bal), "auc": float(a)}
        )
    return (
        np.concatenate(y_true_all),
        np.concatenate(y_pred_all),
        np.concatenate(y_prob_all),
        fold_metrics,
    )


def compute_auc(
    y_true: np.ndarray, y_prob: np.ndarray, n_classes: int
) -> float:
    """Point-estimate AUC: binary → AUC-ROC, multiclass → macro AUC-OVR."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        if n_classes == 2:
            pp = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            return float(roc_auc_score(y_true, pp))
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int,
    n_bootstrap: int = N_BOOTSTRAP_CI,
    seed: int = SEED,
) -> tuple[list[float] | None, int]:
    """Return (95% CI [low, high], n_valid_resamples) for the AUC point."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_prob[idx] if y_prob.ndim == 2 else y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        if n_classes > 2 and len(np.unique(yt)) < n_classes:
            continue
        try:
            if n_classes == 2:
                pp = yp[:, 1] if yp.ndim == 2 else yp
                aucs.append(float(roc_auc_score(yt, pp)))
            else:
                aucs.append(
                    float(roc_auc_score(yt, yp, multi_class="ovr", average="macro"))
                )
        except ValueError:
            continue
    if not aucs:
        return None, 0
    return (
        [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        len(aucs),
    )


# ---------------------------------------------------------------------------
# Arm 1 — Stratify
# ---------------------------------------------------------------------------


def run_arm1(df: pd.DataFrame, target_key: str) -> dict[str, Any]:
    """5-fold CV on PDMEDYN=0 and PDMEDYN=1 separately + bootstrap interaction."""
    spec = TARGETS[target_key]
    n_classes = spec["n_classes"]
    is_ord = spec["is_ordinal"]
    target_col = spec["col"]
    exclude0 = spec["exclude_stage0"]

    logger.info("=" * 70)
    logger.info(f"Arm 1 (stratify) — target={target_key}")
    logger.info("=" * 70)

    out: dict[str, Any] = {
        "arm": 1,
        "arm_name": "stratify",
        "target": target_key,
        "target_col": target_col,
        "seed": SEED,
        "strata": {},
    }

    per_stratum_yprob: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for pdmed_val in (0, 1):
        sub = df[df["pdmedyn"] == pdmed_val]
        X_raw, y, feat_names = prepare_xy(
            sub, target_col, exclude_stage0=exclude0, include_pdmedyn=False
        )
        if len(np.unique(y)) < n_classes:
            logger.warning(
                f"  PDMEDYN={pdmed_val}: observed {len(np.unique(y))} "
                f"classes (< {n_classes}); skipping"
            )
            out["strata"][str(pdmed_val)] = {
                "skipped": True,
                "reason": "<n_classes observed classes in stratum",
                "n": int(len(y)),
            }
            continue
        # Require at least 5 minority-class patients
        min_class = int(np.min(np.bincount(y, minlength=n_classes)))
        if min_class < 5:
            logger.warning(
                f"  PDMEDYN={pdmed_val}: smallest class has {min_class} patients "
                "(< 5); skipping as degenerate"
            )
            out["strata"][str(pdmed_val)] = {
                "skipped": True,
                "reason": f"smallest class n={min_class} < 5 (degenerate)",
                "n": int(len(y)),
            }
            continue

        y_true, y_pred, y_prob, fold_metrics = cv_catboost(
            X_raw, y, n_classes, is_ord, n_folds=5, seed=SEED
        )
        auc_point = compute_auc(y_true, y_prob, n_classes)
        bal_point = float(balanced_accuracy_score(y_true, y_pred))
        ci, n_valid = bootstrap_auc_ci(y_true, y_prob, n_classes, seed=SEED)
        per_stratum_yprob[pdmed_val] = (y_true, y_prob)

        logger.info(
            f"  PDMEDYN={pdmed_val} n={len(y)} AUC={auc_point:.4f} "
            f"[{ci[0]:.4f}, {ci[1]:.4f}]  bal_acc={bal_point:.4f}"
            if ci is not None
            else f"  PDMEDYN={pdmed_val} n={len(y)} AUC={auc_point:.4f}"
        )
        out["strata"][str(pdmed_val)] = {
            "n": int(len(y)),
            "n_features": int(X_raw.shape[1]),
            "class_counts": [int(x) for x in np.bincount(y, minlength=n_classes)],
            "auc_point": auc_point,
            "auc_95ci": ci,
            "auc_bootstrap_n_valid": n_valid,
            "balanced_accuracy": bal_point,
            "fold_metrics": fold_metrics,
        }

    # Interaction test on binary only (AUC is scalar)
    if 0 in per_stratum_yprob and 1 in per_stratum_yprob and n_classes == 2:
        yt_off, yp_off = per_stratum_yprob[0]
        yt_on, yp_on = per_stratum_yprob[1]
        auc_off = compute_auc(yt_off, yp_off, n_classes)
        auc_on = compute_auc(yt_on, yp_on, n_classes)
        delta = auc_off - auc_on
        rng = np.random.default_rng(SEED)
        deltas: list[float] = []
        for _ in range(N_BOOTSTRAP_INTERACTION):
            idx_off = rng.integers(0, len(yt_off), len(yt_off))
            idx_on = rng.integers(0, len(yt_on), len(yt_on))
            if (
                len(np.unique(yt_off[idx_off])) < 2
                or len(np.unique(yt_on[idx_on])) < 2
            ):
                continue
            try:
                a_off = float(
                    roc_auc_score(yt_off[idx_off], yp_off[idx_off, 1])
                )
                a_on = float(roc_auc_score(yt_on[idx_on], yp_on[idx_on, 1]))
                deltas.append(a_off - a_on)
            except ValueError:
                continue
        arr = np.asarray(deltas) if deltas else np.asarray([np.nan])
        ci_low = float(np.nanpercentile(arr, 2.5))
        ci_high = float(np.nanpercentile(arr, 97.5))
        p_two_sided = float(
            2.0 * min((arr >= 0).mean(), (arr <= 0).mean())
        ) if np.isfinite(arr).any() else float("nan")
        out["interaction_test"] = {
            "auc_off_med": auc_off,
            "auc_on_med": auc_on,
            "delta_arm1_off_minus_on": float(delta),
            "bootstrap_delta_95ci": [ci_low, ci_high],
            "bootstrap_two_sided_p": p_two_sided,
            "n_bootstrap_valid": int(np.isfinite(arr).sum()),
            "ci_excludes_zero": bool(
                (ci_low > 0.0 and ci_high > 0.0)
                or (ci_low < 0.0 and ci_high < 0.0)
            ),
        }
        logger.info(
            f"  Interaction: delta={delta:+.4f} [{ci_low:+.4f}, {ci_high:+.4f}] "
            f"p={p_two_sided:.3f}"
        )
    else:
        out["interaction_test"] = {
            "skipped": True,
            "reason": (
                "binary interaction test only run for n_classes=2; "
                "multi-class delta is reported as auc_off - auc_on scalar only"
            ),
        }
        if 0 in per_stratum_yprob and 1 in per_stratum_yprob:
            yt_off, yp_off = per_stratum_yprob[0]
            yt_on, yp_on = per_stratum_yprob[1]
            auc_off = compute_auc(yt_off, yp_off, n_classes)
            auc_on = compute_auc(yt_on, yp_on, n_classes)
            out["interaction_test"].update({
                "auc_off_med": auc_off,
                "auc_on_med": auc_on,
                "delta_arm1_off_minus_on": float(auc_off - auc_on),
            })

    return out


# ---------------------------------------------------------------------------
# Arm 2 — Medication-LOCO
# ---------------------------------------------------------------------------


def run_arm2(df: pd.DataFrame, target_key: str) -> dict[str, Any]:
    """Train on one PDMEDYN stratum, test on the other (both directions)."""
    spec = TARGETS[target_key]
    n_classes = spec["n_classes"]
    target_col = spec["col"]
    exclude0 = spec["exclude_stage0"]

    logger.info("=" * 70)
    logger.info(f"Arm 2 (medication-LOCO) — target={target_key}")
    logger.info("=" * 70)

    out: dict[str, Any] = {
        "arm": 2,
        "arm_name": "medication_loco",
        "target": target_key,
        "target_col": target_col,
        "seed": SEED,
        "directions": {},
    }

    per_direction_yprob: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for train_val, test_val, label in [
        (0, 1, "train_off_test_on"),
        (1, 0, "train_on_test_off"),
    ]:
        tr_df = df[df["pdmedyn"] == train_val]
        te_df = df[df["pdmedyn"] == test_val]
        X_tr, y_tr, _ = prepare_xy(
            tr_df, target_col, exclude_stage0=exclude0, include_pdmedyn=False
        )
        X_te, y_te, _ = prepare_xy(
            te_df, target_col, exclude_stage0=exclude0, include_pdmedyn=False
        )
        if len(np.unique(y_tr)) < n_classes:
            logger.warning(
                f"  {label}: train has only {len(np.unique(y_tr))} classes; skipping"
            )
            out["directions"][label] = {
                "skipped": True,
                "reason": "train has < n_classes observed classes",
                "n_train": int(len(y_tr)),
                "n_test": int(len(y_te)),
            }
            continue
        if len(np.unique(y_te)) < 2:
            logger.warning(
                f"  {label}: test has < 2 classes; skipping"
            )
            out["directions"][label] = {
                "skipped": True,
                "reason": "test has < 2 classes",
                "n_train": int(len(y_tr)),
                "n_test": int(len(y_te)),
            }
            continue
        # Small-train warning
        small_train = len(y_tr) < 100
        if small_train:
            logger.warning(
                f"  {label}: train n={len(y_tr)} < 100 — results likely "
                "under-powered"
            )

        y_pred, y_prob = fit_predict_catboost(
            X_tr, y_tr, X_te, n_classes, seed=SEED
        )
        auc_point = compute_auc(y_te, y_prob, n_classes)
        bal_point = float(balanced_accuracy_score(y_te, y_pred))
        ci, n_valid = bootstrap_auc_ci(y_te, y_prob, n_classes, seed=SEED)
        per_direction_yprob[label] = (y_te, y_prob)

        logger.info(
            f"  {label}: n_train={len(y_tr)} n_test={len(y_te)} "
            f"AUC={auc_point:.4f}"
            + (f" [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "")
            + f"  bal_acc={bal_point:.4f}"
        )
        out["directions"][label] = {
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "small_train_warning": bool(small_train),
            "auc_point": auc_point,
            "auc_95ci": ci,
            "auc_bootstrap_n_valid": n_valid,
            "balanced_accuracy": bal_point,
        }

    # delta_arm2 = AUC(train-off, test-on) - AUC(train-on, test-off)
    a = out["directions"].get("train_off_test_on", {}).get("auc_point")
    b = out["directions"].get("train_on_test_off", {}).get("auc_point")
    if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
        out["delta_arm2"] = float(a - b)
        logger.info(f"  delta_arm2 (off→on minus on→off) = {out['delta_arm2']:+.4f}")
    else:
        out["delta_arm2"] = None

    return out


# ---------------------------------------------------------------------------
# Arm 3 — Covariate
# ---------------------------------------------------------------------------


def run_arm3(df: pd.DataFrame, target_key: str) -> dict[str, Any]:
    """Add PDMEDYN as 23rd feature; compare to WS1.1 22-feature baseline."""
    spec = TARGETS[target_key]
    n_classes = spec["n_classes"]
    is_ord = spec["is_ordinal"]
    target_col = spec["col"]
    exclude0 = spec["exclude_stage0"]

    logger.info("=" * 70)
    logger.info(f"Arm 3 (covariate 23-feat) — target={target_key}")
    logger.info("=" * 70)

    out: dict[str, Any] = {
        "arm": 3,
        "arm_name": "covariate_23feat",
        "target": target_key,
        "target_col": target_col,
        "seed": SEED,
    }

    # 23-feat run (pdmedyn INCLUDED as feature)
    X_raw_23, y_23, feat23 = prepare_xy(
        df, target_col, exclude_stage0=exclude0, include_pdmedyn=True
    )
    y_true_23, y_pred_23, y_prob_23, fold_metrics_23 = cv_catboost(
        X_raw_23, y_23, n_classes, is_ord, n_folds=5, seed=SEED
    )
    auc_23 = compute_auc(y_true_23, y_prob_23, n_classes)
    bal_23 = float(balanced_accuracy_score(y_true_23, y_pred_23))
    ci_23, n_valid_23 = bootstrap_auc_ci(y_true_23, y_prob_23, n_classes, seed=SEED)

    # 22-feat rerun (pdmedyn EXCLUDED) — re-run in-script so delta is from a
    # fresh matched run rather than comparing to a stored JSON (avoids
    # sklearn/catboost version drift creating spurious deltas).
    X_raw_22, y_22, feat22 = prepare_xy(
        df, target_col, exclude_stage0=exclude0, include_pdmedyn=False
    )
    assert len(y_22) == len(y_23), (
        "Arm 3 sanity: 22- and 23-feature cohorts must have identical N "
        f"({len(y_22)} != {len(y_23)})"
    )
    y_true_22, y_pred_22, y_prob_22, fold_metrics_22 = cv_catboost(
        X_raw_22, y_22, n_classes, is_ord, n_folds=5, seed=SEED
    )
    auc_22 = compute_auc(y_true_22, y_prob_22, n_classes)
    bal_22 = float(balanced_accuracy_score(y_true_22, y_pred_22))
    ci_22, n_valid_22 = bootstrap_auc_ci(y_true_22, y_prob_22, n_classes, seed=SEED)

    # Paired bootstrap delta (same seed → same resample indices)
    rng = np.random.default_rng(SEED)
    n = len(y_true_23)
    deltas: list[float] = []
    for _ in range(N_BOOTSTRAP_CI):
        idx = rng.integers(0, n, n)
        yt_23 = y_true_23[idx]
        yp_23 = y_prob_23[idx]
        yt_22 = y_true_22[idx]
        yp_22 = y_prob_22[idx]
        if n_classes == 2:
            if len(np.unique(yt_23)) < 2 or len(np.unique(yt_22)) < 2:
                continue
            try:
                a_23 = float(roc_auc_score(yt_23, yp_23[:, 1]))
                a_22 = float(roc_auc_score(yt_22, yp_22[:, 1]))
            except ValueError:
                continue
        else:
            if (
                len(np.unique(yt_23)) < n_classes
                or len(np.unique(yt_22)) < n_classes
            ):
                continue
            try:
                a_23 = float(
                    roc_auc_score(yt_23, yp_23, multi_class="ovr", average="macro")
                )
                a_22 = float(
                    roc_auc_score(yt_22, yp_22, multi_class="ovr", average="macro")
                )
            except ValueError:
                continue
        deltas.append(a_23 - a_22)

    arr = np.asarray(deltas) if deltas else np.asarray([np.nan])
    ci_delta = (
        [float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))]
        if np.isfinite(arr).any()
        else None
    )
    p_two_sided = (
        float(2.0 * min((arr >= 0).mean(), (arr <= 0).mean()))
        if np.isfinite(arr).any() else float("nan")
    )
    ci_excludes_zero = (
        ci_delta is not None and (
            (ci_delta[0] > 0.0 and ci_delta[1] > 0.0)
            or (ci_delta[0] < 0.0 and ci_delta[1] < 0.0)
        )
    )

    out.update({
        "n": int(len(y_23)),
        "ws1_1_reference_auc": WS1_1_BASELINE_AUC.get(target_key),
        "result_23feat": {
            "n_features": int(X_raw_23.shape[1]),
            "auc_point": auc_23,
            "auc_95ci": ci_23,
            "balanced_accuracy": bal_23,
            "fold_metrics": fold_metrics_23,
        },
        "result_22feat_rerun": {
            "n_features": int(X_raw_22.shape[1]),
            "auc_point": auc_22,
            "auc_95ci": ci_22,
            "balanced_accuracy": bal_22,
            "fold_metrics": fold_metrics_22,
        },
        "delta_arm3": float(auc_23 - auc_22),
        "delta_arm3_paired_bootstrap_95ci": ci_delta,
        "delta_arm3_bootstrap_two_sided_p": p_two_sided,
        "delta_arm3_ci_excludes_zero": bool(ci_excludes_zero),
        "delta_vs_ws1_1_stored": (
            float(auc_23 - WS1_1_BASELINE_AUC[target_key])
            if target_key in WS1_1_BASELINE_AUC and np.isfinite(auc_23) else None
        ),
    })

    logger.info(
        f"  23-feat AUC = {auc_23:.4f}"
        + (f" [{ci_23[0]:.4f}, {ci_23[1]:.4f}]" if ci_23 else "")
        + f" | 22-feat rerun AUC = {auc_22:.4f}"
        + (f" [{ci_22[0]:.4f}, {ci_22[1]:.4f}]" if ci_22 else "")
    )
    logger.info(
        f"  delta_arm3 = {out['delta_arm3']:+.4f} "
        + (f"[{ci_delta[0]:+.4f}, {ci_delta[1]:+.4f}]" if ci_delta else "")
        + f" p={p_two_sided:.3f}"
    )

    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


ARM_RUNNERS = {
    1: run_arm1,
    2: run_arm2,
    3: run_arm3,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--arm",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Which pre-registered sensitivity arm to run",
    )
    p.add_argument(
        "--target",
        type=str,
        choices=list(TARGETS.keys()),
        default="binary",
        help="NSD-ISS target type (default: binary; primary decision target)",
    )
    args = p.parse_args()

    t0 = time.time()
    df = load_joined_frame()
    runner = ARM_RUNNERS[args.arm]
    result = runner(df, args.target)
    result["elapsed_seconds"] = float(time.time() - t0)
    result["run_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    out_path = RESULTS_DIR / f"arm_{args.arm}_{args.target}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(f"Wrote {out_path} (elapsed: {result['elapsed_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
