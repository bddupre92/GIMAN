"""Paper 1 WS1.9 — SHAP + Genetic-Carrier Subgroup Fairness.

Addresses reviewer Q7 for the IEEE JBHI Paper 1 revision:

    Q7. "Provide SHAP feature-importance analysis per CatBoost target and
         report performance stratified by age, sex, and genetic-carrier
         status (LRRK2, GBA, APOE-e4)."

Age + sex stratification is already reported in Analyses A + B of
scripts/paper1/run_confounder_sensitivity.py and §V confounder-sensitivity
subsection of the revision. This runner adds the two remaining pieces:

    Part A — SHAP: TreeSHAP values from a fold-local CatBoost on all 4
             target formulations (binary, 3class, full_ordinal,
             nsd_positive). Reports top-10 features by mean |SHAP| averaged
             across the 5 CV folds + beeswarm plots + DaT-clinical
             interaction effects for the Stage-1 minority class.

    Part B — Genetic-carrier subgroups: re-derives LRRK2 / GBA / APOE-e4
             pathogenic-variant flags from the raw IU Genetic Consensus
             file because the assemble_paper1_features.py extract_genetics
             logic uses str.contains("CARRIER|POSITIVE|YES") which does
             NOT match the actual IU labels ("G2019S", "N409S", "R1441G",
             ...). As a result, lrrk2_carrier and gba_carrier are all-zero
             in features.paper1_features_with_targets (audited 2026-04-23).
             APOE-e4 is correct as-is.

             For each of 4 targets x 4 strata (LRRK2+, GBA+ only,
             APOE-e4+ only, non-carrier reference), refit fold-local
             CatBoost, score on within-stratum held-out predictions,
             compute bootstrap 95% CIs, and do a paired-bootstrap
             interaction delta vs the non-carrier reference. BH-FDR
             correct across the 12 tests.

Decision rule locked in ``outputs/paper1_shap_subgroup/PRE_REGISTRATION.md``
(commit 84d7977) BEFORE this script was written.

Dependencies
------------
- shap >= 0.44        # TreeExplainer + beeswarm plots
- catboost >= 1.2     # model
- scikit-learn        # ROC AUC + StratifiedKFold + imputer
- scipy >= 1.11       # stats.false_discovery_control (BH-FDR)
- pandas, numpy, matplotlib

Usage
-----
    python scripts/paper1/run_shap_subgroup.py --part both
    python scripts/paper1/run_shap_subgroup.py --part shap --target binary
    python scripts/paper1/run_shap_subgroup.py --part subgroup

Outputs are written under ``outputs/paper1_shap_subgroup/``.

Author: Blair Dupre (UND BME)
Date: 2026-04-23
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Project setup
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.staging.target_encoding import (  # noqa: E402
    compute_balanced_weights,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURES_PATH = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
IU_GENETIC_PATH = (
    ROOT
    / "data"
    / "00_raw"
    / "GIMAN"
    / "ppmi_data_csv"
    / "iu_genetic_consensus_20250515_08Oct2025.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "paper1_shap_subgroup"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature set (matches run_fold_local_imputation.py)
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
HIGH_MISS_COLS = {"UPDRS4_TOTAL", "MOCA_TOTAL", "updrs4_total", "moca_total"}

TARGETS: dict[str, dict[str, Any]] = {
    "binary": {
        "column": "target_binary",
        "n_classes": 2,
        "is_ordinal": False,
        "exclude_stage0": False,
    },
    "three_class": {
        "column": "target_3class",
        "n_classes": 3,
        "is_ordinal": True,
        "exclude_stage0": False,
    },
    "full_ordinal": {
        "column": "target_full_ordinal",
        "n_classes": 5,
        "is_ordinal": True,
        "exclude_stage0": False,
    },
    "nsd_positive": {
        "column": "target_nsd_positive",
        "n_classes": 4,
        "is_ordinal": True,
        "exclude_stage0": True,
    },
}

SEED = 42
N_FOLDS = 5
N_BOOTSTRAP = 1000
MIN_STRATUM_SIZE = 20

# Non-carrier IU labels (treated as "0" for carrier derivation).
IU_NEGATIVE_LABELS = {"0", "NA", "", "LRRK2", "GBA"}
# "LRRK2" and "GBA" appear as single-row header-echo artifacts in some IU
# vintages; treat them as missing rather than as variant names.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("paper1_ws1_9")


# ---------------------------------------------------------------------------
# Genetic carrier re-derivation (Part B)
# ---------------------------------------------------------------------------


def load_pathogenic_carrier_flags() -> pd.DataFrame:
    """Re-derive LRRK2 / GBA / APOE-e4 carrier flags from raw IU file.

    Returns a frame indexed by PATNO with columns:

        - lrrk2_pathogenic (int 0/1)
        - gba_pathogenic (int 0/1)
        - apoe_e4_pathogenic (int 0/1)
        - lrrk2_variant (str or "0")
        - gba_variant (str or "0")
        - apoe_genotype (str or "NA")
    """
    if not IU_GENETIC_PATH.exists():
        raise FileNotFoundError(f"IU genetic file not found at {IU_GENETIC_PATH}")

    iu = pd.read_csv(IU_GENETIC_PATH, low_memory=False)
    log.info(
        f"Loaded IU genetic consensus: {len(iu)} rows, cols {list(iu.columns)[:8]}..."
    )

    # Some IU vintages have multiple rows per PATNO for longitudinal sequencing;
    # keep the first (ordered as downloaded).
    iu = iu.drop_duplicates(subset=["PATNO"], keep="first").copy()

    out = iu[["PATNO"]].copy()

    def _to_flag(series: pd.Series) -> np.ndarray:
        s = series.astype(str).str.strip().str.strip('"')
        return (~s.isin(IU_NEGATIVE_LABELS)).astype(int).values

    lrrk = iu["LRRK2"].astype(str).str.strip().str.strip('"')
    gba = iu["GBA"].astype(str).str.strip().str.strip('"')
    apoe = iu["APOE"].astype(str).str.strip().str.strip('"')

    out["lrrk2_pathogenic"] = _to_flag(lrrk)
    out["gba_pathogenic"] = _to_flag(gba)
    # APOE-e4: any genotype string containing the digit 4
    out["apoe_e4_pathogenic"] = apoe.str.contains("4", na=False).astype(int).values
    out["lrrk2_variant"] = lrrk.values
    out["gba_variant"] = gba.values
    out["apoe_genotype"] = apoe.values

    n_lrrk = int(out["lrrk2_pathogenic"].sum())
    n_gba = int(out["gba_pathogenic"].sum())
    n_apoe = int(out["apoe_e4_pathogenic"].sum())
    log.info(
        f"Re-derived IU carrier flags: LRRK2+={n_lrrk}, GBA+={n_gba}, "
        f"APOE-e4+={n_apoe} (from {len(out)} patients)"
    )

    return out


def assign_subgroups(df_with_flags: pd.DataFrame) -> pd.Series:
    """Assign each patient to exactly one of 4 non-overlapping subgroups.

    Precedence (when multiple flags fire): LRRK2+ > GBA+ > APOE-e4+ > Non-carrier.
    This prioritises Mendelian-risk variants (LRRK2/GBA) over APOE and
    collapses clinically-dominant variants first, matching TRIPOD+AI subgroup
    reporting conventions for PD cohorts.
    """

    def _assign(row: pd.Series) -> str:
        if row["lrrk2_pathogenic"] == 1:
            return "LRRK2+"
        if row["gba_pathogenic"] == 1:
            return "GBA+_only"
        if row["apoe_e4_pathogenic"] == 1:
            return "APOE_e4+_only"
        return "Non-carrier"

    return df_with_flags.apply(_assign, axis=1)


# ---------------------------------------------------------------------------
# Data preparation (fold-local-ready)
# ---------------------------------------------------------------------------


def prepare_xy(
    df: pd.DataFrame, target_col: str, exclude_stage0: bool
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Return (X_raw, y, feature_cols, patno) for a target.

    Returns X_raw WITHOUT imputation (fold-local imputation is applied
    inside the CV loop).
    """
    feature_cols = [
        c for c in df.columns if c not in STAGING_COLS and c not in HIGH_MISS_COLS
    ]

    mask = df[target_col] >= 0
    if exclude_stage0:
        mask = mask & (df["nsd_iss_stage"] != "0")

    sub = df[mask].copy()
    y = sub[target_col].values.astype(int)
    patno = sub["PATNO"].values.astype(int) if "PATNO" in sub.columns else sub["patno"].values.astype(int)

    X = sub[feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    return X.values, y, feature_cols, patno


def _fit_catboost(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    random_state: int,
):
    """Factory-build + fit a fresh CatBoost classifier (per WS1.1 recipe)."""
    import catboost as cb

    model = cb.CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        random_seed=random_state,
        auto_class_weights="Balanced",
        verbose=0,
    )
    model.fit(X_train_s, y_train)
    return model


# ---------------------------------------------------------------------------
# Part A — SHAP
# ---------------------------------------------------------------------------


@dataclass
class ShapFoldResult:
    fold_idx: int
    shap_values: np.ndarray  # (n_test, n_features) or (n_test, n_features, n_classes)
    test_idx: np.ndarray
    feature_names: list[str]


def compute_shap_for_target(
    df: pd.DataFrame,
    target_name: str,
    target_spec: dict[str, Any],
) -> dict[str, Any]:
    """Run fold-local CatBoost + TreeSHAP for one target.

    Returns a result dict with per-fold SHAP arrays, aggregate mean |SHAP|
    per feature, top-10 ranking, and rank stability across folds.
    """
    import shap

    log.info(
        f"[SHAP] target={target_name} (n_classes={target_spec['n_classes']}, "
        f"ordinal={target_spec['is_ordinal']})"
    )

    X_raw, y, feat_names, _ = prepare_xy(
        df, target_spec["column"], target_spec["exclude_stage0"]
    )
    log.info(f"  N={len(y)}, features={len(feat_names)}, "
             f"class dist={np.bincount(y, minlength=target_spec['n_classes']).tolist()}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results: list[ShapFoldResult] = []
    # mean_abs[fold, feature]
    mean_abs_per_fold = np.zeros((N_FOLDS, len(feat_names)))
    # Accumulated (test_idx, shap, X_test) for beeswarm on pooled test folds
    all_shap = []
    all_X_test_s = []
    all_test_idx = []

    for fold, (tr, te) in enumerate(skf.split(X_raw, y)):
        X_tr, X_te = X_raw[tr], X_raw[te]
        y_tr = y[tr]

        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_tr)
        X_te_imp = imp.transform(X_te)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_imp)
        X_te_s = sc.transform(X_te_imp)

        model = _fit_catboost(X_tr_s, y_tr, target_spec["n_classes"], SEED)

        # TreeSHAP on held-out fold
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_te_s)
        # For multiclass catboost, TreeExplainer returns ndarray of shape
        # (n_test, n_features, n_classes). For binary, shape (n_test, n_features).
        shap_arr = np.asarray(shap_vals)
        if shap_arr.ndim == 3:
            # Multiclass: reduce over classes for overall importance
            mean_abs = np.mean(np.mean(np.abs(shap_arr), axis=-1), axis=0)
        else:
            mean_abs = np.mean(np.abs(shap_arr), axis=0)

        mean_abs_per_fold[fold] = mean_abs
        fold_results.append(
            ShapFoldResult(
                fold_idx=fold,
                shap_values=shap_arr,
                test_idx=te,
                feature_names=feat_names,
            )
        )
        all_shap.append(shap_arr)
        all_X_test_s.append(X_te_s)
        all_test_idx.append(te)

        log.info(f"  Fold {fold}: mean|SHAP| sum={mean_abs.sum():.3f}")

    # Aggregate across folds
    mean_abs_agg = mean_abs_per_fold.mean(axis=0)
    order = np.argsort(mean_abs_agg)[::-1]
    top10 = [(feat_names[i], float(mean_abs_agg[i])) for i in order[:10]]

    # Rank stability: Spearman rho between each pair of folds on top-10 ranks
    from scipy.stats import spearmanr

    rank_per_fold = np.argsort(np.argsort(-mean_abs_per_fold, axis=1), axis=1)
    rhos = []
    for i in range(N_FOLDS):
        for j in range(i + 1, N_FOLDS):
            rho, _ = spearmanr(rank_per_fold[i], rank_per_fold[j])
            rhos.append(float(rho))
    rho_mean = float(np.mean(rhos))
    rho_std = float(np.std(rhos))

    # Pool across folds for beeswarm figure
    pooled_shap = np.concatenate(all_shap, axis=0)
    pooled_X = np.concatenate(all_X_test_s, axis=0)

    # Stage-1 DaT x clinical interaction effects (full_ordinal only)
    interaction_effects: dict[str, float] | None = None
    if target_name == "full_ordinal":
        interaction_effects = compute_stage1_interaction_effects(
            df, target_spec, feat_names
        )

    # Write raw per-fold arrays to npz
    npz_path = OUTPUT_DIR / f"shap_values_{target_name}.npz"
    np.savez_compressed(
        npz_path,
        pooled_shap=pooled_shap,
        pooled_X=pooled_X,
        mean_abs_per_fold=mean_abs_per_fold,
        mean_abs_agg=mean_abs_agg,
        feature_names=np.array(feat_names, dtype=object),
    )
    log.info(f"  Saved SHAP arrays to {npz_path}")

    # Beeswarm figure
    _plot_beeswarm(
        pooled_shap=pooled_shap,
        pooled_X=pooled_X,
        feat_names=feat_names,
        target_name=target_name,
        n_classes=target_spec["n_classes"],
    )

    out = {
        "target": target_name,
        "n_samples": int(len(y)),
        "n_features": int(len(feat_names)),
        "top10_features": top10,
        "mean_abs_per_feature": {
            feat_names[i]: float(mean_abs_agg[i]) for i in range(len(feat_names))
        },
        "rank_stability_spearman_rho_mean": rho_mean,
        "rank_stability_spearman_rho_std": rho_std,
        "interaction_effects_stage1": interaction_effects,
    }

    return out


def compute_stage1_interaction_effects(
    df: pd.DataFrame,
    target_spec: dict[str, Any],
    feat_names: list[str],
) -> dict[str, float]:
    """Compute SHAP interaction values for Stage-1 DaT x clinical pairs.

    Uses ``shap.TreeExplainer(model).shap_interaction_values(X)`` on a
    single fit of CatBoost over the full Stage-1-preserving cohort. The
    reported metric is mean |interaction SHAP| for each DaT-x-clinical pair,
    restricted to Stage-1 samples.
    """
    import shap

    # Match DaT-SPECT / caudate-SBR features (case-insensitive on name) and
    # the 11 canonical clinical covariates used by the 22-feature Paper 1
    # model. Using c.lower() in {...lowercase...} keeps this robust to
    # whichever case the feature CSV exposes.
    dat_features = [
        c
        for c in feat_names
        if ("sbr" in c.lower()) or ("caudate" in c.lower())
    ]
    clinical_names_lower = {
        "age_at_baseline",
        "sex",
        "updrs1_total",
        "updrs2_total",
        "updrs3_tremor",
        "updrs3_rigidity",
        "updrs3_bradykinesia",
        "updrs3_axial",
        "rbd_total",
        "ess_total",
        "scopa_aut_total",
    }
    clinical_features = [c for c in feat_names if c.lower() in clinical_names_lower]

    X_raw, y, _, _ = prepare_xy(
        df, target_spec["column"], target_spec["exclude_stage0"]
    )
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_raw)
    sc = StandardScaler()
    X_s = sc.fit_transform(X_imp)

    # Single fit on full cohort (interaction values are qualitative; fold-local
    # refits of shap_interaction_values would blow the runtime budget).
    model = _fit_catboost(X_s, y, target_spec["n_classes"], SEED)

    explainer = shap.TreeExplainer(model)
    try:
        inter_vals = explainer.shap_interaction_values(X_s)
    except Exception as exc:  # pragma: no cover — guard against CB multiclass
        log.warning(f"  interaction values unavailable: {exc}")
        return {"error": str(exc)}

    inter_arr = np.asarray(inter_vals)
    # Multi-class: (n_classes, n_samples, n_features, n_features). Binary:
    # (n_samples, n_features, n_features).
    stage1_mask = y == 1
    if not stage1_mask.any():
        return {}

    effects: dict[str, float] = {}
    for dat in dat_features:
        di = feat_names.index(dat)
        for clin in clinical_features:
            ci = feat_names.index(clin)
            if inter_arr.ndim == 4:
                vals = inter_arr[:, stage1_mask, di, ci]
                eff = float(np.mean(np.abs(vals)))
            else:
                vals = inter_arr[stage1_mask, di, ci]
                eff = float(np.mean(np.abs(vals)))
            effects[f"{dat}__x__{clin}"] = eff
    return effects


def _plot_beeswarm(
    pooled_shap: np.ndarray,
    pooled_X: np.ndarray,
    feat_names: list[str],
    target_name: str,
    n_classes: int,
) -> None:
    """Render a SHAP beeswarm with matplotlib fallback for multiclass."""
    import shap

    try:
        if pooled_shap.ndim == 3:
            # Multi-class: plot mean |SHAP| across classes as a bar chart
            # (beeswarm is not informative when collapsed across classes).
            mean_abs = np.mean(np.mean(np.abs(pooled_shap), axis=-1), axis=0)
            order = np.argsort(mean_abs)[::-1][:15]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(
                [feat_names[i] for i in order[::-1]],
                mean_abs[order][::-1],
                color="#1f77b4",
            )
            ax.set_xlabel("Mean |SHAP value| (across classes)")
            ax.set_title(f"Feature importance — {target_name} ({n_classes} classes)")
            plt.tight_layout()
        else:
            # Binary: true beeswarm
            expl_like = shap.Explanation(
                values=pooled_shap,
                data=pooled_X,
                feature_names=feat_names,
            )
            fig = plt.figure(figsize=(8, 6))
            shap.plots.beeswarm(expl_like, max_display=15, show=False)
            plt.title(f"SHAP — {target_name}")
            plt.tight_layout()
    except Exception as exc:
        log.warning(f"  beeswarm plot failed for {target_name}: {exc}")
        return

    pdf = OUTPUT_DIR / f"fig_shap_{target_name}.pdf"
    png = OUTPUT_DIR / f"fig_shap_{target_name}.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, bbox_inches="tight", dpi=200)
    plt.close()
    log.info(f"  Saved beeswarm to {pdf.name} / {png.name}")


# ---------------------------------------------------------------------------
# Part B — Subgroup fairness
# ---------------------------------------------------------------------------


def _compute_subgroup_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int,
) -> float | None:
    """Compute AUC for a subgroup's held-out predictions.

    Binary: standard ROC-AUC on the positive-class probability.
    Multi-class: macro one-vs-rest AUC. Returns None when undefined
    (e.g., only one class present in the stratum).
    """
    unique = np.unique(y_true)
    if n_classes == 2:
        if len(unique) < 2:
            return None
        # y_prob: (n, 2) -> positive class prob in column 1
        return float(roc_auc_score(y_true, y_prob[:, 1]))
    # Multi-class: require every class present for macro-OvR to be defined.
    if len(unique) < n_classes:
        return None
    try:
        return float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
    except ValueError:
        return None


def _bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float] | None:
    """Bootstrap AUC point estimate + 95% CI.

    Returns ``(point, ci_low, ci_high)`` or None if undefined.
    """
    point = _compute_subgroup_auc(y_true, y_prob, n_classes)
    if point is None:
        return None

    boots = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = _compute_subgroup_auc(y_true[idx], y_prob[idx], n_classes)
        if v is not None:
            boots.append(v)
    if not boots:
        return (point, float("nan"), float("nan"))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (point, float(lo), float(hi))


def _bootstrap_delta_auc(
    y_sub: np.ndarray,
    p_sub: np.ndarray,
    y_ref: np.ndarray,
    p_ref: np.ndarray,
    n_classes: int,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Bootstrap ΔAUC = AUC(subgroup) − AUC(reference).

    Subgroup and reference are drawn independently because they are
    disjoint patient sets (the "paired" aspect is that each bootstrap
    resample redraws both together). Returns point estimate + 95% CI +
    two-sided bootstrap p-value (fraction of resamples where sign flips).
    """
    delta_boots = []
    n_s = len(y_sub)
    n_r = len(y_ref)
    for _ in range(n_boot):
        i_s = rng.integers(0, n_s, size=n_s)
        i_r = rng.integers(0, n_r, size=n_r)
        a_s = _compute_subgroup_auc(y_sub[i_s], p_sub[i_s], n_classes)
        a_r = _compute_subgroup_auc(y_ref[i_r], p_ref[i_r], n_classes)
        if a_s is not None and a_r is not None:
            delta_boots.append(a_s - a_r)
    if not delta_boots:
        return {
            "delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
        }
    delta_arr = np.asarray(delta_boots)
    point = _compute_subgroup_auc(y_sub, p_sub, n_classes)
    ref = _compute_subgroup_auc(y_ref, p_ref, n_classes)
    point_delta = (
        point - ref if (point is not None and ref is not None) else float("nan")
    )
    lo, hi = np.percentile(delta_arr, [2.5, 97.5])
    # Two-sided p = 2 * min(P(delta_boots >= 0), P(delta_boots <= 0))
    p_pos = float(np.mean(delta_arr >= 0))
    p_neg = float(np.mean(delta_arr <= 0))
    p_two_sided = 2.0 * min(p_pos, p_neg)
    p_two_sided = max(min(p_two_sided, 1.0), 1.0 / n_boot)

    return {
        "delta": float(point_delta),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(p_two_sided),
    }


def run_subgroup_for_target(
    df: pd.DataFrame,
    target_name: str,
    target_spec: dict[str, Any],
    subgroups: pd.Series,
) -> dict[str, Any]:
    """Run fold-local CatBoost + subgroup AUC bootstrap for one target.

    ``subgroups`` is a Series indexed by PATNO with values in
    {"LRRK2+", "GBA+_only", "APOE_e4+_only", "Non-carrier"}.
    """
    rng = np.random.default_rng(SEED)

    X_raw, y, feat_names, patno = prepare_xy(
        df, target_spec["column"], target_spec["exclude_stage0"]
    )
    n_classes = target_spec["n_classes"]
    log.info(
        f"[SUBGROUP] target={target_name} N={len(y)} classes={n_classes}"
    )

    # Materialise out-of-fold predictions (probabilities) for every patient.
    oof_prob = np.full((len(y), n_classes), np.nan, dtype=float)
    oof_true = np.asarray(y)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(skf.split(X_raw, y)):
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_raw[tr])
        X_te_imp = imp.transform(X_raw[te])
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_imp)
        X_te_s = sc.transform(X_te_imp)

        model = _fit_catboost(X_tr_s, y[tr], n_classes, SEED)
        oof_prob[te] = model.predict_proba(X_te_s)
        log.info(f"  fold {fold}: OOF probabilities written for {len(te)} patients")

    # Map PATNO -> subgroup
    sub_map = subgroups.reindex(patno)
    sub_map = sub_map.fillna("Non-carrier")  # missing IU → treat as non-carrier

    groups_order = ["LRRK2+", "GBA+_only", "APOE_e4+_only", "Non-carrier"]

    per_group: dict[str, dict[str, Any]] = {}
    for g in groups_order:
        mask = (sub_map.values == g)
        n = int(mask.sum())
        y_g = oof_true[mask]
        p_g = oof_prob[mask]
        descriptive = n < MIN_STRATUM_SIZE

        auc_result = _bootstrap_auc_ci(
            y_g, p_g, n_classes, n_boot=N_BOOTSTRAP if not descriptive else 0,
            rng=rng,
        )
        if auc_result is None:
            per_group[g] = {
                "n": n,
                "auc": None,
                "ci_low": None,
                "ci_high": None,
                "descriptive_only": True,
                "reason": "undefined_AUC",
            }
        elif descriptive:
            per_group[g] = {
                "n": n,
                "auc": auc_result[0],
                "ci_low": None,
                "ci_high": None,
                "descriptive_only": True,
                "reason": f"stratum_size_below_{MIN_STRATUM_SIZE}",
            }
        else:
            per_group[g] = {
                "n": n,
                "auc": auc_result[0],
                "ci_low": auc_result[1],
                "ci_high": auc_result[2],
                "descriptive_only": False,
            }

    # Interaction deltas vs non-carrier reference
    ref_mask = (sub_map.values == "Non-carrier")
    y_ref = oof_true[ref_mask]
    p_ref = oof_prob[ref_mask]
    per_group_delta: dict[str, dict[str, Any]] = {}
    for g in ["LRRK2+", "GBA+_only", "APOE_e4+_only"]:
        mask = (sub_map.values == g)
        n = int(mask.sum())
        if (
            per_group[g].get("descriptive_only", False)
            or per_group["Non-carrier"].get("descriptive_only", False)
            or per_group[g]["auc"] is None
            or per_group["Non-carrier"]["auc"] is None
        ):
            per_group_delta[g] = {
                "n": n,
                "delta": None,
                "ci_low": None,
                "ci_high": None,
                "p_raw": None,
                "skipped_reason": "descriptive_or_undefined",
            }
            continue
        y_g = oof_true[mask]
        p_g = oof_prob[mask]
        delta = _bootstrap_delta_auc(
            y_g, p_g, y_ref, p_ref, n_classes, N_BOOTSTRAP, rng
        )
        per_group_delta[g] = {
            "n": n,
            "delta": delta["delta"],
            "ci_low": delta["ci_low"],
            "ci_high": delta["ci_high"],
            "p_raw": delta["p_value"],
            "skipped_reason": None,
        }

    return {
        "target": target_name,
        "n_samples": int(len(y)),
        "per_group_auc": per_group,
        "per_group_delta_vs_reference": per_group_delta,
    }


def bh_fdr_correction(p_values: list[float | None]) -> list[float | None]:
    """Benjamini-Hochberg FDR correction, preserving None for skipped tests."""
    from scipy.stats import false_discovery_control

    idx = [i for i, p in enumerate(p_values) if p is not None]
    raw = np.asarray([p_values[i] for i in idx], dtype=float)
    if len(raw) == 0:
        return p_values
    adj = false_discovery_control(raw, method="bh")
    out: list[float | None] = list(p_values)
    for k, i in enumerate(idx):
        out[i] = float(adj[k])
    return out


def plot_subgroup_forest(subgroup_results: dict[str, Any]) -> None:
    """Render a forest plot of the 9–12 subgroup ΔAUC estimates."""
    rows = []
    for target_name, res in subgroup_results.items():
        for g, d in res["per_group_delta_vs_reference"].items():
            if d["delta"] is None:
                continue
            rows.append(
                dict(
                    label=f"{target_name} / {g}",
                    delta=d["delta"],
                    lo=d["ci_low"],
                    hi=d["ci_high"],
                    p_raw=d["p_raw"],
                    p_fdr=d.get("p_fdr"),
                    n=d["n"],
                )
            )
    if not rows:
        log.warning("No plottable subgroup deltas — skipping forest plot")
        return

    fig, ax = plt.subplots(figsize=(8, max(3, len(rows) * 0.35)))
    ys = np.arange(len(rows))
    for i, r in enumerate(rows):
        color = "#d62728" if (r["p_fdr"] is not None and r["p_fdr"] < 0.05) else "#1f77b4"
        ax.errorbar(
            r["delta"],
            i,
            xerr=[[r["delta"] - r["lo"]], [r["hi"] - r["delta"]]],
            fmt="o",
            color=color,
            capsize=4,
        )
    ax.axvline(0.0, color="black", lw=0.8, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("ΔAUC vs non-carrier reference (95% CI)")
    ax.set_title("Genetic-carrier subgroup fairness (BH-FDR annotated)")
    plt.tight_layout()
    pdf = OUTPUT_DIR / "fig_subgroup_forest.pdf"
    png = OUTPUT_DIR / "fig_subgroup_forest.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, bbox_inches="tight", dpi=200)
    plt.close()
    log.info(f"Saved subgroup forest plot to {pdf.name} / {png.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part",
        choices=["shap", "subgroup", "both"],
        default="both",
        help="Which analysis to run.",
    )
    parser.add_argument(
        "--target",
        choices=list(TARGETS.keys()) + ["all"],
        default="all",
        help="Target formulation (default: all four).",
    )
    args = parser.parse_args()

    targets = list(TARGETS.keys()) if args.target == "all" else [args.target]

    log.info("=" * 70)
    log.info("Paper 1 WS1.9 — SHAP + Genetic-Carrier Subgroup Fairness")
    log.info(f"  Part:   {args.part}")
    log.info(f"  Targets: {targets}")
    log.info(f"  Output:  {OUTPUT_DIR}")
    log.info("=" * 70)

    t0 = time.time()
    df = pd.read_csv(FEATURES_PATH)
    # The CSV mixes lowercase staging/target columns (nsd_iss_stage,
    # target_binary, ...) with uppercase feature columns (AGE_AT_BASELINE,
    # LRRK2_CARRIER, ...). Preserve the native case; STAGING_COLS +
    # HIGH_MISS_COLS already include both vintages.
    log.info(f"Loaded {len(df)} patients, {len(df.columns)} columns")

    # ---------------- Part A — SHAP ----------------
    shap_results: dict[str, Any] = {}
    if args.part in {"shap", "both"}:
        for tn in targets:
            res = compute_shap_for_target(df, tn, TARGETS[tn])
            shap_results[tn] = res
            (OUTPUT_DIR / f"top_features_{tn}.json").write_text(
                json.dumps(res, indent=2), encoding="utf-8"
            )
            log.info(f"  Saved top_features_{tn}.json")

    # ---------------- Part B — Subgroup ----------------
    subgroup_results: dict[str, Any] = {}
    if args.part in {"subgroup", "both"}:
        flags = load_pathogenic_carrier_flags()
        flags_indexed = flags.set_index("PATNO")
        subgroups = assign_subgroups(flags_indexed)

        # Log subgroup sizes at the cohort level
        counts = subgroups.value_counts().to_dict()
        log.info(f"Cohort-level subgroup sizes (IU-file universe): {counts}")

        for tn in targets:
            res = run_subgroup_for_target(df, tn, TARGETS[tn], subgroups)
            subgroup_results[tn] = res

        # Gather raw p-values across the 12-test FDR family
        p_raw_list: list[float | None] = []
        for tn in targets:
            for g in ["LRRK2+", "GBA+_only", "APOE_e4+_only"]:
                p = subgroup_results[tn]["per_group_delta_vs_reference"][g]["p_raw"]
                p_raw_list.append(p)
        p_fdr_list = bh_fdr_correction(p_raw_list)
        i = 0
        for tn in targets:
            for g in ["LRRK2+", "GBA+_only", "APOE_e4+_only"]:
                subgroup_results[tn]["per_group_delta_vs_reference"][g][
                    "p_fdr"
                ] = p_fdr_list[i]
                i += 1

        for tn in targets:
            (OUTPUT_DIR / f"subgroup_{tn}.json").write_text(
                json.dumps(subgroup_results[tn], indent=2, default=str),
                encoding="utf-8",
            )
            log.info(f"  Saved subgroup_{tn}.json")

        plot_subgroup_forest(subgroup_results)

    # ---------------- Consolidated manifest ----------------
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": round(time.time() - t0, 2),
        "part": args.part,
        "targets": targets,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "n_bootstrap": N_BOOTSTRAP,
        "min_stratum_size": MIN_STRATUM_SIZE,
        "pre_registration": str(OUTPUT_DIR / "PRE_REGISTRATION.md"),
        "shap_summary": {
            tn: {
                "top10": r["top10_features"],
                "rank_stability_rho_mean": r["rank_stability_spearman_rho_mean"],
            }
            for tn, r in shap_results.items()
        },
        "subgroup_fdr_family_size": 3 * len(targets)
        if args.part in {"subgroup", "both"}
        else 0,
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    log.info(f"Saved manifest to {OUTPUT_DIR / 'run_manifest.json'}")
    log.info(f"TOTAL runtime: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
