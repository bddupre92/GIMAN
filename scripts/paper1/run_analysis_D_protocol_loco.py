"""Paper 1 Analysis D — DaT-SPECT Protocol-LOCO Sensitivity.

A pre-registered addition to the Paper 1 confounder sensitivity package that
replaces the unobtainable PPMI site-LOSO with a stratification by DaT-SPECT
acquisition protocol (ppmi_raw.datscan_sbr_analysis.protocol).

Rationale
---------
The originally-planned site-LOSO was abandoned because no canonical PPMI
site/center number column is present in our Postgres mirror or LONI IDA
snapshot. Protocol-LOCO isolates scanner/reconstruction drift WITHOUT
confounding by site-specific patient-demographic differences, and is
scientifically more meaningful: protocols correspond to PPMI SPECT
Technical Operations Manual revisions (v3.0 → v4.0, ~2018) which defined
standardized reconstruction pipelines.

Protocol buckets (counts from ppmi_raw.datscan_sbr_analysis)
------------------------------------------------------------
- protocol 001: n=2,267 scans (primary PPMI DaT-SPECT protocol, 2010–~2018 era)
- protocol 002: n=1,840 scans (updated protocol, ~2018+ era)
- protocol 004: n=49  scans (edge protocol)
- protocol T011: n=28 scans (edge protocol)

Baseline scan per PATNO is chosen (earliest datscan_date), with scans whose
`datscan_not_analyzed_reason` is populated excluded. Protocols 001 and 002
are held out in turn. Edge protocols (004 + T011) are reported descriptively
(n~77 combined) but NOT bootstrapped — too small for stable AUC CIs.

Protocol
--------
For each held-out protocol P:
  - Train CatBoost (iterations=1000, depth=6, seed=42, auto_class_weights=Balanced)
    on the OTHER two buckets (e.g. P=001 train on 002 + edge).
  - Predict on the held-out protocol cohort.
  - Bootstrap 1,000 resamples on held-out test set for 95% CI on AUC.
  - Run on target_binary + target_3class (skip full-ordinal and nsd_positive
    per the S-5 Analysis C scope decision).

Expected
--------
Both protocol holdouts should retain AUC ≥ 0.93 on binary, similar to the
Analysis C enrollment-wave envelope, confirming scanner-protocol
generalisability. If one protocol drops substantially, that's a
finding worth surfacing.

Hyperparameters exactly match Table I:
  CatBoost iterations=1000 depth=6 learning_rate=0.05 seed=42
  auto_class_weights="Balanced" n_bootstrap=1000

Author: Blair Dupre
Date: April 2026
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

OUT_DIR = ROOT / "outputs" / "paper1_confounder_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOTSTRAP_CI = 1000

# Match Table I: 22-feature CatBoost, drop high-missingness (UPDRS4, MOCA)
STAGING_COLS = {
    "patno",
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
HIGH_MISS_COLS = {"updrs4_total", "moca_total"}
AUX_COLS = {"protocol", "datscan_date"}


def load_joined_frame() -> pd.DataFrame:
    """Load paper1 features joined to each patient's baseline DaT-SPECT protocol.

    Baseline = earliest datscan_date per PATNO with datscan_not_analyzed_reason IS NULL.
    """
    q = """
    WITH baseline_scan AS (
        SELECT DISTINCT ON (patno)
               patno, protocol, datscan_date
        FROM ppmi_raw.datscan_sbr_analysis
        WHERE datscan_not_analyzed_reason IS NULL OR datscan_not_analyzed_reason = ''
        ORDER BY patno, datscan_date ASC
    )
    SELECT p1.*, b.protocol, b.datscan_date
    FROM features.paper1_features_with_targets p1
    INNER JOIN baseline_scan b ON p1.patno = b.patno
    WHERE p1.target_binary >= 0
    """
    with get_engine().connect() as c:
        df = pd.read_sql_query(text(q), c)

    # Bucket protocols: 001, 002, and "edge" (004 + T011)
    def bucket(p: str) -> str:
        p = str(p).strip()
        if p == "001":
            return "001"
        if p == "002":
            return "002"
        return "edge"  # covers 004, T011

    df["protocol_bucket"] = df["protocol"].apply(bucket)
    logger.info(
        f"Joined cohort: n={len(df)} patients with baseline analyzed DaT-SPECT; "
        f"bucket counts: {df.protocol_bucket.value_counts().to_dict()}"
    )
    raw_counts = df.protocol.value_counts().to_dict()
    logger.info(f"Raw protocol distribution: {raw_counts}")
    return df


def prepare_xy(
    df: pd.DataFrame, target_col: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare (X, y, feature_names) using the same 22-feature pipeline as Table I."""
    feature_cols = [
        c
        for c in df.columns
        if c not in STAGING_COLS
        and c not in HIGH_MISS_COLS
        and c not in AUX_COLS
        and c not in ("protocol_bucket",)
    ]
    mask = df[target_col] >= 0
    sub = df[mask].copy()

    y = sub[target_col].values.astype(int)
    X_raw = sub[feature_cols].copy()
    for col in feature_cols:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_raw.values)
    return X, y, feature_cols


def train_and_eval(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    n_classes: int,
    seed: int = SEED,
    n_bootstrap: int = N_BOOTSTRAP_CI,
) -> dict[str, Any]:
    """Train CatBoost on train fold, evaluate on held-out fold, bootstrap AUC CI.

    Matches Table I hyperparameters exactly.
    """
    import catboost as cb

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    model = cb.CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        random_seed=seed,
        auto_class_weights="Balanced",
        verbose=0,
    )
    model.fit(X_tr_s, y_tr)
    y_prob = model.predict_proba(X_te_s)
    y_pred = np.asarray(model.predict(X_te_s)).ravel().astype(int)
    bal_acc = float(balanced_accuracy_score(y_te, y_pred))

    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    if n_classes == 2:
        prob_pos = y_prob[:, 1]
        try:
            point_auc = float(roc_auc_score(y_te, prob_pos))
        except ValueError:
            point_auc = float("nan")
        macro = None
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(y_te), len(y_te))
            if len(np.unique(y_te[idx])) < 2:
                continue
            try:
                aucs.append(float(roc_auc_score(y_te[idx], prob_pos[idx])))
            except ValueError:
                continue
    else:
        try:
            point_auc = float(
                roc_auc_score(y_te, y_prob, multi_class="ovr", average="macro")
            )
        except ValueError:
            point_auc = float("nan")
        macro = point_auc
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(y_te), len(y_te))
            if len(np.unique(y_te[idx])) < n_classes:
                continue
            try:
                aucs.append(
                    float(
                        roc_auc_score(
                            y_te[idx],
                            y_prob[idx],
                            multi_class="ovr",
                            average="macro",
                        )
                    )
                )
            except ValueError:
                continue

    aucs_arr = np.array(aucs) if aucs else np.array([np.nan])
    ci_low = float(np.nanpercentile(aucs_arr, 2.5)) if np.isfinite(aucs_arr).any() else None
    ci_high = float(np.nanpercentile(aucs_arr, 97.5)) if np.isfinite(aucs_arr).any() else None

    return {
        "n_train": int(len(y_tr)),
        "n_held_out": int(len(y_te)),
        "balanced_accuracy": bal_acc,
        "auc_point": point_auc,
        "macro_auc_ovr": macro,
        "auc_bootstrap_95ci": [ci_low, ci_high] if ci_low is not None else None,
        "n_bootstrap_resamples_valid": int(np.isfinite(aucs_arr).sum()),
    }


def run_analysis_d(df: pd.DataFrame) -> dict[str, Any]:
    """Leave-one-protocol-out across DaT-SPECT acquisition protocols."""
    logger.info("=" * 60)
    logger.info("ANALYSIS D — DaT-SPECT Protocol-LOCO")
    logger.info("=" * 60)

    bucket_counts = df.protocol_bucket.value_counts().to_dict()
    raw_counts = df.protocol.value_counts().to_dict()

    out: dict[str, Any] = {
        "stratification_note": (
            "ppmi_raw.datscan_sbr_analysis.protocol is the DaT-SPECT acquisition "
            "protocol assigned per scan. 001 is the primary PPMI DaT-SPECT protocol "
            "(2010–~2018 era), 002 is the updated protocol (~2018+ era corresponding "
            "to PPMI SPECT Technical Operations Manual v4.0 standardisation), "
            "004/T011 are edge protocols. Protocol-LOCO isolates scanner/"
            "reconstruction drift without confounding by site-specific "
            "patient-demographic differences."
        ),
        "raw_protocol_counts_baseline_scans": raw_counts,
        "bucket_counts": bucket_counts,
        "n_total": int(len(df)),
        "targets": {},
    }

    # Report edge-bucket descriptively
    edge_n = bucket_counts.get("edge", 0)
    if edge_n > 0:
        edge_target_dist = (
            df.loc[df.protocol_bucket == "edge", "target_binary"].value_counts().to_dict()
        )
        out["edge_bucket_description"] = {
            "n": edge_n,
            "target_binary_distribution": {
                int(k): int(v) for k, v in edge_target_dist.items()
            },
            "note": (
                "Edge bucket (protocols 004 + T011) is underpowered; "
                "reported descriptively. It is INCLUDED in the training set "
                "when 001 or 002 is held out, so its signal is absorbed "
                "into the comparator models."
            ),
        }

    # LOCO over 001 and 002 only (edge too small to be held out on its own)
    heldout_protocols = ["001", "002"]

    for target, n_classes in [
        ("target_binary", 2),
        ("target_3class", 3),
    ]:
        logger.info(f"Target: {target}")
        per_protocol: dict[str, Any] = {}
        for held in heldout_protocols:
            held_df = df[df.protocol_bucket == held]
            train_df = df[df.protocol_bucket != held]
            X_tr, y_tr, _ = prepare_xy(train_df, target)
            X_te, y_te, _ = prepare_xy(held_df, target)
            if len(np.unique(y_tr[y_tr >= 0])) < n_classes:
                logger.warning(
                    f"  Protocol {held}: train-set has <{n_classes} classes, skipping"
                )
                continue
            if len(np.unique(y_te[y_te >= 0])) < 2:
                logger.warning(
                    f"  Protocol {held}: held-out has <2 classes, skipping"
                )
                continue
            r = train_and_eval(X_tr, y_tr, X_te, y_te, n_classes, seed=SEED)
            per_protocol[held] = r
            auc_display = f"{r['auc_point']:.4f}"
            if r["auc_bootstrap_95ci"]:
                auc_display += (
                    f" [{r['auc_bootstrap_95ci'][0]:.4f}, "
                    f"{r['auc_bootstrap_95ci'][1]:.4f}]"
                )
            logger.info(
                f"  Hold-out={held} n_te={r['n_held_out']} n_tr={r['n_train']} "
                f"bal_acc={r['balanced_accuracy']:.4f} AUC={auc_display}"
            )

        auc_values = [
            v["auc_point"]
            for v in per_protocol.values()
            if np.isfinite(v["auc_point"])
        ]
        out["targets"][target] = {
            "per_protocol": per_protocol,
            "summary": {
                "n_protocols": len(auc_values),
                "auc_mean": float(np.mean(auc_values)) if auc_values else None,
                "auc_std": float(np.std(auc_values, ddof=1))
                if len(auc_values) > 1
                else None,
                "auc_min": float(np.min(auc_values)) if auc_values else None,
                "auc_max": float(np.max(auc_values)) if auc_values else None,
            },
        }

        # Save per-target per-protocol JSONs
        for protocol, r in per_protocol.items():
            out_json = OUT_DIR / f"analysis_D_{target}_protocol_{protocol}.json"
            out_json.write_text(json.dumps(r, indent=2), encoding="utf-8")

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="Smoke test only (skip train)"
    )
    args = parser.parse_args()

    t0 = time.time()
    df = load_joined_frame()
    if args.dry_run:
        logger.info("Dry run: loaded data, exiting without training.")
        return

    d = run_analysis_d(df)
    result = {
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "n_bootstrap_ci": N_BOOTSTRAP_CI,
        "analysis_D_protocol_loco": d,
    }

    (OUT_DIR / "analysis_D_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    logger.info(f"Done. Total elapsed: {(time.time() - t0) / 60:.2f} min")
    logger.info(
        f"Wrote: {OUT_DIR / 'analysis_D_summary.json'} + per-protocol JSONs"
    )


if __name__ == "__main__":
    main()
