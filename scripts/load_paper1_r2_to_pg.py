#!/usr/bin/env python3
"""Load Paper 1 Round-2 reviewer-response artifacts into Postgres ``features`` schema.

Consumes JSON outputs from the 2026-04-23 R2 work:
  - outputs/paper1_circularity_audit/sensitivity_putamen_ratio.json   (Q2)
  - outputs/paper1_r2_responses/q1_label_var_ablation.json             (Q1)
  - outputs/paper1_r2_responses/q5_saa_stratified.json                 (Q5)
  - outputs/paper1_r2_responses/q7_abstention_rates.json               (Q7)

Creates two tables:
  - features.paper1_r2_sensitivity   one row per (run_id, target, feature_set, stratum)
  - features.paper1_r2_abstention    one row per (cohort, target, model, method, alpha)

Idempotent: on re-run, DELETEs existing rows matching the loaded run_ids/paths
then inserts fresh ones. `--force-reload` TRUNCATEs the two new tables fully.
`--dry-run` prints the plan without hitting the database.

Also optionally updates the CLAUDE.md Schemas registry-freshness table
(features schema row: 8 -> 10) and the audit.claim verdict for the
0.979 -> 0.901 binary headline (manual review; loader prints the SQL).

Usage:
    .venv/bin/python scripts/load_paper1_r2_to_pg.py
    .venv/bin/python scripts/load_paper1_r2_to_pg.py --force-reload
    .venv/bin/python scripts/load_paper1_r2_to_pg.py --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from giman_pipeline.data.db import get_engine

PROJECT_ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")

Q1_JSON = PROJECT_ROOT / "outputs/paper1_r2_responses/q1_label_var_ablation.json"
Q2_JSON = PROJECT_ROOT / "outputs/paper1_circularity_audit/sensitivity_putamen_ratio.json"
Q5_JSON = PROJECT_ROOT / "outputs/paper1_r2_responses/q5_saa_stratified.json"
Q7_JSON = PROJECT_ROOT / "outputs/paper1_r2_responses/q7_abstention_rates.json"

DDL_SENSITIVITY = """
CREATE TABLE IF NOT EXISTS features.paper1_r2_sensitivity (
    run_id          TEXT              NOT NULL,
    target          TEXT              NOT NULL,
    feature_set     TEXT              NOT NULL,
    stratum         TEXT              NOT NULL DEFAULT '',
    n_patients      INTEGER,
    n_features      INTEGER,
    n_folds_used    INTEGER,
    fold_mean_auc   DOUBLE PRECISION,
    fold_std_auc    DOUBLE PRECISION,
    pooled_auc      DOUBLE PRECISION,
    auc_ci95_lo     DOUBLE PRECISION,
    auc_ci95_hi     DOUBLE PRECISION,
    delta_vs_ref    DOUBLE PRECISION,
    ref_label       TEXT,
    verdict         TEXT,
    source_file     TEXT,
    run_timestamp   TIMESTAMPTZ       NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, target, feature_set, stratum)
);
"""

DDL_ABSTENTION = """
CREATE TABLE IF NOT EXISTS features.paper1_r2_abstention (
    cohort                  TEXT               NOT NULL,
    target                  TEXT               NOT NULL,
    model                   TEXT               NOT NULL,
    method                  TEXT               NOT NULL,
    alpha                   DOUBLE PRECISION   NOT NULL,
    confidence_level        DOUBLE PRECISION   NOT NULL,
    confidence_level_pct    INTEGER,
    n                       INTEGER,
    empty_fraction          DOUBLE PRECISION,
    singleton_fraction      DOUBLE PRECISION,
    multi_label_fraction    DOUBLE PRECISION,
    mean_set_size           DOUBLE PRECISION,
    marginal_coverage       DOUBLE PRECISION,
    n_classes               INTEGER,
    n_folds_pooled          INTEGER,
    source_file             TEXT,
    run_timestamp           TIMESTAMPTZ        NOT NULL DEFAULT NOW(),
    PRIMARY KEY (cohort, target, model, method, alpha)
);
"""


def load_q1_rows() -> list[dict]:
    """Q1 label-variable ablation: 4 targets x 2 feature sets + deltas."""
    if not Q1_JSON.exists():
        return []
    d = json.loads(Q1_JSON.read_text())
    verdict = d.get("verdict", {}).get("verdict", "UNKNOWN")
    rows: list[dict] = []
    for fs_label, target_map in [("Path3_21feat", d.get("Path3_21feat", {})),
                                  ("Strict_18feat", d.get("Strict_18feat", {}))]:
        for target, r in target_map.items():
            delta = d["delta"][target]["delta_pooled_auc"] if fs_label == "Path3_21feat" else None
            ci = r.get("pooled_auc_ci95", [None, None])
            rows.append({
                "run_id": "q1_label_var",
                "target": target,
                "feature_set": fs_label,
                "stratum": "",
                "n_patients": r.get("n_samples"),
                "n_features": r.get("n_features"),
                "n_folds_used": None,
                "fold_mean_auc": r.get("fold_mean_auc"),
                "fold_std_auc": r.get("fold_std_auc"),
                "pooled_auc": r.get("pooled_auc"),
                "auc_ci95_lo": ci[0],
                "auc_ci95_hi": ci[1],
                "delta_vs_ref": delta if fs_label == "Path3_21feat" else None,
                "ref_label": "Strict_18feat" if fs_label == "Path3_21feat" else None,
                "verdict": verdict,
                "source_file": str(Q1_JSON.relative_to(PROJECT_ROOT)),
            })
    return rows


def load_q2_rows() -> list[dict]:
    """Q2 putamen-ratio sensitivity: 4 targets x 2 feature sets + MATERIAL verdict.

    JSON keys are SetA_full22 (pre-exclusion) / SetB_no_ratio21 (post-exclusion).
    """
    if not Q2_JSON.exists():
        return []
    d = json.loads(Q2_JSON.read_text())
    verdict_obj = d.get("verdict", {})
    verdict = verdict_obj.get("verdict", "UNKNOWN") if isinstance(verdict_obj, dict) else verdict_obj
    rows: list[dict] = []
    for variant, label in (("SetA_full22", "Path0_22feat"),
                            ("SetB_no_ratio21", "Path3_21feat")):
        v = d.get(variant, {})
        for target, r in v.items():
            ci = r.get("pooled_auc_ci95", [None, None])
            delta_vs_22 = None
            if label == "Path3_21feat" and "delta" in d and target in d["delta"]:
                delta_vs_22 = d["delta"][target].get("delta_pooled_auc")
            rows.append({
                "run_id": "q2_putamen_ratio",
                "target": target,
                "feature_set": label,
                "stratum": "",
                "n_patients": r.get("n_samples"),
                "n_features": r.get("n_features"),
                "n_folds_used": None,
                "fold_mean_auc": r.get("fold_mean_auc"),
                "fold_std_auc": r.get("fold_std_auc"),
                "pooled_auc": r.get("pooled_auc"),
                "auc_ci95_lo": ci[0],
                "auc_ci95_hi": ci[1],
                "delta_vs_ref": delta_vs_22,
                "ref_label": "Path0_22feat" if label == "Path3_21feat" else None,
                "verdict": verdict,
                "source_file": str(Q2_JSON.relative_to(PROJECT_ROOT)),
            })
    return rows


def load_q5_rows() -> list[dict]:
    """Q5 SAA-stratified: 3 targets x {full + 3 strata} + per-target verdicts."""
    if not Q5_JSON.exists():
        return []
    d = json.loads(Q5_JSON.read_text())
    verdict_per_t = d.get("verdict", {}).get("per_target", {})
    rows: list[dict] = []
    # Full-cohort reference rows
    for target, r in d.get("full", {}).items():
        if not r:
            continue
        ci = r.get("pooled_auc_ci95", [None, None])
        rows.append({
            "run_id": "q5_saa_stratified",
            "target": target,
            "feature_set": "Path3_21feat",
            "stratum": "full",
            "n_patients": r.get("n_patients"),
            "n_features": None,
            "n_folds_used": r.get("n_folds_used"),
            "fold_mean_auc": r.get("fold_mean_auc"),
            "fold_std_auc": r.get("fold_std_auc"),
            "pooled_auc": r.get("pooled_auc"),
            "auc_ci95_lo": ci[0],
            "auc_ci95_hi": ci[1],
            "delta_vs_ref": 0.0,
            "ref_label": "full",
            "verdict": str(verdict_per_t.get(target, {})),
            "source_file": str(Q5_JSON.relative_to(PROJECT_ROOT)),
        })
    # Stratum rows
    for target, strata in d.get("strata", {}).items():
        for stratum, r in strata.items():
            if r is None:
                continue
            ci = r.get("pooled_auc_ci95", [None, None])
            delta = (d.get("delta", {}).get(target, {}).get(stratum, {}).get("auc_delta_vs_full"))
            rows.append({
                "run_id": "q5_saa_stratified",
                "target": target,
                "feature_set": "Path3_21feat",
                "stratum": stratum,
                "n_patients": r.get("n_patients"),
                "n_features": None,
                "n_folds_used": r.get("n_folds_used"),
                "fold_mean_auc": r.get("fold_mean_auc"),
                "fold_std_auc": r.get("fold_std_auc"),
                "pooled_auc": r.get("pooled_auc"),
                "auc_ci95_lo": ci[0],
                "auc_ci95_hi": ci[1],
                "delta_vs_ref": delta,
                "ref_label": "full",
                "verdict": str(verdict_per_t.get(target, {})),
                "source_file": str(Q5_JSON.relative_to(PROJECT_ROOT)),
            })
    return rows


def load_q7_rows() -> list[dict]:
    """Q7 abstention rates: ~96 rows spanning internal + external."""
    if not Q7_JSON.exists():
        return []
    d = json.loads(Q7_JSON.read_text())
    rows: list[dict] = []
    for r in d.get("by_cohort_target_model_method_alpha", []):
        rows.append({
            "cohort": r["cohort"],
            "target": r["target"],
            "model": r["model"],
            "method": r["method"],
            "alpha": r["alpha"],
            "confidence_level": r["confidence_level"],
            "confidence_level_pct": r.get("confidence_level_pct"),
            "n": r.get("n"),
            "empty_fraction": r.get("empty_fraction"),
            "singleton_fraction": r.get("singleton_fraction"),
            "multi_label_fraction": r.get("multi_label_fraction"),
            "mean_set_size": r.get("mean_set_size"),
            "marginal_coverage": r.get("marginal_coverage"),
            "n_classes": r.get("n_classes"),
            "n_folds_pooled": r.get("n_folds_pooled"),
            "source_file": r.get("source"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-reload", action="store_true",
                        help="TRUNCATE features.paper1_r2_sensitivity + paper1_r2_abstention first")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan only, no DB writes")
    args = parser.parse_args()

    sens_rows = load_q1_rows() + load_q2_rows() + load_q5_rows()
    abst_rows = load_q7_rows()

    print(f"Loaded from disk:")
    print(f"  sensitivity rows (Q1+Q2+Q5): {len(sens_rows)}")
    print(f"  abstention rows  (Q7):       {len(abst_rows)}")
    if args.dry_run:
        print("\n[DRY-RUN] First 3 sensitivity rows:")
        for r in sens_rows[:3]:
            print(" ", r)
        print("\n[DRY-RUN] First 3 abstention rows:")
        for r in abst_rows[:3]:
            print(" ", r)
        return

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(DDL_SENSITIVITY))
        conn.execute(text(DDL_ABSTENTION))
        if args.force_reload:
            conn.execute(text("TRUNCATE features.paper1_r2_sensitivity"))
            conn.execute(text("TRUNCATE features.paper1_r2_abstention"))
            print("  [FORCE-RELOAD] truncated both tables")
        else:
            run_ids = {r["run_id"] for r in sens_rows}
            for rid in run_ids:
                n = conn.execute(text("DELETE FROM features.paper1_r2_sensitivity WHERE run_id=:r"),
                                 {"r": rid}).rowcount
                print(f"  cleared {n} prior rows for run_id={rid}")
            n = conn.execute(text("DELETE FROM features.paper1_r2_abstention")).rowcount
            print(f"  cleared {n} prior rows from paper1_r2_abstention")

        if sens_rows:
            pd.DataFrame(sens_rows).to_sql("paper1_r2_sensitivity", conn, schema="features",
                                            if_exists="append", index=False)
            print(f"  inserted {len(sens_rows)} rows into features.paper1_r2_sensitivity")
        if abst_rows:
            pd.DataFrame(abst_rows).to_sql("paper1_r2_abstention", conn, schema="features",
                                            if_exists="append", index=False)
            print(f"  inserted {len(abst_rows)} rows into features.paper1_r2_abstention")

    print()
    print("Verify with:")
    print("  psql giman_research -c \"SELECT run_id, target, feature_set, stratum, ROUND(pooled_auc::numeric,4) AS auc, verdict FROM features.paper1_r2_sensitivity ORDER BY run_id, target, feature_set, stratum;\"")
    print("  psql giman_research -c \"SELECT cohort, target, model, method, alpha, ROUND((100*empty_fraction)::numeric,1) AS empty_pct, ROUND((100*multi_label_fraction)::numeric,1) AS multi_pct FROM features.paper1_r2_abstention WHERE model='catboost' AND method IN ('cross','split_cv_aware') ORDER BY cohort, target, alpha;\"")
    print()
    print("Manual audit.claim UPDATE (Path 3 headline change 0.979 -> 0.901):")
    print("  UPDATE audit.claim")
    print("     SET verdict='modified',")
    print("         verdict_notes='commit c484a69: R2 Path 3 commitment — CAUDATE_PUTAMEN_RATIO")
    print("         removed as D-anchor leakage vector; binary NSD+ AUC rerun at 21 features'")
    print("   WHERE paper='P1' AND claim_text LIKE '%0.979%binary%';")


if __name__ == "__main__":
    main()
