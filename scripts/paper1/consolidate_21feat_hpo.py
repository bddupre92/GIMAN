"""Consolidate Paper 1 R2 21-feat nested-CV HPO results + insert into SQL.

Reads outputs/paper1_hpo_21feat/results/nested_{model}_{target}.json for the 8
combinations and writes:
  - outputs/paper1_hpo_21feat/summary.json    (consolidated top-line numbers)
  - Inserts 8 rows into features.paper1_r2_sensitivity with run_id='q_r2_hpo_21feat'

Usage:
    .venv/bin/python scripts/paper1/consolidate_21feat_hpo.py
    .venv/bin/python scripts/paper1/consolidate_21feat_hpo.py --dry-run
    .venv/bin/python scripts/paper1/consolidate_21feat_hpo.py --skip-sql
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import text  # noqa: E402

from giman_pipeline.data.db import get_engine  # noqa: E402

RESULT_DIR = ROOT / "outputs" / "paper1_hpo_21feat" / "results"
SUMMARY_PATH = ROOT / "outputs" / "paper1_hpo_21feat" / "summary.json"

MODELS = ["catboost", "lightgbm"]
TARGETS = ["binary", "3class", "full_ordinal", "nsd_positive"]
RUN_ID = "q_r2_hpo_21feat"


def consolidate() -> dict:
    out = {"run_id": RUN_ID, "combinations": {}}
    missing: list[str] = []
    for m in MODELS:
        for t in TARGETS:
            p = RESULT_DIR / f"nested_{m}_{t}.json"
            if not p.exists():
                missing.append(p.name)
                continue
            d = json.loads(p.read_text())
            key = f"{m}__{t}"
            out["combinations"][key] = {
                "model": m,
                "target": t,
                "n_features": d.get("n_features"),
                "feature_cols": d.get("feature_cols"),
                "fold_mean_auc": d.get("fold_mean_auc"),
                "fold_std_auc": d.get("fold_std_auc"),
                "fold_ci95_bootstrap": d.get("fold_ci95_bootstrap"),
                "pooled_oof_auc": d.get("pooled_oof_auc"),
                "pooled_oof_ci95": d.get("pooled_oof_ci95"),
                "n_oof_patients": d.get("n_oof_patients"),
                "modal_hp": d.get("modal_hp"),
                "per_fold_best_hp": d.get("per_fold_best_hp"),
                "per_fold_test_auc": d.get("per_fold_test_auc"),
                "total_trials": d.get("total_trials"),
                "total_duration_sec": d.get("total_duration_sec"),
            }
    out["n_complete"] = len(out["combinations"])
    out["n_missing"] = len(missing)
    out["missing"] = missing
    out["protocol_ref"] = "scripts/paper1/run_nested_cv_hpo_21feat.py"
    return out


def build_sql_rows(summary: dict) -> list[dict]:
    """Map each combination to one features.paper1_r2_sensitivity row."""
    rows = []
    for key, c in summary["combinations"].items():
        ci = c.get("pooled_oof_ci95") or [None, None]
        rows.append({
            "run_id": RUN_ID,
            "target": c["target"],
            "feature_set": f"Path3_21feat_{c['model']}",
            "stratum": "",
            "n_patients": c.get("n_oof_patients"),
            "n_features": c.get("n_features"),
            "n_folds_used": 5,
            "fold_mean_auc": c.get("fold_mean_auc"),
            "fold_std_auc": c.get("fold_std_auc"),
            "pooled_auc": c.get("pooled_oof_auc"),
            "auc_ci95_lo": ci[0] if ci else None,
            "auc_ci95_hi": ci[1] if ci else None,
            "delta_vs_ref": None,
            "ref_label": "Path0_22feat_R1_22feat_HPO",
            "verdict": "HPO_COMPLETE",
            "source_file": f"outputs/paper1_hpo_21feat/results/nested_{c['model']}_{c['target']}.json",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-sql", action="store_true")
    args = ap.parse_args()

    summary = consolidate()
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[consolidate] {summary['n_complete']}/8 combinations complete")
    if summary["missing"]:
        print(f"[consolidate] MISSING: {summary['missing']}")
    print(f"[consolidate] wrote {SUMMARY_PATH}")

    # Pretty headline table
    print()
    print(f"{'model':<10} {'target':<14} {'n_feat':>6} {'fold_mean':>10} {'pooled_OOF':>12} "
          f"{'CI95_lo':>8} {'CI95_hi':>8}")
    print("-" * 78)
    for key, c in summary["combinations"].items():
        ci = c.get("pooled_oof_ci95") or [None, None]
        print(f"{c['model']:<10} {c['target']:<14} {c.get('n_features') or 0:>6} "
              f"{c.get('fold_mean_auc') or float('nan'):>10.4f} "
              f"{c.get('pooled_oof_auc') or float('nan'):>12.4f} "
              f"{(ci[0] or float('nan')):>8.4f} {(ci[1] or float('nan')):>8.4f}")

    if args.skip_sql:
        print("[consolidate] --skip-sql set; not writing to Postgres")
        return
    if args.dry_run:
        print("[consolidate] --dry-run set; printing SQL rows without writing")
        rows = build_sql_rows(summary)
        for r in rows[:2]:
            print(" ", r)
        print(f"  ... + {len(rows)-2} more rows")
        return
    if summary["n_complete"] == 0:
        print("[consolidate] nothing complete yet — skipping SQL")
        return

    rows = build_sql_rows(summary)
    engine = get_engine()
    with engine.begin() as conn:
        n = conn.execute(
            text("DELETE FROM features.paper1_r2_sensitivity WHERE run_id=:r"),
            {"r": RUN_ID},
        ).rowcount
        print(f"[consolidate] cleared {n} prior rows for run_id={RUN_ID}")
        pd.DataFrame(rows).to_sql(
            "paper1_r2_sensitivity",
            conn,
            schema="features",
            if_exists="append",
            index=False,
        )
        print(f"[consolidate] inserted {len(rows)} rows into features.paper1_r2_sensitivity")


if __name__ == "__main__":
    main()
