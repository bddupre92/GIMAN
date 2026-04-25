"""Load R6-Q1 multiclass calibration results into features.paper1_r2_sensitivity.

Inserts 12 rows: 3 multiclass targets x 4 calibration methods (raw, temperature,
dirichlet, isotonic). The metric reported in pooled_auc is **macro-ECE** (lower
is better, not AUC) so that all rows live alongside the other R-* sensitivity
runs in the same table; the verdict text explains the metric.
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import text

from giman_pipeline.data.db import get_engine

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
JSON_PATH = ROOT / "outputs" / "paper1_r2_responses" / "q_r6_q1_multiclass_calibration.json"
RUN_ID = "q_r6_q1_calibration"

engine = get_engine()
data = json.loads(JSON_PATH.read_text())

with engine.begin() as conn:
    deleted = conn.execute(
        text("DELETE FROM features.paper1_r2_sensitivity WHERE run_id = :r"),
        {"r": RUN_ID},
    ).rowcount
    print(f"[delete] {RUN_ID}: {deleted} prior rows cleared")

rows = []
for target, td in data["by_target"].items():
    n_ext = int(td["n_external"])
    n_classes = int(td["n_classes"])
    raw_ece = float(td["methods"]["raw"]["macro_ece"])
    best_method = td["best_by_macro_ece"]
    for method in ("raw", "temperature", "dirichlet", "isotonic"):
        m = td["methods"][method]
        macro_ece = float(m["macro_ece"])
        macro_brier = float(m["macro_brier"])
        nll = float(m["nll"])
        delta = macro_ece - raw_ece
        is_best = method == best_method
        verdict = (
            f"macro-ECE={macro_ece:.3f} (raw={raw_ece:.3f}, dECE={delta:+.3f}); "
            f"macro-Brier={macro_brier:.3f}; NLL={nll:.3f}"
        )
        if is_best:
            verdict = "BEST | " + verdict
        rows.append({
            "run_id": RUN_ID,
            "target": target,
            "feature_set": "12-feat common (external)",
            "stratum": method,
            "n_patients": n_ext,
            "n_features": 12,
            "n_folds_used": 1,
            "fold_mean_auc": None,
            "fold_std_auc": None,
            # pooled_auc carries macro-ECE for this run (column reused across run_ids)
            "pooled_auc": macro_ece,
            "auc_ci95_lo": None,
            "auc_ci95_hi": None,
            "delta_vs_ref": delta,
            "ref_label": (
                f"raw macro-ECE={raw_ece:.3f}; classes={n_classes}; "
                f"metric in pooled_auc=macro-ECE (lower=better)"
            ),
            "verdict": verdict,
            "source_file": str(JSON_PATH.relative_to(ROOT)),
        })

with engine.begin() as conn:
    for row in rows:
        conn.execute(text("""
            INSERT INTO features.paper1_r2_sensitivity (
                run_id, target, feature_set, stratum,
                n_patients, n_features, n_folds_used,
                fold_mean_auc, fold_std_auc, pooled_auc,
                auc_ci95_lo, auc_ci95_hi, delta_vs_ref, ref_label,
                verdict, source_file
            ) VALUES (
                :run_id, :target, :feature_set, :stratum,
                :n_patients, :n_features, :n_folds_used,
                :fold_mean_auc, :fold_std_auc, :pooled_auc,
                :auc_ci95_lo, :auc_ci95_hi, :delta_vs_ref, :ref_label,
                :verdict, :source_file
            )
        """), row)

print(f"[insert] {RUN_ID}: {len(rows)} rows")

with engine.begin() as conn:
    result = conn.execute(text("""
        SELECT target, stratum, pooled_auc, delta_vs_ref, verdict
        FROM features.paper1_r2_sensitivity
        WHERE run_id = :r
        ORDER BY target, stratum
    """), {"r": RUN_ID}).fetchall()
    for row in result:
        print(f"[verify] {row[0]:<14} {row[1]:<12} ECE={row[2]:.3f} dECE={row[3]:+.3f} | {row[4]}")
