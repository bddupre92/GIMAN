"""Load Q_R5-Q3 striatal-free sensitivity results into features.paper1_r2_sensitivity.

run_id: q_r5_q3_striatal_free
4 rows (one per target). feature_set='Path3_15feat_striatal_free'.
ref_label='Path3_21feat (primary)'. delta_vs_ref is in AUC units (not pp).
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import text

from giman_pipeline.data.db import get_engine

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
JSON_PATH = ROOT / "outputs" / "paper1_r2_responses" / "q_r5_q3_striatal_free.json"

RUN_ID = "q_r5_q3_striatal_free"

engine = get_engine()
data = json.loads(JSON_PATH.read_text())

# Wipe prior rows
with engine.begin() as conn:
    result = conn.execute(
        text("DELETE FROM features.paper1_r2_sensitivity WHERE run_id = :r"),
        {"r": RUN_ID},
    )
    print(f"[delete] {RUN_ID}: {result.rowcount} prior rows cleared")

# Build per-target rows
rows = []
for target, perf in data["performance_comparison"].items():
    rows.append(
        {
            "run_id": RUN_ID,
            "target": target,
            "feature_set": "Path3_15feat_striatal_free",
            "stratum": "",
            "n_patients": int(data["n_patients"]),
            "n_features": int(data["n_features_used"]),
            "n_folds_used": int(data["n_folds"]),
            "fold_mean_auc": float(perf["striatal_free_fold_mean"]),
            "fold_std_auc": float(perf["striatal_free_fold_std"]),
            "pooled_auc": float(perf["striatal_free_pooled_auc"]),
            "auc_ci95_lo": float(perf["striatal_free_ci95"][0]),
            "auc_ci95_hi": float(perf["striatal_free_ci95"][1]),
            "delta_vs_ref": float(
                perf["striatal_free_pooled_auc"] - perf["primary_21feat_pooled_auc"]
            ),
            "ref_label": "Path3_21feat (primary)",
            "verdict": str(perf["verdict"]),
            "source_file": str(JSON_PATH.relative_to(ROOT)),
        }
    )

with engine.begin() as conn:
    for row in rows:
        conn.execute(
            text(
                """
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
            """
            ),
            row,
        )

print(f"[insert] {RUN_ID}: {len(rows)} rows")

# Verify
with engine.begin() as conn:
    result = conn.execute(
        text(
            """
        SELECT target, fold_mean_auc, pooled_auc, delta_vs_ref, verdict
        FROM features.paper1_r2_sensitivity
        WHERE run_id = :r
        ORDER BY target
        """
        ),
        {"r": RUN_ID},
    ).fetchall()
    for r in result:
        print(
            f"[verify] {r[0]}: fold_mean={r[1]:.4f} pooled={r[2]:.4f} Δ={r[3]:+.4f} verdict={r[4]}"
        )
