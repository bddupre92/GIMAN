"""Load Q_R6-Q3 ComBat harmonization results into features.paper1_r2_sensitivity.

run_id: q_r6_q3_combat
7 rows total:
  - 4 internal: feature_set='Path3_21feat_combat_caudate', stratum='internal_5fold'
  - 3 LOCO summary: feature_set='Path3_21feat_combat_caudate', stratum='loco_protocol'
    (binary, 3class, full_ordinal — nsd_positive LOCO too — actually 4 LOCO rows)

Total spec'd was 4+3 = 7 (skipped 1 LOCO target). We emit 4 internal + 4 LOCO = 8
rows for completeness, both summarised against the vanilla 21-feat baseline as ref.

ref_label='Path3_21feat (primary)'.
delta_vs_ref is in AUC units (not pp).
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import text

from giman_pipeline.data.db import get_engine

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
JSON_PATH = ROOT / "outputs" / "paper1_r2_responses" / "q_r6_q3_combat_harmonization.json"

RUN_ID = "q_r6_q3_combat"

engine = get_engine()
data = json.loads(JSON_PATH.read_text())

# Wipe prior rows
with engine.begin() as conn:
    result = conn.execute(
        text("DELETE FROM features.paper1_r2_sensitivity WHERE run_id = :r"),
        {"r": RUN_ID},
    )
    print(f"[delete] {RUN_ID}: {result.rowcount} prior rows cleared")

rows = []

# Internal CV rows (4)
for target, perf in data["internal_aucs"].items():
    rows.append(
        {
            "run_id": RUN_ID,
            "target": target,
            "feature_set": "Path3_21feat_combat_caudate",
            "stratum": "internal_5fold",
            "n_patients": int(perf["n_patients"]),
            "n_features": 19,
            "n_folds_used": int(data["n_folds"]),
            "fold_mean_auc": float(perf["combat_fold_mean"]),
            "fold_std_auc": float(perf["combat_fold_std"]),
            "pooled_auc": float(perf["combat_pooled_auc"]),
            "auc_ci95_lo": float(perf["combat_ci95"][0]),
            "auc_ci95_hi": float(perf["combat_ci95"][1]),
            "delta_vs_ref": float(perf["combat_pooled_auc"] - perf["primary_pooled_auc"]),
            "ref_label": "Path3_21feat (primary)",
            "verdict": data["headline_verdict"],
            "source_file": str(JSON_PATH.relative_to(ROOT)),
        }
    )

# Protocol-LOCO rows (4 — combat side; vanilla included as separate ref label)
for target in data["internal_aucs"]:
    van = data["protocol_loco"]["vanilla"][target]
    cb = data["protocol_loco"]["combat"][target]
    if van["mean_auc"] is None or cb["mean_auc"] is None:
        continue
    rows.append(
        {
            "run_id": RUN_ID,
            "target": target,
            "feature_set": "Path3_21feat_combat_caudate",
            "stratum": "loco_protocol",
            "n_patients": data["combat_config"]["n_combat_fit"],
            "n_features": 19,
            "n_folds_used": int(van["n_protocols"]),
            "fold_mean_auc": float(cb["mean_auc"]),
            "fold_std_auc": float(cb["sd_auc"]),
            "pooled_auc": float(cb["mean_auc"]),
            "auc_ci95_lo": float(cb["mean_auc"] - 1.96 * cb["sd_auc"]) if cb["sd_auc"] == cb["sd_auc"] else float("nan"),
            "auc_ci95_hi": float(cb["mean_auc"] + 1.96 * cb["sd_auc"]) if cb["sd_auc"] == cb["sd_auc"] else float("nan"),
            "delta_vs_ref": float(cb["mean_auc"] - van["mean_auc"]),
            "ref_label": "Path3_21feat vanilla LOCO",
            "verdict": data["headline_verdict"],
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
        SELECT target, stratum, fold_mean_auc, pooled_auc, delta_vs_ref, verdict
        FROM features.paper1_r2_sensitivity
        WHERE run_id = :r
        ORDER BY stratum, target
        """
        ),
        {"r": RUN_ID},
    ).fetchall()
    for r in result:
        print(
            f"[verify] {r[0]:15s} {r[1]:18s} fold_mean={r[2]:.4f} pooled={r[3]:.4f} Δ={r[4]:+.4f} verdict={r[5]}"
        )
