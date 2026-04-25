"""Load Q_R5-Q6 extended fairness results into features.paper1_r2_sensitivity.

run_id: q_r5_q6_fairness
~18 rows: one per (model × axis × subgroup). Skipped subgroups omitted.
feature_set ∈ {Path3_21feat, Common12feat_NSDpos}.
delta_vs_ref is AUC delta relative to the model-overall AUC.
verdict per row repeats the axis-level verdict ("PASS" or "FAIL").
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import text

from giman_pipeline.data.db import get_engine

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
JSON_PATH = ROOT / "outputs" / "paper1_r2_responses" / "q_r5_q6_fairness_extended.json"

RUN_ID = "q_r5_q6_fairness"

MODEL_FEATURE_SET = {
    "21feat_binary": "Path3_21feat",
    "12feat_nsd_positive": "Common12feat_NSDpos",
}
MODEL_TARGET = {
    "21feat_binary": "binary",
    "12feat_nsd_positive": "nsd_positive",
}
MODEL_NFEAT = {
    "21feat_binary": 19,  # CAUDATE_PUTAMEN_RATIO excluded → 19 SQL-source features
    "12feat_nsd_positive": 12,
}

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
for model_label, model_res in data["models"].items():
    feature_set = MODEL_FEATURE_SET[model_label]
    target = MODEL_TARGET[model_label]
    n_feat = MODEL_NFEAT[model_label]
    main_auc = float(model_res["main_auc"])
    for axis_name, axis_res in model_res["axes"].items():
        # Determine axis-level verdict (PASS/FAIL) once per axis
        inter = axis_res["interaction_test"]
        p_raw = inter.get("p_raw")
        max_delta = axis_res["max_abs_delta_vs_main"]
        verdict = (
            "PASS"
            if (max_delta is not None and max_delta < 0.03 and (p_raw is None or p_raw > 0.05))
            else "FAIL"
        )
        for sg_label, sg in axis_res["subgroups"].items():
            if sg.get("skipped") or sg.get("auc") is None:
                continue
            stratum_str = f"{axis_name}::{sg_label}"
            rows.append(
                {
                    "run_id": RUN_ID,
                    "target": target,
                    "feature_set": feature_set,
                    "stratum": stratum_str,
                    "n_patients": int(sg["n"]),
                    "n_features": int(n_feat),
                    "n_folds_used": 5,
                    "fold_mean_auc": None,  # subgroup-pooled, no per-fold breakdown
                    "fold_std_auc": None,
                    "pooled_auc": float(sg["auc"]),
                    "auc_ci95_lo": float(sg["ci95_lo"]) if sg.get("ci95_lo") is not None else None,
                    "auc_ci95_hi": float(sg["ci95_hi"]) if sg.get("ci95_hi") is not None else None,
                    "delta_vs_ref": float(sg["auc"] - main_auc),
                    "ref_label": f"{model_label}_overall (AUC={main_auc:.4f})",
                    "verdict": f"{verdict}|{axis_name}|ECE={sg.get('ece_10bin'):.4f}"
                    if sg.get("ece_10bin") is not None
                    else f"{verdict}|{axis_name}",
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
        SELECT feature_set, stratum, n_patients, pooled_auc, delta_vs_ref, verdict
        FROM features.paper1_r2_sensitivity
        WHERE run_id = :r
        ORDER BY feature_set, stratum
        """
        ),
        {"r": RUN_ID},
    ).fetchall()
    for r in result:
        delta = r[4] if r[4] is not None else float("nan")
        print(
            f"[verify] {r[0]:24s}  {r[1]:30s}  n={r[2]:4d}  "
            f"AUC={r[3]:.4f}  Δ={delta:+.4f}  {r[5]}"
        )
