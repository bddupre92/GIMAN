"""Load Q_R6-Q8 age-decile + BioFIND external results into features.paper1_r2_sensitivity.

run_id: q_r6_q8_subgroup
~25 rows: 10 internal age deciles × 2 models (20) + external sex (1 evaluable, 1 skipped)
+ external age tertile (3) ≈ 24-25 rows. Skipped subgroups are recorded with NaN AUC.

feature_set ∈ {Path3_21feat_decile, Common12feat_NSDpos_decile, Common12feat_BIN_BioFIND}.
delta_vs_ref is AUC delta vs the per-cohort-model overall AUC.
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import text

from giman_pipeline.data.db import get_engine

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
JSON_PATH = ROOT / "outputs" / "paper1_r2_responses" / "q_r6_q8_subgroup_extended.json"

RUN_ID = "q_r6_q8_subgroup"

engine = get_engine()
data = json.loads(JSON_PATH.read_text())

with engine.begin() as conn:
    result = conn.execute(
        text("DELETE FROM features.paper1_r2_sensitivity WHERE run_id = :r"),
        {"r": RUN_ID},
    )
    print(f"[delete] {RUN_ID}: {result.rowcount} prior rows cleared")

rows = []

# --- Internal age-decile rows ---
INTERNAL_FEATURE_SETS = {
    "21feat_binary": ("Path3_21feat_decile", "binary", 19),
    "12feat_nsd_positive": ("Common12feat_NSDpos_decile", "nsd_positive", 12),
}
for model_label, dr in data["internal_age_deciles"].items():
    feat_set, target, n_feat = INTERNAL_FEATURE_SETS[model_label]
    overall = data["internal_overall"][model_label]
    main_auc = float(overall["auc"])
    for sg_label, sg in dr["deciles"].items():
        stratum_str = f"age_decile::{sg_label}"
        if sg.get("skipped") or sg.get("auc") is None:
            rows.append(
                {
                    "run_id": RUN_ID,
                    "target": target,
                    "feature_set": feat_set,
                    "stratum": stratum_str,
                    "n_patients": int(sg["n"]),
                    "n_features": int(n_feat),
                    "n_folds_used": 5,
                    "fold_mean_auc": None,
                    "fold_std_auc": None,
                    "pooled_auc": None,
                    "auc_ci95_lo": None,
                    "auc_ci95_hi": None,
                    "delta_vs_ref": None,
                    "ref_label": f"{model_label}_overall (AUC={main_auc:.4f})",
                    "verdict": f"SKIPPED|age_decile|{sg.get('reason', '?')[:30]}",
                    "source_file": str(JSON_PATH.relative_to(ROOT)),
                }
            )
        else:
            verdict = sg["verdict"]
            rows.append(
                {
                    "run_id": RUN_ID,
                    "target": target,
                    "feature_set": feat_set,
                    "stratum": stratum_str,
                    "n_patients": int(sg["n"]),
                    "n_features": int(n_feat),
                    "n_folds_used": 5,
                    "fold_mean_auc": None,
                    "fold_std_auc": None,
                    "pooled_auc": float(sg["auc"]),
                    "auc_ci95_lo": (
                        float(sg["ci95_lo"]) if sg.get("ci95_lo") is not None else None
                    ),
                    "auc_ci95_hi": (
                        float(sg["ci95_hi"]) if sg.get("ci95_hi") is not None else None
                    ),
                    "delta_vs_ref": float(sg["delta_vs_main"]),
                    "ref_label": f"{model_label}_overall (AUC={main_auc:.4f})",
                    "verdict": (
                        f"{verdict}|age_decile|ECE={sg.get('ece_10bin'):.4f}"
                    ),
                    "source_file": str(JSON_PATH.relative_to(ROOT)),
                }
            )

# --- External BioFIND rows ---
ext = data["external_biofind"]
ext_main_auc = float(ext["main_auc"]) if ext["main_auc"] is not None else None
for axis_name, axis_res in ext["axes"].items():
    for sg_label, sg in axis_res["subgroups"].items():
        stratum_str = f"BioFIND_{axis_name}::{sg_label}"
        if sg.get("skipped") or sg.get("auc") is None:
            rows.append(
                {
                    "run_id": RUN_ID,
                    "target": "binary",
                    "feature_set": "Common12feat_BIN_BioFIND",
                    "stratum": stratum_str,
                    "n_patients": int(sg["n"]),
                    "n_features": 12,
                    "n_folds_used": 1,  # external = single train-on-all-PPMI evaluation
                    "fold_mean_auc": None,
                    "fold_std_auc": None,
                    "pooled_auc": None,
                    "auc_ci95_lo": None,
                    "auc_ci95_hi": None,
                    "delta_vs_ref": None,
                    "ref_label": (
                        f"BioFIND_overall (AUC={ext_main_auc:.4f})"
                        if ext_main_auc is not None
                        else "BioFIND_overall"
                    ),
                    "verdict": f"SKIPPED|BioFIND_{axis_name}|{sg.get('reason', '?')[:30]}",
                    "source_file": str(JSON_PATH.relative_to(ROOT)),
                }
            )
        else:
            verdict = sg["verdict"]
            rows.append(
                {
                    "run_id": RUN_ID,
                    "target": "binary",
                    "feature_set": "Common12feat_BIN_BioFIND",
                    "stratum": stratum_str,
                    "n_patients": int(sg["n"]),
                    "n_features": 12,
                    "n_folds_used": 1,
                    "fold_mean_auc": None,
                    "fold_std_auc": None,
                    "pooled_auc": float(sg["auc"]),
                    "auc_ci95_lo": (
                        float(sg["ci95_lo"]) if sg.get("ci95_lo") is not None else None
                    ),
                    "auc_ci95_hi": (
                        float(sg["ci95_hi"]) if sg.get("ci95_hi") is not None else None
                    ),
                    "delta_vs_ref": float(sg["delta_vs_main"]),
                    "ref_label": (
                        f"BioFIND_overall (AUC={ext_main_auc:.4f})"
                        if ext_main_auc is not None
                        else "BioFIND_overall"
                    ),
                    "verdict": (
                        f"{verdict}|BioFIND_{axis_name}|ECE={sg.get('ece_10bin'):.4f}"
                    ),
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
        auc_str = f"{r[3]:.4f}" if r[3] is not None else "—"
        print(
            f"[verify] {r[0]:30s}  {r[1]:36s}  n={r[2]:4d}  "
            f"AUC={auc_str:>6s}  Δ={delta:+.4f}  {r[5]}"
        )
