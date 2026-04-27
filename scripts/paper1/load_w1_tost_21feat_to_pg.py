"""Load Q_R2_W1 21-feat TOST results into features.paper1_r2_sensitivity.

Two run_ids inserted:
- q_r2_w1_sota_21feat: per-method per-target AUC summary (8 rows: TabPFN + AutoGluon × 4 targets;
  CatBoost-default + CatBoost-HPO + LightGBM-HPO already in q_r2_w3_ablation + q_r2_hpo_21feat).
- q_r2_w1_tost_21feat: per-target TOST verdict summary (4 rows, verdict field encodes counts).

The pairwise comparison details remain in the JSON at
outputs/paper1_r2_responses/q_r2_w1_paired_bootstrap_tost_21feat.json — too granular for a
flat table. The verdict field gives the headline numbers a reviewer needs.
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import text

from giman_pipeline.data.db import get_engine

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
JSON_PATH = ROOT / "outputs" / "paper1_r2_responses" / "q_r2_w1_paired_bootstrap_tost_21feat.json"

engine = get_engine()
data = json.loads(JSON_PATH.read_text())

# Wipe prior rows
with engine.begin() as conn:
    for run_id in ("q_r2_w1_sota_21feat", "q_r2_w1_tost_21feat"):
        result = conn.execute(text("DELETE FROM features.paper1_r2_sensitivity WHERE run_id = :r"),
                              {"r": run_id})
        print(f"[delete] {run_id}: {result.rowcount} prior rows cleared")

# Per-method TabPFN + AG rows (the other 3 methods are already in SQL)
sota_rows = []
tost_rows = []

for target, td in data["by_target"].items():
    n_classes = td["n_classes"]
    summaries = td["per_method_summary"]

    # 1) Per-method rows for the 2 SOTA methods not already covered
    for method in ("tabpfn_21", "autogluon_21"):
        if method not in summaries:
            continue
        s = summaries[method]
        sota_rows.append({
            "run_id": "q_r2_w1_sota_21feat",
            "target": target,
            "feature_set": "Path3_21feat_strict_circularity",
            "stratum": method,
            "n_patients": 2201,
            "n_features": 19,  # 21 minus 2 binary indicators stripped
            "n_folds_used": 5,
            "fold_mean_auc": s["fold_mean"],
            "fold_std_auc": s["fold_std"],
            "pooled_auc": s["fold_mean"],  # pooled not directly available for these methods; use fold_mean
            "auc_ci95_lo": None,
            "auc_ci95_hi": None,
            "delta_vs_ref": None,
            "ref_label": None,
            "verdict": "informative",
            "source_file": str(JSON_PATH.relative_to(ROOT)),
        })

    # 2) Per-target TOST summary
    eq01 = sum(1 for t in td["pairwise_tests"].values() if t["tost_equivalence_eps_0.01"]["equivalent_at_eps"])
    eq02 = sum(1 for t in td["pairwise_tests"].values() if t["tost_equivalence_eps_0.02"]["equivalent_at_eps"])
    npairs = len(td["pairwise_tests"])
    verdict = f"TOST_eq_eps0.01={eq01}/{npairs}; TOST_eq_eps0.02={eq02}/{npairs}"
    if eq02 == npairs:
        outcome = "convergence_at_eps_0.02"
    elif eq02 >= 0.7 * npairs:
        outcome = "majority_convergence_at_eps_0.02"
    else:
        outcome = "partial_convergence_at_eps_0.02"

    tost_rows.append({
        "run_id": "q_r2_w1_tost_21feat",
        "target": target,
        "feature_set": "Path3_21feat_strict_circularity",
        "stratum": "tost_summary",
        "n_patients": 2201,
        "n_features": 19,
        "n_folds_used": 5,
        "fold_mean_auc": None,
        "fold_std_auc": None,
        "pooled_auc": None,
        "auc_ci95_lo": None,
        "auc_ci95_hi": None,
        "delta_vs_ref": None,
        "ref_label": "5-way pairwise (CatBoost-default/HPO + LightGBM-HPO + TabPFN-v2 + AutoGluon)",
        "verdict": f"{outcome}: {verdict}",
        "source_file": str(JSON_PATH.relative_to(ROOT)),
    })

# Insert all rows
with engine.begin() as conn:
    for row in sota_rows + tost_rows:
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

print(f"[insert] q_r2_w1_sota_21feat: {len(sota_rows)} rows")
print(f"[insert] q_r2_w1_tost_21feat: {len(tost_rows)} rows")

# Verify
with engine.begin() as conn:
    result = conn.execute(text("""
        SELECT run_id, COUNT(*), STRING_AGG(DISTINCT target, ', ') AS targets
        FROM features.paper1_r2_sensitivity
        WHERE run_id LIKE 'q_r2_w1%'
        GROUP BY run_id
        ORDER BY run_id
    """)).fetchall()
    for row in result:
        print(f"[verify] {row[0]}: {row[1]} rows, targets={row[2]}")
