"""Load R4-Q3 21-feat graph results into features.paper1_r2_sensitivity.

8 rows (2 architectures × 4 targets) under run_id `q_r4_q3_graph_21feat`.

Companion artifacts:
  outputs/paper1_gat_21feat/{target}/gat_results.json
  outputs/paper1_enhanced_gat_21feat/{target}/enhanced_gat_results.json
  outputs/paper1_r2_responses/q_r4_q3_graph_21feat_aggregate.json (verdicts)

Pattern follows scripts/paper1/load_w1_tost_21feat_to_pg.py.
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import text

from giman_pipeline.data.db import get_engine

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
AGG_PATH = ROOT / "outputs" / "paper1_r2_responses" / "q_r4_q3_graph_21feat_aggregate.json"

RUN_ID = "q_r4_q3_graph_21feat"
TARGETS = ["binary", "three_class", "full_ordinal", "nsd_positive"]

engine = get_engine()
agg = json.loads(AGG_PATH.read_text())

# Wipe prior rows for idempotent reload
with engine.begin() as conn:
    res = conn.execute(
        text("DELETE FROM features.paper1_r2_sensitivity WHERE run_id = :r"),
        {"r": RUN_ID},
    )
    print(f"[delete] {RUN_ID}: {res.rowcount} prior rows cleared")

rows = []
for target in TARGETS:
    n_pat = int(agg["n_patients_by_target"][target])
    for arch_label, src_dir, src_name in [
        ("simple_gat", "paper1_gat_21feat", "gat_results.json"),
        ("mm_gat", "paper1_enhanced_gat_21feat", "enhanced_gat_results.json"),
    ]:
        d = agg["primary_21feat"][arch_label][target]
        baseline = agg["baseline_22feat"][arch_label][target]
        delta = agg["delta_21_minus_22"][arch_label][target]
        # Verdict cell-level: invariant if |delta|<=0.01, sensitive if delta<=-0.02, otherwise mixed
        if abs(delta) <= 0.01:
            verdict = f"invariant_to_circularity_exclusion (delta={delta:+.4f})"
        elif delta <= -0.02:
            verdict = f"sensitive_to_circularity_exclusion (delta={delta:+.4f})"
        elif delta >= 0.02:
            verdict = f"improved_under_strict_circularity (delta={delta:+.4f})"
        else:
            verdict = f"small_shift (delta={delta:+.4f})"

        rows.append({
            "run_id": RUN_ID,
            "target": target,
            "feature_set": "Path3_21feat_strict_circularity",
            "stratum": arch_label,
            "n_patients": n_pat,
            "n_features": 19,  # 21 nominal, 2 binary indicators stripped
            "n_folds_used": 5,
            "fold_mean_auc": float(d["auc"]),
            "fold_std_auc": float(d["auc_std"]),
            "pooled_auc": float(d["auc"]),  # GAT runs are 5-fold mean (no separate pooled estimator)
            "auc_ci95_lo": None,
            "auc_ci95_hi": None,
            "delta_vs_ref": float(delta),
            "ref_label": f"22feat_baseline_{arch_label}_auc={baseline['auc']:.4f}",
            "verdict": verdict,
            "source_file": str((ROOT / "outputs" / src_dir / target / src_name).relative_to(ROOT)),
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
        SELECT target, stratum, fold_mean_auc, delta_vs_ref, verdict
        FROM features.paper1_r2_sensitivity
        WHERE run_id = :r
        ORDER BY target, stratum
    """), {"r": RUN_ID}).fetchall()
    print("\n[verify] Inserted rows:")
    for row in result:
        print(f"  {row[0]:<14} {row[1]:<11} AUC={row[2]:.4f} Δ={row[3]:+.4f}  {row[4]}")
