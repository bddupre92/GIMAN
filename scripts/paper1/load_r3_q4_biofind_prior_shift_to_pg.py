"""Load R3-Q4 BioFIND prior-shift results into features.paper1_r2_sensitivity.

Inserts 6 rows: 3 targets x 2 correction methods (Saerens EM + density-ratio).
The baseline (no-correction) AUC is encoded as ref_label and the post-correction
AUC as pooled_auc; delta_vs_ref carries the AUC delta.

Verdict strings inherit from the per-method per-target verdicts in the JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import text

from giman_pipeline.data.db import get_engine

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
JSON_PATH = ROOT / "outputs" / "paper1_r2_responses" / "q_r3_q4_biofind_prior_shift.json"
RUN_ID = "q_r3_q4_biofind_prior_shift"

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
    base = td["baseline"]
    base_auc = base["auc"]
    n_classes = td["n_classes"]
    n_bf = td["n_patients_biofind"]
    for method_label, method_key in (
        ("saerens_em", "saerens_em_correction"),
        ("density_ratio", "density_ratio_weighting"),
    ):
        m = td[method_key]
        rows.append({
            "run_id": RUN_ID,
            "target": target,
            "feature_set": "12-feat common (external)",
            "stratum": method_label,
            "n_patients": int(n_bf),
            "n_features": 12,
            "n_folds_used": 1,  # Single external evaluation, not k-fold
            "fold_mean_auc": None,
            "fold_std_auc": None,
            "pooled_auc": float(m["auc"]),
            "auc_ci95_lo": None,
            "auc_ci95_hi": None,
            "delta_vs_ref": float(m["auc"] - base_auc),
            "ref_label": (
                f"baseline (no correction): AUC={base_auc:.3f}, ECE={base['ece_10bin']:.3f}, "
                f"BalAcc={base['balanced_accuracy']:.3f}, Cov90={base['conformal_coverage_90pct']:.3f}"
            ),
            "verdict": (
                f"{m['verdict']}: dECE={m['delta_ece']:+.3f}, dBalAcc={m['delta_balanced_acc_pp']:+.2f}pp, "
                f"dAUC={m['delta_auc_pp']:+.2f}pp, dCov={m['delta_coverage_pp']:+.2f}pp"
            ),
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
        print(f"[verify] {row[0]:<14} {row[1]:<14} AUC={row[2]:.3f} dAUC={row[3]:+.3f} | {row[4]}")
