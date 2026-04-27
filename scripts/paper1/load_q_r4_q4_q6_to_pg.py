"""Load R4-Q4 BioFIND SOTA + R4-Q6 internal-12feat results into PostgreSQL.

Two run_ids:
- q_r4_q4_biofind_sota: per-method per-target rows for TabPFN + AutoGluon BioFIND
  (and any tree baselines, for completeness). Up to 6 rows (3 targets × 2 methods)
  loaded based on what JSONs exist.
- q_r4_q6_internal_12feat: per-target CatBoost rows on the 12-feat common subset
  (4 rows: binary, three_class, full_ordinal, nsd_positive).

The verdict field encodes a short reviewer-facing summary.
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import text

from giman_pipeline.data.db import get_engine

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
EV_DIR = ROOT / "outputs" / "external_validation"
RESP_DIR = ROOT / "outputs" / "paper1_r2_responses"

engine = get_engine()

# ── 1) Wipe prior rows ────────────────────────────────────────────────────────
with engine.begin() as conn:
    for run_id in ("q_r4_q4_biofind_sota", "q_r4_q6_internal_12feat"):
        result = conn.execute(
            text(
                "DELETE FROM features.paper1_r2_sensitivity WHERE run_id = :r"
            ),
            {"r": run_id},
        )
        print(f"[delete] {run_id}: {result.rowcount} prior rows cleared")

# ── 2) Q4: TabPFN + AutoGluon BioFIND ────────────────────────────────────────
q4_rows = []
SOTA_METHODS = ["tabpfn", "autogluon"]
TARGETS = ["binary", "three_class", "nsd_positive"]
for target in TARGETS:
    for method in SOTA_METHODS:
        path = EV_DIR / target / f"biofind_{method}_results.json"
        if not path.exists():
            print(f"[skip] {method}/{target}: JSON not found ({path.name})")
            continue
        d = json.loads(path.read_text())
        em = d["external_metrics"]
        verdict_bits = []
        verdict_bits.append(f"bal_acc={em['bal_acc']:.3f}")
        if em["bal_acc_ci"]:
            verdict_bits.append(
                f"CI=[{em['bal_acc_ci'][0]:.3f},{em['bal_acc_ci'][1]:.3f}]"
            )
        if em["auc"] == em["auc"]:  # not NaN
            verdict_bits.append(f"AUC={em['auc']:.3f}")
        else:
            verdict_bits.append("AUC=nan")
        verdict_bits.append(f"QWK={em['qwk']:.3f}")
        verdict_bits.append(f"n={em['n_ground_truth']}")
        q4_rows.append(
            {
                "run_id": "q_r4_q4_biofind_sota",
                "target": target,
                "feature_set": "12feat_common_subset",
                "stratum": method,
                "n_patients": int(em["n_ground_truth"]),
                "n_features": int(d["n_features"]),
                "n_folds_used": None,
                "fold_mean_auc": None,
                "fold_std_auc": None,
                "pooled_auc": float(em["auc"]) if em["auc"] == em["auc"] else None,
                "auc_ci95_lo": float(em["auc_ci"][0]) if em.get("auc_ci") else None,
                "auc_ci95_hi": float(em["auc_ci"][1]) if em.get("auc_ci") else None,
                "delta_vs_ref": None,
                "ref_label": "PPMI 12-feat trained, BioFIND test (Russo 2025 NSD-ISS GT)",
                "verdict": "; ".join(verdict_bits),
                "source_file": str(path.relative_to(ROOT)),
            }
        )
print(f"[stage] q_r4_q4_biofind_sota: {len(q4_rows)} rows ready")

# ── 3) Q6: internal 12-feat CatBoost ─────────────────────────────────────────
q6_path = RESP_DIR / "q_r4_q6_internal_12feat.json"
q6_data = json.loads(q6_path.read_text())
q6_rows = []
for tgt, r in q6_data["per_target"].items():
    pooled_auc = r["pooled_oof_auc"]
    ci = r["pooled_oof_ci95"]
    fold_mean = r["fold_mean_auc"]
    verdict = (
        f"pooled_AUC={pooled_auc:.3f} CI=[{ci[0]:.3f},{ci[1]:.3f}]; "
        f"fold_mean={fold_mean:.3f} ± {r['fold_std_auc']:.3f}; "
        f"bal={r['pooled_oof_balanced_acc']:.3f} qwk={r['pooled_oof_qwk']:.3f}"
    )
    q6_rows.append(
        {
            "run_id": "q_r4_q6_internal_12feat",
            "target": tgt,
            "feature_set": "12feat_common_subset",
            "stratum": "catboost_default",
            "n_patients": int(r["n_patients"]),
            "n_features": int(r["n_features"]),
            "n_folds_used": 5,
            "fold_mean_auc": float(fold_mean),
            "fold_std_auc": float(r["fold_std_auc"]),
            "pooled_auc": float(pooled_auc),
            "auc_ci95_lo": float(ci[0]),
            "auc_ci95_hi": float(ci[1]),
            "delta_vs_ref": None,
            "ref_label": "PPMI 5-fold CV, CatBoost defaults (iters=500, lr=0.05, depth=6, balanced)",
            "verdict": verdict,
            "source_file": str(q6_path.relative_to(ROOT)),
        }
    )
print(f"[stage] q_r4_q6_internal_12feat: {len(q6_rows)} rows ready")

# ── 4) Insert all ────────────────────────────────────────────────────────────
sql = text(
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
)
with engine.begin() as conn:
    for row in q4_rows + q6_rows:
        conn.execute(sql, row)

print(f"[insert] q_r4_q4_biofind_sota: {len(q4_rows)} rows")
print(f"[insert] q_r4_q6_internal_12feat: {len(q6_rows)} rows")

# ── 5) Verify ────────────────────────────────────────────────────────────────
with engine.begin() as conn:
    for run_id in ("q_r4_q4_biofind_sota", "q_r4_q6_internal_12feat"):
        result = conn.execute(
            text(
                """
                SELECT run_id, COUNT(*) AS n, STRING_AGG(DISTINCT target, ', ') AS targets,
                       STRING_AGG(DISTINCT stratum, ', ') AS strata
                FROM features.paper1_r2_sensitivity
                WHERE run_id = :r
                GROUP BY run_id
                """
            ),
            {"r": run_id},
        ).fetchone()
        if result:
            print(
                f"[verify] {result[0]}: {result[1]} rows, targets={result[2]}, strata={result[3]}"
            )
        else:
            print(f"[verify] {run_id}: NO ROWS")
