"""Load R5-Q4 (AUPRC + DCA) results into features.paper1_r2_sensitivity.

Inserts 4 rows: 1 per target (binary, three_class, full_ordinal, nsd_positive).
- pooled_auc encodes the macro AUPRC point estimate
- auc_ci95_{lo,hi} encode the 1,000-resample patient-level bootstrap CI
- verdict encodes the DCA positive-NB threshold range
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import text

from giman_pipeline.data.db import get_engine

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
JSON_PATH = ROOT / "outputs" / "paper1_r2_responses" / "q_r5_q4_auprc_dca.json"
RUN_ID = "q_r5_q4_auprc"

engine = get_engine()
data = json.loads(JSON_PATH.read_text())

with engine.begin() as conn:
    deleted = conn.execute(
        text("DELETE FROM features.paper1_r2_sensitivity WHERE run_id = :r"),
        {"r": RUN_ID},
    ).rowcount
    print(f"[delete] {RUN_ID}: {deleted} prior rows cleared")

rows: list[dict] = []
for target, td in data["by_target"].items():
    n_classes = td["n_classes"]
    n_pts = td["n_patients"]

    if target == "binary":
        rng = td.get("dca_positive_nb_range")
        cal = td.get("calibration_aware_operating_point", {})
        if rng is not None:
            verdict = (
                f"DCA positive NB over pt={rng[0]:.2f}-{rng[1]:.2f}; "
                f"calib-aware op pt={cal.get('threshold'):.2f} NB={cal.get('net_benefit'):.3f}"
            )
        else:
            verdict = "DCA: no positive-NB range vs reference strategies"
    else:
        per_class = td.get("decision_curve_per_class", [])
        labels = td["class_labels"]
        parts: list[str] = []
        for pc in per_class:
            r2 = pc.get("dca_positive_nb_range")
            label = labels[pc["class"]] if pc["class"] < len(labels) else str(pc["class"])
            if r2 is None:
                parts.append(f"{label}: NONE")
            else:
                parts.append(f"{label}: {r2[0]:.2f}-{r2[1]:.2f}")
        verdict = "DCA(OvR) " + "; ".join(parts)

    rows.append({
        "run_id": RUN_ID,
        "target": target,
        "feature_set": "21-feat strict-circularity primary (Path 3)",
        "stratum": "macro_auprc",
        "n_patients": int(n_pts),
        "n_features": 21,
        "n_folds_used": 5,
        "fold_mean_auc": None,
        "fold_std_auc": None,
        "pooled_auc": float(td["auprc_macro"]),
        "auc_ci95_lo": float(td["auprc_macro_ci95_lo"]) if td.get("auprc_macro_ci95_lo") is not None else None,
        "auc_ci95_hi": float(td["auprc_macro_ci95_hi"]) if td.get("auprc_macro_ci95_hi") is not None else None,
        "delta_vs_ref": None,
        "ref_label": f"{n_classes}-class macro AUPRC; bootstrap n={data['n_boot']} seed={data['seed']}",
        "verdict": verdict[:512],  # safety bound
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
        SELECT target, n_patients, pooled_auc, auc_ci95_lo, auc_ci95_hi, verdict
        FROM features.paper1_r2_sensitivity
        WHERE run_id = :r
        ORDER BY target
    """), {"r": RUN_ID}).fetchall()
    for row in result:
        ci = f"[{row[3]:.3f}, {row[4]:.3f}]" if (row[3] is not None and row[4] is not None) else "n/a"
        print(f"[verify] {row[0]:<14} n={row[1]:<5} AUPRC={row[2]:.3f} CI={ci}")
        print(f"         verdict: {row[5]}")
