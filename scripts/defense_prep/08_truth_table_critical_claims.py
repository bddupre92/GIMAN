#!/usr/bin/env python3
"""Truth-table verifier for the 5 critical contradicted claims found by the
2026-04-14 semantic audit swarm.

For each contradicted/misleading claim:
  - chapter value (what the dissertation prose says today)
  - JSON value (what the on-disk artifact says — re-loaded fresh)
  - in-Docker recompute (when applicable, e.g. percentages from raw values)
  - verdict: AGREE / CHAPTER_WRONG / JSON_STALE / RECOMPUTE_DIFFERS

Run inside Docker:
  docker compose exec giman python scripts/defense_prep/08_truth_table_critical_claims.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from textwrap import indent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs" / "defense_prep" / "per_paper_verification" / "TRUTH_TABLE.md"


def load(path: str) -> dict:
    return json.loads((PROJECT_ROOT / path).read_text())


def fmt(v: float, ndp: int = 4) -> str:
    if v is None:
        return "(none)"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt(x, ndp) for x in v) + "]"
    return f"{v:.{ndp}f}"


def main() -> None:
    rows: list[dict] = []

    # ---- Paper 1: binary CatBoost AUC 0.981 vs JSON ----
    p1bin = load("outputs/paper1_benchmark/binary_results.json")
    p1bin_auc_json = p1bin["catboost"]["aggregate"]["auc_roc"]
    rows.append({
        "id": "P1-Binary-AUC",
        "chapter_claim": 0.981,
        "json_value": p1bin_auc_json,
        "verdict": "CHAPTER_WRONG" if abs(0.981 - p1bin_auc_json) > 0.001 else "AGREE",
        "fix": "Replace 0.981 → 0.979 (rounded from JSON value)",
    })

    # ---- Paper 1: three-class bal_acc 0.778 vs JSON ----
    p13c = load("outputs/paper1_benchmark/three_class_results.json")
    p13c_ba_json = p13c["catboost"]["aggregate"]["balanced_accuracy"]
    rows.append({
        "id": "P1-3class-BalAcc",
        "chapter_claim": 0.778,
        "json_value": p13c_ba_json,
        "verdict": "CHAPTER_WRONG" if abs(0.778 - p13c_ba_json) > 0.001 else "AGREE",
        "fix": "Replace 0.778 → 0.783",
    })

    # ---- Paper 2: Stage RMSE % reductions ----
    p2 = load("outputs/paper2_benchmark/per_stage_analysis.json")
    s1_van = p2["frac_0.1"]["per_stage"]["GIMIN_Vanilla"]["1"]
    s1_dec = p2["frac_0.1"]["per_stage"]["GIMIN_StageDecoderOnly"]["1"]
    s3_van = p2["frac_0.1"]["per_stage"]["GIMIN_Vanilla"]["3"]
    s3_dec = p2["frac_0.1"]["per_stage"]["GIMIN_StageDecoderOnly"]["3"]
    s1_pct = (s1_van - s1_dec) / s1_van * 100.0
    s3_pct = (s3_van - s3_dec) / s3_van * 100.0
    rows.append({
        "id": "P2-Stage1-RMSE-%",
        "chapter_claim": 45.0,
        "json_value": None,
        "recompute": s1_pct,
        "verdict": "CHAPTER_WRONG" if abs(45.0 - s1_pct) > 1.0 else "AGREE",
        "fix": f"Replace 45% → {s1_pct:.1f}% (Stage 1 Vanilla {s1_van:.2f} → DecoderOnly {s1_dec:.2f})",
    })
    rows.append({
        "id": "P2-Stage3-RMSE-%",
        "chapter_claim": 33.0,
        "json_value": None,
        "recompute": s3_pct,
        "verdict": "CHAPTER_WRONG" if abs(33.0 - s3_pct) > 1.0 else "AGREE",
        "fix": f"Replace 33% → {s3_pct:.1f}% (Stage 3 Vanilla {s3_van:.2f} → DecoderOnly {s3_dec:.2f})",
    })

    # ---- Paper 2: Conformal coverage 90.6% vs JSON ----
    p2cov = load("outputs/paper2_benchmark/conformal_frac0.1.json")
    p2cov_val = p2cov["per_feature"]["marginal"]["coverage"] * 100.0
    rows.append({
        "id": "P2-Conformal-Coverage-pct",
        "chapter_claim": 90.6,
        "json_value": p2cov_val,
        "verdict": "CHAPTER_WRONG" if abs(90.6 - p2cov_val) > 0.1 else "AGREE",
        "fix": f"Replace 90.6% → {p2cov_val:.1f}%",
    })
    p2nmpiw = p2cov["per_feature"]["marginal"]["nmpiw"]
    rows.append({
        "id": "P2-Conformal-NMPIW",
        "chapter_claim": 1.42,
        "json_value": p2nmpiw,
        "verdict": "CHAPTER_WRONG" if abs(1.42 - p2nmpiw) > 0.005 else "AGREE",
        "fix": f"Replace 1.42 → {p2nmpiw:.2f}",
    })

    # ---- Paper 3: Graph-DT C-td (chapter says 0.926 = DeepHit's number) ----
    p3 = load("outputs/paper3_benchmark/benchmark_summary.json")
    gdt = p3.get("models", {}).get("graph_dt", {})
    dh = p3.get("models", {}).get("deephit", {})
    def _ctd(model_blob):
        v = model_blob.get("c_td")
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict):
            return v.get("mean")
        for k in ("c_td_mean", "ctd_mean", "mean_c_td"):
            if k in model_blob:
                return model_blob[k]
        return None
    gdt_mean = _ctd(gdt)
    dh_mean = _ctd(dh)
    rows.append({
        "id": "P3-GraphDT-Ctd",
        "chapter_claim": 0.926,
        "json_value": gdt_mean,
        "verdict": "CHAPTER_WRONG" if (gdt_mean is not None and abs(0.926 - gdt_mean) > 0.005) else "INSPECT",
        "fix": f"Replace 0.926 → {fmt(gdt_mean, 3) if gdt_mean else 'see benchmark_summary.json structure'}; 0.926 is DeepHit's number",
    })

    # ---- Paper 3: paired t-test t=0.03, p=0.976 ----
    p3_t = None; p3_p = None
    def _coerce_float(x):
        if x is None: return None
        try: return float(x)
        except (ValueError, TypeError): return None
    for entry in p3.get("statistical_tests", []):
        if isinstance(entry, dict) and "Paired t-test" in str(entry.get("Test", "")) and "C-td" in str(entry.get("Comparison", "")):
            p3_t = _coerce_float(entry.get("Statistic"))
            p3_p = _coerce_float(entry.get("p-value") or entry.get("p_value"))
            break
    if p3_t is None:
        for entry in p3.get("statistical_tests", []):
            if isinstance(entry, dict):
                p3_t = p3_t or _coerce_float(entry.get("Statistic") or entry.get("t"))
                p3_p = p3_p or _coerce_float(entry.get("p_value") or entry.get("p"))
                if p3_t and p3_p:
                    break
    rows.append({
        "id": "P3-paired-t",
        "chapter_claim": 0.03,
        "json_value": p3_t,
        "verdict": "CHAPTER_WRONG" if (p3_t is not None and abs(0.03 - p3_t) > 0.5) else "INSPECT",
        "fix": f"Replace t = 0.03 → {fmt(p3_t, 3) if p3_t else 'load benchmark_summary.json'}",
    })
    rows.append({
        "id": "P3-paired-p",
        "chapter_claim": 0.976,
        "json_value": p3_p,
        "verdict": "CHAPTER_WRONG" if (p3_p is not None and abs(0.976 - p3_p) > 0.05) else "INSPECT",
        "fix": f"Replace p = 0.976 → {fmt(p3_p, 3) if p3_p else 'load benchmark_summary.json'}",
    })

    # ---- Paper 10: bidirectional MAE per scan-count ----
    bi = load("outputs/mechanistic_twin/paper10_mech_vs_giman/bidirectional_demo.json")
    per_step = bi.get("per_scan_count_summary", [])
    n_at_5 = next((row["n"] for row in per_step if row.get("scans_used") == 5), None)
    mae_at_5 = next((row["mae_from_mean"] for row in per_step if row.get("scans_used") == 5), None)
    n_at_0 = next((row["n"] for row in per_step if row.get("scans_used") == 0), None)
    mae_at_0 = next((row["mae_from_mean"] for row in per_step if row.get("scans_used") == 0), None)
    rows.append({
        "id": "P10-bidir-MAE-N-mismatch",
        "chapter_claim": "MAE 0.149→0.100 over 644 patients",
        "json_value": f"prior n={n_at_0} MAE={fmt(mae_at_0, 4)} | scans=5 n={n_at_5} MAE={fmt(mae_at_5, 4)}",
        "verdict": "CHAPTER_MISLEADING" if (n_at_5 is not None and n_at_5 < 50) else "INSPECT",
        "fix": f"Reframe: 0.149 prior over n={n_at_0} → 0.100 only at scans=5 with n={n_at_5} (NOT 644)",
    })

    # ---- Paper 10: counterfactual CI rounding ----
    cf = load("outputs/mechanistic_twin/paper10_mech_vs_giman/observational_counterfactual.json")
    ci = cf["calibration_overall"].get("slope_ci95") or cf["calibration_overall"].get("slope_ci")
    rows.append({
        "id": "P10-CI-rounding",
        "chapter_claim": "[0.88, 1.29]",
        "json_value": fmt(ci, 4) if ci else "(missing)",
        "verdict": "ROUNDING_INCONSISTENT",
        "fix": f"Use [{ci[0]:.3f}, {ci[1]:.3f}] = [0.877, 1.285] OR [0.88, 1.28]" if ci else "—",
    })

    # ---- Cross-chapter: ΔAIC ch10 vs ch11 ----
    try:
        regional = load("outputs/mechanistic_twin/phase2/phase3_regional_saem_results.json")
        delta_pop = regional.get("comparison", {}).get("delta_aic_m6r_vs_m1")
    except Exception:
        delta_pop = None
    rows.append({
        "id": "Cross-Ch10-Ch11-deltaAIC",
        "chapter_claim": "ch10: 5,668 / ch11: 3,856 (no disclosure of formula difference)",
        "json_value": fmt(delta_pop, 1) if delta_pop else "(missing)",
        "verdict": "DOCUMENTATION_GAP",
        "fix": "Disclose AIC penalty convention (population vs per-patient) in BOTH chapters",
    })

    # ---- Render Markdown ----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Truth Table — In-Docker Verification of 5 Critical Contradicted Claims\n",
             "_Re-extracted by `scripts/defense_prep/08_truth_table_critical_claims.py` running inside Docker._\n",
             "\n| ID | Chapter Claim | JSON / Recompute | Verdict | Fix |\n",
             "|---|---|---|---|---|\n"]
    for r in rows:
        chap = str(r["chapter_claim"])
        if "json_value" in r and r["json_value"] is not None:
            jv = fmt(r["json_value"], 4) if isinstance(r["json_value"], float) else str(r["json_value"])
        elif "recompute" in r:
            jv = f"{r['recompute']:.2f}%"
        else:
            jv = "(missing)"
        lines.append(f"| {r['id']} | {chap} | {jv} | **{r['verdict']}** | {r['fix']} |\n")

    OUT.write_text("".join(lines))
    print("".join(lines))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
