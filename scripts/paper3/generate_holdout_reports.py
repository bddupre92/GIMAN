#!/usr/bin/env python3
"""Generate the human-readable holdout_report.md files for P3 and P4.

Reads the JSON output of run_deephit_holdout.py, run_graph_dt_holdout.py,
compute_paired_bootstrap_holdout.py, and run_conformal_survival_holdout.py
and emits:

    outputs/paper3_holdout_v1/holdout_report.md
    outputs/paper4_holdout_v1/holdout_report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
HOLDOUT_JSON = DATA_DIR / "06_longitudinal_staging" / "holdout_v1_patnos.json"

P3_DIR = PROJECT_ROOT / "outputs" / "paper3_holdout_v1"
P4_DIR = PROJECT_ROOT / "outputs" / "paper4_holdout_v1"


def _fmt(v, digits=4):
    if v is None:
        return "N/A"
    if isinstance(v, (int,)):
        return str(v)
    try:
        return f"{v:.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def generate_paper3_report() -> str:
    with open(HOLDOUT_JSON) as f:
        split = json.load(f)
    with open(P3_DIR / "deephit_holdout_metrics.json") as f:
        dh = json.load(f)
    with open(P3_DIR / "graph_dt_holdout_metrics.json") as f:
        gdt = json.load(f)
    with open(P3_DIR / "paired_bootstrap.json") as f:
        pb = json.load(f)

    bs = pb["paired_bootstrap"]
    wx = pb["wilcoxon_per_patient"]

    # Determine verdict
    ci_lo = bs["ci_95_lo"]
    ci_hi = bs["ci_95_hi"]
    if ci_lo > 0:
        verdict = "**Graph-DT significantly beats DeepHit** (Δ > 0; 95% CI excludes 0)"
    elif ci_hi < 0:
        verdict = "**DeepHit significantly beats Graph-DT** (Δ < 0; 95% CI excludes 0)"
    else:
        verdict = "**Graph-DT and DeepHit match on holdout** (95% CI contains 0)"

    lines = []
    a = lines.append

    a("# Paper 3 Holdout Report (seed = 2026)")
    a("")
    a("Pre-registered holdout evaluation for submission rigor. The 5-fold ")
    a("stratified CV results reported in the published Paper 3 tables are the ")
    a("primary generalization estimate; this report supplements those with ")
    a("a single-use, never-tuned evaluation on a 380-patient holdout split ")
    a("carved from the 1,900-patient cohort with seed = 2026.")
    a("")
    a("## Split statistics")
    a("")
    a(f"| | Dev | Holdout | Total |")
    a(f"|---|---:|---:|---:|")
    a(f"| Patients | {split['dev_count']} | {split['holdout_count']} | {split['total_patients']} |")
    a(f"| Holdout episodes | — | {dh['n_holdout_episodes']} | — |")
    a(f"| Holdout events   | — | {dh['n_holdout_events']} | — |")
    a("")
    a("Stratification key: `{stage_idx}_{any_transition_observed}` (7 stages × 2 flags).")
    a("")
    a("Per-stratum breakdown:")
    a("")
    a("| Stratum | Dev | Holdout |")
    a("|---|---:|---:|")
    for s in sorted(split["stratum_summary"].keys()):
        v = split["stratum_summary"][s]
        a(f"| {s} | {v['dev']} | {v['holdout']} |")
    a("")
    a("## Headline metrics (holdout set, single evaluation)")
    a("")
    a("| Metric | DeepHit | Graph-DT v5 |")
    a("|---|---:|---:|")
    a(f"| C-td | {_fmt(dh['c_td_holdout'])} | {_fmt(gdt['c_td_holdout'])} |")
    a(f"| IBS  | {_fmt(dh['ibs_holdout'])} | {_fmt(gdt['ibs_holdout'])} |")
    a(f"| Brier @ 5yr | {_fmt(dh['brier_5yr_holdout'])} | {_fmt(gdt['brier_5yr_holdout'])} |")
    a("")
    a("Brier score at each horizon:")
    a("")
    a("| Horizon | DeepHit | Graph-DT v5 |")
    a("|---|---:|---:|")
    for h in ["1yr", "2yr", "5yr", "10yr"]:
        a(f"| {h} | {_fmt(dh['brier_at_horizons'].get(h))} | {_fmt(gdt['brier_at_horizons'].get(h))} |")
    a("")
    a("## Per-transition C-td on holdout")
    a("")
    a("| Destination stage | DeepHit | Graph-DT v5 |")
    a("|---|---:|---:|")
    all_keys = sorted(
        set(dh["per_transition_ctd"].keys()) | set(gdt["per_transition_ctd"].keys())
    )
    for k in all_keys:
        a(
            f"| {k} | "
            f"{_fmt(dh['per_transition_ctd'].get(k))} | "
            f"{_fmt(gdt['per_transition_ctd'].get(k))} |"
        )
    a("")
    a("## Paired bootstrap Δ(Graph-DT − DeepHit)")
    a("")
    a(f"Resamples: {bs['n_bootstrap']} (patient-level, same sub-seed per pair).")
    a("")
    a("| Quantity | Value |")
    a("|---|---:|")
    a(f"| Δ point estimate | {_fmt(bs['point_estimate'])} |")
    a(f"| Δ median         | {_fmt(bs['delta_median'])} |")
    a(f"| 95% CI lower     | {_fmt(bs['ci_95_lo'])} |")
    a(f"| 95% CI upper     | {_fmt(bs['ci_95_hi'])} |")
    a(f"| DeepHit mean bootstrap C-td  | {_fmt(bs['dh_mean'])} |")
    a(f"| Graph-DT mean bootstrap C-td | {_fmt(bs['gdt_mean'])} |")
    a("")
    a("### Wilcoxon signed-rank on per-patient concordance scores")
    a("")
    a("| Quantity | Value |")
    a("|---|---:|")
    a(f"| n patients compared | {wx['n']} |")
    a(f"| DeepHit mean score  | {_fmt(wx.get('dh_mean_score'))} |")
    a(f"| Graph-DT mean score | {_fmt(wx.get('gdt_mean_score'))} |")
    a(f"| Wilcoxon statistic  | {_fmt(wx['wilcoxon_stat'])} |")
    a(f"| Two-sided p-value   | {_fmt(wx['wilcoxon_p'], digits=5)} |")
    a("")
    a("## Verdict")
    a("")
    a(verdict + ".")
    a("")
    a("## Comparison with 5-fold CV (published Paper 3 tables)")
    a("")
    a("| | 5-fold CV | Holdout |")
    a("|---|---|---|")
    a(f"| DeepHit C-td  | 0.926 ± 0.018 | {_fmt(dh['c_td_holdout'])} |")
    a(f"| Graph-DT C-td | 0.920 ± 0.013 | {_fmt(gdt['c_td_holdout'])} |")
    a(f"| Δ (CV: -0.006, p_t=0.108, p_W=0.312) | see Paper 3 | "
      f"Δ = {_fmt(bs['point_estimate'])} "
      f"[{_fmt(bs['ci_95_lo'])}, {_fmt(bs['ci_95_hi'])}] |")
    a("")
    a("Note: Paper 3 5-fold CV numbers were obtained with seed = 42 and ")
    a("feature standardisation per-fold. The holdout uses seed = 2026 for ")
    a("the split and for the internal 80/20 dev → train/val partition; ")
    a("model-init seed is kept at 42 so we test robustness to the COHORT ")
    a("partition, not to weight initialization.")
    a("")
    a("## Reproducibility")
    a("")
    a(f"- Split JSON: `data/06_longitudinal_staging/holdout_v1_patnos.json`")
    a(f"- Generator:  `scripts/paper3/holdout_split_v1.py` (seed=2026)")
    a(f"- DeepHit runner:  `scripts/paper3/run_deephit_holdout.py`")
    a(f"- Graph-DT runner: `scripts/paper3/run_graph_dt_holdout.py`")
    a(f"- Paired bootstrap: `scripts/paper3/compute_paired_bootstrap_holdout.py`")
    a(f"- Checkpoints: `outputs/paper3_checkpoints/holdout_v1/{{deephit,graph_dt}}.pt`")
    a(f"- Predictions: `outputs/paper3_holdout_v1/{{deephit,graph_dt}}_predictions.csv`")
    return "\n".join(lines) + "\n"


def generate_paper4_report() -> str:
    with open(HOLDOUT_JSON) as f:
        split = json.load(f)
    with open(P4_DIR / "conformal_results.json") as f:
        conf = json.load(f)
    with open(P4_DIR / "subgroup_coverage.json") as f:
        sub = json.load(f)

    lines = []
    a = lines.append

    a("# Paper 4 Holdout Report (seed = 2026)")
    a("")
    a("Pre-registered conformal CIF band evaluation on the 380-patient ")
    a("holdout. Calibration set: 50/50 split of the 1,520-patient dev cohort ")
    a("(seed = 2026). Evaluation set: the 380 holdout patients, never seen ")
    a("during Paper 3 model training OR Paper 4 calibration.")
    a("")
    a(f"## Holdout cohort: {split['holdout_count']} patients")
    a("")
    a("## Marginal coverage on holdout")
    a("")
    a("| Model | CL=0.80 | CL=0.90 | CL=0.95 |")
    a("|---|---:|---:|---:|")
    for model_name, r in conf.items():
        row = [model_name]
        by_cl = {entry["confidence_level"]: entry for entry in r["cif_results"]}
        for cl in [0.80, 0.90, 0.95]:
            if cl in by_cl:
                row.append(_fmt(by_cl[cl]["marginal_coverage"]))
            else:
                row.append("N/A")
        a(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    a("")
    a("## Mean band width on holdout")
    a("")
    a("| Model | CL=0.80 | CL=0.90 | CL=0.95 |")
    a("|---|---:|---:|---:|")
    for model_name, r in conf.items():
        row = [model_name]
        by_cl = {entry["confidence_level"]: entry for entry in r["cif_results"]}
        for cl in [0.80, 0.90, 0.95]:
            if cl in by_cl:
                row.append(_fmt(by_cl[cl]["mean_band_width"]))
            else:
                row.append("N/A")
        a(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    a("")
    a("## Conditional conformal coverage by subgroup (CL = 0.90)")
    a("")
    a("| Model | Male | Female | Age < 60 | Age 60-70 | Age ≥ 70 |")
    a("|---|---:|---:|---:|---:|---:|")
    for model_name in sub:
        c = sub[model_name].get("0.90", {})
        row = [
            _fmt(c.get("male")),
            _fmt(c.get("female")),
            _fmt(c.get("age_lt60")),
            _fmt(c.get("age_60_70")),
            _fmt(c.get("age_gte70")),
        ]
        a(f"| {model_name} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")
    a("")
    a("## Comparison with 5-fold CV (Paper 4 published tables)")
    a("")
    a("| | 5-fold CV @ 95% CL | Holdout @ 95% CL |")
    a("|---|---|---|")
    for model_name, r in conf.items():
        by_cl = {entry["confidence_level"]: entry for entry in r["cif_results"]}
        if 0.95 in by_cl:
            cv_ref = {
                "DeepHit": "0.911 ± 0.015",
                "Graph-DT": "0.914 ± 0.013",
            }.get(model_name, "—")
            a(
                f"| {model_name} marginal coverage | "
                f"{cv_ref} | "
                f"{_fmt(by_cl[0.95]['marginal_coverage'])} |"
            )
    a("")
    a("## Reproducibility")
    a("")
    a("- Calibration seed: 2026 (dev 50/50 split)")
    a("- Runner: `scripts/paper4/run_conformal_survival_holdout.py`")
    a("- Checkpoints reused from `outputs/paper3_checkpoints/holdout_v1/`")
    a("- Outputs: `outputs/paper4_holdout_v1/{conformal_results,timing_intervals,subgroup_coverage}.json`")

    return "\n".join(lines) + "\n"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--paper3", action="store_true")
    parser.add_argument("--paper4", action="store_true")
    parser.add_argument("--both", action="store_true")
    args = parser.parse_args()

    if args.both or args.paper3:
        out = P3_DIR / "holdout_report.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(generate_paper3_report())
        print(f"Wrote {out}")

    if args.both or args.paper4:
        out = P4_DIR / "holdout_report.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(generate_paper4_report())
        print(f"Wrote {out}")

    if not (args.paper3 or args.paper4 or args.both):
        print("No report selected. Use --paper3, --paper4, or --both.")


if __name__ == "__main__":
    main()
