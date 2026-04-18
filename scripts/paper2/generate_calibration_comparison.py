#!/usr/bin/env python3
"""Generate the calibration-comparison figure + decision table for Paper 2.

Phase P2-Cal.5 of the calibration comparison.

Reads raw coverage measurements for:
  - D0 (baseline GIMIN, existing 48 checkpoints)
  - B (D0 + per-feature temperature scaling, post-hoc)
  - A (retuned GIMIN with lambda_cal=0.1, warmup=0) + A+B (A + temp scaling)
  - D1 (conformal per-feature residual wrapping from existing JSON)

Produces:
  outputs/paper2_benchmark/calibration_retune/figures/fig_calibration_compare.{pdf,png}
    5-panel figure: reliability diagram at each nominal level, overlaid conditions
  outputs/paper2_benchmark/calibration_retune/calibration_comparison_table.csv
  outputs/paper2_benchmark/calibration_retune/calibration_comparison_summary.json

Usage:
    .venv/bin/python scripts/paper2/generate_calibration_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "paper2_benchmark" / "calibration_retune"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COVERAGE_TARGETS = [0.50, 0.70, 0.80, 0.90, 0.95]


def _load_d0_b_from_measurement(path: Path) -> dict:
    """Load raw (D0) + temperature-scaled (B) coverage from measurement JSON."""
    d = json.load(open(path))
    raw_by_gamma = {g: [] for g in COVERAGE_TARGETS}
    temp_by_gamma = {g: [] for g in COVERAGE_TARGETS}
    for e in d["per_checkpoint"]:
        for g in COVERAGE_TARGETS:
            key = f"gamma_{g:.2f}"
            raw_by_gamma[g].append(e["raw_coverage"][key]["observed_coverage"])
            temp_by_gamma[g].append(e["temp_scaled_coverage"][key]["observed_coverage"])
    return {
        "D0_raw": {g: np.array(raw_by_gamma[g]) for g in COVERAGE_TARGETS},
        "B_temp": {g: np.array(temp_by_gamma[g]) for g in COVERAGE_TARGETS},
    }


def _load_d1_conformal() -> dict:
    """Load D1 conformal per-feature coverage from existing conformal_frac*.json."""
    frac_paths = {
        0.1: ROOT / "outputs" / "paper2_benchmark" / "conformal_frac0.1.json",
        0.2: ROOT / "outputs" / "paper2_benchmark" / "conformal_frac0.2.json",
        0.3: ROOT / "outputs" / "paper2_benchmark" / "conformal_frac0.3.json",
        0.5: ROOT / "outputs" / "paper2_benchmark" / "conformal_frac0.5.json",
    }
    # D1 conformal reports observed coverage ONLY at the 90% target (per-feature mode)
    # but the calibration_curves.marginal field has 5 levels
    coverages_by_gamma = {g: [] for g in COVERAGE_TARGETS}
    for frac, p in frac_paths.items():
        if not p.exists():
            continue
        d = json.load(open(p))
        cc = d.get("calibration_curves", {}).get("marginal", {})
        for g in COVERAGE_TARGETS:
            v = cc.get(f"{g}") or cc.get(f"{g:.2f}")
            if v is not None:
                coverages_by_gamma[g].append(float(v))
    return {
        "D1_conformal": {g: np.array(coverages_by_gamma[g]) for g in COVERAGE_TARGETS},
    }


def _load_a_from_measurement(path: Path) -> dict | None:
    if not path.exists():
        return None
    return _load_d0_b_from_measurement(path)


def _summary_table(conditions: dict) -> dict:
    """Produce {condition -> {gamma -> {mean, std, median, gap_from_target}}}.

    conditions: {name -> {gamma -> np.array of per-checkpoint coverages}}
    """
    summary = {}
    for name, by_gamma in conditions.items():
        summary[name] = {}
        for g in COVERAGE_TARGETS:
            vals = by_gamma.get(g, np.array([]))
            if vals.size == 0:
                summary[name][f"gamma_{g:.2f}"] = None
                continue
            summary[name][f"gamma_{g:.2f}"] = {
                "target": float(g),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "median": float(np.median(vals)),
                "n_checkpoints": int(vals.size),
                "gap_from_target": float(vals.mean() - g),
            }
    return summary


def main():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"matplotlib import failed: {e}")
        sys.exit(1)

    # ── Load conditions ───────────────────────────────────────────────
    conditions = {}

    d0_b_path = OUT_DIR / "raw_coverage_measurement.json"
    if d0_b_path.exists():
        d0_b = _load_d0_b_from_measurement(d0_b_path)
        conditions["D0 raw"] = d0_b["D0_raw"]
        conditions["B temp-scaled"] = d0_b["B_temp"]

    d1 = _load_d1_conformal()
    if any(len(v) > 0 for v in d1["D1_conformal"].values()):
        conditions["D1 conformal"] = d1["D1_conformal"]

    a_path = OUT_DIR / "A_variant_raw_coverage.json"
    a = _load_a_from_measurement(a_path)
    if a is not None:
        conditions["A retuned-raw"] = a["D0_raw"]
        conditions["A+B retuned+temp"] = a["B_temp"]

    print(f"Loaded {len(conditions)} conditions:")
    for name, by_gamma in conditions.items():
        n = sum(len(v) for v in by_gamma.values())
        print(f"  {name}: {n} coverage samples total")

    # ── Summary table ──────────────────────────────────────────────────
    summary = _summary_table(conditions)
    with open(OUT_DIR / "calibration_comparison_summary.json", "w") as fp:
        json.dump(
            {
                "coverage_targets": COVERAGE_TARGETS,
                "conditions": list(conditions.keys()),
                "summary_by_condition": summary,
            },
            fp,
            indent=2,
        )

    # CSV table (readable)
    csv_lines = ["condition,target,mean,std,median,n,gap"]
    for name, by_gamma in summary.items():
        for g in COVERAGE_TARGETS:
            e = by_gamma.get(f"gamma_{g:.2f}")
            if e is None:
                continue
            csv_lines.append(
                f"{name},{g:.2f},{e['mean']:.4f},{e['std']:.4f},{e['median']:.4f},{e['n_checkpoints']},{e['gap_from_target']:+.4f}"
            )
    (OUT_DIR / "calibration_comparison_table.csv").write_text("\n".join(csv_lines))

    # Print readable table
    print("\n" + "=" * 80)
    print("CALIBRATION COMPARISON TABLE")
    print("=" * 80)
    header = f"{'Target':>8}"
    for name in conditions:
        header += f"{name:>20}"
    print(header)
    for g in COVERAGE_TARGETS:
        line = f"{g:>8.2f}"
        for name in conditions:
            e = summary[name].get(f"gamma_{g:.2f}")
            if e is None:
                line += f"{'--':>20}"
            else:
                line += f"{e['mean']:>12.3f} (Δ{e['gap_from_target']:+5.3f})"
        print(line)

    # ── Figure: reliability diagram overlaid ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        [0.4, 1.0], [0.4, 1.0],
        "--", color="gray", linewidth=1.0, label="Ideal calibration",
    )

    colors = {
        "D0 raw": "#d62728",  # red
        "B temp-scaled": "#1f77b4",  # blue
        "D1 conformal": "#2ca02c",  # green
        "A retuned-raw": "#ff7f0e",  # orange
        "A+B retuned+temp": "#9467bd",  # purple
    }
    markers = {
        "D0 raw": "x",
        "B temp-scaled": "o",
        "D1 conformal": "^",
        "A retuned-raw": "s",
        "A+B retuned+temp": "D",
    }

    for name, by_gamma in conditions.items():
        xs = []
        ys = []
        yerr = []
        for g in COVERAGE_TARGETS:
            vals = by_gamma.get(g, np.array([]))
            if vals.size == 0:
                continue
            xs.append(g)
            ys.append(vals.mean())
            yerr.append(vals.std(ddof=0))
        ax.errorbar(
            xs, ys, yerr=yerr,
            color=colors.get(name, "black"),
            marker=markers.get(name, "o"),
            markersize=9, linewidth=1.8,
            label=name,
            capsize=3,
        )

    ax.set_xlabel("Nominal coverage $\\gamma$", fontsize=12)
    ax.set_ylabel("Observed coverage", fontsize=12)
    ax.set_title(
        "Calibration comparison: raw vs temperature-scaled vs conformal\n"
        "(48 checkpoints per condition, mean ± SD)",
        fontsize=13,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_calibration_compare.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_calibration_compare.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote: {FIG_DIR / 'fig_calibration_compare.pdf'}")
    print(f"Wrote: {FIG_DIR / 'fig_calibration_compare.png'}")
    print(f"Wrote: {OUT_DIR / 'calibration_comparison_table.csv'}")


if __name__ == "__main__":
    main()
