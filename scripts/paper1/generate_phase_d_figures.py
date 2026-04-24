"""Paper 1 Phase D — Fig 7 (calibration reliability) + Fig 9 (SHAP + subgroup forest).

Reads JSON outputs from WS1.8 and WS1.9 and renders two publication-quality
figures for the revised manuscript.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAL_DIR = ROOT / "outputs" / "paper1_calibration" / "results"
SHAP_DIR = ROOT / "outputs" / "paper1_shap_subgroup"
FIG_DIR = ROOT / "outputs" / "paper1_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

TARGETS = [
    ("binary", "Binary"),
    ("three_class", "Three-class"),
    ("full_ordinal", "Full ordinal"),
    ("nsd_positive", "NSD+"),
]


def load_cal(name: str) -> dict:
    return json.load(open(CAL_DIR / f"{name}.json"))["calibration"]


def fig7_calibration_bars() -> None:
    """Internal-vs-external ECE and Brier bar panel across 4 targets."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

    internal_ece, external_ece = [], []
    internal_brier, external_brier = [], []
    labels = []
    for key, label in TARGETS:
        i = load_cal(f"internal_{key}")
        e = load_cal(f"external_{key}")
        internal_ece.append(i["ECE"])
        external_ece.append(e["ECE"])
        internal_brier.append(i["Brier"])
        external_brier.append(e["Brier"])
        labels.append(label)

    x = np.arange(len(labels))
    w = 0.38
    blue, orange = "#1f77b4", "#ff7f0e"

    axes[0].bar(x - w / 2, internal_ece, w, label="Internal (PPMI)", color=blue)
    axes[0].bar(x + w / 2, external_ece, w, label="External (BioFIND)", color=orange)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("Expected Calibration Error")
    axes[0].set_title("(a) ECE — 10$\\times$ degradation externally")
    axes[0].axhline(0.05, linestyle=":", color="grey", alpha=0.7, linewidth=0.8)
    axes[0].text(-0.35, 0.055, "well-calibrated threshold (0.05)", fontsize=7, color="grey")
    axes[0].legend(loc="upper left", frameon=False)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 0.5)

    axes[1].bar(x - w / 2, internal_brier, w, label="Internal (PPMI)", color=blue)
    axes[1].bar(x + w / 2, external_brier, w, label="External (BioFIND)", color=orange)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("Brier Score")
    axes[1].set_title("(b) Brier — similar degradation pattern")
    axes[1].legend(loc="upper left", frameon=False)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig7_calibration.{ext}")
    print(f"Wrote fig7 to {FIG_DIR}")


def fig9_shap_subgroup() -> None:
    """SHAP top-10 features + subgroup AUC forest (binary target)."""
    shap_data = json.load(open(SHAP_DIR / "top_features_binary.json"))
    subgroup_data = json.load(open(SHAP_DIR / "subgroup_binary.json"))["per_group_auc"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # Left: SHAP top-10 horizontal bar
    top10 = shap_data["top10_features"]
    names = [t[0] for t in top10]
    values = [t[1] for t in top10]
    y = np.arange(len(names))
    axes[0].barh(y, values, color="#2ca02c")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean |SHAP value|")
    axes[0].set_title("(a) Top-10 SHAP features — CatBoost binary")
    axes[0].grid(axis="x", alpha=0.3)

    # Right: subgroup forest (per-carrier AUC + CI vs Non-carrier reference)
    order = ["LRRK2+", "GBA+_only", "APOE_e4+_only", "Non-carrier"]
    display_names = ["LRRK2+", "GBA+ (only)", "APOEε4+ (only)", "Non-carrier"]
    aucs, ci_los, ci_his, ns = [], [], [], []
    for k in order:
        g = subgroup_data[k]
        aucs.append(g["auc"])
        ci_los.append(g["ci_low"])
        ci_his.append(g["ci_high"])
        ns.append(g["n"])
    y = np.arange(len(order))
    err_lo = [a - lo for a, lo in zip(aucs, ci_los)]
    err_hi = [hi - a for a, hi in zip(aucs, ci_his)]
    axes[1].errorbar(aucs, y, xerr=[err_lo, err_hi], fmt="s", color="#d62728",
                     capsize=3, markersize=7, ecolor="#d62728", elinewidth=1.3)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([f"{nm} (n={n})" for nm, n in zip(display_names, ns)])
    axes[1].invert_yaxis()
    axes[1].axvline(0.9, linestyle=":", color="grey", alpha=0.6, linewidth=0.8)
    axes[1].axvline(1.0, linestyle="--", color="grey", alpha=0.4, linewidth=0.6)
    axes[1].set_xlabel("Binary AUC [95\\% CI]")
    axes[1].set_title("(b) Subgroup fairness by genetic carrier status")
    axes[1].set_xlim(0.88, 1.01)
    axes[1].grid(axis="x", alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig9_shap_subgroup.{ext}")
    print(f"Wrote fig9 to {FIG_DIR}")


if __name__ == "__main__":
    fig7_calibration_bars()
    fig9_shap_subgroup()
