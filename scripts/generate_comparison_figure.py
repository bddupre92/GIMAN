"""Generate Figure 14: Feature Expansion Comparison

Creates a publication-quality grouped bar chart comparing AUC and PR-AUC
across 4 feature configurations of the Fuzzy GIMAN model, with 95%
confidence interval error bars.

Configurations:
    1. Baseline (32 active features)
    2. +Demographics (34: +SEX, AGE_AT_VISIT)
    3. +Full Clinical (38: +SEX, AGE, NP3TOT, NP1RTOT, NHY, MCATOT)
    4. Expanded (40: all above + EDUCYRS, ANYFAMPD)

Output:
    - visualizations/publication_New/Figure14_Feature_Expansion_Comparison.png

Author: GIMAN Research Team
Date: February 2026
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Result file paths
RESULT_PATHS = {
    "Baseline\n(32 features)": PROJECT_ROOT
    / "outputs"
    / "phase9_neuro_fuzzy"
    / "PREP_20260208_SAA_COHORT3_full"
    / "full_training_results.json",
    "+Demographics\n(34 features)": PROJECT_ROOT
    / "outputs"
    / "phase9_expanded"
    / "demographics_run"
    / "full_training_results.json",
    "+Full Clinical\n(38 features)": PROJECT_ROOT
    / "outputs"
    / "phase9_expanded"
    / "clinical_run"
    / "full_training_results.json",
    "Expanded\n(40 features)": PROJECT_ROOT
    / "outputs"
    / "phase9_expanded"
    / "expanded_run"
    / "full_training_results.json",
}

OUTPUT_DIR = PROJECT_ROOT / "visualizations" / "publication_New"


def load_results():
    """Load training results for all configs."""
    results = {}
    for name, path in RESULT_PATHS.items():
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
        else:
            print(f"WARNING: Missing result file: {path}")
    return results


def generate_figure(results: dict):
    """Generate the grouped bar chart."""
    configs = list(results.keys())
    n = len(configs)

    # Extract metrics
    aucs = [results[c]["best_test_auc"] for c in configs]
    pr_aucs = [results[c]["test_pr_auc"] for c in configs]
    ci_lows = [results[c]["auc_ci_95"][0] for c in configs]
    ci_highs = [results[c]["auc_ci_95"][1] for c in configs]

    # Error bar computation (asymmetric)
    auc_err_low = [aucs[i] - ci_lows[i] for i in range(n)]
    auc_err_high = [ci_highs[i] - aucs[i] for i in range(n)]

    # Bar positions
    x = np.arange(n)
    width = 0.35

    # Color scheme
    colors_auc = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    colors_prauc = ["#90CAF9", "#A5D6A7", "#FFE0B2", "#CE93D8"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # AUC bars
    bars1 = ax.bar(
        x - width / 2,
        aucs,
        width,
        label="AUC-ROC",
        color=colors_auc,
        edgecolor="black",
        linewidth=0.8,
        yerr=[auc_err_low, auc_err_high],
        capsize=5,
        error_kw={"linewidth": 1.2, "capthick": 1.2},
    )

    # PR-AUC bars
    bars2 = ax.bar(
        x + width / 2,
        pr_aucs,
        width,
        label="PR-AUC",
        color=colors_prauc,
        edgecolor="black",
        linewidth=0.8,
    )

    # Axis formatting
    ax.set_xlabel("Feature Configuration", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title(
        "Feature Expansion Comparison: Fuzzy GIMAN SAA Classification",
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(configs, ha="center")
    ax.set_ylim(0.0, 1.1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="gray")

    # Annotate values on bars
    for i in range(n):
        # AUC value
        ax.annotate(
            f"{aucs[i]:.3f}",
            xy=(x[i] - width / 2, aucs[i] + auc_err_high[i] + 0.02),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        # PR-AUC value
        ax.annotate(
            f"{pr_aucs[i]:.3f}",
            xy=(x[i] + width / 2, pr_aucs[i] + 0.02),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Highlight best config
    best_idx = np.argmax(aucs)
    ax.annotate(
        "Best",
        xy=(x[best_idx] - width / 2, aucs[best_idx] + auc_err_high[best_idx] + 0.06),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#FF5722",
        arrowprops=dict(arrowstyle="->", color="#FF5722", lw=1.5),
    )

    plt.tight_layout()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "Figure14_Feature_Expansion_Comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\nSaved: {output_path}")
    return output_path


def print_table(results: dict):
    """Print a comparison table for reference."""
    print("\n" + "=" * 80)
    print("FEATURE EXPANSION COMPARISON TABLE")
    print("=" * 80)
    header = f"{'Config':<25} {'AUC':>8} {'95% CI':>22} {'PR-AUC':>8} {'R@P80':>6}"
    print(header)
    print("-" * 80)

    for name, r in results.items():
        name_short = name.replace("\n", " ")
        auc = r["best_test_auc"]
        ci = r["auc_ci_95"]
        prauc = r["test_pr_auc"]
        rp80 = r.get("test_recall_at_precision_80", "N/A")
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        print(f"{name_short:<25} {auc:>8.4f} {ci_str:>22} {prauc:>8.4f} {rp80:>6}")

    print("=" * 80)


def main():
    print("=" * 70)
    print("GENERATING FIGURE 14: FEATURE EXPANSION COMPARISON")
    print("=" * 70)

    results = load_results()

    if len(results) < 4:
        print(f"WARNING: Only {len(results)}/4 configs found")

    print_table(results)
    output_path = generate_figure(results)

    print(f"\nFigure saved to: {output_path}")
    print("Ready for Overleaf upload.")


if __name__ == "__main__":
    main()
