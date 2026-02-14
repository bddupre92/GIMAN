"""Generate publication-quality figures for True GIMAN.

Creates 6 figures covering:
1. Architecture component contribution (progression bar chart)
2. Modality importance (dropout analysis)
3. Sensitivity to missingness
4. Training curves (loss + validation C-index)
5. Architecture investigation (capacity/structure variants)
6. Adaptive fusion gate analysis

Usage:
    python scripts/generate_true_giman_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parents[1]
results_dir = project_root / "results" / "true_giman"
figures_dir = results_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# Publication style
COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#EDC948", "#B07AA1"]
DPI = 350


def load_json(name: str) -> dict:
    return json.loads((results_dir / name).read_text())


def save_figure(fig, name: str):
    """Save figure as both PDF and PNG."""
    fig.savefig(figures_dir / f"{name}.pdf", dpi=DPI, bbox_inches="tight")
    fig.savefig(figures_dir / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.pdf + .png")


def style_axis(ax):
    """Apply publication styling to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


# ---------------------------------------------------------------------------
# Figure 1: Architecture Component Contribution
# ---------------------------------------------------------------------------


def fig1_architecture_progression():
    """Horizontal bar chart showing C-index as components are added."""
    print("Figure 1: Architecture Component Contribution")

    ablation_fc = load_json("ablation_results_full_clinical.json")
    ablation_af = load_json("ablation_adaptive_fusion.json")

    # Build the progression story
    components = [
        ("Simple MLP\n(no graph, no attention)", ablation_fc["simple_mlp"]),
        ("+ Graph Attention\n(no cross-modal)", ablation_fc["no_cross_modal"]),
        ("+ Static Cross-Modal\nAttention", ablation_fc["full_model"]),
        ("+ Strengthened\nAvail. Bias (10x)", ablation_af["C_phase1"]),
        ("+ Observed-Only\nMasking (t=0.3)", ablation_af["D_phase1_2"]),
        ("+ Adaptive Fusion\nGate", ablation_af["E_full_adaptive"]),
    ]

    labels = [c[0] for c in components]
    means = [c[1]["mean"] for c in components]
    ci_lows = [c[1]["ci_low"] for c in components]
    ci_highs = [c[1]["ci_high"] for c in components]
    errors_low = [m - lo for m, lo in zip(means, ci_lows, strict=False)]
    errors_high = [hi - m for m, hi in zip(means, ci_highs, strict=False)]

    # Color gradient from light to dark blue
    n = len(components)
    blues = plt.cm.Blues(np.linspace(0.3, 0.85, n))

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(n)

    bars = ax.barh(
        y_pos,
        means,
        xerr=[errors_low, errors_high],
        color=blues,
        edgecolor="white",
        linewidth=0.5,
        capsize=4,
        error_kw={"lw": 1.2, "color": "black"},
    )

    # Reference line at no-cross-modal baseline
    baseline = ablation_fc["no_cross_modal"]["mean"]
    ax.axvline(
        x=baseline,
        color="#E15759",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label=f"No-CM baseline ({baseline:.3f})",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Concordance Index (C-index)", fontsize=11)
    ax.set_title(
        "Architecture Component Contribution\n(Full Clinical: 34 features, 7 modalities)",
        fontsize=12,
        fontweight="bold",
    )

    # Add value labels
    for i, (m, lo, hi) in enumerate(zip(means, ci_lows, ci_highs, strict=False)):
        ax.text(m + 0.003, i, f"{m:.3f}", va="center", fontsize=8.5, fontweight="bold")

    ax.set_xlim(0.72, 0.85)
    ax.legend(loc="lower right", fontsize=9)
    style_axis(ax)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, "fig1_architecture_progression")


# ---------------------------------------------------------------------------
# Figure 2: Modality Importance
# ---------------------------------------------------------------------------


def fig2_modality_importance():
    """Lollipop chart showing delta C-index when each modality is dropped."""
    print("Figure 2: Modality Importance")

    data = load_json("ablation_results_full_clinical.json")
    full_c = data["full_model"]["mean"]

    # Known missingness rates from the data
    miss_rates = {
        "genetic": 12.0,
        "expanded_clinical": 0.9,
        "structural_imaging": 68.8,
        "csf_biomarkers": 78.1,
        "clinical_biomarkers": 6.1,
        "cortical_thickness": 68.7,
        "motor_cognitive": 2.0,
    }

    # Clean up modality names for display
    display_names = {
        "genetic": "Genetic",
        "expanded_clinical": "Expanded Clinical",
        "structural_imaging": "Structural Imaging",
        "csf_biomarkers": "CSF Biomarkers",
        "clinical_biomarkers": "Clinical Biomarkers",
        "cortical_thickness": "Cortical Thickness",
        "motor_cognitive": "Motor/Cognitive",
    }

    modalities = data["modality_dropout"]
    # Sort by delta (most negative = most important)
    deltas = [
        (
            m["dropped_modality"],
            m["mean"] - full_c,
            m["ci_low"],
            m["ci_high"],
            m["mean"],
        )
        for m in modalities
    ]
    deltas.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(deltas))

    for i, (mod, delta, ci_lo, ci_hi, mean) in enumerate(deltas):
        color = (
            "#E15759" if delta < -0.005 else ("#59A14F" if delta > 0.005 else "#999999")
        )
        ax.barh(i, delta, color=color, alpha=0.85, height=0.6, edgecolor="white")

        # Error bar (CI of the dropped model minus full model mean)
        err_lo = mean - ci_lo
        err_hi = ci_hi - mean
        ax.errorbar(
            delta,
            i,
            xerr=[[err_lo], [err_hi]],
            fmt="none",
            color="black",
            capsize=3,
            lw=1.0,
        )

        # Miss rate annotation
        miss = miss_rates.get(mod, 0)
        side = "left" if delta < 0 else "right"
        x_text = delta - 0.002 if delta < 0 else delta + 0.002
        ha = "right" if delta < 0 else "left"
        ax.text(
            x_text,
            i,
            f"{miss:.0f}% miss",
            va="center",
            ha=ha,
            fontsize=7.5,
            color="#666666",
            style="italic",
        )

    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([display_names.get(d[0], d[0]) for d in deltas], fontsize=10)
    ax.set_xlabel("$\\Delta$ C-index (relative to full model)", fontsize=11)
    ax.set_title(
        "Modality Importance via Dropout\n(Full Clinical config, mask-based ablation)",
        fontsize=12,
        fontweight="bold",
    )

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#E15759", alpha=0.85, label="Important (drop hurts)"),
        Patch(facecolor="#59A14F", alpha=0.85, label="Noise (drop helps)"),
        Patch(facecolor="#999999", alpha=0.85, label="Negligible"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8.5)

    style_axis(ax)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, "fig2_modality_importance")


# ---------------------------------------------------------------------------
# Figure 3: Sensitivity Analysis
# ---------------------------------------------------------------------------


def fig3_sensitivity_analysis():
    """Two-panel figure: missingness sensitivity + threshold sweep."""
    print("Figure 3: Sensitivity Analysis")

    sens = load_json("sensitivity_analysis.json")
    adapt = load_json("ablation_adaptive_fusion.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Panel A: Full vs Clinical-only vs Complete-case ---
    exps = sens["experiments"]
    labels_a = []
    means_a = []
    ci_lo_a = []
    ci_hi_a = []
    colors_a = []

    for exp in exps:
        if exp["mean"] < 0:
            continue  # Skip incomplete
        labels_a.append(exp["label"].replace(" (", "\n("))
        means_a.append(exp["mean"])
        ci_lo_a.append(exp["ci_low"])
        ci_hi_a.append(exp["ci_high"])

    colors_a = [COLORS[0], COLORS[1]]

    x = np.arange(len(labels_a))
    errors_lo = [m - lo for m, lo in zip(means_a, ci_lo_a, strict=False)]
    errors_hi = [hi - m for m, hi in zip(means_a, ci_hi_a, strict=False)]

    bars = ax1.bar(
        x,
        means_a,
        color=colors_a,
        alpha=0.9,
        width=0.5,
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.errorbar(
        x,
        means_a,
        yerr=[errors_lo, errors_hi],
        fmt="none",
        ecolor="black",
        capsize=5,
        lw=1.2,
    )

    # Add "Complete-case" as impossible
    ax1.bar(len(labels_a), 0.0, color="#dddddd", alpha=0.5, width=0.5)
    ax1.text(
        len(labels_a),
        0.4,
        "IMPOSSIBLE\n(0 patients with\nall modalities)",
        ha="center",
        va="center",
        fontsize=8,
        color="#888888",
        fontweight="bold",
    )

    all_labels = labels_a + ["Complete-case\n(all modalities)"]
    ax1.set_xticks(np.arange(len(all_labels)))
    ax1.set_xticklabels(all_labels, fontsize=8.5)
    ax1.set_ylabel("C-index", fontsize=11)
    ax1.set_title(
        "(A) Sensitivity to Modality Completeness", fontsize=11, fontweight="bold"
    )
    ax1.set_ylim(0.0, 0.9)

    # Value labels
    for i, m in enumerate(means_a):
        ax1.text(i, m + 0.015, f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")

    style_axis(ax1)

    # --- Panel B: Threshold sensitivity ---
    thresholds = sorted(adapt["threshold_sweep"].keys(), key=float)
    t_means = [adapt["threshold_sweep"][t]["mean"] for t in thresholds]
    t_ci_lo = [adapt["threshold_sweep"][t]["ci_low"] for t in thresholds]
    t_ci_hi = [adapt["threshold_sweep"][t]["ci_high"] for t in thresholds]
    t_err_lo = [m - lo for m, lo in zip(t_means, t_ci_lo, strict=False)]
    t_err_hi = [hi - m for m, hi in zip(t_means, t_ci_hi, strict=False)]

    x2 = np.arange(len(thresholds))
    bars2 = ax2.bar(
        x2,
        t_means,
        color=COLORS[4],
        alpha=0.85,
        width=0.5,
        edgecolor="white",
        linewidth=0.5,
    )
    ax2.errorbar(
        x2,
        t_means,
        yerr=[t_err_lo, t_err_hi],
        fmt="none",
        ecolor="black",
        capsize=5,
        lw=1.2,
    )

    # Highlight optimal
    best_idx = np.argmax(t_means)
    bars2[best_idx].set_color(COLORS[0])
    bars2[best_idx].set_alpha(1.0)

    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"t = {t}" for t in thresholds], fontsize=9)
    ax2.set_ylabel("C-index", fontsize=11)
    ax2.set_xlabel("Observed-Only Masking Threshold", fontsize=10)
    ax2.set_title(
        "(B) Threshold Sensitivity (Phase 1+2)", fontsize=11, fontweight="bold"
    )
    ax2.set_ylim(0.74, 0.86)

    for i, m in enumerate(t_means):
        ax2.text(i, m + 0.003, f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")

    style_axis(ax2)
    fig.tight_layout()
    save_figure(fig, "fig3_sensitivity_analysis")


# ---------------------------------------------------------------------------
# Figure 4: Training Curves
# ---------------------------------------------------------------------------


def fig4_training_curves():
    """Two-panel figure: training loss and validation C-index over epochs."""
    print("Figure 4: Training Curves")

    curves_path = results_dir / "training_curves.json"
    if not curves_path.exists():
        print(
            "  WARNING: training_curves.json not found. Run capture_training_curves.py first."
        )
        return

    curves = load_json("training_curves.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    fold_colors = [COLORS[0], COLORS[1], COLORS[2], COLORS[3], COLORS[4]]

    # Collect all fold histories
    max_epochs = 0
    fold_data = []
    for fold_key in sorted(curves.keys()):
        fold_info = curves[fold_key]
        history = fold_info["history"]
        epochs = [h["epoch"] for h in history]
        losses = [h["train_loss"] for h in history]
        c_indices = [h["val_c_index"] for h in history]
        max_epochs = max(max_epochs, len(epochs))
        fold_data.append(
            {
                "epochs": epochs,
                "losses": losses,
                "c_indices": c_indices,
                "final_epoch": fold_info["epochs_trained"],
            }
        )

    # --- Panel A: Training Loss ---
    for i, fd in enumerate(fold_data):
        ax1.plot(
            fd["epochs"],
            fd["losses"],
            color=fold_colors[i],
            alpha=0.5,
            linewidth=1.0,
            label=f"Fold {i + 1}",
        )
        # Mark early stopping
        ax1.axvline(
            x=fd["final_epoch"],
            color=fold_colors[i],
            linestyle=":",
            alpha=0.3,
            linewidth=0.8,
        )

    # Compute mean curve (pad shorter folds with NaN)
    all_losses = np.full((5, max_epochs), np.nan)
    for i, fd in enumerate(fold_data):
        all_losses[i, : len(fd["losses"])] = fd["losses"]
    mean_loss = np.nanmean(all_losses, axis=0)
    std_loss = np.nanstd(all_losses, axis=0)
    epoch_range = np.arange(1, max_epochs + 1)

    # Only plot mean where we have at least 2 folds
    valid_mask = np.sum(~np.isnan(all_losses), axis=0) >= 2
    ax1.plot(
        epoch_range[valid_mask],
        mean_loss[valid_mask],
        color="black",
        linewidth=2.0,
        label="Mean",
    )
    ax1.fill_between(
        epoch_range[valid_mask],
        mean_loss[valid_mask] - std_loss[valid_mask],
        mean_loss[valid_mask] + std_loss[valid_mask],
        color="black",
        alpha=0.1,
    )

    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Training Loss (Cox PL)", fontsize=11)
    ax1.set_title("(A) Training Loss", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, ncol=2, loc="upper right")
    style_axis(ax1)

    # --- Panel B: Validation C-index ---
    for i, fd in enumerate(fold_data):
        ax2.plot(
            fd["epochs"],
            fd["c_indices"],
            color=fold_colors[i],
            alpha=0.5,
            linewidth=1.0,
            label=f"Fold {i + 1} (final: {fd['c_indices'][-1]:.3f})",
        )
        ax2.axvline(
            x=fd["final_epoch"],
            color=fold_colors[i],
            linestyle=":",
            alpha=0.3,
            linewidth=0.8,
        )

    all_cidx = np.full((5, max_epochs), np.nan)
    for i, fd in enumerate(fold_data):
        all_cidx[i, : len(fd["c_indices"])] = fd["c_indices"]
    mean_cidx = np.nanmean(all_cidx, axis=0)
    std_cidx = np.nanstd(all_cidx, axis=0)

    valid_mask = np.sum(~np.isnan(all_cidx), axis=0) >= 2
    ax2.plot(
        epoch_range[valid_mask],
        mean_cidx[valid_mask],
        color="black",
        linewidth=2.0,
        label="Mean",
    )
    ax2.fill_between(
        epoch_range[valid_mask],
        mean_cidx[valid_mask] - std_cidx[valid_mask],
        mean_cidx[valid_mask] + std_cidx[valid_mask],
        color="black",
        alpha=0.1,
    )

    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Validation C-index", fontsize=11)
    ax2.set_title("(B) Validation Performance", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=7, ncol=2, loc="lower right")
    style_axis(ax2)

    fig.tight_layout()
    save_figure(fig, "fig4_training_curves")


# ---------------------------------------------------------------------------
# Figure 5: Architecture Investigation
# ---------------------------------------------------------------------------


def fig5_architecture_investigation():
    """Dot plot with error bars for 9 architecture variants."""
    print("Figure 5: Architecture Investigation")

    data = load_json("architecture_investigation.json")
    experiments = data["experiments"]

    # Clean labels
    short_labels = {
        "Reference: Full model": "Full Model\n(321K params)",
        "H1: Reduced capacity GIMAN": "Reduced\n(40K params)",
        "H1b: Minimal capacity GIMAN": "Minimal\n(10K params)",
        "H2a: k=5 graph": "k=5 Graph",
        "H2b: k=20 graph": "k=20 Graph",
        "H3: 1-layer GAT": "1-Layer GAT\n(147K params)",
        "H4: Dropout 0.5": "Dropout 0.5",
        "H5: Lower LR + patience": "Lower LR +\nPatience",
        "H6: Sweet spot config": "Sweet Spot\n(35K params)",
    }

    labels = [short_labels.get(e["label"], e["label"]) for e in experiments]
    means = [e["mean"] for e in experiments]
    ci_lows = [e["ci_low"] for e in experiments]
    ci_highs = [e["ci_high"] for e in experiments]
    params = [e["total_params"] for e in experiments]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(experiments))

    errors = [
        [m - lo for m, lo in zip(means, ci_lows, strict=False)],
        [hi - m for m, hi in zip(means, ci_highs, strict=False)],
    ]

    # Color by parameter count
    param_norm = np.array(params) / max(params)
    colors = plt.cm.YlOrBr(0.2 + 0.6 * param_norm)

    ax.barh(
        y_pos,
        means,
        xerr=errors,
        color=colors,
        alpha=0.85,
        height=0.6,
        capsize=4,
        error_kw={"lw": 1.2, "color": "black"},
        edgecolor="white",
        linewidth=0.5,
    )

    # Reference line
    ref_mean = experiments[0]["mean"]
    ax.axvline(
        x=ref_mean,
        color="#E15759",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label=f"Reference ({ref_mean:.3f})",
    )

    # Value labels with param count
    for i, (m, p) in enumerate(zip(means, params, strict=False)):
        ax.text(m + 0.003, i, f"{m:.3f}", va="center", fontsize=8, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Concordance Index (C-index)", fontsize=11)
    ax.set_title(
        "Architecture Investigation\n(Baseline: 30 features, 6 modalities, 188 events)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(0.70, 0.86)

    # Annotation
    ax.annotate(
        "Events-to-features ratio = 6.3\n(limits architectural differentiation)",
        xy=(0.72, 7.5),
        fontsize=8,
        color="#666666",
        style="italic",
    )

    ax.legend(loc="lower right", fontsize=9)
    style_axis(ax)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, "fig5_architecture_investigation")


# ---------------------------------------------------------------------------
# Figure 6: Gate Analysis
# ---------------------------------------------------------------------------


def fig6_gate_analysis():
    """Box/violin plot of gate values by number of observed modalities."""
    print("Figure 6: Adaptive Fusion Gate Analysis")

    data = load_json("gate_analysis.json")
    all_gate = data["all_gate_data"]

    # Group by modality count
    from collections import defaultdict

    by_count = defaultdict(list)
    for entry in all_gate:
        by_count[entry["n_modalities"]].append(entry["gate_value"])

    counts = sorted(by_count.keys())
    gate_data = [by_count[c] for c in counts]
    n_patients = [len(by_count[c]) for c in counts]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Violin plot
    parts = ax.violinplot(
        gate_data,
        positions=range(len(counts)),
        showmeans=True,
        showmedians=False,
        showextrema=False,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS[i % len(COLORS)])
        pc.set_alpha(0.7)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(1.5)

    # Add box plots inside
    bp = ax.boxplot(
        gate_data,
        positions=range(len(counts)),
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        zorder=3,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(0.8)
    for element in ["whiskers", "caps"]:
        for line in bp[element]:
            line.set_color("black")
            line.set_linewidth(0.8)
    for line in bp["medians"]:
        line.set_color("#E15759")
        line.set_linewidth(1.5)

    # Patient count annotations
    for i, (c, n) in enumerate(zip(counts, n_patients, strict=False)):
        ax.text(
            i,
            ax.get_ylim()[0] + 0.0002,
            f"n={n}",
            ha="center",
            fontsize=8,
            color="#666666",
        )

    # Mean gate annotations
    for i, c in enumerate(counts):
        mean_gate = np.mean(by_count[c])
        ax.text(
            i,
            max(by_count[c]) + 0.001,
            f"{mean_gate:.4f}",
            ha="center",
            fontsize=7.5,
            fontweight="bold",
            color=COLORS[0],
        )

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([f"{c}" for c in counts], fontsize=11)
    ax.set_xlabel("Number of Observed Modalities", fontsize=11)
    ax.set_ylabel("Fusion Gate Value\n(1 = cross-modal, 0 = concat)", fontsize=10)
    ax.set_title(
        "Adaptive Fusion Gate by Patient Modality Availability",
        fontsize=12,
        fontweight="bold",
    )

    # Reference line at 0.5
    ax.axhline(
        y=0.5,
        color="#999999",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
        label="Equal blend (0.5)",
    )
    ax.legend(loc="upper left", fontsize=9)

    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, "fig6_gate_analysis")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("\n" + "=" * 60)
    print("GENERATING PUBLICATION FIGURES — True GIMAN")
    print("=" * 60 + "\n")
    print(f"Output directory: {figures_dir}\n")

    fig1_architecture_progression()
    fig2_modality_importance()
    fig3_sensitivity_analysis()
    fig4_training_curves()
    fig5_architecture_investigation()
    fig6_gate_analysis()

    print(f"\nAll figures saved to: {figures_dir}")
    # List generated files
    files = sorted(figures_dir.glob("*"))
    print(f"Generated {len(files)} files:")
    for f in files:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
