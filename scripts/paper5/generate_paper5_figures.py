#!/usr/bin/env python3
"""
Paper 5: Generate 6 Publication-Quality Figures.

Reads results from outputs/paper5/ and produces:
    Fig 1 — Temporal learning curve (C-td vs training window size)
    Fig 2 — Performance degradation (random CV vs temporal C-td)
    Fig 3 — Covariate shift heatmap (KS statistic per feature x window)
    Fig 4 — Shift-importance interaction scatter
    Fig 5 — DeepHit vs Graph-DT degradation comparison
    Fig 6 — Per-transition temporal stability (C-td across windows)

Usage:
    python scripts/paper5/generate_paper5_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "outputs" / "paper5" / "temporal_validation"
SHIFT_DIR = ROOT / "outputs" / "paper5" / "covariate_shift"
FIGURE_DIR = ROOT / "outputs" / "paper5" / "figures"

# Paper 3 random CV references
PAPER3_DEEPHIT_CTD = 0.924
PAPER3_GRAPHDT_CTD = 0.904

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "deephit": "#2196F3",
    "graph_dt": "#FF5722",
    "reference": "#9E9E9E",
}


def load_temporal_results() -> dict | None:
    """Load temporal validation results JSON."""
    path = RESULTS_DIR / "temporal_validation_results.json"
    if not path.exists():
        print(f"  WARNING: {path} not found. Run run_temporal_validation.py first.")
        return None
    with open(path) as f:
        return json.load(f)


def load_shift_results() -> dict | None:
    """Load covariate shift results."""
    ks_path = SHIFT_DIR / "ks_tests_per_window.json"
    summary_path = SHIFT_DIR / "shift_summary.json"
    if not ks_path.exists():
        print(f"  WARNING: {ks_path} not found. Run run_covariate_shift.py first.")
        return None
    with open(ks_path) as f:
        ks_data = json.load(f)
    summary_data = None
    if summary_path.exists():
        with open(summary_path) as f:
            summary_data = json.load(f)
    return {"ks": ks_data, "summary": summary_data}


def _save_figure(fig, name: str):
    """Save figure as PNG and PDF."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIGURE_DIR / f"{name}.{ext}"
        fig.savefig(path, format=ext)
    print(f"  Saved: {FIGURE_DIR / name}.{{png,pdf}}")
    plt.close(fig)


# ── Figure 1: Temporal Learning Curve ────────────────────────────────

def fig1_learning_curve(results: dict):
    """C-td vs temporal window for both models.

    Uses window labels on x-axis (not training size) because W4 is a stress
    test with fewer training patients (950) than W3 (1520).
    """
    windows = sorted(k for k in results if k.startswith("W"))
    n_trains = [results[w]["n_train"] for w in windows]

    dh_ctds = [results[w]["deephit"]["c_td"] for w in windows]
    has_gdt = all("graph_dt" in results[w] for w in windows)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(windows))

    # Plot W1-W3 as expanding window (connected) + W4 as stress test (separate)
    # W1-W3 = expanding windows, W4 = 50/50 stress test
    expand_idx = [i for i, w in enumerate(windows) if w != "W4"]
    stress_idx = [i for i, w in enumerate(windows) if w == "W4"]

    # DeepHit
    ax.plot([x[i] for i in expand_idx], [dh_ctds[i] for i in expand_idx],
            "o-", color=COLORS["deephit"], linewidth=2, markersize=7,
            label="Dynamic-DeepHit")
    if stress_idx:
        ax.plot([x[i] for i in stress_idx], [dh_ctds[i] for i in stress_idx],
                "o", color=COLORS["deephit"], markersize=9, markerfacecolor="white",
                markeredgewidth=2)

    if has_gdt:
        gdt_ctds = [results[w]["graph_dt"]["c_td"] for w in windows]
        ax.plot([x[i] for i in expand_idx], [gdt_ctds[i] for i in expand_idx],
                "s-", color=COLORS["graph_dt"], linewidth=2, markersize=7,
                label="Graph-DT")
        if stress_idx:
            ax.plot([x[i] for i in stress_idx], [gdt_ctds[i] for i in stress_idx],
                    "s", color=COLORS["graph_dt"], markersize=9, markerfacecolor="white",
                    markeredgewidth=2)

    # Reference lines (Paper 3 random CV)
    ax.axhline(PAPER3_DEEPHIT_CTD, color=COLORS["deephit"], linestyle="--",
               alpha=0.4, label=f"DeepHit 5-CV ({PAPER3_DEEPHIT_CTD:.3f})")
    if has_gdt:
        ax.axhline(PAPER3_GRAPHDT_CTD, color=COLORS["graph_dt"], linestyle="--",
                    alpha=0.4, label=f"Graph-DT 5-CV ({PAPER3_GRAPHDT_CTD:.3f})")

    ax.set_xlabel("Temporal Window")
    ax.set_ylabel("Time-Dependent Concordance Index (C-td)")
    ax.set_title("Temporal Validation: Learning Curve")
    ax.legend(loc="lower left", fontsize=8)

    # Dynamic y-axis to accommodate all data points
    all_ctds = dh_ctds + (gdt_ctds if has_gdt else [])
    ymin = max(0.5, min(all_ctds) - 0.05)
    ax.set_ylim(ymin, 1.0)
    ax.grid(True, alpha=0.3)

    # Window labels with training size
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w}\n(n={n_trains[i]})" for i, w in enumerate(windows)],
                        fontsize=8)

    # Annotate W4 as stress test
    if stress_idx:
        si = stress_idx[0]
        ax.annotate("50/50\nstress test",
                    (x[si], min(dh_ctds[si], gdt_ctds[si] if has_gdt else dh_ctds[si])),
                    textcoords="offset points", xytext=(30, -15),
                    ha="center", fontsize=7, color="gray",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    _save_figure(fig, "fig1_temporal_learning_curve")


# ── Figure 2: Performance Degradation ────────────────────────────────

def fig2_degradation(results: dict):
    """Grouped bar: random CV C-td vs per-window temporal C-td."""
    windows = sorted(k for k in results if k.startswith("W"))
    has_gdt = all("graph_dt" in results[w] for w in windows)

    dh_temporal = [results[w]["deephit"]["c_td"] for w in windows]

    labels = windows + ["Paper 3\n(5-CV)"]
    x = np.arange(len(labels))

    if has_gdt:
        gdt_temporal = [results[w]["graph_dt"]["c_td"] for w in windows]
        width = 0.35

        fig, ax = plt.subplots(figsize=(7, 4.5))
        dh_vals = dh_temporal + [PAPER3_DEEPHIT_CTD]
        gdt_vals = gdt_temporal + [PAPER3_GRAPHDT_CTD]

        bars1 = ax.bar(x - width / 2, dh_vals, width, color=COLORS["deephit"],
                        label="Dynamic-DeepHit", alpha=0.85)
        bars2 = ax.bar(x + width / 2, gdt_vals, width, color=COLORS["graph_dt"],
                        label="Graph-DT", alpha=0.85)

        # Add value labels
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    else:
        width = 0.5
        fig, ax = plt.subplots(figsize=(6, 4.5))
        dh_vals = dh_temporal + [PAPER3_DEEPHIT_CTD]
        bars1 = ax.bar(x, dh_vals, width, color=COLORS["deephit"],
                        label="Dynamic-DeepHit", alpha=0.85)
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Temporal Window")
    ax.set_ylabel("C-td")
    ax.set_title("Performance: Temporal Validation vs Random Cross-Validation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # Dynamic y-axis to accommodate W4's low C-td
    all_vals = dh_vals + (gdt_vals if has_gdt else [])
    ymin = max(0.5, min(all_vals) - 0.05)
    ax.set_ylim(ymin, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Vertical line separating temporal from CV
    ax.axvline(len(windows) - 0.5, color="gray", linestyle=":", alpha=0.5)

    _save_figure(fig, "fig2_performance_degradation")


# ── Figure 3: Covariate Shift Heatmap ────────────────────────────────

def fig3_shift_heatmap(shift_data: dict):
    """KS statistic per feature x 4 temporal windows."""
    ks_data = shift_data["ks"]
    windows = sorted(ks_data.keys())

    # Collect all features
    features = [r["feature"] for r in ks_data[windows[0]]]
    n_features = len(features)
    n_windows = len(windows)

    # Build matrix: features x windows
    ks_matrix = np.zeros((n_features, n_windows))
    shifted_matrix = np.zeros((n_features, n_windows), dtype=bool)

    for wi, wname in enumerate(windows):
        for fi, feat_result in enumerate(ks_data[wname]):
            ks_val = feat_result["ks_statistic"]
            ks_matrix[fi, wi] = ks_val if ks_val is not None and not np.isnan(ks_val) else 0.0
            shifted_matrix[fi, wi] = feat_result["shifted"]

    fig, ax = plt.subplots(figsize=(5, max(6, n_features * 0.35)))

    im = ax.imshow(ks_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.3)

    ax.set_xticks(range(n_windows))
    ax.set_xticklabels(windows)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(features, fontsize=7)

    # Mark shifted cells with asterisk
    for fi in range(n_features):
        for wi in range(n_windows):
            if shifted_matrix[fi, wi]:
                ax.text(wi, fi, "*", ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")

    ax.set_xlabel("Temporal Window")
    ax.set_title("Covariate Shift: KS Statistic per Feature")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("KS Statistic")

    fig.tight_layout()
    _save_figure(fig, "fig3_covariate_shift_heatmap")


# ── Figure 4: Shift-Importance Interaction ───────────────────────────

def fig4_shift_importance(shift_data: dict, results: dict):
    """Scatter: mean KS statistic (x) vs feature name, highlighting shifted features.

    NOTE: True feature importance requires gradient-based or permutation
    analysis on DeepHit. For now, we plot mean KS across windows per feature
    as a proxy for shift magnitude, ranked.
    """
    ks_data = shift_data["ks"]
    windows = sorted(ks_data.keys())
    features = [r["feature"] for r in ks_data[windows[0]]]

    # Mean KS across windows per feature
    mean_ks = []
    any_shifted = []
    for fi, feat in enumerate(features):
        ks_vals = []
        shifted = False
        for wname in windows:
            ks_val = ks_data[wname][fi]["ks_statistic"]
            if ks_val is not None and not np.isnan(ks_val):
                ks_vals.append(ks_val)
            if ks_data[wname][fi]["shifted"]:
                shifted = True
        mean_ks.append(np.mean(ks_vals) if ks_vals else 0.0)
        any_shifted.append(shifted)

    # Sort by mean KS descending
    order = np.argsort(mean_ks)[::-1]

    fig, ax = plt.subplots(figsize=(6, max(5, len(features) * 0.3)))

    y_pos = np.arange(len(features))
    colors = ["#FF5722" if any_shifted[i] else "#2196F3" for i in order]

    ax.barh(y_pos, [mean_ks[i] for i in order], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([features[i] for i in order], fontsize=7)
    ax.set_xlabel("Mean KS Statistic (across 4 windows)")
    ax.set_title("Feature Shift Magnitude")

    # Legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="#FF5722", alpha=0.8, label="Shifted (p<0.001 or PSI>0.25)"),
            Patch(facecolor="#2196F3", alpha=0.8, label="Not shifted"),
        ],
        loc="lower right", fontsize=8,
    )

    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    _save_figure(fig, "fig4_shift_importance_interaction")


# ── Figure 5: DeepHit vs Graph-DT Degradation ───────────────────────

def fig5_model_degradation(results: dict):
    """Compare degradation patterns: DeepHit vs Graph-DT."""
    windows = sorted(k for k in results if k.startswith("W"))
    has_gdt = all("graph_dt" in results[w] for w in windows)

    if not has_gdt:
        print("  Skipping fig5 — no Graph-DT results.")
        return

    dh_deg = [results[w]["deephit"]["degradation_from_cv"] for w in windows]
    gdt_deg = [results[w]["graph_dt"]["degradation_from_cv"] for w in windows]

    x = np.arange(len(windows))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 4))

    bars1 = ax.bar(x - width / 2, dh_deg, width, color=COLORS["deephit"],
                    label="Dynamic-DeepHit", alpha=0.85)
    bars2 = ax.bar(x + width / 2, gdt_deg, width, color=COLORS["graph_dt"],
                    label="Graph-DT", alpha=0.85)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        sign = "+" if h > 0 else ""
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.002,
                f"{sign}{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Temporal Window")
    ax.set_ylabel("C-td Degradation from Random CV")
    ax.set_title("Performance Degradation: Temporal vs Random Split")
    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    _save_figure(fig, "fig5_model_degradation_comparison")


# ── Figure 6: Per-Transition Temporal Stability ──────────────────────

def fig6_per_transition_stability(results: dict):
    """Per-transition C-td across 4 windows (line plot)."""
    windows = sorted(k for k in results if k.startswith("W"))

    # Collect all transition keys
    all_transitions = set()
    for w in windows:
        if "deephit" in results[w]:
            all_transitions.update(results[w]["deephit"]["per_transition_ctd"].keys())

    if not all_transitions:
        print("  Skipping fig6 — no per-transition data.")
        return

    transitions = sorted(all_transitions)
    colors_list = plt.cm.Set1(np.linspace(0, 1, max(len(transitions), 1)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    has_gdt = all("graph_dt" in results[w] for w in windows)

    for ax_idx, (model_key, title) in enumerate([
        ("deephit", "Dynamic-DeepHit"),
        ("graph_dt", "Graph-DT"),
    ]):
        if ax_idx == 1 and not has_gdt:
            axes[1].set_visible(False)
            break

        ax = axes[ax_idx]
        for ti, trans in enumerate(transitions):
            ctds = []
            for w in windows:
                if model_key in results[w]:
                    val = results[w][model_key]["per_transition_ctd"].get(trans)
                    ctds.append(val if val is not None else np.nan)
                else:
                    ctds.append(np.nan)

            # Skip transitions that are all NaN
            if all(np.isnan(c) for c in ctds):
                continue

            ax.plot(range(len(windows)), ctds, "o-",
                    color=colors_list[ti], label=trans, linewidth=1.5, markersize=5)

        ax.set_xlabel("Temporal Window")
        ax.set_xticks(range(len(windows)))
        ax.set_xticklabels(windows)
        ax.set_title(title)
        ax.set_ylim(0.3, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")

    axes[0].set_ylabel("Per-Transition C-td")
    fig.suptitle("Per-Transition C-td Stability Across Temporal Windows", y=1.02)
    fig.tight_layout()
    _save_figure(fig, "fig6_per_transition_stability")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    print("Loading results...")
    results = load_temporal_results()
    shift_data = load_shift_results()

    figures_generated = 0

    if results is not None:
        print("\nGenerating Figure 1: Temporal Learning Curve...")
        fig1_learning_curve(results)
        figures_generated += 1

        print("Generating Figure 2: Performance Degradation...")
        fig2_degradation(results)
        figures_generated += 1

    if shift_data is not None:
        print("Generating Figure 3: Covariate Shift Heatmap...")
        fig3_shift_heatmap(shift_data)
        figures_generated += 1

        if results is not None:
            print("Generating Figure 4: Shift-Importance Interaction...")
            fig4_shift_importance(shift_data, results)
            figures_generated += 1

    if results is not None:
        print("Generating Figure 5: Model Degradation Comparison...")
        fig5_model_degradation(results)
        figures_generated += 1

        print("Generating Figure 6: Per-Transition Stability...")
        fig6_per_transition_stability(results)
        figures_generated += 1

    print(f"\nGenerated {figures_generated} figures in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
