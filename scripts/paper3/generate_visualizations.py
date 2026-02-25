#!/usr/bin/env python3
"""
Paper 3 Visualization Suite: Graph-Informed Digital Twins for NSD-ISS Transitions.

Generates publication-quality figures demonstrating Graph-DT's novelty:
  1. Patient similarity graph (colored by stage)
  2. Transition matrix heatmap
  3. Individual patient trajectory spaghetti plots
  4. Model comparison bar charts (C-td, IBS)
  5. Per-transition C-td comparison (Graph-DT vs DeepHit)
  6. Brier score at horizons comparison
  7. Sojourn time comparison (Markov vs Simuni 2025)
  8. Stage distribution over time
  9. "Patients Like You" trajectory overlay (from graph neighbors)
  10. Node embeddings colored by stage (UMAP/t-SNE)
  11. Gate activation analysis (requires one training fold)
  12. Individual CIF predictions (requires one training fold)

Usage:
    python scripts/paper3/generate_visualizations.py
    python scripts/paper3/generate_visualizations.py --skip-training   # static figs only
    python scripts/paper3/generate_visualizations.py --training-only   # model figs only

Outputs:  outputs/paper3_figures/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper3_figures"
FEATURES_PATH = DATA_DIR / "07_paper3_features" / "longitudinal_features.csv"
TRANSITIONS_PATH = DATA_DIR / "06_longitudinal_staging" / "transition_events.csv"
BENCHMARK_PATH = PROJECT_ROOT / "outputs" / "paper3_benchmark" / "benchmark_summary.json"
MARKOV_PATH = PROJECT_ROOT / "outputs" / "paper3_markov" / "markov_results.json"
DEEPHIT_PATH = PROJECT_ROOT / "outputs" / "paper3_deephit" / "deephit_results.json"
GRAPHDT_PATH = PROJECT_ROOT / "outputs" / "paper3_graph_dt" / "graph_dt_results.json"

# Stage labels and colors (consistent across all figures)
STAGE_LABELS = ["0", "1", "2B", "3", "4", "5", "6"]
STAGE_NUMERIC = [0, 1, 2.5, 3, 4, 5, 6]
STAGE_COLORS = {
    "0": "#2ecc71",   # green — no markers
    "1": "#3498db",   # blue — early biological
    "2B": "#f39c12",  # orange — clinical onset
    "3": "#e74c3c",   # red — mild impairment
    "4": "#9b59b6",   # purple — moderate
    "5": "#34495e",   # dark grey — severe
    "6": "#1a1a2e",   # near-black — very severe
}

# Matplotlib style for publication
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def load_data():
    """Load all data needed for visualizations."""
    print("Loading data...")
    features = pd.read_csv(FEATURES_PATH, low_memory=False)
    transitions = pd.read_csv(TRANSITIONS_PATH, low_memory=False)

    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)
    with open(MARKOV_PATH) as f:
        markov = json.load(f)
    with open(DEEPHIT_PATH) as f:
        deephit = json.load(f)
    with open(GRAPHDT_PATH) as f:
        graphdt = json.load(f)

    print(f"  Features: {len(features)} obs, {features['PATNO'].nunique()} patients")
    print(f"  Transitions: {len(transitions)}")
    return features, transitions, benchmark, markov, deephit, graphdt


# =====================================================================
# FIGURE 1: Patient Similarity Graph
# =====================================================================

def fig_patient_graph(features, output_dir):
    """Visualize the kNN patient similarity graph colored by baseline NSD-ISS stage."""
    import networkx as nx

    print("\n[Fig 1] Patient similarity graph...")

    # Get baseline data
    baseline = features[features["months_from_baseline"] == 0.0].copy()
    baseline = baseline.drop_duplicates(subset="PATNO", keep="first")
    baseline = baseline[baseline["nsd_stage"].notna()]

    # Build kNN graph (same logic as model)
    from giman_pipeline.paper3.graph_digital_twin import GRAPH_FEATURES, build_patient_graph

    patient_ids = sorted(baseline["PATNO"].unique().tolist())
    edge_index, edge_weight, node_baseline = build_patient_graph(
        features, patient_ids, k_neighbors=15
    )
    ei = edge_index.numpy()
    ew = edge_weight.numpy()

    # Map patient index to baseline stage
    pat_to_idx = {p: i for i, p in enumerate(patient_ids)}
    pat_stages = {}
    for _, row in baseline.iterrows():
        pat = int(row["PATNO"])
        if pat in pat_to_idx:
            pat_stages[pat_to_idx[pat]] = str(row["nsd_stage"])

    # Build networkx graph
    G = nx.Graph()
    G.add_nodes_from(range(len(patient_ids)))
    for e in range(ei.shape[1]):
        s, d = int(ei[0, e]), int(ei[1, e])
        if s < d:  # avoid duplicates
            G.add_edge(s, d, weight=float(ew[e]))

    # Node colors by stage
    node_colors = [STAGE_COLORS.get(pat_stages.get(i, "0"), "#cccccc")
                   for i in range(len(patient_ids))]

    # Stage counts
    stage_counts = {}
    for idx, stg in pat_stages.items():
        stage_counts[stg] = stage_counts.get(stg, 0) + 1

    # Layout — spring layout with moderate iterations
    print("  Computing spring layout (may take ~30s for 1900 nodes)...")
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42, weight="weight")

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Draw edges (very faint)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.02, edge_color="#cccccc",
                           width=0.3)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=15, node_color=node_colors,
                           edgecolors="none", alpha=0.8)

    # Legend
    legend_patches = []
    for stage in STAGE_LABELS:
        n = stage_counts.get(stage, 0)
        if n > 0:
            legend_patches.append(
                mpatches.Patch(color=STAGE_COLORS[stage],
                               label=f"Stage {stage} (n={n})")
            )
    ax.legend(handles=legend_patches, loc="upper left", framealpha=0.9,
              fontsize=10, title="NSD-ISS Stage", title_fontsize=11)

    ax.set_title("Patient Similarity Graph (k=15 Nearest Neighbors)\n"
                 "Colored by Baseline NSD-ISS Stage", fontsize=14)
    ax.axis("off")

    fig.savefig(output_dir / "fig1_patient_similarity_graph.png")
    fig.savefig(output_dir / "fig1_patient_similarity_graph.pdf")
    plt.close(fig)
    print(f"  Saved: fig1_patient_similarity_graph.png/pdf")

    # Also save a zoomed version showing inter-stage connectivity
    return G, pos, pat_stages, patient_ids


# =====================================================================
# FIGURE 2: Transition Matrix Heatmap
# =====================================================================

def fig_transition_matrix(benchmark, output_dir):
    """Heatmap of observed stage transitions."""
    print("\n[Fig 2] Transition matrix heatmap...")

    tm = benchmark["cohort"]["cohort"]["transition_matrix"]
    stages = ["0", "1", "2B", "3", "4", "5", "6"]
    matrix = np.zeros((len(stages), len(stages)), dtype=int)
    for i, s_from in enumerate(stages):
        for j, s_to in enumerate(stages):
            matrix[i, j] = tm.get(s_from, {}).get(s_to, 0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Use log scale for the colormap since values vary greatly
    mask = matrix == 0
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=stages, yticklabels=stages,
        ax=ax, mask=mask, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Number of Transitions"},
        annot_kws={"size": 11},
    )
    # Add zeros for masked cells
    for i in range(len(stages)):
        for j in range(len(stages)):
            if matrix[i, j] == 0:
                ax.text(j + 0.5, i + 0.5, "0", ha="center", va="center",
                        fontsize=9, color="#cccccc")

    ax.set_xlabel("Destination Stage", fontsize=12)
    ax.set_ylabel("Source Stage", fontsize=12)
    ax.set_title("Observed NSD-ISS Stage Transitions (N=2,859)", fontsize=14)

    # Add annotations for forward vs backward
    total_fwd = benchmark["cohort"]["forward_transitions"]
    total_bwd = benchmark["cohort"]["backward_transitions"]
    ax.text(0.02, -0.08,
            f"Forward: {total_fwd} ({total_fwd/(total_fwd+total_bwd)*100:.1f}%)  |  "
            f"Backward: {total_bwd} ({total_bwd/(total_fwd+total_bwd)*100:.1f}%)",
            transform=ax.transAxes, fontsize=10, style="italic")

    fig.tight_layout()
    fig.savefig(output_dir / "fig2_transition_matrix.png")
    fig.savefig(output_dir / "fig2_transition_matrix.pdf")
    plt.close(fig)
    print(f"  Saved: fig2_transition_matrix.png/pdf")


# =====================================================================
# FIGURE 3: Individual Patient Trajectories (Spaghetti Plot)
# =====================================================================

def fig_patient_trajectories(features, output_dir):
    """Spaghetti plot of individual patient stage trajectories over time."""
    print("\n[Fig 3] Patient trajectory spaghetti plot...")

    # Find patients with interesting trajectories
    # Progressive: goes from 2B → 3 → 4
    # Regressive: goes backward (4 → 3 or 3 → 2B)
    # Stable: stays at same stage

    patients = features.groupby("PATNO").agg(
        n_visits=("EVENT_ID", "count"),
        stages=("nsd_stage_numeric", lambda x: list(x.dropna())),
        times=("months_from_baseline", lambda x: list(x)),
        first_stage=("nsd_stage_numeric", "first"),
        last_stage=("nsd_stage_numeric", "last"),
    ).reset_index()

    # Filter to patients with enough data
    patients = patients[patients["n_visits"] >= 4]
    patients["stage_change"] = patients["last_stage"] - patients["first_stage"]
    patients["n_unique_stages"] = patients["stages"].apply(lambda x: len(set(x)))

    # Select examples
    progressors = patients[(patients["stage_change"] > 0) & (patients["n_unique_stages"] >= 3)]
    regressors = patients[(patients["stage_change"] < 0) & (patients["n_unique_stages"] >= 2)]
    stable = patients[(patients["stage_change"] == 0) & (patients["n_visits"] >= 6)]
    oscillators = patients[patients["n_unique_stages"] >= 3]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: All patients (random subset, faded)
    ax = axes[0, 0]
    rng = np.random.RandomState(42)
    sample_pats = rng.choice(patients["PATNO"].values, size=min(200, len(patients)), replace=False)
    for pat in sample_pats:
        pat_data = features[features["PATNO"] == pat].sort_values("months_from_baseline")
        times = pat_data["months_from_baseline"].values
        stages = pat_data["nsd_stage_numeric"].values
        valid = ~np.isnan(stages)
        if valid.sum() >= 2:
            ax.plot(times[valid] / 12, stages[valid], alpha=0.08, linewidth=0.5,
                    color="#3498db")
    ax.set_ylabel("NSD-ISS Stage")
    ax.set_xlabel("Years from Baseline")
    ax.set_title("A. Population Overview (n=200 random)")
    ax.set_yticks(STAGE_NUMERIC)
    ax.set_yticklabels(STAGE_LABELS)
    ax.set_xlim(-0.5, 15)
    ax.grid(True, alpha=0.3)

    # Panel B: Progressive patients
    ax = axes[0, 1]
    if len(progressors) > 0:
        for _, row in progressors.head(8).iterrows():
            pat_data = features[features["PATNO"] == row["PATNO"]].sort_values("months_from_baseline")
            times = pat_data["months_from_baseline"].values
            stages = pat_data["nsd_stage_numeric"].values
            valid = ~np.isnan(stages)
            if valid.sum() >= 2:
                ax.plot(times[valid] / 12, stages[valid], alpha=0.7, linewidth=1.5,
                        marker="o", markersize=3)
    ax.set_ylabel("NSD-ISS Stage")
    ax.set_xlabel("Years from Baseline")
    ax.set_title("B. Progressive Patients")
    ax.set_yticks(STAGE_NUMERIC)
    ax.set_yticklabels(STAGE_LABELS)
    ax.set_xlim(-0.5, 15)
    ax.grid(True, alpha=0.3)

    # Panel C: Regressive patients
    ax = axes[1, 0]
    if len(regressors) > 0:
        for _, row in regressors.head(8).iterrows():
            pat_data = features[features["PATNO"] == row["PATNO"]].sort_values("months_from_baseline")
            times = pat_data["months_from_baseline"].values
            stages = pat_data["nsd_stage_numeric"].values
            valid = ~np.isnan(stages)
            if valid.sum() >= 2:
                ax.plot(times[valid] / 12, stages[valid], alpha=0.7, linewidth=1.5,
                        marker="o", markersize=3)
    ax.set_ylabel("NSD-ISS Stage")
    ax.set_xlabel("Years from Baseline")
    ax.set_title(f"C. Regressive Patients (treatment-driven)")
    ax.set_yticks(STAGE_NUMERIC)
    ax.set_yticklabels(STAGE_LABELS)
    ax.set_xlim(-0.5, 15)
    ax.grid(True, alpha=0.3)

    # Panel D: Oscillating patients (forward + backward)
    ax = axes[1, 1]
    osc = oscillators[(oscillators["n_unique_stages"] >= 3) & (oscillators["n_visits"] >= 6)]
    if len(osc) > 0:
        for _, row in osc.head(8).iterrows():
            pat_data = features[features["PATNO"] == row["PATNO"]].sort_values("months_from_baseline")
            times = pat_data["months_from_baseline"].values
            stages = pat_data["nsd_stage_numeric"].values
            valid = ~np.isnan(stages)
            if valid.sum() >= 2:
                ax.plot(times[valid] / 12, stages[valid], alpha=0.7, linewidth=1.5,
                        marker="o", markersize=3)
    ax.set_ylabel("NSD-ISS Stage")
    ax.set_xlabel("Years from Baseline")
    ax.set_title("D. Oscillating Patients (bidirectional)")
    ax.set_yticks(STAGE_NUMERIC)
    ax.set_yticklabels(STAGE_LABELS)
    ax.set_xlim(-0.5, 15)
    ax.grid(True, alpha=0.3)

    fig.suptitle("NSD-ISS Stage Trajectories in PPMI (N=1,900 Patients)", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_patient_trajectories.png")
    fig.savefig(output_dir / "fig3_patient_trajectories.pdf")
    plt.close(fig)
    print(f"  Saved: fig3_patient_trajectories.png/pdf")


# =====================================================================
# FIGURE 4: Model Comparison (C-td, IBS with error bars)
# =====================================================================

def fig_model_comparison(deephit, graphdt, output_dir):
    """Bar chart comparing DeepHit vs Graph-DT with per-fold error bars."""
    print("\n[Fig 4] Model comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = ["DeepHit", "Graph-DT"]
    colors = ["#3498db", "#e74c3c"]

    # C-td comparison
    ax = axes[0]
    ctds = [deephit["c_td"], graphdt["c_td"]]
    stds = [deephit["c_td_std"], graphdt["c_td_std"]]
    bars = ax.bar(models, ctds, yerr=stds, color=colors, alpha=0.85,
                  capsize=8, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Time-Dependent Concordance (C-td)")
    ax.set_title("A. Discriminative Performance")
    ax.set_ylim(0.88, 0.96)
    # Add fold dots
    for i, (folds, color) in enumerate(zip(
        [deephit["c_td_per_fold"], graphdt["c_td_per_fold"]], colors
    )):
        x_jitter = np.random.RandomState(42).normal(0, 0.04, len(folds))
        ax.scatter([i + xj for xj in x_jitter], folds, color=color,
                   edgecolor="black", linewidth=0.5, s=40, zorder=5, alpha=0.7)
    ax.axhline(y=0.5, color="#cccccc", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.text(0.5, 0.02, "p = 0.108 (paired t-test)\nNot statistically significant",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic",
            color="#666666")
    ax.grid(True, axis="y", alpha=0.3)

    # IBS comparison
    ax = axes[1]
    ibs_vals = [deephit["ibs"], graphdt["ibs"]]
    ibs_stds = [deephit["ibs_std"], graphdt["ibs_std"]]
    bars = ax.bar(models, ibs_vals, yerr=ibs_stds, color=colors, alpha=0.85,
                  capsize=8, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Integrated Brier Score (IBS)")
    ax.set_title("B. Calibration Performance")
    ax.set_ylim(0.0, 0.010)
    for i, (folds, color) in enumerate(zip(
        [deephit["ibs_per_fold"], graphdt["ibs_per_fold"]], colors
    )):
        x_jitter = np.random.RandomState(42).normal(0, 0.04, len(folds))
        ax.scatter([i + xj for xj in x_jitter], folds, color=color,
                   edgecolor="black", linewidth=0.5, s=40, zorder=5, alpha=0.7)
    ax.text(0.5, 0.95, "Lower is better",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic",
            va="top", color="#666666")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Deep Learning Model Comparison (5-Fold CV)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_model_comparison.png")
    fig.savefig(output_dir / "fig4_model_comparison.pdf")
    plt.close(fig)
    print(f"  Saved: fig4_model_comparison.png/pdf")


# =====================================================================
# FIGURE 5: Per-Transition C-td Comparison
# =====================================================================

def fig_per_transition_ctd(deephit, graphdt, output_dir):
    """Grouped bar chart comparing per-transition C-td."""
    print("\n[Fig 5] Per-transition C-td comparison...")

    dh_trans = deephit.get("per_transition_ctd", {})
    gdt_trans = graphdt.get("per_transition_ctd", {})

    # Filter to transitions with meaningful sample sizes
    key_transitions = ["→0", "→2B", "→3", "→4", "→5"]
    labels = []
    dh_vals = []
    gdt_vals = []
    for t in key_transitions:
        dh_v = dh_trans.get(t)
        gdt_v = gdt_trans.get(t)
        if dh_v is not None and gdt_v is not None:
            labels.append(t)
            dh_vals.append(dh_v)
            gdt_vals.append(gdt_v)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bars1 = ax.bar(x - width/2, dh_vals, width, label="DeepHit",
                   color="#3498db", alpha=0.85, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, gdt_vals, width, label="Graph-DT",
                   color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("C-td (Time-Dependent Concordance)")
    ax.set_xlabel("Transition Destination")
    ax.set_title("Per-Transition Discriminative Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.5, 1.05)
    ax.axhline(y=0.5, color="#cccccc", linestyle="--", linewidth=1, label="Random")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate key transitions
    stage_descriptions = {
        "→0": "Regression\nto preclinical",
        "→2B": "Onset of\nclinical PD",
        "→3": "Mild\nimpairment",
        "→4": "Moderate\nimpairment",
        "→5": "Severe\nimpairment",
    }
    for i, label in enumerate(labels):
        if label in stage_descriptions:
            ax.text(i, 0.48, stage_descriptions[label], ha="center",
                    fontsize=8, style="italic", color="#888888")

    fig.tight_layout()
    fig.savefig(output_dir / "fig5_per_transition_ctd.png")
    fig.savefig(output_dir / "fig5_per_transition_ctd.pdf")
    plt.close(fig)
    print(f"  Saved: fig5_per_transition_ctd.png/pdf")


# =====================================================================
# FIGURE 6: Brier Score at Horizons
# =====================================================================

def fig_brier_horizons(deephit, graphdt, output_dir):
    """Line plot of Brier scores at different prediction horizons."""
    print("\n[Fig 6] Brier score at horizons...")

    horizons = ["1yr", "2yr", "5yr", "10yr"]
    years = [1, 2, 5, 10]

    dh_brier = [deephit["brier_at_horizons"][h] for h in horizons]
    gdt_brier = [graphdt["brier_at_horizons"][h] for h in horizons]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(years, dh_brier, "o-", color="#3498db", linewidth=2, markersize=8,
            label="DeepHit", alpha=0.85)
    ax.plot(years, gdt_brier, "s-", color="#e74c3c", linewidth=2, markersize=8,
            label="Graph-DT", alpha=0.85)

    ax.set_xlabel("Prediction Horizon (years)")
    ax.set_ylabel("Brier Score (lower is better)")
    ax.set_title("Calibration at Different Prediction Horizons")
    ax.legend(fontsize=11)
    ax.set_xticks(years)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(max(dh_brier), max(gdt_brier)) * 1.5)

    fig.tight_layout()
    fig.savefig(output_dir / "fig6_brier_horizons.png")
    fig.savefig(output_dir / "fig6_brier_horizons.pdf")
    plt.close(fig)
    print(f"  Saved: fig6_brier_horizons.png/pdf")


# =====================================================================
# FIGURE 7: Sojourn Time Comparison
# =====================================================================

def fig_sojourn_times(markov, output_dir):
    """Bar chart comparing Markov sojourn times vs Simuni 2025 KM."""
    print("\n[Fig 7] Sojourn time comparison...")

    stages = ["2B", "3", "4"]
    markov_sojourn = [markov["sojourn_times"][s] for s in stages]

    # Simuni 2025 KM median transition times
    simuni_km = [1.19, 4.98, 9.77]  # 2B→3, 3→4, 4→5

    # Our KM (from benchmark)
    our_km = [0.6, 5.7, 10.0]  # from benchmark_summary.json kaplan_meier

    x = np.arange(len(stages))
    width = 0.25

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    bars1 = ax.bar(x - width, markov_sojourn, width, label="Markov Sojourn Time",
                   color="#3498db", alpha=0.85, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, simuni_km, width, label="Simuni 2025 KM Median",
                   color="#2ecc71", alpha=0.85, edgecolor="black", linewidth=0.5)
    bars3 = ax.bar(x + width, our_km, width, label="Our KM Median",
                   color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.5)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.15,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Time (years)")
    ax.set_xlabel("NSD-ISS Stage Transition")
    ax.set_title("Stage Transition Timing: Markov vs Kaplan-Meier")
    ax.set_xticks(x)
    ax.set_xticklabels(["2B → 3", "3 → 4", "4 → 5"], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Add CIs for Markov
    if "bootstrap_ci" in markov:
        ci = markov["bootstrap_ci"]["sojourn_ci"]
        for i, s in enumerate(stages):
            lo = ci[s]["ci_lower"]
            hi = ci[s]["ci_upper"]
            ax.errorbar(x[i] - width, markov_sojourn[i],
                        yerr=[[markov_sojourn[i] - lo], [hi - markov_sojourn[i]]],
                        fmt="none", color="black", capsize=4)

    fig.tight_layout()
    fig.savefig(output_dir / "fig7_sojourn_comparison.png")
    fig.savefig(output_dir / "fig7_sojourn_comparison.pdf")
    plt.close(fig)
    print(f"  Saved: fig7_sojourn_comparison.png/pdf")


# =====================================================================
# FIGURE 8: Stage Distribution Over Time
# =====================================================================

def fig_stage_distribution(features, output_dir):
    """Stacked area chart of stage distribution over follow-up time."""
    print("\n[Fig 8] Stage distribution over time...")

    # Bin time into 6-month intervals
    features = features.copy()
    features["time_bin"] = (features["months_from_baseline"] / 6).round() * 6

    # Get stage at each time bin
    valid = features[features["nsd_stage"].notna()].copy()
    valid["stage"] = valid["nsd_stage"].astype(str)

    # Count stages per time bin
    time_bins = sorted(valid["time_bin"].unique())
    time_bins = [t for t in time_bins if 0 <= t <= 180]

    stage_props = {s: [] for s in STAGE_LABELS}
    time_years = []

    for t in time_bins:
        chunk = valid[valid["time_bin"] == t]
        n = len(chunk)
        if n < 10:
            continue
        time_years.append(t / 12)
        for s in STAGE_LABELS:
            stage_props[s].append((chunk["stage"] == s).sum() / n)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    bottoms = np.zeros(len(time_years))
    for stage in STAGE_LABELS:
        values = np.array(stage_props[stage])
        ax.fill_between(time_years, bottoms, bottoms + values,
                        color=STAGE_COLORS[stage], alpha=0.8, label=f"Stage {stage}")
        bottoms += values

    ax.set_xlabel("Years from Baseline", fontsize=12)
    ax.set_ylabel("Proportion of Patients", fontsize=12)
    ax.set_title("NSD-ISS Stage Distribution Over Follow-Up", fontsize=14)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "fig8_stage_distribution.png")
    fig.savefig(output_dir / "fig8_stage_distribution.pdf")
    plt.close(fig)
    print(f"  Saved: fig8_stage_distribution.png/pdf")


# =====================================================================
# FIGURE 9: "Patients Like You" Trajectory Overlay
# =====================================================================

def fig_patients_like_you(features, output_dir):
    """For select patients, show their k=15 graph neighbors' trajectories.

    This is the KEY visualization for Graph-DT novelty — it shows how
    population context enriches individual predictions.
    """
    print("\n[Fig 9] 'Patients Like You' trajectory overlay...")

    from giman_pipeline.paper3.graph_digital_twin import build_patient_graph

    # Build graph
    patient_ids = sorted(features["PATNO"].unique().tolist())
    edge_index, edge_weight, node_baseline = build_patient_graph(
        features, patient_ids, k_neighbors=15
    )
    ei = edge_index.numpy()
    ew = edge_weight.numpy()
    pat_to_idx = {p: i for i, p in enumerate(patient_ids)}
    idx_to_pat = {i: p for p, i in pat_to_idx.items()}

    # Build adjacency: for each node, get its neighbors + weights
    neighbors = {}
    for e in range(ei.shape[1]):
        s, d = int(ei[0, e]), int(ei[1, e])
        w = float(ew[e])
        if s not in neighbors:
            neighbors[s] = []
        neighbors[s].append((d, w))

    # Find interesting "anchor" patients
    baseline = features[features["months_from_baseline"] == 0.0].copy()
    baseline = baseline.drop_duplicates(subset="PATNO", keep="first")

    # Select patients from different stages with enough follow-up
    pat_info = features.groupby("PATNO").agg(
        n_visits=("EVENT_ID", "count"),
        max_time=("months_from_baseline", "max"),
        first_stage=("nsd_stage", "first"),
    ).reset_index()
    pat_info = pat_info[pat_info["n_visits"] >= 5]

    # Pick one from each active stage
    anchor_patients = {}
    for target_stage in ["2B", "3", "4"]:
        candidates = pat_info[pat_info["first_stage"] == target_stage]
        candidates = candidates.sort_values("n_visits", ascending=False)
        if len(candidates) > 0:
            anchor_patients[target_stage] = int(candidates.iloc[0]["PATNO"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax_i, (stage, anchor_pat) in enumerate(anchor_patients.items()):
        ax = axes[ax_i]
        anchor_idx = pat_to_idx.get(anchor_pat)
        if anchor_idx is None:
            continue

        # Get neighbors
        nbrs = neighbors.get(anchor_idx, [])
        nbrs = sorted(nbrs, key=lambda x: -x[1])[:15]  # top 15 by weight

        # Plot neighbor trajectories (faded)
        for nbr_idx, weight in nbrs:
            nbr_pat = idx_to_pat[nbr_idx]
            nbr_data = features[features["PATNO"] == nbr_pat].sort_values("months_from_baseline")
            times = nbr_data["months_from_baseline"].values / 12
            stages = nbr_data["nsd_stage_numeric"].values
            valid = ~np.isnan(stages)
            if valid.sum() >= 2:
                ax.plot(times[valid], stages[valid],
                        alpha=min(weight * 0.6, 0.5), linewidth=1.2,
                        color="#3498db", zorder=2)

        # Plot anchor patient trajectory (bold, on top)
        anchor_data = features[features["PATNO"] == anchor_pat].sort_values("months_from_baseline")
        times = anchor_data["months_from_baseline"].values / 12
        stages = anchor_data["nsd_stage_numeric"].values
        valid = ~np.isnan(stages)
        ax.plot(times[valid], stages[valid], linewidth=3, color="#e74c3c",
                marker="o", markersize=6, zorder=5, label=f"Patient {anchor_pat}")

        # Add neighbor count annotation
        nbr_stages = []
        for nbr_idx, _ in nbrs:
            nbr_pat = idx_to_pat[nbr_idx]
            nbr_bl = baseline[baseline["PATNO"] == nbr_pat]
            if len(nbr_bl) > 0:
                nbr_stages.append(str(nbr_bl.iloc[0]["nsd_stage"]))

        from collections import Counter
        stage_dist = Counter(nbr_stages)
        dist_str = ", ".join(f"S{s}: {n}" for s, n in sorted(stage_dist.items()))

        ax.set_ylabel("NSD-ISS Stage")
        ax.set_xlabel("Years from Baseline")
        ax.set_title(f"Anchor: Stage {stage} Patient\n"
                     f"(15 similar patients shown in blue)")
        ax.set_yticks(STAGE_NUMERIC)
        ax.set_yticklabels(STAGE_LABELS)
        ax.set_xlim(-0.5, 15)
        ax.grid(True, alpha=0.3)

        # Add text box with neighbor info
        textstr = f"Neighbor stages: {dist_str}"
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat",
                                                   alpha=0.5))

    fig.suptitle("\"Patients Like You\": Graph Neighbors' Trajectories\n"
                 "(Graph-DT uses these similar patients to enrich predictions)",
                 fontsize=14, y=1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "fig9_patients_like_you.png")
    fig.savefig(output_dir / "fig9_patients_like_you.pdf")
    plt.close(fig)
    print(f"  Saved: fig9_patients_like_you.png/pdf")


# =====================================================================
# FIGURE 10: Cross-Stage Graph Connectivity Analysis
# =====================================================================

def fig_cross_stage_connectivity(features, output_dir):
    """Analyze how the graph connects patients ACROSS stages.

    This is key for understanding Graph-DT's value: patients in early stages
    are connected to patients in later stages, providing "future trajectory"
    information.
    """
    print("\n[Fig 10] Cross-stage connectivity analysis...")

    from giman_pipeline.paper3.graph_digital_twin import build_patient_graph

    patient_ids = sorted(features["PATNO"].unique().tolist())
    edge_index, edge_weight, node_baseline = build_patient_graph(
        features, patient_ids, k_neighbors=15
    )
    ei = edge_index.numpy()
    ew = edge_weight.numpy()
    pat_to_idx = {p: i for i, p in enumerate(patient_ids)}

    # Get baseline stage per patient
    baseline = features[features["months_from_baseline"] == 0.0].copy()
    baseline = baseline.drop_duplicates(subset="PATNO", keep="first")
    idx_to_stage = {}
    for _, row in baseline.iterrows():
        pat = int(row["PATNO"])
        if pat in pat_to_idx:
            idx_to_stage[pat_to_idx[pat]] = str(row["nsd_stage"])

    # Build stage connectivity matrix
    active_stages = ["0", "1", "2B", "3", "4"]
    n_stages = len(active_stages)
    stage_to_si = {s: i for i, s in enumerate(active_stages)}

    # Count edges between stages
    cross_matrix = np.zeros((n_stages, n_stages), dtype=int)
    weight_matrix = np.zeros((n_stages, n_stages), dtype=float)
    for e in range(ei.shape[1]):
        s_node, d_node = int(ei[0, e]), int(ei[1, e])
        s_stage = idx_to_stage.get(s_node)
        d_stage = idx_to_stage.get(d_node)
        if s_stage in stage_to_si and d_stage in stage_to_si:
            si, di = stage_to_si[s_stage], stage_to_si[d_stage]
            cross_matrix[si, di] += 1
            weight_matrix[si, di] += float(ew[e])

    # Normalize by number of source nodes per stage
    stage_counts_arr = np.zeros(n_stages)
    for idx, stg in idx_to_stage.items():
        if stg in stage_to_si:
            stage_counts_arr[stage_to_si[stg]] += 1

    # Average weight (similarity) per stage pair
    avg_weight = np.zeros_like(weight_matrix)
    for i in range(n_stages):
        for j in range(n_stages):
            if cross_matrix[i, j] > 0:
                avg_weight[i, j] = weight_matrix[i, j] / cross_matrix[i, j]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Edge count heatmap
    ax = axes[0]
    # Normalize by source stage count for proportion
    norm_matrix = np.zeros_like(cross_matrix, dtype=float)
    for i in range(n_stages):
        total = cross_matrix[i].sum()
        if total > 0:
            norm_matrix[i] = cross_matrix[i] / total

    sns.heatmap(norm_matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=active_stages, yticklabels=active_stages,
                ax=ax, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Proportion of Edges"})
    ax.set_xlabel("Neighbor Stage")
    ax.set_ylabel("Patient Stage")
    ax.set_title("A. Edge Distribution Across Stages\n(row-normalized)")

    # Panel B: Average similarity weight
    ax = axes[1]
    mask_zero = avg_weight == 0
    sns.heatmap(avg_weight, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=active_stages, yticklabels=active_stages,
                ax=ax, mask=mask_zero, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Average Cosine Similarity"})
    ax.set_xlabel("Neighbor Stage")
    ax.set_ylabel("Patient Stage")
    ax.set_title("B. Average Edge Weight (Similarity)")

    fig.suptitle("Cross-Stage Graph Connectivity\n"
                 "(How the patient similarity graph connects patients across NSD-ISS stages)",
                 fontsize=14, y=1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "fig10_cross_stage_connectivity.png")
    fig.savefig(output_dir / "fig10_cross_stage_connectivity.pdf")
    plt.close(fig)
    print(f"  Saved: fig10_cross_stage_connectivity.png/pdf")


# =====================================================================
# FIGURE 11: Markov Transition Probability Trajectories
# =====================================================================

def fig_markov_trajectories(markov, output_dir):
    """Show Markov model predicted stage occupation probabilities over time."""
    print("\n[Fig 11] Markov transition probability trajectories...")

    # Load trajectory predictions
    traj_path = PROJECT_ROOT / "outputs" / "paper3_markov" / "trajectory_predictions.csv"
    if not traj_path.exists():
        print("  [SKIP] trajectory_predictions.csv not found")
        return

    traj = pd.read_csv(traj_path)

    start_stages = ["2B", "3", "4"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_i, start in enumerate(start_stages):
        ax = axes[ax_i]
        sub = traj[traj["start_stage"] == start]
        if len(sub) == 0:
            continue

        years = sub["years"].values
        for s in STAGE_LABELS:
            # Column might be named "p_0" or just "0"
            col = f"p_{s}" if f"p_{s}" in sub.columns else s
            if col in sub.columns:
                vals = sub[col].values.astype(float)
                if vals.max() > 0.01:  # only plot non-trivial stages
                    ax.plot(years, vals, label=f"Stage {s}",
                            color=STAGE_COLORS[s], linewidth=2, alpha=0.85)

        ax.set_xlabel("Years")
        ax.set_ylabel("Probability")
        ax.set_title(f"Starting from Stage {start}")
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Markov Model: Stage Occupation Probabilities Over Time",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig11_markov_trajectories.png")
    fig.savefig(output_dir / "fig11_markov_trajectories.pdf")
    plt.close(fig)
    print(f"  Saved: fig11_markov_trajectories.png/pdf")


# =====================================================================
# FIGURE 12: Fold-Level Variance Comparison
# =====================================================================

def fig_fold_variance(deephit, graphdt, output_dir):
    """Box plot showing Graph-DT's lower variance across folds."""
    print("\n[Fig 12] Fold-level variance comparison...")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    data = [deephit["c_td_per_fold"], graphdt["c_td_per_fold"]]
    bp = ax.boxplot(data, tick_labels=["DeepHit", "Graph-DT"],
                    patch_artist=True, widths=0.4,
                    showmeans=True, meanprops={"marker": "D", "markerfacecolor": "black",
                                                "markersize": 8})

    colors = ["#3498db", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual fold points
    for i, (folds, color) in enumerate(zip(data, colors)):
        x_jitter = np.random.RandomState(42).normal(i + 1, 0.05, len(folds))
        ax.scatter(x_jitter, folds, color=color, edgecolor="black",
                   linewidth=0.5, s=60, zorder=5, alpha=0.8)

    ax.set_ylabel("C-td (Time-Dependent Concordance)")
    ax.set_title("Fold-Level Performance Stability\n"
                 f"DeepHit std={deephit['c_td_std']:.4f} vs "
                 f"Graph-DT std={graphdt['c_td_std']:.4f}")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate
    ax.text(0.5, 0.02,
            "Graph-DT shows lower fold-to-fold variance\n"
            "(graph regularization provides more stable predictions)",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic",
            color="#666666")

    fig.tight_layout()
    fig.savefig(output_dir / "fig12_fold_variance.png")
    fig.savefig(output_dir / "fig12_fold_variance.pdf")
    plt.close(fig)
    print(f"  Saved: fig12_fold_variance.png/pdf")


# =====================================================================
# MODEL-BASED VISUALIZATIONS (require one training fold)
# =====================================================================

def run_training_for_visualizations(features, output_dir):
    """Train ONE fold of Graph-DT and save artifacts for visualization.

    Saves: gate activations, node embeddings, individual CIF predictions,
    attention weights, training history.
    """
    print("\n" + "=" * 70)
    print("TRAINING ONE FOLD FOR VISUALIZATION ARTIFACTS")
    print("=" * 70)

    import torch
    from giman_pipeline.paper3.graph_digital_twin import (
        build_patient_graph, GraphDigitalTwin, GraphDeepHitDataset,
        graph_collate_fn, train_graph_model, predict_all_graph,
        GRAPH_FEATURES,
    )
    from giman_pipeline.paper3.dynamic_deephit import (
        build_patient_arrays, extract_episodes, compute_feature_stats,
        _get_time_bin, TIME_BIN_ENDS, N_TIME_BINS,
    )
    from giman_pipeline.paper3.multistate_markov import N_STATES
    from sklearn.model_selection import StratifiedKFold

    device = torch.device("mps") if torch.backends.mps.is_available() else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"  Device: {device}")

    # Prepare data
    patient_arrays, col_names = build_patient_arrays(features)
    input_dim = len(col_names)
    episodes = extract_episodes(features, verbose=True)

    all_patient_ids = sorted(set(e.patno for e in episodes))
    edge_index, edge_weight, node_baseline = build_patient_graph(
        features, all_patient_ids, k_neighbors=15
    )
    pat_to_gidx = {p: i for i, p in enumerate(all_patient_ids)}

    # Use first fold only
    pat_info = {}
    for ep in episodes:
        if ep.patno not in pat_info:
            pat_info[ep.patno] = (ep.current_stage_idx, False)
        if not ep.censored:
            s, _ = pat_info[ep.patno]
            pat_info[ep.patno] = (s, True)

    patnos = sorted(pat_info.keys())
    strat = [f"{pat_info[p][0]}_{int(pat_info[p][1])}" for p in patnos]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = list(skf.split(patnos, strat))[0]

    train_pats_all = set(np.array(patnos)[train_idx])
    test_pats = set(np.array(patnos)[test_idx])

    rng = np.random.RandomState(42)
    tlist = sorted(train_pats_all)
    rng.shuffle(tlist)
    n_val = max(1, len(tlist) // 5)
    val_pats = set(tlist[:n_val])
    atrain_pats = set(tlist[n_val:])

    train_eps = [e for e in episodes if e.patno in atrain_pats]
    val_eps = [e for e in episodes if e.patno in val_pats]
    test_eps = [e for e in episodes if e.patno in test_pats]

    means, stds = compute_feature_stats(patient_arrays, atrain_pats)

    train_ds = GraphDeepHitDataset(train_eps, patient_arrays, means, stds, pat_to_gidx)
    val_ds = GraphDeepHitDataset(val_eps, patient_arrays, means, stds, pat_to_gidx)
    test_ds = GraphDeepHitDataset(test_eps, patient_arrays, means, stds, pat_to_gidx)

    torch.manual_seed(42)
    model = GraphDigitalTwin(
        input_dim=input_dim,
        n_baseline_features=node_baseline.size(1),
        hidden_dim=128,
        n_gru_layers=2,
        gat_heads=4,
        gat_layers=2,
        dropout=0.3,
    ).to(device)

    print(f"  Training fold 1: {len(train_eps)} train, {len(val_eps)} val, {len(test_eps)} test")

    history, best_val = train_graph_model(
        model, train_ds, val_ds, device,
        node_baseline, edge_index, edge_weight,
        n_epochs=100, batch_size=64, lr=5e-4,
        patience=20, alpha=0.1, verbose=True,
    )

    # === Save training history ===
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(history["train_loss"], label="Train Loss", alpha=0.8)
    ax.plot(history["val_loss"], label="Validation Loss", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Graph-DT Training Curve (Fold 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig13_training_curve.png")
    fig.savefig(output_dir / "fig13_training_curve.pdf")
    plt.close(fig)
    print(f"  Saved: fig13_training_curve.png/pdf")

    # === Extract gate activations ===
    print("  Extracting gate activations...")
    model.eval()
    nb = node_baseline.to(device)
    ei = edge_index.to(device)
    ew = edge_weight.to(device)

    graph_feats = model.compute_graph_features(nb, ei, ew)

    gate_values = []
    stage_labels_list = []

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             collate_fn=graph_collate_fn, num_workers=0)

    with torch.no_grad():
        for batch in test_loader:
            seqs = batch["sequences"].to(device)
            slens = batch["seq_lens"]
            sidxs = batch["stage_idxs"].to(device)
            gidxs = batch["graph_idxs"].to(device)

            # Manual forward pass to extract gate values
            sorted_lens, sort_idx = slens.sort(descending=True)
            sorted_seqs = seqs[sort_idx]
            sorted_lens_clamped = sorted_lens.clamp(min=1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                sorted_seqs, sorted_lens_clamped.cpu(), batch_first=True
            )
            gru_out, h_n = model.gru(packed)
            from torch.nn.utils.rnn import pad_packed_sequence
            gru_out_padded, _ = pad_packed_sequence(gru_out, batch_first=True)
            _, unsort_idx = sort_idx.sort()
            gru_out_unsorted = gru_out_padded[unsort_idx]
            last_hidden = h_n[-1][unsort_idx]

            seq_lens_dev = slens.to(device)
            attn_ctx = model.temporal_attn(gru_out_unsorted, seq_lens_dev)
            temporal = attn_ctx + last_hidden

            graph_feat = graph_feats[gidxs]
            gate_input = torch.cat([temporal, graph_feat], dim=-1)
            g = torch.sigmoid(model.gate_linear(gate_input))

            gate_values.append(g.cpu().numpy())
            stage_labels_list.append(batch["stage_idxs"].numpy())

    gate_values = np.concatenate(gate_values, axis=0)  # (n_test, hidden_dim)
    stage_labels_arr = np.concatenate(stage_labels_list)

    # === Figure 14: Gate Activation Distribution ===
    print("  Plotting gate activations...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Overall gate distribution
    ax = axes[0]
    mean_gate = gate_values.mean(axis=1)  # mean gate per episode
    ax.hist(mean_gate, bins=50, color="#e74c3c", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(x=mean_gate.mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_gate.mean():.3f}")
    ax.set_xlabel("Mean Gate Activation")
    ax.set_ylabel("Count (episodes)")
    ax.set_title("A. Gate Activation Distribution\n(0 = pure temporal, 1 = full graph)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Gate by stage
    ax = axes[1]
    from giman_pipeline.paper3.multistate_markov import STAGE_LABELS as SL
    stage_gate_means = {}
    for si in range(len(SL)):
        mask = stage_labels_arr == si
        if mask.sum() > 0:
            stage_gate_means[SL[si]] = mean_gate[mask]

    bp_data = []
    bp_labels = []
    bp_colors = []
    for s in STAGE_LABELS:
        if s in stage_gate_means and len(stage_gate_means[s]) > 5:
            bp_data.append(stage_gate_means[s])
            bp_labels.append(f"Stage {s}\n(n={len(stage_gate_means[s])})")
            bp_colors.append(STAGE_COLORS[s])

    if bp_data:
        bp = ax.boxplot(bp_data, tick_labels=bp_labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], bp_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    ax.set_ylabel("Mean Gate Activation")
    ax.set_title("B. Gate Activation by Current Stage")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Warm-Start Gate Analysis: When Does the Model Use Graph Context?",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig14_gate_activations.png")
    fig.savefig(output_dir / "fig14_gate_activations.pdf")
    plt.close(fig)
    print(f"  Saved: fig14_gate_activations.png/pdf")

    # === Figure 15: Node Embeddings (t-SNE) ===
    print("  Computing t-SNE on node embeddings...")
    try:
        from sklearn.manifold import TSNE

        node_emb = graph_feats.detach().cpu().numpy()  # (1900, 128)

        # Get stage labels for all patients
        baseline = features[features["months_from_baseline"] == 0.0].copy()
        baseline = baseline.drop_duplicates(subset="PATNO", keep="first")
        node_stage_labels = []
        for pat in all_patient_ids:
            bl = baseline[baseline["PATNO"] == pat]
            if len(bl) > 0 and pd.notna(bl.iloc[0]["nsd_stage"]):
                node_stage_labels.append(str(bl.iloc[0]["nsd_stage"]))
            else:
                node_stage_labels.append("unknown")

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        emb_2d = tsne.fit_transform(node_emb)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        for stage in STAGE_LABELS:
            mask = np.array(node_stage_labels) == stage
            if mask.sum() > 0:
                ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                           c=STAGE_COLORS[stage], s=15, alpha=0.7,
                           label=f"Stage {stage} (n={mask.sum()})",
                           edgecolors="none")

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("GAT Node Embeddings (t-SNE)\n"
                     "Graph-enriched patient representations colored by NSD-ISS stage")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=10,
                  title="NSD-ISS Stage")
        ax.axis("equal")

        fig.tight_layout()
        fig.savefig(output_dir / "fig15_node_embeddings_tsne.png")
        fig.savefig(output_dir / "fig15_node_embeddings_tsne.pdf")
        plt.close(fig)
        print(f"  Saved: fig15_node_embeddings_tsne.png/pdf")
    except Exception as ex:
        print(f"  [SKIP] t-SNE failed: {ex}")

    # === Figure 16: Example CIF Predictions ===
    print("  Generating individual CIF predictions...")
    preds = predict_all_graph(model, test_ds, device,
                              node_baseline, edge_index, edge_weight)

    cif = preds["cif"].numpy()  # (n_test, n_causes, n_time_bins)
    event_idxs = preds["event_idxs"].numpy()
    time_bins = preds["time_bins"].numpy()
    censored_arr = preds["censored"].numpy()
    stage_idxs_arr = preds["stage_idxs"].numpy()

    # Pick 4 uncensored patients with different transitions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    target_events = [3, 4, 2, 0]  # transitions to Stage 3, 4, 2B, 0
    panel_labels = ["A", "B", "C", "D"]

    time_months = np.array(TIME_BIN_ENDS)

    for ax_idx, target_event in enumerate(target_events):
        ax = axes[ax_idx // 2, ax_idx % 2]

        # Find an example with this transition
        mask = (~censored_arr) & (event_idxs == target_event)
        candidates = np.where(mask)[0]

        if len(candidates) == 0:
            ax.text(0.5, 0.5, "No examples", transform=ax.transAxes, ha="center")
            continue

        # Pick the one with highest predicted CIF (most confident)
        best = candidates[0]
        for c in candidates:
            if cif[c, target_event, time_bins[c]] > cif[best, target_event, time_bins[best]]:
                best = c

        # Plot CIF curves for each cause
        for k in range(min(N_STATES, 7)):
            if cif[best, k, :].max() > 0.01:
                label = f"→ Stage {STAGE_LABELS[k]}"
                ax.plot(time_months / 12, cif[best, k, :],
                        color=STAGE_COLORS[STAGE_LABELS[k]],
                        linewidth=2, alpha=0.8, label=label)

        # Mark actual event
        actual_time = time_months[time_bins[best]] / 12
        ax.axvline(x=actual_time, color="black", linestyle="--", linewidth=1,
                   alpha=0.5)
        ax.scatter([actual_time], [cif[best, target_event, time_bins[best]]],
                   color="black", s=100, zorder=5, marker="*",
                   label="Actual transition")

        from_stage = STAGE_LABELS[stage_idxs_arr[best]]
        to_stage = STAGE_LABELS[target_event]
        ax.set_xlabel("Years")
        ax.set_ylabel("Cumulative Incidence")
        ax.set_title(f"{panel_labels[ax_idx]}. From Stage {from_stage}: "
                     f"Actual → Stage {to_stage}")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Graph-DT Individual Predictions: Cumulative Incidence Functions\n"
                 "(competing risks: probability of transitioning to each stage over time)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig16_individual_cif.png")
    fig.savefig(output_dir / "fig16_individual_cif.pdf")
    plt.close(fig)
    print(f"  Saved: fig16_individual_cif.png/pdf")

    print("\n  Model-based visualization artifacts complete!")


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Paper 3 figures")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip model-based visualizations (no retraining)")
    parser.add_argument("--training-only", action="store_true",
                        help="Only run model-based visualizations")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features, transitions, benchmark, markov, deephit, graphdt = load_data()

    if not args.training_only:
        print("\n" + "=" * 70)
        print("STATIC VISUALIZATIONS (from saved data)")
        print("=" * 70)

        # 1. Patient similarity graph
        fig_patient_graph(features, OUTPUT_DIR)

        # 2. Transition matrix heatmap
        fig_transition_matrix(benchmark, OUTPUT_DIR)

        # 3. Patient trajectories
        fig_patient_trajectories(features, OUTPUT_DIR)

        # 4. Model comparison
        fig_model_comparison(deephit, graphdt, OUTPUT_DIR)

        # 5. Per-transition C-td
        fig_per_transition_ctd(deephit, graphdt, OUTPUT_DIR)

        # 6. Brier score at horizons
        fig_brier_horizons(deephit, graphdt, OUTPUT_DIR)

        # 7. Sojourn time comparison
        fig_sojourn_times(markov, OUTPUT_DIR)

        # 8. Stage distribution over time
        fig_stage_distribution(features, OUTPUT_DIR)

        # 9. "Patients Like You"
        fig_patients_like_you(features, OUTPUT_DIR)

        # 10. Cross-stage connectivity
        fig_cross_stage_connectivity(features, OUTPUT_DIR)

        # 11. Markov trajectories
        fig_markov_trajectories(markov, OUTPUT_DIR)

        # 12. Fold variance
        fig_fold_variance(deephit, graphdt, OUTPUT_DIR)

    if not args.skip_training:
        run_training_for_visualizations(features, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print(f"ALL FIGURES SAVED TO: {OUTPUT_DIR}")
    print("=" * 70)

    # Summary
    figs = sorted(OUTPUT_DIR.glob("fig*.png"))
    print(f"\nGenerated {len(figs)} figures:")
    for f in figs:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
