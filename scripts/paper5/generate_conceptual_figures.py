"""Generate two conceptual figures for the JAMIA submission of Paper 5.

Fig S3 (pccp_monitoring_schematic):
    Flowchart of the FDA SaMD Predetermined Change Control Plan monitoring loop:
        new batch of patient data -> per-feature KS + PSI + multivariate MMD
        shift tests -> decision gate (fraction shifted >30%?) -> retrain + revalidate
        OR continue monitoring. Maps the prose in Discussion section to a picture.

Fig S4 (inductive_graph_schematic):
    Two-panel schematic:
        (a) Training-only kNN patient similarity graph (16 nodes, k=4)
        (b) Same graph with 4 test nodes added; each test node connects to its
            k=4 nearest training neighbours; CRITICAL: zero test-test edges.
    Illustrates the inductive extension described in Methods §Inductive graph
    extension.

Outputs:
    outputs/mechanistic_twin/paper5_submission/figures/figS3_pccp_monitoring.{png,pdf}
    outputs/mechanistic_twin/paper5_submission/figures/figS4_inductive_graph.{png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "mechanistic_twin" / "paper5_submission" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

OI_BLACK = "#000000"
OI_VERMILLION = "#D55E00"
OI_BLUE_GREEN = "#009E73"
OI_SKY_BLUE = "#56B4E9"
OI_ORANGE = "#E69F00"
OI_BLUE = "#0072B2"
OI_REDDISH_PURPLE = "#CC79A7"
OI_YELLOW = "#F0E442"
OI_WHITE = "#FFFFFF"
OI_GRAY = "#888888"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _box(ax, x, y, w, h, text, facecolor, edgecolor=OI_BLACK,
         text_color=OI_BLACK, fontsize=9, fontweight="normal",
         style="round,pad=0.02,rounding_size=0.02"):
    """Draw a rounded-rectangle box with centred text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=style, linewidth=1.3,
                         facecolor=facecolor, edgecolor=edgecolor)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_color,
            wrap=True)


def _arrow(ax, x1, y1, x2, y2, color=OI_BLACK, label=None, label_offset=(0, 0),
           style="-|>", linestyle="-"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, mutation_scale=18,
                            linewidth=1.6, color=color, linestyle=linestyle,
                            shrinkA=2, shrinkB=2)
    ax.add_patch(arrow)
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=8, style="italic", color=color,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.85))


def fig_pccp_monitoring() -> None:
    """PCCP monitoring loop flowchart."""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # ── Row 1: incoming data / prior model ──
    _box(ax, 2.0, 7.0, 2.8, 0.9,
         "New batch of\npatient data\n(next enrollment cohort)",
         OI_SKY_BLUE, fontsize=9, fontweight="bold")
    _box(ax, 2.0, 5.2, 2.8, 0.9,
         "Prior deployed\nmodel + training\nfeature distributions",
         OI_SKY_BLUE, fontsize=9, fontweight="bold")

    # ── Row 2: three shift tests in parallel ──
    _box(ax, 6.5, 7.0, 2.2, 0.7, "KS tests\n(per-feature, $p<0.001$)",
         OI_ORANGE, fontsize=8)
    _box(ax, 6.5, 6.1, 2.2, 0.7, "PSI\n(per-feature, $>0.25$)",
         OI_ORANGE, fontsize=8)
    _box(ax, 6.5, 5.2, 2.2, 0.7, "MMD permutation test\n(multivariate, $p<0.001$)",
         OI_ORANGE, fontsize=8)

    # ── Row 3: aggregation ──
    _box(ax, 10.5, 6.1, 2.6, 0.9,
         "Aggregate shift verdict\nfraction features shifted\n+ multivariate significance",
         OI_BLUE_GREEN, fontsize=9, fontweight="bold")

    # ── Row 4: decision gate ──
    _box(ax, 10.5, 4.0, 3.0, 1.1,
         ">30% features shifted\nOR MMD $p<0.001$?",
         OI_YELLOW, fontsize=10, fontweight="bold",
         style="round,pad=0.05,rounding_size=0.03")

    # ── Row 5: retrain + continue paths ──
    _box(ax, 6.5, 2.0, 3.4, 1.0,
         "Retrain on expanded cohort\n+ revalidate temporally ($W_{\\mathrm{new}}$)\n+ PCCP re-approval workflow",
         OI_VERMILLION, fontsize=9, fontweight="bold", text_color=OI_WHITE)
    _box(ax, 12.0, 2.0, 2.5, 1.0,
         "Continue with deployed\nmodel; log shift metrics",
         OI_BLUE_GREEN, fontsize=9, text_color=OI_WHITE, fontweight="bold")

    # ── Row 6: feedback loop ──
    _box(ax, 2.0, 2.0, 2.8, 1.0,
         "Redeploy + re-audit\n(W1-W4 degradation\nenvelope)",
         OI_REDDISH_PURPLE, fontsize=9, text_color=OI_WHITE, fontweight="bold")

    # Arrows
    # Inputs into the 3 tests
    for y in (5.2, 6.1, 7.0):
        _arrow(ax, 3.5, 7.0 if y > 6 else y, 5.4, y, color=OI_GRAY)
        _arrow(ax, 3.5, 5.2, 5.4, y, color=OI_GRAY, linestyle=":")

    # Tests -> aggregator
    for y in (5.2, 6.1, 7.0):
        _arrow(ax, 7.7, y, 9.2, 6.1, color=OI_GRAY)

    # Aggregator -> decision
    _arrow(ax, 10.5, 5.55, 10.5, 4.6, color=OI_BLACK)

    # Decision -> retrain (YES branch)
    _arrow(ax, 9.0, 4.0, 8.3, 2.5, color=OI_VERMILLION,
           label="Yes  trigger retrain", label_offset=(-0.5, 0.25))
    # Decision -> continue (NO branch)
    _arrow(ax, 12.0, 4.0, 12.0, 2.6, color=OI_BLUE_GREEN,
           label="No  continue",
           label_offset=(0.6, 0))

    # Retrain -> redeploy
    _arrow(ax, 4.8, 2.0, 3.5, 2.0, color=OI_VERMILLION)
    # Redeploy -> data (closing loop)
    _arrow(ax, 2.0, 2.6, 2.0, 6.5, color=OI_REDDISH_PURPLE, linestyle="--",
           label="next enrollment\ncohort", label_offset=(-1.5, 0))

    # Titles
    ax.text(7.0, 7.8,
            "Paper 5 monitoring triad",
            ha="center", va="bottom", fontsize=10, style="italic", color=OI_GRAY)
    ax.text(7.0, 0.45,
            "Mapped onto FDA PCCP requirements: monitoring triggers (Paper 5) "
            "+ retraining protocol (PCCP) + revalidation (temporal $W_{\\mathrm{new}}$) "
            "+ re-audit loop (Paper 5 W1-W4 envelope).",
            ha="center", va="center", fontsize=8.5, style="italic",
            color=OI_BLACK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                      edgecolor=OI_BLACK, linewidth=0.5))

    fig.suptitle("Covariate-shift-triggered retraining loop for a SaMD-compliant PCCP",
                 fontsize=12, fontweight="bold", y=0.97)

    for ext in ("png", "pdf"):
        p = OUT / f"figS3_pccp_monitoring.{ext}"
        fig.savefig(p)
        print(f"  wrote {p.relative_to(REPO)}")
    plt.close(fig)


def _knn_edges(xy, k, exclude_self=True):
    """Return list of (i, j) edges connecting each node i to its k nearest other nodes."""
    n = len(xy)
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1))
    if exclude_self:
        np.fill_diagonal(d, np.inf)
    edges = set()
    for i in range(n):
        nn = np.argsort(d[i])[:k]
        for j in nn:
            edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def fig_inductive_graph() -> None:
    """Two-panel schematic: training graph | training + inductive test nodes."""
    rng = np.random.default_rng(20260418)
    n_train = 16
    n_test = 4
    k = 4

    # Training nodes: cluster-ish layout in 2D feature space
    train_xy = rng.normal(size=(n_train, 2)) * 1.2
    # Test nodes: placed deliberately so some are close to training clusters
    test_xy = np.array([
        [1.3, 1.6],    # near upper-right cluster
        [-1.9, -0.3],  # near left
        [0.5, -1.8],   # near lower-middle
        [-0.2, 2.2],   # near top
    ])

    all_xy = np.vstack([train_xy, test_xy])

    # Edges in training-only graph
    train_edges = _knn_edges(train_xy, k)
    # Inductive extension: for each test node, connect to k nearest training nodes
    d_test_to_train = np.sqrt(((test_xy[:, None, :] - train_xy[None, :, :]) ** 2).sum(axis=-1))
    test_to_train_edges = []
    for i in range(n_test):
        nn = np.argsort(d_test_to_train[i])[:k]
        for j in nn:
            test_to_train_edges.append((n_train + i, int(j)))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5.5))

    # ── Panel (a): training-only graph ──
    ax_a.set_title("(a) Training-only $k$-NN patient similarity graph",
                   fontsize=10, fontweight="bold")
    for (i, j) in train_edges:
        x1, y1 = train_xy[i]; x2, y2 = train_xy[j]
        ax_a.plot([x1, x2], [y1, y2], color=OI_GRAY, linewidth=0.9, alpha=0.6, zorder=1)
    ax_a.scatter(train_xy[:, 0], train_xy[:, 1], s=280, color=OI_BLUE,
                 edgecolor=OI_BLACK, linewidth=1.0, zorder=3,
                 label=f"Training patient ($n={n_train}$)")
    for idx, (x, y) in enumerate(train_xy):
        ax_a.text(x, y, str(idx+1), ha="center", va="center",
                  fontsize=8, color=OI_WHITE, fontweight="bold", zorder=4)

    ax_a.set_xlabel("Baseline feature 1 (demo axis)")
    ax_a.set_ylabel("Baseline feature 2 (demo axis)")
    ax_a.legend(loc="upper right", fontsize=8)
    ax_a.set_aspect("equal")
    ax_a.grid(alpha=0.15)
    ax_a.set_xlim(-3.2, 3.2); ax_a.set_ylim(-3.2, 3.2)

    # ── Panel (b): training + test inductive extension ──
    ax_b.set_title("(b) Inductive extension: test nodes added with $k{=}4$ train-only edges",
                   fontsize=10, fontweight="bold")
    # training edges (lighter)
    for (i, j) in train_edges:
        x1, y1 = train_xy[i]; x2, y2 = train_xy[j]
        ax_b.plot([x1, x2], [y1, y2], color=OI_GRAY, linewidth=0.7, alpha=0.35, zorder=1)
    # inductive (test -> train) edges, highlighted
    for (t_idx, tr_idx) in test_to_train_edges:
        x1, y1 = all_xy[t_idx]; x2, y2 = all_xy[tr_idx]
        ax_b.plot([x1, x2], [y1, y2], color=OI_VERMILLION,
                  linewidth=1.6, alpha=0.9, zorder=2)

    ax_b.scatter(train_xy[:, 0], train_xy[:, 1], s=220, color=OI_BLUE,
                 edgecolor=OI_BLACK, linewidth=1.0, zorder=3,
                 label="Training patient")
    ax_b.scatter(test_xy[:, 0], test_xy[:, 1], s=260,
                 facecolor=OI_ORANGE, edgecolor=OI_VERMILLION, linewidth=2.0,
                 marker="s", zorder=4,
                 label=f"Test patient ($n={n_test}$, later enrollment)")
    for idx, (x, y) in enumerate(train_xy):
        ax_b.text(x, y, str(idx+1), ha="center", va="center",
                  fontsize=7, color=OI_WHITE, fontweight="bold", zorder=5)
    for idx, (x, y) in enumerate(test_xy):
        ax_b.text(x, y, f"T{idx+1}", ha="center", va="center",
                  fontsize=7, color=OI_BLACK, fontweight="bold", zorder=5)

    # Annotate the CRITICAL constraint
    ax_b.text(0.02, 0.98,
              "Each test node connects to\n"
              "$k{=}4$ nearest TRAINING\n"
              "neighbours (vermillion edges).\n"
              "ZERO test-test edges\n"
              "(prevents future-patient leakage).",
              transform=ax_b.transAxes, va="top", ha="left",
              fontsize=8, color=OI_BLACK, linespacing=1.25, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                        edgecolor=OI_VERMILLION, linewidth=1.0, alpha=0.92))

    ax_b.set_xlabel("Baseline feature 1 (demo axis)")
    ax_b.set_ylabel("")
    ax_b.legend(loc="lower right", fontsize=8)
    ax_b.set_aspect("equal")
    ax_b.grid(alpha=0.15)
    ax_b.set_xlim(-3.2, 3.2); ax_b.set_ylim(-3.2, 3.2)

    fig.suptitle("Inductive graph extension for deploying Graph-DT on future patients",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = OUT / f"figS4_inductive_graph.{ext}"
        fig.savefig(p)
        print(f"  wrote {p.relative_to(REPO)}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating conceptual figures for Paper 5...")
    fig_pccp_monitoring()
    fig_inductive_graph()
    print("Done.")
