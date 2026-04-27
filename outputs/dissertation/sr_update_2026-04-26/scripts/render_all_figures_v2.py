"""Render all 6 ch02 figures as PDFs with the updated 60-paper synthesis.

Outputs (replacing the original chapter TikZ figures):
- sr_prisma_flow_60.pdf (Figure 1 — PRISMA flow with cumulative 1,654 → 60)
- sr_architecture_60.pdf (Figure 2 — model architecture distribution, 60 papers)
- sr_probast_grid_60.pdf (Figure 3 — PROBAST grid, 59 assessable studies)
- sr_validation_60.pdf (Figure 4 — validation tier pie + reporting gaps for 60 papers)
- sr_heterogeneity_60.pdf (Figure 5 — heterogeneity 5-criteria for 60 papers)
- sr_harvest_60.pdf (Figure 6 — extended harvest plot, 11 head-to-head studies)
"""
import csv, json, math
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon, FancyArrowPatch

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025/outputs/dissertation/sr_update_2026-04-26")
FIGS = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025/outputs/dissertation/figures")

# Color scheme (matches LaTeX)
LOW = "#4CAF50"; MOD = "#FF9800"; HIGH = "#F44336"; LM = "#A5D6A7"; NA = "#BDBDBD"
TIER2 = "#43A047"; TIER1 = "#FB8C00"; TIER0 = "#FBC02D"

# ============================================================================
# DATA LOAD
# ============================================================================
def load_data():
    with open(ROOT / "data_extraction.csv") as f:
        rows = list(csv.DictReader(f))
    deep = {}
    for batch in (ROOT / "extractions").glob("batch_*.json"):
        try:
            d = json.loads(batch.read_text())
            if isinstance(d, dict):
                d = d.get("papers", d.get("extractions", [d]))
            if isinstance(d, list):
                for p in d:
                    if isinstance(p, dict) and p.get("row"):
                        deep[int(p["row"])] = p
        except: pass
    return rows, deep


def get_probast(r, deep):
    """Get PROBAST scores with deep override."""
    rn = int(r["row"])
    d = deep.get(rn, {})
    pr = d.get("probast_revised", {})
    return (pr.get("D1", r["probast_d1"]),
            pr.get("D2", r["probast_d2"]),
            pr.get("D3", r["probast_d3"]),
            pr.get("D4", r["probast_d4"]),
            pr.get("OVERALL", r["probast_overall"]))


# ============================================================================
# FIGURE 1: PRISMA FLOW (cumulative)
# ============================================================================
def render_prisma_flow():
    fig, ax = plt.subplots(figsize=(9, 11.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 14); ax.axis("off")

    def box(x, y, w, h, text, fill="#E3F2FD", edge="#1565C0", weight="normal", fontsize=8.5):
        p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.04,rounding_size=0.06",
                           linewidth=1.0, edgecolor=edge, facecolor=fill, zorder=2)
        ax.add_patch(p)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.0, color="#444", zorder=1))

    fig.suptitle("Figure 1. PRISMA-2020 Flow Diagram",
                 fontsize=11, fontweight="bold", y=0.97)

    # Identification
    ax.text(0.5, 12.7, "IDENTIFICATION", fontsize=10, fontweight="bold", color="#1565C0")
    box(6, 12.0, 9, 1.1, "Records identified across seven databases\n(Jan 2016 -- Apr 2026)\n$n = 1{,}654$\n(systematic search + AI-assisted + snowball forward citations)",
        fill="#FFF3E0", edge="#F57C00", weight="bold")
    arrow(6, 11.4, 6, 10.6)

    # Screening
    ax.text(0.5, 10.4, "SCREENING", fontsize=10, fontweight="bold", color="#1565C0")
    box(6, 10.0, 6, 0.7, "Records after dedup + PD-relevance filter:  $n = 334$",
        fill="#F3E5F5", edge="#7B1FA2", weight="bold")
    arrow(6, 9.6, 6, 9.0)
    box(6, 8.6, 6, 0.65,
        "Title/abstract excluded:  $n = 236$",
        fill="#FAFAFA", edge="#9E9E9E", fontsize=8)
    arrow(6, 8.27, 6, 7.7)

    # Eligibility
    ax.text(0.5, 7.5, "ELIGIBILITY", fontsize=10, fontweight="bold", color="#1565C0")
    box(4.0, 7.2, 5, 0.7,
        "Full-text articles assessed:\n$n = 98$",
        fill="#F3E5F5", edge="#7B1FA2", weight="bold")
    box(9.5, 7.2, 5, 1.7,
        "Full-text excluded:  $n = 38$\n\n  E1 (no empirical validation):  7\n  E1-circular (UPDRS leakage):  7\n  E2 (conference abstract):     5\n  E3 (not prognostic):         11\n  E4 (not a pred. model):       4\n  E5 (duplicate / overlap):     4",
        fill="#FFEBEE", edge="#C62828", fontsize=7.3)
    arrow(6.5, 7.2, 7, 7.2)

    # Included
    ax.text(0.5, 5.6, "INCLUDED", fontsize=10, fontweight="bold", color="#1565C0")
    box(6, 5.2, 6, 0.95,
        "Studies INCLUDED in qualitative + quantitative synthesis\n$n = 60$",
        fill="#C8E6C9", edge="#1B5E20", weight="bold", fontsize=10)
    arrow(6, 4.7, 6, 4.2)

    # Final breakdown
    box(6, 3.6, 8, 1.0,
        "Pooled meta-analysis:  $n = 11$ head-to-head studies\n"
        "Tier-2 external validation:  16 studies  $\\cdot$  PROBAST L:M:H = 20:61:17 (\\%)\n"
        "Mechanistic-DT (NASEM-partial):  Hemedan 2026 (3/4)  $\\cdot$  Matsui 2026 (2/4)",
        fill="#FFF8E1", edge="#F9A825", fontsize=8.5)

    # Footer
    ax.text(6, 0.5, "1,654 records  $\\rightarrow$  334 after dedup  $\\rightarrow$  98 full-text  $\\rightarrow$  38 excluded  $\\rightarrow$  60 included.",
            ha="center", fontsize=8, style="italic", color="#555")
    ax.text(6, 0.15, "PROSPERO CRD420261293572",
            ha="center", fontsize=7, style="italic", color="#777")

    out = FIGS / "sr_prisma_flow_60.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Wrote {out.name}")


# ============================================================================
# FIGURE 2: MODEL ARCHITECTURE DISTRIBUTION (60 papers)
# ============================================================================
def render_architecture(rows, deep):
    """Bar chart of model_class across 60 papers (uses canonical CSV model_class column)."""
    bucket_map = {
        "static-ML":           "Static ML (RF / XGBoost / SVM / ensemble)",
        "dynamic-temporal":    "Dynamic temporal (RNN / LSTM / Transformer / HMM / GAT)",
        "dynamic-RL":          "Dynamic — Reinforcement Learning",
        "dynamic-hybrid":      "Dynamic hybrid (LLM + MCTS + RAG)",
        "dynamic-mechanistic": "Dynamic mechanistic (Gaussian Process + clustering)",
        "mechanistic":         "Mechanistic (PK/PD ODE, Neural-mass, SDE)",
        "mechanistic-hybrid":  "Mechanistic / hybrid (Neural ODE, NLME, SReFT)",
        "mechanistic-prob":    "Mechanistic — probabilistic (Bayesian DT)",
        "n/a":                 "Protocol-only (excluded from PROBAST)",
    }
    classes = [bucket_map.get(r["model_class"].strip(), "Other / unknown") for r in rows]

    counts = Counter(classes)
    total = sum(counts.values())
    sorted_items = counts.most_common()

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [k for k, v in sorted_items]
    values = [v for k, v in sorted_items]
    pcts = [100 * v / total for v in values]
    colors = ["#1976D2", "#43A047", "#FB8C00", "#8E24AA", "#E53935", "#0097A7", "#7E57C2", "#9E9E9E"][:len(labels)]

    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Number of studies", fontsize=10)
    ax.set_xlim(0, max(values) * 1.25)
    for i, (v, p) in enumerate(zip(values, pcts)):
        ax.text(v + 0.5, i, f"{v}  ({p:.0f}\\%)", va="center", fontsize=9.5, fontweight="bold")

    # NASEM annotation
    ax.axhline(y=len(labels) - 0.4, color="red", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.text(max(values) * 0.5, len(labels) - 0.5,
            f"NASEM digital-twin partial-compliance: 2 / {total}  (Hemedan 3/4 + Matsui 2/4)",
            fontsize=8.5, color="red", style="italic", ha="center", bbox=dict(facecolor="white", alpha=0.85, edgecolor="red", pad=2))

    n_static = sum(v for k, v in counts.items() if k.startswith("Static"))
    n_dynamic = sum(v for k, v in counts.items() if k.startswith("Dynamic"))
    n_mech = sum(v for k, v in counts.items() if k.startswith("Mechanistic"))
    ax.set_title(f"Figure 2. Model architecture distribution across {total} included studies\n"
                 f"Static ML: {n_static}/{total} = {100*n_static/total:.0f}\\%  $\\cdot$  "
                 f"Dynamic temporal: {n_dynamic}/{total} = {100*n_dynamic/total:.0f}\\%  $\\cdot$  "
                 f"Mechanistic / hybrid: {n_mech}/{total} = {100*n_mech/total:.0f}\\%",
                 fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    out = FIGS / "sr_architecture_60.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Wrote {out.name}")


# ============================================================================
# FIGURE 3: PROBAST GRID (59 STUDIES)
# ============================================================================
def render_probast_grid(rows, deep):
    color_map = {"L": LOW, "M": MOD, "H": HIGH, "L_M": LM, "L-M": LM, "n/a": NA, "": NA}

    grid = []
    for r in rows:
        rn = int(r["row"])
        if rn == 30:  # Bloem protocol — not assessed
            continue
        d1, d2, d3, d4, ov = get_probast(r, deep)
        grid.append((rn, r["first_author_year"], d1, d2, d3, d4, ov))

    n = len(grid)
    fig, ax = plt.subplots(figsize=(10, 13.5))

    cell_w, cell_h = 1.0, 0.4
    domains = ["D1: Participants", "D2: Predictors", "D3: Outcome", "D4: Analysis", "Overall"]
    x_offsets = [3.5, 4.6, 5.7, 6.8, 7.95]

    # Domain column headers (rotated)
    for i, dom in enumerate(domains):
        ax.text(x_offsets[i] + cell_w / 2, n * cell_h + 0.4, dom,
                rotation=45, ha="left", va="bottom", fontsize=8.5, fontweight="bold")

    # Rows
    for i, (rn, label, d1, d2, d3, d4, ov) in enumerate(grid):
        y = (n - i - 1) * cell_h
        ax.text(3.4, y + cell_h / 2, f"{rn}. {label[:22]}", ha="right", va="center", fontsize=7)
        for j, val in enumerate([d1, d2, d3, d4, ov]):
            color = color_map.get(val.strip().replace("-", "_"), NA) if val else NA
            edge = "black" if j == 4 else "#666"
            lw = 1.2 if j == 4 else 0.5
            rect = Rectangle((x_offsets[j], y), cell_w * 0.8, cell_h * 0.8,
                             facecolor=color, edgecolor=edge, linewidth=lw)
            ax.add_patch(rect)

    # Aggregate stats footer
    overall_counter = Counter(p[6] for p in grid)
    total = len(grid)
    stats = (f"$n = {total}$ assessable studies (Bloem 2019 protocol excluded)  ·  "
             f"Low: {overall_counter.get('L', 0)} ({100*overall_counter.get('L', 0)/total:.0f}\\%)  ·  "
             f"Moderate: {overall_counter.get('M', 0)} ({100*overall_counter.get('M', 0)/total:.0f}\\%)  ·  "
             f"High: {overall_counter.get('H', 0)} ({100*overall_counter.get('H', 0)/total:.0f}\\%)")
    ax.text((x_offsets[0] + x_offsets[-1]) / 2 + 0.5, -0.8, stats,
            ha="center", fontsize=8.5, fontweight="bold")

    # Legend
    legend_y = -1.6
    legend_items = [(LOW, "Low risk"), (MOD, "Moderate risk"), (HIGH, "High risk"), (LM, "Borderline (L–M)")]
    for i, (color, label) in enumerate(legend_items):
        x = 3.5 + i * 1.5
        ax.add_patch(Rectangle((x, legend_y), 0.4, 0.3, facecolor=color, edgecolor="black", linewidth=0.5))
        ax.text(x + 0.5, legend_y + 0.15, label, va="center", fontsize=8)

    ax.set_xlim(2, 10)
    ax.set_ylim(-2.2, n * cell_h + 1.5)
    ax.axis("off")
    ax.set_title(f"Figure 3. PROBAST risk-of-bias grid ($n={n}$ studies)",
                 fontsize=12, fontweight="bold", pad=20)

    out = FIGS / "sr_probast_grid_60.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Wrote {out.name}")
    return overall_counter


# ============================================================================
# FIGURE 4: VALIDATION TIER PIE + REPORTING GAPS (60 papers)
# ============================================================================
def render_validation_60(rows, deep):
    # Collect tier counts
    tier_counts = Counter()
    for r in rows:
        v = r["validation_tier"].strip()
        if v == "0": tier_counts["Tier 0 (Internal CV)"] += 1
        elif v == "1": tier_counts["Tier 1 (Holdout)"] += 1
        elif v == "2": tier_counts["Tier 2 (External)"] += 1
        elif v.lower() in ("inversion", "fitting", "n/a"): tier_counts["Other (Inversion/Fitting/N/A)"] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # Panel A: Pie chart
    labels = ["Tier 2 (External)", "Tier 1 (Holdout)", "Tier 0 (Internal CV)", "Other (Inversion/Fitting/N/A)"]
    sizes = [tier_counts.get(l, 0) for l in labels]
    colors_p = [TIER2, TIER1, TIER0, "#BDBDBD"]

    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors_p, autopct="%1.0f%%",
                                        startangle=90, pctdistance=0.78, textprops={"fontsize": 9.5})
    for at in autotexts:
        at.set_fontsize(11); at.set_fontweight("bold"); at.set_color("white")
    ax1.set_title(f"(A) Validation tier distribution\n$n = {sum(sizes)}$ studies", fontsize=11, fontweight="bold")

    # Panel B: Reporting gaps (chapter-aligned metrics; see §3.3-§3.4 for definitions)
    # External validation: Tier 2 count = 16/60 = 27%
    # Calibration (formal: HL/slope/intercept/Brier): 2/59 = 3.4%
    # 95% CIs reported: 25/59 = 42% (per chapter §3.5 Heterogeneity Criterion B)
    # Sample size >100: 36/60 = 60%
    # TRIPOD+AI explicit compliance (Vamvakas 2026, Wang 2026 Front Aging, Duan 2026): 3/30 update = 10%, ~5% across 60
    has_n_gt100 = sum(1 for r in rows
                      if (r["n_pd"].replace(",", "").replace("+", "").replace("~", "").split()[0] if r["n_pd"] else "").isdigit()
                      and int(r["n_pd"].replace(",", "").replace("+", "").replace("~", "").split()[0]) > 100)
    metrics = [
        ("External\nvalidation",  27,                              TIER2),
        ("Formal\ncalibration",    3,                              MOD),
        ("95\\% CIs\nreported",   42,                              TIER2),
        ("Sample size\n$>$100",   round(100 * has_n_gt100 / 60),  "#0097A7"),
        ("TRIPOD+AI\ncompliant",   5,                              MOD),
    ]

    x = list(range(len(metrics)))
    vals = [m[1] for m in metrics]
    colors_b = [m[2] for m in metrics]

    ax2.bar(x, vals, width=0.6, color=colors_b, edgecolor="black", linewidth=0.5)
    for xi, v in enumerate(vals):
        ax2.text(xi, v + 1, f"{v:.0f}\\%", ha="center", fontsize=9, fontweight="bold")

    ax2.axhline(50, color="gray", linestyle="--", alpha=0.5)
    ax2.text(len(metrics) - 0.5, 51, "50\\%", fontsize=8, color="gray", va="bottom")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m[0] for m in metrics], fontsize=9)
    ax2.set_ylabel("\\% of Studies", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.set_title("(B) Reporting quality indicators ($n=60$)", fontsize=11, fontweight="bold")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.suptitle("Figure 4. Validation quality and reporting gaps",
                 fontsize=12, fontweight="bold", y=1.02)

    out = FIGS / "sr_validation_60.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Wrote {out.name}")


# ============================================================================
# FIGURE 5: HETEROGENEITY 5-CRITERIA (60 papers)
# ============================================================================
def render_heterogeneity_60(rows):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6),
                                          gridspec_kw={"width_ratios": [1, 1, 1.2]})

    fig.suptitle("Figure 5. Heterogeneity assessment for meta-analysis feasibility",
                 fontsize=11, fontweight="bold", y=1.02)

    # ===== Panel A: Outcome metric distribution =====
    metrics = Counter()
    for r in rows:
        m = r["primary_metric"].lower()
        if "auc" in m or "auroc" in m: metrics["AUC / AUROC"] += 1
        elif "c-index" in m or "concordance" in m: metrics["C-index"] += 1
        elif "rmse" in m or "mae" in m or "smape" in m: metrics["RMSE / MAE / SMAPE"] += 1
        elif "hr" in m or "hazard" in m: metrics["Hazard Ratio"] += 1
        elif "accuracy" in m or "f1" in m: metrics["Accuracy / F1"] += 1
        elif "r²" in m or "r2" in m: metrics["$R^2$"] += 1
        else: metrics["Other / custom"] += 1

    sorted_m = sorted(metrics.items(), key=lambda x: -x[1])
    labels_m = [k for k, v in sorted_m]
    vals_m = [v for k, v in sorted_m]
    colors_m = ["#1976D2", "#0097A7", "#43A047", "#FB8C00", "#E53935", "#8E24AA", "#9E9E9E"][:len(labels_m)]

    ax1.barh(range(len(labels_m)), vals_m, color=colors_m, edgecolor="black", linewidth=0.5)
    ax1.set_yticks(range(len(labels_m)))
    ax1.set_yticklabels(labels_m, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Number of studies", fontsize=9)
    ax1.set_title("(A) Outcome metrics", fontsize=10, fontweight="bold")
    for i, v in enumerate(vals_m):
        ax1.text(v + 0.3, i, str(v), va="center", fontsize=9, fontweight="bold")
    ax1.set_xlim(0, max(vals_m) * 1.15)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # ===== Panel B: 95% CI reporting (chapter-aligned: 25/59 = 42%) =====
    # Using chapter §3.5 Heterogeneity Criterion B counts: 25 of 59 = 42% of PROBAST-eligible studies
    has_ci = 25
    no_ci = 34
    sizes_ci = [has_ci, no_ci]
    labels_ci = [f"95\\% CI\nreported\n({has_ci})", f"No CI\n({no_ci})"]
    colors_ci = ["#43A047", "#E53935"]
    ax2.pie(sizes_ci, labels=labels_ci, colors=colors_ci, autopct="%1.0f%%",
            startangle=90, textprops={"fontsize": 9}, pctdistance=0.7)
    ax2.set_title("(B) Variance reporting ($n=59$)\n95\\% CIs reported on primary metric",
                  fontsize=10, fontweight="bold")

    # ===== Panel C: 5-criteria checklist =====
    criteria = [
        ("(i)  Common outcome metric ($\\geq$10 share)", "MET", "AUC/AUROC: 23 studies"),
        ("(ii)  Variance/CIs reported", "PARTIAL", f"{has_ci}/60 = 42\\%"),
        ("(iii) Homogeneous prediction horizons", "UNMET", "$\\sim$90\\% unspecified"),
        ("(iv) Comparable populations", "MET", "PPMI 60\\%$\\rightarrow$40\\%"),
        ("(v)  $\\geq$3 studies per cell", "PARTIAL", "8 cells of 25 unique"),
    ]
    status_color = {"MET": LOW, "PARTIAL": MOD, "UNMET": HIGH}
    status_marker = {"MET": "$\\checkmark$", "PARTIAL": "?", "UNMET": "$\\times$"}

    for i, (crit, status, detail) in enumerate(criteria):
        y = len(criteria) - i - 1
        # Status icon
        ax3.add_patch(Rectangle((0, y), 0.7, 0.7, facecolor=status_color[status], edgecolor="black", linewidth=0.6))
        ax3.text(0.35, y + 0.35, status_marker[status], ha="center", va="center",
                 fontsize=14, color="white", fontweight="bold")
        # Criterion text
        ax3.text(0.85, y + 0.5, crit, va="center", fontsize=9.5, fontweight="bold")
        # Detail
        ax3.text(0.85, y + 0.18, detail, va="center", fontsize=8, color="#555", style="italic")

    ax3.set_xlim(-0.1, 6.5)
    ax3.set_ylim(-0.3, len(criteria))
    ax3.axis("off")
    ax3.set_title("(C) Meta-analysis feasibility checklist\n[Verdict: STRATIFIED MA FEASIBLE]",
                  fontsize=10, fontweight="bold", color="#1B5E20")
    ax3.text(3.2, -0.5,
             "4 of 5 criteria are \\textit{Met} or \\textit{Partial} for the dynamic-vs-static head-to-head pool",
             ha="center", fontsize=8.5, style="italic", color="#1565C0")

    out = FIGS / "sr_heterogeneity_60.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Wrote {out.name}")


# ============================================================================
# FIGURE 6: HARVEST PLOT (11 head-to-head studies, extending original 8)
# ============================================================================
def render_harvest_60():
    # 11-study head-to-head data (matching meta_analysis_results.json)
    studies = [
        # (study, year_label, tier, delta_abs, delta_pct, color, era)
        ("Venuto",      "2023", 2, 0.064,  9.8, TIER2, "original"),
        ("Severson",    "2021", 2, None,   12.4, TIER2, "original"),
        ("Dadu",        "2022", 2, 0.070,  9.0, TIER2, "original"),
        ("Lian",        "2024", 2, 0.079,  8.9, TIER2, "original"),
        ("Pan",         "2026", 2, 0.110, 14.1, TIER2, "update"),
        ("Dai",         "2026", 2, 0.090, 13.2, TIER2, "update"),
        ("Basaia",      "2025", 1, 0.070, 10.8, TIER1, "original"),
        ("Hemedan (DT)","2026", 0, 0.285, 42.9, TIER0, "update"),
        ("Mohammadi",   "2026", 0, 0.246, 43.9, TIER0, "update"),
        ("Junaid",      "2023", 0, 0.064,  7.6, TIER0, "original"),
        ("Nilashi",     "2022", 0, 0.052,  6.1, "#9E9E9E", "original"),  # equivalence
        ("Chahine",     "2019", 0, 0.056, 46.7, TIER0, "original"),
    ]

    fig, ax = plt.subplots(figsize=(13, 7))

    tier_groups = [(2, "Tier 2 (External validation)", TIER2),
                    (1, "Tier 1 (Holdout)", TIER1),
                    (0, "Tier 0 (Internal CV)", TIER0)]
    x_pos = 0
    tick_xs = []; tick_labels = []
    group_centers = []; group_starts = []
    for tier, group_label, group_color in tier_groups:
        group_studies = [s for s in studies if s[2] == tier]
        if not group_studies: continue
        gs_x_start = x_pos
        for (study, year, t, delta, pct, color, era) in group_studies:
            ax.bar(x_pos, pct, color=color, edgecolor="#333", linewidth=0.8, width=0.7)
            label = f"{study} {year}"
            ax.text(x_pos, -2, label, rotation=45, ha="right", va="top", fontsize=8.5)
            ax.text(x_pos, pct + 1, f"+{pct:.1f}\\%", ha="center", fontsize=8.5)
            tick_xs.append(x_pos); tick_labels.append(label)
            x_pos += 1
        gs_x_end = x_pos - 1
        group_centers.append((gs_x_start + gs_x_end) / 2)
        group_starts.append(gs_x_end + 0.5)
        x_pos += 0.7  # gap between groups

    # Group brackets
    for i, ((tier, group_label, _), center) in enumerate(zip(tier_groups, group_centers)):
        ax.text(center, 53, group_label, ha="center", fontsize=10, fontweight="bold", color="#333")

    # Median line
    ax.axhline(11.7, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(x_pos - 0.5, 12.5, "Median +11.7\\%  (n=11)", color="red", fontsize=9, ha="right", fontweight="bold")

    # Pooled estimate annotation
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.text(x_pos - 0.5, 50,
            "Pooled $\\Delta$AUC $= +0.117$\n[95\\% CI 0.054–0.179]\n$I^2 = 71\\%$, $n=11$",
            ha="right", fontsize=9, fontweight="bold",
            bbox=dict(facecolor="#E3F2FD", edgecolor="#1565C0", boxstyle="round,pad=0.4"))

    ax.set_ylabel("Relative Improvement (\\%)", fontsize=10)
    ax.set_ylim(-15, 55)
    ax.set_xticks([])
    ax.set_title("Figure 6. Harvest plot --- Effect direction by validation tier ($n=11$ head-to-head pool)",
                 fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Custom legend
    legend_handles = [
        mpatches.Patch(color=TIER2, label="Tier 2 (External, n=6 studies, all favour dynamic)"),
        mpatches.Patch(color=TIER1, label="Tier 1 (Holdout, n=1 favours dynamic)"),
        mpatches.Patch(color=TIER0, label="Tier 0 (Internal CV, n=4 favour dynamic + 1 equivalence)"),
        mpatches.Patch(color="#9E9E9E", label="Equivalence (Nilashi 2022)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5)

    out = FIGS / "sr_harvest_60.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Wrote {out.name}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    rows, deep = load_data()
    print(f"Loaded {len(rows)} CSV rows + {len(deep)} deep extractions\n")

    print("Rendering Figure 1 (PRISMA flow)...")
    render_prisma_flow()

    print("Rendering Figure 2 (Architecture distribution)...")
    render_architecture(rows, deep)

    print("Rendering Figure 3 (PROBAST grid)...")
    render_probast_grid(rows, deep)

    print("Rendering Figure 4 (Validation tier + reporting)...")
    render_validation_60(rows, deep)

    print("Rendering Figure 5 (Heterogeneity 5-criteria)...")
    render_heterogeneity_60(rows)

    print("Rendering Figure 6 (Harvest plot, 11 studies)...")
    render_harvest_60()

    print("\nAll figures rendered. Forest plot (Figure 7) already exists at sr_meta_analysis_forest.pdf.")


if __name__ == "__main__":
    main()
