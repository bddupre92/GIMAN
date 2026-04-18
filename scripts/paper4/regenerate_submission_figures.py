"""Regenerate Fig 1 (CIF bands with visible shading via dual-axis zoom) and
Fig 11 (conformal method comparison with colorblind-safe palette) for the
npj Digital Medicine submission.

Also regenerate Fig 12 (directional coverage) with colorblind-safe palette
since the original red/green pairing is a colorblind failure.

Input data:
    outputs/paper4/expanded/patient_case_studies.json
    outputs/paper4/expanded/conformal_baselines.json
    outputs/paper4/expanded/directional_analysis.json

Output:
    outputs/paper4_submission/figures/fig1_cif_bands_submission.png
    outputs/paper4_submission/figures/fig1_cif_bands_submission.pdf
    outputs/paper4_submission/figures/fig11_conformal_baselines_submission.png
    outputs/paper4_submission/figures/fig11_conformal_baselines_submission.pdf
    outputs/paper4_submission/figures/fig12_directional_submission.png
    outputs/paper4_submission/figures/fig12_directional_submission.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "outputs" / "paper4"
OUT_DIR = REPO / "outputs" / "paper4_submission" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe palette
OI_VERMILLION = "#D55E00"
OI_BLUE_GREEN = "#009E73"
OI_SKY_BLUE   = "#56B4E9"
OI_ORANGE     = "#E69F00"
OI_REDDISH_PURPLE = "#CC79A7"
OI_YELLOW     = "#F0E442"
OI_BLUE       = "#0072B2"
OI_BLACK      = "#000000"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def save_pair(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(out)
        print(f"  wrote {out.relative_to(REPO)}")
    plt.close(fig)


def fig1_cif_bands() -> None:
    """Two-row layout:
    Top row (3 panels):  CIF curve + conformal bands on full [0,1] scale (contextual view)
    Bottom row (3 panels): SAME data zoomed to make bands visible
    """
    cases_path = DATA_DIR / "expanded" / "patient_case_studies.json"
    with open(cases_path) as f:
        all_cases = json.load(f)["cases"]

    target = [3785, 3207, 3960]
    pmap = {c["patno"]: c for c in all_cases}
    selected = [pmap[p] for p in target if p in pmap]
    if len(selected) < 3:
        selected = all_cases[:3]

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5),
                             gridspec_kw={"height_ratios": [1.3, 1.0]})

    panel_labels = ["a", "b", "c"]
    line_color = OI_BLUE
    band_color = OI_SKY_BLUE
    actual_color = OI_VERMILLION
    ti_color = OI_BLUE_GREEN

    for col, case in enumerate(selected):
        time_bins = np.array(case["time_bins_months"])
        cif = np.array(case["cif_curve"])
        lo = np.array(case["band_lower"])
        hi = np.array(case["band_upper"])
        actual_t = case["actual_duration_months"]
        ti = case["timing_interval"]
        src = case["source_stage"]
        dst = case["dest_stage"]
        patno = case["patno"]

        t_plot = np.concatenate([[0], time_bins])
        cif_plot = np.concatenate([[0], cif])
        lo_plot = np.concatenate([[0], lo])
        hi_plot = np.concatenate([[0], hi])

        # ---- TOP: contextual full-scale CIF ----
        ax_top = axes[0, col]
        ax_top.fill_between(t_plot, lo_plot, hi_plot, alpha=0.35, color=band_color,
                            edgecolor=line_color, linewidth=0.5, label="90% conformal band")
        ax_top.plot(t_plot, cif_plot, "-", color=line_color, linewidth=2.0,
                    label=f"CIF: {src}$\\to${dst}", zorder=3)
        ax_top.plot(t_plot, cif_plot, "o", color=line_color, markersize=3, zorder=4)
        if actual_t > 0:
            ax_top.axvline(actual_t, ls="--", color=actual_color, alpha=0.8,
                           linewidth=1.5, label=f"Actual: {actual_t:.0f} mo")
        ax_top.set_title(f"({panel_labels[col]}) Patient {patno}:  {src}$\\to${dst}",
                         fontsize=10, fontweight="bold")
        ax_top.set_ylabel("Cumulative incidence" if col == 0 else "")
        ax_top.set_ylim(-0.03, 1.08)
        ax_top.set_xlim(-2, max(time_bins) + 5)
        ax_top.grid(True, alpha=0.2)
        if col == 0:
            ax_top.legend(fontsize=7, loc="center right", framealpha=0.9)

        # ---- BOTTOM: zoomed early window making bands visible ----
        ax_bot = axes[1, col]
        # Zoom x to 0-60 months (or tighter near actual transition)
        zoom_xlim = (0, min(60, max(time_bins) + 5))
        # Zoom y to whatever makes bands visible: get the vertical extent of the band
        band_width_max = float(np.max(hi_plot - lo_plot))
        # find y-center where bands exist
        mask_zoom = (t_plot >= zoom_xlim[0]) & (t_plot <= zoom_xlim[1])
        if mask_zoom.any():
            y_min_zoom = max(0.0, float(np.min(lo_plot[mask_zoom])) - 0.02)
            y_max_zoom = min(1.05, float(np.max(hi_plot[mask_zoom])) + 0.02)
            # If band is ultra-narrow, force a minimum vertical window so it's visible
            if (y_max_zoom - y_min_zoom) < 0.15:
                y_center = (y_max_zoom + y_min_zoom) / 2
                y_min_zoom = max(0.0, y_center - 0.08)
                y_max_zoom = min(1.05, y_center + 0.08)
        else:
            y_min_zoom, y_max_zoom = -0.02, 0.2

        ax_bot.fill_between(t_plot, lo_plot, hi_plot, alpha=0.45, color=band_color,
                            edgecolor=line_color, linewidth=0.5)
        ax_bot.plot(t_plot, cif_plot, "-o", color=line_color, linewidth=1.8,
                    markersize=4)
        if actual_t > 0 and zoom_xlim[0] <= actual_t <= zoom_xlim[1]:
            ax_bot.axvline(actual_t, ls="--", color=actual_color, alpha=0.8, linewidth=1.5)

        # Timing interval bracket at bottom of zoomed window
        ti_lo = ti["lower_months"]
        ti_hi = ti["upper_months"]
        bracket_y = y_min_zoom + 0.015 * (y_max_zoom - y_min_zoom)
        ax_bot.plot([ti_lo, ti_hi], [bracket_y, bracket_y], "-",
                    color=ti_color, linewidth=3.0, solid_capstyle="butt")
        ax_bot.plot([ti_lo, ti_lo], [bracket_y - 0.01, bracket_y + 0.01],
                    "-", color=ti_color, linewidth=2.0)
        ax_bot.plot([ti_hi, ti_hi], [bracket_y - 0.01, bracket_y + 0.01],
                    "-", color=ti_color, linewidth=2.0)
        ax_bot.text((ti_lo + ti_hi) / 2, bracket_y + 0.02 * (y_max_zoom - y_min_zoom),
                    f"TI: [{ti_lo:.0f}, {ti_hi:.0f}] mo", ha="center", va="bottom",
                    fontsize=8, color=ti_color, fontweight="bold")
        ax_bot.set_xlim(*zoom_xlim)
        ax_bot.set_ylim(y_min_zoom, y_max_zoom)
        ax_bot.set_xlabel("Months since baseline")
        ax_bot.set_ylabel("CIF (zoomed)" if col == 0 else "")
        ax_bot.set_title(f"zoom: y$\\in$[{y_min_zoom:.2f}, {y_max_zoom:.2f}],  "
                         f"band w$\\leq${band_width_max:.3f}",
                         fontsize=8, style="italic")
        ax_bot.grid(True, alpha=0.2)

    fig.suptitle("Conformal CIF prediction bands (90% confidence level)",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout()
    save_pair(fig, "fig1_cif_bands_submission")


def fig11_conformal_baselines() -> None:
    """Two-panel bar chart, Okabe-Ito colorblind-safe palette, width annotations on bars."""
    path = DATA_DIR / "expanded" / "conformal_baselines.json"
    with open(path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Okabe-Ito: proposed = bluish-green; baselines = blue/orange; bonferroni = vermillion
    palette = {
        "IPCW (proposed)":  OI_BLUE_GREEN,
        "Marginal (pooled)": OI_BLUE,
        "Naive (no IPCW)":  OI_ORANGE,
        "Bonferroni":       OI_VERMILLION,
    }
    short_names = {
        "IPCW (proposed)": "IPCW\n(proposed)",
        "Marginal (pooled)": "Marginal\n(pooled)",
        "Naive (no IPCW)": "Naive\n(no IPCW)",
        "Bonferroni": "Bonferroni",
    }

    for ax_idx, cl_key in enumerate(["CL=0.9", "CL=0.95"]):
        if cl_key not in data:
            continue
        cl_data = data[cl_key]
        methods = list(cl_data.keys())
        x = np.arange(len(methods))
        covs = [cl_data[m]["mean_coverage"] for m in methods]
        cov_stds = [cl_data[m]["std_coverage"] for m in methods]
        widths = [cl_data[m]["mean_width"] for m in methods]
        bar_colors = [palette.get(m, OI_SKY_BLUE) for m in methods]

        ax = axes[ax_idx]
        ax.bar(x, covs, yerr=cov_stds, color=bar_colors, alpha=0.85, capsize=4,
               edgecolor=OI_BLACK, linewidth=0.5)

        cl_val = float(cl_key.split("=")[1])
        ax.axhline(y=cl_val, color=OI_BLACK, linestyle=":", alpha=0.7,
                   linewidth=1.3, label=f"Target = {cl_val}")

        # width annotation above each bar (not crammed under error bar)
        for i, w in enumerate(widths):
            upper_y = covs[i] + cov_stds[i] + 0.018
            ax.text(i, upper_y, f"w={w:.4f}", ha="center", va="bottom",
                    fontsize=8, style="italic", color=OI_BLACK)

        ax.set_xticks(x)
        ax.set_xticklabels([short_names.get(m, m) for m in methods], fontsize=9)
        ax.set_ylabel("Marginal coverage")
        ax.set_title(f"Conformal methods ({cl_key})", fontsize=10, fontweight="bold")
        ax.set_ylim(0.70, 1.08)
        ax.grid(True, axis="y", alpha=0.2)
        ax.legend(fontsize=8, loc="lower center")

    fig.suptitle("Conformal method comparison: coverage vs band width (Okabe-Ito palette)",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    save_pair(fig, "fig11_conformal_baselines_submission")


def fig12_directional_coverage() -> None:
    """Recolored directional coverage with colorblind-safe palette."""
    path = DATA_DIR / "expanded" / "directional_analysis.json"
    with open(path) as f:
        data = json.load(f)

    # Aggregate per-fold entries into mean/std per (model, direction)
    per_fold = data["per_fold"]
    def _stats(model: str, direction: str):
        vals = [e["coverage"] for e in per_fold
                if e["model"] == model and e["direction"] == direction]
        return float(np.mean(vals)), float(np.std(vals, ddof=1)), vals

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Palette: forward = bluish-green; backward = orange (distinguishable in all colorblind types)
    forward_color = OI_BLUE_GREEN
    backward_color = OI_ORANGE

    # Left: coverage by direction × model
    models = ["DeepHit", "Graph-DT"]
    width = 0.35
    x = np.arange(len(models))

    fwd_cov, fwd_std = [], []
    bwd_cov, bwd_std = [], []
    for m in models:
        fm, fs, _ = _stats(m, "forward")
        bm, bs, _ = _stats(m, "backward")
        fwd_cov.append(fm); fwd_std.append(fs)
        bwd_cov.append(bm); bwd_std.append(bs)

    n_fwd_total = sum(e["n_patients"] for e in per_fold
                      if e["direction"] == "forward" and e["model"] == "DeepHit")
    n_bwd_total = sum(e["n_patients"] for e in per_fold
                      if e["direction"] == "backward" and e["model"] == "DeepHit")

    ax = axes[0]
    ax.bar(x - width/2, fwd_cov, width, yerr=fwd_std, color=forward_color,
           label="Forward (progression)", capsize=4, edgecolor=OI_BLACK, linewidth=0.5)
    ax.bar(x + width/2, bwd_cov, width, yerr=bwd_std, color=backward_color,
           label="Backward (regression)", capsize=4, edgecolor=OI_BLACK, linewidth=0.5)
    ax.axhline(0.90, color=OI_BLACK, linestyle=":", alpha=0.7,
               linewidth=1.3, label="90% target")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage by transition direction (90% CL)",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(0.50, 1.0)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.2)

    # Right: distribution
    ax = axes[1]
    n_fwd = n_fwd_total
    n_bwd = n_bwd_total
    total = n_fwd + n_bwd
    bars = ax.bar(["Forward\n(progression)", "Backward\n(regression)"],
                  [n_fwd, n_bwd], color=[forward_color, backward_color],
                  edgecolor=OI_BLACK, linewidth=0.5)
    for b, count in zip(bars, [n_fwd, n_bwd]):
        pct = 100 * count / total
        ax.text(b.get_x() + b.get_width()/2, b.get_height() * 1.01,
                f"{count}\n({pct:.0f}%)", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylabel("Total evaluation patients")
    ax.set_title("Transition direction distribution",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(n_fwd, n_bwd) * 1.15)
    ax.grid(True, axis="y", alpha=0.2)

    fig.suptitle("Directional analysis: forward vs backward conformal coverage",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    save_pair(fig, "fig12_directional_submission")


if __name__ == "__main__":
    print("Regenerating submission figures...")
    fig1_cif_bands()
    fig11_conformal_baselines()
    fig12_directional_coverage()
    print("Done.")
