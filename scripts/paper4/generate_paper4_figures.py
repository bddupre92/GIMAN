#!/usr/bin/env python3
"""Generate all Paper 4 publication figures.

10 figures covering conformal CIF bands, calibration, and subgroup equity.

Outputs:
    outputs/paper4/figures/
        fig1_conformal_cif_bands.{png,pdf}
        fig2_coverage_calibration.{png,pdf}
        fig3_interval_width_by_transition.{png,pdf}
        fig4_timing_intervals.{png,pdf}
        fig5_reliability_diagram.{png,pdf}
        fig6_ece_comparison.{png,pdf}
        fig7_subgroup_forest_plot.{png,pdf}
        fig8_subgroup_interaction.{png,pdf}
        fig9_gate_activation.{png,pdf}
        fig10_conditional_coverage.{png,pdf}
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.paper3.multistate_markov import N_STATES, STAGE_LABELS

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper4" / "figures"
DATA_DIR = PROJECT_ROOT / "outputs" / "paper4"

# Publication style
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

COLORS = {
    "DeepHit": "#2196F3",
    "Graph-DT": "#FF5722",
}
STAGE_COLORS = plt.cm.Set2(np.linspace(0, 1, N_STATES))


def save_fig(fig, name):
    """Save figure in both PNG and PDF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.png")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  Saved {name}.{{png,pdf}}")


# --- Figure 1: Conformal CIF bands for example patient ---


def fig1_conformal_cif_bands():
    """Example patient CIF curves with 90% conformal prediction bands.

    Uses pre-computed case studies from patient_case_studies.json which contain
    actual CIF curves, conformal band bounds, and timing intervals.
    """
    cases_path = DATA_DIR / "expanded" / "patient_case_studies.json"
    if not cases_path.exists():
        print("  SKIP fig1: patient_case_studies.json not found")
        return

    with open(cases_path) as f:
        all_cases = json.load(f)["cases"]

    if len(all_cases) < 3:
        print(f"  SKIP fig1: only {len(all_cases)} case studies (need 3)")
        return

    # Select 3 visually diverse patients:
    #   - Patient 3785: slow 2B->3 (54 months, beautiful S-curve)
    #   - Patient 3207: rapid 2B->3 (7.2 months, sharp rise)
    #   - Patient 3960: 2B->4 (18 months, different destination)
    target_patnos = [3785, 3207, 3960]
    patno_map = {c["patno"]: c for c in all_cases}
    selected = [patno_map[p] for p in target_patnos if p in patno_map]

    # Fallback: use first 3 cases if target patients not found
    if len(selected) < 3:
        selected = all_cases[:3]

    panel_colors = ["#2196F3", "#4CAF50", "#FF9800"]
    panel_labels = ["(a)", "(b)", "(c)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax_idx, case in enumerate(selected):
        ax = axes[ax_idx]
        color = panel_colors[ax_idx]

        time_bins = np.array(case["time_bins_months"])
        cif = np.array(case["cif_curve"])
        lo = np.array(case["band_lower"])
        hi = np.array(case["band_upper"])
        actual_t = case["actual_duration_months"]
        ti = case["timing_interval"]
        src = case["source_stage"]
        dst = case["dest_stage"]
        patno = case["patno"]

        # Prepend origin (0, 0) for smooth plotting from time 0
        t_plot = np.concatenate([[0], time_bins])
        cif_plot = np.concatenate([[0], cif])
        lo_plot = np.concatenate([[0], lo])
        hi_plot = np.concatenate([[0], hi])

        # Conformal band (shaded region)
        ax.fill_between(
            t_plot,
            lo_plot,
            hi_plot,
            alpha=0.25,
            color=color,
            label="90% conformal band",
        )

        # CIF curve
        ax.plot(
            t_plot,
            cif_plot,
            "-o",
            color=color,
            markersize=3,
            linewidth=1.8,
            label=f"CIF: {src} $\\to$ {dst}",
            zorder=3,
        )

        # Actual transition time (vertical dashed line)
        if actual_t > 0:
            ax.axvline(
                actual_t,
                ls="--",
                color="red",
                alpha=0.7,
                linewidth=1.2,
                label=f"Actual: {actual_t:.0f} mo",
            )
        else:
            # Transition at baseline — mark with arrow at x=0
            ax.annotate(
                "Actual\n(baseline)",
                xy=(1, 0.5),
                fontsize=7,
                color="red",
                ha="left",
                va="center",
            )

        # Timing interval (horizontal bracket)
        ti_lo = ti["lower_months"]
        ti_hi = ti["upper_months"]
        bracket_y = 0.08
        ax.plot(
            [ti_lo, ti_hi],
            [bracket_y, bracket_y],
            "-",
            color="darkred",
            linewidth=2.5,
            alpha=0.6,
            solid_capstyle="butt",
        )
        ax.plot(
            [ti_lo, ti_lo],
            [bracket_y - 0.03, bracket_y + 0.03],
            "-",
            color="darkred",
            linewidth=1.5,
            alpha=0.6,
        )
        ax.plot(
            [ti_hi, ti_hi],
            [bracket_y - 0.03, bracket_y + 0.03],
            "-",
            color="darkred",
            linewidth=1.5,
            alpha=0.6,
        )
        ax.text(
            (ti_lo + ti_hi) / 2,
            bracket_y + 0.06,
            f"TI: [{ti_lo:.0f}, {ti_hi:.0f}] mo",
            ha="center",
            va="bottom",
            fontsize=7,
            color="darkred",
        )

        ax.set_title(
            f"{panel_labels[ax_idx]} Patient {patno}: Stage {src} $\\to$ {dst}",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("Months since baseline")
        if ax_idx == 0:
            ax.set_ylabel("Cumulative Incidence")
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlim(-2, max(time_bins) + 5)
        ax.legend(fontsize=7, loc="center right")
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Conformal CIF Prediction Bands (90% Confidence Level)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "fig1_conformal_cif_bands")


# --- Figure 2: Coverage calibration across confidence levels ---


def fig2_coverage_calibration():
    """Predicted vs actual coverage across alpha levels."""
    agg_path = DATA_DIR / "conformal" / "aggregate_summary.json"
    if not agg_path.exists():
        print("  SKIP fig2: aggregate summary not found")
        return

    with open(agg_path) as f:
        agg = json.load(f)

    fig, ax = plt.subplots(figsize=(5, 4.5))

    for model_name, color in COLORS.items():
        if model_name not in agg:
            continue
        cov_data = agg[model_name]["marginal_coverage"]
        targets = []
        means = []
        stds = []
        for cl_str, vals in sorted(cov_data.items()):
            targets.append(float(cl_str))
            means.append(vals["mean"])
            stds.append(vals["std"])

        ax.errorbar(
            targets,
            means,
            yerr=stds,
            fmt="o-",
            color=color,
            label=model_name,
            capsize=4,
            markersize=6,
        )

    ax.plot([0.75, 1.0], [0.75, 1.0], "k--", alpha=0.5, label="Ideal")
    ax.set_xlabel("Target Coverage (1 - α)")
    ax.set_ylabel("Actual Marginal Coverage")
    ax.set_title("Coverage Calibration")
    ax.legend()
    ax.set_xlim(0.75, 1.0)
    ax.set_ylim(0.75, 1.0)
    ax.set_aspect("equal")

    fig.tight_layout()
    save_fig(fig, "fig2_coverage_calibration")


# --- Figure 3: Interval width by transition ---


def fig3_interval_width_by_transition():
    """Box plot of conformal band widths per destination stage."""
    agg_path = DATA_DIR / "conformal" / "aggregate_summary.json"
    if not agg_path.exists():
        print("  SKIP fig3: data not found")
        return

    # Load per-fold results for width distributions
    fig, ax = plt.subplots(figsize=(8, 4))

    for model_file, model_name in [
        ("conformal_results_deephit.json", "DeepHit"),
        ("conformal_results_graph_dt.json", "Graph-DT"),
    ]:
        path = DATA_DIR / "conformal" / model_file
        if not path.exists():
            continue
        with open(path) as f:
            results = json.load(f)

        # Get 90% results
        r90 = [r for r in results if abs(r["confidence_level"] - 0.90) < 0.01]
        if not r90:
            continue

        widths_per_cause = {}
        for r in r90:
            for k_str, w in r["per_cause_band_width"].items():
                k = int(k_str)
                widths_per_cause.setdefault(k, []).append(w)

        causes = sorted(widths_per_cause.keys())
        x = np.arange(len(causes))
        offset = -0.2 if model_name == "DeepHit" else 0.2
        means = [np.mean(widths_per_cause[k]) for k in causes]
        stds = [np.std(widths_per_cause[k]) for k in causes]

        ax.bar(
            x + offset,
            means,
            0.35,
            yerr=stds,
            label=model_name,
            color=COLORS[model_name],
            alpha=0.8,
            capsize=3,
        )

    ax.set_xticks(range(len(causes)))
    ax.set_xticklabels([STAGE_LABELS[k] for k in causes])
    ax.set_xlabel("Destination Stage")
    ax.set_ylabel("Mean Band Width")
    ax.set_title("Conformal Band Width by Transition Destination (90% CL)")
    ax.legend()

    fig.tight_layout()
    save_fig(fig, "fig3_interval_width_by_transition")


# --- Figure 4: Timing intervals per cause ---


def fig4_timing_intervals():
    """Per-cause timing interval coverage and width for major transitions."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for model_file, model_name in [
        ("timing_intervals_deephit.json", "DeepHit"),
        ("timing_intervals_graph_dt.json", "Graph-DT"),
    ]:
        path = DATA_DIR / "conformal" / model_file
        if not path.exists():
            print(f"  SKIP fig4: {model_file} not found")
            continue

        with open(path) as f:
            results = json.load(f)

        # Get 90% results
        r90 = [r for r in results if abs(r["confidence_level"] - 0.90) < 0.01]
        if not r90:
            continue

        # Aggregate per cause across folds
        cov_per_cause = {}
        width_per_cause = {}
        n_per_cause = {}
        for r in r90:
            for k_str, cov in r["timing_coverage"].items():
                k = int(k_str)
                cov_per_cause.setdefault(k, []).append(cov)
            for k_str, w in r["median_interval_width_months"].items():
                k = int(k_str)
                if w != float("inf") and w < 1000:
                    width_per_cause.setdefault(k, []).append(w)
            for k_str, n in r["n_uncensored_per_cause"].items():
                k = int(k_str)
                n_per_cause.setdefault(k, []).append(n)

        # Only plot causes with sufficient events
        major = [
            k
            for k in sorted(cov_per_cause.keys())
            if k in n_per_cause and np.mean(n_per_cause[k]) >= 10
        ]
        if not major:
            continue

        x = np.arange(len(major))
        offset = -0.18 if model_name == "DeepHit" else 0.18

        # Coverage panel
        covs = [np.mean(cov_per_cause[k]) for k in major]
        cov_stds = [np.std(cov_per_cause[k]) for k in major]
        axes[0].bar(
            x + offset,
            covs,
            0.32,
            yerr=cov_stds,
            color=COLORS[model_name],
            alpha=0.8,
            capsize=3,
            label=model_name,
        )

        # Width panel
        widths = [np.mean(width_per_cause.get(k, [0])) for k in major]
        width_stds = [np.std(width_per_cause.get(k, [0])) for k in major]
        axes[1].bar(
            x + offset,
            widths,
            0.32,
            yerr=width_stds,
            color=COLORS[model_name],
            alpha=0.8,
            capsize=3,
            label=model_name,
        )

    if not major:
        plt.close(fig)
        return

    labels = [STAGE_LABELS[k] for k in major]
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].axhline(y=0.90, color="red", linestyle="--", alpha=0.5, label="90% target")
    axes[0].set_ylabel("Timing Coverage")
    axes[0].set_title("Transition Timing Coverage (90% CL)")
    axes[0].legend(fontsize=7)
    axes[0].set_ylim(0, 1.05)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Median Interval Width (months)")
    axes[1].set_title("Timing Interval Width")
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    save_fig(fig, "fig4_timing_intervals")


# --- Figure 5: Reliability diagram ---


def fig5_reliability_diagram():
    """Predicted CIF vs observed proportion at 1, 3, 5 years."""
    for model_file, model_name in [
        ("calibration_results_deephit.json", "DeepHit"),
        ("calibration_results_graph_dt.json", "Graph-DT"),
    ]:
        path = DATA_DIR / "calibration" / model_file
        if not path.exists():
            print(f"  SKIP fig5: {model_file} not found")
            continue

        with open(path) as f:
            results = json.load(f)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        horizons = ["1yr", "3yr", "5yr"]

        for ax_idx, h in enumerate(horizons):
            ax = axes[ax_idx]

            # Aggregate reliability data across folds
            all_bins = {}  # cause -> list of (pred, obs) per bin
            for r in results:
                if h not in r.get("reliability_data", {}):
                    continue
                for row in r["reliability_data"][h]:
                    k = row["cause"]
                    bc = row["bin_center"]
                    key = (k, round(bc, 2))
                    if key not in all_bins:
                        all_bins[key] = {"preds": [], "obs": []}
                    all_bins[key]["preds"].append(row["predicted"])
                    all_bins[key]["obs"].append(row["observed"])

            # Plot per major cause
            for k in [2, 3, 4]:
                bins = [
                    (bc, np.mean(v["preds"]), np.mean(v["obs"]))
                    for (cause, bc), v in sorted(all_bins.items())
                    if cause == k
                ]
                if not bins:
                    continue
                bcs, preds, obs = zip(*bins, strict=False)
                ax.plot(
                    preds,
                    obs,
                    "o-",
                    color=STAGE_COLORS[k],
                    label=STAGE_LABELS[k],
                    markersize=4,
                    alpha=0.8,
                )

            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=0.8)
            ax.set_xlabel("Predicted CIF")
            ax.set_ylabel("Observed Proportion")
            ax.set_title(f"{h} Horizon")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect("equal")
            if ax_idx == 0:
                ax.legend(fontsize=7)

        fig.suptitle(f"Reliability Diagrams — {model_name}", fontsize=12, y=1.02)
        fig.tight_layout()
        save_fig(
            fig, f"fig5_reliability_diagram_{model_name.lower().replace('-', '_')}"
        )


# --- Figure 6: ECE comparison ---


def fig6_ece_comparison():
    """Bar chart: DeepHit vs Graph-DT ECE per cause."""
    agg_path = DATA_DIR / "calibration" / "aggregate_ece.json"
    if not agg_path.exists():
        print("  SKIP fig6: data not found")
        return

    with open(agg_path) as f:
        agg = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    horizons = ["1yr", "3yr", "5yr"]

    for ax_idx, h in enumerate(horizons):
        ax = axes[ax_idx]

        for model_name, color in COLORS.items():
            if model_name not in agg or h not in agg[model_name]:
                continue
            val = agg[model_name][h]
            ax.bar(
                model_name,
                val["mean"],
                yerr=val["std"],
                color=color,
                alpha=0.8,
                capsize=4,
            )

        ax.set_title(f"{h} Horizon")
        ax.set_ylabel("ECE")

    fig.suptitle("Expected Calibration Error by Model and Horizon", fontsize=12, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig6_ece_comparison")


# --- Figure 7: Subgroup forest plot ---


def fig7_subgroup_forest_plot():
    """Per-subgroup C-td with error bars for both models."""
    path = DATA_DIR / "subgroup" / "subgroup_ctd.json"
    if not path.exists():
        print("  SKIP fig7: data not found")
        return

    with open(path) as f:
        ctd_data = json.load(f)

    # Aggregate across folds
    agg = {}
    for r in ctd_data:
        model = r["model_name"]
        var = r["subgroup_var"]
        for group, ctd in r["per_group_ctd"].items():
            key = (model, var, group)
            agg.setdefault(key, []).append(ctd)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    vars_list = ["lrrk2", "gba", "sex", "age"]

    for ax_idx, var_name in enumerate(vars_list):
        ax = axes[ax_idx // 2, ax_idx % 2]

        groups = sorted(set(g for (m, v, g) in agg if v == var_name))
        y_pos = np.arange(len(groups))

        for model_idx, (model_name, color) in enumerate(COLORS.items()):
            offset = -0.15 + model_idx * 0.3
            for g_idx, group in enumerate(groups):
                key = (model_name, var_name, group)
                if key in agg:
                    vals = agg[key]
                    mean = np.mean(vals)
                    ci = 1.96 * np.std(vals) / max(np.sqrt(len(vals)), 1)
                    ax.errorbar(
                        mean,
                        g_idx + offset,
                        xerr=ci,
                        fmt="o",
                        color=color,
                        capsize=3,
                        markersize=5,
                        label=model_name if g_idx == 0 else "",
                    )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(groups)
        ax.set_xlabel("C-td")
        ax.set_title(f"Subgroup: {var_name.upper()}")
        ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.3)
        if ax_idx == 0:
            ax.legend()

    fig.suptitle("Per-Subgroup C-td (Forest Plot)", fontsize=12, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig7_subgroup_forest_plot")


# --- Figure 8: Subgroup interaction (delta C-td) ---


def fig8_subgroup_interaction():
    """Delta C-td (Graph-DT minus DeepHit) by subgroup."""
    path = DATA_DIR / "subgroup" / "interaction_tests.json"
    if not path.exists():
        print("  SKIP fig8: interaction_tests.json not found")
        return

    with open(path) as f:
        int_data = json.load(f)

    # Aggregate delta per (var, group)
    agg = {}
    for r in int_data:
        var = r["subgroup_var"]
        for group, delta in r["delta_ctd_per_group"].items():
            key = (var, group)
            agg.setdefault(key, []).append(delta)

    if not agg:
        print("  SKIP fig8: no interaction data")
        return

    vars_list = sorted(set(v for (v, _) in agg))
    fig, ax = plt.subplots(figsize=(8, 5))

    y_labels = []
    y_pos = []
    pos = 0
    for var_name in vars_list:
        groups = sorted(set(g for (v, g) in agg if v == var_name))
        for group in groups:
            key = (var_name, group)
            vals = agg[key]
            mean = np.mean(vals)
            ci = 1.96 * np.std(vals) / max(np.sqrt(len(vals)), 1)
            color = "#4CAF50" if mean > 0 else "#F44336"
            ax.errorbar(
                mean, pos, xerr=ci, fmt="o", color=color, capsize=4, markersize=6
            )
            y_labels.append(f"{var_name}: {group}")
            y_pos.append(pos)
            pos += 1
        pos += 0.5  # gap between vars

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("$\\Delta$ C-td (Graph-DT $-$ DeepHit)")
    ax.set_title("Subgroup Interaction: Model Advantage by Subgroup")
    ax.invert_yaxis()

    fig.tight_layout()
    save_fig(fig, "fig8_subgroup_interaction")


# --- Figure 9: Gate activation by subgroup ---


def fig9_gate_activation():
    """Gate activation distribution from Graph-DT by subgroup."""
    # Gate activations need checkpoint loading — use subgroup C-td as proxy
    # if gate data not available separately
    import torch

    CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "paper3_checkpoints"
    FEATURES_PATH = (
        PROJECT_ROOT / "data" / "07_paper3_features" / "longitudinal_features.csv"
    )

    gdt_path = CHECKPOINT_DIR / "graph_dt" / "fold0_graph_dt.pt"
    if not gdt_path.exists():
        print("  SKIP fig9: Graph-DT checkpoint not found")
        return

    try:
        from giman_pipeline.paper3.dynamic_deephit import (
            build_patient_arrays,
            extract_episodes,
        )
        from giman_pipeline.paper3.graph_digital_twin import (
            GraphDeepHitDataset,
            load_graph_dt_checkpoint,
        )
        from giman_pipeline.paper4.subgroup import assign_subgroups

        features_df = pd.read_csv(FEATURES_PATH, low_memory=False)
        patient_arrays, _ = build_patient_arrays(features_df)
        episodes = extract_episodes(features_df, verbose=False)

        model, cp = load_graph_dt_checkpoint(gdt_path)
        test_pats = set(cp["test_pats"])
        means, stds = cp["means"], cp["stds"]
        pat_to_gidx = cp["pat_to_gidx"]

        test_eps = [e for e in episodes if e.patno in test_pats]
        test_ds = GraphDeepHitDataset(
            test_eps, patient_arrays, means, stds, pat_to_gidx
        )

        # Extract gate activations
        device = next(model.parameters()).device
        gate_vals = []
        pat_ids = []

        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        from torch.utils.data import DataLoader

        from giman_pipeline.paper3.graph_digital_twin import graph_collate_fn

        loader = DataLoader(
            test_ds, batch_size=64, shuffle=False, collate_fn=graph_collate_fn
        )

        # Pre-compute GAT graph features once (same as model.compute_graph_features)
        node_baseline = cp["node_baseline"].to(device)
        edge_index = cp["edge_index"].to(device)
        edge_weight = cp["edge_weight"].to(device)

        model.eval()
        with torch.no_grad():
            graph_node_features = model.compute_graph_features(
                node_baseline,
                edge_index,
                edge_weight,
            )  # (N_patients, hidden_dim)

            for batch in loader:
                sequences = batch["sequences"].to(device)
                seq_lens = batch["seq_lens"].to(device)
                graph_idxs = batch["graph_idxs"].to(device)

                # 1. Temporal encoding (mirrors model.forward())
                sorted_lens, sort_idx = seq_lens.sort(descending=True)
                sorted_seqs = sequences[sort_idx]
                sorted_lens_clamped = sorted_lens.clamp(min=1)
                packed = pack_padded_sequence(
                    sorted_seqs,
                    sorted_lens_clamped.cpu(),
                    batch_first=True,
                )
                gru_out, h_n = model.gru(packed)
                gru_out_padded, _ = pad_packed_sequence(gru_out, batch_first=True)
                _, unsort_idx = sort_idx.sort()
                gru_out_unsorted = gru_out_padded[unsort_idx]
                last_hidden = h_n[-1][unsort_idx]

                attn_ctx = model.temporal_attn(gru_out_unsorted, seq_lens)
                temporal = attn_ctx + last_hidden

                # 2. Graph features for this batch
                graph_feat = graph_node_features[graph_idxs]

                # 3. Gate activation (sigmoid of gate_linear)
                gate_input = torch.cat([temporal, graph_feat], dim=-1)
                g = torch.sigmoid(model.gate_linear(gate_input))

                # Average gate across hidden_dim for a scalar per patient
                gate_scalar = g.mean(dim=-1).cpu().numpy().tolist()
                gate_vals.extend(gate_scalar)
                pat_ids.extend(batch["patnos"])

        subgroup_assignments = assign_subgroups(pat_ids, features_df)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax_idx, var_name in enumerate(["sex", "age"]):
            ax = axes[ax_idx]
            if var_name not in subgroup_assignments:
                continue
            pat_groups = subgroup_assignments[var_name]
            groups = sorted(set(pat_groups.values()))

            data_per_group = []
            labels = []
            for group in groups:
                g_vals = [
                    gate_vals[i]
                    for i, p in enumerate(pat_ids)
                    if pat_groups.get(p) == group
                ]
                if g_vals:
                    data_per_group.append(g_vals)
                    labels.append(group)

            if data_per_group:
                parts = ax.violinplot(
                    data_per_group,
                    positions=range(len(labels)),
                    showmedians=True,
                    showextrema=False,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(COLORS["Graph-DT"])
                    pc.set_alpha(0.6)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels)
                ax.set_ylabel("Gate Activation (σ)")
                ax.set_title(f"Gate by {var_name.upper()}")

        fig.suptitle(
            "Graph-DT Gate Activation by Subgroup (Fold 0)", fontsize=12, y=1.02
        )
        fig.tight_layout()
        save_fig(fig, "fig9_gate_activation")

    except Exception as e:
        print(f"  SKIP fig9: {e}")


# --- Figure 10: Conditional coverage by subgroup ---


def fig10_conditional_coverage():
    """Conditional conformal coverage per subgroup (equity analysis)."""
    path = DATA_DIR / "subgroup" / "conditional_coverage.json"
    if not path.exists():
        print("  SKIP fig10: data not found")
        return

    with open(path) as f:
        cc_data = json.load(f)

    # Parse results
    agg = {}
    for key, coverages in cc_data.items():
        parts = key.split("_")
        # Format: "ModelName_varname_foldN"
        model = parts[0]
        fold_str = parts[-1]
        var_name = "_".join(parts[1:-1])

        for group, cov in coverages.items():
            agg_key = (model, var_name, group)
            agg.setdefault(agg_key, []).append(cov)

    if not agg:
        print("  SKIP fig10: no data parsed")
        return

    vars_list = sorted(set(v for (_, v, _) in agg))

    fig, axes = plt.subplots(1, len(vars_list), figsize=(4 * len(vars_list), 4))
    if len(vars_list) == 1:
        axes = [axes]

    for ax_idx, var_name in enumerate(vars_list):
        ax = axes[ax_idx]
        groups = sorted(set(g for (m, v, g) in agg if v == var_name))
        x = np.arange(len(groups))

        for model_idx, (model_name, color) in enumerate(COLORS.items()):
            offset = -0.15 + model_idx * 0.3
            means = []
            for group in groups:
                key = (model_name, var_name, group)
                if key in agg:
                    means.append(np.mean(agg[key]))
                else:
                    means.append(0)
            ax.bar(
                x + offset,
                means,
                0.25,
                color=color,
                alpha=0.8,
                label=model_name if ax_idx == 0 else "",
            )

        ax.axhline(
            y=0.90,
            color="red",
            linestyle="--",
            alpha=0.5,
            label="Target" if ax_idx == 0 else "",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha="right")
        ax.set_ylabel("Coverage")
        ax.set_title(var_name.upper())
        ax.set_ylim(0.5, 1.0)

    if len(vars_list) > 0:
        axes[0].legend(fontsize=7)

    fig.suptitle("Conditional Conformal Coverage by Subgroup", fontsize=12, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig10_conditional_coverage")


# --- Figure 11: Conformal Baselines Comparison ---


def fig11_conformal_baselines():
    """Bar chart comparing 4 conformal methods: coverage vs band width."""
    path = DATA_DIR / "expanded" / "conformal_baselines.json"
    if not path.exists():
        print("  SKIP fig11: conformal_baselines.json not found")
        return

    with open(path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, cl_key in enumerate(["CL=0.9", "CL=0.95"]):
        if cl_key not in data:
            continue

        cl_data = data[cl_key]
        methods = list(cl_data.keys())
        # Short names for display
        short_names = {
            "IPCW (proposed)": "IPCW\n(proposed)",
            "Marginal (pooled)": "Marginal\n(pooled)",
            "Naive (no IPCW)": "Naive\n(no IPCW)",
            "Bonferroni": "Bonferroni",
        }

        x = np.arange(len(methods))
        covs = [cl_data[m]["mean_coverage"] for m in methods]
        cov_stds = [cl_data[m]["std_coverage"] for m in methods]
        widths = [cl_data[m]["mean_width"] for m in methods]
        width_stds = [cl_data[m]["std_width"] for m in methods]

        # Use log scale for width since Bonferroni is 70x larger
        bar_colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]

        ax = axes[ax_idx]
        bars = ax.bar(x, covs, yerr=cov_stds, color=bar_colors, alpha=0.8, capsize=4)

        # Add width annotation below each bar
        for i, (w, ws) in enumerate(zip(widths, width_stds, strict=False)):
            ax.text(
                i,
                covs[i] - cov_stds[i] - 0.035,
                f"w={w:.4f}",
                ha="center",
                va="top",
                fontsize=7,
                style="italic",
            )

        cl_val = float(cl_key.split("=")[1])
        ax.axhline(
            y=cl_val, color="red", linestyle="--", alpha=0.5, label=f"Target ({cl_val})"
        )
        ax.set_xticks(x)
        ax.set_xticklabels([short_names.get(m, m) for m in methods], fontsize=8)
        ax.set_ylabel("Marginal Coverage")
        ax.set_title(f"Conformal Methods ({cl_key})")
        ax.set_ylim(0.65, 1.05)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Conformal Method Comparison: Coverage vs Band Width", fontsize=12, y=1.02
    )
    fig.tight_layout()
    save_fig(fig, "fig11_conformal_baselines")


# --- Figure 12: Directional Coverage (Forward vs Backward) ---


def fig12_directional_coverage():
    """Coverage comparison for forward vs backward transitions."""
    path = DATA_DIR / "expanded" / "directional_analysis.json"
    if not path.exists():
        print("  SKIP fig12: directional_analysis.json not found")
        return

    with open(path) as f:
        data = json.load(f)

    per_fold = data["per_fold"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: Coverage by direction and model
    for model_idx, model_name in enumerate(["DeepHit", "Graph-DT"]):
        for dir_idx, direction in enumerate(["forward", "backward"]):
            vals = [
                r["coverage"]
                for r in per_fold
                if r["model"] == model_name and r["direction"] == direction
            ]
            x_pos = model_idx + dir_idx * 0.35 - 0.15
            color = "#4CAF50" if direction == "forward" else "#FF5722"
            axes[0].bar(
                x_pos,
                np.mean(vals),
                0.30,
                yerr=np.std(vals),
                color=color,
                alpha=0.8,
                capsize=4,
                label=f"{direction.capitalize()}" if model_idx == 0 else "",
            )

    axes[0].axhline(y=0.90, color="red", linestyle="--", alpha=0.5)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["DeepHit", "Graph-DT"])
    axes[0].set_ylabel("Coverage")
    axes[0].set_title("Coverage by Transition Direction")
    axes[0].legend()
    axes[0].set_ylim(0.5, 1.0)

    # Panel B: Number of patients by direction
    summary = data["summary"]
    dirs = ["forward", "backward"]
    counts = [summary[d]["total_patients"] for d in dirs]
    colors = ["#4CAF50", "#FF5722"]
    axes[1].bar(range(2), counts, color=colors, alpha=0.8)
    axes[1].set_xticks(range(2))
    axes[1].set_xticklabels(["Forward\n(progression)", "Backward\n(regression)"])
    axes[1].set_ylabel("Total Evaluation Patients")
    axes[1].set_title("Transition Direction Distribution")

    for i, c in enumerate(counts):
        pct = c / sum(counts) * 100
        axes[1].text(i, c + 10, f"{c}\n({pct:.0f}%)", ha="center", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "fig12_directional_coverage")


# --- Figure 13: Patient Case Studies ---


def fig13_patient_case_studies():
    """Individual patient CIF curves with conformal bands."""
    path = DATA_DIR / "expanded" / "patient_case_studies.json"
    if not path.exists():
        print("  SKIP fig13: patient_case_studies.json not found")
        return

    with open(path) as f:
        data = json.load(f)

    cases = data.get("cases", [])
    if not cases:
        print("  SKIP fig13: no case studies found")
        return

    n_cases = min(len(cases), 5)
    fig, axes = plt.subplots(1, n_cases, figsize=(3.5 * n_cases, 4))
    if n_cases == 1:
        axes = [axes]

    for ax_idx, case in enumerate(cases[:n_cases]):
        ax = axes[ax_idx]
        time_bins = case["time_bins_months"]
        cif = case["cif_curve"]
        lo = case["band_lower"]
        hi = case["band_upper"]
        actual = case["actual_duration_months"]

        ax.fill_between(
            time_bins, lo, hi, alpha=0.25, color=COLORS["DeepHit"], label="90% CI band"
        )
        ax.plot(
            time_bins,
            cif,
            "-o",
            color=COLORS["DeepHit"],
            markersize=3,
            label=f"CIF → {case['dest_stage']}",
        )
        ax.axvline(
            x=actual,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Actual: {actual:.0f}mo",
        )

        # Timing interval
        ti = case.get("timing_interval")
        if ti and ti.get("lower_months") is not None:
            ax.axvspan(
                ti["lower_months"],
                ti["upper_months"],
                alpha=0.1,
                color="green",
                label="Timing CI",
            )

        ax.set_xlabel("Months")
        ax.set_ylabel("CIF")
        ax.set_title(
            f"Pt {case['patno']}\n"
            f"{case['source_stage']}→{case['dest_stage']}, "
            f"{case['sex']}, {case['age_at_baseline']:.0f}yr",
            fontsize=9,
        )
        ax.set_ylim(-0.05, 1.05)
        if ax_idx == 0:
            ax.legend(fontsize=6, loc="upper left")

    fig.suptitle(
        "Individual Patient CIF with Conformal Bands (DeepHit, 90% CL)",
        fontsize=11,
        y=1.04,
    )
    fig.tight_layout()
    save_fig(fig, "fig13_patient_case_studies")


def main():
    print("=" * 60)
    print("GENERATING PAPER 4 FIGURES")
    print("=" * 60)

    fig1_conformal_cif_bands()
    fig2_coverage_calibration()
    fig3_interval_width_by_transition()
    fig4_timing_intervals()
    fig5_reliability_diagram()
    fig6_ece_comparison()
    fig7_subgroup_forest_plot()
    fig8_subgroup_interaction()
    fig9_gate_activation()
    fig10_conditional_coverage()
    fig11_conformal_baselines()
    fig12_directional_coverage()
    fig13_patient_case_studies()

    print("\nDone!")


if __name__ == "__main__":
    main()
