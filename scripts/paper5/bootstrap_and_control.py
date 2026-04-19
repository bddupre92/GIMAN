#!/usr/bin/env python3
"""Paper 5 robustness analysis: window-level bootstrap CIs + W4 shift-vs-n control.

Analysis A — Window-level bootstrap CIs
    For each of 4 windows x 2 models (DeepHit, Graph-DT), load the per-window
    checkpoint from outputs/paper5/checkpoints/window{1..4}_{model}.pt, run
    inference on the held-out test set, and compute 1,000-resample bootstrap
    C-td 95% CIs over the episode index.

Analysis B — W4 random-split control
    W4 = 50/50 enrollment-ordered (train 950 / test 950, C-td ~0.69).
    This confounds sample-size drop (950 vs 760+ training) with maximal temporal
    distance. We re-train with a RANDOM 50/50 split (same PATNO universe) to
    isolate the two effects. N_runs = 5 bootstrap reshuffles, same hyperparams,
    identical episode + feature pipeline.

Outputs:
    outputs/mechanistic_twin/paper5_submission/revision_analyses/
        bootstrap_and_control.json          # all numerical results
        fig5_with_cis.pdf                   # 4-window learning curve with CIs
        figS5_w4_shift_vs_n.pdf             # W4 shift vs random-split control
        p5_bootstrap_summary.md             # short verdict report

Runtime: ~5 min inference (Analysis A) + ~10-25 min retraining (Analysis B, 5
reshuffles x 2 models with early stopping on MPS).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.paper3.dynamic_deephit import (  # noqa: E402
    DeepHitDataset,
    build_patient_arrays,
    compute_feature_stats,
    extract_episodes,
    load_deephit_checkpoint,
    predict_all,
)
from giman_pipeline.paper3.graph_digital_twin import (  # noqa: E402
    GraphDeepHitDataset,
    load_graph_dt_checkpoint,
    predict_all_graph,
)
from giman_pipeline.paper5.inductive_graph import (  # noqa: E402
    InductiveGraphExtender,
    extract_test_baseline_features,
)
from giman_pipeline.paper5.train_per_window import (  # noqa: E402
    train_deephit_on_window,
    train_graph_dt_on_window,
)

# ── Paths ─────────────────────────────────────────────────────────────

FEATURES_PATH = ROOT / "data" / "07_paper3_features" / "longitudinal_features.csv"
CKPT_DIR = ROOT / "outputs" / "paper5" / "checkpoints"
OUT_DIR = (
    ROOT
    / "outputs"
    / "mechanistic_twin"
    / "paper5_submission"
    / "revision_analyses"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Bootstrap helper ──────────────────────────────────────────────────


def _compute_ctd_vectorized(
    cif: np.ndarray,
    events: np.ndarray,
    tbins: np.ndarray,
    cens: np.ndarray,
    n_pairs: int = 50_000,
    seed: int = 42,
) -> float:
    """Same-cause time-dependent concordance on pre-sampled pairs (vectorized).

    Identical math to Paper 3's ``compute_ctd`` but fully vectorized via NumPy.
    """
    uncensored = np.where(~cens)[0]
    if len(uncensored) < 2:
        return 0.5
    rng = np.random.RandomState(seed)
    n_max = min(n_pairs, len(uncensored) * (len(uncensored) - 1) // 2)
    # Sample pair indices
    ii = rng.choice(uncensored, size=n_max, replace=True)
    jj = rng.choice(uncensored, size=n_max, replace=True)
    # Drop same-index and same-tbin pairs first
    mask = (ii != jj) & (events[ii] == events[jj]) & (tbins[ii] != tbins[jj])
    if not mask.any():
        return 0.5
    ii = ii[mask]
    jj = jj[mask]
    # Order so i has earlier time
    later_first = tbins[ii] > tbins[jj]
    ii2 = np.where(later_first, jj, ii)
    jj2 = np.where(later_first, ii, jj)
    k = events[ii2]
    t = tbins[ii2]
    rng_idx = np.arange(len(ii2))
    ci = cif[ii2, k, t]
    cj = cif[jj2, k, t]
    concordant = int(np.sum(ci > cj))
    discordant = int(np.sum(ci < cj))
    tied = int(np.sum(ci == cj))
    total = concordant + discordant + 0.5 * tied
    return float(concordant / total) if total > 0 else 0.5


def _bootstrap_ctd_from_preds(
    preds: dict,
    n_bootstrap: int = 1000,
    seed: int = 123,
    n_pairs: int = 50_000,
) -> tuple[float, float, float, list[float]]:
    """Bootstrap C-td over the episode index (no model re-inference).

    Resamples *episodes* with replacement; computes C-td on each draw using the
    vectorized implementation. Returns (mean_c_td, ci_lo, ci_hi, samples).
    """
    cif = preds["cif"].numpy()
    events = preds["event_idxs"].numpy()
    tbins = preds["time_bins"].numpy()
    cens = preds["censored"].numpy()
    n_ep = len(events)

    rng = np.random.RandomState(seed)
    samples = []
    for b in range(n_bootstrap):
        idx = rng.choice(n_ep, size=n_ep, replace=True)
        ctd = _compute_ctd_vectorized(
            cif[idx],
            events[idx],
            tbins[idx],
            cens[idx],
            n_pairs=n_pairs,
            seed=42 + b,
        )
        samples.append(ctd)

    samples_arr = np.array(samples)
    mean = float(np.mean(samples_arr))
    lo = float(np.percentile(samples_arr, 2.5))
    hi = float(np.percentile(samples_arr, 97.5))
    return mean, lo, hi, samples


# ── Analysis A: per-window inference + bootstrap CIs ─────────────────


def run_window_inference(
    window_idx: int,
    features_df: pd.DataFrame,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """Load window checkpoints, run inference on test set, compute preds dict."""
    w_num = window_idx + 1
    out = {"window_idx": window_idx, "window_name": f"W{w_num}"}

    # DeepHit
    dh_path = CKPT_DIR / f"window{w_num}_deephit.pt"
    if verbose:
        print(f"  [W{w_num}] loading DeepHit from {dh_path.name}")
    dh_model, dh_cp = load_deephit_checkpoint(dh_path, device=device)
    train_patnos = dh_cp["train_pats"]
    test_patnos = dh_cp["test_pats"]
    means = dh_cp["means"]
    stds = dh_cp["stds"]

    all_patnos = set(train_patnos) | set(test_patnos)
    subset_df = features_df[features_df["PATNO"].isin(all_patnos)]
    episodes = extract_episodes(subset_df, verbose=False)
    patient_arrays, _ = build_patient_arrays(subset_df)
    test_set = set(test_patnos)
    test_eps = [e for e in episodes if e.patno in test_set]

    dh_test_ds = DeepHitDataset(test_eps, patient_arrays, means, stds)
    dh_preds = predict_all(dh_model, dh_test_ds, device)
    out["deephit"] = {
        "n_train_patients": len(train_patnos),
        "n_test_patients": len(test_patnos),
        "n_test_episodes": len(test_eps),
        "point_c_td": dh_cp.get("c_td"),
    }
    out["_deephit_preds"] = dh_preds  # kept for bootstrap

    # Graph-DT — need extended graph
    gdt_path = CKPT_DIR / f"window{w_num}_graph_dt.pt"
    if verbose:
        print(f"  [W{w_num}] loading Graph-DT from {gdt_path.name}")
    gdt_model, gdt_cp = load_graph_dt_checkpoint(gdt_path, device=device)
    train_edge_index = gdt_cp["edge_index"]
    train_edge_weight = gdt_cp["edge_weight"]
    train_node_baseline = gdt_cp["node_baseline"]
    train_pat_to_gidx = gdt_cp["pat_to_gidx"]
    k_neighbors = gdt_cp.get("k_neighbors", 15)

    extender = InductiveGraphExtender(
        train_node_baseline=train_node_baseline,
        train_edge_index=train_edge_index,
        train_edge_weight=train_edge_weight,
        train_pat_to_gidx=train_pat_to_gidx,
        k_neighbors=k_neighbors,
    )
    test_baseline = extract_test_baseline_features(
        features_df, test_patnos, train_means=None, train_stds=None
    )
    (
        ext_baseline,
        ext_edge_index,
        ext_edge_weight,
        full_pat_to_gidx,
    ) = extender.extend_for_test(test_baseline, test_patnos)

    gdt_test_ds = GraphDeepHitDataset(
        test_eps,
        patient_arrays,
        gdt_cp["means"],
        gdt_cp["stds"],
        full_pat_to_gidx,
    )
    gdt_preds = predict_all_graph(
        gdt_model,
        gdt_test_ds,
        device,
        node_baseline=ext_baseline,
        edge_index=ext_edge_index,
        edge_weight=ext_edge_weight,
    )
    out["graph_dt"] = {
        "n_test_episodes": len(test_eps),
        "point_c_td": gdt_cp.get("c_td"),
    }
    out["_graph_dt_preds"] = gdt_preds

    return out


def run_analysis_a(
    features_df: pd.DataFrame,
    device: torch.device,
    n_bootstrap: int = 1000,
    verbose: bool = True,
) -> dict:
    """Per-window inference + bootstrap CIs for both models across W1-W4."""
    results = {}
    for wi in range(4):
        w_name = f"W{wi + 1}"
        t_start = time.time()
        win_out = run_window_inference(wi, features_df, device, verbose=verbose)

        # Bootstrap DeepHit
        dh_mean, dh_lo, dh_hi, _ = _bootstrap_ctd_from_preds(
            win_out["_deephit_preds"], n_bootstrap=n_bootstrap, seed=1000 + wi
        )
        win_out["deephit"]["bootstrap_mean_c_td"] = dh_mean
        win_out["deephit"]["bootstrap_ci_lo"] = dh_lo
        win_out["deephit"]["bootstrap_ci_hi"] = dh_hi

        # Bootstrap Graph-DT
        gdt_mean, gdt_lo, gdt_hi, _ = _bootstrap_ctd_from_preds(
            win_out["_graph_dt_preds"], n_bootstrap=n_bootstrap, seed=2000 + wi
        )
        win_out["graph_dt"]["bootstrap_mean_c_td"] = gdt_mean
        win_out["graph_dt"]["bootstrap_ci_lo"] = gdt_lo
        win_out["graph_dt"]["bootstrap_ci_hi"] = gdt_hi

        # CI overlap test
        overlap = not (dh_hi < gdt_lo or gdt_hi < dh_lo)
        win_out["ci_overlap"] = bool(overlap)
        win_out["deephit_minus_graphdt_point"] = float(
            win_out["deephit"]["point_c_td"] - win_out["graph_dt"]["point_c_td"]
        )

        # Clean up prediction tensors before serialization
        del win_out["_deephit_preds"]
        del win_out["_graph_dt_preds"]

        elapsed = time.time() - t_start
        win_out["elapsed_seconds"] = round(elapsed, 1)
        if verbose:
            print(
                f"  [W{wi + 1}] DeepHit  {dh_mean:.4f}"
                f" [{dh_lo:.4f}, {dh_hi:.4f}] "
                f"| Graph-DT {gdt_mean:.4f} [{gdt_lo:.4f}, {gdt_hi:.4f}]"
                f" | overlap: {overlap} | {elapsed:.1f}s"
            )
        results[w_name] = win_out
    return results


# ── Analysis B: W4 shift vs sample-size ───────────────────────────────


def run_analysis_b(
    features_df: pd.DataFrame,
    n_resamples: int = 5,
    verbose: bool = True,
) -> dict:
    """Random-split control for W4 (50/50 random vs 50/50 enrollment-ordered).

    For each of n_resamples:
      1. Randomly permute the 1,900 PATNOs
      2. Train/test = first 950 / last 950
      3. Train DeepHit and Graph-DT from scratch with same hyperparams as W4
      4. Record test-set C-td
    """
    # Pool of all patients used in W4 temporal split (union of train+test)
    w4_dh_cp = torch.load(
        CKPT_DIR / "window4_deephit.pt",
        map_location="cpu",
        weights_only=False,
    )
    all_patnos = sorted(set(w4_dh_cp["train_pats"]) | set(w4_dh_cp["test_pats"]))
    n_total = len(all_patnos)
    half = n_total // 2
    if verbose:
        print(f"  Random-split pool: {n_total} patients, {half}/{n_total - half} split")

    resample_results = []
    for r in range(n_resamples):
        rng = np.random.RandomState(50_000 + r)
        shuffled = np.array(all_patnos, dtype=int)
        rng.shuffle(shuffled)
        train_pats = shuffled[:half].tolist()
        test_pats = shuffled[half:].tolist()

        if verbose:
            print(
                f"\n  [random-split resample {r + 1}/{n_resamples}] "
                f"seed={50_000 + r}, train={len(train_pats)}, test={len(test_pats)}"
            )

        # Seed isolation for DeepHit vs Graph-DT
        dh_result = train_deephit_on_window(
            features_df=features_df,
            train_patnos=train_pats,
            test_patnos=test_pats,
            checkpoint_path=None,
            seed=50_000 + r,
            verbose=False,
        )
        gdt_result = train_graph_dt_on_window(
            features_df=features_df,
            train_patnos=train_pats,
            test_patnos=test_pats,
            checkpoint_path=None,
            seed=50_000 + r,
            verbose=False,
        )
        resample_results.append(
            {
                "resample_idx": r,
                "seed": 50_000 + r,
                "n_train": len(train_pats),
                "n_test": len(test_pats),
                "deephit_c_td": dh_result["c_td"],
                "deephit_ibs": dh_result["ibs"],
                "graph_dt_c_td": gdt_result["c_td"],
                "graph_dt_ibs": gdt_result["ibs"],
            }
        )
        if verbose:
            print(
                f"    DeepHit C-td = {dh_result['c_td']:.4f} | "
                f"Graph-DT C-td = {gdt_result['c_td']:.4f}"
            )

    dh_random = np.array([r["deephit_c_td"] for r in resample_results])
    gdt_random = np.array([r["graph_dt_c_td"] for r in resample_results])

    w4_dh_ctd = w4_dh_cp.get("c_td")
    w4_gdt_cp = torch.load(
        CKPT_DIR / "window4_graph_dt.pt",
        map_location="cpu",
        weights_only=False,
    )
    w4_gdt_ctd = w4_gdt_cp.get("c_td")

    summary = {
        "n_resamples": n_resamples,
        "n_train": half,
        "n_test": n_total - half,
        "resample_results": resample_results,
        "random_split": {
            "deephit": {
                "mean_c_td": float(dh_random.mean()),
                "std_c_td": float(dh_random.std(ddof=1)) if len(dh_random) > 1 else 0.0,
                "min_c_td": float(dh_random.min()),
                "max_c_td": float(dh_random.max()),
            },
            "graph_dt": {
                "mean_c_td": float(gdt_random.mean()),
                "std_c_td": float(gdt_random.std(ddof=1))
                if len(gdt_random) > 1
                else 0.0,
                "min_c_td": float(gdt_random.min()),
                "max_c_td": float(gdt_random.max()),
            },
        },
        "enrollment_ordered_w4": {
            "deephit_c_td": float(w4_dh_ctd) if w4_dh_ctd is not None else None,
            "graph_dt_c_td": float(w4_gdt_ctd) if w4_gdt_ctd is not None else None,
        },
        "decomposition_pp": {
            "deephit_total_gap_from_cv_0924": float(0.924 - w4_dh_ctd),
            "deephit_sample_size_effect": float(0.924 - dh_random.mean()),
            "deephit_shift_effect": float(dh_random.mean() - w4_dh_ctd),
            "graph_dt_total_gap_from_cv_0904": float(0.904 - w4_gdt_ctd),
            "graph_dt_sample_size_effect": float(0.904 - gdt_random.mean()),
            "graph_dt_shift_effect": float(gdt_random.mean() - w4_gdt_ctd),
        },
    }
    return summary


# ── Figures ───────────────────────────────────────────────────────────


def make_fig5_with_cis(analysis_a: dict, out_path: Path) -> None:
    """Learning-curve figure with 95% CI error bars."""
    order = ["W1", "W2", "W3", "W4"]
    n_train = [analysis_a[w]["deephit"]["n_train_patients"] for w in order]

    dh_mean = [analysis_a[w]["deephit"]["bootstrap_mean_c_td"] for w in order]
    dh_lo = [analysis_a[w]["deephit"]["bootstrap_ci_lo"] for w in order]
    dh_hi = [analysis_a[w]["deephit"]["bootstrap_ci_hi"] for w in order]

    gdt_mean = [analysis_a[w]["graph_dt"]["bootstrap_mean_c_td"] for w in order]
    gdt_lo = [analysis_a[w]["graph_dt"]["bootstrap_ci_lo"] for w in order]
    gdt_hi = [analysis_a[w]["graph_dt"]["bootstrap_ci_hi"] for w in order]

    dh_err = np.array([
        [m - lo for m, lo in zip(dh_mean, dh_lo)],
        [hi - m for hi, m in zip(dh_hi, dh_mean)],
    ])
    gdt_err = np.array([
        [m - lo for m, lo in zip(gdt_mean, gdt_lo)],
        [hi - m for hi, m in zip(gdt_hi, gdt_mean)],
    ])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(
        n_train,
        dh_mean,
        yerr=dh_err,
        fmt="o-",
        capsize=4,
        linewidth=1.8,
        markersize=7,
        color="#d95f02",
        label="DeepHit",
    )
    ax.errorbar(
        [n + 10 for n in n_train],
        gdt_mean,
        yerr=gdt_err,
        fmt="s--",
        capsize=4,
        linewidth=1.8,
        markersize=7,
        color="#1b9e77",
        label="Graph-DT",
    )

    ax.axhline(0.924, color="#d95f02", ls=":", alpha=0.5, label="DeepHit CV (0.924)")
    ax.axhline(0.904, color="#1b9e77", ls=":", alpha=0.5, label="Graph-DT CV (0.904)")

    for i, w in enumerate(order):
        ax.annotate(
            w,
            (n_train[i], max(dh_mean[i], gdt_mean[i]) + 0.015),
            ha="center",
            fontsize=9,
            color="#333",
        )

    ax.set_xlabel("Training-set size (patients)")
    ax.set_ylabel("Test-set C-td (bootstrap mean, 95% CI)")
    ax.set_title("Temporal-validation learning curve with bootstrap CIs")
    ax.set_ylim(0.55, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def make_figS5_shift_vs_n(analysis_b: dict, out_path: Path) -> None:
    """W4 shift vs sample-size decomposition."""
    dh_random = [r["deephit_c_td"] for r in analysis_b["resample_results"]]
    gdt_random = [r["graph_dt_c_td"] for r in analysis_b["resample_results"]]
    w4_dh = analysis_b["enrollment_ordered_w4"]["deephit_c_td"]
    w4_gdt = analysis_b["enrollment_ordered_w4"]["graph_dt_c_td"]

    dh_random_mean = float(np.mean(dh_random))
    gdt_random_mean = float(np.mean(gdt_random))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, random_samples, random_mean, w4_point, paper3_cv, title in [
        (axes[0], dh_random, dh_random_mean, w4_dh, 0.924, "DeepHit"),
        (axes[1], gdt_random, gdt_random_mean, w4_gdt, 0.904, "Graph-DT"),
    ]:
        # Scatter resamples
        jitter = np.random.RandomState(42).uniform(-0.1, 0.1, size=len(random_samples))
        ax.scatter(
            1 + jitter,
            random_samples,
            color="#7570b3",
            s=40,
            alpha=0.7,
            zorder=3,
            label="Random 50/50 resamples",
        )
        ax.hlines(
            random_mean,
            0.6,
            1.4,
            colors="#7570b3",
            linewidth=2,
            label=f"Random mean = {random_mean:.3f}",
        )
        ax.scatter(
            [2],
            [w4_point],
            color="#d95f02",
            s=80,
            marker="D",
            zorder=3,
            label=f"W4 enrollment-ordered = {w4_point:.3f}",
        )
        ax.hlines(
            paper3_cv,
            0.5,
            2.5,
            colors="#1b9e77",
            linestyles=":",
            linewidth=2,
            label=f"Paper 3 CV = {paper3_cv:.3f}",
        )

        # Arrow showing shift
        if random_mean != w4_point:
            ax.annotate(
                f"Δ_shift = {random_mean - w4_point:+.3f}",
                xy=(2, w4_point + 0.01),
                xytext=(1.3, (random_mean + w4_point) / 2),
                fontsize=8,
                ha="center",
                arrowprops=dict(arrowstyle="->", color="black", alpha=0.6),
            )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Random\n50/50", "Enrollment-ordered\n50/50 (W4)"])
        ax.set_title(title)
        ax.set_ylim(0.55, 1.0)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="lower left", fontsize=7)

    axes[0].set_ylabel("Test-set C-td")
    fig.suptitle(
        "W4 shift vs sample-size control (N={n} resamples)".format(
            n=len(dh_random)
        )
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ── Markdown summary ──────────────────────────────────────────────────


def write_summary_md(results: dict, out_path: Path) -> None:
    a = results["analysis_a"]
    b = results.get("analysis_b")

    lines = [
        "# Paper 5 — Bootstrap CI + W4 Shift vs. Sample-Size Control",
        "",
        "Generated by `scripts/paper5/bootstrap_and_control.py`.",
        "",
        "## Analysis A — Window-level bootstrap 95% CIs",
        "",
        "| Window | n_train | DeepHit C-td [95% CI] | Graph-DT C-td [95% CI] | CI overlap |",
        "|---|---|---|---|---|",
    ]
    for w in ["W1", "W2", "W3", "W4"]:
        wr = a[w]
        dh = wr["deephit"]
        gdt = wr["graph_dt"]
        lines.append(
            f"| {w} | {dh['n_train_patients']} | "
            f"{dh['bootstrap_mean_c_td']:.4f} "
            f"[{dh['bootstrap_ci_lo']:.4f}, {dh['bootstrap_ci_hi']:.4f}] | "
            f"{gdt['bootstrap_mean_c_td']:.4f} "
            f"[{gdt['bootstrap_ci_lo']:.4f}, {gdt['bootstrap_ci_hi']:.4f}] | "
            f"{'YES' if wr['ci_overlap'] else 'NO'} |"
        )

    lines.extend([
        "",
        "### CI overlap verdict",
        "",
    ])
    overlap_windows = [w for w in ["W1", "W2", "W3", "W4"] if a[w]["ci_overlap"]]
    non_overlap_windows = [
        w for w in ["W1", "W2", "W3", "W4"] if not a[w]["ci_overlap"]
    ]
    lines.append(
        f"- Overlapping CIs (models NOT distinguishable): "
        f"{', '.join(overlap_windows) if overlap_windows else 'none'}"
    )
    lines.append(
        f"- Non-overlapping CIs (models statistically distinguishable): "
        f"{', '.join(non_overlap_windows) if non_overlap_windows else 'none'}"
    )

    if b is None:
        lines.append("")
        lines.append("## Analysis B — W4 shift-vs-n decomposition")
        lines.append("")
        lines.append("_Skipped (use without --skip-b to run)._")
        out_path.write_text("\n".join(lines))
        return

    lines.extend([
        "",
        "## Analysis B — W4 shift-vs-n decomposition",
        "",
        f"Pool: {b['n_train'] + b['n_test']} patients (all W4 participants)",
        f"Resamples: {b['n_resamples']} random 50/50 splits",
        "",
        "| Quantity | DeepHit | Graph-DT |",
        "|---|---|---|",
        f"| Paper 3 CV (reference) | 0.924 | 0.904 |",
        f"| Random 50/50 mean | "
        f"{b['random_split']['deephit']['mean_c_td']:.4f} "
        f"(SD {b['random_split']['deephit']['std_c_td']:.4f}) | "
        f"{b['random_split']['graph_dt']['mean_c_td']:.4f} "
        f"(SD {b['random_split']['graph_dt']['std_c_td']:.4f}) |",
        f"| W4 enrollment-ordered | "
        f"{b['enrollment_ordered_w4']['deephit_c_td']:.4f} | "
        f"{b['enrollment_ordered_w4']['graph_dt_c_td']:.4f} |",
        f"| Sample-size effect (CV - random) | "
        f"{b['decomposition_pp']['deephit_sample_size_effect']:+.4f} | "
        f"{b['decomposition_pp']['graph_dt_sample_size_effect']:+.4f} |",
        f"| Shift effect (random - W4) | "
        f"{b['decomposition_pp']['deephit_shift_effect']:+.4f} | "
        f"{b['decomposition_pp']['graph_dt_shift_effect']:+.4f} |",
        f"| Total gap (CV - W4) | "
        f"{b['decomposition_pp']['deephit_total_gap_from_cv_0924']:+.4f} | "
        f"{b['decomposition_pp']['graph_dt_total_gap_from_cv_0904']:+.4f} |",
        "",
        "### Interpretation",
        "",
    ])

    dh_shift = b["decomposition_pp"]["deephit_shift_effect"]
    dh_ss = b["decomposition_pp"]["deephit_sample_size_effect"]
    gdt_shift = b["decomposition_pp"]["graph_dt_shift_effect"]
    gdt_ss = b["decomposition_pp"]["graph_dt_sample_size_effect"]

    def _dom(shift, ss):
        frac = abs(shift) / max(abs(shift) + abs(ss), 1e-9)
        if frac >= 0.67:
            return f"shift dominates ({frac*100:.0f}%)"
        if frac <= 0.33:
            return f"sample-size dominates ({(1-frac)*100:.0f}%)"
        return f"mixed (shift {frac*100:.0f}% / sample-size {(1-frac)*100:.0f}%)"

    lines.append(
        f"- DeepHit: {_dom(dh_shift, dh_ss)} "
        f"(shift {dh_shift:+.3f} vs. sample-size {dh_ss:+.3f})"
    )
    lines.append(
        f"- Graph-DT: {_dom(gdt_shift, gdt_ss)} "
        f"(shift {gdt_shift:+.3f} vs. sample-size {gdt_ss:+.3f})"
    )
    lines.append("")
    lines.append(
        "> Positive *shift effect* means the temporal W4 split degrades accuracy "
        "beyond what random 50/50 subsampling alone produces — i.e. the "
        "distribution drift between 2010-2018 and 2018-2024 enrollees is a "
        "first-order contributor to the W4 collapse."
    )
    lines.append("")

    out_path.write_text("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Paper 5 bootstrap CIs + W4 shift-vs-n control"
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Episode-level bootstrap samples (default: 1000)",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=5,
        help="Random-split resamples for Analysis B (default: 5)",
    )
    parser.add_argument(
        "--skip-b",
        action="store_true",
        help="Skip Analysis B (only bootstrap CIs).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device: cpu|cuda|mps (default: auto).",
    )
    args = parser.parse_args()

    if args.device is None:
        device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Loading features from {FEATURES_PATH}")
    features_df = pd.read_csv(FEATURES_PATH)

    print("\n=== ANALYSIS A: Per-window inference + bootstrap CIs ===")
    a_start = time.time()
    analysis_a = run_analysis_a(
        features_df, device, n_bootstrap=args.n_bootstrap, verbose=True
    )
    a_elapsed = time.time() - a_start
    print(f"Analysis A complete in {a_elapsed / 60:.1f} min")

    analysis_b = None
    if not args.skip_b:
        print("\n=== ANALYSIS B: W4 shift-vs-n random-split control ===")
        b_start = time.time()
        analysis_b = run_analysis_b(
            features_df, n_resamples=args.n_resamples, verbose=True
        )
        b_elapsed = time.time() - b_start
        print(f"Analysis B complete in {b_elapsed / 60:.1f} min")

    results = {
        "n_bootstrap": args.n_bootstrap,
        "analysis_a": analysis_a,
        "analysis_b": analysis_b,
    }

    # Save JSON
    out_json = OUT_DIR / "bootstrap_and_control.json"

    def _cvt(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=_cvt)
    print(f"\nSaved JSON: {out_json}")

    # Figures
    fig5_path = OUT_DIR / "fig5_with_cis.pdf"
    make_fig5_with_cis(analysis_a, fig5_path)
    print(f"Saved figure: {fig5_path}")

    if analysis_b is not None:
        figS5_path = OUT_DIR / "figS5_w4_shift_vs_n.pdf"
        make_figS5_shift_vs_n(analysis_b, figS5_path)
        print(f"Saved figure: {figS5_path}")

    # Markdown summary
    md_path = OUT_DIR / "p5_bootstrap_summary.md"
    write_summary_md(results, md_path)
    print(f"Saved summary: {md_path}")


if __name__ == "__main__":
    main()
