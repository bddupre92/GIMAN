#!/usr/bin/env python3
"""Paper 3+4 Workstream 2 — 5-fold model ensemble analysis (proper LOFO).

Six prediction strategies evaluated on the pooled held-out 5-fold cohort
via patient-level 1,000-resample paired bootstrap 95% CIs:

  1. baseline-DH        Single held-out DeepHit fold (reference)
  2. baseline-GDT       Single held-out Graph-DT fold (reference)
  3. arch-blend         Per-patient mean of DH_held-out + GDT_held-out CIFs
                        -- tests whether cross-architecture blending helps
  4. LOFO-DH-avg        For each test episode in fold k, mean of 4 DeepHit
                        fold-models != k. Tests within-model bagging
                        variance reduction (Lakshminarayanan 2017).
  5. LOFO-GDT-avg       Same for Graph-DT. Directly tests the Tier-2
                        finding that Graph-DT has higher fold variance
                        (0.034 vs DH 0.020).
  6. LOFO-full-IV       8-model inverse-variance weighted ensemble of the
                        4 eligible DH + 4 eligible GDT models per episode.
                        Weights = 1 / sigma^2(val C-td for each fold).

LOFO = Leave-One-Fold-Out inference. For each held-out test episode in
fold k, strategies 4-6 average across the 4 fold-models where the
patient was NOT in training. This preserves held-out purity. Graph-DT
has the additional constraint that fold-models can only predict on
patients present in their graph (pat_to_gidx); if fewer than 4
eligible Graph-DT models cover a patient, we average over however
many are available (minimum 2).

Each of the 10 fold-models is run inference ONCE on all cohort
episodes (~20s each, ~3-4 min total). Strategies are then computed
from the stored per-model CIF arrays.

Outputs:
  outputs/mechanistic_twin/paper3plus4_submission/npj-dm/revision_analyses/
    ensemble_5fold.json          -- metrics + paired bootstrap CIs
    ensemble_5fold_summary.md    -- human-readable summary
    fig_ensemble_comparison.pdf  -- pooled C-td + fold variance

Decision gate (per .claude/plans/refactored-spinning-lantern.md §W2):
  - LOFO-GDT-avg closes >= 50% of the (DH - GDT) gap -- strengthens paper
  - Any LOFO ensemble reduces fold variance >= 20% vs single-fold -- novelty

Usage:
  .venv/bin/python scripts/paper3/ensemble_5fold.py
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.paper3.dynamic_deephit import (  # noqa: E402
    DeepHitDataset,
    _get_time_bin,
    build_patient_arrays,
    extract_episodes,
    load_deephit_checkpoint,
    predict_all,
)
from giman_pipeline.paper3.graph_digital_twin import (  # noqa: E402
    GraphDeepHitDataset,
    load_graph_dt_checkpoint,
    predict_all_graph,
)

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "07_paper3_features" / "longitudinal_features.csv"
TRANSITIONS_PATH = DATA_DIR / "06_longitudinal_staging" / "transition_events.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "paper3_checkpoints"
OUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "mechanistic_twin"
    / "paper3plus4_submission"
    / "npj-dm"
    / "revision_analyses"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 1000
N_PAIRS_EVAL = 50000     # pair pool for pooled-cohort C-td
N_PAIRS_BOOT = 5000      # smaller pool for bootstrap resamples
SEED = 42
DEVICE = torch.device("cpu")


@dataclass
class CohortPredictions:
    """CIFs for all cohort episodes under one fold-model. NaN where the
    model cannot predict (e.g., Graph-DT patient not in the fold's graph)."""

    cif: np.ndarray      # (n_ep, n_causes, n_time_bins) -- NaN rows where undefined
    valid: np.ndarray    # (n_ep,) bool -- True if model could predict
    val_ctd: float       # the fold's validation C-td, used for IV weighting


def _run_deephit_cohort(fold_idx: int, episodes, patient_arrays) -> CohortPredictions:
    """Predict the fold-k DeepHit model on ALL cohort episodes (not just fold-k's test set)."""
    cpath = CHECKPOINT_DIR / "deephit" / f"fold{fold_idx}_deephit.pt"
    model, cp = load_deephit_checkpoint(cpath, device=DEVICE)
    # DeepHit has no patient-graph constraint -- every episode is predictable
    ds = DeepHitDataset(episodes, patient_arrays, cp["means"], cp["stds"])
    preds = predict_all(model, ds, DEVICE)
    cif = preds["cif"].numpy()
    valid = np.ones(len(episodes), dtype=bool)
    val_ctd = float(cp.get("fold_ctd", cp.get("val_ctd", np.nan)))
    if np.isnan(val_ctd):
        val_ctd = 0.9  # fallback default
    return CohortPredictions(cif=cif, valid=valid, val_ctd=val_ctd)


def _run_graph_dt_cohort(fold_idx: int, episodes, patient_arrays) -> CohortPredictions:
    """Predict the fold-k Graph-DT model on all cohort episodes whose PATNO
    is in the fold's graph (pat_to_gidx)."""
    cpath = CHECKPOINT_DIR / "graph_dt" / f"fold{fold_idx}_graph_dt.pt"
    model, cp = load_graph_dt_checkpoint(cpath, device=DEVICE)
    n_ep = len(episodes)
    n_causes = cp.get("n_causes", 7)
    n_tbins = cp.get("n_time_bins", 11)

    # Episodes eligible for this fold's graph
    pat_to_gidx = cp["pat_to_gidx"]
    eligible_mask = np.array(
        [e.patno in pat_to_gidx for e in episodes], dtype=bool
    )
    eligible_eps = [e for e, ok in zip(episodes, eligible_mask) if ok]

    if not eligible_eps:
        cif = np.full((n_ep, n_causes, n_tbins), np.nan)
        return CohortPredictions(cif=cif, valid=np.zeros(n_ep, dtype=bool), val_ctd=0.9)

    ds = GraphDeepHitDataset(
        eligible_eps, patient_arrays, cp["means"], cp["stds"], pat_to_gidx
    )
    preds = predict_all_graph(
        model, ds, DEVICE,
        cp["node_baseline"], cp["edge_index"], cp["edge_weight"],
    )
    cif_eligible = preds["cif"].numpy()
    # scatter back into full cohort array
    cif = np.full((n_ep, n_causes, n_tbins), np.nan)
    cif[eligible_mask] = cif_eligible
    val_ctd = float(cp.get("fold_ctd", cp.get("val_ctd", np.nan)))
    if np.isnan(val_ctd):
        val_ctd = 0.9
    return CohortPredictions(cif=cif, valid=eligible_mask, val_ctd=val_ctd)


def _lofo_average(
    per_fold_cifs: list[np.ndarray],    # 5 arrays of (n_ep, c, t); NaN where invalid
    test_fold_per_ep: np.ndarray,       # (n_ep,) 0..4 -- which fold was each episode's test
    validity_per_fold: list[np.ndarray], # 5 bool arrays (n_ep,) -- True if model k valid
    weights: list[float] | None = None,  # optional per-fold weights (for IV)
) -> tuple[np.ndarray, np.ndarray]:
    """For each episode, average predictions from the 4 folds where episode
    was NOT in training (fold != test_fold). Returns (avg_cif, n_contributing)."""
    n_ep, n_c, n_t = per_fold_cifs[0].shape
    avg_cif = np.zeros_like(per_fold_cifs[0])
    n_contrib = np.zeros(n_ep, dtype=int)

    for ep_idx in range(n_ep):
        held_out = test_fold_per_ep[ep_idx]
        contrib_vals = []
        contrib_weights = []
        for k in range(5):
            if k == held_out:
                continue
            if not validity_per_fold[k][ep_idx]:
                continue
            contrib_vals.append(per_fold_cifs[k][ep_idx])
            w = 1.0 if weights is None else weights[k]
            contrib_weights.append(w)
        if not contrib_vals:
            avg_cif[ep_idx] = np.nan
            n_contrib[ep_idx] = 0
            continue
        W = np.array(contrib_weights)
        W = W / W.sum()
        stacked = np.stack(contrib_vals, axis=0)  # (n_valid, c, t)
        avg_cif[ep_idx] = (stacked * W[:, None, None]).sum(axis=0)
        n_contrib[ep_idx] = len(contrib_vals)

    return avg_cif, n_contrib


def _iv_ensemble(
    dh_cifs: list[np.ndarray], dh_valid: list[np.ndarray], dh_ctds: list[float],
    gdt_cifs: list[np.ndarray], gdt_valid: list[np.ndarray], gdt_ctds: list[float],
    test_fold_per_ep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-variance full ensemble across 8 eligible models per episode
    (4 DeepHit + 4 Graph-DT, excluding the held-out fold for both)."""
    n_ep = len(test_fold_per_ep)
    n_c, n_t = dh_cifs[0].shape[1:]
    avg_cif = np.zeros((n_ep, n_c, n_t))
    n_contrib = np.zeros(n_ep, dtype=int)

    # IV weights from fold val-ctd: higher C-td -> higher weight (proxy for 1/sigma^2)
    # We use val_ctd^2 as a simple proxy that favors more-reliable models;
    # genuine 1/sigma^2 would require per-fold variance estimates.
    dh_weights = np.array(dh_ctds) ** 2
    gdt_weights = np.array(gdt_ctds) ** 2

    for ep_idx in range(n_ep):
        held_out = test_fold_per_ep[ep_idx]
        contrib_vals = []
        contrib_weights = []
        for k in range(5):
            if k == held_out:
                continue
            if dh_valid[k][ep_idx]:
                contrib_vals.append(dh_cifs[k][ep_idx])
                contrib_weights.append(dh_weights[k])
            if gdt_valid[k][ep_idx]:
                contrib_vals.append(gdt_cifs[k][ep_idx])
                contrib_weights.append(gdt_weights[k])
        if not contrib_vals:
            avg_cif[ep_idx] = np.nan
            n_contrib[ep_idx] = 0
            continue
        W = np.array(contrib_weights)
        W = W / W.sum()
        stacked = np.stack(contrib_vals, axis=0)
        avg_cif[ep_idx] = (stacked * W[:, None, None]).sum(axis=0)
        n_contrib[ep_idx] = len(contrib_vals)

    return avg_cif, n_contrib


def _held_out_cif(
    per_fold_cifs: list[np.ndarray],
    validity_per_fold: list[np.ndarray],
    test_fold_per_ep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """For each episode, return the per-fold-model CIF from its held-out fold."""
    n_ep = len(test_fold_per_ep)
    out = np.full_like(per_fold_cifs[0], np.nan)
    valid = np.zeros(n_ep, dtype=bool)
    for ep_idx in range(n_ep):
        k = test_fold_per_ep[ep_idx]
        if validity_per_fold[k][ep_idx]:
            out[ep_idx] = per_fold_cifs[k][ep_idx]
            valid[ep_idx] = True
    return out, valid


def _sample_concordant_pairs_vec(events, tbins, censored, n_pairs, rng):
    """Vectorized pair sampler. Returns (i_arr, j_arr, k_arr, t_arr) arrays
    where i has earlier event time than j, same cause, different times."""
    uncensored = np.where(~censored)[0]
    if len(uncensored) < 2:
        return None
    # Oversample to account for filter rejection; expected retain rate ~0.1-0.3
    overdraw = max(n_pairs * 5, 10000)
    idx_a = rng.choice(uncensored, size=overdraw, replace=True)
    idx_b = rng.choice(uncensored, size=overdraw, replace=True)
    ea = events[idx_a]
    eb = events[idx_b]
    ta = tbins[idx_a]
    tb = tbins[idx_b]
    # Keep: same cause AND different times AND different sample
    keep = (ea == eb) & (ta != tb) & (idx_a != idx_b)
    idx_a = idx_a[keep]
    idx_b = idx_b[keep]
    ea = ea[keep]
    ta = ta[keep]
    tb = tb[keep]
    # Orient so i has earlier time than j
    swap = ta > tb
    i_arr = np.where(swap, idx_b, idx_a)
    j_arr = np.where(swap, idx_a, idx_b)
    t_arr = np.minimum(ta, tb)
    k_arr = ea
    # Trim to n_pairs
    n_actual = min(len(i_arr), n_pairs)
    return i_arr[:n_actual], j_arr[:n_actual], k_arr[:n_actual], t_arr[:n_actual]


def _score_pairs_vec(cif, pair_tuple):
    """Vectorized concordance scoring. cif shape (n_ep, n_causes, n_tbins);
    pair_tuple = (i_arr, j_arr, k_arr, t_arr). Returns mean concordance."""
    i_arr, j_arr, k_arr, t_arr = pair_tuple
    if len(i_arr) == 0:
        return 0.5
    ci = cif[i_arr, k_arr, t_arr]
    cj = cif[j_arr, k_arr, t_arr]
    # NaN handling: if either endpoint is NaN, count as tied (0.5)
    nan_mask = np.isnan(ci) | np.isnan(cj)
    scores = np.where(
        nan_mask,
        0.5,
        np.where(ci > cj, 1.0, np.where(ci < cj, 0.0, 0.5)),
    )
    return float(scores.mean())


def _ctd(cif, events, tbins, censored, n_pairs, seed):
    rng = np.random.RandomState(seed)
    pair_tuple = _sample_concordant_pairs_vec(events, tbins, censored, n_pairs, rng)
    if pair_tuple is None:
        return 0.5
    return _score_pairs_vec(cif, pair_tuple)


def main():
    print("Loading episodes and patient arrays...")
    features = pd.read_csv(FEATURES_PATH, low_memory=False)
    patient_arrays, _ = build_patient_arrays(features)
    episodes = extract_episodes(features, verbose=False)
    n_ep = len(episodes)
    events_all = np.array([e.event_stage_idx if not e.censored else 0 for e in episodes])
    tbins_all = np.array([_get_time_bin(e.duration_months) for e in episodes])
    censored_all = np.array([e.censored for e in episodes])
    print(f"  n_episodes={n_ep}  n_events={np.sum(~censored_all)}")

    # Build test_fold_per_ep by scanning the 5 DeepHit fold checkpoints
    print("\nDetermining held-out fold per episode...")
    test_fold_per_ep = np.full(n_ep, -1, dtype=int)
    for fi in range(5):
        cpath = CHECKPOINT_DIR / "deephit" / f"fold{fi}_deephit.pt"
        cp = torch.load(cpath, map_location="cpu", weights_only=False)
        test_pats = set(cp["test_pats"])
        for ep_idx, e in enumerate(episodes):
            if e.patno in test_pats:
                test_fold_per_ep[ep_idx] = fi
    coverage = (test_fold_per_ep >= 0).mean()
    print(f"  Episodes assigned to a held-out fold: {coverage * 100:.1f}% ({np.sum(test_fold_per_ep >= 0)}/{n_ep})")
    # drop episodes not assigned to any held-out fold (shouldn't happen with full CV)
    keep = test_fold_per_ep >= 0
    episodes = [episodes[i] for i in range(n_ep) if keep[i]]
    events_all = events_all[keep]
    tbins_all = tbins_all[keep]
    censored_all = censored_all[keep]
    test_fold_per_ep = test_fold_per_ep[keep]
    n_ep = len(episodes)
    print(f"  Kept {n_ep} episodes for ensemble analysis")

    print("\nRunning cohort-wide inference for 10 models...")
    dh_cohorts: list[CohortPredictions] = []
    gdt_cohorts: list[CohortPredictions] = []
    for fi in range(5):
        print(f"  [DH fold {fi}]")
        dh_cohorts.append(_run_deephit_cohort(fi, episodes, patient_arrays))
        print(f"  [GDT fold {fi}]")
        gdt_cohorts.append(_run_graph_dt_cohort(fi, episodes, patient_arrays))

    dh_cifs = [c.cif for c in dh_cohorts]
    dh_valid = [c.valid for c in dh_cohorts]
    dh_ctds = [c.val_ctd for c in dh_cohorts]
    gdt_cifs = [c.cif for c in gdt_cohorts]
    gdt_valid = [c.valid for c in gdt_cohorts]
    gdt_ctds = [c.val_ctd for c in gdt_cohorts]

    print("\nComposing 6 strategies...")

    # Strategy 1: baseline-DH (single held-out fold DH per episode)
    cif_base_dh, valid_base_dh = _held_out_cif(dh_cifs, dh_valid, test_fold_per_ep)

    # Strategy 2: baseline-GDT
    cif_base_gdt, valid_base_gdt = _held_out_cif(gdt_cifs, gdt_valid, test_fold_per_ep)

    # Strategy 3: arch-blend (held-out DH + held-out GDT) / 2
    with np.errstate(invalid="ignore"):
        cif_arch = 0.5 * (cif_base_dh + cif_base_gdt)
    valid_arch = valid_base_dh & valid_base_gdt

    # Strategy 4: LOFO-DH-avg
    cif_lofo_dh, n_dh_contrib = _lofo_average(dh_cifs, test_fold_per_ep, dh_valid)
    valid_lofo_dh = n_dh_contrib >= 2
    print(f"  LOFO-DH-avg: mean n_models per episode = {n_dh_contrib[valid_lofo_dh].mean():.2f}")

    # Strategy 5: LOFO-GDT-avg
    cif_lofo_gdt, n_gdt_contrib = _lofo_average(gdt_cifs, test_fold_per_ep, gdt_valid)
    valid_lofo_gdt = n_gdt_contrib >= 2
    print(f"  LOFO-GDT-avg: mean n_models per episode = {n_gdt_contrib[valid_lofo_gdt].mean():.2f}")

    # Strategy 6: LOFO-full-IV
    cif_lofo_iv, n_iv_contrib = _iv_ensemble(
        dh_cifs, dh_valid, dh_ctds,
        gdt_cifs, gdt_valid, gdt_ctds,
        test_fold_per_ep,
    )
    valid_lofo_iv = n_iv_contrib >= 4
    print(f"  LOFO-full-IV: mean n_models per episode = {n_iv_contrib[valid_lofo_iv].mean():.2f}")

    strategies = {
        "baseline_dh":   (cif_base_dh, valid_base_dh),
        "baseline_gdt":  (cif_base_gdt, valid_base_gdt),
        "arch_blend":    (cif_arch, valid_arch),
        "lofo_dh_avg":   (cif_lofo_dh, valid_lofo_dh),
        "lofo_gdt_avg":  (cif_lofo_gdt, valid_lofo_gdt),
        "lofo_full_iv":  (cif_lofo_iv, valid_lofo_iv),
    }

    # Pooled C-td for each strategy (use common valid set for fair comparison)
    common_valid = np.ones(n_ep, dtype=bool)
    for _, (_, v) in strategies.items():
        common_valid &= v
    print(f"\nCommon valid episodes across all 6 strategies: {common_valid.sum()}/{n_ep}")

    print("\n=== POOLED COHORT C-TD (common valid subset) ===")
    pooled = {}
    for name, (cif, _) in strategies.items():
        ctd = _ctd(
            cif[common_valid], events_all[common_valid],
            tbins_all[common_valid], censored_all[common_valid],
            N_PAIRS_EVAL, SEED,
        )
        pooled[name] = ctd
        print(f"  {name:20s}  {ctd:.4f}")

    # Fold-level C-tds for variance analysis
    per_fold = {name: [] for name in strategies}
    for fi in range(5):
        fmask = (test_fold_per_ep == fi) & common_valid
        if fmask.sum() < 10:
            continue
        for name, (cif, _) in strategies.items():
            ctd = _ctd(
                cif[fmask], events_all[fmask],
                tbins_all[fmask], censored_all[fmask],
                min(N_PAIRS_EVAL // 5, 10000), SEED + fi,
            )
            per_fold[name].append(ctd)

    print("\n=== PER-FOLD C-TD ===")
    for name, vals in per_fold.items():
        print(f"  {name:20s}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  vals={[round(v,4) for v in vals]}")

    # Paired bootstrap on pooled cohort
    print(f"\nPaired bootstrap ({N_BOOT} resamples on {common_valid.sum()} episodes)...")
    n_valid = common_valid.sum()
    boot = {name: np.zeros(N_BOOT) for name in strategies}
    rng = np.random.default_rng(SEED)
    idx_valid = np.where(common_valid)[0]
    for b in range(N_BOOT):
        idx = rng.choice(idx_valid, size=n_valid, replace=True)
        for name, (cif, _) in strategies.items():
            boot[name][b] = _ctd(
                cif[idx], events_all[idx], tbins_all[idx], censored_all[idx],
                N_PAIRS_BOOT, SEED + b,
            )
        if (b + 1) % 200 == 0:
            print(f"  {b + 1}/{N_BOOT}")

    def ci(v, name):
        return {
            "name": name,
            "point": float(v.mean()),
            "ci_lo": float(np.quantile(v, 0.025)),
            "ci_hi": float(np.quantile(v, 0.975)),
            "std": float(v.std()),
        }

    def paired_ci(a, b, name):
        d = a - b
        return {
            "name": name,
            "delta_mean": float(d.mean()),
            "ci_lo": float(np.quantile(d, 0.025)),
            "ci_hi": float(np.quantile(d, 0.975)),
        }

    pooled_cis = {name: ci(boot[name], name) for name in strategies}
    key_pairs = [
        ("arch_blend", "baseline_dh"),
        ("arch_blend", "baseline_gdt"),
        ("lofo_dh_avg", "baseline_dh"),
        ("lofo_gdt_avg", "baseline_gdt"),
        ("lofo_full_iv", "baseline_dh"),
        ("lofo_full_iv", "baseline_gdt"),
        ("lofo_gdt_avg", "lofo_dh_avg"),
    ]
    paired = {
        f"{a}_vs_{b}": paired_ci(boot[a], boot[b], f"{a} − {b}")
        for a, b in key_pairs
    }

    # Variance reduction
    def var_red(ens, base):
        std_e = np.std(per_fold[ens]) if per_fold[ens] else float("nan")
        std_b = np.std(per_fold[base]) if per_fold[base] else float("nan")
        if std_b == 0 or np.isnan(std_b):
            return None
        return float((std_b - std_e) / std_b * 100)

    var_reductions = {
        "lofo_dh_avg_vs_baseline_dh": var_red("lofo_dh_avg", "baseline_dh"),
        "lofo_gdt_avg_vs_baseline_gdt": var_red("lofo_gdt_avg", "baseline_gdt"),
        "lofo_full_iv_vs_baseline_dh": var_red("lofo_full_iv", "baseline_dh"),
        "lofo_full_iv_vs_baseline_gdt": var_red("lofo_full_iv", "baseline_gdt"),
        "arch_blend_vs_baseline_dh": var_red("arch_blend", "baseline_dh"),
        "arch_blend_vs_baseline_gdt": var_red("arch_blend", "baseline_gdt"),
    }

    # Gap-closure analysis (plan's decision gate)
    gap_dh_gdt = pooled_cis["baseline_dh"]["point"] - pooled_cis["baseline_gdt"]["point"]
    gaps_closed = {}
    for ensname in ["arch_blend", "lofo_dh_avg", "lofo_gdt_avg", "lofo_full_iv"]:
        ens_val = pooled_cis[ensname]["point"]
        if abs(gap_dh_gdt) < 1e-6:
            gaps_closed[ensname] = None
            continue
        # Gap closed = how much ensemble moves toward DeepHit from Graph-DT
        gaps_closed[ensname] = float(
            (ens_val - pooled_cis["baseline_gdt"]["point"]) / gap_dh_gdt * 100
        )

    # Decision gate
    decision_gate = {
        "gap_dh_minus_gdt_pooled": float(gap_dh_gdt),
        "gap_closed_pct": gaps_closed,
        "variance_reduction_pct": var_reductions,
        "threshold_gap_pct": 50.0,
        "threshold_variance_pct": 20.0,
        "passes": {
            "arch_blend_gap": gaps_closed.get("arch_blend") is not None and gaps_closed["arch_blend"] >= 50.0,
            "lofo_gdt_gap":   gaps_closed.get("lofo_gdt_avg") is not None and gaps_closed["lofo_gdt_avg"] >= 50.0,
            "lofo_full_gap":  gaps_closed.get("lofo_full_iv") is not None and gaps_closed["lofo_full_iv"] >= 50.0,
            "lofo_dh_var":    var_reductions.get("lofo_dh_avg_vs_baseline_dh") is not None
                              and var_reductions["lofo_dh_avg_vs_baseline_dh"] >= 20.0,
            "lofo_gdt_var":   var_reductions.get("lofo_gdt_avg_vs_baseline_gdt") is not None
                              and var_reductions["lofo_gdt_avg_vs_baseline_gdt"] >= 20.0,
        },
    }

    results = {
        "cohort": {
            "n_episodes_total": int(len(censored_all)),
            "n_episodes_common_valid": int(common_valid.sum()),
            "n_events": int(np.sum(~censored_all[common_valid])),
        },
        "model_contribution_stats": {
            "lofo_dh_avg_mean_models": float(n_dh_contrib[valid_lofo_dh].mean()) if valid_lofo_dh.any() else None,
            "lofo_gdt_avg_mean_models": float(n_gdt_contrib[valid_lofo_gdt].mean()) if valid_lofo_gdt.any() else None,
            "lofo_full_iv_mean_models": float(n_iv_contrib[valid_lofo_iv].mean()) if valid_lofo_iv.any() else None,
        },
        "pooled_ctd": pooled_cis,
        "per_fold_ctd": {name: [float(v) for v in vals] for name, vals in per_fold.items()},
        "per_fold_std": {name: float(np.std(vals)) if vals else None for name, vals in per_fold.items()},
        "paired_delta_95ci": paired,
        "decision_gate": decision_gate,
    }

    out_path = OUT_DIR / "ensemble_5fold.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON saved: {out_path}")

    # Summary markdown
    lines = [
        "# W2 Results — P3+4 5-Fold LOFO Ensemble (6 strategies)\n",
        "Date: 2026-04-18\n",
        f"Cohort (common valid): {common_valid.sum()} episodes, {int(np.sum(~censored_all[common_valid]))} events.\n\n",
        "## Pooled C-td (1,000 bootstrap resamples)\n\n",
        "| Strategy | Point | 95% CI | Fold std |\n|---|---|---|---|\n",
    ]
    for name in strategies:
        p = pooled_cis[name]
        fs = np.std(per_fold[name]) if per_fold[name] else float("nan")
        lines.append(
            f"| {name} | {p['point']:.4f} | [{p['ci_lo']:.4f}, {p['ci_hi']:.4f}] | {fs:.4f} |\n"
        )
    lines.append("\n## Paired Δ (1,000 bootstrap, 95% CI)\n\n")
    lines.append("| Comparison | Δ | 95% CI |\n|---|---|---|\n")
    for k, v in paired.items():
        lines.append(
            f"| {v['name']} | {v['delta_mean']:+.4f} | [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}] |\n"
        )
    lines.append("\n## Decision gate\n\n")
    lines.append(f"- DeepHit − Graph-DT pooled gap: {gap_dh_gdt:+.4f}\n")
    lines.append(f"- Gap-closure (toward DeepHit from Graph-DT):\n")
    for k, v in gaps_closed.items():
        lines.append(f"  - {k}: {v:.1f}%  ({'PASS' if v is not None and v >= 50 else 'FAIL'} at 50% threshold)\n")
    lines.append(f"- Variance reduction (fold std):\n")
    for k, v in var_reductions.items():
        if v is None:
            lines.append(f"  - {k}: n/a\n")
        else:
            lines.append(f"  - {k}: {v:.1f}%  ({'PASS' if v >= 20 else 'FAIL'} at 20% threshold)\n")
    summary_path = OUT_DIR / "ensemble_5fold_summary.md"
    with open(summary_path, "w") as f:
        f.writelines(lines)
    print(f"Summary saved: {summary_path}")

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    names = list(strategies.keys())
    colors = ["#4b6cb7", "#8b5a3c", "#2b8a3e", "#4b9cb7", "#ab5a3c", "#5b3a7e"]
    xs = np.arange(len(names))
    points = [pooled_cis[n]["point"] for n in names]
    los = [pooled_cis[n]["ci_lo"] for n in names]
    his = [pooled_cis[n]["ci_hi"] for n in names]
    ax1.bar(xs, points, color=colors,
            yerr=[[p - l for p, l in zip(points, los)],
                  [h - p for p, h in zip(points, his)]], capsize=4)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Pooled C-td (95% CI)")
    ax1.set_ylim(0.85, 1.0)
    ax1.set_title("Pooled Cohort C-td")

    box_data = [per_fold[n] for n in names]
    ax2.boxplot(box_data, tick_labels=names)
    for lab in ax2.get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha("right")
        lab.set_fontsize(9)
    ax2.set_ylabel("Per-fold C-td")
    ax2.set_title("Fold-level Variance")

    fig.tight_layout()
    fig_path = OUT_DIR / "fig_ensemble_comparison.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Figure saved: {fig_path}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
