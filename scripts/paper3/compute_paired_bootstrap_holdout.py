#!/usr/bin/env python3
"""Paired bootstrap Δ(Graph-DT − DeepHit) on the pre-registered holdout.

Reads:
    outputs/paper3_holdout_v1/deephit_predictions.csv
    outputs/paper3_holdout_v1/graph_dt_predictions.csv
    outputs/paper3_checkpoints/holdout_v1/{deephit,graph_dt}.pt  — for raw CIFs

Writes:
    outputs/paper3_holdout_v1/paired_bootstrap.json — paired Δ + 95% CI +
                                                     Wilcoxon signed-rank p
    outputs/paper3_holdout_v1/holdout_report.md     — full holdout report

Patient-level paired bootstrap: each resample draws patients (with replacement),
computes a single-pass C-td for both models on the drawn episodes, records the
delta. 1,000 resamples → 95% CI via percentile method.

Wilcoxon signed-rank is evaluated on per-patient contributions: for each
patient, count the # of concordant minus # discordant pairs among the
uncensored holdout pool with that patient as the early-event member — this
gives a sign per patient, and Wilcoxon tests if the sign distribution is
centered at zero.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.paper3.dynamic_deephit import (  # noqa: E402
    DeepHitDataset,
    build_patient_arrays,
    extract_episodes,
    predict_all,
)
from giman_pipeline.paper3.graph_digital_twin import (  # noqa: E402
    GraphDeepHitDataset,
    predict_all_graph,
)
from giman_pipeline.paper3.multistate_markov import STAGE_LABELS  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "07_paper3_features" / "longitudinal_features.csv"
HOLDOUT_JSON = DATA_DIR / "06_longitudinal_staging" / "holdout_v1_patnos.json"

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "paper3_checkpoints" / "holdout_v1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper3_holdout_v1"

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 2026


def _compute_ctd_sampled(preds_or_cif, ev, tb, cens, rng, n_pairs: int = 50000):
    """Vectorised C-td on a fixed set using sampled pairs (pair-level)."""
    cif = preds_or_cif
    uncensored = np.where(~cens)[0]
    if len(uncensored) < 2:
        return 0.5

    # Sample n_pairs (i, j) pairs uniformly from uncensored indices
    n_max = min(n_pairs, len(uncensored) * (len(uncensored) - 1) // 2)
    a_idx = rng.randint(0, len(uncensored), size=n_max)
    b_idx = rng.randint(0, len(uncensored), size=n_max)
    # Drop self-pairs
    ok = a_idx != b_idx
    a_idx = uncensored[a_idx[ok]]
    b_idx = uncensored[b_idx[ok]]
    if len(a_idx) == 0:
        return 0.5

    # Only keep pairs with same event type and different time bin
    same_ev = ev[a_idx] == ev[b_idx]
    diff_t = tb[a_idx] != tb[b_idx]
    keep = same_ev & diff_t
    a_idx = a_idx[keep]
    b_idx = b_idx[keep]
    if len(a_idx) == 0:
        return 0.5

    # Ensure a is earlier event (tb[a] < tb[b])
    swap = tb[a_idx] > tb[b_idx]
    i_arr = np.where(swap, b_idx, a_idx)
    j_arr = np.where(swap, a_idx, b_idx)

    k_arr = ev[i_arr]
    t_arr = tb[i_arr]

    # Gather CIFs using advanced indexing
    ci_vals = cif[i_arr, k_arr, t_arr]
    cj_vals = cif[j_arr, k_arr, t_arr]

    conc = int((ci_vals > cj_vals).sum())
    disc = int((ci_vals < cj_vals).sum())
    tied = int((ci_vals == cj_vals).sum())
    tot = conc + disc + 0.5 * tied
    return conc / tot if tot > 0 else 0.5


def load_holdout_inference(device: torch.device):
    """Run both models on the holdout, return aligned CIFs + metadata."""
    features_df = pd.read_csv(FEATURES_PATH, low_memory=False)
    with open(HOLDOUT_JSON) as f:
        split = json.load(f)

    patient_arrays, _ = build_patient_arrays(features_df)
    episodes = extract_episodes(features_df, verbose=False)
    holdout_pats = set(split["holdout"])
    holdout_eps = [e for e in episodes if e.patno in holdout_pats]
    print(f"  {len(holdout_eps)} holdout episodes")

    # --- DeepHit ---
    from giman_pipeline.paper3.dynamic_deephit import (
        DynamicDeepHit, N_STATES, N_TIME_BINS,
    )
    dh_ckpt = torch.load(
        CHECKPOINT_DIR / "deephit.pt", weights_only=False, map_location=device
    )
    dh_model = DynamicDeepHit(
        input_dim=dh_ckpt["input_dim"],
        hidden_dim=dh_ckpt["hidden_dim"],
        n_gru_layers=dh_ckpt["n_gru_layers"],
        n_causes=N_STATES,
        n_time_bins=N_TIME_BINS,
        dropout=dh_ckpt.get("dropout", 0.3),
    )
    dh_model.load_state_dict(dh_ckpt["model_state_dict"])
    dh_model.to(device)
    dh_model.eval()
    dh_ds = DeepHitDataset(
        holdout_eps, patient_arrays, dh_ckpt["means"], dh_ckpt["stds"]
    )
    dh_preds = predict_all(dh_model, dh_ds, device)

    # --- Graph-DT ---
    from giman_pipeline.paper3.graph_digital_twin import GraphDigitalTwin
    gdt_ckpt = torch.load(
        CHECKPOINT_DIR / "graph_dt.pt", weights_only=False, map_location=device
    )
    gdt_model = GraphDigitalTwin(
        input_dim=gdt_ckpt["input_dim"],
        n_baseline_features=gdt_ckpt["n_baseline_features"],
        hidden_dim=gdt_ckpt["hidden_dim"],
        n_gru_layers=gdt_ckpt["n_gru_layers"],
        gat_heads=gdt_ckpt.get("gat_heads", 4),
        gat_layers=gdt_ckpt.get("gat_layers", 2),
        n_causes=N_STATES,
        n_time_bins=N_TIME_BINS,
        dropout=gdt_ckpt.get("dropout", 0.3),
    )
    gdt_model.load_state_dict(gdt_ckpt["model_state_dict"])
    gdt_model.to(device)
    gdt_model.eval()
    gdt_ds = GraphDeepHitDataset(
        holdout_eps,
        patient_arrays,
        gdt_ckpt["means"],
        gdt_ckpt["stds"],
        gdt_ckpt["pat_to_gidx"],
    )
    gdt_preds = predict_all_graph(
        gdt_model,
        gdt_ds,
        device,
        gdt_ckpt["node_baseline"],
        gdt_ckpt["edge_index"],
        gdt_ckpt["edge_weight"],
    )

    cif_dh = dh_preds["cif"].numpy()
    cif_gdt = gdt_preds["cif"].numpy()
    ev = dh_preds["event_idxs"].numpy()
    tb = dh_preds["time_bins"].numpy()
    cens = dh_preds["censored"].numpy()
    # DeepHit and Graph-DT iterate episodes in the same order because
    # DataLoader shuffle=False + same episode list
    assert (gdt_preds["event_idxs"].numpy() == ev).all(), (
        "Episode ordering mismatch between DeepHit and Graph-DT predictions"
    )

    patnos = np.array([e.patno for e in holdout_eps])
    return cif_dh, cif_gdt, ev, tb, cens, patnos, holdout_eps


def paired_bootstrap_ctd_delta(
    cif_dh: np.ndarray,
    cif_gdt: np.ndarray,
    ev: np.ndarray,
    tb: np.ndarray,
    cens: np.ndarray,
    patnos: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
):
    """Patient-level paired bootstrap. Returns (delta_point, ci_lo, ci_hi, deltas)."""
    unique_patnos = np.unique(patnos)
    rng = np.random.RandomState(seed)

    # Precompute episode-index map
    pat_to_eps_idx: dict[int, list[int]] = {}
    for i, p in enumerate(patnos):
        pat_to_eps_idx.setdefault(int(p), []).append(i)

    deltas = []
    dh_vals = []
    gdt_vals = []

    for b in range(n_boot):
        sampled = rng.choice(unique_patnos, size=len(unique_patnos), replace=True)
        # Collect episode indices whose patno is in the sampled set
        idx_list = []
        for sp in sampled:
            idx_list.extend(pat_to_eps_idx[int(sp)])
        idx_arr = np.array(idx_list)
        cif_dh_b = cif_dh[idx_arr]
        cif_gdt_b = cif_gdt[idx_arr]
        ev_b = ev[idx_arr]
        tb_b = tb[idx_arr]
        cens_b = cens[idx_arr]

        sub_rng = np.random.RandomState(seed + b + 1)
        dh_ctd = _compute_ctd_sampled(cif_dh_b, ev_b, tb_b, cens_b, sub_rng)
        sub_rng = np.random.RandomState(seed + b + 1)  # same pair sampling
        gdt_ctd = _compute_ctd_sampled(cif_gdt_b, ev_b, tb_b, cens_b, sub_rng)
        deltas.append(gdt_ctd - dh_ctd)
        dh_vals.append(dh_ctd)
        gdt_vals.append(gdt_ctd)

    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "point_estimate": float(np.mean(deltas)),
        "delta_median": float(np.median(deltas)),
        "ci_95_lo": float(ci_lo),
        "ci_95_hi": float(ci_hi),
        "dh_mean": float(np.mean(dh_vals)),
        "gdt_mean": float(np.mean(gdt_vals)),
        "dh_median": float(np.median(dh_vals)),
        "gdt_median": float(np.median(gdt_vals)),
        "n_bootstrap": int(n_boot),
    }


def wilcoxon_per_patient(
    cif_dh: np.ndarray,
    cif_gdt: np.ndarray,
    ev: np.ndarray,
    tb: np.ndarray,
    cens: np.ndarray,
    patnos: np.ndarray,
):
    """Wilcoxon on per-patient (DeepHit − Graph-DT) concordance contributions.

    For each patient i (uncensored), we pick all uncensored episodes j with a
    DIFFERENT time bin. Concordance contribution = 1 if CIF for patient's event
    is correctly ordered, 0 if tied, -1 if discordant. Sum across j to get a
    per-patient score. Compare DeepHit vs Graph-DT scores.
    """
    # Precompute uncensored indices
    uncens = np.where(~cens)[0]
    if len(uncens) < 2:
        return {"n": 0, "wilcoxon_stat": None, "wilcoxon_p": None}

    dh_scores = []
    gdt_scores = []
    for i in uncens:
        k = ev[i]
        t = tb[i]
        # partners with different time bin
        mates = uncens[(ev[uncens] == k) & (tb[uncens] != t) & (uncens != i)]
        if len(mates) == 0:
            continue
        # patient i should have HIGHER CIF if their event is EARLIER than j
        dh_contrib = 0
        gdt_contrib = 0
        for j in mates:
            if t < tb[j]:  # i earlier → i's CIF should be > j's at t
                diff_dh = cif_dh[i, k, t] - cif_dh[j, k, t]
                diff_gdt = cif_gdt[i, k, t] - cif_gdt[j, k, t]
            else:  # i later → i's CIF at tb[j] should be LARGER (CIFs are monotone)
                diff_dh = cif_dh[j, k, tb[j]] - cif_dh[i, k, tb[j]]
                diff_gdt = cif_gdt[j, k, tb[j]] - cif_gdt[i, k, tb[j]]
            dh_contrib += 1 if diff_dh > 0 else (-1 if diff_dh < 0 else 0)
            gdt_contrib += 1 if diff_gdt > 0 else (-1 if diff_gdt < 0 else 0)
        dh_scores.append(dh_contrib / len(mates))
        gdt_scores.append(gdt_contrib / len(mates))

    dh_arr = np.array(dh_scores)
    gdt_arr = np.array(gdt_scores)

    if len(dh_arr) == 0 or np.allclose(dh_arr, gdt_arr):
        return {
            "n": int(len(dh_arr)),
            "wilcoxon_stat": None,
            "wilcoxon_p": None,
            "dh_mean_score": float(np.mean(dh_arr)) if len(dh_arr) else None,
            "gdt_mean_score": float(np.mean(gdt_arr)) if len(gdt_arr) else None,
        }

    try:
        stat, p = stats.wilcoxon(gdt_arr, dh_arr, alternative="two-sided")
    except ValueError:
        stat, p = None, None

    return {
        "n": int(len(dh_arr)),
        "wilcoxon_stat": float(stat) if stat is not None else None,
        "wilcoxon_p": float(p) if p is not None else None,
        "dh_mean_score": float(np.mean(dh_arr)),
        "gdt_mean_score": float(np.mean(gdt_arr)),
    }


def main():
    t0 = time.time()
    device = torch.device("cpu")  # CPU for reproducibility
    print("Loading inference on holdout...")
    cif_dh, cif_gdt, ev, tb, cens, patnos, holdout_eps = load_holdout_inference(device)

    print("\nRunning paired bootstrap (1,000 resamples)...")
    bs = paired_bootstrap_ctd_delta(cif_dh, cif_gdt, ev, tb, cens, patnos)
    print(f"  Δ(Graph-DT − DeepHit) point estimate: {bs['point_estimate']:.4f}")
    print(f"  95% CI: [{bs['ci_95_lo']:.4f}, {bs['ci_95_hi']:.4f}]")
    print(f"  DeepHit mean C-td:   {bs['dh_mean']:.4f}")
    print(f"  Graph-DT mean C-td:  {bs['gdt_mean']:.4f}")

    print("\nWilcoxon signed-rank on per-patient concordance scores...")
    wx = wilcoxon_per_patient(cif_dh, cif_gdt, ev, tb, cens, patnos)
    print(f"  n = {wx['n']}  stat = {wx['wilcoxon_stat']}  p = {wx['wilcoxon_p']}")

    # Load headline metrics from individual metric JSONs (written by training)
    with open(OUTPUT_DIR / "deephit_holdout_metrics.json") as f:
        dh_metrics = json.load(f)
    with open(OUTPUT_DIR / "graph_dt_holdout_metrics.json") as f:
        gdt_metrics = json.load(f)

    payload = {
        "paired_bootstrap": bs,
        "wilcoxon_per_patient": wx,
        "deephit_headline_ctd": dh_metrics["c_td_holdout"],
        "graph_dt_headline_ctd": gdt_metrics["c_td_holdout"],
        "n_holdout_patients": len(np.unique(patnos)),
        "n_holdout_episodes": int(len(patnos)),
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    out_path = OUTPUT_DIR / "paired_bootstrap.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWrote {out_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
