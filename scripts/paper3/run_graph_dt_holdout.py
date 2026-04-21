#!/usr/bin/env python3
"""Pre-registered holdout rerun of Graph Digital Twin v5 for Paper 3 rigor.

Forks scripts/paper3/run_graph_dt.py. Trains ONCE on the 1,520-patient dev set
(internal 80/20 val split, seed=2026) and evaluates on the 380-patient holdout
(seed=2026) from data/06_longitudinal_staging/holdout_v1_patnos.json.

Graph-DT v5 (the final winning architecture): attention pool + warm-start
gated fusion (bias=-5.0) + 18 baseline features + graph smoothing (λ=0.01).
Hyperparameters match the Paper 3 published run:
    hidden_dim=128, n_gru_layers=2, gat_heads=4, gat_layers=2, k=15,
    lr=1e-3, batch=64, dropout=0.3, alpha=0.1, patience=15, n_epochs=100.

Outputs
-------
outputs/paper3_checkpoints/holdout_v1/graph_dt.pt        — single trained model
outputs/paper3_holdout_v1/graph_dt_predictions.csv       — per-patient predictions
outputs/paper3_holdout_v1/graph_dt_holdout_metrics.json  — C-td/IBS/gate/per-transition
outputs/paper3_holdout_v1/graph_dt_training_history.json — loss curves
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.paper3.dynamic_deephit import (  # noqa: E402
    N_TIME_BINS,
    TIME_BIN_ENDS,
    build_patient_arrays,
    compute_brier_score,
    compute_ctd,
    compute_feature_stats,
    compute_ibs,
    extract_episodes,
)
from giman_pipeline.paper3.graph_digital_twin import (  # noqa: E402
    GraphDeepHitDataset,
    GraphDigitalTwin,
    build_patient_graph,
    predict_all_graph,
    train_graph_model,
)
from giman_pipeline.paper3.multistate_markov import N_STATES, STAGE_LABELS  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "07_paper3_features" / "longitudinal_features.csv"
HOLDOUT_JSON = DATA_DIR / "06_longitudinal_staging" / "holdout_v1_patnos.json"

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "paper3_checkpoints" / "holdout_v1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper3_holdout_v1"

HOLDOUT_SEED = 2026
INTERNAL_VAL_SEED = 2026
MODEL_INIT_SEED = 42


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _per_transition_ctd_single(preds: dict) -> dict[str, float]:
    """Per-destination-stage C-td, single pass."""
    results = {}
    ev = preds["event_idxs"].numpy()
    tb = preds["time_bins"].numpy()
    cens = preds["censored"].numpy()
    cif = preds["cif"].numpy()
    for k in range(N_STATES):
        conc = disc = tied_n = 0
        unc_k = np.where((~cens) & (ev == k))[0]
        if len(unc_k) < 2:
            continue
        rng = np.random.RandomState(42)
        n_s = min(len(unc_k), 200)
        samp = rng.choice(unc_k, size=n_s, replace=False)
        for a in range(len(samp)):
            for b in range(a + 1, len(samp)):
                i, j = samp[a], samp[b]
                if tb[i] == tb[j]:
                    continue
                if tb[i] > tb[j]:
                    i, j = j, i
                ci_ = cif[i, k, tb[i]]
                cj_ = cif[j, k, tb[i]]
                if ci_ > cj_:
                    conc += 1
                elif ci_ < cj_:
                    disc += 1
                else:
                    tied_n += 1
        tot = conc + disc + 0.5 * tied_n
        if tot > 0:
            results[f"→{STAGE_LABELS[k]}"] = conc / tot
    return results


def _extract_predicted_time_months(cif_cause: np.ndarray) -> float | None:
    time_bins_months = np.array(TIME_BIN_ENDS, dtype=float)
    above_half = np.where(cif_cause > 0.5)[0]
    if len(above_half) > 0:
        return float(time_bins_months[above_half[0]])
    pmf = np.diff(np.concatenate([[0.0], cif_cause]))
    if pmf.max() > 0.01:
        return float(time_bins_months[pmf.argmax()])
    return None


def build_predictions_csv(preds: dict, episodes_ordered: list) -> pd.DataFrame:
    cif = preds["cif"].numpy()
    time_bins = preds["time_bins"].numpy()
    rows = []
    j_5yr = 7  # TIME_BIN_ENDS[7] = 60 months

    for i, ep in enumerate(episodes_ordered):
        src = STAGE_LABELS[ep.current_stage_idx]
        dst = STAGE_LABELS[ep.event_stage_idx] if not ep.censored else "(censored)"
        k_for_cif = (
            ep.event_stage_idx
            if ep.event_stage_idx >= 0
            else int(cif[i].sum(axis=-1).argmax())
        )
        cif_5yr = float(cif[i, k_for_cif, min(j_5yr, N_TIME_BINS - 1)])
        mode_cause = int(cif[i].max(axis=-1).argmax())
        t_pred = _extract_predicted_time_months(cif[i, mode_cause, :])

        rows.append(
            {
                "patno": int(ep.patno),
                "source_stage": src,
                "dest_stage": dst,
                "observed_time_months": float(ep.duration_months),
                "observed_time_bin": int(time_bins[i]),
                "censored": bool(ep.censored),
                "cif_5yr": cif_5yr,
                "pred_cause_idx": int(mode_cause),
                "pred_cause_stage": STAGE_LABELS[mode_cause],
                "pred_time_months_50th": t_pred if t_pred is not None else np.nan,
                "pred_time_months_90ci_lo": np.nan,
                "pred_time_months_90ci_hi": np.nan,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Graph-DT holdout rerun (seed 2026)")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-gru-layers", type=int, default=2)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument("--k-neighbors", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--graph-smooth-weight", type=float, default=0.01)
    parser.add_argument("--model-init-seed", type=int, default=MODEL_INIT_SEED)
    args = parser.parse_args()

    t0 = time.time()

    print("Loading features...")
    features_df = pd.read_csv(FEATURES_PATH, low_memory=False)
    with open(HOLDOUT_JSON) as f:
        split = json.load(f)

    dev_pats = set(split["dev"])
    holdout_pats = set(split["holdout"])
    print(f"  Dev:     {len(dev_pats)} patients")
    print(f"  Holdout: {len(holdout_pats)} patients")

    patient_arrays, col_names = build_patient_arrays(features_df)
    input_dim = len(col_names)
    episodes = extract_episodes(features_df, verbose=False)

    dev_episodes = [e for e in episodes if e.patno in dev_pats]
    holdout_episodes = [e for e in episodes if e.patno in holdout_pats]
    print(f"  Episodes: dev={len(dev_episodes)}  holdout={len(holdout_episodes)}")

    # Build the patient graph on ALL patients (no label leakage — uses baseline
    # features only, same policy as Paper 3 cross_validate). This INCLUDES the
    # holdout patients' baseline features as graph nodes, but the holdout
    # EVENTS are never seen by the model during training — same transductive
    # assumption as the published Paper 3 results.
    all_patnos = sorted(set(e.patno for e in episodes))
    print(f"  Building kNN graph on {len(all_patnos)} patients (k={args.k_neighbors})...")
    edge_index, edge_weight, node_baseline = build_patient_graph(
        features_df, all_patnos, k_neighbors=args.k_neighbors
    )
    pat_to_gidx = {p: i for i, p in enumerate(all_patnos)}
    n_baseline_features = node_baseline.size(1)
    print(
        f"  Graph: {len(all_patnos)} nodes, {edge_index.size(1)} edges, "
        f"{n_baseline_features} baseline feats"
    )

    # Internal train/val split of dev (80/20, seed=2026)
    rng = np.random.RandomState(INTERNAL_VAL_SEED)
    dev_list = sorted(dev_pats)
    rng.shuffle(dev_list)
    n_val = max(1, len(dev_list) // 5)
    val_pats = set(dev_list[:n_val])
    atrain_pats = set(dev_list[n_val:])
    print(f"  Internal train: {len(atrain_pats)} / val: {len(val_pats)}")

    train_eps = [e for e in dev_episodes if e.patno in atrain_pats]
    val_eps = [e for e in dev_episodes if e.patno in val_pats]

    means, stds = compute_feature_stats(patient_arrays, atrain_pats)

    train_ds = GraphDeepHitDataset(
        train_eps, patient_arrays, means, stds, pat_to_gidx
    )
    val_ds = GraphDeepHitDataset(
        val_eps, patient_arrays, means, stds, pat_to_gidx
    )
    holdout_ds = GraphDeepHitDataset(
        holdout_episodes, patient_arrays, means, stds, pat_to_gidx
    )

    device = _get_device()
    print(f"  Device: {device}")

    torch.manual_seed(args.model_init_seed)
    model = GraphDigitalTwin(
        input_dim=input_dim,
        n_baseline_features=n_baseline_features,
        hidden_dim=args.hidden_dim,
        n_gru_layers=args.n_gru_layers,
        gat_heads=args.gat_heads,
        gat_layers=args.gat_layers,
        dropout=args.dropout,
    ).to(device)

    print("\nTraining Graph-DT v5 on dev set...")
    history, best_val, best_state = train_graph_model(
        model,
        train_ds,
        val_ds,
        device,
        node_baseline,
        edge_index,
        edge_weight,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        alpha=args.alpha,
        graph_smooth_weight=args.graph_smooth_weight,
        verbose=True,
    )

    # --- Evaluate on holdout ---
    print("\nEvaluating on holdout...")
    preds = predict_all_graph(
        model,
        holdout_ds,
        device,
        node_baseline,
        edge_index,
        edge_weight,
    )
    ctd = compute_ctd(preds)
    ibs = compute_ibs(preds)
    brier_5yr = compute_brier_score(preds, 7)
    per_trans = _per_transition_ctd_single(preds)

    eval_horizons = {"1yr": 12, "2yr": 24, "5yr": 60, "10yr": 120}
    brier_hz = {}
    for label, months in eval_horizons.items():
        j = next(i for i, t in enumerate(TIME_BIN_ENDS) if months <= t)
        b = compute_brier_score(preds, j)
        brier_hz[label] = float(b) if not np.isnan(b) else None

    print(f"  C-td (holdout):  {ctd:.4f}")
    print(f"  IBS  (holdout):  {ibs:.4f}")
    print(f"  Brier@5yr:       {brier_5yr:.4f}")
    print("  Per-transition C-td:")
    for k, v in sorted(per_trans.items()):
        print(f"    {k}: {v:.4f}")

    # --- Gate activation summary on holdout ---
    with torch.no_grad():
        nb = node_baseline.to(device)
        ei = edge_index.to(device)
        ew = edge_weight.to(device)
        graph_feats = model.compute_graph_features(nb, ei, ew)
        gate_means_per_stage: dict[int, list[float]] = {}
        from torch.utils.data import DataLoader
        from giman_pipeline.paper3.graph_digital_twin import graph_collate_fn

        loader = DataLoader(
            holdout_ds,
            batch_size=128,
            shuffle=False,
            collate_fn=graph_collate_fn,
        )
        for batch in loader:
            seqs = batch["sequences"].to(device)
            slens = batch["seq_lens"]
            sidxs = batch["stage_idxs"].to(device)
            gidxs = batch["graph_idxs"].to(device)
            # Recover `temporal` the same way forward() does
            from torch.nn.utils.rnn import (
                pack_padded_sequence,
                pad_packed_sequence,
            )
            sorted_lens, sort_idx = slens.sort(descending=True)
            sorted_seqs = seqs[sort_idx]
            packed = pack_padded_sequence(
                sorted_seqs, sorted_lens.clamp(min=1).cpu(), batch_first=True
            )
            gru_out, h_n = model.gru(packed)
            gru_out_p, _ = pad_packed_sequence(gru_out, batch_first=True)
            _, unsort = sort_idx.sort()
            gru_out_u = gru_out_p[unsort]
            last_h = h_n[-1][unsort]
            attn_ctx = model.temporal_attn(gru_out_u, slens.to(device))
            temporal = attn_ctx + last_h
            graph_feat = graph_feats[gidxs]
            gate_input = torch.cat([temporal, graph_feat], dim=-1)
            g = torch.sigmoid(model.gate_linear(gate_input)).mean(dim=-1)
            for stage_idx_val, gate_val in zip(sidxs.cpu().numpy(), g.cpu().numpy()):
                gate_means_per_stage.setdefault(int(stage_idx_val), []).append(
                    float(gate_val)
                )
        gate_summary = {
            STAGE_LABELS[k]: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "n": len(v),
            }
            for k, v in gate_means_per_stage.items()
        }

    print("  Gate activation (mean per source stage):")
    for s, v in gate_summary.items():
        print(f"    {s}: {v['mean']:.4f} ± {v['std']:.4f} (n={v['n']})")

    # --- Save checkpoint ---
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "graph_dt.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "n_gru_layers": args.n_gru_layers,
            "n_baseline_features": n_baseline_features,
            "gat_heads": args.gat_heads,
            "gat_layers": args.gat_layers,
            "dropout": args.dropout,
            "n_causes": N_STATES,
            "n_time_bins": N_TIME_BINS,
            "means": means,
            "stds": stds,
            "dev_pats": sorted(dev_pats),
            "holdout_pats": sorted(holdout_pats),
            "train_pats": sorted(atrain_pats),
            "val_pats": sorted(val_pats),
            "col_names": col_names,
            "edge_index": edge_index.cpu(),
            "edge_weight": edge_weight.cpu(),
            "node_baseline": node_baseline.cpu(),
            "pat_to_gidx": pat_to_gidx,
            "k_neighbors": args.k_neighbors,
            "holdout_seed": HOLDOUT_SEED,
            "internal_val_seed": INTERNAL_VAL_SEED,
            "model_init_seed": args.model_init_seed,
            "holdout_ctd": ctd,
            "holdout_ibs": ibs,
            "best_val_loss": best_val,
            "variant": "graph_dt_v5_holdout_v1",
        },
        ckpt_path,
    )
    print(f"\nCheckpoint: {ckpt_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_df = build_predictions_csv(preds, holdout_episodes)
    pred_csv = OUTPUT_DIR / "graph_dt_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)
    print(f"Predictions: {pred_csv} ({len(pred_df)} rows)")

    metrics = {
        "model": "Graph-DT v5",
        "split_version": "holdout_v1",
        "split_seed": HOLDOUT_SEED,
        "model_init_seed": args.model_init_seed,
        "internal_val_seed": INTERNAL_VAL_SEED,
        "n_dev_patients": len(dev_pats),
        "n_holdout_patients": len(holdout_pats),
        "n_holdout_episodes": len(holdout_episodes),
        "n_holdout_events": int(sum(1 for e in holdout_episodes if not e.censored)),
        "c_td_holdout": float(ctd),
        "ibs_holdout": float(ibs),
        "brier_5yr_holdout": float(brier_5yr)
        if not np.isnan(brier_5yr)
        else None,
        "brier_at_horizons": brier_hz,
        "per_transition_ctd": per_trans,
        "gate_mean_per_stage": gate_summary,
        "best_val_loss": float(best_val),
        "graph_stats": {
            "n_nodes": len(all_patnos),
            "n_edges": int(edge_index.size(1)),
            "avg_degree": float(edge_index.size(1) / len(all_patnos)),
            "k_neighbors": args.k_neighbors,
            "n_baseline_features": int(n_baseline_features),
        },
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "n_gru_layers": args.n_gru_layers,
            "gat_heads": args.gat_heads,
            "gat_layers": args.gat_layers,
            "k_neighbors": args.k_neighbors,
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "dropout": args.dropout,
            "alpha": args.alpha,
            "patience": args.patience,
            "graph_smooth_weight": args.graph_smooth_weight,
        },
        "elapsed_seconds": time.time() - t0,
    }
    metrics_json = OUTPUT_DIR / "graph_dt_holdout_metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics:     {metrics_json}")

    hist_json = OUTPUT_DIR / "graph_dt_training_history.json"
    with open(hist_json, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History:     {hist_json}")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
