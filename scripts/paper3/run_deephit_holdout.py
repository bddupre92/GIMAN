#!/usr/bin/env python3
"""Pre-registered holdout rerun of Dynamic-DeepHit for Paper 3 submission rigor.

Forks scripts/paper3/run_deephit.py. Instead of 5-fold stratified CV, trains
ONCE on the 1,520-patient dev set (internal 80/20 val split, seed=2026) and
evaluates on the pre-registered 380-patient holdout split (seed=2026) from
data/06_longitudinal_staging/holdout_v1_patnos.json.

Winning hyperparameters from the 5-fold CV are FIXED:
    hidden_dim=128, n_gru_layers=2, lr=1e-3, batch=64, dropout=0.3,
    alpha=0.1, patience=15, n_epochs=100
(see paper3/run_deephit.py defaults — those are the numbers reported in the
published Paper 3 tables.)

Outputs
-------
outputs/paper3_checkpoints/holdout_v1/deephit.pt        — single trained model
outputs/paper3_holdout_v1/deephit_predictions.csv       — per-patient predictions
outputs/paper3_holdout_v1/deephit_holdout_metrics.json  — C-td/IBS/per-transition
outputs/paper3_holdout_v1/deephit_training_history.json — loss curves
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
    DeepHitDataset,
    DynamicDeepHit,
    N_TIME_BINS,
    TIME_BIN_ENDS,
    _get_device,
    build_patient_arrays,
    compute_brier_score,
    compute_ctd,
    compute_feature_stats,
    compute_ibs,
    extract_episodes,
    predict_all,
    train_model,
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
INTERNAL_VAL_SEED = 2026  # for dev → train/val split
MODEL_INIT_SEED = 42  # KEEP existing model init seed for apples-to-apples comparison


def _per_transition_ctd_single(preds: dict) -> dict[str, float]:
    """Per-destination-stage C-td, single pass (no folds)."""
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
    """First t-bin where CIF > 0.5; fallback to PMF argmax."""
    time_bins_months = np.array(TIME_BIN_ENDS, dtype=float)
    above_half = np.where(cif_cause > 0.5)[0]
    if len(above_half) > 0:
        return float(time_bins_months[above_half[0]])
    pmf = np.diff(np.concatenate([[0.0], cif_cause]))
    if pmf.max() > 0.01:
        return float(time_bins_months[pmf.argmax()])
    return None


def build_predictions_csv(
    preds: dict,
    episodes_ordered: list,
) -> pd.DataFrame:
    """Build per-episode prediction CSV with CIFs and predicted timing.

    The predict_all() helper evaluates episodes in dataset order — we recover
    that order by inspecting `predictions["event_idxs"]` alongside the episodes
    list used to build the DeepHitDataset. As long as DataLoader has
    shuffle=False (which predict_all does), the order is preserved.

    Columns:
        patno, source_stage, dest_stage, observed_time_months,
        cif_5yr (at the observed event's destination),
        pred_time_months_50th (median predicted time to most-likely cause),
        pred_time_months_90ci_lo, pred_time_months_90ci_hi  (placeholders — P4 fills)
    """
    cif = preds["cif"].numpy()  # (n, K, J)
    event_idxs = preds["event_idxs"].numpy()
    time_bins = preds["time_bins"].numpy()
    censored = preds["censored"].numpy()

    rows = []
    j_5yr = 7  # 60-month bin; TIME_BIN_ENDS[7] = 60

    for i, ep in enumerate(episodes_ordered):
        src = STAGE_LABELS[ep.current_stage_idx]
        if ep.censored:
            dst = "(censored)"
        else:
            dst = STAGE_LABELS[ep.event_stage_idx]

        # CIF at 5yr for the ACTUAL dest stage (or best-guess if censored)
        k_for_cif = ep.event_stage_idx if ep.event_stage_idx >= 0 else int(cif[i].sum(axis=-1).argmax())
        cif_5yr = float(cif[i, k_for_cif, min(j_5yr, N_TIME_BINS - 1)])

        # Predicted time is taken from the most-likely cause's CIF curve
        mode_cause = int(cif[i].max(axis=-1).argmax())  # cause with highest peak CIF
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
                # Conformal bands are computed by run_conformal_survival_holdout.py
                "pred_time_months_90ci_lo": np.nan,
                "pred_time_months_90ci_hi": np.nan,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="DeepHit holdout rerun (seed 2026)")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-gru-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--model-init-seed", type=int, default=MODEL_INIT_SEED)
    args = parser.parse_args()

    t0 = time.time()

    # --- Load data + split ---
    print("Loading features...")
    features_df = pd.read_csv(FEATURES_PATH, low_memory=False)
    with open(HOLDOUT_JSON) as f:
        split = json.load(f)

    dev_pats = set(split["dev"])
    holdout_pats = set(split["holdout"])
    print(f"  Dev:     {len(dev_pats)} patients")
    print(f"  Holdout: {len(holdout_pats)} patients")

    # --- Prep data ---
    patient_arrays, col_names = build_patient_arrays(features_df)
    input_dim = len(col_names)
    episodes = extract_episodes(features_df, verbose=False)
    print(f"  Episodes: {len(episodes)} total")

    dev_episodes = [e for e in episodes if e.patno in dev_pats]
    holdout_episodes = [e for e in episodes if e.patno in holdout_pats]
    print(f"  Dev episodes:     {len(dev_episodes)}")
    print(f"  Holdout episodes: {len(holdout_episodes)}")

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

    # Feature standardisation from TRAIN patients only
    means, stds = compute_feature_stats(patient_arrays, atrain_pats)

    train_ds = DeepHitDataset(train_eps, patient_arrays, means, stds)
    val_ds = DeepHitDataset(val_eps, patient_arrays, means, stds)
    holdout_ds = DeepHitDataset(holdout_episodes, patient_arrays, means, stds)

    # --- Build + train model ---
    device = _get_device()
    print(f"  Device: {device}")

    torch.manual_seed(args.model_init_seed)
    model = DynamicDeepHit(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        n_gru_layers=args.n_gru_layers,
        dropout=args.dropout,
    ).to(device)

    print("\nTraining DeepHit on dev set...")
    history, best_val, best_state = train_model(
        model,
        train_ds,
        val_ds,
        device,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        alpha=args.alpha,
        verbose=True,
    )

    # --- Evaluate on holdout ---
    print("\nEvaluating on holdout...")
    preds = predict_all(model, holdout_ds, device)
    ctd = compute_ctd(preds)
    ibs = compute_ibs(preds)
    brier_5yr = compute_brier_score(preds, 7)  # bin 7 = 60 months
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

    # --- Save checkpoint ---
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "deephit.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "n_gru_layers": args.n_gru_layers,
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
            "holdout_seed": HOLDOUT_SEED,
            "internal_val_seed": INTERNAL_VAL_SEED,
            "model_init_seed": args.model_init_seed,
            "holdout_ctd": ctd,
            "holdout_ibs": ibs,
            "best_val_loss": best_val,
            "variant": "deephit_holdout_v1",
        },
        ckpt_path,
    )
    print(f"\nCheckpoint: {ckpt_path}")

    # --- Save predictions CSV ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_df = build_predictions_csv(preds, holdout_episodes)
    pred_csv = OUTPUT_DIR / "deephit_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)
    print(f"Predictions: {pred_csv} ({len(pred_df)} rows)")

    # --- Save metrics JSON ---
    metrics = {
        "model": "DeepHit",
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
        "best_val_loss": float(best_val),
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "n_gru_layers": args.n_gru_layers,
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "dropout": args.dropout,
            "alpha": args.alpha,
            "patience": args.patience,
        },
        "elapsed_seconds": time.time() - t0,
    }
    metrics_json = OUTPUT_DIR / "deephit_holdout_metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics:     {metrics_json}")

    # --- Save training history ---
    hist_json = OUTPUT_DIR / "deephit_training_history.json"
    with open(hist_json, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History:     {hist_json}")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
