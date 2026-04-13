#!/usr/bin/env python3
"""Step 5: Dynamic-DeepHit for NSD-ISS Stage Transitions.

Runs 5-fold stratified CV of a GRU-based competing-risks survival model
on the longitudinal NSD-ISS staging data.

Usage:
    python scripts/paper3/run_deephit.py [--n-epochs 100] [--hidden-dim 128]

Outputs:
    outputs/paper3_deephit/
        deephit_results.json   — C-td, IBS, per-fold and per-transition metrics
        per_transition_ctd.csv — C-td broken down by destination stage
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.paper3.dynamic_deephit import (
    cross_validate,
    save_deephit_results,
)

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "07_paper3_features" / "longitudinal_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper3_deephit"


def main():
    parser = argparse.ArgumentParser(description="Run Dynamic-DeepHit model")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-gru-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.1, help="Ranking loss weight")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Save per-fold checkpoints to this directory",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    features = pd.read_csv(FEATURES_PATH, low_memory=False)
    print(f"  {len(features)} observations from {features['PATNO'].nunique()} patients")
    print(
        f"  Stage distribution:\n{features['nsd_stage'].value_counts().sort_index().to_string()}"
    )

    # Run cross-validation
    print("\n" + "=" * 70)
    print("DYNAMIC-DEEPHIT: 5-Fold Stratified CV")
    print("=" * 70)

    t0 = time.time()
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    result = cross_validate(
        features,
        n_folds=args.n_folds,
        hidden_dim=args.hidden_dim,
        n_gru_layers=args.n_gru_layers,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        alpha=args.alpha,
        patience=args.patience,
        verbose=True,
        seed=args.seed,
        checkpoint_dir=checkpoint_dir,
    )
    elapsed = time.time() - t0

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  C-td:  {result.c_td:.4f} ± {result.c_td_std:.4f}")
    print(f"  IBS:   {result.ibs:.4f} ± {result.ibs_std:.4f}")
    print(f"  Folds: {result.c_td_per_fold}")

    print("\n  Brier scores at horizons:")
    for h, bs in result.brier_at_horizons.items():
        print(f"    {h}: {bs:.4f}")

    if result.per_transition_ctd:
        print("\n  Per-transition C-td:")
        for trans, ctd in sorted(result.per_transition_ctd.items()):
            print(f"    {trans}: {ctd:.4f}")

    print(
        f"\n  Episodes: {result.n_episodes} "
        f"({result.n_events} events, {result.n_censored} censored)"
    )
    print(f"  Total time: {elapsed:.1f}s")

    # Save
    save_deephit_results(result, OUTPUT_DIR)

    # Compare with Markov baseline
    markov_path = PROJECT_ROOT / "outputs" / "paper3_markov" / "markov_results.json"
    if markov_path.exists():
        with open(markov_path) as f:
            markov = json.load(f)
        print("\n  Comparison with Markov baseline:")
        print(f"    Markov NLL:    {markov.get('log_likelihood', 'N/A')}")
        print(f"    DeepHit C-td:  {result.c_td:.4f}")
        print("    (Direct comparison requires computing C-td for Markov model)")

    print("\nDone!")


if __name__ == "__main__":
    main()
