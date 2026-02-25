#!/usr/bin/env python3
"""
Step 6: Graph-Informed Digital Twin for NSD-ISS Stage Transitions.

Runs 5-fold stratified CV of the spatio-temporal Graph-DT model
(GRU temporal encoder + GAT graph attention).

Usage:
    python scripts/paper3/run_graph_dt.py [--n-epochs 100] [--k-neighbors 15]

Outputs:
    outputs/paper3_graph_dt/
        graph_dt_results.json  — C-td, IBS, graph stats, per-fold metrics
        per_transition_ctd.csv — Per-transition C-td breakdown
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.paper3.graph_digital_twin import (
    cross_validate,
    save_graph_dt_results,
)

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "07_paper3_features" / "longitudinal_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper3_graph_dt"


def main():
    parser = argparse.ArgumentParser(description="Run Graph Digital Twin model")
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
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Save per-fold checkpoints to this directory")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    features = pd.read_csv(FEATURES_PATH, low_memory=False)
    print(f"  {len(features)} observations from {features['PATNO'].nunique()} patients")

    print("\n" + "=" * 70)
    print("GRAPH DIGITAL TWIN: 5-Fold Stratified CV")
    print("=" * 70)

    t0 = time.time()
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    result = cross_validate(
        features,
        n_folds=args.n_folds,
        hidden_dim=args.hidden_dim,
        n_gru_layers=args.n_gru_layers,
        gat_heads=args.gat_heads,
        gat_layers=args.gat_layers,
        k_neighbors=args.k_neighbors,
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

    print(f"\n  Graph: {result.graph_stats}")
    print(f"  Episodes: {result.n_episodes} ({result.n_events} events)")
    print(f"  Total time: {elapsed:.1f}s")

    save_graph_dt_results(result, OUTPUT_DIR)

    # Compare with baselines
    for name, path in [
        ("Markov", "paper3_markov/markov_results.json"),
        ("DeepHit", "paper3_deephit/deephit_results.json"),
    ]:
        fpath = PROJECT_ROOT / "outputs" / path
        if fpath.exists():
            with open(fpath) as f:
                d = json.load(f)
            ctd = d.get("c_td", "N/A")
            ibs = d.get("ibs", "N/A")
            print(f"\n  {name}: C-td={ctd}, IBS={ibs}")

    print(f"\n  Graph-DT: C-td={result.c_td:.4f}, IBS={result.ibs:.4f}")
    print("\nDone!")


if __name__ == "__main__":
    main()
