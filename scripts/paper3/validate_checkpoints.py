#!/usr/bin/env python3
"""Validate per-fold checkpoints by loading, reconstructing, and verifying C-td.

Loads each of the 10 checkpoints (5 DeepHit + 5 Graph-DT), reconstructs
the model, runs predict_cif on the fold's test patients, and checks that
the C-td matches the saved fold_ctd within tolerance.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.paper3.dynamic_deephit import (
    DeepHitDataset,
    build_patient_arrays,
    compute_ctd,
    extract_episodes,
    load_deephit_checkpoint,
    predict_all,
)
from giman_pipeline.paper3.graph_digital_twin import (
    GraphDeepHitDataset,
    load_graph_dt_checkpoint,
    predict_all_graph,
)

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "07_paper3_features" / "longitudinal_features.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "paper3_checkpoints"
TOLERANCE = 0.003


def validate_deephit_checkpoints(features_df, episodes, patient_arrays):
    """Validate all DeepHit fold checkpoints."""
    ckpt_dir = CHECKPOINT_DIR / "deephit"
    print("\n" + "=" * 60)
    print("VALIDATING DEEPHIT CHECKPOINTS")
    print("=" * 60)

    all_ok = True
    for fi in range(5):
        path = ckpt_dir / f"fold{fi}_deephit.pt"
        if not path.exists():
            print(f"  MISSING: {path}")
            all_ok = False
            continue

        model, cp = load_deephit_checkpoint(path)
        test_pats = set(cp["test_pats"])
        means = cp["means"]
        stds = cp["stds"]
        saved_ctd = cp["fold_ctd"]

        test_eps = [e for e in episodes if e.patno in test_pats]
        test_ds = DeepHitDataset(test_eps, patient_arrays, means, stds)

        device = next(model.parameters()).device
        preds = predict_all(model, test_ds, device)
        recomputed_ctd = compute_ctd(preds)

        delta = abs(recomputed_ctd - saved_ctd)
        ok = delta <= TOLERANCE
        status = "OK" if ok else "FAIL"
        print(
            f"  Fold {fi}: saved={saved_ctd:.4f}  recomputed={recomputed_ctd:.4f}  "
            f"delta={delta:.4f}  [{status}]"
        )
        if not ok:
            all_ok = False

    return all_ok


def validate_graph_dt_checkpoints(features_df, episodes, patient_arrays):
    """Validate all Graph-DT fold checkpoints."""
    ckpt_dir = CHECKPOINT_DIR / "graph_dt"
    print("\n" + "=" * 60)
    print("VALIDATING GRAPH-DT CHECKPOINTS")
    print("=" * 60)

    all_ok = True
    for fi in range(5):
        path = ckpt_dir / f"fold{fi}_graph_dt.pt"
        if not path.exists():
            print(f"  MISSING: {path}")
            all_ok = False
            continue

        model, cp = load_graph_dt_checkpoint(path)
        test_pats = set(cp["test_pats"])
        means = cp["means"]
        stds = cp["stds"]
        pat_to_gidx = cp["pat_to_gidx"]
        edge_index = cp["edge_index"]
        edge_weight = cp["edge_weight"]
        node_baseline = cp["node_baseline"]
        saved_ctd = cp["fold_ctd"]

        test_eps = [e for e in episodes if e.patno in test_pats]
        test_ds = GraphDeepHitDataset(
            test_eps, patient_arrays, means, stds, pat_to_gidx
        )

        device = next(model.parameters()).device
        preds = predict_all_graph(
            model,
            test_ds,
            device,
            node_baseline,
            edge_index,
            edge_weight,
        )
        recomputed_ctd = compute_ctd(preds)

        delta = abs(recomputed_ctd - saved_ctd)
        ok = delta <= TOLERANCE
        status = "OK" if ok else "FAIL"
        print(
            f"  Fold {fi}: saved={saved_ctd:.4f}  recomputed={recomputed_ctd:.4f}  "
            f"delta={delta:.4f}  [{status}]"
        )
        if not ok:
            all_ok = False

    return all_ok


def main():
    print("Loading data...")
    features_df = pd.read_csv(FEATURES_PATH, low_memory=False)
    patient_arrays, _ = build_patient_arrays(features_df)
    episodes = extract_episodes(features_df, verbose=False)
    print(f"  {len(episodes)} episodes from {features_df['PATNO'].nunique()} patients")

    dh_ok = validate_deephit_checkpoints(features_df, episodes, patient_arrays)
    gdt_ok = validate_graph_dt_checkpoints(features_df, episodes, patient_arrays)

    print("\n" + "=" * 60)
    if dh_ok and gdt_ok:
        print("ALL CHECKPOINTS VALIDATED SUCCESSFULLY")
    else:
        print("SOME CHECKPOINTS FAILED VALIDATION")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
