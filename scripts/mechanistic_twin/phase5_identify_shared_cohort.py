#!/usr/bin/env python3
"""Phase 5 Task 2: Identify shared cohort for head-to-head benchmark.

The shared cohort is the intersection of:
  1. GIMAN Graph-DT/DeepHit patient lists (train + val + test across 5 folds)
  2. Phase 2 posterior store (1,065 patients in HDF5 from Task 1)
  3. Canonical parquet patients with ≥2 paired ON-OFF visits

This is the analysis set for:
  - Task 4: Head-to-head on common endpoint (time-to-NP4OFF≥1)
  - Task 5: Bidirectional update demo (subset with ≥3 DaT scans)
  - Task 6: Observational counterfactual (subset with LEDD escalations)

Run:
    .venv/bin/python scripts/mechanistic_twin/phase5_identify_shared_cohort.py

Output:
    outputs/mechanistic_twin/paper10_mech_vs_giman/shared_cohort.json

Author: Blair Dupre
Date: 2026-04-13
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.mechanistic_twin_v2.posterior_store import PosteriorStore
from scripts.mechanistic_twin._reproducibility import capture_provenance

CHECKPOINTS = PROJECT_ROOT / "outputs/paper3_checkpoints"
CANONICAL = (
    PROJECT_ROOT
    / "outputs/mechanistic_twin/paper10_mech_vs_giman/canonical_assembled_v2.parquet"
)
POSTERIORS_H5 = (
    PROJECT_ROOT
    / "outputs/mechanistic_twin/paper10_mech_vs_giman/phase2_posteriors_full_samples.h5"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs/mechanistic_twin/paper10_mech_vs_giman"


def load_giman_patnos() -> tuple[set[int], dict[int, dict[str, int]]]:
    """Load patient lists from all Graph-DT + DeepHit fold checkpoints.

    Returns:
        (giman_all, per_fold_counts) where giman_all is the union of all
        train + val + test patients across all folds and both models.
    """
    all_patnos: set[int] = set()
    per_fold: dict[int, dict[str, int]] = {}

    for fold in range(5):
        fold_pats: set[int] = set()
        fold_stats = {}
        for model in ["graph_dt", "deephit"]:
            cp_path = CHECKPOINTS / model / f"fold{fold}_{model}.pt"
            if not cp_path.exists():
                print(f"  WARN: missing {cp_path}")
                continue
            cp = torch.load(cp_path, map_location="cpu", weights_only=False)
            train = {int(p) for p in cp.get("train_pats", [])}
            val = {int(p) for p in cp.get("val_pats", [])}
            test = {int(p) for p in cp.get("test_pats", [])}
            fold_pats.update(train)
            fold_pats.update(val)
            fold_pats.update(test)
            fold_stats[f"{model}_train"] = len(train)
            fold_stats[f"{model}_val"] = len(val)
            fold_stats[f"{model}_test"] = len(test)
        per_fold[fold] = fold_stats | {"fold_total": len(fold_pats)}
        all_patnos.update(fold_pats)

    return all_patnos, per_fold


def load_mechanistic_patnos() -> set[int]:
    """Load PATNOs from HDF5 posterior store."""
    store = PosteriorStore(POSTERIORS_H5)
    return set(store.list_patients())


def load_multi_pair_patnos(min_pairs: int = 2) -> tuple[set[int], dict]:
    """Load canonical parquet, filter to patients with >=min_pairs paired visits.

    Returns:
        (patnos, stats) — patnos is the set; stats has count breakdowns.
    """
    df = pd.read_parquet(CANONICAL)
    df["PATNO"] = df["PATNO"].astype(int)  # canonical stores as string
    paired = df.dropna(subset=["updrs3_on", "updrs3_off"])
    counts = paired.groupby("PATNO").size()
    patnos = set(counts[counts >= min_pairs].index.tolist())
    stats = {
        "total_rows": len(df),
        "paired_rows": len(paired),
        "patients_with_1plus_pair": int((counts >= 1).sum()),
        "patients_with_2plus_pairs": int((counts >= 2).sum()),
        "patients_with_3plus_pairs": int((counts >= 3).sum()),
        "patients_with_5plus_pairs": int((counts >= 5).sum()),
        "min_pairs_filter": min_pairs,
    }
    return patnos, stats


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prov = capture_provenance(
        script_path=Path(__file__),
        repo_root=PROJECT_ROOT,
        input_files=[
            CHECKPOINTS / "graph_dt" / "fold0_graph_dt.pt",
            CHECKPOINTS / "deephit" / "fold0_deephit.pt",
            CANONICAL,
            POSTERIORS_H5,
        ],
    )

    print("Loading GIMAN patient lists (Graph-DT + DeepHit × 5 folds)...")
    giman_all, per_fold = load_giman_patnos()
    print(f"  GIMAN total (union across folds): {len(giman_all):,}")

    print("\nLoading mechanistic posterior patients...")
    mech_patnos = load_mechanistic_patnos()
    print(f"  HDF5 store: {len(mech_patnos):,}")

    print("\nLoading canonical paired visit patients (>=2 paired ON-OFF)...")
    multi_patnos, pair_stats = load_multi_pair_patnos(min_pairs=2)
    print(f"  >=2 paired: {len(multi_patnos):,}")
    print(f"  >=3 paired: {pair_stats['patients_with_3plus_pairs']:,}")

    # Intersections
    giman_and_mech = giman_all & mech_patnos
    giman_and_paired = giman_all & multi_patnos
    mech_and_paired = mech_patnos & multi_patnos
    shared_all_three = giman_all & mech_patnos & multi_patnos

    print(f"\n=== Intersections ===")
    print(f"GIMAN ∩ Mechanistic: {len(giman_and_mech):,}")
    print(f"GIMAN ∩ Paired≥2: {len(giman_and_paired):,}")
    print(f"Mechanistic ∩ Paired≥2: {len(mech_and_paired):,}")
    print(f"Shared (all three): {len(shared_all_three):,}")

    # Also report shared with stricter filters (for Task 5)
    _, triple_pair_stats = load_multi_pair_patnos(min_pairs=3)
    triple_patnos, _ = load_multi_pair_patnos(min_pairs=3)
    shared_triple = giman_all & mech_patnos & triple_patnos

    summary = {
        "giman_total": len(giman_all),
        "mechanistic_total": len(mech_patnos),
        "multi_pair_total": len(multi_patnos),
        "shared_giman_mech": len(giman_and_mech),
        "shared_giman_paired": len(giman_and_paired),
        "shared_mech_paired": len(mech_and_paired),
        "shared_all_three": len(shared_all_three),
        "shared_all_three_triple_pairs": len(shared_triple),
        "shared_patnos": sorted(shared_all_three),
        "shared_triple_patnos": sorted(shared_triple),
        "per_fold_counts": per_fold,
        "paired_visit_stats": pair_stats,
        "_provenance": prov,
    }
    summary_path = OUTPUT_DIR / "shared_cohort.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {summary_path}")

    # Scale check: Phase 4 Path B had 772 patients in analysis
    # Our shared = GIMAN × mech × paired≥2 should be similar order
    print(f"\n=== Task 4 analysis set candidates ===")
    print(f"Phase 4 Path B reference: 772 patients")
    print(f"Shared cohort (this task): {len(shared_all_three):,}")
    print(f"Shared with ≥3 paired (Task 5 candidate): {len(shared_triple):,}")


if __name__ == "__main__":
    main()
