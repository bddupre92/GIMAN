#!/usr/bin/env python3
"""Pre-registered holdout split (seed=2026) for Paper 3 / Paper 4 submission rigor.

Motivation
----------
The 5-fold stratified CV used throughout Paper 3 and Paper 4 is defensible
generalization estimation, but reviewers following a Kovatchev-style clinical-
trial discipline will want a single, pre-registered holdout that was never
touched during model development. This script carves an 80/20 patient-level
stratified split of the 1,900-patient Paper 3 cohort using a DIFFERENT random
seed (2026) from the one the CV uses (42), so the holdout is guaranteed to be
distinct from any CV fold in terms of partition assignment.

Stratification strata
---------------------
For each patient we build the same (current_stage_idx, any_transition) tuple
used by the Paper 3 cross_validate routine:

    stratum_key = f"{first_stage_idx}_{int(any_transition_observed)}"

With 7 stages × 2 transition flags = up to 14 strata, stratified 80/20 split.

Outputs
-------
data/06_longitudinal_staging/holdout_v1_patnos.json :

    {
        "seed": 2026,
        "version": "v1",
        "dev": [<PATNO>, ...],          # ~1,520 patients
        "holdout": [<PATNO>, ...],      # ~380 patients
        "stratum_summary": {"0_0": {"dev": N, "holdout": N}, ...},
        "total_patients": 1900,
        "generated_from": "data/07_paper3_features/longitudinal_features.csv",
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.paper3.dynamic_deephit import extract_episodes  # noqa: E402

FEATURES_PATH = (
    PROJECT_ROOT / "data" / "07_paper3_features" / "longitudinal_features.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "06_longitudinal_staging" / "holdout_v1_patnos.json"
)

HOLDOUT_SEED = 2026
HOLDOUT_FRACTION = 0.20


def build_patient_strata(features_df: pd.DataFrame) -> tuple[list[int], list[str]]:
    """Derive (patno, stratum_key) lists using the same logic as cross_validate."""
    episodes = extract_episodes(features_df, verbose=False)

    # Replicate cross_validate pat_info logic: first-episode stage + any uncensored transition
    pat_info: dict[int, tuple[int, bool]] = {}
    for ep in episodes:
        if ep.patno not in pat_info:
            pat_info[ep.patno] = (ep.current_stage_idx, False)
        if not ep.censored:
            s, _ = pat_info[ep.patno]
            pat_info[ep.patno] = (s, True)

    patnos = sorted(pat_info.keys())
    strat = [f"{pat_info[p][0]}_{int(pat_info[p][1])}" for p in patnos]
    return patnos, strat


def split_dev_holdout(
    patnos: list[int],
    strat: list[str],
    seed: int = HOLDOUT_SEED,
    holdout_fraction: float = HOLDOUT_FRACTION,
) -> tuple[list[int], list[int]]:
    """Stratified train/holdout split. Rare strata fall back to unstratified."""
    strat_arr = np.array(strat)
    counts = pd.Series(strat_arr).value_counts()

    # StratifiedShuffleSplit requires each stratum to have at least 2 samples.
    # Collapse the "_transitionObserved" flag for rare strata — fall back to
    # stratification by stage only. If even that is too rare, merge into a
    # single "_rare" bucket that has >=2 members.
    MIN_STRATUM = 2
    rare = set(counts.index[counts < MIN_STRATUM].tolist())
    if rare:
        collapsed = np.array(
            [s.split("_")[0] if s in rare else s for s in strat]
        )
        stage_counts = pd.Series(collapsed).value_counts()
        still_rare = set(stage_counts.index[stage_counts < MIN_STRATUM].tolist())
        if still_rare:
            strat_arr = np.array(
                [
                    ("__rare" if c in still_rare else c)
                    for c in collapsed
                ]
            )
            # Final check: if "__rare" is still a singleton, merge into first
            # non-rare stratum (maintains exchangeability — we only move ≤1 pt).
            rare_count = int((strat_arr == "__rare").sum())
            if rare_count < MIN_STRATUM and rare_count > 0:
                majority_key = pd.Series(strat_arr[strat_arr != "__rare"]).mode()[0]
                strat_arr = np.where(
                    strat_arr == "__rare", majority_key, strat_arr
                )
        else:
            strat_arr = collapsed

    dev_pats, holdout_pats = train_test_split(
        patnos,
        test_size=holdout_fraction,
        random_state=seed,
        stratify=strat_arr,
    )
    return sorted(dev_pats), sorted(holdout_pats)


def summarize_strata(
    patnos: list[int],
    strat: list[str],
    dev_pats: list[int],
    holdout_pats: list[int],
) -> dict[str, dict[str, int]]:
    """Count per-stratum dev/holdout sizes for transparency."""
    pat_to_strat = dict(zip(patnos, strat))
    dev_set = set(dev_pats)
    holdout_set = set(holdout_pats)

    summary: dict[str, dict[str, int]] = {}
    for p, s in pat_to_strat.items():
        if s not in summary:
            summary[s] = {"dev": 0, "holdout": 0}
        if p in dev_set:
            summary[s]["dev"] += 1
        elif p in holdout_set:
            summary[s]["holdout"] += 1
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate the pre-registered Paper 3/4 holdout split."
    )
    parser.add_argument("--seed", type=int, default=HOLDOUT_SEED)
    parser.add_argument(
        "--holdout-fraction", type=float, default=HOLDOUT_FRACTION,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output JSON path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary but do not write JSON",
    )
    args = parser.parse_args()

    print("Loading features...")
    features_df = pd.read_csv(FEATURES_PATH, low_memory=False)
    n_visits = len(features_df)
    n_pats = features_df["PATNO"].nunique()
    print(f"  {n_visits} visits from {n_pats} patients")

    print("Building stratification keys...")
    patnos, strat = build_patient_strata(features_df)
    assert len(patnos) == n_pats, (
        f"Stratification covers {len(patnos)} patients but CSV has {n_pats}"
    )

    print(f"Splitting 80/20 with seed={args.seed}...")
    dev_pats, holdout_pats = split_dev_holdout(
        patnos,
        strat,
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
    )
    print(f"  Dev:     {len(dev_pats)} patients")
    print(f"  Holdout: {len(holdout_pats)} patients")

    stratum_summary = summarize_strata(patnos, strat, dev_pats, holdout_pats)
    print("\nPer-stratum breakdown (stage_transitionObserved -> {dev, holdout}):")
    for s in sorted(stratum_summary.keys()):
        v = stratum_summary[s]
        print(f"  {s}: dev={v['dev']:4d}  holdout={v['holdout']:4d}")

    if args.dry_run:
        print("\n[dry-run] Not writing output.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "version": "v1",
        "holdout_fraction": args.holdout_fraction,
        "dev": dev_pats,
        "holdout": holdout_pats,
        "stratum_summary": stratum_summary,
        "total_patients": n_pats,
        "dev_count": len(dev_pats),
        "holdout_count": len(holdout_pats),
        "generated_from": str(FEATURES_PATH.relative_to(PROJECT_ROOT)),
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, default=int)
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
