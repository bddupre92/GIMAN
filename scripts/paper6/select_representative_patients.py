#!/usr/bin/env python3
"""Paper 6: Select Representative Patients for Unified Pipeline Demo.

Criteria:
    1. At least 3 longitudinal visits
    2. At least 1 observed NSD-ISS transition (ground truth)
    3. Diverse initial stages (0, 2B/3, 3/4)
    4. At least 1 backward transition for regression demo
    5. At least 1 genetic carrier (LRRK2 or GBA) if available
    6. Feature completeness >60% at baseline for key clinical features

Outputs:
    outputs/paper6/patient_selection.json — Selected PATNOs + summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

FEATURES_PATH = ROOT / "data" / "07_paper3_features" / "longitudinal_features.csv"
TRANSITIONS_PATH = ROOT / "data" / "06_longitudinal_staging" / "transition_events.csv"
PAPER1_PATH = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
OUTPUT_DIR = ROOT / "outputs" / "paper6"

# Key clinical features to check completeness
CLINICAL_FEATURES = [
    "updrs1_total", "updrs2_total", "updrs3_total", "hy_stage",
    "moca_total", "ess_total", "rbd_total", "scopa_aut_total",
]


def select_patients() -> dict:
    """Select 5 representative patients for the unified pipeline demo."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    features_df = pd.read_csv(FEATURES_PATH)
    transitions_df = pd.read_csv(TRANSITIONS_PATH)
    paper1_df = pd.read_csv(PAPER1_PATH)

    print(f"Loaded: {features_df['PATNO'].nunique()} patients, "
          f"{len(features_df)} visits, {len(transitions_df)} transitions")

    # Patients with transitions (922 unique)
    trans_pats = set(transitions_df["PATNO"].unique())

    # Per-patient stats
    pat_stats = []
    for patno, grp in features_df.groupby("PATNO"):
        if patno not in trans_pats:
            continue

        n_visits = len(grp)
        if n_visits < 3:
            continue

        baseline = grp[grp["months_from_baseline"] == 0.0]
        if baseline.empty:
            continue

        initial_stage = baseline.iloc[0]["nsd_stage"]
        stages_visited = grp["nsd_stage"].unique().tolist()

        # Feature completeness at baseline
        bl_row = baseline.iloc[0]
        n_avail = sum(1 for f in CLINICAL_FEATURES if pd.notna(bl_row.get(f)))
        completeness = n_avail / len(CLINICAL_FEATURES)

        if completeness < 0.6:
            continue

        # Transition details
        pt_trans = transitions_df[transitions_df["PATNO"] == patno]
        n_transitions = len(pt_trans)
        has_backward = (pt_trans["direction"] == "backward").any()
        has_forward = (pt_trans["direction"] == "forward").any()
        transition_types = [
            f"{r['source_stage']}→{r['dest_stage']}"
            for _, r in pt_trans.iterrows()
        ]

        # Genetics
        is_lrrk2 = bool(bl_row.get("lrrk2_carrier", 0))
        is_gba = bool(bl_row.get("gba_carrier", 0))

        pat_stats.append({
            "patno": int(patno),
            "n_visits": n_visits,
            "initial_stage": str(initial_stage),
            "stages_visited": [str(s) for s in stages_visited],
            "n_transitions": n_transitions,
            "has_backward": has_backward,
            "has_forward": has_forward,
            "transition_types": transition_types,
            "completeness": completeness,
            "is_lrrk2": is_lrrk2,
            "is_gba": is_gba,
            "follow_up_months": float(grp["months_from_baseline"].max()),
        })

    df_candidates = pd.DataFrame(pat_stats)
    print(f"\nCandidates: {len(df_candidates)} patients with ≥3 visits, "
          f"≥1 transition, ≥60% baseline completeness")

    # Selection strategy: pick 5 diverse patients
    selected = []

    # 1. Patient starting at Stage 0 with forward progression
    stage0 = df_candidates[
        (df_candidates["initial_stage"] == "0") &
        (df_candidates["has_forward"])
    ].sort_values("n_transitions", ascending=False)
    if len(stage0) > 0:
        selected.append(stage0.iloc[0]["patno"])
        print(f"  Stage 0 progressor: PATNO {selected[-1]}")

    # 2. Patient starting at Stage 2B with forward progression to 3+
    stage2b = df_candidates[
        (df_candidates["initial_stage"] == "2B") &
        (df_candidates["has_forward"]) &
        (~df_candidates["patno"].isin(selected))
    ].sort_values("n_transitions", ascending=False)
    if len(stage2b) > 0:
        selected.append(stage2b.iloc[0]["patno"])
        print(f"  Stage 2B progressor: PATNO {selected[-1]}")

    # 3. Patient with backward transition (regression)
    backward = df_candidates[
        (df_candidates["has_backward"]) &
        (~df_candidates["patno"].isin(selected))
    ].sort_values("n_transitions", ascending=False)
    if len(backward) > 0:
        selected.append(backward.iloc[0]["patno"])
        print(f"  Backward transition: PATNO {selected[-1]}")

    # 4. Patient starting at Stage 3 or 4 (advanced)
    advanced = df_candidates[
        (df_candidates["initial_stage"].isin(["3", "4"])) &
        (~df_candidates["patno"].isin(selected))
    ].sort_values("n_transitions", ascending=False)
    if len(advanced) > 0:
        selected.append(advanced.iloc[0]["patno"])
        print(f"  Advanced stage: PATNO {selected[-1]}")

    # 5. Genetic carrier if available, else longest follow-up
    carriers = df_candidates[
        ((df_candidates["is_lrrk2"]) | (df_candidates["is_gba"])) &
        (~df_candidates["patno"].isin(selected))
    ].sort_values("n_transitions", ascending=False)
    if len(carriers) > 0:
        selected.append(carriers.iloc[0]["patno"])
        print(f"  Genetic carrier: PATNO {selected[-1]}")
    else:
        remaining = df_candidates[
            ~df_candidates["patno"].isin(selected)
        ].sort_values("follow_up_months", ascending=False)
        if len(remaining) > 0:
            selected.append(remaining.iloc[0]["patno"])
            print(f"  Longest follow-up: PATNO {selected[-1]}")

    # Build output
    result = {"selected_patnos": [int(p) for p in selected], "patients": []}

    for patno in selected:
        info = df_candidates[df_candidates["patno"] == patno].iloc[0].to_dict()
        # Get Paper 1 staging info
        p1_row = paper1_df[paper1_df["PATNO"] == patno]
        if len(p1_row) > 0:
            info["paper1_stage"] = str(p1_row.iloc[0]["nsd_iss_stage"])
        else:
            info["paper1_stage"] = "N/A"
        result["patients"].append(info)

    # Summary
    result["summary"] = {
        "n_candidates": len(df_candidates),
        "n_selected": len(selected),
        "initial_stages": [
            df_candidates[df_candidates["patno"] == p].iloc[0]["initial_stage"]
            for p in selected
        ],
        "total_transitions": sum(
            df_candidates[df_candidates["patno"] == p].iloc[0]["n_transitions"]
            for p in selected
        ),
    }

    # Save
    out_path = OUTPUT_DIR / "patient_selection.json"

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_convert)

    print(f"\nSaved patient selection to {out_path}")
    print(f"\nSelected {len(selected)} patients:")
    for i, patno in enumerate(selected):
        info = df_candidates[df_candidates["patno"] == patno].iloc[0]
        print(f"  {i+1}. PATNO {patno}: Stage {info['initial_stage']}, "
              f"{info['n_visits']} visits, {info['n_transitions']} transitions, "
              f"{info['follow_up_months']:.0f}mo follow-up")
        print(f"     Transitions: {', '.join(info['transition_types'])}")

    return result


if __name__ == "__main__":
    select_patients()
