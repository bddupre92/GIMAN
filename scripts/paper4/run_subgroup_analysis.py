#!/usr/bin/env python3
"""Run subgroup equity analysis across all checkpoints.

Evaluates per-subgroup C-td, bootstrap interaction tests, and
conditional conformal coverage for LRRK2, GBA, sex, and age subgroups.

Outputs:
    outputs/paper4/subgroup/
        subgroup_ctd.json
        interaction_tests.json
        conditional_coverage.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.paper3.dynamic_deephit import (
    DeepHitDataset,
    build_patient_arrays,
    extract_episodes,
    load_deephit_checkpoint,
    predict_all,
)
from giman_pipeline.paper3.graph_digital_twin import (
    GraphDeepHitDataset,
    load_graph_dt_checkpoint,
    predict_all_graph,
)
from giman_pipeline.paper4.conformal_survival import CauseSpecificConformal
from giman_pipeline.paper4.subgroup import (
    SUBGROUP_VARS,
    apply_fdr_correction,
    assign_subgroups,
    bootstrap_interaction_test,
    compute_conditional_coverage,
    compute_subgroup_ctd,
    interaction_test_to_dict,
    subgroup_ctd_to_dict,
)

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "07_paper3_features" / "longitudinal_features.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "paper3_checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper4" / "subgroup"


def get_test_predictions(model_type, fi, episodes, patient_arrays):
    """Load checkpoint and get test predictions for a fold."""
    if model_type == "deephit":
        path = CHECKPOINT_DIR / "deephit" / f"fold{fi}_deephit.pt"
        model, cp = load_deephit_checkpoint(path)
        test_pats = set(cp["test_pats"])
        means, stds = cp["means"], cp["stds"]
        test_eps = [e for e in episodes if e.patno in test_pats]
        test_ds = DeepHitDataset(test_eps, patient_arrays, means, stds)
        device = next(model.parameters()).device
        preds = predict_all(model, test_ds, device)
    else:
        path = CHECKPOINT_DIR / "graph_dt" / f"fold{fi}_graph_dt.pt"
        model, cp = load_graph_dt_checkpoint(path)
        test_pats = set(cp["test_pats"])
        means, stds = cp["means"], cp["stds"]
        pat_to_gidx = cp["pat_to_gidx"]
        test_eps = [e for e in episodes if e.patno in test_pats]
        test_ds = GraphDeepHitDataset(
            test_eps, patient_arrays, means, stds, pat_to_gidx
        )
        device = next(model.parameters()).device
        preds = predict_all_graph(
            model,
            test_ds,
            device,
            cp["node_baseline"],
            cp["edge_index"],
            cp["edge_weight"],
        )

    patnos = [ep.patno for ep in test_eps]
    return preds, patnos, test_eps


def main():
    t0 = time.time()
    print("=" * 60)
    print("PAPER 4: SUBGROUP EQUITY ANALYSIS")
    print("=" * 60)

    print("\nLoading data...")
    features_df = pd.read_csv(FEATURES_PATH, low_memory=False)
    patient_arrays, _ = build_patient_arrays(features_df)
    episodes = extract_episodes(features_df, verbose=False)
    print(f"  {len(episodes)} episodes")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Per-subgroup C-td ---
    print("\n--- PER-SUBGROUP C-td ---")
    all_ctd_results = []

    for fi in range(5):
        for model_type, model_name in [
            ("deephit", "DeepHit"),
            ("graph_dt", "Graph-DT"),
        ]:
            preds, patnos, test_eps = get_test_predictions(
                model_type,
                fi,
                episodes,
                patient_arrays,
            )

            subgroup_assignments = assign_subgroups(patnos, features_df)

            for var_name, pat_groups in subgroup_assignments.items():
                ctd_result = compute_subgroup_ctd(
                    preds,
                    patnos,
                    pat_groups,
                    model_name,
                    var_name,
                )
                all_ctd_results.append(ctd_result)

        print(f"  Fold {fi} done")

    # Save C-td results
    with open(OUTPUT_DIR / "subgroup_ctd.json", "w") as f:
        json.dump(
            [subgroup_ctd_to_dict(r) for r in all_ctd_results],
            f,
            indent=2,
            default=str,
        )

    # Print summary
    print("\n  Per-subgroup C-td (averaged across folds):")
    for var_name in SUBGROUP_VARS:
        print(f"\n  {var_name}:")
        for model_name in ["DeepHit", "Graph-DT"]:
            fold_results = [
                r
                for r in all_ctd_results
                if r.model_name == model_name and r.subgroup_var == var_name
            ]
            if not fold_results:
                continue
            # Get all groups
            all_groups = set()
            for r in fold_results:
                all_groups.update(r.per_group_ctd.keys())
            for group in sorted(all_groups):
                vals = [r.per_group_ctd.get(group, float("nan")) for r in fold_results]
                valid = [v for v in vals if not np.isnan(v)]
                if valid:
                    print(
                        f"    {model_name} [{group}]: C-td={np.mean(valid):.4f} ± {np.std(valid):.4f}"
                    )

    # --- Bootstrap interaction tests ---
    print("\n--- BOOTSTRAP INTERACTION TESTS ---")
    interaction_results = []

    for fi in range(5):
        dh_preds, dh_patnos, _ = get_test_predictions(
            "deephit", fi, episodes, patient_arrays
        )
        gdt_preds, gdt_patnos, _ = get_test_predictions(
            "graph_dt", fi, episodes, patient_arrays
        )

        # Align predictions (same patients, same order)
        assert dh_patnos == gdt_patnos, f"Patient mismatch in fold {fi}"

        subgroup_assignments = assign_subgroups(dh_patnos, features_df)

        for var_name, pat_groups in subgroup_assignments.items():
            result = bootstrap_interaction_test(
                dh_preds,
                gdt_preds,
                dh_patnos,
                pat_groups,
                var_name,
                n_bootstrap=500,
                random_state=42 + fi,
            )
            interaction_results.append(result)

        print(f"  Fold {fi} done")

    # FDR correction
    interaction_results = apply_fdr_correction(interaction_results)

    with open(OUTPUT_DIR / "interaction_tests.json", "w") as f:
        json.dump(
            [interaction_test_to_dict(r) for r in interaction_results],
            f,
            indent=2,
            default=str,
        )

    print("\n  Interaction test results:")
    for var_name in SUBGROUP_VARS:
        fold_results = [r for r in interaction_results if r.subgroup_var == var_name]
        p_values = [
            r.interaction_p_value
            for r in fold_results
            if not np.isnan(r.interaction_p_value)
        ]
        if p_values:
            print(
                f"    {var_name}: avg p={np.mean(p_values):.4f}, "
                f"range=[{min(p_values):.4f}, {max(p_values):.4f}]"
            )

    # --- Conditional coverage ---
    print("\n--- CONDITIONAL CONFORMAL COVERAGE ---")
    cond_coverage = {}

    for fi in range(5):
        for model_type, model_name in [
            ("deephit", "DeepHit"),
            ("graph_dt", "Graph-DT"),
        ]:
            preds, patnos, test_eps = get_test_predictions(
                model_type,
                fi,
                episodes,
                patient_arrays,
            )

            cif = preds["cif"].numpy()
            durations = np.array([ep.duration_months for ep in test_eps])
            event_idxs = preds["event_idxs"].numpy()
            censored = preds["censored"].numpy()

            # Split 50/50 for conformal
            rng = np.random.RandomState(42 + fi)
            n = len(durations)
            idx = rng.permutation(n)
            n_cal = n // 2

            csc = CauseSpecificConformal(confidence_level=0.90)
            csc.calibrate(
                cif[idx[:n_cal]],
                durations[idx[:n_cal]],
                event_idxs[idx[:n_cal]],
                censored[idx[:n_cal]],
            )
            bands = csc.predict_bands(cif[idx[n_cal:]])

            eval_patnos = [patnos[i] for i in idx[n_cal:]]
            subgroup_assignments = assign_subgroups(eval_patnos, features_df)

            for var_name, pat_groups in subgroup_assignments.items():
                cc = compute_conditional_coverage(
                    cif[idx[n_cal:]],
                    bands,
                    durations[idx[n_cal:]],
                    event_idxs[idx[n_cal:]],
                    censored[idx[n_cal:]],
                    eval_patnos,
                    pat_groups,
                )
                key = f"{model_name}_{var_name}_fold{fi}"
                cond_coverage[key] = cc

    with open(OUTPUT_DIR / "conditional_coverage.json", "w") as f:
        json.dump(cond_coverage, f, indent=2, default=str)

    print(f"\n  Total time: {time.time() - t0:.1f}s")
    print(f"  Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
