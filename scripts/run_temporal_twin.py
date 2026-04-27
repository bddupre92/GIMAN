"""Temporal Digital Twin Pipeline Runner.

Demonstrates the full digital twin lifecycle:
1. Load TemporalGIMAN ensemble from trained checkpoints
2. Generate baseline trajectories for all (or subset of) patients
3. Simulate incremental updates (remove last visit, then re-add)
4. Run counterfactual sensitivity analysis on top-risk patients
5. Save all states, visualizations, and summary

Usage:
    python scripts/run_temporal_twin.py
    python scripts/run_temporal_twin.py --max_patients 10
    python scripts/run_temporal_twin.py --config config/digital_twin/temporal_twin.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.giman_pipeline.data_processing.longitudinal_assembler import (  # noqa: E402
    PatientSequence,
)
from src.giman_pipeline.digital_twin.state import CounterfactualSpec  # noqa: E402
from src.giman_pipeline.digital_twin.temporal_twin_engine import (  # noqa: E402
    TemporalTwinEngine,
    truncate_to_n_visits,
)
from src.giman_pipeline.digital_twin.twin_registry import TwinRegistry  # noqa: E402
from src.giman_pipeline.digital_twin.twin_visualizer import TwinVisualizer  # noqa: E402
from src.giman_pipeline.training.temporal_data_loaders import (  # noqa: E402
    load_longitudinal_dataset,
)


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration."""
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run temporal digital twin pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(root / "config" / "digital_twin" / "temporal_twin.yaml"),
    )
    parser.add_argument(
        "--max_patients",
        type=int,
        default=0,
        help="Max patients for baseline generation (0=all)",
    )
    parser.add_argument(
        "--n_update_demos",
        type=int,
        default=0,
        help="Override number of update demo patients (0=use config)",
    )
    parser.add_argument(
        "--n_counterfactual",
        type=int,
        default=0,
        help="Override number of counterfactual patients (0=use config)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    start_time = time.time()
    config = load_config(args.config)

    print("=" * 60)
    print("TEMPORAL DIGITAL TWIN PIPELINE")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Load engine ──
    engine_cfg = config["engine"]
    engine = TemporalTwinEngine(
        checkpoint_dir=root / engine_cfg["checkpoint_dir"],
        feature_config_path=root / engine_cfg["feature_config"],
        training_config_path=root / engine_cfg["training_config"],
        checkpoint_pattern=engine_cfg.get(
            "checkpoint_pattern", "model_fold*_*.pt"
        ),
        device=engine_cfg.get("device", "cpu"),
    )

    # ── Load dataset ──
    data_cfg = config.get("data", {})
    dataset = load_longitudinal_dataset(
        data_dir=root / data_cfg.get(
            "data_dir", "data/03_prodromal/longitudinal_training"
        ),
        endpoints_path=root / data_cfg.get(
            "endpoints", "data/prodromal_cohort/prodromal_survival_data.csv"
        ),
        min_visits=data_cfg.get("min_visits", 2),
    )

    # ── Registry ──
    registry_cfg = config.get("registry", {})
    registry_dir = root / registry_cfg.get(
        "output_dir", "results/digital_twin_states"
    )
    registry = TwinRegistry(registry_dir)

    # ── Visualization output ──
    vis_cfg = config.get("visualization", {})
    vis_dir = root / vis_cfg.get("output_dir", "visualizations/digital_twin")
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ── Demo settings ──
    demo_cfg = config.get("demo", {})
    max_patients = args.max_patients or demo_cfg.get("max_patients", 0)
    n_update_demos = args.n_update_demos or demo_cfg.get("n_update_demos", 10)
    n_counterfactual = args.n_counterfactual or demo_cfg.get(
        "n_counterfactual", 10
    )

    # ── 1. Generate baseline twins ──
    print(f"\n{'─' * 40}")
    print("PHASE 1: Baseline Trajectory Generation")
    print(f"{'─' * 40}")

    patnos = list(dataset.patnos)
    if max_patients > 0:
        patnos = patnos[:max_patients]

    print(f"Generating baselines for {len(patnos)} patients...")
    baseline_start = time.time()

    for i, patno in enumerate(patnos):
        seq = dataset.sequences[patno]
        state = engine.generate_baseline(seq)
        registry.save_state(state)

        if (i + 1) % 100 == 0 or (i + 1) == len(patnos):
            elapsed = time.time() - baseline_start
            rate = (i + 1) / elapsed
            print(
                f"  [{i + 1}/{len(patnos)}] "
                f"{rate:.1f} patients/sec, "
                f"elapsed {elapsed:.0f}s"
            )

    # Load all generated states
    all_states = registry.load_latest_states()
    print(f"\nBaseline twins generated: {len(all_states)}")

    # Cohort summary
    summary = engine.generate_cohort_summary(all_states)
    print(f"  Risk score: {summary['risk_score_mean']:.4f} "
          f"+/- {summary['risk_score_std']:.4f}")
    print(f"  Range: [{summary['risk_score_min']:.4f}, "
          f"{summary['risk_score_max']:.4f}]")

    # Plot cohort distribution
    TwinVisualizer.plot_cohort_risk_distribution(
        list(all_states.values()),
        vis_dir / "cohort_risk_distribution.png",
    )

    # Plot a few example trajectories
    example_patnos = patnos[:min(3, len(patnos))]
    for patno in example_patnos:
        if patno in all_states:
            TwinVisualizer.plot_patient_trajectory(
                all_states[patno],
                vis_dir / f"patient_{patno}_trajectory.png",
            )

    # ── 2. Incremental update demonstration ──
    print(f"\n{'─' * 40}")
    print("PHASE 2: Incremental Update Demonstration")
    print(f"{'─' * 40}")

    # Select patients with >=3 visits for meaningful demo
    update_candidates = [
        p for p in patnos
        if dataset.sequences[p].n_visits >= 3
    ][:n_update_demos]

    print(f"Demonstrating updates for {len(update_candidates)} patients...")
    update_deltas = []

    for patno in update_candidates:
        seq = dataset.sequences[patno]

        # Generate baseline from all visits except the last
        partial_seq = truncate_to_n_visits(seq, seq.n_visits - 1)
        partial_state = engine.generate_baseline(partial_seq)

        # Update with last visit
        updated_state = engine.update_twin(
            partial_state,
            new_features=seq.features[-1],
            new_obs_mask=seq.obs_mask[-1],
            new_time_month=float(seq.time_months[-1]),
            new_visit_id=seq.visit_ids[-1],
            original_sequence=partial_seq,
        )

        delta = updated_state.risk_score - partial_state.risk_score
        update_deltas.append(delta)

        print(
            f"  Patient {patno}: "
            f"{partial_state.n_visits}→{updated_state.n_visits} visits, "
            f"risk {partial_state.risk_score:.4f}→{updated_state.risk_score:.4f} "
            f"(delta={delta:+.4f})"
        )

        # Visualize update history
        TwinVisualizer.plot_update_history(
            [partial_state, updated_state],
            vis_dir / f"patient_{patno}_update.png",
        )

    if update_deltas:
        print(f"\n  Update deltas: mean={np.mean(update_deltas):+.4f}, "
              f"std={np.std(update_deltas):.4f}")

    # ── 3. Counterfactual analysis ──
    print(f"\n{'─' * 40}")
    print("PHASE 3: Counterfactual Sensitivity Analysis")
    print(f"{'─' * 40}")

    # Build specs from config
    cf_cfg = config.get("counterfactual", {})
    specs = []
    for spec_dict in cf_cfg.get("default_specs", []):
        specs.append(
            CounterfactualSpec(
                feature_name=spec_dict["feature_name"],
                delta=float(spec_dict["delta"]),
            )
        )
    if not specs:
        specs = [
            CounterfactualSpec(feature_name="UPDRS_I", delta=-0.5),
            CounterfactualSpec(feature_name="UPSIT_SCORE", delta=0.5),
        ]

    print(f"Interventions: {len(specs)}")
    for s in specs:
        print(f"  {s.feature_name}: {s.delta:+.3f}")

    # Select top-risk patients
    sorted_states = sorted(
        all_states.values(), key=lambda s: s.risk_score, reverse=True
    )
    cf_patients = sorted_states[:n_counterfactual]

    print(f"\nRunning counterfactuals on {len(cf_patients)} top-risk patients...")
    all_deltas: dict[str, list[float]] = {}

    for state in cf_patients:
        seq = dataset.sequences[state.patno]
        cf_result = engine.simulate_counterfactual(seq, specs)

        for key, delta in cf_result.delta_risk.items():
            all_deltas.setdefault(key, []).append(delta)

        TwinVisualizer.plot_counterfactual_comparison(
            cf_result,
            vis_dir / f"patient_{state.patno}_counterfactual.png",
        )

    # Print sensitivity summary
    print("\nCounterfactual Sensitivity Summary:")
    print(f"  {'Intervention':<35} {'Mean Delta':>12} {'Std':>10}")
    print("  " + "-" * 57)
    for key, deltas in sorted(
        all_deltas.items(), key=lambda x: abs(np.mean(x[1])), reverse=True
    ):
        print(
            f"  {key:<35} {np.mean(deltas):>+12.4f} {np.std(deltas):>10.4f}"
        )

    # ── 4. Validation checks ──
    print(f"\n{'─' * 40}")
    print("PHASE 4: Validation Checks")
    print(f"{'─' * 40}")

    # Zero-delta invariance
    test_patno = patnos[0]
    test_seq = dataset.sequences[test_patno]
    zero_spec = CounterfactualSpec(
        feature_name=engine.feature_names[0], delta=0.0
    )
    zero_cf = engine.simulate_counterfactual(test_seq, [zero_spec])
    zero_key = list(zero_cf.delta_risk.keys())[0]
    zero_delta = abs(zero_cf.delta_risk[zero_key])
    zero_pass = zero_delta < 1e-6
    print(f"  Zero-delta invariance: {'PASS' if zero_pass else 'FAIL'} "
          f"(delta={zero_delta:.2e})")

    # Monotonicity check on ensemble-averaged trajectories
    monotone_count = 0
    total_checked = 0
    for state in all_states.values():
        traj = np.array(state.risk_trajectory)
        if len(traj) >= 2:
            diffs = np.diff(traj)
            if np.all(diffs >= -1e-6):
                monotone_count += 1
            total_checked += 1
    monotonicity_rate = monotone_count / total_checked if total_checked > 0 else 0
    print(f"  Monotonicity rate: {monotonicity_rate:.1%} "
          f"({monotone_count}/{total_checked})")

    # State persistence round-trip
    test_state = all_states[patnos[0]]
    registry.save_state(test_state)
    loaded = registry.load_state(patnos[0])
    persist_pass = (
        loaded is not None
        and loaded.patno == test_state.patno
        and abs(loaded.risk_score - test_state.risk_score) < 1e-9
    )
    print(f"  State persistence round-trip: "
          f"{'PASS' if persist_pass else 'FAIL'}")

    # ── 5. Save summary ──
    output_dir = root / "results" / "digital_twin"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_summary = {
        "timestamp": timestamp,
        "n_patients": len(all_states),
        "ensemble_size": engine.n_models,
        "model_version": engine.model_version,
        "cohort_stats": summary,
        "counterfactual_sensitivity": {
            key: {
                "mean_delta": float(np.mean(deltas)),
                "std_delta": float(np.std(deltas)),
                "n_patients": len(deltas),
            }
            for key, deltas in all_deltas.items()
        },
        "validation": {
            "zero_delta_invariance": bool(zero_pass),
            "monotonicity_rate": float(monotonicity_rate),
            "state_persistence": bool(persist_pass),
        },
        "update_demo": {
            "n_patients": len(update_deltas),
            "mean_delta": float(np.mean(update_deltas))
            if update_deltas
            else 0.0,
            "std_delta": float(np.std(update_deltas))
            if update_deltas
            else 0.0,
        },
        "elapsed_seconds": time.time() - start_time,
    }

    summary_path = output_dir / f"twin_summary_{timestamp}.json"
    summary_path.write_text(
        json.dumps(final_summary, indent=2), encoding="utf-8"
    )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("DIGITAL TWIN PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Patients: {len(all_states)}")
    print(f"  Ensemble: {engine.n_models} models")
    print(f"  Registry: {registry_dir}")
    print(f"  Visualizations: {vis_dir}")
    print(f"  Summary: {summary_path}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
