from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from giman_pipeline.digital_twin.simulator import DataDrivenTwinSimulator  # noqa: E402
from giman_pipeline.digital_twin.state import CounterfactualSpec  # noqa: E402

DEFAULT_HORIZONS = [0, 6, 12, 18, 24]


def _sha256_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _build_patno_index(sim: DataDrivenTwinSimulator) -> dict[int, int]:
    pat = sim.data.patno.detach().cpu().numpy().astype(int)
    return {int(v): int(i) for i, v in enumerate(pat)}


def _trajectory_points(
    sim: DataDrivenTwinSimulator, patient_idx: int
) -> list[dict[str, float | int]]:
    states = sim.simulate_patient(patient_idx=patient_idx, horizons=DEFAULT_HORIZONS)
    return [
        {
            "patno": int(s.patno),
            "month": int(s.t_month),
            "risk_saa": float(s.risk_saa),
            "risk_survival": float(s.risk_survival),
            "unc_low": float(s.uncertainty_low),
            "unc_high": float(s.uncertainty_high),
        }
        for s in states
    ]


def _compute_patient_deltas(
    baseline_sim: DataDrivenTwinSimulator,
    updated_sim: DataDrivenTwinSimulator,
    common_patnos: list[int],
) -> pd.DataFrame:
    base_idx = _build_patno_index(baseline_sim)
    upd_idx = _build_patno_index(updated_sim)

    rows: list[dict[str, float | int]] = []
    for pat in common_patnos:
        btraj = baseline_sim.simulate_patient(base_idx[pat], horizons=DEFAULT_HORIZONS)
        utraj = updated_sim.simulate_patient(upd_idx[pat], horizons=DEFAULT_HORIZONS)
        b_final = float(btraj[-1].risk_saa)
        u_final = float(utraj[-1].risk_saa)
        delta = u_final - b_final
        rows.append(
            {
                "PATNO": int(pat),
                "baseline_final_risk": b_final,
                "updated_final_risk": u_final,
                "delta_risk": delta,
                "abs_delta_risk": abs(delta),
            }
        )

    return pd.DataFrame(rows).sort_values("abs_delta_risk", ascending=False)


def _compute_sensitivity(
    sim: DataDrivenTwinSimulator,
    patient_indices: list[int],
    specs: list[CounterfactualSpec],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for pidx in patient_indices:
        result = sim.simulate_counterfactual(
            patient_idx=pidx,
            specs=specs,
            horizons=DEFAULT_HORIZONS,
        )
        for intervention, delta in result.delta_risk.items():
            rows.append(
                {
                    "patient_idx": int(pidx),
                    "intervention": intervention,
                    "delta_risk": float(delta),
                    "abs_delta_risk": float(abs(delta)),
                }
            )
    return pd.DataFrame(rows)


def _plot_patient_before_after(
    baseline_sim: DataDrivenTwinSimulator,
    updated_sim: DataDrivenTwinSimulator,
    patno: int,
    output_path: Path,
) -> None:
    b_idx = _build_patno_index(baseline_sim)[patno]
    u_idx = _build_patno_index(updated_sim)[patno]
    b = baseline_sim.simulate_patient(b_idx, horizons=DEFAULT_HORIZONS)
    u = updated_sim.simulate_patient(u_idx, horizons=DEFAULT_HORIZONS)

    months = [s.t_month for s in b]
    b_risk = [s.risk_saa for s in b]
    u_risk = [s.risk_saa for s in u]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(months, b_risk, "o-", label="baseline", color="#4E79A7")
    ax.plot(months, u_risk, "o-", label="updated", color="#E15759")
    ax.set_ylim(0, 1)
    ax.set_title(f"Patient {patno}: Digital Twin Before/After Update")
    ax.set_xlabel("month")
    ax.set_ylabel("predicted SAA risk")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_reference_before_after(
    baseline_sim: DataDrivenTwinSimulator,
    updated_sim: DataDrivenTwinSimulator,
    output_path: Path,
) -> None:
    base_map = _build_patno_index(baseline_sim)
    upd_map = _build_patno_index(updated_sim)
    base_pat = sorted(base_map.keys())[0]
    upd_pat = sorted(upd_map.keys())[0]

    b = baseline_sim.simulate_patient(base_map[base_pat], horizons=DEFAULT_HORIZONS)
    u = updated_sim.simulate_patient(upd_map[upd_pat], horizons=DEFAULT_HORIZONS)

    months = [s.t_month for s in b]
    b_risk = [s.risk_saa for s in b]
    u_risk = [s.risk_saa for s in u]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(
        months, b_risk, "o-", label=f"baseline_ref PATNO={base_pat}", color="#4E79A7"
    )
    ax.plot(months, u_risk, "o-", label=f"updated_ref PATNO={upd_pat}", color="#E15759")
    ax.set_ylim(0, 1)
    ax.set_title("Reference Trajectories (No Common PATNO Across Pulls)")
    ax.set_xlabel("month")
    ax.set_ylabel("predicted SAA risk")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_cohort_delta_distribution(delta_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    if delta_df.empty:
        ax.axis("off")
        ax.text(
            0.05,
            0.55,
            "No common PATNO across baseline and updated pull.\n"
            "Patient-level delta distribution is unavailable.",
            fontsize=11,
            va="center",
        )
    else:
        ax.hist(delta_df["delta_risk"], bins=20, color="#59A14F", alpha=0.9)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title("Cohort Delta Distribution (Updated - Baseline)")
        ax.set_xlabel("delta risk")
        ax.set_ylabel("count")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_sensitivity_shift(summary_df: pd.DataFrame, output_path: Path) -> None:
    df = summary_df.sort_values("delta_mean_abs_delta", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.barh(df["intervention"][::-1], df["delta_mean_abs_delta"][::-1], color="#F28E2B")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Intervention Sensitivity Shift by Pull")
    ax.set_xlabel("updated - baseline mean |delta risk|")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse CLI args for one digital-twin refresh cycle."""
    parser = argparse.ArgumentParser(
        description="Run digital twin update cycle on external pull"
    )
    parser.add_argument("--pull-id", type=str, required=True)
    parser.add_argument(
        "--baseline-data",
        type=Path,
        default=root
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_run"
        / "test_data.pt",
    )
    parser.add_argument(
        "--baseline-metadata",
        type=Path,
        default=root
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_run"
        / "pyg_data_metadata.json",
    )
    parser.add_argument(
        "--imaging-feature-csv",
        type=Path,
        default=None,
        help="Optional PATNO-level imaging delta feature CSV to inject into twin state outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute deterministic digital-twin refresh and write cycle artifacts."""
    args = parse_args()
    pull_id = args.pull_id

    external_data = (
        root / "data" / "04_external_validation" / pull_id / "external_data.pt"
    )
    external_meta = (
        root / "data" / "04_external_validation" / pull_id / "external_metadata.json"
    )
    if not external_data.exists() or not external_meta.exists():
        raise FileNotFoundError(
            f"Missing external artifacts for pull '{pull_id}'. Expected {external_data} and {external_meta}"
        )

    baseline_sim = DataDrivenTwinSimulator(
        data_path=args.baseline_data,
        metadata_path=args.baseline_metadata,
        temperature=2.5,
    )
    updated_sim = DataDrivenTwinSimulator(
        data_path=external_data,
        metadata_path=external_meta,
        temperature=2.5,
    )

    baseline_pat = set(_build_patno_index(baseline_sim).keys())
    updated_pat = set(_build_patno_index(updated_sim).keys())
    common_patnos = sorted(baseline_pat & updated_pat)
    has_common_patnos = len(common_patnos) > 0

    out_dir = root / "outputs" / "digital_twin_updates" / pull_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if has_common_patnos:
        delta_df = _compute_patient_deltas(baseline_sim, updated_sim, common_patnos)
        changed_df = delta_df[delta_df["abs_delta_risk"] > 1e-6].copy()
    else:
        delta_df = pd.DataFrame(
            columns=[
                "PATNO",
                "baseline_final_risk",
                "updated_final_risk",
                "delta_risk",
                "abs_delta_risk",
            ]
        )
        changed_df = delta_df.copy()

    all_delta_path = out_dir / "all_patient_deltas.csv"
    changed_path = out_dir / "changed_patients.csv"
    delta_df.to_csv(all_delta_path, index=False)
    changed_df.to_csv(changed_path, index=False)

    # Optional PATNO-level imaging feature injection for twin state sidecar
    twin_state_with_imaging = None
    imaging_join_coverage = None
    if args.imaging_feature_csv is not None:
        if args.imaging_feature_csv.exists():
            imaging_df = pd.read_csv(args.imaging_feature_csv)
            if "PATNO" in imaging_df.columns:
                imaging_df["PATNO"] = pd.to_numeric(
                    imaging_df["PATNO"], errors="coerce"
                ).astype("Int64")
                state_df = delta_df.copy()
                if not state_df.empty:
                    state_df["PATNO"] = pd.to_numeric(
                        state_df["PATNO"], errors="coerce"
                    ).astype("Int64")
                twin_state_with_imaging = state_df.merge(
                    imaging_df, on="PATNO", how="left"
                )
                twin_state_path = out_dir / "twin_state_with_imaging.csv"
                twin_state_with_imaging.to_csv(twin_state_path, index=False)
                matched = int(
                    twin_state_with_imaging.filter(items=["PATNO"]).dropna().shape[0]
                )
                imaging_cols = [c for c in imaging_df.columns if c != "PATNO"]
                non_null_any = int(
                    twin_state_with_imaging.filter(items=imaging_cols)
                    .notna()
                    .any(axis=1)
                    .sum()
                )
                imaging_join_coverage = {
                    "n_rows": int(len(twin_state_with_imaging)),
                    "n_patno_rows": matched,
                    "n_rows_with_any_imaging_feature": non_null_any,
                    "imaging_feature_columns": int(max(len(imaging_df.columns) - 1, 0)),
                    "path": str(twin_state_path),
                }
            else:
                imaging_join_coverage = {
                    "error": "imaging_feature_csv_missing_PATNO",
                    "path": str(args.imaging_feature_csv),
                }
        else:
            imaging_join_coverage = {
                "error": "imaging_feature_csv_not_found",
                "path": str(args.imaging_feature_csv),
            }

    # Counterfactual sensitivity comparison
    preferred_specs = [
        CounterfactualSpec(feature_name="UPDRS_I", delta=-0.5),
        CounterfactualSpec(feature_name="SCOPA_AUT_SCORE", delta=-0.5),
        CounterfactualSpec(feature_name="TREMOR_SCORE", delta=-0.5),
        CounterfactualSpec(feature_name="UPSIT_SCORE", delta=0.5),
        CounterfactualSpec(feature_name="ALPHA_SYNUCLEIN", delta=-0.5),
        CounterfactualSpec(feature_name="LRRK2", delta=-0.5),
    ]
    feature_set = set(baseline_sim.feature_names)
    specs = [s for s in preferred_specs if s.feature_name in feature_set]
    spec_warnings: list[str] = []
    if not specs and baseline_sim.feature_names:
        specs = [
            CounterfactualSpec(feature_name=baseline_sim.feature_names[0], delta=-0.1)
        ]
        spec_warnings.append(
            "preferred_specs_missing_in_feature_set_using_fallback_spec"
        )

    base_idx = _build_patno_index(baseline_sim)
    upd_idx = _build_patno_index(updated_sim)
    if has_common_patnos:
        selected_patnos = common_patnos[: min(25, len(common_patnos))]
        base_indices = [base_idx[p] for p in selected_patnos]
        upd_indices = [upd_idx[p] for p in selected_patnos]
    else:
        base_indices = list(base_idx.values())[: min(25, len(base_idx))]
        upd_indices = list(upd_idx.values())[: min(25, len(upd_idx))]

    sens_base = _compute_sensitivity(baseline_sim, base_indices, specs)
    sens_upd = _compute_sensitivity(updated_sim, upd_indices, specs)

    if sens_base.empty:
        agg_base = pd.DataFrame(
            {
                "intervention": [f"{s.feature_name}:{s.delta:+.3f}" for s in specs],
                "baseline_mean_abs_delta": [0.0] * len(specs),
            }
        )
    else:
        agg_base = (
            sens_base.groupby("intervention", as_index=False)["abs_delta_risk"]
            .mean()
            .rename(columns={"abs_delta_risk": "baseline_mean_abs_delta"})
        )

    if sens_upd.empty:
        agg_upd = pd.DataFrame(
            {
                "intervention": [f"{s.feature_name}:{s.delta:+.3f}" for s in specs],
                "updated_mean_abs_delta": [0.0] * len(specs),
            }
        )
    else:
        agg_upd = (
            sens_upd.groupby("intervention", as_index=False)["abs_delta_risk"]
            .mean()
            .rename(columns={"abs_delta_risk": "updated_mean_abs_delta"})
        )
    sens_summary = agg_base.merge(agg_upd, on="intervention", how="outer").fillna(0.0)
    sens_summary["delta_mean_abs_delta"] = (
        sens_summary["updated_mean_abs_delta"] - sens_summary["baseline_mean_abs_delta"]
    )
    sens_summary_path = out_dir / "counterfactual_delta_summary.csv"
    sens_summary.to_csv(sens_summary_path, index=False)

    # Zero-delta invariance check
    probe_pat = common_patnos[0] if has_common_patnos else sorted(base_idx.keys())[0]
    probe_idx = base_idx[probe_pat]
    zero_delta_pass = False
    if baseline_sim.feature_names:
        zero_spec = CounterfactualSpec(
            feature_name=baseline_sim.feature_names[0], delta=0.0
        )
        zero_result = baseline_sim.simulate_counterfactual(
            patient_idx=probe_idx,
            specs=[zero_spec],
            horizons=DEFAULT_HORIZONS,
        )
        if zero_result.counterfactual_paths:
            zero_key = list(zero_result.counterfactual_paths.keys())[0]
            baseline_final = float(zero_result.baseline_path[-1].risk_saa)
            cf_final = float(zero_result.counterfactual_paths[zero_key][-1].risk_saa)
            zero_delta_pass = abs(cf_final - baseline_final) <= 1e-9

    # Determinism check
    if has_common_patnos:
        hash1 = _sha256_json(
            {
                "mean_abs_delta": float(delta_df["abs_delta_risk"].mean()),
                "max_abs_delta": float(delta_df["abs_delta_risk"].max()),
                "n_changed": int(len(changed_df)),
            }
        )
        delta_df_repeat = _compute_patient_deltas(
            baseline_sim, updated_sim, common_patnos
        )
        hash2 = _sha256_json(
            {
                "mean_abs_delta": float(delta_df_repeat["abs_delta_risk"].mean()),
                "max_abs_delta": float(delta_df_repeat["abs_delta_risk"].max()),
                "n_changed": int(np.sum(delta_df_repeat["abs_delta_risk"] > 1e-6)),
            }
        )
        deterministic = hash1 == hash2
    else:
        hash1 = _sha256_json(
            {
                "baseline_sens": sens_base.to_dict(orient="records"),
                "updated_sens": sens_upd.to_dict(orient="records"),
            }
        )
        sens_base_repeat = _compute_sensitivity(baseline_sim, base_indices, specs)
        sens_upd_repeat = _compute_sensitivity(updated_sim, upd_indices, specs)
        hash2 = _sha256_json(
            {
                "baseline_sens": sens_base_repeat.to_dict(orient="records"),
                "updated_sens": sens_upd_repeat.to_dict(orient="records"),
            }
        )
        deterministic = hash1 == hash2

    # Visuals
    vis_dir = root / "visualizations" / "appendix" / "digital_twin_updates" / pull_id
    vis_dir.mkdir(parents=True, exist_ok=True)

    patient_fig = vis_dir / "patient_before_after_trajectory.png"
    if has_common_patnos:
        _plot_patient_before_after(baseline_sim, updated_sim, probe_pat, patient_fig)
    else:
        _plot_reference_before_after(baseline_sim, updated_sim, patient_fig)

    cohort_fig = vis_dir / "cohort_delta_distribution.png"
    _plot_cohort_delta_distribution(delta_df, cohort_fig)

    sens_fig = vis_dir / "intervention_sensitivity_shift.png"
    _plot_sensitivity_shift(sens_summary, sens_fig)

    summary = {
        "pull_id": pull_id,
        "n_patients_refreshed": int(len(common_patnos)),
        "n_baseline_patients": int(len(baseline_pat)),
        "n_updated_patients": int(len(updated_pat)),
        "n_changed_patients": int(len(changed_df)),
        "mean_abs_delta_risk": (
            float(delta_df["abs_delta_risk"].mean()) if not delta_df.empty else 0.0
        ),
        "max_abs_delta_risk": (
            float(delta_df["abs_delta_risk"].max()) if not delta_df.empty else 0.0
        ),
        "calibration_shift": {
            "note": "computed in external validation stage",
            "external_metrics_path": str(
                root
                / "outputs"
                / "external_validation"
                / pull_id
                / "external_metrics.json"
            ),
        },
        "warnings": ([] if has_common_patnos else ["no_common_patients"])
        + spec_warnings,
        "validation": {
            "zero_delta_counterfactual_pass": bool(zero_delta_pass),
            "deterministic_rerun_pass": bool(deterministic),
        },
        "artifacts": {
            "all_patient_deltas": str(all_delta_path),
            "changed_patients": str(changed_path),
            "counterfactual_delta_summary": str(sens_summary_path),
            "patient_trajectory_figure": str(patient_fig),
            "cohort_delta_figure": str(cohort_fig),
            "sensitivity_shift_figure": str(sens_fig),
        },
    }
    if imaging_join_coverage is not None:
        summary["imaging_feature_injection"] = imaging_join_coverage
        if "path" in imaging_join_coverage:
            summary["artifacts"]["twin_state_with_imaging"] = imaging_join_coverage[
                "path"
            ]

    summary_path = out_dir / "twin_refresh_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Digital twin update cycle complete")
    print(f"pull_id={pull_id}")
    print(f"summary={summary_path}")
    print(f"changed_patients={changed_path}")
    print(f"counterfactual_delta_summary={sens_summary_path}")


if __name__ == "__main__":
    main()
