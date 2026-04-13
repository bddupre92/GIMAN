from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .simulator import DataDrivenTwinSimulator, save_simulation_result
from .state import CounterfactualSpec


@dataclass(frozen=True)
class TwinRunConfig:
    patient_idx: int
    specs: list[CounterfactualSpec]
    horizons: list[int]
    temperature: float = 2.5


def run_twin_counterfactual(
    data_path: Path,
    metadata_path: Path,
    output_json: Path,
    output_figure: Path,
    config: TwinRunConfig,
) -> None:
    sim = DataDrivenTwinSimulator(
        data_path=data_path,
        metadata_path=metadata_path,
        temperature=config.temperature,
    )
    result = sim.simulate_counterfactual(
        patient_idx=config.patient_idx,
        specs=config.specs,
        horizons=config.horizons,
    )
    save_simulation_result(result, output_json)

    t = [x.t_month for x in result.baseline_path]
    y_base = [x.risk_saa for x in result.baseline_path]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, y_base, "o-", label="baseline", color="#4E79A7")
    for key, path in result.counterfactual_paths.items():
        y = [x.risk_saa for x in path]
        ax.plot(t, y, "o--", label=key)

    ax.set_title("Digital Twin v1 Counterfactual Trajectories")
    ax.set_xlabel("month")
    ax.set_ylabel("predicted SAA risk")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure, dpi=300)
    plt.close(fig)


def run_twin_sensitivity_scan(
    data_path: Path,
    metadata_path: Path,
    output_csv: Path,
    output_figure: Path,
    specs: list[CounterfactualSpec],
    patient_indices: list[int],
    horizons: list[int],
    temperature: float = 2.5,
) -> pd.DataFrame:
    sim = DataDrivenTwinSimulator(
        data_path=data_path,
        metadata_path=metadata_path,
        temperature=temperature,
    )

    rows: list[dict[str, float | str | int]] = []
    for patient_idx in patient_indices:
        result = sim.simulate_counterfactual(
            patient_idx=patient_idx,
            specs=specs,
            horizons=horizons,
        )
        for key, delta in result.delta_risk.items():
            rows.append(
                {
                    "patient_idx": int(patient_idx),
                    "intervention": key,
                    "delta_risk": float(delta),
                    "abs_delta_risk": float(abs(delta)),
                }
            )

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    agg = (
        df.groupby("intervention", as_index=False)["abs_delta_risk"]
        .mean()
        .sort_values("abs_delta_risk", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        agg["intervention"][::-1],
        agg["abs_delta_risk"][::-1],
        color="#4E79A7",
    )
    ax.set_title("Digital Twin Sensitivity (Mean |Δ risk| over patients)")
    ax.set_xlabel("mean absolute delta in final-horizon SAA risk")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure, dpi=300)
    plt.close(fig)
    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    run_twin_counterfactual(
        data_path=root
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_run"
        / "test_data.pt",
        metadata_path=root
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_run"
        / "pyg_data_metadata.json",
        output_json=root / "outputs" / "digital_twin" / "patient_0_counterfactual.json",
        output_figure=root
        / "visualizations"
        / "appendix"
        / "digital_twin"
        / "patient_0_counterfactual.png",
        config=TwinRunConfig(
            patient_idx=0,
            specs=[
                CounterfactualSpec(feature_name="UPDRS_I", delta=-0.5),
                CounterfactualSpec(feature_name="SCOPA_AUT_SCORE", delta=-0.5),
            ],
            horizons=[0, 6, 12, 18, 24],
            temperature=2.5,
        ),
    )
