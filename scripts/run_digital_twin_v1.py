from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from giman_pipeline.digital_twin.counterfactual import (
    TwinRunConfig,
    run_twin_sensitivity_scan,
    run_twin_counterfactual,
)
from giman_pipeline.digital_twin.state import CounterfactualSpec

if __name__ == "__main__":
    data_path = root / "data" / "03_prodromal" / "final_pyg_data_sota_run" / "test_data.pt"
    metadata_path = (
        root / "data" / "03_prodromal" / "final_pyg_data_sota_run" / "pyg_data_metadata.json"
    )
    specs = [
        CounterfactualSpec(feature_name="UPDRS_I", delta=-0.5),
        CounterfactualSpec(feature_name="SCOPA_AUT_SCORE", delta=-0.5),
        CounterfactualSpec(feature_name="TREMOR_SCORE", delta=-0.5),
        CounterfactualSpec(feature_name="UPSIT_SCORE", delta=0.5),
        CounterfactualSpec(feature_name="ALPHA_SYNUCLEIN", delta=-1.0),
        CounterfactualSpec(feature_name="LRRK2", delta=-1.0),
    ]
    horizons = [0, 6, 12, 18, 24]

    run_twin_counterfactual(
        data_path=data_path,
        metadata_path=metadata_path,
        output_json=root / "outputs" / "digital_twin" / "patient_0_counterfactual.json",
        output_figure=root
        / "visualizations"
        / "appendix"
        / "digital_twin"
        / "patient_0_counterfactual.png",
        config=TwinRunConfig(
            patient_idx=0,
            specs=specs,
            horizons=horizons,
            temperature=2.5,
        ),
    )

    run_twin_sensitivity_scan(
        data_path=data_path,
        metadata_path=metadata_path,
        output_csv=root / "outputs" / "digital_twin" / "sensitivity_scan.csv",
        output_figure=root
        / "visualizations"
        / "appendix"
        / "digital_twin"
        / "sensitivity_scan.png",
        specs=specs,
        patient_indices=list(range(25)),
        horizons=horizons,
        temperature=2.5,
    )
