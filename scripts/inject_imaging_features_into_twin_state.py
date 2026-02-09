#!/usr/bin/env python3
"""Inject PATNO-level imaging delta features into current digital twin state outputs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.digital_twin.simulator import DataDrivenTwinSimulator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join imaging delta features into twin state artifacts for the canonical dataset."
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=f"TWIN_IMG_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "data/03_prodromal/final_pyg_data_sota_run/test_data.pt",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=ROOT
        / "data/03_prodromal/final_pyg_data_sota_run/pyg_data_metadata.json",
    )
    parser.add_argument(
        "--imaging-feature-csv",
        type=Path,
        default=ROOT
        / "outputs/sota_lift/IMAGING_20260208_RUN2/imaging_delta_features.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / "outputs" / "digital_twin" / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = ROOT / "visualizations" / "appendix" / "digital_twin" / args.run_tag
    vis_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    twin_df: pd.DataFrame
    try:
        sim = DataDrivenTwinSimulator(
            data_path=args.data_path,
            metadata_path=args.metadata_path,
            temperature=2.5,
        )
        patnos = sim.data.patno.detach().cpu().numpy().astype(int)
        rows = []
        for idx, pat in enumerate(patnos):
            states = sim.simulate_patient(
                patient_idx=int(idx), horizons=[0, 6, 12, 18, 24]
            )
            rows.append(
                {
                    "PATNO": int(pat),
                    "risk_saa_month0": float(states[0].risk_saa),
                    "risk_saa_month24": float(states[-1].risk_saa),
                    "risk_survival_month24": float(states[-1].risk_survival),
                    "unc_low_month24": float(states[-1].uncertainty_low),
                    "unc_high_month24": float(states[-1].uncertainty_high),
                }
            )
        twin_df = pd.DataFrame(rows)
    except Exception as exc:
        warnings.append(f"simulator_load_failed:{exc}")
        data = torch.load(args.data_path, weights_only=False, map_location="cpu")
        patno = pd.Series(
            data.patno.detach().cpu().numpy()
            if hasattr(data, "patno")
            else np.arange(data.x.shape[0])
        )
        twin_df = pd.DataFrame({"PATNO": patno.astype(int)})
        if hasattr(data, "time"):
            twin_df["time"] = data.time.detach().cpu().numpy()
        if hasattr(data, "event"):
            twin_df["event"] = data.event.detach().cpu().numpy()
        if hasattr(data, "saa_label"):
            twin_df["saa_label"] = data.saa_label.detach().cpu().numpy()
        twin_df["risk_saa_month0"] = np.nan
        twin_df["risk_saa_month24"] = np.nan
        twin_df["risk_survival_month24"] = np.nan
        twin_df["unc_low_month24"] = np.nan
        twin_df["unc_high_month24"] = np.nan

    img_df = pd.read_csv(args.imaging_feature_csv)
    img_df["PATNO"] = pd.to_numeric(img_df["PATNO"], errors="coerce").astype("Int64")
    twin_df["PATNO"] = pd.to_numeric(twin_df["PATNO"], errors="coerce").astype("Int64")

    merged = twin_df.merge(img_df, on="PATNO", how="left")
    merged_path = out_dir / "twin_state_with_imaging.csv"
    merged.to_csv(merged_path, index=False)

    imaging_cols = [c for c in img_df.columns if c != "PATNO"]
    merged["has_imaging_features"] = merged[imaging_cols].notna().any(axis=1)

    plot_col = "risk_saa_month24"
    if merged[plot_col].isna().all() and "saa_label" in merged.columns:
        plot_col = "saa_label"

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    with_img = (
        pd.to_numeric(
            merged.loc[merged["has_imaging_features"], plot_col], errors="coerce"
        )
        .dropna()
        .values
    )
    no_img = (
        pd.to_numeric(
            merged.loc[~merged["has_imaging_features"], plot_col], errors="coerce"
        )
        .dropna()
        .values
    )
    bins = (
        np.linspace(0.0, 1.0, 20)
        if plot_col != "saa_label"
        else np.array([-0.5, 0.5, 1.5])
    )
    if len(with_img) > 0:
        ax.hist(
            with_img, bins=bins, alpha=0.75, label=f"with imaging (n={len(with_img)})"
        )
    if len(no_img) > 0:
        ax.hist(
            no_img, bins=bins, alpha=0.75, label=f"without imaging (n={len(no_img)})"
        )
    if len(with_img) == 0 and len(no_img) == 0:
        ax.text(0.05, 0.6, "No plottable twin risk values available.", fontsize=11)
    ax.set_title("Twin State by Imaging Feature Availability")
    ax.set_xlabel(plot_col)
    ax.set_ylabel("count")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig_path = vis_dir / "twin_risk_by_imaging_availability.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "run_tag": args.run_tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "data_path": str(args.data_path),
            "metadata_path": str(args.metadata_path),
            "imaging_feature_csv": str(args.imaging_feature_csv),
        },
        "coverage": {
            "n_twin_patients": int(len(twin_df)),
            "n_imaging_feature_patients": int(img_df["PATNO"].dropna().nunique()),
            "n_patients_with_joined_imaging": int(merged["has_imaging_features"].sum()),
            "n_patients_without_joined_imaging": int(
                (~merged["has_imaging_features"]).sum()
            ),
        },
        "artifacts": {
            "twin_state_with_imaging_csv": str(merged_path),
            "risk_by_imaging_plot": str(fig_path),
        },
        "warnings": warnings,
    }
    summary_path = out_dir / "twin_imaging_injection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Twin imaging feature injection complete")
    print(f"summary={summary_path}")
    print(f"twin_state_with_imaging={merged_path}")


if __name__ == "__main__":
    main()
