#!/usr/bin/env python3
"""Generate stage-correlated synthetic wearable data.

THIS SCRIPT GENERATES SYNTHETIC DATA ONLY.
It does NOT use or read any real patient sensor data.
All values are computationally generated based on published
clinical ranges for each NSD-ISS stage.

Usage:
    python generator.py              # Generate for 5 representative patients
    python generator.py --patnos 3203 3434 3785 3476 3960
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Stage-specific parameter ranges (from clinical literature)
STAGE_PARAMS = {
    "0": {
        "gait_speed": {"mean": 1.1, "std": 0.08},
        "tremor_amplitude": {"mean": 0.06, "std": 0.02},
        "sleep_hours": {"mean": 7.2, "std": 0.5},
    },
    "1": {
        "gait_speed": {"mean": 1.05, "std": 0.10},
        "tremor_amplitude": {"mean": 0.08, "std": 0.03},
        "sleep_hours": {"mean": 7.0, "std": 0.6},
    },
    "2B": {
        "gait_speed": {"mean": 0.90, "std": 0.10},
        "tremor_amplitude": {"mean": 0.12, "std": 0.04},
        "sleep_hours": {"mean": 6.5, "std": 0.7},
    },
    "3": {
        "gait_speed": {"mean": 0.72, "std": 0.12},
        "tremor_amplitude": {"mean": 0.18, "std": 0.05},
        "sleep_hours": {"mean": 6.0, "std": 0.9},
    },
    "4": {
        "gait_speed": {"mean": 0.55, "std": 0.12},
        "tremor_amplitude": {"mean": 0.28, "std": 0.08},
        "sleep_hours": {"mean": 5.2, "std": 1.0},
    },
    "5": {
        "gait_speed": {"mean": 0.40, "std": 0.10},
        "tremor_amplitude": {"mean": 0.35, "std": 0.10},
        "sleep_hours": {"mean": 4.5, "std": 1.2},
    },
    "6": {
        "gait_speed": {"mean": 0.30, "std": 0.08},
        "tremor_amplitude": {"mean": 0.40, "std": 0.12},
        "sleep_hours": {"mean": 4.0, "std": 1.5},
    },
}

# Default representative patients (from Paper 6)
DEFAULT_PATIENTS = {
    3203: "3",
    3434: "2B",
    3785: "2B",
    3476: "3",
    3960: "4",
}


def generate_wearable_data(
    patno: int,
    stage: str,
    n_days: int = 90,
    seed: int | None = None,
) -> dict:
    """Generate synthetic wearable data for a single patient."""
    rng = np.random.default_rng(seed or patno)
    params = STAGE_PARAMS.get(stage, STAGE_PARAMS["3"])

    days = list(range(n_days))

    # Add slow trend + daily noise + occasional events
    def gen_series(mean, std, trend_per_day=0.0):
        base = rng.normal(mean, std, n_days)
        trend = np.arange(n_days) * trend_per_day
        # Add weekly periodicity
        weekly = 0.02 * np.sin(2 * np.pi * np.arange(n_days) / 7)
        return np.clip(base + trend + weekly, 0, None).tolist()

    gait = gen_series(
        params["gait_speed"]["mean"],
        params["gait_speed"]["std"],
        trend_per_day=-0.0005,  # Slight decline over 90 days
    )
    tremor = gen_series(
        params["tremor_amplitude"]["mean"],
        params["tremor_amplitude"]["std"],
        trend_per_day=0.0002,
    )
    sleep = gen_series(
        params["sleep_hours"]["mean"],
        params["sleep_hours"]["std"],
    )

    # Round for readability
    gait = [round(v, 3) for v in gait]
    tremor = [round(v, 4) for v in tremor]
    sleep = [round(v, 1) for v in sleep]

    return {
        "patno": patno,
        "stage_at_generation": stage,
        "n_days": n_days,
        "source": "SYNTHETIC",
        "disclaimer": (
            "This is computationally generated synthetic wearable data "
            "for demonstration purposes only. It does NOT represent real "
            "patient sensor readings."
        ),
        "sensor_data": {
            "gait_speed": {"dates": days, "values": gait},
            "tremor_amplitude": {"dates": days, "values": tremor},
            "sleep_hours": {"dates": days, "values": sleep},
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic wearable data")
    parser.add_argument("--patnos", nargs="+", type=int, default=list(DEFAULT_PATIENTS.keys()))
    parser.add_argument("--n-days", type=int, default=90)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "samples")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for patno in args.patnos:
        stage = DEFAULT_PATIENTS.get(patno, "3")
        data = generate_wearable_data(patno, stage, n_days=args.n_days)

        path = args.output_dir / f"patient_{patno}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  Generated: {path.name} (Stage {stage}, {args.n_days} days)")

    print(f"\nGenerated {len(args.patnos)} synthetic wearable files in {args.output_dir}")
    print("WARNING: These are SYNTHETIC data files — NOT real patient data.")


if __name__ == "__main__":
    main()
