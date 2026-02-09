from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

QUICK_SCRIPT = ROOT / "archive/development/phase9/phase9_neuro_fuzzy_implementation.py"
MULTITASK_SCRIPT = ROOT / "archive/development/phase9/phase9_multitask_learning.py"


ABLATIONS: dict[str, list[str]] = {
    "baseline": [],
    "no_csf": ["ALPHA_SYNUCLEIN", "TOTAL_TAU", "ABETA42", "PTAU181"],
    "no_ess": ["ESS_TOTAL"],
    "no_lrrk2": ["LRRK2"],
    "no_csf_no_ess": [
        "ALPHA_SYNUCLEIN",
        "TOTAL_TAU",
        "ABETA42",
        "PTAU181",
        "ESS_TOTAL",
    ],
    "no_csf_no_ess_no_lrrk2": [
        "ALPHA_SYNUCLEIN",
        "TOTAL_TAU",
        "ABETA42",
        "PTAU181",
        "ESS_TOTAL",
        "LRRK2",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run targeted Phase 9 ablations + retuning for run3 tensors."
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=f"ABLATE_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--train-data-path",
        type=Path,
        default=ROOT / "data/03_prodromal/final_pyg_data_sota_run/train_data.pt",
    )
    parser.add_argument(
        "--test-data-path",
        type=Path,
        default=ROOT / "data/03_prodromal/final_pyg_data_sota_run/test_data.pt",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=ROOT
        / "data/03_prodromal/final_pyg_data_sota_run/pyg_data_metadata.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iters", type=int, default=600)
    return parser.parse_args()


def _run_cmd(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return {
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-5000:],
        "stderr_tail": proc.stderr[-5000:],
    }


def _quick_configs(seed: int, bootstrap_iters: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "quick_a",
            "args": [
                "--seed",
                str(seed),
                "--epochs",
                "100",
                "--learning-rate",
                "0.001",
                "--num-rules",
                "16",
                "--bootstrap-iters",
                str(bootstrap_iters),
                "--classification-loss",
                "focal",
                "--focal-gamma",
                "2.0",
            ],
        },
        {
            "name": "quick_b",
            "args": [
                "--seed",
                str(seed),
                "--epochs",
                "140",
                "--learning-rate",
                "0.0007",
                "--num-rules",
                "24",
                "--bootstrap-iters",
                str(bootstrap_iters),
                "--classification-loss",
                "focal",
                "--focal-gamma",
                "2.0",
            ],
        },
        {
            "name": "quick_c",
            "args": [
                "--seed",
                str(seed),
                "--epochs",
                "160",
                "--learning-rate",
                "0.0005",
                "--num-rules",
                "32",
                "--feature-dropout-rate",
                "0.02",
                "--bootstrap-iters",
                str(bootstrap_iters),
                "--classification-loss",
                "focal",
                "--focal-gamma",
                "2.5",
            ],
        },
    ]


def _multitask_configs(seed: int, bootstrap_iters: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "multitask_a",
            "args": [
                "--seed",
                str(seed),
                "--epochs",
                "180",
                "--learning-rate",
                "0.005",
                "--num-rules",
                "32",
                "--classification-loss-weight",
                "1.0",
                "--bootstrap-iters",
                str(bootstrap_iters),
                "--classification-loss",
                "focal",
                "--focal-gamma",
                "2.0",
            ],
        },
        {
            "name": "multitask_b",
            "args": [
                "--seed",
                str(seed),
                "--epochs",
                "220",
                "--learning-rate",
                "0.003",
                "--num-rules",
                "32",
                "--classification-loss-weight",
                "1.5",
                "--bootstrap-iters",
                str(bootstrap_iters),
                "--classification-loss",
                "focal",
                "--focal-gamma",
                "2.0",
            ],
        },
        {
            "name": "multitask_c",
            "args": [
                "--seed",
                str(seed),
                "--epochs",
                "220",
                "--learning-rate",
                "0.003",
                "--num-rules",
                "24",
                "--classification-loss-weight",
                "2.0",
                "--feature-dropout-rate",
                "0.02",
                "--bootstrap-iters",
                str(bootstrap_iters),
                "--classification-loss",
                "focal",
                "--focal-gamma",
                "2.5",
            ],
        },
    ]


def _extract_metric(model_type: str, payload: dict[str, Any]) -> float:
    if model_type == "quick":
        return float(
            max(payload.get("saa_auc", 0.0), payload.get("best_saa_auc_seen", 0.0))
        )
    return float(
        max(payload.get("final_saa_auc", 0.0), payload.get("best_saa_auc_seen", 0.0))
    )


def run_suite(args: argparse.Namespace) -> None:
    out_root = ROOT / "outputs" / "phase9_ablations" / args.run_tag
    out_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []

    common_args = [
        "--train-data-path",
        str(args.train_data_path),
        "--test-data-path",
        str(args.test_data_path),
        "--metadata-path",
        str(args.metadata_path),
    ]

    quick_cfgs = _quick_configs(seed=args.seed, bootstrap_iters=args.bootstrap_iters)
    multi_cfgs = _multitask_configs(
        seed=args.seed, bootstrap_iters=args.bootstrap_iters
    )

    for ablation_name, feature_list in ABLATIONS.items():
        feature_csv = ",".join(feature_list)

        for cfg in quick_cfgs:
            trial_id = f"{ablation_name}__{cfg['name']}"
            trial_out = out_root / "quick" / trial_id
            cmd = [
                str(PYTHON),
                str(QUICK_SCRIPT),
                *common_args,
                "--output-dir",
                str(trial_out),
                "--feature-blacklist",
                feature_csv,
                *cfg["args"],
            ]
            run = _run_cmd(cmd)
            logs.append({"trial_id": trial_id, "model_type": "quick", **run})
            result_file = trial_out / "quick_neuro_fuzzy_results.json"
            if run["returncode"] != 0 or not result_file.exists():
                records.append(
                    {
                        "trial_id": trial_id,
                        "model_type": "quick",
                        "ablation": ablation_name,
                        "config": cfg["name"],
                        "feature_blacklist": feature_csv,
                        "status": "failed",
                        "saa_auc_metric": None,
                    }
                )
                continue

            payload = json.loads(result_file.read_text(encoding="utf-8"))
            records.append(
                {
                    "trial_id": trial_id,
                    "model_type": "quick",
                    "ablation": ablation_name,
                    "config": cfg["name"],
                    "feature_blacklist": feature_csv,
                    "status": "ok",
                    "saa_auc_metric": _extract_metric("quick", payload),
                    "final_saa_auc": payload.get("saa_auc"),
                    "best_saa_auc_seen": payload.get(
                        "best_saa_auc_seen", payload.get("saa_auc")
                    ),
                    "accuracy": payload.get("accuracy"),
                }
            )

        for cfg in multi_cfgs:
            trial_id = f"{ablation_name}__{cfg['name']}"
            trial_out = out_root / "multitask" / trial_id
            cmd = [
                str(PYTHON),
                str(MULTITASK_SCRIPT),
                *common_args,
                "--output-dir",
                str(trial_out),
                "--feature-blacklist",
                feature_csv,
                *cfg["args"],
            ]
            run = _run_cmd(cmd)
            logs.append({"trial_id": trial_id, "model_type": "multitask", **run})
            result_file = trial_out / "multitask_training_results.json"
            if run["returncode"] != 0 or not result_file.exists():
                records.append(
                    {
                        "trial_id": trial_id,
                        "model_type": "multitask",
                        "ablation": ablation_name,
                        "config": cfg["name"],
                        "feature_blacklist": feature_csv,
                        "status": "failed",
                        "saa_auc_metric": None,
                    }
                )
                continue

            payload = json.loads(result_file.read_text(encoding="utf-8"))
            records.append(
                {
                    "trial_id": trial_id,
                    "model_type": "multitask",
                    "ablation": ablation_name,
                    "config": cfg["name"],
                    "feature_blacklist": feature_csv,
                    "status": "ok",
                    "saa_auc_metric": _extract_metric("multitask", payload),
                    "final_saa_auc": payload.get("final_saa_auc"),
                    "best_saa_auc_seen": payload.get(
                        "best_saa_auc_seen", payload.get("final_saa_auc")
                    ),
                    "final_c_index": payload.get("final_c_index"),
                }
            )

    runs_df = pd.DataFrame(records)
    logs_df = pd.DataFrame(logs)
    runs_csv = out_root / "ablation_trials.csv"
    logs_json = out_root / "ablation_logs.json"
    runs_df.to_csv(runs_csv, index=False)
    logs_json.write_text(logs_df.to_json(orient="records", indent=2), encoding="utf-8")

    ok_df = runs_df[runs_df["status"] == "ok"].copy()
    if ok_df.empty:
        raise RuntimeError(f"No successful ablation runs. See {logs_json}")

    best_idx = ok_df["saa_auc_metric"].astype(float).idxmax()
    best_row = ok_df.loc[best_idx].to_dict()

    best_by_model = (
        ok_df.sort_values("saa_auc_metric", ascending=False)
        .groupby("model_type", as_index=False)
        .first()
    )
    best_by_ablation = (
        ok_df.sort_values("saa_auc_metric", ascending=False)
        .groupby(["model_type", "ablation"], as_index=False)
        .first()
    )

    summary = {
        "run_tag": args.run_tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_trials_total": int(len(runs_df)),
        "n_trials_success": int(len(ok_df)),
        "best_overall": best_row,
        "best_by_model": best_by_model.to_dict(orient="records"),
        "best_by_model_and_ablation": best_by_ablation.to_dict(orient="records"),
        "artifacts": {
            "trials_csv": str(runs_csv),
            "logs_json": str(logs_json),
        },
    }
    summary_json = out_root / "ablation_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    viz_dir = ROOT / "visualizations" / "sota_lift" / args.run_tag
    viz_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)
    plot_df = (
        best_by_ablation[["model_type", "ablation", "saa_auc_metric"]]
        .pivot(index="ablation", columns="model_type", values="saa_auc_metric")
        .fillna(0.0)
        .sort_index()
    )
    plot_df.plot(kind="bar", ax=ax)
    ax.axhline(0.9, linestyle="--", color="black", linewidth=1, label="Target 0.90")
    ax.set_ylabel("Best SAA AUC (per ablation)")
    ax.set_title("Targeted Ablations + Retuning (Quick vs Multitask)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig_path = viz_dir / "phase9_targeted_ablations_best_by_group.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Ablation suite complete: {summary_json}")
    print(f"✓ Visualization: {fig_path}")


if __name__ == "__main__":
    run_suite(parse_args())
