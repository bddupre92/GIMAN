from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .contracts import assert_classification_contract, load_contract_from_metadata
from .metrics import (
    auc_with_ci,
    brier,
    c_index_with_ci,
    decision_curve_net_benefit,
    expected_calibration_error,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


REQUIRED_TENSOR_FIELDS = ("x", "edge_index", "time", "event", "saa_label", "patno")


def _require_tensor_fields(obj: Any, obj_name: str) -> None:
    missing = [name for name in REQUIRED_TENSOR_FIELDS if not hasattr(obj, name)]
    if missing:
        raise ValueError(f"{obj_name} missing required tensor fields: {missing}")


def _load_tensor_data(train_path: Path, test_path: Path) -> dict[str, np.ndarray]:
    train = torch.load(train_path, weights_only=False)
    test = torch.load(test_path, weights_only=False)
    _require_tensor_fields(train, "train_data")
    _require_tensor_fields(test, "test_data")

    y_train = train.saa_label.detach().cpu().numpy().astype(int)
    y_test = test.saa_label.detach().cpu().numpy().astype(int)
    e_train = train.event.detach().cpu().numpy().astype(int)
    e_test = test.event.detach().cpu().numpy().astype(int)
    t_train = train.time.detach().cpu().numpy().astype(float)
    t_test = test.time.detach().cpu().numpy().astype(float)
    pat_train = train.patno.detach().cpu().numpy().astype(int)
    pat_test = test.patno.detach().cpu().numpy().astype(int)

    if np.any(t_train < 0) or np.any(t_test < 0):
        raise ValueError("time must be non-negative for all samples")

    overlap = np.intersect1d(np.unique(pat_train), np.unique(pat_test))
    if overlap.size > 0:
        raise ValueError(f"patient leakage detected: {overlap.size} overlapping PATNOs")

    if np.array_equal(y_train, e_train) and np.array_equal(y_test, e_test):
        raise ValueError(
            "classification target appears to be survival event proxy; saa_label must be distinct from event"
        )

    return {
        "x_train": train.x.detach().cpu().numpy(),
        "x_test": test.x.detach().cpu().numpy(),
        "y_train": y_train,
        "y_test": y_test,
        "e_train": e_train,
        "e_test": e_test,
        "t_train": t_train,
        "t_test": t_test,
        "patno_train": pat_train,
        "patno_test": pat_test,
    }


def _evaluate_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> dict[str, Any]:
    auc = auc_with_ci(y_true, y_prob, n_bootstrap=300, seed=42)
    cidx = c_index_with_ci(y_prob, time, event, n_bootstrap=300, seed=42)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    brier_score = brier(y_true, y_prob)
    nb = decision_curve_net_benefit(y_true, y_prob)
    return {
        "auc": auc.value,
        "auc_ci_95": [auc.ci_low, auc.ci_high],
        "c_index": cidx.value,
        "c_index_ci_95": [cidx.ci_low, cidx.ci_high],
        "ece": ece,
        "brier": brier_score,
        "decision_curve": nb,
    }


def _build_baselines(data: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    t_test = data["t_test"]
    e_test = data["e_test"]

    out: dict[str, dict[str, Any]] = {}

    lr = LogisticRegression(max_iter=1500, random_state=42)
    lr.fit(x_train, y_train)
    prob_lr = lr.predict_proba(x_test)[:, 1]
    out["logistic_regression"] = _evaluate_classifier(y_test, prob_lr, t_test, e_test)

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(x_train, y_train)
    prob_rf = rf.predict_proba(x_test)[:, 1]
    out["random_forest"] = _evaluate_classifier(y_test, prob_rf, t_test, e_test)

    svm = SVC(probability=True, random_state=42)
    svm.fit(x_train, y_train)
    prob_svm = svm.predict_proba(x_test)[:, 1]
    out["svm_rbf"] = _evaluate_classifier(y_test, prob_svm, t_test, e_test)

    return out


def _load_fuzzy_artifacts(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    p_full = (
        root
        / "outputs"
        / "phase9_neuro_fuzzy_sota_run_from50ckpt"
        / "full_training_results.json"
    )
    p_multi = (
        root
        / "outputs"
        / "phase9_neuro_fuzzy_sota_run_from50ckpt"
        / "multitask_training_results.json"
    )
    p8 = root / "outputs" / "phase8_2_final_training_sota_run" / "training_results.json"

    if p_full.exists():
        result["fuzzy_full"] = json.loads(p_full.read_text(encoding="utf-8"))
    if p_multi.exists():
        result["fuzzy_multitask"] = json.loads(p_multi.read_text(encoding="utf-8"))
    if p8.exists():
        result["phase8_survival"] = json.loads(p8.read_text(encoding="utf-8"))

    return result


def _plot_benchmark(metrics: dict[str, dict[str, Any]], output_dir: Path) -> Path:
    names = list(metrics.keys())
    auc_vals = [metrics[n]["auc"] for n in names]
    cidx_vals = [metrics[n]["c_index"] for n in names]

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, auc_vals, width, label="AUC", color="#4E79A7")
    ax.bar(x + width / 2, cidx_vals, width, label="C-index", color="#F28E2B")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Internal Benchmark: Baselines on Canonical Patient Split")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "internal_benchmark_metrics.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def _plot_calibration(metrics: dict[str, dict[str, Any]], output_dir: Path) -> Path:
    labels = list(metrics.keys())
    ece_vals = [metrics[k]["ece"] for k in labels]
    brier_vals = [metrics[k]["brier"] for k in labels]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, ece_vals, width, label="ECE", color="#59A14F")
    ax.bar(x + width / 2, brier_vals, width, label="Brier", color="#E15759")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Calibration Diagnostics (Lower is Better)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "internal_calibration_metrics.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def run_internal_sota_lock(
    train_data_path: Path,
    test_data_path: Path,
    metadata_path: Path,
    output_json_path: Path,
    output_report_path: Path,
    figure_dir: Path,
) -> dict[str, Any]:
    root = _repo_root()

    contract = load_contract_from_metadata(metadata_path)
    assert_classification_contract(contract.classification_key, contract.event_key)

    tensors = _load_tensor_data(train_data_path, test_data_path)
    baseline_metrics = _build_baselines(tensors)
    fuzzy = _load_fuzzy_artifacts(root)

    benchmark_fig = _plot_benchmark(baseline_metrics, figure_dir)
    calibration_fig = _plot_calibration(baseline_metrics, figure_dir)

    manuscript_paths = [
        root / "Archive_New" / "pipeline_cleanup_2026-02-06" / "main.tex",
        root / "Docs" / "audit" / "SOTA_GAP_CLOSURE_SPRINT1.md",
    ]
    banned_phrases = ["clinical sota", "state-of-the-art", "clinically sota"]
    governance_hits: dict[str, list[str]] = {}
    for p in manuscript_paths:
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore").lower()
        hits = [phrase for phrase in banned_phrases if phrase in txt]
        governance_hits[str(p)] = hits

    payload = {
        "contract": contract.as_dict(),
        "baselines": baseline_metrics,
        "fuzzy_artifacts": fuzzy,
        "validation_checks": {
            "required_tensor_fields": list(REQUIRED_TENSOR_FIELDS),
            "patient_disjoint": True,
            "time_non_negative": True,
            "classification_not_proxy_event": True,
        },
        "claim_governance": governance_hits,
        "figures": {
            "benchmark": str(benchmark_fig),
            "calibration": str(calibration_fig),
        },
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _fmt_metric(m: dict[str, Any]) -> str:
        return (
            f"AUC {m['auc']:.4f} [95% CI {m['auc_ci_95'][0]:.4f}, {m['auc_ci_95'][1]:.4f}], "
            f"C-index {m['c_index']:.4f} [95% CI {m['c_index_ci_95'][0]:.4f}, {m['c_index_ci_95'][1]:.4f}], "
            f"ECE {m['ece']:.4f}, Brier {m['brier']:.4f}"
        )

    lines = [
        "# SOTA Internal Lock Report",
        "",
        "## Scope",
        "Internal evidence lock on canonical patient-level split. Clinical-superiority claims remain blocked pending external validation.",
        "",
        "## Contract",
        f"- Survival time key: `{contract.time_key}`",
        f"- Survival event key: `{contract.event_key}`",
        f"- Classification key: `{contract.classification_key}`",
        f"- Split hash: `{contract.split_hash}`",
        "",
        "## Validation Checks",
        "- Required tensor fields present: `x, edge_index, time, event, saa_label, patno`",
        "- Patient-level split disjointness: `pass`",
        "- Non-negative survival times: `pass`",
        "- Classification label proxy check (`saa_label` vs `event`): `pass`",
        "",
        "## Baseline Results (Artifact-backed)",
    ]

    for model_name, m in baseline_metrics.items():
        lines.append(f"- `{model_name}`: {_fmt_metric(m)}")

    lines += [
        "",
        "## FUZZY GIMAN Artifacts",
        f"- Phase 9 full artifact present: `{('fuzzy_full' in fuzzy)}`",
        f"- Phase 9 multitask artifact present: `{('fuzzy_multitask' in fuzzy)}`",
        f"- Phase 8 survival artifact present: `{('phase8_survival' in fuzzy)}`",
        "",
        "## Claim Governance Scan",
    ]

    for p, hits in governance_hits.items():
        if hits:
            lines.append(f"- `{p}`: banned phrase hit(s) -> `{hits}`")
        else:
            lines.append(f"- `{p}`: no banned phrase hits")

    lines += [
        "",
        "## Governance",
        "- This report is internal-only and does not authorize clinical-superiority wording.",
        "- External validation artifact is required before clinical-readiness claims.",
        "",
        "## Figures",
        f"- Benchmark metrics: `{benchmark_fig}`",
        f"- Calibration metrics: `{calibration_fig}`",
        "",
        "## Outputs",
        f"- JSON payload: `{output_json_path}`",
    ]

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text("\n".join(lines), encoding="utf-8")

    return payload


if __name__ == "__main__":
    root = _repo_root()
    payload_path = root / "outputs" / "sota_lock" / "internal_sota_lock.json"
    report_path = root / "Docs" / "audit" / "SOTA_INTERNAL_LOCK_REPORT.md"
    fig_dir = root / "visualizations" / "publication_internal"
    run_internal_sota_lock(
        train_data_path=root
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_run"
        / "train_data.pt",
        test_data_path=root
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_run"
        / "test_data.pt",
        metadata_path=root
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_run"
        / "pyg_data_metadata.json",
        output_json_path=payload_path,
        output_report_path=report_path,
        figure_dir=fig_dir,
    )
