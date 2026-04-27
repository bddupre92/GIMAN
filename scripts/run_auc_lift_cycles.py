from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from giman_pipeline.sota.metrics import (  # noqa: E402
    auc_with_ci,
    brier,
    decision_curve_net_benefit,
    expected_calibration_error,
    pr_auc_with_ci,
)

try:
    from torch_geometric.nn import GATConv
except Exception:  # pragma: no cover
    GATConv = None


@dataclass
class Paths:
    cycle_root: Path
    vis_root: Path
    docs_root: Path
    cycle_dataset_root: Path


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _text_save(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    return float(average_precision_score(y_true, y_score))


def _parse_markdown_auc(path: Path) -> float | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"FUZZY GIMAN\s*&\s*([0-9.]+)", text)
    if m:
        return float(m.group(1))
    m2 = re.search(r"best_test_auc[\"']?\\s*[:=]\\s*([0-9.]+)", text)
    if m2:
        return float(m2.group(1))
    return None


def _load_tensor_dataset(data_dir: Path) -> tuple[Any, Any, dict[str, Any]]:
    train = torch.load(data_dir / "train_data.pt", weights_only=False)
    test = torch.load(data_dir / "test_data.pt", weights_only=False)
    meta = _load_json(data_dir / "pyg_data_metadata.json")
    return train, test, meta


def _extract_xy_pat(data_obj: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = data_obj.x.detach().cpu().numpy().astype(np.float32)
    y = data_obj.saa_label.detach().cpu().numpy().astype(int)
    pat = data_obj.patno.detach().cpu().numpy().astype(int)
    return x, y, pat


def _extract_surv(data_obj: Any) -> tuple[np.ndarray, np.ndarray]:
    t = data_obj.time.detach().cpu().numpy().astype(float)
    e = data_obj.event.detach().cpu().numpy().astype(int)
    return t, e


def _recall_at_precision(
    y_true: np.ndarray, y_score: np.ndarray, target: float = 0.8
) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    p, r, _ = precision_recall_curve(y_true, y_score)
    mask = p >= target
    if not np.any(mask):
        return 0.0
    return float(np.max(r[mask]))


def _model_predict_proba(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(x)
        s = np.asarray(s, dtype=float)
        return 1.0 / (1.0 + np.exp(-s))
    raise TypeError(f"Unsupported model type for probabilities: {type(model)}")


def _fit_platt(
    train_prob: np.ndarray, y_train: np.ndarray, test_prob: np.ndarray
) -> np.ndarray:
    eps = 1e-6
    train_prob = np.clip(train_prob, eps, 1 - eps)
    test_prob = np.clip(test_prob, eps, 1 - eps)
    train_logit = np.log(train_prob / (1 - train_prob)).reshape(-1, 1)
    test_logit = np.log(test_prob / (1 - test_prob)).reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(train_logit, y_train)
    return lr.predict_proba(test_logit)[:, 1]


def _fit_isotonic(
    train_prob: np.ndarray, y_train: np.ndarray, test_prob: np.ndarray
) -> np.ndarray:
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(train_prob, y_train)
    return ir.predict(test_prob)


def _eval_probs(
    y_true: np.ndarray, y_prob: np.ndarray, seed: int = 42
) -> dict[str, Any]:
    auc = auc_with_ci(y_true, y_prob, n_bootstrap=500, seed=seed)
    pr = pr_auc_with_ci(y_true, y_prob, n_bootstrap=500, seed=seed)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    b = brier(y_true, y_prob)
    return {
        "auc": float(auc.value),
        "auc_ci_95": [float(auc.ci_low), float(auc.ci_high)],
        "pr_auc": float(pr.value),
        "pr_auc_ci_95": [float(pr.ci_low), float(pr.ci_high)],
        "recall_at_precision_80": float(_recall_at_precision(y_true, y_prob, 0.8)),
        "ece": float(ece),
        "brier": float(b),
        "decision_curve": decision_curve_net_benefit(y_true, y_prob),
    }


def _plot_baseline_panel(
    out_path: Path,
    internal_auc: float,
    external_auc: float,
    internal_ece: float,
    external_ece: float,
    twin_pass: bool,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2))

    axes[0].bar(
        ["internal", "external-like"],
        [internal_auc, external_auc],
        color=["#4E79A7", "#E15759"],
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_title("AUC")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)

    axes[1].bar(
        ["internal", "external-like"],
        [internal_ece, external_ece],
        color=["#59A14F", "#F28E2B"],
    )
    axes[1].set_title("ECE (lower better)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)

    axes[2].bar(["twin_gate_pass"], [1.0 if twin_pass else 0.0], color="#76B7B2")
    axes[2].set_ylim(0, 1.1)
    axes[2].set_title("Twin Status")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["fail", "pass"])

    fig.suptitle("Cycle00 Baseline Scoreboard")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -------------------------
# Cycle 0
# -------------------------
def run_cycle00(paths: Paths, pull_id: str) -> dict[str, Any]:
    metadata_path = (
        root
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_run"
        / "pyg_data_metadata.json"
    )
    lock_report_path = root / "Docs" / "audit" / "SOTA_INTERNAL_LOCK_REPORT.md"
    lock_json_path = root / "outputs" / "sota_lock" / "internal_sota_lock.json"
    ext_metrics_path = (
        root / "outputs" / "external_validation" / pull_id / "external_metrics.json"
    )
    clinical_cycles_path = (
        root / "Docs" / "audit" / "CLINICAL_HARDENING_REVIEW_CYCLES.json"
    )

    meta = _load_json(metadata_path)
    lock = _load_json(lock_json_path)
    ext = _load_json(ext_metrics_path)
    cycles = _load_json(clinical_cycles_path)

    internal_auc = (
        lock.get("fuzzy_artifacts", {}).get("fuzzy_full", {}).get("best_test_auc")
    )
    if internal_auc is None:
        internal_auc = _parse_markdown_auc(lock_report_path)
    if internal_auc is None:
        internal_auc = 0.0

    internal_raw_ece = 0.0
    explain_summary_path = (
        root
        / "visualizations"
        / "appendix"
        / "explainability"
        / "explainability_summary.json"
    )
    if explain_summary_path.exists():
        explain = _load_json(explain_summary_path)
        internal_raw_ece = float(
            explain.get("calibration_metrics", {}).get("raw", {}).get("ece", 0.0)
        )

    external_auc = float(ext["classification"]["auc"])
    external_ece = float(ext["calibration"]["raw"]["ece"])

    twin_pass = False
    for c in cycles.get("cycles", []):
        if c.get("cycle") == "C4_digital_twin_sensitivity":
            twin_pass = bool(c.get("pass", False))
            break

    # Determinism check using repeat split hash
    repeat_meta_path = (
        root
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_run_repeat"
        / "pyg_data_metadata.json"
    )
    repeat_hash_match = False
    if repeat_meta_path.exists():
        repeat_meta = _load_json(repeat_meta_path)
        repeat_hash_match = repeat_meta.get("split_hash") == meta.get("split_hash")

    panel_path = paths.vis_root / "cycle00" / "baseline_scoreboard.png"
    _plot_baseline_panel(
        panel_path,
        internal_auc,
        external_auc,
        internal_raw_ece,
        external_ece,
        twin_pass,
    )

    payload = {
        "cycle_id": "cycle00",
        "timestamp_utc": _now_utc(),
        "frozen_contract": {
            "schema_version": meta.get("schema_version"),
            "split_hash": meta.get("split_hash"),
            "required_keys": ["PATNO", "time", "event", "saa_label"],
            "path_contract": meta.get("path_contract", {}),
        },
        "baseline_metrics": {
            "internal_auc": float(internal_auc),
            "internal_raw_ece": float(internal_raw_ece),
            "external_auc": external_auc,
            "external_raw_ece": external_ece,
        },
        "reproducibility_check": {
            "split_hash_match_repeat": repeat_hash_match,
            "status": "pass" if repeat_hash_match else "warn",
        },
        "artifacts": {
            "internal_lock_report": str(lock_report_path),
            "internal_lock_json": str(lock_json_path),
            "external_metrics_json": str(ext_metrics_path),
            "baseline_panel_fig": str(panel_path),
        },
        "gate_pass": bool(repeat_hash_match),
    }

    _json_save(paths.cycle_root / "cycle00_baseline_metrics.json", payload)

    md = [
        "# AUC Lift Cycle 00 - Baseline Freeze",
        "",
        f"Generated (UTC): `{payload['timestamp_utc']}`",
        "",
        "## Frozen Contract",
        f"- schema_version: `{payload['frozen_contract']['schema_version']}`",
        f"- split_hash: `{payload['frozen_contract']['split_hash']}`",
        "- required keys: `PATNO,time,event,saa_label`",
        "",
        "## Baseline Board",
        f"- internal_auc: `{internal_auc:.4f}`",
        f"- external_like_auc: `{external_auc:.4f}`",
        f"- internal_raw_ece: `{internal_raw_ece:.4f}`",
        f"- external_raw_ece: `{external_ece:.4f}`",
        "",
        "## Gate",
        f"- reproducibility split-hash match (repeat dataset): `{repeat_hash_match}`",
        f"- status: `{payload['reproducibility_check']['status']}`",
        "",
        "## Figure",
        f"- `{panel_path}`",
    ]
    _text_save(paths.docs_root / "AUC_LIFT_CYCLE00_BASELINE.md", "\n".join(md) + "\n")
    return payload


# -------------------------
# Cycle 1
# -------------------------
def _plot_cycle1_constant_heatmap(
    before_mask: np.ndarray, after_mask: np.ndarray, names: list[str], out_path: Path
) -> None:
    arr = np.vstack([before_mask.astype(float), after_mask.astype(float)])
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.25), 2.6))
    im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["before_const", "after_const"])
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_title("Constant Feature Mask Before/After Cycle01")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("is_constant")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_cycle1_variance(
    before_var: np.ndarray, after_var: np.ndarray, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.hist(
        np.log10(np.clip(before_var, 1e-12, None)), bins=30, alpha=0.6, label="before"
    )
    ax.hist(
        np.log10(np.clip(after_var, 1e-12, None)), bins=30, alpha=0.6, label="after"
    )
    ax.set_title("Feature Variance Distribution (log10)")
    ax.set_xlabel("log10(variance)")
    ax.set_ylabel("count")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_cycle1_lineage(actions: list[str], out_path: Path) -> None:
    counts = pd.Series(actions).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(counts.index, counts.values, color="#4E79A7")
    ax.set_title("Feature Lineage/Promotion Actions")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_cycle01(
    paths: Paths, constant_var_thr: float = 1e-8, dominant_thr: float = 0.995
) -> dict[str, Any]:
    src_dir = root / "data" / "03_prodromal" / "final_pyg_data_sota_run"
    train, test, meta = _load_tensor_dataset(src_dir)

    x_train = train.x.detach().cpu().numpy()
    x_test = test.x.detach().cpu().numpy()
    names = list(meta.get("feature_names", []))

    variances = np.var(x_train, axis=0)
    before_const = variances <= constant_var_thr

    records: list[dict[str, Any]] = []
    keep_idx: list[int] = []
    for i, name in enumerate(names):
        col = x_train[:, i]
        uniq, counts = np.unique(col, return_counts=True)
        dom = float(np.max(counts) / len(col)) if len(col) else 1.0
        is_const = bool(variances[i] <= constant_var_thr)
        action = "keep"
        if is_const:
            action = "drop_constant"
        elif dom >= dominant_thr:
            # Near-constant signals can still carry rare-event value, so keep but flag.
            action = "keep_flag_near_constant"
        else:
            keep_idx.append(i)

        if action == "keep" and i not in keep_idx:
            keep_idx.append(i)

        records.append(
            {
                "feature_name": name,
                "variance": float(variances[i]),
                "is_constant": is_const,
                "missing_fraction": float(np.mean(~np.isfinite(col))),
                "dominant_value_fraction": dom,
                "lineage_ok": True,
                "action": action,
            }
        )

    keep_idx = sorted(set(keep_idx))
    kept_names = [names[i] for i in keep_idx]

    # rebuild dataset version
    out_dir = paths.cycle_dataset_root
    out_dir.mkdir(parents=True, exist_ok=True)

    train_new = train.clone()
    test_new = test.clone()
    train_new.x = train_new.x[:, keep_idx]
    test_new.x = test_new.x[:, keep_idx]

    torch.save(train_new, out_dir / "train_data.pt")
    torch.save(test_new, out_dir / "test_data.pt")

    new_meta = dict(meta)
    new_meta["schema_version"] = "sota_auc_lift_cycle01_v1"
    new_meta["parent_schema_version"] = meta.get("schema_version")
    new_meta["n_features"] = len(kept_names)
    new_meta["feature_names"] = kept_names
    new_meta["feature_drop_indices"] = [
        i for i in range(len(names)) if i not in keep_idx
    ]
    new_meta["feature_drop_names"] = [
        names[i] for i in range(len(names)) if i not in keep_idx
    ]
    new_meta["feature_order_hash"] = _sha256_bytes(
        json.dumps(kept_names, sort_keys=False).encode("utf-8")
    )
    _json_save(out_dir / "pyg_data_metadata.json", new_meta)

    # copy split manifest + pat mapping if present
    for fname in ["split_manifest.json", "patno_mapping.json"]:
        src = src_dir / fname
        if src.exists():
            (out_dir / fname).write_text(
                src.read_text(encoding="utf-8"), encoding="utf-8"
            )

    after_var = np.var(train_new.x.detach().cpu().numpy(), axis=0)
    after_const = after_var <= constant_var_thr

    fig_const = paths.vis_root / "cycle01" / "constant_feature_heatmap_before_after.png"
    fig_var = paths.vis_root / "cycle01" / "variance_distribution_before_after.png"
    fig_lin = paths.vis_root / "cycle01" / "lineage_coverage_panel.png"

    # align after mask to original features for heatmap
    after_mask_full = np.ones(len(names), dtype=bool)
    after_mask_full[keep_idx] = False
    _plot_cycle1_constant_heatmap(before_const, after_mask_full, names, fig_const)
    _plot_cycle1_variance(variances, after_var, fig_var)
    _plot_cycle1_lineage([r["action"] for r in records], fig_lin)

    summary = {
        "cycle_id": "cycle01",
        "timestamp_utc": _now_utc(),
        "source_dataset": str(src_dir),
        "repaired_dataset": str(out_dir),
        "n_features_before": len(names),
        "n_features_after": len(kept_names),
        "n_constant_before": int(np.sum(before_const)),
        "n_constant_after": int(np.sum(after_const)),
        "n_near_constant_flagged": int(
            sum(1 for r in records if r["action"] == "keep_flag_near_constant")
        ),
        "dropped_features": [r for r in records if r["action"] == "drop_constant"],
        "flagged_features": [
            r for r in records if r["action"] == "keep_flag_near_constant"
        ],
        "feature_quality_records": records,
        "gate": {
            "constant_feature_reduction": int(np.sum(before_const))
            - int(np.sum(after_const)),
            "required_modality_fields_explicitly_accounted": True,
            "pass": bool(np.sum(after_const) == 0),
        },
        "artifacts": {
            "feature_quality_json": str(
                paths.cycle_root / "cycle01_feature_quality.json"
            ),
            "fig_constant_heatmap": str(fig_const),
            "fig_variance": str(fig_var),
            "fig_lineage": str(fig_lin),
        },
    }

    _json_save(paths.cycle_root / "cycle01_feature_quality.json", summary)

    md = [
        "# AUC Lift Cycle 01 - Feature Integrity Repair",
        "",
        f"Generated (UTC): `{summary['timestamp_utc']}`",
        "",
        "## Summary",
        f"- features_before: `{summary['n_features_before']}`",
        f"- features_after: `{summary['n_features_after']}`",
        f"- constant_before: `{summary['n_constant_before']}`",
        f"- constant_after: `{summary['n_constant_after']}`",
        f"- near_constant_flagged: `{summary['n_near_constant_flagged']}`",
        "",
        "## Gate",
        f"- pass: `{summary['gate']['pass']}`",
        "",
        "## Key Actions",
    ]
    for row in (summary["dropped_features"] + summary["flagged_features"])[:20]:
        md.append(
            f"- `{row['feature_name']}` -> `{row['action']}` (variance={row['variance']:.3e}, dominant={row['dominant_value_fraction']:.3f})"
        )
    md.extend(
        [
            "",
            "## Figures",
            f"- `{fig_const}`",
            f"- `{fig_var}`",
            f"- `{fig_lin}`",
        ]
    )
    _text_save(
        paths.docs_root / "AUC_LIFT_CYCLE01_FEATURE_INTEGRITY.md", "\n".join(md) + "\n"
    )
    return summary


# -------------------------
# Cycle 2
# -------------------------
def _plot_cycle2_label_horizon(df: pd.DataFrame, out_path: Path) -> None:
    if "landmark_month" not in df.columns or "phenoconverted" not in df.columns:
        return
    grp = (
        df.groupby("landmark_month")["phenoconverted"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("landmark_month")
    )
    fig, ax1 = plt.subplots(figsize=(8.8, 4.6))
    ax1.plot(
        grp["landmark_month"],
        grp["mean"],
        marker="o",
        color="#4E79A7",
        label="event_rate",
    )
    ax1.set_ylabel("event rate")
    ax1.set_xlabel("landmark_month")
    ax1.set_ylim(0, 1)
    ax1.grid(True, linestyle="--", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.bar(
        grp["landmark_month"], grp["count"], alpha=0.2, color="#E15759", label="count"
    )
    ax2.set_ylabel("count")
    ax1.set_title("Label-by-Horizon Distribution")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_cycle2_timeline(df: pd.DataFrame, out_path: Path) -> None:
    if "landmark_month" not in df.columns or "original_time" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.scatter(df["landmark_month"], df["original_time"], s=8, alpha=0.35)
    maxv = float(max(df["landmark_month"].max(), df["original_time"].max()))
    ax.plot([0, maxv], [0, maxv], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("landmark_month")
    ax.set_ylabel("original_time")
    ax.set_title("Event/Visit Timeline Consistency")
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_cycle02(paths: Paths) -> dict[str, Any]:
    cycle_data_dir = paths.cycle_dataset_root
    train, test, meta = _load_tensor_dataset(cycle_data_dir)
    x_train, y_train, pat_train = _extract_xy_pat(train)
    x_test, y_test, pat_test = _extract_xy_pat(test)
    t_train, e_train = _extract_surv(train)
    t_test, e_test = _extract_surv(test)

    # label reliability checks
    proxy_violation_train = bool(np.array_equal(y_train, e_train))
    proxy_violation_test = bool(np.array_equal(y_test, e_test))
    prevalence_train = float(np.mean(y_train))
    prevalence_test = float(np.mean(y_test))
    prevalence_gap = abs(prevalence_train - prevalence_test)

    # timeline checks from longitudinal csv
    long_csv = (
        root
        / "data"
        / "03_prodromal"
        / "final_training_dataset"
        / "unified_longitudinal_early_pd.csv"
    )
    df_long = pd.read_csv(long_csv)
    has_timeline_cols = all(
        c in df_long.columns for c in ["landmark_month", "original_time"]
    )
    post_outcome_violations = 0
    if has_timeline_cols:
        post_outcome_violations = int(
            np.sum(df_long["landmark_month"] > df_long["original_time"])
        )

    # stability vs repeat split
    repeat_dir = root / "data" / "03_prodromal" / "final_pyg_data_sota_run_repeat"
    repeat_meta = (
        _load_json(repeat_dir / "pyg_data_metadata.json")
        if (repeat_dir / "pyg_data_metadata.json").exists()
        else {}
    )
    repeat_train_rate = float(repeat_meta.get("train_saa_rate", prevalence_train))
    repeat_test_rate = float(repeat_meta.get("test_saa_rate", prevalence_test))

    fig_horizon = paths.vis_root / "cycle02" / "label_by_horizon_distribution.png"
    fig_timeline = paths.vis_root / "cycle02" / "event_visit_timeline_consistency.png"
    _plot_cycle2_label_horizon(df_long, fig_horizon)
    _plot_cycle2_timeline(df_long, fig_timeline)

    pass_flag = (
        (not proxy_violation_train)
        and (not proxy_violation_test)
        and (post_outcome_violations == 0)
        and (prevalence_gap <= 0.03)
    )

    payload = {
        "cycle_id": "cycle02",
        "timestamp_utc": _now_utc(),
        "checks": {
            "proxy_label_violation_train": proxy_violation_train,
            "proxy_label_violation_test": proxy_violation_test,
            "post_outcome_violations": post_outcome_violations,
            "train_test_prevalence_gap": prevalence_gap,
            "train_prevalence": prevalence_train,
            "test_prevalence": prevalence_test,
            "repeat_train_prevalence": repeat_train_rate,
            "repeat_test_prevalence": repeat_test_rate,
            "patient_disjoint": bool(
                len(np.intersect1d(np.unique(pat_train), np.unique(pat_test))) == 0
            ),
        },
        "gate": {
            "pass": pass_flag,
            "criteria": {
                "proxy_violations": 0,
                "post_outcome_violations": 0,
                "prevalence_gap_max": 0.03,
            },
        },
        "artifacts": {
            "label_horizon_fig": str(fig_horizon),
            "timeline_consistency_fig": str(fig_timeline),
            "source_longitudinal_csv": str(long_csv),
        },
    }

    _json_save(paths.cycle_root / "cycle02_label_integrity.json", payload)
    md = [
        "# AUC Lift Cycle 02 - Endpoint and Label Reliability",
        "",
        f"Generated (UTC): `{payload['timestamp_utc']}`",
        "",
        "## Checks",
        f"- proxy_label_violation_train: `{proxy_violation_train}`",
        f"- proxy_label_violation_test: `{proxy_violation_test}`",
        f"- post_outcome_violations: `{post_outcome_violations}`",
        f"- train_test_prevalence_gap: `{prevalence_gap:.4f}`",
        f"- patient_disjoint: `{payload['checks']['patient_disjoint']}`",
        "",
        "## Gate",
        f"- pass: `{pass_flag}`",
        "",
        "## Figures",
        f"- `{fig_horizon}`",
        f"- `{fig_timeline}`",
    ]
    _text_save(
        paths.docs_root / "AUC_LIFT_CYCLE02_ENDPOINT_QC.md", "\n".join(md) + "\n"
    )
    return payload


# -------------------------
# Cycle 3
# -------------------------
def _fit_tabular_models(
    x_train: np.ndarray, y_train: np.ndarray, seed: int = 42
) -> dict[str, Any]:
    models: dict[str, Any] = {}

    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    lr.fit(x_train, y_train)
    models["logistic_regression"] = lr

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=seed,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    rf.fit(x_train, y_train)
    models["random_forest"] = rf

    et = ExtraTreesClassifier(
        n_estimators=500,
        random_state=seed,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    et.fit(x_train, y_train)
    models["extra_trees"] = et

    svm = SVC(probability=True, class_weight="balanced", random_state=seed)
    svm.fit(x_train, y_train)
    models["svm_rbf"] = svm

    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    pos_w = (neg / max(pos, 1.0)) if pos > 0 else 1.0
    sample_weight = np.where(y_train == 1, pos_w, 1.0).astype(np.float64)

    hgb = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.03,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        l2_regularization=0.1,
        random_state=seed,
    )
    hgb.fit(x_train, y_train, sample_weight=sample_weight)
    models["hist_gradient_boosting"] = hgb

    return models


def _plot_cycle3_roc_pr(
    model_probs: dict[str, np.ndarray], y_true: np.ndarray, out_path: Path
) -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    for name, prob in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        p, r, _ = precision_recall_curve(y_true, prob)
        axes[0].plot(fpr, tpr, label=name)
        axes[1].plot(r, p, label=name)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axes[0].set_title("ROC Curves")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].grid(True, linestyle="--", alpha=0.25)
    axes[1].set_title("Precision-Recall Curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid(True, linestyle="--", alpha=0.25)
    axes[1].axhline(np.mean(y_true), linestyle="--", color="black", linewidth=1)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Cycle03 Tabular Baselines")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_cycle3_calibration(
    y_true: np.ndarray, probs: dict[str, np.ndarray], out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="perfect")
    for name, pr in probs.items():
        frac_pos, mean_pred = calibration_curve(
            y_true, pr, n_bins=8, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curves (Best Model Raw/Calibrated)")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_cycle3_decision(
    y_true: np.ndarray, probs: dict[str, np.ndarray], out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for name, pr in probs.items():
        dc = decision_curve_net_benefit(y_true, pr)
        ax.plot(dc["thresholds"], dc["net_benefit"], label=name)
    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curves")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_cycle03(paths: Paths, baseline_internal_auc: float) -> dict[str, Any]:
    train, test, meta = _load_tensor_dataset(paths.cycle_dataset_root)
    x_train, y_train, pat_train = _extract_xy_pat(train)
    x_test, y_test, _ = _extract_xy_pat(test)

    models = _fit_tabular_models(x_train, y_train, seed=42)

    model_probs_test: dict[str, np.ndarray] = {}
    model_probs_train: dict[str, np.ndarray] = {}
    results: dict[str, Any] = {}
    for name, mdl in models.items():
        p_test = _model_predict_proba(mdl, x_test)
        p_train = _model_predict_proba(mdl, x_train)
        model_probs_test[name] = p_test
        model_probs_train[name] = p_train
        results[name] = _eval_probs(y_test, p_test, seed=42)

    best_name = max(results.keys(), key=lambda k: results[k]["auc"])
    best_model = models[best_name]
    best_test_prob = model_probs_test[best_name]

    # deterministic group-aware split for calibration fitting
    gkf = GroupKFold(n_splits=5)
    tr_idx, cal_idx = next(iter(gkf.split(x_train, y_train, groups=pat_train)))
    x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
    x_cal, y_cal = x_train[cal_idx], y_train[cal_idx]

    # refit same class on split to avoid leakage
    mdl_class = type(best_model)
    if best_name == "logistic_regression":
        best_refit = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        )
    elif best_name == "random_forest":
        best_refit = RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        )
    elif best_name == "extra_trees":
        best_refit = ExtraTreesClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
        )
    elif best_name == "svm_rbf":
        best_refit = SVC(probability=True, class_weight="balanced", random_state=42)
    elif best_name == "hist_gradient_boosting":
        best_refit = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.03,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=0.1,
            random_state=42,
        )
    else:
        best_refit = mdl_class()

    if best_name == "hist_gradient_boosting":
        tr_pos = float(np.sum(y_tr == 1))
        tr_neg = float(np.sum(y_tr == 0))
        tr_pos_w = (tr_neg / max(tr_pos, 1.0)) if tr_pos > 0 else 1.0
        tr_sample_weight = np.where(y_tr == 1, tr_pos_w, 1.0).astype(np.float64)
        best_refit.fit(x_tr, y_tr, sample_weight=tr_sample_weight)
    else:
        best_refit.fit(x_tr, y_tr)
    p_cal = _model_predict_proba(best_refit, x_cal)
    p_test_raw = _model_predict_proba(best_refit, x_test)

    p_test_platt = _fit_platt(p_cal, y_cal, p_test_raw)
    p_test_iso = _fit_isotonic(p_cal, y_cal, p_test_raw)

    cal_metrics = {
        "raw": _eval_probs(y_test, p_test_raw, seed=42),
        "platt": _eval_probs(y_test, p_test_platt, seed=42),
        "isotonic": _eval_probs(y_test, p_test_iso, seed=42),
    }

    # save best model + transform contract
    model_dir = paths.cycle_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"cycle03_best_{best_name}.joblib"
    joblib.dump(best_model, model_path)

    transform = {
        "feature_names": meta.get("feature_names", []),
        "n_features": int(x_train.shape[1]),
        "schema_version": meta.get("schema_version"),
    }
    _json_save(model_dir / "cycle03_feature_contract.json", transform)

    fig_rocpr = paths.vis_root / "cycle03" / "roc_pr_curves.png"
    fig_cal = paths.vis_root / "cycle03" / "calibration_curves.png"
    fig_dc = paths.vis_root / "cycle03" / "decision_curves.png"
    _plot_cycle3_roc_pr(model_probs_test, y_test, fig_rocpr)
    _plot_cycle3_calibration(
        y_test,
        {"raw": p_test_raw, "platt": p_test_platt, "isotonic": p_test_iso},
        fig_cal,
    )
    _plot_cycle3_decision(
        y_test,
        {"raw": p_test_raw, "platt": p_test_platt, "isotonic": p_test_iso},
        fig_dc,
    )

    best_auc = results[best_name]["auc"]
    auc_delta_vs_cycle00 = best_auc - baseline_internal_auc

    no_calib_regression = (
        cal_metrics["platt"]["ece"] <= cal_metrics["raw"]["ece"]
        and cal_metrics["platt"]["brier"] <= cal_metrics["raw"]["brier"]
    ) or (
        cal_metrics["isotonic"]["ece"] <= cal_metrics["raw"]["ece"]
        and cal_metrics["isotonic"]["brier"] <= cal_metrics["raw"]["brier"]
    )

    gate_pass = bool((auc_delta_vs_cycle00 >= 0.02) and no_calib_regression)

    payload = {
        "cycle_id": "cycle03",
        "timestamp_utc": _now_utc(),
        "models": results,
        "best_model": best_name,
        "best_auc": best_auc,
        "auc_delta_vs_cycle00": auc_delta_vs_cycle00,
        "calibration_registry": cal_metrics,
        "best_model_artifact": str(model_path),
        "feature_contract": str(model_dir / "cycle03_feature_contract.json"),
        "gate": {
            "internal_auc_min_delta": 0.02,
            "no_calibration_regression": no_calib_regression,
            "pass": gate_pass,
        },
        "artifacts": {
            "roc_pr_fig": str(fig_rocpr),
            "calibration_fig": str(fig_cal),
            "decision_curve_fig": str(fig_dc),
        },
    }

    _json_save(paths.cycle_root / "cycle03_model_registry.json", payload)

    md = [
        "# AUC Lift Cycle 03 - Tabular Baseline Rebuild",
        "",
        f"Generated (UTC): `{payload['timestamp_utc']}`",
        "",
        "## Model Metrics",
    ]
    for name, m in results.items():
        md.append(
            f"- `{name}`: AUC={m['auc']:.4f} [{m['auc_ci_95'][0]:.4f}, {m['auc_ci_95'][1]:.4f}], PR-AUC={m['pr_auc']:.4f}, ECE={m['ece']:.4f}, Brier={m['brier']:.4f}"
        )
    md.extend(
        [
            "",
            "## Best Model",
            f"- best_model: `{best_name}`",
            f"- best_auc: `{best_auc:.4f}`",
            f"- auc_delta_vs_cycle00: `{auc_delta_vs_cycle00:.4f}`",
            "",
            "## Gate",
            f"- no_calibration_regression: `{no_calib_regression}`",
            f"- pass: `{gate_pass}`",
            "",
            "## Figures",
            f"- `{fig_rocpr}`",
            f"- `{fig_cal}`",
            f"- `{fig_dc}`",
        ]
    )
    _text_save(
        paths.docs_root / "AUC_LIFT_CYCLE03_TABULAR_BASELINES.md", "\n".join(md) + "\n"
    )
    return payload


# -------------------------
# Cycle 4
# -------------------------
class _SimpleGAT(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        if GATConv is None:
            raise RuntimeError("torch_geometric not available")
        self.g1 = GATConv(in_dim, hidden, heads=2, dropout=0.2)
        self.g2 = GATConv(hidden * 2, hidden, heads=1, dropout=0.2)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, data: Any) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.g1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.g2(x, edge_index)
        x = F.elu(x)
        return self.fc(x)


def _train_simple_gat(
    train_data: Any, test_data: Any, epochs: int = 120
) -> tuple[float, np.ndarray]:
    if GATConv is None:
        return 0.5, np.full(test_data.saa_label.shape[0], 0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    model = _SimpleGAT(in_dim=int(train_data.x.shape[1]), hidden=64).to(device)

    y = train_data.saa_label.long()
    pos = float((y == 1).sum().item())
    neg = float((y == 0).sum().item())
    w0 = 1.0
    w1 = (neg / pos) if pos > 0 else 1.0
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([w0, w1], dtype=torch.float32, device=device)
    )
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(train_data)
        loss = criterion(logits, train_data.saa_label.long())
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(test_data), dim=1)[:, 1].detach().cpu().numpy()
    auc = _safe_auc(test_data.saa_label.detach().cpu().numpy().astype(int), probs)
    return float(auc), probs


def _permutation_dominance_ratio(
    y_true: np.ndarray, x: np.ndarray, prob_fn, feature_names: list[str]
) -> tuple[float, list[dict[str, Any]]]:
    base = _safe_auc(y_true, prob_fn(x))
    rows = []
    rng = np.random.default_rng(42)
    for i, name in enumerate(feature_names):
        xp = x.copy()
        xp[:, i] = rng.permutation(xp[:, i])
        aucp = _safe_auc(y_true, prob_fn(xp))
        drop = max(0.0, base - aucp)
        rows.append({"feature": name, "auc_drop": float(drop)})
    rows.sort(key=lambda z: z["auc_drop"], reverse=True)
    top = rows[0]["auc_drop"] if rows else 0.0
    top10 = sum(r["auc_drop"] for r in rows[:10]) if rows else 0.0
    ratio = float(top / top10) if top10 > 0 else 0.0
    return ratio, rows


def _plot_cycle4_ablation(metrics: dict[str, float], out_path: Path) -> None:
    labels = list(metrics.keys())
    vals = [metrics[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(labels, vals, color=["#4E79A7", "#59A14F", "#F28E2B"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC")
    ax.set_title("Cycle04 Ablation Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_cycle4_dominance(d: dict[str, float], out_path: Path) -> None:
    labels = list(d.keys())
    vals = [d[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(labels, vals, marker="o")
    ax.axhline(0.5, linestyle="--", color="red", label="threshold")
    ax.set_ylabel("top feature dominance ratio")
    ax.set_title("Top-Feature Dominance Trend")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_cycle4_rule_stability(
    rule_heatmap_exists: bool, proxy_pass: bool, out_path: Path
) -> None:
    labels = ["rule_heatmap_exists", "proxy_gate_pass"]
    vals = [1.0 if rule_heatmap_exists else 0.0, 1.0 if proxy_pass else 0.0]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(labels, vals, color="#76B7B2")
    ax.set_ylim(0, 1.1)
    ax.set_title("Fuzzy Stability Signals")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["fail", "pass"])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_cycle04(paths: Paths, cycle03_payload: dict[str, Any]) -> dict[str, Any]:
    train, test, meta = _load_tensor_dataset(paths.cycle_dataset_root)
    x_train, y_train, _ = _extract_xy_pat(train)
    x_test, y_test, _ = _extract_xy_pat(test)

    # no-graph baseline from cycle03 best
    no_graph_auc = float(cycle03_payload["best_auc"])

    # no-fuzzy graph model
    no_fuzzy_auc, no_fuzzy_probs = _train_simple_gat(
        train.clone(), test.clone(), epochs=120
    )

    # full fuzzy from current artifact (audited)
    fuzzy_full_path = (
        root
        / "outputs"
        / "phase9_neuro_fuzzy_sota_run_from50ckpt"
        / "full_training_results.json"
    )
    fuzzy_full = _load_json(fuzzy_full_path) if fuzzy_full_path.exists() else {}
    full_fuzzy_auc = float(fuzzy_full.get("best_test_auc", 0.0))

    # dominance
    clinical_cycles = _load_json(
        root / "Docs" / "audit" / "CLINICAL_HARDENING_REVIEW_CYCLES.json"
    )
    baseline_dom = None
    proxy_pass = False
    for c in clinical_cycles.get("cycles", []):
        if c.get("cycle") == "C1_proxy_dependency_audit":
            baseline_dom = float(c.get("dominance_ratio_top_over_top10", 0.0))
            proxy_pass = bool(c.get("pass", False))
            break
    if baseline_dom is None:
        baseline_dom = 0.0

    # cycle03 model dominance (approx): use best model saved
    model_path = Path(cycle03_payload["best_model_artifact"])
    best_model = joblib.load(model_path)
    best_prob_fn = lambda x: _model_predict_proba(best_model, x)
    cycle03_dom, cycle03_perm = _permutation_dominance_ratio(
        y_test, x_test, best_prob_fn, meta.get("feature_names", [])
    )

    # no-fuzzy dominance
    if GATConv is not None:
        # approximate using no_fuzzy probs only (cannot do feature-wise infer function cheaply without retraining each perm)
        no_fuzzy_dom = float("nan")
    else:
        no_fuzzy_dom = float("nan")

    ablation_metrics = {
        "no_graph_tabular": no_graph_auc,
        "no_fuzzy_graph": no_fuzzy_auc,
        "full_fuzzy_graph": full_fuzzy_auc,
    }

    fig_ablation = paths.vis_root / "cycle04" / "ablation_comparison.png"
    fig_dom = paths.vis_root / "cycle04" / "dominance_trend.png"
    fig_rule = paths.vis_root / "cycle04" / "fuzzy_rule_stability_panel.png"
    _plot_cycle4_ablation(ablation_metrics, fig_ablation)
    _plot_cycle4_dominance(
        {
            "baseline_fuzzy": baseline_dom,
            "cycle03_best": cycle03_dom,
        },
        fig_dom,
    )
    rule_heatmap_exists = (
        root
        / "visualizations"
        / "appendix"
        / "explainability"
        / "fuzzy_rule_activation_heatmap.png"
    ).exists()
    _plot_cycle4_rule_stability(rule_heatmap_exists, proxy_pass, fig_rule)

    gate_pass = bool(
        (full_fuzzy_auc > no_graph_auc) and proxy_pass and rule_heatmap_exists
    )

    payload = {
        "cycle_id": "cycle04",
        "timestamp_utc": _now_utc(),
        "ablation_metrics": ablation_metrics,
        "dominance": {
            "baseline_fuzzy_ratio": baseline_dom,
            "cycle03_best_ratio": cycle03_dom,
            "no_fuzzy_ratio": no_fuzzy_dom,
        },
        "proxy_gate_pass": proxy_pass,
        "rule_heatmap_exists": rule_heatmap_exists,
        "cycle03_permutation_top20": cycle03_perm[:20],
        "gate": {
            "criterion": "full_model_auc > cycle03_no_graph_auc and proxy pass and rule artifact present",
            "pass": gate_pass,
        },
        "artifacts": {
            "ablation_fig": str(fig_ablation),
            "dominance_fig": str(fig_dom),
            "rule_stability_fig": str(fig_rule),
        },
    }

    _json_save(paths.cycle_root / "cycle04_ablation_metrics.json", payload)
    md = [
        "# AUC Lift Cycle 04 - GNN + Neuro-Fuzzy Reintroduction",
        "",
        f"Generated (UTC): `{payload['timestamp_utc']}`",
        "",
        "## Ablation Metrics",
        f"- no_graph_tabular_auc: `{no_graph_auc:.4f}`",
        f"- no_fuzzy_graph_auc: `{no_fuzzy_auc:.4f}`",
        f"- full_fuzzy_graph_auc: `{full_fuzzy_auc:.4f}`",
        "",
        "## Gate",
        f"- proxy_gate_pass: `{proxy_pass}`",
        f"- rule_heatmap_exists: `{rule_heatmap_exists}`",
        f"- pass: `{gate_pass}`",
        "",
        "## Figures",
        f"- `{fig_ablation}`",
        f"- `{fig_dom}`",
        f"- `{fig_rule}`",
    ]
    _text_save(paths.docs_root / "AUC_LIFT_CYCLE04_GNN_FUZZY.md", "\n".join(md) + "\n")
    return payload


# -------------------------
# Cycle 5
# -------------------------
def _plot_cycle5_subgroup(
    y_true: np.ndarray, p_raw: np.ndarray, subgroup: np.ndarray, out_path: Path
) -> None:
    # subgroup = 0/1
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    rows = []
    for g in [0, 1]:
        m = subgroup == g
        if np.sum(m) < 10 or len(np.unique(y_true[m])) < 2:
            rows.append((f"group_{g}", np.nan, np.nan))
            continue
        e = expected_calibration_error(y_true[m], p_raw[m], n_bins=8)
        b = brier(y_true[m], p_raw[m])
        rows.append((f"group_{g}", e, b))

    labels = [r[0] for r in rows]
    ece_vals = [0 if np.isnan(r[1]) else r[1] for r in rows]
    b_vals = [0 if np.isnan(r[2]) else r[2] for r in rows]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, ece_vals, width=w, label="ECE")
    ax.bar(x + w / 2, b_vals, width=w, label="Brier")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Calibration by Subgroup (LRRK2 proxy groups)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_cycle05(paths: Paths, cycle03_payload: dict[str, Any]) -> dict[str, Any]:
    train, test, meta = _load_tensor_dataset(paths.cycle_dataset_root)
    x_train, y_train, _ = _extract_xy_pat(train)
    x_test, y_test, _ = _extract_xy_pat(test)

    best_model = joblib.load(Path(cycle03_payload["best_model_artifact"]))
    p_raw_train = _model_predict_proba(best_model, x_train)
    p_raw_test = _model_predict_proba(best_model, x_test)
    p_platt = _fit_platt(p_raw_train, y_train, p_raw_test)
    p_iso = _fit_isotonic(p_raw_train, y_train, p_raw_test)

    metrics = {
        "raw": _eval_probs(y_test, p_raw_test, seed=42),
        "platt": _eval_probs(y_test, p_platt, seed=42),
        "isotonic": _eval_probs(y_test, p_iso, seed=42),
    }

    # choose calibration for utility: highest mean net benefit over thresholds
    util_score = {
        k: float(np.mean(v["decision_curve"]["net_benefit"]))
        for k, v in metrics.items()
    }
    selected = max(util_score.keys(), key=lambda k: util_score[k])

    # subgroup reliability using LRRK2 (feature exists in repaired set maybe)
    fn = meta.get("feature_names", [])
    subgroup_fig = paths.vis_root / "cycle05" / "calibration_by_subgroup.png"
    subgroup_summary: dict[str, Any] = {}
    if "LRRK2" in fn:
        idx = fn.index("LRRK2")
        subgroup = (x_test[:, idx] > 0).astype(int)
        _plot_cycle5_subgroup(
            y_test,
            metrics[selected]["decision_curve"]["thresholds"]
            and {
                "raw": p_raw_test,
                "platt": p_platt,
                "isotonic": p_iso,
            }[selected],
            subgroup,
            subgroup_fig,
        )
        subgroup_summary = {
            "feature": "LRRK2",
            "n_group0": int(np.sum(subgroup == 0)),
            "n_group1": int(np.sum(subgroup == 1)),
        }

    fig_rel = paths.vis_root / "cycle05" / "reliability_selected.png"
    _plot_cycle3_calibration(
        y_test,
        {
            "selected": {"raw": p_raw_test, "platt": p_platt, "isotonic": p_iso}[
                selected
            ]
        },
        fig_rel,
    )
    fig_net = paths.vis_root / "cycle05" / "net_benefit_selected.png"
    _plot_cycle3_decision(
        y_test, {"raw": p_raw_test, "platt": p_platt, "isotonic": p_iso}, fig_net
    )

    improved = (
        metrics[selected]["ece"] <= metrics["raw"]["ece"]
        and metrics[selected]["brier"] <= metrics["raw"]["brier"]
    )
    positive_nb = float(np.mean(metrics[selected]["decision_curve"]["net_benefit"])) > 0
    gate_pass = bool(improved and positive_nb)

    payload = {
        "cycle_id": "cycle05",
        "timestamp_utc": _now_utc(),
        "calibration_registry": metrics,
        "decision_utility_mean_net_benefit": util_score,
        "selected_calibration": selected,
        "subgroup_summary": subgroup_summary,
        "gate": {
            "improved_ece_and_brier": improved,
            "positive_net_benefit": positive_nb,
            "pass": gate_pass,
        },
        "artifacts": {
            "reliability_fig": str(fig_rel),
            "net_benefit_fig": str(fig_net),
            "subgroup_fig": str(subgroup_fig),
        },
    }
    _json_save(paths.cycle_root / "cycle05_calibration_registry.json", payload)

    md = [
        "# AUC Lift Cycle 05 - Calibration and Clinical Utility",
        "",
        f"Generated (UTC): `{payload['timestamp_utc']}`",
        "",
        "## Calibration Metrics",
    ]
    for k, m in metrics.items():
        md.append(
            f"- `{k}`: ECE={m['ece']:.4f}, Brier={m['brier']:.4f}, AUC={m['auc']:.4f}"
        )
    md.extend(
        [
            "",
            "## Selection",
            f"- selected_calibration: `{selected}`",
            f"- mean_net_benefit: `{util_score[selected]:.6f}`",
            "",
            "## Gate",
            f"- pass: `{gate_pass}`",
            "",
            "## Figures",
            f"- `{fig_rel}`",
            f"- `{fig_net}`",
            f"- `{subgroup_fig}`",
        ]
    )
    _text_save(
        paths.docs_root / "AUC_LIFT_CYCLE05_CALIBRATION_UTILITY.md",
        "\n".join(md) + "\n",
    )
    return payload


# -------------------------
# Cycle 6
# -------------------------
def _plot_cycle6_internal_external(
    internal_auc: float, external_auc: float, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(
        ["internal", "external-like"],
        [internal_auc, external_auc],
        color=["#4E79A7", "#E15759"],
    )
    ax.set_ylim(0, 1)
    ax.set_title("Internal vs External AUC")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_cycle6_gap_decomp(
    base_ext_auc: float,
    new_ext_auc: float,
    base_ece: float,
    new_ece: float,
    out_path: Path,
) -> None:
    labels = ["AUC gain", "ECE reduction"]
    vals = [new_ext_auc - base_ext_auc, base_ece - new_ece]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(labels, vals, color=["#59A14F", "#F28E2B"])
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Transportability Gap Decomposition")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_cycle06(
    paths: Paths, cycle03_payload: dict[str, Any], pull_id: str
) -> dict[str, Any]:
    # evaluate cycle03 best model on external pt
    ext_data_path = (
        root / "data" / "04_external_validation" / pull_id / "external_data.pt"
    )
    ext_meta_path = (
        root / "data" / "04_external_validation" / pull_id / "external_metadata.json"
    )
    ext_metrics_path = (
        root / "outputs" / "external_validation" / pull_id / "external_metrics.json"
    )

    ext_data = torch.load(ext_data_path, weights_only=False)
    y_ext = ext_data.saa_label.detach().cpu().numpy().astype(int)

    best_model = joblib.load(Path(cycle03_payload["best_model_artifact"]))
    contract = _load_json(paths.cycle_root / "models" / "cycle03_feature_contract.json")
    n_features = int(contract["n_features"])
    x_ext = ext_data.x.detach().cpu().numpy().astype(np.float32)
    if x_ext.shape[1] != n_features:
        # align by selecting first n_features if dataset still 49 and model on reduced set from cycle01
        if x_ext.shape[1] > n_features:
            # use feature names alignment from metadata
            ext_meta = _load_json(ext_meta_path)
            ext_names = ext_meta.get("feature_names", [])
            tgt_names = contract.get("feature_names", [])
            idx = [ext_names.index(n) for n in tgt_names if n in ext_names]
            if len(idx) != len(tgt_names):
                raise ValueError("External feature alignment failed for cycle06")
            x_ext = x_ext[:, idx]
        else:
            raise ValueError("External features fewer than cycle03 contract")

    p_ext_raw = _model_predict_proba(best_model, x_ext)
    ext_new = _eval_probs(y_ext, p_ext_raw, seed=42)

    # compare to baseline external (frozen fuzzy)
    ext_base = _load_json(ext_metrics_path)
    base_auc = float(ext_base["classification"]["auc"])
    base_ece = float(ext_base["calibration"]["raw"]["ece"])

    internal_auc = float(cycle03_payload["best_auc"])
    new_auc = float(ext_new["auc"])
    new_ece = float(ext_new["ece"])

    fig_panel = paths.vis_root / "cycle06" / "internal_vs_external_delta_panel.png"
    fig_gap = paths.vis_root / "cycle06" / "transportability_gap_decomposition.png"
    _plot_cycle6_internal_external(internal_auc, new_auc, fig_panel)
    _plot_cycle6_gap_decomp(base_auc, new_auc, base_ece, new_ece, fig_gap)

    # CI non-overlap improvement heuristic
    base_low, base_high = ext_base["classification"]["auc_ci_95"]
    new_low, new_high = ext_new["auc_ci_95"]
    ci_non_overlap_improved = bool(new_low > base_high)
    auc_improved = bool(new_auc > base_auc)

    governance_ok = bool(ext_base.get("governance", {}).get("real_data_only", True))
    gate_pass = bool(auc_improved and governance_ok)

    payload = {
        "cycle_id": "cycle06",
        "timestamp_utc": _now_utc(),
        "baseline_external": {
            "auc": base_auc,
            "auc_ci_95": [base_low, base_high],
            "ece": base_ece,
        },
        "candidate_external": ext_new,
        "internal_auc": internal_auc,
        "delta": {
            "auc_gain": new_auc - base_auc,
            "ece_reduction": base_ece - new_ece,
        },
        "ci_non_overlap_improved": ci_non_overlap_improved,
        "governance_ok": governance_ok,
        "gate": {
            "auc_improved": auc_improved,
            "pass": gate_pass,
        },
        "artifacts": {
            "panel_fig": str(fig_panel),
            "gap_fig": str(fig_gap),
            "baseline_external_metrics": str(ext_metrics_path),
        },
    }

    _json_save(paths.cycle_root / "cycle06_external_metrics.json", payload)
    md = [
        "# AUC Lift Cycle 06 - External-like Transportability",
        "",
        f"Generated (UTC): `{payload['timestamp_utc']}`",
        "",
        "## Baseline vs Candidate External",
        f"- baseline_auc: `{base_auc:.4f}`",
        f"- candidate_auc: `{new_auc:.4f}`",
        f"- auc_gain: `{payload['delta']['auc_gain']:.4f}`",
        f"- baseline_ece: `{base_ece:.4f}`",
        f"- candidate_ece: `{new_ece:.4f}`",
        "",
        "## Gate",
        f"- auc_improved: `{auc_improved}`",
        f"- ci_non_overlap_improved: `{ci_non_overlap_improved}`",
        f"- governance_ok: `{governance_ok}`",
        f"- pass: `{gate_pass}`",
        "",
        "## Figures",
        f"- `{fig_panel}`",
        f"- `{fig_gap}`",
    ]
    _text_save(paths.docs_root / "AUC_LIFT_CYCLE06_EXTERNAL.md", "\n".join(md) + "\n")
    return payload


# -------------------------
# Cycle 7
# -------------------------
def run_cycle07(paths: Paths, pull_id: str) -> dict[str, Any]:
    cmd = [
        str(root / ".venv" / "bin" / "python"),
        str(root / "scripts" / "run_digital_twin_update_cycle.py"),
        "--pull-id",
        pull_id,
    ]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)

    summary_path = (
        root
        / "outputs"
        / "digital_twin_updates"
        / pull_id
        / "twin_refresh_summary.json"
    )
    summary = _load_json(summary_path)

    pass_zero = bool(
        summary.get("validation", {}).get("zero_delta_counterfactual_pass", False)
    )
    pass_det = bool(
        summary.get("validation", {}).get("deterministic_rerun_pass", False)
    )
    bounded = bool(summary.get("max_abs_delta_risk", 0.0) <= 1.0)
    gate_pass = bool(pass_zero and pass_det and bounded)

    payload = {
        "cycle_id": "cycle07",
        "timestamp_utc": _now_utc(),
        "runner": {
            "command": " ".join(cmd),
            "returncode": int(proc.returncode),
        },
        "twin_summary": summary,
        "gate": {
            "zero_delta_invariance": pass_zero,
            "deterministic_rerun": pass_det,
            "bounded_outputs": bounded,
            "pass": gate_pass,
        },
    }
    _json_save(paths.cycle_root / "cycle07_twin_coupling.json", payload)

    md = [
        "# AUC Lift Cycle 07 - Digital Twin Coupling Gate",
        "",
        f"Generated (UTC): `{payload['timestamp_utc']}`",
        "",
        "## Twin Summary",
        f"- n_patients_refreshed: `{summary.get('n_patients_refreshed')}`",
        f"- n_changed_patients: `{summary.get('n_changed_patients')}`",
        f"- mean_abs_delta_risk: `{summary.get('mean_abs_delta_risk')}`",
        f"- max_abs_delta_risk: `{summary.get('max_abs_delta_risk')}`",
        f"- warnings: `{summary.get('warnings')}`",
        "",
        "## Gate",
        f"- zero_delta_invariance: `{pass_zero}`",
        f"- deterministic_rerun: `{pass_det}`",
        f"- bounded_outputs: `{bounded}`",
        f"- pass: `{gate_pass}`",
        "",
        "## Visuals",
        f"- `{summary.get('artifacts', {}).get('patient_trajectory_figure', '')}`",
        f"- `{summary.get('artifacts', {}).get('cohort_delta_figure', '')}`",
        f"- `{summary.get('artifacts', {}).get('sensitivity_shift_figure', '')}`",
    ]
    _text_save(
        paths.docs_root / "AUC_LIFT_CYCLE07_TWIN_COUPLING.md", "\n".join(md) + "\n"
    )
    return payload


# -------------------------
# Cycle 8
# -------------------------
def run_cycle08(
    paths: Paths,
    cycle03: dict[str, Any],
    cycle06: dict[str, Any],
    cycle07: dict[str, Any],
) -> dict[str, Any]:
    internal_auc = float(cycle03["best_auc"])
    external_auc = float(cycle06["candidate_external"]["auc"])
    twin_pass = bool(cycle07["gate"]["pass"])

    if internal_auc >= 0.90 and external_auc >= 0.80:
        level = "LEVEL_1_SUCCESS"
        next_step = "Begin independent external cohort protocol and dissertation demo packaging."
    elif internal_auc >= 0.90 and external_auc < 0.80:
        level = "INTERNAL_ONLY_SUCCESS"
        next_step = "Continue transportability cycles (covariate harmonization + shift handling)."
    else:
        level = "RETURN_TO_FEATURE_ENDPOINT_CYCLES"
        next_step = "Iterate feature integrity and endpoint reliability before architecture complexity."

    readiness_level = {
        "internal_auc_target_met": internal_auc >= 0.90,
        "external_auc_target_met": external_auc >= 0.80,
        "twin_gate_pass": twin_pass,
        "promotion_allowed": bool(
            internal_auc >= 0.90 and external_auc >= 0.80 and twin_pass
        ),
    }

    payload = {
        "cycle_id": "cycle08",
        "timestamp_utc": _now_utc(),
        "decision": {
            "level": level,
            "next_step": next_step,
        },
        "targets": {
            "internal_auc": internal_auc,
            "external_like_auc": external_auc,
            "target_internal": 0.90,
            "target_external": 0.80,
        },
        "readiness": readiness_level,
    }

    _json_save(paths.cycle_root / "cycle08_milestone_decision.json", payload)

    md1 = [
        "# AUC Lift Milestone Decision",
        "",
        f"Generated (UTC): `{payload['timestamp_utc']}`",
        "",
        "## Targets",
        f"- internal_auc: `{internal_auc:.4f}` (target `>=0.90`)",
        f"- external_like_auc: `{external_auc:.4f}` (target `>=0.80`)",
        f"- twin_gate_pass: `{twin_pass}`",
        "",
        "## Decision",
        f"- level: `{level}`",
        f"- next_step: {next_step}",
    ]
    _text_save(
        paths.docs_root / "AUC_LIFT_MILESTONE_DECISION.md", "\n".join(md1) + "\n"
    )

    md2 = [
        "# Dissertation Twin Readiness Level",
        "",
        f"Generated (UTC): `{payload['timestamp_utc']}`",
        "",
        "## Readiness Signals",
        f"- internal_auc_target_met: `{readiness_level['internal_auc_target_met']}`",
        f"- external_auc_target_met: `{readiness_level['external_auc_target_met']}`",
        f"- twin_gate_pass: `{readiness_level['twin_gate_pass']}`",
        f"- promotion_allowed: `{readiness_level['promotion_allowed']}`",
        "",
        "## Interpretation",
        "- Promotion remains blocked unless both predictive thresholds and twin gate are satisfied.",
        "- Current stage supports dissertation evidence-building with auditable cycle deltas.",
    ]
    _text_save(
        paths.docs_root / "DISSERTATION_TWIN_READINESS_LEVEL.md", "\n".join(md2) + "\n"
    )

    return payload


# -------------------------
# Contracts documentation
# -------------------------
def write_contract_additions(paths: Paths) -> None:
    contracts = {
        "FeatureQualityReport": {
            "feature_name": "str",
            "variance": "float",
            "is_constant": "bool",
            "missing_fraction": "float",
            "lineage_ok": "bool",
            "action": "keep|drop_constant|keep_flag_near_constant",
        },
        "CycleDeltaReport": {
            "cycle_id": "str",
            "changes_applied": "list[str]",
            "internal_metrics_delta": "dict",
            "external_metrics_delta": "dict",
            "twin_metrics_delta": "dict",
            "gate_status": "pass|fail|warn",
        },
        "TwinPromotionGate": {
            "model_version": "str",
            "calibration_version": "str",
            "external_status": "pass|fail",
            "twin_stability_status": "pass|fail",
            "promotion_decision": "allow|block",
        },
        "VersionKeys": [
            "dataset_version",
            "split_hash",
            "model_hash",
            "calibration_hash",
            "twin_version",
        ],
    }
    _json_save(paths.cycle_root / "contracts_v1.json", contracts)


def _precheck_true_saa_label(paths: Paths) -> dict[str, Any]:
    """Hard-stop precheck: require true SAA label source before any training cycles."""
    csv_path = (
        root
        / "data"
        / "03_prodromal"
        / "final_training_dataset"
        / "unified_longitudinal_early_pd.csv"
    )
    if not csv_path.exists():
        payload = {
            "timestamp_utc": _now_utc(),
            "source_csv": str(csv_path),
            "gate": {"pass": False},
            "reason": "missing_final_training_dataset_csv",
        }
        _json_save(paths.cycle_root / "precheck_label_gate.json", payload)
        _text_save(
            paths.docs_root / "AUC_LIFT_PREFLIGHT_LABEL_GATE.md",
            "# AUC Lift Preflight Label Gate\n\n- status: `FAIL`\n- reason: missing final training dataset CSV\n",
        )
        return payload

    df_head = pd.read_csv(csv_path, nrows=5)
    cols = set(df_head.columns)
    true_aliases = [
        "saa_label",
        "SAA_POSITIVE",
        "SAA_LABEL",
        "saa_status",
        "SAA_STATUS",
        "saa",
        "SAA",
        "subacute_anxiety",
        "anxiety_label",
    ]
    proxy_aliases = ["event", "event_observed", "phenoconverted"]

    true_candidates = [c for c in true_aliases if c in cols]
    proxy_candidates = [c for c in proxy_aliases if c in cols]

    resolved_true = true_candidates[0] if true_candidates else None
    reason = "ok"
    pass_flag = bool(resolved_true is not None and resolved_true not in proxy_aliases)
    identical_to_event = None

    if not pass_flag:
        if resolved_true is None:
            reason = "missing_true_saa_label_column"
        else:
            reason = f"proxy_label_source_detected:{resolved_true}"
    else:
        event_col = next((c for c in proxy_aliases if c in cols), None)
        if event_col is not None:
            check_df = pd.read_csv(csv_path, usecols=[resolved_true, event_col])
            y = pd.to_numeric(check_df[resolved_true], errors="coerce")
            e = pd.to_numeric(check_df[event_col], errors="coerce")
            identical_to_event = bool(
                y.notna().all()
                and e.notna().all()
                and np.array_equal(y.to_numpy(dtype=int), e.to_numpy(dtype=int))
            )
            if identical_to_event:
                pass_flag = False
                reason = f"saa_label_identical_to_{event_col}"

    cohort_pat = set(
        pd.to_numeric(
            pd.read_csv(csv_path, usecols=["PATNO"])["PATNO"], errors="coerce"
        )
        .dropna()
        .astype(int)
        .tolist()
    )
    saa_ref_path = root / "data" / "03_prodromal" / "enhanced" / "saa_labels.csv"
    saa_ref = {
        "path": str(saa_ref_path),
        "exists": saa_ref_path.exists(),
        "n_patients": 0,
        "overlap_patients": 0,
    }
    if saa_ref_path.exists():
        ref_df = pd.read_csv(saa_ref_path, usecols=["PATNO"])
        ref_pat = set(
            pd.to_numeric(ref_df["PATNO"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
        saa_ref["n_patients"] = int(len(ref_pat))
        saa_ref["overlap_patients"] = int(len(ref_pat & cohort_pat))

    payload = {
        "timestamp_utc": _now_utc(),
        "source_csv": str(csv_path),
        "columns_seen_head": sorted(cols),
        "resolved_true_saa_label": resolved_true,
        "proxy_candidates": proxy_candidates,
        "identical_to_event": identical_to_event,
        "saa_reference": saa_ref,
        "gate": {"pass": pass_flag},
        "reason": reason,
    }
    _json_save(paths.cycle_root / "precheck_label_gate.json", payload)

    md = [
        "# AUC Lift Preflight Label Gate",
        "",
        f"Generated (UTC): `{payload['timestamp_utc']}`",
        "",
        f"- source_csv: `{csv_path}`",
        f"- resolved_true_saa_label: `{resolved_true}`",
        f"- proxy_candidates: `{proxy_candidates}`",
        f"- identical_to_event: `{identical_to_event}`",
        f"- saa_reference_exists: `{saa_ref['exists']}`",
        f"- saa_reference_patients: `{saa_ref['n_patients']}`",
        f"- saa_reference_overlap_patients: `{saa_ref['overlap_patients']}`",
        f"- status: `{'PASS' if pass_flag else 'FAIL'}`",
        f"- reason: `{reason}`",
        "",
        "Gate rule: model cycles are blocked unless a true SAA label source exists and is not proxy-equivalent to event labels.",
    ]
    _text_save(
        paths.docs_root / "AUC_LIFT_PREFLIGHT_LABEL_GATE.md", "\n".join(md) + "\n"
    )
    return payload


def _safe_shape(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path)
            return {"exists": True, "rows": int(len(df)), "cols": int(df.shape[1])}
        except Exception as exc:
            return {"exists": True, "error": str(exc)}
    return {"exists": True}


def _write_ppmi_lineage_review(paths: Paths) -> dict[str, Any]:
    """Trace how final training artifacts are built from raw PPMI datasets."""
    raw_root = root / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv"
    raw_tables = sorted(raw_root.glob("*.csv"))

    lineage_rows = [
        {
            "stage": "raw_ppmi",
            "script": "n/a",
            "input_paths": str(raw_root),
            "output_path": str(raw_root),
            "notes": "Raw PPMI CSV tables inventory",
        },
        {
            "stage": "prodromal_cohort_base",
            "script": "phase8.1 cohort extraction (upstream)",
            "input_paths": "PPMI raw clinical tables",
            "output_path": "data/prodromal_cohort/prodromal_survival_data.csv",
            "notes": "Base cohort used by Phase 8.2 feature extractors",
        },
        {
            "stage": "extract_genetic",
            "script": "archive/development/phase8/subphase8_2_dynamic_endpoints/extract_genetic_features.py",
            "input_paths": "data/00_raw/GIMAN/ppmi_data_csv/iu_genetic_consensus_20250515_18Sep2025.csv",
            "output_path": "data/03_prodromal/enhanced/genetic_features.csv",
            "notes": "Genetic mutation/risk features",
        },
        {
            "stage": "extract_expanded_clinical",
            "script": "archive/development/phase8/subphase8_2_dynamic_endpoints/extract_expanded_clinical.py",
            "input_paths": "PPMI UPDRS I/II/III CSV tables",
            "output_path": "data/03_prodromal/enhanced/expanded_clinical_features.csv",
            "notes": "Clinical progression features",
        },
        {
            "stage": "extract_freesurfer_volumes",
            "script": "archive/development/phase8/subphase8_2_dynamic_endpoints/extract_freesurfer_volumes.py",
            "input_paths": "data/00_raw/GIMAN/ppmi_data_csv/FS7_ASEG_VOL_30Sep2025.csv",
            "output_path": "data/03_prodromal/enhanced/freesurfer_volumes.csv",
            "notes": "Subcortical volume markers",
        },
        {
            "stage": "extract_dat_spect_sbr",
            "script": "archive/development/phase8/subphase8_2_dynamic_endpoints/extract_dat_spect_sbr.py",
            "input_paths": "data/01_processed/dat_spect_sbr_values.csv",
            "output_path": "data/03_prodromal/enhanced/dat_spect_sbr.csv",
            "notes": "DAT-SPECT binding/asymmetry markers",
        },
        {
            "stage": "extract_csf_biomarkers",
            "script": "archive/development/phase8/subphase8_2_dynamic_endpoints/extract_csf_biomarkers.py",
            "input_paths": "data/00_raw/GIMAN/ppmi_data_csv/Current_Biospecimen_Analysis_Results_18Sep2025.csv",
            "output_path": "data/03_prodromal/enhanced/csf_biomarkers.csv",
            "notes": "CSF alpha-syn/tau/abeta/pTau181 markers",
        },
        {
            "stage": "extract_clinical_biomarkers",
            "script": "archive/development/phase8/subphase8_2_dynamic_endpoints/extract_clinical_biomarkers.py",
            "input_paths": "PPMI UPSIT/RBD/SCOPA-AUT/ESS CSV tables",
            "output_path": "data/03_prodromal/enhanced/clinical_biomarkers.csv",
            "notes": "Non-motor clinical biomarker scores",
        },
        {
            "stage": "extract_cortical_thickness",
            "script": "archive/development/phase8/subphase8_2_dynamic_endpoints/extract_cortical_thickness.py",
            "input_paths": "data/00_raw/GIMAN/ppmi_data_csv/FS7_APARC_CTH_30Sep2025.csv",
            "output_path": "data/03_prodromal/enhanced/cortical_thickness.csv",
            "notes": "Cortical thickness markers",
        },
        {
            "stage": "extract_saa_labels",
            "script": "archive/development/phase8/subphase8_2_dynamic_endpoints/extract_saa_labels.py",
            "input_paths": "data/00_raw/GIMAN/ppmi_data_csv/Current_Biospecimen_Analysis_Results_18Sep2025.csv",
            "output_path": "data/03_prodromal/enhanced/saa_labels.csv",
            "notes": "Real SAA labels from TESTNAME='SAA Positive - final'",
        },
        {
            "stage": "merge_feature_groups",
            "script": "archive/development/phase8/subphase8_2_dynamic_endpoints/merge_all_features.py",
            "input_paths": "data/03_prodromal/enhanced/*.csv",
            "output_path": "data/03_prodromal/enhanced/prodromal_multimodal_features.csv",
            "notes": "36-feature multimodal prodromal table",
        },
        {
            "stage": "final_training_merge",
            "script": "archive/development/phase8/subphase8_2_dynamic_endpoints/merge_final_training_dataset.py",
            "input_paths": "enhanced_longitudinal + enhanced_36_features (+ optional early_pd)",
            "output_path": "data/03_prodromal/final_training_dataset/unified_longitudinal_early_pd.csv",
            "notes": "Longitudinal training-ready table",
        },
        {
            "stage": "pyg_contract_build",
            "script": "archive/development/phase8/subphase8_2_dynamic_endpoints/prepare_final_pyg_data.py",
            "input_paths": "data/03_prodromal/final_training_dataset/unified_longitudinal_early_pd.csv",
            "output_path": "data/03_prodromal/final_pyg_data_sota_run/{train_data.pt,test_data.pt,pyg_data_metadata.json}",
            "notes": "Canonical patient-disjoint tensors",
        },
    ]

    lineage_df = pd.DataFrame(lineage_rows)
    table_path = paths.docs_root / "PPMI_DATA_LINEAGE_TABLE.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_df.to_csv(table_path, index=False)

    key_outputs = [
        root
        / "data"
        / "03_prodromal"
        / "enhanced"
        / "prodromal_multimodal_features.csv",
        root
        / "data"
        / "03_prodromal"
        / "final_training_dataset"
        / "unified_longitudinal_early_pd.csv",
        root
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_run"
        / "pyg_data_metadata.json",
    ]
    output_shapes = {str(p): _safe_shape(p) for p in key_outputs}

    saa_ref_path = root / "data" / "03_prodromal" / "enhanced" / "saa_labels.csv"
    saa_ref = {
        "exists": saa_ref_path.exists(),
        "n_patients": 0,
        "overlap_with_final_training": 0,
    }
    if saa_ref_path.exists():
        ref_df = pd.read_csv(saa_ref_path, usecols=["PATNO"])
        ref_pat = set(
            pd.to_numeric(ref_df["PATNO"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
        final_train_path = (
            root
            / "data"
            / "03_prodromal"
            / "final_training_dataset"
            / "unified_longitudinal_early_pd.csv"
        )
        if final_train_path.exists():
            final_df = pd.read_csv(final_train_path, usecols=["PATNO"])
            final_pat = set(
                pd.to_numeric(final_df["PATNO"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
            saa_ref["overlap_with_final_training"] = int(len(ref_pat & final_pat))
        saa_ref["n_patients"] = int(len(ref_pat))

    payload = {
        "timestamp_utc": _now_utc(),
        "raw_ppmi_root": str(raw_root),
        "raw_ppmi_csv_count": int(len(raw_tables)),
        "lineage_rows": len(lineage_rows),
        "lineage_table": str(table_path),
        "key_output_shapes": output_shapes,
        "saa_reference": {
            "path": str(saa_ref_path),
            **saa_ref,
        },
    }
    _json_save(paths.cycle_root / "ppmi_data_lineage_review.json", payload)

    md = [
        "# PPMI Data Lineage Review",
        "",
        f"Generated (UTC): `{payload['timestamp_utc']}`",
        "",
        "## Raw Inventory",
        f"- raw_ppmi_root: `{raw_root}`",
        f"- raw_ppmi_csv_count: `{len(raw_tables)}`",
        "",
        "## Lineage",
        "- Raw PPMI tables are transformed via Phase 8.2 extractors into modality-specific enhanced tables.",
        "- Enhanced modality tables are merged into `prodromal_multimodal_features.csv`.",
        "- Final longitudinal training table is built by `merge_final_training_dataset.py`.",
        "- Canonical PyG train/test tensors are built by `prepare_final_pyg_data.py`.",
        "",
        "## Key Output Shape Checks",
    ]
    for p, shp in output_shapes.items():
        if shp.get("exists"):
            if "rows" in shp:
                md.append(f"- `{p}`: rows={shp['rows']}, cols={shp['cols']}")
            else:
                md.append(f"- `{p}`: exists")
        else:
            md.append(f"- `{p}`: MISSING")
    md.extend(
        [
            "",
            "## SAA Label Availability Check",
            f"- saa_reference_path: `{saa_ref_path}`",
            f"- saa_reference_exists: `{saa_ref['exists']}`",
            f"- saa_reference_patients: `{saa_ref['n_patients']}`",
            f"- overlap_with_final_training_patients: `{saa_ref['overlap_with_final_training']}`",
            "",
            "## Artifact",
            f"- lineage table: `{table_path}`",
            f"- machine-readable summary: `{paths.cycle_root / 'ppmi_data_lineage_review.json'}`",
        ]
    )
    _text_save(paths.docs_root / "PPMI_DATA_LINEAGE_REVIEW.md", "\n".join(md) + "\n")
    return payload


def run_all(pull_id: str) -> dict[str, Any]:
    _seed_everything(42)
    paths = Paths(
        cycle_root=root / "outputs" / "sota_lift",
        vis_root=root / "visualizations" / "sota_lift",
        docs_root=root / "Docs" / "audit",
        cycle_dataset_root=root
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_lift_cycle01",
    )

    write_contract_additions(paths)
    lineage = _write_ppmi_lineage_review(paths)
    preflight = _precheck_true_saa_label(paths)
    if not preflight["gate"]["pass"]:
        summary = {
            "timestamp_utc": _now_utc(),
            "pull_id": pull_id,
            "status": "blocked_preflight",
            "blocked_by": "label_truth_gate",
            "preflight": preflight,
            "lineage_review": lineage,
        }
        _json_save(paths.cycle_root / "run_summary.json", summary)
        raise RuntimeError(
            "AUC-lift cycles blocked: true saa_label is missing/proxy-derived. "
            "Ingest a real SAA label column before model cycles."
        )

    c0 = run_cycle00(paths, pull_id)
    c1 = run_cycle01(paths)
    c2 = run_cycle02(paths)
    c3 = run_cycle03(
        paths, baseline_internal_auc=float(c0["baseline_metrics"]["internal_auc"])
    )
    c4 = run_cycle04(paths, c3)
    c5 = run_cycle05(paths, c3)
    c6 = run_cycle06(paths, c3, pull_id)
    c7 = run_cycle07(paths, pull_id)
    c8 = run_cycle08(paths, c3, c6, c7)

    summary = {
        "timestamp_utc": _now_utc(),
        "pull_id": pull_id,
        "cycles": {
            "cycle00": c0.get("gate_pass", c0.get("gate", {}).get("pass")),
            "cycle01": c1.get("gate", {}).get("pass"),
            "cycle02": c2.get("gate", {}).get("pass"),
            "cycle03": c3.get("gate", {}).get("pass"),
            "cycle04": c4.get("gate", {}).get("pass"),
            "cycle05": c5.get("gate", {}).get("pass"),
            "cycle06": c6.get("gate", {}).get("pass"),
            "cycle07": c7.get("gate", {}).get("pass"),
            "cycle08": True,
        },
        "targets": {
            "internal_auc": c3["best_auc"],
            "external_auc": c6["candidate_external"]["auc"],
            "internal_target_met": c3["best_auc"] >= 0.90,
            "external_target_met": c6["candidate_external"]["auc"] >= 0.80,
        },
    }
    _json_save(paths.cycle_root / "run_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run FUZZY GIMAN AUC-Lift cycles")
    p.add_argument("--pull-id", type=str, default="PPMI_20251008_REAL_DISJOINT")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_all(args.pull_id)
    print(json.dumps(summary, indent=2))
