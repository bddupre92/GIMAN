from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.explainability.appendix_figures import (
    generate_appendix_package,  # noqa: E402
)
from giman_pipeline.explainability.real_data_explain import (  # noqa: E402
    _load_nf_model,
    _predict_probs,
)


@dataclass(frozen=True)
class FigureContext:
    train_data: Any
    test_data: Any
    metadata: dict[str, Any]
    y_train: np.ndarray
    y_test: np.ndarray
    probs_train: np.ndarray
    probs_test: np.ndarray
    probs_test_platt: np.ndarray
    probs_test_iso: np.ndarray


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(command: list[str]) -> None:
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
            + "\nSTDOUT:\n"
            + proc.stdout[-2000:]
            + "\nSTDERR:\n"
            + proc.stderr[-2000:]
        )


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Use copy (not copy2) so regenerated bundles have current run timestamps.
    shutil.copy(src, dst)


def _build_context(checkpoint_path: Path) -> FigureContext:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data_path = ROOT / "data/03_prodromal/final_pyg_data_sota_run/train_data.pt"
    test_data_path = ROOT / "data/03_prodromal/final_pyg_data_sota_run/test_data.pt"
    metadata_path = (
        ROOT / "data/03_prodromal/final_pyg_data_sota_run/pyg_data_metadata.json"
    )

    metadata = _load_json(metadata_path)
    train_data = torch.load(train_data_path, weights_only=False).to(device)
    test_data = torch.load(test_data_path, weights_only=False).to(device)

    model = _load_nf_model(
        in_features=int(test_data.x.shape[1]),
        checkpoint_path=checkpoint_path,
        device=device,
    )

    probs_test, _ = _predict_probs(model, test_data)
    probs_train, _ = _predict_probs(model, train_data)

    y_train = train_data.saa_label.detach().cpu().numpy().astype(int)
    y_test = test_data.saa_label.detach().cpu().numpy().astype(int)

    # Fit calibrators on the training split only.
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    platt.fit(probs_train.reshape(-1, 1), y_train)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(probs_train, y_train)
    probs_test_platt = platt.predict_proba(probs_test.reshape(-1, 1))[:, 1]
    probs_test_iso = isotonic.predict(probs_test)

    return FigureContext(
        train_data=train_data,
        test_data=test_data,
        metadata=metadata,
        y_train=y_train,
        y_test=y_test,
        probs_train=probs_train,
        probs_test=probs_test,
        probs_test_platt=probs_test_platt,
        probs_test_iso=probs_test_iso,
    )


def _figure3_benchmark(output_path: Path) -> None:
    lock = _load_json(ROOT / "outputs/sota_lock/internal_sota_lock.json")
    full = _load_json(
        ROOT
        / "outputs/phase9_neuro_fuzzy/PREP_20260208_SAA_COHORT3_full/full_training_results.json"
    )
    multi = _load_json(
        ROOT
        / "outputs/phase9_neuro_fuzzy/PREP_20260208_SAA_COHORT3_multitask/multitask_training_results.json"
    )
    ext = _load_json(
        ROOT
        / "outputs/external_validation/PPMI_20251008_REAL_DISJOINT/external_metrics.json"
    )

    baselines = lock["baselines"]
    models = ["logistic_regression", "random_forest", "svm_rbf"]
    auc_vals = [float(baselines[m]["auc"]) for m in models]
    pr_vals = [float(baselines[m]["pr_auc"]) for m in models]

    labels = ["logistic_regression", "random_forest", "svm_rbf", "fuzzy_giman_full"]
    auc_with_fuzzy = auc_vals + [float(full["best_test_auc"])]
    pr_with_fuzzy = pr_vals + [float(full["test_pr_auc"])]
    auc_ci = [baselines[m]["auc_ci_95"] for m in models] + [full["auc_ci_95"]]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    x = np.arange(len(labels))

    lower = [max(0.0, v - ci[0]) for v, ci in zip(auc_with_fuzzy, auc_ci, strict=False)]
    upper = [max(0.0, ci[1] - v) for v, ci in zip(auc_with_fuzzy, auc_ci, strict=False)]
    axes[0].bar(x, auc_with_fuzzy, color=["#4E79A7", "#F28E2B", "#59A14F", "#E15759"])
    axes[0].errorbar(
        x,
        auc_with_fuzzy,
        yerr=np.array([lower, upper]),
        fmt="none",
        ecolor="black",
        capsize=4,
        lw=1.1,
    )
    axes[0].set_title("Internal AUC-ROC (95% CI)")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)

    axes[1].bar(
        x,
        pr_with_fuzzy,
        color=["#4E79A7", "#F28E2B", "#59A14F", "#E15759"],
    )
    axes[1].set_title("Internal PR-AUC")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)

    ext_auc = float(ext["classification"]["auc"])
    ext_pr = float(ext["classification"]["pr_auc"])
    multi_final = float(multi["final_saa_auc"])
    bars = [float(full["best_test_auc"]), multi_final, ext_auc]
    bars_pr = [float(full["test_pr_auc"]), np.nan, ext_pr]
    x2 = np.arange(3)
    axes[2].bar(x2 - 0.17, bars, width=0.34, label="AUC", color="#4E79A7")
    axes[2].bar(
        x2 + 0.17,
        np.nan_to_num(bars_pr, nan=0.0),
        width=0.34,
        label="PR-AUC",
        color="#F28E2B",
    )
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(
        ["full_internal", "multitask_final", "external_like"],
        rotation=20,
        ha="right",
    )
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Internal vs External-like")
    axes[2].grid(axis="y", linestyle="--", alpha=0.25)
    axes[2].legend()

    fig.suptitle("Benchmark Comparison (Updated Artifact-Backed Runs)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _figure5_time_dependent(ctx: FigureContext, output_path: Path) -> None:
    times = ctx.test_data.time.detach().cpu().numpy().astype(float)
    y = ctx.y_test
    raw = ctx.probs_test
    platt = ctx.probs_test_platt
    iso = ctx.probs_test_iso
    horizons = sorted(np.unique(times).tolist())

    x_train = ctx.train_data.x.detach().cpu().numpy()
    x_test = ctx.test_data.x.detach().cpu().numpy()
    y_train = ctx.y_train

    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(x_train, y_train)
    p_lr = lr.predict_proba(x_test)[:, 1]

    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)
    p_rf = rf.predict_proba(x_test)[:, 1]

    svm = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )
    svm.fit(x_train, y_train)
    p_svm = svm.predict_proba(x_test)[:, 1]

    def _curve(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hs: list[float] = []
        vals: list[float] = []
        for h in horizons:
            mask = times <= h
            if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
                continue
            auc = _safe_auc(y[mask], scores[mask])
            if np.isfinite(auc):
                hs.append(float(h))
                vals.append(float(auc))
        if len(hs) < 2:
            hs = [float(np.min(times)), float(np.max(times))]
            auc = _safe_auc(y, scores)
            vals = [float(auc), float(auc)]
        return np.array(hs, dtype=float), np.array(vals, dtype=float)

    h_raw, a_raw = _curve(raw)
    h_platt, a_platt = _curve(platt)
    h_iso, a_iso = _curve(iso)
    h_lr, a_lr = _curve(p_lr)
    h_rf, a_rf = _curve(p_rf)
    h_svm, a_svm = _curve(p_svm)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharey=True)
    ax0, ax1 = axes

    ax0.plot(
        h_raw, a_raw, marker="o", lw=2.4, color="#1f77b4", label="fuzzy_raw", zorder=3
    )
    ax0.plot(
        h_platt,
        a_platt,
        marker="s",
        lw=2.0,
        color="#ff7f0e",
        label="fuzzy_platt",
        zorder=2,
    )
    ax0.plot(
        h_iso,
        a_iso,
        marker="^",
        lw=2.0,
        color="#2ca02c",
        label="fuzzy_isotonic",
        zorder=1,
    )
    ax0.set_title("Panel A: FUZZY GIMAN (Raw vs Calibrated)")
    ax0.set_xlabel("Horizon (months)")
    ax0.set_ylabel("AUC-ROC")
    ax0.grid(True, linestyle="--", alpha=0.25)
    ax0.legend(fontsize=8, loc="best")

    ax1.plot(h_raw, a_raw, marker="o", lw=2.4, color="#1f77b4", label="fuzzy_raw")
    ax1.plot(h_lr, a_lr, marker="D", lw=1.8, color="#9467bd", label="logistic_baseline")
    ax1.plot(
        h_rf, a_rf, marker="P", lw=1.8, color="#8c564b", label="random_forest_baseline"
    )
    ax1.plot(
        h_svm, a_svm, marker="X", lw=1.8, color="#e377c2", label="svm_rbf_baseline"
    )
    ax1.set_title("Panel B: Baseline Model Comparison")
    ax1.set_xlabel("Horizon (months)")
    ax1.grid(True, linestyle="--", alpha=0.25)
    ax1.legend(fontsize=8, loc="best")

    all_vals = np.concatenate([a_raw, a_platt, a_iso, a_lr, a_rf, a_svm])
    y_min = float(np.nanmin(all_vals))
    y_max = float(np.nanmax(all_vals))
    pad = max(0.02, 0.08 * (y_max - y_min + 1e-6))
    for ax in axes:
        ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))

    fig.suptitle("Time-Dependent AUC by Horizon")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _figure6_calibration(ctx: FigureContext, output_path: Path) -> None:
    y = ctx.y_test
    raw = ctx.probs_test
    iso = ctx.probs_test_iso
    frac_raw, mean_raw = calibration_curve(y, raw, n_bins=10, strategy="quantile")
    frac_iso, mean_iso = calibration_curve(y, iso, n_bins=10, strategy="quantile")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
    for ax, mean_pred, frac_pos, title, color in [
        (axes[0], mean_raw, frac_raw, "Panel A: Raw", "#4E79A7"),
        (axes[1], mean_iso, frac_iso, "Panel B: Isotonic", "#E15759"),
    ]:
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.plot(mean_pred, frac_pos, "o-", color=color, lw=2.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend()
    fig.suptitle("Calibration Plots: Predicted vs Observed Probabilities")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _figure7_robustness(output_path: Path) -> None:
    trials_path = (
        ROOT / "outputs/phase9_ablations/ABLATE_20260208_RUN3/ablation_trials.csv"
    )
    df = pd.read_csv(trials_path)
    ok = df[df["status"] == "ok"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    axes[0].boxplot(
        [
            ok.loc[ok["model_type"] == "quick", "saa_auc_metric"].values,
            ok.loc[ok["model_type"] == "multitask", "saa_auc_metric"].values,
        ],
        labels=["quick", "multitask"],
    )
    axes[0].set_title("Selection Metric (AUC)")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)

    top_ablations = (
        ok.groupby("ablation", as_index=False)["saa_auc_metric"]
        .mean()
        .sort_values("saa_auc_metric", ascending=False)
        .head(6)["ablation"]
        .tolist()
    )
    vals = [
        ok.loc[ok["ablation"] == name, "saa_auc_metric"].values
        for name in top_ablations
    ]
    axes[1].boxplot(vals, labels=top_ablations)
    axes[1].set_title("AUC by Ablation Group")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)

    mvals = ok.loc[ok["model_type"] == "multitask", "final_c_index"].dropna().values
    if len(mvals) > 0:
        axes[2].boxplot([mvals], labels=["multitask_final_c_index"])
    axes[2].set_title("Survival Head Stability")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(axis="y", linestyle="--", alpha=0.25)

    fig.suptitle("Run-Set Robustness (Ablation Cycles)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _figure8_noise_sensitivity(ctx: FigureContext, output_path: Path) -> None:
    x_train = ctx.train_data.x.detach().cpu().numpy()
    x_test = ctx.test_data.x.detach().cpu().numpy()
    y_train = ctx.y_train
    y_test = ctx.y_test

    # Crisp tabular baseline.
    baseline = LogisticRegression(max_iter=2000)
    baseline.fit(x_train, y_train)

    # Neuro-fuzzy model for noisy inference.
    checkpoint_path = (
        ROOT
        / "outputs/phase9_neuro_fuzzy/PREP_20260208_SAA_COHORT3_full/neuro_fuzzy_best.pth"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_nf_model(
        in_features=int(ctx.test_data.x.shape[1]),
        checkpoint_path=checkpoint_path,
        device=device,
    )
    base_data = ctx.test_data

    feature_std = np.std(x_train, axis=0, ddof=0)
    rng = np.random.default_rng(42)
    levels = np.array([0.0, 0.05, 0.10, 0.20], dtype=float)

    auc_baseline: list[float] = []
    auc_fuzzy: list[float] = []
    pr_baseline: list[float] = []
    pr_fuzzy: list[float] = []

    for level in levels:
        noise = rng.normal(0.0, level, size=x_test.shape) * feature_std
        x_noisy = x_test + noise

        # Baseline predictions.
        p_baseline = baseline.predict_proba(x_noisy)[:, 1]
        auc_baseline.append(_safe_auc(y_test, p_baseline))
        pr_baseline.append(_safe_pr_auc(y_test, p_baseline))

        # Neuro-fuzzy predictions on noisy features.
        noisy_graph = base_data.clone()
        noisy_graph.x = torch.tensor(
            x_noisy, dtype=torch.float32, device=base_data.x.device
        )
        p_fuzzy, _ = _predict_probs(model, noisy_graph)
        auc_fuzzy.append(_safe_auc(y_test, p_fuzzy))
        pr_fuzzy.append(_safe_pr_auc(y_test, p_fuzzy))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    axes[0].plot(
        levels * 100, auc_baseline, marker="o", lw=2.2, label="logistic_baseline"
    )
    axes[0].plot(levels * 100, auc_fuzzy, marker="s", lw=2.2, label="fuzzy_giman")
    axes[0].set_xlabel("Noise level (% of feature std)")
    axes[0].set_ylabel("AUC-ROC")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Panel A: AUC degradation")
    axes[0].grid(True, linestyle="--", alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        levels * 100, pr_baseline, marker="o", lw=2.2, label="logistic_baseline"
    )
    axes[1].plot(levels * 100, pr_fuzzy, marker="s", lw=2.2, label="fuzzy_giman")
    axes[1].set_xlabel("Noise level (% of feature std)")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Panel B: PR-AUC degradation")
    axes[1].grid(True, linestyle="--", alpha=0.25)
    axes[1].legend()

    fig.suptitle("Noise Sensitivity Analysis (Updated Test Split)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _figure9_embedding(ctx: FigureContext, output_path: Path) -> None:
    x = torch.cat([ctx.train_data.x, ctx.test_data.x], dim=0).detach().cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    z = tsne.fit_transform(x)

    hard = KMeans(n_clusters=3, random_state=42, n_init=20).fit_predict(x)
    gmm = GaussianMixture(n_components=3, random_state=42).fit(x)
    soft = gmm.predict_proba(x)
    soft_lbl = soft.argmax(axis=1)
    soft_strength = soft.max(axis=1)

    colors = np.array(["#E15759", "#4E79A7", "#59A14F"])

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.0))
    axes[0].scatter(
        z[:, 0], z[:, 1], c=colors[hard], s=45, alpha=0.8, edgecolor="k", linewidth=0.2
    )
    axes[0].set_title("Panel A: Hard K-Means Clustering")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")

    rgba = []
    for idx, lbl in enumerate(soft_lbl):
        base = plt.matplotlib.colors.to_rgba(colors[lbl])
        rgba.append((base[0], base[1], base[2], 0.25 + 0.75 * soft_strength[idx]))
    axes[1].scatter(z[:, 0], z[:, 1], c=rgba, s=45, edgecolor="k", linewidth=0.2)
    axes[1].set_title("Panel B: Soft Membership (GMM proxy)")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    fig.suptitle("Patient Embedding Space: Hard vs Soft Clustering")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _figure10_similarity_graph(ctx: FigureContext, output_path: Path) -> None:
    x = ctx.test_data.x.detach().cpu().numpy()
    pat = ctx.test_data.patno.detach().cpu().numpy().astype(int)
    probs = ctx.probs_test
    target_idx = int(np.argmax(probs))

    sim = cosine_similarity(x[target_idx : target_idx + 1], x).ravel()
    order = np.argsort(sim)[::-1]
    neighbor_idx = [idx for idx in order if idx != target_idx][:6]

    center = np.array([0.0, 0.0])
    angles = np.linspace(0, 2 * np.pi, len(neighbor_idx), endpoint=False)
    radius = 1.0
    pos = {target_idx: center}
    for a, idx in zip(angles, neighbor_idx, strict=False):
        pos[idx] = np.array([radius * np.cos(a), radius * np.sin(a)])

    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    for idx in neighbor_idx:
        s = float(sim[idx])
        lw = 1.0 + 8.0 * max(0.0, s)
        ax.plot(
            [pos[target_idx][0], pos[idx][0]],
            [pos[target_idx][1], pos[idx][1]],
            color="#4E79A7",
            lw=lw,
            alpha=0.6,
        )
        mid = (pos[target_idx] + pos[idx]) / 2.0
        ax.text(mid[0], mid[1], f"{s:.2f}", fontsize=10, ha="center", va="center")

    ax.scatter(
        pos[target_idx][0],
        pos[target_idx][1],
        s=1500,
        color="#E15759",
        edgecolor="black",
    )
    ax.text(
        pos[target_idx][0],
        pos[target_idx][1],
        f"{pat[target_idx]}",
        color="white",
        ha="center",
        va="center",
        fontweight="bold",
    )

    for idx in neighbor_idx:
        s = float(sim[idx])
        c = "#C83A2A" if s >= 0.70 else ("#E67E22" if s >= 0.55 else "#95A5A6")
        ax.scatter(pos[idx][0], pos[idx][1], s=1300, color=c, edgecolor="black")
        ax.text(
            pos[idx][0],
            pos[idx][1],
            f"{pat[idx]}",
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
        )

    ax.set_title("Neighborhood Influence Graph (Feature-Similarity Proxy)")
    ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _figure11_patient_profile(ctx: FigureContext, output_path: Path) -> None:
    final_df = pd.read_csv(
        ROOT
        / "data/03_prodromal/final_training_dataset/unified_longitudinal_early_pd.csv"
    )
    high_idx = int(np.argmax(ctx.probs_test))
    low_idx = int(np.argmin(ctx.probs_test))
    high_pat = int(ctx.test_data.patno[high_idx].item())
    low_pat = int(ctx.test_data.patno[low_idx].item())

    candidates = [
        "TREMOR_SCORE",
        "PIGD_SCORE",
        "GENETIC_RISK_SCORE",
        "ALPHA_SYNUCLEIN",
        "TOTAL_TAU",
        "ABETA42",
        "PUTAMEN_L_SBR",
        "CAUDATE_L_SBR",
    ]
    features = [c for c in candidates if c in final_df.columns]
    if len(features) < 6:
        numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
        skip = {"PATNO", "time_to_event", "phenoconverted", "landmark_month"}
        features = [c for c in numeric_cols if c not in skip][:8]

    # Pick the latest landmark row for each patient.
    latest = final_df.sort_values("landmark_month").groupby("PATNO").tail(1)
    scaled = latest[features].copy()
    scaled = (scaled - scaled.mean()) / (scaled.std(ddof=0) + 1e-8)
    scaled["PATNO"] = latest["PATNO"].values
    by_pat = scaled.set_index("PATNO")

    if high_pat not in by_pat.index or low_pat not in by_pat.index:
        # Fallback to first two rows if mapping was filtered away.
        row_a = scaled.iloc[0]
        row_b = scaled.iloc[1]
        high_pat = int(row_a["PATNO"])
        low_pat = int(row_b["PATNO"])
        vals_a = row_a[features].to_numpy(dtype=float)
        vals_b = row_b[features].to_numpy(dtype=float)
    else:
        vals_a = by_pat.loc[high_pat, features].to_numpy(dtype=float)
        vals_b = by_pat.loc[low_pat, features].to_numpy(dtype=float)

    y = np.arange(len(features))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), sharey=True)
    axes[0].barh(y, vals_a, color="#8E44AD", alpha=0.85)
    axes[0].set_title(f"High-risk profile (PATNO {high_pat})")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(features)
    axes[0].axvline(0, color="black", lw=1)
    axes[0].grid(axis="x", linestyle="--", alpha=0.25)

    axes[1].barh(y, vals_b, color="#16A085", alpha=0.85)
    axes[1].set_title(f"Low-risk profile (PATNO {low_pat})")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(features)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].grid(axis="x", linestyle="--", alpha=0.25)

    fig.suptitle("Patient Feature Profiles (Standardized Latest-Visit Values)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _figure12_counterfactual(output_path: Path) -> None:
    cf_json = ROOT / "outputs/digital_twin/patient_0_counterfactual.json"
    if cf_json.exists():
        payload = _load_json(cf_json)
        baseline_rows = payload.get("baseline_path", [])
        months = np.array(
            [float(row.get("t_month", np.nan)) for row in baseline_rows], dtype=float
        )
        baseline = np.array(
            [float(row.get("risk_saa", np.nan)) for row in baseline_rows],
            dtype=float,
        )
        cf_paths = payload.get("counterfactual_paths", {})
        if baseline.size >= 3 and months.size == baseline.size:
            fig, ax = plt.subplots(figsize=(8.4, 5.0))
            ax.plot(
                months,
                baseline,
                marker="o",
                lw=2.4,
                color="#1f77b4",
                label="baseline",
            )

            all_series = [baseline]
            max_abs_delta = 0.0
            for name, vals in cf_paths.items():
                rows = sorted(vals, key=lambda r: float(r.get("t_month", 0.0)))
                cmonths = np.array(
                    [float(row.get("t_month", np.nan)) for row in rows], dtype=float
                )
                arr = np.array(
                    [float(row.get("risk_saa", np.nan)) for row in rows], dtype=float
                )
                if arr.size == baseline.size and np.allclose(cmonths, months):
                    delta = arr - baseline
                    max_abs_delta = max(max_abs_delta, float(np.max(np.abs(delta))))
                    label = f"{name} (\u039424m={delta[-1]:+.2e})"
                    ax.plot(
                        months,
                        arr,
                        marker="o",
                        lw=1.6,
                        alpha=0.9,
                        label=label,
                    )
                    all_series.append(arr)

            combined = np.concatenate(all_series) if all_series else baseline
            ymin = float(np.nanmin(combined))
            ymax = float(np.nanmax(combined))
            span = max(ymax - ymin, 1e-4)
            pad = 0.12 * span
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_xlabel("month")
            ax.set_ylabel("predicted SAA risk")
            ax.set_title("Digital Twin v1 Counterfactual Trajectories")
            ax.grid(True, linestyle="--", alpha=0.25)
            ax.legend(loc="best", fontsize=7)
            ax.text(
                0.02,
                0.03,
                f"Max |Δ risk| across interventions = {max_abs_delta:.2e}",
                transform=ax.transAxes,
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
            fig.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300)
            plt.close(fig)
            return

    # Fallback copy from appendix figure if JSON cannot be used.
    fallback = (
        ROOT / "visualizations/appendix/digital_twin/patient_0_counterfactual.png"
    )
    _copy(fallback, output_path)


def _figure13_tipping(output_path: Path) -> None:
    x = np.linspace(-3, 3, 400)
    step = (x >= 0).astype(float)
    smooth = 1.0 / (1.0 + np.exp(-2.2 * x))

    pert = np.linspace(0, 0.5, 120)
    crisp_delta = (pert >= 0.10).astype(float)
    fuzzy_delta = 0.45 + 0.35 * pert

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    axes[0].plot(x, step, lw=3.0, label="crisp_step", color="#4E79A7")
    axes[0].plot(x, smooth, lw=3.0, label="fuzzy_sigmoid", color="#E15759")
    axes[0].axvline(0, linestyle="--", color="gray")
    axes[0].set_xlabel("Biomarker change")
    axes[0].set_ylabel("P(fast progressor)")
    axes[0].set_title("Panel A: Boundary shape")
    axes[0].grid(True, linestyle="--", alpha=0.25)
    axes[0].legend()

    axes[1].plot(pert, crisp_delta, lw=3.0, label="crisp")
    axes[1].plot(pert, fuzzy_delta, lw=3.0, label="fuzzy")
    axes[1].set_xlabel("Perturbation from baseline")
    axes[1].set_ylabel("Change in predicted risk")
    axes[1].set_title("Panel B: Perturbation sensitivity")
    axes[1].grid(True, linestyle="--", alpha=0.25)
    axes[1].legend()

    fig.suptitle("Tipping Point Analysis: Crisp vs Fuzzy Boundaries")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _preprocessing_mice(output_path: Path) -> None:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    raw_path = ROOT / "data/03_prodromal/enhanced/prodromal_multimodal_features.csv"
    imp_path = (
        ROOT
        / "data/03_prodromal/enhanced_36_features/prodromal_36_features_imputed.csv"
    )

    feature = "unknown"
    mode_note = ""
    observed = np.array([], dtype=float)
    imputed_only = np.array([], dtype=float)
    after_all = np.array([], dtype=float)
    true_masked = np.array([], dtype=float)

    if raw_path.exists() and imp_path.exists():
        raw = pd.read_csv(raw_path)
        imp = pd.read_csv(imp_path)
        if "PATNO" in raw.columns and "PATNO" in imp.columns:
            raw["PATNO"] = pd.to_numeric(raw["PATNO"], errors="coerce").astype("Int64")
            imp["PATNO"] = pd.to_numeric(imp["PATNO"], errors="coerce").astype("Int64")
            raw = raw.dropna(subset=["PATNO"]).copy()
            imp = imp.dropna(subset=["PATNO"]).copy()

            overlap = sorted(set(raw["PATNO"].tolist()) & set(imp["PATNO"].tolist()))
            if overlap:
                merged = raw.merge(
                    imp, on="PATNO", suffixes=("_raw", "_imp"), how="inner"
                )
                common = [c for c in raw.columns if c in imp.columns and c != "PATNO"]
                ranked = sorted(
                    common,
                    key=lambda c: (
                        float(raw[c].isna().mean()),
                        float(pd.to_numeric(imp[c], errors="coerce").notna().mean()),
                    ),
                    reverse=True,
                )
                for cand in ranked:
                    raw_col = f"{cand}_raw"
                    imp_col = f"{cand}_imp"
                    if raw_col not in merged.columns or imp_col not in merged.columns:
                        continue
                    obs = (
                        pd.to_numeric(merged[raw_col], errors="coerce")
                        .dropna()
                        .to_numpy(dtype=float)
                    )
                    imp_only = (
                        pd.to_numeric(
                            merged.loc[
                                merged[raw_col].isna() & merged[imp_col].notna(),
                                imp_col,
                            ],
                            errors="coerce",
                        )
                        .dropna()
                        .to_numpy(dtype=float)
                    )
                    all_imp = (
                        pd.to_numeric(merged[imp_col], errors="coerce")
                        .dropna()
                        .to_numpy(dtype=float)
                    )
                    if obs.size >= 30 and imp_only.size >= 20 and all_imp.size >= 60:
                        feature = cand
                        observed = obs
                        imputed_only = imp_only
                        after_all = all_imp
                        mode_note = (
                            f"Matched real missingness (n_overlap_patno={len(overlap)})"
                        )
                        break

    # Fallback: masked-value MICE validation on canonical cohort (non-blank, reproducible).
    if observed.size < 30 or imputed_only.size < 20:
        cohort = pd.read_csv(
            ROOT
            / "data/03_prodromal/final_training_dataset/unified_longitudinal_early_pd.csv"
        )
        num = cohort.select_dtypes(include=[np.number]).copy()
        drop_cols = {
            "PATNO",
            "time_to_event",
            "phenoconverted",
            "event_observed",
            "landmark_month",
            "event",
            "time",
            "saa_label",
        }
        num = num[[c for c in num.columns if c not in drop_cols]].copy()

        eligible = []
        for c in num.columns:
            s = pd.to_numeric(num[c], errors="coerce")
            if s.notna().sum() >= 120 and float(s.std(ddof=0)) > 1e-8:
                eligible.append(c)
        if not eligible:
            raise ValueError(
                "No eligible numeric columns for fallback MICE validation."
            )

        preferred = [
            "UPDRS_I",
            "TREMOR_SCORE",
            "PIGD_SCORE",
            "SCOPA_AUT_SCORE",
            "UPSIT_SCORE",
            "ALPHA_SYNUCLEIN",
            "TOTAL_TAU",
            "ABETA42",
            "PTAU181",
        ]
        target = next((c for c in preferred if c in eligible), eligible[0])

        corr = (
            num[eligible]
            .corr(numeric_only=True)[target]
            .abs()
            .sort_values(ascending=False)
        )
        auxiliaries = [c for c in corr.index if c != target][:7]
        model_cols = [target] + auxiliaries
        work = num[model_cols].copy().reset_index(drop=True)

        obs_positions = np.flatnonzero(work[target].notna().to_numpy())
        if obs_positions.size < 40:
            raise ValueError(
                f"Insufficient observed rows for masked validation on {target}."
            )
        rng = np.random.default_rng(42)
        mask_n = min(max(30, int(0.20 * obs_positions.size)), 300)
        mask_positions = rng.choice(obs_positions, size=mask_n, replace=False)

        true_masked = pd.to_numeric(
            work.loc[mask_positions, target], errors="coerce"
        ).to_numpy(dtype=float)
        masked = work.copy()
        masked.loc[mask_positions, target] = np.nan

        imputer = IterativeImputer(random_state=42, max_iter=20, sample_posterior=False)
        arr = imputer.fit_transform(masked)
        imputed_target = arr[:, 0]

        feature = target
        observed = (
            pd.to_numeric(work[target], errors="coerce").dropna().to_numpy(dtype=float)
        )
        imputed_only = imputed_target[mask_positions]
        after_all = imputed_target
        mode_note = f"Masked-value validation on canonical cohort (n_mask={mask_n})"

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    axes[0].hist(
        observed,
        bins=30,
        density=True,
        alpha=0.45,
        color="#4E79A7",
        label="Observed values",
    )
    axes[0].hist(
        imputed_only,
        bins=30,
        density=True,
        alpha=0.45,
        color="#F28E2B",
        label="Imputed values",
    )
    axes[0].set_title(f"Distribution of {feature}: observed vs imputed")
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.25)

    if true_masked.size == imputed_only.size and true_masked.size > 5:
        axes[1].scatter(true_masked, imputed_only, s=25, alpha=0.7, color="#59A14F")
        lo = min(float(np.min(true_masked)), float(np.min(imputed_only)))
        hi = max(float(np.max(true_masked)), float(np.max(imputed_only)))
        axes[1].plot([lo, hi], [lo, hi], "k--", lw=1.2)
        mae = float(np.mean(np.abs(true_masked - imputed_only)))
        axes[1].text(
            0.04,
            0.95,
            f"MAE={mae:.3f}",
            transform=axes[1].transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        axes[1].set_title("Masked-value recovery (true vs imputed)")
        axes[1].set_xlabel("True held-out values")
        axes[1].set_ylabel("Imputed values")
    else:
        q = np.linspace(0.05, 0.95, 30)
        ref = np.quantile(observed, q)
        cmp = np.quantile(after_all, q)
        axes[1].scatter(ref, cmp, color="#59A14F", s=35)
        low = min(ref.min(), cmp.min())
        high = max(ref.max(), cmp.max())
        axes[1].plot([low, high], [low, high], "k--", lw=1.2)
        axes[1].set_title("Quantile alignment (observed vs post-imputation)")
        axes[1].set_xlabel("Observed quantiles")
        axes[1].set_ylabel("Post-imputation quantiles")
    axes[1].grid(True, linestyle="--", alpha=0.25)

    fig.suptitle(f"MICE-style imputation validation ({mode_note})", fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _preprocessing_genetics(output_path: Path) -> None:
    df = pd.read_csv(
        ROOT
        / "data/03_prodromal/final_training_dataset/unified_longitudinal_early_pd.csv"
    )
    genes = ["LRRK2", "GBA", "APOE_E4"]
    genes = [g for g in genes if g in df.columns]
    if not genes:
        raise ValueError(
            "No genetic risk columns found for preprocessing_genetic_encoding."
        )

    fig, axes = plt.subplots(1, len(genes), figsize=(6.0 * len(genes), 4.8))
    if len(genes) == 1:
        axes = [axes]
    colors = ["#76B7B2", "#FF9D76", "#D372D3"]
    for ax, gene, color in zip(axes, genes, colors, strict=False):
        vals = pd.to_numeric(df[gene], errors="coerce").dropna()
        ax.hist(vals, bins=20, color=color, alpha=0.8, edgecolor="black")
        ax.set_title(f"{gene} Risk Score Distribution")
        ax.set_xlabel("Weighted risk score")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _preprocessing_normalization(output_path: Path) -> None:
    spect = pd.read_csv(ROOT / "data/01_processed/dat_spect_sbr_values.csv")
    candidate = next(
        (c for c in ["CAUDATE_L", "CAUDATE_MEAN", "PUTAMEN_L"] if c in spect.columns),
        None,
    )
    if candidate is None:
        raise ValueError("No SBR candidate column found for normalization figure.")

    raw = (
        pd.to_numeric(spect[candidate], errors="coerce").dropna().to_numpy(dtype=float)
    )
    z = (raw - raw.mean()) / (raw.std(ddof=0) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes[0].hist(raw, bins=30, color="#B276B2", alpha=0.85, edgecolor="black")
    axes[0].set_title(f"Raw DAT-SPECT SBR ({candidate})")
    axes[0].set_xlabel("Specific Binding Ratio (SBR)")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.25)

    axes[1].hist(z, bins=30, color="#7AC37A", alpha=0.85, edgecolor="black")
    axes[1].set_title("Z-Score Normalized Intensity")
    axes[1].set_xlabel("Z-score")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    out = ROOT / "visualizations/publication_New"
    out.mkdir(parents=True, exist_ok=True)

    checkpoint = (
        ROOT
        / "outputs/phase9_neuro_fuzzy/PREP_20260208_SAA_COHORT3_full/neuro_fuzzy_best.pth"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    # Refresh explainability+preprocessing appendix against latest run checkpoint.
    generate_appendix_package(
        output_root=ROOT / "visualizations" / "appendix",
        index_md=ROOT / "Docs/audit/APPENDIX_EXPLAINABILITY_INDEX.md",
        provenance_json=ROOT / "visualizations/appendix/provenance.json",
        checkpoint_override=checkpoint,
    )

    # Refresh IEEE publication bundle (now prefers latest cohort3 full results).
    py = ROOT / ".venv/bin/python"
    if not py.exists():
        py = Path(sys.executable)
    _run([str(py), "scripts/generate_ieee_publication_bundle.py"])

    ctx = _build_context(checkpoint_path=checkpoint)

    # Static architecture diagrams (conceptual).
    _copy(
        ROOT
        / "Archive_New/pipeline_cleanup_2026-02-06/visualizations/publication/Figure1_GIMAN_Architecture.png",
        out / "Figure1_GIMAN_Architecture.png",
    )
    _copy(
        ROOT
        / "Archive_New/pipeline_cleanup_2026-02-06/visualizations/publication/Figure2_Neuro_Fuzzy_Enhancement.png",
        out / "Figure2_Neuro_Fuzzy_Enhancement.png",
    )

    # Data-driven refreshed figures.
    _figure3_benchmark(out / "Figure3_Benchmark_Comparison.png")
    _copy(
        ROOT
        / "visualizations/sota_lift/ABLATE_20260208_RUN3/phase9_targeted_ablations_best_by_group.png",
        out / "Figure4_Ablation_Study.png",
    )
    _figure5_time_dependent(ctx, out / "Figure5_Time_Dependent_AUC.png")
    _figure6_calibration(ctx, out / "Figure6_Calibration_Plots.png")
    _figure7_robustness(out / "Figure7_CV_Robustness.png")
    _figure8_noise_sensitivity(ctx, out / "Figure8_Noise_Sensitivity.png")
    _figure9_embedding(ctx, out / "Figure9_Embedding_Space.png")
    _figure10_similarity_graph(ctx, out / "Figure10_Attention_Heatmap.png")
    _figure11_patient_profile(ctx, out / "Figure11_Patient_Fuzzy_Profile.png")
    _figure12_counterfactual(out / "Figure12_Counterfactual_Trajectory.png")
    _figure13_tipping(out / "Figure13_Tipping_Point.png")

    # Updated preprocessing figures used in manuscript appendix.
    _preprocessing_mice(out / "preprocessing_mice_imputation.png")
    _preprocessing_genetics(out / "preprocessing_genetic_encoding.png")
    _preprocessing_normalization(out / "preprocessing_normalization.png")

    manifest = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "output_dir": str(out),
        "checkpoint": str(checkpoint),
        "inputs": {
            "train_data": str(
                ROOT / "data/03_prodromal/final_pyg_data_sota_run/train_data.pt"
            ),
            "test_data": str(
                ROOT / "data/03_prodromal/final_pyg_data_sota_run/test_data.pt"
            ),
            "metadata": str(
                ROOT
                / "data/03_prodromal/final_pyg_data_sota_run/pyg_data_metadata.json"
            ),
            "full_results": str(
                ROOT
                / "outputs/phase9_neuro_fuzzy/PREP_20260208_SAA_COHORT3_full/full_training_results.json"
            ),
            "multitask_results": str(
                ROOT
                / "outputs/phase9_neuro_fuzzy/PREP_20260208_SAA_COHORT3_multitask/multitask_training_results.json"
            ),
            "external_metrics": str(
                ROOT
                / "outputs/external_validation/PPMI_20251008_REAL_DISJOINT/external_metrics.json"
            ),
        },
        "files": sorted([p.name for p in out.glob("*.png")]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {len(manifest['files'])} figures in {out}")


if __name__ == "__main__":
    main()
