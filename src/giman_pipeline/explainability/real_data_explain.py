from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from giman_pipeline.sota.metrics import brier, expected_calibration_error, safe_pr_auc


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _load_nf_model(in_features: int, checkpoint_path: Path, device: torch.device):
    root = _repo_root()
    phase8_dir = (
        root / "archive" / "development" / "phase8" / "subphase8_2_dynamic_endpoints"
    )
    if str(phase8_dir) not in sys.path:
        sys.path.append(str(phase8_dir))
    if str(root) not in sys.path:
        sys.path.append(str(root))

    from train_final_giman_survival import GIMANSurvivalGAT

    from archive.development.phase9.neuro_fuzzy import NeuroFuzzyGIMAN

    gat = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128)
    model = NeuroFuzzyGIMAN(gat, num_classes=2, num_rules=32).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def _predict_probs(model, data) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits, rule_weights = model(data)
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    return probs, rule_weights.detach().cpu().numpy()


def _fit_calibrators(
    probs: np.ndarray,
    y_true: np.ndarray,
) -> tuple[LogisticRegression, IsotonicRegression]:
    x = probs.reshape(-1, 1)
    platt = LogisticRegression(C=1e6, solver="lbfgs")
    platt.fit(x, y_true.astype(int))

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(probs, y_true.astype(int))
    return platt, isotonic


def _calibrate_scores(
    probs: np.ndarray,
    platt: LogisticRegression,
    isotonic: IsotonicRegression,
) -> tuple[np.ndarray, np.ndarray]:
    platt_probs = platt.predict_proba(probs.reshape(-1, 1))[:, 1]
    isotonic_probs = isotonic.predict(probs)
    return platt_probs, isotonic_probs


def _permutation_importance(
    model,
    data,
    y_true: np.ndarray,
    baseline_auc: float,
    feature_names: list[str],
) -> pd.DataFrame:
    x = data.x.detach().cpu().numpy()
    drops: list[float] = []
    rng = np.random.default_rng(42)

    for i in range(x.shape[1]):
        x_perm = x.copy()
        rng.shuffle(x_perm[:, i])
        temp = data.clone()
        temp.x = torch.tensor(x_perm, dtype=torch.float32, device=data.x.device)
        with torch.no_grad():
            logits, _ = model(temp)
            prob = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        auc_perm = _safe_auc(y_true, prob)
        drops.append(float(baseline_auc - auc_perm))

    return pd.DataFrame(
        {
            "feature": feature_names,
            "auc_drop": drops,
        }
    ).sort_values("auc_drop", ascending=False)


def run_real_data_explainability(
    train_data_path: Path,
    test_data_path: Path,
    metadata_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = metadata.get("feature_names", [])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = torch.load(train_data_path, weights_only=False).to(device)
    test_data = torch.load(test_data_path, weights_only=False).to(device)
    if not hasattr(train_data, "patno"):
        raise ValueError(
            "train_data must include patno for patient-level calibration split"
        )

    y_true = test_data.saa_label.detach().cpu().numpy().astype(int)
    y_train = train_data.saa_label.detach().cpu().numpy().astype(int)

    model = _load_nf_model(
        in_features=int(test_data.x.shape[1]),
        checkpoint_path=checkpoint_path,
        device=device,
    )

    probs, rule_weights = _predict_probs(model, test_data)
    probs_train, _ = _predict_probs(model, train_data)

    auc = _safe_auc(y_true, probs)
    pr_auc = safe_pr_auc(y_true, probs)

    # Fit calibration models on patient-disjoint calibration subset from training graph.
    pat_train = train_data.patno.detach().cpu().numpy().astype(int)
    patient_df = pd.DataFrame({"patno": pat_train, "label": y_train}).groupby(
        "patno", as_index=False
    )["label"].max()
    stratify = patient_df["label"].to_numpy() if patient_df["label"].nunique() > 1 else None
    try:
        val_pat, _ = train_test_split(
            patient_df["patno"].to_numpy(),
            test_size=0.5,
            random_state=42,
            stratify=stratify,
        )
    except ValueError:
        # Fallback when a class has too few samples for stratified split.
        val_pat, _ = train_test_split(
            patient_df["patno"].to_numpy(),
            test_size=0.5,
            random_state=42,
            stratify=None,
        )
    cal_mask = np.isin(pat_train, val_pat)
    probs_cal = probs_train[cal_mask]
    y_cal = y_train[cal_mask]
    platt, isotonic = _fit_calibrators(probs_cal, y_cal)
    probs_platt, probs_iso = _calibrate_scores(probs, platt, isotonic)

    # Permutation importance
    pi_df = _permutation_importance(model, test_data, y_true, auc, feature_names)
    pi_path = output_dir / "feature_permutation_importance.csv"
    pi_df.to_csv(pi_path, index=False)

    # Top feature bar plot
    top = pi_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"][::-1], top["auc_drop"][::-1], color="#4E79A7")
    ax.set_title("Permutation Importance (AUC Drop)")
    ax.set_xlabel("AUC drop after feature permutation")
    fig.tight_layout()
    top_fig = output_dir / "feature_importance_top20.png"
    fig.savefig(top_fig, dpi=300)
    plt.close(fig)

    # Fuzzy rule activation heatmap
    rw = rule_weights
    n_rows = min(40, rw.shape[0])
    idx = np.argsort(probs)[-n_rows:]
    rule_var = np.var(rw, axis=0)
    top_rules = np.argsort(rule_var)[::-1][:12]
    heat = rw[idx][:, top_rules]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(heat, aspect="auto", cmap="viridis")
    ax.set_title("Top Rule Activations (Highest-risk patients)")
    ax.set_xlabel("Rule ID")
    ax.set_ylabel("Patient rank")
    ax.set_xticks(np.arange(len(top_rules)))
    ax.set_xticklabels([str(int(r)) for r in top_rules], rotation=45)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    rule_fig = output_dir / "fuzzy_rule_activation_heatmap.png"
    fig.savefig(rule_fig, dpi=300)
    plt.close(fig)

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(
        y_true, probs, n_bins=10, strategy="quantile"
    )
    frac_pos_platt, mean_pred_platt = calibration_curve(
        y_true, probs_platt, n_bins=10, strategy="quantile"
    )
    frac_pos_iso, mean_pred_iso = calibration_curve(
        y_true, probs_iso, n_bins=10, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.plot(mean_pred, frac_pos, "o-", color="#E15759", label="FUZZY GIMAN (raw)")
    ax.plot(
        mean_pred_platt,
        frac_pos_platt,
        "o-",
        color="#4E79A7",
        label="Platt calibrated",
    )
    ax.plot(
        mean_pred_iso,
        frac_pos_iso,
        "o-",
        color="#59A14F",
        label="Isotonic calibrated",
    )
    ax.set_title("SAA Calibration Curve (Test split)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend()
    fig.tight_layout()
    cal_fig = output_dir / "saa_calibration_curve.png"
    fig.savefig(cal_fig, dpi=300)
    plt.close(fig)

    summary = {
        "n_test": int(len(y_true)),
        "auc": auc,
        "pr_auc": pr_auc,
        "calibration_metrics": {
            "raw": {
                "ece": expected_calibration_error(y_true, probs, n_bins=10),
                "brier": brier(y_true, probs),
            },
            "platt": {
                "ece": expected_calibration_error(y_true, probs_platt, n_bins=10),
                "brier": brier(y_true, probs_platt),
            },
            "isotonic": {
                "ece": expected_calibration_error(y_true, probs_iso, n_bins=10),
                "brier": brier(y_true, probs_iso),
            },
        },
        "checkpoint": str(checkpoint_path),
        "feature_importance_csv": str(pi_path),
        "feature_importance_fig": str(top_fig),
        "rule_heatmap_fig": str(rule_fig),
        "calibration_fig": str(cal_fig),
    }
    (output_dir / "explainability_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    root = _repo_root()
    run_real_data_explainability(
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
        checkpoint_path=root
        / "outputs"
        / "phase9_neuro_fuzzy_sota_run_from50ckpt"
        / "neuro_fuzzy_best.pth",
        output_dir=root / "visualizations" / "appendix" / "explainability",
    )
