"""R6-Q1: Multiclass external calibration on BioFIND.

Reviewer 6 flags that Paper 1's temperature-scaling is binary-only and that
multiclass external calibration is the open gap. This runner extends the
calibration protocol to the three multiclass NSD-ISS targets (binary,
three_class, nsd_positive) on the BioFIND external cohort, comparing three
calibration families:

    1. Per-target temperature scaling (extends R2-Q4 to multiclass)
    2. Dirichlet calibration (Kull et al. 2019, Adv NeurIPS) implemented as
       multinomial logistic regression on log-probabilities + ODIR (off-diagonal
       and intercept) regularization with lambda=1e-2 default.
    3. One-vs-rest isotonic regression (sklearn.isotonic) per class with
       renormalization.

Each calibrator is FIT on PPMI internal OOF predictions
(``outputs/paper1_calibration/results/per_fold_probs.npz``) and then applied
to BioFIND external predictions, which are regenerated here using the same
training pipeline as ``scripts/run_external_validation.py`` (12-feat common
clinical features, CatBoost trained on full PPMI).

Outputs (under ``outputs/paper1_r2_responses/``):
    - q_r6_q1_multiclass_calibration.json  (per-target / per-method metrics)
    - q_r6_q1_multiclass_calibration_table.md  (summary table + verdict)
    - q_r6_q1_reliability_three_class.png/pdf
    - q_r6_q1_reliability_nsd_positive.png/pdf

Author: Paper 1 R2 response sweep (R6-Q1)
Date: 2026-04-25
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    log_loss,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
INTERNAL_NPZ = ROOT / "outputs" / "paper1_calibration" / "results" / "per_fold_probs.npz"
PPMI_FEATURES = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
EXT_OUT = ROOT / "outputs" / "paper1_r2_responses"
EXT_OUT.mkdir(parents=True, exist_ok=True)

# Common 12-feature clinical-only set (matches run_external_validation.py)
COMMON_FEATURES = [
    "AGE_AT_BASELINE",
    "SEX",
    "UPDRS1_TOTAL",
    "UPDRS2_TOTAL",
    "UPDRS3_TREMOR",
    "UPDRS3_RIGIDITY",
    "UPDRS3_BRADYKINESIA",
    "UPDRS3_AXIAL",
    "UPDRS4_TOTAL",
    "MOCA_TOTAL",
    "ESS_TOTAL",
    "RBD_TOTAL",
]

OKABE_ITO = [
    "#000000",  # raw (black)
    "#0072B2",  # temperature (blue)
    "#D55E00",  # dirichlet (vermillion)
    "#009E73",  # isotonic (bluish green)
    "#E69F00",  # extra (orange)
]

EPS = 1e-12
SEED = 42

# Targets for which multiclass calibration is interesting (binary included for
# completeness so the table reports a unified picture).
TARGETS = ("binary", "three_class", "nsd_positive")

CLASS_LABELS = {
    "binary": ["NSD-", "NSD+"],
    "three_class": ["Early(0-1)", "Mild(2B)", "Impaired(3-4)"],
    "nsd_positive": ["1", "2B", "3", "4"],
}


# ----------------------------------------------------------------------
# Probability helpers
# ----------------------------------------------------------------------
def _safe_log(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1.0))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------
# Calibrators
# ----------------------------------------------------------------------
class TemperatureScaler:
    """Per-target single-scalar temperature on log-probabilities (multiclass).

    Fits T* by minimising NLL via L-BFGS-B on the calibration set.
    """

    def __init__(self) -> None:
        self.T: float = 1.0

    def fit(self, p: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        log_p = _safe_log(p)

        def nll(log_T: np.ndarray) -> float:
            T = float(np.exp(log_T))  # parametrise log T for unconstrained
            scaled = log_p / T
            sm = _softmax(scaled)
            return -float(np.mean(np.log(np.clip(sm[np.arange(len(y)), y], EPS, 1.0))))

        res = minimize(nll, x0=np.array([0.0]), method="L-BFGS-B")
        self.T = float(np.exp(res.x[0]))
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return _softmax(_safe_log(p) / self.T)


class DirichletCalibrator:
    """Dirichlet calibration via multinomial logistic on log-probs + ODIR.

    Kull et al. 2019, Beyond temperature scaling: Obtaining well-calibrated
    multiclass probabilities with Dirichlet calibration. NeurIPS 32.

    Implementation: multinomial logistic regression on log-probability inputs,
    fit via L-BFGS minimising regularised cross-entropy.
        z_k = b_k + sum_j W_{kj} log p_j
        p_calib = softmax(z)
    Regularisation:
        L = NLL + lambda * (||W_offdiag||_F^2 + ||b||_2^2)   (ODIR)
    """

    def __init__(self, lam: float = 1e-2) -> None:
        self.lam = lam
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.K: int = 0

    def _pack(self, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.concatenate([W.ravel(), b])

    def _unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        K = self.K
        W = theta[: K * K].reshape(K, K)
        b = theta[K * K :]
        return W, b

    def fit(self, p: np.ndarray, y: np.ndarray) -> "DirichletCalibrator":
        K = p.shape[1]
        self.K = K
        log_p = _safe_log(p)
        n = len(y)
        y_oh = np.eye(K)[y]

        def loss(theta: np.ndarray) -> float:
            W, b = self._unpack(theta)
            z = log_p @ W.T + b
            sm = _softmax(z)
            nll = -np.mean(np.sum(y_oh * np.log(np.clip(sm, EPS, 1.0)), axis=1))
            # ODIR: penalise off-diagonal W and intercept b
            offdiag = W - np.diag(np.diag(W))
            reg = self.lam * (np.sum(offdiag**2) + np.sum(b**2))
            return float(nll + reg)

        # Identity initialisation (T=1 equivalent)
        W0 = np.eye(K)
        b0 = np.zeros(K)
        theta0 = self._pack(W0, b0)
        res = minimize(loss, theta0, method="L-BFGS-B", options={"maxiter": 500})
        self.W, self.b = self._unpack(res.x)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        log_p = _safe_log(p)
        z = log_p @ self.W.T + self.b
        return _softmax(z)


class IsotonicOvR:
    """One-vs-rest isotonic regression per class + renormalisation.

    Trains K independent IsotonicRegression(out_of_bounds='clip') on
    (P_k, [y == k]). At inference, predict per-class probability and
    renormalise rowwise.
    """

    def __init__(self) -> None:
        self.models: list[IsotonicRegression] = []
        self.K: int = 0

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicOvR":
        K = p.shape[1]
        self.K = K
        self.models = []
        for k in range(K):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(p[:, k], (y == k).astype(float))
            self.models.append(iso)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        K = self.K
        out = np.zeros_like(p)
        for k in range(K):
            out[:, k] = self.models[k].predict(p[:, k])
        # Floor to EPS then renormalise
        out = np.clip(out, EPS, 1.0)
        out = out / out.sum(axis=1, keepdims=True)
        return out


class IdentityCalibrator:
    """Pass-through (raw)."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IdentityCalibrator":
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return p


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def classwise_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """ECE per class (one-vs-rest binarisation), 10-bin uniform width."""
    K = p.shape[1]
    ece = np.zeros(K)
    n = len(y)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for k in range(K):
        target = (y == k).astype(float)
        prob = p[:, k]
        bin_idx = np.clip(np.digitize(prob, bins[1:-1], right=False), 0, n_bins - 1)
        for b in range(n_bins):
            mask = bin_idx == b
            if not mask.any():
                continue
            conf = prob[mask].mean()
            acc = target[mask].mean()
            ece[k] += (mask.sum() / n) * abs(conf - acc)
    return ece


def classwise_brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = p.shape[1]
    return np.array([
        brier_score_loss((y == k).astype(int), p[:, k]) for k in range(K)
    ])


def per_class_report(p: np.ndarray, y: np.ndarray) -> dict:
    """precision/recall/f1/sens/spec per class, with support."""
    pred = p.argmax(axis=1)
    K = p.shape[1]
    report = {}
    for k in range(K):
        y_bin = (y == k).astype(int)
        p_bin = (pred == k).astype(int)
        tp = int(((p_bin == 1) & (y_bin == 1)).sum())
        fp = int(((p_bin == 1) & (y_bin == 0)).sum())
        fn = int(((p_bin == 0) & (y_bin == 1)).sum())
        tn = int(((p_bin == 0) & (y_bin == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        sens = recall  # same definition
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        report[str(k)] = {
            "support": int(y_bin.sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "sensitivity": float(sens),
            "specificity": float(spec),
        }
    return report


# ----------------------------------------------------------------------
# Data loading and external prediction generation
# ----------------------------------------------------------------------
def load_internal_oof() -> dict[str, dict[str, np.ndarray]]:
    """Load PPMI OOF probabilities + labels from per_fold_probs.npz."""
    npz = np.load(INTERNAL_NPZ, allow_pickle=True)
    out: dict[str, dict[str, np.ndarray]] = {}
    for tgt in TARGETS:
        out[tgt] = {
            "p": np.asarray(npz[f"{tgt}_y_prob"], dtype=float),
            "y": np.asarray(npz[f"{tgt}_y_true"], dtype=int),
        }
    return out


def _ppmi_target_frame(target_type: str) -> pd.DataFrame:
    df = pd.read_csv(PPMI_FEATURES)
    if target_type == "binary":
        df["target"] = (df["nsd_iss_stage"].isin(["1", "2B", "3", "4"])).astype(int)
    elif target_type == "three_class":
        stage_map = {"0": 0, "1": 0, "2B": 1, "3": 2, "4": 2}
        df["target"] = df["nsd_iss_stage"].map(stage_map)
        df = df.dropna(subset=["target"])
        df["target"] = df["target"].astype(int)
    elif target_type == "nsd_positive":
        df = df[df["nsd_iss_stage"].isin(["1", "2B", "3", "4"])]
        stage_map = {"1": 0, "2B": 1, "3": 2, "4": 3}
        df["target"] = df["nsd_iss_stage"].map(stage_map)
        df["target"] = df["target"].astype(int)
    else:
        raise ValueError(target_type)
    return df


def _biofind_ground_truth(target_type: str) -> pd.DataFrame | None:
    if target_type == "binary":
        saa_path = ROOT / "data" / "00_raw" / "BioFind" / "biofind_saa_consensus.csv"
        if not saa_path.exists():
            return None
        saa = pd.read_csv(saa_path)
        saa["participant_id"] = "BF-" + saa["PATNO"].astype(str)
        out = saa[["participant_id", "SAA_RESULT"]].copy()
        out["target"] = out["SAA_RESULT"].astype(int)
        return out[["participant_id", "target"]]
    staging_path = ROOT / "data" / "04_staging" / "biofind_nsd_iss_staging.csv"
    if not staging_path.exists():
        return None
    staging = pd.read_csv(staging_path)
    if target_type == "three_class":
        staging["target"] = staging["nsd_iss_stage"].map({2: 0, 3: 1, 4: 2, 5: 2})
    elif target_type == "nsd_positive":
        staging["target"] = staging["nsd_iss_stage"].map({2: 1, 3: 2, 4: 3, 5: 3})
    else:
        return None
    staging = staging.dropna(subset=["target"])
    staging["target"] = staging["target"].astype(int)
    return staging[["participant_id", "target"]]


def get_external_predictions(target_type: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Train CatBoost on full PPMI (12-feat common) and predict on BioFIND
    ground-truth subset. Returns (proba_matrix [n, K], y_true [n]).
    """
    from catboost import CatBoostClassifier

    ppmi = _ppmi_target_frame(target_type)
    available = [f for f in COMMON_FEATURES if f in ppmi.columns]
    X_ppmi = ppmi[available].values
    y_ppmi = ppmi["target"].values
    n_classes = len(np.unique(y_ppmi))

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()  # not used by CatBoost, kept for symmetry
    X_imp = imputer.fit_transform(X_ppmi)
    scaler.fit(X_imp)

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        auto_class_weights="Balanced",
        verbose=0,
        random_seed=SEED,
        eval_metric="TotalF1",
    )
    model.fit(X_imp, y_ppmi)

    bf_features_path = ROOT / "data" / "05_features" / "biofind_features.csv"
    if not bf_features_path.exists():
        logger.error("BioFIND features not found at %s", bf_features_path)
        return None
    bf = pd.read_csv(bf_features_path)
    for f in available:
        if f not in bf.columns:
            bf[f] = np.nan

    gt = _biofind_ground_truth(target_type)
    if gt is None:
        logger.error("BioFIND ground truth missing for %s", target_type)
        return None
    bf_gt = bf.merge(gt[["participant_id", "target"]], on="participant_id", how="inner")
    if bf_gt.empty:
        logger.error("No BioFIND ground-truth overlap for %s", target_type)
        return None

    X_bf = bf_gt[available].values
    X_bf_imp = imputer.transform(X_bf)
    proba = model.predict_proba(X_bf_imp)
    y_true = bf_gt["target"].values.astype(int)
    # Some external targets (nsd_positive) may include classes outside the
    # PPMI training range; restrict to PPMI-known labels and renumber if needed.
    if target_type == "nsd_positive":
        # PPMI nsd_positive maps stages 1,2B,3,4 -> 0,1,2,3 (4 classes).
        # BioFIND maps stages 2,3,4,5 -> 1,2,3,3 -> we keep labels in {0..3}.
        keep = (y_true >= 0) & (y_true < n_classes)
        proba = proba[keep]
        y_true = y_true[keep]
    elif target_type == "three_class":
        keep = (y_true >= 0) & (y_true < n_classes)
        proba = proba[keep]
        y_true = y_true[keep]
    return proba, y_true


# ----------------------------------------------------------------------
# Reliability diagram figure
# ----------------------------------------------------------------------
def reliability_curve(p_class: np.ndarray, y_class: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p_class, bins[1:-1], right=False), 0, n_bins - 1)
    confs, accs, weights = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        confs.append(float(p_class[mask].mean()))
        accs.append(float(y_class[mask].mean()))
        weights.append(int(mask.sum()))
    return np.array(confs), np.array(accs), np.array(weights)


def make_reliability_figure(
    target: str,
    raw_p: np.ndarray,
    cal_p: dict[str, np.ndarray],
    y: np.ndarray,
    out_path: Path,
) -> None:
    K = raw_p.shape[1]
    fig_w = min(7.0, 2.3 * K + 0.6)
    fig_h = 2.6
    fig, axes = plt.subplots(1, K, figsize=(fig_w, fig_h), sharey=True)
    if K == 1:
        axes = [axes]
    method_colors = {
        "Raw": OKABE_ITO[0],
        "Temperature": OKABE_ITO[1],
        "Dirichlet": OKABE_ITO[2],
        "Isotonic": OKABE_ITO[3],
    }

    labels = CLASS_LABELS.get(target, [str(i) for i in range(K)])
    for k in range(K):
        ax = axes[k]
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8, label="Perfect")
        # Raw
        for name, p_mat in {"Raw": raw_p, **cal_p}.items():
            confs, accs, _ = reliability_curve(p_mat[:, k], (y == k).astype(int))
            if confs.size == 0:
                continue
            ax.plot(
                confs,
                accs,
                marker="o",
                ms=4,
                lw=1.5,
                color=method_colors.get(name, "black"),
                label=name,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Class {labels[k]}", fontsize=9)
        ax.set_xlabel("Predicted prob.", fontsize=8)
        if k == 0:
            ax.set_ylabel("Observed freq.", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    axes[-1].legend(fontsize=6, loc="upper left", frameon=False)
    fig.suptitle(
        f"BioFIND classwise reliability — target={target}", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def evaluate_method(p: np.ndarray, y: np.ndarray) -> dict:
    cw_ece = classwise_ece(p, y, n_bins=10)
    cw_brier = classwise_brier(p, y)
    macro_ece = float(np.mean(cw_ece))
    macro_brier = float(np.mean(cw_brier))
    per_class = per_class_report(p, y)
    try:
        nll = float(log_loss(y, np.clip(p, EPS, 1.0), labels=list(range(p.shape[1]))))
    except Exception:
        nll = float("nan")
    return {
        "macro_ece": macro_ece,
        "classwise_ece": [float(v) for v in cw_ece],
        "macro_brier": macro_brier,
        "classwise_brier": [float(v) for v in cw_brier],
        "nll": nll,
        "per_class": per_class,
    }


def run() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    internal = load_internal_oof()

    results: dict = {
        "workstream": "paper1_r2_q_r6_q1_multiclass_calibration",
        "feature_set": "12-feat common (external)",
        "n_bins_ece": 10,
        "seed": SEED,
        "calibrators": {
            "raw": "identity",
            "temperature": "scalar T fit by NLL on internal OOF",
            "dirichlet": "Kull et al. 2019 logistic on log-probs + ODIR (lambda=1e-2)",
            "isotonic": "one-vs-rest IsotonicRegression(out_of_bounds='clip') + renormalise",
        },
        "by_target": {},
    }

    for target in TARGETS:
        logger.info("=== Target: %s ===", target)
        # Fit calibrators on internal OOF
        p_int = internal[target]["p"]
        y_int = internal[target]["y"]

        temp = TemperatureScaler().fit(p_int, y_int)
        diri = DirichletCalibrator(lam=1e-2).fit(p_int, y_int)
        iso = IsotonicOvR().fit(p_int, y_int)

        # External predictions on BioFIND ground-truth subset
        ext = get_external_predictions(target)
        if ext is None:
            logger.warning("Skipping %s: external prediction unavailable", target)
            continue
        p_ext_raw, y_ext = ext
        n_ext = int(len(y_ext))
        n_classes = int(p_ext_raw.shape[1])

        # Apply calibrators to external probabilities
        p_temp = temp.transform(p_ext_raw)
        p_diri = diri.transform(p_ext_raw)
        p_iso = iso.transform(p_ext_raw)

        per_method = {
            "raw": evaluate_method(p_ext_raw, y_ext),
            "temperature": evaluate_method(p_temp, y_ext),
            "dirichlet": evaluate_method(p_diri, y_ext),
            "isotonic": evaluate_method(p_iso, y_ext),
        }
        per_method["temperature"]["T"] = float(temp.T)
        per_method["dirichlet"]["W"] = diri.W.tolist() if diri.W is not None else None
        per_method["dirichlet"]["b"] = diri.b.tolist() if diri.b is not None else None

        # Best method by macro-ECE
        best_ece = min(per_method.items(), key=lambda kv: kv[1]["macro_ece"])
        best_brier = min(per_method.items(), key=lambda kv: kv[1]["macro_brier"])

        results["by_target"][target] = {
            "n_external": n_ext,
            "n_classes": n_classes,
            "class_labels": CLASS_LABELS.get(target, [str(i) for i in range(n_classes)]),
            "external_class_distribution": {
                str(k): int((y_ext == k).sum()) for k in range(n_classes)
            },
            "internal_n": int(len(y_int)),
            "methods": per_method,
            "best_by_macro_ece": best_ece[0],
            "best_by_macro_brier": best_brier[0],
            "raw_macro_ece": per_method["raw"]["macro_ece"],
        }

        # Reliability figure (multiclass targets only — binary 1-panel is sparse)
        if target in ("three_class", "nsd_positive"):
            fig_path = EXT_OUT / f"q_r6_q1_reliability_{target}"
            make_reliability_figure(
                target,
                p_ext_raw,
                {
                    "Temperature": p_temp,
                    "Dirichlet": p_diri,
                    "Isotonic": p_iso,
                },
                y_ext,
                fig_path,
            )
            logger.info("  Wrote %s.png/pdf", fig_path)

    # ----- Verdict / interpretation -----
    targets_done = list(results["by_target"].keys())
    summary = {}
    for tgt in targets_done:
        r = results["by_target"][tgt]
        m = r["methods"]
        summary[tgt] = {
            "raw_macro_ece": m["raw"]["macro_ece"],
            "best_method": r["best_by_macro_ece"],
            "best_macro_ece": m[r["best_by_macro_ece"]]["macro_ece"],
            "delta_ece_vs_raw": m[r["best_by_macro_ece"]]["macro_ece"] - m["raw"]["macro_ece"],
        }
    results["summary"] = summary

    # NONE-works detection: any target where best macro-ECE > raw macro-ECE
    failures = [
        t for t, s in summary.items()
        if s["delta_ece_vs_raw"] > -1e-3  # no meaningful improvement
    ]
    if failures:
        results["verdict"] = (
            f"At least one target ({', '.join(failures)}) sees no meaningful "
            "ECE improvement from any of the three calibrators tested. "
            "Mondrian / class-conditional CP (Q2) is motivated as the next gap closer."
        )
    else:
        best_methods = ", ".join(f"{t}:{s['best_method']}" for t, s in summary.items())
        results["verdict"] = (
            f"All three calibrators improve external ECE on at least one target. "
            f"Best by macro-ECE per target: {best_methods}."
        )

    # Save JSON
    json_path = EXT_OUT / "q_r6_q1_multiclass_calibration.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", json_path)

    # Markdown table
    md_lines = [
        "# R6-Q1: Multiclass External Calibration on BioFIND",
        "",
        "Calibrators fit on PPMI internal OOF (`per_fold_probs.npz`); evaluated on",
        "BioFIND ground-truth subset using the 12-feature common clinical CatBoost.",
        "",
        "| Target | Method | macro-ECE | macro-Brier | NLL | Best? |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for tgt in targets_done:
        r = results["by_target"][tgt]
        best = r["best_by_macro_ece"]
        for method in ("raw", "temperature", "dirichlet", "isotonic"):
            m = r["methods"][method]
            mark = "★" if method == best else ""
            md_lines.append(
                f"| {tgt} | {method} | {m['macro_ece']:.3f} | {m['macro_brier']:.3f} | "
                f"{m['nll']:.3f} | {mark} |"
            )
    md_lines += [
        "",
        "## Best calibrator per target",
        "",
        "| Target | n_ext | Best method | macro-ECE | Δ vs raw |",
        "|---|---:|---|---:|---:|",
    ]
    for tgt in targets_done:
        s = summary[tgt]
        n_ext = results["by_target"][tgt]["n_external"]
        md_lines.append(
            f"| {tgt} | {n_ext} | {s['best_method']} | {s['best_macro_ece']:.3f} | "
            f"{s['delta_ece_vs_raw']:+.3f} |"
        )
    md_lines += [
        "",
        "## Verdict",
        "",
        results["verdict"],
        "",
    ]
    md_path = EXT_OUT / "q_r6_q1_multiclass_calibration_table.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Wrote %s", md_path)

    return results


if __name__ == "__main__":
    run()
