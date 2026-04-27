"""R5-Q4: AUPRC, per-class precision/recall/F1, and decision-curve analysis (DCA).

Reviewer 5 asks:
  "Can you report AUPRC and clinically relevant operating points
   (sensitivity/specificity) per class, and decision-curve analysis to
   quantify net benefit across thresholds?"

Source artefact: outputs/paper1_calibration/results/per_fold_probs.npz
  (per-fold OOF predictions of the 21-feat strict-circularity primary CatBoost
   across binary, three-class, full-ordinal, NSD-positive targets).

Outputs:
  - outputs/paper1_r2_responses/q_r5_q4_auprc_dca.json
  - outputs/paper1_r2_responses/q_r5_q4_auprc_dca_table.md
  - outputs/paper1_r2_responses/q_r5_q4_dca_curves.png

Reproducibility: random_state=42, 1,000 patient-level bootstrap resamples for
AUPRC CIs.  DCA implements Vickers (2006):
    NB = (TP / N) - (FP / N) * (pt / (1 - pt))
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
NPZ_PATH = ROOT / "outputs" / "paper1_calibration" / "results" / "per_fold_probs.npz"
OUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUT_JSON = OUT_DIR / "q_r5_q4_auprc_dca.json"
OUT_MD = OUT_DIR / "q_r5_q4_auprc_dca_table.md"
OUT_PNG = OUT_DIR / "q_r5_q4_dca_curves.png"

RNG = np.random.default_rng(42)
N_BOOT = 1000

# Okabe-Ito palette (colour-blind safe)
OK_BLACK = "#000000"
OK_ORANGE = "#E69F00"
OK_SKY = "#56B4E9"
OK_GREEN = "#009E73"
OK_YELLOW = "#F0E442"
OK_BLUE = "#0072B2"
OK_VERMILLION = "#D55E00"
OK_MAGENTA = "#CC79A7"
OK_GREY = "#808080"

TARGETS = ("binary", "three_class", "full_ordinal", "nsd_positive")
CLASS_LABELS = {
    "binary": ["NSD-", "NSD+"],
    "three_class": ["Early (0-1)", "Mild (2B)", "Impaired (3-4)"],
    "full_ordinal": ["0", "1", "2B", "3", "4"],
    "nsd_positive": ["1", "2B", "3", "4"],
}


# ---------- helpers ---------------------------------------------------------

def bootstrap_auprc(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float, np.ndarray]:
    """Patient-level bootstrap CI for binary AUPRC."""
    point = float(average_precision_score(y_true, y_score))
    n = len(y_true)
    boot_vals: list[float] = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            boot_vals.append(float(average_precision_score(y_true[idx], y_score[idx])))
        except Exception:
            continue
    arr = np.array(boot_vals) if boot_vals else np.array([np.nan])
    lo = float(np.nanpercentile(arr, 2.5))
    hi = float(np.nanpercentile(arr, 97.5))
    return point, lo, hi, arr


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float]:
    """Operating point that maximises Youden's J = sens + spec - 1."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    best = int(np.argmax(j))
    return float(thr[best]), float(tpr[best]), float(1.0 - fpr[best])


def per_class_metrics_at_youden(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> list[dict]:
    """One-vs-rest precision / recall / F1 / sens / spec at Youden's J."""
    out: list[dict] = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            out.append({
                "class": int(c),
                "support": int(y_bin.sum()),
                "threshold": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "sens": None,
                "spec": None,
                "ppv": None,
                "npv": None,
            })
            continue
        score_c = y_prob[:, c]
        thr, sens, spec = youden_threshold(y_bin, score_c)
        y_hat = (score_c >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_bin, y_hat, labels=[0, 1]).ravel()
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan")
        out.append({
            "class": int(c),
            "support": int(y_bin.sum()),
            "threshold": float(thr),
            "precision": float(precision_score(y_bin, y_hat, zero_division=0)),
            "recall": float(recall_score(y_bin, y_hat, zero_division=0)),
            "f1": float(f1_score(y_bin, y_hat, zero_division=0)),
            "sens": float(sens),
            "spec": float(spec),
            "ppv": ppv,
            "npv": npv,
        })
    return out


def decision_curve(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Net benefit curve (Vickers 2006).

    NB(model) = (TP / N) - (FP / N) * (pt / (1 - pt))
    NB(treat-all) = prev - (1 - prev) * (pt / (1 - pt))
    NB(treat-none) = 0
    """
    n = len(y_true)
    prev = float(y_true.mean())
    thresholds = np.arange(0.01, 1.00, 0.01)
    nb_model: list[float] = []
    nb_all: list[float] = []
    for pt in thresholds:
        y_hat = (y_score >= pt).astype(int)
        tp = float(((y_hat == 1) & (y_true == 1)).sum())
        fp = float(((y_hat == 1) & (y_true == 0)).sum())
        odds = pt / (1.0 - pt)
        nb_model.append(tp / n - fp / n * odds)
        nb_all.append(prev - (1.0 - prev) * odds)
    return {
        "thresholds": [float(t) for t in thresholds],
        "net_benefit": [float(v) for v in nb_model],
        "net_benefit_treat_all": [float(v) for v in nb_all],
        "net_benefit_treat_none": [0.0 for _ in thresholds],
    }


def positive_nb_range(dca: dict) -> tuple[float, float] | None:
    """Find the contiguous threshold range where model NB > max(0, treat-all NB)."""
    thr = np.asarray(dca["thresholds"])
    nb_m = np.asarray(dca["net_benefit"])
    nb_a = np.asarray(dca["net_benefit_treat_all"])
    superior = nb_m > np.maximum(nb_a, 0.0)
    if not superior.any():
        return None
    idxs = np.where(superior)[0]
    return float(thr[idxs.min()]), float(thr[idxs.max()])


# ---------- main per-target analysis ----------------------------------------

def analyse_target(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    n_classes = y_prob.shape[1]
    out: dict = {
        "n_patients": int(len(y_true)),
        "n_classes": int(n_classes),
        "class_labels": CLASS_LABELS[name],
    }

    # ---------- AUPRC (per class + macro) -----------------------------------
    auprc_per_class: list[dict] = []
    boot_macro: list[float] = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0:
            auprc_per_class.append({
                "class": c,
                "auprc": None,
                "ci95_lo": None,
                "ci95_hi": None,
                "support": int(y_bin.sum()),
            })
            continue
        point, lo, hi, _ = bootstrap_auprc(y_bin, y_prob[:, c])
        auprc_per_class.append({
            "class": int(c),
            "auprc": point,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "support": int(y_bin.sum()),
        })

    # macro AUPRC bootstrap (resample once, average across classes per resample)
    n = len(y_true)
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        per_class_aps: list[float] = []
        for c in range(n_classes):
            y_bin = (y_true[idx] == c).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue
            try:
                per_class_aps.append(float(average_precision_score(y_bin, y_prob[idx, c])))
            except Exception:
                continue
        if per_class_aps:
            boot_macro.append(float(np.mean(per_class_aps)))

    # macro point estimate
    macro_pts = [a["auprc"] for a in auprc_per_class if a["auprc"] is not None]
    out["auprc_macro"] = float(np.mean(macro_pts)) if macro_pts else None
    if boot_macro:
        out["auprc_macro_ci95_lo"] = float(np.percentile(boot_macro, 2.5))
        out["auprc_macro_ci95_hi"] = float(np.percentile(boot_macro, 97.5))
    else:
        out["auprc_macro_ci95_lo"] = None
        out["auprc_macro_ci95_hi"] = None
    out["auprc_per_class"] = auprc_per_class

    # ---------- per-class precision / recall / F1 / sens / spec --------------
    out["per_class_at_youden"] = per_class_metrics_at_youden(y_true, y_prob, n_classes)

    # ---------- DCA ---------------------------------------------------------
    if n_classes == 2:
        dca = decision_curve(y_true, y_prob[:, 1])
        out["decision_curve"] = dca
        rng = positive_nb_range(dca)
        out["dca_positive_nb_range"] = rng
        # ROC + PR curves for binary
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        prec, rec, _ = precision_recall_curve(y_true, y_prob[:, 1])
        out["roc_curve"] = {"fpr": [float(x) for x in fpr], "tpr": [float(x) for x in tpr]}
        out["pr_curve"] = {"recall": [float(x) for x in rec], "precision": [float(x) for x in prec]}
        # Calibration-aware operating point: among thresholds where the model
        # strictly beats both reference strategies, pick the one with the
        # highest net benefit.  Falls back to the global argmax if no threshold
        # beats both references (degenerate at very low pt where treat-all
        # dominates by construction).
        thr_arr = np.asarray(dca["thresholds"])
        nb_arr = np.asarray(dca["net_benefit"])
        nb_all = np.asarray(dca["net_benefit_treat_all"])
        superior = nb_arr > np.maximum(nb_all, 0.0)
        if superior.any():
            cand_idx = np.where(superior)[0]
            best_idx = int(cand_idx[np.argmax(nb_arr[cand_idx])])
        else:
            best_idx = int(np.argmax(nb_arr))
        out["calibration_aware_operating_point"] = {
            "threshold": float(thr_arr[best_idx]),
            "net_benefit": float(nb_arr[best_idx]),
        }
    else:
        # one-vs-rest DCA per class (most clinically actionable per class)
        per_class_dca: list[dict] = []
        for c in range(n_classes):
            y_bin = (y_true == c).astype(int)
            if y_bin.sum() < 2:
                per_class_dca.append({"class": c, "decision_curve": None, "dca_positive_nb_range": None})
                continue
            dca = decision_curve(y_bin, y_prob[:, c])
            per_class_dca.append({
                "class": int(c),
                "support": int(y_bin.sum()),
                "decision_curve": dca,
                "dca_positive_nb_range": positive_nb_range(dca),
            })
        out["decision_curve_per_class"] = per_class_dca

    return out


# ---------- markdown table --------------------------------------------------

def fmt_ci(lo, hi) -> str:
    if lo is None or hi is None:
        return "n/a"
    return f"[{lo:.3f}, {hi:.3f}]"


def build_markdown(results: dict) -> str:
    lines: list[str] = []
    lines.append("# R5-Q4 — AUPRC, Operating Points, and Decision-Curve Analysis")
    lines.append("")
    lines.append(
        "21-feat strict-circularity primary CatBoost, OOF predictions from the "
        "5-fold cross-validation reported in §V.A.  All metrics computed at the "
        "patient level; bootstrap CIs use 1,000 resamples (seed=42)."
    )
    lines.append("")

    # AUPRC table
    lines.append("## AUPRC (macro, 95% bootstrap CI)")
    lines.append("")
    lines.append("| Target | n | classes | AUPRC (macro) | 95% CI |")
    lines.append("|---|---|---|---|---|")
    for tgt in TARGETS:
        r = results[tgt]
        lines.append(
            f"| {tgt} | {r['n_patients']} | {r['n_classes']} | "
            f"{r['auprc_macro']:.3f} | {fmt_ci(r['auprc_macro_ci95_lo'], r['auprc_macro_ci95_hi'])} |"
        )
    lines.append("")

    # Per-class AUPRC + operating-point table
    lines.append("## Per-class AUPRC and operating point at Youden's J")
    lines.append("")
    lines.append("| Target | Class | n_pos | AUPRC [95% CI] | Threshold | Sens | Spec | Prec | F1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for tgt in TARGETS:
        r = results[tgt]
        labels = r["class_labels"]
        for ap, mc in zip(r["auprc_per_class"], r["per_class_at_youden"]):
            label = labels[mc["class"]] if mc["class"] < len(labels) else str(mc["class"])
            ap_str = "n/a" if ap["auprc"] is None else f"{ap['auprc']:.3f} {fmt_ci(ap['ci95_lo'], ap['ci95_hi'])}"
            thr = "n/a" if mc["threshold"] is None else f"{mc['threshold']:.3f}"
            sens = "n/a" if mc["sens"] is None else f"{mc['sens']:.3f}"
            spec = "n/a" if mc["spec"] is None else f"{mc['spec']:.3f}"
            prec = "n/a" if mc["precision"] is None else f"{mc['precision']:.3f}"
            f1 = "n/a" if mc["f1"] is None else f"{mc['f1']:.3f}"
            lines.append(
                f"| {tgt} | {label} | {ap['support']} | {ap_str} | {thr} | {sens} | {spec} | {prec} | {f1} |"
            )
    lines.append("")

    # DCA summary paragraph
    lines.append("## Decision-Curve Analysis (DCA)")
    lines.append("")
    lines.append(
        "Net benefit was computed across threshold probabilities 0.01–0.99 (step 0.01) "
        "following Vickers (2006): NB = (TP/N) - (FP/N) * pt/(1-pt), with reference "
        "strategies treat-all (NB = prev - (1-prev) * pt/(1-pt)) and treat-none (NB = 0)."
    )
    lines.append("")
    bin_r = results["binary"]
    rng = bin_r.get("dca_positive_nb_range")
    if rng is not None:
        lines.append(
            f"**Binary (NSD+ vs NSD-)** — the model provides positive net benefit "
            f"relative to the better of treat-all / treat-none across threshold "
            f"probabilities **{rng[0]:.2f} – {rng[1]:.2f}**, with peak NB "
            f"{bin_r['calibration_aware_operating_point']['net_benefit']:.3f} at "
            f"threshold {bin_r['calibration_aware_operating_point']['threshold']:.2f}."
        )
    lines.append("")

    # Multiclass DCA range summary
    for tgt in ("three_class", "full_ordinal", "nsd_positive"):
        r = results[tgt]
        per_class = r.get("decision_curve_per_class", [])
        labels = r["class_labels"]
        ranges = []
        for pc in per_class:
            rng = pc.get("dca_positive_nb_range")
            label = labels[pc["class"]] if pc["class"] < len(labels) else str(pc["class"])
            if rng is None:
                ranges.append(f"{label}: NONE")
            else:
                ranges.append(f"{label}: {rng[0]:.2f}–{rng[1]:.2f}")
        lines.append(f"**{tgt}** one-vs-rest DCA positive-NB ranges — " + "; ".join(ranges))
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "Reproducibility: `scripts/paper1/run_q_r5_q4_auprc_dca.py`, "
        "OOF source `outputs/paper1_calibration/results/per_fold_probs.npz`, "
        "seed=42, sklearn.metrics, n_boot=1000."
    )
    return "\n".join(lines) + "\n"


# ---------- figure ----------------------------------------------------------

def render_figure(results: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.6), constrained_layout=True)
    panels = list(zip(TARGETS, axes.ravel()))

    for tgt, ax in panels:
        r = results[tgt]
        labels = r["class_labels"]
        if tgt == "binary":
            dca = r["decision_curve"]
            thr = np.asarray(dca["thresholds"])
            ax.plot(thr, dca["net_benefit"], color=OK_VERMILLION, lw=2.0, label="CatBoost (21-feat)")
            ax.plot(thr, dca["net_benefit_treat_all"], color=OK_GREY, lw=1.0, ls="--", label="Treat all")
            ax.plot(thr, dca["net_benefit_treat_none"], color=OK_BLACK, lw=1.0, ls=":", label="Treat none")
        else:
            per_class = r.get("decision_curve_per_class", [])
            colors = [OK_VERMILLION, OK_BLUE, OK_GREEN, OK_ORANGE, OK_MAGENTA]
            for pc in per_class:
                if pc.get("decision_curve") is None:
                    continue
                dca = pc["decision_curve"]
                thr = np.asarray(dca["thresholds"])
                cls = pc["class"]
                label = labels[cls] if cls < len(labels) else str(cls)
                ax.plot(thr, dca["net_benefit"], color=colors[cls % len(colors)], lw=1.4, label=label)
                # treat-all reference for the class with the largest support
            # Generic treat-none reference
            ax.axhline(0.0, color=OK_BLACK, lw=0.8, ls=":")
        ax.set_title(tgt, fontsize=9)
        ax.set_xlabel("Threshold probability", fontsize=8)
        ax.set_ylabel("Net benefit", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xlim(0.0, 1.0)
        # Sensible y-limits: clip extremes so positive NB is visible
        if tgt == "binary":
            nb_vals = list(r["decision_curve"]["net_benefit"])
        else:
            nb_vals = []
            for pc in r.get("decision_curve_per_class", []):
                if pc.get("decision_curve") is not None:
                    nb_vals.extend(pc["decision_curve"]["net_benefit"])
        ymax = max(0.05, float(np.nanmax(nb_vals)) * 1.10) if nb_vals else 0.4
        ax.set_ylim(-0.05, ymax)
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.legend(fontsize=6, loc="upper right", frameon=False)

    fig.suptitle("Decision-curve analysis (21-feat primary CatBoost, OOF)", fontsize=10)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------- driver ----------------------------------------------------------

def main() -> None:
    npz = np.load(NPZ_PATH)
    results: dict = {
        "workstream": "r5_q4_auprc_dca",
        "feature_spec": "21-feat strict-circularity primary CatBoost (Path 3)",
        "model": "catboost_21feat_default",
        "source": str(NPZ_PATH.relative_to(ROOT)),
        "seed": 42,
        "n_boot": N_BOOT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "by_target": {},
    }

    for tgt in TARGETS:
        y_true = npz[f"{tgt}_y_true"]
        y_prob = npz[f"{tgt}_y_prob"]
        results["by_target"][tgt] = analyse_target(tgt, y_true, y_prob)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    OUT_MD.write_text(build_markdown(results["by_target"]))
    render_figure(results["by_target"])

    print(f"[json] {OUT_JSON.relative_to(ROOT)}")
    print(f"[md]   {OUT_MD.relative_to(ROOT)}")
    print(f"[png]  {OUT_PNG.relative_to(ROOT)}")
    for tgt in TARGETS:
        r = results["by_target"][tgt]
        ci = fmt_ci(r["auprc_macro_ci95_lo"], r["auprc_macro_ci95_hi"])
        print(f"[summary] {tgt:<14} macro AUPRC = {r['auprc_macro']:.3f} {ci}")


if __name__ == "__main__":
    main()
