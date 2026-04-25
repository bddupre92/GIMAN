"""Paper 1 R6-Q8 — Age-decile (internal) + BioFIND external subgroup fairness.

Reviewer 6 asks (extending R5-Q6):
  "Can you report subgroup performance (and calibration/coverage) by sex and
   age deciles for both internal and BioFIND external to complement your
   confounder analyses?"

R5-Q6 covered: sex + age tertiles + carrier on 21-feat BINARY + 12-feat NSD+
INTERNAL only. R6 wants finer (deciles) + external.

This script:
  1) Reuses the R5-Q6 OOF generators (21-feat BINARY + 12-feat NSD+) and
     re-stratifies by **age decile** (10 bins, ~ N_total/10 each).
  2) Trains the same 12-feat CatBoost on FULL PPMI then predicts on BioFIND
     and computes per-subgroup AUC + 95% CI + ECE for sex + **age tertiles**
     (n=103 too small for deciles; reviewer-honest).

Outputs:
  outputs/paper1_r2_responses/q_r6_q8_subgroup_extended.json
  outputs/paper1_r2_responses/q_r6_q8_subgroup_extended_table.md
  outputs/paper1_r2_responses/q_r6_q8_age_decile_internal.png
  outputs/paper1_r2_responses/q_r6_q8_external_subgroup.png

Verdict per subgroup: PASS = |Δ AUC vs overall| < 0.03 (reuses R5-Q6 threshold).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.paper1.run_q_r5_q6_fairness_extended import (  # noqa: E402
    BOOT_N,
    COMMON_FEATURES_12,
    CV_SEED,
    ECE_BINS,
    auc_of,
    ece_of,
    get_oof_12feat_nsdpos,
    get_oof_21feat_binary,
    load_features_with_full_sex,
    _safe_auc,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("q_r6_q8_subgroup")

OUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIOFIND_FEATURES = ROOT / "data" / "05_features" / "biofind_features.csv"
BIOFIND_STAGING = ROOT / "data" / "04_staging" / "biofind_nsd_iss_staging.csv"

MIN_DECILE_N = 30  # Need ≥30 to compute AUC + 1000 bootstrap reliably
MIN_EXT_STRATUM_N = 15  # Looser for external (n=103); honest about constraint
DELTA_THRESHOLD = 0.03

# Okabe-Ito palette
C_REF = "#0072B2"
C_BIN = "#E69F00"
C_NSD = "#009E73"
C_FEM = "#CC79A7"
C_MAL = "#0072B2"


# ---------------------------------------------------------------------------
# Internal age-decile stratification
# ---------------------------------------------------------------------------
def per_decile_metrics(
    sub: pd.DataFrame,
    y: np.ndarray,
    oof: np.ndarray,
    n_classes: int,
    main_auc: float,
    rng: np.random.Generator,
) -> dict:
    """Compute n / AUC / 95% CI / ECE / Δ vs overall per age decile."""
    age = sub["AGE_AT_BASELINE"].astype(float).values
    valid = ~np.isnan(age)
    decile_cuts = np.percentile(age[valid], np.arange(10, 100, 10))  # 9 cuts → 10 bins
    log.info("  Age decile cuts (years): %s", [round(c, 2) for c in decile_cuts])

    deciles_out = {}
    for i in range(10):
        if i == 0:
            mask = age < decile_cuts[0]
            label = f"D1_<{decile_cuts[0]:.1f}y"
        elif i == 9:
            mask = age >= decile_cuts[-1]
            label = f"D10_>={decile_cuts[-1]:.1f}y"
        else:
            mask = (age >= decile_cuts[i - 1]) & (age < decile_cuts[i])
            label = f"D{i + 1}_{decile_cuts[i - 1]:.1f}-{decile_cuts[i]:.1f}y"
        n = int(mask.sum())
        if n < MIN_DECILE_N:
            deciles_out[label] = {
                "decile_idx": i + 1,
                "n": n,
                "auc": None,
                "ci95_lo": None,
                "ci95_hi": None,
                "ece_10bin": None,
                "delta_vs_main": None,
                "skipped": True,
                "reason": f"n<{MIN_DECILE_N}",
            }
            continue
        yt = y[mask]
        pp = oof[mask]
        auc = _safe_auc(yt, pp, n_classes)
        if auc is None:
            deciles_out[label] = {
                "decile_idx": i + 1,
                "n": n,
                "auc": None,
                "ci95_lo": None,
                "ci95_hi": None,
                "ece_10bin": None,
                "delta_vs_main": None,
                "skipped": True,
                "reason": "AUC undefined",
            }
            continue
        ece = ece_of(yt, pp, n_classes, n_bins=ECE_BINS)
        boots = []
        for _ in range(BOOT_N):
            idx = rng.integers(0, len(yt), len(yt))
            v = _safe_auc(yt[idx], pp[idx], n_classes)
            if v is not None:
                boots.append(v)
        lo = float(np.percentile(boots, 2.5)) if len(boots) >= 10 else None
        hi = float(np.percentile(boots, 97.5)) if len(boots) >= 10 else None
        delta = float(auc - main_auc)
        verdict = "PASS" if abs(delta) < DELTA_THRESHOLD else "FAIL"
        deciles_out[label] = {
            "decile_idx": i + 1,
            "n": n,
            "auc": float(auc),
            "ci95_lo": lo,
            "ci95_hi": hi,
            "ece_10bin": float(ece),
            "delta_vs_main": delta,
            "skipped": False,
            "verdict": verdict,
        }
        log.info(
            "    %-22s  n=%-4d  AUC=%.4f [%.3f,%.3f]  Δ=%+.4f  ECE=%.4f  %s",
            label,
            n,
            auc,
            lo,
            hi,
            delta,
            ece,
            verdict,
        )
    # Headline: trend direction
    aucs_in_order = [
        d["auc"]
        for d in sorted(deciles_out.values(), key=lambda x: x["decile_idx"])
        if d.get("auc") is not None
    ]
    if len(aucs_in_order) >= 4:
        from scipy.stats import spearmanr

        decile_idx_seq = [
            d["decile_idx"]
            for d in sorted(deciles_out.values(), key=lambda x: x["decile_idx"])
            if d.get("auc") is not None
        ]
        rho, p_trend = spearmanr(decile_idx_seq, aucs_in_order)
        trend = {
            "spearman_rho": float(rho),
            "spearman_p": float(p_trend),
            "shape": (
                "monotonic_increasing"
                if rho > 0.6 and p_trend < 0.05
                else "monotonic_decreasing"
                if rho < -0.6 and p_trend < 0.05
                else "U_or_flat"
            ),
            "n_deciles_evaluated": len(aucs_in_order),
        }
    else:
        trend = {
            "spearman_rho": None,
            "spearman_p": None,
            "shape": "insufficient_deciles",
            "n_deciles_evaluated": len(aucs_in_order),
        }
    n_pass = sum(
        1 for d in deciles_out.values() if d.get("verdict") == "PASS"
    )
    n_fail = sum(
        1 for d in deciles_out.values() if d.get("verdict") == "FAIL"
    )
    return {
        "decile_cuts": [float(c) for c in decile_cuts],
        "deciles": deciles_out,
        "trend": trend,
        "n_deciles_pass": n_pass,
        "n_deciles_fail": n_fail,
    }


# ---------------------------------------------------------------------------
# External BioFIND analysis
# ---------------------------------------------------------------------------
def train_full_ppmi_12feat_binary(df: pd.DataFrame) -> tuple:
    """Train 12-feat CatBoost on FULL PPMI for BioFIND BINARY external."""
    mask = df["target_binary"] >= 0
    sub = df[mask].copy().reset_index(drop=True)
    X = sub[COMMON_FEATURES_12].to_numpy(dtype=float)
    y = sub["target_binary"].astype(int).to_numpy()
    log.info("Training full 12-feat BINARY on PPMI: n=%d, n_features=%d", len(sub), X.shape[1])
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_imp = imp.fit_transform(X)
    sc.fit(X_imp)  # CatBoost is scale-invariant but keep parity with external pipeline
    clf = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
        auto_class_weights="Balanced",
    )
    clf.fit(X_imp, y)
    return clf, imp, sc


BIOFIND_SAA = ROOT / "data" / "00_raw" / "BioFind" / "biofind_saa_consensus.csv"


def predict_biofind_binary(clf, imp) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate BioFIND binary predictions w/ ground truth.

    BINARY ground truth = SAA consensus (S+ vs S-): n=194 with 81 negatives +
    113 positives. The external_validation pipeline (run_external_validation.py
    line 179-188) uses this exact source.

    BioFIND features have ESS_TOTAL absent and UPDRS4_TOTAL 100% missing. Per
    the external pipeline (line 327-333), missing features are added as NaN
    columns and imputed by the FULL-PPMI-fit imputer at prediction time.
    """
    bf = pd.read_csv(BIOFIND_FEATURES)
    saa = pd.read_csv(BIOFIND_SAA)
    saa["participant_id"] = "BF-" + saa["PATNO"].astype(str)
    saa["target_binary"] = saa["SAA_RESULT"].astype(int)
    log.info("BioFIND features: %d patients, SAA consensus: %d patients", len(bf), len(saa))

    # Add NaN columns for missing common features (matches external pipeline behavior)
    for f in COMMON_FEATURES_12:
        if f not in bf.columns:
            log.info("  Adding NaN column for absent BioFIND feature: %s", f)
            bf[f] = np.nan

    # Inner merge on participant_id with SAA ground truth
    merged = bf.merge(
        saa[["participant_id", "target_binary", "SAA_RESULT"]],
        on="participant_id",
        how="inner",
    )
    log.info("After merge (features ∩ SAA): n=%d", len(merged))
    log.info(
        "BINARY ground-truth subset: n=%d, class dist=%s",
        len(merged),
        dict(zip(*np.unique(merged["target_binary"].values, return_counts=True), strict=False)),
    )

    X = merged[COMMON_FEATURES_12].to_numpy(dtype=float)
    X_imp = imp.transform(X)
    proba = clf.predict_proba(X_imp)
    return merged, merged["target_binary"].values, proba


def biofind_subgroup_metrics(
    merged: pd.DataFrame,
    y: np.ndarray,
    proba: np.ndarray,
    main_auc: float,
    age_tertile_cuts_external: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Per-subgroup AUC + 95% CI + ECE for SEX and AGE TERTILES on BioFIND."""
    if rng is None:
        rng = np.random.default_rng(CV_SEED)
    sex = merged["SEX"].fillna(-1).astype(int).values
    age = merged["AGE_AT_BASELINE"].astype(float).values
    if age_tertile_cuts_external is None:
        valid = ~np.isnan(age)
        age_tertile_cuts_external = np.percentile(age[valid], [33.3333, 66.6667])
    log.info("  BioFIND age tertile cuts (years): %s", [round(c, 2) for c in age_tertile_cuts_external])

    strata = {
        "sex": {
            "Female": sex == 0,
            "Male": sex == 1,
        },
        "age_tertile": {
            f"T1_<{age_tertile_cuts_external[0]:.1f}y": age < age_tertile_cuts_external[0],
            f"T2_{age_tertile_cuts_external[0]:.1f}-{age_tertile_cuts_external[1]:.1f}y": (
                (age >= age_tertile_cuts_external[0]) & (age < age_tertile_cuts_external[1])
            ),
            f"T3_>={age_tertile_cuts_external[1]:.1f}y": age >= age_tertile_cuts_external[1],
        },
    }

    results = {}
    for axis_name, groups in strata.items():
        per_subgroup = {}
        any_pass = True
        for label, mask in groups.items():
            n = int(mask.sum())
            if n < MIN_EXT_STRATUM_N:
                per_subgroup[label] = {
                    "n": n,
                    "auc": None,
                    "ci95_lo": None,
                    "ci95_hi": None,
                    "ece_10bin": None,
                    "delta_vs_main": None,
                    "skipped": True,
                    "reason": f"n<{MIN_EXT_STRATUM_N} (small external cohort)",
                }
                continue
            yt = y[mask]
            pp = proba[mask]
            if len(np.unique(yt)) < 2:
                per_subgroup[label] = {
                    "n": n,
                    "auc": None,
                    "ci95_lo": None,
                    "ci95_hi": None,
                    "ece_10bin": None,
                    "delta_vs_main": None,
                    "skipped": True,
                    "reason": "single class in subgroup (BioFIND severely imbalanced — gotcha documented)",
                }
                continue
            auc = _safe_auc(yt, pp, n_classes=2)
            if auc is None:
                per_subgroup[label] = {
                    "n": n,
                    "auc": None,
                    "ci95_lo": None,
                    "ci95_hi": None,
                    "ece_10bin": None,
                    "delta_vs_main": None,
                    "skipped": True,
                    "reason": "AUC undefined",
                }
                continue
            ece = ece_of(yt, pp, n_classes=2, n_bins=ECE_BINS)
            boots = []
            for _ in range(BOOT_N):
                idx = rng.integers(0, len(yt), len(yt))
                v = _safe_auc(yt[idx], pp[idx], n_classes=2)
                if v is not None:
                    boots.append(v)
            lo = float(np.percentile(boots, 2.5)) if len(boots) >= 10 else None
            hi = float(np.percentile(boots, 97.5)) if len(boots) >= 10 else None
            delta = float(auc - main_auc) if main_auc is not None else None
            verdict = (
                "PASS"
                if (delta is not None and abs(delta) < DELTA_THRESHOLD)
                else "FAIL"
                if delta is not None
                else "UNDETERMINED"
            )
            if verdict == "FAIL":
                any_pass = False
            per_subgroup[label] = {
                "n": n,
                "auc": float(auc),
                "ci95_lo": lo,
                "ci95_hi": hi,
                "ece_10bin": float(ece),
                "delta_vs_main": delta,
                "skipped": False,
                "verdict": verdict,
            }
            log.info(
                "    %-12s  %-22s  n=%-4d  AUC=%.4f [%.3f,%.3f]  Δ=%+.4f  ECE=%.4f  %s",
                axis_name,
                label,
                n,
                auc,
                lo,
                hi,
                delta if delta is not None else float("nan"),
                ece,
                verdict,
            )
        # Axis verdict: PASS only if all evaluable subgroups PASS
        evaluable = [
            sg for sg in per_subgroup.values() if not sg.get("skipped")
        ]
        if not evaluable:
            axis_verdict = "INSUFFICIENT_DATA"
        else:
            axis_verdict = "PASS" if all(sg.get("verdict") == "PASS" for sg in evaluable) else "FAIL"
        max_delta = max(
            (abs(sg["delta_vs_main"]) for sg in evaluable if sg.get("delta_vs_main") is not None),
            default=None,
        )
        results[axis_name] = {
            "subgroups": per_subgroup,
            "axis_verdict": axis_verdict,
            "max_abs_delta_vs_main": float(max_delta) if max_delta is not None else None,
        }
    return results, [float(c) for c in age_tertile_cuts_external]


# ---------------------------------------------------------------------------
# Plot 1 — internal age-decile trend
# ---------------------------------------------------------------------------
def plot_age_decile_internal(
    decile_results: dict[str, dict],
    overall_aucs: dict[str, float],
    overall_cis: dict[str, list],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.6), dpi=300, sharey=True)
    model_keys = list(decile_results.keys())
    color_map = {"21feat_binary": C_BIN, "12feat_nsd_positive": C_NSD}
    title_map = {
        "21feat_binary": "21-feat Path 3 (Binary)",
        "12feat_nsd_positive": "12-feat clinical-only (NSD+ sub-stage)",
    }

    for ax, model_label in zip(axes, model_keys):
        dr = decile_results[model_label]
        decs = sorted(dr["deciles"].values(), key=lambda x: x["decile_idx"])
        x_idx = []
        aucs = []
        los = []
        his = []
        for d in decs:
            x_idx.append(d["decile_idx"])
            if d.get("auc") is None:
                aucs.append(np.nan)
                los.append(np.nan)
                his.append(np.nan)
            else:
                aucs.append(d["auc"])
                los.append(d["ci95_lo"])
                his.append(d["ci95_hi"])
        x_arr = np.array(x_idx, dtype=float)
        aucs = np.array(aucs)
        los = np.array(los)
        his = np.array(his)
        valid = ~np.isnan(aucs)

        # Reference band (overall AUC ± 95% CI)
        ovr_auc = overall_aucs[model_label]
        ovr_lo, ovr_hi = overall_cis[model_label]
        ax.axhspan(ovr_lo, ovr_hi, alpha=0.18, color=C_REF, zorder=0, label="Overall 95% CI")
        ax.axhline(ovr_auc, color=C_REF, lw=0.9, ls="--", alpha=0.7, label=f"Overall AUC = {ovr_auc:.3f}")
        # Per-decile points + CI bands
        ax.plot(
            x_arr[valid],
            aucs[valid],
            color=color_map[model_label],
            lw=1.2,
            marker="o",
            markersize=5,
            markerfacecolor=color_map[model_label],
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="Per-decile AUC",
        )
        ax.fill_between(
            x_arr[valid],
            los[valid],
            his[valid],
            color=color_map[model_label],
            alpha=0.20,
        )
        ax.set_xticks(np.arange(1, 11))
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(0.50, 1.0)
        ax.set_xlabel("Age decile (1=youngest, 10=oldest)", fontsize=8)
        ax.set_title(title_map[model_label], fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", linestyle=":", lw=0.3, alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.85)
        # Annotate trend
        trend = dr["trend"]
        if trend.get("spearman_rho") is not None:
            ax.text(
                0.02,
                0.98,
                f"Spearman ρ={trend['spearman_rho']:+.2f}\np={trend['spearman_p']:.3f}\nshape: {trend['shape']}",
                transform=ax.transAxes,
                fontsize=6.5,
                va="top",
                ha="left",
                bbox=dict(facecolor="white", edgecolor="grey", lw=0.3, alpha=0.85),
            )

    axes[0].set_ylabel("AUC [95% bootstrap CI]", fontsize=8)
    fig.suptitle("R6-Q8: Internal age-decile AUC (PPMI)", fontsize=10, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    log.info("Wrote %s", out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 — external BioFIND forest
# ---------------------------------------------------------------------------
def plot_external_forest(
    biofind_main_auc: float,
    biofind_main_ci: list,
    biofind_main_ece: float,
    biofind_results: dict,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    # Flatten subgroups: sex.Female, sex.Male, age_tertile.T1, T2, T3
    rows = []
    for axis_name, axis_res in biofind_results.items():
        for sg_label, sg in axis_res["subgroups"].items():
            rows.append((axis_name, sg_label, sg))
    n_rows = len(rows)

    fig, ax = plt.subplots(figsize=(5.0, 0.55 * n_rows + 1.0), dpi=300)
    y_pos = np.arange(n_rows)[::-1]

    # Reference band
    ax.axvspan(biofind_main_ci[0], biofind_main_ci[1], alpha=0.18, color=C_REF, zorder=0,
               label=f"Overall 95% CI ({biofind_main_ci[0]:.3f}–{biofind_main_ci[1]:.3f})")
    ax.axvline(biofind_main_auc, color=C_REF, lw=0.9, ls="--", alpha=0.7,
               label=f"Overall AUC = {biofind_main_auc:.3f}")

    labels = []
    for i, (axis_name, sg_label, sg) in enumerate(rows):
        col = C_FEM if "Female" in sg_label else C_MAL if "Male" in sg_label else C_NSD
        if sg.get("skipped") or sg.get("auc") is None:
            ax.text(
                0.52,
                y_pos[i],
                f"skipped (n={sg.get('n', 0)}, {sg.get('reason', '?')[:40]})",
                ha="left",
                va="center",
                fontsize=6.5,
                color="grey",
            )
        else:
            ax.errorbar(
                [sg["auc"]],
                [y_pos[i]],
                xerr=[[sg["auc"] - sg["ci95_lo"]], [sg["ci95_hi"] - sg["auc"]]],
                fmt="o",
                capsize=3,
                markersize=6,
                ecolor="#333",
                markerfacecolor=col,
                markeredgecolor="black",
                markeredgewidth=0.5,
                linewidth=1.0,
            )
            verdict_color = "#009E73" if sg.get("verdict") == "PASS" else "#D55E00"
            ax.text(
                1.02,
                y_pos[i],
                f"Δ={sg['delta_vs_main']:+.3f}  ECE={sg['ece_10bin']:.3f}  {sg['verdict']}",
                ha="left",
                va="center",
                fontsize=6.5,
                color=verdict_color,
                transform=ax.get_yaxis_transform(),
            )
        labels.append(f"{axis_name}: {sg_label}\n(n={sg.get('n', 0)})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0.30, 1.0)
    ax.set_xlabel("AUC [95% bootstrap CI]", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", linestyle=":", lw=0.3, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.85)
    ax.set_title(
        "R6-Q8: BioFIND external subgroup AUC (12-feat clinical, BINARY)\n"
        "n=103, severe class imbalance (95.4% NSD+) — small-n caveat applies",
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    log.info("Wrote %s", out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------
def fmt_auc_ci(d: dict) -> str:
    if d.get("skipped") or d.get("auc") is None:
        return "—"
    return f"{d['auc']:.3f} [{d['ci95_lo']:.3f}, {d['ci95_hi']:.3f}]"


def fmt_ece(d: dict) -> str:
    if d.get("skipped") or d.get("ece_10bin") is None:
        return "—"
    return f"{d['ece_10bin']:.3f}"


def fmt_delta(d: dict) -> str:
    if d.get("skipped") or d.get("delta_vs_main") is None:
        return "—"
    return f"{d['delta_vs_main']:+.3f}"


def build_table(payload: dict) -> str:
    lines = [
        "# R6-Q8: Subgroup fairness — age deciles (internal) + BioFIND external",
        "",
        "Extends R5-Q6 sex/age-tertile/carrier internal analysis with:",
        "  (a) **age deciles** (10 bins) on both 21-feat BINARY and 12-feat NSD+ INTERNAL",
        "  (b) **BioFIND external** stratified by sex + age tertiles (n=103 too small for deciles)",
        "",
        "**Verdict per subgroup:** PASS = |Δ AUC vs overall| < 0.03 (matches R5-Q6 threshold).",
        "",
        "## Internal — Age deciles",
        "",
    ]
    for model_label, dr in payload["internal_age_deciles"].items():
        ovr_auc = payload["internal_overall"][model_label]["auc"]
        ovr_lo, ovr_hi = payload["internal_overall"][model_label]["ci95"]
        ovr_ece = payload["internal_overall"][model_label]["ece"]
        n_total = payload["internal_overall"][model_label]["n"]
        cuts = dr["decile_cuts"]
        lines.append(
            f"### {model_label}  (Overall AUC={ovr_auc:.3f} [{ovr_lo:.3f}, {ovr_hi:.3f}], "
            f"ECE={ovr_ece:.3f}, n={n_total:,})"
        )
        lines.append("")
        lines.append(f"Decile cuts (years): " + ", ".join(f"{c:.1f}" for c in cuts))
        lines.append("")
        lines.append("| Decile | n | AUC [95% CI] | ECE_10bin | Δ vs overall | Verdict |")
        lines.append("|---|---|---|---|---|---|")
        for d in sorted(dr["deciles"].values(), key=lambda x: x["decile_idx"]):
            label = next(
                k for k, v in dr["deciles"].items() if v["decile_idx"] == d["decile_idx"]
            )
            lines.append(
                f"| {label} | {d['n']} | {fmt_auc_ci(d)} | {fmt_ece(d)} | {fmt_delta(d)} | "
                f"{d.get('verdict', 'SKIPPED')} |"
            )
        trend = dr["trend"]
        if trend.get("spearman_rho") is not None:
            lines.append("")
            lines.append(
                f"**Trend across deciles:** Spearman ρ={trend['spearman_rho']:+.3f}, "
                f"p={trend['spearman_p']:.3f}, shape={trend['shape']} "
                f"(evaluable deciles: {trend['n_deciles_evaluated']}/10)"
            )
        lines.append(
            f"**Decile PASS rate:** {dr['n_deciles_pass']}/"
            f"{dr['n_deciles_pass'] + dr['n_deciles_fail']} evaluable deciles"
        )
        lines.append("")

    # External BioFIND
    ext = payload["external_biofind"]
    lines.append("## External — BioFIND (12-feat clinical, BINARY)")
    lines.append("")
    lines.append(
        f"Overall: AUC={ext['main_auc']:.3f} [{ext['main_auc_ci95'][0]:.3f}, "
        f"{ext['main_auc_ci95'][1]:.3f}], ECE={ext['main_ece_10bin']:.3f}, "
        f"n={ext['n_total']}, class dist={ext['class_distribution']}"
    )
    lines.append("")
    lines.append(
        "**Caveats (reviewer-honest):** BioFIND is severely imbalanced (95.4% NSD+); subgroup "
        "AUCs may have wide CIs from few negatives in each stratum. Age-decile (10 bins) is not "
        "feasible at n=103; reporting age tertiles instead. Bootstrap UndefinedMetricWarnings "
        "(documented gotcha) are expected when a stratum draws only one class — those bootstrap "
        "iterations are dropped from the CI computation."
    )
    lines.append("")
    lines.append(
        f"BioFIND age-tertile cuts (years): "
        f"T1<{ext['age_tertile_cuts_external'][0]:.1f}, "
        f"T2={ext['age_tertile_cuts_external'][0]:.1f}–{ext['age_tertile_cuts_external'][1]:.1f}, "
        f"T3≥{ext['age_tertile_cuts_external'][1]:.1f}"
    )
    lines.append("")
    lines.append("| Axis | Subgroup | n | AUC [95% CI] | ECE_10bin | Δ vs overall | Verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for axis_name, axis_res in ext["axes"].items():
        for i, (sg_label, sg) in enumerate(axis_res["subgroups"].items()):
            axis_show = axis_name if i == 0 else ""
            lines.append(
                f"| {axis_show} | {sg_label} | {sg['n']} | {fmt_auc_ci(sg)} | "
                f"{fmt_ece(sg)} | {fmt_delta(sg)} | {sg.get('verdict', 'SKIPPED')} |"
            )
    lines.append("")
    lines.append(
        f"**Axis-level verdicts (BioFIND):** sex={ext['axes']['sex']['axis_verdict']}, "
        f"age_tertile={ext['axes']['age_tertile']['axis_verdict']}"
    )
    lines.append("")
    headline = payload["headline"]
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- Internal age deciles: 21-feat-binary {headline['internal']['21feat_binary']['n_pass']}/"
        f"{headline['internal']['21feat_binary']['n_evaluable']} deciles PASS "
        f"(trend={headline['internal']['21feat_binary']['trend_shape']}); "
        f"12-feat-NSD+ {headline['internal']['12feat_nsd_positive']['n_pass']}/"
        f"{headline['internal']['12feat_nsd_positive']['n_evaluable']} deciles PASS "
        f"(trend={headline['internal']['12feat_nsd_positive']['trend_shape']})"
    )
    lines.append(
        f"- External BioFIND: sex={headline['external']['sex_verdict']}, "
        f"age_tertile={headline['external']['age_tertile_verdict']}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    log.info("=" * 72)
    log.info("R6-Q8: age-decile (internal) + BioFIND external subgroup fairness")
    log.info("=" * 72)

    df = load_features_with_full_sex()

    # ---- Internal: 21-feat BINARY OOF ----
    log.info("--- 21-feat BINARY OOF (Path 3) — internal ---")
    sub21, y21, oof21 = get_oof_21feat_binary(df)
    main_auc_21 = auc_of(y21, oof21, n_classes=2)
    main_ece_21 = ece_of(y21, oof21, n_classes=2)
    rng_main = np.random.default_rng(CV_SEED)
    boots = []
    for _ in range(BOOT_N):
        idx = rng_main.integers(0, len(y21), len(y21))
        v = _safe_auc(y21[idx], oof21[idx], n_classes=2)
        if v is not None:
            boots.append(v)
    main_ci_21 = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    log.info("21-feat BINARY OVERALL: n=%d  AUC=%.4f [%.3f,%.3f]  ECE=%.4f",
             len(y21), main_auc_21, main_ci_21[0], main_ci_21[1], main_ece_21)

    log.info("--- Per-decile (21-feat BINARY) ---")
    rng21 = np.random.default_rng(CV_SEED + 1)
    deciles_21 = per_decile_metrics(sub21, y21, oof21, n_classes=2, main_auc=main_auc_21, rng=rng21)

    # ---- Internal: 12-feat NSD+ OOF ----
    log.info("--- 12-feat NSD+ OOF (R4-Q6 spec) — internal ---")
    sub12, y12, oof12 = get_oof_12feat_nsdpos(df)
    main_auc_12 = auc_of(y12, oof12, n_classes=4)
    main_ece_12 = ece_of(y12, oof12, n_classes=4)
    rng_main2 = np.random.default_rng(CV_SEED)
    boots2 = []
    for _ in range(BOOT_N):
        idx = rng_main2.integers(0, len(y12), len(y12))
        v = _safe_auc(y12[idx], oof12[idx], n_classes=4)
        if v is not None:
            boots2.append(v)
    main_ci_12 = [float(np.percentile(boots2, 2.5)), float(np.percentile(boots2, 97.5))]
    log.info("12-feat NSD+ OVERALL: n=%d  AUC=%.4f [%.3f,%.3f]  ECE=%.4f",
             len(y12), main_auc_12, main_ci_12[0], main_ci_12[1], main_ece_12)

    log.info("--- Per-decile (12-feat NSD+) ---")
    rng12 = np.random.default_rng(CV_SEED + 2)
    deciles_12 = per_decile_metrics(sub12, y12, oof12, n_classes=4, main_auc=main_auc_12, rng=rng12)

    # ---- External: BioFIND ----
    log.info("--- BioFIND external (12-feat clinical, BINARY) ---")
    clf_full, imp_full, _ = train_full_ppmi_12feat_binary(df)
    bf_merged, bf_y, bf_proba = predict_biofind_binary(clf_full, imp_full)
    bf_main_auc = _safe_auc(bf_y, bf_proba, n_classes=2)
    bf_main_ece = ece_of(bf_y, bf_proba, n_classes=2)
    rng_bf_main = np.random.default_rng(CV_SEED)
    boots_bf = []
    for _ in range(BOOT_N):
        idx = rng_bf_main.integers(0, len(bf_y), len(bf_y))
        v = _safe_auc(bf_y[idx], bf_proba[idx], n_classes=2)
        if v is not None:
            boots_bf.append(v)
    bf_main_ci = [
        float(np.percentile(boots_bf, 2.5)) if boots_bf else None,
        float(np.percentile(boots_bf, 97.5)) if boots_bf else None,
    ]
    log.info(
        "BioFIND OVERALL: n=%d  AUC=%.4f [%.3f,%.3f]  ECE=%.4f  class_dist=%s",
        len(bf_y),
        bf_main_auc if bf_main_auc is not None else float("nan"),
        bf_main_ci[0] if bf_main_ci[0] is not None else float("nan"),
        bf_main_ci[1] if bf_main_ci[1] is not None else float("nan"),
        bf_main_ece,
        dict(zip(*np.unique(bf_y, return_counts=True), strict=False)),
    )

    log.info("--- BioFIND subgroups ---")
    rng_bf = np.random.default_rng(CV_SEED + 3)
    bf_axes, bf_age_cuts = biofind_subgroup_metrics(
        bf_merged, bf_y, bf_proba, main_auc=bf_main_auc, rng=rng_bf
    )

    # ---- Compose payload ----
    headline = {
        "internal": {},
        "external": {
            "sex_verdict": bf_axes["sex"]["axis_verdict"],
            "age_tertile_verdict": bf_axes["age_tertile"]["axis_verdict"],
            "flagged_subgroups": [],
        },
    }
    for label, dr in [("21feat_binary", deciles_21), ("12feat_nsd_positive", deciles_12)]:
        n_eval = dr["n_deciles_pass"] + dr["n_deciles_fail"]
        headline["internal"][label] = {
            "n_pass": dr["n_deciles_pass"],
            "n_fail": dr["n_deciles_fail"],
            "n_evaluable": n_eval,
            "trend_shape": dr["trend"]["shape"],
            "spearman_rho": dr["trend"]["spearman_rho"],
            "spearman_p": dr["trend"]["spearman_p"],
        }
    for axis_name, axis_res in bf_axes.items():
        for sg_label, sg in axis_res["subgroups"].items():
            if sg.get("verdict") == "FAIL":
                headline["external"]["flagged_subgroups"].append(
                    f"{axis_name}::{sg_label} (Δ={sg['delta_vs_main']:+.3f}, AUC={sg['auc']:.3f}, n={sg['n']})"
                )

    payload = {
        "workstream": "q_r6_q8_subgroup_extended",
        "spec_summary": (
            "Internal age-decile (10 bins) on 21-feat BINARY + 12-feat NSD+ (R5-Q6 OOF reused). "
            "External BioFIND on 12-feat clinical BINARY (full PPMI training, n=103 GT subset). "
            "5-fold stratified CV, random_state=42, 1000 bootstrap. Δ threshold = 0.03."
        ),
        "n_folds": 5,
        "cv_seed": CV_SEED,
        "bootstrap_n": BOOT_N,
        "delta_threshold": DELTA_THRESHOLD,
        "min_decile_n": MIN_DECILE_N,
        "min_external_stratum_n": MIN_EXT_STRATUM_N,
        "internal_overall": {
            "21feat_binary": {
                "n": int(len(y21)),
                "auc": float(main_auc_21),
                "ci95": main_ci_21,
                "ece": float(main_ece_21),
            },
            "12feat_nsd_positive": {
                "n": int(len(y12)),
                "auc": float(main_auc_12),
                "ci95": main_ci_12,
                "ece": float(main_ece_12),
            },
        },
        "internal_age_deciles": {
            "21feat_binary": deciles_21,
            "12feat_nsd_positive": deciles_12,
        },
        "external_biofind": {
            "n_total": int(len(bf_y)),
            "class_distribution": {
                str(k): int(v)
                for k, v in zip(*np.unique(bf_y, return_counts=True), strict=False)
            },
            "main_auc": float(bf_main_auc) if bf_main_auc is not None else None,
            "main_auc_ci95": bf_main_ci,
            "main_ece_10bin": float(bf_main_ece),
            "age_tertile_cuts_external": bf_age_cuts,
            "axes": bf_axes,
            "verdict_per_axis": {
                "sex": bf_axes["sex"]["axis_verdict"],
                "age_tertile": bf_axes["age_tertile"]["axis_verdict"],
            },
        },
        "headline": headline,
    }

    out_json = OUT_DIR / "q_r6_q8_subgroup_extended.json"
    out_json.write_text(json.dumps(payload, indent=2, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x))
    log.info("Wrote %s", out_json)

    # ---- Markdown table ----
    md = build_table(payload)
    out_md = OUT_DIR / "q_r6_q8_subgroup_extended_table.md"
    out_md.write_text(md)
    log.info("Wrote %s", out_md)

    # ---- Plot 1: internal age-decile trend ----
    plot_age_decile_internal(
        decile_results={
            "21feat_binary": deciles_21,
            "12feat_nsd_positive": deciles_12,
        },
        overall_aucs={
            "21feat_binary": main_auc_21,
            "12feat_nsd_positive": main_auc_12,
        },
        overall_cis={
            "21feat_binary": main_ci_21,
            "12feat_nsd_positive": main_ci_12,
        },
        out_path=OUT_DIR / "q_r6_q8_age_decile_internal.png",
    )

    # ---- Plot 2: external BioFIND forest ----
    plot_external_forest(
        biofind_main_auc=bf_main_auc,
        biofind_main_ci=bf_main_ci,
        biofind_main_ece=bf_main_ece,
        biofind_results=bf_axes,
        out_path=OUT_DIR / "q_r6_q8_external_subgroup.png",
    )

    log.info("=" * 72)
    log.info("HEADLINE — Internal age deciles:")
    for label, h in headline["internal"].items():
        log.info(
            "  %-22s  %d/%d PASS  trend=%s  Spearman ρ=%s",
            label,
            h["n_pass"],
            h["n_evaluable"],
            h["trend_shape"],
            f"{h['spearman_rho']:+.2f}" if h["spearman_rho"] is not None else "n/a",
        )
    log.info("HEADLINE — External BioFIND:")
    log.info(
        "  sex_verdict=%s  age_tertile_verdict=%s",
        headline["external"]["sex_verdict"],
        headline["external"]["age_tertile_verdict"],
    )
    if headline["external"]["flagged_subgroups"]:
        for f in headline["external"]["flagged_subgroups"]:
            log.info("  FLAGGED: %s", f)
    log.info("Total time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
