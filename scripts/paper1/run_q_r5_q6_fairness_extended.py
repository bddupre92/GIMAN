"""Paper 1 R5-Q6 — Extended subgroup fairness on 21-feat BINARY + 12-feat NSD+.

Reviewer 5 asks: "Can you provide stratified performance and calibration by
sex, age tertiles, and key genetic subgroups (LRRK2, GBA), particularly for the
clinical-only sub-staging model?"

Existing data covered:
  - Fig 8 = per-genotype subgroup AUC for the **21-feat primary BINARY** target
  - q9_extended_subgroup.json has age-band stratification (not tertiles) on 21-feat

Gap (this script closes):
  - Equivalent fairness analysis for **12-feat clinical-only NSD+ sub-staging**
    (the deployment-relevant CatBoost-12 model from R4-Q6, AUC 0.899)
  - Age-tertile stratification on BOTH models (q9 used pre-set bands <60/60-70/>=70)
  - Calibration ECE per subgroup
  - Side-by-side subgroup AUCs in one JSON / one MD / one 6-panel PNG

Outputs:
  outputs/paper1_r2_responses/q_r5_q6_fairness_extended.json
  outputs/paper1_r2_responses/q_r5_q6_fairness_extended_table.md
  outputs/paper1_r2_responses/q_r5_q6_fairness_extended.png

Verdict per axis (sex/age/carrier × 2 models = 6 axes):
  PASS = max |Δ AUC vs main| < 0.03 AND interaction p > 0.05
  FAIL = either threshold breached
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
from scipy.stats import chi2
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.paper1.run_fold_local_imputation import (  # noqa: E402
    FEATURES_PATH,
    HIGH_MISS_COLS,
    STAGING_COLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("q_r5_q6_fairness")

OUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEMOGRAPHICS_PATH = ROOT / "data" / "00_raw" / "Demographics_08Feb2026.csv"

N_FOLDS = 5
CV_SEED = 42
BOOT_N = 1000
MIN_STRATUM_N = 30  # Slightly relaxed (vs 40 in q9) because NSD+ subgroup is smaller (n=779)
ECE_BINS = 10
PATH3_EXCLUDE = {"CAUDATE_PUTAMEN_RATIO"}

# 12-feature clinical common subset (matches q_r4_q6)
COMMON_FEATURES_12 = [
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

# Okabe-Ito palette
C_REF = "#0072B2"  # blue (reference / overall)
C_LRRK2 = "#E69F00"  # orange
C_GBA = "#CC79A7"  # pink
C_APOE = "#009E73"  # green
C_NEUTRAL = "#56B4E9"  # sky blue (sex / age tertiles)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_features_with_full_sex() -> pd.DataFrame:
    """Load paper1_features_with_targets and patch SEX from Demographics."""
    df = pd.read_csv(FEATURES_PATH)
    log.info("Loaded %d patients × %d cols from features", len(df), df.shape[1])
    # Patch SEX (849 NaN in features file — gotcha documented in CLAUDE.md)
    if df["SEX"].isna().sum() > 0:
        demo = pd.read_csv(DEMOGRAPHICS_PATH)
        sex_per_pat = (
            demo.dropna(subset=["SEX"]).groupby("PATNO")["SEX"].first().astype(int)
        )
        n_before = df["SEX"].notna().sum()
        df["SEX"] = df["PATNO"].map(sex_per_pat).fillna(df["SEX"])
        n_after = df["SEX"].notna().sum()
        log.info(
            "Patched SEX from Demographics: coverage %d -> %d (PPMI SCREEN: 0=Female, 1=Male)",
            n_before,
            n_after,
        )
    return df


def auc_of(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float:
    if n_classes == 2:
        return roc_auc_score(y_true, y_proba[:, 1])
    return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")


def ece_of(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int, n_bins: int = ECE_BINS) -> float:
    """Expected Calibration Error.

    For binary: top-class probability vs binary label.
    For multiclass: max-confidence top-1 ECE (most common multiclass formulation).
    """
    if len(y_true) == 0:
        return float("nan")
    if n_classes == 2:
        confidence = y_proba[:, 1]
        correctness = (y_true == 1).astype(float)
    else:
        pred = np.argmax(y_proba, axis=1)
        confidence = np.max(y_proba, axis=1)
        correctness = (pred == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (confidence >= bin_edges[i]) & (confidence < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (confidence >= bin_edges[i]) & (confidence <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = confidence[mask].mean()
        bin_acc = correctness[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Per-model OOF prediction generators
# ---------------------------------------------------------------------------
def get_oof_21feat_binary(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Reproduce 21-feat Path 3 BINARY OOF (matches Fig 8 / q_r2_w3 setup)."""
    feat_cols = [
        c
        for c in df.columns
        if c not in STAGING_COLS and c not in HIGH_MISS_COLS and c not in PATH3_EXCLUDE
    ]
    mask = df["target_binary"] >= 0
    sub = df[mask].copy().reset_index(drop=True)
    X = sub[feat_cols].to_numpy(dtype=float)
    y = sub["target_binary"].astype(int).to_numpy()
    log.info("21-feat BINARY: n=%d, n_features=%d", len(sub), X.shape[1])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    oof = np.zeros((len(y), 2))
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X[tr])
        X_te = imp.transform(X[te])
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        clf = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            random_seed=42,
            verbose=False,
            auto_class_weights="Balanced",
        )
        clf.fit(X_tr, y[tr])
        oof[te] = clf.predict_proba(X_te)
    return sub, y, oof


def get_oof_12feat_nsdpos(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Reproduce 12-feat NSD+ CatBoost OOF (matches q_r4_q6 setup)."""
    sub = df[df["nsd_iss_stage"].isin(["1", "2B", "3", "4"])].copy().reset_index(drop=True)
    m = {"1": 0, "2B": 1, "3": 2, "4": 3}
    sub["target"] = sub["nsd_iss_stage"].map(m).astype(int)
    X = sub[COMMON_FEATURES_12].to_numpy(dtype=float)
    y = sub["target"].to_numpy(dtype=int)
    n_classes = 4
    log.info("12-feat NSD+: n=%d, n_classes=%d", len(sub), n_classes)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    oof = np.zeros((len(y), n_classes))
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X[tr])
        X_te = imp.transform(X[te])
        # NOTE: q_r4_q6 does NOT use StandardScaler (CatBoost is scale-invariant)
        clf = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            auto_class_weights="Balanced",
            verbose=0,
            random_seed=42,
            eval_metric="TotalF1",
        )
        clf.fit(X_tr, y[tr])
        oof[te] = clf.predict_proba(X_te)
    return sub, y, oof


# ---------------------------------------------------------------------------
# Stratification builders
# ---------------------------------------------------------------------------
def build_strata(sub: pd.DataFrame, age_tertile_cuts: np.ndarray | None = None) -> dict:
    """Returns a dict {axis_name: {label: bool_mask}}.

    Sex: PPMI SCREEN convention 0=Female, 1=Male.
    Age: tertiles computed from this cohort (or from passed cuts to share across models).
    Carrier: mutually-exclusive priority LRRK2 > GBA > APOE > Non-carrier (matches Fig 8).
    """
    sex = sub["SEX"].fillna(-1).astype(int).values
    age = sub["AGE_AT_BASELINE"].astype(float).values
    if age_tertile_cuts is None:
        valid_age = age[~np.isnan(age)]
        age_tertile_cuts = np.percentile(valid_age, [33.3333, 66.6667])

    lrrk2 = sub["LRRK2_CARRIER"].fillna(0).astype(bool).values
    gba = sub["GBA_CARRIER"].fillna(0).astype(bool).values
    apoe = sub["APOE_E4_CARRIER"].fillna(0).astype(bool).values

    strata = {
        "sex": {
            "Female": sex == 0,
            "Male": sex == 1,
        },
        "age_tertile": {
            f"T1_<{age_tertile_cuts[0]:.1f}y": age < age_tertile_cuts[0],
            f"T2_{age_tertile_cuts[0]:.1f}-{age_tertile_cuts[1]:.1f}y": (
                age >= age_tertile_cuts[0]
            )
            & (age < age_tertile_cuts[1]),
            f"T3_>={age_tertile_cuts[1]:.1f}y": age >= age_tertile_cuts[1],
        },
        "carrier": {
            "LRRK2+": lrrk2,
            "GBA+ only": gba & ~lrrk2,
            "APOE-e4+ only": apoe & ~lrrk2 & ~gba,
            "Non-carrier": ~lrrk2 & ~gba & ~apoe,
        },
    }
    return strata, age_tertile_cuts


def _safe_auc(yt: np.ndarray, pp: np.ndarray, n_classes: int) -> float | None:
    """AUC that gracefully handles classes absent from a subgroup.

    For binary, requires both classes present. For multiclass, computes one-vs-rest
    macro AUC over only the classes that ARE present in y_true (drop missing-class
    columns to avoid 0-positive errors). Returns None if fewer than 2 classes present.
    """
    classes_present = np.unique(yt)
    if len(classes_present) < 2:
        return None
    if n_classes == 2:
        try:
            return float(roc_auc_score(yt, pp[:, 1]))
        except ValueError:
            return None
    # Multiclass: compute macro OVR over the classes that ARE present
    aucs_per_class = []
    for c in classes_present:
        y_bin = (yt == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        try:
            aucs_per_class.append(roc_auc_score(y_bin, pp[:, c]))
        except ValueError:
            continue
    if len(aucs_per_class) == 0:
        return None
    return float(np.mean(aucs_per_class))


def stratum_metrics(
    y: np.ndarray, oof: np.ndarray, n_classes: int, mask: np.ndarray, rng: np.random.Generator
) -> dict:
    """Return n, AUC (per-subgroup-classes-present), 95% CI, ECE for a subgroup."""
    yt = y[mask]
    pp = oof[mask]
    n_present = int(len(np.unique(yt)))
    if mask.sum() < MIN_STRATUM_N or n_present < 2:
        return {
            "n": int(mask.sum()),
            "auc": None,
            "ci95_lo": None,
            "ci95_hi": None,
            "ece_10bin": None,
            "n_classes_present": n_present,
            "skipped": True,
            "reason": f"n<{MIN_STRATUM_N} or <2 classes",
        }
    auc = _safe_auc(yt, pp, n_classes)
    if auc is None:
        return {
            "n": int(mask.sum()),
            "auc": None,
            "ci95_lo": None,
            "ci95_hi": None,
            "ece_10bin": None,
            "n_classes_present": n_present,
            "skipped": True,
            "reason": "AUC undefined",
        }
    ece = ece_of(yt, pp, n_classes, n_bins=ECE_BINS)
    boots = []
    for _ in range(BOOT_N):
        idx = rng.integers(0, len(yt), len(yt))
        v = _safe_auc(yt[idx], pp[idx], n_classes)
        if v is not None:
            boots.append(v)
    lo = float(np.percentile(boots, 2.5)) if len(boots) >= 10 else None
    hi = float(np.percentile(boots, 97.5)) if len(boots) >= 10 else None
    return {
        "n": int(mask.sum()),
        "auc": float(auc),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "ece_10bin": ece,
        "skipped": False,
        "n_classes_present": n_present,
        "auc_method": "macro_OVR_classes_present" if n_classes > 2 else "binary",
    }


def interaction_chi2(stratum_aucs: list[float], stratum_cis: list[tuple[float, float]]) -> dict:
    """Chi2 interaction test using CI half-widths as SE proxy (matches q9 method)."""
    if len(stratum_aucs) < 2:
        return {"chi2_stat": None, "df": 0, "p_raw": None}
    aucs = np.array(stratum_aucs)
    mean_auc = float(np.mean(aucs))
    ses = np.array([(hi - lo) / (2 * 1.96) for lo, hi in stratum_cis])
    z_stat = float(np.sum(((aucs - mean_auc) / np.maximum(ses, 1e-6)) ** 2))
    p_raw = float(1 - chi2.cdf(z_stat, df=len(aucs) - 1))
    return {"chi2_stat": z_stat, "df": int(len(aucs) - 1), "p_raw": p_raw}


# ---------------------------------------------------------------------------
# Model run wrapper
# ---------------------------------------------------------------------------
def analyze_model(
    model_label: str,
    sub: pd.DataFrame,
    y: np.ndarray,
    oof: np.ndarray,
    n_classes: int,
    age_tertile_cuts: np.ndarray | None = None,
) -> dict:
    main_auc = auc_of(y, oof, n_classes)
    main_ece = ece_of(y, oof, n_classes)
    rng = np.random.default_rng(CV_SEED)
    boots = []
    for _ in range(BOOT_N):
        idx = rng.integers(0, len(y), len(y))
        v = _safe_auc(y[idx], oof[idx], n_classes)
        if v is not None:
            boots.append(v)
    main_ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    log.info(
        "%s OVERALL: n=%d  AUC=%.4f [%.3f,%.3f]  ECE=%.4f",
        model_label,
        len(y),
        main_auc,
        main_ci[0],
        main_ci[1],
        main_ece,
    )

    strata, cuts_used = build_strata(sub, age_tertile_cuts=age_tertile_cuts)

    out = {
        "model_spec": model_label,
        "n_total": int(len(y)),
        "n_classes": n_classes,
        "main_auc": float(main_auc),
        "main_auc_ci95": main_ci,
        "main_ece_10bin": float(main_ece),
        "age_tertile_cuts": [float(c) for c in cuts_used],
        "axes": {},
    }

    rng2 = np.random.default_rng(CV_SEED)
    for axis_name, groups in strata.items():
        per_subgroup = {}
        cur_aucs = []
        cur_cis = []
        for label, mask in groups.items():
            r = stratum_metrics(y, oof, n_classes, mask, rng2)
            per_subgroup[label] = r
            if not r["skipped"] and r["auc"] is not None:
                cur_aucs.append(r["auc"])
                cur_cis.append((r["ci95_lo"], r["ci95_hi"]))
                log.info(
                    "  %-12s  %-22s  n=%-4d  AUC=%.4f [%.3f,%.3f]  Δ=%+.4f  ECE=%.4f",
                    axis_name,
                    label,
                    r["n"],
                    r["auc"],
                    r["ci95_lo"],
                    r["ci95_hi"],
                    r["auc"] - main_auc,
                    r["ece_10bin"],
                )
            else:
                log.info(
                    "  %-12s  %-22s  SKIPPED (%s)",
                    axis_name,
                    label,
                    r.get("reason", "?"),
                )
        inter = interaction_chi2(cur_aucs, cur_cis)
        max_delta = max((abs(a - main_auc) for a in cur_aucs), default=None)
        out["axes"][axis_name] = {
            "subgroups": per_subgroup,
            "interaction_test": inter,
            "max_abs_delta_vs_main": float(max_delta) if max_delta is not None else None,
        }
    return out


# ---------------------------------------------------------------------------
# Markdown table builder
# ---------------------------------------------------------------------------
def fmt_auc_ci(r: dict) -> str:
    if r.get("skipped") or r.get("auc") is None:
        return "—"
    return f"{r['auc']:.3f} [{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]"


def fmt_ece(r: dict) -> str:
    if r.get("skipped") or r.get("ece_10bin") is None:
        return "—"
    return f"{r['ece_10bin']:.3f}"


def build_table(model_results: dict[str, dict]) -> str:
    lines = [
        "# R5-Q6: Extended subgroup fairness — 21-feat BINARY vs 12-feat NSD+ sub-staging",
        "",
        "Side-by-side stratified AUC + 10-bin ECE across **sex, age tertiles, and "
        "mutually-exclusive genetic carrier strata** (LRRK2 > GBA > APOE > Non-carrier).",
        "",
        "**Verdict per axis:** PASS = max |Δ AUC vs overall| < 0.03 AND interaction p > 0.05.",
        "",
    ]
    for label, res in model_results.items():
        lines.append(f"## {label}  (Overall AUC = {res['main_auc']:.3f} "
                     f"[{res['main_auc_ci95'][0]:.3f}, {res['main_auc_ci95'][1]:.3f}], "
                     f"ECE = {res['main_ece_10bin']:.3f}, n = {res['n_total']:,})")
        lines.append("")
        lines.append(
            f"Age tertile cuts (years): "
            f"T1<{res['age_tertile_cuts'][0]:.1f}, "
            f"T2={res['age_tertile_cuts'][0]:.1f}–{res['age_tertile_cuts'][1]:.1f}, "
            f"T3≥{res['age_tertile_cuts'][1]:.1f}"
        )
        lines.append("")
        lines.append(
            "| Axis | Subgroup | n | AUC [95% CI] | ECE_10bin | Δ vs overall | Interaction p | Verdict |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for axis_name, axis_res in res["axes"].items():
            inter = axis_res["interaction_test"]
            p_raw = inter.get("p_raw")
            max_delta = axis_res["max_abs_delta_vs_main"]
            verdict = (
                "PASS"
                if (
                    max_delta is not None
                    and max_delta < 0.03
                    and (p_raw is None or p_raw > 0.05)
                )
                else "FAIL"
            )
            for i, (sg_label, sg) in enumerate(axis_res["subgroups"].items()):
                axis_show = axis_name if i == 0 else ""
                p_show = f"{p_raw:.3f}" if (i == 0 and p_raw is not None) else ""
                v_show = verdict if i == 0 else ""
                delta_show = (
                    f"{sg['auc'] - res['main_auc']:+.3f}"
                    if (not sg.get("skipped") and sg.get("auc") is not None)
                    else "—"
                )
                lines.append(
                    f"| {axis_show} | {sg_label} | {sg.get('n', 0)} | "
                    f"{fmt_auc_ci(sg)} | {fmt_ece(sg)} | {delta_show} | {p_show} | {v_show} |"
                )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Forest plot (6 panels = sex/age/carrier × 21-feat-binary/12-feat-NSD+)
# ---------------------------------------------------------------------------
def plot_forest(model_results: dict[str, dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.5), dpi=300, sharex=False)
    axis_order = ["sex", "age_tertile", "carrier"]
    model_order = list(model_results.keys())
    color_map = {
        "sex": [C_NEUTRAL, C_NEUTRAL],
        "age_tertile": [C_NEUTRAL, C_NEUTRAL, C_NEUTRAL],
        "carrier": [C_LRRK2, C_GBA, C_APOE, C_REF],
    }
    axis_titles = {
        "sex": "Sex",
        "age_tertile": "Age tertile",
        "carrier": "Genetic carrier",
    }

    for col_i, model_label in enumerate(model_order):
        res = model_results[model_label]
        main_auc = res["main_auc"]
        main_lo, main_hi = res["main_auc_ci95"]
        for row_i, axis_name in enumerate(axis_order):
            ax = axes[row_i, col_i]
            axis_res = res["axes"][axis_name]
            subgroups = list(axis_res["subgroups"].keys())
            aucs = []
            los = []
            his = []
            ns = []
            labels = []
            colors = []
            for k, label in enumerate(subgroups):
                sg = axis_res["subgroups"][label]
                if sg.get("skipped") or sg.get("auc") is None:
                    aucs.append(np.nan)
                    los.append(np.nan)
                    his.append(np.nan)
                else:
                    aucs.append(sg["auc"])
                    los.append(sg["ci95_lo"])
                    his.append(sg["ci95_hi"])
                ns.append(sg.get("n", 0))
                labels.append(f"{label}\n(n={sg.get('n', 0)})")
                colors.append(color_map[axis_name][min(k, len(color_map[axis_name]) - 1)])

            y_pos = np.arange(len(subgroups))[::-1]
            # Reference band
            ax.axvspan(main_lo, main_hi, alpha=0.18, color=C_REF, zorder=0)
            ax.axvline(main_auc, color=C_REF, lw=0.8, alpha=0.6, ls="--", zorder=1)
            # Points + CIs
            for i, (a, lo, hi, c) in enumerate(zip(aucs, los, his, colors)):
                if np.isnan(a):
                    ax.text(0.5, y_pos[i], "skipped (n<30)", ha="center", va="center",
                            fontsize=7, color="grey", transform=ax.get_yaxis_transform())
                    continue
                ax.errorbar(
                    [a],
                    [y_pos[i]],
                    xerr=[[a - lo], [hi - a]],
                    fmt="o",
                    capsize=3,
                    markersize=7,
                    ecolor="#333",
                    markerfacecolor=c,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    linewidth=1.0,
                )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=7)
            # Verdict in title
            inter = axis_res["interaction_test"]
            p_raw = inter.get("p_raw")
            md = axis_res["max_abs_delta_vs_main"]
            verdict = (
                "PASS"
                if (md is not None and md < 0.03 and (p_raw is None or p_raw > 0.05))
                else "FAIL"
            )
            v_color = "#009E73" if verdict == "PASS" else "#D55E00"
            title_top = f"{axis_titles[axis_name]} — {verdict}"
            if row_i == 0:
                title_top = f"{model_label}\n{title_top}"
            ax.set_title(title_top, fontsize=8.5, color=v_color if verdict == "FAIL" else "black")
            ax.set_xlim(0.50, 1.0)
            ax.axvline(0.5, color="grey", lw=0.4, alpha=0.4)
            if row_i == 2:
                ax.set_xlabel("AUC [95% bootstrap CI]", fontsize=8)
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="x", linestyle=":", linewidth=0.3, alpha=0.4)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle(
        "R5-Q6: Subgroup fairness — sex, age tertile, genetic carrier × 2 models",
        fontsize=10,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    log.info("Wrote %s", out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    log.info("=" * 72)
    log.info("R5-Q6 extended subgroup fairness (21-feat BINARY + 12-feat NSD+)")
    log.info("=" * 72)

    df = load_features_with_full_sex()

    # ---- Model 1: 21-feat BINARY ----
    log.info("--- Building OOF for 21-feat BINARY (Path 3) ---")
    sub21, y21, oof21 = get_oof_21feat_binary(df)
    res21 = analyze_model("21feat_binary", sub21, y21, oof21, n_classes=2)

    # Use SAME age tertile cuts on the NSD+ subgroup so cohort-comparable
    age_cuts = np.array(res21["age_tertile_cuts"])

    # ---- Model 2: 12-feat NSD+ sub-staging ----
    log.info("--- Building OOF for 12-feat NSD+ sub-staging (R4-Q6 spec) ---")
    sub12, y12, oof12 = get_oof_12feat_nsdpos(df)
    # NOTE: NSD+ subgroup has its own age distribution; reuse 21-feat cuts to keep tertile
    # boundaries cohort-comparable (reviewer-friendly).
    res12 = analyze_model(
        "12feat_nsd_positive", sub12, y12, oof12, n_classes=4, age_tertile_cuts=age_cuts
    )

    model_results = {
        "21feat_binary": res21,
        "12feat_nsd_positive": res12,
    }

    # ---- Compute per-axis PASS/FAIL counts ----
    axes_summary = {}
    pass_count = 0
    total_count = 0
    fails = []
    worst_per_model = {}
    for model_label, res in model_results.items():
        worst_axis = None
        worst_delta = -1.0
        worst_label = None
        for axis_name, axis_res in res["axes"].items():
            inter = axis_res["interaction_test"]
            p_raw = inter.get("p_raw")
            md = axis_res["max_abs_delta_vs_main"]
            verdict = (
                "PASS"
                if (md is not None and md < 0.03 and (p_raw is None or p_raw > 0.05))
                else "FAIL"
            )
            axes_summary[f"{model_label}::{axis_name}"] = {
                "verdict": verdict,
                "max_abs_delta_vs_main": md,
                "interaction_p_raw": p_raw,
            }
            total_count += 1
            if verdict == "PASS":
                pass_count += 1
            else:
                fails.append(f"{model_label}::{axis_name} (Δ={md:.3f}, p={p_raw:.3f})")
            # Find worst |Δ| subgroup
            for sg_label, sg in axis_res["subgroups"].items():
                if sg.get("skipped") or sg.get("auc") is None:
                    continue
                d = abs(sg["auc"] - res["main_auc"])
                if d > worst_delta:
                    worst_delta = d
                    worst_axis = axis_name
                    worst_label = sg_label
        worst_per_model[model_label] = {
            "axis": worst_axis,
            "subgroup": worst_label,
            "abs_delta": float(worst_delta) if worst_delta >= 0 else None,
        }

    payload = {
        "workstream": "q_r5_q6_fairness_extended",
        "spec_summary": (
            "21-feat Path3 CatBoost binary + 12-feat clinical-only CatBoost NSD+ "
            "(R4-Q6 spec). 5-fold stratified CV, random_state=42, 1000 bootstrap. "
            "Carrier strata mutually exclusive (LRRK2 > GBA > APOE > Non-carrier)."
        ),
        "n_folds": N_FOLDS,
        "cv_seed": CV_SEED,
        "bootstrap_n": BOOT_N,
        "min_stratum_n": MIN_STRATUM_N,
        "ece_n_bins": ECE_BINS,
        "models": model_results,
        "axes_verdict_summary": axes_summary,
        "headline": {
            "n_axes_total": total_count,
            "n_axes_pass": pass_count,
            "n_axes_fail": total_count - pass_count,
            "fails": fails,
            "worst_per_model": worst_per_model,
        },
    }

    out_json = OUT_DIR / "q_r5_q6_fairness_extended.json"
    out_json.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s", out_json)

    md = build_table(model_results)
    out_md = OUT_DIR / "q_r5_q6_fairness_extended_table.md"
    out_md.write_text(md)
    log.info("Wrote %s", out_md)

    out_png = OUT_DIR / "q_r5_q6_fairness_extended.png"
    plot_forest(model_results, out_png)

    log.info("=" * 72)
    log.info("HEADLINE: %d/%d axes PASS", pass_count, total_count)
    if fails:
        for f in fails:
            log.info("  FAIL: %s", f)
    for ml, w in worst_per_model.items():
        log.info(
            "  Worst |Δ| in %s: %s/%s (|Δ|=%.3f)",
            ml,
            w["axis"],
            w["subgroup"],
            w["abs_delta"] if w["abs_delta"] is not None else float("nan"),
        )
    log.info("Total time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
