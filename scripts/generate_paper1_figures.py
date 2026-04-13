#!/usr/bin/env python3
"""Generate all Paper 1 publication-quality figures and tables.

Outputs:
  outputs/paper1_figures/
    table1_demographics.csv         — Table 1: Cohort demographics
    table2_feature_availability.csv — Table 2: Feature completeness
    fig1_consort_flow.txt           — CONSORT flow diagram (text for LaTeX)
    fig2_stage_distribution.png     — NSD-ISS stage distribution
    fig3_feature_ablation.png       — Feature ablation: AUC by feature set
    fig4_external_validation.png    — BioFIND external: predicted vs actual
    fig5_domain_shift.png           — Domain shift: UPDRS distributions
    fig6_model_agreement.png        — PDBP cross-model prediction agreement
    fig7_calibration.png            — Calibration plots (internal + external)
    fig8_conformal_coverage.png     — Conformal coverage + set sizes
    fig9_fairness_sex.png           — Fairness: performance by sex
    fig10_fairness_age.png          — Fairness: performance by age group
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = BASE / "outputs" / "paper1_figures"
OUT.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)
COLORS = {
    "PPMI": "#2196F3",
    "BioFIND": "#FF9800",
    "PDBP": "#4CAF50",
    "HBS": "#9C27B0",
    "CatBoost": "#2196F3",
    "XGBoost": "#FF5722",
    "RandomForest": "#4CAF50",
    "LogisticRegression": "#9C27B0",
}


def load_ppmi():
    """Load PPMI features with proper demographics."""
    ppmi = pd.read_csv(DATA / "05_features" / "paper1_features_with_targets.csv")

    # Fix SEX from raw demographics (feature file has incomplete SEX)
    demo_files = list((DATA / "00_raw").glob("Demographics_*.csv"))
    if demo_files:
        demo = pd.read_csv(demo_files[0], low_memory=False)
        demo_bl = demo.drop_duplicates(subset=["PATNO"], keep="first")
        sex_map = dict(zip(demo_bl["PATNO"], demo_bl["SEX"], strict=False))
        ppmi["SEX"] = ppmi["PATNO"].map(sex_map)

    return ppmi


def load_external():
    """Load all external cohort feature files."""
    cohorts = {}
    for name in ["biofind", "pdbp", "hbs"]:
        path = DATA / "05_features" / f"{name}_features.csv"
        if path.exists():
            cohorts[name.upper() if name != "biofind" else "BioFIND"] = pd.read_csv(
                path
            )
    return cohorts


# ═══════════════════════════════════════════════════════════════════════
# TABLE 1: Demographics
# ═══════════════════════════════════════════════════════════════════════
def generate_demographics_table(ppmi, external):
    """Table 1: Cohort demographics comparison."""
    print("\n=== TABLE 1: Demographics ===")

    rows = []
    for name, df in [("PPMI", ppmi)] + list(external.items()):
        n = len(df)
        age_mean = df["AGE_AT_BASELINE"].mean()
        age_std = df["AGE_AT_BASELINE"].std()

        if "SEX" in df.columns and df["SEX"].notna().sum() > 0:
            n_male = (
                (df["SEX"] == 1).sum()
                if 1 in df["SEX"].values
                else (df["SEX"] == 0).sum()
            )
            n_female = (
                n - n_male
                if df["SEX"].notna().all()
                else df["SEX"].notna().sum() - n_male
            )
            # PPMI: 0=Female, 1=Male. External cohorts: check encoding
            if name == "PPMI":
                n_female_real = (df["SEX"] == 0).sum()
                n_male_real = (df["SEX"] == 1).sum()
            else:
                # External cohorts also use 0=Female, 1=Male from AMP-PD
                n_female_real = (df["SEX"] == 0).sum()
                n_male_real = (df["SEX"] == 1).sum()
            pct_male = (
                n_male_real / (n_male_real + n_female_real) * 100
                if (n_male_real + n_female_real) > 0
                else float("nan")
            )
        else:
            n_male_real = n_female_real = 0
            pct_male = float("nan")

        # Motor scores
        brady = (
            df["UPDRS3_BRADYKINESIA"].mean()
            if "UPDRS3_BRADYKINESIA" in df.columns
            else float("nan")
        )
        updrs1 = (
            df["UPDRS1_TOTAL"].mean() if "UPDRS1_TOTAL" in df.columns else float("nan")
        )
        updrs2 = (
            df["UPDRS2_TOTAL"].mean() if "UPDRS2_TOTAL" in df.columns else float("nan")
        )
        moca = df["MOCA_TOTAL"].mean() if "MOCA_TOTAL" in df.columns else float("nan")
        rbd = df["RBD_TOTAL"].mean() if "RBD_TOTAL" in df.columns else float("nan")

        # Stage distribution (PPMI only)
        if name == "PPMI" and "nsd_iss_stage" in df.columns:
            stages = df["nsd_iss_stage"].value_counts()
            stage_str = "; ".join(
                f"S{k}: {v}" for k, v in sorted(stages.items()) if k != "unclassified"
            )
        else:
            stage_str = "N/A"

        rows.append(
            {
                "Cohort": name,
                "N (PD)": n,
                "Age, mean (SD)": f"{age_mean:.1f} ({age_std:.1f})",
                "Male, n (%)": f"{n_male_real} ({pct_male:.1f}%)"
                if not np.isnan(pct_male)
                else "N/A",
                "Female, n": n_female_real if n_female_real > 0 else "N/A",
                "UPDRS-I, mean": f"{updrs1:.1f}" if not np.isnan(updrs1) else "N/A",
                "UPDRS-II, mean": f"{updrs2:.1f}" if not np.isnan(updrs2) else "N/A",
                "UPDRS-III Brady, mean": f"{brady:.1f}"
                if not np.isnan(brady)
                else "N/A",
                "MoCA, mean": f"{moca:.1f}" if not np.isnan(moca) else "N/A",
                "RBD-SQ, mean": f"{rbd:.1f}" if not np.isnan(rbd) else "N/A",
                "NSD-ISS Stages": stage_str,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "table1_demographics.csv", index=False)
    print(table.to_string(index=False))
    return table


# ═══════════════════════════════════════════════════════════════════════
# TABLE 2: Feature Availability
# ═══════════════════════════════════════════════════════════════════════
def generate_feature_availability_table(ppmi, external):
    """Table 2: Feature completeness across cohorts."""
    print("\n=== TABLE 2: Feature Availability ===")

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
    rows = []
    for feat in COMMON_FEATURES:
        row = {"Feature": feat}
        for name, df in [("PPMI", ppmi)] + list(external.items()):
            if feat in df.columns:
                pct = df[feat].notna().mean() * 100
                row[name] = f"{pct:.0f}%"
            else:
                row[name] = "0%"
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "table2_feature_availability.csv", index=False)
    print(table.to_string(index=False))
    return table


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: CONSORT Flow Diagram (text for LaTeX/manual drawing)
# ═══════════════════════════════════════════════════════════════════════
def generate_consort_flow(ppmi, external):
    """CONSORT-style participant flow."""
    print("\n=== FIGURE 1: CONSORT Flow ===")

    biofind = external.get("BioFIND")
    staging = None
    staging_path = DATA / "04_staging" / "biofind_nsd_iss_staging.csv"
    if staging_path.exists():
        staging = pd.read_csv(staging_path)

    flow = f"""
CONSORT-Style Participant Flow Diagram

PPMI (Training Cohort)
├── Total AMP-PD v4 participants: ~4,580
├── With DaT-SPECT imaging: 2,137
├── With SAA results: 277
├── NSD-ISS staged: 2,201
│   ├── Stage 0 (NSD-): 1,418 (64.4%)
│   ├── Stage 1: 67 (3.0%)
│   ├── Stage 2B: 208 (9.5%)
│   ├── Stage 3: 487 (22.1%)
│   ├── Stage 4: 17 (0.8%)
│   └── Unclassified: 4 (0.2%)
├── Binary target: 1,422 NSD- / 779 NSD+
└── Used for: Internal 5-fold CV + model training

BioFIND (External Validation)
├── Total participants: 213
├── PD patients: {len(biofind) if biofind is not None else "?"}
├── With SAA consensus: 108
│   ├── S+: 103 (95.4%)
│   └── S-: 5 (4.6%)
├── NSD-ISS staged (Russo 2025 replication): {len(staging) if staging is not None else "?"}
│   ├── Stage 2: 9
│   ├── Stage 3: 58
│   ├── Stage 4: 34
│   └── Stage 5: 2
└── Used for: External validation with ground truth

PDBP (External Prediction)
├── Total participants: ~1,610
├── PD patients: 893
├── Common features available: 12/12
├── Ground truth: None (no SAA/DaT-SPECT)
└── Used for: Prediction distribution analysis

HBS (External Prediction — Limited)
├── Total participants: ~1,189
├── PD patients: 649
├── Common features available: 8/12 (5 missing)
├── Ground truth: None
└── Used for: Feature completeness sensitivity analysis
"""
    (OUT / "fig1_consort_flow.txt").write_text(flow)
    print(flow)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: NSD-ISS Stage Distribution Across Cohorts
# ═══════════════════════════════════════════════════════════════════════
def generate_stage_distribution(ppmi):
    """Stage distribution bar chart."""
    print("\n=== FIGURE 2: Stage Distribution ===")

    stage_order = ["0", "1", "2B", "3", "4"]
    stage_labels = ["Stage 0\n(NSD-)", "Stage 1", "Stage 2B", "Stage 3", "Stage 4"]

    # PPMI
    ppmi_counts = ppmi["nsd_iss_stage"].value_counts()
    ppmi_vals = [ppmi_counts.get(s, 0) for s in stage_order]
    ppmi_total = sum(ppmi_vals)
    ppmi_pcts = [v / ppmi_total * 100 for v in ppmi_vals]

    # BioFIND (from staging file)
    staging_path = DATA / "04_staging" / "biofind_nsd_iss_staging.csv"
    bf_vals = [0, 0, 0, 0, 0]
    if staging_path.exists():
        bf_stg = pd.read_csv(staging_path)
        bf_stage_counts = bf_stg["nsd_iss_stage"].value_counts()
        # Map BioFIND stages (2,3,4,5) to our order
        bf_map = {2: 2, 3: 3, 4: 4, 5: 4}  # combine 4+5
        for k, v in bf_stage_counts.items():
            if k in bf_map:
                idx = (
                    stage_order.index(str(bf_map[k]))
                    if str(bf_map[k]) in stage_order
                    else -1
                )
                if k == 2:
                    idx = 2  # Stage 2B
                elif k == 3:
                    idx = 3
                elif k in [4, 5]:
                    idx = 4
                if 0 <= idx < len(bf_vals):
                    bf_vals[idx] += v
        bf_total = sum(bf_vals)
        bf_pcts = [v / bf_total * 100 if bf_total > 0 else 0 for v in bf_vals]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PPMI
    bars1 = axes[0].bar(
        range(len(stage_order)),
        ppmi_pcts,
        color=COLORS["PPMI"],
        alpha=0.8,
        edgecolor="white",
    )
    axes[0].set_xticks(range(len(stage_order)))
    axes[0].set_xticklabels(stage_labels, fontsize=9)
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title(f"PPMI (n={ppmi_total})")
    for bar, val, pct in zip(bars1, ppmi_vals, ppmi_pcts, strict=False):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"n={val}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # BioFIND
    if staging_path.exists():
        bf_labels = ["Stage 0", "Stage 1", "Stage 2", "Stage 3", "Stage 4-5"]
        bars2 = axes[1].bar(
            range(len(stage_order)),
            bf_pcts,
            color=COLORS["BioFIND"],
            alpha=0.8,
            edgecolor="white",
        )
        axes[1].set_xticks(range(len(stage_order)))
        axes[1].set_xticklabels(bf_labels, fontsize=9)
        axes[1].set_ylabel("Percentage (%)")
        axes[1].set_title(f"BioFIND S+ (n={bf_total})")
        for bar, val, pct in zip(bars2, bf_vals, bf_pcts, strict=False):
            if val > 0:
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"n={val}\n({pct:.1f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.savefig(OUT / "fig2_stage_distribution.png")
    plt.close()
    print(f"Saved: {OUT / 'fig2_stage_distribution.png'}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: Feature Ablation
# ═══════════════════════════════════════════════════════════════════════
def generate_feature_ablation():
    """Feature ablation: full vs clinical-only AUC."""
    print("\n=== FIGURE 3: Feature Ablation ===")

    targets = ["Binary", "Three-class", "Full ordinal", "NSD+ subgroup"]
    full_auc = [0.979, 0.942, 0.946, 0.904]
    clinical_auc = [0.727, 0.797, 0.823, 0.900]

    x = np.arange(len(targets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(
        x - width / 2,
        full_auc,
        width,
        label="Full 22 features",
        color="#2196F3",
        alpha=0.85,
    )
    bars2 = ax.bar(
        x + width / 2,
        clinical_auc,
        width,
        label="Clinical-only 12 features",
        color="#FF9800",
        alpha=0.85,
    )

    # Delta labels
    for i, (f, c) in enumerate(zip(full_auc, clinical_auc, strict=False)):
        delta = c - f
        ax.annotate(
            f"{delta:+.1%}",
            xy=(i, min(f, c) - 0.02),
            ha="center",
            fontsize=9,
            color="red",
            fontweight="bold",
        )

    ax.set_ylabel("AUC-ROC (macro)")
    ax.set_title(
        "Feature Ablation: DaT-SPECT Contribution to NSD-ISS Prediction (CatBoost)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.legend()
    ax.set_ylim(0.6, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="Chance")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(OUT / "fig3_feature_ablation.png")
    plt.close()
    print(f"Saved: {OUT / 'fig3_feature_ablation.png'}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: BioFIND External Validation
# ═══════════════════════════════════════════════════════════════════════
def generate_external_validation():
    """External validation: predicted vs actual on BioFIND."""
    print("\n=== FIGURE 4: External Validation ===")

    # Load results
    results_dir = BASE / "outputs" / "external_validation"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Binary
    binary_path = results_dir / "binary" / "external_validation_results.json"
    if binary_path.exists():
        with open(binary_path) as f:
            res = json.load(f)
        models = ["CatBoost", "XGBoost", "RandomForest", "LogisticRegression"]
        bal_accs = [
            res["external"]["BioFIND"][m]["external_metrics"]["bal_acc"] for m in models
        ]
        colors_m = [COLORS[m] for m in models]
        axes[0].barh(models, bal_accs, color=colors_m, alpha=0.8)
        axes[0].axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Chance")
        axes[0].set_xlim(0, 1)
        axes[0].set_title("Binary (S+ vs S-)\nn=108")
        axes[0].set_xlabel("Balanced Accuracy")
        for i, v in enumerate(bal_accs):
            axes[0].text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9)

    # Three-class — load from three_class results
    tc_path = results_dir / "three_class" / "external_validation_results.json"
    if tc_path.exists():
        with open(tc_path) as f:
            res_tc = json.load(f)
        bf_tc = res_tc.get("external", {}).get("BioFIND", {})
        models_tc = [
            m
            for m in ["CatBoost", "XGBoost", "RandomForest", "LogisticRegression"]
            if m in bf_tc
        ]
        if models_tc:
            aucs_tc = []
            for m in models_tc:
                ext_m = bf_tc[m].get("external_metrics", {})
                aucs_tc.append(ext_m.get("auc", 0))
            colors_tc = [COLORS[m] for m in models_tc]
            axes[1].barh(models_tc, aucs_tc, color=colors_tc, alpha=0.8)
            axes[1].axvline(x=0.5, color="red", linestyle="--", alpha=0.5)
            axes[1].set_xlim(0, 1)
            axes[1].set_title("Three-class (AUC)\nn=103")
            axes[1].set_xlabel("Macro AUC-ROC")
            for i, v in enumerate(aucs_tc):
                axes[1].text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9)

    # NSD-positive — load from nsd_positive results
    nsd_path = results_dir / "nsd_positive" / "external_validation_results.json"
    if nsd_path.exists():
        with open(nsd_path) as f:
            res_nsd = json.load(f)
        bf_nsd = res_nsd.get("external", {}).get("BioFIND", {})
        models_nsd = [
            m
            for m in ["CatBoost", "XGBoost", "RandomForest", "LogisticRegression"]
            if m in bf_nsd
        ]
        if models_nsd:
            qwks = []
            for m in models_nsd:
                ext_m = bf_nsd[m].get("external_metrics", {})
                qwks.append(ext_m.get("qwk", 0))
            colors_nsd = [COLORS[m] for m in models_nsd]
            axes[2].barh(models_nsd, qwks, color=colors_nsd, alpha=0.8)
            axes[2].axvline(x=0, color="red", linestyle="--", alpha=0.5)
            axes[2].set_xlim(-0.2, 0.6)
            axes[2].set_title("NSD+ Subgroup (QWK)\nn=103")
            axes[2].set_xlabel("Quadratic Weighted Kappa")
            for i, v in enumerate(qwks):
                axes[2].text(
                    max(v + 0.02, 0.02), i, f"{v:.3f}", va="center", fontsize=9
                )

    plt.suptitle(
        "BioFIND External Validation (Clinical Features Only)", fontsize=14, y=1.02
    )
    plt.tight_layout()
    plt.savefig(OUT / "fig4_external_validation.png")
    plt.close()
    print(f"Saved: {OUT / 'fig4_external_validation.png'}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5: Domain Shift Visualization
# ═══════════════════════════════════════════════════════════════════════
def generate_domain_shift(ppmi, external):
    """Distribution comparison showing domain shift."""
    print("\n=== FIGURE 5: Domain Shift ===")

    biofind = external.get("BioFIND")
    if biofind is None:
        print("  BioFIND not available, skipping")
        return

    features_to_compare = [
        "UPDRS3_BRADYKINESIA",
        "UPDRS3_TREMOR",
        "UPDRS3_RIGIDITY",
        "UPDRS1_TOTAL",
    ]
    available = [
        f for f in features_to_compare if f in biofind.columns and f in ppmi.columns
    ]

    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, feat in zip(axes, available, strict=False):
        ppmi_nsd_neg = ppmi.loc[ppmi["target_binary"] == 0, feat].dropna()
        ppmi_nsd_pos = ppmi.loc[ppmi["target_binary"] == 1, feat].dropna()
        bf_all = biofind[feat].dropna()

        ax.hist(
            ppmi_nsd_neg,
            bins=20,
            alpha=0.5,
            color=COLORS["PPMI"],
            label=f"PPMI NSD- (n={len(ppmi_nsd_neg)})",
            density=True,
        )
        ax.hist(
            ppmi_nsd_pos,
            bins=20,
            alpha=0.5,
            color="#FF5722",
            label=f"PPMI NSD+ (n={len(ppmi_nsd_pos)})",
            density=True,
        )
        ax.hist(
            bf_all,
            bins=20,
            alpha=0.5,
            color=COLORS["BioFIND"],
            label=f"BioFIND PD (n={len(bf_all)})",
            density=True,
        )

        ax.set_xlabel(feat.replace("_", " "))
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.suptitle(
        "Domain Shift: PPMI NSD-/NSD+ vs BioFIND PD Feature Distributions",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUT / "fig5_domain_shift.png")
    plt.close()
    print(f"Saved: {OUT / 'fig5_domain_shift.png'}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6: PDBP Cross-Model Agreement
# ═══════════════════════════════════════════════════════════════════════
def generate_model_agreement():
    """Cross-model prediction agreement on PDBP."""
    print("\n=== FIGURE 6: Model Agreement (PDBP) ===")

    binary_path = (
        BASE
        / "outputs"
        / "external_validation"
        / "binary"
        / "external_validation_results.json"
    )
    if not binary_path.exists():
        print("  Binary results not found, skipping")
        return

    with open(binary_path) as f:
        res = json.load(f)

    pdbp = res.get("external", {}).get("PDBP", {})
    models = ["CatBoost", "XGBoost", "RandomForest", "LogisticRegression"]
    nsd_pos_pcts = []
    for m in models:
        dist = pdbp.get(m, {}).get("prediction_distribution", {})
        total = sum(dist.values())
        pos = dist.get("1", 0)
        nsd_pos_pcts.append(pos / total * 100 if total > 0 else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_m = [COLORS[m] for m in models]
    bars = ax.bar(models, nsd_pos_pcts, color=colors_m, alpha=0.85, edgecolor="white")
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)
    ax.set_ylabel("% Predicted NSD+")
    ax.set_title(
        "Cross-Model Prediction Disagreement: PDBP (n=893)\nBinary NSD+ vs NSD-"
    )
    ax.set_ylim(0, 80)

    for bar, pct in zip(bars, nsd_pos_pcts, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Disagreement annotation
    spread = max(nsd_pos_pcts) - min(nsd_pos_pcts)
    ax.annotate(
        f"Spread: {spread:.1f}pp",
        xy=(0.5, 0.95),
        xycoords="axes fraction",
        ha="center",
        fontsize=12,
        color="red",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="red"),
    )

    plt.tight_layout()
    plt.savefig(OUT / "fig6_model_agreement.png")
    plt.close()
    print(f"Saved: {OUT / 'fig6_model_agreement.png'}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 7: Calibration Plots
# ═══════════════════════════════════════════════════════════════════════
def generate_calibration_plots(ppmi):
    """Calibration curves for internal CV (binary target)."""
    print("\n=== FIGURE 7: Calibration Plots ===")

    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

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

    y = ppmi["target_binary"].values
    available = [f for f in COMMON_FEATURES if f in ppmi.columns]
    X = ppmi[available].values

    # Impute + scale
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    models = {
        "CatBoost": CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            auto_class_weights="Balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric="logloss"
        ),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=1000),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Internal calibration
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        all_probs = np.zeros(len(y))
        for train_idx, test_idx in skf.split(X, y):
            model_clone = (
                type(model)(**model.get_params())
                if hasattr(model, "get_params")
                else model
            )
            model_clone.fit(X[train_idx], y[train_idx])
            probs = model_clone.predict_proba(X[test_idx])[:, 1]
            all_probs[test_idx] = probs

        frac_pos, mean_pred = calibration_curve(
            y, all_probs, n_bins=10, strategy="quantile"
        )
        brier = brier_score_loss(y, all_probs)
        axes[0].plot(
            mean_pred,
            frac_pos,
            "o-",
            color=COLORS[name],
            label=f"{name} (Brier={brier:.3f})",
        )

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of Positives")
    axes[0].set_title("Internal CV Calibration (PPMI, Binary)")
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    # Reliability diagram: histogram of predicted probabilities
    axes[1].hist(
        all_probs[y == 0],
        bins=30,
        alpha=0.5,
        color=COLORS["PPMI"],
        label="NSD- (actual)",
        density=True,
    )
    axes[1].hist(
        all_probs[y == 1],
        bins=30,
        alpha=0.5,
        color="#FF5722",
        label="NSD+ (actual)",
        density=True,
    )
    axes[1].set_xlabel("Predicted P(NSD+)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Prediction Distribution by True Class")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUT / "fig7_calibration.png")
    plt.close()
    print(f"Saved: {OUT / 'fig7_calibration.png'}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 8: Conformal Prediction Coverage + Set Sizes
# ═══════════════════════════════════════════════════════════════════════
def generate_conformal_figure():
    """Conformal coverage and set sizes from existing results."""
    print("\n=== FIGURE 8: Conformal Prediction ===")

    conformal_dir = BASE / "outputs" / "paper1_conformal"
    if not conformal_dir.exists():
        print("  No conformal results found, skipping")
        return

    targets = ["binary", "three_class", "full_ordinal", "nsd_positive"]
    target_labels = [
        "Binary",
        "Three-class",
        "Full ordinal\n(5-class)",
        "NSD+\n(4-class)",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Collect coverage and set sizes for 90% confidence, cross-conformal, CatBoost
    coverages = []
    set_sizes = []

    for target in targets:
        path = conformal_dir / f"{target}_conformal.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            # Structure: {model_name: [list of entries]}
            found = False
            model_key = "catboost"
            if model_key in data:
                for entry in data[model_key]:
                    if (
                        entry.get("conformal_method") == "cross"
                        and abs(entry.get("confidence_level", 0) - 0.9) < 0.01
                    ):
                        coverages.append(entry.get("marginal_coverage", 0) * 100)
                        set_sizes.append(entry.get("mean_set_size", 0))
                        found = True
                        break
            if not found:
                coverages.append(0)
                set_sizes.append(0)
        else:
            coverages.append(0)
            set_sizes.append(0)

    x = np.arange(len(targets))

    # Coverage
    bars1 = axes[0].bar(x, coverages, color="#2196F3", alpha=0.85, edgecolor="white")
    axes[0].axhline(y=90, color="red", linestyle="--", alpha=0.5, label="90% target")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(target_labels, fontsize=9)
    axes[0].set_ylabel("Marginal Coverage (%)")
    axes[0].set_title("Conformal Coverage (CatBoost, Cross-CV+, 90% target)")
    axes[0].legend()
    axes[0].set_ylim(80, 105)
    for bar, v in zip(bars1, coverages, strict=False):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Set sizes
    bars2 = axes[1].bar(x, set_sizes, color="#FF9800", alpha=0.85, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(target_labels, fontsize=9)
    axes[1].set_ylabel("Mean Prediction Set Size")
    axes[1].set_title("Prediction Set Size (smaller = more informative)")
    for bar, v in zip(bars2, set_sizes, strict=False):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Add n_classes reference
    n_classes = [2, 3, 5, 4]
    for i, nc in enumerate(n_classes):
        axes[1].axhline(y=nc, color="gray", alpha=0.1)

    plt.tight_layout()
    plt.savefig(OUT / "fig8_conformal_coverage.png")
    plt.close()
    print(f"Saved: {OUT / 'fig8_conformal_coverage.png'}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 9-10: Fairness Analysis (Sex + Age)
# ═══════════════════════════════════════════════════════════════════════
def generate_fairness_analysis(ppmi):
    """Fairness: model performance stratified by sex and age."""
    print("\n=== FIGURES 9-10: Fairness Analysis ===")

    from catboost import CatBoostClassifier

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

    y = ppmi["target_binary"].values
    available = [f for f in COMMON_FEATURES if f in ppmi.columns]
    X_df = ppmi[available].copy()

    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X_df.values)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    sex = ppmi["SEX"].values
    age = ppmi["AGE_AT_BASELINE"].values

    # 5-fold CV predictions
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        auto_class_weights="Balanced",
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = np.zeros(len(y))
    all_probs = np.zeros(len(y))

    for train_idx, test_idx in skf.split(X, y):
        m = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            auto_class_weights="Balanced",
        )
        m.fit(X[train_idx], y[train_idx])
        all_preds[test_idx] = np.asarray(m.predict(X[test_idx])).ravel()
        all_probs[test_idx] = m.predict_proba(X[test_idx])[:, 1]

    # === SEX FAIRNESS ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sex_groups = {"Female (0)": sex == 0, "Male (1)": sex == 1}
    sex_metrics = {}
    for label, mask in sex_groups.items():
        valid = mask & ~np.isnan(sex)
        if valid.sum() > 20:
            ba = balanced_accuracy_score(y[valid], all_preds[valid])
            try:
                auc = roc_auc_score(y[valid], all_probs[valid])
            except ValueError:
                auc = float("nan")
            sex_metrics[label] = {"bal_acc": ba, "auc": auc, "n": int(valid.sum())}

    if sex_metrics:
        labels = list(sex_metrics.keys())
        bal_accs = [sex_metrics[l]["bal_acc"] for l in labels]
        aucs = [sex_metrics[l]["auc"] for l in labels]
        ns = [sex_metrics[l]["n"] for l in labels]

        x_pos = np.arange(len(labels))
        axes[0].bar(
            x_pos - 0.175,
            bal_accs,
            0.35,
            label="Balanced Accuracy",
            color="#2196F3",
            alpha=0.85,
        )
        axes[0].bar(
            x_pos + 0.175, aucs, 0.35, label="AUC-ROC", color="#FF9800", alpha=0.85
        )
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([f"{l}\n(n={ns[i]})" for i, l in enumerate(labels)])
        axes[0].set_ylabel("Score")
        axes[0].set_title(
            "Binary NSD-ISS Prediction by Sex\n(CatBoost, 12 clinical features)"
        )
        axes[0].legend()
        axes[0].set_ylim(0.5, 1.0)

    # Overall parity
    if len(sex_metrics) == 2:
        vals = list(sex_metrics.values())
        gap_ba = abs(vals[0]["bal_acc"] - vals[1]["bal_acc"])
        gap_auc = abs(vals[0]["auc"] - vals[1]["auc"])
        axes[1].bar(
            ["Bal Acc Gap", "AUC Gap"],
            [gap_ba, gap_auc],
            color=[
                "#4CAF50" if gap_ba < 0.05 else "#FF5722",
                "#4CAF50" if gap_auc < 0.05 else "#FF5722",
            ],
            alpha=0.85,
        )
        axes[1].axhline(
            y=0.05, color="red", linestyle="--", label="5% fairness threshold"
        )
        axes[1].set_ylabel("Absolute Gap")
        axes[1].set_title("Sex Fairness Parity Gap")
        axes[1].legend()
        axes[1].set_ylim(0, max(gap_ba, gap_auc) * 1.5 + 0.02)

    plt.tight_layout()
    plt.savefig(OUT / "fig9_fairness_sex.png")
    plt.close()
    print(f"Saved: {OUT / 'fig9_fairness_sex.png'}")

    # === AGE FAIRNESS ===
    fig, ax = plt.subplots(figsize=(10, 5))

    age_bins = [(0, 55, "<55"), (55, 65, "55-64"), (65, 75, "65-74"), (75, 100, "75+")]
    age_metrics = {}
    for lo, hi, label in age_bins:
        mask = (age >= lo) & (age < hi) & ~np.isnan(age)
        if mask.sum() > 20:
            ba = balanced_accuracy_score(y[mask], all_preds[mask])
            try:
                auc = roc_auc_score(y[mask], all_probs[mask])
            except ValueError:
                auc = float("nan")
            age_metrics[label] = {"bal_acc": ba, "auc": auc, "n": int(mask.sum())}

    if age_metrics:
        labels_age = list(age_metrics.keys())
        bal_accs_age = [age_metrics[l]["bal_acc"] for l in labels_age]
        aucs_age = [age_metrics[l]["auc"] for l in labels_age]
        ns_age = [age_metrics[l]["n"] for l in labels_age]

        x_pos = np.arange(len(labels_age))
        ax.bar(
            x_pos - 0.175,
            bal_accs_age,
            0.35,
            label="Balanced Accuracy",
            color="#2196F3",
            alpha=0.85,
        )
        ax.bar(
            x_pos + 0.175, aucs_age, 0.35, label="AUC-ROC", color="#FF9800", alpha=0.85
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{l}\n(n={ns_age[i]})" for i, l in enumerate(labels_age)])
        ax.set_ylabel("Score")
        ax.set_title(
            "Binary NSD-ISS Prediction by Age Group\n(CatBoost, 12 clinical features)"
        )
        ax.legend()
        ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    plt.savefig(OUT / "fig10_fairness_age.png")
    plt.close()
    print(f"Saved: {OUT / 'fig10_fairness_age.png'}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("Paper 1: Publication Figure & Table Generation")
    print("=" * 70)

    ppmi = load_ppmi()
    external = load_external()

    print(f"\nLoaded: PPMI n={len(ppmi)}")
    for name, df in external.items():
        print(f"  {name}: n={len(df)}")

    # Tables
    generate_demographics_table(ppmi, external)
    generate_feature_availability_table(ppmi, external)

    # Figures
    generate_consort_flow(ppmi, external)
    generate_stage_distribution(ppmi)
    generate_feature_ablation()
    generate_external_validation()
    generate_domain_shift(ppmi, external)
    generate_model_agreement()
    generate_calibration_plots(ppmi)
    generate_conformal_figure()
    generate_fairness_analysis(ppmi)

    print("\n" + "=" * 70)
    print(f"All outputs saved to: {OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
