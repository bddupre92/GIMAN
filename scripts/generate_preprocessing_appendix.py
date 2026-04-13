from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(
    "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025"
)
TODAY = date.today().isoformat()


RAW_MULTIMODAL_PATH = BASE_DIR / "data" / "03_prodromal" / "enhanced" / "prodromal_multimodal_features.csv"
IMPUTED_36_PATH = BASE_DIR / "data" / "03_prodromal" / "enhanced_36_features" / "prodromal_36_features_imputed.csv"
FINAL_PATH = BASE_DIR / "data" / "03_prodromal" / "final_training_dataset" / "unified_longitudinal_early_pd.csv"
PYG_META_PATH = BASE_DIR / "data" / "03_prodromal" / "final_pyg_data" / "pyg_data_metadata.json"


plt.style.use("seaborn-v0_8-whitegrid")


def _safe_feature_subset(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def _save_stage_summary(raw_df: pd.DataFrame, imputed_df: pd.DataFrame, final_df: pd.DataFrame) -> pd.DataFrame:
    stage_summary = pd.DataFrame(
        [
            {
                "stage": "raw_multimodal",
                "n_rows": len(raw_df),
                "n_patients": raw_df["PATNO"].nunique(),
                "n_columns": raw_df.shape[1],
            },
            {
                "stage": "imputed_36",
                "n_rows": len(imputed_df),
                "n_patients": imputed_df["PATNO"].nunique(),
                "n_columns": imputed_df.shape[1],
            },
            {
                "stage": "final_longitudinal",
                "n_rows": len(final_df),
                "n_patients": final_df["PATNO"].nunique(),
                "n_columns": final_df.shape[1],
            },
        ]
    )
    stage_summary.to_csv(TABLE_DIR / "stage_summary.csv", index=False)
    return stage_summary


def _plot_stage_overview(stage_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = ["n_rows", "n_patients", "n_columns"]
    titles = ["Rows", "Unique Patients", "Columns"]

    for ax, metric, title in zip(axes, metrics, titles):
        ax.bar(stage_summary["stage"], stage_summary[metric], color=["#4e79a7", "#f28e2b", "#59a14f"])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("GIMAN Preprocessing Stage Overview")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_stage_overview.png", dpi=300)
    plt.close(fig)


def _plot_missingness_before_after(raw_df: pd.DataFrame, imputed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    non_id_cols = [c for c in raw_df.columns if c != "PATNO"]

    raw_missing = raw_df[non_id_cols].isna().mean().sort_values(ascending=False).rename("raw_missing_rate")
    imputed_missing = (
        imputed_df[_safe_feature_subset(imputed_df, non_id_cols)]
        .isna()
        .mean()
        .reindex(raw_missing.index)
        .fillna(np.nan)
        .rename("imputed_missing_rate")
    )

    missing_compare = pd.concat([raw_missing, imputed_missing], axis=1)
    missing_compare.to_csv(TABLE_DIR / "missingness_before_after.csv")

    top = missing_compare.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(top))
    ax.barh(y + 0.2, top["raw_missing_rate"], height=0.35, label="Before imputation", color="#e15759")
    ax.barh(y - 0.2, top["imputed_missing_rate"], height=0.35, label="After imputation", color="#76b7b2")
    ax.set_yticks(y)
    ax.set_yticklabels(top.index)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Missing proportion")
    ax.set_title("Top Missing Features Before vs After Imputation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_missingness_before_after.png", dpi=300)
    plt.close(fig)

    return raw_missing.to_frame(), imputed_missing.to_frame()


def _plot_missing_indicator_prevalence(final_df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [c for c in final_df.columns if c.endswith("_missing")]
    missing_rates = final_df[missing_cols].mean().sort_values(ascending=False)
    missing_rates.to_frame("rate").to_csv(TABLE_DIR / "missing_indicator_prevalence.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(missing_rates.index, missing_rates.values, color="#edc948")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Prevalence of Missingness Indicators in Final Model Inputs")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_missing_indicator_prevalence.png", dpi=300)
    plt.close(fig)

    return missing_rates.to_frame("rate")


def _plot_event_structure(final_df: pd.DataFrame) -> pd.DataFrame:
    event_summary = (
        final_df.groupby("landmark_month", as_index=False)
        .agg(n_rows=("PATNO", "size"), n_events=("phenoconverted", "sum"))
        .assign(event_rate=lambda x: x["n_events"] / x["n_rows"])
        .sort_values("landmark_month")
    )
    event_summary.to_csv(TABLE_DIR / "event_summary_by_landmark.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(event_summary["landmark_month"].astype(str), event_summary["event_rate"], color="#59a14f")
    axes[0].set_title("Event Rate by Landmark Month")
    axes[0].set_ylabel("Event rate")

    event_vals = final_df.loc[final_df["phenoconverted"] == 1, "time_to_event"]
    censor_vals = final_df.loc[final_df["phenoconverted"] == 0, "time_to_event"]
    axes[1].hist(censor_vals, bins=20, alpha=0.7, label="Censored", color="#4e79a7")
    axes[1].hist(event_vals, bins=20, alpha=0.7, label="Events", color="#e15759")
    axes[1].set_title("time_to_event Distribution")
    axes[1].set_xlabel("Months")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_event_structure.png", dpi=300)
    plt.close(fig)

    return event_summary


def _plot_observed_vs_imputed(raw_df: pd.DataFrame, imputed_df: pd.DataFrame) -> pd.DataFrame:
    candidates = ["ALPHA_SYNUCLEIN", "TOTAL_TAU", "ABETA42", "PTAU181", "UPDRS_I", "UPDRS_II"]
    candidates = [c for c in candidates if c in raw_df.columns and c in imputed_df.columns]

    records: list[dict] = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for i, col in enumerate(candidates[:6]):
        ax = axes[i]
        observed = raw_df[col].dropna()
        all_vals = imputed_df[col].dropna()
        n_observed = len(observed)
        n_total = len(all_vals)

        if n_observed > 0:
            ax.hist(observed, bins=25, alpha=0.7, density=True, label="Observed", color="#4e79a7")
        ax.hist(all_vals, bins=25, alpha=0.5, density=True, label="After imputation", color="#f28e2b")

        ax.set_title(col)
        if n_observed == 0:
            ax.text(0.5, 0.9, "Fully imputed", transform=ax.transAxes, ha="center", fontsize=9)

        ax.legend(fontsize=8)
        records.append(
            {
                "feature": col,
                "observed_count_raw": n_observed,
                "total_count_imputed": n_total,
                "raw_missing_rate": float(raw_df[col].isna().mean()),
            }
        )

    for j in range(len(candidates[:6]), 6):
        axes[j].axis("off")

    fig.suptitle("Observed vs Post-Imputation Feature Distributions")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_observed_vs_imputed_distributions.png", dpi=300)
    plt.close(fig)

    comp_df = pd.DataFrame(records)
    comp_df.to_csv(TABLE_DIR / "observed_vs_imputed_summary.csv", index=False)
    return comp_df


def _plot_feature_correlation(final_df: pd.DataFrame, feature_names: list[str]) -> None:
    core = [
        "GENETIC_RISK_SCORE",
        "UPDRS_I",
        "UPDRS_II",
        "PIGD_SCORE",
        "TREMOR_SCORE",
        "CAUDATE_L_VOL",
        "PUTAMEN_L_VOL",
        "CAUDATE_L_SBR",
        "PUTAMEN_L_SBR",
        "ALPHA_SYNUCLEIN",
        "TOTAL_TAU",
        "ABETA42",
    ]
    cols = [c for c in core if c in feature_names and c in final_df.columns]
    corr = final_df[cols].corr(numeric_only=True)
    corr.to_csv(TABLE_DIR / "core_feature_correlation.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    ax.set_title("Core Input Feature Correlation Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_core_feature_correlation.png", dpi=300)
    plt.close(fig)


def _save_descriptive_stats(final_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    numeric_cols = [c for c in feature_names if c in final_df.columns and not c.endswith("_missing")]
    desc = final_df[numeric_cols].describe().T
    desc.to_csv(TABLE_DIR / "final_feature_descriptive_stats.csv")
    return desc


def _write_report(
    stage_summary: pd.DataFrame,
    raw_missing: pd.DataFrame,
    imputed_missing: pd.DataFrame,
    missing_indicators: pd.DataFrame,
    event_summary: pd.DataFrame,
    obs_imputed_summary: pd.DataFrame,
    feature_names: list[str],
) -> None:
    always_missing_raw = int((raw_missing["raw_missing_rate"] == 1.0).sum())
    indicators_full = int((missing_indicators["rate"] == 1.0).sum())
    event_rate = float((event_summary["n_events"].sum() / event_summary["n_rows"].sum()))

    lines = [
        f"# GIMAN Preprocessing Appendix Package ({TODAY})",
        "",
        "## Scope",
        "This appendix package documents preprocessing outputs feeding the current Phase 8+ survival/GAT pipeline.",
        "",
        "## Key Transparency Findings",
        f"- Stage sizes: raw multimodal={int(stage_summary.loc[stage_summary.stage == 'raw_multimodal', 'n_rows'].iloc[0])} rows, imputed_36={int(stage_summary.loc[stage_summary.stage == 'imputed_36', 'n_rows'].iloc[0])} rows, final_longitudinal={int(stage_summary.loc[stage_summary.stage == 'final_longitudinal', 'n_rows'].iloc[0])} rows.",
        f"- Raw multimodal features with 100% missingness: {always_missing_raw}.",
        f"- Missing-indicator columns at 100% prevalence in final model inputs: {indicators_full}.",
        f"- Final longitudinal event rate: {event_rate:.4f}.",
        f"- Feature channels consumed by model: {len(feature_names)}.",
        "",
        "## Figure Index",
        f"1. `{FIG_DIR / '01_stage_overview.png'}`",
        f"2. `{FIG_DIR / '02_missingness_before_after.png'}`",
        f"3. `{FIG_DIR / '03_missing_indicator_prevalence.png'}`",
        f"4. `{FIG_DIR / '04_event_structure.png'}`",
        f"5. `{FIG_DIR / '05_observed_vs_imputed_distributions.png'}`",
        f"6. `{FIG_DIR / '06_core_feature_correlation.png'}`",
        "",
        "## Table Index",
        f"- `{TABLE_DIR / 'stage_summary.csv'}`",
        f"- `{TABLE_DIR / 'missingness_before_after.csv'}`",
        f"- `{TABLE_DIR / 'missing_indicator_prevalence.csv'}`",
        f"- `{TABLE_DIR / 'event_summary_by_landmark.csv'}`",
        f"- `{TABLE_DIR / 'observed_vs_imputed_summary.csv'}`",
        f"- `{TABLE_DIR / 'core_feature_correlation.csv'}`",
        f"- `{TABLE_DIR / 'final_feature_descriptive_stats.csv'}`",
        "",
        "## Notes for Paper Methods Appendix",
        "- Report both imputation completion and missing-indicator prevalence so readers can see which modalities were largely inferred.",
        "- In methods text, explicitly separate patient-level imputation from longitudinal row expansion.",
        "- In figure captions, clarify that event labels are sparse and represented at landmark-expanded row level.",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate preprocessing appendix figures and tables.")
    parser.add_argument(
        "--output-root",
        default="Docs/audit",
        help="Workspace-relative output root. Example: visualizations/preprocessing_appendix",
    )
    args = parser.parse_args()

    output_root = BASE_DIR / args.output_root
    fig_dir = output_root / "appendix_figures" / f"preprocessing_{TODAY}"
    table_dir = output_root / "appendix_tables" / f"preprocessing_{TODAY}"
    report_path = output_root / f"GIMAN_PREPROCESSING_APPENDIX_{TODAY}.md"

    global FIG_DIR, TABLE_DIR, REPORT_PATH
    FIG_DIR = fig_dir
    TABLE_DIR = table_dir
    REPORT_PATH = report_path

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(RAW_MULTIMODAL_PATH)
    imputed_df = pd.read_csv(IMPUTED_36_PATH)
    final_df = pd.read_csv(FINAL_PATH)

    with PYG_META_PATH.open("r", encoding="utf-8") as f:
        pyg_meta = json.load(f)
    feature_names = pyg_meta["feature_names"]

    stage_summary = _save_stage_summary(raw_df, imputed_df, final_df)
    _plot_stage_overview(stage_summary)

    raw_missing, imputed_missing = _plot_missingness_before_after(raw_df, imputed_df)
    missing_indicators = _plot_missing_indicator_prevalence(final_df)
    event_summary = _plot_event_structure(final_df)
    obs_imputed_summary = _plot_observed_vs_imputed(raw_df, imputed_df)
    _plot_feature_correlation(final_df, feature_names)
    _save_descriptive_stats(final_df, feature_names)

    _write_report(
        stage_summary,
        raw_missing,
        imputed_missing,
        missing_indicators,
        event_summary,
        obs_imputed_summary,
        feature_names,
    )

    print("Appendix preprocessing package created:")
    print(f"- Figures: {FIG_DIR}")
    print(f"- Tables:  {TABLE_DIR}")
    print(f"- Report:  {REPORT_PATH}")


if __name__ == "__main__":
    main()
