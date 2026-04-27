"""Generate two new publication figures for the JAMIA submission of Paper 5.

Fig 5 (enrollment_timeline_splits):
    Shows the 14-year PPMI enrollment period with the four temporal-window
    boundaries (W1/W2/W3/W4) overlaid. Makes the expanding-window design
    concrete for reviewers.

Fig 6 (pdmedyn_over_time):
    Shows baseline PD-medication prevalence (pdmedyn) as a function of
    enrollment date. Directly visualises the dominant covariate-shift driver
    identified by the PSI analysis (PSI 0.26-0.29 across W2-W4).

Inputs:
    data/00_raw/GIMAN/ppmi_data_csv/Demographics_30Sep2025.csv (INFODT per PATNO)
    data/06_longitudinal_staging/longitudinal_nsd_iss.csv (pdmedyn at baseline)

Outputs:
    outputs/mechanistic_twin/paper5_submission/figures/fig5_enrollment_timeline_splits.{png,pdf}
    outputs/mechanistic_twin/paper5_submission/figures/fig6_pdmedyn_over_time.{png,pdf}

Palette: Okabe-Ito colourblind-safe.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv" / "Demographics_30Sep2025.csv"
LONG = REPO / "data" / "06_longitudinal_staging" / "longitudinal_nsd_iss.csv"
OUT = REPO / "outputs" / "mechanistic_twin" / "paper5_submission" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito palette
OI_BLACK = "#000000"
OI_VERMILLION = "#D55E00"
OI_BLUE_GREEN = "#009E73"
OI_SKY_BLUE   = "#56B4E9"
OI_ORANGE     = "#E69F00"
OI_BLUE       = "#0072B2"
OI_REDDISH_PURPLE = "#CC79A7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_enrollment_dates() -> pd.DataFrame:
    """Return one-row-per-patient DataFrame: [PATNO, enroll_date] for patients that
    appear in the Paper 3 longitudinal cohort.
    """
    demo = pd.read_csv(DEMO, usecols=["PATNO", "EVENT_ID", "INFODT"], dtype={"PATNO": "Int64"})
    # INFODT is MM/YYYY; take the earliest per patient as enrollment
    demo["enroll_date"] = pd.to_datetime(demo["INFODT"], format="%m/%Y", errors="coerce")
    demo = demo.dropna(subset=["enroll_date"])
    enroll = demo.groupby("PATNO", as_index=False)["enroll_date"].min()

    long_df = pd.read_csv(LONG, usecols=["PATNO"], dtype={"PATNO": "Int64"})
    cohort = long_df["PATNO"].dropna().unique()
    enroll = enroll[enroll["PATNO"].isin(cohort)].reset_index(drop=True)
    return enroll


def load_pdmedyn_baseline() -> pd.DataFrame:
    """Baseline pdmedyn (first visit per patient) joined with enrollment date."""
    long_df = pd.read_csv(LONG, usecols=["PATNO", "EVENT_ID", "pdmedyn", "months_from_baseline"],
                          dtype={"PATNO": "Int64"})
    # baseline = row with minimum months_from_baseline per patient
    long_df = long_df.sort_values(["PATNO", "months_from_baseline"])
    base = long_df.groupby("PATNO", as_index=False).first()
    base = base[["PATNO", "pdmedyn"]]
    enroll = load_enrollment_dates()
    return base.merge(enroll, on="PATNO", how="inner").dropna(subset=["pdmedyn"])


def compute_window_boundaries(enroll: pd.DataFrame) -> dict:
    """Return enrollment-date cut-points that define the 4 temporal windows.
    The implementation in Paper 5 sorts patients by enrollment date and cuts
    by rank, so we reproduce that here.
    """
    sorted_dates = enroll["enroll_date"].sort_values().values
    n = len(sorted_dates)
    def q(frac): return sorted_dates[int(round(frac * n)) - 1] if frac > 0 else sorted_dates[0]
    return {
        "W1_train_end": q(0.40),
        "W1_test_end":  q(0.60),
        "W2_train_end": q(0.60),
        "W2_test_end":  q(0.80),
        "W3_train_end": q(0.80),
        "W3_test_end":  q(1.00),
        "W4_train_end": q(0.50),
        "W4_test_end":  q(1.00),
        "cohort_start": q(0.001),
        "cohort_end":   q(1.00),
        "n": n,
    }


def fig5_enrollment_timeline() -> None:
    enroll = load_enrollment_dates()
    bounds = compute_window_boundaries(enroll)
    n = bounds["n"]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 5.8),
                                          gridspec_kw={"height_ratios": [1.0, 0.7]})

    # --- Top: histogram of enrollment dates with window boundaries ---
    dates = pd.to_datetime(enroll["enroll_date"]).values.astype("datetime64[M]")
    years = pd.to_datetime(enroll["enroll_date"]).dt.year + pd.to_datetime(enroll["enroll_date"]).dt.month / 12.0
    bins = np.arange(years.min(), years.max() + 0.5, 0.5)  # 6-month bins
    ax_top.hist(years, bins=bins, color=OI_SKY_BLUE, edgecolor=OI_BLUE, alpha=0.8)
    ax_top.set_xlabel("Enrollment year")
    ax_top.set_ylabel("Number of patients (6-month bins)")
    ax_top.set_title(f"PPMI enrollment timeline (n={n} patients, 2010-2024)",
                     fontsize=11, fontweight="bold")

    # Overlay window boundaries
    W1_end = pd.Timestamp(bounds["W1_train_end"]).year + pd.Timestamp(bounds["W1_train_end"]).month/12
    W2_end = pd.Timestamp(bounds["W2_train_end"]).year + pd.Timestamp(bounds["W2_train_end"]).month/12
    W3_end = pd.Timestamp(bounds["W3_train_end"]).year + pd.Timestamp(bounds["W3_train_end"]).month/12
    for x, label, color in [(W1_end, "40% (W1 train|test)", OI_VERMILLION),
                             (W2_end, "60% (W2 train|test)", OI_ORANGE),
                             (W3_end, "80% (W3 train|test)", OI_BLUE_GREEN)]:
        ax_top.axvline(x, ls="--", color=color, linewidth=1.8, alpha=0.85)
        ax_top.text(x, ax_top.get_ylim()[1] * 0.95, label, rotation=90,
                    va="top", ha="right", fontsize=7, color=color, fontweight="bold")

    # --- Bottom: window bars showing train/test range ---
    rows = [
        ("W1", bounds["cohort_start"], bounds["W1_train_end"], bounds["W1_test_end"]),
        ("W2", bounds["cohort_start"], bounds["W2_train_end"], bounds["W2_test_end"]),
        ("W3", bounds["cohort_start"], bounds["W3_train_end"], bounds["W3_test_end"]),
        ("W4", bounds["cohort_start"], bounds["W4_train_end"], bounds["W4_test_end"]),
    ]
    y_positions = [3, 2, 1, 0]
    for (label, start, train_end, test_end), y in zip(rows, y_positions):
        start_y = pd.Timestamp(start).year + pd.Timestamp(start).month/12
        train_end_y = pd.Timestamp(train_end).year + pd.Timestamp(train_end).month/12
        test_end_y = pd.Timestamp(test_end).year + pd.Timestamp(test_end).month/12
        ax_bot.barh(y, train_end_y - start_y, left=start_y, height=0.6,
                    color=OI_BLUE, edgecolor=OI_BLACK, label="Train" if y == 3 else None)
        ax_bot.barh(y, test_end_y - train_end_y, left=train_end_y, height=0.6,
                    color=OI_VERMILLION, edgecolor=OI_BLACK, label="Test" if y == 3 else None)
        ax_bot.text(start_y - 0.2, y, label, ha="right", va="center",
                    fontsize=10, fontweight="bold")

    ax_bot.set_yticks([])
    ax_bot.set_xlim(ax_top.get_xlim())
    ax_bot.set_xlabel("Enrollment year")
    ax_bot.set_title("Expanding-window temporal splits (train = blue, test = vermillion)",
                     fontsize=10)
    ax_bot.legend(loc="upper right", fontsize=8, ncol=2)
    ax_bot.spines[["top", "right", "left"]].set_visible(False)
    ax_bot.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = OUT / f"fig5_enrollment_timeline_splits.{ext}"
        fig.savefig(p)
        print(f"  wrote {p.relative_to(REPO)}")
    plt.close(fig)


def fig6_pdmedyn_over_time() -> None:
    df = load_pdmedyn_baseline()
    df["pdmedyn"] = pd.to_numeric(df["pdmedyn"], errors="coerce")
    df = df.dropna(subset=["pdmedyn"])
    df["year_bin"] = pd.cut(pd.to_datetime(df["enroll_date"]).dt.year,
                             bins=np.arange(2010, 2026, 2),
                             labels=[f"{y}-{y+1}" for y in range(2010, 2024, 2)],
                             include_lowest=True, right=False)

    g = df.groupby("year_bin", observed=True)["pdmedyn"].agg(["mean", "count"])
    g["se"] = np.sqrt(g["mean"] * (1 - g["mean"]) / g["count"].clip(lower=1))
    g = g.reset_index()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(g))
    ax.bar(x, g["mean"], yerr=g["se"], color=OI_VERMILLION, alpha=0.85,
           edgecolor=OI_BLACK, linewidth=0.6, capsize=4,
           label="Baseline PD-medication prevalence")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in g["year_bin"]], fontsize=9, rotation=0)
    ax.set_ylabel("Proportion of patients on PD medication at baseline")
    ax.set_xlabel("Enrollment year bin (2-year windows)")
    ax.set_title("Covariate shift driver: baseline $\\mathtt{pdmedyn}$ prevalence across the "
                 "14-year PPMI enrollment period",
                 fontsize=11, fontweight="bold")
    # Annotate counts
    for i, (_, row) in enumerate(g.iterrows()):
        ax.text(i, row["mean"] + row["se"] + 0.015,
                f"{row['mean']*100:.0f}%\nn={int(row['count'])}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylim(0, min(1.0, g["mean"].max() + g["se"].max() + 0.12))
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    # Note about why this matters
    ax.text(0.02, 0.97,
            "PSI = 0.26-0.29 across W2-W4\n"
            "reflects evolving PD-medication\n"
            "prescribing practices (2010-2024).",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8, style="italic", color=OI_BLACK,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=OI_BLACK, linewidth=0.5, alpha=0.85))

    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = OUT / f"fig6_pdmedyn_over_time.{ext}"
        fig.savefig(p)
        print(f"  wrote {p.relative_to(REPO)}")
    plt.close(fig)


if __name__ == "__main__":
    print("Regenerating Paper 5 submission figures...")
    fig5_enrollment_timeline()
    fig6_pdmedyn_over_time()
    print("Done.")
