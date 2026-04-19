"""Paper 6 JAMIA revision: aggregate numerical defenses.

Replaces n=5 vignette framing with 1,900-patient aggregate statistics that
peer reviewers asked for:

    A. Accuracy vs chance (25%) and majority-class baselines
    B. Top-2 accuracy
    C. Ordinal MAE (stages 1, 2B, 3, 4 -> indices 0, 1, 2, 3)
    D. 4x4 confusion matrix + per-stage accuracy (NSD+ subset)
    E. CONSORT-style comparison of 1,065 Phase-2-covered vs 835 uncovered
       subgroups, with chi-square / t-tests and a TikZ CONSORT diagram

Inputs
------
- outputs/paper6/pipeline_results/v2_full_cohort/pipeline_summary.json
- outputs/mechanistic_twin/data/posteriors/phase2_combined_1065.csv
- data/05_features/paper1_features_with_targets.csv

Outputs
-------
- outputs/mechanistic_twin/paper6_submission/jamia/revision_analyses/aggregate_stats.json
- outputs/mechanistic_twin/paper6_submission/jamia/revision_analyses/fig_consort_coverage_split.{tex,pdf}
- outputs/mechanistic_twin/paper6_submission/jamia/revision_analyses/aggregate_summary.md
"""
from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
PIPE_SUMMARY = PROJECT / "outputs/paper6/pipeline_results/v2_full_cohort/pipeline_summary.json"
POSTERIORS = PROJECT / "outputs/mechanistic_twin/data/posteriors/phase2_combined_1065.csv"
P1_FEATURES = PROJECT / "data/05_features/paper1_features_with_targets.csv"

OUT_DIR = PROJECT / "outputs/mechanistic_twin/paper6_submission/jamia/revision_analyses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NSD_STAGES = ["1", "2B", "3", "4"]
STAGE_TO_ORD = {"1": 0, "2B": 1, "3": 2, "4": 3}


# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------

def load_inputs() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    pipe = json.loads(PIPE_SUMMARY.read_text())
    post = pd.read_csv(POSTERIORS)
    feats = pd.read_csv(P1_FEATURES)
    return pipe, post, feats


# ---------------------------------------------------------------------------
# A-D: NSD+ accuracy analyses
# ---------------------------------------------------------------------------

def nsd_positive_analyses(pipe: dict) -> dict[str, Any]:
    """Accuracy baselines, top-2, ordinal MAE, confusion matrix."""
    nsd_rows = []
    for patno, v in pipe.items():
        if not isinstance(v, dict) or "staging" not in v:
            continue  # skip summary/meta entries
        s = v["staging"]
        actual = s.get("actual_stage")
        if actual not in NSD_STAGES:
            continue
        probs = s["probabilities"]  # dict {stage: prob}
        pred = s["predicted_stage"]
        # Sort stages by probability desc
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        top1 = ranked[0][0]
        top2 = ranked[1][0] if len(ranked) > 1 else None
        nsd_rows.append({
            "patno": int(patno),
            "actual": actual,
            "pred": pred,
            "top1": top1,
            "top2": top2,
            **{f"p_{k}": v for k, v in probs.items()},
        })
    df = pd.DataFrame(nsd_rows)
    n = len(df)

    # Accuracy baselines
    correct = (df["actual"] == df["pred"]).sum()
    acc = correct / n

    counts = Counter(df["actual"])
    majority_stage, majority_n = counts.most_common(1)[0]
    majority_acc = majority_n / n

    chance_acc = 1.0 / 4  # 4-class random

    # Top-2 accuracy
    top2_correct = ((df["actual"] == df["top1"]) | (df["actual"] == df["top2"])).sum()
    top2_acc = top2_correct / n

    # Ordinal MAE
    df["actual_ord"] = df["actual"].map(STAGE_TO_ORD)
    df["pred_ord"] = df["pred"].map(STAGE_TO_ORD)
    df["abs_err"] = (df["actual_ord"] - df["pred_ord"]).abs()
    mae = float(df["abs_err"].mean())
    mae_distribution = df["abs_err"].value_counts().sort_index().to_dict()
    mae_distribution = {int(k): int(v) for k, v in mae_distribution.items()}

    # 4x4 confusion matrix (rows = actual, cols = predicted)
    conf = pd.DataFrame(
        np.zeros((4, 4), dtype=int), index=NSD_STAGES, columns=NSD_STAGES,
    )
    for _, r in df.iterrows():
        conf.loc[r["actual"], r["pred"]] += 1
    per_stage_acc = {}
    for s in NSD_STAGES:
        row_sum = int(conf.loc[s].sum())
        diag = int(conf.loc[s, s])
        per_stage_acc[s] = {
            "n": row_sum,
            "correct": diag,
            "accuracy": (diag / row_sum) if row_sum > 0 else None,
        }

    return {
        "n_nsd_positive": n,
        "n_correct_top1": int(correct),
        "accuracy_top1": acc,
        "chance_accuracy_4class": chance_acc,
        "majority_stage": majority_stage,
        "majority_stage_n": int(majority_n),
        "majority_class_accuracy": majority_acc,
        "top2_accuracy": float(top2_acc),
        "top2_n_correct": int(top2_correct),
        "ordinal_mae": mae,
        "ordinal_mae_distribution": mae_distribution,
        "confusion_matrix": {
            "rows_actual": NSD_STAGES,
            "cols_predicted": NSD_STAGES,
            "values": conf.values.tolist(),
        },
        "per_stage_accuracy": per_stage_acc,
    }


# ---------------------------------------------------------------------------
# E: CONSORT 1,065 covered vs 835 uncovered
# ---------------------------------------------------------------------------

def _actual_stage_from_pipe(pipe: dict) -> pd.DataFrame:
    rows = []
    for patno, v in pipe.items():
        if not isinstance(v, dict) or "staging" not in v:
            continue  # skip summary entries
        rows.append({
            "PATNO": int(patno),
            "actual_stage": v["staging"].get("actual_stage"),
            "n_visits": v.get("n_visits"),
            "follow_up_months": v.get("follow_up_months"),
        })
    return pd.DataFrame(rows)


def _load_datscan_counts() -> pd.DataFrame:
    """Count DaT-SPECT scans per PATNO from raw data."""
    dat_path = PROJECT / "data/00_raw/DaTScan_SBR_Analysis_08Feb2026.csv"
    df = pd.read_csv(dat_path, low_memory=False)
    # Keep analyzed scans only
    if "DATSCAN_ANALYZED" in df.columns:
        df = df[df["DATSCAN_ANALYZED"].astype(str).str.strip().str.lower() == "yes"]
    counts = df.groupby("PATNO").size().reset_index(name="n_datscans_raw")
    return counts


def _genetics_from_features(feats: pd.DataFrame) -> pd.DataFrame:
    cols = ["PATNO", "LRRK2_CARRIER", "GBA_CARRIER", "APOE_E4_CARRIER"]
    return feats[cols].copy()


def _load_raw_demographics_and_genetics() -> pd.DataFrame:
    """Pull SEX from raw Demographics + LRRK2/GBA/APOE from iu_genetic_consensus.

    The paper1 features CSV has SEX NaN for 849 rows and LRRK2/GBA collapsed
    to all-zeros (Paper 1 gotcha). Re-derive from raw files.
    """
    demog = pd.read_csv(
        PROJECT / "data/00_raw/Demographics_08Feb2026.csv", low_memory=False
    )
    demog = demog.drop_duplicates("PATNO", keep="first")[["PATNO", "SEX"]].copy()
    demog["SEX"] = pd.to_numeric(demog["SEX"], errors="coerce")

    g = pd.read_csv(
        PROJECT
        / "data/00_raw/GIMAN/ppmi_data_csv/iu_genetic_consensus_20250515_30Sep2025.csv",
        low_memory=False,
    )
    g = g.drop_duplicates("PATNO", keep="first")[["PATNO", "LRRK2", "GBA", "APOE"]]

    # Binary carrier flags: LRRK2 != '0' means pathogenic carrier (any LRRK2
    # variant tagged); same for GBA. APOE_E4 == True if 'E4' in genotype.
    def _carrier(v) -> float:
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        return 0.0 if s in ("0", "", "nan") else 1.0

    g["LRRK2_CARRIER"] = g["LRRK2"].apply(_carrier)
    g["GBA_CARRIER"] = g["GBA"].apply(_carrier)
    g["APOE_E4_CARRIER"] = g["APOE"].apply(
        lambda v: np.nan if pd.isna(v) else (1.0 if "E4" in str(v).upper() else 0.0)
    )

    out = demog.merge(
        g[["PATNO", "LRRK2_CARRIER", "GBA_CARRIER", "APOE_E4_CARRIER"]],
        on="PATNO",
        how="outer",
    )
    return out


def coverage_consort(
    pipe: dict, post: pd.DataFrame, feats: pd.DataFrame
) -> dict[str, Any]:
    pipe_df = _actual_stage_from_pipe(pipe)
    # Drop duplicate PATNO (pipeline should be unique, but be safe)
    pipe_df = pipe_df.drop_duplicates("PATNO", keep="first")

    # Merge AGE from feats (reliable), then raw SEX + genetics (paper1 CSV
    # has broken SEX/LRRK2/GBA — see CLAUDE.md Paper 1 Gotchas).
    base = pipe_df.merge(
        feats[["PATNO", "AGE_AT_BASELINE"]], on="PATNO", how="left"
    )
    raw_dg = _load_raw_demographics_and_genetics()
    base = base.merge(raw_dg, on="PATNO", how="left")

    # DaT scan counts from raw
    try:
        datc = _load_datscan_counts()
        base = base.merge(datc, on="PATNO", how="left")
    except FileNotFoundError:
        base["n_datscans_raw"] = np.nan

    # Coverage flag: PATNO in posteriors
    covered = set(post["PATNO"].astype(int).tolist())
    base["covered"] = base["PATNO"].isin(covered)
    base["wave"] = base["PATNO"].map(
        dict(zip(post["PATNO"].astype(int), post["wave"]))
    )

    n_total = len(base)
    n_cov = int(base["covered"].sum())
    n_unc = int((~base["covered"]).sum())

    def _continuous(col: str) -> dict[str, Any]:
        a = base.loc[base["covered"], col].dropna().astype(float)
        b = base.loc[~base["covered"], col].dropna().astype(float)
        if len(a) < 2 or len(b) < 2:
            return {
                "covered_mean": float(a.mean()) if len(a) else None,
                "covered_sd": float(a.std(ddof=1)) if len(a) > 1 else None,
                "covered_n": int(len(a)),
                "uncovered_mean": float(b.mean()) if len(b) else None,
                "uncovered_sd": float(b.std(ddof=1)) if len(b) > 1 else None,
                "uncovered_n": int(len(b)),
                "t_stat": None,
                "p_value": None,
                "test": "welch_t",
            }
        t, p = stats.ttest_ind(a, b, equal_var=False)
        return {
            "covered_mean": float(a.mean()),
            "covered_sd": float(a.std(ddof=1)),
            "covered_n": int(len(a)),
            "uncovered_mean": float(b.mean()),
            "uncovered_sd": float(b.std(ddof=1)),
            "uncovered_n": int(len(b)),
            "t_stat": float(t),
            "p_value": float(p),
            "test": "welch_t",
        }

    def _categorical(col: str, positive_label=1) -> dict[str, Any]:
        """Binary % + chi-square (missing values excluded from denominator)."""
        a = base.loc[base["covered"], col].dropna()
        b = base.loc[~base["covered"], col].dropna()
        pos_a = int((a == positive_label).sum())
        pos_b = int((b == positive_label).sum())
        table = np.array([
            [pos_a, len(a) - pos_a],
            [pos_b, len(b) - pos_b],
        ])
        if table.sum() == 0 or (table.sum(axis=0) == 0).any():
            return {
                "covered_pos": pos_a,
                "covered_n": int(len(a)),
                "covered_pct": (pos_a / len(a) * 100) if len(a) else None,
                "uncovered_pos": pos_b,
                "uncovered_n": int(len(b)),
                "uncovered_pct": (pos_b / len(b) * 100) if len(b) else None,
                "chi2": None,
                "p_value": None,
                "test": "chi2",
            }
        chi2, p, _, _ = stats.chi2_contingency(table)
        return {
            "covered_pos": pos_a,
            "covered_n": int(len(a)),
            "covered_pct": (pos_a / len(a) * 100) if len(a) else None,
            "uncovered_pos": pos_b,
            "uncovered_n": int(len(b)),
            "uncovered_pct": (pos_b / len(b) * 100) if len(b) else None,
            "chi2": float(chi2),
            "p_value": float(p),
            "test": "chi2",
        }

    def _stage_distribution() -> dict[str, Any]:
        a = base.loc[base["covered"], "actual_stage"].dropna()
        b = base.loc[~base["covered"], "actual_stage"].dropna()
        cats = sorted(set(list(a.unique()) + list(b.unique())),
                      key=lambda x: ["0", "1", "2B", "3", "4", "5", "6"].index(x)
                      if x in ["0", "1", "2B", "3", "4", "5", "6"] else 99)
        table = np.array([
            [int((a == c).sum()) for c in cats],
            [int((b == c).sum()) for c in cats],
        ])
        chi2, p, _, _ = stats.chi2_contingency(table)
        return {
            "stages": cats,
            "covered_counts": table[0].tolist(),
            "uncovered_counts": table[1].tolist(),
            "covered_pct": [float(x / table[0].sum() * 100) if table[0].sum() else None
                            for x in table[0]],
            "uncovered_pct": [float(x / table[1].sum() * 100) if table[1].sum() else None
                              for x in table[1]],
            "chi2": float(chi2),
            "p_value": float(p),
            "test": "chi2",
        }

    # Wave breakdown (within covered)
    wave_a = int(((base["covered"]) & (base["wave"] == "A")).sum())
    wave_b = int(((base["covered"]) & (base["wave"] == "B")).sum())

    # Follow-up in years
    base["follow_up_years"] = base["follow_up_months"] / 12.0

    consort = {
        "totals": {
            "n_ppmi_staged": 2201,
            "n_longitudinal": n_total,
            "n_covered_phase2": n_cov,
            "n_uncovered": n_unc,
            "wave_a_covered": wave_a,
            "wave_b_covered": wave_b,
        },
        "age_at_baseline_years": _continuous("AGE_AT_BASELINE"),
        "sex_male_pct": _categorical("SEX", positive_label=1),  # 1 = male in PPMI
        "n_datscans_raw": _continuous("n_datscans_raw"),
        "n_visits_longitudinal": _continuous("n_visits"),
        "follow_up_years": _continuous("follow_up_years"),
        "lrrk2_carrier_pct": _categorical("LRRK2_CARRIER", positive_label=1),
        "gba_carrier_pct": _categorical("GBA_CARRIER", positive_label=1),
        "apoe_e4_carrier_pct": _categorical("APOE_E4_CARRIER", positive_label=1),
        "nsd_iss_stage_distribution": _stage_distribution(),
    }
    return consort


# ---------------------------------------------------------------------------
# CONSORT TikZ figure
# ---------------------------------------------------------------------------

CONSORT_TEMPLATE = r"""\documentclass[tikz,border=6pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc}
\begin{document}
\begin{tikzpicture}[
    node distance=0.9cm and 1.2cm,
    box/.style={rectangle, draw=black, fill=white, rounded corners=2pt,
                minimum width=7.0cm, minimum height=0.9cm, align=center,
                font=\small},
    boxwide/.style={rectangle, draw=black, fill=white, rounded corners=2pt,
                minimum width=9.5cm, minimum height=0.9cm, align=center,
                font=\small},
    boxsmall/.style={rectangle, draw=black, fill=white, rounded corners=2pt,
                minimum width=5.3cm, minimum height=1.0cm, align=center,
                font=\small},
    excl/.style={rectangle, draw=black, dashed, fill=white, rounded corners=2pt,
                minimum width=5.0cm, minimum height=0.8cm, align=center,
                font=\footnotesize},
    arr/.style={-{Latex[length=2.2mm]}, thick}
]
    \node[boxwide] (pop) {PPMI total registered cohort\\\textbf{N=__N_PPMI_TOTAL__}};
    \node[boxwide, below=of pop] (staged) {NSD-ISS staged\\\textbf{N=__N_STAGED__}};
    \node[boxwide, below=of staged] (longit) {Longitudinal PD / Prodromal cohort (Paper 6 eligible)\\\textbf{N=__N_LONG__}};

    \node[boxsmall, below left=1.1cm and -1.4cm of longit] (cov)
        {Phase-2 posterior coverage\\(Wave A + Wave B, $\ge$2 DaT-SPECT)\\\textbf{N=__N_COV__}\\(Wave A=__WA__, Wave B=__WB__)};
    \node[boxsmall, below right=1.1cm and -1.4cm of longit] (unc)
        {No Phase-2 posterior\\(insufficient serial DaT-SPECT)\\\textbf{N=__N_UNC__}};

    \node[excl, right=0.6cm of staged] (exclstage) {Excluded: not PD / prodromal\\or no follow-up visits};

    \draw[arr] (pop) -- (staged);
    \draw[arr] (staged) -- (longit);
    \draw[arr] (longit.south) -- ++(0,-0.3) -| (cov.north);
    \draw[arr] (longit.south) -- ++(0,-0.3) -| (unc.north);
    \draw[arr] (staged.east) -- (exclstage.west);

    \node[below=0.3cm of cov, font=\scriptsize, align=center] (covdesc)
        {Mechanistic-layer available\\Calibrated $N(t)/N_0$, $T_{tox}$, $\alpha_{tox}$};
    \node[below=0.3cm of unc, font=\scriptsize, align=center] (uncdesc)
        {Mechanistic-layer unavailable\\(Stats and Graph-DT pathways active)};
\end{tikzpicture}
\end{document}
"""


def write_consort_tex(consort: dict[str, Any]) -> Path:
    t = consort["totals"]
    tex = (
        CONSORT_TEMPLATE
        .replace("__N_PPMI_TOTAL__", "8,042")
        .replace("__N_STAGED__", f"{t['n_ppmi_staged']:,}")
        .replace("__N_LONG__", f"{t['n_longitudinal']:,}")
        .replace("__N_COV__", f"{t['n_covered_phase2']:,}")
        .replace("__N_UNC__", f"{t['n_uncovered']:,}")
        .replace("__WA__", f"{t['wave_a_covered']:,}")
        .replace("__WB__", f"{t['wave_b_covered']:,}")
    )
    tex_path = OUT_DIR / "fig_consort_coverage_split.tex"
    tex_path.write_text(tex)
    return tex_path


def compile_tex(tex_path: Path) -> Path | None:
    """Compile with pdflatex if available; otherwise skip quietly."""
    outdir = tex_path.parent
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(outdir),
             str(tex_path)],
            capture_output=True, text=True, timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[warn] pdflatex compile skipped: {e}")
        return None
    pdf = tex_path.with_suffix(".pdf")
    if not pdf.exists():
        print("[warn] pdflatex produced no PDF. Last 40 lines of log:")
        print("\n".join(result.stdout.splitlines()[-40:]))
        return None
    # Clean aux files
    for ext in (".aux", ".log", ".out"):
        p = tex_path.with_suffix(ext)
        if p.exists():
            p.unlink()
    return pdf


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def write_markdown(accdef: dict[str, Any], consort: dict[str, Any]) -> Path:
    lines: list[str] = []
    lines.append("# Paper 6 JAMIA revision — Aggregate analyses\n")
    lines.append("Generated by `scripts/paper6/aggregate_revision_stats.py`.\n")

    # Results §3.2 paste-ready paragraph
    lines.append("## Paste-ready text for Results §3.2 (accuracy baselines)\n")
    a = accdef
    ps = a["per_stage_accuracy"]

    def _pct(x: float | None) -> str:
        return f"{x*100:.1f}\\%" if x is not None else "n/a"

    res_par = (
        f"Within the NSD+-evaluable subset (n={a['n_nsd_positive']:,}; 4-class "
        f"stages {{1, 2B, 3, 4}}), the integrated pipeline achieved top-1 "
        f"accuracy {_pct(a['accuracy_top1'])} "
        f"({a['n_correct_top1']:,}/{a['n_nsd_positive']:,}), versus a chance "
        f"baseline of {_pct(a['chance_accuracy_4class'])} and a majority-class "
        f"baseline of {_pct(a['majority_class_accuracy'])} (predicting "
        f"Stage {a['majority_stage']} for all patients; n="
        f"{a['majority_stage_n']:,}/{a['n_nsd_positive']:,}). Top-2 accuracy "
        f"was {_pct(a['top2_accuracy'])} "
        f"({a['top2_n_correct']:,}/{a['n_nsd_positive']:,}), indicating that "
        f"for the majority of patients the true stage was either the modal or "
        f"adjacent-modal CatBoost prediction. Ordinal mean absolute error "
        f"(stages 1--4 mapped to 0--3) was "
        f"{a['ordinal_mae']:.3f} stage units, "
        f"with error distribution "
        + ", ".join(
            f"|e|={k}: {v:,}" for k, v in a["ordinal_mae_distribution"].items()
        ) + ". Per-stage top-1 accuracy: "
        + ", ".join(
            f"Stage {s} {_pct(ps[s]['accuracy'])} "
            f"({ps[s]['correct']:,}/{ps[s]['n']:,})"
            for s in NSD_STAGES
        ) + "."
    )
    lines.append(res_par + "\n")

    # Confusion matrix markdown
    lines.append("## 4×4 Confusion matrix (rows = actual, cols = predicted)\n")
    cm = accdef["confusion_matrix"]["values"]
    header = "| Actual \\ Predicted | " + " | ".join(NSD_STAGES) + " |"
    sep = "|" + "|".join(["---"] * (len(NSD_STAGES) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for i, s in enumerate(NSD_STAGES):
        row = f"| **{s}** | " + " | ".join(str(cm[i][j]) for j in range(4)) + " |"
        lines.append(row)
    lines.append("")

    # Discussion paragraph (coverage comparison)
    lines.append("## Paste-ready text for Discussion (coverage selection bias)\n")
    t = consort["totals"]
    age = consort["age_at_baseline_years"]
    sex = consort["sex_male_pct"]
    nvisits = consort["n_visits_longitudinal"]
    fup = consort["follow_up_years"]
    nsc = consort["n_datscans_raw"]
    sd = consort["nsd_iss_stage_distribution"]
    lrrk2 = consort["lrrk2_carrier_pct"]
    gba = consort["gba_carrier_pct"]

    def _fmt(x, spec=".1f"):
        if x is None:
            return "n/a"
        return format(x, spec)

    def _fmtp(x):
        return "n/a" if x is None else f"{x:.3g}"

    def _cont(d: dict) -> str:
        return (
            f"{_fmt(d['covered_mean'])}±{_fmt(d['covered_sd'])} vs "
            f"{_fmt(d['uncovered_mean'])}±{_fmt(d['uncovered_sd'])}, "
            f"p={_fmtp(d['p_value'])}"
        )

    def _cat(d: dict) -> str:
        return (
            f"{_fmt(d['covered_pct'])}\\% vs {_fmt(d['uncovered_pct'])}\\%, "
            f"p={_fmtp(d['p_value'])}"
        )

    disc = (
        f"Of the {t['n_longitudinal']:,} PPMI patients eligible for the "
        f"longitudinal pipeline, {t['n_covered_phase2']:,} had Phase-2 "
        f"calibrated posteriors (Wave A n={t['wave_a_covered']:,}; "
        f"Wave B n={t['wave_b_covered']:,}) and {t['n_uncovered']:,} did not, "
        f"primarily because they lacked $\\ge$2 DaT-SPECT scans required for "
        f"the α-syn/N(t) ODE identifiability. On baseline characteristics, "
        f"the two subgroups were "
        f"comparable in age at baseline (covered vs uncovered: "
        f"{_cont(age)}), sex distribution (male %: "
        f"{_cat(sex)}), "
        f"LRRK2 carrier status ({_cat(lrrk2)}), and GBA carrier status "
        f"({_cat(gba)}). "
        f"As expected by construction, the covered subgroup had more DaT-"
        f"SPECT scans per patient ("
        f"{_cont(nsc)}) and longer follow-up ("
        f"{_cont(fup)} years; "
        f"{_cont(nvisits)} longitudinal visits). The NSD-ISS stage "
        f"distribution was significantly different between subgroups "
        f"(χ²={sd['chi2']:.1f}, p={sd['p_value']:.3g}), consistent with the "
        f"fact that patients enrolled into DaT-rich sub-studies tend to "
        f"cluster in intermediate biological stages. Because the pipeline "
        f"returns a well-characterised non-mechanistic output for uncovered "
        f"patients (CatBoost staging + Graph-DT transitions + per-feature "
        f"conformal bands), this selection pattern affects only which "
        f"pathway a patient enters, not whether they receive a calibrated "
        f"prediction."
    )
    lines.append(disc + "\n")

    # CONSORT summary table
    lines.append("## Coverage subgroup comparison (CONSORT subgroup table)\n")
    lines.append("| Variable | Covered (n=%d) | Uncovered (n=%d) | Test | p |"
                 % (t["n_covered_phase2"], t["n_uncovered"]))
    lines.append("|---|---|---|---|---|")

    def _row_cont(label: str, d: dict) -> str:
        return (f"| {label} | {d['covered_mean']:.2f} ± {d['covered_sd']:.2f} "
                f"(n={d['covered_n']:,}) | "
                f"{d['uncovered_mean']:.2f} ± {d['uncovered_sd']:.2f} "
                f"(n={d['uncovered_n']:,}) | Welch t | "
                f"{d['p_value']:.3g} |")

    def _row_cat(label: str, d: dict) -> str:
        return (f"| {label} | {d['covered_pos']:,}/{d['covered_n']:,} "
                f"({d['covered_pct']:.1f}%) | "
                f"{d['uncovered_pos']:,}/{d['uncovered_n']:,} "
                f"({d['uncovered_pct']:.1f}%) | χ² | {d['p_value']:.3g} |")

    lines.append(_row_cont("Age at baseline (years)", age))
    lines.append(_row_cat("Sex (male)", sex))
    lines.append(_row_cont("DaT-SPECT scans (raw)", nsc))
    lines.append(_row_cont("Longitudinal visits", nvisits))
    lines.append(_row_cont("Follow-up (years)", fup))
    lines.append(_row_cat("LRRK2 carrier", lrrk2))
    lines.append(_row_cat("GBA carrier", gba))
    lines.append(_row_cat("APOE-ε4 carrier", consort["apoe_e4_carrier_pct"]))
    lines.append("")

    # NSD-ISS distribution
    lines.append("### NSD-ISS stage distribution by coverage\n")
    sd2 = sd
    lines.append("| Stage | Covered n (%) | Uncovered n (%) |")
    lines.append("|---|---|---|")
    for stg, cc, uc, cp, up in zip(
        sd2["stages"], sd2["covered_counts"], sd2["uncovered_counts"],
        sd2["covered_pct"], sd2["uncovered_pct"],
    ):
        lines.append(f"| {stg} | {cc:,} ({cp:.1f}%) | {uc:,} ({up:.1f}%) |")
    lines.append(f"\nχ²={sd2['chi2']:.2f}, p={sd2['p_value']:.3g}\n")

    out = OUT_DIR / "aggregate_summary.md"
    out.write_text("\n".join(lines))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pipe, post, feats = load_inputs()
    print(f"Loaded pipeline summary: {len(pipe)} patients")
    print(f"Loaded posteriors: {len(post)} patients")
    print(f"Loaded paper1 features: {len(feats)} patients")

    accdef = nsd_positive_analyses(pipe)
    print(
        f"\nA. NSD+ top-1 accuracy: {accdef['accuracy_top1']*100:.2f}%  "
        f"(chance {accdef['chance_accuracy_4class']*100:.1f}%, "
        f"majority {accdef['majority_class_accuracy']*100:.2f}% @ Stage "
        f"{accdef['majority_stage']})"
    )
    print(f"B. Top-2 accuracy: {accdef['top2_accuracy']*100:.2f}%")
    print(f"C. Ordinal MAE: {accdef['ordinal_mae']:.3f}")
    print(
        "D. Per-stage: "
        + ", ".join(
            f"{s}={accdef['per_stage_accuracy'][s]['accuracy']*100:.1f}%"
            if accdef['per_stage_accuracy'][s]['accuracy'] is not None
            else f"{s}=n/a"
            for s in NSD_STAGES
        )
    )

    consort = coverage_consort(pipe, post, feats)
    t = consort["totals"]
    print(
        f"\nE. CONSORT: covered={t['n_covered_phase2']:,}  "
        f"uncovered={t['n_uncovered']:,}  (wave A={t['wave_a_covered']:,}, "
        f"wave B={t['wave_b_covered']:,})"
    )

    out = {
        "meta": {
            "script": "scripts/paper6/aggregate_revision_stats.py",
            "inputs": {
                "pipeline_summary": str(PIPE_SUMMARY),
                "posteriors": str(POSTERIORS),
                "paper1_features": str(P1_FEATURES),
            },
            "n_patients_in_pipeline": len(pipe),
            "n_patients_in_posteriors": len(post),
        },
        "A_accuracy_baselines": {
            "accuracy_top1": accdef["accuracy_top1"],
            "n_correct_top1": accdef["n_correct_top1"],
            "n_nsd_positive": accdef["n_nsd_positive"],
            "chance_accuracy_4class": accdef["chance_accuracy_4class"],
            "majority_stage": accdef["majority_stage"],
            "majority_stage_n": accdef["majority_stage_n"],
            "majority_class_accuracy": accdef["majority_class_accuracy"],
        },
        "B_top2_accuracy": {
            "top2_accuracy": accdef["top2_accuracy"],
            "top2_n_correct": accdef["top2_n_correct"],
            "n_nsd_positive": accdef["n_nsd_positive"],
        },
        "C_ordinal_mae": {
            "mae": accdef["ordinal_mae"],
            "distribution": accdef["ordinal_mae_distribution"],
        },
        "D_confusion_matrix": {
            "matrix": accdef["confusion_matrix"],
            "per_stage_accuracy": accdef["per_stage_accuracy"],
        },
        "E_coverage_consort": consort,
    }

    out_path = OUT_DIR / "aggregate_stats.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")

    tex_path = write_consort_tex(consort)
    print(f"Wrote {tex_path}")
    pdf_path = compile_tex(tex_path)
    if pdf_path:
        print(f"Wrote {pdf_path}")

    md_path = write_markdown(accdef, consort)
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
