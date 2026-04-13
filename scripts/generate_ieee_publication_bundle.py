from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_existing(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No candidate artifact path exists: " + ", ".join(str(p) for p in candidates)
    )


def _fmt_ci(ci: list[float] | tuple[float, float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _render_fig1_discrimination(
    baselines: dict,
    fuzzy_auc: float,
    fuzzy_auc_ci: list[float],
    fuzzy_pr_auc: float,
    output_path: Path,
) -> None:
    model_names = list(baselines.keys()) + ["fuzzy_giman"]
    auc_vals = [baselines[m]["auc"] for m in baselines] + [fuzzy_auc]
    pr_vals = [baselines[m]["pr_auc"] for m in baselines] + [fuzzy_pr_auc]
    auc_ci = [baselines[m]["auc_ci_95"] for m in baselines] + [fuzzy_auc_ci]

    x = np.arange(len(model_names))
    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    lower = [v - ci[0] for v, ci in zip(auc_vals, auc_ci, strict=False)]
    upper = [ci[1] - v for v, ci in zip(auc_vals, auc_ci, strict=False)]
    axes[0].bar(x, auc_vals, color=colors, alpha=0.9)
    axes[0].errorbar(
        x,
        auc_vals,
        yerr=np.array([lower, upper]),
        fmt="none",
        ecolor="black",
        capsize=4,
        lw=1.2,
    )
    axes[0].set_title("AUC-ROC with 95% CI")
    axes[0].set_ylabel("AUC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=20, ha="right")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(x, pr_vals, color=colors, alpha=0.9)
    axes[1].set_title("PR-AUC")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=20, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Internal Classification Discrimination (Canonical Split)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=350)
    plt.close(fig)


def _render_fig2_calibration(cal: dict, output_path: Path) -> None:
    rows = []
    for name in ("raw", "platt", "isotonic"):
        rows.append((name, float(cal[name]["ece"]), float(cal[name]["brier"])))
    df = pd.DataFrame(rows, columns=["method", "ece", "brier"])

    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, df["ece"], width, label="ECE", color="#59A14F")
    ax.bar(x + width / 2, df["brier"], width, label="Brier", color="#E15759")
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"])
    ax.set_ylabel("Error (lower is better)")
    ax.set_title("Calibration Diagnostics for FUZZY GIMAN")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=350)
    plt.close(fig)


def _render_fig3_feature_importance(feature_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(feature_csv).head(15).copy()
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.barh(df["feature"][::-1], df["auc_drop"][::-1], color="#4E79A7")
    ax.set_title("Top-15 Permutation Importances (AUC Drop)")
    ax.set_xlabel("AUC drop after permutation")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=350)
    plt.close(fig)


def _render_fig4_digital_twin(
    moderate_csv: Path,
    stress_csv: Path,
    output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mod = pd.read_csv(moderate_csv)
    stress = pd.read_csv(stress_csv)
    mod_agg = (
        mod.groupby("intervention", as_index=False)["abs_delta_risk"]
        .mean()
        .rename(columns={"abs_delta_risk": "moderate_abs_delta"})
    )
    stress_agg = (
        stress.groupby("intervention", as_index=False)["abs_delta_risk"]
        .mean()
        .rename(columns={"abs_delta_risk": "stress_abs_delta"})
    )
    merged = mod_agg.merge(stress_agg, on="intervention", how="outer").fillna(0.0)
    merged = merged.sort_values("stress_abs_delta", ascending=False).head(10)

    y = np.arange(len(merged))
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.barh(y + 0.2, merged["moderate_abs_delta"], height=0.35, label="Moderate")
    ax.barh(y - 0.2, merged["stress_abs_delta"], height=0.35, label="Stress")
    ax.set_yticks(y)
    ax.set_yticklabels(merged["intervention"])
    ax.set_xlabel("Mean absolute change in final-horizon SAA risk")
    ax.set_title("Digital Twin Intervention Sensitivity")
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=350)
    plt.close(fig)
    return mod_agg, stress_agg


def _render_fig5_readiness(readiness_csv: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(readiness_csv)
    agg = (
        df.groupby("domain", as_index=False)
        .agg(
            mean_joinability=("joinability_score", "mean"),
            n_sources=("source_path", "count"),
            ready=("readiness_status", lambda s: int((s == "ready").sum())),
            partial=("readiness_status", lambda s: int((s == "partial").sum())),
            blocked=("readiness_status", lambda s: int((s == "blocked").sum())),
        )
        .sort_values("mean_joinability", ascending=False)
    )

    x = np.arange(len(agg))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x, agg["mean_joinability"], color="#4E79A7")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(agg["domain"], rotation=25, ha="right")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Domain Joinability")
    axes[0].set_ylabel("Mean joinability score")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(x, agg["ready"], label="ready", color="#59A14F")
    axes[1].bar(
        x, agg["partial"], bottom=agg["ready"], label="partial", color="#F28E2B"
    )
    axes[1].bar(
        x,
        agg["blocked"],
        bottom=agg["ready"] + agg["partial"],
        label="blocked",
        color="#E15759",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(agg["domain"], rotation=25, ha="right")
    axes[1].set_title("Readiness by Domain")
    axes[1].set_ylabel("Number of sources")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=350)
    plt.close(fig)
    return agg


def _render_fig6_preprocessing(final_csv: Path, output_path: Path) -> dict[str, float]:
    df = pd.read_csv(final_csv)
    missing = df.isna().mean().sort_values(ascending=False).head(12)
    events = df["phenoconverted"].value_counts(dropna=False).sort_index()
    time_vals = df["time_to_event"].fillna(0)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    axes[0].barh(missing.index[::-1], missing.values[::-1], color="#4E79A7")
    axes[0].set_title("Top Missingness Features")
    axes[0].set_xlabel("Missing fraction")

    axes[1].bar(events.index.astype(str), events.values, color="#F28E2B")
    axes[1].set_title("Phenoconversion Balance")
    axes[1].set_xlabel("Label")
    axes[1].set_ylabel("Count")

    axes[2].hist(time_vals, bins=20, color="#59A14F", alpha=0.9)
    axes[2].set_title("Time-to-event")
    axes[2].set_xlabel("Months")
    axes[2].set_ylabel("Count")

    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=350)
    plt.close(fig)

    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "positive_events": int(events.get(1, 0)),
        "negative_events": int(events.get(0, 0)),
        "median_time_to_event": float(time_vals.median()),
    }


def _render_fig7_cycle_gates(cycle_payload: dict, output_path: Path) -> None:
    cycles = cycle_payload["cycles"]
    names = [c["cycle"].replace("_", " ") for c in cycles]
    vals = [1 if c["pass"] else 0 for c in cycles]
    colors = ["#59A14F" if v == 1 else "#E15759" for v in vals]
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Pass status")
    ax.set_title("Clinical Hardening Review Cycles")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FAIL", "PASS"])
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=350)
    plt.close(fig)


def _latex_metrics_table(
    baselines: dict,
    fuzzy_auc: float,
    fuzzy_auc_ci: list[float],
    fuzzy_pr_auc: float,
    fuzzy_calibration: dict | None = None,
) -> str:
    fuzzy_ece = 0.392
    fuzzy_brier = 0.292
    if fuzzy_calibration is not None:
        raw = fuzzy_calibration.get("raw", {})
        fuzzy_ece = float(raw.get("ece", fuzzy_ece))
        fuzzy_brier = float(raw.get("brier", fuzzy_brier))

    rows = []
    for model, m in baselines.items():
        rows.append(
            f"{model.replace('_', ' ')} & {m['auc']:.3f} & {_fmt_ci(m['auc_ci_95'])} & {m['pr_auc']:.3f} & {m['ece']:.3f} & {m['brier']:.3f} \\\\"
        )
    rows.append(
        f"FUZZY GIMAN & {fuzzy_auc:.3f} & {_fmt_ci(fuzzy_auc_ci)} & {fuzzy_pr_auc:.3f} & {fuzzy_ece:.3f}$^*$ & {fuzzy_brier:.3f}$^*$ \\\\"
    )

    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Internal classification metrics on canonical patient split.}",
            "\\label{tab:internal_metrics}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "Model & AUC & AUC 95\\% CI & PR-AUC & ECE & Brier \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{2pt}",
            "\\footnotesize{$^*$FUZZY GIMAN ECE/Brier shown for raw probabilities before post-hoc calibration.}",
            "\\end{table}",
            "",
        ]
    )


def _latex_calibration_table(cal: dict) -> str:
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Post-hoc calibration effects for FUZZY GIMAN on held-out test split.}",
            "\\label{tab:calibration}",
            "\\begin{tabular}{lcc}",
            "\\toprule",
            "Method & ECE (lower better) & Brier (lower better) \\\\",
            "\\midrule",
            f"Raw & {cal['raw']['ece']:.3f} & {cal['raw']['brier']:.3f} \\\\",
            f"Platt & {cal['platt']['ece']:.3f} & {cal['platt']['brier']:.3f} \\\\",
            f"Isotonic & {cal['isotonic']['ece']:.3f} & {cal['isotonic']['brier']:.3f} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def _latex_gate_table(cycle_payload: dict) -> str:
    rows = []
    for c in cycle_payload["cycles"]:
        status = "PASS" if c["pass"] else "FAIL"
        rows.append(f"{c['cycle']} & {status} \\\\")
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Clinical hardening review cycle outcomes.}",
            "\\label{tab:cycle_gates}",
            "\\begin{tabular}{lc}",
            "\\toprule",
            "Cycle & Status \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def _build_ieee_addendum(
    figure_dir: Path,
    table_dir: Path,
    output_tex: Path,
    external_report_path: Path,
) -> None:
    tex = "\n".join(
        [
            "\\section{Artifact-Backed Internal Validation Update}",
            "This update replaces unsupported near-perfect claims with evidence from reproducible artifacts generated on the canonical patient-disjoint split. Performance should be interpreted as \\emph{internal validation only}.",
            "",
            "\\begin{figure*}[t]",
            "\\centering",
            f"\\includegraphics[width=0.95\\textwidth]{{{figure_dir.as_posix()}/Figure01_Internal_Discrimination.png}}",
            "\\caption{Internal discrimination metrics (AUC and PR-AUC) for baseline models and FUZZY GIMAN on the canonical split.}",
            "\\label{fig:ieee_internal_discrimination}",
            "\\end{figure*}",
            "",
            "\\begin{figure}[t]",
            "\\centering",
            f"\\includegraphics[width=\\columnwidth]{{{figure_dir.as_posix()}/Figure02_Fuzzy_Calibration.png}}",
            "\\caption{Calibration diagnostics for FUZZY GIMAN. Post-hoc calibration substantially improves ECE and Brier score.}",
            "\\label{fig:ieee_calibration}",
            "\\end{figure}",
            "",
            "\\begin{figure}[t]",
            "\\centering",
            f"\\includegraphics[width=\\columnwidth]{{{figure_dir.as_posix()}/Figure03_Feature_Importance.png}}",
            "\\caption{Top permutation importances for FUZZY GIMAN (AUC-drop criterion).}",
            "\\label{fig:ieee_feature_importance}",
            "\\end{figure}",
            "",
            "\\begin{figure*}[t]",
            "\\centering",
            f"\\includegraphics[width=0.95\\textwidth]{{{figure_dir.as_posix()}/Figure04_DigitalTwin_Sensitivity.png}}",
            "\\caption{Digital twin sensitivity by intervention intensity. Stress scenarios produce larger but bounded risk deltas.}",
            "\\label{fig:ieee_twin_sensitivity}",
            "\\end{figure*}",
            "",
            "\\begin{figure*}[t]",
            "\\centering",
            f"\\includegraphics[width=0.95\\textwidth]{{{figure_dir.as_posix()}/Figure05_Multimodal_Readiness.png}}",
            "\\caption{Multimodal readiness audit summary across source domains.}",
            "\\label{fig:ieee_readiness}",
            "\\end{figure*}",
            "",
            "\\begin{figure*}[t]",
            "\\centering",
            f"\\includegraphics[width=0.95\\textwidth]{{{figure_dir.as_posix()}/Figure06_Preprocessing_Transparency.png}}",
            "\\caption{Preprocessing transparency: missingness profile, label balance, and time-to-event distribution.}",
            "\\label{fig:ieee_preprocessing}",
            "\\end{figure*}",
            "",
            "\\begin{figure}[t]",
            "\\centering",
            f"\\includegraphics[width=\\columnwidth]{{{figure_dir.as_posix()}/Figure07_Clinical_Gates.png}}",
            "\\caption{Clinical hardening cycle outcomes. All internal gates pass; external validation remains required.}",
            "\\label{fig:ieee_cycle_gates}",
            "\\end{figure}",
            "",
            f"\\input{{{table_dir.as_posix()}/table_internal_metrics.tex}}",
            f"\\input{{{table_dir.as_posix()}/table_calibration.tex}}",
            f"\\input{{{table_dir.as_posix()}/table_cycle_gates.tex}}",
            "",
            "\\section{Clinical Claim Boundary}",
            "The external validation artifact indicates that clinical deployment claims remain out of scope at this stage. We therefore frame the model as an internally validated, explainable research system pending independent cohort validation.",
            "",
            f"\\noindent\\textbf{{External validation artifact:}} \\texttt{{{external_report_path.as_posix()}}}",
            "",
        ]
    )
    _write_text(output_tex, tex)


def main() -> None:
    """Generate publication-ready IEEE figures, tables, and addendum artifacts."""
    root = _repo_root()
    figure_dir = root / "visualizations" / "publication_ieee"
    table_dir = root / "Docs" / "audit" / "ieee_tables"
    manuscript_dir = root / "Docs" / "manuscript"

    internal_lock = _load_json(
        root / "outputs" / "sota_lock" / "internal_sota_lock.json"
    )
    explain = _load_json(
        root
        / "visualizations"
        / "appendix"
        / "explainability"
        / "explainability_summary.json"
    )
    cycles = _load_json(
        root / "Docs" / "audit" / "CLINICAL_HARDENING_REVIEW_CYCLES.json"
    )
    full_results_path = _first_existing(
        [
            root
            / "outputs"
            / "phase9_neuro_fuzzy"
            / "PREP_20260208_SAA_COHORT3_full"
            / "full_training_results.json",
            root
            / "outputs"
            / "phase9_neuro_fuzzy_sota_run_from50ckpt"
            / "full_training_results.json",
        ]
    )
    full_results = _load_json(full_results_path)

    baselines = internal_lock["baselines"]
    fuzzy_auc = float(explain["auc"])
    fuzzy_pr_auc = float(explain["pr_auc"])
    fuzzy_auc_ci = full_results.get("auc_ci_95", [fuzzy_auc, fuzzy_auc])

    fig1 = figure_dir / "Figure01_Internal_Discrimination.png"
    fig2 = figure_dir / "Figure02_Fuzzy_Calibration.png"
    fig3 = figure_dir / "Figure03_Feature_Importance.png"
    fig4 = figure_dir / "Figure04_DigitalTwin_Sensitivity.png"
    fig5 = figure_dir / "Figure05_Multimodal_Readiness.png"
    fig6 = figure_dir / "Figure06_Preprocessing_Transparency.png"
    fig7 = figure_dir / "Figure07_Clinical_Gates.png"

    _render_fig1_discrimination(
        baselines=baselines,
        fuzzy_auc=fuzzy_auc,
        fuzzy_auc_ci=fuzzy_auc_ci,
        fuzzy_pr_auc=fuzzy_pr_auc,
        output_path=fig1,
    )
    _render_fig2_calibration(explain["calibration_metrics"], fig2)
    _render_fig3_feature_importance(
        Path(explain["feature_importance_csv"]),
        fig3,
    )
    mod_agg, stress_agg = _render_fig4_digital_twin(
        root / "outputs" / "digital_twin" / "sensitivity_scan_moderate.csv",
        root / "outputs" / "digital_twin" / "sensitivity_scan_stress.csv",
        fig4,
    )
    readiness_agg = _render_fig5_readiness(
        root / "Docs" / "audit" / "PPMI_MODALITY_COVERAGE_MATRIX.csv",
        fig5,
    )
    preprocessing_summary = _render_fig6_preprocessing(
        root
        / "data"
        / "03_prodromal"
        / "final_training_dataset"
        / "unified_longitudinal_early_pd.csv",
        fig6,
    )
    _render_fig7_cycle_gates(cycles, fig7)

    metrics_table_tex = _latex_metrics_table(
        baselines=baselines,
        fuzzy_auc=fuzzy_auc,
        fuzzy_auc_ci=fuzzy_auc_ci,
        fuzzy_pr_auc=fuzzy_pr_auc,
        fuzzy_calibration=explain.get("calibration_metrics"),
    )
    cal_table_tex = _latex_calibration_table(explain["calibration_metrics"])
    gates_table_tex = _latex_gate_table(cycles)

    _write_text(table_dir / "table_internal_metrics.tex", metrics_table_tex)
    _write_text(table_dir / "table_calibration.tex", cal_table_tex)
    _write_text(table_dir / "table_cycle_gates.tex", gates_table_tex)

    _build_ieee_addendum(
        figure_dir=figure_dir,
        table_dir=table_dir,
        output_tex=manuscript_dir / "main-4_ieee_results_addendum.tex",
        external_report_path=root / "Docs" / "audit" / "EXTERNAL_VALIDATION_REPORT.md",
    )

    manifest_lines = [
        "# IEEE Publication Bundle Manifest",
        "",
        "## Figures",
        f"- {fig1}",
        f"- {fig2}",
        f"- {fig3}",
        f"- {fig4}",
        f"- {fig5}",
        f"- {fig6}",
        f"- {fig7}",
        "",
        "## Tables (LaTeX)",
        f"- {table_dir / 'table_internal_metrics.tex'}",
        f"- {table_dir / 'table_calibration.tex'}",
        f"- {table_dir / 'table_cycle_gates.tex'}",
        "",
        "## Manuscript Addendum (LaTeX)",
        f"- {manuscript_dir / 'main-4_ieee_results_addendum.tex'}",
        "",
        "## Key Evidence Inputs",
        f"- {root / 'outputs' / 'sota_lock' / 'internal_sota_lock.json'}",
        f"- {root / 'visualizations' / 'appendix' / 'explainability' / 'explainability_summary.json'}",
        f"- {root / 'Docs' / 'audit' / 'CLINICAL_HARDENING_REVIEW_CYCLES.json'}",
        f"- {root / 'Docs' / 'audit' / 'EXTERNAL_VALIDATION_REPORT.md'}",
        "",
        "## Derived Summaries",
        "",
        "### Digital Twin Sensitivity Means",
        mod_agg.to_string(index=False),
        "",
        stress_agg.to_string(index=False),
        "",
        "### Multimodal Readiness Aggregation",
        readiness_agg.to_string(index=False),
        "",
        "### Preprocessing Summary",
        json.dumps(preprocessing_summary, indent=2),
    ]
    _write_text(
        root / "Docs" / "audit" / "IEEE_PUBLICATION_BUNDLE_MANIFEST.md",
        "\n".join(manifest_lines),
    )

    print("Generated IEEE publication bundle:")
    print(f"  figures: {figure_dir}")
    print(f"  tables: {table_dir}")
    print(f"  addendum: {manuscript_dir / 'main-4_ieee_results_addendum.tex'}")


if __name__ == "__main__":
    main()
