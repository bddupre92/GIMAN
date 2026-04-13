from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))


def _load_nf_model(in_features: int, checkpoint_path: Path):
    phase8_dir = (
        root / "archive" / "development" / "phase8" / "subphase8_2_dynamic_endpoints"
    )
    if str(phase8_dir) not in sys.path:
        sys.path.append(str(phase8_dir))
    if str(root) not in sys.path:
        sys.path.append(str(root))

    from train_final_giman_survival import GIMANSurvivalGAT

    from archive.development.phase9.neuro_fuzzy import NeuroFuzzyGIMAN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gat = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128)
    model = NeuroFuzzyGIMAN(gat, num_classes=2, num_rules=32).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model, device


def _predict_prob(model, data) -> np.ndarray:
    with torch.no_grad():
        logits, _ = model(data)
        return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()


def _cycle1_proxy_dependency(checkpoint_path: Path) -> dict[str, Any]:
    from giman_pipeline.sota.metrics import safe_auc

    explain_summary = json.loads(
        (
            root
            / "visualizations"
            / "appendix"
            / "explainability"
            / "explainability_summary.json"
        ).read_text(encoding="utf-8")
    )
    pi_path = Path(explain_summary["feature_importance_csv"])
    pi = pd.read_csv(pi_path)
    top = pi.iloc[0]
    top2 = pi.iloc[1] if len(pi) > 1 else top
    top_drop = float(top["auc_drop"])
    second_drop = float(top2["auc_drop"])
    top10_sum = float(pi.head(10)["auc_drop"].clip(lower=0).sum())
    dominance_ratio = float(top_drop / top10_sum) if top10_sum > 0 else 0.0

    metadata = json.loads(
        (
            root
            / "data"
            / "03_prodromal"
            / "final_pyg_data_sota_run"
            / "pyg_data_metadata.json"
        ).read_text(encoding="utf-8")
    )
    feature_names = metadata.get("feature_names", [])
    top_feature = str(top["feature"])

    test_data = torch.load(
        root / "data" / "03_prodromal" / "final_pyg_data_sota_run" / "test_data.pt",
        weights_only=False,
    )
    train_data = torch.load(
        root / "data" / "03_prodromal" / "final_pyg_data_sota_run" / "train_data.pt",
        weights_only=False,
    )
    y_test = test_data.saa_label.detach().cpu().numpy().astype(int)
    model, device = _load_nf_model(
        in_features=int(test_data.x.shape[1]),
        checkpoint_path=checkpoint_path,
    )
    test_data = test_data.to(device)
    baseline_probs = _predict_prob(model, test_data)
    baseline_auc = safe_auc(y_test, baseline_probs)

    ablation_auc = baseline_auc
    auc_drop_ablation = 0.0
    if top_feature in feature_names:
        idx = feature_names.index(top_feature)
        train_x = train_data.x.detach().cpu().numpy()
        median_val = float(np.median(train_x[:, idx]))
        x_ablate = test_data.x.detach().cpu().numpy().copy()
        x_ablate[:, idx] = median_val
        temp = test_data.clone()
        temp.x = torch.tensor(x_ablate, dtype=torch.float32, device=device)
        ablation_probs = _predict_prob(model, temp)
        ablation_auc = safe_auc(y_test, ablation_probs)
        auc_drop_ablation = float(baseline_auc - ablation_auc)

    passed = bool(dominance_ratio < 0.50 and auc_drop_ablation < 0.20)
    return {
        "cycle": "C1_proxy_dependency_audit",
        "pass": passed,
        "top_feature": top_feature,
        "top_auc_drop": top_drop,
        "second_auc_drop": second_drop,
        "dominance_ratio_top_over_top10": dominance_ratio,
        "ablation_auc": ablation_auc,
        "baseline_auc": baseline_auc,
        "ablation_auc_drop": auc_drop_ablation,
        "notes": (
            "Fail indicates probable proxy/single-feature dependency. "
            "Gate thresholds: dominance_ratio<0.50 and ablation_auc_drop<0.20."
        ),
    }


def _cycle2_imbalance_and_objective() -> dict[str, Any]:
    train_data = torch.load(
        root / "data" / "03_prodromal" / "final_pyg_data_sota_run" / "train_data.pt",
        weights_only=False,
    )
    y = train_data.saa_label.detach().cpu().numpy().astype(int)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    prevalence = float(np.mean(y))
    imbalance_ratio = float((neg / pos) if pos > 0 else np.inf)

    text_full = (
        root / "archive" / "development" / "phase9" / "phase9_full_training.py"
    ).read_text(encoding="utf-8", errors="ignore")
    text_multi = (
        root / "archive" / "development" / "phase9" / "phase9_multitask_learning.py"
    ).read_text(encoding="utf-8", errors="ignore")
    supports_focal = ("classification-loss" in text_full) and (
        "classification-loss" in text_multi
    )
    supports_class_weights = ("CrossEntropyLoss(weight=" in text_full) and (
        "CrossEntropyLoss(weight=" in text_multi
    )

    passed = bool(
        pos > 0
        and neg > 0
        and supports_focal
        and supports_class_weights
        and prevalence <= 0.30
    )
    return {
        "cycle": "C2_imbalance_and_objective",
        "pass": passed,
        "train_positive": pos,
        "train_negative": neg,
        "prevalence_positive": prevalence,
        "imbalance_ratio_neg_over_pos": imbalance_ratio,
        "supports_focal_loss_flags": supports_focal,
        "supports_class_weights": supports_class_weights,
        "notes": "Fail indicates imbalance handling pipeline is incomplete for clinical-risk classification.",
    }


def _cycle3_calibration() -> dict[str, Any]:
    summary_path = (
        root
        / "visualizations"
        / "appendix"
        / "explainability"
        / "explainability_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw = summary["calibration_metrics"]["raw"]
    platt = summary["calibration_metrics"]["platt"]
    isotonic = summary["calibration_metrics"]["isotonic"]

    best_ece = min(raw["ece"], platt["ece"], isotonic["ece"])
    best_brier = min(raw["brier"], platt["brier"], isotonic["brier"])
    improved = (best_ece < raw["ece"]) and (best_brier < raw["brier"])
    passed = bool(improved and best_ece <= 0.10)
    return {
        "cycle": "C3_calibration_review",
        "pass": passed,
        "raw": raw,
        "platt": platt,
        "isotonic": isotonic,
        "best_ece": best_ece,
        "best_brier": best_brier,
        "notes": "Pass requires calibration method to improve both ECE and Brier against raw scores.",
    }


def _cycle4_digital_twin_sensitivity() -> dict[str, Any]:
    from giman_pipeline.digital_twin.counterfactual import run_twin_sensitivity_scan
    from giman_pipeline.digital_twin.state import CounterfactualSpec

    data_path = (
        root / "data" / "03_prodromal" / "final_pyg_data_sota_run" / "test_data.pt"
    )
    metadata_path = (
        root
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_run"
        / "pyg_data_metadata.json"
    )
    specs_moderate = [
        CounterfactualSpec(feature_name="UPDRS_I", delta=-0.5),
        CounterfactualSpec(feature_name="SCOPA_AUT_SCORE", delta=-0.5),
        CounterfactualSpec(feature_name="TREMOR_SCORE", delta=-0.5),
        CounterfactualSpec(feature_name="UPSIT_SCORE", delta=0.5),
        CounterfactualSpec(feature_name="ALPHA_SYNUCLEIN", delta=-0.5),
        CounterfactualSpec(feature_name="LRRK2", delta=-0.5),
    ]
    specs_stress = [
        CounterfactualSpec(feature_name="UPDRS_I", delta=-3.0),
        CounterfactualSpec(feature_name="SCOPA_AUT_SCORE", delta=-3.0),
        CounterfactualSpec(feature_name="TREMOR_SCORE", delta=-3.0),
        CounterfactualSpec(feature_name="UPSIT_SCORE", delta=3.0),
        CounterfactualSpec(feature_name="ALPHA_SYNUCLEIN", delta=-3.0),
        CounterfactualSpec(feature_name="LRRK2", delta=-3.0),
    ]

    out_csv_mod = root / "outputs" / "digital_twin" / "sensitivity_scan_moderate.csv"
    out_fig_mod = (
        root
        / "visualizations"
        / "appendix"
        / "digital_twin"
        / "sensitivity_scan_moderate.png"
    )
    out_csv_stress = root / "outputs" / "digital_twin" / "sensitivity_scan_stress.csv"
    out_fig_stress = (
        root
        / "visualizations"
        / "appendix"
        / "digital_twin"
        / "sensitivity_scan_stress.png"
    )

    df_mod = run_twin_sensitivity_scan(
        data_path=data_path,
        metadata_path=metadata_path,
        output_csv=out_csv_mod,
        output_figure=out_fig_mod,
        specs=specs_moderate,
        patient_indices=list(range(25)),
        horizons=[0, 6, 12, 18, 24],
        temperature=2.5,
    )
    df_stress = run_twin_sensitivity_scan(
        data_path=data_path,
        metadata_path=metadata_path,
        output_csv=out_csv_stress,
        output_figure=out_fig_stress,
        specs=specs_stress,
        patient_indices=list(range(25)),
        horizons=[0, 6, 12, 18, 24],
        temperature=2.5,
    )
    mean_abs_mod = float(df_mod["abs_delta_risk"].mean()) if len(df_mod) else 0.0
    max_abs_mod = float(df_mod["abs_delta_risk"].max()) if len(df_mod) else 0.0
    mean_abs_stress = (
        float(df_stress["abs_delta_risk"].mean()) if len(df_stress) else 0.0
    )
    max_abs_stress = float(df_stress["abs_delta_risk"].max()) if len(df_stress) else 0.0
    passed = bool(
        max_abs_mod >= 0.005 and mean_abs_stress >= 0.002 and max_abs_stress >= 0.02
    )
    return {
        "cycle": "C4_digital_twin_sensitivity",
        "pass": passed,
        "n_rows_moderate": int(len(df_mod)),
        "n_rows_stress": int(len(df_stress)),
        "mean_abs_delta_risk_moderate": mean_abs_mod,
        "max_abs_delta_risk_moderate": max_abs_mod,
        "mean_abs_delta_risk_stress": mean_abs_stress,
        "max_abs_delta_risk_stress": max_abs_stress,
        "sensitivity_csv_moderate": str(out_csv_mod),
        "sensitivity_fig_moderate": str(out_fig_mod),
        "sensitivity_csv_stress": str(out_csv_stress),
        "sensitivity_fig_stress": str(out_fig_stress),
        "notes": "Pass requires detectable moderate-response and non-trivial stress-response.",
    }


def _cycle5_metric_contract() -> dict[str, Any]:
    payload = json.loads(
        (root / "outputs" / "sota_lock" / "internal_sota_lock.json").read_text(
            encoding="utf-8"
        )
    )
    baselines = payload.get("baselines", {})
    has_pr = all("pr_auc" in v for v in baselines.values())
    has_no_class_cindex = all("c_index" not in v for v in baselines.values())
    has_survival_block = "survival_metrics" in payload
    passed = bool(has_pr and has_no_class_cindex and has_survival_block)
    return {
        "cycle": "C5_metric_contract_separation",
        "pass": passed,
        "classification_models": list(baselines.keys()),
        "classification_has_pr_auc": has_pr,
        "classification_excludes_c_index": has_no_class_cindex,
        "has_survival_metrics_block": has_survival_block,
        "notes": "Pass requires explicit split between classification and survival metrics.",
    }


def _cycle6_clinical_gate(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    fail_cycles = [c["cycle"] for c in cycles if not c["pass"]]
    report_candidates = sorted(
        (root / "Docs" / "audit").glob("EXTERNAL_VALIDATION_REPORT_EV_*.md")
    )
    metrics_candidates = sorted(
        (root / "outputs" / "external_validation").glob("*/external_metrics.json")
    )
    external_validation_artifact = report_candidates[-1] if report_candidates else None
    external_metrics_artifact = metrics_candidates[-1] if metrics_candidates else None

    has_external_validation = (
        external_validation_artifact is not None
        and external_metrics_artifact is not None
    )

    real_data_only = False
    if external_metrics_artifact is not None:
        try:
            payload = json.loads(external_metrics_artifact.read_text(encoding="utf-8"))
            governance = payload.get("governance", {})
            real_data_only = bool(
                governance.get("real_data_only", False)
                and not governance.get("synthetic_label_generation", True)
            )
        except Exception:
            real_data_only = False

    passed = len(fail_cycles) == 0 and has_external_validation and real_data_only
    return {
        "cycle": "C6_clinical_readiness_gate",
        "pass": passed,
        "failed_dependencies": fail_cycles,
        "external_validation_artifact": (
            str(external_validation_artifact)
            if external_validation_artifact is not None
            else ""
        ),
        "external_metrics_artifact": (
            str(external_metrics_artifact)
            if external_metrics_artifact is not None
            else ""
        ),
        "has_external_validation_artifact": has_external_validation,
        "external_validation_real_data_only": real_data_only,
        "notes": (
            "Clinical readiness is blocked unless all prior cycles pass and "
            "a run-tagged external validation artifact (real-data-only) exists."
        ),
    }


def _write_report(payload: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Clinical Hardening Review Cycles",
        "",
        "Rigorous stepwise review for FUZZY GIMAN internal hardening.",
        "",
    ]
    for c in payload["cycles"]:
        status = "PASS" if c["pass"] else "FAIL"
        lines += [f"## {c['cycle']} — {status}", ""]
        for k, v in c.items():
            if k in {"cycle", "pass"}:
                continue
            lines.append(f"- {k}: `{v}`")
        lines.append("")

    lines += [
        "## Summary",
        f"- overall_pass: `{payload['overall_pass']}`",
        "- external_validation_required: `True`",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run hardening cycles and write machine-readable plus markdown reports."""
    from giman_pipeline.explainability.appendix_figures import generate_appendix_package
    from giman_pipeline.sota.benchmark import run_internal_sota_lock

    hardened_ckpt = (
        root / "outputs" / "phase9_neuro_fuzzy_hardened" / "neuro_fuzzy_best.pth"
    )
    checkpoint_path = (
        hardened_ckpt
        if hardened_ckpt.exists()
        else root
        / "outputs"
        / "phase9_neuro_fuzzy_sota_run_from50ckpt"
        / "neuro_fuzzy_best.pth"
    )

    # Generate latest appendix + benchmark artifacts before cycle checks
    generate_appendix_package(
        output_root=root / "visualizations" / "appendix",
        index_md=root / "Docs" / "audit" / "APPENDIX_EXPLAINABILITY_INDEX.md",
        provenance_json=root / "visualizations" / "appendix" / "provenance.json",
        checkpoint_override=checkpoint_path,
    )
    run_internal_sota_lock(
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
        output_json_path=root / "outputs" / "sota_lock" / "internal_sota_lock.json",
        output_report_path=root / "Docs" / "audit" / "SOTA_INTERNAL_LOCK_REPORT.md",
        figure_dir=root / "visualizations" / "publication_internal",
    )

    cycles: list[dict[str, Any]] = []
    cycles.append(_cycle1_proxy_dependency(checkpoint_path=checkpoint_path))
    cycles.append(_cycle2_imbalance_and_objective())
    cycles.append(_cycle3_calibration())
    cycles.append(_cycle4_digital_twin_sensitivity())
    cycles.append(_cycle5_metric_contract())
    cycles.append(_cycle6_clinical_gate(cycles))

    overall_pass = all(c["pass"] for c in cycles[:-1]) and cycles[-1]["pass"]
    payload = {
        "overall_pass": overall_pass,
        "cycles": cycles,
    }
    out_json = root / "Docs" / "audit" / "CLINICAL_HARDENING_REVIEW_CYCLES.json"
    out_md = root / "Docs" / "audit" / "CLINICAL_HARDENING_REVIEW_CYCLES.md"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(payload, out_md)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
