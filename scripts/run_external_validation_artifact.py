from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


def _extract_metric(text: str, pattern: str, default: str = "n/a") -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else default


def main() -> None:
    """Generate Docs/audit external-validation artifact from Phase 6 analysis."""
    root = Path(__file__).resolve().parents[1]
    phase6_md = (
        root
        / "archive"
        / "development"
        / "phase6"
        / "PHASE6_REAL_PPMI_VALIDATION_ANALYSIS.md"
    )
    phase6_json = (
        root
        / "archive"
        / "development"
        / "phase6"
        / "phase6_real_ppmi_validation_results.json"
    )
    phase6_json_note = (
        "exists but appears partially truncated; used as reference only"
        if phase6_json.exists()
        else "missing"
    )

    md_text = phase6_md.read_text(encoding="utf-8", errors="ignore")

    n_patients = _extract_metric(md_text, r"Patients\*\*:\s*([0-9]+)")
    n_features = _extract_metric(md_text, r"Features\*\*:\s*([0-9]+)")
    motor_r2 = _extract_metric(md_text, r"Motor Prediction \(R²\):\s*([^\n]+)")
    motor_acc = _extract_metric(md_text, r"Motor Clinical Accuracy:\s*([^\n]+)")
    cog_auc = _extract_metric(md_text, r"Cognitive Classification \(AUC\):\s*([^\n]+)")
    cog_acc = _extract_metric(md_text, r"Cognitive Accuracy:\s*([^\n]+)")
    sensitivity = _extract_metric(md_text, r"Sensitivity:\s*([^\n]+)")
    specificity = _extract_metric(md_text, r"Specificity:\s*([^\n]+)")

    out = root / "Docs" / "audit" / "EXTERNAL_VALIDATION_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# External Validation Report",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "This report captures currently available out-of-sample validation evidence for FUZZY GIMAN.",
        "",
        "## Validation Sources",
        "- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/archive/development/phase6/PHASE6_REAL_PPMI_VALIDATION_ANALYSIS.md`",
        f"- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/archive/development/phase6/phase6_real_ppmi_validation_results.json` ({phase6_json_note})",
        "",
        "## External-Like Validation Summary (Phase 6 Real PPMI Analysis)",
        f"- Cohort size: `{n_patients}`",
        f"- Feature count: `{n_features}`",
        f"- Motor prediction (R²): `{motor_r2}`",
        f"- Motor clinical accuracy: `{motor_acc}`",
        f"- Cognitive AUC: `{cog_auc}`",
        f"- Cognitive accuracy: `{cog_acc}`",
        f"- Sensitivity: `{sensitivity}`",
        f"- Specificity: `{specificity}`",
        "",
        "## Interpretation",
        "- External-like generalization is currently weak on the available Phase 6 real-PPMI validation artifact.",
        "- This does not support clinical deployment claims at this stage.",
        "- Internal hardening can pass while clinical-readiness remains conditional on stronger external evidence.",
        "",
        "## Validation Tier and Caveats",
        "- This artifact is treated as provisional external-like evidence (not a fully independent, locked multi-site cohort validation).",
        "- A fully independent external cohort protocol is still recommended for publication-grade clinical claims.",
        "",
        "## Status",
        "- external_validation_artifact_present: `True`",
        "- clinical_deployment_ready: `False`",
        "",
        "## Next Required External Validation Steps",
        "1. Run locked protocol on an independent cohort with fixed preprocessing and model checkpoint.",
        "2. Report AUC/PR-AUC/C-index with confidence intervals and calibration metrics.",
        "3. Include subgroup robustness and transportability analysis.",
        "4. Attach full reproducibility manifest (data hash, split hash, model hash, code commit).",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
