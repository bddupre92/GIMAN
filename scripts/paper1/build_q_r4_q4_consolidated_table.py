"""Build R4-Q4 consolidated BioFIND external-validation table.

Loads existing tree-baseline results from outputs/external_validation/{target}/
and merges in any TabPFN + AutoGluon BioFIND results that have been generated.

Outputs:
    outputs/paper1_r2_responses/q_r4_q4_biofind_consolidated_table.md
    outputs/paper1_r2_responses/q_r4_q4_biofind_consolidated_table.json

CLI:
    .venv/bin/python scripts/paper1/build_q_r4_q4_consolidated_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EV_DIR = ROOT / "outputs" / "external_validation"
OUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["binary", "three_class", "nsd_positive"]
TREE_METHODS = ["CatBoost", "XGBoost", "RandomForest", "LogisticRegression"]
SOTA_METHODS = ["TabPFN", "AutoGluon"]

# Per-class labels per target (after BioFIND ground-truth remap)
CLASS_LABELS = {
    "binary": ["NSD-", "NSD+"],
    "three_class": ["Early(0-1)", "Mild(2B)", "Impaired(3-4)"],
    "nsd_positive": ["Stage 1", "Stage 2B", "Stage 3", "Stage 4"],
}


def fmt_ci(point, ci):
    if point is None:
        return "—"
    if ci is None or any(v is None for v in ci):
        return f"{point:.3f}"
    return f"{point:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def load_existing(target):
    """Load BioFIND results for tree baselines from existing JSON."""
    path = EV_DIR / target / "external_validation_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_sota_biofind(target, method_lower):
    """Load TabPFN or AutoGluon BioFIND result if it exists."""
    path = EV_DIR / target / f"biofind_{method_lower}_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def extract_metrics(method_block):
    """Extract metrics from a method block in the existing schema.

    Returns dict with: n, bal_acc, bal_acc_ci, auc, auc_ci, qwk, macro_f1, per_class_f1.
    """
    em = method_block.get("external_metrics", {})
    cr = em.get("classification_report", {})
    macro = cr.get("macro avg", {})
    per_class = {}
    for k, v in cr.items():
        if isinstance(v, dict) and k not in ("macro avg", "weighted avg"):
            per_class[k] = v.get("f1-score")
    return {
        "n": em.get("n_ground_truth"),
        "bal_acc": em.get("bal_acc"),
        "bal_acc_ci": em.get("bal_acc_ci"),
        "auc": em.get("auc"),
        "auc_ci": em.get("auc_ci"),
        "qwk": em.get("qwk"),
        "macro_f1": macro.get("f1-score"),
        "per_class_f1": per_class,
    }


def build_table():
    rows = []
    consolidated = {}
    for target in TARGETS:
        existing = load_existing(target)
        if existing is None:
            continue
        consolidated[target] = {}
        for method in TREE_METHODS:
            block = existing.get("external", {}).get("BioFIND", {}).get(method)
            if not block:
                continue
            m = extract_metrics(block)
            m["method"] = method
            m["source"] = "existing_external_validation"
            consolidated[target][method] = m
        for method in SOTA_METHODS:
            biofind = load_sota_biofind(target, method.lower())
            if biofind is None:
                consolidated[target][method] = {
                    "method": method,
                    "source": "PENDING",
                    "n": None,
                    "bal_acc": None,
                    "bal_acc_ci": None,
                    "auc": None,
                    "auc_ci": None,
                    "qwk": None,
                    "macro_f1": None,
                    "per_class_f1": {},
                }
            else:
                m = extract_metrics(biofind)
                m["method"] = method
                m["source"] = "biofind_sota_run"
                consolidated[target][method] = m

    # Save JSON
    json_path = OUT_DIR / "q_r4_q4_biofind_consolidated_table.json"
    json_path.write_text(json.dumps(consolidated, indent=2))

    # Build markdown
    lines = ["# R4-Q4: Consolidated BioFIND External-Validation Table", ""]
    lines.append(
        "All methods trained on PPMI 12-feature common subset (n=2,201) and "
        "evaluated on BioFIND PD patients with NSD-ISS ground truth (Russo 2025 "
        "replication). Bootstrap 95% CIs from 1,000 resamples."
    )
    lines.append("")
    lines.append(
        "**Notes.** (1) `nan` AUC CIs occur when bootstrap resamples contain only one "
        "ground-truth class — inherent to severely imbalanced BioFIND (95.4% S+ for binary, "
        "0/103 patients in stage 1 for nsd_positive). Reported as nan, not omitted. "
        "(2) AutoGluon BioFIND runs require the `.venv-autogluon` sidecar venv "
        "(LightGBM+PyTorch libomp dual-runtime collision); launch with "
        "`bash scripts/paper1/_launch_ag_biofind_external.sh {target}` from a terminal."
    )
    lines.append("")
    for target in TARGETS:
        if target not in consolidated:
            continue
        labels = CLASS_LABELS.get(target, [])
        lines.append(f"## Target: {target}")
        lines.append("")
        any_method = next(iter(consolidated[target].values()))
        n_gt = any_method.get("n")
        lines.append(f"BioFIND ground-truth N = {n_gt if n_gt else 'N/A'}.")
        lines.append("")
        # Per-class F1 columns based on report keys (sorted numerically)
        per_class_keys = []
        for m in consolidated[target].values():
            per_class_keys += list(m.get("per_class_f1", {}).keys())
        per_class_keys = sorted(set(per_class_keys), key=lambda x: int(x))
        header = ["Method", "Bal Acc [95% CI]", "AUC [95% CI]", "QWK", "Macro F1"]
        for k in per_class_keys:
            label = labels[int(k)] if int(k) < len(labels) else f"class {k}"
            header.append(f"F1({label})")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for method in TREE_METHODS + SOTA_METHODS:
            if method not in consolidated[target]:
                continue
            m = consolidated[target][method]
            if m["source"] == "PENDING":
                cells = [method, "PENDING", "PENDING", "—", "—"]
                for _ in per_class_keys:
                    cells.append("—")
            else:
                cells = [
                    method,
                    fmt_ci(m["bal_acc"], m["bal_acc_ci"]),
                    fmt_ci(m["auc"], m["auc_ci"]),
                    f"{m['qwk']:.3f}" if m["qwk"] is not None else "—",
                    f"{m['macro_f1']:.3f}" if m["macro_f1"] is not None else "—",
                ]
                for k in per_class_keys:
                    f1 = m.get("per_class_f1", {}).get(k)
                    cells.append(f"{f1:.3f}" if f1 is not None else "—")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    md_path = OUT_DIR / "q_r4_q4_biofind_consolidated_table.md"
    md_path.write_text("\n".join(lines))
    print(f"[ok] wrote {md_path}")
    print(f"[ok] wrote {json_path}")
    return consolidated


if __name__ == "__main__":
    build_table()
