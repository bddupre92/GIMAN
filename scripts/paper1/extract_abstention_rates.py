"""Paper 1 R2-Q7 — Extract empty/singleton/multi-label fractions from archived conformal JSONs.

Reviewer Q7: What fraction of cases receive empty or multi-label prediction sets
at 80/90/95% CLs internally and externally?

ERRATUM (2026-04-23): Prior v1 of this script only mined
outputs/paper1_external_conformal/results/*.json and concluded per-patient
distributions for INTERNAL were not archived. That was wrong. Internal
`set_size_distribution` is present in outputs/paper1_conformal/*.json
(empty_set_rate + singleton_rate + full_set_rate + explicit histogram).
External per_fold_metrics[*] also has internal_size_distribution +
external_size_distribution at each of 5 folds × 4 alphas which aggregate
cleanly.

Sources:
  1. INTERNAL (CV+ on full PPMI): outputs/paper1_conformal/{binary,three_class,
     full_ordinal,nsd_positive}_conformal.json — 3 models × split/cross × 3 CLs.
  2. EXTERNAL (PPMI→BioFIND): outputs/paper1_external_conformal/results/
     {binary,3class,nsd_positive}.json — per_fold_metrics across 5 folds × 4 alphas.

Output: outputs/paper1_r2_responses/q7_abstention_rates.json
Plus publication table as markdown.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("q7_abstention")

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INTERNAL_DIR = ROOT / "outputs" / "paper1_conformal"
EXTERNAL_DIR = ROOT / "outputs" / "paper1_external_conformal" / "results"

TARGETS_INTERNAL = ["binary", "three_class", "full_ordinal", "nsd_positive"]
TARGETS_EXTERNAL = ["binary", "3class", "nsd_positive"]


def distribution_to_fractions(dist: dict, n_total: int) -> dict:
    """Convert {size: count} histogram to fractions."""
    empty = dist.get("0", dist.get(0, 0))
    single = dist.get("1", dist.get(1, 0))
    multi = sum(int(v) for k, v in dist.items() if int(k) >= 2)
    n = empty + single + multi
    if n == 0 or n != n_total:
        log.warning("  count mismatch: empty+single+multi=%d vs n_total=%d", n, n_total)
    n = max(n, n_total, 1)
    return {
        "n": int(n),
        "empty_fraction": round(empty / n, 4),
        "singleton_fraction": round(single / n, 4),
        "multi_label_fraction": round(multi / n, 4),
    }


def mine_internal(target: str) -> list[dict]:
    """Mine internal cross-conformal (CV+) and split-conformal rows."""
    path = INTERNAL_DIR / f"{target}_conformal.json"
    if not path.exists():
        log.warning("missing internal: %s", path)
        return []
    d = json.load(open(path))
    rows = []
    for model in d:
        for entry in d[model]:
            n_total = int(entry["n_test"])
            dist = entry.get("set_size_distribution", {})
            empty = int(dist.get("0", dist.get(0, 0)))
            single = int(dist.get("1", dist.get(1, 0)))
            multi = sum(int(v) for k, v in dist.items() if int(k) >= 2)
            rows.append({
                "cohort": "internal",
                "target": target,
                "model": model,
                "method": entry["conformal_method"],
                "alpha": round(1 - entry["confidence_level"], 2),
                "confidence_level": entry["confidence_level"],
                "confidence_level_pct": int(entry["confidence_level"] * 100),
                "n": n_total,
                "empty_fraction": round(empty / n_total, 4),
                "singleton_fraction": round(single / n_total, 4),
                "multi_label_fraction": round(multi / n_total, 4),
                "mean_set_size": round(entry["mean_set_size"], 4),
                "marginal_coverage": round(entry["marginal_coverage"], 4),
                "n_classes": int(entry["n_classes"]),
                "source": str(path.relative_to(ROOT)),
            })
    return rows


def mine_external(target: str) -> list[dict]:
    """Mine external (PPMI→BioFIND). Aggregate 5 folds per alpha."""
    path = EXTERNAL_DIR / f"{target}.json"
    if not path.exists():
        log.warning("missing external: %s", path)
        return []
    d = json.load(open(path))
    per_fold = d.get("per_fold_metrics", [])
    # Group per-fold rows by alpha, then sum distributions across folds
    by_alpha_cohort: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    by_alpha_cohort_n: dict = defaultdict(lambda: defaultdict(int))
    by_alpha_cov: dict = defaultdict(lambda: defaultdict(list))
    by_alpha_mss: dict = defaultdict(lambda: defaultdict(list))
    for entry in per_fold:
        alpha = entry["alpha"]
        for cohort_key, cov_key, mss_key, size_key, n_key in [
            ("internal", "internal_coverage", "internal_mean_set_size",
             "internal_size_distribution", "n_test_internal"),
            ("external", "external_coverage", "external_mean_set_size",
             "external_size_distribution", "n_test_external"),
        ]:
            size_dist = entry.get(size_key, {})
            for k, v in size_dist.items():
                by_alpha_cohort[alpha][cohort_key][int(k)] += int(v)
            by_alpha_cohort_n[alpha][cohort_key] += int(entry[n_key])
            by_alpha_cov[alpha][cohort_key].append(float(entry[cov_key]))
            by_alpha_mss[alpha][cohort_key].append(float(entry[mss_key]))
    # Emit aggregated rows
    rows = []
    for alpha in sorted(by_alpha_cohort):
        for cohort in ("internal", "external"):
            dist = dict(by_alpha_cohort[alpha][cohort])
            n_total = by_alpha_cohort_n[alpha][cohort]
            empty = dist.get(0, 0)
            single = dist.get(1, 0)
            multi = sum(int(v) for k, v in dist.items() if int(k) >= 2)
            cov_mean = sum(by_alpha_cov[alpha][cohort]) / len(by_alpha_cov[alpha][cohort])
            mss_mean = sum(by_alpha_mss[alpha][cohort]) / len(by_alpha_mss[alpha][cohort])
            rows.append({
                "cohort": f"external_eval::{cohort}",  # disambiguate from internal CV+ runs
                "target": target,
                "model": "catboost",
                "method": "split_cv_aware",  # external uses 5-fold split-conformal
                "alpha": round(alpha, 2),
                "confidence_level": round(1 - alpha, 2),
                "confidence_level_pct": int(round((1 - alpha) * 100)),
                "n": int(n_total),
                "empty_fraction": round(empty / max(n_total, 1), 4),
                "singleton_fraction": round(single / max(n_total, 1), 4),
                "multi_label_fraction": round(multi / max(n_total, 1), 4),
                "mean_set_size": round(mss_mean, 4),
                "marginal_coverage": round(cov_mean, 4),
                "n_folds_pooled": 5,
                "source": str(path.relative_to(ROOT)),
            })
    return rows


def summarize_catboost_primary(rows: list[dict]) -> dict:
    """Primary reviewer table — CatBoost cross-conformal (internal) + external split."""
    pri = [r for r in rows if r["model"] == "catboost"
           and (r["method"] == "cross"
                or r["method"] == "split_cv_aware")
           and r["confidence_level"] in (0.8, 0.9, 0.95)]
    pri.sort(key=lambda r: (r["target"], r["cohort"], r["confidence_level"]))
    return {"catboost_primary": pri}


def main() -> None:
    all_rows: list[dict] = []
    for tgt in TARGETS_INTERNAL:
        r = mine_internal(tgt)
        log.info("internal %s: %d rows", tgt, len(r))
        all_rows.extend(r)
    for tgt in TARGETS_EXTERNAL:
        r = mine_external(tgt)
        log.info("external %s: %d rows", tgt, len(r))
        all_rows.extend(r)

    summary = {
        "by_cohort_target_model_method_alpha": all_rows,
        "n_rows_total": len(all_rows),
        "sources": {
            "internal": str(INTERNAL_DIR.relative_to(ROOT)),
            "external": str(EXTERNAL_DIR.relative_to(ROOT)),
        },
        "note": (
            "Abstention = empty set (|C|=0); multi-label = |C|>=2. "
            "Internal from paper1_conformal/*.json (3 models x split/cross x 3 CLs). "
            "External from paper1_external_conformal/results/*.json (5-fold split "
            "pooled across folds per alpha). Full archaeology closed Q7 without "
            "needing a rerun."
        ),
        **summarize_catboost_primary(all_rows),
    }

    out = OUTPUT_DIR / "q7_abstention_rates.json"
    out.write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s (%d rows)", out, len(all_rows))

    # Publication markdown table for the primary reviewer-facing view
    lines = [
        "# Q7 — Empty / singleton / multi-label fractions (CatBoost)",
        "",
        "Cross-conformal on PPMI internal (n=2,201 binary / 2,197 3class+full_ord / 779 NSD+). "
        "Split-conformal 5-fold on PPMI→BioFIND external (pooled distribution).",
        "",
        "| Cohort | Target | CL | n | Empty | Singleton | Multi | Mean\\|C\\| | Coverage |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["catboost_primary"]:
        lines.append(
            f"| {r['cohort']} | {r['target']} | {int(r['confidence_level']*100)}% | "
            f"{r['n']} | {100*r['empty_fraction']:.1f}% | "
            f"{100*r['singleton_fraction']:.1f}% | "
            f"{100*r['multi_label_fraction']:.1f}% | "
            f"{r['mean_set_size']:.3f} | {100*r['marginal_coverage']:.1f}% |"
        )
    md_out = OUTPUT_DIR / "q7_abstention_table.md"
    md_out.write_text("\n".join(lines) + "\n")
    log.info("Wrote %s", md_out)


if __name__ == "__main__":
    main()
