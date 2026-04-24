"""Paper 1 R2-Q7 — Extract abstention / multi-label rates from conformal JSONs.

Reviewer Q7: What fraction of cases receive empty or multi-label prediction sets
at 80/90/95% CLs internally and externally?

Data mines:
  outputs/paper1_conformal/ (internal 4 targets × 3 CLs)
  outputs/paper1_external_conformal/results/*.json (external binary/3class/nsd_positive)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("q7_abstention")

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def summarize_set_sizes(set_sizes: list, cohort: str, target: str, alpha: float) -> dict:
    s = np.asarray(set_sizes)
    n = len(s)
    if n == 0:
        return {"cohort": cohort, "target": target, "alpha": alpha, "n": 0}
    return {
        "cohort": cohort,
        "target": target,
        "alpha": alpha,
        "confidence_level_pct": int((1 - alpha) * 100),
        "n": int(n),
        "empty_fraction": float(np.mean(s == 0)),
        "singleton_fraction": float(np.mean(s == 1)),
        "multi_label_fraction": float(np.mean(s > 1)),
        "mean_set_size": float(np.mean(s)),
        "median_set_size": float(np.median(s)),
        "max_set_size": int(np.max(s)),
    }


def mine_external(path: Path) -> list[dict]:
    """outputs/paper1_external_conformal/results/<target>.json contains per-alpha, per-fold set_sizes."""
    d = json.load(open(path))
    target = d["target"]
    records = []
    if "aggregates" in d:
        for alpha_key, tiers in d["aggregates"].items():
            alpha = float(alpha_key.replace("alpha_", ""))
            for cohort_name in ("internal", "external"):
                t = tiers.get(cohort_name, {})
                if "set_size_distribution" in t:
                    sizes = t["set_size_distribution"]
                    records.append(summarize_set_sizes(sizes, cohort_name, target, alpha))
                elif "mean_set_size_mean" in t:
                    # Aggregate-only; fall back to derived stats
                    records.append({
                        "cohort": cohort_name, "target": target, "alpha": alpha,
                        "confidence_level_pct": int((1 - alpha) * 100),
                        "mean_set_size": t.get("mean_set_size_mean"),
                        "coverage": t.get("coverage_mean"),
                        "note": "from aggregate stats; raw distribution not archived",
                    })
    return records


def main():
    records = []
    # Internal + external from paper1_external_conformal
    for tgt in ("binary", "3class", "nsd_positive"):
        p = ROOT / "outputs" / "paper1_external_conformal" / "results" / f"{tgt}.json"
        if p.exists():
            records.extend(mine_external(p))
            log.info("mined %s", p.name)
        else:
            log.warning("missing: %s", p)

    # Aggregate into pivot-style summary
    summary = {"by_cohort_target_alpha": records, "note": (
        "Abstention = empty set; multi-label = set size > 1. "
        "Extracted from existing conformal JSONs; per-patient set-size "
        "distributions stored only for external cohorts via "
        "paper1_external_conformal runs (aggregates for internal). "
        "Internal-only per-patient distribution requires rerun of cross-conformal "
        "with per-patient export; flagged as follow-up."
    )}
    # Concise table
    tbl = []
    for r in records:
        row = (
            f"{r['cohort']:<10} {r['target']:<14} alpha={r['alpha']:.2f} "
            f"CL={r.get('confidence_level_pct', '?')}%  "
            f"empty={r.get('empty_fraction', 'n/a') if isinstance(r.get('empty_fraction'), (int, float)) else 'n/a':<8} "
            f"multi={r.get('multi_label_fraction', 'n/a') if isinstance(r.get('multi_label_fraction'), (int, float)) else 'n/a':<8} "
            f"mean|C|={r.get('mean_set_size', 0.0):.3f}"
        )
        tbl.append(row)
        print(row)
    summary["printable_table"] = tbl

    out = OUTPUT_DIR / "q7_abstention_rates.json"
    out.write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
