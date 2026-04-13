"""Merge Paper 2 benchmark results from multiple runs.

Combines authoritative GIMIN + classical baseline results with new DL baseline
results (GAIN, SAITS, MIWAE) into a single combined JSON.

Usage:
    python scripts/merge_benchmark_results.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "outputs" / "paper2_benchmark"

# Source runs
OLD_RUN = BENCHMARK_DIR / "runs" / "full_benchmark_20260222_160247"
NEW_RUN = BENCHMARK_DIR / "runs" / "dl_baselines_v4"
OUTPUT = BENCHMARK_DIR / "imputation_benchmark_results_combined.json"

# Models to take from each source
OLD_MODELS = {
    "Mean",
    "Median",
    "KNN",
    "MICE",
    "MissForest",
    "GIMIN_Vanilla",
    "GIMIN_StageConditioned",
    "GIMIN_StageGraphOnly",
    "GIMIN_StageDecoderOnly",
}
NEW_MODELS = {"GAIN", "SAITS", "MIWAE"}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def merge_section(old_section: dict, new_section: dict, section_name: str) -> dict:
    """Merge a summary or raw section across fractions."""
    merged = {}
    for frac_key in sorted(set(old_section) | set(new_section)):
        merged[frac_key] = {}
        old_frac = old_section.get(frac_key, {})
        new_frac = new_section.get(frac_key, {})

        # Add old models
        for model in OLD_MODELS:
            if model in old_frac:
                merged[frac_key][model] = old_frac[model]

        # Add new DL models
        for model in NEW_MODELS:
            if model in new_frac:
                merged[frac_key][model] = new_frac[model]
            else:
                print(
                    f"  WARNING: {model} missing from {section_name}/{frac_key} in new run"
                )

    return merged


def main():
    if not OLD_RUN.exists():
        print(f"ERROR: Old run not found: {OLD_RUN}")
        sys.exit(1)
    if not NEW_RUN.exists():
        print(f"ERROR: New run not found: {NEW_RUN}")
        sys.exit(1)

    old_results = load_json(OLD_RUN / "imputation_benchmark_results.json")
    new_results = load_json(NEW_RUN / "imputation_benchmark_results.json")

    combined = {
        "summary": merge_section(
            old_results["summary"], new_results["summary"], "summary"
        ),
        "raw": merge_section(old_results["raw"], new_results["raw"], "raw"),
    }

    # Verify all 12 models present
    for frac_key in combined["summary"]:
        models = set(combined["summary"][frac_key].keys())
        expected = OLD_MODELS | NEW_MODELS
        missing = expected - models
        if missing:
            print(f"WARNING: {frac_key} missing models: {missing}")
        else:
            print(f"  {frac_key}: {len(models)} models OK")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nCombined results saved to: {OUTPUT}")
    print(
        f"Total models: {len(combined['summary'][list(combined['summary'].keys())[0]])}"
    )

    # Print summary table
    print("\n" + "=" * 70)
    print("Combined Results Summary (frac=0.1)")
    print("=" * 70)
    frac = combined["summary"].get("frac_0.1", {})
    for model in sorted(frac.keys()):
        rmse = frac[model].get("rmse_mean", frac[model].get("rmse", "N/A"))
        r2 = frac[model].get("r2_mean", frac[model].get("r2", "N/A"))
        if isinstance(rmse, (int, float)):
            print(f"  {model:30s}: RMSE={rmse:.4f}, R²={r2:.4f}")
        else:
            print(f"  {model:30s}: {rmse}")


if __name__ == "__main__":
    main()
