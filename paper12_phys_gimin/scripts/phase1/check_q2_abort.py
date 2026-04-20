"""CLI wrapper for Q2 abort gate. Reads smoke_summary.json, writes gate_q2_verdict.json."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from phys_gimin.q2_gate import evaluate_from_smoke_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("smoke_run_dir", type=Path, help="Directory containing smoke_summary.json")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/paper12_phys_gimin/gate_q2_verdict.json"))
    args = parser.parse_args()

    summary_path = args.smoke_run_dir / "smoke_summary.json"
    verdict = evaluate_from_smoke_summary(summary_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    verdict_dict = asdict(verdict)
    args.output.write_text(json.dumps(verdict_dict, indent=2))

    # Print aggregate verdict
    print("\n=== Q2 Gate — Aggregate ===")
    aggregate_keys = {k: v for k, v in verdict_dict.items() if k != "per_fraction"}
    print(json.dumps(aggregate_keys, indent=2))

    # Print per-fraction verdicts if present
    if verdict_dict.get("per_fraction"):
        print("\n=== Q2 Gate — Per-Fraction ===")
        for frac_key, frac_data in verdict_dict["per_fraction"].items():
            decision = frac_data["decision"]
            phys_rmse = frac_data.get("phys_rmse_median", float("nan"))
            mean_rmse = frac_data.get("mean_rmse_median", float("nan"))
            cv = frac_data.get("cv_phys", float("nan"))
            print(
                f"  frac={frac_key}: {decision:30s} "
                f"phys_rmse={phys_rmse:.4f}  mean_rmse={mean_rmse:.4f}  cv={cv:.3f}"
            )

    print(f"\nOverall decision: {verdict.decision}")
    print(f"Verdict written to: {args.output}")


if __name__ == "__main__":
    main()
