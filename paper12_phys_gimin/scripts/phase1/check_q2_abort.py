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
    args.output.write_text(json.dumps(asdict(verdict), indent=2))
    print(json.dumps(asdict(verdict), indent=2))


if __name__ == "__main__":
    main()
