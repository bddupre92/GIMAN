#!/usr/bin/env python3
"""Sensitivity analysis: Paper 10 counterfactual calibration at alternative LEDD thresholds.

Reviewer-facing question: is the 1.074 [0.88, 1.29] calibration slope
sensitive to the choice of 200 mg LEDD-escalation threshold? This script
re-runs the Paper 10 Task 6 calibration at 5 thresholds
(100, 150, 200, 300, 400 mg) and reports whether the calibration-slope CI
still contains 1.0 at each.

Output: outputs/defense_prep/sensitivity_ledd_threshold.json and console table.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "mechanistic_twin" / "phase5_observational_counterfactual.py"
OUT = PROJECT_ROOT / "outputs" / "defense_prep" / "sensitivity_ledd_threshold.json"

THRESHOLDS = [100.0, 150.0, 200.0, 300.0, 400.0]


def run_with_threshold(mg: float) -> dict:
    """Run the Paper 10 counterfactual script with a patched threshold via env monkey-patch."""
    src = SCRIPT.read_text()
    patched = src.replace(
        "LEDD_ESCALATION_THRESHOLD_MG = 200.0",
        f"LEDD_ESCALATION_THRESHOLD_MG = {mg}",
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=SCRIPT.parent) as fh:
        fh.write(patched)
        tmp_path = Path(fh.name)
    try:
        result = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            check=True,
        )
        json_path = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "paper10_mech_vs_giman" / "observational_counterfactual.json"
        payload = json.loads(json_path.read_text())
        return {
            "threshold_mg": mg,
            "n_events": payload["counts"]["n_events"],
            "n_patients": payload["counts"]["n_patients_with_escalation"],
            "slope": payload["calibration"]["slope"],
            "slope_ci": payload["calibration"]["slope_ci_95"],
            "intercept": payload["calibration"]["intercept"],
            "intercept_ci": payload["calibration"]["intercept_ci_95"],
            "r2": payload["calibration"]["r2"],
            "mae": payload["calibration"]["mae"],
            "slope_contains_1": payload["calibration"]["slope_ci_95"][0] <= 1.0 <= payload["calibration"]["slope_ci_95"][1],
        }
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for mg in THRESHOLDS:
        print(f"[sensitivity] Running at threshold={mg:.0f} mg...")
        try:
            row = run_with_threshold(mg)
            rows.append(row)
            print(f"  n_events={row['n_events']} slope={row['slope']:.3f} CI={row['slope_ci']} contains_1={row['slope_contains_1']}")
        except subprocess.CalledProcessError as e:
            print(f"  FAILED at threshold={mg}: {e.stderr[:200]}")
            rows.append({"threshold_mg": mg, "error": e.stderr[:500]})

    OUT.write_text(json.dumps({"thresholds": rows}, indent=2))
    print(f"\nWrote {OUT}")
    print("\n=== Sensitivity Summary ===")
    print(f"{'Threshold':>10s} {'N events':>9s} {'Slope':>7s} {'CI':>20s} {'Contains 1.0?':>15s}")
    for r in rows:
        if "error" in r:
            print(f"{r['threshold_mg']:>10.0f} ERROR")
            continue
        ci = f"[{r['slope_ci'][0]:.2f}, {r['slope_ci'][1]:.2f}]"
        flag = "YES" if r["slope_contains_1"] else "NO"
        print(f"{r['threshold_mg']:>10.0f} {r['n_events']:>9d} {r['slope']:>7.3f} {ci:>20s} {flag:>15s}")


if __name__ == "__main__":
    main()
