"""Paper 1 WS1.6 — Medication Sensitivity Decision Rule Summariser.

Reads ``outputs/paper1_medication_sensitivity/results/arm_{1,2,3}_{target}.json``,
extracts the three per-arm deltas on the primary ``binary`` target, and writes
the pre-registered decision verdict to
``outputs/paper1_medication_sensitivity/decision_rule_verdict.json``.

Decision rule (locked in PRE_REGISTRATION.md):

    max_abs_delta = max(|delta_arm1|, |delta_arm2|, |delta_arm3|)

    PROMOTE-TO-ANALYSIS-F if max_abs_delta > 0.02 AND the 95% bootstrap CI for
    at least one of the three deltas excludes 0.

    REPORT-NULL-IN-S-5.8 otherwise.

The summariser also prints a secondary-target comparison for the 3-class,
full-ordinal, and nsd_positive JSONs if they are present.

Author: Blair Dupre (UND BME)
Date: April 2026
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "paper1_medication_sensitivity"
RESULTS_DIR = OUT_DIR / "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROMOTE_THRESHOLD = 0.02


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_arm_delta(
    target: str,
) -> dict[str, Any]:
    """Return a dict with delta_arm1, delta_arm2, delta_arm3, their CIs, and
    whether each CI excludes 0. Missing JSONs contribute NaN deltas (not fatal,
    so partial runs can still generate a provisional verdict)."""
    a1 = _load(RESULTS_DIR / f"arm_1_{target}.json")
    a2 = _load(RESULTS_DIR / f"arm_2_{target}.json")
    a3 = _load(RESULTS_DIR / f"arm_3_{target}.json")

    # Arm 1 — interaction test delta = AUC(off) - AUC(on)
    delta1 = None
    ci1 = None
    ci1_excludes = False
    if a1 is not None:
        it = a1.get("interaction_test") or {}
        if not it.get("skipped"):
            delta1 = it.get("delta_arm1_off_minus_on")
            ci1 = it.get("bootstrap_delta_95ci")
            ci1_excludes = bool(it.get("ci_excludes_zero", False))
        else:
            # Multi-class fallback — use the non-bootstrap scalar if present
            delta1 = it.get("delta_arm1_off_minus_on")

    # Arm 2 — medication-LOCO delta = AUC(train-off,test-on) - AUC(train-on,test-off)
    delta2 = None
    ci2 = None
    ci2_excludes = False  # No per-delta bootstrap CI is computed for Arm 2; the
    # per-direction AUCs each have their own CI. We conservatively set
    # ci2_excludes=False so Arm 2 alone never triggers PROMOTE; the decision
    # rule requires AT LEAST ONE arm's CI to exclude 0.
    if a2 is not None:
        delta2 = a2.get("delta_arm2")

    # Arm 3 — covariate delta = AUC(23-feat) - AUC(22-feat)
    delta3 = None
    ci3 = None
    ci3_excludes = False
    if a3 is not None:
        delta3 = a3.get("delta_arm3")
        ci3 = a3.get("delta_arm3_paired_bootstrap_95ci")
        ci3_excludes = bool(a3.get("delta_arm3_ci_excludes_zero", False))

    return {
        "target": target,
        "delta_arm1": delta1,
        "delta_arm1_95ci": ci1,
        "delta_arm1_ci_excludes_zero": ci1_excludes,
        "delta_arm2": delta2,
        "delta_arm2_95ci": ci2,
        "delta_arm2_ci_excludes_zero": ci2_excludes,
        "delta_arm3": delta3,
        "delta_arm3_95ci": ci3,
        "delta_arm3_ci_excludes_zero": ci3_excludes,
        "arm_json_found": {
            "arm_1": a1 is not None,
            "arm_2": a2 is not None,
            "arm_3": a3 is not None,
        },
    }


def _apply_decision_rule(deltas: dict[str, Any]) -> dict[str, Any]:
    """Apply the locked PROMOTE-vs-NULL decision rule."""
    ds = [
        deltas.get("delta_arm1"),
        deltas.get("delta_arm2"),
        deltas.get("delta_arm3"),
    ]
    ds_finite = [abs(d) for d in ds if d is not None and np.isfinite(d)]
    max_abs_delta = float(max(ds_finite)) if ds_finite else float("nan")

    any_ci_excludes_zero = any(
        bool(deltas.get(f"delta_arm{k}_ci_excludes_zero", False))
        for k in (1, 2, 3)
    )

    promote = (
        np.isfinite(max_abs_delta)
        and max_abs_delta > PROMOTE_THRESHOLD
        and any_ci_excludes_zero
    )

    return {
        "max_abs_delta": max_abs_delta,
        "threshold": PROMOTE_THRESHOLD,
        "any_ci_excludes_zero": any_ci_excludes_zero,
        "promote_to_analysis_f": bool(promote),
        "report_null_in_s5_8": bool(not promote),
        "decision_verdict": (
            "PROMOTE-TO-ANALYSIS-F" if promote else "REPORT-NULL-IN-S-5.8"
        ),
    }


def main() -> None:
    logger.info("=" * 70)
    logger.info("Paper 1 WS1.6 — Medication Sensitivity Decision Summary")
    logger.info("=" * 70)

    primary = _extract_arm_delta("binary")
    verdict = _apply_decision_rule(primary)

    # Secondary targets: extract deltas but do NOT feed them into the decision
    # rule (per PRE_REGISTRATION.md the decision is binary-only).
    secondary = {
        t: _extract_arm_delta(t) for t in ("3class", "full_ordinal", "nsd_positive")
    }

    payload = {
        "primary_target": "binary",
        "primary": primary,
        "decision": verdict,
        "secondary": secondary,
        "pre_registration": "outputs/paper1_medication_sensitivity/PRE_REGISTRATION.md",
    }
    out_path = OUT_DIR / "decision_rule_verdict.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info("")
    logger.info(f"Primary target (binary):")
    for k in (1, 2, 3):
        d = primary.get(f"delta_arm{k}")
        ci = primary.get(f"delta_arm{k}_95ci")
        ex = primary.get(f"delta_arm{k}_ci_excludes_zero")
        ci_str = (
            f" [{ci[0]:+.4f}, {ci[1]:+.4f}]"
            if ci is not None and ci[0] is not None else ""
        )
        d_str = f"{d:+.4f}" if d is not None and np.isfinite(d) else "NA"
        logger.info(
            f"  delta_arm{k} = {d_str}{ci_str}  ci_excludes_zero={bool(ex)}"
        )
    logger.info("")
    logger.info(f"max_abs_delta = {verdict['max_abs_delta']:.4f}  "
                f"threshold = {verdict['threshold']}")
    logger.info(f"any_ci_excludes_zero = {verdict['any_ci_excludes_zero']}")
    logger.info(f"")
    logger.info(f"VERDICT: {verdict['decision_verdict']}")
    logger.info(f"")
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
