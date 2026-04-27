"""WS-P3-6: Markov multi-state model predictive metrics at fixed horizons.

Computes time-dependent concordance (C-td) and Integrated Brier Score (IBS) for
the multi-state Markov model already fit in `outputs/paper3_markov/markov_results.json`,
evaluated at 1/2/5/10 yr horizons on each Paper 3 cross-validation test fold.

Per the WS-P3-6 plan in
``Docs/superpowers/plans/2026-04-23-paper3plus4-reviewer-response-execution.md``:

  1. Load the Q matrix (continuous-time Markov chain intensity matrix; 7 states:
     0, 1, 2B, 3, 4, 5, 6) from ``markov_results.json``.
  2. For each (fold, patient) compute the cumulative incidence
     P(reach any non-baseline stage by time t) at t in {1, 2, 5, 10} yr via
     P(t) = expm(Q * t). Patient-independent under the homogeneous CTMC.
  3. Compute C-td (Uno 2007) for each fold against observed transitions, and
     IBS (Graf 1999) at the four horizons.
  4. Write ``outputs/paper3plus4_revision/markov_metrics/markov_ctd_ibs_at_horizons.json``.

Output schema:
    {
      "model": "Markov",
      "horizons_yr": [1, 2, 5, 10],
      "per_fold": [{"fold_idx": 0, "n_test": 380, "ctd": ..., "ibs": ...}, ...],
      "aggregate": {"ctd_mean": ..., "ctd_std": ..., "ibs_mean": ..., "ibs_std": ...},
      "method_note": "..."
    }

Reproducibility:
    Run from project root::

        .venv/bin/python scripts/paper3plus4/run_markov_predictive_metrics.py

    All inputs are pre-computed (Q, transitions, fold assignments). Wall time
    < 30 seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import expm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MARKOV_JSON = PROJECT_ROOT / "outputs/paper3_markov/markov_results.json"
LONG_NSDISS_CSV = PROJECT_ROOT / "data/06_longitudinal_staging/longitudinal_nsd_iss.csv"
TRANSITIONS_CSV = PROJECT_ROOT / "data/06_longitudinal_staging/transition_events.csv"
DEEPHIT_CKPT_DIR = PROJECT_ROOT / "outputs/paper3_checkpoints/deephit"

OUTPUT_DIR = PROJECT_ROOT / "outputs/paper3plus4_revision/markov_metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS_YR = [1, 2, 5, 10]
HORIZONS_MO = [int(yr * 12) for yr in HORIZONS_YR]

# Stage codes used in markov_results.json (7 states)
STAGE_LABELS = ["0", "1", "2B", "3", "4", "5", "6"]
STAGE_TO_IDX = {label: i for i, label in enumerate(STAGE_LABELS)}


def load_q_matrix() -> np.ndarray:
    """Load the Markov intensity matrix Q from the canonical results JSON."""
    with open(MARKOV_JSON) as f:
        d = json.load(f)
    Q = np.array(d["Q"])
    assert Q.shape == (7, 7), f"Expected 7-state Q matrix, got {Q.shape}"
    # Sanity: rows of Q should sum to ~0 (CTMC property)
    assert np.allclose(Q.sum(axis=1), 0, atol=1e-10), "Q row sums must be 0"
    return Q


def load_fold_test_patients() -> list[list[int]]:
    """Load per-fold test patient lists from DeepHit checkpoints (canonical)."""
    fold_pats: list[list[int]] = []
    for fold_idx in range(5):
        ckpt_path = DEEPHIT_CKPT_DIR / f"fold{fold_idx}_deephit.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        cp = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        test_pats = [int(x) for x in cp["test_pats"]]
        fold_pats.append(test_pats)
    return fold_pats


def get_baseline_stages(test_pats: list[int]) -> dict[int, int]:
    """Return {patno: baseline_stage_idx} for the requested patients.

    The "baseline" is each patient's earliest staged visit in the longitudinal
    table. Stage strings (``0/1/2B/3/4/5/6``) map to indices 0..6.
    """
    df = pd.read_csv(LONG_NSDISS_CSV)
    # Take earliest visit per patient (column is months_from_baseline; canonical)
    df = df.sort_values(["PATNO", "months_from_baseline"])
    baseline = df.groupby("PATNO").first().reset_index()
    baseline = baseline[baseline["PATNO"].isin(test_pats)]
    out = {}
    for _, row in baseline.iterrows():
        stage = str(row["nsd_stage"])
        if stage in STAGE_TO_IDX:
            out[int(row["PATNO"])] = STAGE_TO_IDX[stage]
    return out


def get_observed_transitions(test_pats: list[int]) -> dict[int, list[dict]]:
    """Return {patno: [{time_months, dest_stage_idx}, ...]} of observed transitions."""
    df = pd.read_csv(TRANSITIONS_CSV)
    df = df[df["PATNO"].isin(test_pats)]
    out: dict[int, list[dict]] = {p: [] for p in test_pats}
    for _, row in df.iterrows():
        pat = int(row["PATNO"])
        dest = str(row["dest_stage"])
        if dest not in STAGE_TO_IDX:
            continue
        out.setdefault(pat, []).append(
            dict(
                time_months=float(row["months_from_baseline_dst"]),
                dest_stage_idx=STAGE_TO_IDX[dest],
            )
        )
    return out


def cif_at_horizon(Q: np.ndarray, baseline_stage: int, t_yr: float) -> np.ndarray:
    """CIF[k] = P(in stage k at time t | baseline = baseline_stage)."""
    P = expm(Q * t_yr)
    return P[baseline_stage]  # row of P, length 7


def compute_ctd(
    Q: np.ndarray,
    baseline_stages: dict[int, int],
    transitions: dict[int, list[dict]],
    horizon_yr: float,
) -> float:
    """Time-dependent concordance (Uno 2007 style) at a single horizon.

    For each pair (i, j) with patient i transitioning before t and patient j
    not transitioning before t, count concordance if the predicted CIF for
    i's destination cause is higher for patient i than for patient j at the
    same horizon. Uses a simple destination-cause-aware concordance.

    Returns NaN if there are too few comparable pairs (< 50).
    """
    horizon_mo = horizon_yr * 12.0
    pats = list(baseline_stages.keys())
    if len(pats) < 2:
        return float("nan")

    # Pre-compute CIF row for each baseline stage (cache 7 rows)
    cif_rows = {s: cif_at_horizon(Q, s, horizon_yr) for s in set(baseline_stages.values())}

    # Identify cases (transition by horizon) and controls (no transition by horizon)
    cases = []  # list of (pat, dest_idx, observed_time)
    controls = []  # list of pat
    for p in pats:
        trs = transitions.get(p, [])
        in_window = [t for t in trs if t["time_months"] <= horizon_mo]
        if in_window:
            # First transition in window
            t = sorted(in_window, key=lambda x: x["time_months"])[0]
            cases.append((p, t["dest_stage_idx"], t["time_months"]))
        else:
            controls.append(p)

    if not cases or not controls:
        return float("nan")

    n_pairs = 0
    n_concordant = 0
    rng = np.random.RandomState(42)
    # Subsample if pair-space is too large (cases × controls)
    max_pairs = 50_000
    pair_indices = []
    if len(cases) * len(controls) > max_pairs:
        # Random pair sample
        for _ in range(max_pairs):
            ci = rng.randint(len(cases))
            di = rng.randint(len(controls))
            pair_indices.append((ci, di))
    else:
        pair_indices = [(ci, di) for ci in range(len(cases)) for di in range(len(controls))]

    for ci, di in pair_indices:
        case_pat, dest_idx, _ = cases[ci]
        ctrl_pat = controls[di]
        case_pred = cif_rows[baseline_stages[case_pat]][dest_idx]
        ctrl_pred = cif_rows[baseline_stages[ctrl_pat]][dest_idx]
        n_pairs += 1
        if case_pred > ctrl_pred:
            n_concordant += 1
        elif case_pred == ctrl_pred:
            n_concordant += 0.5

    if n_pairs < 50:
        return float("nan")
    return n_concordant / n_pairs


def compute_ibs(
    Q: np.ndarray,
    baseline_stages: dict[int, int],
    transitions: dict[int, list[dict]],
    horizons_yr: list[float],
) -> float:
    """Integrated Brier Score across given horizons.

    For each horizon t in horizons_yr:
      BS(t) = (1/N) * sum_i ( I(transition_i <= t) - CIF_i(any non-baseline, t) )^2
    The IBS is the trapezoidal integral of BS(t) over horizons_yr divided by
    the integration interval.
    """
    pats = list(baseline_stages.keys())
    if not pats:
        return float("nan")

    cif_cache = {
        (s, t): cif_at_horizon(Q, s, t)
        for s in set(baseline_stages.values())
        for t in horizons_yr
    }

    bs_per_horizon = []
    for t_yr in horizons_yr:
        t_mo = t_yr * 12.0
        sq = 0.0
        for p in pats:
            base = baseline_stages[p]
            cif_row = cif_cache[(base, t_yr)]
            # Probability of being in any state OTHER than the baseline state by t
            pred = 1.0 - cif_row[base]
            obs = 1.0 if any(tr["time_months"] <= t_mo for tr in transitions.get(p, [])) else 0.0
            sq += (obs - pred) ** 2
        bs_per_horizon.append(sq / len(pats))

    # Trapezoidal integration over [horizons_yr[0], horizons_yr[-1]]
    if len(horizons_yr) < 2:
        return float("nan")
    return float(np.trapezoid(bs_per_horizon, x=horizons_yr) / (horizons_yr[-1] - horizons_yr[0]))


def main() -> int:
    print(f"[WS-P3-6] Loading Markov Q matrix from {MARKOV_JSON}")
    Q = load_q_matrix()

    print("[WS-P3-6] Loading per-fold test patient lists from DeepHit checkpoints")
    fold_test_pats = load_fold_test_patients()

    print("[WS-P3-6] Computing per-fold metrics ...")
    per_fold = []
    for fold_idx, test_pats in enumerate(fold_test_pats):
        baseline_stages = get_baseline_stages(test_pats)
        transitions = get_observed_transitions(test_pats)

        # C-td at each horizon
        ctd_per_horizon = {
            f"ctd_{t}yr": compute_ctd(Q, baseline_stages, transitions, t)
            for t in HORIZONS_YR
        }
        ibs = compute_ibs(Q, baseline_stages, transitions, HORIZONS_YR)
        ctd_overall = float(np.nanmean(list(ctd_per_horizon.values())))

        per_fold.append(
            dict(
                fold_idx=fold_idx,
                n_test=len(test_pats),
                n_with_baseline=len(baseline_stages),
                n_with_transitions=sum(1 for p in test_pats if transitions.get(p)),
                ctd_overall=ctd_overall,
                **ctd_per_horizon,
                ibs=float(ibs),
            )
        )
        print(
            f"  fold {fold_idx}: n_test={len(test_pats)}, n_baseline={len(baseline_stages)}, "
            f"C-td_overall={ctd_overall:.4f}, IBS={ibs:.4f}"
        )

    # Aggregate
    ctd_means = [f["ctd_overall"] for f in per_fold if not np.isnan(f["ctd_overall"])]
    ibs_means = [f["ibs"] for f in per_fold if not np.isnan(f["ibs"])]
    aggregate = dict(
        ctd_overall_mean=float(np.mean(ctd_means)) if ctd_means else float("nan"),
        ctd_overall_std=float(np.std(ctd_means, ddof=1)) if len(ctd_means) > 1 else float("nan"),
        ibs_mean=float(np.mean(ibs_means)) if ibs_means else float("nan"),
        ibs_std=float(np.std(ibs_means, ddof=1)) if len(ibs_means) > 1 else float("nan"),
    )
    for t in HORIZONS_YR:
        key = f"ctd_{t}yr"
        vals = [f[key] for f in per_fold if not np.isnan(f[key])]
        if vals:
            aggregate[f"{key}_mean"] = float(np.mean(vals))
            aggregate[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")

    out = dict(
        model="Multi-state Markov (homogeneous CTMC)",
        horizons_yr=HORIZONS_YR,
        per_fold=per_fold,
        aggregate=aggregate,
        method_note=(
            "C-td computed via destination-cause-aware concordance with up to 50,000 "
            "case-control pair samples (random_state=42). IBS computed via trapezoidal "
            "integration of per-horizon Brier scores over the four horizons. The Markov "
            "CIF is patient-independent given baseline stage (homogeneous CTMC), so "
            "discrimination is bounded by the variation in baseline-stage assignment. "
            "Results are reported as the parametric sojourn-time reference per "
            "WS-P3-S3 in the npj-DM Table II Markov row."
        ),
        source_q_matrix=str(MARKOV_JSON.relative_to(PROJECT_ROOT)),
        source_fold_assignments=str(DEEPHIT_CKPT_DIR.relative_to(PROJECT_ROOT)),
    )

    out_path = OUTPUT_DIR / "markov_ctd_ibs_at_horizons.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[WS-P3-6] Wrote {out_path}")
    print(f"[WS-P3-6] Aggregate: C-td={aggregate['ctd_overall_mean']:.4f} ± "
          f"{aggregate['ctd_overall_std']:.4f}; IBS={aggregate['ibs_mean']:.4f} ± "
          f"{aggregate['ibs_std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
