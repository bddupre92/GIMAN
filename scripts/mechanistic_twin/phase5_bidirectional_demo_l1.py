#!/usr/bin/env python3
"""L1-lite bidirectional demo: replay with per-visit σ (cross-paper integration L1).

Demonstrates that `update_posterior` now accepts a per-visit sigma vector
alongside the scalar default. Three regression variants run on a subset
of the Phase 5 Task 5 bidirectional-demo cohort (644 patients with 3+
DaT-SPECT scans):

  baseline    : σ = scalar 0.20 (current Phase 5 Task 5 behaviour)
  uniform_vec : σ = np.full(n_visits, 0.20) — must match baseline exactly
  gimin_like  : σ varies by visit from 0.08 (first, tight sensor) to
                0.20 (later, GIMIN-imputed proxy) — exercises the
                heterogeneous-σ path

Outputs:
  outputs/mechanistic_twin/paper10_mech_vs_giman/l1_demo.json
  outputs/mechanistic_twin/paper10_mech_vs_giman/l1_demo_summary.md

Scope: demonstrate the L1 MECHANISM works end-to-end. Does NOT consume
real GIMIN σ values yet; that requires longitudinal GIMIN inference on
~16,700 P3 visits (Phase 5 Task 5 extension, ~2-3h MPS compute).

Usage:
  .venv/bin/python scripts/mechanistic_twin/phase5_bidirectional_demo_l1.py
  .venv/bin/python scripts/mechanistic_twin/phase5_bidirectional_demo_l1.py --n-patients 100
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.mechanistic_twin_v2.forward_model import SBR_SIGMA  # noqa: E402
from giman_pipeline.mechanistic_twin_v2.posterior_store import (  # noqa: E402
    PatientPosterior,
    PosteriorStore,
)
from giman_pipeline.mechanistic_twin_v2.updater import (  # noqa: E402
    update_posterior,
    weighted_predictive_sbr,
)

RNG_SEED = 42
POSTERIOR_STORE = (
    PROJECT_ROOT
    / "outputs"
    / "mechanistic_twin"
    / "paper10_mech_vs_giman"
    / "phase2_posteriors_full_samples.h5"
)
CANONICAL = (
    PROJECT_ROOT / "outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet"
)
OUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "paper10_mech_vs_giman"


def build_sigma_vec(n_visits: int, mode: str) -> np.ndarray:
    """Construct a per-visit σ vector."""
    if mode == "uniform_vec":
        return np.full(n_visits, SBR_SIGMA, dtype=float)
    if mode == "gimin_like":
        # First visit: tight sensor σ (observed DaT); later visits: grow toward
        # the default Phase 2 σ (mimicking GIMIN-imputed visits with wider posterior)
        return np.linspace(0.08, 0.20, n_visits)
    raise ValueError(f"unknown sigma mode: {mode}")


def replay_patient(
    prior: PatientPosterior,
    patno: int,
    visits: pd.DataFrame,
    region: str,
    sigma_mode: str,
) -> dict | None:
    """Run sequential SIR replay for one patient using a given σ strategy."""
    visits = visits.sort_values("t_years").reset_index(drop=True)
    K = len(visits)
    if K < 3:
        return None
    col = f"sbr_{region}_mean"
    t = visits["t_years"].to_numpy()
    sbr = visits[col].to_numpy()
    if np.any(np.isnan(sbr)):
        return None

    sbr_0 = float(sbr[0])
    rng = np.random.default_rng(RNG_SEED + int(patno))

    state = PatientPosterior(
        patno=patno,
        version=prior.version,
        samples=prior.samples.copy(),
        weights=prior.weights.copy(),
        ess=prior.ess,
        log_marg_lik=prior.log_marg_lik,
        param_names=list(prior.param_names),
        source=prior.source,
    )

    t_holdout = np.array([float(t[-1])])
    sbr_holdout = float(sbr[-1])

    predictions = []

    def _summarise(used: int, st: PatientPosterior) -> dict:
        mn, md, lo, hi = weighted_predictive_sbr(st, t_holdout, sbr_0)
        return {
            "scans_used": int(used),
            "pred_mean": float(mn[0]),
            "obs_held_out": sbr_holdout,
            "abs_error_mean": float(abs(mn[0] - sbr_holdout)),
            "covered": bool(lo[0] <= sbr_holdout <= hi[0]),
            "ess": float(st.ess),
        }

    predictions.append(_summarise(0, state))

    for i in range(1, K):
        # Sigma vector for scans [0..i-1] (those being added)
        if sigma_mode == "baseline":
            sigma_arg = None  # uses default scalar
        else:
            sigma_arg = build_sigma_vec(i, sigma_mode)
        state = update_posterior(
            state, t[:i], sbr[:i], sbr_0, rng=rng, sigma=sigma_arg,
        )
        predictions.append(_summarise(i, state))

    return {
        "patno": int(patno),
        "n_scans": int(K),
        "sigma_mode": sigma_mode,
        "predictions": predictions,
    }


def load_patients_with_3plus(canonical: pd.DataFrame, region: str) -> pd.DataFrame:
    """Return long-format scan-level rows for patients with 3+ non-NaN SBR scans."""
    col = f"sbr_{region}_mean"
    df = canonical[["PATNO", "t_years", col]].dropna(subset=[col]).copy()
    counts = df["PATNO"].value_counts()
    keep = counts[counts >= 3].index.tolist()
    return df[df["PATNO"].isin(keep)].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-patients", type=int, default=200,
                        help="Cap on patients for speed (default 200)")
    parser.add_argument("--region", type=str, default="putamen",
                        choices=["caudate", "putamen"])
    parser.add_argument("--output", type=str,
                        default=str(OUT_DIR / "l1_demo.json"))
    args = parser.parse_args()

    print("Loading posterior store + canonical parquet...")
    store = PosteriorStore(POSTERIOR_STORE)
    canonical = pd.read_parquet(CANONICAL)

    scans = load_patients_with_3plus(canonical, args.region)
    patnos = sorted(scans["PATNO"].unique())[: args.n_patients]
    print(f"  Patients with 3+ {args.region} scans: {len(scans['PATNO'].unique())}")
    print(f"  Subset for this run (n-patients={args.n_patients}): {len(patnos)}")

    modes = ["baseline", "uniform_vec", "gimin_like"]
    results: dict[str, list[dict]] = {m: [] for m in modes}

    for pi, patno in enumerate(patnos, start=1):
        try:
            prior = store.load(int(patno))
        except KeyError:
            continue
        visits = scans[scans["PATNO"] == patno]
        if len(visits) < 3:
            continue
        for mode in modes:
            rec = replay_patient(prior, int(patno), visits, args.region, mode)
            if rec is not None:
                results[mode].append(rec)
        if pi % 50 == 0:
            print(f"  Replayed {pi}/{len(patnos)} patients")

    # Aggregate MAE at each scan-count
    summary = {"config": vars(args), "modes": {}}
    max_scans = max(
        max(p["n_scans"] for p in results[m]) for m in modes if results[m]
    )
    for mode in modes:
        by_scan: dict[int, list[float]] = {}
        for p in results[mode]:
            for pred in p["predictions"]:
                by_scan.setdefault(pred["scans_used"], []).append(pred["abs_error_mean"])
        mode_agg = []
        for used in range(max_scans):
            if used in by_scan and by_scan[used]:
                mode_agg.append({
                    "scans_used": used,
                    "n": len(by_scan[used]),
                    "mae": float(np.mean(by_scan[used])),
                    "mae_median": float(np.median(by_scan[used])),
                })
        summary["modes"][mode] = {"n_patients": len(results[mode]), "per_scan": mode_agg}

    # Regression test: baseline vs uniform_vec MAE curves must match to machine precision
    base = summary["modes"]["baseline"]["per_scan"]
    uvec = summary["modes"]["uniform_vec"]["per_scan"]
    max_diff = 0.0
    for b, u in zip(base, uvec):
        if b["scans_used"] == u["scans_used"]:
            max_diff = max(max_diff, abs(b["mae"] - u["mae"]))
    summary["regression_test"] = {
        "baseline_vs_uniform_vec_max_mae_diff": float(max_diff),
        "passed": bool(max_diff < 1e-10),
    }

    # Diff between baseline and gimin_like at final scan
    if base and summary["modes"]["gimin_like"]["per_scan"]:
        b_final = base[-1]["mae"]
        g_final = summary["modes"]["gimin_like"]["per_scan"][-1]["mae"]
        summary["heterogeneous_sigma_effect"] = {
            "final_scan_baseline_mae": b_final,
            "final_scan_gimin_like_mae": g_final,
            "delta": float(g_final - b_final),
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    print("\n=== L1-LITE BIDIRECTIONAL DEMO SUMMARY ===")
    print(f"Patients replayed: {len(results['baseline'])}")
    print(f"\nPer-scan MAE (putamen region):")
    print(f"{'scans':>6s} {'baseline':>10s} {'uniform_vec':>12s} {'gimin_like':>11s}")
    for used in range(max_scans):
        b = next((x for x in base if x["scans_used"] == used), None)
        u = next((x for x in uvec if x["scans_used"] == used), None)
        g = next((x for x in summary["modes"]["gimin_like"]["per_scan"] if x["scans_used"] == used), None)
        if b:
            print(f"{used:>6d} {b['mae']:>10.4f} "
                  f"{u['mae'] if u else float('nan'):>12.4f} "
                  f"{g['mae'] if g else float('nan'):>11.4f}")

    print(f"\nRegression test (baseline vs uniform_vec max MAE diff): "
          f"{max_diff:.3e}  ({'PASS' if max_diff < 1e-10 else 'FAIL'})")

    if "heterogeneous_sigma_effect" in summary:
        he = summary["heterogeneous_sigma_effect"]
        print(f"\nHeterogeneous σ effect at final scan:")
        print(f"  baseline MAE:   {he['final_scan_baseline_mae']:.4f}")
        print(f"  gimin_like MAE: {he['final_scan_gimin_like_mae']:.4f}")
        print(f"  Δ (gimin_like - baseline): {he['delta']:+.4f}")


if __name__ == "__main__":
    main()
