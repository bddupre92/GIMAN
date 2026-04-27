#!/usr/bin/env python3
"""Phase 5 Task 5 — Bidirectional update demo (THE TWIN PROOF).

Shows that the mechanistic patient-specific model can assimilate new DaT-SPECT
observations as they arrive (NASEM bidirectional-flow criterion) by
re-weighting a patient's posterior via SIR rather than re-running the full
Julia IS calibration.

Design (Simulated Prospective Replay):
    1. Draw N=50,000 posterior samples from Phase 2 population priors (shared
       across all patients).
    2. For each patient with K>=3 longitudinal DaT-SPECT scans:
       - v1: reweight using scan 1 only
       - v2: reweight using scans 1-2
       - v_{K-1}: reweight using scans 1..K-1
       Each step predicts the held-out LAST scan (K) and we record MAE,
       residual, 90% CI coverage.
    3. Report: does MAE decrease monotonically with update count? Does
       coverage stay at >=0.90?

Output: outputs/mechanistic_twin/paper10_mech_vs_giman/bidirectional_demo.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.mechanistic_twin_v2.forward_model import (
    T_TOX_CONST,
)
from giman_pipeline.mechanistic_twin_v2.posterior_store import PatientPosterior
from giman_pipeline.mechanistic_twin_v2.updater import (
    update_posterior,
    weighted_predictive_sbr,
)

SBR_LONGITUDINAL = (
    PROJECT_ROOT / "outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet"
)
OUTPUT_JSON = (
    PROJECT_ROOT
    / "outputs/mechanistic_twin/paper10_mech_vs_giman/bidirectional_demo.json"
)

# Phase 2 priors (match step_2_6v4 exactly)
PRIOR_KN_LOGMEAN = np.log(1e-4)
PRIOR_KN_LOGSD = 1.5
PRIOR_ALPHA_LOGMEAN = np.log(1.8e-5)
PRIOR_ALPHA_LOGSD = 2.0
N_PRIOR = 50_000
RNG_SEED = 202604131


@dataclass
class PatientReplayRecord:
    patno: int
    n_scans: int
    t_years: list[float]
    sbr_observed: list[float]
    sbr_0: float
    predictions: list[dict[str, Any]] = field(default_factory=list)


def draw_population_prior(n: int, seed: int) -> PatientPosterior:
    rng = np.random.default_rng(seed)
    k_n = rng.lognormal(PRIOR_KN_LOGMEAN, PRIOR_KN_LOGSD, size=n)
    alpha = rng.lognormal(PRIOR_ALPHA_LOGMEAN, PRIOR_ALPHA_LOGSD, size=n)
    T_tox = alpha * k_n * T_TOX_CONST
    samples = np.stack([k_n, alpha, T_tox], axis=1)
    return PatientPosterior(
        patno=-1,
        version=0,
        samples=samples,
        weights=np.full(n, 1.0 / n),
        ess=float(n),
        log_marg_lik=0.0,
        param_names=["k_n", "alpha_tox", "T_tox"],
        source="population_prior",
    )


def replay_patient(
    prior: PatientPosterior,
    patno: int,
    visits: pd.DataFrame,
    region: str = "putamen",
) -> PatientReplayRecord | None:
    """Run sequential SIR replay on one patient's longitudinal SBR scans."""
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

    rec = PatientReplayRecord(
        patno=int(patno),
        n_scans=int(K),
        t_years=[float(x) for x in t],
        sbr_observed=[float(x) for x in sbr],
        sbr_0=sbr_0,
    )

    # Held-out scan = last visit (K-th)
    t_holdout = np.array([float(t[-1])])
    sbr_holdout = float(sbr[-1])

    def _summarise(ver: int, used: int, st: PatientPosterior) -> dict:
        mn, md, lo, hi = weighted_predictive_sbr(st, t_holdout, sbr_0)
        return {
            "version": int(ver),
            "scans_used": int(used),
            "pred_mean": float(mn[0]),
            "pred_median": float(md[0]),
            "pred_lo90": float(lo[0]),
            "pred_hi90": float(hi[0]),
            "obs_held_out": sbr_holdout,
            "abs_error_mean": float(abs(mn[0] - sbr_holdout)),
            "abs_error_median": float(abs(md[0] - sbr_holdout)),
            "covered": bool(lo[0] <= sbr_holdout <= hi[0]),
            "ess": st.ess,
        }

    rec.predictions.append(_summarise(0, 0, state))
    for i in range(1, K):
        state = update_posterior(state, t[:i], sbr[:i], sbr_0, rng=rng)
        rec.predictions.append(_summarise(state.version, i, state))

    return rec


def summarise(records: list[PatientReplayRecord]) -> dict[str, Any]:
    """Aggregate MAE, coverage, ESS by number of scans used."""
    rows = []
    for rec in records:
        for p in rec.predictions:
            rows.append(
                {
                    "patno": rec.patno,
                    "scans_used": p["scans_used"],
                    "abs_error_mean": p["abs_error_mean"],
                    "abs_error_median": p["abs_error_median"],
                    "covered": p["covered"],
                    "ess": p["ess"],
                }
            )
    df = pd.DataFrame(rows)
    by_sc = (
        df.groupby("scans_used")
        .agg(
            n=("patno", "size"),
            mae_from_mean=("abs_error_mean", "mean"),
            mae_from_median=("abs_error_median", "mean"),
            median_ae_from_mean=("abs_error_mean", "median"),
            median_ae_from_median=("abs_error_median", "median"),
            coverage=("covered", "mean"),
            ess_median=("ess", "median"),
        )
        .reset_index()
    )
    return {
        "per_scan_count_summary": by_sc.to_dict(orient="records"),
        "n_patients": int(df["patno"].nunique()),
        "n_records": int(len(df)),
    }


def main() -> None:
    print("[Phase 5 Task 5] Loading longitudinal SBR data...")
    df = pd.read_parquet(SBR_LONGITUDINAL)
    print(f"  visits={len(df)} patients={df['PATNO'].nunique()}")

    eligible = df.groupby("PATNO").filter(lambda g: len(g) >= 3)
    elig_pts = eligible["PATNO"].unique()
    print(f"  eligible (>=3 scans): {len(elig_pts)} patients")

    print(f"[Phase 5 Task 5] Drawing population prior N={N_PRIOR} samples...")
    prior = draw_population_prior(N_PRIOR, RNG_SEED)

    records: list[PatientReplayRecord] = []
    print(f"[Phase 5 Task 5] Running SIR replay on {len(elig_pts)} patients...")
    for i, patno in enumerate(elig_pts):
        visits = eligible[eligible["PATNO"] == patno]
        rec = replay_patient(prior, int(patno), visits, region="putamen")
        if rec is not None:
            records.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(elig_pts)}")

    print(f"[Phase 5 Task 5] Completed {len(records)} patients")
    summary = summarise(records)
    summary["config"] = {
        "n_prior": N_PRIOR,
        "seed": RNG_SEED,
        "prior_kn_logmean": PRIOR_KN_LOGMEAN,
        "prior_kn_logsd": PRIOR_KN_LOGSD,
        "prior_alpha_logmean": PRIOR_ALPHA_LOGMEAN,
        "prior_alpha_logsd": PRIOR_ALPHA_LOGSD,
        "region": "putamen",
        "held_out_scan": "last (K-th)",
    }
    summary["records"] = [asdict(r) for r in records]

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Phase 5 Task 5] Wrote {OUTPUT_JSON}")

    # Print per-scan-count summary
    print("\nPer-scan-count summary:")
    print(pd.DataFrame(summary["per_scan_count_summary"]).to_string(index=False))


if __name__ == "__main__":
    main()
