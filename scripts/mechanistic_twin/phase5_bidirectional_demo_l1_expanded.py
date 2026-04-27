#!/usr/bin/env python3
"""L1-full expanded-cohort bidirectional demo.

Extends phase5_bidirectional_demo_l1.py to use the L1 bridge v2 parquet
(outputs/mechanistic_twin/l1_gimin_bridge/gimin_dat_sbr_baseline_v2.parquet)
to identify patients whose baseline DaT-SBR is GIMIN-IMPUTED rather than
observed. For those patients, the bidirectional update uses:

  visit 0 (baseline): sigma = GIMIN temperature-scaled σ (imputed)
  visits 1..K-1:      sigma = sensor σ (observed scans)

This is the core cross-paper integration L1 claim: GIMIN's σ expands the
calibratable cohort of the mechanistic twin beyond the observed-DaT subset.

We report three cohorts:

  1. obs_only    : Phase 5 Task 5 baseline (all scans observed, scalar σ=0.20)
  2. l1_uniform  : Phase 5 Task 5 replay with per-visit σ vector (uniform 0.20)
                   regression test — must match obs_only bit-exact
  3. l1_expanded : patients whose baseline DaT is GIMIN-imputed; baseline
                   uses GIMIN σ, later visits use sensor σ. THIS IS THE NEW
                   COHORT L1 ENABLES.

Outputs:
  outputs/mechanistic_twin/paper10_mech_vs_giman/l1_expanded_demo.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
SENSOR_SIGMA = 0.08  # fine-grained sensor σ for observed DaT (vs 0.20 default)
POSTERIOR_STORE = (
    PROJECT_ROOT
    / "outputs/mechanistic_twin/paper10_mech_vs_giman/phase2_posteriors_full_samples.h5"
)
CANONICAL_SBR = (
    PROJECT_ROOT / "outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet"
)
L1_BRIDGE = (
    PROJECT_ROOT
    / "outputs/mechanistic_twin/l1_gimin_bridge/gimin_dat_sbr_baseline_v2.parquet"
)
OUT_DIR = PROJECT_ROOT / "outputs/mechanistic_twin/paper10_mech_vs_giman"


def load_patients_with_scans(scans_df: pd.DataFrame, region: str, min_scans: int = 2) -> dict[int, pd.DataFrame]:
    """Group scans by PATNO; keep patients with ≥min_scans."""
    col = f"sbr_{region}_mean"
    df = scans_df[["PATNO", "t_years", col]].dropna(subset=[col]).copy()
    counts = df["PATNO"].value_counts()
    keep = counts[counts >= min_scans].index.tolist()
    return {int(p): df[df["PATNO"] == p].sort_values("t_years").reset_index(drop=True)
            for p in keep}


def replay_observed_only(
    prior: PatientPosterior, patno: int, visits: pd.DataFrame, region: str,
) -> dict | None:
    """Classic Phase 5 Task 5 replay (scalar σ, observed scans)."""
    if len(visits) < 2:
        return None
    col = f"sbr_{region}_mean"
    t = visits["t_years"].to_numpy()
    sbr = visits[col].to_numpy()
    sbr_0 = float(sbr[0])
    rng = np.random.default_rng(RNG_SEED + int(patno))

    state = _clone_posterior(prior)
    t_holdout = np.array([float(t[-1])])
    sbr_holdout = float(sbr[-1])

    preds = [_summarise(state, t_holdout, sbr_0, sbr_holdout, 0)]
    for i in range(1, len(visits)):
        state = update_posterior(state, t[:i], sbr[:i], sbr_0, rng=rng)
        preds.append(_summarise(state, t_holdout, sbr_0, sbr_holdout, i))
    return {"patno": int(patno), "mode": "obs_only", "n_scans": len(visits), "preds": preds}


def replay_l1_expanded(
    prior: PatientPosterior,
    patno: int,
    visits: pd.DataFrame,
    region: str,
    baseline_mu: float,
    baseline_sigma: float,
    baseline_is_observed: bool,
) -> dict | None:
    """L1-expanded replay: baseline uses GIMIN σ if imputed, later uses sensor σ.

    For patients whose baseline DaT is imputed (baseline_is_observed=False), we
    insert a synthetic baseline scan at t=0 with the GIMIN posterior mean + σ.
    Subsequent observed scans use the sensor σ.
    """
    if len(visits) < 2:
        return None
    col = f"sbr_{region}_mean"
    t_obs = visits["t_years"].to_numpy()
    sbr_obs = visits[col].to_numpy()

    if baseline_is_observed:
        # Use observed baseline as anchor; all visits at sensor σ
        t_all = t_obs
        sbr_all = sbr_obs
        sigma_all = np.full_like(sbr_obs, SENSOR_SIGMA, dtype=float)
        sbr_0 = float(sbr_obs[0])
    else:
        # Inject imputed baseline at t=0 with GIMIN σ; followed by observed scans
        if t_obs[0] == 0.0:
            # Already has a t=0 scan but we're told it's imputed; skip
            return None
        t_all = np.concatenate([[0.0], t_obs])
        sbr_all = np.concatenate([[baseline_mu], sbr_obs])
        # sqrt(gimin^2 + sensor^2) for baseline; sensor for later scans
        sigma_baseline = float(np.sqrt(baseline_sigma ** 2 + SENSOR_SIGMA ** 2))
        sigma_all = np.concatenate([[sigma_baseline], np.full_like(sbr_obs, SENSOR_SIGMA)])
        sbr_0 = float(baseline_mu)

    if len(sbr_all) < 2:
        return None

    rng = np.random.default_rng(RNG_SEED + int(patno))
    state = _clone_posterior(prior)
    t_holdout = np.array([float(t_all[-1])])
    sbr_holdout = float(sbr_all[-1])

    preds = [_summarise(state, t_holdout, sbr_0, sbr_holdout, 0)]
    for i in range(1, len(t_all)):
        state = update_posterior(
            state, t_all[:i], sbr_all[:i], sbr_0, rng=rng, sigma=sigma_all[:i],
        )
        preds.append(_summarise(state, t_holdout, sbr_0, sbr_holdout, i))
    return {
        "patno": int(patno),
        "mode": "l1_expanded",
        "n_scans": int(len(t_all)),
        "baseline_is_observed": bool(baseline_is_observed),
        "baseline_sigma": float(sigma_all[0]),
        "preds": preds,
    }


def _clone_posterior(prior: PatientPosterior) -> PatientPosterior:
    return PatientPosterior(
        patno=prior.patno,
        version=prior.version,
        samples=prior.samples.copy(),
        weights=prior.weights.copy(),
        ess=prior.ess,
        log_marg_lik=prior.log_marg_lik,
        param_names=list(prior.param_names),
        source=prior.source,
    )


def _summarise(state: PatientPosterior, t_holdout, sbr_0, sbr_holdout, used: int) -> dict:
    mn, md, lo, hi = weighted_predictive_sbr(state, t_holdout, sbr_0)
    return {
        "scans_used": int(used),
        "pred_mean": float(mn[0]),
        "obs_held_out": float(sbr_holdout),
        "abs_error_mean": float(abs(mn[0] - sbr_holdout)),
        "covered": bool(lo[0] <= sbr_holdout <= hi[0]),
        "ess": float(state.ess),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-patients", type=int, default=1000,
                        help="Cap for speed (default 1000)")
    parser.add_argument("--region", type=str, default="putamen",
                        choices=["caudate", "putamen"])
    args = parser.parse_args()

    print("Loading inputs...")
    store = PosteriorStore(POSTERIOR_STORE)
    scans_df = pd.read_parquet(CANONICAL_SBR)
    bridge = pd.read_parquet(L1_BRIDGE)
    print(f"  posterior store: {POSTERIOR_STORE.name}")
    print(f"  longitudinal SBR scans: {scans_df.shape}")
    print(f"  L1 bridge (baseline imputations): {bridge.shape}")

    # Map: PATNO → (baseline_mu, baseline_sigma, baseline_is_observed) for bilateral mean
    bridge_col_mean = f"{args.region.upper()}_MEAN_SBR_imputed_mean"
    bridge_col_std = f"{args.region.upper()}_MEAN_SBR_imputed_std"
    bridge_col_obs = f"{args.region.upper()}_MEAN_SBR_is_observed"
    bridge_info = {
        int(r["PATNO"]): (
            float(r[bridge_col_mean]),
            float(r[bridge_col_std]),
            bool(r[bridge_col_obs]),
        )
        for _, r in bridge.iterrows()
    }

    patients = load_patients_with_scans(scans_df, args.region, min_scans=2)
    print(f"  Patients with ≥2 {args.region} observed scans: {len(patients)}")

    # Find the L1-enabled cohort: patients whose baseline DaT is imputed
    l1_enabled = [p for p in patients if p in bridge_info and not bridge_info[p][2]]
    obs_baseline = [p for p in patients if p in bridge_info and bridge_info[p][2]]
    print(f"  Cohort split: {len(obs_baseline)} with observed baseline, {len(l1_enabled)} with GIMIN-imputed baseline")

    patnos = sorted(patients.keys())[: args.n_patients]
    print(f"  Subset for this run: {len(patnos)}")

    records_obs_only = []
    records_l1 = []
    for pi, patno in enumerate(patnos, start=1):
        try:
            prior = store.load(int(patno))
        except KeyError:
            continue
        visits = patients[patno]
        r_obs = replay_observed_only(prior, patno, visits, args.region)
        if r_obs is not None:
            records_obs_only.append(r_obs)
        if patno in bridge_info:
            bmu, bsig, bobs = bridge_info[patno]
            r_l1 = replay_l1_expanded(prior, patno, visits, args.region, bmu, bsig, bobs)
            if r_l1 is not None:
                records_l1.append(r_l1)
        if pi % 200 == 0:
            print(f"  Replayed {pi}/{len(patnos)}")

    # Aggregate per-scan MAE for each mode + for the L1-enabled subcohort
    def aggregate(records):
        by_scan: dict[int, list[float]] = {}
        for r in records:
            for pred in r["preds"]:
                by_scan.setdefault(pred["scans_used"], []).append(pred["abs_error_mean"])
        return [
            {"scans_used": used, "n": len(by_scan[used]),
             "mae": float(np.mean(by_scan[used])),
             "mae_median": float(np.median(by_scan[used]))}
            for used in sorted(by_scan)
        ]

    # Filter L1 records to the imputed-baseline subcohort
    l1_imputed_only = [r for r in records_l1 if not r.get("baseline_is_observed", True)]

    summary = {
        "config": vars(args),
        "cohort_sizes": {
            "obs_only_total": len(records_obs_only),
            "l1_total": len(records_l1),
            "l1_imputed_baseline": len(l1_imputed_only),
        },
        "modes": {
            "obs_only": aggregate(records_obs_only),
            "l1_all": aggregate(records_l1),
            "l1_imputed_baseline": aggregate(l1_imputed_only),
        },
    }

    out_path = OUT_DIR / "l1_expanded_demo.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    print("\n=== L1 EXPANDED DEMO HEADLINE ===")
    print(f"Cohort totals:")
    print(f"  obs_only:              n={len(records_obs_only)}")
    print(f"  l1_all:                n={len(records_l1)}")
    print(f"  l1_imputed-baseline:   n={len(l1_imputed_only)}  (GIMIN σ enabled these)")

    print("\nPer-scan MAE (obs_only vs l1_imputed-baseline):")
    obs_tbl = {x["scans_used"]: x for x in summary["modes"]["obs_only"]}
    l1_tbl = {x["scans_used"]: x for x in summary["modes"]["l1_imputed_baseline"]}
    all_scans = sorted(set(obs_tbl) | set(l1_tbl))
    print(f"{'scans':>6s}  {'obs_only n':>12s}  {'obs MAE':>10s}  {'l1_imp n':>10s}  {'l1_imp MAE':>12s}")
    for s in all_scans:
        o = obs_tbl.get(s, {})
        l = l1_tbl.get(s, {})
        print(f"{s:>6d}  {o.get('n', 0):>12d}  {o.get('mae', float('nan')):>10.4f}  "
              f"{l.get('n', 0):>10d}  {l.get('mae', float('nan')):>12.4f}")


if __name__ == "__main__":
    main()
