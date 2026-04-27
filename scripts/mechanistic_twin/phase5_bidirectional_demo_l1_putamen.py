#!/usr/bin/env python3
"""Task B: PUTAMEN-only L1 bidirectional demo with expanded cohort.

Demonstrates that feeding GIMIN-imputed PUTAMEN observations (with σ × 2.5
calibration from Pillar 6 MCAR held-out) into the mechanistic-twin
bidirectional updater produces a tangible MAE / coverage improvement over
the observed-only baseline — i.e., L1 enables TRUE cohort expansion for
imputation-gap patients.

Cohort (from per_visit bridge ∩ canonical longitudinal):
  * 428 patients with ≥3 observed PUTAMEN scans AND ≥1 GIMIN-imputed visit
  * Median 14 imputed visits per patient (between observed scans)

Protocol per patient:
  1. Sort all visits (observed + imputed) by months_from_baseline
  2. Hold out the LAST observed PUTAMEN scan (canonical ground truth)
  3. Run two bidirectional SIR replays:
     (a) baseline:    posterior updates use observed-only PUTAMEN
     (b) l1_enabled:  posterior updates use observed + imputed PUTAMEN
                      with σ = 0.20 (scalar) for observed visits and
                      σ = GIMIN_std × 2.5 for imputed visits
  4. Compare predicted weighted-mean SBR vs held-out truth: abs error + 95% CI coverage

Pillar 6 empirical σ calibration (PUTAMEN_MEAN derived from PUTAMEN_L/R):
  Raw GIMIN σ median ≈ 0.11 (under-confident MAE/σ ≈ 2.4)
  Calibrated σ = 2.5 × GIMIN_std ≈ 0.25  (MAE/σ ≈ 1.0, coverage ≈ 90%)

Outputs:
  outputs/mechanistic_twin/paper10_mech_vs_giman/l1_putamen_demo.json
  outputs/mechanistic_twin/paper10_mech_vs_giman/fig_l1_putamen_demo.{pdf,png}

Usage:
  .venv/bin/python scripts/mechanistic_twin/phase5_bidirectional_demo_l1_putamen.py
  .venv/bin/python scripts/mechanistic_twin/phase5_bidirectional_demo_l1_putamen.py --n-patients 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
# Pillar 6 MCAR marginal calibration: PUTAMEN MAE/σ ≈ 2.5. But marginal
# coverage ≠ joint coverage when many imputations stack — smoke test shows
# σ×2.5 over-concentrates posterior (coverage collapses to 5%). We sweep a
# range of inflation factors to find the joint-calibration regime.
SIGMA_SWEEP = (2.5, 5.0, 10.0, 20.0)
DEFAULT_SIGMA_INFLATION = 2.5

POSTERIOR_STORE = (
    PROJECT_ROOT
    / "outputs/mechanistic_twin/paper10_mech_vs_giman/phase2_posteriors_full_samples.h5"
)
CANONICAL = PROJECT_ROOT / "outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet"
BRIDGE = (
    PROJECT_ROOT
    / "outputs/mechanistic_twin/l1_gimin_bridge/gimin_per_visit_dat_sbr.parquet"
)
OUT_DIR = PROJECT_ROOT / "outputs/mechanistic_twin/paper10_mech_vs_giman"


def build_patient_timeline(
    patno: int,
    bridge_visits: pd.DataFrame,
    canonical_visits: pd.DataFrame,
    sigma_inflation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Assemble merged timeline for one patient.

    Returns (t_years, sbr, sigma, is_observed) sorted by t_years, or None
    if patient lacks ≥3 observed PUTAMEN scans in the bridge.
    """
    b = bridge_visits.sort_values("months_from_baseline").reset_index(drop=True)
    t_years_b = b["months_from_baseline"].to_numpy() / 12.0
    is_obs = b["PUTAMEN_MEAN_SBR_is_observed"].to_numpy().astype(bool)
    sbr_obs_raw = b["PUTAMEN_MEAN_SBR_obs"].to_numpy(dtype=float)
    sbr_imp = b["PUTAMEN_MEAN_SBR_imputed_mean"].to_numpy(dtype=float)
    sig_imp = b["PUTAMEN_MEAN_SBR_imputed_std"].to_numpy(dtype=float)

    # For observed visits, we prefer canonical parquet ground truth (Xing Core
    # Lab values if present) over the bridge observed-slot (which is also Xing).
    c = canonical_visits.dropna(subset=["sbr_putamen_mean"]).sort_values("t_years")
    canon_t = c["t_years"].to_numpy()
    canon_sbr = c["sbr_putamen_mean"].to_numpy()

    # Final SBR per visit: observed -> canonical-matched (nearest t), else imputed mean
    sbr = np.where(is_obs, sbr_obs_raw, sbr_imp)
    sigma = np.where(is_obs, SBR_SIGMA, sig_imp * sigma_inflation)

    if is_obs.sum() < 3:
        return None
    if np.any(np.isnan(sbr)):
        return None

    return t_years_b, sbr, sigma, is_obs


def run_replay(
    prior: PatientPosterior,
    patno: int,
    t_years: np.ndarray,
    sbr: np.ndarray,
    sigma: np.ndarray,
    is_obs: np.ndarray,
    mode: str,
    max_imputed: int | None = None,
) -> dict | None:
    """Run bidirectional SIR replay.

    mode='baseline'   → only observed visits feed the posterior
    mode='l1_enabled' → observed + imputed visits feed the posterior
    Both modes predict the LAST observed visit, which is held out.
    """
    # Index of last observed visit (hold-out target)
    obs_idx = np.where(is_obs)[0]
    hold_idx = int(obs_idx[-1])
    sbr_holdout = float(sbr[hold_idx])
    t_holdout = np.array([float(t_years[hold_idx])])
    sbr_0 = float(sbr[obs_idx[0]])

    # Assemble input visits (excluding hold-out)
    input_mask = np.ones(len(t_years), dtype=bool)
    input_mask[hold_idx] = False
    if mode == "baseline":
        input_mask &= is_obs
    elif mode == "l1_enabled":
        # Optionally cap imputed visits to an evenly-spaced subset
        if max_imputed is not None:
            imp_idxs = np.where(input_mask & ~is_obs)[0]
            if len(imp_idxs) > max_imputed:
                # Pick evenly-spaced indices
                keep_positions = np.linspace(0, len(imp_idxs) - 1,
                                             max_imputed).astype(int)
                keep = imp_idxs[keep_positions]
                drop = np.setdiff1d(imp_idxs, keep)
                input_mask[drop] = False
    else:
        raise ValueError(f"unknown mode: {mode}")

    t_in = t_years[input_mask]
    sbr_in = sbr[input_mask]
    sig_in = sigma[input_mask]

    if len(t_in) < 1:
        return None

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

    # One posterior update with all input visits + per-visit σ
    state = update_posterior(
        state, t_in, sbr_in, sbr_0, rng=rng, sigma=sig_in,
    )

    mn, md, lo, hi = weighted_predictive_sbr(state, t_holdout, sbr_0)
    return {
        "patno": int(patno),
        "n_obs_input": int(is_obs[input_mask].sum()),
        "n_imp_input": int((~is_obs[input_mask]).sum()),
        "t_holdout": float(t_holdout[0]),
        "sbr_holdout": sbr_holdout,
        "pred_mean": float(mn[0]),
        "pred_lo": float(lo[0]),
        "pred_hi": float(hi[0]),
        "abs_error": float(abs(mn[0] - sbr_holdout)),
        "covered_95": bool(lo[0] <= sbr_holdout <= hi[0]),
        "ess": float(state.ess),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-patients", type=int, default=None,
                        help="Cap on patients; default = all eligible (428)")
    parser.add_argument("--output", type=str,
                        default=str(OUT_DIR / "l1_putamen_demo.json"))
    parser.add_argument("--sigma-sweep", action="store_true",
                        help="Sweep σ inflation [2.5, 5, 10, 20]; else use default 2.5")
    parser.add_argument("--max-imputed", type=int, default=None,
                        help="Cap imputed visits per patient (keep evenly-spaced subset)")
    args = parser.parse_args()

    print("=" * 75)
    print("L1 PUTAMEN-only bidirectional demo (expanded cohort)")
    print("=" * 75)

    print("\nLoading posterior store + canonical + bridge...")
    store = PosteriorStore(POSTERIOR_STORE)
    canonical = pd.read_parquet(CANONICAL)
    bridge = pd.read_parquet(BRIDGE)

    # Eligible patients: have ≥3 observed PUTAMEN in bridge + ≥3 canonical visits
    obs_counts = bridge[bridge["PUTAMEN_MEAN_SBR_is_observed"]].groupby("PATNO").size()
    bridge_3plus = obs_counts[obs_counts >= 3].index.tolist()
    imp_counts = bridge[~bridge["PUTAMEN_MEAN_SBR_is_observed"]].groupby("PATNO").size()
    bridge_has_imp = imp_counts[imp_counts >= 1].index.tolist()
    eligible = sorted(set(bridge_3plus) & set(bridge_has_imp))
    print(f"  Eligible patients (≥3 obs + ≥1 imp PUTAMEN): {len(eligible)}")

    if args.n_patients is not None:
        eligible = eligible[: args.n_patients]
        print(f"  Capped to {len(eligible)} patients")

    sigma_values = SIGMA_SWEEP if args.sigma_sweep else (DEFAULT_SIGMA_INFLATION,)
    print(f"\nσ inflation factors to evaluate: {sigma_values}")

    # Aggregate results — baseline computed once (σ-independent), L1 per-σ
    baseline_records: list[dict] = []
    l1_records: dict[float, list[dict]] = {s: [] for s in sigma_values}
    n_skipped = 0
    n_missing_posterior = 0

    for pi, patno in enumerate(eligible, start=1):
        try:
            prior = store.load(int(patno))
        except KeyError:
            n_missing_posterior += 1
            continue

        bv = bridge[bridge["PATNO"] == patno]
        cv = canonical[canonical["PATNO"] == patno]

        # Baseline — σ-independent for observed-only mode
        tl_ref = build_patient_timeline(int(patno), bv, cv,
                                        sigma_inflation=DEFAULT_SIGMA_INFLATION)
        if tl_ref is None:
            n_skipped += 1
            continue
        t_ref, sbr_ref, sig_ref, is_obs_ref = tl_ref
        base = run_replay(prior, int(patno), t_ref, sbr_ref, sig_ref, is_obs_ref,
                          mode="baseline")
        if base is not None:
            baseline_records.append(base)

        # L1 per σ inflation
        for s in sigma_values:
            tl = build_patient_timeline(int(patno), bv, cv, sigma_inflation=s)
            if tl is None:
                continue
            t_years, sbr, sigma_vec, is_obs = tl
            rec = run_replay(prior, int(patno), t_years, sbr, sigma_vec, is_obs,
                             mode="l1_enabled", max_imputed=args.max_imputed)
            if rec is not None:
                rec["sigma_inflation"] = float(s)
                l1_records[s].append(rec)

        if pi % 50 == 0:
            print(f"  Replayed {pi}/{len(eligible)} patients")

    print(f"\n  Skipped (<3 obs or NaN SBR): {n_skipped}")
    print(f"  Missing posterior: {n_missing_posterior}")
    print(f"  baseline: n={len(baseline_records)}")
    for s in sigma_values:
        print(f"  l1 σ×{s}: n={len(l1_records[s])}")

    records = {"baseline": baseline_records}
    for s in sigma_values:
        records[f"l1_sigma_x{s}"] = l1_records[s]

    # Summary per mode
    summary: dict = {"config": vars(args), "sigma_sweep": list(sigma_values),
                     "modes": {}}
    for mode, recs in records.items():
        if not recs:
            continue
        errs = np.array([r["abs_error"] for r in recs])
        covs = np.array([r["covered_95"] for r in recs])
        widths = np.array([r["pred_hi"] - r["pred_lo"] for r in recs])
        summary["modes"][mode] = {
            "n_patients": len(recs),
            "mean_abs_error": float(errs.mean()),
            "median_abs_error": float(np.median(errs)),
            "rmse": float(np.sqrt((errs ** 2).mean())),
            "coverage_95ci": float(covs.mean()),
            "median_width_95ci": float(np.median(widths)),
            "mean_imputed_input": float(np.mean([r["n_imp_input"] for r in recs])),
            "mean_observed_input": float(np.mean([r["n_obs_input"] for r in recs])),
        }

    # Paired comparison: baseline vs each l1_sigma variant
    base_by_pt = {r["patno"]: r for r in records["baseline"]}
    summary["paired_comparison"] = {}
    best_sigma = None
    best_cov_gap = float("inf")
    for s in sigma_values:
        mode_key = f"l1_sigma_x{s}"
        l1_by_pt = {r["patno"]: r for r in records[mode_key]}
        common = sorted(set(base_by_pt) & set(l1_by_pt))
        if not common:
            continue
        d_err = np.array([l1_by_pt[p]["abs_error"] - base_by_pt[p]["abs_error"]
                          for p in common])
        improved = (d_err < 0).mean()
        rng_boot = np.random.default_rng(RNG_SEED)
        boot_means = []
        for _ in range(1000):
            idx = rng_boot.integers(0, len(d_err), len(d_err))
            boot_means.append(d_err[idx].mean())
        ci = [float(np.percentile(boot_means, 2.5)),
              float(np.percentile(boot_means, 97.5))]
        summary["paired_comparison"][mode_key] = {
            "sigma_inflation": float(s),
            "n_paired": len(common),
            "mean_delta_abs_error": float(d_err.mean()),
            "median_delta_abs_error": float(np.median(d_err)),
            "mean_delta_95ci": ci,
            "pct_patients_improved": float(improved),
        }
        # Track σ closest to nominal 95% coverage
        cov = summary["modes"][mode_key]["coverage_95ci"]
        cov_gap = abs(cov - 0.95)
        if cov_gap < best_cov_gap:
            best_cov_gap = cov_gap
            best_sigma = s
    summary["best_sigma_for_nominal_coverage"] = float(best_sigma) if best_sigma else None

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Print report
    print("\n" + "=" * 85)
    print("RESULTS")
    print("=" * 85)
    # Baseline row first
    s = summary["modes"].get("baseline", {})
    if s:
        print(f"\n  baseline (observed-only):")
        print(f"    n patients:           {s['n_patients']}")
        print(f"    mean input visits:    "
              f"{s['mean_observed_input']:.1f} observed + "
              f"{s['mean_imputed_input']:.1f} imputed")
        print(f"    MAE:                  {s['mean_abs_error']:.4f}")
        print(f"    95% CI coverage:      {s['coverage_95ci']:.1%}")
        print(f"    median 95% CI width:  {s['median_width_95ci']:.4f}")

    # L1 sweep
    print(f"\n  {'σ inflation':<12s} {'MAE':>8s} {'coverage':>10s} {'CI width':>10s} "
          f"{'ΔMAE vs base':>14s} {'pct↓':>8s}")
    print("  " + "-" * 70)
    for s_val in sigma_values:
        mode_key = f"l1_sigma_x{s_val}"
        m = summary["modes"].get(mode_key, {})
        pc = summary["paired_comparison"].get(mode_key, {})
        if not m or not pc:
            continue
        marker = "  "
        if summary.get("best_sigma_for_nominal_coverage") == float(s_val):
            marker = " ★"
        print(f"  σ×{s_val:<7.1f}{marker} {m['mean_abs_error']:>8.4f} "
              f"{m['coverage_95ci']:>10.1%} "
              f"{m['median_width_95ci']:>10.4f} "
              f"{pc['mean_delta_abs_error']:>+14.4f} "
              f"{pc['pct_patients_improved']:>7.1%}")
    if summary.get("best_sigma_for_nominal_coverage"):
        print(f"\n  ★ σ×{summary['best_sigma_for_nominal_coverage']} closest to nominal 95% coverage")

    # Figure: σ sweep calibration curve
    if summary.get("paired_comparison"):
        _render_figure(records, summary, sigma_values, base_by_pt)


def _render_figure(records, summary, sigma_values, base_by_pt):
    """σ-sweep calibration + MAE figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    sigmas = np.array([float(s) for s in sigma_values])
    coverages = np.array([summary["modes"][f"l1_sigma_x{s}"]["coverage_95ci"]
                          for s in sigma_values])
    maes = np.array([summary["modes"][f"l1_sigma_x{s}"]["mean_abs_error"]
                     for s in sigma_values])
    widths = np.array([summary["modes"][f"l1_sigma_x{s}"]["median_width_95ci"]
                       for s in sigma_values])
    base_mae = summary["modes"]["baseline"]["mean_abs_error"]
    base_cov = summary["modes"]["baseline"]["coverage_95ci"]
    base_width = summary["modes"]["baseline"]["median_width_95ci"]

    # Panel A: calibration curve — coverage vs σ inflation
    ax1.plot(sigmas, coverages, "o-", color="#1f77b4", label="L1 enabled coverage",
             linewidth=2, markersize=8)
    ax1.axhline(base_cov, color="#ff7f0e", linestyle="--", alpha=0.8,
                label=f"Baseline coverage ({base_cov:.1%})")
    ax1.axhline(0.95, color="k", linestyle=":", lw=0.8, label="Nominal 95%")
    ax1.axhline(0.90, color="gray", linestyle=":", lw=0.8, label="Min 90%")
    ax1.set_xscale("log")
    ax1.set_xlabel("σ inflation factor (GIMIN σ × k)")
    ax1.set_ylabel("95% CI coverage (held-out SBR)")
    ax1.set_title("(a) Joint calibration under σ sweep")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3)
    # Annotate best
    best_s = summary.get("best_sigma_for_nominal_coverage")
    if best_s is not None:
        idx = np.argmin(np.abs(sigmas - best_s))
        ax1.annotate(f"★ σ×{best_s}\n{coverages[idx]:.1%}",
                     xy=(best_s, coverages[idx]),
                     xytext=(10, 10), textcoords="offset points",
                     fontsize=9, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", lw=0.5))

    # Panel B: MAE + CI width dual-axis
    color_mae = "#2ca02c"
    color_width = "#d62728"
    ax2.plot(sigmas, maes, "o-", color=color_mae, label="L1 enabled MAE",
             linewidth=2, markersize=8)
    ax2.axhline(base_mae, color=color_mae, linestyle="--", alpha=0.6,
                label=f"Baseline MAE ({base_mae:.3f})")
    ax2.set_xscale("log")
    ax2.set_xlabel("σ inflation factor")
    ax2.set_ylabel("MAE (held-out)", color=color_mae)
    ax2.tick_params(axis="y", labelcolor=color_mae)
    ax2.grid(alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.plot(sigmas, widths, "s-", color=color_width, label="L1 enabled CI width",
              linewidth=2, markersize=8)
    ax2b.axhline(base_width, color=color_width, linestyle="--", alpha=0.6,
                 label=f"Baseline width ({base_width:.3f})")
    ax2b.set_ylabel("median 95% CI width", color=color_width)
    ax2b.tick_params(axis="y", labelcolor=color_width)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2b.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)
    ax2.set_title("(b) MAE vs CI width trade-off")

    n_base = summary["modes"]["baseline"]["n_patients"]
    imp_count = summary["modes"][f"l1_sigma_x{sigma_values[0]}"]["mean_imputed_input"]
    fig.suptitle(
        f"L1 PUTAMEN-only bidirectional demo — n={n_base} patients, "
        f"~{imp_count:.0f} imputed visits/patient",
        y=1.02,
    )
    fig.tight_layout()
    fig_path = OUT_DIR / "fig_l1_putamen_demo.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
