"""WS-P3-14b Mondrian per-stratum conformal recalibration on carrier strata.

The marginal IPCW conformal calibration in
``run_subgroup_with_lrrk2_gba_fix.py`` produces a single quantile cube per fold
that is applied to all patients regardless of carrier status. The §S-3 H3
verdict (PARTIAL) flagged that conditional coverage in the carrier strata
deviates 3-9 pp from nominal in 4/5 folds — exactly the Mondrian regime that
Bostr"om et al. (2025) describe.

Mondrian conformal prediction (this script): for each carrier stratum,
recalibrate ``CauseSpecificConformal`` on the calibration set restricted to
that stratum. Apply the stratum-specific quantiles to the eval-set patients of
that stratum. Report per-stratum coverage and mean band width.

Reproducibility: identical fold splits and identical 50/50 cal/eval permutation
seed (``args.seed + fold_idx``) as the marginal runner, so the comparison is
apples-to-apples (same eval patients, different calibration scope).

Outputs:

  ``outputs/paper4/subgroup_carriers/conditional_coverage_carriers_mondrian.json``
      Per-(model, fold, CL, stratum) Mondrian coverage + band width.

  ``outputs/paper4/subgroup_carriers/mondrian_summary.json``
      Aggregate per-stratum mean ± std coverage + band width, plus the
      Mondrian-vs-marginal delta computed against the existing
      ``conditional_coverage_carriers.json`` (marginal baseline).

Run::

    .venv/bin/python scripts/paper3plus4/run_mondrian_carriers.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from giman_pipeline.data.db import read_sql  # noqa: E402
from giman_pipeline.paper3.dynamic_deephit import (  # noqa: E402
    TIME_BIN_ENDS,
    build_patient_arrays,
    extract_episodes,
)
from giman_pipeline.paper4.conformal_survival import CauseSpecificConformal  # noqa: E402
from giman_pipeline.paper4.subgroup import (  # noqa: E402
    CARRIER_STRATA,
    assign_carrier_subgroups,
)
from scripts.paper3plus4.run_subgroup_with_lrrk2_gba_fix import (  # noqa: E402
    OUTPUT_DIR,
    get_test_predictions,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("mondrian_carriers")

FEATURES_CSV = PROJECT_ROOT / "data" / "07_paper3_features" / "longitudinal_features.csv"
SEED = 42

# Mondrian per-stratum minimum calibration N. Below this, the empirical
# stratum quantile is too noisy to be informative; we report N/A and fall
# back to the marginal quantile in the output JSON for transparency.
MIN_CAL_PER_STRATUM = 30


def _coverage_and_width(
    cif: np.ndarray,
    bands: np.ndarray,
    durations: np.ndarray,
    event_idxs: np.ndarray,
    censored: np.ndarray,
    indices: list[int],
) -> tuple[float, float, int]:
    """Compute marginal coverage + mean band width over a subset of patients.

    Returns ``(coverage, mean_width, n_evaluated_tuples)``. NaN coverage
    if no tuples were evaluable (empty stratum).
    """
    if not indices:
        return float("nan"), float("nan"), 0

    time_bins_months = np.array(TIME_BIN_ENDS, dtype=float)
    n_causes = cif.shape[1]
    n_tbins = len(time_bins_months)

    covered = 0
    total = 0
    width_sum = 0.0
    width_n = 0

    for i in indices:
        for k in range(n_causes):
            for t_idx in range(n_tbins):
                t_months = time_bins_months[t_idx]

                if censored[i] and durations[i] < t_months:
                    continue

                if (
                    not censored[i]
                    and event_idxs[i] == k
                    and durations[i] <= t_months
                ):
                    cif_obs = 1.0
                elif (
                    not censored[i]
                    and event_idxs[i] != k
                    and durations[i] <= t_months
                ):
                    cif_obs = 0.0
                else:
                    cif_obs = 0.0

                lo = bands[i, k, t_idx, 0]
                hi = bands[i, k, t_idx, 1]
                total += 1
                if lo <= cif_obs <= hi:
                    covered += 1
                width_sum += hi - lo
                width_n += 1

    coverage = covered / max(total, 1)
    mean_width = width_sum / max(width_n, 1)
    return coverage, mean_width, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    log.info("WS-P3-14b Mondrian per-stratum conformal recalibration")
    log.info("  OUTPUT_DIR=%s, MIN_CAL_PER_STRATUM=%d, seed=%d",
             OUTPUT_DIR, MIN_CAL_PER_STRATUM, args.seed)

    log.info("Loading carrier flags from features.paper1_features_with_targets ...")
    carrier_df = read_sql(
        "SELECT patno, lrrk2_carrier, gba_carrier, apoe_e4_carrier, sex, age_at_baseline "
        "FROM features.paper1_features_with_targets"
    )
    carrier_df = carrier_df.rename(columns={"patno": "PATNO"})
    carrier_df["patno"] = carrier_df["PATNO"]

    log.info("Loading Paper 3 longitudinal features ...")
    features_df = pd.read_csv(FEATURES_CSV, low_memory=False)
    patient_arrays, _ = build_patient_arrays(features_df)
    episodes = extract_episodes(features_df, verbose=False)
    log.info("  %d episodes across %d patnos",
             len(episodes), features_df["PATNO"].nunique())

    # Per (fold, model, CL, stratum) Mondrian recalibration
    mondrian_records: list[dict] = []
    log.info("Running Mondrian recalibration (5 folds × 2 models × 2 CL × 4 strata) ...")

    for fold_idx in range(5):
        for model_type, model_name in (("deephit", "DeepHit"), ("graph_dt", "Graph-DT")):
            preds, patnos, test_eps = get_test_predictions(
                model_type, fold_idx, episodes, patient_arrays
            )
            cif = preds["cif"].numpy()
            durations = np.array([ep.duration_months for ep in test_eps])
            event_idxs = preds["event_idxs"].numpy()
            censored = preds["censored"].numpy()
            n = len(durations)

            # Carrier assignment for ALL fold patnos (cal + eval)
            full_assignments = assign_carrier_subgroups(patnos, carrier_df)

            for cl in (0.90, 0.95):
                rng = np.random.RandomState(args.seed + fold_idx)
                idx = rng.permutation(n)
                n_cal = n // 2
                cal_idx_global = idx[:n_cal]
                eval_idx_global = idx[n_cal:]

                # Same marginal calibration as the existing runner — used as the
                # fallback for any stratum below the MIN_CAL_PER_STRATUM floor.
                csc_marginal = CauseSpecificConformal(confidence_level=cl)
                csc_marginal.calibrate(
                    cif[cal_idx_global],
                    durations[cal_idx_global],
                    event_idxs[cal_idx_global],
                    censored[cal_idx_global],
                )
                bands_marginal = csc_marginal.predict_bands(cif[eval_idx_global])

                # Pre-fetch eval-set carrier assignments
                eval_patnos = [patnos[i] for i in eval_idx_global]
                eval_assignments = assign_carrier_subgroups(eval_patnos, carrier_df)

                for stratum in CARRIER_STRATA:
                    # Cal-set indices restricted to this stratum (within global cif)
                    cal_stratum_global = [
                        gi for gi in cal_idx_global
                        if full_assignments.get(patnos[gi]) == stratum
                    ]
                    n_cal_stratum = len(cal_stratum_global)

                    # Eval-set indices restricted to this stratum (within eval slice)
                    eval_stratum_local = [
                        local_i for local_i, gi in enumerate(eval_idx_global)
                        if full_assignments.get(patnos[gi]) == stratum
                    ]
                    n_eval_stratum = len(eval_stratum_local)

                    record: dict = dict(
                        model=model_name,
                        fold_idx=fold_idx,
                        confidence_level=cl,
                        stratum=stratum,
                        n_cal=n_cal_stratum,
                        n_eval=n_eval_stratum,
                    )

                    if n_eval_stratum == 0:
                        record.update(
                            mondrian_used=False,
                            reason="no eval patients in stratum",
                            coverage_marginal=float("nan"),
                            coverage_mondrian=float("nan"),
                            width_marginal=float("nan"),
                            width_mondrian=float("nan"),
                        )
                        mondrian_records.append(record)
                        continue

                    # Marginal-quantile coverage on this stratum's eval patients
                    cov_marg, w_marg, _ = _coverage_and_width(
                        cif[eval_idx_global],
                        bands_marginal,
                        durations[eval_idx_global],
                        event_idxs[eval_idx_global],
                        censored[eval_idx_global],
                        eval_stratum_local,
                    )

                    if n_cal_stratum < MIN_CAL_PER_STRATUM:
                        # Honest fallback — Mondrian quantile too noisy
                        record.update(
                            mondrian_used=False,
                            reason=(
                                f"n_cal={n_cal_stratum} below "
                                f"MIN_CAL_PER_STRATUM={MIN_CAL_PER_STRATUM}"
                            ),
                            coverage_marginal=cov_marg,
                            coverage_mondrian=float("nan"),
                            width_marginal=w_marg,
                            width_mondrian=float("nan"),
                        )
                        mondrian_records.append(record)
                        continue

                    # Mondrian: stratum-specific calibration
                    csc_mondrian = CauseSpecificConformal(confidence_level=cl)
                    csc_mondrian.calibrate(
                        cif[cal_stratum_global],
                        durations[cal_stratum_global],
                        event_idxs[cal_stratum_global],
                        censored[cal_stratum_global],
                    )
                    bands_mondrian = csc_mondrian.predict_bands(cif[eval_idx_global])

                    cov_mond, w_mond, _ = _coverage_and_width(
                        cif[eval_idx_global],
                        bands_mondrian,
                        durations[eval_idx_global],
                        event_idxs[eval_idx_global],
                        censored[eval_idx_global],
                        eval_stratum_local,
                    )
                    record.update(
                        mondrian_used=True,
                        reason="ok",
                        coverage_marginal=cov_marg,
                        coverage_mondrian=cov_mond,
                        width_marginal=w_marg,
                        width_mondrian=w_mond,
                    )
                    mondrian_records.append(record)

            log.info(
                "  fold=%d model=%s — %d strata × 2 CL processed",
                fold_idx,
                model_name,
                len(CARRIER_STRATA),
            )

    # Persist per-record JSON
    out_records = OUTPUT_DIR / "conditional_coverage_carriers_mondrian.json"
    with open(out_records, "w") as f:
        json.dump(
            dict(
                method="Mondrian per-stratum CauseSpecificConformal recalibration",
                citation="Boström et al. 2025; Vovk et al. 2022",
                min_cal_per_stratum=MIN_CAL_PER_STRATUM,
                random_state=args.seed,
                n_records=len(mondrian_records),
                records=mondrian_records,
            ),
            f,
            indent=2,
        )
    log.info("Wrote %s (%d records)", out_records, len(mondrian_records))

    # Aggregate: per-(model, CL, stratum) mean ± std across folds
    log.info("Aggregating per-(model, CL, stratum) across 5 folds ...")
    agg: dict[str, list[dict]] = {}
    for rec in mondrian_records:
        key = f"{rec['model']}__cl{int(rec['confidence_level'] * 100)}__{rec['stratum']}"
        agg.setdefault(key, []).append(rec)

    summary: list[dict] = []
    for key, recs in agg.items():
        cov_marg = np.array([r["coverage_marginal"] for r in recs], dtype=float)
        cov_mond = np.array([r["coverage_mondrian"] for r in recs], dtype=float)
        w_marg = np.array([r["width_marginal"] for r in recs], dtype=float)
        w_mond = np.array([r["width_mondrian"] for r in recs], dtype=float)
        n_used = sum(1 for r in recs if r["mondrian_used"])
        summary.append(
            dict(
                key=key,
                model=recs[0]["model"],
                confidence_level=recs[0]["confidence_level"],
                stratum=recs[0]["stratum"],
                n_folds=len(recs),
                n_folds_mondrian_used=n_used,
                cov_marginal_mean=float(np.nanmean(cov_marg)),
                cov_marginal_std=float(np.nanstd(cov_marg)),
                cov_mondrian_mean=float(np.nanmean(cov_mond)),
                cov_mondrian_std=float(np.nanstd(cov_mond)),
                cov_delta_mean=float(np.nanmean(cov_mond - cov_marg)),
                width_marginal_mean=float(np.nanmean(w_marg)),
                width_mondrian_mean=float(np.nanmean(w_mond)),
                width_inflation_pct=float(
                    100.0 * (np.nanmean(w_mond) - np.nanmean(w_marg))
                    / np.nanmean(w_marg)
                ),
            )
        )

    out_summary = OUTPUT_DIR / "mondrian_summary.json"
    with open(out_summary, "w") as f:
        json.dump(
            dict(
                method="Aggregate of Mondrian per-stratum recalibration",
                random_state=args.seed,
                summary=summary,
            ),
            f,
            indent=2,
            default=str,
        )
    log.info("Wrote %s", out_summary)

    # Pretty log: by CL, by model, sorted by stratum
    log.info("Summary table (Mondrian vs marginal coverage):")
    for cl in (0.90, 0.95):
        log.info("  Confidence level %.2f:", cl)
        for model in ("DeepHit", "Graph-DT"):
            for stratum in CARRIER_STRATA:
                key = f"{model}__cl{int(cl * 100)}__{stratum}"
                rec = next((s for s in summary if s["key"] == key), None)
                if rec is None:
                    continue
                if rec["n_folds_mondrian_used"] == 0:
                    log.info(
                        "    %s × %s: mondrian N/A (n_cal too small in all folds)",
                        model, stratum,
                    )
                    continue
                target_cov = cl
                marg_dev = rec["cov_marginal_mean"] - target_cov
                mond_dev = rec["cov_mondrian_mean"] - target_cov
                log.info(
                    "    %-8s × %-13s | marg %.3f (%+.3f) | mondrian %.3f (%+.3f) | "
                    "Δ=%+0.3f | width inflate %+5.1f%% | folds_mond_used=%d/%d",
                    model, stratum,
                    rec["cov_marginal_mean"], marg_dev,
                    rec["cov_mondrian_mean"], mond_dev,
                    rec["cov_delta_mean"],
                    rec["width_inflation_pct"],
                    rec["n_folds_mondrian_used"], rec["n_folds"],
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
