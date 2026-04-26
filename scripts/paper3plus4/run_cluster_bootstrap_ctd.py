"""WS-P3-2 Subject-level (cluster) bootstrap of C-td on per-fold predictions.

Reviewer3.com #5 flagged that the existing per-fold C-td 95\\% CIs were
computed via *episode-level* bootstrap (``bootstrap_ctd_ci`` in
``src/giman_pipeline/paper4/subgroup.py``), treating each stage-occupancy
episode as an independent observation. Because 1{,}900 patients contribute
4{,}792 episodes (mean 2.5 episodes per patient), within-patient
correlation between episodes biases iid bootstrap CIs to be too narrow.

This runner re-computes per-fold C-td 95\\% CIs using the
*subject-level* (cluster) bootstrap (Davison & Hinkley 1997 §3.8;
Field & Welsh 2007): unique patnos are resampled with replacement, and
all of each sampled patno's episodes are included in the resample.

Output:

    outputs/paper4/cluster_bootstrap/cluster_bootstrap_ctd.json

      Per-(model, fold) iid vs cluster CIs + cluster vs iid CI-width
      ratio + observed C-td.

Run::

    .venv/bin/python scripts/paper3plus4/run_cluster_bootstrap_ctd.py
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

from giman_pipeline.paper3.dynamic_deephit import (  # noqa: E402
    build_patient_arrays,
    compute_ctd,
    extract_episodes,
)
from giman_pipeline.paper4.subgroup import (  # noqa: E402
    bootstrap_ctd_ci,
    cluster_bootstrap_ctd_ci,
)
from scripts.paper3plus4.run_subgroup_with_lrrk2_gba_fix import (  # noqa: E402
    get_test_predictions,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("cluster_bootstrap_ctd")

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper4" / "cluster_bootstrap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = PROJECT_ROOT / "data" / "07_paper3_features" / "longitudinal_features.csv"
SEED = 42
N_BOOTSTRAP = 1000


def _ci(arr: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap (alpha/2, 1-alpha/2) CI."""
    return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    log.info("WS-P3-2 cluster-bootstrap C-td CIs")
    log.info("  OUTPUT_DIR=%s, n_bootstrap=%d, seed=%d",
             OUTPUT_DIR, args.n_bootstrap, args.seed)

    log.info("Loading Paper 3 longitudinal features ...")
    features_df = pd.read_csv(FEATURES_CSV, low_memory=False)
    patient_arrays, _ = build_patient_arrays(features_df)
    episodes = extract_episodes(features_df, verbose=False)
    log.info("  %d episodes across %d patnos",
             len(episodes), features_df["PATNO"].nunique())

    records: list[dict] = []
    for fold_idx in range(5):
        for model_type, model_name in (("deephit", "DeepHit"), ("graph_dt", "Graph-DT")):
            preds, patnos, _ = get_test_predictions(
                model_type, fold_idx, episodes, patient_arrays
            )
            n_episodes = len(patnos)
            n_unique = len(set(patnos))
            obs_ctd = float(compute_ctd(preds))
            log.info(
                "  fold=%d model=%-8s | n_episodes=%4d n_patnos=%4d | observed C-td=%.4f",
                fold_idx, model_name, n_episodes, n_unique, obs_ctd,
            )

            log.info("    iid (episode-level) bootstrap (B=%d) ...", args.n_bootstrap)
            iid_arr = bootstrap_ctd_ci(
                preds, n_bootstrap=args.n_bootstrap,
                random_state=args.seed + fold_idx,
            )
            iid_lo, iid_hi = _ci(iid_arr)
            iid_width = iid_hi - iid_lo

            log.info("    cluster (patno-level) bootstrap (B=%d) ...", args.n_bootstrap)
            clu_arr = cluster_bootstrap_ctd_ci(
                preds, patnos, n_bootstrap=args.n_bootstrap,
                random_state=args.seed + fold_idx,
            )
            clu_lo, clu_hi = _ci(clu_arr)
            clu_width = clu_hi - clu_lo

            inflation_pct = 100.0 * (clu_width - iid_width) / max(iid_width, 1e-12)

            record = dict(
                model=model_name,
                fold_idx=fold_idx,
                n_episodes=n_episodes,
                n_unique_patnos=n_unique,
                episodes_per_patient=n_episodes / max(n_unique, 1),
                observed_ctd=obs_ctd,
                iid_lo=iid_lo,
                iid_hi=iid_hi,
                iid_width=iid_width,
                iid_mean=float(iid_arr.mean()),
                iid_std=float(iid_arr.std()),
                cluster_lo=clu_lo,
                cluster_hi=clu_hi,
                cluster_width=clu_width,
                cluster_mean=float(clu_arr.mean()),
                cluster_std=float(clu_arr.std()),
                ci_width_inflation_pct=inflation_pct,
            )
            records.append(record)
            log.info(
                "    iid 95%% CI=[%.4f, %.4f] (w=%.4f) | cluster 95%% CI=[%.4f, %.4f] (w=%.4f) | inflation=%+.1f%%",
                iid_lo, iid_hi, iid_width,
                clu_lo, clu_hi, clu_width,
                inflation_pct,
            )

    # Per-model aggregate
    log.info("Aggregate per-model (across 5 folds):")
    summary: dict[str, dict] = {}
    for model_name in ("DeepHit", "Graph-DT"):
        recs = [r for r in records if r["model"] == model_name]
        obs = np.array([r["observed_ctd"] for r in recs])
        iid_lo = np.array([r["iid_lo"] for r in recs])
        iid_hi = np.array([r["iid_hi"] for r in recs])
        clu_lo = np.array([r["cluster_lo"] for r in recs])
        clu_hi = np.array([r["cluster_hi"] for r in recs])
        clu_w = np.array([r["cluster_width"] for r in recs])
        iid_w = np.array([r["iid_width"] for r in recs])
        infl = np.array([r["ci_width_inflation_pct"] for r in recs])
        summary[model_name] = dict(
            n_folds=len(recs),
            observed_ctd_mean=float(obs.mean()),
            observed_ctd_std=float(obs.std()),
            iid_ci_lo_mean=float(iid_lo.mean()),
            iid_ci_hi_mean=float(iid_hi.mean()),
            iid_width_mean=float(iid_w.mean()),
            cluster_ci_lo_mean=float(clu_lo.mean()),
            cluster_ci_hi_mean=float(clu_hi.mean()),
            cluster_width_mean=float(clu_w.mean()),
            ci_width_inflation_pct_mean=float(infl.mean()),
            ci_width_inflation_pct_min=float(infl.min()),
            ci_width_inflation_pct_max=float(infl.max()),
        )
        log.info(
            "  %s | obs %.4f ± %.4f | iid 95%%CI [%.4f, %.4f] (w=%.4f) | "
            "cluster 95%%CI [%.4f, %.4f] (w=%.4f) | inflation %+.1f%% (range %+.1f%%-%+.1f%%)",
            model_name,
            obs.mean(), obs.std(),
            iid_lo.mean(), iid_hi.mean(), iid_w.mean(),
            clu_lo.mean(), clu_hi.mean(), clu_w.mean(),
            infl.mean(), infl.min(), infl.max(),
        )

    out_path = OUTPUT_DIR / "cluster_bootstrap_ctd.json"
    with open(out_path, "w") as f:
        json.dump(
            dict(
                method=(
                    "Per-fold C-td bootstrap CIs: iid (episode-level) vs "
                    "cluster (subject-level, patno) variants"
                ),
                citation_iid="standard percentile bootstrap (Efron 1979)",
                citation_cluster="Davison & Hinkley 1997 §3.8; Field & Welsh 2007",
                n_bootstrap=args.n_bootstrap,
                random_state=args.seed,
                ci_alpha=0.05,
                records=records,
                summary=summary,
            ),
            f,
            indent=2,
            default=str,
        )
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
