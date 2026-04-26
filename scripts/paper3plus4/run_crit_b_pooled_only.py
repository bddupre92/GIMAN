"""WS-P3-CRIT-B Pooled-OOF only — skips the slow per-fold bootstrap.

The full runner ``run_subgroup_with_lrrk2_gba_fix.py`` does:
  (a) per-carrier C-td (5 folds × 2 models, ~50 min, ALREADY DONE on disk)
  (b) per-fold bootstrap interaction tests (5 folds × 2 models × 3 strata, slow)
  (c) pooled-OOF interaction tests (6 cells, the WS-P3-CRIT-B canonical step)
  (d) conditional conformal coverage (separate)

This helper runs ONLY (c). It loads the existing checkpoints, regenerates the
per-fold predictions (~5-10 min, just inference), populates per_fold_interactions
with the minimal data the pooled function needs, then runs the pooled-OOF test
for each of 6 (model × stratum) cells (~30-60 min).

Output: writes a refreshed ``interaction_tests_carriers.json`` with both pooled
and Fisher-legacy fields, plus BH-FDR on pooled.

Reproducibility: random_state=42 throughout. Reads checkpoints from
outputs/paper3_checkpoints/{deephit,graph_dt}/ (read-only).

Run with nohup so it survives shell/IDE resets::

    nohup .venv/bin/python scripts/paper3plus4/run_crit_b_pooled_only.py \
        > /tmp/crit_b_pooled.log 2>&1 &
"""

from __future__ import annotations

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
    build_patient_arrays,
    extract_episodes,
)
from giman_pipeline.paper4.subgroup import (  # noqa: E402
    CARRIER_STRATA,
    apply_fdr_correction_scipy,
    assign_carrier_subgroups,
)

# Reuse the helpers from the full runner — they're already correct
from scripts.paper3plus4.run_subgroup_with_lrrk2_gba_fix import (  # noqa: E402
    OUTPUT_DIR,
    REFERENCE_STRATUM,
    compute_pooled_interaction_test,
    get_test_predictions,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("crit_b_pooled")

FEATURES_CSV = PROJECT_ROOT / "data" / "07_paper3_features" / "longitudinal_features.csv"
SEED = 42
N_BOOTSTRAP_POOLED = 500


def main() -> int:
    log.info("WS-P3-CRIT-B pooled-only re-run — OUTPUT_DIR=%s", OUTPUT_DIR)

    log.info("Loading carrier flags from features.paper1_features_with_targets ...")
    carrier_df = read_sql(
        "SELECT patno, lrrk2_carrier, gba_carrier, apoe_e4_carrier, sex, age_at_baseline "
        "FROM features.paper1_features_with_targets"
    )
    lrrk2_count = int((carrier_df["lrrk2_carrier"].fillna(0) == 1).sum())
    gba_count = int((carrier_df["gba_carrier"].fillna(0) == 1).sum())
    apoe_count = int((carrier_df["apoe_e4_carrier"].fillna(0) == 1).sum())
    log.info("  feature-table counts: LRRK2+=%d, GBA+=%d, APOE+=%d",
             lrrk2_count, gba_count, apoe_count)
    assert lrrk2_count == 175 and gba_count == 111 and apoe_count == 441, (
        f"SQL counts drift! Expected 175/111/441; got {lrrk2_count}/{gba_count}/{apoe_count}."
    )

    log.info("Loading Paper 3 longitudinal features ...")
    features_df = pd.read_csv(FEATURES_CSV, low_memory=False)
    patient_arrays, _ = build_patient_arrays(features_df)
    episodes = extract_episodes(features_df, verbose=False)
    log.info("  %d episodes across %d patnos",
             len(episodes), features_df["PATNO"].nunique())

    # Harmonise carrier_df column case
    carrier_df = carrier_df.rename(columns={"patno": "PATNO"})
    carrier_df["patno"] = carrier_df["PATNO"]

    # Build per_fold_interactions with minimal keys for compute_pooled_interaction_test
    log.info("Loading checkpoints + computing per-fold predictions (5 folds × 2 models) ...")
    per_fold_interactions: dict[tuple[str, str], list[dict]] = {
        (model_name, stratum): []
        for model_name in ("DeepHit", "Graph-DT")
        for stratum in CARRIER_STRATA
        if stratum != REFERENCE_STRATUM
    }
    for fold_idx in range(5):
        for model_type, model_name in (("deephit", "DeepHit"), ("graph_dt", "Graph-DT")):
            preds, patnos, _ = get_test_predictions(
                model_type, fold_idx, episodes, patient_arrays
            )
            carrier_assignments = assign_carrier_subgroups(patnos, carrier_df)
            for stratum in CARRIER_STRATA:
                if stratum == REFERENCE_STRATUM:
                    continue
                per_fold_interactions[(model_name, stratum)].append(
                    dict(
                        fold_idx=fold_idx,
                        preds=preds,
                        patnos=patnos,
                        assignments=carrier_assignments,
                        delta_ctd=float("nan"),
                        p_value=float("nan"),
                    )
                )
            log.info("  fold=%d model=%s — predictions cached", fold_idx, model_name)

    log.info("Pooled-OOF interaction tests (B=%d, seed=%d) per (model, stratum) ...",
             N_BOOTSTRAP_POOLED, SEED)
    fdr_records = []
    for (model, stratum), folds in per_fold_interactions.items():
        log.info("  computing pooled-OOF for (%s, %s) ...", model, stratum)
        pooled = compute_pooled_interaction_test(
            per_fold_interactions,
            model=model,
            stratum=stratum,
            reference_label=REFERENCE_STRATUM,
            n_bootstrap=N_BOOTSTRAP_POOLED,
            random_state=SEED,
        )
        log.info(
            "    delta_ctd=%+0.4f, p_pooled=%0.4f, n_pool=%d (n_carrier=%d, n_ref=%d)",
            pooled["delta_ctd"], pooled["p_value"], pooled["n_pool"],
            pooled["n_carrier"], pooled["n_reference"],
        )
        per_fold_for_json = [
            {k: v for k, v in f.items() if k not in ("preds", "patnos", "assignments")}
            for f in folds
        ]
        fdr_records.append(
            dict(
                model=model,
                stratum=stratum,
                delta_ctd_pooled=pooled["delta_ctd"],
                p_value_pooled=pooled["p_value"],
                n_carrier_pooled=pooled["n_carrier"],
                n_reference_pooled=pooled["n_reference"],
                n_pool=pooled["n_pool"],
                n_folds_pooled=pooled["n_folds_pooled"],
                n_bootstrap_pooled=pooled["n_bootstrap"],
                # Legacy Fisher fields are NaN here — no per-fold bootstrap was run.
                # Pre-fix Fisher values preserved in
                # outputs/paper4/subgroup_carriers/_pre_crit_b/interaction_tests_carriers.json
                delta_mean=float("nan"),
                p_raw_mean=float("nan"),
                p_fisher_legacy=float("nan"),
                per_fold=per_fold_for_json,
            )
        )

    raw_p_pooled = [r["p_value_pooled"] for r in fdr_records]
    p_fdr = apply_fdr_correction_scipy(raw_p_pooled)
    for r, p_adj in zip(fdr_records, p_fdr):
        r["p_fdr"] = float(p_adj)

    out_path = OUTPUT_DIR / "interaction_tests_carriers.json"
    with open(out_path, "w") as f:
        json.dump(
            dict(
                n_hypotheses=len(fdr_records),
                fdr_method="BH (scipy.stats.false_discovery_control, method='bh')",
                p_combination=(
                    "Pooled-OOF single bootstrap test per (model, stratum) "
                    "(WS-P3-CRIT-B; replaces Fisher across 5 folds — see "
                    "_pre_crit_b/ for pre-fix outputs). Note: per-fold "
                    "bootstrap NOT re-run in this helper — Fisher-legacy "
                    "fields are NaN. Pre-fix Fisher values preserved in "
                    "_pre_crit_b/interaction_tests_carriers.json for the "
                    "side-by-side §S-CRIT-B table."
                ),
                random_state=SEED,
                n_bootstrap_pooled=N_BOOTSTRAP_POOLED,
                records=fdr_records,
            ),
            f,
            indent=2,
        )
    log.info("Wrote %s", out_path)

    log.info("Summary (BH-FDR sorted by p_fdr ascending):")
    for r in sorted(fdr_records, key=lambda r: r["p_fdr"]):
        sig = "*" if r["p_fdr"] < 0.05 else " "
        log.info(
            " %s (%s, %s): delta=%+0.4f p_pooled=%0.4f p_fdr=%0.4f n_pool=%d",
            sig, r["model"], r["stratum"],
            r["delta_ctd_pooled"], r["p_value_pooled"], r["p_fdr"], r["n_pool"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
