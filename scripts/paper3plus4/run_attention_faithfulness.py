"""WS-P3-15 attention-faithfulness experiment for Graph-DT.

For each test patient (per fold), compares the CIF prediction shift
when masking the top-k attended neighbors (sorted by edge weight) vs
masking k random neighbors. The faithfulness gap (top-k − random-k)
quantifies whether the kNN edge weights are informative attribution
signals or diffuse regularization.

Subject of WS-P3-15 (reviewer3.com): substitute for full Koh-Liang
influence functions, which are intractable on the 1{,}900-patient
deep survival model with Mac compute. The deletion experiment is
the field-standard alternative for graph-explanation faithfulness
(Yuan et al. 2022; DeYoung et al. 2020 ERASER).

Output:

    outputs/paper4/faithfulness/attention_faithfulness.json

      Per-(model, fold, patient) record + per-fold aggregate +
      per-model aggregate (mean ± SD across folds of per-patient
      faithfulness-gap mean and Spearman ρ).

Run::

    .venv/bin/python scripts/paper3plus4/run_attention_faithfulness.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from giman_pipeline.paper3.dynamic_deephit import (  # noqa: E402
    build_patient_arrays,
    extract_episodes,
)
from giman_pipeline.paper3.graph_digital_twin import (  # noqa: E402
    GraphDeepHitDataset,
    graph_collate_fn,
    load_graph_dt_checkpoint,
)
from giman_pipeline.paper4.faithfulness import (  # noqa: E402
    compute_faithfulness_for_patient,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("attention_faithfulness")

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper4" / "faithfulness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "paper3_checkpoints"
FEATURES_CSV = PROJECT_ROOT / "data" / "07_paper3_features" / "longitudinal_features.csv"
SEED = 42
K_MASK_VALUES = (1, 3, 5)
N_RANDOM_SEEDS = 3
# Per-fold patient subsample (for compute budget — set to 0 to use all test
# patients). With ~380 patients × 5 folds × 13 inferences × 50ms ≈ 20 min,
# the full run is tractable on Mac.
PATIENT_SUBSAMPLE = 0  # 0 = no subsample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--patient-subsample", type=int, default=PATIENT_SUBSAMPLE)
    args = parser.parse_args()

    log.info("WS-P3-15 attention-faithfulness experiment (Graph-DT)")
    log.info(
        "  OUTPUT_DIR=%s, k_mask_values=%s, n_random_seeds=%d, seed=%d, subsample=%d",
        OUTPUT_DIR, K_MASK_VALUES, N_RANDOM_SEEDS, args.seed, args.patient_subsample,
    )

    log.info("Loading Paper 3 longitudinal features ...")
    features_df = pd.read_csv(FEATURES_CSV, low_memory=False)
    patient_arrays, _ = build_patient_arrays(features_df)
    episodes = extract_episodes(features_df, verbose=False)
    log.info("  %d episodes across %d patnos",
             len(episodes), features_df["PATNO"].nunique())

    rng_subsample = np.random.RandomState(args.seed)
    all_records: list[dict] = []

    for fold_idx in range(5):
        ckpt = CHECKPOINT_DIR / "graph_dt" / f"fold{fold_idx}_graph_dt.pt"
        log.info("fold=%d loading %s ...", fold_idx, ckpt.name)
        model, cp = load_graph_dt_checkpoint(ckpt, device=torch.device("cpu"))
        device = next(model.parameters()).device

        test_pats = set(cp["test_pats"])
        means, stds = cp["means"], cp["stds"]
        pat_to_gidx = cp["pat_to_gidx"]
        node_baseline = cp["node_baseline"]
        edge_index = cp["edge_index"]
        edge_weight = cp["edge_weight"]

        test_eps = [e for e in episodes if e.patno in test_pats]
        ds = GraphDeepHitDataset(test_eps, patient_arrays, means, stds, pat_to_gidx)

        # Subsample patients for compute budget if requested
        n_total = len(test_eps)
        if args.patient_subsample > 0 and args.patient_subsample < n_total:
            select_idx = rng_subsample.choice(
                n_total, size=args.patient_subsample, replace=False
            )
            select_idx = sorted(select_idx.tolist())
        else:
            select_idx = list(range(n_total))

        log.info(
            "  fold=%d test_eps=%d (running on %d patient-episodes)",
            fold_idx, n_total, len(select_idx),
        )

        loader = DataLoader(
            ds, batch_size=1, shuffle=False, collate_fn=graph_collate_fn,
            num_workers=0,
        )
        # Materialize all single-element batches; then iterate select_idx
        all_batches = list(loader)

        t0 = time.time()
        n_done = 0
        for ds_idx in select_idx:
            batch = all_batches[ds_idx]
            ep = test_eps[ds_idx]
            try:
                rec = compute_faithfulness_for_patient(
                    model=model,
                    patno=ep.patno,
                    gidx=int(batch["graph_idxs"].item()),
                    sequences=batch["sequences"],
                    seq_lens=batch["seq_lens"],
                    stage_idxs=batch["stage_idxs"],
                    graph_idxs=batch["graph_idxs"],
                    node_baseline=node_baseline,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    k_mask_values=K_MASK_VALUES,
                    n_random_seeds=N_RANDOM_SEEDS,
                    rng_seed=args.seed,
                    device=device,
                )
            except Exception as exc:
                log.warning(
                    "  fold=%d patno=%s SKIPPED (%s)", fold_idx, ep.patno, exc,
                )
                continue

            all_records.append(
                dict(
                    model="Graph-DT",
                    fold_idx=fold_idx,
                    patno=rec.patno,
                    gidx=rec.gidx,
                    n_neighbors=rec.n_neighbors,
                    baseline_l1=rec.baseline_l1,
                    top_k_shift_l1={str(k): v for k, v in rec.top_k_shift_l1.items()},
                    random_k_shift_l1_mean={
                        str(k): v for k, v in rec.random_k_shift_l1_mean.items()
                    },
                    random_k_shift_l1_std={
                        str(k): v for k, v in rec.random_k_shift_l1_std.items()
                    },
                    gap_l1={str(k): v for k, v in rec.gap_l1.items()},
                    spearman_attn_vs_delete=rec.spearman_attn_vs_delete,
                )
            )

            n_done += 1
            if n_done % 50 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                remaining = (len(select_idx) - n_done) / max(rate, 1e-9)
                log.info(
                    "  fold=%d progress=%d/%d, %.1f pat/s, ETA fold %.1f min",
                    fold_idx, n_done, len(select_idx), rate, remaining / 60.0,
                )
        log.info(
            "  fold=%d done (%d records, %.1f min)",
            fold_idx, n_done, (time.time() - t0) / 60.0,
        )

    # Per-fold aggregate
    log.info("Aggregating across folds ...")
    fold_summary: list[dict] = []
    for fold_idx in range(5):
        recs = [r for r in all_records if r["fold_idx"] == fold_idx]
        if not recs:
            continue
        for k in K_MASK_VALUES:
            top = np.array([r["top_k_shift_l1"][str(k)] for r in recs])
            rnd = np.array([r["random_k_shift_l1_mean"][str(k)] for r in recs])
            gap = np.array([r["gap_l1"][str(k)] for r in recs])
            top_clean = top[~np.isnan(top)]
            rnd_clean = rnd[~np.isnan(rnd)]
            gap_clean = gap[~np.isnan(gap)]
            fold_summary.append(
                dict(
                    fold_idx=fold_idx,
                    k_mask=k,
                    n_patients=len(recs),
                    n_eligible=len(top_clean),
                    top_k_shift_l1_mean=float(top_clean.mean()) if len(top_clean) else float("nan"),
                    random_k_shift_l1_mean=float(rnd_clean.mean()) if len(rnd_clean) else float("nan"),
                    gap_l1_mean=float(gap_clean.mean()) if len(gap_clean) else float("nan"),
                    gap_l1_std=float(gap_clean.std()) if len(gap_clean) else float("nan"),
                    fraction_positive_gap=float((gap_clean > 0).mean()) if len(gap_clean) else float("nan"),
                )
            )

    # Per-model aggregate (across all folds)
    spear = np.array([r["spearman_attn_vs_delete"] for r in all_records])
    spear_clean = spear[~np.isnan(spear)]
    model_summary = {"Graph-DT": {}}
    for k in K_MASK_VALUES:
        top = np.array([r["top_k_shift_l1"][str(k)] for r in all_records])
        rnd = np.array([r["random_k_shift_l1_mean"][str(k)] for r in all_records])
        gap = np.array([r["gap_l1"][str(k)] for r in all_records])
        top_clean = top[~np.isnan(top)]
        rnd_clean = rnd[~np.isnan(rnd)]
        gap_clean = gap[~np.isnan(gap)]
        model_summary["Graph-DT"][f"k_mask_{k}"] = dict(
            n_patients=len(top_clean),
            top_k_shift_l1_mean=float(top_clean.mean()) if len(top_clean) else float("nan"),
            random_k_shift_l1_mean=float(rnd_clean.mean()) if len(rnd_clean) else float("nan"),
            gap_l1_mean=float(gap_clean.mean()) if len(gap_clean) else float("nan"),
            gap_l1_std=float(gap_clean.std()) if len(gap_clean) else float("nan"),
            fraction_positive_gap=float((gap_clean > 0).mean()) if len(gap_clean) else float("nan"),
        )
    model_summary["Graph-DT"]["spearman_attn_vs_delete_mean"] = (
        float(spear_clean.mean()) if len(spear_clean) else float("nan")
    )
    model_summary["Graph-DT"]["spearman_attn_vs_delete_std"] = (
        float(spear_clean.std()) if len(spear_clean) else float("nan")
    )
    model_summary["Graph-DT"]["fraction_positive_spearman"] = (
        float((spear_clean > 0).mean()) if len(spear_clean) else float("nan")
    )
    model_summary["Graph-DT"]["n_patients_total"] = len(all_records)

    out_path = OUTPUT_DIR / "attention_faithfulness.json"
    with open(out_path, "w") as f:
        json.dump(
            dict(
                method=(
                    "Per-patient deletion-shift faithfulness for Graph-DT "
                    "kNN attention pathway: top-k vs random-k edge masking."
                ),
                citation_yuan="Yuan et al. 2022 (graph-explanation taxonomy)",
                citation_deyoung="DeYoung et al. 2020 (ERASER comprehensiveness)",
                citation_jain="Jain & Wallace 2019 (attention is not explanation)",
                citation_wiegreffe="Wiegreffe & Pinter 2019 (counter-argument)",
                k_mask_values=list(K_MASK_VALUES),
                n_random_seeds=N_RANDOM_SEEDS,
                random_state=args.seed,
                fold_summary=fold_summary,
                model_summary=model_summary,
                records=all_records,
            ),
            f,
            indent=2,
            default=str,
        )
    log.info("Wrote %s (%d patient records)", out_path, len(all_records))

    # Pretty log
    log.info("Aggregate (Graph-DT, across all folds and patients):")
    for k in K_MASK_VALUES:
        s = model_summary["Graph-DT"][f"k_mask_{k}"]
        log.info(
            "  k_mask=%d | top mean=%.4f | random mean=%.4f | gap mean=%.4f ± %.4f | "
            "fraction positive gap=%.3f | n=%d",
            k,
            s["top_k_shift_l1_mean"], s["random_k_shift_l1_mean"],
            s["gap_l1_mean"], s["gap_l1_std"],
            s["fraction_positive_gap"], s["n_patients"],
        )
    log.info(
        "  Spearman ρ (per-neighbor edge_weight vs delete shift): %.3f ± %.3f, fraction positive=%.3f",
        model_summary["Graph-DT"]["spearman_attn_vs_delete_mean"],
        model_summary["Graph-DT"]["spearman_attn_vs_delete_std"],
        model_summary["Graph-DT"]["fraction_positive_spearman"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
