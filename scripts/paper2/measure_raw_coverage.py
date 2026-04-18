#!/usr/bin/env python3
"""Measure raw heteroscedastic+MC-dropout coverage on the 48 Paper 2 checkpoints.

Phase P2-Cal.0 of the calibration comparison.

For each of 4 GIMIN variants x 4 mask fractions x 3 runs = 48 checkpoints:
  1. Reproduce the training-time artificial mask (deterministic from seed=42+run_idx)
  2. Load the checkpoint state_dict and reconstruct the model
  3. Run inference with MC dropout (T=20) to get mean + total_std in original scale
  4. Compute raw empirical coverage at nominal levels {0.50, 0.70, 0.80, 0.90, 0.95}
  5. Also compute per-feature temperature-scaled coverage (P2-Cal.B) as a bonus
  6. Save a per-checkpoint JSON entry

Produces: outputs/paper2_benchmark/calibration_retune/raw_coverage_measurement.json

Usage:
    .venv/bin/python scripts/paper2/measure_raw_coverage.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm as scipy_norm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "GIMImpN_imputation"))
sys.path.insert(0, str(ROOT / "scripts"))

# Re-use the benchmark's data/config/scaler/graph pipeline
from run_paper2_experiments import (  # noqa: E402
    MODALITY_DIMS,
    STAGE_NAMES,
    build_gimin_config,
    build_scaler_from_config,
    create_artificial_mask,
    evaluate_gimin_model,
    load_data,
)

import argparse

DEFAULT_CKPT_DIR = (
    ROOT
    / "outputs"
    / "paper2_benchmark"
    / "runs"
    / "full_benchmark_20260222_160247"
    / "checkpoints"
)
OUT_DIR = ROOT / "outputs" / "paper2_benchmark" / "calibration_retune"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["Vanilla", "StageGraphOnly", "StageDecoderOnly", "StageConditioned"]
MASK_FRACS = [0.1, 0.2, 0.3, 0.5]
NUM_RUNS = 3
SEED = 42
COVERAGE_TARGETS = [0.50, 0.70, 0.80, 0.90, 0.95]
MC_SAMPLES = 20


def _build_model(variant: str, cfg):
    """Instantiate the correct GIMIN variant matching the saved checkpoint."""
    from giman_pipeline.imputation.stage_conditioned_gimin import (
        StageConditionedGIMIN,
        VanillaGIMIN,
    )

    # VanillaGIMIN (no stage anything) is used for two variants:
    #   - Vanilla: vanilla graph, no stage conditioning
    #   - StageGraphOnly: stage-aware graph, no stage conditioning in model
    # StageConditionedGIMIN (has stage_embedding + stage_decoder) is used for:
    #   - StageDecoderOnly: vanilla graph, stage-conditioned decoder
    #   - StageConditioned: stage-aware graph, stage-conditioned decoder
    if variant in ("Vanilla", "StageGraphOnly"):
        return VanillaGIMIN(
            modality_dims=MODALITY_DIMS,
            embed_dim=64,
            num_gnn_layers=3,
            num_heads=4,
            mc_dropout=0.1,
        ), False
    return StageConditionedGIMIN(
        modality_dims=MODALITY_DIMS,
        embed_dim=64,
        num_gnn_layers=3,
        num_heads=4,
        mc_dropout=0.1,
        num_stages=6,
        stage_embed_dim=16,
    ), True


def _compute_raw_coverage(
    mean: np.ndarray,
    std: np.ndarray,
    truth: np.ndarray,
    eval_mask: np.ndarray,
    targets: list[float],
) -> dict:
    """Empirical coverage at each nominal level assuming Gaussian.

    Returns dict with per-target observed coverage and mean interval width.
    """
    # Clamp std to avoid division by zero
    std_safe = np.maximum(std, 1e-9)
    residuals = np.abs(truth - mean)
    eval_bool = eval_mask > 0
    res_flat = residuals[eval_bool]
    std_flat = std_safe[eval_bool]

    out = {"n_eval": int(eval_bool.sum())}
    for gamma in targets:
        z = scipy_norm.ppf(0.5 + gamma / 2)  # two-sided quantile
        inside = res_flat <= z * std_flat
        mean_width = float((2 * z * std_flat).mean())
        out[f"gamma_{gamma:.2f}"] = {
            "target": gamma,
            "observed_coverage": float(inside.mean()),
            "z_quantile": float(z),
            "mean_interval_width": mean_width,
        }
    return out


def _compute_temperature_scaled_coverage(
    mean: np.ndarray,
    std: np.ndarray,
    truth: np.ndarray,
    eval_mask: np.ndarray,
    targets: list[float],
    cal_frac: float = 0.5,
) -> dict:
    """Compute coverage after per-feature temperature scaling.

    Splits evaluation positions 50/50 into calibration (fit T_f) and test.
    Learns scalar T_f per feature such that |residual|_f / (T_f * std_f)
    matches a unit normal at 90% nominal. Uses the empirical 90% quantile
    of |residual_f / std_f| / z_0.90 as T_f.

    Returns dict with per-target observed coverage on the test split.
    """
    n, f = mean.shape
    eval_bool = eval_mask > 0
    std_safe = np.maximum(std, 1e-9)

    rng = np.random.RandomState(0)
    cal_mask = np.zeros_like(eval_mask, dtype=bool)
    test_mask = np.zeros_like(eval_mask, dtype=bool)
    for j in range(f):
        col_positions = np.flatnonzero(eval_bool[:, j])
        if len(col_positions) < 10:
            continue
        rng.shuffle(col_positions)
        cut = int(cal_frac * len(col_positions))
        cal_mask[col_positions[:cut], j] = True
        test_mask[col_positions[cut:], j] = True

    # Fit per-feature T_f at 90% nominal
    z90 = scipy_norm.ppf(0.95)  # two-sided 90% CI
    T = np.ones(f)
    for j in range(f):
        cal_positions = np.flatnonzero(cal_mask[:, j])
        if len(cal_positions) < 5:
            continue
        res_j = np.abs(truth[cal_positions, j] - mean[cal_positions, j])
        std_j = std_safe[cal_positions, j]
        # Empirical scale ratio: we want |res|/(T*std) to have 90%-ile = z90
        ratio = res_j / std_j
        T[j] = max(np.quantile(ratio, 0.90) / z90, 1e-3)

    # Evaluate on test split with scaled std
    out = {
        "n_test": int(test_mask.sum()),
        "per_feature_T": T.tolist(),
        "T_median": float(np.median(T)),
        "T_mean": float(T.mean()),
    }
    for gamma in targets:
        z = scipy_norm.ppf(0.5 + gamma / 2)
        inside = []
        total = 0
        for j in range(f):
            positions = np.flatnonzero(test_mask[:, j])
            if len(positions) == 0:
                continue
            res_j = np.abs(truth[positions, j] - mean[positions, j])
            std_j = std_safe[positions, j] * T[j]
            inside.append((res_j <= z * std_j).sum())
            total += len(positions)
        inside_total = int(sum(inside))
        out[f"gamma_{gamma:.2f}"] = {
            "target": gamma,
            "observed_coverage": inside_total / max(total, 1),
            "n_test": total,
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CKPT_DIR,
        help="Directory of GIMIN state_dict .pt files",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="raw_coverage_measurement.json",
        help="Output JSON filename (written to outputs/paper2_benchmark/calibration_retune/)",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="raw_coverage_summary.json",
        help="Aggregated summary JSON filename",
    )
    args = parser.parse_args()

    ckpt_dir_arg = args.checkpoint_dir
    out_name = args.output_name
    sum_name = args.summary_name

    device = torch.device("cpu")  # MC dropout on CPU for reproducibility
    torch.manual_seed(SEED)

    print("=" * 70)
    print(f"P2-Cal.0: Raw coverage measurement on checkpoints at {ckpt_dir_arg}")
    print(f"  Output: {OUT_DIR / out_name}")
    print("=" * 70)

    # ── 1. Load data exactly as the benchmark does ───────────────────
    features, mask, stages, feature_names = load_data()
    n, f = features.shape
    print(f"\nData: n={n}, f={f}, {100 * (1 - mask.mean()):.1f}% missing")

    cfg = build_gimin_config()
    scaler = build_scaler_from_config(cfg)

    features_t_raw = torch.tensor(features, dtype=torch.float32)
    mask_t = torch.tensor(mask, dtype=torch.float32)

    # Fit scaler on observed values (matches benchmark discipline)
    scaler.fit(features_t_raw, mask_t)
    features_norm_t = scaler.transform(features_t_raw, mask_t)

    stages_t = torch.tensor(stages, dtype=torch.long)

    # ── 2. Build both graphs once (stage-aware + vanilla) ────────────
    from giman_pipeline.imputation.stage_graph_builder import StageAwareGraphBuilder

    all_graphs = {}
    features_norm_np = features_norm_t.numpy()

    results = []

    for frac in MASK_FRACS:
        print(f"\n{'─' * 70}")
        print(f"Mask fraction: {frac:.0%}")
        print(f"{'─' * 70}")

        for run in range(NUM_RUNS):
            seed = SEED + run
            corrupted_mask, eval_mask = create_artificial_mask(
                mask,
                frac,
                seed=seed,
            )
            corrupted_mask_t = torch.tensor(corrupted_mask, dtype=torch.float32)

            # Build graphs (vanilla + stage-aware) for THIS corrupted_mask
            vanilla_key = f"vanilla_frac{frac}_run{run}"
            stage_key = f"stage_frac{frac}_run{run}"

            if vanilla_key not in all_graphs:
                vanilla_builder = StageAwareGraphBuilder(
                    k_neighbors=15, min_overlap=3, stage_affinity_beta=0.0
                )
                all_graphs[vanilla_key] = vanilla_builder.build_full_graph(
                    features_norm_np * corrupted_mask.astype(np.float32),
                    corrupted_mask,
                    stages=None,
                )
            if stage_key not in all_graphs:
                stage_builder = StageAwareGraphBuilder(
                    k_neighbors=15, min_overlap=3, stage_affinity_beta=0.3
                )
                all_graphs[stage_key] = stage_builder.build_full_graph(
                    features_norm_np * corrupted_mask.astype(np.float32),
                    corrupted_mask,
                    stages=stages,
                )

            for variant in VARIANTS:
                ckpt_fname = f"frac{frac:.1f}_run{run}_GIMIN_{variant}.pt"
                ckpt_path = ckpt_dir_arg / ckpt_fname
                if not ckpt_path.exists():
                    print(f"  [MISS] {ckpt_fname}")
                    continue

                t0 = time.time()
                model, is_stage = _build_model(variant, cfg)
                state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
                try:
                    model.load_state_dict(state)
                except RuntimeError as e:
                    print(f"  [LOAD-ERR] {ckpt_fname}: {e}")
                    continue
                model.to(device).eval()

                # Pick graph: Stage-aware graph for StageGraphOnly + StageConditioned,
                # vanilla graph for Vanilla + StageDecoderOnly.
                use_stage_graph = variant in ("StageGraphOnly", "StageConditioned")
                graph = all_graphs[stage_key if use_stage_graph else vanilla_key]

                edge_index = graph["edge_index"].to(device)
                edge_weight = graph["edge_weight"].to(device)
                overlap_frac = graph["overlap_frac"].to(device)

                # Inference
                try:
                    _, _, mean_pred, total_std = evaluate_gimin_model(
                        model,
                        features_norm_t.to(device),
                        mask_t.to(device),
                        corrupted_mask_t.to(device),
                        eval_mask,
                        edge_index,
                        edge_weight,
                        overlap_frac,
                        stages,
                        features_original=features,
                        scaler=scaler,
                        stages_t=stages_t.to(device) if is_stage else None,
                        is_stage_conditioned=is_stage,
                        mc_samples=MC_SAMPLES,
                    )
                except Exception as e:
                    print(f"  [INF-ERR] {ckpt_fname}: {e}")
                    continue

                # Compute D0 (raw) coverage
                raw_cov = _compute_raw_coverage(
                    mean_pred, total_std, features, eval_mask, COVERAGE_TARGETS
                )

                # Compute B (temperature-scaled) coverage
                temp_cov = _compute_temperature_scaled_coverage(
                    mean_pred, total_std, features, eval_mask, COVERAGE_TARGETS
                )

                entry = {
                    "variant": variant,
                    "frac": frac,
                    "run": run,
                    "checkpoint": ckpt_fname,
                    "n_masked": int(eval_mask.sum()),
                    "elapsed_seconds": round(time.time() - t0, 2),
                    "raw_coverage": raw_cov,
                    "temp_scaled_coverage": temp_cov,
                }
                results.append(entry)

                print(
                    f"  [{variant:>18s}] frac={frac:.1f} run={run} "
                    f"raw γ=0.90 → {raw_cov['gamma_0.90']['observed_coverage']:.3f}  "
                    f"tempB γ=0.90 → {temp_cov['gamma_0.90']['observed_coverage']:.3f}  "
                    f"({entry['elapsed_seconds']:.1f}s)"
                )

                # Incremental save
                with open(OUT_DIR / out_name, "w") as fp:
                    json.dump(
                        {
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "n_checkpoints_processed": len(results),
                            "targets": COVERAGE_TARGETS,
                            "mc_samples": MC_SAMPLES,
                            "seed": SEED,
                            "per_checkpoint": results,
                        },
                        fp,
                        indent=2,
                    )

    # ── 3. Aggregate across checkpoints at each γ ────────────────────
    summary = {}
    for gamma in COVERAGE_TARGETS:
        key = f"gamma_{gamma:.2f}"
        raw_values = [e["raw_coverage"][key]["observed_coverage"] for e in results]
        temp_values = [
            e["temp_scaled_coverage"][key]["observed_coverage"] for e in results
        ]
        summary[key] = {
            "target": gamma,
            "raw_mean": float(np.mean(raw_values)),
            "raw_std": float(np.std(raw_values)),
            "raw_median": float(np.median(raw_values)),
            "temp_mean": float(np.mean(temp_values)),
            "temp_std": float(np.std(temp_values)),
            "temp_median": float(np.median(temp_values)),
        }

    with open(OUT_DIR / sum_name, "w") as fp:
        json.dump(
            {
                "n_checkpoints": len(results),
                "variants": VARIANTS,
                "mask_fractions": MASK_FRACS,
                "num_runs": NUM_RUNS,
                "mc_samples": MC_SAMPLES,
                "summary_across_all_checkpoints": summary,
            },
            fp,
            indent=2,
        )

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"  Processed {len(results)} of {len(VARIANTS) * len(MASK_FRACS) * NUM_RUNS} checkpoints")
    print(f"  Raw + temp-scaled coverage per checkpoint: {OUT_DIR / 'raw_coverage_measurement.json'}")
    print(f"  Aggregated summary: {OUT_DIR / 'raw_coverage_summary.json'}")

    print("\n  Aggregated observed coverage (raw vs temperature-scaled):")
    print(f"  {'target':>8}  {'D0 raw':>12}  {'B temp':>12}  Δ")
    for gamma in COVERAGE_TARGETS:
        s = summary[f"gamma_{gamma:.2f}"]
        delta = s["temp_mean"] - s["raw_mean"]
        print(
            f"  {gamma:>8.2f}  {s['raw_mean']:>8.3f}±{s['raw_std']:.3f}"
            f"  {s['temp_mean']:>8.3f}±{s['temp_std']:.3f}  {delta:+.3f}"
        )


if __name__ == "__main__":
    main()
