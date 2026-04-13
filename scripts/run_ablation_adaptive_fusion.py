"""Ablation studies for Adaptive Cross-Modal Attention.

Tests each phase incrementally to isolate contributions:
  A. No cross-modal (concat only) — current best baseline (0.796)
  B. Current full GIMAN (pre-Phase-1 behavior via scale=1 + post-scaling)
  C. Phase 1: Strengthened bias (10x) + no post-scaling
  D. Phase 1+2: + observed-only masking (threshold=0.3)
  E. Full adaptive: + learned fusion gate

Also includes:
  - Threshold sensitivity sweep (0.1, 0.2, 0.3, 0.5)
  - Gate interpretability analysis (mean gate by modality count)

Usage:
    python scripts/run_ablation_adaptive_fusion.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.giman_pipeline.training.multi_task_trainer import (
    MultiTaskConfig,
    MultiTaskTrainer,
    build_knn_graph,
)

# ---------------------------------------------------------------------------
# Data loading and imputation (reused from run_ablation_full_clinical.py)
# ---------------------------------------------------------------------------


def load_data():
    """Load full_clinical dataset."""
    data_path = (
        project_root
        / "data/03_prodromal/final_training_dataset/prodromal_full_clinical.csv"
    )
    config_path = project_root / "config/feature_configs/full_clinical.yaml"
    df = pd.read_csv(data_path)
    config = yaml.safe_load(open(config_path))
    feature_cols = []
    modality_dims = {}
    for mod_name, mod_config in config["modalities"].items():
        modality_dims[mod_name] = mod_config["dim"]
        feature_cols.extend(mod_config["features"])
    features = df[feature_cols].values.astype(np.float32)
    times = df["time_to_event"].values.astype(np.float32)
    events = df["phenoconverted"].values.astype(np.int64)
    return df, features, times, events, feature_cols, modality_dims


def tiered_impute_fold(features_train, features_val, modality_dims):
    """Tiered imputation: MICE for low-missing, placeholder for high-missing."""
    obs_mask_train = (~np.isnan(features_train)).astype(np.float32)
    obs_mask_val = (~np.isnan(features_val)).astype(np.float32)
    offset = 0
    tier_3_cols, tier_12_cols = [], []
    for mod_name, dim in modality_dims.items():
        miss_rate = np.isnan(features_train[:, offset : offset + dim]).mean()
        cols = list(range(offset, offset + dim))
        (tier_3_cols if miss_rate > 0.50 else tier_12_cols).extend(cols)
        offset += dim
    for col in tier_3_cols:
        observed = features_train[:, col][~np.isnan(features_train[:, col])]
        fill_val = observed.mean() if len(observed) > 0 else 0.0
        features_train[:, col] = np.where(
            np.isnan(features_train[:, col]), fill_val, features_train[:, col]
        )
        features_val[:, col] = np.where(
            np.isnan(features_val[:, col]), fill_val, features_val[:, col]
        )
    if tier_12_cols and np.isnan(features_train[:, tier_12_cols]).any():
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=10, max_depth=5, random_state=42, n_jobs=-1
            ),
            max_iter=10,
            random_state=42,
            sample_posterior=False,
        )
        features_train[:, tier_12_cols] = imputer.fit_transform(
            features_train[:, tier_12_cols]
        )
        features_val[:, tier_12_cols] = imputer.transform(features_val[:, tier_12_cols])
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_val = scaler.transform(features_val)
    return features_train, features_val, obs_mask_train, obs_mask_val


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95):
    rng = np.random.RandomState(42)
    bootstrapped = [
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ]
    bootstrapped = np.array(bootstrapped)
    alpha = (1 - ci) / 2
    return (
        np.mean(values),
        np.percentile(bootstrapped, 100 * alpha),
        np.percentile(bootstrapped, 100 * (1 - alpha)),
    )


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------


def make_config(
    modality_dims, cross_modal_heads=None, adaptive_fusion=False, observed_threshold=0.0
):
    """Build a MultiTaskConfig from the YAML base + overrides."""
    train_config = yaml.safe_load(
        open(project_root / "config/training/true_giman.yaml")
    )
    n_modalities = len(modality_dims)
    embed_dim = train_config["model"]["modality_embed_dim"]
    if cross_modal_heads is None:
        cross_modal_heads = train_config["model"]["cross_modal_heads"]
    fused_dim = train_config["model"]["fused_dim"]
    if cross_modal_heads == 0:
        fused_dim = n_modalities * embed_dim
    return MultiTaskConfig(
        modality_dims=modality_dims,
        modality_embed_dim=embed_dim,
        cross_modal_heads=cross_modal_heads,
        fused_dim=fused_dim,
        gat_hidden_dim=train_config["model"]["gat_hidden_dim"],
        gat_output_dim=train_config["model"]["gat_output_dim"],
        gat_heads=train_config["model"]["gat_heads"],
        gat_layers=train_config["model"]["gat_layers"],
        dropout=train_config["model"]["dropout"],
        num_subtypes=train_config["model"]["num_subtypes"],
        num_diagnostic_classes=train_config["model"]["num_diagnostic_classes"],
        lr=train_config["training"]["lr"],
        weight_decay=float(train_config["training"]["weight_decay"]),
        max_epochs=train_config["training"]["max_epochs"],
        patience=train_config["training"]["patience"],
        grad_clip=train_config["training"]["grad_clip"],
        graph_k=train_config["data"]["graph_k"],
        warmup_survival=0,
        warmup_subtype=0,
        warmup_diagnostic=0,
        adaptive_fusion=adaptive_fusion,
        observed_threshold=observed_threshold,
    )


# ---------------------------------------------------------------------------
# CV runners
# ---------------------------------------------------------------------------


def run_giman_cv(
    features,
    times,
    events,
    modality_dims,
    config,
    label,
    k_folds=5,
    collect_gate_values=False,
):
    """Run GIMAN 5-fold CV. Optionally collect per-patient gate values."""
    trainer = MultiTaskTrainer(config)
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    c_indices = []
    all_gate_data = []  # (gate_value, n_modalities_observed) per patient

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(np.arange(len(features)), events)
    ):
        feat_train, feat_val = features[train_idx].copy(), features[val_idx].copy()
        feat_train, feat_val, mask_train, mask_val = tiered_impute_fold(
            feat_train, feat_val, modality_dims
        )
        k = min(config.graph_k, len(val_idx) - 1)
        train_data = Data(
            x=torch.tensor(feat_train, dtype=torch.float32),
            edge_index=build_knn_graph(
                feat_train, k=config.graph_k, obs_mask=mask_train
            ),
            time=torch.tensor(times[train_idx], dtype=torch.float32),
            event=torch.tensor(events[train_idx], dtype=torch.long),
            obs_mask=torch.tensor(mask_train, dtype=torch.float32),
        )
        val_data = Data(
            x=torch.tensor(feat_val, dtype=torch.float32),
            edge_index=build_knn_graph(feat_val, k=k, obs_mask=mask_val),
            time=torch.tensor(times[val_idx], dtype=torch.float32),
            event=torch.tensor(events[val_idx], dtype=torch.long),
            obs_mask=torch.tensor(mask_val, dtype=torch.float32),
        )
        result = trainer.train_fold(train_data, val_data, fold=fold, verbose=False)
        c_indices.append(result["final_metrics"].get("c_index", -1))

        # Collect gate values from validation set
        if collect_gate_values and hasattr(result, "model") or collect_gate_values:
            try:
                model = trainer.model
                model.eval()
                with torch.no_grad():
                    device = next(model.parameters()).device
                    vd = val_data.to(device)
                    obs = getattr(vd, "obs_mask", None)
                    out = model(
                        vd.x, vd.edge_index, obs_mask=obs, return_attention=True
                    )
                if out.fusion_gate_values is not None:
                    gate_vals = out.fusion_gate_values.squeeze().cpu().numpy()
                    # Count observed modalities per patient
                    offset = 0
                    for patient_i in range(len(val_idx)):
                        n_obs = 0
                        col = 0
                        for mod_name, dim in modality_dims.items():
                            mod_avail = mask_val[patient_i, col : col + dim].mean()
                            if mod_avail > 0.3:
                                n_obs += 1
                            col += dim
                        all_gate_data.append(
                            {
                                "gate_value": float(gate_vals[patient_i])
                                if gate_vals.ndim > 0
                                else float(gate_vals),
                                "n_modalities": n_obs,
                            }
                        )
            except Exception as e:
                print(f"    Warning: could not collect gate values: {e}")

    mean, ci_low, ci_high = bootstrap_ci(np.array(c_indices))
    result = {
        "name": label,
        "c_indices": c_indices,
        "mean": float(mean),
        "std": float(np.std(c_indices)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }
    if collect_gate_values and all_gate_data:
        result["gate_analysis"] = analyze_gate_values(all_gate_data)
    return result


def analyze_gate_values(gate_data):
    """Analyze gate values stratified by number of observed modalities."""
    from collections import defaultdict

    by_count = defaultdict(list)
    for entry in gate_data:
        by_count[entry["n_modalities"]].append(entry["gate_value"])

    analysis = {}
    for n_mod in sorted(by_count.keys()):
        vals = by_count[n_mod]
        analysis[f"{n_mod}_modalities"] = {
            "n_patients": len(vals),
            "mean_gate": float(np.mean(vals)),
            "std_gate": float(np.std(vals)),
            "min_gate": float(np.min(vals)),
            "max_gate": float(np.max(vals)),
        }
    return analysis


# ---------------------------------------------------------------------------
# Main experiments
# ---------------------------------------------------------------------------


def main():
    start_time = time.time()
    print("\n" + "=" * 70)
    print("ADAPTIVE CROSS-MODAL ATTENTION — INCREMENTAL ABLATION")
    print("=" * 70 + "\n")

    df, features, times, events, feature_cols, modality_dims = load_data()
    print(f"Dataset: {len(df)} patients, {events.sum()} events ({events.mean():.1%})")
    print(f"Features: {len(feature_cols)} across {len(modality_dims)} modalities")
    print(f"Modalities: {list(modality_dims.keys())}\n")

    results = {}

    # -----------------------------------------------------------------------
    # Experiment 1: Incremental ablation
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("EXPERIMENT 1: INCREMENTAL ABLATION")
    print("=" * 70 + "\n")

    # A. No cross-modal (baseline to beat)
    print("-" * 50)
    print("A. No cross-modal (concat only) — BASELINE TO BEAT")
    print("-" * 50)
    config_a = make_config(modality_dims, cross_modal_heads=0)
    results["A_no_crossmodal"] = run_giman_cv(
        features, times, events, modality_dims, config_a, "No cross-modal (concat only)"
    )
    print(
        f"  C-index: {results['A_no_crossmodal']['mean']:.4f} ± {results['A_no_crossmodal']['std']:.4f}\n"
    )

    # B. Current full GIMAN (pre-Phase-1: no adaptive, threshold=0.0)
    # Note: We can't truly revert the bias scale change without modifying code,
    # but threshold=0.0 and adaptive=False exercises the Phase 1 code path
    # (strengthened bias + no post-scaling). This IS the Phase 1 result.
    print("-" * 50)
    print("C. Phase 1: Strengthened bias (10x) + no post-scaling")
    print("-" * 50)
    config_c = make_config(modality_dims, adaptive_fusion=False, observed_threshold=0.0)
    results["C_phase1"] = run_giman_cv(
        features,
        times,
        events,
        modality_dims,
        config_c,
        "Phase 1: Strengthened bias + no post-scaling",
    )
    print(
        f"  C-index: {results['C_phase1']['mean']:.4f} ± {results['C_phase1']['std']:.4f}\n"
    )

    # D. Phase 1+2: + observed-only masking
    print("-" * 50)
    print("D. Phase 1+2: + observed-only masking (threshold=0.3)")
    print("-" * 50)
    config_d = make_config(modality_dims, adaptive_fusion=False, observed_threshold=0.3)
    results["D_phase1_2"] = run_giman_cv(
        features,
        times,
        events,
        modality_dims,
        config_d,
        "Phase 1+2: + observed-only masking (t=0.3)",
    )
    print(
        f"  C-index: {results['D_phase1_2']['mean']:.4f} ± {results['D_phase1_2']['std']:.4f}\n"
    )

    # E. Full adaptive: + learned fusion gate
    print("-" * 50)
    print("E. Full adaptive: Phase 1+2+3 (fusion gate)")
    print("-" * 50)
    config_e = make_config(modality_dims, adaptive_fusion=True, observed_threshold=0.3)
    results["E_full_adaptive"] = run_giman_cv(
        features,
        times,
        events,
        modality_dims,
        config_e,
        "Full adaptive (Phase 1+2+3)",
        collect_gate_values=True,
    )
    print(
        f"  C-index: {results['E_full_adaptive']['mean']:.4f} ± {results['E_full_adaptive']['std']:.4f}\n"
    )

    # -----------------------------------------------------------------------
    # Experiment 2: Threshold sensitivity
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("EXPERIMENT 2: THRESHOLD SENSITIVITY")
    print("=" * 70 + "\n")

    threshold_results = {}
    for threshold in [0.1, 0.2, 0.3, 0.5]:
        print(f"  Threshold = {threshold}...")
        config_t = make_config(
            modality_dims, adaptive_fusion=False, observed_threshold=threshold
        )
        res = run_giman_cv(
            features,
            times,
            events,
            modality_dims,
            config_t,
            f"Phase 1+2 (threshold={threshold})",
        )
        threshold_results[str(threshold)] = res
        print(f"    C-index: {res['mean']:.4f} ± {res['std']:.4f}")
    results["threshold_sweep"] = threshold_results
    print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - start_time
    baseline_c = results["A_no_crossmodal"]["mean"]

    print("=" * 70)
    print("ADAPTIVE FUSION ABLATION SUMMARY")
    print("=" * 70)
    print(f"\n{'Incremental Ablation':}")
    print(f"{'Condition':<55s} {'C-index':>8s} {'95% CI':>18s} {'Δ':>8s}")
    print("-" * 95)
    for key in ["A_no_crossmodal", "C_phase1", "D_phase1_2", "E_full_adaptive"]:
        r = results[key]
        delta = "" if key == "A_no_crossmodal" else f"{r['mean'] - baseline_c:>+.4f}"
        ci_str = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        print(f"  {r['name']:<53s} {r['mean']:>8.4f} {ci_str:>18s} {delta:>8s}")

    print(f"\n{'Threshold Sensitivity':}")
    print(f"{'Threshold':<20s} {'C-index':>8s} {'95% CI':>18s}")
    print("-" * 50)
    for t, r in threshold_results.items():
        ci_str = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        print(f"  {t:<18s} {r['mean']:>8.4f} {ci_str:>18s}")

    # Gate analysis
    if "gate_analysis" in results.get("E_full_adaptive", {}):
        print(f"\n{'Gate Interpretability (Full Adaptive)':}")
        print(
            f"{'# Modalities':>15s} {'N patients':>12s} {'Mean Gate':>12s} {'Std':>8s}"
        )
        print("-" * 55)
        for key, vals in results["E_full_adaptive"]["gate_analysis"].items():
            n_mod = key.split("_")[0]
            print(
                f"  {n_mod:>13s} {vals['n_patients']:>12d} {vals['mean_gate']:>12.4f} {vals['std_gate']:>8.4f}"
            )

    print(f"\n  Total time: {elapsed:.0f}s ({elapsed / 60:.1f}min)")

    # Save results
    output_dir = project_root / "results" / "true_giman"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "ablation_adaptive_fusion.json"
    with open(results_path, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
            default=lambda x: x.tolist() if hasattr(x, "tolist") else x,
        )
    print(f"  Results saved: {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
