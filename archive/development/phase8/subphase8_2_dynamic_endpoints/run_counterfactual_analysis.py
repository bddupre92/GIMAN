"""
Counterfactual Analysis: Direct model perturbation for feature sensitivity.

Bypasses the Digital Twin v1 simulator (which dilutes single-patient perturbations
through shared graph attention). Instead, perturbs features directly in sigma-scaled
units and re-runs the trained neuro-fuzzy model to measure actual prediction changes.

Produces:
  - outputs/counterfactual_analysis/counterfactual_results.json
  - visualizations/publication_New/Figure12_Counterfactual_Trajectory.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(
    0,
    str(
        PROJECT_ROOT
        / "archive"
        / "development"
        / "phase8"
        / "subphase8_2_dynamic_endpoints"
    ),
)

from giman_pipeline.explainability.real_data_explain import (  # noqa: E402
    _load_nf_model,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/phase9_neuro_fuzzy/PREP_20260208_SAA_COHORT3_full/neuro_fuzzy_best.pth"
)
TRAIN_DATA = PROJECT_ROOT / "data/03_prodromal/final_pyg_data_sota_run/train_data.pt"
TEST_DATA = PROJECT_ROOT / "data/03_prodromal/final_pyg_data_sota_run/test_data.pt"
METADATA = (
    PROJECT_ROOT / "data/03_prodromal/final_pyg_data_sota_run/pyg_data_metadata.json"
)

OUTPUT_JSON = PROJECT_ROOT / "outputs/counterfactual_analysis/counterfactual_results.json"
OUTPUT_FIGURE = (
    PROJECT_ROOT
    / "visualizations/publication_New/Figure12_Counterfactual_Trajectory.png"
)

# Features to test, grouped by modality for colour-coding
FEATURE_GROUPS = {
    "CSF": ["ALPHA_SYNUCLEIN", "TOTAL_TAU", "ABETA42", "PTAU181"],
    "Clinical": ["TREMOR_SCORE", "PIGD_SCORE", "SCOPA_AUT_TOTAL", "ESS_TOTAL",
                  "UPSIT_TOTAL", "RBD_TOTAL"],
    "Genetic": ["GBA", "SNCA", "LRRK2", "APOE_E4", "GENETIC_RISK_SCORE"],
    "Imaging": [
        "CAUDATE_L_VOL", "CAUDATE_R_VOL", "PUTAMEN_L_VOL", "PUTAMEN_R_VOL",
        "HIPPOCAMPUS_L_VOL", "HIPPOCAMPUS_R_VOL",
        "CAUDATE_L_SBR", "CAUDATE_R_SBR", "PUTAMEN_L_SBR", "PUTAMEN_R_SBR",
        "CAUDATE_ASYMMETRY", "PUTAMEN_ASYMMETRY",
        "ENTORHINAL_L_CTH", "ENTORHINAL_R_CTH",
        "CINGULATE_L_CTH", "CINGULATE_R_CTH",
        "PRECENTRAL_L_CTH", "PRECENTRAL_R_CTH",
    ],
}

# Perturbation magnitudes in sigma units
SIGMA_LEVELS = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])

# Modality colours for the bar chart
MODALITY_COLORS = {
    "CSF": "#2ca02c",       # green
    "Clinical": "#ff7f0e",  # orange
    "Genetic": "#d62728",   # red
    "Imaging": "#1f77b4",   # blue
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _predict_probs_from_model(model, data) -> np.ndarray:
    """Return SAA class-1 probabilities for all samples."""
    with torch.no_grad():
        logits, _ = model(data)
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    return probs


def _feature_to_modality(fname: str) -> str:
    for mod, feats in FEATURE_GROUPS.items():
        if fname in feats:
            return mod
    return "Other"


# ---------------------------------------------------------------------------
# Core counterfactual sweep
# ---------------------------------------------------------------------------
def run_counterfactual_sweep(
    model,
    test_data,
    feature_names: list[str],
    feature_stds: np.ndarray,
) -> dict:
    """
    For every (feature, sigma_level, patient), perturb and record Δprob.
    Returns a structured dict of results.
    """
    device = test_data.x.device
    x_orig = test_data.x.detach().cpu().numpy()
    n_samples = x_orig.shape[0]

    # Baseline predictions
    baseline_probs = _predict_probs_from_model(model, test_data)

    # Only test non-modality-indicator features that have non-zero std
    skip_features = {
        "modality_present_imaging",
        "modality_present_genetic",
        "modality_present_csf",
        "modality_present_clinical",
    }
    testable_features = []
    for i, fname in enumerate(feature_names):
        if fname in skip_features:
            continue
        if feature_stds[i] < 1e-8:
            continue
        testable_features.append((i, fname))

    # Results storage
    # sweep[feature_name][sigma_level_str] = list of Δprob per patient
    sweep: dict[str, dict[str, list[float]]] = {}

    for feat_idx, feat_name in testable_features:
        sweep[feat_name] = {}
        for sigma in SIGMA_LEVELS:
            sigma_key = f"{sigma:+.1f}"
            deltas: list[float] = []

            if sigma == 0.0:
                deltas = [0.0] * n_samples
            else:
                # Perturb each patient independently
                for patient_i in range(n_samples):
                    x_pert = x_orig.copy()
                    x_pert[patient_i, feat_idx] += sigma * feature_stds[feat_idx]

                    temp_data = test_data.clone()
                    temp_data.x = torch.tensor(
                        x_pert, dtype=torch.float32, device=device
                    )
                    pert_probs = _predict_probs_from_model(model, temp_data)
                    deltas.append(float(pert_probs[patient_i] - baseline_probs[patient_i]))

            sweep[feat_name][sigma_key] = deltas

    # Aggregate: mean |Δprob| at ±1σ for ranking
    sensitivity_at_1sigma: dict[str, dict] = {}
    for feat_name, sigma_dict in sweep.items():
        deltas_neg1 = np.array(sigma_dict["-1.0"])
        deltas_pos1 = np.array(sigma_dict["+1.0"])
        abs_deltas = np.concatenate([np.abs(deltas_neg1), np.abs(deltas_pos1)])
        sensitivity_at_1sigma[feat_name] = {
            "mean_abs_delta": float(np.mean(abs_deltas)),
            "std_abs_delta": float(np.std(abs_deltas)),
            "median_abs_delta": float(np.median(abs_deltas)),
            "max_abs_delta": float(np.max(abs_deltas)),
            "modality": _feature_to_modality(feat_name),
        }

    # Find highest-risk SAA+ patient for Panel A
    saa_labels = test_data.saa_label.detach().cpu().numpy().astype(int)
    saa_pos_mask = saa_labels == 1
    if saa_pos_mask.any():
        saa_pos_indices = np.where(saa_pos_mask)[0]
        best_idx = saa_pos_indices[np.argmax(baseline_probs[saa_pos_indices])]
    else:
        best_idx = int(np.argmax(baseline_probs))

    patno = (
        int(test_data.patno[best_idx].item())
        if hasattr(test_data, "patno")
        else int(best_idx)
    )

    return {
        "baseline_probs": baseline_probs.tolist(),
        "saa_labels": saa_labels.tolist(),
        "sweep": sweep,
        "sensitivity_at_1sigma": sensitivity_at_1sigma,
        "representative_patient_idx": int(best_idx),
        "representative_patno": patno,
        "sigma_levels": SIGMA_LEVELS.tolist(),
        "feature_names": feature_names,
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------
def generate_figure(results: dict) -> None:
    """Two-panel publication figure for counterfactual analysis."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "figure.dpi": 300,
    })

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Counterfactual Feature Sensitivity Analysis",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    sweep = results["sweep"]
    sensitivity = results["sensitivity_at_1sigma"]
    patient_idx = results["representative_patient_idx"]
    patno = results["representative_patno"]
    sigma_levels = np.array(results["sigma_levels"])
    baseline_prob = results["baseline_probs"][patient_idx]

    # ---- Panel A: Representative patient trajectories ----
    # Top 6 features by sensitivity
    ranked_features = sorted(
        sensitivity.keys(), key=lambda f: sensitivity[f]["mean_abs_delta"], reverse=True
    )
    top_features = ranked_features[:6]

    colors_panel_a = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    for i, feat_name in enumerate(top_features):
        probs_at_sigmas = []
        for sigma in sigma_levels:
            sigma_key = f"{sigma:+.1f}"
            if sigma == 0.0:
                probs_at_sigmas.append(baseline_prob)
            else:
                delta = sweep[feat_name][sigma_key][patient_idx]
                probs_at_sigmas.append(baseline_prob + delta)

        ax_a.plot(
            sigma_levels,
            probs_at_sigmas,
            marker="o",
            markersize=5,
            linewidth=2.0,
            color=colors_panel_a[i],
            label=feat_name,
            alpha=0.85,
        )

    # Baseline reference line
    ax_a.axhline(
        y=baseline_prob,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.5,
        label=f"Baseline (p={baseline_prob:.3f})",
    )
    ax_a.axvline(x=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)

    ax_a.set_xlabel("Perturbation Magnitude (σ units)", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Predicted SAA Probability", fontsize=11, fontweight="bold")
    ax_a.set_title(
        f"Panel A: What-If Trajectories (Patient {patno})",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax_a.set_xlim(-2.3, 2.3)
    ax_a.grid(True, alpha=0.25, linestyle="--")
    ax_a.legend(loc="best", fontsize=8, frameon=True, shadow=True)

    # ---- Panel B: Population-level feature sensitivity ----
    # Show top 12 features sorted by effect size
    top_for_bar = ranked_features[:12]
    top_for_bar.reverse()  # So highest is at top

    bar_vals = np.array([sensitivity[f]["mean_abs_delta"] for f in top_for_bar])
    bar_errs_raw = np.array([sensitivity[f]["std_abs_delta"] for f in top_for_bar])
    # Clip lower error bar so bars don't extend below zero
    bar_errs_lower = np.minimum(bar_errs_raw, bar_vals)
    bar_errs = np.array([bar_errs_lower, bar_errs_raw])
    bar_colors = [MODALITY_COLORS.get(sensitivity[f]["modality"], "#7f7f7f") for f in top_for_bar]

    y_pos = np.arange(len(top_for_bar))
    ax_b.barh(
        y_pos,
        bar_vals,
        xerr=bar_errs,
        color=bar_colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        capsize=3,
    )
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(top_for_bar, fontsize=9)
    ax_b.set_xlabel(
        "Mean |Δ Probability| at ±1σ", fontsize=11, fontweight="bold"
    )
    ax_b.set_title(
        "Panel B: Population Feature Sensitivity (n=32)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax_b.set_xlim(left=0)
    ax_b.grid(True, axis="x", alpha=0.25, linestyle="--")

    # Modality legend for Panel B
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=MODALITY_COLORS["CSF"], edgecolor="black", linewidth=0.5, label="CSF"),
        Patch(facecolor=MODALITY_COLORS["Clinical"], edgecolor="black", linewidth=0.5, label="Clinical"),
        Patch(facecolor=MODALITY_COLORS["Genetic"], edgecolor="black", linewidth=0.5, label="Genetic"),
        Patch(facecolor=MODALITY_COLORS["Imaging"], edgecolor="black", linewidth=0.5, label="Imaging"),
    ]
    ax_b.legend(handles=legend_elements, loc="lower right", fontsize=8, frameon=True)

    plt.tight_layout()
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved figure: {OUTPUT_FIGURE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("COUNTERFACTUAL ANALYSIS: Direct Model Perturbation")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load metadata
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    feature_names: list[str] = metadata["feature_names"]
    n_features = len(feature_names)
    print(f"Features: {n_features}")

    # Load data
    train_data = torch.load(TRAIN_DATA, weights_only=False).to(device)
    test_data = torch.load(TEST_DATA, weights_only=False).to(device)
    print(f"Train: {train_data.x.shape[0]} samples, Test: {test_data.x.shape[0]} samples")

    # Compute feature standard deviations from training data
    x_train = train_data.x.detach().cpu().numpy()
    feature_stds = np.std(x_train, axis=0, ddof=0)
    print(f"Feature stds range: [{feature_stds.min():.4f}, {feature_stds.max():.4f}]")
    nonzero_std = (feature_stds > 1e-8).sum()
    print(f"Features with non-zero std: {nonzero_std}/{n_features}")

    # Load model
    model = _load_nf_model(
        in_features=n_features,
        checkpoint_path=CHECKPOINT,
        device=device,
    )
    print("Model loaded successfully")

    # Run counterfactual sweep
    print("\nRunning counterfactual sweep...")
    results = run_counterfactual_sweep(model, test_data, feature_names, feature_stds)

    # Print top features by sensitivity
    print("\n--- Top 10 Features by Sensitivity (mean |Δprob| at ±1σ) ---")
    sensitivity = results["sensitivity_at_1sigma"]
    ranked = sorted(
        sensitivity.items(), key=lambda kv: kv[1]["mean_abs_delta"], reverse=True
    )
    for rank, (feat, stats) in enumerate(ranked[:10], 1):
        print(
            f"  {rank:2d}. {feat:25s}  mean|Δ|={stats['mean_abs_delta']:.4f}  "
            f"std={stats['std_abs_delta']:.4f}  max={stats['max_abs_delta']:.4f}  "
            f"({stats['modality']})"
        )

    patient_idx = results["representative_patient_idx"]
    patno = results["representative_patno"]
    baseline_p = results["baseline_probs"][patient_idx]
    print(f"\nRepresentative patient: PATNO {patno} (idx={patient_idx}, baseline p={baseline_p:.4f})")

    # Save results JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Convert sweep to serializable form (already is, but be safe)
    serializable = {
        k: v for k, v in results.items()
    }
    OUTPUT_JSON.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"\nSaved results: {OUTPUT_JSON}")

    # Generate figure
    print("\nGenerating Figure 12...")
    generate_figure(results)

    print("\nDone!")


if __name__ == "__main__":
    main()
