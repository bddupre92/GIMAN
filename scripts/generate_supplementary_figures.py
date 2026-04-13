"""Generate supplementary publication figures:
- Figure 2: Accurate Neuro-Fuzzy Architecture Diagram
- Appendix: ROC & PR Curves
- Appendix: Confusion Matrix at Optimal Threshold
- Appendix: Permutation Importance Ranking
- Appendix: Training Convergence Curves (re-train short run to capture history)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(
    0,
    str(ROOT / "archive/development/phase8/subphase8_2_dynamic_endpoints"),
)

from giman_pipeline.explainability.real_data_explain import (
    _load_nf_model,
)

OUT = ROOT / "visualizations/publication_New"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINT = (
    ROOT
    / "outputs/phase9_neuro_fuzzy/PREP_20260208_SAA_COHORT3_full/neuro_fuzzy_best.pth"
)
TRAIN_DATA = ROOT / "data/03_prodromal/final_pyg_data_sota_run/train_data.pt"
TEST_DATA = ROOT / "data/03_prodromal/final_pyg_data_sota_run/test_data.pt"
METADATA = ROOT / "data/03_prodromal/final_pyg_data_sota_run/pyg_data_metadata.json"
PERM_IMP_CSV = (
    ROOT / "visualizations/appendix/explainability/feature_permutation_importance.csv"
)


def _setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "figure.dpi": 300,
        }
    )


# =========================================================================
# Figure 2: Accurate Neuro-Fuzzy Architecture Diagram
# =========================================================================
def generate_figure2():
    """Regenerate Figure 2 to match the actual Phase 9 architecture."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.suptitle(
        "Neuro-Fuzzy GIMAN Architecture (Phase 9)",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    # Color scheme
    c_input = "#E8D5F5"  # lavender - input
    c_gat = "#B39DDB"  # purple - GAT
    c_fuzz = "#FFE0B2"  # orange - fuzzy
    c_rule = "#FFCC80"  # darker orange - rules
    c_ts = "#C8E6C9"  # green - Takagi-Sugeno
    c_out = "#FFCDD2"  # red - output
    edge = "black"

    def box(x, y, w, h, color, label, sublabel="", fontsize=10):
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor=edge,
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2 + (0.12 if sublabel else 0),
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
        )
        if sublabel:
            ax.text(
                x + w / 2,
                y + h / 2 - 0.22,
                sublabel,
                ha="center",
                va="center",
                fontsize=8,
                fontstyle="italic",
                color="#555555",
            )

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"),
        )

    # Layout: left to right
    # 1. Input features
    box(
        0.3,
        1.8,
        1.8,
        1.4,
        c_input,
        "Input\nFeatures",
        "x \u2208 \u211d\u00b3\u2077",
        11,
    )

    # 2. Patient Similarity Graph
    box(2.6, 1.8, 1.6, 1.4, c_input, "kNN Graph", "k=10, cosine", 10)
    arrow(2.1, 2.5, 2.6, 2.5)

    # 3. GAT Encoder (3-layer)
    box(4.7, 1.8, 2.0, 1.4, c_gat, "GAT Encoder", "3 layers, H=128, 4 heads", 11)
    arrow(4.2, 2.5, 4.7, 2.5)

    # 4. Fuzzification
    box(7.2, 1.8, 1.8, 1.4, c_fuzz, "Fuzzification", "\u03bc = Gaussian(c, \u03c3)", 10)
    arrow(6.7, 2.5, 7.2, 2.5)

    # 5. Rule Evaluation
    box(9.5, 1.8, 1.6, 1.4, c_rule, "Rule Layer", "R=32, mean T-norm", 10)
    arrow(9.0, 2.5, 9.5, 2.5)

    # 6. Takagi-Sugeno Defuzzification
    box(11.6, 1.8, 1.8, 1.4, c_ts, "Defuzzification", "Takagi-Sugeno", 10)
    arrow(11.1, 2.5, 11.6, 2.5)

    # Outputs (two boxes)
    box(11.0, 0.1, 1.2, 0.9, c_out, "SAA\nClass", "", 9)
    box(12.4, 0.1, 1.2, 0.9, c_out, "Risk\nScore", "", 9)
    arrow(12.5, 1.8, 11.6, 1.0)
    arrow(12.5, 1.8, 13.0, 1.0)

    # Annotations
    ax.text(
        5.7,
        4.2,
        "Embedding: h\u1d62 \u2208 \u211d\u00b9\u00b2\u2078",
        fontsize=9,
        ha="center",
        color="#6A1B9A",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="#6A1B9A", alpha=0.9
        ),
    )
    arrow(5.7, 4.0, 5.7, 3.2)

    ax.text(
        8.1,
        4.2,
        "Membership:\n\u03bc \u2208 [0,1]\u00b9\u00b2\u2078\u02e3\u00b3\u00b2",
        fontsize=8,
        ha="center",
        color="#E65100",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="#E65100", alpha=0.9
        ),
    )
    arrow(8.1, 3.8, 8.1, 3.2)

    ax.text(
        10.3,
        4.2,
        "Normalized\nfiring strengths",
        fontsize=8,
        ha="center",
        color="#BF360C",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="#BF360C", alpha=0.9
        ),
    )
    arrow(10.3, 3.8, 10.3, 3.2)

    # Legend
    legend_items = [
        mpatches.Patch(facecolor=c_input, edgecolor=edge, label="Input/Graph"),
        mpatches.Patch(facecolor=c_gat, edgecolor=edge, label="Neural (GAT)"),
        mpatches.Patch(facecolor=c_fuzz, edgecolor=edge, label="Fuzzy Layer"),
        mpatches.Patch(facecolor=c_ts, edgecolor=edge, label="Defuzzification"),
        mpatches.Patch(facecolor=c_out, edgecolor=edge, label="Output"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower left",
        fontsize=9,
        ncol=5,
        frameon=True,
        fancybox=True,
    )

    plt.tight_layout()
    path = OUT / "Figure2_Neuro_Fuzzy_Enhancement.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {path}")


# =========================================================================
# ROC & PR Curves
# =========================================================================
def generate_roc_pr_curves(model, test_data, y_test, probs):
    """Two-panel: ROC curve + PR curve for main model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Classification Performance Curves (Neuro-Fuzzy GIMAN)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # ROC
    fpr, tpr, thresholds_roc = roc_curve(y_test, probs)
    auc_val = roc_auc_score(y_test, probs)
    ax1.plot(
        fpr, tpr, lw=2.5, color="#1f77b4", label=f"Fuzzy GIMAN (AUC = {auc_val:.3f})"
    )
    ax1.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random (AUC = 0.500)")
    ax1.fill_between(fpr, tpr, alpha=0.15, color="#1f77b4")

    # Find optimal threshold (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds_roc[best_idx]
    ax1.scatter(
        fpr[best_idx],
        tpr[best_idx],
        s=120,
        c="red",
        zorder=5,
        edgecolor="black",
        linewidth=1.5,
    )
    ax1.annotate(
        f"Optimal\n(t={best_thresh:.2f})",
        xy=(fpr[best_idx], tpr[best_idx]),
        xytext=(fpr[best_idx] + 0.15, tpr[best_idx] - 0.15),
        fontsize=9,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.9
        ),
    )

    ax1.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    ax1.set_title("Panel A: ROC Curve", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.legend(loc="lower right", fontsize=9, frameon=True)

    # PR Curve
    precision, recall, thresholds_pr = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    prevalence = y_test.mean()
    ax2.plot(
        recall, precision, lw=2.5, color="#2ca02c", label=f"Fuzzy GIMAN (AP = {ap:.3f})"
    )
    ax2.axhline(
        y=prevalence,
        color="gray",
        linestyle="--",
        lw=1.2,
        alpha=0.6,
        label=f"Baseline (prevalence = {prevalence:.2f})",
    )
    ax2.fill_between(recall, precision, alpha=0.15, color="#2ca02c")

    ax2.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax2.set_title(
        "Panel B: Precision-Recall Curve", fontsize=11, fontweight="bold", pad=10
    )
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.05)
    ax2.grid(True, alpha=0.25, linestyle="--")
    ax2.legend(loc="upper right", fontsize=9, frameon=True)

    plt.tight_layout()
    path = OUT / "appendix_roc_pr_curves.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {path}")
    return best_thresh


# =========================================================================
# Confusion Matrix
# =========================================================================
def generate_confusion_matrix(y_test, probs, threshold):
    """Confusion matrix at optimal (Youden's J) threshold."""
    y_pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.3]}
    )
    fig.suptitle(
        f"Classification at Optimal Threshold (t = {threshold:.3f})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Heatmap
    im = ax1.imshow(cm, interpolation="nearest", cmap="Blues")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["SAA-", "SAA+"], fontsize=11)
    ax1.set_yticklabels(["SAA-", "SAA+"], fontsize=11)
    ax1.set_xlabel("Predicted", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Actual", fontsize=11, fontweight="bold")
    ax1.set_title("Panel A: Confusion Matrix", fontsize=11, fontweight="bold", pad=10)

    # Annotate cells
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax1.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color=color,
            )

    fig.colorbar(im, ax=ax1, shrink=0.7)

    # Classification report as text table
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0

    metrics = [
        ("Sensitivity (Recall)", f"{sensitivity:.3f}"),
        ("Specificity", f"{specificity:.3f}"),
        ("PPV (Precision)", f"{ppv:.3f}"),
        ("NPV", f"{npv:.3f}"),
        ("Accuracy", f"{accuracy:.3f}"),
        ("F1 Score", f"{f1:.3f}"),
        ("", ""),
        ("True Positives", f"{tp}"),
        ("True Negatives", f"{tn}"),
        ("False Positives", f"{fp}"),
        ("False Negatives", f"{fn}"),
    ]

    ax2.axis("off")
    ax2.set_title(
        "Panel B: Classification Metrics", fontsize=11, fontweight="bold", pad=10
    )
    table = ax2.table(
        cellText=[[m, v] for m, v in metrics],
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="center",
        colWidths=[0.55, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif row == 8:  # Separator row
            cell.set_facecolor("#f0f0f0")

    plt.tight_layout()
    path = OUT / "appendix_confusion_matrix.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {path}")


# =========================================================================
# Permutation Importance
# =========================================================================
def generate_permutation_importance():
    """Horizontal bar chart of permutation importance from existing CSV."""
    df = pd.read_csv(PERM_IMP_CSV)
    df = df.sort_values("auc_drop", ascending=True)

    # Color by positive/negative contribution
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in df["auc_drop"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(df))
    ax.barh(
        y_pos,
        df["auc_drop"],
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"], fontsize=9)
    ax.set_xlabel("AUC Drop When Feature Permuted", fontsize=11, fontweight="bold")
    ax.set_title(
        "Permutation Feature Importance (Neuro-Fuzzy GIMAN)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
    ax.grid(True, axis="x", alpha=0.25, linestyle="--")

    # Annotate top feature
    top_feat = df.iloc[-1]
    ax.annotate(
        f"  {top_feat['auc_drop']:.3f}",
        xy=(top_feat["auc_drop"], len(df) - 1),
        fontsize=9,
        fontweight="bold",
        color="#2ca02c",
        va="center",
    )

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="#2ca02c",
            edgecolor="black",
            linewidth=0.5,
            label="Positive importance (AUC drops)",
        ),
        Patch(
            facecolor="#d62728",
            edgecolor="black",
            linewidth=0.5,
            label="Negative (AUC improves when permuted)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, frameon=True)

    plt.tight_layout()
    path = OUT / "appendix_permutation_importance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {path}")


# =========================================================================
# Training Convergence Curves
# =========================================================================
def generate_convergence_curves(model, train_data, test_data, y_train, y_test):
    """Run a short re-training (50 epochs) from checkpoint to capture
    loss/AUC convergence curves for illustration.
    Instead of full 220 epochs, we show the characteristic convergence shape.
    """
    from sklearn.metrics import roc_auc_score

    device = train_data.x.device

    # We'll record metrics during a short training run from scratch
    # (to show the characteristic convergence)
    # Load a fresh model and train briefly
    from train_final_giman_survival import GIMANSurvivalGAT

    from archive.development.phase9.neuro_fuzzy import NeuroFuzzyGIMAN

    gat = GIMANSurvivalGAT(in_features=int(train_data.x.shape[1]), hidden_dim=128)
    fresh_model = NeuroFuzzyGIMAN(gat, num_classes=2, num_rules=32).to(device)

    # Load the actual checkpoint
    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    fresh_model.load_state_dict(state)

    optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=5e-3, weight_decay=1e-4)

    # Record convergence from fine-tuning around the checkpoint
    epochs_list = []
    train_losses = []
    test_aucs = []
    train_aucs = []

    for epoch in range(1, 61):
        fresh_model.train()
        optimizer.zero_grad()
        logits, _ = fresh_model(train_data)
        labels = train_data.saa_label.long().to(device)

        # Focal loss
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** 2.0 * ce).mean()
        focal.backward()
        optimizer.step()

        # Eval
        fresh_model.eval()
        with torch.no_grad():
            train_logits, _ = fresh_model(train_data)
            train_probs = F.softmax(train_logits, dim=1)[:, 1].cpu().numpy()
            test_logits, _ = fresh_model(test_data)
            test_probs = F.softmax(test_logits, dim=1)[:, 1].cpu().numpy()

        t_auc = (
            roc_auc_score(y_train, train_probs) if len(np.unique(y_train)) > 1 else 0.5
        )
        e_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 0.5

        epochs_list.append(epoch)
        train_losses.append(float(focal.item()))
        train_aucs.append(t_auc)
        test_aucs.append(e_auc)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Training Convergence (Neuro-Fuzzy GIMAN, from checkpoint)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    ax1.plot(epochs_list, train_losses, lw=2.0, color="#1f77b4", label="Focal Loss")
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax1.set_title("Panel A: Training Loss", fontsize=11, fontweight="bold", pad=10)
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.legend(fontsize=9)

    ax2.plot(
        epochs_list, train_aucs, lw=2.0, color="#2ca02c", label="Train AUC", alpha=0.8
    )
    ax2.plot(
        epochs_list, test_aucs, lw=2.0, color="#d62728", label="Test AUC", alpha=0.8
    )
    ax2.axhline(
        y=0.8646,
        color="gray",
        linestyle="--",
        lw=1.2,
        alpha=0.6,
        label="Reported Best (0.865)",
    )
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("AUC-ROC", fontsize=11, fontweight="bold")
    ax2.set_title("Panel B: AUC Convergence", fontsize=11, fontweight="bold", pad=10)
    ax2.set_ylim(0.4, 1.02)
    ax2.grid(True, alpha=0.25, linestyle="--")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    path = OUT / "appendix_training_convergence.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {path}")


# =========================================================================
# Main
# =========================================================================
def main():
    _setup_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("GENERATING SUPPLEMENTARY FIGURES")
    print("=" * 70)

    # 1. Figure 2
    print("\n[1/5] Figure 2: Neuro-Fuzzy Architecture...")
    generate_figure2()

    # Load model and data for remaining figures
    print("\n    Loading model and data...")
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    model = _load_nf_model(
        in_features=len(metadata["feature_names"]),
        checkpoint_path=CHECKPOINT,
        device=device,
    )
    train_data = torch.load(TRAIN_DATA, weights_only=False).to(device)
    test_data = torch.load(TEST_DATA, weights_only=False).to(device)
    y_train = train_data.saa_label.detach().cpu().numpy().astype(int)
    y_test = test_data.saa_label.detach().cpu().numpy().astype(int)

    with torch.no_grad():
        logits, _ = model(test_data)
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    # 2. ROC/PR Curves
    print("\n[2/5] ROC & PR Curves...")
    best_thresh = generate_roc_pr_curves(model, test_data, y_test, probs)

    # 3. Confusion Matrix
    print(f"\n[3/5] Confusion Matrix (threshold={best_thresh:.3f})...")
    generate_confusion_matrix(y_test, probs, best_thresh)

    # 4. Permutation Importance
    print("\n[4/5] Permutation Importance...")
    generate_permutation_importance()

    # 5. Training Convergence
    print("\n[5/5] Training Convergence Curves...")
    generate_convergence_curves(model, train_data, test_data, y_train, y_test)

    print("\n" + "=" * 70)
    print("ALL SUPPLEMENTARY FIGURES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
