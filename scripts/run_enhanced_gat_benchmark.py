#!/usr/bin/env python3
"""Run Enhanced Multimodal GAT benchmark on NSD-ISS stage prediction targets.

Uses the EXISTING GIMAN codebase architecture:
  - PyG GATConv layers (from graph_attention_network.py)
  - Cross-modal attention fusion (from cross_modal_attention.py)
  - k-NN patient similarity graphs (from patient_similarity.py)

Splits 22 features into 2 clinically meaningful modalities:
  - Clinical modality (12): demographics + motor + cognitive + sleep + autonomic
  - Biomarker modality (10): DaT-SPECT SBR + olfaction + genetics

Each modality → embedding → 3-layer PyG GATConv → cross-modal attention → classifier

Targets: binary, three_class, full_ordinal, nsd_positive
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops, to_undirected

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = BASE / "outputs" / "paper1_enhanced_gat"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Modality-split feature sets ──
# Modality 1: Clinical (demographics + motor + cognitive + sleep + autonomic)
CLINICAL_FEATURES = [
    "SEX",
    "HANDED",
    "AGE_AT_BASELINE",
    "UPDRS1_TOTAL",
    "UPDRS2_TOTAL",
    "UPDRS3_TOTAL",
    "UPDRS4_TOTAL",
    "UPDRS3_RIGIDITY",
    "UPDRS3_BRADYKINESIA",
    "UPDRS3_TREMOR",
    "UPDRS3_POSTURE_GAIT",
    "MOCA_TOTAL",
    "ESS_TOTAL",
    "RBD_TOTAL",
    "SCOPA_AUT_TOTAL",
]

# Modality 2: Biomarker (DaT-SPECT + olfaction + genetics)
BIOMARKER_FEATURES = [
    "CAUDATE_LEFT_SBR",
    "CAUDATE_RIGHT_SBR",
    "CAUDATE_MEAN_SBR",
    "PUTAMEN_LEFT_SBR",
    "PUTAMEN_RIGHT_SBR",
    "PUTAMEN_MEAN_SBR",
    "UPSIT_TOTAL",
    # "GBA_CARRIER", "LRRK2_CARRIER", "APOE_GENOTYPE",
    # ^ commented out — these are in the 22-feature set but have very low variance
]

# Also run on clinical-only (single modality) for comparison
CLINICAL_ONLY = CLINICAL_FEATURES.copy()

TARGETS = {
    "binary": "target_binary",
    "three_class": "target_3class",
    "full_ordinal": "target_full_ordinal",
    "nsd_positive": "target_nsd_positive",
}


# ═══════════════════════════════════════════════════════════════════════
# Graph Construction (PyG format)
# ═══════════════════════════════════════════════════════════════════════
def build_knn_graph_pyg(X: np.ndarray, k: int = 10) -> torch.Tensor:
    """Build k-NN graph as PyG edge_index from feature matrix using cosine similarity."""
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0)
    N = X.shape[0]

    src, dst = [], []
    for i in range(N):
        topk = np.argsort(sim[i])[-k:]
        for j in topk:
            src.append(i)
            dst.append(j)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    return edge_index


# ═══════════════════════════════════════════════════════════════════════
# Enhanced Multimodal GAT for NSD-ISS Classification
# ═══════════════════════════════════════════════════════════════════════
class ModalityEncoder(nn.Module):
    """Project raw features to embedding space."""

    def __init__(self, input_dim: int, embed_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiModalGATClassifier(nn.Module):
    """Adapted from GIMAN's MultiModalGraphAttention + CrossModalTransformer.

    Architecture:
      1. Modality-specific encoders (raw features → embeddings)
      2. Per-modality 3-layer PyG GATConv on patient similarity graph
      3. Cross-modal attention fusion (nn.MultiheadAttention)
      4. Classification head for NSD-ISS stages

    Uses PyG GATConv (not custom attention) — same layers as
    src/giman_pipeline/models/graph_attention_network.py
    """

    def __init__(
        self,
        clinical_dim: int,
        biomarker_dim: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_gat_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_modalities = 2

        # Modality encoders
        self.clinical_encoder = ModalityEncoder(clinical_dim, embed_dim, dropout)
        self.biomarker_encoder = ModalityEncoder(biomarker_dim, embed_dim, dropout)

        # Per-modality GAT layers (PyG GATConv)
        self.clinical_gat_layers = nn.ModuleList()
        self.biomarker_gat_layers = nn.ModuleList()
        self.clinical_norms = nn.ModuleList()
        self.biomarker_norms = nn.ModuleList()

        for i in range(num_gat_layers):
            in_dim = embed_dim
            if i < num_gat_layers - 1:
                out_per_head = hidden_dim // num_heads
                self.clinical_gat_layers.append(
                    GATConv(
                        in_dim,
                        out_per_head,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True,
                    )
                )
                self.biomarker_gat_layers.append(
                    GATConv(
                        in_dim,
                        out_per_head,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True,
                    )
                )
                self.clinical_norms.append(nn.LayerNorm(hidden_dim))
                self.biomarker_norms.append(nn.LayerNorm(hidden_dim))
            else:
                # Final layer: single head, output embed_dim
                self.clinical_gat_layers.append(
                    GATConv(in_dim, embed_dim, heads=1, dropout=dropout, concat=False)
                )
                self.biomarker_gat_layers.append(
                    GATConv(in_dim, embed_dim, heads=1, dropout=dropout, concat=False)
                )
                self.clinical_norms.append(nn.LayerNorm(embed_dim))
                self.biomarker_norms.append(nn.LayerNorm(embed_dim))

        self.dropout = nn.Dropout(dropout)

        # Cross-modal attention fusion
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_modal_norm = nn.LayerNorm(embed_dim)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _apply_gat_layers(self, x, edge_index, gat_layers, norms):
        """Apply GAT layers with residual connections."""
        for i, (gat, norm) in enumerate(zip(gat_layers, norms, strict=False)):
            h = gat(x, edge_index)
            h = norm(h)
            if i < len(gat_layers) - 1:
                h = F.elu(h)
                h = self.dropout(h)
            # Residual if shapes match
            if h.shape == x.shape:
                h = h + x
            x = h
        return x

    def forward(self, x_clinical, x_biomarker, edge_index):
        """Forward pass.

        Args:
            x_clinical: [N, clinical_dim] raw clinical features
            x_biomarker: [N, biomarker_dim] raw biomarker features
            edge_index: [2, num_edges] PyG edge index
        """
        # 1. Encode modalities
        h_clin = self.clinical_encoder(x_clinical)
        h_bio = self.biomarker_encoder(x_biomarker)

        # 2. Per-modality GAT
        h_clin = self._apply_gat_layers(
            h_clin, edge_index, self.clinical_gat_layers, self.clinical_norms
        )
        h_bio = self._apply_gat_layers(
            h_bio, edge_index, self.biomarker_gat_layers, self.biomarker_norms
        )

        # 3. Cross-modal attention
        stacked = torch.stack([h_clin, h_bio], dim=1)  # [N, 2, embed_dim]
        attended, attn_weights = self.cross_modal_attention(stacked, stacked, stacked)
        attended = self.cross_modal_norm(attended + stacked)  # residual

        # 4. Fusion
        fused = attended.flatten(start_dim=1)  # [N, 2*embed_dim]
        fused = self.fusion(fused)  # [N, embed_dim]

        # 5. Classify
        logits = self.classifier(fused)
        return logits, attn_weights


class SingleModalGATClassifier(nn.Module):
    """Single-modality GAT for clinical-only baseline."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_gat_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = ModalityEncoder(input_dim, embed_dim, dropout)

        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_gat_layers):
            in_dim = embed_dim
            if i < num_gat_layers - 1:
                out_per_head = hidden_dim // num_heads
                self.gat_layers.append(
                    GATConv(
                        in_dim,
                        out_per_head,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True,
                    )
                )
                self.norms.append(nn.LayerNorm(hidden_dim))
            else:
                self.gat_layers.append(
                    GATConv(in_dim, embed_dim, heads=1, dropout=dropout, concat=False)
                )
                self.norms.append(nn.LayerNorm(embed_dim))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        h = self.encoder(x)
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms, strict=False)):
            h_new = gat(h, edge_index)
            h_new = norm(h_new)
            if i < len(self.gat_layers) - 1:
                h_new = F.elu(h_new)
                h_new = self.dropout(h_new)
            if h_new.shape == h.shape:
                h_new = h_new + h
            h = h_new
        return self.classifier(h), None


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════
def train_model(
    model,
    X_clin_tr,
    X_bio_tr,
    y_tr,
    edge_index_tr,
    X_clin_val,
    X_bio_val,
    y_val,
    edge_index_val,
    num_classes,
    epochs=300,
    lr=5e-4,
    patience=40,
    is_multimodal=True,
):
    """Train with early stopping."""
    model = model.to(DEVICE)

    # Class weights
    class_counts = np.bincount(y_tr.astype(int), minlength=num_classes)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    class_weights = torch.FloatTensor(weights).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5
    )

    y_tr_t = torch.LongTensor(y_tr.astype(int)).to(DEVICE)
    y_val_t = torch.LongTensor(y_val.astype(int)).to(DEVICE)
    ei_tr = edge_index_tr.to(DEVICE)
    ei_val = edge_index_val.to(DEVICE)

    if is_multimodal:
        xc_tr = torch.FloatTensor(X_clin_tr).to(DEVICE)
        xb_tr = torch.FloatTensor(X_bio_tr).to(DEVICE)
        xc_val = torch.FloatTensor(X_clin_val).to(DEVICE)
        xb_val = torch.FloatTensor(X_bio_val).to(DEVICE)
    else:
        xc_tr = torch.FloatTensor(X_clin_tr).to(DEVICE)
        xc_val = torch.FloatTensor(X_clin_val).to(DEVICE)
        xb_tr = xb_val = None

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if is_multimodal:
            logits, _ = model(xc_tr, xb_tr, ei_tr)
        else:
            logits, _ = model(xc_tr, ei_tr)
        loss = criterion(logits, y_tr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            if is_multimodal:
                val_logits, _ = model(xc_val, xb_val, ei_val)
            else:
                val_logits, _ = model(xc_val, ei_val)
            val_loss = criterion(val_logits, y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_model(model, X_clin, X_bio, edge_index, is_multimodal=True):
    """Get predictions."""
    model.eval()
    with torch.no_grad():
        if is_multimodal:
            xc = torch.FloatTensor(X_clin).to(DEVICE)
            xb = torch.FloatTensor(X_bio).to(DEVICE)
            logits, attn = model(xc, xb, edge_index.to(DEVICE))
        else:
            xc = torch.FloatTensor(X_clin).to(DEVICE)
            logits, attn = model(xc, edge_index.to(DEVICE))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
    return preds, probs


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════
def run_benchmark(
    df, y, target_name, n_classes, model_type, feature_config, n_folds=5, k_neighbors=10
):
    """Run 5-fold CV for a given model configuration."""
    is_multimodal = model_type == "multimodal"
    feat_label = feature_config["label"]
    print(f"\n--- Enhanced GAT [{model_type}]: {target_name} ({feat_label}) ---")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Prepare features
    clin_cols = [c for c in feature_config["clinical"] if c in df.columns]
    X_clin_raw = df[clin_cols].values

    if is_multimodal:
        bio_cols = [c for c in feature_config["biomarker"] if c in df.columns]
        X_bio_raw = df[bio_cols].values
        X_all = np.hstack([X_clin_raw, X_bio_raw])
    else:
        bio_cols = []
        X_bio_raw = None
        X_all = X_clin_raw

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y)):
        # Split
        X_clin_train, X_clin_test = X_clin_raw[train_idx], X_clin_raw[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if is_multimodal:
            X_bio_train, X_bio_test = X_bio_raw[train_idx], X_bio_raw[test_idx]

        # Impute + scale (per-fold to prevent leakage)
        imp_clin = SimpleImputer(strategy="median")
        X_clin_train = imp_clin.fit_transform(X_clin_train)
        X_clin_test = imp_clin.transform(X_clin_test)

        scaler_clin = StandardScaler()
        X_clin_train = scaler_clin.fit_transform(X_clin_train)
        X_clin_test = scaler_clin.transform(X_clin_test)

        if is_multimodal:
            imp_bio = SimpleImputer(strategy="median")
            X_bio_train = imp_bio.fit_transform(X_bio_train)
            X_bio_test = imp_bio.transform(X_bio_test)

            scaler_bio = StandardScaler()
            X_bio_train = scaler_bio.fit_transform(X_bio_train)
            X_bio_test = scaler_bio.transform(X_bio_test)

        # Build per-fold k-NN graphs
        X_graph_train = (
            np.hstack([X_clin_train, X_bio_train]) if is_multimodal else X_clin_train
        )
        X_graph_test = (
            np.hstack([X_clin_test, X_bio_test]) if is_multimodal else X_clin_test
        )

        edge_index_train = build_knn_graph_pyg(X_graph_train, k=k_neighbors)
        edge_index_test = build_knn_graph_pyg(
            X_graph_test, k=min(k_neighbors, len(X_graph_test) - 1)
        )

        # Train/val split within training fold
        n_val = max(int(len(X_clin_train) * 0.2), 1)
        rng = np.random.RandomState(fold)
        val_mask = rng.choice(len(X_clin_train), n_val, replace=False)
        tr_mask = np.array([i for i in range(len(X_clin_train)) if i not in val_mask])

        X_clin_tr, X_clin_v = X_clin_train[tr_mask], X_clin_train[val_mask]
        y_tr, y_v = y_train[tr_mask], y_train[val_mask]
        X_graph_tr = X_graph_train[tr_mask]
        X_graph_v = X_graph_train[val_mask]

        ei_tr = build_knn_graph_pyg(X_graph_tr, k=k_neighbors)
        ei_v = build_knn_graph_pyg(X_graph_v, k=min(k_neighbors, len(X_graph_v) - 1))

        if is_multimodal:
            X_bio_tr, X_bio_v = X_bio_train[tr_mask], X_bio_train[val_mask]
            model = MultiModalGATClassifier(
                clinical_dim=len(clin_cols),
                biomarker_dim=len(bio_cols),
                embed_dim=128,
                hidden_dim=128,
                num_heads=4,
                num_gat_layers=3,
                num_classes=n_classes,
                dropout=0.2,
            )
        else:
            X_bio_tr = X_bio_v = None
            model = SingleModalGATClassifier(
                input_dim=len(clin_cols),
                embed_dim=128,
                hidden_dim=128,
                num_heads=4,
                num_gat_layers=3,
                num_classes=n_classes,
                dropout=0.2,
            )

        t0 = time.time()
        model = train_model(
            model,
            X_clin_tr,
            X_bio_tr,
            y_tr,
            ei_tr,
            X_clin_v,
            X_bio_v,
            y_v,
            ei_v,
            num_classes=n_classes,
            epochs=300,
            lr=5e-4,
            patience=40,
            is_multimodal=is_multimodal,
        )
        train_time = time.time() - t0

        # Predict on test fold
        if is_multimodal:
            preds, probs = predict_model(
                model, X_clin_test, X_bio_test, edge_index_test, True
            )
        else:
            preds, probs = predict_model(
                model, X_clin_test, None, edge_index_test, False
            )

        ba = balanced_accuracy_score(y_test, preds)
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_test, probs[:, 1])
            else:
                auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")
        qwk = cohen_kappa_score(y_test, preds, weights="quadratic")

        fold_metrics.append(
            {
                "fold": fold,
                "bal_acc": ba,
                "auc": auc,
                "qwk": qwk,
                "train_time": train_time,
            }
        )
        print(
            f"  Fold {fold}: bal_acc={ba:.4f}, AUC={auc:.4f}, QWK={qwk:.4f} ({train_time:.1f}s)"
        )

    mean_ba = np.mean([m["bal_acc"] for m in fold_metrics])
    mean_auc = np.nanmean([m["auc"] for m in fold_metrics])
    mean_qwk = np.mean([m["qwk"] for m in fold_metrics])
    std_ba = np.std([m["bal_acc"] for m in fold_metrics])
    std_auc = np.nanstd([m["auc"] for m in fold_metrics])

    result = {
        "model": f"Enhanced-GAT-{model_type}",
        "target": target_name,
        "feature_set": feat_label,
        "n_clinical_features": len(clin_cols),
        "n_biomarker_features": len(bio_cols) if is_multimodal else 0,
        "n_total_features": len(clin_cols) + (len(bio_cols) if is_multimodal else 0),
        "n_patients": len(y),
        "n_classes": n_classes,
        "architecture": "MultiModalGATClassifier"
        if is_multimodal
        else "SingleModalGATClassifier",
        "pyg_gatconv_layers": 3,
        "num_heads": 4,
        "embed_dim": 128,
        "k_neighbors": 10,
        "bal_acc": round(mean_ba, 4),
        "bal_acc_std": round(std_ba, 4),
        "auc": round(mean_auc, 4),
        "auc_std": round(std_auc, 4),
        "qwk": round(mean_qwk, 4),
        "fold_metrics": fold_metrics,
    }
    print(
        f"  MEAN: bal_acc={mean_ba:.4f}±{std_ba:.4f}, AUC={mean_auc:.4f}±{std_auc:.4f}, QWK={mean_qwk:.4f}"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("Enhanced Multimodal GAT Benchmark on NSD-ISS Targets")
    print("Architecture: PyG GATConv + Cross-Modal Attention Fusion")
    print("=" * 70)

    ppmi = pd.read_csv(DATA / "05_features" / "paper1_features_with_targets.csv")
    print(f"Loaded PPMI: {len(ppmi)} patients")

    all_results = {}

    for target_name, target_col in TARGETS.items():
        target_dir = OUT / target_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # Filter valid samples AND exclude -1 (unclassified)
        mask = ppmi[target_col].notna() & (ppmi[target_col] >= 0)
        df = ppmi[mask].copy()
        y_raw = df[target_col].values.astype(int)

        # Remap labels to consecutive 0..K-1
        unique_labels = sorted(np.unique(y_raw))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y = np.array([label_map[v] for v in y_raw])
        n_classes = len(unique_labels)

        print(f"\n{'=' * 60}")
        print(f"Target: {target_name} | N={len(df)} | Classes={n_classes}")
        print(
            f"Class dist: {dict(zip(*np.unique(y, return_counts=True), strict=False))}"
        )
        print(f"Label map: {label_map}")
        print(f"{'=' * 60}")

        target_results = []

        # Configuration 1: Multimodal (clinical + biomarker)
        config_mm = {
            "label": "multimodal_22",
            "clinical": CLINICAL_FEATURES,
            "biomarker": BIOMARKER_FEATURES,
        }
        result = run_benchmark(
            df,
            y,
            target_name,
            n_classes,
            model_type="multimodal",
            feature_config=config_mm,
        )
        target_results.append(result)

        # Configuration 2: Clinical-only (single modality)
        config_clin = {
            "label": "clinical_only_15",
            "clinical": CLINICAL_ONLY,
            "biomarker": [],
        }
        result = run_benchmark(
            df,
            y,
            target_name,
            n_classes,
            model_type="single",
            feature_config=config_clin,
        )
        target_results.append(result)

        # Save per-target
        with open(target_dir / "enhanced_gat_results.json", "w") as f:
            json.dump(target_results, f, indent=2, default=str)
        print(f"Saved: {target_dir / 'enhanced_gat_results.json'}")

        all_results[target_name] = target_results

    # Save comprehensive results
    with open(OUT / "enhanced_gat_benchmark_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 80)
    print("ENHANCED MULTIMODAL GAT BENCHMARK SUMMARY")
    print("=" * 80)
    print(
        f"{'Target':<16} {'Model':<20} {'Features':<18} {'Bal Acc':<14} {'AUC':<14} {'QWK':<10}"
    )
    print("-" * 92)
    for target, results in all_results.items():
        for r in results:
            print(
                f"{target:<16} {r['model']:<20} {r['feature_set']:<18} "
                f"{r['bal_acc']:.4f}±{r['bal_acc_std']:.4f}  "
                f"{r['auc']:.4f}±{r['auc_std']:.4f}  {r['qwk']:.4f}"
            )

    print(f"\nAll results saved to: {OUT}")


if __name__ == "__main__":
    main()
