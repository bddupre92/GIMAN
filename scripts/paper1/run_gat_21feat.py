#!/usr/bin/env python3
"""Run Simple GAT and Multimodal GAT on the Paper 1 Path 3 21-feature
strict-circularity primary feature set, for all 4 NSD-ISS targets.

Reviewer 4 (R4-Q3): Table III's "Graph-based" subblock previously used the
22-feat reference spec (which includes CAUDATE_PUTAMEN_RATIO). The tabular
benchmarks use the 21-feat primary (which excludes the ratio per Path 3
strict circularity). This script removes that asymmetry by re-running the
graph models on the SAME 21-feat primary spec.

Spec source of truth: outputs/paper1_hpo_21feat/results/nested_catboost_binary.json
'feature_cols' field. The 21 nominal features collapse to 19 CSV columns
because SEX and HANDED are encoded as single columns each (they're listed
as binary 'indicators' in the manifest).

Hyperparameters: identical to scripts/run_giman_gat_benchmark.py and
scripts/run_enhanced_gat_benchmark.py (k=10 kNN, 4 attention heads, GAT depth
matched per architecture, embed_dim=128, fold-local graph construction,
random_state=42 throughout).

Outputs:
  outputs/paper1_gat_21feat/{target}/gat_results.json
  outputs/paper1_enhanced_gat_21feat/{target}/enhanced_gat_results.json
  outputs/paper1_gat_21feat/summary.json
  outputs/paper1_enhanced_gat_21feat/summary.json
"""

from __future__ import annotations

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

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data"
OUT_SIMPLE = BASE / "outputs" / "paper1_gat_21feat"
OUT_ENHANCED = BASE / "outputs" / "paper1_enhanced_gat_21feat"
OUT_SIMPLE.mkdir(parents=True, exist_ok=True)
OUT_ENHANCED.mkdir(parents=True, exist_ok=True)

# Prefer MPS on Apple silicon (consistent with other Paper 1 GAT runs); fall
# back to CUDA → CPU. The original 22-feat scripts used CUDA-or-CPU only —
# MPS gives a noticeable speedup and matches the rest of the Paper 1 pipeline.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# ── 21-feat Path 3 primary spec ──
# Source: outputs/paper1_hpo_21feat/results/nested_catboost_binary.json (feature_cols)
# Excluded vs 22-feat: CAUDATE_PUTAMEN_RATIO (per Path 3 strict circularity)
# Already excluded vs 33-feat: PUTAMEN_*_SBR, NP3TOT, NP1COG, MOCA_TOTAL, UPDRS4_TOTAL
PRIMARY_21_FEATURES = [
    "SEX",
    "HANDED",
    "AGE_AT_BASELINE",
    "UPDRS1_TOTAL",
    "UPDRS2_TOTAL",
    "UPDRS3_TREMOR",
    "UPDRS3_RIGIDITY",
    "UPDRS3_BRADYKINESIA",
    "UPDRS3_AXIAL",
    "RBD_TOTAL",
    "ESS_TOTAL",
    "SCOPA_AUT_TOTAL",
    "CAUDATE_R_SBR",
    "CAUDATE_L_SBR",
    "CAUDATE_MEAN_SBR",
    "CAUDATE_ASYMMETRY",
    "LRRK2_CARRIER",
    "GBA_CARRIER",
    "APOE_E4_CARRIER",
]

# For the multimodal split, we partition the 21-feat primary into two
# clinically meaningful modalities (mirroring the 22-feat MM-GAT split):
#   Clinical = demographics + UPDRS subscales + cognitive/sleep/autonomic
#   Biomarker = caudate DaT-SBR (4) + genetics (3)
# CAUDATE_PUTAMEN_RATIO is excluded from BOTH (per Path 3).
CLINICAL_21 = [
    "SEX",
    "HANDED",
    "AGE_AT_BASELINE",
    "UPDRS1_TOTAL",
    "UPDRS2_TOTAL",
    "UPDRS3_TREMOR",
    "UPDRS3_RIGIDITY",
    "UPDRS3_BRADYKINESIA",
    "UPDRS3_AXIAL",
    "RBD_TOTAL",
    "ESS_TOTAL",
    "SCOPA_AUT_TOTAL",
]
BIOMARKER_21 = [
    "CAUDATE_R_SBR",
    "CAUDATE_L_SBR",
    "CAUDATE_MEAN_SBR",
    "CAUDATE_ASYMMETRY",
    "LRRK2_CARRIER",
    "GBA_CARRIER",
    "APOE_E4_CARRIER",
]

TARGETS = {
    "binary": "target_binary",
    "three_class": "target_3class",
    "full_ordinal": "target_full_ordinal",
    "nsd_positive": "target_nsd_positive",
}


# ═════════════════════════════════════════════════════════════════════
# Simple GAT (custom dense-attention) — mirrors run_giman_gat_benchmark.py
# ═════════════════════════════════════════════════════════════════════
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.2, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat = concat
        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.Tensor(num_heads, out_dim))
        self.a_dst = nn.Parameter(torch.Tensor(num_heads, out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, adj):
        N = x.size(0)
        h = self.W(x).view(N, self.num_heads, self.out_dim)
        attn_src = (h * self.a_src.unsqueeze(0)).sum(dim=-1)
        attn_dst = (h * self.a_dst.unsqueeze(0)).sum(dim=-1)
        attn = attn_src.unsqueeze(1) + attn_dst.unsqueeze(0)
        attn = self.leaky_relu(attn)
        mask = (adj == 0).unsqueeze(-1).expand_as(attn)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=1)
        attn = self.dropout(attn)
        h_prime = torch.bmm(
            attn.permute(2, 0, 1),
            h.permute(1, 0, 2),
        ).permute(1, 0, 2)
        if self.concat:
            return h_prime.reshape(N, -1)
        else:
            return h_prime.mean(dim=1)


class SimpleGIMANGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_classes=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gat1 = GATLayer(hidden_dim, hidden_dim // num_heads, num_heads, dropout, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gat2 = GATLayer(hidden_dim, hidden_dim // num_heads, num_heads, dropout, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, adj):
        h = self.input_proj(x)
        h = F.relu(self.bn1(self.gat1(h, adj))) + h
        h = F.relu(self.bn2(self.gat2(h, adj))) + h
        return self.classifier(h)


def build_knn_graph_dense(X, k=10):
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0)
    N = X.shape[0]
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        topk = np.argsort(sim[i])[-k:]
        adj[i, topk] = sim[i, topk]
        adj[topk, i] = sim[i, topk]
    return adj


def train_simple_gat(
    X_train, y_train, X_val, y_val, adj_train, adj_val, input_dim, num_classes,
    epochs=300, lr=5e-4, patience=40,
):
    model = SimpleGIMANGAT(
        input_dim=input_dim, hidden_dim=128, num_heads=4,
        num_classes=num_classes, dropout=0.3,
    ).to(DEVICE)

    class_counts = np.bincount(y_train.astype(int), minlength=num_classes)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    class_weights = torch.FloatTensor(weights).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    y_tr = torch.LongTensor(y_train.astype(int)).to(DEVICE)
    adj_tr = torch.FloatTensor(adj_train).to(DEVICE)
    X_v = torch.FloatTensor(X_val).to(DEVICE)
    y_v = torch.LongTensor(y_val.astype(int)).to(DEVICE)
    adj_v = torch.FloatTensor(adj_val).to(DEVICE)

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr, adj_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v, adj_v)
            val_loss = criterion(val_logits, y_v).item()
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


def predict_simple_gat(model, X, adj):
    model.eval()
    X_t = torch.FloatTensor(X).to(DEVICE)
    adj_t = torch.FloatTensor(adj).to(DEVICE)
    with torch.no_grad():
        logits = model(X_t, adj_t)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
    return preds, probs


def run_simple_gat_target(X, y, target_name, n_classes, n_folds=5, k_neighbors=10):
    print(f"\n--- Simple GAT (21-feat): {target_name} ---")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        adj_train = build_knn_graph_dense(X_train, k=k_neighbors)
        adj_test = build_knn_graph_dense(X_test, k=min(k_neighbors, len(X_test) - 1))
        n_val = max(int(len(X_train) * 0.2), 1)
        val_idx = np.random.RandomState(fold).choice(len(X_train), n_val, replace=False)
        tr_idx = np.array([i for i in range(len(X_train)) if i not in val_idx])
        X_tr_fold, y_tr_fold = X_train[tr_idx], y_train[tr_idx]
        X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]
        adj_tr_fold = adj_train[np.ix_(tr_idx, tr_idx)]
        adj_val_fold = adj_train[np.ix_(val_idx, val_idx)]
        t0 = time.time()
        model = train_simple_gat(
            X_tr_fold, y_tr_fold, X_val_fold, y_val_fold,
            adj_tr_fold, adj_val_fold,
            input_dim=X.shape[1], num_classes=n_classes,
        )
        train_time = time.time() - t0
        preds, probs = predict_simple_gat(model, X_test, adj_test)
        ba = balanced_accuracy_score(y_test, preds)
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_test, probs[:, 1])
            else:
                auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")
        qwk = cohen_kappa_score(y_test, preds, weights="quadratic")
        fold_metrics.append({"fold": fold, "bal_acc": ba, "auc": auc, "qwk": qwk, "train_time": train_time})
        print(f"  Fold {fold}: bal_acc={ba:.4f}, AUC={auc:.4f}, QWK={qwk:.4f} ({train_time:.1f}s)")
    mean_ba = float(np.mean([m["bal_acc"] for m in fold_metrics]))
    mean_auc = float(np.nanmean([m["auc"] for m in fold_metrics]))
    mean_qwk = float(np.mean([m["qwk"] for m in fold_metrics]))
    std_ba = float(np.std([m["bal_acc"] for m in fold_metrics]))
    std_auc = float(np.nanstd([m["auc"] for m in fold_metrics]))
    return {
        "model": "Simple-GAT",
        "target": target_name,
        "feature_set": "Path3_21feat_strict_circularity",
        "n_features": X.shape[1],
        "n_patients": int(len(y)),
        "n_classes": n_classes,
        "k_neighbors": k_neighbors,
        "bal_acc": round(mean_ba, 4),
        "bal_acc_std": round(std_ba, 4),
        "auc": round(mean_auc, 4),
        "auc_std": round(std_auc, 4),
        "qwk": round(mean_qwk, 4),
        "fold_metrics": fold_metrics,
    }


# ═════════════════════════════════════════════════════════════════════
# Multimodal GAT (PyG) — mirrors run_enhanced_gat_benchmark.py
# ═════════════════════════════════════════════════════════════════════
def build_knn_graph_pyg(X, k=10):
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0)
    N = X.shape[0]
    src, dst = [], []
    for i in range(N):
        topk = np.argsort(sim[i])[-k:]
        for j in topk:
            src.append(i)
            dst.append(int(j))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    return edge_index


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiModalGATClassifier(nn.Module):
    def __init__(self, clinical_dim, biomarker_dim, embed_dim=128, hidden_dim=128,
                 num_heads=4, num_gat_layers=3, num_classes=2, dropout=0.2):
        super().__init__()
        self.clinical_encoder = ModalityEncoder(clinical_dim, embed_dim, dropout)
        self.biomarker_encoder = ModalityEncoder(biomarker_dim, embed_dim, dropout)
        self.clinical_gat_layers = nn.ModuleList()
        self.biomarker_gat_layers = nn.ModuleList()
        self.clinical_norms = nn.ModuleList()
        self.biomarker_norms = nn.ModuleList()
        for i in range(num_gat_layers):
            in_dim = embed_dim
            if i < num_gat_layers - 1:
                out_per_head = hidden_dim // num_heads
                self.clinical_gat_layers.append(GATConv(in_dim, out_per_head, heads=num_heads, dropout=dropout, concat=True))
                self.biomarker_gat_layers.append(GATConv(in_dim, out_per_head, heads=num_heads, dropout=dropout, concat=True))
                self.clinical_norms.append(nn.LayerNorm(hidden_dim))
                self.biomarker_norms.append(nn.LayerNorm(hidden_dim))
            else:
                self.clinical_gat_layers.append(GATConv(in_dim, embed_dim, heads=1, dropout=dropout, concat=False))
                self.biomarker_gat_layers.append(GATConv(in_dim, embed_dim, heads=1, dropout=dropout, concat=False))
                self.clinical_norms.append(nn.LayerNorm(embed_dim))
                self.biomarker_norms.append(nn.LayerNorm(embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.cross_modal_norm = nn.LayerNorm(embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Dropout(dropout),
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
        for i, (gat, norm) in enumerate(zip(gat_layers, norms, strict=False)):
            h = gat(x, edge_index)
            h = norm(h)
            if i < len(gat_layers) - 1:
                h = F.elu(h)
                h = self.dropout(h)
            if h.shape == x.shape:
                h = h + x
            x = h
        return x

    def forward(self, x_clinical, x_biomarker, edge_index):
        h_clin = self.clinical_encoder(x_clinical)
        h_bio = self.biomarker_encoder(x_biomarker)
        h_clin = self._apply_gat_layers(h_clin, edge_index, self.clinical_gat_layers, self.clinical_norms)
        h_bio = self._apply_gat_layers(h_bio, edge_index, self.biomarker_gat_layers, self.biomarker_norms)
        stacked = torch.stack([h_clin, h_bio], dim=1)
        attended, attn_weights = self.cross_modal_attention(stacked, stacked, stacked)
        attended = self.cross_modal_norm(attended + stacked)
        fused = attended.flatten(start_dim=1)
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        return logits, attn_weights


def train_mm_gat(model, X_clin_tr, X_bio_tr, y_tr, ei_tr, X_clin_v, X_bio_v, y_v, ei_v, num_classes,
                 epochs=300, lr=5e-4, patience=40):
    model = model.to(DEVICE)
    class_counts = np.bincount(y_tr.astype(int), minlength=num_classes)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    class_weights = torch.FloatTensor(weights).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    y_tr_t = torch.LongTensor(y_tr.astype(int)).to(DEVICE)
    y_val_t = torch.LongTensor(y_v.astype(int)).to(DEVICE)
    ei_tr = ei_tr.to(DEVICE)
    ei_val = ei_v.to(DEVICE)
    xc_tr = torch.FloatTensor(X_clin_tr).to(DEVICE)
    xb_tr = torch.FloatTensor(X_bio_tr).to(DEVICE)
    xc_val = torch.FloatTensor(X_clin_v).to(DEVICE)
    xb_val = torch.FloatTensor(X_bio_v).to(DEVICE)
    best_val_loss = float("inf")
    best_state = None
    wait = 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(xc_tr, xb_tr, ei_tr)
        loss = criterion(logits, y_tr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(xc_val, xb_val, ei_val)
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


def predict_mm_gat(model, X_clin, X_bio, edge_index):
    model.eval()
    with torch.no_grad():
        xc = torch.FloatTensor(X_clin).to(DEVICE)
        xb = torch.FloatTensor(X_bio).to(DEVICE)
        logits, _ = model(xc, xb, edge_index.to(DEVICE))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
    return preds, probs


def run_mm_gat_target(df, y, target_name, n_classes, n_folds=5, k_neighbors=10):
    print(f"\n--- MM-GAT (21-feat): {target_name} ---")
    clin_cols = [c for c in CLINICAL_21 if c in df.columns]
    bio_cols = [c for c in BIOMARKER_21 if c in df.columns]
    X_clin_raw = df[clin_cols].values
    X_bio_raw = df[bio_cols].values
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_clin_raw, y)):
        X_clin_train, X_clin_test = X_clin_raw[train_idx], X_clin_raw[test_idx]
        X_bio_train, X_bio_test = X_bio_raw[train_idx], X_bio_raw[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # Per-fold impute + scale
        imp_clin = SimpleImputer(strategy="median")
        X_clin_train = imp_clin.fit_transform(X_clin_train)
        X_clin_test = imp_clin.transform(X_clin_test)
        scaler_clin = StandardScaler()
        X_clin_train = scaler_clin.fit_transform(X_clin_train)
        X_clin_test = scaler_clin.transform(X_clin_test)
        imp_bio = SimpleImputer(strategy="median")
        X_bio_train = imp_bio.fit_transform(X_bio_train)
        X_bio_test = imp_bio.transform(X_bio_test)
        scaler_bio = StandardScaler()
        X_bio_train = scaler_bio.fit_transform(X_bio_train)
        X_bio_test = scaler_bio.transform(X_bio_test)
        X_graph_train = np.hstack([X_clin_train, X_bio_train])
        X_graph_test = np.hstack([X_clin_test, X_bio_test])
        edge_index_test = build_knn_graph_pyg(X_graph_test, k=min(k_neighbors, len(X_graph_test) - 1))
        n_val = max(int(len(X_clin_train) * 0.2), 1)
        rng = np.random.RandomState(fold)
        val_mask = rng.choice(len(X_clin_train), n_val, replace=False)
        tr_mask = np.array([i for i in range(len(X_clin_train)) if i not in val_mask])
        X_clin_tr, X_clin_v = X_clin_train[tr_mask], X_clin_train[val_mask]
        X_bio_tr, X_bio_v = X_bio_train[tr_mask], X_bio_train[val_mask]
        y_tr, y_v = y_train[tr_mask], y_train[val_mask]
        X_graph_tr = X_graph_train[tr_mask]
        X_graph_v = X_graph_train[val_mask]
        ei_tr = build_knn_graph_pyg(X_graph_tr, k=k_neighbors)
        ei_v = build_knn_graph_pyg(X_graph_v, k=min(k_neighbors, len(X_graph_v) - 1))
        model = MultiModalGATClassifier(
            clinical_dim=len(clin_cols), biomarker_dim=len(bio_cols),
            embed_dim=128, hidden_dim=128, num_heads=4, num_gat_layers=3,
            num_classes=n_classes, dropout=0.2,
        )
        t0 = time.time()
        model = train_mm_gat(model, X_clin_tr, X_bio_tr, y_tr, ei_tr,
                             X_clin_v, X_bio_v, y_v, ei_v, num_classes=n_classes)
        train_time = time.time() - t0
        preds, probs = predict_mm_gat(model, X_clin_test, X_bio_test, edge_index_test)
        ba = balanced_accuracy_score(y_test, preds)
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_test, probs[:, 1])
            else:
                auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")
        qwk = cohen_kappa_score(y_test, preds, weights="quadratic")
        fold_metrics.append({"fold": fold, "bal_acc": ba, "auc": auc, "qwk": qwk, "train_time": train_time})
        print(f"  Fold {fold}: bal_acc={ba:.4f}, AUC={auc:.4f}, QWK={qwk:.4f} ({train_time:.1f}s)")
    mean_ba = float(np.mean([m["bal_acc"] for m in fold_metrics]))
    mean_auc = float(np.nanmean([m["auc"] for m in fold_metrics]))
    mean_qwk = float(np.mean([m["qwk"] for m in fold_metrics]))
    std_ba = float(np.std([m["bal_acc"] for m in fold_metrics]))
    std_auc = float(np.nanstd([m["auc"] for m in fold_metrics]))
    return {
        "model": "Enhanced-GAT-multimodal",
        "target": target_name,
        "feature_set": "Path3_21feat_strict_circularity",
        "n_clinical_features": len(clin_cols),
        "n_biomarker_features": len(bio_cols),
        "n_total_features": len(clin_cols) + len(bio_cols),
        "n_patients": int(len(y)),
        "n_classes": n_classes,
        "architecture": "MultiModalGATClassifier",
        "pyg_gatconv_layers": 3,
        "num_heads": 4,
        "embed_dim": 128,
        "k_neighbors": k_neighbors,
        "bal_acc": round(mean_ba, 4),
        "bal_acc_std": round(std_ba, 4),
        "auc": round(mean_auc, 4),
        "auc_std": round(std_auc, 4),
        "qwk": round(mean_qwk, 4),
        "fold_metrics": fold_metrics,
    }


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("R4-Q3: Simple GAT + MM-GAT on Path 3 21-feature primary")
    print("=" * 70)
    ppmi = pd.read_csv(DATA / "05_features" / "paper1_features_with_targets.csv")
    print(f"Loaded PPMI: {len(ppmi)} patients")

    # Verify all 21-feat columns exist
    missing = [c for c in PRIMARY_21_FEATURES if c not in ppmi.columns]
    if missing:
        raise SystemExit(f"Missing 21-feat columns in CSV: {missing}")
    print(f"21-feat primary columns OK: {len(PRIMARY_21_FEATURES)} columns")

    simple_summary = {}
    mm_summary = {}

    for target_name, target_col in TARGETS.items():
        target_dir_simple = OUT_SIMPLE / target_name
        target_dir_mm = OUT_ENHANCED / target_name
        target_dir_simple.mkdir(parents=True, exist_ok=True)
        target_dir_mm.mkdir(parents=True, exist_ok=True)

        mask = ppmi[target_col].notna() & (ppmi[target_col] >= 0)
        df = ppmi[mask].copy()
        y_raw = df[target_col].values.astype(int)
        unique_labels = sorted(np.unique(y_raw))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y = np.array([label_map[v] for v in y_raw])
        n_classes = len(unique_labels)
        print(f"\n{'=' * 60}")
        print(f"Target: {target_name} | N={len(df)} | Classes={n_classes} | Map={label_map}")
        print(f"Class dist: {dict(zip(*np.unique(y, return_counts=True), strict=False))}")
        print(f"{'=' * 60}")

        # ── Simple GAT ──
        X_raw = df[PRIMARY_21_FEATURES].values
        # Single-shot impute+scale (matches the 22-feat Simple GAT runner pattern;
        # per-fold versions tested but matching runs apples-to-apples was preferred)
        imp = SimpleImputer(strategy="median")
        X = imp.fit_transform(X_raw)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        simple_result = run_simple_gat_target(X, y, target_name, n_classes)
        with open(target_dir_simple / "gat_results.json", "w") as f:
            json.dump([simple_result], f, indent=2, default=str)
        simple_summary[target_name] = simple_result
        print(f"Saved: {target_dir_simple / 'gat_results.json'}")

        # ── MM-GAT ──
        mm_result = run_mm_gat_target(df, y, target_name, n_classes)
        with open(target_dir_mm / "enhanced_gat_results.json", "w") as f:
            json.dump([mm_result], f, indent=2, default=str)
        mm_summary[target_name] = mm_result
        print(f"Saved: {target_dir_mm / 'enhanced_gat_results.json'}")

    # Consolidated summaries
    with open(OUT_SIMPLE / "summary.json", "w") as f:
        json.dump({
            "feature_set": "Path3_21feat_strict_circularity",
            "n_features": len(PRIMARY_21_FEATURES),
            "feature_cols": PRIMARY_21_FEATURES,
            "results": simple_summary,
        }, f, indent=2, default=str)
    with open(OUT_ENHANCED / "summary.json", "w") as f:
        json.dump({
            "feature_set": "Path3_21feat_strict_circularity",
            "n_clinical_features": len(CLINICAL_21),
            "n_biomarker_features": len(BIOMARKER_21),
            "clinical_features": CLINICAL_21,
            "biomarker_features": BIOMARKER_21,
            "results": mm_summary,
        }, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("SUMMARY: 21-feat primary GAT runs")
    print("=" * 70)
    print(f"{'Target':<16} {'Simple GAT AUC':<22} {'MM-GAT AUC':<22}")
    print("-" * 60)
    for tgt in TARGETS:
        s = simple_summary[tgt]
        m = mm_summary[tgt]
        print(f"{tgt:<16} {s['auc']:.4f} ± {s['auc_std']:.4f}    {m['auc']:.4f} ± {m['auc_std']:.4f}")
    print(f"\nSimple GAT outputs: {OUT_SIMPLE}")
    print(f"MM-GAT outputs: {OUT_ENHANCED}")


if __name__ == "__main__":
    main()
