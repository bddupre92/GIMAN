#!/usr/bin/env python3
"""Run GIMAN GAT (Graph Attention Network) on NSD-ISS stage prediction targets.

This is the graph model novelty layer on top of the tabular baselines.
Uses patient similarity graphs with multi-head GAT for stage classification.

Models:
  1. GIMAN-GAT: Our multi-head GAT with patient similarity graph
  2. AdaMedGraph: APPNP + AdaBoost baseline (already benchmarked)

Targets: binary, three_class, full_ordinal, nsd_positive
Features: 22-feature (full) and 12-feature (clinical-only) variants

Outputs: outputs/paper1_gat/{target_type}/gat_results.json
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

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = BASE / "outputs" / "paper1_gat"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Feature sets ──
FULL_FEATURES = [
    "SEX",
    "HANDED",
    "AGE_AT_BASELINE",
    "UPDRS1_TOTAL",
    "UPDRS2_TOTAL",
    "UPDRS3_TREMOR",
    "UPDRS3_RIGIDITY",
    "UPDRS3_BRADYKINESIA",
    "UPDRS3_AXIAL",
    "UPDRS4_TOTAL",
    "MOCA_TOTAL",
    "RBD_TOTAL",
    "ESS_TOTAL",
    "SCOPA_AUT_TOTAL",
    "CAUDATE_R_SBR",
    "CAUDATE_L_SBR",
    "CAUDATE_MEAN_SBR",
    "CAUDATE_ASYMMETRY",
    "CAUDATE_PUTAMEN_RATIO",
    "LRRK2_CARRIER",
    "GBA_CARRIER",
    "APOE_E4_CARRIER",
]

COMMON_FEATURES = [
    "AGE_AT_BASELINE",
    "SEX",
    "UPDRS1_TOTAL",
    "UPDRS2_TOTAL",
    "UPDRS3_TREMOR",
    "UPDRS3_RIGIDITY",
    "UPDRS3_BRADYKINESIA",
    "UPDRS3_AXIAL",
    "UPDRS4_TOTAL",
    "MOCA_TOTAL",
    "ESS_TOTAL",
    "RBD_TOTAL",
]

TARGETS = {
    "binary": "target_binary",
    "three_class": "target_3class",
    "full_ordinal": "target_full_ordinal",
    "nsd_positive": "target_nsd_positive",
}


# ═══════════════════════════════════════════════════════════════════════
# Simple GAT for Tabular Data
# ═══════════════════════════════════════════════════════════════════════
class GATLayer(nn.Module):
    """Single Graph Attention layer."""

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
        """x: (N, in_dim)
        adj: (N, N) adjacency matrix (dense)
        """
        N = x.size(0)
        h = self.W(x).view(N, self.num_heads, self.out_dim)  # (N, heads, out_dim)

        # Attention scores
        attn_src = (h * self.a_src.unsqueeze(0)).sum(dim=-1)  # (N, heads)
        attn_dst = (h * self.a_dst.unsqueeze(0)).sum(dim=-1)  # (N, heads)

        # Pairwise attention: e_ij = LeakyReLU(a_src * h_i + a_dst * h_j)
        attn = attn_src.unsqueeze(1) + attn_dst.unsqueeze(0)  # (N, N, heads)
        attn = self.leaky_relu(attn)

        # Mask non-edges
        mask = (adj == 0).unsqueeze(-1).expand_as(attn)
        attn = attn.masked_fill(mask, float("-inf"))

        # Softmax
        attn = F.softmax(attn, dim=1)  # (N, N, heads)
        attn = self.dropout(attn)

        # Aggregate: h' = sum_j attn_ij * h_j
        # (N, N, heads) x (N, heads, out_dim) -> (N, heads, out_dim)
        h_prime = torch.bmm(
            attn.permute(2, 0, 1),  # (heads, N, N)
            h.permute(1, 0, 2),  # (heads, N, out_dim)
        ).permute(1, 0, 2)  # (N, heads, out_dim)

        if self.concat:
            return h_prime.reshape(N, -1)  # (N, heads * out_dim)
        else:
            return h_prime.mean(dim=1)  # (N, out_dim)


class GIMANGAT(nn.Module):
    """GIMAN Graph Attention Network for NSD-ISS stage classification."""

    def __init__(
        self, input_dim, hidden_dim=64, num_heads=4, num_classes=2, dropout=0.3
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gat1 = GATLayer(
            hidden_dim, hidden_dim // num_heads, num_heads, dropout, concat=True
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gat2 = GATLayer(
            hidden_dim, hidden_dim // num_heads, num_heads, dropout, concat=True
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, adj):
        h = self.input_proj(x)
        h = F.relu(self.bn1(self.gat1(h, adj))) + h  # residual
        h = F.relu(self.bn2(self.gat2(h, adj))) + h  # residual
        return self.classifier(h)


# ═══════════════════════════════════════════════════════════════════════
# Graph Construction
# ═══════════════════════════════════════════════════════════════════════
def build_knn_graph(X, k=10):
    """Build k-NN graph from feature matrix using cosine similarity."""
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0)

    N = X.shape[0]
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        topk = np.argsort(sim[i])[-k:]
        adj[i, topk] = sim[i, topk]
        adj[topk, i] = sim[i, topk]  # symmetric

    return adj


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════
def train_gat(
    X_train,
    y_train,
    X_val,
    y_val,
    adj_train,
    adj_val,
    input_dim,
    num_classes,
    epochs=300,
    lr=5e-4,
    patience=40,
):
    """Train GIMAN-GAT model with early stopping."""
    model = GIMANGAT(
        input_dim=input_dim,
        hidden_dim=128,
        num_heads=4,
        num_classes=num_classes,
        dropout=0.3,
    ).to(DEVICE)

    # Class weights
    class_counts = np.bincount(y_train.astype(int), minlength=num_classes)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    class_weights = torch.FloatTensor(weights).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5
    )

    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    y_tr = torch.LongTensor(y_train.astype(int)).to(DEVICE)
    adj_tr = torch.FloatTensor(adj_train).to(DEVICE)

    X_v = torch.FloatTensor(X_val).to(DEVICE)
    y_v = torch.LongTensor(y_val.astype(int)).to(DEVICE)
    adj_v = torch.FloatTensor(adj_val).to(DEVICE)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr, adj_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()

        # Validation
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


def predict_gat(model, X, adj):
    """Get predictions and probabilities from trained GAT."""
    model.eval()
    X_t = torch.FloatTensor(X).to(DEVICE)
    adj_t = torch.FloatTensor(adj).to(DEVICE)
    with torch.no_grad():
        logits = model(X_t, adj_t)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
    return preds, probs


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════
def run_gat_benchmark(
    X, y, target_name, n_classes, feature_set_name, n_folds=5, k_neighbors=10
):
    """Run GIMAN-GAT with stratified k-fold CV."""
    print(
        f"\n--- GIMAN-GAT: {target_name} ({feature_set_name}, {X.shape[1]} features) ---"
    )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Build per-fold graphs
        adj_train = build_knn_graph(X_train, k=k_neighbors)
        adj_test = build_knn_graph(X_test, k=min(k_neighbors, len(X_test) - 1))

        # Split train into train/val (80/20 within training fold)
        n_val = max(int(len(X_train) * 0.2), 1)
        val_idx = np.random.RandomState(fold).choice(len(X_train), n_val, replace=False)
        tr_idx = np.array([i for i in range(len(X_train)) if i not in val_idx])

        X_tr_fold, y_tr_fold = X_train[tr_idx], y_train[tr_idx]
        X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]
        adj_tr_fold = adj_train[np.ix_(tr_idx, tr_idx)]
        adj_val_fold = adj_train[np.ix_(val_idx, val_idx)]

        # Train
        t0 = time.time()
        model = train_gat(
            X_tr_fold,
            y_tr_fold,
            X_val_fold,
            y_val_fold,
            adj_tr_fold,
            adj_val_fold,
            input_dim=X.shape[1],
            num_classes=n_classes,
            epochs=300,
            lr=5e-4,
            patience=40,
        )
        train_time = time.time() - t0

        # Predict on test
        preds, probs = predict_gat(model, X_test, adj_test)

        # Metrics
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

    # Aggregate
    mean_ba = np.mean([m["bal_acc"] for m in fold_metrics])
    mean_auc = np.nanmean([m["auc"] for m in fold_metrics])
    mean_qwk = np.mean([m["qwk"] for m in fold_metrics])
    std_ba = np.std([m["bal_acc"] for m in fold_metrics])
    std_auc = np.nanstd([m["auc"] for m in fold_metrics])

    result = {
        "model": "GIMAN-GAT",
        "target": target_name,
        "feature_set": feature_set_name,
        "n_features": X.shape[1],
        "n_patients": len(y),
        "n_classes": n_classes,
        "k_neighbors": k_neighbors,
        "bal_acc": round(mean_ba, 4),
        "bal_acc_std": round(std_ba, 4),
        "auc": round(mean_auc, 4),
        "auc_std": round(std_auc, 4),
        "qwk": round(mean_qwk, 4),
        "fold_metrics": fold_metrics,
    }

    print(
        f"  MEAN: bal_acc={mean_ba:.4f}+-{std_ba:.4f}, AUC={mean_auc:.4f}+-{std_auc:.4f}, QWK={mean_qwk:.4f}"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("GIMAN-GAT Benchmark on NSD-ISS Targets")
    print("=" * 70)

    # Load data
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
        print(f"Label mapping: {label_map}")

        print(f"\n{'=' * 50}")
        print(f"Target: {target_name} | N={len(df)} | Classes={n_classes}")
        print(
            f"Class distribution: {dict(zip(*np.unique(y, return_counts=True), strict=False))}"
        )
        print(f"{'=' * 50}")

        target_results = []

        # Run on both feature sets
        for feat_name, feat_list in [
            ("full_22", FULL_FEATURES),
            ("clinical_12", COMMON_FEATURES),
        ]:
            available = [f for f in feat_list if f in df.columns]
            X_raw = df[available].values

            # Impute + scale
            imp = SimpleImputer(strategy="median")
            X = imp.fit_transform(X_raw)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            result = run_gat_benchmark(
                X,
                y,
                target_name=target_name,
                n_classes=n_classes,
                feature_set_name=feat_name,
                n_folds=5,
                k_neighbors=10,
            )
            target_results.append(result)

        # Save per-target
        with open(target_dir / "gat_results.json", "w") as f:
            json.dump(target_results, f, indent=2, default=str)
        print(f"Saved: {target_dir / 'gat_results.json'}")

        all_results[target_name] = target_results

    # Save comprehensive results
    with open(OUT / "gat_benchmark_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print("\n" + "=" * 70)
    print("GIMAN-GAT BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Target':<16} {'Features':<14} {'Bal Acc':<12} {'AUC':<12} {'QWK':<10}")
    print("-" * 64)
    for target, results in all_results.items():
        for r in results:
            print(
                f"{target:<16} {r['feature_set']:<14} {r['bal_acc']:.4f}+-{r['bal_acc_std']:.4f} "
                f"{r['auc']:.4f}+-{r['auc_std']:.4f} {r['qwk']:.4f}"
            )

    print(f"\nAll results saved to: {OUT}")


if __name__ == "__main__":
    main()
