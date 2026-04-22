#!/usr/bin/env python3
"""Paper 1 — 3-Modality 32-Feature GAT sensitivity variant (Supplementary S-3).

Forks the 2-modality Enhanced MM-GAT benchmark (``scripts/run_enhanced_gat_benchmark.py``)
to a 3-modality architecture with pairwise cross-modal attention:

  * Modality 1 (clinical, 14d): AGE, SEX, HANDED, UPDRS1/2/4 totals, UPDRS3 subscales (4),
    MOCA, RBD_TOTAL, ESS_TOTAL, SCOPA_AUT_TOTAL
  * Modality 2 (biomarker, 8d): 5 caudate DaT-SPECT-derived + 3 genetic carrier flags
  * Modality 3 (extended, 10d): 6 cortical thickness + 4 CSF biomarkers

Paper 1's canonical GRS_TOTAL is not pre-computed in ``iu_genetic_consensus_*.csv`` so the
extended modality is 10d not 11d; the sensitivity variant is therefore **32 features across
3 modalities**, not 33. See ``scripts/paper1/assemble_extended_33_feat.py`` for the ETL.

Matches Enhanced MM-GAT hyperparameters exactly (seed=42, k=10, 5-fold stratified CV,
150 epochs + early-stop patience 15, AdamW lr=1e-3 wd=1e-4, dropout 0.3, hidden=128,
4 heads x 3 layers). Per-fold kNN graph built on training set only, inductively extended
to test patients (nearest-neighbour projection).

Targets: binary, three_class, full_ordinal, nsd_positive.

Output: ``outputs/paper1_enhanced_gat_3mod/{target}_results.json``
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
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops, to_undirected

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = BASE / "outputs" / "paper1_enhanced_gat_3mod"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"Device: {DEVICE}")

# ==============================================================================
# Three-modality feature partition
# ==============================================================================
CLINICAL_FEATURES = [  # 14
    "AGE_AT_BASELINE",
    "SEX",
    "HANDED",
    "UPDRS1_TOTAL",
    "UPDRS2_TOTAL",
    "UPDRS4_TOTAL",
    "UPDRS3_TREMOR",
    "UPDRS3_RIGIDITY",
    "UPDRS3_BRADYKINESIA",
    "UPDRS3_AXIAL",
    "MOCA_TOTAL",
    "RBD_TOTAL",
    "ESS_TOTAL",
    "SCOPA_AUT_TOTAL",
]

BIOMARKER_FEATURES = [  # 8
    "CAUDATE_L_SBR",
    "CAUDATE_R_SBR",
    "CAUDATE_MEAN_SBR",
    "CAUDATE_ASYMMETRY",
    "CAUDATE_PUTAMEN_RATIO",
    "LRRK2_CARRIER",
    "GBA_CARRIER",
    "APOE_E4_CARRIER",
]

EXTENDED_FEATURES = [  # 10 (6 cortical thickness + 4 CSF; GRS not precomputed)
    "CTH_ENTORHINAL_L",
    "CTH_ENTORHINAL_R",
    "CTH_POSTCINGULATE_L",
    "CTH_POSTCINGULATE_R",
    "CTH_PRECENTRAL_L",
    "CTH_PRECENTRAL_R",
    "CSF_ASYN",
    "CSF_ABETA42",
    "CSF_PTAU181",
    "CSF_TTAU",
]

TARGETS = {
    "binary": "target_binary",
    "three_class": "target_3class",
    "full_ordinal": "target_full_ordinal",
    "nsd_positive": "target_nsd_positive",
}

# Hyperparams (match Enhanced MM-GAT)
SEED = 42
K_NEIGHBOURS = 10
N_EPOCHS = 150
EARLY_STOP_PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
HIDDEN_DIM = 128
EMBED_DIM = 128
N_HEADS = 4
N_GAT_LAYERS = 3
N_FOLDS = 5


# ==============================================================================
# Graph construction (k-NN, training-only; inductive extension for test)
# ==============================================================================
def build_knn_graph(X_train: np.ndarray, k: int = K_NEIGHBOURS) -> torch.Tensor:
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(X_train)
    np.fill_diagonal(sim, 0.0)
    N = X_train.shape[0]
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


def inductive_extend_graph(
    X_train: np.ndarray, X_test: np.ndarray, k: int = K_NEIGHBOURS
) -> tuple[torch.Tensor, np.ndarray]:
    """Return edge_index for combined N_train+N_test nodes.

    Test nodes are connected to their top-k nearest training nodes via cosine similarity.
    Training nodes retain their internal k-NN graph only (no leakage).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    n_tr, n_te = X_train.shape[0], X_test.shape[0]
    # Train-internal graph
    sim_tr = cosine_similarity(X_train)
    np.fill_diagonal(sim_tr, 0.0)
    src, dst = [], []
    for i in range(n_tr):
        topk = np.argsort(sim_tr[i])[-k:]
        for j in topk:
            src.append(i)
            dst.append(int(j))

    # Test-to-train edges (directed; we symmetrise via to_undirected)
    sim_te_tr = cosine_similarity(X_test, X_train)
    for i in range(n_te):
        topk = np.argsort(sim_te_tr[i])[-k:]
        for j in topk:
            src.append(n_tr + i)
            dst.append(int(j))

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=n_tr + n_te)
    return edge_index, np.arange(n_tr, n_tr + n_te)


# ==============================================================================
# Three-modality GAT with pairwise cross-modal attention (B.i)
# ==============================================================================
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = EMBED_DIM, dropout: float = DROPOUT):
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


class ThreeModalGAT(nn.Module):
    """3-modality GAT with pairwise cross-modal MHA (B.i pairwise)."""

    def __init__(
        self,
        clinical_dim: int,
        biomarker_dim: int,
        extended_dim: int,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = N_HEADS,
        num_gat_layers: int = N_GAT_LAYERS,
        num_classes: int = 2,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.num_modalities = 3

        self.clinical_encoder = ModalityEncoder(clinical_dim, embed_dim, dropout)
        self.biomarker_encoder = ModalityEncoder(biomarker_dim, embed_dim, dropout)
        self.extended_encoder = ModalityEncoder(extended_dim, embed_dim, dropout)

        # Per-modality GAT stacks
        self.clinical_gat, self.clinical_norms = self._make_gat_stack(
            embed_dim, hidden_dim, num_heads, num_gat_layers, dropout
        )
        self.biomarker_gat, self.biomarker_norms = self._make_gat_stack(
            embed_dim, hidden_dim, num_heads, num_gat_layers, dropout
        )
        self.extended_gat, self.extended_norms = self._make_gat_stack(
            embed_dim, hidden_dim, num_heads, num_gat_layers, dropout
        )

        # Pairwise cross-modal MHA (3 blocks)
        self.attn_CB = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_CS = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_BS = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )
        self._init_weights()

    def _make_gat_stack(self, embed_dim, hidden_dim, num_heads, num_layers, dropout):
        layers = nn.ModuleList()
        norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = embed_dim
            if i < num_layers - 1:
                out_per_head = hidden_dim // num_heads
                layers.append(
                    GATConv(in_dim, out_per_head, heads=num_heads, dropout=dropout, concat=True)
                )
                norms.append(nn.LayerNorm(hidden_dim))
            else:
                layers.append(GATConv(in_dim, embed_dim, heads=1, dropout=dropout, concat=False))
                norms.append(nn.LayerNorm(embed_dim))
        return layers, norms

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _apply_gat(self, h, edge_index, layers, norms):
        x = h
        for i, (gat, norm) in enumerate(zip(layers, norms, strict=False)):
            h_new = gat(x, edge_index)
            h_new = norm(h_new)
            if i < len(layers) - 1:
                h_new = F.elu(h_new)
            if h_new.shape == x.shape:
                h_new = h_new + x
            x = h_new
        return x

    def forward(self, x_clin, x_bio, x_ext, edge_index):
        h_c = self.clinical_encoder(x_clin)
        h_b = self.biomarker_encoder(x_bio)
        h_s = self.extended_encoder(x_ext)

        h_c = self._apply_gat(h_c, edge_index, self.clinical_gat, self.clinical_norms)
        h_b = self._apply_gat(h_b, edge_index, self.biomarker_gat, self.biomarker_norms)
        h_s = self._apply_gat(h_s, edge_index, self.extended_gat, self.extended_norms)

        # Pairwise cross-modal attention; each modality queried against another, residual
        def pair(q, kv, attn):
            q_, _ = attn(q.unsqueeze(1), kv.unsqueeze(1), kv.unsqueeze(1))
            return self.cross_norm(q.unsqueeze(1) + q_).squeeze(1)

        h_c_upd = pair(h_c, h_b, self.attn_CB)
        h_c_upd = pair(h_c_upd, h_s, self.attn_CS)
        h_b_upd = pair(h_b, h_s, self.attn_BS)
        h_s_upd = h_s  # extended stream receives from the other two via pair()

        fused = self.fusion(torch.cat([h_c_upd, h_b_upd, h_s_upd], dim=-1))
        return self.classifier(fused)


# ==============================================================================
# Training utilities
# ==============================================================================
def _forward_pass(
    model: ThreeModalGAT,
    X: dict,
    edge_index: torch.Tensor,
    idx: np.ndarray,
) -> torch.Tensor:
    logits = model(X["clin"], X["bio"], X["ext"], edge_index)
    return logits[idx]


def train_one_fold(
    Xtr: dict,
    Xte: dict,
    ytr: np.ndarray,
    yte: np.ndarray,
    edge_index: torch.Tensor,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    num_classes: int,
    class_weights: torch.Tensor | None = None,
) -> dict:
    """Single-fold training loop with early stopping on a holdout 10% of train."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = ThreeModalGAT(
        clinical_dim=Xtr["clin"].shape[1],
        biomarker_dim=Xtr["bio"].shape[1],
        extended_dim=Xtr["ext"].shape[1],
        num_classes=num_classes,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Build graph-on-full-union tensors (already concatenated outside)
    X_full = {k: torch.tensor(np.vstack([Xtr[k], Xte[k]]), dtype=torch.float32).to(DEVICE) for k in Xtr}
    edge_index = edge_index.to(DEVICE)
    y_full = np.concatenate([ytr, yte])
    y_full_t = torch.tensor(y_full, dtype=torch.long).to(DEVICE)

    # Internal val split on train indices
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(train_idx))
    val_sz = max(1, int(0.1 * len(train_idx)))
    val_idx = train_idx[perm[:val_sz]]
    tr_idx = train_idx[perm[val_sz:]]

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE) if class_weights is not None else None)

    best_val, bad_epochs, best_state = float("inf"), 0, None
    for ep in range(N_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(X_full["clin"], X_full["bio"], X_full["ext"], edge_index)
        loss = criterion(logits[tr_idx], y_full_t[tr_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_all = model(X_full["clin"], X_full["bio"], X_full["ext"], edge_index)
            val_loss = criterion(logits_all[val_idx], y_full_t[val_idx]).item()
        if val_loss < best_val - 1e-4:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs > EARLY_STOP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(X_full["clin"], X_full["bio"], X_full["ext"], edge_index)
        probs = F.softmax(logits[test_idx], dim=-1).cpu().numpy()
    preds = probs.argmax(axis=1)
    return {"probs": probs, "preds": preds, "y_true": yte}


# ==============================================================================
# Main
# ==============================================================================
def prepare_features(df: pd.DataFrame) -> dict:
    """Return {'clin','bio','ext'} arrays, imputing missing per-column with median."""
    X = {}
    for name, cols in [
        ("clin", CLINICAL_FEATURES),
        ("bio", BIOMARKER_FEATURES),
        ("ext", EXTENDED_FEATURES),
    ]:
        sub = df[cols].apply(pd.to_numeric, errors="coerce")
        imputer = SimpleImputer(strategy="median")
        X[name] = imputer.fit_transform(sub.values).astype(np.float32)
    return X


def bootstrap_ci(metric_fn, y_true, y_pred_or_proba, n_boot=1000, alpha=0.05, rng=None):
    rng = rng or np.random.RandomState(0)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            v = metric_fn(y_true[idx], y_pred_or_proba[idx])
        except ValueError:
            continue
        vals.append(v)
    vals = np.sort(vals)
    lo = np.quantile(vals, alpha / 2)
    hi = np.quantile(vals, 1 - alpha / 2)
    return float(np.mean(vals)), float(lo), float(hi)


def run_target(df: pd.DataFrame, target_key: str, target_col: str) -> dict:
    print(f"\n{'=' * 72}\nTARGET: {target_key} ({target_col})\n{'=' * 72}")

    # Filter valid targets; for NSD+, drop Stage 0
    exclude_stage0 = target_key == "nsd_positive"
    mask = df[target_col] >= 0
    if exclude_stage0:
        mask &= df["nsd_iss_stage"] != "0"
    sub = df[mask].reset_index(drop=True)
    y_raw = sub[target_col].values.astype(int)
    # Remap to consecutive 0..K-1
    uniq = sorted(np.unique(y_raw))
    y = np.array([uniq.index(v) for v in y_raw])
    K = len(uniq)
    print(f"  n={len(sub)}, K={K}, class counts={np.bincount(y).tolist()}")

    # Class weights (balanced)
    from sklearn.utils.class_weight import compute_class_weight

    cw = compute_class_weight("balanced", classes=np.arange(K), y=y)
    class_weights = torch.tensor(cw, dtype=torch.float32)

    X_all = prepare_features(sub)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_preds = []
    fold_probs = []
    fold_y = []
    fold_bal = []
    fold_auc = []
    fold_qwk = []

    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        print(f"  Fold {fold+1}/{N_FOLDS}: n_train={len(tr)}, n_test={len(te)}")

        # Per-fold scaling (fit on train)
        X_tr = {k: X_all[k][tr] for k in X_all}
        X_te = {k: X_all[k][te] for k in X_all}
        for k in X_tr:
            sc = StandardScaler().fit(X_tr[k])
            X_tr[k] = sc.transform(X_tr[k]).astype(np.float32)
            X_te[k] = sc.transform(X_te[k]).astype(np.float32)

        # Build graph on concatenated clinical+biomarker+extended (as 32d vector),
        # training-only then inductively extended
        X_tr_concat = np.concatenate([X_tr["clin"], X_tr["bio"], X_tr["ext"]], axis=1)
        X_te_concat = np.concatenate([X_te["clin"], X_te["bio"], X_te["ext"]], axis=1)
        edge_index, _ = inductive_extend_graph(X_tr_concat, X_te_concat, k=K_NEIGHBOURS)

        train_idx = np.arange(len(tr))
        test_idx = np.arange(len(tr), len(tr) + len(te))

        out = train_one_fold(
            X_tr,
            X_te,
            y[tr],
            y[te],
            edge_index,
            train_idx,
            test_idx,
            num_classes=K,
            class_weights=class_weights,
        )
        fold_preds.append(out["preds"])
        fold_probs.append(out["probs"])
        fold_y.append(out["y_true"])
        fold_bal.append(balanced_accuracy_score(out["y_true"], out["preds"]))
        try:
            if K == 2:
                auc = roc_auc_score(out["y_true"], out["probs"][:, 1])
            else:
                auc = roc_auc_score(
                    out["y_true"],
                    out["probs"],
                    multi_class="ovr",
                    average="macro",
                )
            fold_auc.append(auc)
        except ValueError:
            fold_auc.append(np.nan)
        fold_qwk.append(
            cohen_kappa_score(out["y_true"], out["preds"], weights="quadratic")
        )
        print(f"    bal_acc={fold_bal[-1]:.4f}  auc={fold_auc[-1]:.4f}  qwk={fold_qwk[-1]:.4f}")

    y_all = np.concatenate(fold_y)
    p_all = np.concatenate(fold_preds)
    prob_all = np.concatenate(fold_probs)

    rng = np.random.RandomState(SEED)
    bal_mean, bal_lo, bal_hi = bootstrap_ci(balanced_accuracy_score, y_all, p_all, rng=rng)
    if K == 2:
        auc_mean, auc_lo, auc_hi = bootstrap_ci(
            lambda yt, pp: roc_auc_score(yt, pp),
            y_all,
            prob_all[:, 1],
            rng=np.random.RandomState(SEED + 1),
        )
    else:
        auc_mean, auc_lo, auc_hi = bootstrap_ci(
            lambda yt, pp: roc_auc_score(yt, pp, multi_class="ovr", average="macro"),
            y_all,
            prob_all,
            rng=np.random.RandomState(SEED + 1),
        )
    qwk_mean, qwk_lo, qwk_hi = bootstrap_ci(
        lambda yt, pp: cohen_kappa_score(yt, pp, weights="quadratic"),
        y_all,
        p_all,
        rng=np.random.RandomState(SEED + 2),
    )

    result = {
        "target": target_key,
        "n_patients": int(len(sub)),
        "n_classes": int(K),
        "class_counts": np.bincount(y).tolist(),
        "per_fold": {
            "balanced_accuracy": [float(v) for v in fold_bal],
            "auc": [float(v) for v in fold_auc],
            "qwk": [float(v) for v in fold_qwk],
        },
        "aggregate": {
            "balanced_accuracy": {
                "mean": float(np.mean(fold_bal)),
                "std": float(np.std(fold_bal)),
                "bootstrap_mean": bal_mean,
                "bootstrap_ci": [bal_lo, bal_hi],
            },
            "auc": {
                "mean": float(np.nanmean(fold_auc)),
                "std": float(np.nanstd(fold_auc)),
                "bootstrap_mean": auc_mean,
                "bootstrap_ci": [auc_lo, auc_hi],
            },
            "qwk": {
                "mean": float(np.mean(fold_qwk)),
                "std": float(np.std(fold_qwk)),
                "bootstrap_mean": qwk_mean,
                "bootstrap_ci": [qwk_lo, qwk_hi],
            },
        },
    }
    out_path = OUT / f"{target_key}_results.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved → {out_path}")
    return result


def main():
    feat_path = DATA / "05_features" / "paper1_features_extended_33.csv"
    if not feat_path.exists():
        raise SystemExit(
            "Run scripts/paper1/assemble_extended_33_feat.py first to produce "
            f"{feat_path}"
        )
    df = pd.read_csv(feat_path)
    print(f"Loaded extended feature table: {len(df)} rows x {len(df.columns)} cols")

    all_results = {}
    t0 = time.time()
    for key, col in TARGETS.items():
        all_results[key] = run_target(df, key, col)
    print(f"\nAll targets complete in {time.time() - t0:.1f} s")

    summary_path = OUT / "summary.json"
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
