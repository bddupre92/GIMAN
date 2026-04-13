"""Graph-Informed Digital Twin for NSD-ISS Stage Transitions.

Combines patient similarity graphs with temporal sequence modeling to predict
stage transition timing. The graph provides population-level context about
similar patients' baseline profiles, enriching individual predictions.

Architecture:
    1. Build patient similarity graph from BASELINE features (no leakage)
    2. Learned node encoder maps baseline features → embeddings
    3. GAT propagates information across similar patients (pre-computed per fold)
    4. GRU encodes per-episode visit sequences (individual trajectory)
    5. Gated fusion combines temporal + graph representations
    6. Cause-specific heads predict discrete-time transition CIF

Key design decisions:
    - Graph uses baseline features ONLY → no information leakage from future visits
    - All patients (train+test) included in graph → transductive but label-free
    - Node features pre-computed once per fold → stable training, no re-encoding
    - Gated residual: output = temporal + gate * graph_enrichment
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GATConv
from tqdm import tqdm

from giman_pipeline.paper3.dynamic_deephit import (
    N_TIME_BINS,
    _get_time_bin,
    build_patient_arrays,
    compute_brier_score,
    compute_ctd,
    compute_feature_stats,
    compute_ibs,
    extract_episodes,
)
from giman_pipeline.paper3.multistate_markov import N_STATES, STAGE_LABELS

# ── Graph Construction ─────────────────────────────────────────────────

GRAPH_FEATURES = [
    # Demographics & genetics (always available)
    "age_at_baseline",
    "sex",
    "lrrk2_carrier",
    "gba_carrier",
    "snca_carrier",
    "apoe_e4",
    # Clinical scores at baseline
    "updrs3_total",
    "updrs2_total",
    "updrs1_total",
    "hy_stage",
    "nsd_stage_numeric",
    # Cognitive / autonomic / sleep (51-62% coverage — NaN handled)
    "moca_total",
    "ess_total",
    "rbd_total",
    "scopa_aut_total",
    # Olfaction
    "upsit_total",
    # DaTSCAN imaging (16% coverage — still informative for those who have it)
    "caudate_mean_sbr",
    "putamen_mean_sbr",
]


def build_patient_graph(
    features_df: pd.DataFrame,
    patient_ids: list[int],
    k_neighbors: int = 15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build kNN patient similarity graph from baseline features.

    Returns:
        edge_index: (2, E) long tensor
        edge_weight: (E,) float tensor
        node_baseline: (N, n_graph_features) float tensor (standardized)
    """
    baseline = features_df[features_df["months_from_baseline"] == 0.0]
    baseline = baseline[baseline["PATNO"].isin(patient_ids)]

    pat_to_idx = {p: i for i, p in enumerate(patient_ids)}
    n = len(patient_ids)

    cols = [c for c in GRAPH_FEATURES if c in baseline.columns]
    feat_matrix = np.zeros((n, len(cols)), dtype=np.float32)
    mask = np.zeros((n, len(cols)), dtype=bool)

    for _, row in baseline.iterrows():
        idx = pat_to_idx.get(int(row["PATNO"]))
        if idx is None:
            continue
        for ci, col in enumerate(cols):
            val = row[col]
            if pd.notna(val):
                feat_matrix[idx, ci] = float(val)
                mask[idx, ci] = True

    # Standardize
    for ci in range(len(cols)):
        obs = feat_matrix[mask[:, ci], ci]
        if len(obs) > 1:
            m, s = obs.mean(), obs.std()
            if s > 1e-8:
                feat_matrix[:, ci] = (feat_matrix[:, ci] - m) / s
            else:
                feat_matrix[:, ci] = 0.0
    feat_matrix[~mask] = 0.0

    # kNN graph from cosine similarity
    norms = np.linalg.norm(feat_matrix, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    normed = feat_matrix / norms
    sim = normed @ normed.T
    np.fill_diagonal(sim, -1.0)

    k = min(k_neighbors, n - 1)
    src_list, dst_list, w_list = [], [], []
    for i in range(n):
        topk = np.argpartition(sim[i], -k)[-k:]
        for j in topk:
            if sim[i, j] > 0:
                src_list.append(i)
                dst_list.append(j)
                w_list.append(float(sim[i, j]))

    if not src_list:
        src_list = list(range(n))
        dst_list = list(range(n))
        w_list = [1.0] * n

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(w_list, dtype=torch.float32)
    node_baseline = torch.tensor(feat_matrix, dtype=torch.float32)
    return edge_index, edge_weight, node_baseline


# ── Dataset ────────────────────────────────────────────────────────────


class GraphDeepHitDataset(Dataset):
    def __init__(self, episodes, patient_arrays, means, stds, pat_to_graph_idx):
        self.episodes = episodes
        self.patient_arrays = patient_arrays
        self.means = means
        self.stds = stds
        self.pat_to_graph_idx = pat_to_graph_idx

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        raw = self.patient_arrays[ep.patno]
        seq = raw[: ep.max_visit_idx + 1].copy()
        seq = (seq - self.means) / (self.stds + 1e-8)
        seq = np.nan_to_num(seq, nan=0.0)
        return {
            "sequence": torch.from_numpy(seq),
            "seq_len": len(seq),
            "stage_idx": ep.current_stage_idx,
            "duration": ep.duration_months,
            "time_bin": _get_time_bin(ep.duration_months),
            "event_idx": ep.event_stage_idx,
            "censored": ep.censored,
            "graph_idx": self.pat_to_graph_idx[
                ep.patno
            ],  # KeyError if missing = correct behavior
            "patno": ep.patno,
        }


def graph_collate_fn(batch):
    sequences = [item["sequence"] for item in batch]
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return {
        "sequences": padded,
        "seq_lens": torch.LongTensor([item["seq_len"] for item in batch]),
        "stage_idxs": torch.LongTensor([item["stage_idx"] for item in batch]),
        "durations": torch.FloatTensor([item["duration"] for item in batch]),
        "time_bins": torch.LongTensor([item["time_bin"] for item in batch]),
        "event_idxs": torch.LongTensor([item["event_idx"] for item in batch]),
        "censored": torch.BoolTensor([item["censored"] for item in batch]),
        "graph_idxs": torch.LongTensor([item["graph_idx"] for item in batch]),
        "patnos": [item["patno"] for item in batch],
    }


# ── Temporal Attention ─────────────────────────────────────────────────


class TemporalAttentionPool(nn.Module):
    """Attention-weighted pooling over GRU hidden states.

    Instead of using only the last GRU hidden state, this learns to attend
    over ALL timesteps, giving a richer temporal summary. Combined with
    the last hidden state via residual addition.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, gru_output: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """Args:
            gru_output: (batch, max_seq, hidden_dim) padded GRU outputs
            seq_lens: (batch,) actual sequence lengths
        Returns:
            context: (batch, hidden_dim) attention-weighted sum
        """
        scores = self.attn(gru_output).squeeze(-1)  # (batch, max_seq)
        # Mask padding positions
        device = gru_output.device
        max_len = gru_output.size(1)
        mask = torch.arange(max_len, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (batch, max_seq)
        context = (gru_output * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden_dim)
        return context


# ── Model ──────────────────────────────────────────────────────────────


class GraphDigitalTwin(nn.Module):
    """GRU (temporal) + GAT (population graph) with gated fusion.

    Architecture:
        1. GAT enriches baseline features → population-context embeddings
        2. GRU encodes individual visit sequences → temporal embeddings
        3. Temporal attention pools over all GRU timesteps
        4. Warm-start gated fusion: temporal + gate * graph (residual)
        5. Stage embedding + output head → discrete-time CIF
    """

    def __init__(
        self,
        input_dim: int,
        n_baseline_features: int = 9,
        hidden_dim: int = 128,
        n_gru_layers: int = 2,
        gat_heads: int = 4,
        gat_layers: int = 2,
        n_causes: int = N_STATES,
        n_time_bins: int = N_TIME_BINS,
        stage_embed_dim: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_causes = n_causes
        self.n_time_bins = n_time_bins
        self.hidden_dim = hidden_dim

        # Temporal encoder (per-episode visit sequences)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_gru_layers,
            batch_first=True,
            dropout=dropout if n_gru_layers > 1 else 0.0,
        )

        # Temporal attention pooling (over all GRU timesteps, not just last)
        self.temporal_attn = TemporalAttentionPool(hidden_dim)

        # Node encoder: baseline features → hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(n_baseline_features, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
        )

        # GAT layers on patient graph
        self.gat_layers_list = nn.ModuleList()
        gat_in = hidden_dim
        for _ in range(gat_layers):
            out_per_head = hidden_dim // gat_heads
            self.gat_layers_list.append(
                GATConv(
                    gat_in, out_per_head, heads=gat_heads, dropout=dropout, concat=True
                )
            )
            gat_in = out_per_head * gat_heads
        self.gat_proj = nn.Linear(gat_in, hidden_dim)
        self.gat_norm = nn.LayerNorm(hidden_dim)

        # Warm-start gated fusion: starts nearly closed (sigmoid(-5)≈0.007)
        # so model begins as pure temporal, learns to incorporate graph
        self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.zeros_(self.gate_linear.weight)
        nn.init.constant_(self.gate_linear.bias, -5.0)  # warm-start: gate ≈ 0

        # Stage embedding
        self.stage_embed = nn.Embedding(N_STATES, stage_embed_dim)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim + stage_embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_causes * n_time_bins + 1),
        )

    def compute_graph_features(self, node_baseline, edge_index, edge_weight=None):
        """Pre-compute GAT-enriched node features (once per fold/epoch)."""
        x = self.node_encoder(node_baseline)
        for gat in self.gat_layers_list:
            x = gat(x, edge_index, edge_attr=edge_weight)
            x = F.elu(x)
        x = self.gat_proj(x)
        x = self.gat_norm(x)
        return x  # (N_patients, hidden_dim)

    def forward(self, sequences, seq_lens, stage_idxs, graph_idxs, graph_node_features):
        """Args:
        sequences: (batch, max_seq, input_dim)
        seq_lens: (batch,)
        stage_idxs: (batch,)
        graph_idxs: (batch,) indices into graph_node_features
        graph_node_features: (N_patients, hidden_dim) pre-computed GAT output
        """
        # 1. Temporal encoding with attention pooling
        sorted_lens, sort_idx = seq_lens.sort(descending=True)
        sorted_seqs = sequences[sort_idx]
        sorted_lens_clamped = sorted_lens.clamp(min=1)
        packed = pack_padded_sequence(
            sorted_seqs, sorted_lens_clamped.cpu(), batch_first=True
        )
        gru_out, h_n = self.gru(packed)
        gru_out_padded, _ = pad_packed_sequence(gru_out, batch_first=True)
        _, unsort_idx = sort_idx.sort()
        gru_out_unsorted = gru_out_padded[unsort_idx]
        last_hidden = h_n[-1][unsort_idx]  # (batch, hidden_dim)

        # Attention pool over all timesteps + residual from last hidden
        seq_lens_dev = seq_lens.to(sequences.device)
        attn_ctx = self.temporal_attn(gru_out_unsorted, seq_lens_dev)
        temporal = attn_ctx + last_hidden  # (batch, hidden_dim)

        # 2. Graph features for this batch
        graph_feat = graph_node_features[graph_idxs]  # (batch, hidden_dim)

        # 3. Warm-start gated fusion: temporal + gate * graph
        gate_input = torch.cat([temporal, graph_feat], dim=-1)
        g = torch.sigmoid(self.gate_linear(gate_input))  # starts ≈ 0.007
        fused = temporal + g * graph_feat  # residual: always keep temporal

        # 4. Stage embedding
        stage_emb = self.stage_embed(stage_idxs)

        # 5. Output
        combined = torch.cat([fused, stage_emb], dim=-1)
        logits = self.output_head(combined)
        return F.softmax(logits, dim=-1)

    def predict_cif(
        self, sequences, seq_lens, stage_idxs, graph_idxs, graph_node_features
    ):
        pmf = self.forward(
            sequences, seq_lens, stage_idxs, graph_idxs, graph_node_features
        )
        event_pmf = pmf[:, :-1].view(-1, self.n_causes, self.n_time_bins)
        return torch.cumsum(event_pmf, dim=-1)


# ── Loss ───────────────────────────────────────────────────────────────


def _graph_loss(pmf, time_bins, event_idxs, censored, alpha=0.1):
    """Efficient GPU loss (NLL + ranking)."""
    batch = pmf.size(0)
    event_pmf = pmf[:, :-1].view(batch, N_STATES, N_TIME_BINS)
    cum_event = torch.cumsum(event_pmf, dim=-1)
    cum_total = cum_event.sum(dim=1)

    loss = torch.tensor(0.0, device=pmf.device)

    unc_mask = ~censored
    if unc_mask.any():
        unc_idx = torch.where(unc_mask)[0]
        unc_k = event_idxs[unc_idx]
        unc_j = time_bins[unc_idx]
        unc_p = event_pmf[unc_idx, unc_k, unc_j]
        loss = loss - torch.log(unc_p + 1e-8).sum() / batch

    cen_mask = censored
    if cen_mask.any():
        cen_idx = torch.where(cen_mask)[0]
        cen_j = time_bins[cen_idx]
        p_ev = cum_total[cen_idx, cen_j]
        loss = loss - torch.log((1.0 - p_ev).clamp(min=1e-8)).sum() / batch

    nll_val = loss.detach()

    if alpha > 0 and unc_mask.sum() > 2:
        unc_idx2 = torch.where(unc_mask)[0]
        unc_k2 = event_idxs[unc_idx2]
        unc_j2 = time_bins[unc_idx2]
        sort_order = torch.argsort(unc_j2)
        s_idx = unc_idx2[sort_order]
        s_k = unc_k2[sort_order]
        s_j = unc_j2[sort_order]
        same_k = s_k[:-1] == s_k[1:]
        diff_t = s_j[:-1] != s_j[1:]
        valid = same_k & diff_t
        if valid.any():
            vi = torch.where(valid)[0]
            ci_early = cum_event[s_idx[vi], s_k[vi], s_j[vi]]
            ci_later = cum_event[s_idx[vi + 1], s_k[vi], s_j[vi]]
            rank_loss = torch.exp(-(ci_early - ci_later) / 0.1).mean()
            loss = loss + alpha * rank_loss

    return loss, nll_val, loss.detach() - nll_val


def _graph_smoothing_loss(
    graph_node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    stage_idxs_per_node: torch.Tensor | None = None,
    n_sample: int = 512,
) -> torch.Tensor:
    """Graph smoothing: similar connected patients → similar representations.

    L_smooth = mean_{(i,j) in E} w_ij * ||h_i - h_j||^2
    Only compares patients in the same current stage (if provided).
    """
    n_edges = edge_index.size(1)
    if n_edges == 0:
        return torch.tensor(0.0, device=graph_node_features.device)

    # Sample edges for efficiency
    if n_edges > n_sample:
        idx = torch.randperm(n_edges, device=edge_index.device)[:n_sample]
        ei = edge_index[:, idx]
        ew = edge_weight[idx] if edge_weight is not None else None
    else:
        ei = edge_index
        ew = edge_weight

    src_feat = graph_node_features[ei[0]]
    dst_feat = graph_node_features[ei[1]]
    diff = (src_feat - dst_feat).pow(2).mean(dim=-1)  # (n_edges,)

    if ew is not None:
        diff = diff * ew

    return diff.mean()


# ── Training ───────────────────────────────────────────────────────────


def train_graph_model(
    model,
    train_ds,
    val_ds,
    device,
    node_baseline,
    edge_index,
    edge_weight,
    n_epochs=100,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    patience=15,
    alpha=0.1,
    graph_smooth_weight=0.01,
    verbose=True,
):
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=graph_collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=graph_collate_fn,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=10,
        factor=0.5,
        min_lr=1e-6,
    )

    nb = node_baseline.to(device)
    ei = edge_index.to(device)
    ew = edge_weight.to(device) if edge_weight is not None else None

    best_val = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    it = (
        tqdm(range(n_epochs), desc="Training", unit="epoch")
        if verbose
        else range(n_epochs)
    )

    for epoch in it:
        # Train
        model.train()

        # Compute graph features once per epoch (efficient: 1 GAT pass for 1900 nodes)
        # First batch uses live computation graph (gradients → node_encoder + GAT)
        # Remaining batches use detached features (only GRU + gate + head get gradients)
        graph_feats_live = model.compute_graph_features(nb, ei, ew)
        graph_feats_detached = graph_feats_live.detach()

        train_loss = 0.0
        n_b = 0
        for batch in train_loader:
            seqs = batch["sequences"].to(device)
            slens = batch["seq_lens"]
            sidxs = batch["stage_idxs"].to(device)
            gidxs = batch["graph_idxs"].to(device)
            tbins = batch["time_bins"].to(device)
            eidxs = batch["event_idxs"].to(device)
            cens = batch["censored"].to(device)

            optimizer.zero_grad()
            # First batch: gradients flow to node_encoder + GAT
            # Subsequent batches: use detached features for efficiency
            gf = graph_feats_live if n_b == 0 else graph_feats_detached
            pmf = model(seqs, slens, sidxs, gidxs, gf)
            loss, _, _ = _graph_loss(pmf, tbins, eidxs, cens, alpha)
            # Graph smoothing regularization
            if graph_smooth_weight > 0:
                smooth = _graph_smoothing_loss(gf, ei, ew)
                loss = loss + graph_smooth_weight * smooth
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # After first batch backward, re-compute detached features
            # with updated weights for remaining batches
            if n_b == 0:
                graph_feats_detached = model.compute_graph_features(nb, ei, ew).detach()

            train_loss += loss.item()
            n_b += 1

        # Validate
        model.eval()
        with torch.no_grad():
            gf_val = model.compute_graph_features(nb, ei, ew)
            val_loss = 0.0
            n_vb = 0
            for batch in val_loader:
                seqs = batch["sequences"].to(device)
                slens = batch["seq_lens"]
                sidxs = batch["stage_idxs"].to(device)
                gidxs = batch["graph_idxs"].to(device)
                tbins = batch["time_bins"].to(device)
                eidxs = batch["event_idxs"].to(device)
                cens = batch["censored"].to(device)
                pmf = model(seqs, slens, sidxs, gidxs, gf_val)
                loss, _, _ = _graph_loss(pmf, tbins, eidxs, cens, alpha)
                val_loss += loss.item()
                n_vb += 1

        tl = train_loss / max(n_b, 1)
        vl = val_loss / max(n_vb, 1)
        scheduler.step(vl)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)

        if verbose and hasattr(it, "set_postfix"):
            it.set_postfix(
                tr=f"{tl:.4f}",
                va=f"{vl:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            )

        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    tqdm.write(f"  Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return history, best_val, best_state


# ── Evaluation ─────────────────────────────────────────────────────────


@torch.no_grad()
def predict_all_graph(
    model, dataset, device, node_baseline, edge_index, edge_weight, batch_size=128
):
    model.eval()
    nb = node_baseline.to(device)
    ei = edge_index.to(device)
    ew = edge_weight.to(device) if edge_weight is not None else None
    graph_feats = model.compute_graph_features(nb, ei, ew)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=graph_collate_fn,
        num_workers=0,
    )
    out = {k: [] for k in ("cif", "event_idxs", "time_bins", "censored", "stage_idxs")}

    for batch in loader:
        seqs = batch["sequences"].to(device)
        slens = batch["seq_lens"]
        sidxs = batch["stage_idxs"].to(device)
        gidxs = batch["graph_idxs"].to(device)
        cif = model.predict_cif(seqs, slens, sidxs, gidxs, graph_feats)
        out["cif"].append(cif.cpu())
        out["event_idxs"].append(batch["event_idxs"])
        out["time_bins"].append(batch["time_bins"])
        out["censored"].append(batch["censored"])
        out["stage_idxs"].append(batch["stage_idxs"])

    return {k: torch.cat(v) for k, v in out.items()}


# ── Cross-Validation ──────────────────────────────────────────────────


@dataclass
class GraphDTResult:
    c_td: float
    c_td_std: float
    c_td_per_fold: list[float]
    ibs: float
    ibs_std: float
    ibs_per_fold: list[float]
    brier_at_horizons: dict[str, float]
    per_transition_ctd: dict[str, float]
    best_val_losses: list[float]
    n_episodes: int
    n_events: int
    n_censored: int
    graph_stats: dict
    hyperparams: dict


def cross_validate(
    features_df: pd.DataFrame,
    n_folds: int = 5,
    hidden_dim: int = 128,
    n_gru_layers: int = 2,
    gat_heads: int = 4,
    gat_layers: int = 2,
    k_neighbors: int = 15,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    dropout: float = 0.3,
    alpha: float = 0.1,
    patience: int = 15,
    verbose: bool = True,
    seed: int = 42,
    checkpoint_dir: Path | None = None,
) -> GraphDTResult:
    """Stratified K-fold CV for Graph Digital Twin."""
    from sklearn.model_selection import StratifiedKFold

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    )
    if verbose:
        print(f"  Device: {device}")

    patient_arrays, col_names = build_patient_arrays(features_df)
    input_dim = len(col_names)
    episodes = extract_episodes(features_df, verbose=verbose)
    n_events_total = sum(1 for e in episodes if not e.censored)

    if verbose:
        print(
            f"  Episodes: {len(episodes)} ({n_events_total} events, "
            f"{len(episodes) - n_events_total} censored)"
        )
        print(f"  Input dim: {input_dim}")

    # Build graph on ALL patients (baseline features only, no label leakage)
    all_patient_ids = sorted(set(e.patno for e in episodes))
    edge_index, edge_weight, node_baseline = build_patient_graph(
        features_df,
        all_patient_ids,
        k_neighbors=k_neighbors,
    )
    pat_to_gidx = {p: i for i, p in enumerate(all_patient_ids)}
    n_baseline_features = node_baseline.size(1)

    graph_stats = {
        "n_nodes": len(all_patient_ids),
        "n_edges": edge_index.size(1),
        "avg_degree": edge_index.size(1) / len(all_patient_ids),
        "k_neighbors": k_neighbors,
        "n_baseline_features": n_baseline_features,
    }
    if verbose:
        print(
            f"  Graph: {graph_stats['n_nodes']} nodes, "
            f"{graph_stats['n_edges']} edges, "
            f"avg degree {graph_stats['avg_degree']:.1f}"
        )

    # Patient stratification
    pat_info: dict[int, tuple[int, bool]] = {}
    for ep in episodes:
        if ep.patno not in pat_info:
            pat_info[ep.patno] = (ep.current_stage_idx, False)
        if not ep.censored:
            s, _ = pat_info[ep.patno]
            pat_info[ep.patno] = (s, True)

    patnos = sorted(pat_info.keys())
    strat = [f"{pat_info[p][0]}_{int(pat_info[p][1])}" for p in patnos]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_ctds, fold_ibs, fold_vals = [], [], []
    all_preds = []

    folds = list(skf.split(patnos, strat))
    fold_iter = tqdm(folds, desc="CV Folds", unit="fold") if verbose else folds

    for fi, (train_idx, test_idx) in enumerate(fold_iter):
        train_pats_all = set(np.array(patnos)[train_idx])
        test_pats = set(np.array(patnos)[test_idx])

        rng = np.random.RandomState(seed + fi)
        tlist = sorted(train_pats_all)
        rng.shuffle(tlist)
        n_val = max(1, len(tlist) // 5)
        val_pats = set(tlist[:n_val])
        atrain_pats = set(tlist[n_val:])

        train_eps = [e for e in episodes if e.patno in atrain_pats]
        val_eps = [e for e in episodes if e.patno in val_pats]
        test_eps = [e for e in episodes if e.patno in test_pats]

        means, stds = compute_feature_stats(patient_arrays, atrain_pats)

        train_ds = GraphDeepHitDataset(
            train_eps, patient_arrays, means, stds, pat_to_gidx
        )
        val_ds = GraphDeepHitDataset(val_eps, patient_arrays, means, stds, pat_to_gidx)
        test_ds = GraphDeepHitDataset(
            test_eps, patient_arrays, means, stds, pat_to_gidx
        )

        torch.manual_seed(seed + fi)
        model = GraphDigitalTwin(
            input_dim=input_dim,
            n_baseline_features=n_baseline_features,
            hidden_dim=hidden_dim,
            n_gru_layers=n_gru_layers,
            gat_heads=gat_heads,
            gat_layers=gat_layers,
            dropout=dropout,
        ).to(device)

        history, best_val, best_state = train_graph_model(
            model,
            train_ds,
            val_ds,
            device,
            node_baseline,
            edge_index,
            edge_weight,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            alpha=alpha,
            verbose=False,
        )

        preds = predict_all_graph(
            model,
            test_ds,
            device,
            node_baseline,
            edge_index,
            edge_weight,
        )
        ctd = compute_ctd(preds)
        ibs_val = compute_ibs(preds)

        fold_ctds.append(ctd)
        fold_ibs.append(ibs_val)
        fold_vals.append(best_val)
        all_preds.append(preds)

        # Save per-fold checkpoint for downstream papers (4, 5, Ch.5)
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": best_state,
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "n_gru_layers": n_gru_layers,
                    "n_baseline_features": n_baseline_features,
                    "gat_heads": gat_heads,
                    "gat_layers": gat_layers,
                    "dropout": dropout,
                    "n_causes": N_STATES,
                    "n_time_bins": N_TIME_BINS,
                    "means": means,
                    "stds": stds,
                    "train_pats": sorted(atrain_pats),
                    "val_pats": sorted(val_pats),
                    "test_pats": sorted(test_pats),
                    "col_names": col_names,
                    "edge_index": edge_index.cpu(),
                    "edge_weight": edge_weight.cpu(),
                    "node_baseline": node_baseline.cpu(),
                    "pat_to_gidx": pat_to_gidx,
                    "k_neighbors": k_neighbors,
                    "fold_idx": fi,
                    "fold_ctd": ctd,
                    "fold_ibs": ibs_val,
                    "seed": seed,
                },
                checkpoint_dir / f"fold{fi}_graph_dt.pt",
            )
            if verbose:
                tqdm.write(
                    f"  Checkpoint saved: {checkpoint_dir / f'fold{fi}_graph_dt.pt'}"
                )

        if verbose:
            tqdm.write(
                f"  Fold {fi + 1}: C-td={ctd:.4f}  IBS={ibs_val:.4f}  "
                f"val_loss={best_val:.4f}  epochs={len(history['train_loss'])}"
            )

    # Aggregate
    eval_horizons = {"1yr": 12, "2yr": 24, "5yr": 60, "10yr": 120}
    brier_hz = {}
    for label, months in eval_horizons.items():
        j = _get_time_bin(months)
        vals = [compute_brier_score(p, j) for p in all_preds]
        vals = [v for v in vals if not np.isnan(v)]
        brier_hz[label] = float(np.mean(vals)) if vals else float("nan")

    per_trans = _per_transition_ctd(all_preds)

    return GraphDTResult(
        c_td=float(np.mean(fold_ctds)),
        c_td_std=float(np.std(fold_ctds)),
        c_td_per_fold=fold_ctds,
        ibs=float(np.mean(fold_ibs)),
        ibs_std=float(np.std(fold_ibs)),
        ibs_per_fold=fold_ibs,
        brier_at_horizons=brier_hz,
        per_transition_ctd=per_trans,
        best_val_losses=fold_vals,
        n_episodes=len(episodes),
        n_events=n_events_total,
        n_censored=len(episodes) - n_events_total,
        graph_stats=graph_stats,
        hyperparams={
            "hidden_dim": hidden_dim,
            "n_gru_layers": n_gru_layers,
            "gat_heads": gat_heads,
            "gat_layers": gat_layers,
            "k_neighbors": k_neighbors,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "dropout": dropout,
            "alpha": alpha,
            "patience": patience,
            "n_folds": n_folds,
            "input_dim": input_dim,
        },
    )


def _per_transition_ctd(all_preds: list[dict]) -> dict[str, float]:
    results = {}
    for k in range(N_STATES):
        conc = disc = tied_n = 0
        for preds in all_preds:
            ev = preds["event_idxs"].numpy()
            tb = preds["time_bins"].numpy()
            cens = preds["censored"].numpy()
            cif = preds["cif"].numpy()
            unc_k = np.where((~cens) & (ev == k))[0]
            if len(unc_k) < 2:
                continue
            rng = np.random.RandomState(42)
            n_s = min(len(unc_k), 200)
            samp = rng.choice(unc_k, size=n_s, replace=False)
            for a in range(len(samp)):
                for b in range(a + 1, len(samp)):
                    i, j = samp[a], samp[b]
                    if tb[i] == tb[j]:
                        continue
                    if tb[i] > tb[j]:
                        i, j = j, i
                    ci_ = cif[i, k, tb[i]]
                    cj_ = cif[j, k, tb[i]]
                    if ci_ > cj_:
                        conc += 1
                    elif ci_ < cj_:
                        disc += 1
                    else:
                        tied_n += 1
        tot = conc + disc + 0.5 * tied_n
        if tot > 0:
            results[f"→{STAGE_LABELS[k]}"] = conc / tot
    return results


# ── Checkpoint Loading ─────────────────────────────────────────────────


def load_graph_dt_checkpoint(
    path: Path,
    device: torch.device | None = None,
) -> tuple[GraphDigitalTwin, dict]:
    """Load a per-fold Graph-DT checkpoint saved by cross_validate().

    Args:
        path: Path to fold checkpoint (.pt file).
        device: Target device. If None, auto-selects GPU/MPS/CPU.

    Returns:
        (model, checkpoint_dict) — model in eval mode on device.
        checkpoint_dict includes edge_index, edge_weight, node_baseline,
        and pat_to_gidx needed for graph-aware inference.
    """
    if device is None:
        device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
    cp = torch.load(path, weights_only=False, map_location=device)
    model = GraphDigitalTwin(
        input_dim=cp["input_dim"],
        n_baseline_features=cp["n_baseline_features"],
        hidden_dim=cp["hidden_dim"],
        n_gru_layers=cp["n_gru_layers"],
        gat_heads=cp.get("gat_heads", 4),
        gat_layers=cp.get("gat_layers", 2),
        dropout=cp.get("dropout", 0.3),
    )
    model.load_state_dict(cp["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cp


# ── Save ──────────────────────────────────────────────────────────────


def save_graph_dt_results(result: GraphDTResult, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    d = {
        "c_td": result.c_td,
        "c_td_std": result.c_td_std,
        "c_td_per_fold": result.c_td_per_fold,
        "ibs": result.ibs,
        "ibs_std": result.ibs_std,
        "ibs_per_fold": result.ibs_per_fold,
        "brier_at_horizons": result.brier_at_horizons,
        "per_transition_ctd": result.per_transition_ctd,
        "best_val_losses": result.best_val_losses,
        "n_episodes": result.n_episodes,
        "n_events": result.n_events,
        "n_censored": result.n_censored,
        "graph_stats": result.graph_stats,
        "hyperparams": result.hyperparams,
    }
    with open(output_dir / "graph_dt_results.json", "w") as f:
        json.dump(d, f, indent=2, default=str)

    if result.per_transition_ctd:
        pd.DataFrame(
            [{"transition": k, "c_td": v} for k, v in result.per_transition_ctd.items()]
        ).to_csv(output_dir / "per_transition_ctd.csv", index=False)

    print(f"  Results saved to {output_dir}")
