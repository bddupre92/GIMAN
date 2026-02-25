"""
Dynamic-DeepHit for NSD-ISS Stage Transitions.

Implements a GRU-based deep survival model for competing-risks stage transition
prediction, adapted from Lee et al. (2019) "Dynamic-DeepHit".

Architecture:
    Visit features → GRU encoder → patient representation
    [patient_repr, stage_embedding] → shared FC → cause-specific CIF

Key adaptations for NSD-ISS:
    - Bidirectional transitions (forward + backward)
    - Stage-conditioned outputs
    - Episode-level training (one episode = one stage occupancy)
    - Missingness indicators for partially observed features
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from giman_pipeline.paper3.multistate_markov import STAGE_LABELS, STAGE_TO_IDX, N_STATES

# ── Feature Configuration ──────────────────────────────────────────────

TIME_VARYING_FEATURES = [
    "updrs1_total",
    "updrs2_total",
    "updrs3_total",
    "hy_stage",
    "moca_total",
    "ess_total",
    "rbd_total",
    "scopa_aut_total",
    "pdmedyn",
    "nsd_stage_numeric",
    "months_from_baseline",
    "time_in_current_stage_months",
]

STATIC_FEATURES = [
    "age_at_baseline",
    "sex",
    "lrrk2_carrier",
    "gba_carrier",
]

ALL_FEATURES = TIME_VARYING_FEATURES + STATIC_FEATURES

# Features with >5% missingness that get binary mask indicators
FEATURES_WITH_MISSING = [
    "updrs3_total",
    "moca_total",
    "ess_total",
    "rbd_total",
    "scopa_aut_total",
    "pdmedyn",
]

# Discrete time bins (upper bound in months)
TIME_BIN_ENDS = [3, 6, 12, 18, 24, 36, 48, 60, 84, 120, 180]
N_TIME_BINS = len(TIME_BIN_ENDS)


# ── Data Preparation ───────────────────────────────────────────────────

@dataclass
class Episode:
    """A stage-occupancy episode for one patient."""
    patno: int
    current_stage_idx: int
    event_stage_idx: int  # destination stage; -1 if censored
    duration_months: float
    censored: bool
    max_visit_idx: int  # last visit index (into patient array) for this episode


def _get_time_bin(duration_months: float) -> int:
    """Map duration to its discrete time bin index."""
    for j, t_end in enumerate(TIME_BIN_ENDS):
        if duration_months <= t_end:
            return j
    return N_TIME_BINS - 1


def extract_episodes(features_df: pd.DataFrame, verbose: bool = True) -> list[Episode]:
    """Extract stage-occupancy episodes from longitudinal data."""
    episodes = []
    grouped = features_df.groupby("PATNO")
    iterator = tqdm(grouped, desc="Extracting episodes", unit="patient") if verbose else grouped

    for patno, pdf in iterator:
        pdf = pdf.sort_values("months_from_baseline").reset_index(drop=True)
        if len(pdf) < 1:
            continue

        current_stage = pdf.iloc[0]["nsd_stage"]
        ep_start_time = pdf.iloc[0]["months_from_baseline"]
        current_stage_idx = STAGE_TO_IDX.get(current_stage, 0)

        for i in range(1, len(pdf)):
            new_stage = pdf.iloc[i]["nsd_stage"]
            if new_stage != current_stage:
                duration = pdf.iloc[i]["months_from_baseline"] - ep_start_time
                episodes.append(Episode(
                    patno=patno,
                    current_stage_idx=current_stage_idx,
                    event_stage_idx=STAGE_TO_IDX.get(new_stage, 0),
                    duration_months=max(duration, 0.01),
                    censored=False,
                    max_visit_idx=i,
                ))
                current_stage = new_stage
                current_stage_idx = STAGE_TO_IDX.get(new_stage, 0)
                ep_start_time = pdf.iloc[i]["months_from_baseline"]

        # Last episode is censored
        duration = pdf.iloc[-1]["months_from_baseline"] - ep_start_time
        episodes.append(Episode(
            patno=patno,
            current_stage_idx=current_stage_idx,
            event_stage_idx=-1,
            duration_months=max(duration, 0.01),
            censored=True,
            max_visit_idx=len(pdf) - 1,
        ))

    return episodes


def build_patient_arrays(
    features_df: pd.DataFrame,
) -> tuple[dict[int, np.ndarray], list[str]]:
    """Build per-patient feature arrays with missingness indicators.

    Returns:
        patient_arrays: {patno: (n_visits, n_input_features) float32}
        column_names: list of column names matching the array columns
    """
    df = features_df.copy()

    # Create missingness indicators
    mask_cols = []
    for f in FEATURES_WITH_MISSING:
        if f in df.columns:
            mc = f"_m_{f}"
            df[mc] = df[f].isna().astype(np.float32)
            mask_cols.append(mc)

    # Fill missing feature values with 0 (will be standardised later)
    for col in ALL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    all_cols = [c for c in ALL_FEATURES if c in df.columns] + mask_cols

    patient_arrays: dict[int, np.ndarray] = {}
    for patno, group in df.groupby("PATNO"):
        group = group.sort_values("months_from_baseline")
        patient_arrays[patno] = group[all_cols].values.astype(np.float32)

    return patient_arrays, all_cols


def compute_feature_stats(
    patient_arrays: dict[int, np.ndarray],
    patnos: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean/std from a subset of patients."""
    arrs = [patient_arrays[p] for p in patnos if p in patient_arrays]
    if not arrs:
        dim = next(iter(patient_arrays.values())).shape[1]
        return np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32)
    stacked = np.concatenate(arrs, axis=0)
    means = np.nanmean(stacked, axis=0).astype(np.float32)
    stds = np.nanstd(stacked, axis=0).astype(np.float32)
    stds[stds < 1e-8] = 1.0
    return means, stds


# ── Dataset / DataLoader ───────────────────────────────────────────────

class DeepHitDataset(Dataset):
    def __init__(
        self,
        episodes: list[Episode],
        patient_arrays: dict[int, np.ndarray],
        means: np.ndarray,
        stds: np.ndarray,
    ):
        self.episodes = episodes
        self.patient_arrays = patient_arrays
        self.means = means
        self.stds = stds

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
            "patno": ep.patno,
        }


def collate_fn(batch):
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
        "patnos": [item["patno"] for item in batch],
    }


# ── Model ──────────────────────────────────────────────────────────────

class DynamicDeepHit(nn.Module):
    """GRU encoder + cause-specific discrete-time hazard heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_gru_layers: int = 2,
        n_causes: int = N_STATES,
        n_time_bins: int = N_TIME_BINS,
        stage_embed_dim: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_causes = n_causes
        self.n_time_bins = n_time_bins

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_gru_layers,
            batch_first=True,
            dropout=dropout if n_gru_layers > 1 else 0.0,
        )

        self.stage_embed = nn.Embedding(N_STATES, stage_embed_dim)

        shared_in = hidden_dim + stage_embed_dim
        self.shared_net = nn.Sequential(
            nn.Linear(shared_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Joint distribution over (cause, time_bin) + 1 "no-event" category
        self.output_net = nn.Linear(64, n_causes * n_time_bins + 1)

    def forward(self, sequences, seq_lens, stage_idxs):
        """Returns PMF of shape (batch, n_causes * n_time_bins + 1)."""
        # Sort by length for packing
        sorted_lens, sort_idx = seq_lens.sort(descending=True)
        sorted_seqs = sequences[sort_idx]
        sorted_lens = sorted_lens.clamp(min=1)

        packed = pack_padded_sequence(sorted_seqs, sorted_lens.cpu(), batch_first=True)
        _, h_n = self.gru(packed)
        patient_repr = h_n[-1]  # (batch, hidden_dim)

        # Unsort
        _, unsort_idx = sort_idx.sort()
        patient_repr = patient_repr[unsort_idx]

        stage_emb = self.stage_embed(stage_idxs)
        combined = torch.cat([patient_repr, stage_emb], dim=-1)
        shared = self.shared_net(combined)
        logits = self.output_net(shared)
        return F.softmax(logits, dim=-1)

    def predict_cif(self, sequences, seq_lens, stage_idxs):
        """Cause-specific CIF: (batch, n_causes, n_time_bins)."""
        pmf = self.forward(sequences, seq_lens, stage_idxs)
        event_pmf = pmf[:, :-1].view(-1, self.n_causes, self.n_time_bins)
        return torch.cumsum(event_pmf, dim=-1)


# ── Loss ───────────────────────────────────────────────────────────────

def _nll_loss(pmf, time_bins, event_idxs, censored):
    """Vectorized NLL: -log P(observed outcome)."""
    batch = pmf.size(0)
    event_pmf = pmf[:, :-1].view(batch, N_STATES, N_TIME_BINS)

    # Build cumulative event probability up to each time bin (inclusive)
    cum_event = torch.cumsum(event_pmf, dim=-1)  # (B, K, J)
    # Total event prob up to time_bin j across all causes
    cum_total = cum_event.sum(dim=1)  # (B, J)

    # Uncensored loss: -log P(event=k, time=j)
    unc_mask = ~censored
    if unc_mask.any():
        unc_k = event_idxs[unc_mask]  # destination stage indices
        unc_j = time_bins[unc_mask]
        # Gather P(event=k, time=j) for each uncensored sample
        unc_probs = event_pmf[unc_mask]  # (n_unc, K, J)
        unc_p = unc_probs[torch.arange(unc_k.size(0)), unc_k, unc_j]
        unc_loss = -torch.log(unc_p + 1e-8).sum()
    else:
        unc_loss = torch.tensor(0.0, device=pmf.device)

    # Censored loss: -log P(T > t_c) = -log(1 - sum of event probs up to t_c)
    cen_mask = censored
    if cen_mask.any():
        cen_j = time_bins[cen_mask]
        cen_cum = cum_total[cen_mask]  # (n_cen, J)
        p_event_before = cen_cum[torch.arange(cen_j.size(0)), cen_j]
        cen_loss = -torch.log((1.0 - p_event_before).clamp(min=1e-8)).sum()
    else:
        cen_loss = torch.tensor(0.0, device=pmf.device)

    return (unc_loss + cen_loss) / batch


def _ranking_loss(pmf, time_bins, event_idxs, censored, sigma=0.1):
    """Vectorized ranking loss on sampled concordant pairs."""
    batch = pmf.size(0)
    event_pmf = pmf[:, :-1].view(batch, N_STATES, N_TIME_BINS)
    cif = torch.cumsum(event_pmf, dim=-1)  # (B, K, J)

    uncensored = torch.where(~censored)[0]
    if len(uncensored) < 2:
        return torch.tensor(0.0, device=pmf.device)

    # Sample pairs: pick N uncensored, form all (i,j) pairs
    n_sample = min(len(uncensored), 64)
    perm = torch.randperm(len(uncensored), device=pmf.device)[:n_sample]
    sampled = uncensored[perm]

    s_events = event_idxs[sampled]   # (N,)
    s_tbins = time_bins[sampled]     # (N,)

    # All pairs (i, j) with i < j
    ii = torch.arange(n_sample, device=pmf.device)
    row, col = torch.meshgrid(ii, ii, indexing="ij")
    mask = row < col
    ri, ci = row[mask], col[mask]

    # Filter: same event type, different time
    same_event = s_events[ri] == s_events[ci]
    diff_time = s_tbins[ri] != s_tbins[ci]
    valid = same_event & diff_time

    if not valid.any():
        return torch.tensor(0.0, device=pmf.device)

    ri, ci = ri[valid], ci[valid]

    # Ensure ri has earlier event
    swap = s_tbins[ri] > s_tbins[ci]
    ri_f = torch.where(swap, ci, ri)
    ci_f = torch.where(swap, ri, ci)

    idx_i = sampled[ri_f]
    idx_j = sampled[ci_f]
    k = s_events[ri_f]
    t = s_tbins[ri_f]

    cif_i = cif[idx_i, k, t]
    cif_j = cif[idx_j, k, t]

    loss = torch.exp(-(cif_i - cif_j) / sigma).mean()
    return loss


def deephit_loss(pmf, time_bins, event_idxs, censored, alpha=0.1, sigma=0.1):
    # Move to CPU for loss computation (avoids MPS kernel overhead on indexing)
    pmf_cpu = pmf.cpu() if pmf.device.type != "cpu" else pmf
    tb_cpu = time_bins.cpu() if time_bins.device.type != "cpu" else time_bins
    ei_cpu = event_idxs.cpu() if event_idxs.device.type != "cpu" else event_idxs
    ce_cpu = censored.cpu() if censored.device.type != "cpu" else censored

    nll = _nll_loss(pmf_cpu, tb_cpu, ei_cpu, ce_cpu)
    rank = _ranking_loss(pmf_cpu, tb_cpu, ei_cpu, ce_cpu, sigma)
    total = nll + alpha * rank

    # Move back to original device for backward pass
    if pmf.device.type != "cpu":
        # Re-derive loss from original pmf to maintain gradient graph
        # We use the CPU-computed indices to do a single gather on GPU
        return _gpu_loss(pmf, time_bins, event_idxs, censored, alpha, sigma)
    return total, nll, rank


def _gpu_loss(pmf, time_bins, event_idxs, censored, alpha=0.1, sigma=0.1):
    """Efficient GPU loss using minimal indexing."""
    batch = pmf.size(0)
    event_pmf = pmf[:, :-1].view(batch, N_STATES, N_TIME_BINS)
    cum_event = torch.cumsum(event_pmf, dim=-1)
    cum_total = cum_event.sum(dim=1)  # (B, J)

    # NLL
    unc_mask = ~censored
    loss = torch.tensor(0.0, device=pmf.device)

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

    # Ranking: simplified — skip pair enumeration, use a batch-level proxy
    # Sort uncensored by time, compare adjacent predictions
    if alpha > 0 and unc_mask.sum() > 2:
        cif = cum_event  # (B, K, J)
        unc_idx = torch.where(unc_mask)[0]
        unc_k = event_idxs[unc_idx]
        unc_j = time_bins[unc_idx]

        # Sort by time within same cause
        sort_order = torch.argsort(unc_j)
        s_idx = unc_idx[sort_order]
        s_k = unc_k[sort_order]
        s_j = unc_j[sort_order]

        # Compare consecutive pairs with same cause
        same_k = s_k[:-1] == s_k[1:]
        diff_t = s_j[:-1] != s_j[1:]
        valid = same_k & diff_t
        if valid.any():
            vi = torch.where(valid)[0]
            # Earlier event patient should have higher CIF
            ci_early = cif[s_idx[vi], s_k[vi], s_j[vi]]
            ci_later = cif[s_idx[vi + 1], s_k[vi], s_j[vi]]
            rank_loss = torch.exp(-(ci_early - ci_later) / 0.1).mean()
            loss = loss + alpha * rank_loss

    return loss, nll_val, loss.detach() - nll_val


# ── Training ───────────────────────────────────────────────────────────

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_epoch(model, loader, device, optimizer=None, alpha=0.1):
    """Train or evaluate for one epoch."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = total_nll = total_rank = 0.0
    n_batches = 0

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for batch in loader:
            seqs = batch["sequences"].to(device)
            slens = batch["seq_lens"]
            sidxs = batch["stage_idxs"].to(device)
            tbins = batch["time_bins"].to(device)
            eidxs = batch["event_idxs"].to(device)
            cens = batch["censored"].to(device)

            pmf = model(seqs, slens, sidxs)
            loss, nll, rank = deephit_loss(pmf, tbins, eidxs, cens, alpha=alpha)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            total_nll += nll.item()
            total_rank += rank.item()
            n_batches += 1

    d = max(n_batches, 1)
    return {"loss": total_loss / d, "nll": total_nll / d, "ranking": total_rank / d}


def train_model(
    model, train_ds, val_ds, device,
    n_epochs=100, batch_size=64, lr=1e-3, weight_decay=1e-4,
    patience=15, alpha=0.1, verbose=True,
):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=7, factor=0.5, min_lr=1e-6,
    )

    best_val = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    it = tqdm(range(n_epochs), desc="Training", unit="epoch") if verbose else range(n_epochs)

    for epoch in it:
        train_m = _run_epoch(model, train_loader, device, optimizer, alpha)
        val_m = _run_epoch(model, val_loader, device, alpha=alpha)
        scheduler.step(val_m["loss"])

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])

        if verbose and hasattr(it, "set_postfix"):
            it.set_postfix(tr=f"{train_m['loss']:.4f}", va=f"{val_m['loss']:.4f}",
                           lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
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
def predict_all(model, dataset, device, batch_size=128):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    out = {k: [] for k in ("cif", "event_idxs", "time_bins", "censored", "stage_idxs")}

    for batch in loader:
        seqs = batch["sequences"].to(device)
        slens = batch["seq_lens"]
        sidxs = batch["stage_idxs"].to(device)
        cif = model.predict_cif(seqs, slens, sidxs)
        out["cif"].append(cif.cpu())
        out["event_idxs"].append(batch["event_idxs"])
        out["time_bins"].append(batch["time_bins"])
        out["censored"].append(batch["censored"])
        out["stage_idxs"].append(batch["stage_idxs"])

    return {k: torch.cat(v) for k, v in out.items()}


def compute_ctd(preds: dict) -> float:
    """Time-dependent concordance index (sampled pairs)."""
    cif = preds["cif"].numpy()
    events = preds["event_idxs"].numpy()
    tbins = preds["time_bins"].numpy()
    cens = preds["censored"].numpy()

    uncensored = np.where(~cens)[0]
    if len(uncensored) < 2:
        return 0.5

    rng = np.random.RandomState(42)
    concordant = discordant = tied = 0
    n_max = min(50_000, len(uncensored) * (len(uncensored) - 1) // 2)

    for _ in range(n_max):
        idx = rng.choice(uncensored, size=2, replace=False)
        i, j = idx
        if events[i] != events[j] or tbins[i] == tbins[j]:
            continue
        if tbins[i] > tbins[j]:
            i, j = j, i
        k, t = events[i], tbins[i]
        ci, cj = cif[i, k, t], cif[j, k, t]
        if ci > cj:
            concordant += 1
        elif ci < cj:
            discordant += 1
        else:
            tied += 1

    total = concordant + discordant + 0.5 * tied
    return concordant / total if total > 0 else 0.5


def compute_brier_score(preds: dict, eval_bin: int) -> float:
    """Brier score at a specific time bin."""
    cif = preds["cif"].numpy()
    events = preds["event_idxs"].numpy()
    tbins = preds["time_bins"].numpy()
    cens = preds["censored"].numpy()

    bs = 0.0
    n = 0
    for i in range(len(events)):
        if cens[i] and tbins[i] < eval_bin:
            continue  # censored before horizon
        n += 1
        for k in range(N_STATES):
            indicator = 1.0 if (not cens[i] and events[i] == k and tbins[i] <= eval_bin) else 0.0
            pred = cif[i, k, min(eval_bin, N_TIME_BINS - 1)]
            bs += (indicator - pred) ** 2

    return bs / (n * N_STATES) if n > 0 else float("nan")


def compute_ibs(preds: dict) -> float:
    """Integrated Brier score across all time bins."""
    scores, weights = [], []
    prev = 0
    for j, t_end in enumerate(TIME_BIN_ENDS):
        b = compute_brier_score(preds, j)
        if not np.isnan(b):
            dt = t_end - prev
            scores.append(b * dt)
            weights.append(dt)
        prev = t_end
    return sum(scores) / sum(weights) if weights else float("nan")


# ── Cross-Validation ──────────────────────────────────────────────────

@dataclass
class DeepHitResult:
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
    hyperparams: dict


def cross_validate(
    features_df: pd.DataFrame,
    n_folds: int = 5,
    hidden_dim: int = 128,
    n_gru_layers: int = 2,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    dropout: float = 0.3,
    alpha: float = 0.1,
    patience: int = 15,
    verbose: bool = True,
    seed: int = 42,
    checkpoint_dir: Path | None = None,
) -> DeepHitResult:
    """Stratified K-fold cross-validation for Dynamic-DeepHit."""
    from sklearn.model_selection import StratifiedKFold

    device = _get_device()
    if verbose:
        print(f"  Device: {device}")

    # One-time data prep
    patient_arrays, col_names = build_patient_arrays(features_df)
    input_dim = len(col_names)
    episodes = extract_episodes(features_df, verbose=verbose)
    n_events_total = sum(1 for e in episodes if not e.censored)

    if verbose:
        print(f"  Episodes: {len(episodes)} ({n_events_total} events, "
              f"{len(episodes) - n_events_total} censored)")
        print(f"  Input dim: {input_dim} ({len(ALL_FEATURES)} features "
              f"+ {len(FEATURES_WITH_MISSING)} masks)")

    # Patient-level stratification
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
        train_pats = set(np.array(patnos)[train_idx])
        test_pats = set(np.array(patnos)[test_idx])

        # Split train → actual_train + val (80/20)
        rng = np.random.RandomState(seed + fi)
        tlist = sorted(train_pats)
        rng.shuffle(tlist)
        n_val = max(1, len(tlist) // 5)
        val_pats = set(tlist[:n_val])
        atrain_pats = set(tlist[n_val:])

        train_eps = [e for e in episodes if e.patno in atrain_pats]
        val_eps = [e for e in episodes if e.patno in val_pats]
        test_eps = [e for e in episodes if e.patno in test_pats]

        # Per-fold standardisation from train patients only
        means, stds = compute_feature_stats(patient_arrays, atrain_pats)

        train_ds = DeepHitDataset(train_eps, patient_arrays, means, stds)
        val_ds = DeepHitDataset(val_eps, patient_arrays, means, stds)
        test_ds = DeepHitDataset(test_eps, patient_arrays, means, stds)

        torch.manual_seed(seed + fi)
        model = DynamicDeepHit(
            input_dim=input_dim, hidden_dim=hidden_dim,
            n_gru_layers=n_gru_layers, dropout=dropout,
        ).to(device)

        history, best_val, best_state = train_model(
            model, train_ds, val_ds, device,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            patience=patience, alpha=alpha, verbose=False,
        )

        preds = predict_all(model, test_ds, device)
        ctd = compute_ctd(preds)
        ibs = compute_ibs(preds)

        fold_ctds.append(ctd)
        fold_ibs.append(ibs)
        fold_vals.append(best_val)
        all_preds.append(preds)

        # Save per-fold checkpoint for downstream papers (4, 5, Ch.5)
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": best_state,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "n_gru_layers": n_gru_layers,
                "dropout": dropout,
                "n_causes": N_STATES,
                "n_time_bins": N_TIME_BINS,
                "means": means,
                "stds": stds,
                "train_pats": sorted(atrain_pats),
                "val_pats": sorted(val_pats),
                "test_pats": sorted(test_pats),
                "col_names": col_names,
                "fold_idx": fi,
                "fold_ctd": ctd,
                "fold_ibs": ibs,
                "seed": seed,
            }, checkpoint_dir / f"fold{fi}_deephit.pt")
            if verbose:
                tqdm.write(f"  Checkpoint saved: {checkpoint_dir / f'fold{fi}_deephit.pt'}")

        if verbose:
            tqdm.write(f"  Fold {fi+1}: C-td={ctd:.4f}  IBS={ibs:.4f}  "
                       f"val_loss={best_val:.4f}  epochs={len(history['train_loss'])}")

    # Aggregate metrics
    eval_horizons = {"1yr": 12, "2yr": 24, "5yr": 60, "10yr": 120}
    brier_hz = {}
    for label, months in eval_horizons.items():
        j = _get_time_bin(months)
        vals = [compute_brier_score(p, j) for p in all_preds]
        vals = [v for v in vals if not np.isnan(v)]
        brier_hz[label] = float(np.mean(vals)) if vals else float("nan")

    per_trans = _per_transition_ctd(all_preds)

    return DeepHitResult(
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
        hyperparams={
            "hidden_dim": hidden_dim, "n_gru_layers": n_gru_layers,
            "n_epochs": n_epochs, "batch_size": batch_size,
            "lr": lr, "dropout": dropout, "alpha": alpha,
            "patience": patience, "n_folds": n_folds, "input_dim": input_dim,
        },
    )


def _per_transition_ctd(all_preds: list[dict]) -> dict[str, float]:
    """Per-destination-stage C-td across folds."""
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

def load_deephit_checkpoint(
    path: Path,
    device: torch.device | None = None,
) -> tuple["DynamicDeepHit", dict]:
    """Load a per-fold DeepHit checkpoint saved by cross_validate().

    Args:
        path: Path to fold checkpoint (.pt file).
        device: Target device. If None, auto-selects GPU/MPS/CPU.

    Returns:
        (model, checkpoint_dict) — model in eval mode on device.
    """
    if device is None:
        device = _get_device()
    cp = torch.load(path, weights_only=False, map_location=device)
    model = DynamicDeepHit(
        input_dim=cp["input_dim"],
        hidden_dim=cp["hidden_dim"],
        n_gru_layers=cp["n_gru_layers"],
        dropout=cp.get("dropout", 0.3),
    )
    model.load_state_dict(cp["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cp


# ── Save ──────────────────────────────────────────────────────────────

def save_deephit_results(result: DeepHitResult, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    d = {
        "c_td": result.c_td, "c_td_std": result.c_td_std,
        "c_td_per_fold": result.c_td_per_fold,
        "ibs": result.ibs, "ibs_std": result.ibs_std,
        "ibs_per_fold": result.ibs_per_fold,
        "brier_at_horizons": result.brier_at_horizons,
        "per_transition_ctd": result.per_transition_ctd,
        "best_val_losses": result.best_val_losses,
        "n_episodes": result.n_episodes,
        "n_events": result.n_events,
        "n_censored": result.n_censored,
        "hyperparams": result.hyperparams,
    }

    with open(output_dir / "deephit_results.json", "w") as f:
        json.dump(d, f, indent=2, default=str)

    if result.per_transition_ctd:
        pd.DataFrame([
            {"transition": k, "c_td": v}
            for k, v in result.per_transition_ctd.items()
        ]).to_csv(output_dir / "per_transition_ctd.csv", index=False)

    print(f"  Results saved to {output_dir}")
