"""Smoke benchmark runner — importable + CLI-wrapped.

Functions in this module are importable for unit-testing and also invoked
by scripts/phase1/run_smoke_benchmark.py as CLI entry points.

Pipeline per seed:
  1. Load PPMI 33-feature dataset (real, or a mock dataset fixture for tests).
  2. Apply MCAR mask at specified mask_fraction using the seed RNG.
  3. Train phys-GIMIN-lit for n_epochs via PhysGIMINTrainer.
  4. Compute RMSE on masked-but-observed entries.
  5. Run Mean baseline for reference (fills missing with column means).
  6. Emit per-seed JSON: outputs/paper12_phys_gimin/runs/{name}_seed{N}_{ts}/results.json.

Multi-seed + multi-fraction orchestration is done by run_multi_seed() which
dispatches single-seed runs in sequence.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from phys_gimin.model import PhysGIMIN
from phys_gimin.priors.literature import LiteraturePriorProvider
from phys_gimin.regularizer import PhysicsRegularizer
from phys_gimin.trajectory_cache import TrajectoryCache
from phys_gimin.training import PhysGIMINTrainer
from phys_gimin.utils.paths import get_project_root, get_device


@dataclass
class SmokeRunResult:
    method: str                   # "phys_gimin_lit" or "mean"
    seed: int
    mask_fraction: float
    final_rmse: float
    wall_time_s: float
    run_dir: str
    n_epochs: int
    n_patients: int
    n_features: int
    status: str                   # "completed" or "failed"


# ── Real PPMI data loader (reuses Paper 2 pipeline) ────────────────────────────

def _load_real_ppmi_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[int]]:
    """Load the Paper 2 canonical 33-feature PPMI dataset via the existing pipeline.

    This reuses scripts/run_paper2_experiments.py::load_data() without modification.
    The Paper 2 script is importable as-is because argparse lives inside parse_args()
    and CLI dispatch is guarded by `if __name__ == "__main__"`.

    Returns:
        features_np: (2197, 33) float32  — un-normalized raw features
        mask_np:     (2197, 33) float32  — 1 = observed, 0 = missing
        stages_np:   (2197,) int         — NSD-ISS stages (0/1/2/3/4; 5 = unclassified)
        feature_names: list of feature names (up to 33, depending on availability)
        patnos:      list[int]           — PPMI PATNO per patient (same row order)
    """
    _project_root = get_project_root()
    for _p in [_project_root / "scripts",
               _project_root / "src",
               _project_root / "GIMImpN_imputation"]:
        _ps = str(_p)
        if _ps not in sys.path:
            sys.path.insert(0, _ps)

    # Import without triggering argparse (argparse is in parse_args(); CLI is __main__-guarded).
    from run_paper2_experiments import load_data as _paper2_load_data, STAGE_MAP

    features_np, mask_np, stages_np, feature_names = _paper2_load_data()

    # Reconstruct PATNOs by replaying the same filtering logic as load_data().
    # We load the same parquet + staging CSV to extract PATNO in the merged order.
    import pandas as pd
    data_dir = _project_root / "GIMImpN_imputation" / "outputs"
    staging_path = _project_root / "data" / "04_staging" / "nsd_iss_staging_results.csv"

    features_df = pd.read_parquet(data_dir / "ppmi_full_cohort.parquet")
    staging_df = pd.read_csv(staging_path)
    staging_df["stage_encoded"] = staging_df["nsd_iss_stage"].astype(str).map(STAGE_MAP)
    staging_df["stage_encoded"] = staging_df["stage_encoded"].fillna(5).astype(int)
    staging_valid = staging_df[staging_df["nsd_iss_stage"] != "unclassified"].copy()
    staged_patnos = set(staging_valid["PATNO"].values)

    staged_mask = features_df.index.isin(staged_patnos)
    features_staged = features_df[staged_mask].copy().reset_index()
    features_staged = features_staged.merge(
        staging_valid[["PATNO", "stage_encoded", "nsd_iss_stage"]],
        on="PATNO",
        how="inner",
    )
    patnos = features_staged["PATNO"].tolist()

    return features_np, mask_np, stages_np, feature_names, patnos


# ── Real-data graph builder ─────────────────────────────────────────────────────

def _build_real_graph(
    features_np: np.ndarray,
    mask_np: np.ndarray,
    stages_np: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Build a stage-aware k-NN patient similarity graph (Paper 2 method).

    Tries to import StageAwareGraphBuilder from the main project.
    Falls back to a plain cosine-similarity k-NN (k=15) via sklearn if the
    import fails — still a real graph, just not stage-aware.

    Args:
        features_np: (N, D) float32 un-normalized features
        mask_np:     (N, D) float32 observation mask (1=observed, 0=missing)
        stages_np:   (N,)   int NSD-ISS stage array

    Returns:
        edge_index:   (2, n_edges) long tensor
        edge_weight:  (n_edges,) float tensor
        overlap_frac: (n_edges,) float tensor
        builder_name: str label for provenance logging
    """
    _project_root = get_project_root()
    main_src = str(_project_root / "src")
    if main_src not in sys.path:
        sys.path.insert(0, main_src)

    try:
        from giman_pipeline.imputation.stage_graph_builder import StageAwareGraphBuilder

        builder = StageAwareGraphBuilder(
            k_neighbors=15,
            min_overlap=3,
            stage_affinity_beta=0.3,
            use_stratified=False,
        )
        result = builder.build_full_graph(features_np, mask_np, stages=stages_np)
        edge_index = result["edge_index"]
        edge_weight = result["edge_weight"]
        overlap_frac = result["overlap_frac"]
        builder_name = "stage_aware_knn_k15"

    except ImportError:
        # Fallback: plain cosine k-NN via sklearn (still a real graph)
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import normalize

        # Z-score normalize observed features (zero-fill missing)
        col_means = np.nanmean(
            np.where(mask_np > 0, features_np, np.nan), axis=0
        )
        col_stds = np.nanstd(
            np.where(mask_np > 0, features_np, np.nan), axis=0
        )
        col_stds = np.where(col_stds < 1e-8, 1.0, col_stds)
        feat_norm = (features_np - col_means) * mask_np / col_stds

        nbrs = NearestNeighbors(n_neighbors=16, metric="cosine", algorithm="brute")
        nbrs.fit(feat_norm)
        distances, indices = nbrs.kneighbors(feat_norm)

        N, D = features_np.shape
        src_list, dst_list, weight_list = [], [], []
        for i in range(N):
            for j_pos in range(1, 16):  # skip self (position 0)
                j = indices[i, j_pos]
                w = float(1.0 - distances[i, j_pos])
                if w > 0:
                    src_list.append(i)
                    dst_list.append(j)
                    weight_list.append(w)

        # Symmetrize
        edge_dict: dict[tuple[int, int], float] = {}
        for s, d, w in zip(src_list, dst_list, weight_list, strict=False):
            key = (min(s, d), max(s, d))
            if key not in edge_dict or w > edge_dict[key]:
                edge_dict[key] = w

        final_src, final_dst, final_weight = [], [], []
        for (u, v), w in edge_dict.items():
            final_src.extend([u, v])
            final_dst.extend([v, u])
            final_weight.extend([w, w])

        edge_index = torch.tensor([final_src, final_dst], dtype=torch.long)
        edge_weight = torch.tensor(final_weight, dtype=torch.float32)

        # overlap_frac: fraction of features observed on both ends
        mask_f32 = mask_np.astype(np.float32)
        src_arr = edge_index[0].numpy()
        dst_arr = edge_index[1].numpy()
        overlaps = (mask_f32[src_arr] * mask_f32[dst_arr]).sum(axis=1) / D
        overlap_frac = torch.tensor(overlaps, dtype=torch.float32)
        builder_name = "cosine_knn_k15_fallback"

    return edge_index, edge_weight, overlap_frac, builder_name


# ── Dataset classes ─────────────────────────────────────────────────────────────

class _RealDataDataset(Dataset):
    """Wrapper for REAL PPMI data matching PhysGIMIN's expected batch dict.

    Constructor takes pre-computed graph + real metadata. The smoke benchmark's
    run_single_seed builds the graph via the main-project StageAwareGraphBuilder
    and threads it through here.

    All three data-fidelity properties are guaranteed by the constructor:
      - stage_ids: real NSD-ISS stages from Paper 2 load_data()
      - sbr_0_per_patient: real per-patient striatal DaT-SBR baseline
      - edge_index/edge_weight: k-NN patient similarity graph (k=15, not a chain)
    """

    def __init__(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        stages: torch.Tensor,                        # REAL NSD-ISS stages (long, shape=(n_pts,))
        sbr_0_per_patient: list[float],              # REAL baseline striatal SBR
        edge_index: torch.Tensor,                    # (2, n_edges) from main-project builder
        edge_weight: torch.Tensor,                   # (n_edges,)
        overlap_frac: torch.Tensor,                  # (n_edges,)
        patnos: list[int],                           # patient IDs (real PATNO or sequential)
        ode_feature_indices: list[int],              # which feature columns are DaT-SBR targets
    ):
        self.features = features if isinstance(features, torch.Tensor) else torch.from_numpy(features).float()
        self.mask = mask if isinstance(mask, torch.Tensor) else torch.from_numpy(mask).float()
        self.stage_ids = stages if isinstance(stages, torch.Tensor) else torch.tensor(stages, dtype=torch.long)
        self.sbr_0 = sbr_0_per_patient
        self._edge_index = edge_index
        self._edge_weight = edge_weight
        self._overlap_frac = overlap_frac
        self.patnos = patnos
        self.n_patients = self.features.shape[0]
        # 2 time points match len(ode_feature_indices) = 2
        self.t_years = [np.array([0.0, 1.0])] * self.n_patients
        self._ode_feature_indices = ode_feature_indices

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": self.features,
            "mask": self.mask,
            "edge_index": self._edge_index,
            "edge_weight": self._edge_weight,
            "overlap_frac": self._overlap_frac,
            "stage_ids": self.stage_ids,
            "patnos": self.patnos,
            "t_years_per_patient": self.t_years,
            "sbr_0_per_patient": self.sbr_0,
            "ode_feature_indices": self._ode_feature_indices,
        }


class _MockDataset(Dataset):
    """Mock dataset for smoke benchmark unit tests.

    n_patients × n_features synthetic data.
    ode_feature_indices=[0, 1] — first 2 features treated as DaT-SBR.
    t_years has 2 elements to match len(ode_feature_indices).
    """

    def __init__(self, n_patients: int = 50, n_features: int = 33, seed: int = 1001):
        rng = np.random.default_rng(seed)
        self.features = torch.from_numpy(rng.standard_normal((n_patients, n_features))).float()
        self.mask = torch.ones(n_patients, n_features)
        self.stage_ids = torch.from_numpy(rng.integers(0, 6, (n_patients,))).long()
        self.patnos = list(range(3000, 3000 + n_patients))
        # t_years has 2 elements — matches len(ode_feature_indices)=2
        self.t_years = [np.array([0.0, 1.0])] * n_patients
        self.sbr_0 = [2.5] * n_patients
        self.n_patients = n_patients
        self.n_features = n_features

    def __len__(self) -> int:
        return 1  # one full-graph batch

    def __getitem__(self, idx: int) -> dict:
        n = self.n_patients
        # build a minimal connected edge list: chain 0→1→2→...→n-1 (undirected)
        srcs = list(range(n - 1)) + list(range(1, n))
        dsts = list(range(1, n)) + list(range(n - 1))
        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        edge_weight = torch.ones(len(srcs))
        overlap_frac = torch.ones(len(srcs))
        return {
            "features": self.features,
            "mask": self.mask,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "overlap_frac": overlap_frac,
            "stage_ids": self.stage_ids,
            "patnos": self.patnos,
            "t_years_per_patient": self.t_years,
            "sbr_0_per_patient": self.sbr_0,
            "ode_feature_indices": [0, 1],
        }


def _collate_passthrough(batch: list[dict]) -> dict:
    """batch is a length-1 list of dicts; return the dict directly."""
    return batch[0]


def _apply_mcar_mask(features: torch.Tensor, mask_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MCAR mask to features. Returns (masked_features, mask).

    mask[i, j] = 1 means observed; mask[i, j] = 0 means missing.
    masked_features has 0.0 at missing positions.
    """
    rng = np.random.default_rng(seed)
    n, f = features.shape
    mask = torch.ones(n, f)
    # For each entry independently, set to missing with probability mask_fraction
    missing_indicator = rng.random((n, f)) < mask_fraction
    mask[missing_indicator] = 0.0
    masked_features = features.clone()
    masked_features[mask == 0] = 0.0
    return masked_features, mask


def _compute_rmse(predicted: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute RMSE on originally-masked entries (mask==0 positions)."""
    missing = mask == 0
    if missing.sum() == 0:
        # No masked entries — evaluate on all
        diff = predicted - true
    else:
        diff = predicted[missing] - true[missing]
    return float(torch.sqrt((diff ** 2).mean()).item())


def _run_mean_baseline(
    true_features: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Fill missing entries with per-column mean from observed entries."""
    n, f = true_features.shape
    filled = true_features.clone()
    for j in range(f):
        col_obs_mask = mask[:, j] == 1
        if col_obs_mask.any():
            col_mean = true_features[col_obs_mask, j].mean()
        else:
            col_mean = torch.tensor(0.0)
        missing_rows = mask[:, j] == 0
        filled[missing_rows, j] = col_mean
    return filled


def _build_normalization_scaler(
    true_features: torch.Tensor,
    observed_mask: torch.Tensor,
    mock_data: bool,
) -> "Any | None":
    """Build and fit a ModalityAwareScaler on observed-only entries.

    Uses Paper 2's build_gimin_config + build_scaler_from_config so the
    normalization strategy exactly matches the Paper 2 benchmark.

    Returns None for mock data (no meaningful scale issue there).
    RMSE reported on normalized scale when scaler is applied (labelled
    rmse_zscore in outputs). The directional verdict (phys > Mean or phys < Mean)
    is unchanged by a scale transform.
    """
    if mock_data:
        return None

    # Import Paper 2 pipeline (sys.path already extended in _load_real_ppmi_data)
    from run_paper2_experiments import build_gimin_config as _build_gimin_config
    from gimin.data.scaler import build_scaler_from_config as _build_scaler

    cfg = _build_gimin_config(old_to_new_index=None)  # default 33-feature schema
    scaler = _build_scaler(cfg)
    # Fit on all observed entries of the full (un-MCAR-masked) data.
    scaler.fit(true_features, observed_mask)
    return scaler


def _extract_sbr_0_per_patient(
    features_np: np.ndarray,
    mask_np: np.ndarray,
    feature_names: list[str],
) -> tuple[list[float], int]:
    """Extract per-patient baseline striatal DaT-SBR from the raw (un-normalized) features.

    Uses the mean of observed CAUDATE_L_SBR, CAUDATE_R_SBR, PUTAMEN_L_SBR,
    PUTAMEN_R_SBR columns. Missing SBR columns fall back to cohort median.

    Args:
        features_np: (N, D) float32 raw (un-normalized) features
        mask_np:     (N, D) float32 observation mask (1=observed, 0=missing)
        feature_names: list of column names matching columns of features_np

    Returns:
        sbr_0_per_pt: list[float] of per-patient baseline SBR, length N
        n_fallback:   int count of patients who got cohort-median fallback
    """
    sbr_col_names = ["CAUDATE_L_SBR", "CAUDATE_R_SBR", "PUTAMEN_L_SBR", "PUTAMEN_R_SBR"]
    sbr_col_indices = [
        feature_names.index(name)
        for name in sbr_col_names
        if name in feature_names
    ]

    if len(sbr_col_indices) == 0:
        # No DaT-SBR columns present — use a constant placeholder
        return [2.5] * features_np.shape[0], features_np.shape[0]

    sbr_cols = features_np[:, sbr_col_indices]          # (N, up to 4)
    obs_for_sbr = mask_np[:, sbr_col_indices]           # (N, up to 4)

    with np.errstate(invalid="ignore", divide="ignore"):
        row_sums = (sbr_cols * obs_for_sbr).sum(axis=1)
        row_counts = np.clip(obs_for_sbr.sum(axis=1), 1, None)
        sbr_0_raw = np.where(obs_for_sbr.sum(axis=1) > 0, row_sums / row_counts, np.nan)

    cohort_median = float(np.nanmedian(sbr_0_raw))
    n_fallback = int(np.isnan(sbr_0_raw).sum())
    sbr_0_clean = np.where(np.isnan(sbr_0_raw), cohort_median, sbr_0_raw)

    return sbr_0_clean.tolist(), n_fallback


def run_single_seed(
    method: str,
    seed: int,
    mask_fraction: float,
    n_epochs: int,
    n_patients: int | None,
    n_features: int,
    output_dir: Path,
    mock_data: bool = True,
    device: str | None = None,
) -> SmokeRunResult:
    """Run one seed at one mask fraction. Returns result + writes JSON to output_dir."""
    t0 = time.time()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"{method}_seed{seed}_{ts}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # For provenance — will be populated in the real-data phys_gimin_lit path
    graph_stats: dict | None = None
    sbr_0_stats: dict | None = None

    # Resolve the actual number of patients for result reporting
    _actual_n_patients: int = n_patients if n_patients is not None else 50  # updated below for real data

    try:
        # 1. Build data
        if mock_data:
            _mock_n = n_patients if n_patients is not None else 50
            dataset = _MockDataset(n_patients=_mock_n, n_features=n_features, seed=seed)
            true_features = dataset.features  # shape (n_patients, n_features)
            original_mask = torch.ones_like(true_features)  # mock data is fully observed
            _actual_n_patients = _mock_n
        else:
            # Real-data path (Paper 2 canonical loader reused verbatim)
            features_np, mask_np, stages_np, feature_names, patnos_list = _load_real_ppmi_data()
            n_patients_real = features_np.shape[0]

            # Optional subsampling for test runs (keeps first n_patients if caller requested fewer)
            # When n_patients is None, use the full cohort.
            if n_patients is not None and n_patients < n_patients_real:
                rng = np.random.default_rng(seed)
                idx = rng.choice(n_patients_real, size=n_patients, replace=False)
                features_np = features_np[idx]
                mask_np = mask_np[idx]
                stages_np = stages_np[idx]
                patnos_list = [patnos_list[i] for i in idx]

            true_features = torch.from_numpy(features_np).float()
            original_mask = torch.from_numpy(mask_np).float()
            _actual_n_patients = features_np.shape[0]  # actual count after optional subsampling

        # 2. Apply MCAR mask
        masked_features, mask = _apply_mcar_mask(true_features, mask_fraction, seed=seed + 10000)

        # 2b. Normalization (matches Paper 2 §V.D) — fit on full observed entries.
        # Scaler is shared across method calls within one seed; caller builds it here.
        # RMSE is computed on z-scored scale (Option A from spec).
        scaler = _build_normalization_scaler(true_features, original_mask, mock_data)

        if scaler is not None:
            # Transform both true and masked features to normalized scale.
            # The observed_mask (original_mask) governs which values the scaler
            # uses — we transform using the MCAR mask so missing positions stay 0.
            true_features_norm = scaler.transform(true_features, original_mask)
            masked_features_norm = scaler.transform(masked_features, mask)
        else:
            true_features_norm = true_features
            masked_features_norm = masked_features

        if method == "mean":
            # 3+4. Mean baseline: fill with column means on observed data, compute RMSE.
            # Evaluated on normalized scale for apples-to-apples vs phys_gimin_lit.
            predicted = _run_mean_baseline(masked_features_norm, mask)
            final_rmse = _compute_rmse(predicted, true_features_norm, mask)
            actual_epochs = 1

        elif method == "phys_gimin_lit":
            # 3. Build PhysGIMIN + trainer; train on normalized features.
            torch.manual_seed(seed)
            model = PhysGIMIN(
                modality_dims=[2, 5, 6, 6, 4, 4, 6],
                embed_dim=64,
                num_gnn_layers=2,
                num_heads=4,
                mc_dropout=0.1,
                num_stages=6,
                stage_embed_dim=16,
            )
            provider = LiteraturePriorProvider()
            regularizer = PhysicsRegularizer(provider=provider, beta=0.5)
            cache = TrajectoryCache(provider=provider)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            resolved_device = get_device(override=device)
            trainer = PhysGIMINTrainer(
                model=model,
                regularizer=regularizer,
                trajectory_cache=cache,
                optimizer=optimizer,
                seed=seed,
                warmup_epochs=1,
                lambda_phys_target=1.0,
                min_recon_fraction=0.30,
                output_dir=run_dir,
                run_name="phys_gimin_lit",
                deterministic=True,
                device=resolved_device,
            )

            if mock_data:
                # Mock data: use _MockDataset (chain graph is fine for smoke tests)
                train_dataset: Dataset = _MockDataset(
                    n_patients=n_patients, n_features=n_features, seed=seed
                )
                train_dataset.features = masked_features_norm  # type: ignore[attr-defined]
                train_dataset.mask = mask                       # type: ignore[attr-defined]
                eval_dataset: Dataset = _MockDataset(
                    n_patients=n_patients, n_features=n_features, seed=seed
                )
                eval_dataset.features = masked_features_norm   # type: ignore[attr-defined]
                eval_dataset.mask = mask                        # type: ignore[attr-defined]

            else:
                # Real data: build stage-aware k-NN graph + extract real sbr_0
                # sbr_0 extracted from UN-NORMALIZED features (physics ODE needs real SBR units)
                sbr_0_list, n_fallback = _extract_sbr_0_per_patient(
                    features_np, mask_np, feature_names
                )
                sbr_0_vals = np.array(sbr_0_list)
                sbr_0_stats = {
                    "mean": float(np.mean(sbr_0_vals)),
                    "std": float(np.std(sbr_0_vals)),
                    "min": float(np.min(sbr_0_vals)),
                    "max": float(np.max(sbr_0_vals)),
                    "n_total": len(sbr_0_list),
                    "n_fallback": n_fallback,
                }

                # Build stage-aware k-NN graph (Paper 2 method)
                # Use original mask (not MCAR mask) for graph construction —
                # same convention as Paper 2 which builds graph before artificial masking.
                edge_index, edge_weight, overlap_frac, builder_name = _build_real_graph(
                    features_np, mask_np, stages_np
                )
                n_nodes = features_np.shape[0]
                n_edges = edge_index.shape[1]
                avg_degree = float(n_edges / max(n_nodes, 1))
                graph_stats = {
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    "avg_degree": avg_degree,
                    "builder": builder_name,
                }

                # Identify DaT-SBR feature indices for physics regularizer
                sbr_col_names = ["CAUDATE_L_SBR", "CAUDATE_R_SBR", "PUTAMEN_L_SBR", "PUTAMEN_R_SBR"]
                ode_feature_indices = [
                    feature_names.index(name)
                    for name in sbr_col_names
                    if name in feature_names
                ][:2]  # use first 2 (CAUDATE_L, CAUDATE_R) to match ode_feature_indices length

                real_stages = torch.tensor(stages_np, dtype=torch.long)

                real_dataset = _RealDataDataset(
                    features=masked_features_norm,
                    mask=mask,
                    stages=real_stages,
                    sbr_0_per_patient=sbr_0_list,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    overlap_frac=overlap_frac,
                    patnos=patnos_list,
                    ode_feature_indices=ode_feature_indices if ode_feature_indices else [0, 1],
                )
                train_dataset = real_dataset
                eval_dataset = real_dataset

            loader = DataLoader(train_dataset, batch_size=1, collate_fn=_collate_passthrough)

            # 4. Train
            train_result = trainer.fit(loader, loader, n_epochs=n_epochs, patience=n_epochs + 10)

            if train_result.status == "failed":
                raise RuntimeError(f"Training failed: {train_result.failure_reason}")

            # 5. Compute RMSE on masked entries (normalized scale)
            model.eval()
            full_loader = DataLoader(eval_dataset, batch_size=1, collate_fn=_collate_passthrough)

            _infer_device = torch.device(resolved_device)
            with torch.no_grad():
                batch = next(iter(full_loader))
                output = model(
                    batch["features"].to(_infer_device),
                    batch["mask"].to(_infer_device),
                    batch["edge_index"].to(_infer_device),
                    batch["edge_weight"].to(_infer_device),
                    batch["overlap_frac"].to(_infer_device),
                    batch["stage_ids"].to(_infer_device),
                )
            predicted = output["imputed_mean"].cpu()
            # RMSE on normalized scale (Option A — directional verdict unchanged)
            final_rmse = _compute_rmse(predicted, true_features_norm, mask)
            actual_epochs = train_result.final_epoch + 1

        else:
            raise ValueError(f"Unknown method: {method!r}. Must be 'phys_gimin_lit' or 'mean'.")

        wall_time_s = time.time() - t0
        status = "completed"

    except Exception as exc:
        wall_time_s = time.time() - t0
        status = "failed"
        final_rmse = float("nan")
        actual_epochs = 0
        # Write error info
        import traceback as _tb
        (run_dir / "error.txt").write_text(f"{exc}\n\n{_tb.format_exc()}")

    result = SmokeRunResult(
        method=method,
        seed=seed,
        mask_fraction=mask_fraction,
        final_rmse=final_rmse,
        wall_time_s=wall_time_s,
        run_dir=str(run_dir),
        n_epochs=actual_epochs,
        n_patients=_actual_n_patients,
        n_features=n_features,
        status=status,
    )

    # Write per-seed JSON
    (run_dir / "results.json").write_text(json.dumps(asdict(result), indent=2))

    # Merge graph_stats + sbr_0_stats into provenance.json if they were computed
    if graph_stats is not None or sbr_0_stats is not None:
        prov_path = run_dir / "provenance.json"
        if prov_path.exists():
            existing = json.loads(prov_path.read_text())
        else:
            existing = {}
        if graph_stats is not None:
            existing["graph_stats"] = graph_stats
        if sbr_0_stats is not None:
            existing["sbr_0_stats"] = sbr_0_stats
        prov_path.write_text(json.dumps(existing, indent=2))

    return result


def run_multi_seed(
    method: str,
    seeds: list[int],
    mask_fraction: float,
    n_epochs: int,
    n_patients: int | None = None,
    n_features: int = 33,
    output_dir: Path = Path("outputs/paper12_phys_gimin/runs/smoke_test"),
    mock_data: bool = True,
    device: str | None = None,
) -> list[SmokeRunResult]:
    """Run multiple seeds sequentially; return list of results."""
    results = []
    for seed in seeds:
        result = run_single_seed(
            method=method,
            seed=seed,
            mask_fraction=mask_fraction,
            n_epochs=n_epochs,
            n_patients=n_patients,
            n_features=n_features,
            output_dir=output_dir,
            mock_data=mock_data,
            device=device,
        )
        results.append(result)
    return results


def rmse_cv(results: list[SmokeRunResult]) -> float:
    """Coefficient of variation of final_rmse across a list of seed results."""
    rmses = np.array([r.final_rmse for r in results if r.status == "completed"])
    if len(rmses) == 0 or np.mean(rmses) == 0:
        return float("inf")
    return float(np.std(rmses) / np.mean(rmses))
