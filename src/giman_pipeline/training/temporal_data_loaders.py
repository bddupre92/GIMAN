"""Temporal data loaders for Temporal GIMAN.

Provides PyTorch Dataset and collate functions for variable-length
patient temporal sequences. Handles ragged sequences (patients have
different numbers of visits) via padding and masking.

Classes:
    TemporalPPMIDataset: PyTorch Dataset wrapping PatientSequence objects.

Functions:
    collate_temporal_fn: Custom collate for variable-length temporal batches.
    load_longitudinal_dataset: Load pickled sequences + endpoints.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class TemporalPPMIDataset(Dataset):
    """PyTorch Dataset for variable-length patient temporal sequences.

    Each item returns:
        features: [T_i, F] float tensor (feature values, NaN -> 0)
        obs_mask: [T_i, F] float tensor (1=observed, 0=missing)
        time_months: [T_i] float tensor (months from baseline)
        time_to_event: scalar float (survival time)
        event: scalar float (1=phenoconverted, 0=censored)
        patno: int (patient ID)
        n_visits: int (number of visits for this patient)
    """

    def __init__(
        self,
        sequences: dict[int, Any],  # patno -> PatientSequence
        endpoints: dict[int, tuple[float, float]],  # patno -> (time_to_event, event)
        min_visits: int = 1,
    ) -> None:
        """
        Args:
            sequences: Dict mapping PATNO to PatientSequence objects.
            endpoints: Dict mapping PATNO to (time_to_event, event_indicator).
            min_visits: Minimum number of visits required (patients below this are dropped).
        """
        # Filter to patients present in both sequences and endpoints
        valid_patnos = sorted(
            p for p in sequences
            if p in endpoints and sequences[p].n_visits >= min_visits
        )

        self.patnos = valid_patnos
        self.sequences = sequences
        self.endpoints = endpoints

    def __len__(self) -> int:
        return len(self.patnos)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        patno = self.patnos[idx]
        seq = self.sequences[patno]
        time_to_event, event = self.endpoints[patno]

        return {
            "features": torch.from_numpy(seq.features).float(),  # [T, F]
            "obs_mask": torch.from_numpy(seq.obs_mask).float(),  # [T, F]
            "time_months": torch.from_numpy(seq.time_months).float(),  # [T]
            "time_to_event": torch.tensor(time_to_event, dtype=torch.float32),
            "event": torch.tensor(event, dtype=torch.float32),
            "patno": patno,
            "n_visits": seq.n_visits,
        }


def collate_temporal_fn(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Custom collate function for variable-length temporal sequences.

    Pads all sequences to the maximum length in the batch.
    Creates a seq_mask [N, T_max] indicating valid timesteps.

    Returns dict with:
        features: [N, T_max, F] padded feature tensor
        obs_mask: [N, T_max, F] padded observation mask
        time_months: [N, T_max] padded time tensor
        seq_mask: [N, T_max] binary mask (1=valid timestep, 0=padding)
        time_to_event: [N] survival times
        event: [N] event indicators
        patnos: list[int] patient IDs
        n_visits: [N] number of real visits per patient
    """
    N = len(batch)
    F = batch[0]["features"].shape[1]
    T_max = max(b["n_visits"] for b in batch)

    features = torch.zeros(N, T_max, F)
    obs_mask = torch.zeros(N, T_max, F)
    time_months = torch.zeros(N, T_max)
    seq_mask = torch.zeros(N, T_max)
    time_to_event = torch.zeros(N)
    event = torch.zeros(N)
    patnos = []
    n_visits = torch.zeros(N, dtype=torch.long)

    for i, b in enumerate(batch):
        T_i = b["n_visits"]
        features[i, :T_i, :] = b["features"]
        obs_mask[i, :T_i, :] = b["obs_mask"]
        time_months[i, :T_i] = b["time_months"]
        seq_mask[i, :T_i] = 1.0
        time_to_event[i] = b["time_to_event"]
        event[i] = b["event"]
        patnos.append(b["patno"])
        n_visits[i] = T_i

    return {
        "features": features,
        "obs_mask": obs_mask,
        "time_months": time_months,
        "seq_mask": seq_mask,
        "time_to_event": time_to_event,
        "event": event,
        "patnos": patnos,
        "n_visits": n_visits,
    }


def load_longitudinal_dataset(
    data_dir: str | Path,
    endpoints_path: str | Path | None = None,
    min_visits: int = 1,
) -> TemporalPPMIDataset:
    """Load pickled longitudinal sequences and create a TemporalPPMIDataset.

    Args:
        data_dir: Directory containing longitudinal_sequences.pkl.
        endpoints_path: Path to prodromal_survival_data.csv.
                        If None, looks in standard location.
        min_visits: Minimum visits per patient.

    Returns:
        TemporalPPMIDataset ready for DataLoader.
    """
    import pandas as pd

    data_dir = Path(data_dir)
    seq_path = data_dir / "longitudinal_sequences.pkl"

    with open(seq_path, "rb") as f:
        sequences = pickle.load(f)

    # Load endpoints
    if endpoints_path is None:
        project_root = Path(__file__).resolve().parents[3]
        endpoints_path = (
            project_root / "data" / "prodromal_cohort" / "prodromal_survival_data.csv"
        )
    endpoints_df = pd.read_csv(endpoints_path)

    # Build endpoint dict: patno -> (time_to_event, event)
    endpoints_dict: dict[int, tuple[float, float]] = {}
    for _, row in endpoints_df.iterrows():
        endpoints_dict[int(row["PATNO"])] = (
            float(row["time_to_event"]),
            float(row["phenoconverted"]),
        )

    dataset = TemporalPPMIDataset(
        sequences=sequences,
        endpoints=endpoints_dict,
        min_visits=min_visits,
    )

    print(
        f"Loaded TemporalPPMIDataset: {len(dataset)} patients "
        f"(min_visits={min_visits})"
    )
    return dataset
