"""HDF5-backed persistent store for per-patient posterior samples.

Enables bidirectional updating: when a new observation arrives, we reweight
the stored samples via SIR (sequential importance resampling) rather than
re-running the full IS calibration.

HDF5 schema:
    /patient_<patno>/v<N>/
        ├─ samples (n, d)        # d-dim posterior samples
        ├─ weights (n,)          # normalized weights (sum to 1)
        └─ attrs:
            ess: effective sample size
            log_marg_lik: log marginal likelihood
            param_names: list of parameter names
            source: "phase2_is_v5", "phase2_is_v5_waveb", or "update_v{N}"

The first version (v1) is seeded from the existing Phase 2 IS chain parquets
at outputs/mechanistic_twin/data/posteriors/chains_is_v5{,_waveb}/ which
contain resampled equal-weight posteriors for 1,065 patients.

Subsequent versions (v2, v3, ...) are produced by `update_posterior()` in
updater.py when new observations arrive.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np


@dataclass
class PatientPosterior:
    """Per-patient posterior with samples + IS weights.

    After update via SIR reweighting, weights may no longer be uniform.
    Effective sample size (ESS) tracks the quality of the reweighted posterior.
    """

    patno: int
    version: int
    samples: np.ndarray  # (n, d) — d posterior parameters
    weights: np.ndarray  # (n,) — normalized, sum to 1
    ess: float
    log_marg_lik: float
    param_names: list[str]
    source: str = "unknown"

    def __post_init__(self) -> None:
        if self.samples.ndim != 2:
            raise ValueError(
                f"samples must be 2D (n, d), got shape {self.samples.shape}"
            )
        if self.weights.ndim != 1 or self.weights.shape[0] != self.samples.shape[0]:
            raise ValueError(
                f"weights shape {self.weights.shape} != samples n={self.samples.shape[0]}"
            )
        if len(self.param_names) != self.samples.shape[1]:
            raise ValueError(
                f"param_names ({len(self.param_names)}) != samples d ({self.samples.shape[1]})"
            )

    def posterior_mean(self) -> np.ndarray:
        """Weighted posterior mean. Shape (d,)."""
        return np.sum(self.samples * self.weights[:, None], axis=0)

    def posterior_quantile(self, q: float) -> np.ndarray:
        """Weighted quantile across each parameter. Shape (d,)."""
        from scipy.stats import rankdata

        # For each parameter, compute weighted quantile
        d = self.samples.shape[1]
        result = np.empty(d)
        for j in range(d):
            order = np.argsort(self.samples[:, j])
            sorted_samples = self.samples[order, j]
            sorted_weights = self.weights[order]
            cumw = np.cumsum(sorted_weights)
            # Find first index where cumulative weight >= q
            idx = np.searchsorted(cumw, q)
            idx = min(idx, len(sorted_samples) - 1)
            result[j] = sorted_samples[idx]
        return result


class PosteriorStore:
    """HDF5-backed persistent store for patient posteriors.

    Supports versioning: each update_posterior() call creates a new version
    while preserving history. Use version="latest" for the most recent.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def save(self, posterior: PatientPosterior) -> None:
        """Save or overwrite a patient's posterior at the given version."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.path, "a") as f:
            key = f"patient_{posterior.patno}/v{posterior.version}"
            if key in f:
                del f[key]
            grp = f.create_group(key)
            grp.create_dataset(
                "samples", data=posterior.samples, compression="gzip", compression_opts=4
            )
            grp.create_dataset(
                "weights", data=posterior.weights, compression="gzip", compression_opts=4
            )
            grp.attrs["ess"] = float(posterior.ess)
            grp.attrs["log_marg_lik"] = float(posterior.log_marg_lik)
            grp.attrs["param_names"] = [n.encode("utf-8") for n in posterior.param_names]
            grp.attrs["source"] = posterior.source.encode("utf-8")

    def load(
        self, patno: int, version: int | Literal["latest"] = "latest"
    ) -> PatientPosterior:
        """Load a patient's posterior at a specific version."""
        with h5py.File(self.path, "r") as f:
            pat_key = f"patient_{patno}"
            if pat_key not in f:
                raise KeyError(f"Patient {patno} not in store at {self.path}")
            pat_grp = f[pat_key]
            if version == "latest":
                versions = sorted(
                    int(k[1:]) for k in pat_grp.keys() if k.startswith("v")
                )
                if not versions:
                    raise KeyError(f"No versions for patient {patno}")
                version = versions[-1]
            if f"v{version}" not in pat_grp:
                raise KeyError(f"Patient {patno} has no v{version}")
            grp = pat_grp[f"v{version}"]
            return PatientPosterior(
                patno=int(patno),
                version=int(version),
                samples=grp["samples"][:],
                weights=grp["weights"][:],
                ess=float(grp.attrs["ess"]),
                log_marg_lik=float(grp.attrs["log_marg_lik"]),
                param_names=[
                    n.decode("utf-8") if isinstance(n, bytes) else str(n)
                    for n in grp.attrs["param_names"]
                ],
                source=(
                    grp.attrs["source"].decode("utf-8")
                    if isinstance(grp.attrs["source"], bytes)
                    else str(grp.attrs["source"])
                ),
            )

    def list_patients(self) -> list[int]:
        """List all patient PATNOs in the store."""
        if not self.path.exists():
            return []
        with h5py.File(self.path, "r") as f:
            return sorted(
                int(k.split("_")[1]) for k in f.keys() if k.startswith("patient_")
            )

    def versions(self, patno: int) -> list[int]:
        """List all versions for a given patient."""
        with h5py.File(self.path, "r") as f:
            pat_key = f"patient_{patno}"
            if pat_key not in f:
                return []
            pat_grp = f[pat_key]
            return sorted(
                int(k[1:]) for k in pat_grp.keys() if k.startswith("v")
            )

    def __contains__(self, patno: int) -> bool:
        return patno in self.list_patients()

    def __len__(self) -> int:
        return len(self.list_patients())
