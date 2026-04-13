"""JSON-backed registry for patient digital twin states.

Provides versioned persistence, loading, and audit trail retrieval
for TemporalTwinState objects. Each save creates a timestamped JSON
file, enabling full history reconstruction and reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .temporal_twin_state import TemporalTwinState


class TwinRegistry:
    """JSON-backed registry for patient digital twin states.

    Each patient's state is saved as a timestamped JSON file:
        twin_{patno}_{updated_at}.json

    Loading returns the most recent state. Full audit trails are
    reconstructable from the file history.

    Args:
        registry_dir: Directory for state files. Created if absent.
    """

    def __init__(self, registry_dir: Path) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def save_state(self, state: TemporalTwinState) -> Path:
        """Save a twin state as versioned JSON.

        Args:
            state: The twin state to persist.

        Returns:
            Path to the saved JSON file.
        """
        # Sanitize timestamp for filename (replace colons)
        safe_ts = state.updated_at.replace(":", "-").replace(".", "-")
        filename = f"twin_{state.patno}_{safe_ts}.json"
        path = self.registry_dir / filename

        payload = asdict(state)
        # Ensure JSON-safe (convert numpy types)
        payload = _make_json_safe(payload)

        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def load_state(self, patno: int) -> TemporalTwinState | None:
        """Load the most recent twin state for a patient.

        Args:
            patno: Patient number.

        Returns:
            Most recent TemporalTwinState, or None if not found.
        """
        pattern = f"twin_{patno}_*.json"
        files = sorted(self.registry_dir.glob(pattern), reverse=True)
        if not files:
            return None

        data = json.loads(files[0].read_text(encoding="utf-8"))
        return TemporalTwinState(**data)

    def load_latest_states(self) -> dict[int, TemporalTwinState]:
        """Load the most recent state for every patient in the registry.

        Returns:
            Dict mapping patno -> TemporalTwinState.
        """
        # Collect all patient numbers from filenames
        states: dict[int, TemporalTwinState] = {}
        seen_patnos: set[int] = set()

        for path in sorted(self.registry_dir.glob("twin_*.json"), reverse=True):
            parts = path.stem.split("_")
            if len(parts) < 3:
                continue
            try:
                patno = int(parts[1])
            except ValueError:
                continue

            if patno in seen_patnos:
                continue  # Already have the most recent
            seen_patnos.add(patno)

            data = json.loads(path.read_text(encoding="utf-8"))
            states[patno] = TemporalTwinState(**data)

        return states

    def get_audit_trail(self, patno: int) -> list[dict[str, Any]]:
        """Get the full update history for a patient.

        Returns all update_log entries across all saved states,
        ordered chronologically.

        Args:
            patno: Patient number.

        Returns:
            List of audit log entries.
        """
        pattern = f"twin_{patno}_*.json"
        files = sorted(self.registry_dir.glob(pattern))  # chronological

        all_logs: list[dict[str, Any]] = []
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            logs = data.get("update_log", [])
            all_logs.extend(logs)

        return all_logs

    def list_patients(self) -> list[int]:
        """List all patient numbers with saved states."""
        patnos: set[int] = set()
        for path in self.registry_dir.glob("twin_*.json"):
            parts = path.stem.split("_")
            if len(parts) >= 3:
                try:
                    patnos.add(int(parts[1]))
                except ValueError:
                    continue
        return sorted(patnos)

    @staticmethod
    def compute_model_version_hash(checkpoint_paths: list[Path]) -> str:
        """Deterministic hash of model checkpoints for versioning.

        Uses file sizes and modification times (not full content hash)
        for speed. Sufficient for version tracking.

        Args:
            checkpoint_paths: Paths to model checkpoint files.

        Returns:
            Hex digest string.
        """
        h = hashlib.sha256()
        for path in sorted(checkpoint_paths):
            h.update(path.name.encode("utf-8"))
            h.update(str(path.stat().st_size).encode("utf-8"))
        return h.hexdigest()[:16]

    @staticmethod
    def compute_data_version_hash(
        features: np.ndarray,
        time_months: np.ndarray,
        patno: int,
    ) -> str:
        """Deterministic hash of patient data for versioning.

        Args:
            features: [T, F] feature matrix.
            time_months: [T] visit times.
            patno: Patient number.

        Returns:
            Hex digest string.
        """
        h = hashlib.sha256()
        h.update(str(patno).encode("utf-8"))
        h.update(features.tobytes())
        h.update(time_months.tobytes())
        return h.hexdigest()[:16]


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types to Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
