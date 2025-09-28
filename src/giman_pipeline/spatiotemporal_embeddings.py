"""Spatiotemporal Embedding Provider for GIMAN

Loads patient-level spatiotemporal embeddings from the real-data output file.

- Loads from: archive/development/phase2/embeddings_output/giman_spatiotemporal_embeddings.json
- Provides: get_patient_embedding(patient_id) and get_all_embeddings()

Embeddings are 256-dimensional numpy arrays, indexed by patient ID (string or int).

Usage:
    from giman_pipeline.spatiotemporal_embeddings import get_patient_embedding, get_all_embeddings
    emb = get_patient_embedding('12345')
    all_embs = get_all_embeddings()
"""

import json
import os
from typing import Any

import numpy as np

# Path to the real-data embedding file (JSON format)
EMBEDDING_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "archive",
    "development",
    "phase2",
    "embeddings_output",
    "giman_spatiotemporal_embeddings.json",
)

_embeddings_cache: dict[str, np.ndarray] = {}


def _load_embeddings() -> dict[str, np.ndarray]:
    """Load all patient embeddings from the JSON file as a dict of numpy arrays."""
    global _embeddings_cache
    if _embeddings_cache:
        return _embeddings_cache
    with open(EMBEDDING_PATH) as f:
        data = json.load(f)
    # The generator output is {"embeddings": {"session_key": [float, ...], ...}, "metadata": {...}}
    emb_dict = data["embeddings"]
    _embeddings_cache = {
        str(pid): np.array(vec, dtype=np.float32) for pid, vec in emb_dict.items()
    }
    return _embeddings_cache


def get_patient_embedding(patient_id: str | int) -> np.ndarray:
    """Retrieve the 256-dim embedding for a given patient ID.

    Args:
        patient_id: Patient identifier (string or int)

    Returns:
        Numpy array of shape (256,)

    Raises:
        KeyError: If patient_id not found in embeddings
    """
    embs = _load_embeddings()
    pid = str(patient_id)
    if pid not in embs:
        raise KeyError(f"Patient ID {pid} not found in embeddings.")
    return embs[pid]


def get_all_embeddings() -> dict[str, np.ndarray]:
    """Retrieve all patient embeddings as a dict {patient_id: embedding}.

    Returns:
        Dict mapping patient_id (str) to numpy array (256,)
    """
    return _load_embeddings()


# Additional functions for integration test compatibility
def get_all_spatiotemporal_embeddings() -> dict[str, np.ndarray]:
    """Alias for get_all_embeddings() - integration test compatibility."""
    return get_all_embeddings()


def get_spatiotemporal_embedding(
    patient_id: str | int, session: str = None
) -> np.ndarray:
    """Get spatiotemporal embedding for a patient session.

    Args:
        patient_id: Patient ID
        session: Session type ('baseline', 'followup_1', etc.)

    Returns:
        Numpy array of shape (256,)
    """
    if session:
        session_key = f"{patient_id}_{session}"
    else:
        session_key = str(patient_id)
    return get_patient_embedding(session_key)


def get_embedding_info() -> dict[str, Any]:
    """Get information about available embeddings."""
    embs = _load_embeddings()

    # Parse patient IDs and sessions
    available_patients = set()
    for session_key in embs.keys():
        if "_" in session_key:
            patient_id = session_key.split("_")[0]
            available_patients.add(patient_id)
        else:
            available_patients.add(session_key)

    return {
        "num_sessions": len(embs),
        "available_patients": sorted(list(available_patients)),
        "metadata": {"embedding_dim": 256, "embedding_type": "spatiotemporal_cnn_gru"},
    }


class SpatiotemporalProvider:
    """Provider class for integration test compatibility."""

    @staticmethod
    def get_available_patients():
        """Get list of available patient IDs."""
        info = get_embedding_info()
        return info["available_patients"]

    @staticmethod
    def get_patient_embeddings(patient_id: str | int) -> dict[str, np.ndarray]:
        """Get all embeddings for a specific patient."""
        embs = _load_embeddings()
        patient_embs = {}

        for session_key, embedding in embs.items():
            if session_key.startswith(str(patient_id) + "_") or session_key == str(
                patient_id
            ):
                patient_embs[session_key] = embedding

        return patient_embs


# Global provider instance for integration test
spatiotemporal_provider = SpatiotemporalProvider()
