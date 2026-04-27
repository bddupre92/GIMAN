"""Wearable data service: loads synthetic wearable data (STRICTLY separated)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import SYNTHETIC_WEARABLE_DIR

logger = logging.getLogger(__name__)


def load_wearable_data(patno: int) -> dict | None:
    """Load synthetic wearable data for a patient.

    IMPORTANT: This function ONLY reads from the synthetic_data directory.
    It NEVER accesses real patient data.
    """
    path = SYNTHETIC_WEARABLE_DIR / f"patient_{patno}.json"
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    # Always attach synthetic source marker
    data["source"] = "SYNTHETIC"
    data["disclaimer"] = (
        "This is computationally generated synthetic wearable data "
        "for demonstration purposes only. It does NOT represent real "
        "patient sensor readings."
    )

    return data
