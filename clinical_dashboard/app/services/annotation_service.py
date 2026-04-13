"""Annotation service: per-patient clinician notes stored as JSON files."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.config import ANNOTATIONS_DIR
from app.data.schema import Annotation, AnnotationList

logger = logging.getLogger(__name__)


def _annotation_path(patno: int) -> Path:
    return ANNOTATIONS_DIR / f"patient_{patno}.json"


def load_annotations(patno: int) -> AnnotationList:
    """Load all annotations for a patient."""
    path = _annotation_path(patno)
    annotations = []
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        annotations = [Annotation(**a) for a in data.get("annotations", [])]
    return AnnotationList(patno=patno, annotations=annotations)


def save_annotation(patno: int, annotation: Annotation) -> AnnotationList:
    """Add an annotation and return the updated list."""
    # Ensure directory exists
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Set timestamp if not provided
    if not annotation.timestamp:
        annotation.timestamp = datetime.now(timezone.utc).isoformat()

    # Load existing
    existing = load_annotations(patno)
    existing.annotations.append(annotation)

    # Save
    path = _annotation_path(patno)
    with open(path, "w") as f:
        json.dump(
            {"patno": patno, "annotations": [a.model_dump() for a in existing.annotations]},
            f,
            indent=2,
        )

    logger.info(f"Saved annotation for patient {patno} (total: {len(existing.annotations)})")
    return existing
