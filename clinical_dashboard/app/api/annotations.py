"""Annotations API: clinician notes per patient."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.data.schema import Annotation

router = APIRouter()


@router.get("/{patno}")
async def get_annotations(patno: int):
    """Get all annotations for a patient."""
    from app.services.annotation_service import load_annotations
    return load_annotations(patno)


@router.post("/{patno}")
async def add_annotation(patno: int, annotation: Annotation):
    """Add a clinician annotation for a patient."""
    from app.services.annotation_service import save_annotation
    return save_annotation(patno, annotation)
