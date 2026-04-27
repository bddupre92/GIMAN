"""Explainability API: SHAP values + gate activation."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


@router.get("/feature_importance/{patno}")
async def feature_importance(patno: int):
    """Get feature importance and gate activation for a patient."""
    from app.main import model_registry, patient_store
    from app.services.explainability_service import explain_prediction

    result = explain_prediction(
        patno, model_registry=model_registry, patient_store=patient_store
    )
    if result is None:
        raise HTTPException(status_code=400, detail="Could not compute explanation")
    return result
