"""Staging API: CatBoost NSD-ISS stage prediction."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.data.schema import StagingRequest

router = APIRouter()


@router.post("/predict")
async def predict_stage(req: StagingRequest):
    """Run CatBoost staging prediction for a patient or custom features."""
    from app.main import model_registry, patient_store
    from app.services.staging_service import predict_staging

    result = predict_staging(
        req, model_registry=model_registry, patient_store=patient_store
    )
    if result is None:
        raise HTTPException(status_code=400, detail="Could not compute staging prediction")
    return result
