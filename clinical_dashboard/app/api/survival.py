"""Survival API: DeepHit + Graph-DT CIF prediction."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.data.schema import SurvivalRequest

router = APIRouter()


@router.post("/predict")
async def predict_survival(req: SurvivalRequest):
    """Run DeepHit and Graph-DT survival prediction for a patient."""
    from app.main import model_registry, patient_store
    from app.services.survival_service import predict_survival

    result = predict_survival(
        req.patno, model_registry=model_registry, patient_store=patient_store
    )
    if result is None:
        raise HTTPException(status_code=400, detail="Could not compute survival prediction")
    return result
