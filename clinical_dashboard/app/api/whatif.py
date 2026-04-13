"""What-If API: counterfactual scenario engine."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.data.schema import WhatIfRequest

router = APIRouter()


@router.post("/simulate")
async def simulate_whatif(req: WhatIfRequest):
    """Run what-if simulation: modify features and compare predictions."""
    from app.main import model_registry, patient_store
    from app.services.whatif_service import run_whatif_simulation

    result = run_whatif_simulation(
        req, model_registry=model_registry, patient_store=patient_store
    )
    if result is None:
        raise HTTPException(status_code=400, detail="Could not run what-if simulation")
    return result
