"""Patient API: list, search, and detail endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


def _get_store():
    from app.main import patient_store
    return patient_store


@router.get("")
async def list_patients(
    search: str | None = Query(None, description="Search by patient ID"),
    stage: str | None = Query(None, description="Filter by current NSD-ISS stage"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List patients with optional search/filter."""
    store = _get_store()
    results, total = store.list_patients(search=search, stage=stage, limit=limit, offset=offset)
    return {"patients": results, "total": total, "limit": limit, "offset": offset}


@router.get("/{patno}")
async def get_patient(patno: int):
    """Full patient detail for the dashboard."""
    store = _get_store()
    detail = store.get_patient_detail(patno)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Patient {patno} not found")
    return detail


@router.get("/{patno}/features")
async def get_patient_features(patno: int):
    """12 CatBoost features for a patient's latest visit."""
    store = _get_store()
    features = store.get_patient_features_for_staging(patno)
    if not features:
        raise HTTPException(status_code=404, detail=f"No features found for patient {patno}")
    return {"patno": patno, "features": features}


@router.get("/{patno}/neighbors")
async def get_patient_neighbors(patno: int, k: int = Query(10, ge=1, le=30)):
    """Find k similar patients from Graph-DT graph."""
    from app.main import model_registry, patient_store
    from app.services.similarity_service import find_similar_patients

    neighbors = find_similar_patients(
        patno, model_registry=model_registry, patient_store=patient_store, k=k
    )
    return neighbors
