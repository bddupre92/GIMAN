"""Wearable API: synthetic wearable data (strictly separated)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/{patno}")
async def get_wearable_data(patno: int):
    """Get synthetic wearable data for a patient.

    IMPORTANT: This endpoint ONLY returns synthetic data.
    Every response includes source='SYNTHETIC' and a disclaimer.
    """
    from app.services.wearable_service import load_wearable_data

    data = load_wearable_data(patno)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No wearable data available for patient {patno}",
        )
    return data
