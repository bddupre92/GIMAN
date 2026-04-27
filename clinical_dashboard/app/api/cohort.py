"""Cohort API: population-level statistics and Markov trajectories."""

from __future__ import annotations

from fastapi import APIRouter, Query

router = APIRouter()


@router.get("/summary")
async def cohort_summary():
    """Population-level summary: stage distribution, sojourn times."""
    from app.main import patient_store
    return patient_store.get_cohort_summary()


@router.get("/markov")
async def markov_trajectories(
    start_stage: str = Query("2B", description="Starting NSD-ISS stage"),
    horizon_years: float = Query(10.0, ge=0.5, le=20.0),
):
    """Markov trajectory: stage probabilities over time from a starting stage."""
    from app.main import patient_store
    from app.services.markov_service import compute_markov_trajectory

    result = compute_markov_trajectory(
        patient_store.markov_data, start_stage=start_stage, horizon_years=horizon_years
    )
    return result
