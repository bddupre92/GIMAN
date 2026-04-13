"""CatBoost staging prediction service."""

from __future__ import annotations

import logging

import numpy as np

from app.config import CATBOOST_12_FEATURES, NSD_POSITIVE_LABELS
from app.data.schema import StagingRequest, StagingPrediction

logger = logging.getLogger(__name__)


def predict_staging(
    req: StagingRequest,
    model_registry,
    patient_store,
) -> StagingPrediction | None:
    """Run CatBoost NSD-positive staging (4 classes: 1, 2B, 3, 4)."""

    # Get features: from request or from patient store
    if req.features:
        features = req.features
    elif req.patno is not None:
        features = patient_store.get_patient_features_for_staging(req.patno)
    else:
        return None

    if not features:
        return None

    # Build feature vector in correct order
    X = np.zeros((1, len(CATBOOST_12_FEATURES)))
    for i, feat_name in enumerate(CATBOOST_12_FEATURES):
        val = features.get(feat_name)
        if val is not None:
            X[0, i] = val
        else:
            # Median imputation for missing
            X[0, i] = patient_store.catboost_medians[i] if patient_store.catboost_medians is not None else 0.0

    # Predict
    if model_registry.catboost_model is None:
        logger.warning("CatBoost model not loaded — returning None")
        return None

    proba = model_registry.catboost_model.predict_proba(X)[0]
    pred_class = int(np.argmax(proba))
    pred_stage = NSD_POSITIVE_LABELS.get(pred_class, str(pred_class))

    # Build probabilities dict
    probabilities = {}
    for idx, stage_label in NSD_POSITIVE_LABELS.items():
        probabilities[stage_label] = round(float(proba[idx]), 4) if idx < len(proba) else 0.0

    # Simple conformal set: stages where probability > threshold
    # (Using a simple threshold-based approach for MVP)
    threshold = 0.10
    conformal_set = [
        NSD_POSITIVE_LABELS[idx]
        for idx in range(len(proba))
        if proba[idx] > threshold and idx in NSD_POSITIVE_LABELS
    ]
    if not conformal_set:
        conformal_set = [pred_stage]

    # Features actually used
    features_used = {feat_name: round(float(X[0, i]), 2) for i, feat_name in enumerate(CATBOOST_12_FEATURES)}

    return StagingPrediction(
        predicted_class=pred_class,
        predicted_stage=pred_stage,
        probabilities=probabilities,
        conformal_set=conformal_set,
        confidence_level=0.90,
        features_used=features_used,
    )
