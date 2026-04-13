"""What-If scenario engine: modify features, re-run all models, compare."""

from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np

from app.config import CATBOOST_12_FEATURES, LONGITUDINAL_TO_PAPER1
from app.data.schema import (
    StagingRequest,
    WhatIfRequest,
)
from app.services.staging_service import predict_staging
from app.services.survival_service import predict_survival

logger = logging.getLogger(__name__)


def run_whatif_simulation(
    req: WhatIfRequest,
    model_registry,
    patient_store,
) -> dict | None:
    """Clone patient features, apply modifications, re-predict, compute delta."""

    patno = req.patno
    modifications = {m.feature: m.new_value for m in req.modifications}

    if not modifications:
        return None

    # ── Baseline predictions ────────────────────────────────
    baseline_staging = predict_staging(
        StagingRequest(patno=patno),
        model_registry=model_registry,
        patient_store=patient_store,
    )

    baseline_survival = predict_survival(
        patno, model_registry=model_registry, patient_store=patient_store
    )

    if baseline_staging is None and baseline_survival is None:
        return None

    # ── Build modified features ─────────────────────────────
    modified_features = dict(baseline_staging.features_used) if baseline_staging else {}

    # Apply modifications (handle both uppercase Paper 1 names and lowercase longitudinal names)
    reverse_map = {v: k for k, v in LONGITUDINAL_TO_PAPER1.items()}
    for feat_name, new_val in modifications.items():
        if feat_name in CATBOOST_12_FEATURES:
            modified_features[feat_name] = new_val
        elif feat_name.upper() in CATBOOST_12_FEATURES:
            modified_features[feat_name.upper()] = new_val
        elif feat_name in LONGITUDINAL_TO_PAPER1:
            modified_features[LONGITUDINAL_TO_PAPER1[feat_name]] = new_val

    # ── Counterfactual staging ──────────────────────────────
    cf_staging = predict_staging(
        StagingRequest(features=modified_features),
        model_registry=model_registry,
        patient_store=patient_store,
    )

    # ── Counterfactual survival ─────────────────────────────
    # For survival, we need to modify the longitudinal visit sequence
    # For the MVP, we re-run survival with original data
    # (full counterfactual survival would require modifying visit history)
    cf_survival = baseline_survival  # TODO: implement full counterfactual survival

    # ── Build explanation ───────────────────────────────────
    staging_changed = False
    explanation_parts = []

    if baseline_staging and cf_staging:
        if baseline_staging.predicted_stage != cf_staging.predicted_stage:
            staging_changed = True
            explanation_parts.append(
                f"Stage prediction changed: {baseline_staging.predicted_stage} -> "
                f"{cf_staging.predicted_stage}"
            )
        else:
            explanation_parts.append(
                f"Stage prediction unchanged: {cf_staging.predicted_stage}"
            )

        # Probability deltas
        for stage, base_p in baseline_staging.probabilities.items():
            cf_p = cf_staging.probabilities.get(stage, 0)
            delta = cf_p - base_p
            if abs(delta) > 0.01:
                direction = "+" if delta > 0 else ""
                explanation_parts.append(
                    f"  Stage {stage}: {direction}{delta:.1%} "
                    f"({base_p:.1%} -> {cf_p:.1%})"
                )

    explanation = "\n".join(explanation_parts) if explanation_parts else "No significant changes."

    return {
        "baseline_staging": baseline_staging.model_dump() if baseline_staging else None,
        "counterfactual_staging": cf_staging.model_dump() if cf_staging else None,
        "baseline_survival": baseline_survival,
        "counterfactual_survival": cf_survival,
        "staging_changed": staging_changed,
        "modifications_applied": modifications,
        "explanation": explanation,
    }
