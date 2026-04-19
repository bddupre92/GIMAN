"""PhysGIMIN — identity subclass of StageConditionedGIMIN with physics-regularized loss.

PhysGIMIN extends StageConditionedGIMIN by composing physics-informed regularization
at the LOSS level, not the architecture level. The forward pass is identical to the
parent StageConditionedGIMIN, preserving the heteroscedastic decoder σ-head contract
so that Paper 10 bidirectional integration and main-project inference paths can consume
PhysGIMIN weights without modification.

This design choice (identity subclass) exists so that:
1. Checkpoints carry class identity for provenance tracking (logged in training JSON).
2. Downstream code can isinstance-check for physics-regularized models.
3. Future architectural divergence (e.g., learnable constraint manifolds) is branch-local.

Physics regularization is composed via:
- src/phys_gimin/regularizer.py — modular regularization terms
- src/phys_gimin/loss.py — combined loss function

See the Week 1 scaffold docstrings for details on boundary-condition penalties,
state-variable clipping, and stop-grad operations.
"""

from __future__ import annotations

from giman_pipeline.imputation.stage_conditioned_gimin import StageConditionedGIMIN


class PhysGIMIN(StageConditionedGIMIN):
    """Identity subclass over StageConditionedGIMIN with physics-regularized loss composition.

    Architecture
    ============
    Identical to parent StageConditionedGIMIN:
    - ModalityEncoderBank (per-modality encoding)
    - CrossModalImputationAttention (missingness-gated fusion)
    - GNN message-passing layers (graph-informed propagation)
    - StageConditionedDecoder (heteroscedastic μ, σ output with stage conditioning)

    Physics Integration
    ===================
    Physics constraints are composed at the loss function level:
    - Boundary conditions (priors on extreme values)
    - State-variable clipping (enforce biological bounds)
    - Stage-stratified regularization (modulate penalty by NSD-ISS stage)
    - Stop-grad operations (prevent optimizer from "cheating" via gradient flow)

    All constraints are implemented in src/phys_gimin/{loss.py,regularizer.py}
    and composed during training, not at inference time.

    Checkpoint Compatibility
    ========================
    PhysGIMIN checkpoints are bit-identical to StageConditionedGIMIN:
    - Same state_dict schema
    - Same forward() output contract
    - Same model_class logging for provenance

    This ensures that:
    1. Paper 3 fold0 graph-DT checkpoint loading code works unchanged.
    2. Paper 10 bidirectional posterior updates consume PhysGIMIN weights.
    3. Main-project inference scripts (e.g., paper6/unified_pipeline_demo_v2.py)
       can run inference on PhysGIMIN weights without modification.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize PhysGIMIN as a StageConditionedGIMIN subclass.

        All arguments are passed through to the parent class.
        The variant_tag attribute is set for runtime identification.
        """
        super().__init__(*args, **kwargs)
        self.variant_tag: str = "phys_gimin"
