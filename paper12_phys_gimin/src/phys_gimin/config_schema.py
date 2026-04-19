"""Pydantic schemas for Paper 12 Hydra configs.

Validates config correctness before a 300-run batch is launched.
Catches typos (e.g., `seed: "1001"` as string instead of int) early.

Robustness contract (Layer 1): experiment must include seed (int, required),
deterministic (bool, default True), output_schema_version (str).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ExperimentConfig(BaseModel):
    name: str
    seed: int  # required per robustness contract Layer 1
    deterministic: bool = True
    output_root: Path
    output_schema_version: str = "1.0"

    @field_validator("seed")
    @classmethod
    def _seed_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"seed must be >= 0, got {v}")
        return v


class DataConfig(BaseModel):
    feature_schema: int = 33
    modality_dims: list[int]
    ppmi_cohort_filter: Literal["all", "pd_only"] = "all"
    mask_fractions: list[float] = Field(default_factory=lambda: [0.10, 0.25, 0.50, 0.75])

    @field_validator("modality_dims")
    @classmethod
    def _modality_dims_sum_matches_schema(cls, v: list[int], info) -> list[int]:
        data = info.data
        schema = data.get("feature_schema", 33)
        if sum(v) != schema:
            raise ValueError(f"sum(modality_dims)={sum(v)} != feature_schema={schema}")
        return v

    @field_validator("mask_fractions")
    @classmethod
    def _mask_fracs_in_unit_interval(cls, v: list[float]) -> list[float]:
        for f in v:
            if not (0.0 < f < 1.0):
                raise ValueError(f"mask_fraction {f} must be in (0, 1)")
        return v


class ModelConfig(BaseModel):
    embed_dim: int = 64
    num_gnn_layers: int = 3
    num_heads: int = 4
    mc_dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    num_stages: int = 6
    stage_embed_dim: int = 16
    use_stage_attention_bias: bool = True


class TrainingConfig(BaseModel):
    n_epochs: int = Field(default=300, gt=0)
    patience: int = Field(default=20, gt=0)
    batch_size: int = Field(default=256, gt=0)
    lr: float = Field(default=1e-3, gt=0.0)
    lambda_phys_target: float = Field(default=1.0, ge=0.0)
    warmup_epochs: int = Field(default=5, ge=0)
    min_recon_fraction: float = Field(default=0.30, ge=0.0, le=1.0)
    floor_violation_decay: float = Field(default=0.5, gt=0.0, le=1.0)
    ema_alpha: float = Field(default=0.9, ge=0.0, le=1.0)


class RegularizerConfig(BaseModel):
    variant: Literal["literature", "self"]
    beta: float = Field(default=0.5, gt=0.0)
    eps: float = Field(default=1e-3, gt=0.0)
    posterior_hdf5_path: Path | None = None

    @field_validator("posterior_hdf5_path")
    @classmethod
    def _hdf5_required_for_self(cls, v: Path | None, info) -> Path | None:
        data = info.data
        if data.get("variant") == "self" and v is None:
            raise ValueError("posterior_hdf5_path is required for variant='self'")
        return v


class PhysGIMINConfig(BaseModel):
    """Top-level config for Paper 12 phys-GIMIN runs."""
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    regularizer: RegularizerConfig
