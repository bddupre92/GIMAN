"""Baseline adapters for Paper 12 phys-GIMIN competitor comparisons.

Each adapter wraps an external method and exposes the phys-GIMIN baseline interface:
  (imputed_mean, imputed_sigma) = adapter.impute(features, mask)

Available adapters:
  - DeRooijImputer: wraps de Rooij et al. 2025 UDE physiology-informed regularization
    (vendored at baselines/derooij_2025/, VENDOR_NOTES.md for attribution)
"""
from phys_gimin.baseline_adapters.derooij_adapter import DeRooijImputer

__all__ = ["DeRooijImputer"]
