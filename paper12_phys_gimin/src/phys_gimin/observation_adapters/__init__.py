"""Observation adapters for cross-paper integration L1 (GIMIN → mechanistic twin).

Task 2 (2026-04-19): PerVisitSbrLikelihood wraps main-project loglik_sbr to accept
per-visit sigma vectors from GIMIN imputation.

Task 3 (pending): multichannel adapters (GFAP, NfL, Amprion SAA) + Path B (ON-OFF gap).
"""
from phys_gimin.observation_adapters.sbr import PerVisitSbrLikelihood

__all__ = ["PerVisitSbrLikelihood"]
