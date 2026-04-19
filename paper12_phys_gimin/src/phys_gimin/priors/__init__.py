"""Prior providers for the physics regularizer.

Two variants, constructor-injected via the PriorProvider Protocol:

- LiteraturePriorProvider: fixed ODE parameters from Fearnley-Lees, Lee 2019,
  Iljina 2016. Zero leakage from project posteriors. Safe against any
  downstream target.
- PosteriorStorePriorProvider: per-patient posteriors from the main project's
  HDF5 posterior store. Partially tautological vs downstream targets trained
  on the same posteriors (Papers 7/9/10).

Runtime type-check enforces which variant is active; label is recorded in
every output JSON to prevent silent contamination.
"""

from phys_gimin.priors.base import PriorProvider

__all__ = ["PriorProvider"]
