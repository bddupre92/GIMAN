"""Observation adapters — wrappers that feed phys-GIMIN σ into the mechanistic twin.

Week 5+ scope. Each adapter imports from the main project but does not
modify it. See scoping plan "σ-contract via wrappers" section for the
patch-via-wrapper discipline.

Planned modules:
- sbr.py — Per-visit SBR likelihood wrapper on mechanistic_twin_v2.observations.
- multichannel.py — Ch 9.6 σ-vector SAEM wrapper.
- path_b.py — Errors-in-variables regression on Paper 9 Path B GAP.
"""
