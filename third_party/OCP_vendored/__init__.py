"""Vendored xrty/OCP algorithm functions.

See ocp.py docstring for provenance, licence caveat, and citation.
"""

from .ocp import (
    brute_force_predict_set,
    cdf_naive_ordinal_prediction,
    evaluate_sets,
    fix_randomness,
    get_qhat_ordinal_aps,
    ordinal_aps_prediction,
    sliding_window_predict_set,
)

__all__ = [
    "brute_force_predict_set",
    "cdf_naive_ordinal_prediction",
    "evaluate_sets",
    "fix_randomness",
    "get_qhat_ordinal_aps",
    "ordinal_aps_prediction",
    "sliding_window_predict_set",
]

# Pinned upstream provenance for audit trail
UPSTREAM_REPO = "https://github.com/xrty/OCP"
UPSTREAM_SHA = "676fbca8aeb7a2ede586d821eaddc687a4fa78fc"
UPSTREAM_DATE = "2025-11-16T23:04:55Z"
VENDORED_DATE = "2026-04-23"
