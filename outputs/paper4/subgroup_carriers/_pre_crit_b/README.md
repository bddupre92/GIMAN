# Pre-WS-P3-CRIT-B snapshot

This directory contains the WS-P3-14 carrier-subgroup outputs *before* the
WS-P3-CRIT-B fix to the Fisher's-method-on-CV-folds combination of per-fold
interaction p-values.

**Why preserved.** Reviewer #3 (reviewer3.com) flagged that combining per-fold
interaction p-values via Fisher's method violates the procedure's strict
independence assumption (5-fold CV test sets are disjoint, but trained models
share 60% training data → positively correlated p-values → inflated Type I
error). The fix replaces the per-fold-then-Fisher-combine pattern with a
SINGLE patient-level bootstrap interaction test on the pooled out-of-fold
predictions across all 5 folds.

**Use.** Compare against the post-fix outputs in the parent directory to see
the magnitude of the p-value shift. The fold assignments, model checkpoints,
random seed (42), and per-fold delta-C-td values are unchanged; only the
combination method changed.

**Snapshot timestamp:** committed alongside the WS-P3-CRIT-B fix commit.
