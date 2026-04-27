# OCP_vendored

Verbatim vendor of the Min-CPS / Min-RCPS ordinal conformal-prediction algorithm
from https://github.com/xrty/OCP (commit `676fbca8`, 2025-11-16).

**Why vendored, not pip-installed / submoduled:** the upstream repo has no
setup.py / pyproject.toml / LICENSE and uses a per-dataset script layout
rather than a Python package. See `ocp.py` docstring for the full provenance
+ licence-caveat note.

**Citation:** Zhang Z et al. (2025). *Provably Minimum-Length Conformal
Prediction Sets for Ordinal Classification.* arXiv:2511.16845.

**Callable algorithm functions** (public API for this vendor dir):

- `sliding_window_predict_set(val_scores, qhat)` — Algorithm 1 (Min-CPS), O(K)
- `get_qhat_ordinal_aps(fn, cal_scores, cal_labels, alpha)` — Algorithm 2, binary-search calibration
- `cdf_naive_ordinal_prediction`, `ordinal_aps_prediction`, `brute_force_predict_set` — ablation / baseline variants
- `evaluate_sets(fn, scores, labels, qhat, alpha)` — coverage + mean size diagnostic

Used by: `scripts/paper1/run_ordinal_conformal.py` (Paper 1 WS1.5 ordinal CP analysis).
