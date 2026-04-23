"""Vendored reference implementation of Min-CPS / Min-RCPS ordinal CP.

# Provenance

- **Source repository:** https://github.com/xrty/OCP
- **Upstream path:** IMDB/ocp.py (identical copies also live in UTKFace/, Avocado/,
  Temperature/ — xrty/OCP is per-dataset script layout rather than a Python package)
- **Upstream commit SHA:** 676fbca8aeb7a2ede586d821eaddc687a4fa78fc
- **Upstream commit date:** 2025-11-16T23:04:55Z
- **Vendored on:** 2026-04-23 (Paper 1 WS1.5 session)

# Why vendored and not pip-installed / submoduled

xrty/OCP has NO README-documented API, NO setup.py / pyproject.toml, NO LICENSE
file, and a per-dataset script layout (each of UTKFace/, Avocado/, Temperature/,
IMDB/ contains its own copy of ocp.py + ocpPipeline.py + files/score.npy etc).
It cannot be imported as a package. Per project CONVENTIONS.md §7.9a
("wrap canonical reference code, don't hand-roll the algorithm"), this file
is a verbatim copy of the upstream IMDB/ocp.py at the SHA above, with ONLY
the following changes by us:

1. This provenance docstring prepended (no algorithm changes).
2. Removed the `pdb` import (dev-only, unused).
3. Removed the `if __name__ == "__main__":` script harness (dataset-specific
   .npy file paths not relevant to our invocation). All pure algorithm
   functions are preserved bit-for-bit.

The algorithm functions below are the reference implementations of:

- **Algorithm 1 (Min-CPS):** `sliding_window_predict_set` — O(K) sliding-window
  search for the shortest contiguous interval [l,u] with cumulative softmax
  mass >= qhat that contains the mode argmax(f).
- **Algorithm 2 (calibration):** `get_qhat_ordinal_aps` — binary-search tau on
  the calibration set for the smallest threshold achieving target coverage.
- **Ablation / baseline variants:** `cdf_naive_ordinal_prediction`,
  `ordinal_aps_prediction` (Lu-Angelopoulos-Pomerantz 2022 MICCAI adaptive
  APS variant), `brute_force_predict_set` (O(K^2) validation reference for
  the O(K) sliding window).

# Citation

Zhang Z et al. (2025). Provably Minimum-Length Conformal Prediction Sets for
Ordinal Classification. arXiv:2511.16845 (submitted to AAAI 2026).

# License

Upstream repo does NOT declare a LICENSE as of the pinned SHA. We vendor
under an assumed academic-fair-use basis given: (a) the arXiv paper publicly
describes these exact algorithms, (b) attribution is explicit here, (c) our
use is non-commercial research reproduction. If the upstream authors add a
restrictive license, this file should be deleted and the algorithm
re-implemented from the paper (which is mathematically trivial — both
algorithms are O(K) with ~30 lines each and the paper gives full pseudo-code).
"""

from __future__ import annotations

import random

import numpy as np


def fix_randomness(seed=0):
    np.random.seed(seed=seed)
    random.seed(seed)


def get_qhat_ordinal_aps(prediction_function, cal_scores, cal_labels, alpha, tol=1e-6):
    n = cal_scores.shape[0]
    left, right = 0.001, 0.999
    # left, right = 0.001, 1.3
    best_q = right
    target_coverage = np.ceil((n + 1) * (1 - alpha)) / n
    while right - left > tol:
        mid = (left + right) / 2
        coverage, _, _, _, _ = evaluate_sets(
            prediction_function, np.copy(cal_scores), np.copy(cal_labels), mid, alpha
        )
        if coverage >= target_coverage:
            best_q = mid
            right = mid
        else:
            left = mid
    return best_q


def cdf_naive_ordinal_prediction(val_scores, qhat):
    cumsum = val_scores.cumsum(axis=1)
    argmaxes = val_scores.argmax(axis=1)
    maxes = cumsum[np.arange(val_scores.shape[0]), argmaxes]
    prediction_set = (cumsum >= (maxes[:, None] - qhat)) & (
        cumsum <= (maxes[:, None] + qhat)
    )
    return prediction_set


def ordinal_aps_prediction(val_scores, qhat):
    # if qhat > 1:  # bug somewhere?
    # return np.ones_like(val_scores).astype(bool)
    P = val_scores == val_scores.max(axis=1)[:, None]
    idx_construction_incomplete = (val_scores * P.astype(float)).sum(
        axis=1
    ) <= qhat  # Places where top-1 isn't correct
    while idx_construction_incomplete.sum() > 0:
        P_inc = P[idx_construction_incomplete]
        scores_inc = val_scores[idx_construction_incomplete]
        set_cumsum = P_inc.cumsum(axis=1)
        lower_edge_idx = (P_inc > 0).argmax(axis=1)
        upper_edge_idx = set_cumsum.argmax(axis=1)

        # Where the lower edge is both valid and also has a higher softmax score than the upper edge
        lower_edge_wins = ((lower_edge_idx - 1) >= 0) & (
            (upper_edge_idx + 1 > scores_inc.shape[1] - 1)
            | (
                scores_inc[
                    np.arange(scores_inc.shape[0]), np.maximum(lower_edge_idx - 1, 0)
                ]
                > scores_inc[
                    np.arange(scores_inc.shape[0]),
                    np.minimum(upper_edge_idx + 1, scores_inc.shape[1] - 1),
                ]
            )
        )
        P_inc[lower_edge_wins, lower_edge_idx[lower_edge_wins] - 1] = True
        P_inc[~lower_edge_wins, upper_edge_idx[~lower_edge_wins] + 1] = True  # IndexError here when alpha is too small
        P[idx_construction_incomplete] = P_inc
        idx_construction_incomplete = (val_scores * P.astype(float)).sum(axis=1) <= qhat
    return P


def sliding_window_predict_set(val_scores, qhat):
    N, K = val_scores.shape
    P = np.zeros((N, K), dtype=bool)
    for i in range(N):
        f = val_scores[i]
        y_star = np.argmax(f)
        prefix = np.zeros(K + 1)
        for j in range(K):
            prefix[j + 1] = prefix[j] + f[j]

        best_len = float("inf")
        best_l = best_u = -1
        l = 0
        for u in range(y_star, K):
            while l <= y_star:
                prob_sum = prefix[u + 1] - prefix[l]
                dist_penalty = abs(y_star - l) + abs(y_star - u)
                score = prob_sum
                if y_star >= l and y_star <= u and score >= qhat:
                    if u - l < best_len:
                        best_len = u - l
                        best_l, best_u = l, u

                if score >= qhat:
                    l += 1
                else:
                    break
        if best_l != -1 and best_u != -1:
            P[i, best_l : best_u + 1] = True
        else:
            P[i, 0:K] = True
    return P


def brute_force_predict_set(val_scores, qhat):
    N, K = val_scores.shape
    P = np.zeros((N, K), dtype=bool)

    for i in range(N):
        f = val_scores[i]
        y_star = np.argmax(f)

        best_len = float("inf")
        best_l = best_u = -1

        for l in range(0, y_star + 1):
            for u in range(y_star, K):
                prob_sum = np.sum(f[l : u + 1])
                score = prob_sum
                if score >= qhat:
                    if u - l < best_len:
                        best_len = u - l
                        best_l, best_u = l, u

        if best_l != -1 and best_u != -1:
            P[i, best_l : best_u + 1] = True

    return P


def evaluate_sets(
    prediction_function, val_scores, val_labels, qhat, alpha, print_bool=False
):
    sets = prediction_function(val_scores, qhat)
    sizes = sets.sum(axis=1)
    sizes_distribution = np.array([(sizes == i).mean() for i in range(5)])
    covered = sets[np.arange(val_labels.shape[0]), val_labels]
    coverage = covered.mean()
    label_stratified_coverage = [
        covered[val_labels == j].mean() for j in range(np.unique(val_labels).max() + 1)
    ]
    label_distribution = [
        (val_labels == j).mean() for j in range(np.unique(val_labels).max() + 1)
    ]
    if print_bool is True:
        print(
            f"alpha: {alpha} | coverage: {coverage:.4f} | average size: {sizes.mean():.4f} | qhat: {qhat:.4f}"
        )
    return (
        coverage,
        label_stratified_coverage,
        sizes_distribution,
        sizes.mean(),
        label_distribution,
    )
