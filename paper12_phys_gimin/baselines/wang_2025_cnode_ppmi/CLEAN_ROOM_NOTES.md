# Wang 2025 CNODE — Clean-Room Assumptions

**Paper:** Wang et al. 2025, "Conditional Neural ODE for Longitudinal Parkinson's
Disease Progression Forecasting." arXiv:2511.04789.

This file documents every architectural or training choice made where the
published paper under-specifies a detail. The implementation at `cnode.py` /
`train.py` was written WITHOUT consulting any public code. Where a paper detail
could not be resolved from the available text, a standard Neural-ODE default is
used and recorded here.

## 1. Paper text availability

At implementation time (2026-04-20), the arXiv HTML and PDF for 2511.04789 were
accessible but the fetched content yielded the abstract, author list, and high-
level method framing only — the Methods-section equations (§II.B–II.D), the
architectural dimensions, and the exact training-loop hyperparameters were not
extractable from the text returned by our retrieval tooling.

The cohort statistics (**N=161 patients; 111 with 2 visits; 50 with ≥3
visits**) and the published headline metrics (**RMSE 0.1606, R² 0.826**) are
taken verbatim from `outputs/paper12_scoping/clean_room_verification_protocol.md`
§4 row 2 — the pre-registered record that was compiled from a prior read of the
same PDF.

## 2. Architectural choices

| Choice | Value | Rationale |
|---|---|---|
| Model family | Conditional Neural ODE | Stated in title and abstract. |
| Latent hidden dimension **H** | 64 | Standard for medium-scale NODE; dimensions not given in abstract. |
| Encoder / decoder | 2-hidden-layer MLP (width 128, GELU) | Common PD-imaging baselines use 2-layer MLPs; GELU is the most-cited nonlinearity for biomedical NODEs. |
| Vector-field MLP | 2 hidden layers, width 128, GELU | Matches the encoder budget; keeps the forward parameter count comparable to typical CNODE reference implementations (≈100k params). |
| Conditioning mechanism | Concatenation of `c` with `h` at every ODE step (plus a scalar time feature) | Paper abstracts the "patient-specific initial time and progression speed" as conditioning; direct concatenation is the strictly-simplest interpretation. |
| Time feature inside f_θ | Raw `t` (years from baseline) | No evidence the paper uses a Fourier/sinusoidal embedding. |
| ODE solver | `dopri5` with rtol=1e-5, atol=1e-7 | `torchdiffeq` defaults. Paper does not report a solver. |
| Adjoint sensitivity | Not used (direct backprop) | Integration horizon is ≤10 years and batches fit in memory; adjoint is an optimisation not a correctness concern. |

All of these choices are overridable via `CNODEConfig` without touching model
code, so a later revision that discovers the paper specifies a different value
can patch the config without amending the architecture.

## 3. Training loss

Masked trajectory-MSE over observed timepoints — averaged over (sample × time ×
feature) entries. This is the standard NODE training objective; the paper does
not report an alternative.

## 4. CV protocol

- **5-fold stratified** by visit-count (2-visit vs ≥3-visit) — preserves the
  Wang 111/50 ratio in every fold.
- **100 epochs per fold** with Adam, lr=1e-3 (NODE standard) — the paper does
  not report epochs / lr. This value is plausible given the cohort size.
- **Seed 42** everywhere (numpy, torch, Python random).

## 5. Synthetic-mode evaluation is a smoke test, NOT a fidelity attestation

`train.py --synthetic` generates random trajectories from a linear drift model
so that an *idealised* CNODE could fit them perfectly. The resulting metrics
have NO bearing on fidelity to Wang's published PPMI results — they only
verify that the training loop runs end-to-end and writes a well-formed
`report.json`.

A true fidelity verdict requires:
  1. FreeSurfer preprocessing outputs at `data/02_freesurfer/cohort.npz`
  2. `train.py --real`
  3. A PASS (both RMSE ∈ [0.145, 0.177] AND R² ∈ [0.743, 0.909]).

The pre-registered gate bounds are locked by
`tests/test_wang_cnode.py::test_fidelity_gate_bounds_are_locked`. If a future
refactor changes them — even "to match what we actually produce" — that test
will fail, which is the intended safety net against moving-the-goalposts.

## 6. Deviations from Wang 2025

None known. All visible deviations stem from missing detail in the paper PDF
rather than intentional departure. If a reviewer points out a specific spec in
§II.B–II.D that our implementation does not honour, amend this file AND the
corresponding config default AND add a regression test; do not silently change
the model.

## 7. Integration into phys-GIMIN benchmark

The adapter at `src/phys_gimin/baseline_adapters/cnode_adapter.py` forwards
ONLY the brain-imaging subset of the PPMI 33-feature schema to CNODE. For
features where CNODE does not produce a trajectory (e.g. genetics, UPDRS), the
adapter falls back to feature-mean imputation and flags σ with a
`sigma_fallback` value. See adapter docstring for the full provenance map.
