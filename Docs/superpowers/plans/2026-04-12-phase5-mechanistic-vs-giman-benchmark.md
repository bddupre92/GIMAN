# Phase 5: Bidirectional-Ready Mechanistic Model + External Validation (Paper 10) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a bidirectional-ready mechanistic patient-specific model for Parkinson's disease that (1) updates Bayesian posteriors when new observations arrive, (2) externally validates the SBR decay finding on LCC cohort, (3) benchmarks against GIMAN Graph-DT on a common clinical endpoint, and (4) honestly audits NASEM digital twin criteria.

**Architecture:** Persistent posterior sample storage (HDF5), `update_posterior()` API via SIR/IS reweighting with MCMC rejuvenation, patient state versioning, prediction logging + validation loop. Forward simulation uses Phase 4 interaction model coefficients. External validation on LCC DaT-SPECT data. Head-to-head on time-to-wearing-off endpoint.

**Tech Stack:** Python 3.12, PyTorch 2.8 (checkpoint loading), PyMC/scipy (SIR reweighting), h5py (posterior storage), lifelines (survival), pandas, numpy, statsmodels, matplotlib/seaborn, existing `_reproducibility.py` helper.

---

## Revision History

**v1 (original):** Archived in git history at commit `7dfb7e6` ([GitHub link](https://github.com/bddupre92/PD_PHD/blob/7dfb7e6/Docs/superpowers/plans/2026-04-12-phase5-mechanistic-vs-giman-benchmark.md)). Contains full code blocks for data assembly, GIMAN prediction loading, shared cohort identification — referenced throughout this v2 for boilerplate reuse. Run `git show 7dfb7e6:Docs/superpowers/plans/2026-04-12-phase5-mechanistic-vs-giman-benchmark.md` to view.

**v2 (2026-04-13):** Comprehensive revision after deep review (3 parallel research agents + NASEM 2024 report + CPT:PSP credibility framework).

**Key changes from v1:**

- Dropped "benchmark + counterfactual" framing (critical thinking agent flagged as overclaim — incommensurable metrics, regression extrapolation not mechanism)
- Added **bidirectional architecture** as primary contribution (NASEM-aligned)
- Replaced synthetic 50-patient counterfactual with **observational LEDD-escalation validation**
- Added **external validation on LCC** (addresses "only works on PPMI" critique)
- Added **NASEM criteria audit** (owns partial implementation honestly)
- Fixed data lineage: canonical parquet with ON+OFF rows (Path B inherits from main assembly)
- Target venue shift: **npj Parkinson's Disease** or **Journal of Parkinson's Disease** (from CPT:PSP — better fit for methodological emphasis)

**Rationale:** Papers 10+11 cannot deliver a full NASEM-compliant digital twin with current observational data (no continuous sensors, no intervention ground truth, no prospective re-imaging). But they CAN deliver a bidirectional-ready architecture + external validation + honest NASEM audit — a defense-grade PhD contribution.

## Literature Validation (Closed-Loop Stage 1 — COMPLETE 2026-04-13)

| Search | Finding |
|---|---|
| NASEM 2024 Report | VVUQ is the critical gap; bidirectional flow is defining criterion |
| Musuamba 2021 (CPT:PSP) | Risk-informed model credibility framework (ASME V&V 40) |
| Friedrich 2016 (CPT:PSP) | QSP model qualification method (MQM) |
| Viceconti 2020 | In silico trials VVUQ regulatory framework |
| npj Digital Medicine 2025 | VVUQ for precision medicine digital twins |
| arxiv 2405.05301 | NASEM-compliant critical illness DT design reference |
| Hicks 2015 (648 cites) | V&V best practices — field standard |
| GitHub research | Hybrid-ODE-NeurIPS-2021, AlaaLab/med-real2sim, auton-survival — reusable patterns |

## Data Lineage Fix (Task 0 — CRITICAL)

**Issue discovered:** The main `phase4_assembled_data.parquet` has only 40 ON-state rows because Task 1 of Phase 4 filtered to OFF during assembly. Path B re-extracts from raw Part III CSV to get the 4,203 paired ON-OFF visits. This creates **two data pipelines** — violates canonical-source principle.

**Fix:** Rebuild the assembled parquet to include BOTH ON and OFF rows with added columns `updrs3_on`, `updrs3_off`, `gap` (where paired). Path A filters in-memory. Path B uses paired rows directly. Paper 10 uses ONE canonical source.

## NASEM Criteria Self-Audit (Honest Scope)

| Criterion | Current State | After Paper 10 | Gap |
|---|---|---|---|
| Physiological constraints | Partial (N(t) ODE + Hill PD) | Same | Simplified vs full 5-module coupled ODE |
| Bidirectional data flow | NO | **YES (episodic)** | Continuous updating requires sensors |
| Continuous updating | NO | Episodic (per-visit) | True continuous = Phase 6 MindMend |
| Patient-level validation | Partial | **External (LCC) + replay harness** | No prospective interventional validation |

**Honest framing:** Paper 10 delivers a **bidirectional-ready mechanistic patient-specific model**, NOT a full NASEM-compliant digital twin. Phase 6 is acknowledged future work.

---

## File Structure

```
src/giman_pipeline/mechanistic_twin_v2/   # NEW package for bidirectional arch
├── __init__.py
├── state.py                # PatientState dataclass (version, samples, weights, ess)
├── posterior_store.py      # HDF5-backed: /patno/version → datasets
├── updater.py              # update_posterior() via SIR + MCMC rejuvenation
├── simulator.py            # Forward simulation with posterior uncertainty
├── counterfactual.py       # Extends existing src/giman_pipeline/digital_twin/
├── validation.py           # PredictionLog + calibration_report
├── forward_model.py        # Ports Phase 2 ODE from Julia module
└── observations.py         # Per-observation-type likelihoods

scripts/mechanistic_twin/
├── phase5_rebuild_canonical_parquet.py     # Task 0: fix data lineage
├── phase5_persist_full_posteriors.py       # Task 1: rerun IS with full samples
├── phase5_identify_shared_cohort.py        # Task 2: GIMAN × mechanistic × Phase 4
├── phase5_external_validation_lcc.py       # Task 3: LCC SBR decay validation
├── phase5_headtohead_wearing_off.py        # Task 4: common endpoint benchmark
├── phase5_bidirectional_demo.py            # Task 5: fit scans 1-2, predict scan 3
├── phase5_observational_counterfactual.py  # Task 6: LEDD escalation validation
├── phase5_nasem_audit.py                   # Task 7: NASEM criteria scoring
├── phase5_generate_figures.py              # Task 8: publication figures

tests/mechanistic_twin_v2/
├── test_state.py
├── test_posterior_store.py
├── test_updater.py
├── test_simulator.py

outputs/mechanistic_twin/paper10_mech_vs_giman/
├── canonical_assembled_v2.parquet          # Task 0 output (ON+OFF)
├── phase2_posteriors_full_samples.h5       # Task 1 output
├── shared_cohort.json                      # Task 2 output
├── external_validation_lcc.json            # Task 3 output
├── headtohead_wearing_off.json             # Task 4 output
├── bidirectional_demo.json                 # Task 5 output
├── observational_counterfactual.json       # Task 6 output
├── nasem_audit.json                        # Task 7 output
├── figures/                                # Task 8 output
├── latex/main.tex                          # Paper 10 manuscript
```

---

### Task 0: Rebuild Canonical Assembled Parquet (Data Lineage Fix)

**Files:**

- Create: `scripts/mechanistic_twin/phase5_rebuild_canonical_parquet.py`
- Read: `data/00_raw/MDS-UPDRS Part IV/MDS-UPDRS_Part_III_12Apr2026.csv`
- Read: `data/00_raw/MDS-UPDRS Part IV/MDS-UPDRS_Part_IV__Motor_Complications_12Apr2026.csv`
- Read: `data/00_raw/LEDD_Concomitant_Medication_Log_12Apr2026.csv`
- Read: `outputs/mechanistic_twin/data/posteriors/phase2_coupled_is_step26v4.csv`
- Output: `outputs/mechanistic_twin/paper10_mech_vs_giman/canonical_assembled_v2.parquet`

**Context:** Fix the Phase 4 data lineage. Current parquet has only 40 ON rows; Path B rebuilds pairs ad-hoc. New canonical parquet must have ALL UPDRS-III rows (ON + OFF + unstated) with columns for both states paired by PATNO+EVENT_ID.

- [ ] **Step 1: Write failing test for canonical parquet schema**

```python
# tests/mechanistic_twin_v2/test_canonical_parquet.py
import pandas as pd
from pathlib import Path


def test_canonical_has_both_on_and_off():
    """Canonical parquet must have >3000 ON rows AND >7000 OFF rows."""
    path = Path("outputs/mechanistic_twin/paper10_mech_vs_giman/canonical_assembled_v2.parquet")
    df = pd.read_parquet(path)
    on_count = df["updrs3_on"].notna().sum()
    off_count = df["updrs3_off"].notna().sum()
    assert on_count > 3000, f"Expected >3000 ON rows, got {on_count}"
    assert off_count > 7000, f"Expected >7000 OFF rows, got {off_count}"


def test_canonical_has_gap_column():
    """Canonical parquet must have gap column where paired."""
    df = pd.read_parquet("outputs/mechanistic_twin/paper10_mech_vs_giman/canonical_assembled_v2.parquet")
    assert "gap" in df.columns
    paired = df.dropna(subset=["updrs3_on", "updrs3_off"])
    assert (abs(paired["gap"] - (paired["updrs3_off"] - paired["updrs3_on"])) < 0.01).all()


def test_path_b_pair_count_matches():
    """Canonical parquet paired count should be ~4,203 (Phase 4 Path B)."""
    df = pd.read_parquet("outputs/mechanistic_twin/paper10_mech_vs_giman/canonical_assembled_v2.parquet")
    paired_count = df.dropna(subset=["updrs3_on", "updrs3_off"]).shape[0]
    assert paired_count >= 4000, f"Expected ~4,203 paired visits, got {paired_count}"
```

- [ ] **Step 2: Implement rebuild script** — see v1 plan Task 1 for LEDD computation logic to reuse. Must load UPDRS-III WITHOUT filtering PDSTATE, pair ON-OFF, merge Part IV + LEDD + posteriors, compute n_frac using compound decay `(1 - pct_loss/100)^years`.

- [ ] **Step 3: Run + verify**

```bash
.venv/bin/python scripts/mechanistic_twin/phase5_rebuild_canonical_parquet.py
.venv/bin/python -m pytest tests/mechanistic_twin_v2/test_canonical_parquet.py -v
```

- [ ] **Step 4: Cross-check Path B reproducibility** — Re-run Phase 4 Path B using canonical parquet. Verify interaction p-value is still ≈0.044 after severity control.

- [ ] **Step 5: Commit**

```bash
git add scripts/mechanistic_twin/phase5_rebuild_canonical_parquet.py tests/mechanistic_twin_v2/test_canonical_parquet.py
git commit -m "feat: Phase 5 Task 0 — rebuild canonical parquet with ON+OFF (fix data lineage)"
```

---

### Task 1: Persist Full Posterior Samples (Bidirectional Infrastructure)

**Files:**

- Create: `scripts/mechanistic_twin/phase5_persist_full_posteriors.py`
- Create: `src/giman_pipeline/mechanistic_twin_v2/posterior_store.py`
- Create: `tests/mechanistic_twin_v2/test_posterior_store.py`
- Output: `outputs/mechanistic_twin/paper10_mech_vs_giman/phase2_posteriors_full_samples.h5`

**Context:** Currently `phase2_coupled_is_step26v4.csv` has only summary stats. Bidirectional updating requires full 10,000 posterior samples per patient + weights for SIR reweighting when new observations arrive. THIS IS THE CRITICAL INFRASTRUCTURE STEP.

**Data structure (HDF5):**

```text
phase2_posteriors_full_samples.h5
├── /patient_3001/v1/
│   ├── samples (10000, 5)      # (k_n, k_alpha, alpha_tox, T_tox, N0)
│   ├── weights (10000,)        # IS weights
│   ├── @ess, @log_marg_lik
```

- [ ] **Step 1: Implement `PosteriorStore` class**

```python
# src/giman_pipeline/mechanistic_twin_v2/posterior_store.py
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import h5py
import numpy as np


@dataclass
class PatientPosterior:
    patno: int
    version: int
    samples: np.ndarray      # (N, D) D=5 params
    weights: np.ndarray      # (N,) normalized
    ess: float
    log_marg_lik: float
    param_names: list[str]


class PosteriorStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def save(self, posterior: PatientPosterior) -> None:
        with h5py.File(self.path, "a") as f:
            key = f"patient_{posterior.patno}/v{posterior.version}"
            if key in f:
                del f[key]
            grp = f.create_group(key)
            grp.create_dataset("samples", data=posterior.samples, compression="gzip")
            grp.create_dataset("weights", data=posterior.weights)
            grp.attrs["ess"] = posterior.ess
            grp.attrs["log_marg_lik"] = posterior.log_marg_lik
            grp.attrs["param_names"] = [n.encode("utf-8") for n in posterior.param_names]

    def load(self, patno: int, version: int | Literal["latest"] = "latest") -> PatientPosterior:
        with h5py.File(self.path, "r") as f:
            pat_grp = f[f"patient_{patno}"]
            if version == "latest":
                versions = sorted([int(k[1:]) for k in pat_grp.keys() if k.startswith("v")])
                version = versions[-1]
            grp = pat_grp[f"v{version}"]
            return PatientPosterior(
                patno=patno, version=version,
                samples=grp["samples"][:], weights=grp["weights"][:],
                ess=float(grp.attrs["ess"]),
                log_marg_lik=float(grp.attrs["log_marg_lik"]),
                param_names=[n.decode("utf-8") for n in grp.attrs["param_names"]],
            )

    def list_patients(self) -> list[int]:
        with h5py.File(self.path, "r") as f:
            return sorted([int(k.split("_")[1]) for k in f.keys() if k.startswith("patient_")])
```

- [ ] **Step 2: Write tests**

```python
# tests/mechanistic_twin_v2/test_posterior_store.py
import numpy as np
from giman_pipeline.mechanistic_twin_v2.posterior_store import PosteriorStore, PatientPosterior


def test_save_load_roundtrip(tmp_path):
    store = PosteriorStore(tmp_path / "test.h5")
    post = PatientPosterior(
        patno=3001, version=1,
        samples=np.random.randn(1000, 5),
        weights=np.ones(1000) / 1000,
        ess=900.0, log_marg_lik=-42.0,
        param_names=["k_n", "k_alpha", "alpha_tox", "T_tox", "N0"],
    )
    store.save(post)
    loaded = store.load(3001, version=1)
    assert loaded.patno == 3001
    assert loaded.samples.shape == (1000, 5)
    assert loaded.ess == 900.0


def test_latest_version(tmp_path):
    store = PosteriorStore(tmp_path / "test.h5")
    for v in [1, 2, 3]:
        post = PatientPosterior(
            patno=3001, version=v, samples=np.random.randn(100, 5),
            weights=np.ones(100) / 100, ess=float(v * 100), log_marg_lik=-v * 10.0,
            param_names=["k_n", "k_alpha", "alpha_tox", "T_tox", "N0"],
        )
        store.save(post)
    latest = store.load(3001, version="latest")
    assert latest.version == 3
```

- [ ] **Step 3: Re-run Phase 2 IS with full sample persistence**

Modify `scripts/mechanistic_twin/step_2_6_v4_is_weighted_posterior.py` to call `PosteriorStore.save()` alongside existing summary CSV output. Expect ~1-2 days compute on 1,065 patients.

- [ ] **Step 4: Verify samples reproduce published medians**

For each patient, compute posterior median from stored samples. Compare to `phase2_coupled_is_step26v4.csv` medians. Must match within floating-point tolerance.

- [ ] **Step 5: Commit**

```bash
git add src/giman_pipeline/mechanistic_twin_v2/posterior_store.py tests/mechanistic_twin_v2/test_posterior_store.py scripts/mechanistic_twin/phase5_persist_full_posteriors.py
git commit -m "feat: Phase 5 Task 1 — PosteriorStore HDF5 infra for bidirectional updating"
```

---

### Task 2: Identify Shared Cohort (3-way Overlap)

**Files:**

- Create: `scripts/mechanistic_twin/phase5_identify_shared_cohort.py`
- Read: canonical parquet, Phase 2 posteriors, Graph-DT/DeepHit checkpoints
- Output: `outputs/mechanistic_twin/paper10_mech_vs_giman/shared_cohort.json`

**Context:** Identify patients in (1) GIMAN checkpoint train OR test, (2) Phase 2 posteriors, (3) canonical parquet with 2+ ON-OFF pairs. This is the ~280-patient head-to-head analysis set.

- [ ] **Step 1: Implement script** (see prior version for boilerplate; key change: filter by 2+ ON-OFF pairs in canonical parquet)

- [ ] **Step 2: Run + verify count >=200**

- [ ] **Step 3: Commit**

```bash
git add scripts/mechanistic_twin/phase5_identify_shared_cohort.py
git commit -m "feat: Phase 5 Task 2 — identify shared cohort (GIMAN × mechanistic × paired)"
```

---

### Task 3: External Validation on LCC Cohort

**Files:**

- Create: `scripts/mechanistic_twin/phase5_external_validation_lcc.py`
- Read: `data/00_raw/LCC/DaTSCAN_SBR.csv`
- Output: `outputs/mechanistic_twin/paper10_mech_vs_giman/external_validation_lcc.json`

**Context:** LCC (Lewy Cohort) has N=638 patients with DaT-SPECT SBR. Re-fit Phase 1 SBR decay on LCC, compare rate distribution to PPMI's 3.29%/yr median. THIS IS THE EXTERNAL VALIDATION — addresses "only works on PPMI" critique.

**Analysis:**

1. Fit exponential SBR decay per-patient on LCC (reuse Phase 1 methodology at `scripts/mechanistic_twin/step_1_5_sbr_decay_calibration.py`)
2. Extract pct_loss_per_yr distribution (N=638 or however many have 2+ scans)
3. KS test: LCC vs PPMI distributions
4. Mann-Whitney U test on medians
5. Per-stage breakdown (if NSD-ISS staging available for LCC)

- [ ] **Step 1: Check LCC DaT-SPECT data structure**

```bash
head -5 data/00_raw/LCC/DaTSCAN_SBR.csv
wc -l data/00_raw/LCC/DaTSCAN_SBR.csv
```

- [ ] **Step 2: Identify patients with 2+ scans**

- [ ] **Step 3: Fit per-patient exponential decay** (reuse Phase 1 code, adapt to LCC column names)

- [ ] **Step 4: Compare distributions**

```python
from scipy.stats import ks_2samp, mannwhitneyu
ks_stat, ks_p = ks_2samp(ppmi_pct_loss, lcc_pct_loss)
mw_stat, mw_p = mannwhitneyu(ppmi_pct_loss, lcc_pct_loss, alternative="two-sided")
```

- [ ] **Step 5: Interpret**

- If KS p > 0.05 AND median difference < 1%/yr → STRONG external validation
- If KS p < 0.05 but medians similar → cohort composition differences, report transparently
- If medians very different → genuine difference, discuss as limitation

- [ ] **Step 6: Commit**

```bash
git add scripts/mechanistic_twin/phase5_external_validation_lcc.py
git commit -m "feat: Phase 5 Task 3 — external validation of SBR decay on LCC cohort"
```

---

### Task 4: Head-to-Head on Common Endpoint (Time-to-Wearing-Off)

**Files:**

- Create: `scripts/mechanistic_twin/phase5_headtohead_wearing_off.py`
- Read: canonical parquet, Graph-DT checkpoints, Phase 2 posteriors
- Output: `outputs/mechanistic_twin/paper10_mech_vs_giman/headtohead_wearing_off.json`

**Context:** Previous v1 plan compared incommensurable metrics (C-td vs R²). **Fixed:** both models predict **time-to-NP4OFF≥1** (wearing-off onset). C-index for BOTH. Paired bootstrap.

**Setup:**

- Outcome: time from treatment initiation to first NP4OFF ≥ 1
- Graph-DT: Cox-adjusted hazard from transition predictions (adapt since Graph-DT predicts transitions, not wearing-off directly)
- Mechanistic: Phase 4 Path B gap trajectory crossing threshold (e.g., 5 UPDRS points) → predicts wearing-off
- Common cohort: 280 shared patients

- [ ] **Step 1: Extract wearing-off events from canonical parquet**

```python
paired = canonical.dropna(subset=["updrs3_on", "updrs3_off"])
wearing_off_events = canonical[canonical["NP4OFF"] >= 1].groupby("PATNO").first()
# Compute time from treatment_start to first NP4OFF >= 1
```

- [ ] **Step 2: Load Graph-DT predictions for shared cohort**

Use `load_graph_dt_checkpoint()` at `src/giman_pipeline/paper3/graph_digital_twin.py:961`. For each patient, extract CIF predictions for all causes, derive hazard ratio for wearing-off.

- [ ] **Step 3: Compute mechanistic predictions**

For each patient, compute gap trajectory using Phase 4 β coefficients. Find predicted year when gap < threshold. This is the mechanistic wearing-off prediction.

- [ ] **Step 4: Compute C-index for both models**

```python
from lifelines.utils import concordance_index
ci_graph_dt = concordance_index(times, graph_dt_risk_scores, events)
ci_mechanistic = concordance_index(times, mechanistic_risk_scores, events)
```

- [ ] **Step 5: Paired bootstrap (1000 resamples)**

For each of 1000 bootstrap samples, compute both C-indices. Report mean, 95% CI, and p-value for difference.

- [ ] **Step 6: Commit**

```bash
git add scripts/mechanistic_twin/phase5_headtohead_wearing_off.py
git commit -m "feat: Phase 5 Task 4 — head-to-head on time-to-wearing-off (common endpoint)"
```

---

### Task 5: Bidirectional Update Demo (The Twin Proof)

**Files:**

- Create: `scripts/mechanistic_twin/phase5_bidirectional_demo.py`
- Create: `src/giman_pipeline/mechanistic_twin_v2/updater.py`
- Create: `src/giman_pipeline/mechanistic_twin_v2/state.py`
- Create: `src/giman_pipeline/mechanistic_twin_v2/forward_model.py`
- Create: `src/giman_pipeline/mechanistic_twin_v2/observations.py`
- Create: `src/giman_pipeline/mechanistic_twin_v2/simulator.py`
- Create: `src/giman_pipeline/mechanistic_twin_v2/validation.py`
- Create: `tests/mechanistic_twin_v2/test_updater.py`
- Read: `phase2_posteriors_full_samples.h5` (from Task 1)
- Output: `outputs/mechanistic_twin/paper10_mech_vs_giman/bidirectional_demo.json`

**Context:** THIS IS THE KEY TASK. Demonstrates the bidirectional capability that distinguishes our work from pure regression.

**Design (Option B "Simulated Prospective"):**

1. For each of ~400 patients with 3+ DaT-SPECT scans:
2. Initialize posterior using baseline scan only (t=0)
3. Freeze predictions for t+1yr, t+3yr (including N(t), UPDRS, NP4OFF)
4. When second scan arrives: `update_posterior()` via SIR + rejuvenation
5. Compare frozen predictions to observed outcomes
6. Repeat for third scan
7. Report: does MAE decrease with update count? Coverage maintained?

- [ ] **Step 1: Implement `PatientState` dataclass** (state.py — see v2 architecture doc for signature)

- [ ] **Step 2: Implement `forward_model.py`**

Port Phase 2 ODE solver from Julia. For each posterior sample (theta), predict N(t), α-syn trajectories. Must exactly reproduce Julia results (validated via round-trip test).

- [ ] **Step 3: Implement `observations.py` — likelihood functions**

```python
def loglik_dat_spect(theta, sbr_observed, t, covariates) -> float:
    """p(SBR_obs | theta, t)"""
    N_predicted = forward_simulate_N(theta, t)
    sbr_predicted = N_predicted * covariates["sbr_per_neuron"]
    return -0.5 * ((sbr_observed - sbr_predicted) / sigma_sbr)**2
```

- [ ] **Step 4: Implement `update_posterior()` function**

```python
# src/giman_pipeline/mechanistic_twin_v2/updater.py
import numpy as np
from .posterior_store import PosteriorStore, PatientPosterior
from .observations import compute_log_likelihood


def update_posterior(
    prior: PatientPosterior,
    observation: dict,
    ess_threshold: float = 0.5,
) -> PatientPosterior:
    """Update posterior via SIR reweighting."""
    log_lik = compute_log_likelihood(prior.samples, observation)
    log_weights_new = np.log(prior.weights + 1e-300) + log_lik
    log_weights_new -= log_weights_new.max()
    weights_new = np.exp(log_weights_new)
    weights_new /= weights_new.sum()

    ess = 1.0 / (weights_new ** 2).sum()
    if ess / len(weights_new) < ess_threshold:
        samples_new = resample_and_rejuvenate(prior.samples, weights_new, observation)
        weights_new = np.ones(len(samples_new)) / len(samples_new)
    else:
        samples_new = prior.samples

    return PatientPosterior(
        patno=prior.patno,
        version=prior.version + 1,
        samples=samples_new,
        weights=weights_new,
        ess=float(ess),
        log_marg_lik=prior.log_marg_lik + np.log((prior.weights * np.exp(log_lik)).sum() + 1e-300),
        param_names=prior.param_names,
    )


def resample_and_rejuvenate(samples, weights, obs, n_mcmc=10):
    """Systematic resampling + small MCMC moves."""
    N = len(samples)
    indices = np.random.choice(N, size=N, p=weights)
    resampled = samples[indices].copy()
    # Small random walk MCMC steps (Metropolis-Hastings)
    # ... (implementation detail)
    return resampled
```

- [ ] **Step 5: Write tests for updater**

```python
def test_update_preserves_valid_posterior():
    """Weights must sum to 1 after update."""
    ...

def test_update_exact_matches_full_is():
    """Update on all observations should match one-shot IS within tolerance."""
    # Critical validation: if we run update_posterior() sequentially with all
    # observations, result should match the original phase2_coupled_is_step26v4
    # run that used all observations at once.
    ...
```

- [ ] **Step 6: Implement replay harness**

```python
# scripts/mechanistic_twin/phase5_bidirectional_demo.py
"""Replay 400+ multi-scan patients, log prediction improvements per update."""
for patno in multi_scan_patients:
    # 1. Load baseline posterior (one-shot IS on scan 1 only)
    state_v1 = initialize_posterior(patno, first_scan_only=True)
    log_predictions(state_v1, horizons=[1, 3, 5])  # Frozen predictions

    # 2. Update with scan 2
    state_v2 = update_posterior(state_v1, observation_scan_2)
    log_predictions(state_v2, horizons=[1, 3])

    # 3. Update with scan 3
    state_v3 = update_posterior(state_v2, observation_scan_3)
    log_predictions(state_v3, horizons=[1])

    # 4. Compare frozen predictions to actual future observations
    validation_records.append(compare_predictions_to_outcomes(patno))

# Analysis: does MAE decrease with update count? Coverage stable?
```

- [ ] **Step 7: Analyze and report**

Primary endpoints:

- MAE at horizon Δ, stratified by update count (should decrease monotonically)
- 90% CI coverage (should stay calibrated at ≥90%)
- CRPS at each horizon

- [ ] **Step 8: Commit**

```bash
git add src/giman_pipeline/mechanistic_twin_v2/ scripts/mechanistic_twin/phase5_bidirectional_demo.py tests/mechanistic_twin_v2/
git commit -m "feat: Phase 5 Task 5 — bidirectional update demo (the twin proof)"
```

---

### Task 6: Observational Counterfactual Calibration

**Files:**

- Create: `scripts/mechanistic_twin/phase5_observational_counterfactual.py`
- Read: canonical parquet, LEDD log, Phase 2 posteriors
- Output: `outputs/mechanistic_twin/paper10_mech_vs_giman/observational_counterfactual.json`

**Context:** Replace synthetic 50-patient counterfactual (v1 plan) with **observational validation**. Identify PPMI patients who actually escalated LEDD by ≥200mg between visits. Compare model-predicted gap change (using Phase 4 β=-12.57) to observed gap change.

**This is a real validation, not extrapolation.**

- [ ] **Step 1: Extract LEDD escalation events**

```python
# For each patient, find consecutive visits where LEDD increased by >=200mg
# AND both visits have ON-OFF gap measurements
escalation_events = []
for patno, grp in canonical.groupby("PATNO"):
    grp_sorted = grp.sort_values("months_from_baseline")
    for i in range(1, len(grp_sorted)):
        prev, curr = grp_sorted.iloc[i-1], grp_sorted.iloc[i]
        if (curr["ledd_total"] - prev["ledd_total"] >= 200
            and pd.notna(prev["gap"]) and pd.notna(curr["gap"])):
            escalation_events.append({
                "PATNO": patno,
                "months_prev": prev["months_from_baseline"],
                "months_curr": curr["months_from_baseline"],
                "ledd_change": curr["ledd_total"] - prev["ledd_total"],
                "gap_prev": prev["gap"],
                "gap_curr": curr["gap"],
                "observed_delta_gap": curr["gap"] - prev["gap"],
                "n_frac_prev": prev["n_frac"],
                "n_frac_curr": curr["n_frac"],
            })
```

- [ ] **Step 2: Compute predicted gap change using interaction model**

```python
# Using Phase 4 confounding-controlled coefficients (M2 from phase4_confounding_control.json):
# beta_interaction = 1.410 (survives severity control, p=0.044)
beta_interaction = 1.410
beta_nfrac = -12.57  # From mixed-effects model
beta_ledd = 0.003

for event in escalation_events:
    event["predicted_delta_gap"] = (
        beta_ledd * (event["ledd_change"] / 500)
        + beta_interaction * event["n_frac_prev"] * (event["ledd_change"] / 500)
    )
```

- [ ] **Step 3: Compare predicted vs observed**

- Correlation between predicted and observed delta_gap (Pearson + Spearman)
- RMSE, MAE
- Scatter plot for figures
- Direction accuracy: sign(predicted) == sign(observed)?

- [ ] **Step 4: Sensitivity analyses**

- Does prediction quality depend on time between visits?
- Does it depend on N(t) at escalation (sub-EC50 vs higher)?
- Stratify by LEDD escalation magnitude

- [ ] **Step 5: Frame neuroprotection scenarios as hypothesis-generating only**

No ground truth for neuroprotection counterfactuals in PPMI. Report them but label clearly as "regression-based scenario projection, not observationally validated."

- [ ] **Step 6: Commit**

```bash
git add scripts/mechanistic_twin/phase5_observational_counterfactual.py
git commit -m "feat: Phase 5 Task 6 — observational counterfactual (LEDD escalation validation)"
```

---

### Task 7: NASEM Criteria Audit

**Files:**

- Create: `scripts/mechanistic_twin/phase5_nasem_audit.py`
- Output: `outputs/mechanistic_twin/paper10_mech_vs_giman/nasem_audit.json`

**Context:** Pure methodological. Map our work against NASEM 2024 digital twin criteria. Own the partial implementation honestly.

**7 NASEM criteria to score (0=absent, 1=partial, 2=substantial, 3=complete):**

1. Virtual representation (physiological model)
2. Bidirectional flow (data → model → decisions)
3. Predictive capability
4. Uncertainty quantification
5. Validation (V&V)
6. Fitness-for-purpose (context of use)
7. Governance (ethics, privacy, reproducibility)

- [ ] **Step 1: Implement audit script**

```python
# scripts/mechanistic_twin/phase5_nasem_audit.py
nasem_audit = {
    "virtual_representation": {
        "score": 2,  # substantial
        "evidence": [
            "Phase 2 coupled ODE (α-syn + N(t))",
            "Phase 4 Hill PD + N(t)×LEDD interaction",
            "Per-patient calibrated via IS on 1,065 patients",
        ],
        "gaps": [
            "Simplified vs full 5-module coupled ODE (no Lewy body propagation, no levodopa PK)",
        ],
    },
    "bidirectional_flow": {
        "score": 2,  # substantial after Task 5
        "evidence": ["Task 5 update_posterior() API", "Task 5 replay harness on 400 patients"],
        "gaps": ["Episodic updates (1-2 years) not continuous; no closed-loop re-treatment"],
    },
    "predictive_capability": {
        "score": 2,
        "evidence": ["Phase 4 Path B gap prediction", "Paper 4 conformal bands 91% coverage"],
        "gaps": ["Fixed-effects R²=0.051 is modest"],
    },
    "uncertainty_quantification": {
        "score": 3,  # complete (via Paper 4)
        "evidence": ["Paper 4 IPCW conformal bands", "Phase 2 posterior CIs", "PPC in Task 5"],
        "gaps": [],
    },
    "validation": {
        "score": 2,
        "evidence": [
            "Task 3 external validation on LCC",
            "Task 4 head-to-head vs Graph-DT",
            "Task 6 observational counterfactual",
        ],
        "gaps": ["No prospective interventional validation"],
    },
    "fitness_for_purpose": {
        "score": 2,
        "evidence": ["Context: PD progression prediction + treatment response", "Paper 9 NASEM audit table"],
        "gaps": ["Not qualified for regulatory decision-making"],
    },
    "governance": {
        "score": 3,  # complete
        "evidence": [
            "Closed-Loop Methodology v1.5",
            "Documentation Lifecycle Protocol v1.0",
            "All code + data provenance via _reproducibility.py",
        ],
        "gaps": [],
    },
}

total_score = sum(c["score"] for c in nasem_audit.values())
max_score = 7 * 3  # 21
compliance_pct = total_score / max_score * 100
```

- [ ] **Step 2: Generate summary table** for paper

- [ ] **Step 3: Commit**

---

### Task 8: Publication Figures

**Files:**

- Create: `scripts/mechanistic_twin/phase5_generate_figures.py`
- Output: `outputs/mechanistic_twin/paper10_mech_vs_giman/figures/*.{png,pdf}`

**Figures (9 total):**

1. **Fig 1:** Architecture diagram (PPMI → IS calibration → PosteriorStore → updater → counterfactual → validation)
2. **Fig 2:** NASEM criteria radar chart (our coverage + gaps)
3. **Fig 3:** Bidirectional demo — MAE vs update count (the twin proof)
4. **Fig 4:** External validation — LCC vs PPMI SBR decay distributions (KS test annotation)
5. **Fig 5:** Head-to-head C-index comparison (paired bootstrap CIs)
6. **Fig 6:** Observational counterfactual — predicted vs observed gap change (LEDD escalation events)
7. **Fig 7:** Patient case studies (3 fast progressors, 3 slow, 3 medium — use EDA findings)
8. **Fig 8:** Calibration plot (posterior predictive checks)
9. **Fig 9:** Dissertation arc — how Paper 10 fits with Papers 1-9

- [ ] Follow publication standards from Paper 9 (300 DPI, colorblind palette, no titles)
- [ ] Commit

---

### Task 9: Documentation Lifecycle (Cycle B)

Per `Docs/documentation_lifecycle_protocol.md`:

- [ ] **Update `outputs/defense_prep/mechanistic_digital_twin_roadmap.md`** Phase 5 section with v2 architecture
- [ ] **Update `CLAUDE.md`** Phase 5 status + data lineage fix note
- [ ] **Update `outputs/mechanistic_twin/phase2/REPRODUCIBILITY_MANIFEST.md`** with Task 0 provenance
- [ ] **Add NASEM 2024 + Musuamba 2021 + Friedrich 2016 + Viceconti 2020 + Hicks 2015 to bibliography.tex**
- [ ] **Write Paper 10 LaTeX manuscript** at `outputs/mechanistic_twin/paper10_mech_vs_giman/latex/main.tex`
  - Structure follows Atsou 2025 / Valderrama 2024 CPT:PSP conventions
  - Include NASEM audit table as methods subsection
  - "Study Highlights" box (4 items)
  - Honest labeling: "mechanistic patient-specific model" not "digital twin"
- [ ] **Compile PDF + open in Preview**
- [ ] **Final commit + push**

---

## Self-Review Checklist

**1. Spec coverage:** All 5 sections from the deep review recommendations are covered:

- §2 NASEM audit → Task 7
- §3 Bidirectional demo → Task 5
- §4 External validation LCC → Task 3
- §5 Head-to-head common endpoint → Task 4
- §6 Observational counterfactual → Task 6
- Plus Task 0 (data lineage), Task 1 (posterior store), Task 2 (cohort), Task 8 (figures), Task 9 (docs)

**2. Placeholder scan:** Some "see v1 plan Task 1 for boilerplate" and "detailed code omitted here for space" references remain. These are intentional — preserves v1's detailed code blocks (available in git history `5130cd8`) without duplication.

**3. Type consistency:** `PatientPosterior` dataclass used consistently in Tasks 1, 5. `PosteriorStore` API consistent throughout.

---

## Execution Options

**1. Subagent-Driven (recommended)** — Dispatch fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Critical path:**

- Task 0 (1-2 days) → Task 1 (3 days, blocks Task 5) → Tasks 2, 3, 4 in parallel → Task 5 (2 weeks) → Task 6 (3 days) → Task 7 (2 days) → Task 8 (1 week) → Task 9 (1 week)
- **Total: ~4 months** (fits Paper 10 target scope)

**Start recommendation:** Task 0 first (unblocks everything by providing canonical data source), then Task 1 (unblocks Task 5), then parallelize 2, 3, 4.
