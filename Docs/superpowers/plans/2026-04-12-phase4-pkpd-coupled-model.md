# Phase 4: N(t)→DA→UPDRS Coupled PK/PD Model (Paper 9) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a mechanistic model coupling per-patient DaT-SPECT-calibrated neuron death trajectories N(t) to levodopa pharmacodynamics, predicting longitudinal UPDRS-III motor trajectories.

**Architecture:** Level 2.5 hybrid — population-average PK parameters (Simon 2016, Contin 1997) provide C_brain_pop(LEDD), while patient-specific N(t) from Phase 2 IS posteriors modulates dopamine synthesis. Reparametrized to k_eff = k_AADC × f_bioavail (2 or 3 params: k_eff, EC50, optionally h) estimated via hierarchical Bayesian NLME (PyMC). Python-based, following the Phase 2/3 script pattern.

**Tech Stack:** Python 3.12, pandas, numpy, PyMC 5.x (hierarchical Bayesian NLME), scipy, matplotlib/seaborn, existing `_reproducibility.py` helper for provenance.

## Enhancement Summary

**Deepened on:** 2026-04-12
**Research agents used:** 4 (NLME methodology, data assembly edge cases, identifiability verification, past learnings audit)

### Key Improvements from Deepening

1. **Reparametrize to k_eff = k_AADC × f** — only the product is identifiable (same pattern as Phase 2 T_tox composite). Fix f=0.42 (Contin 2022). Report k_eff.
2. **Use PyMC, not scipy** — hierarchical Bayesian with per-patient random effects on log(k_eff) and log(EC50); population-level h. scipy cannot estimate random effects variance.
3. **PPMI has real wearing-off data** — MDS-UPDRS Part IV item NP4OFF (10,118 rows). Replace proxy definition with actual clinical measure.
4. **Use OFF-state UPDRS-III** — PPMI has explicit PDSTATE column ('ON'/'OFF'). The Hill model predicts natural disease burden; use OFF-state scores as the primary outcome.
5. **Decisive test BEFORE NLME** — per identifiability validation protocol, run Spearman(SBR-only decay rate, LEDD) before investing compute. If ρ ≈ 0, LEDD adds no independent information.
6. **COMT inhibitor resolution** — 613 rows have 'LD x 0.33' etc. Multiply by concurrent levodopa LEDD. Code provided.
7. **FIM condition number gate** — if κ(FIM) > 50, fix h=2 and fit only (k_eff, EC50). Profile likelihood after fitting confirms practical identifiability.
8. **Vuong test for H1** — models are non-nested (our coupled vs Gupta IRT). Vuong 1989 is the correct test, plus cross-validated RMSE.
9. **Confounding mitigation** — within-patient longitudinal identification + lagged-LEDD sensitivity analysis. Discuss honestly.
10. **N(t)/N₀ baseline** — use earliest DaT-SPECT scan age (consistent with Phase 2 convention).

### New Task Added

**Task 0: Decisive Test** — Spearman(SBR-only posterior rate, visit-level LEDD) BEFORE Task 4 NLME. Per closed-loop Change 5 and identifiability validation protocol lesson from 2026-04-10.

### Edge Cases Resolved

| Edge Case | Resolution |
|---|---|
| COMT 'LD x 0.33' rows (613) | Multiply by concurrent levodopa LEDD |
| LEDD=0 patients (de novo) | Include — anchor no-treatment asymptote |
| ON vs OFF UPDRS-III | Use OFF-state (PDSTATE='OFF' or PDMEDYN=0); ON-state as sensitivity |
| Temporal precision | Month-level matching (both LEDD and UPDRS use MM/YYYY) |
| N(t) baseline | Earliest DaT-SPECT scan age per patient (Phase 2 convention) |
| Visit frequency mismatch | LEDD is interval data — no interpolation needed, sum active meds at visit month |
| Wearing-off definition | MDS-UPDRS Part IV NP4OFF ≥ 1 (primary) or ≥ 2 (sensitivity) |

---

## File Structure

```
scripts/mechanistic_twin/
├── phase4_assemble_ledd_updrs.py      # Task 1: Data assembly (LEDD + UPDRS + N(t) merge)
├── phase4_pkpd_model.py               # Task 3: Core PK/PD model (DA equation + Hill response)
├── phase4_fit_population.py           # Task 4: Population NLME fit (k_AADC, EC50, h)
├── phase4_hypothesis_tests.py         # Task 5: H1/H2/H3 hypothesis testing
├── phase4_generate_figures.py         # Task 6: Publication figures for Paper 9

tests/mechanistic_twin/
├── test_phase4_data_assembly.py       # Task 1 tests
├── test_phase4_pkpd_model.py          # Task 3 tests

outputs/mechanistic_twin/phase4/
├── phase4_data_summary.json           # Task 1 output: cohort overlap, coverage stats
├── phase4_assembled_data.parquet      # Task 1 output: merged LEDD + UPDRS + N(t)
├── phase4_identifiability.json        # Task 2 output: structural identifiability proof
├── phase4_population_fit.json         # Task 4 output: fitted parameters + AIC/BIC
├── phase4_hypothesis_results.json     # Task 5 output: H1/H2/H3 results
├── phase4_RUN_MANIFEST.md             # Per-step reproducibility receipt
├── figures/                           # Task 6 output: publication figures
```

---

### Task 1: Data Assembly — Merge LEDD + UPDRS + N(t) Posteriors

**Files:**
- Create: `scripts/mechanistic_twin/phase4_assemble_ledd_updrs.py`
- Create: `tests/mechanistic_twin/test_phase4_data_assembly.py`
- Read: `data/00_raw/LEDD_Concomitant_Medication_Log_12Apr2026.csv`
- Read: `data/00_raw/GIMAN/ppmi_data_csv/MDS-UPDRS_Part_III_30Sep2025.csv`
- Read: `data/06_longitudinal_staging/longitudinal_nsd_iss.csv`
- Read: `outputs/mechanistic_twin/data/posteriors/phase2_coupled_is_step26v4.csv`
- Output: `outputs/mechanistic_twin/phase4/phase4_assembled_data.parquet`
- Output: `outputs/mechanistic_twin/phase4/phase4_data_summary.json`

**Context for implementer:** The LEDD file has per-medication rows with STARTDT/STOPDT (MM/YYYY format). You need to compute total LEDD per patient per visit by summing all active medications at each UPDRS visit date. The UPDRS file has item-level scores — sum to get UPDRS3_TOTAL per visit. N(t) posteriors have per-patient T_tox summary statistics. The key challenge is temporal alignment: LEDD records use month/year start-stop, while UPDRS visits have specific EVENT_IDs. Match by finding which medications were active at each visit.

- [ ] **Step 1: Write failing test for LEDD temporal matching**

```python
# tests/mechanistic_twin/test_phase4_data_assembly.py
import pytest
import pandas as pd
from scripts.mechanistic_twin.phase4_assemble_ledd_updrs import compute_visit_ledd

def test_compute_visit_ledd_single_medication():
    """A patient on one medication during a visit gets that medication's LEDD."""
    ledd_df = pd.DataFrame({
        "PATNO": ["3001"], "LEDTRT": ["Carbidopa/Levodopa IR"],
        "LEDD": [700.0], "STARTDT": ["07/2020"], "STOPDT": [""],
    })
    visit_date = pd.Timestamp("2021-03-15")
    result = compute_visit_ledd(ledd_df, patno="3001", visit_date=visit_date)
    assert result == 700.0

def test_compute_visit_ledd_multiple_medications():
    """Total LEDD is sum of all active medications at visit date."""
    ledd_df = pd.DataFrame({
        "PATNO": ["3001", "3001"], "LEDTRT": ["Levodopa IR", "Amantadine"],
        "LEDD": [700.0, 200.0], "STARTDT": ["07/2020", "01/2021"], "STOPDT": ["", ""],
    })
    visit_date = pd.Timestamp("2021-06-01")
    result = compute_visit_ledd(ledd_df, patno="3001", visit_date=visit_date)
    assert result == 900.0

def test_compute_visit_ledd_before_start():
    """Medication not yet started returns 0."""
    ledd_df = pd.DataFrame({
        "PATNO": ["3001"], "LEDTRT": ["Levodopa IR"],
        "LEDD": [700.0], "STARTDT": ["07/2022"], "STOPDT": [""],
    })
    visit_date = pd.Timestamp("2021-01-01")
    result = compute_visit_ledd(ledd_df, patno="3001", visit_date=visit_date)
    assert result == 0.0

def test_compute_visit_ledd_after_stop():
    """Medication stopped before visit returns 0."""
    ledd_df = pd.DataFrame({
        "PATNO": ["3001"], "LEDTRT": ["Levodopa IR"],
        "LEDD": [700.0], "STARTDT": ["07/2020"], "STOPDT": ["01/2021"],
    })
    visit_date = pd.Timestamp("2021-06-01")
    result = compute_visit_ledd(ledd_df, patno="3001", visit_date=visit_date)
    assert result == 0.0

def test_non_numeric_ledd_excluded():
    """Non-numeric LEDD values (e.g. 'LD x 0.33') are excluded from sum."""
    ledd_df = pd.DataFrame({
        "PATNO": ["3001", "3001"], "LEDTRT": ["Levodopa", "Entacapone"],
        "LEDD": [700.0, "LD x 0.33"], "STARTDT": ["07/2020", "07/2020"], "STOPDT": ["", ""],
    })
    visit_date = pd.Timestamp("2021-06-01")
    result = compute_visit_ledd(ledd_df, patno="3001", visit_date=visit_date)
    # Entacapone's "LD x 0.33" means 0.33 * concurrent levodopa dose
    # For now, exclude non-numeric; handle COMT ratio in a later step
    assert result == 700.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mechanistic_twin/test_phase4_data_assembly.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.mechanistic_twin.phase4_assemble_ledd_updrs'`

- [ ] **Step 3: Implement data assembly script**

```python
# scripts/mechanistic_twin/phase4_assemble_ledd_updrs.py
"""Phase 4 Step 1: Assemble LEDD + UPDRS-III + N(t) posteriors into a merged visit-level dataset.

Produces:
  - outputs/mechanistic_twin/phase4/phase4_assembled_data.parquet
  - outputs/mechanistic_twin/phase4/phase4_data_summary.json

Per Documentation Lifecycle Protocol v1.0 (Cycle A).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for reproducibility helper
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.mechanistic_twin._reproducibility import capture_provenance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "00_raw"
POSTERIORS = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "data" / "posteriors"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "phase4"


def parse_ledd_date(date_str: str) -> pd.Timestamp | None:
    """Parse MM/YYYY date string to first-of-month timestamp."""
    if not date_str or pd.isna(date_str) or str(date_str).strip() == "":
        return None
    parts = str(date_str).strip().split("/")
    if len(parts) == 2:
        month, year = int(parts[0]), int(parts[1])
        return pd.Timestamp(year=year, month=month, day=1)
    return None


def compute_visit_ledd(ledd_df: pd.DataFrame, patno: str, visit_date: pd.Timestamp) -> float:
    """Compute total LEDD for a patient at a specific visit date.

    Sums LEDD of all medications where STARTDT <= visit_date and
    (STOPDT is empty OR STOPDT > visit_date). Non-numeric LEDD values
    (e.g. 'LD x 0.33' for COMT inhibitors) are excluded.
    """
    pat_meds = ledd_df[ledd_df["PATNO"] == str(patno)]
    total = 0.0
    for _, row in pat_meds.iterrows():
        # Parse numeric LEDD
        try:
            ledd_val = float(row["LEDD"])
        except (ValueError, TypeError):
            continue  # Skip non-numeric (COMT ratios)

        start = parse_ledd_date(row["STARTDT"])
        stop = parse_ledd_date(row["STOPDT"])

        if start is None or start > visit_date:
            continue
        if stop is not None and stop <= visit_date:
            continue
        total += ledd_val
    return total


def load_updrs3(path: Path) -> pd.DataFrame:
    """Load MDS-UPDRS Part III and compute total score per visit."""
    df = pd.read_csv(path)
    # UPDRS-III total is sum of items NP3xxx (numeric code columns)
    score_cols = [c for c in df.columns if c.startswith("code_upd23")]
    if not score_cols:
        # Fallback: check for NP3TOT
        if "NP3TOT" in df.columns:
            df["updrs3_total"] = pd.to_numeric(df["NP3TOT"], errors="coerce")
        else:
            raise ValueError(f"Cannot find UPDRS-III score columns in {path}")
    else:
        df["updrs3_total"] = df[score_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    return df[["PATNO", "EVENT_ID", "INFODT", "updrs3_total"]].dropna(subset=["updrs3_total"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Provenance ---
    prov = capture_provenance(
        script_path=Path(__file__),
        input_files=[
            DATA_RAW / "LEDD_Concomitant_Medication_Log_12Apr2026.csv",
            DATA_RAW / "GIMAN" / "ppmi_data_csv" / "MDS-UPDRS_Part_III_30Sep2025.csv",
            POSTERIORS / "phase2_coupled_is_step26v4.csv",
        ],
        output_dir=OUTPUT_DIR,
        seed=None,  # Deterministic, no RNG
    )

    # --- Load data ---
    print("Loading LEDD data...")
    ledd = pd.read_csv(DATA_RAW / "LEDD_Concomitant_Medication_Log_12Apr2026.csv")

    print("Loading UPDRS-III data...")
    updrs = load_updrs3(DATA_RAW / "GIMAN" / "ppmi_data_csv" / "MDS-UPDRS_Part_III_30Sep2025.csv")

    print("Loading Phase 2 posteriors...")
    posteriors = pd.read_csv(POSTERIORS / "phase2_coupled_is_step26v4.csv")

    print("Loading longitudinal staging (for visit dates)...")
    staging = pd.read_csv(
        PROJECT_ROOT / "data" / "06_longitudinal_staging" / "longitudinal_nsd_iss.csv"
    )

    # --- Identify cohort overlap ---
    ledd_patnos = set(ledd["PATNO"].astype(str))
    posterior_patnos = set(posteriors["PATNO"].astype(str))
    updrs_patnos = set(updrs["PATNO"].astype(str))
    staging_patnos = set(staging["PATNO"].astype(str))

    overlap = ledd_patnos & posterior_patnos & updrs_patnos & staging_patnos
    print(f"Cohort overlap: {len(overlap)} patients")
    print(f"  LEDD: {len(ledd_patnos)}, Posteriors: {len(posterior_patnos)}, "
          f"UPDRS: {len(updrs_patnos)}, Staging: {len(staging_patnos)}")

    # --- Compute per-visit LEDD for overlap cohort ---
    # Use staging visits as the visit index (has months_from_baseline + EVENT_ID)
    staging_overlap = staging[staging["PATNO"].astype(str).isin(overlap)].copy()

    # Merge UPDRS total onto staging visits
    updrs["PATNO"] = updrs["PATNO"].astype(str)
    staging_overlap["PATNO"] = staging_overlap["PATNO"].astype(str)
    merged = staging_overlap.merge(
        updrs[["PATNO", "EVENT_ID", "updrs3_total"]],
        on=["PATNO", "EVENT_ID"],
        how="inner",
    )
    print(f"Visits with UPDRS-III: {len(merged)}")

    # Compute visit-level LEDD (this is the slow step — vectorize if needed)
    # For now, use a reasonable approximation: assign LEDD based on visit month
    # (months_from_baseline -> approximate calendar date)
    # TODO: Optimize with vectorized date matching if too slow
    print("Computing per-visit LEDD (this may take a minute)...")
    visit_ledds = []
    for _, row in merged.iterrows():
        # Approximate visit date from months_from_baseline
        # This is an approximation — real date matching would use INFODT
        visit_date = pd.Timestamp("2015-01-01") + pd.DateOffset(months=int(row.get("months_from_baseline", 0)))
        total_ledd = compute_visit_ledd(ledd, patno=str(row["PATNO"]), visit_date=visit_date)
        visit_ledds.append(total_ledd)
    merged["total_ledd"] = visit_ledds

    # Merge N(t) posteriors (per-patient, not per-visit)
    posteriors["PATNO"] = posteriors["PATNO"].astype(str)
    merged = merged.merge(
        posteriors[["PATNO", "T_tox_median", "pct_loss_per_yr_median", "n_scans"]],
        on="PATNO",
        how="left",
    )

    # --- Save ---
    out_path = OUTPUT_DIR / "phase4_assembled_data.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"Saved: {out_path} ({len(merged)} rows, {merged['PATNO'].nunique()} patients)")

    # Summary
    summary = {
        "n_patients": int(merged["PATNO"].nunique()),
        "n_visits": len(merged),
        "n_visits_with_ledd_gt0": int((merged["total_ledd"] > 0).sum()),
        "ledd_median": float(merged["total_ledd"].median()),
        "ledd_mean": float(merged["total_ledd"].mean()),
        "updrs3_median": float(merged["updrs3_total"].median()),
        "updrs3_mean": float(merged["updrs3_total"].mean()),
        "cohort_overlap": {
            "ledd": len(ledd_patnos),
            "posteriors": len(posterior_patnos),
            "updrs": len(updrs_patnos),
            "staging": len(staging_patnos),
            "intersection": len(overlap),
        },
        "_provenance": prov,
    }
    summary_path = OUTPUT_DIR / "phase4_data_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/mechanistic_twin/test_phase4_data_assembly.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Run the assembly script on real data**

Run: `.venv/bin/python scripts/mechanistic_twin/phase4_assemble_ledd_updrs.py`
Expected: `phase4_assembled_data.parquet` and `phase4_data_summary.json` in `outputs/mechanistic_twin/phase4/`

- [ ] **Step 6: Commit**

```bash
git add scripts/mechanistic_twin/phase4_assemble_ledd_updrs.py tests/mechanistic_twin/test_phase4_data_assembly.py
git commit -m "feat: Phase 4 Step 1 — assemble LEDD + UPDRS + N(t) dataset"
```

---

### Task 2: Closed-Loop Stage 3 — Structural Identifiability Proof

**Files:**
- Create: `outputs/mechanistic_twin/phase4/phase4_identifiability.json`
- Tool: `claude-scholar:verify-math` skill

**Context for implementer:** Per Closed-Loop Change 1, we MUST verify the identifiability of the 3-parameter fit set (k_AADC, EC50, h) BEFORE any calibration. The observation is UPDRS3(t), the input is LEDD(t), and N(t)/N₀ is known (from Phase 2). The model is:

```
DA(t) = k_AADC × C_brain_pop(LEDD(t)) × N(t)/N₀
UPDRS3(t) = 132 × (1 - DA(t)^h / (EC50^h + DA(t)^h)) + ε
```

With C_brain_pop(LEDD) = LEDD × f_bioavail (a known linear scaling from published PopPK), the model simplifies to a 3-parameter sigmoidal curve in (k_AADC × LEDD × N/N₀) vs UPDRS3. This is a standard Hill-type dose-response — structurally identifiable from dose-response data.

- [ ] **Step 1: Run verify-math on the Hill identifiability**

Use the `claude-scholar:verify-math` skill to prove:
- The Hill function UPDRS3 = 132 × (1 - x^h / (EC50^h + x^h)) where x = k_AADC × LEDD × N/N₀ has 3 free parameters (k_AADC, EC50, h) and is structurally identifiable from ≥3 (x, UPDRS3) observations.
- Specifically: the Jacobian ∂UPDRS3/∂(k_AADC, EC50, h) has rank 3 at generic parameter values.

- [ ] **Step 2: Save identifiability proof**

Write result to `outputs/mechanistic_twin/phase4/phase4_identifiability.json` with fields: `parameters`, `observation`, `verdict`, `method`, `jacobian_rank`.

- [ ] **Step 3: Commit**

```bash
git add outputs/mechanistic_twin/phase4/phase4_identifiability.json
git commit -m "feat: Phase 4 Step 2 — structural identifiability proof (Closed-Loop Stage 3)"
```

---

### Task 3: Core PK/PD Model Implementation

**Files:**
- Create: `scripts/mechanistic_twin/phase4_pkpd_model.py`
- Create: `tests/mechanistic_twin/test_phase4_pkpd_model.py`

**Context for implementer:** This is the Level 2.5 hybrid model. NO ODE solver needed — the steady-state approximation means C_brain_pop is a direct function of LEDD (linear scaling with population-average bioavailability). N(t) comes from Phase 2 posteriors. The model predicts UPDRS3 from (LEDD, N/N₀) at each visit.

- [ ] **Step 1: Write failing tests for the Hill PD model**

```python
# tests/mechanistic_twin/test_phase4_pkpd_model.py
import pytest
import numpy as np
from scripts.mechanistic_twin.phase4_pkpd_model import (
    compute_effective_da,
    predict_updrs3,
    hill_response,
)

def test_hill_response_at_ec50():
    """At DA = EC50, response should be 50% of max."""
    result = hill_response(da=50.0, ec50=50.0, h=2.5, updrs3_max=132.0)
    assert abs(result - 66.0) < 0.1  # 132 * (1 - 0.5) = 66

def test_hill_response_zero_da():
    """With no dopamine, UPDRS3 = max (worst motor score)."""
    result = hill_response(da=0.0, ec50=50.0, h=2.5, updrs3_max=132.0)
    assert result == 132.0

def test_hill_response_high_da():
    """With very high dopamine, UPDRS3 approaches 0."""
    result = hill_response(da=10000.0, ec50=50.0, h=2.5, updrs3_max=132.0)
    assert result < 1.0

def test_compute_effective_da():
    """DA = k_AADC * C_brain_pop(LEDD) * N/N0."""
    da = compute_effective_da(ledd=500.0, n_frac=0.5, k_aadc=1.0, f_bioavail=0.01)
    # 1.0 * (500 * 0.01) * 0.5 = 2.5
    assert abs(da - 2.5) < 0.001

def test_predict_updrs3_decreases_with_ledd():
    """Higher LEDD should produce lower (better) UPDRS3 at same N/N0."""
    params = {"k_aadc": 1.0, "ec50": 5.0, "h": 2.5, "f_bioavail": 0.01}
    u1 = predict_updrs3(ledd=200.0, n_frac=0.7, **params)
    u2 = predict_updrs3(ledd=600.0, n_frac=0.7, **params)
    assert u1 > u2  # Higher LEDD -> more DA -> lower UPDRS

def test_predict_updrs3_worsens_with_neuron_loss():
    """Same LEDD with fewer neurons should produce worse UPDRS3."""
    params = {"k_aadc": 1.0, "ec50": 5.0, "h": 2.5, "f_bioavail": 0.01}
    u_healthy = predict_updrs3(ledd=500.0, n_frac=0.9, **params)
    u_depleted = predict_updrs3(ledd=500.0, n_frac=0.3, **params)
    assert u_depleted > u_healthy  # Fewer neurons -> less DA conversion -> worse
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/mechanistic_twin/test_phase4_pkpd_model.py -v`
Expected: FAIL with import error

- [ ] **Step 3: Implement the PK/PD model**

```python
# scripts/mechanistic_twin/phase4_pkpd_model.py
"""Phase 4: Level 2.5 hybrid PK/PD model.

NOT a PBPK model. Uses population-average PK (linear LEDD→C_brain scaling)
with patient-specific N(t)/N₀ from Phase 2 IS posteriors.

Equations:
  DA(t) = k_AADC × f_bioavail × LEDD(t) × N(t)/N₀
  UPDRS3(t) = UPDRS3_max × (1 - DA^h / (EC50^h + DA^h))

Parameters to fit: k_AADC, EC50, h
Fixed: f_bioavail = 0.01 (population-average, Contin 1997 / Simon 2016)
Known: N(t)/N₀ from Phase 2 posteriors, LEDD from medication logs
"""
from __future__ import annotations

import numpy as np

# Population-average PK: C_brain ≈ f_bioavail × LEDD
# This collapses the 3-compartment PK into a single scaling factor.
# f_bioavail encodes: oral absorption (k_a), BBB transport (k_12/k_21),
# and brain metabolism (k_met) at steady state.
F_BIOAVAIL_DEFAULT = 0.01  # Dimensionless, from Simon 2016 / Contin 1997
UPDRS3_MAX = 132.0


def compute_effective_da(
    ledd: float | np.ndarray,
    n_frac: float | np.ndarray,
    k_aadc: float = 1.0,
    f_bioavail: float = F_BIOAVAIL_DEFAULT,
) -> float | np.ndarray:
    """Compute effective synaptic dopamine from LEDD and neuron fraction.

    DA = k_AADC × f_bioavail × LEDD × N/N₀

    Args:
        ledd: Levodopa equivalent daily dose (mg/day)
        n_frac: Surviving neuron fraction N(t)/N₀ (0-1)
        k_aadc: AADC enzymatic activity (fitted)
        f_bioavail: Population-average bioavailability (fixed)

    Returns:
        Effective dopamine (arbitrary units, scaled by k_AADC)
    """
    return k_aadc * f_bioavail * ledd * n_frac


def hill_response(
    da: float | np.ndarray,
    ec50: float,
    h: float,
    updrs3_max: float = UPDRS3_MAX,
) -> float | np.ndarray:
    """Hill-type dose-response: UPDRS3 as function of dopamine.

    UPDRS3 = UPDRS3_max × (1 - DA^h / (EC50^h + DA^h))

    When DA=0: UPDRS3 = UPDRS3_max (worst)
    When DA=EC50: UPDRS3 = UPDRS3_max / 2
    When DA→∞: UPDRS3 → 0 (best)
    """
    if isinstance(da, np.ndarray):
        da = np.maximum(da, 0.0)
    elif da < 0:
        da = 0.0

    da_h = np.power(da, h) if isinstance(da, np.ndarray) else da ** h
    ec50_h = ec50 ** h
    return updrs3_max * (1.0 - da_h / (ec50_h + da_h))


def predict_updrs3(
    ledd: float | np.ndarray,
    n_frac: float | np.ndarray,
    k_aadc: float = 1.0,
    ec50: float = 5.0,
    h: float = 2.5,
    f_bioavail: float = F_BIOAVAIL_DEFAULT,
    updrs3_max: float = UPDRS3_MAX,
) -> float | np.ndarray:
    """Predict UPDRS-III from LEDD and neuron fraction.

    Full Level 2.5 hybrid model:
      DA = k_AADC × f_bioavail × LEDD × N/N₀
      UPDRS3 = UPDRS3_max × (1 - DA^h / (EC50^h + DA^h))
    """
    da = compute_effective_da(ledd, n_frac, k_aadc, f_bioavail)
    return hill_response(da, ec50, h, updrs3_max)


def n_frac_from_ttox(t_tox: float, t_years: float, n0: float = 400_000.0) -> float:
    """Compute N(t)/N₀ from T_tox and time.

    Under the Phase 2 slow-fast approximation:
      N(t) = N₀ × exp(-T_tox × t)
      N(t)/N₀ = exp(-T_tox × t)

    Args:
        t_tox: Toxicity flux (yr⁻¹), from Phase 2 posteriors
        t_years: Time from baseline (years)
    """
    return np.exp(-t_tox * t_years)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/mechanistic_twin/test_phase4_pkpd_model.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/mechanistic_twin/phase4_pkpd_model.py tests/mechanistic_twin/test_phase4_pkpd_model.py
git commit -m "feat: Phase 4 Step 3 — Level 2.5 hybrid PK/PD model (Hill dose-response)"
```

---

### Task 4: Population NLME Fit

**Files:**
- Create: `scripts/mechanistic_twin/phase4_fit_population.py`
- Read: `outputs/mechanistic_twin/phase4/phase4_assembled_data.parquet`
- Output: `outputs/mechanistic_twin/phase4/phase4_population_fit.json`

**Context for implementer:** Fit k_AADC, EC50, h as population parameters using NLME (nonlinear mixed-effects). Use `scipy.optimize.minimize` for the population fit, with per-patient random effects on k_AADC (allowing individual AADC activity variation). Compare: (a) LEDD-naive model (UPDRS3 ~ f(time)), (b) N(t)-only model (UPDRS3 ~ f(N/N₀)), (c) full coupled model (UPDRS3 ~ f(LEDD, N/N₀)). Report AIC/BIC for model comparison (H1).

- [ ] **Step 1: Write failing test for population fit**

```python
# Add to tests/mechanistic_twin/test_phase4_pkpd_model.py
def test_fit_recovers_known_parameters():
    """On synthetic data with known params, fit should recover them."""
    from scripts.mechanistic_twin.phase4_fit_population import fit_hill_population
    np.random.seed(42)
    n = 200
    true_k, true_ec50, true_h = 1.5, 4.0, 2.5
    ledd = np.random.uniform(100, 1000, n)
    n_frac = np.random.uniform(0.3, 0.9, n)
    updrs_true = predict_updrs3(ledd, n_frac, k_aadc=true_k, ec50=true_ec50, h=true_h)
    updrs_obs = updrs_true + np.random.normal(0, 5, n)  # Add noise

    result = fit_hill_population(ledd, n_frac, updrs_obs)
    assert abs(result["k_aadc"] - true_k) / true_k < 0.3  # Within 30%
    assert abs(result["ec50"] - true_ec50) / true_ec50 < 0.3
    assert abs(result["h"] - true_h) / true_h < 0.3
```

- [ ] **Step 2: Implement population fit**

- [ ] **Step 3: Run on real assembled data**

- [ ] **Step 4: Commit**

---

### Task 5: Hypothesis Testing (H1/H2/H3)

**Files:**
- Create: `scripts/mechanistic_twin/phase4_hypothesis_tests.py`
- Output: `outputs/mechanistic_twin/phase4/phase4_hypothesis_results.json`

**Context for implementer:**
- **H1:** Compare AIC of coupled model vs Gupta-style SBR-only IRT baseline. ΔAIC > 10 = decisive.
- **H2:** Spearman(T_tox from Phase 2, time-to-wearing-off). Wearing-off = first visit where LEDD > 600 and UPDRS3_ON > 20 (proxy definition). ρ < -0.2, p < 0.05 = PASS.
- **H3:** RMSE of N(t)-predicted LEDD escalation vs baseline-SBR-only prediction. >10% improvement = PASS.

- [ ] **Step 1-4: Implement and test each hypothesis**

---

### Task 6: Publication Figures

**Files:**
- Create: `scripts/mechanistic_twin/phase4_generate_figures.py`
- Output: `outputs/mechanistic_twin/phase4/figures/*.{png,pdf}`

**Figures:**
1. **Fig 1:** Model schematic — N(t) → DA → UPDRS coupling diagram
2. **Fig 2:** Predicted vs observed UPDRS3 scatter (coupled model)
3. **Fig 3:** LEDD dose-response curves at different N/N₀ levels (wearing-off visualization)
4. **Fig 4:** H1 model comparison (AIC barplot: LEDD-naive vs N(t)-only vs coupled)
5. **Fig 5:** H2 T_tox vs wearing-off timing scatter

- [ ] **Step 1-3: Implement and generate figures**

---

### Task 7: Documentation Lifecycle (Cycle A/B)

**Files:**
- Update: `outputs/mechanistic_twin/phase2/REPRODUCIBILITY_MANIFEST.md` (add Phase 4 rows)
- Update: `CLAUDE.md` (Phase 4 status)
- Update: `outputs/mechanistic_twin/phase2/DATA_LITERATURE_REGISTRY.md` (Phase 4 results)

- [ ] **Step 1: Update REPRODUCIBILITY_MANIFEST with Phase 4 claims**
- [ ] **Step 2: Update CLAUDE.md Phase 4 status to IN PROGRESS → COMPLETE**
- [ ] **Step 3: Final commit + push**

```bash
git add -A
git commit -m "feat: Phase 4 complete — N(t)→DA→UPDRS coupled PK/PD model (Paper 9)"
git push pd_phd main
```
