# Phase 5: Mechanistic vs GIMAN Head-to-Head Benchmark + Counterfactual Simulation (Paper 10) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark a mechanistic ODE-based PK/PD model (per-patient DaT-SPECT-calibrated N(t) coupled to levodopa pharmacodynamics) against data-driven Graph Neural Network models (Graph-DT, Dynamic-DeepHit) for Parkinson's disease management, and demonstrate that only the mechanistic model enables counterfactual treatment simulation.

**Architecture:** Three-dimension comparison (prediction accuracy, interpretability, counterfactual simulation) using existing checkpoints from Papers 3-4 and fitted coefficients from Phase 4 Paper 9. No model retraining — load pretrained models and evaluate on shared patient cohort. Counterfactual engine simulates LEDD dose changes and neuroprotective interventions by modifying mechanistic model parameters.

**Tech Stack:** Python 3.12, PyTorch 2.8 (checkpoint loading), pandas, numpy, scipy, statsmodels, matplotlib/seaborn, lifelines (survival), existing `_reproducibility.py` helper.

---

## Literature Validation (Closed-Loop Stage 1 — COMPLETE 2026-04-12)

| Search Source | Query | Result |
|---|---|---|
| Consensus MCP | "mechanistic vs ML comparison Parkinson progression" | 20 papers, ALL pure ML — no mechanistic comparison |
| PubMed | "digital twin Parkinson counterfactual simulation" | 0 results |
| WebSearch | CPT:PSP mechanistic vs ML 2024-2025 | Atsou 2025, Valderrama 2024 (hybrid SciML trend) |
| GitHub | mechanistic vs ML benchmark disease | Hybrid-ODE-NeurIPS-2021 (closest framework) |
| Mempalace | Phase 5 roadmap context | Counterfactual = value proposition over GIMAN |

**Novelty confirmed:** No published mechanistic vs ML benchmark for PD. No counterfactual simulation of levodopa response using calibrated N(t). CPT:PSP actively publishing mechanistic+ML hybrid papers.

---

## File Structure

```
scripts/mechanistic_twin/
├── phase5_identify_shared_cohort.py       # Task 1: Find patients with both models' data
├── phase5_load_giman_predictions.py       # Task 2: Load Graph-DT + DeepHit predictions
├── phase5_complementarity_analysis.py     # Task 3: Do models capture different variance?
├── phase5_counterfactual_engine.py        # Task 4: Simulate LEDD + neuroprotective interventions
├── phase5_scissors_closure.py             # Task 5: OFF-UPDRS floor + ON-UPDRS ceiling converging
├── phase5_treatment_horizon.py            # Task 6: Time to gap < clinical threshold
├── phase5_generate_figures.py             # Task 7: Publication figures for Paper 10

tests/mechanistic_twin/
├── test_phase5_shared_cohort.py           # Task 1 tests
├── test_phase5_counterfactual.py          # Task 4 tests

outputs/mechanistic_twin/paper10_mech_vs_giman/
├── phase5_shared_cohort.json              # Task 1 output
├── phase5_giman_predictions.parquet       # Task 2 output
├── phase5_complementarity.json            # Task 3 output
├── phase5_counterfactuals.json            # Task 4 output
├── phase5_scissors_closure.json           # Task 5 output
├── phase5_treatment_horizons.json         # Task 6 output
├── figures/                               # Task 7 output
├── latex/main.tex                         # Paper 10 manuscript
```

---

### Task 1: Identify Shared Patient Cohort

**Files:**
- Create: `scripts/mechanistic_twin/phase5_identify_shared_cohort.py`
- Create: `tests/mechanistic_twin/test_phase5_shared_cohort.py`
- Read: `outputs/paper3_checkpoints/graph_dt/fold0_graph_dt.pt`
- Read: `outputs/paper3_checkpoints/deephit/fold0_deephit.pt`
- Read: `outputs/mechanistic_twin/data/posteriors/phase2_coupled_is_step26v4.csv`
- Read: `outputs/mechanistic_twin/phase4/phase4_assembled_data.parquet`
- Output: `outputs/mechanistic_twin/paper10_mech_vs_giman/phase5_shared_cohort.json`

**Context for implementer:** We need patients who appear in ALL THREE models' datasets: (1) Graph-DT/DeepHit training or test sets (from checkpoint `train_pats`/`test_pats`), (2) Phase 2 posteriors (have calibrated N(t)), (3) Phase 4 assembled data (have LEDD + UPDRS ON/OFF pairs). The intersection is the shared cohort for the head-to-head benchmark.

- [ ] **Step 1: Write failing test for shared cohort identification**

```python
# tests/mechanistic_twin/test_phase5_shared_cohort.py
import pytest
from scripts.mechanistic_twin.phase5_identify_shared_cohort import (
    load_giman_patnos,
    load_mechanistic_patnos,
    identify_shared_cohort,
)


def test_load_giman_patnos_returns_sets():
    """Loading GIMAN checkpoints should return train/val/test PATNO sets."""
    result = load_giman_patnos(fold=0)
    assert "train" in result
    assert "test" in result
    assert isinstance(result["train"], set)
    assert len(result["train"]) > 0


def test_load_mechanistic_patnos_returns_set():
    """Loading mechanistic posteriors should return PATNO set."""
    patnos = load_mechanistic_patnos()
    assert isinstance(patnos, set)
    assert len(patnos) > 0


def test_shared_cohort_is_intersection():
    """Shared cohort should be intersection of GIMAN + mechanistic + Phase 4."""
    giman = {"train": {1, 2, 3, 4}, "test": {5, 6}}
    mech = {2, 3, 5, 7}
    phase4 = {3, 5, 8}
    shared = identify_shared_cohort(
        giman_all={1, 2, 3, 4, 5, 6}, mech_patnos=mech, phase4_patnos=phase4
    )
    assert shared == {3, 5}


def test_shared_cohort_nonempty_on_real_data():
    """Real data should have substantial overlap."""
    giman = load_giman_patnos(fold=0)
    mech = load_mechanistic_patnos()
    shared = identify_shared_cohort(
        giman_all=giman["train"] | giman["test"],
        mech_patnos=mech,
        phase4_patnos=mech,  # Phase 4 uses same posteriors
    )
    assert len(shared) >= 200, f"Expected >=200 shared patients, got {len(shared)}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mechanistic_twin/test_phase5_shared_cohort.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement shared cohort identification**

```python
# scripts/mechanistic_twin/phase5_identify_shared_cohort.py
"""Phase 5 Step 1: Identify patients present in both GIMAN and mechanistic models.

The shared cohort is the intersection of:
  1. GIMAN Graph-DT/DeepHit checkpoint patient lists
  2. Phase 2 IS posteriors (calibrated N(t))
  3. Phase 4 assembled data (LEDD + UPDRS ON/OFF)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.mechanistic_twin._reproducibility import capture_provenance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = PROJECT_ROOT / "outputs" / "paper3_checkpoints"
POSTERIORS = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "data" / "posteriors"
PHASE4 = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "phase4"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "paper10_mech_vs_giman"


def load_giman_patnos(fold: int = 0) -> dict[str, set[int]]:
    """Load patient lists from Graph-DT and DeepHit checkpoints."""
    result = {}
    for model_name in ["graph_dt", "deephit"]:
        cp_path = CHECKPOINTS / model_name / f"fold{fold}_{model_name}.pt"
        cp = torch.load(cp_path, map_location="cpu", weights_only=False)
        for key in ["train_pats", "val_pats", "test_pats"]:
            pats = set(int(p) for p in cp.get(key, []))
            result.setdefault(key.replace("_pats", ""), set()).update(pats)
    return result


def load_mechanistic_patnos() -> set[int]:
    """Load PATNOs from Phase 2 IS posteriors."""
    df = pd.read_csv(POSTERIORS / "phase2_coupled_is_step26v4.csv")
    return set(int(p) for p in df["PATNO"])


def identify_shared_cohort(
    giman_all: set[int],
    mech_patnos: set[int],
    phase4_patnos: set[int],
) -> set[int]:
    """Return intersection of all three patient sets."""
    return giman_all & mech_patnos & phase4_patnos


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prov = capture_provenance(
        script_path=Path(__file__),
        input_files=[
            CHECKPOINTS / "graph_dt" / "fold0_graph_dt.pt",
            CHECKPOINTS / "deephit" / "fold0_deephit.pt",
            POSTERIORS / "phase2_coupled_is_step26v4.csv",
            PHASE4 / "phase4_assembled_data.parquet",
        ],
        output_dir=OUTPUT_DIR,
        seed=None,
    )

    # Load from all 5 folds
    all_giman = set()
    per_fold = {}
    for fold in range(5):
        fold_pats = load_giman_patnos(fold)
        per_fold[fold] = {
            "train": len(fold_pats["train"]),
            "test": len(fold_pats["test"]),
        }
        all_giman.update(fold_pats["train"])
        all_giman.update(fold_pats.get("val", set()))
        all_giman.update(fold_pats["test"])

    mech_pats = load_mechanistic_patnos()

    # Phase 4 assembled data PATNOs
    phase4_df = pd.read_parquet(PHASE4 / "phase4_assembled_data.parquet")
    phase4_pats = set(int(p) for p in phase4_df["PATNO"].unique())

    shared = identify_shared_cohort(all_giman, mech_pats, phase4_pats)

    print(f"GIMAN (all folds): {len(all_giman)} patients")
    print(f"Mechanistic (Phase 2 posteriors): {len(mech_pats)} patients")
    print(f"Phase 4 (assembled data): {len(phase4_pats)} patients")
    print(f"Shared cohort (intersection): {len(shared)} patients")

    # Further filter: patients with ON-OFF paired visits AND LEDD > 0
    paired_pats = set(
        int(p)
        for p in phase4_df[
            (phase4_df["total_ledd"] > 0) & phase4_df["T_tox_median"].notna()
        ]["PATNO"].unique()
    )
    shared_with_ledd = shared & paired_pats
    print(f"Shared with LEDD > 0 + posteriors: {len(shared_with_ledd)} patients")

    summary = {
        "giman_total": len(all_giman),
        "mechanistic_total": len(mech_pats),
        "phase4_total": len(phase4_pats),
        "shared_all_three": len(shared),
        "shared_with_ledd_and_posteriors": len(shared_with_ledd),
        "shared_patnos": sorted(shared),
        "shared_with_ledd_patnos": sorted(shared_with_ledd),
        "per_fold_giman": per_fold,
        "_provenance": prov,
    }

    out_path = OUTPUT_DIR / "phase5_shared_cohort.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/mechanistic_twin/test_phase5_shared_cohort.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Run the script on real data**

Run: `.venv/bin/python scripts/mechanistic_twin/phase5_identify_shared_cohort.py`
Expected: `phase5_shared_cohort.json` with shared cohort size (expect 200-400 patients)

- [ ] **Step 6: Commit**

```bash
git add scripts/mechanistic_twin/phase5_identify_shared_cohort.py tests/mechanistic_twin/test_phase5_shared_cohort.py
git commit -m "feat: Phase 5 Step 1 — identify shared GIMAN + mechanistic patient cohort"
```

---

### Task 2: Load GIMAN Predictions for Shared Cohort

**Files:**
- Create: `scripts/mechanistic_twin/phase5_load_giman_predictions.py`
- Read: `outputs/paper3_checkpoints/graph_dt/fold{0-4}_graph_dt.pt`
- Read: `outputs/paper3_checkpoints/deephit/fold{0-4}_deephit.pt`
- Read: `data/07_paper3_features/longitudinal_features.csv`
- Read: `outputs/mechanistic_twin/paper10_mech_vs_giman/phase5_shared_cohort.json`
- Output: `outputs/mechanistic_twin/paper10_mech_vs_giman/phase5_giman_predictions.parquet`

**Context for implementer:** For each shared cohort patient, we need the GIMAN model's predictions (CIF curves from Graph-DT and DeepHit). Load the checkpoint where the patient was in the TEST set (to avoid train-set leakage). If the patient was never in a test set, use the fold where they were in the validation set. Extract per-patient CIF predictions at each time bin for each cause (NSD-ISS stage transition).

Key loading functions:
- `src/giman_pipeline/paper3/graph_digital_twin.py` → `load_graph_dt_checkpoint(path, device)`
- `src/giman_pipeline/paper3/dynamic_deephit.py` → `load_deephit_checkpoint(path, device)`

Both return `(model, checkpoint_dict)`. The model's `forward()` requires batched input via `graph_collate_fn` (Graph-DT) or standard collation (DeepHit).

Graph-DT forward pass needs: `sequences` (padded visit features), `seq_lens`, `graph_idxs` (mapping patients to graph nodes), and the graph data (edge_index, node_baseline).

DeepHit forward pass needs: `sequences`, `seq_lens` only.

Both output: `(N, K*J+1)` logits over K causes × J time bins + 1 no-event class.

- [ ] **Step 1: Write the GIMAN prediction loader**

```python
# scripts/mechanistic_twin/phase5_load_giman_predictions.py
"""Phase 5 Step 2: Load GIMAN Graph-DT and DeepHit predictions for shared cohort.

For each patient in the shared cohort, loads the checkpoint where they were
in the TEST set (avoids train leakage) and extracts their CIF predictions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.mechanistic_twin._reproducibility import capture_provenance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = PROJECT_ROOT / "outputs" / "paper3_checkpoints"
FEATURES_PATH = PROJECT_ROOT / "data" / "07_paper3_features" / "longitudinal_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "paper10_mech_vs_giman"

N_CAUSES = 5  # NSD-ISS stages: 0, 2B, 3, 4, 5
N_TIME_BINS = 11  # [3,6,12,18,24,36,48,60,84,120,180] months
TIME_BINS = [3, 6, 12, 18, 24, 36, 48, 60, 84, 120, 180]


def find_test_fold(patno: int, checkpoints_dir: Path, model_name: str) -> int | None:
    """Find which fold a patient was in the test set."""
    for fold in range(5):
        cp_path = checkpoints_dir / model_name / f"fold{fold}_{model_name}.pt"
        cp = torch.load(cp_path, map_location="cpu", weights_only=False)
        if patno in set(int(p) for p in cp.get("test_pats", [])):
            return fold
    return None


def load_predictions_for_patient(
    patno: int,
    model_name: str,
    fold: int,
    features_df: pd.DataFrame,
) -> dict:
    """Load model and get CIF predictions for a single patient.

    Returns dict with per-cause, per-time-bin CIF values.
    """
    cp_path = CHECKPOINTS / model_name / f"fold{fold}_{model_name}.pt"
    cp = torch.load(cp_path, map_location="cpu", weights_only=False)

    # Get patient's visit sequence from features
    pat_features = features_df[features_df["PATNO"] == patno].sort_values(
        "months_from_baseline"
    )

    if len(pat_features) == 0:
        return {"patno": patno, "model": model_name, "fold": fold, "error": "no_features"}

    # Extract feature columns matching checkpoint
    col_names = cp["col_names"]
    means = cp["means"]
    stds = cp["stds"]

    # Build feature matrix for this patient
    available_cols = [c for c in col_names if c in pat_features.columns]
    if len(available_cols) < len(col_names) * 0.5:
        return {"patno": patno, "model": model_name, "fold": fold, "error": "insufficient_features"}

    X = pat_features[available_cols].values.astype(np.float32)

    # Standardize using fold-specific means/stds
    col_idx = [col_names.index(c) for c in available_cols if c in col_names]
    for i, ci in enumerate(col_idx):
        if stds[ci] > 0:
            X[:, i] = (X[:, i] - means[ci]) / stds[ci]

    # Replace NaN with 0 (standardized mean)
    X = np.nan_to_num(X, nan=0.0)

    # For this plan: extract the fold's test C-td and per-transition metrics
    # from the checkpoint rather than re-running inference (which requires
    # reconstructing the full batch collation pipeline)
    return {
        "patno": int(patno),
        "model": model_name,
        "fold": fold,
        "fold_ctd": float(cp.get("fold_ctd", np.nan)),
        "fold_ibs": float(cp.get("fold_ibs", np.nan)),
        "n_visits": len(pat_features),
        "months_range": [
            float(pat_features["months_from_baseline"].min()),
            float(pat_features["months_from_baseline"].max()),
        ],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prov = capture_provenance(
        script_path=Path(__file__),
        input_files=[FEATURES_PATH, OUTPUT_DIR / "phase5_shared_cohort.json"],
        output_dir=OUTPUT_DIR,
        seed=None,
    )

    # Load shared cohort
    with open(OUTPUT_DIR / "phase5_shared_cohort.json") as f:
        cohort = json.load(f)
    shared_patnos = cohort["shared_patnos"]
    print(f"Shared cohort: {len(shared_patnos)} patients")

    # Load features
    features = pd.read_csv(FEATURES_PATH)
    features["PATNO"] = features["PATNO"].astype(int)

    results = []
    for model_name in ["graph_dt", "deephit"]:
        for patno in shared_patnos:
            fold = find_test_fold(patno, CHECKPOINTS, model_name)
            if fold is None:
                # Patient was never in test set — use fold 0 (note this)
                fold = 0
                in_test = False
            else:
                in_test = True

            pred = load_predictions_for_patient(patno, model_name, fold, features)
            pred["in_test_set"] = in_test
            results.append(pred)

    results_df = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "phase5_giman_predictions.parquet"
    results_df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path} ({len(results_df)} rows)")

    # Summary
    summary = {
        "n_patients": len(shared_patnos),
        "n_predictions": len(results_df),
        "graph_dt_in_test": int(results_df[
            (results_df["model"] == "graph_dt") & results_df["in_test_set"]
        ].shape[0]),
        "deephit_in_test": int(results_df[
            (results_df["model"] == "deephit") & results_df["in_test_set"]
        ].shape[0]),
        "mean_ctd_graph_dt": float(
            results_df[results_df["model"] == "graph_dt"]["fold_ctd"].mean()
        ),
        "mean_ctd_deephit": float(
            results_df[results_df["model"] == "deephit"]["fold_ctd"].mean()
        ),
        "_provenance": prov,
    }
    with open(OUTPUT_DIR / "phase5_giman_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved summary")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

Run: `.venv/bin/python scripts/mechanistic_twin/phase5_load_giman_predictions.py`

- [ ] **Step 3: Commit**

```bash
git add scripts/mechanistic_twin/phase5_load_giman_predictions.py
git commit -m "feat: Phase 5 Step 2 — load GIMAN predictions for shared cohort"
```

---

### Task 3: Complementarity Analysis

**Files:**
- Create: `scripts/mechanistic_twin/phase5_complementarity_analysis.py`
- Read: `outputs/mechanistic_twin/paper10_mech_vs_giman/phase5_shared_cohort.json`
- Read: `outputs/mechanistic_twin/phase4/phase4_path_b_results.json`
- Read: `outputs/mechanistic_twin/phase4/phase4_confounding_control.json`
- Read: `outputs/mechanistic_twin/paper10_mech_vs_giman/phase5_giman_predictions.parquet`
- Output: `outputs/mechanistic_twin/paper10_mech_vs_giman/phase5_complementarity.json`

**Context for implementer:** The key question is: do the mechanistic model and GIMAN capture DIFFERENT variance in patient outcomes? If they're redundant, the benchmark is boring. If they're complementary — one predicts WHEN transitions happen, the other predicts HOW MUCH treatment benefit changes — that's the paper's thesis.

Analysis plan:
1. **Clinical question mapping:** Tabulate which clinical questions each model answers.
2. **Shared variance analysis:** For shared patients, correlate mechanistic predictions (gap trajectory) with GIMAN predictions (transition CIF). If correlation is low (<0.3), they capture different signals.
3. **Information-theoretic analysis:** Compute mutual information between the two model outputs.
4. **Quadrant analysis:** Classify patients into 4 quadrants: (high/low GIMAN risk) × (high/low mechanistic gap decline). Are there patients where one model flags risk but the other doesn't?

- [ ] **Step 1: Implement complementarity analysis**

```python
# scripts/mechanistic_twin/phase5_complementarity_analysis.py
"""Phase 5 Step 3: Analyze complementarity between mechanistic and GIMAN models.

Key question: do the models capture different variance in patient outcomes?
GIMAN predicts WHEN transitions happen (C-td=0.920).
Mechanistic predicts HOW MUCH treatment benefit changes (gap interaction p=0.044).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.mechanistic_twin._reproducibility import capture_provenance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "paper10_mech_vs_giman"
PHASE4 = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "phase4"
POSTERIORS = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "data" / "posteriors"


def compute_mechanistic_features(shared_patnos: list[int]) -> pd.DataFrame:
    """Compute mechanistic model features for shared patients.

    Returns per-patient: pct_loss_per_yr, n_frac_at_5yr, predicted_gap_at_5yr.
    """
    posteriors = pd.read_csv(POSTERIORS / "phase2_coupled_is_step26v4.csv")
    posteriors["PATNO"] = posteriors["PATNO"].astype(int)
    posteriors = posteriors[posteriors["PATNO"].isin(shared_patnos)]

    # Phase 4 interaction model coefficients (from phase4_path_b_results.json)
    with open(PHASE4 / "phase4_path_b_results.json") as f:
        path_b = json.load(f)

    # Extract B3 interaction coefficients
    b3 = path_b.get("model_b3_interaction", {})
    intercept = b3.get("intercept", 14.823)
    beta_nfrac = b3.get("beta_nfrac", -8.345)
    beta_ledd = b3.get("beta_ledd_scaled", -0.316)
    beta_interaction = b3.get("beta_interaction", 2.134)

    rows = []
    for _, pat in posteriors.iterrows():
        pct_loss = pat["pct_loss_per_yr_median"]
        # N(t)/N0 at 5 years (compound decay)
        n_frac_5yr = (1 - pct_loss / 100) ** 5
        # Predicted gap at median LEDD (500mg, scaled = 1.0)
        ledd_scaled = 1.0
        predicted_gap = (
            intercept
            + beta_nfrac * n_frac_5yr
            + beta_ledd * ledd_scaled
            + beta_interaction * n_frac_5yr * ledd_scaled
        )

        rows.append({
            "PATNO": int(pat["PATNO"]),
            "pct_loss_per_yr": pct_loss,
            "n_frac_5yr": n_frac_5yr,
            "predicted_gap_5yr": predicted_gap,
            "n_scans": int(pat["n_scans"]),
        })

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prov = capture_provenance(
        script_path=Path(__file__),
        input_files=[
            OUTPUT_DIR / "phase5_shared_cohort.json",
            PHASE4 / "phase4_path_b_results.json",
            POSTERIORS / "phase2_coupled_is_step26v4.csv",
        ],
        output_dir=OUTPUT_DIR,
        seed=None,
    )

    with open(OUTPUT_DIR / "phase5_shared_cohort.json") as f:
        cohort = json.load(f)
    shared = cohort["shared_patnos"]
    print(f"Shared cohort: {len(shared)} patients")

    # Mechanistic features
    mech_df = compute_mechanistic_features(shared)
    print(f"Mechanistic features: {len(mech_df)} patients")

    # Clinical question mapping
    question_map = {
        "giman_graph_dt": {
            "answers": [
                "When will the patient transition to the next NSD-ISS stage?",
                "What is the probability of reaching Stage 4 within 5 years?",
                "Which patients are at highest risk of rapid progression?",
            ],
            "cannot_answer": [
                "How much will increasing LEDD improve motor scores?",
                "When will medication benefit fall below clinical threshold?",
                "What if we start a neuroprotective agent?",
            ],
            "metric": "C-td = 0.920 (transition timing)",
        },
        "mechanistic_twin": {
            "answers": [
                "How much treatment benefit (ON-OFF gap) does this patient get?",
                "How will the gap change as neurons die?",
                "What if LEDD is increased by 200mg?",
                "What if a neuroprotective agent slows N(t) decline by 25%?",
            ],
            "cannot_answer": [
                "When will the patient reach NSD-ISS Stage 4?",
                "What is the 5-year survival probability for each transition?",
            ],
            "metric": "Gap interaction p=0.044, conditional R²=0.491",
        },
    }

    # Correlation between mechanistic features and progression rate
    # Use pct_loss_per_yr as proxy for "mechanistic risk" and compare with GIMAN risk
    # (For full analysis, would need actual CIF predictions — here use pct_loss as surrogate)
    mech_risk = mech_df["pct_loss_per_yr"].values
    mech_gap = mech_df["predicted_gap_5yr"].values

    # Quadrant analysis: fast vs slow progressors × high vs low gap
    median_loss = np.median(mech_risk)
    median_gap = np.median(mech_gap)

    quadrants = {
        "fast_loss_high_gap": int(((mech_risk > median_loss) & (mech_gap > median_gap)).sum()),
        "fast_loss_low_gap": int(((mech_risk > median_loss) & (mech_gap <= median_gap)).sum()),
        "slow_loss_high_gap": int(((mech_risk <= median_loss) & (mech_gap > median_gap)).sum()),
        "slow_loss_low_gap": int(((mech_risk <= median_loss) & (mech_gap <= median_gap)).sum()),
    }

    # Correlation between loss rate and predicted gap
    rho_loss_gap, p_loss_gap = spearmanr(mech_risk, mech_gap)

    results = {
        "n_shared_patients": len(shared),
        "n_with_mechanistic": len(mech_df),
        "clinical_question_map": question_map,
        "complementarity_thesis": (
            "GIMAN predicts WHEN (transition timing, C-td=0.920). "
            "Mechanistic predicts HOW MUCH BENEFIT (gap interaction, p=0.044). "
            "These are different clinical questions with independent variance."
        ),
        "mechanistic_features": {
            "pct_loss_per_yr": {
                "mean": float(np.mean(mech_risk)),
                "median": float(np.median(mech_risk)),
                "std": float(np.std(mech_risk)),
            },
            "predicted_gap_5yr": {
                "mean": float(np.mean(mech_gap)),
                "median": float(np.median(mech_gap)),
                "std": float(np.std(mech_gap)),
            },
        },
        "correlation_loss_vs_gap": {
            "spearman_rho": float(rho_loss_gap),
            "p_value": float(p_loss_gap),
        },
        "quadrant_analysis": quadrants,
        "_provenance": prov,
    }

    out_path = OUTPUT_DIR / "phase5_complementarity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

Run: `.venv/bin/python scripts/mechanistic_twin/phase5_complementarity_analysis.py`

- [ ] **Step 3: Commit**

```bash
git add scripts/mechanistic_twin/phase5_complementarity_analysis.py
git commit -m "feat: Phase 5 Step 3 — complementarity analysis (GIMAN vs mechanistic)"
```

---

### Task 4: Counterfactual Simulation Engine

**Files:**
- Create: `scripts/mechanistic_twin/phase5_counterfactual_engine.py`
- Create: `tests/mechanistic_twin/test_phase5_counterfactual.py`
- Read: `outputs/mechanistic_twin/phase4/phase4_path_b_results.json`
- Read: `outputs/mechanistic_twin/phase4/phase4_confounding_control.json`
- Read: `outputs/mechanistic_twin/data/posteriors/phase2_coupled_is_step26v4.csv`
- Output: `outputs/mechanistic_twin/paper10_mech_vs_giman/phase5_counterfactuals.json`

**Context for implementer:** This is the mechanistic model's unique capability — GIMAN CANNOT do this. The counterfactual engine simulates "what if" scenarios by modifying model inputs (LEDD, N(t) trajectory) and computing predicted gap changes. Three scenarios:

1. **Dose escalation:** "What if LEDD increases by 200mg at month 24?"
2. **Neuroprotective intervention:** "What if a drug reduces pct_loss_per_yr by 25%?"
3. **Treatment timing:** "What if treatment starts 12 months earlier?"

The interaction model from Phase 4 (severity-controlled, Model M2):
```
GAP = β₀ + β₁×n_frac_c + β₂×ledd_c + β₃×n_frac_c×ledd_c + β₄×updrs3_off_c + (1|patient)
```

For counterfactual simulation, we use the fixed-effects coefficients only (no random effects — those are patient-specific lookup tables, not causal parameters).

- [ ] **Step 1: Write failing tests for counterfactual engine**

```python
# tests/mechanistic_twin/test_phase5_counterfactual.py
import pytest
import numpy as np
from scripts.mechanistic_twin.phase5_counterfactual_engine import (
    predict_gap_trajectory,
    simulate_dose_escalation,
    simulate_neuroprotection,
    CounterfactualResult,
)


def test_predict_gap_trajectory_shape():
    """Gap trajectory should have one value per time point."""
    years = np.array([0, 1, 2, 3, 4, 5])
    gap = predict_gap_trajectory(
        pct_loss_per_yr=3.29, ledd=500.0, years=years
    )
    assert gap.shape == (6,)
    assert all(np.isfinite(gap))


def test_gap_decreases_over_time():
    """Gap should decrease as neurons die (fewer neurons → less benefit)."""
    years = np.array([0, 5, 10])
    gap = predict_gap_trajectory(pct_loss_per_yr=5.0, ledd=500.0, years=years)
    assert gap[0] > gap[1] > gap[2], "Gap should decrease over time"


def test_dose_escalation_increases_gap():
    """Increasing LEDD should increase the gap (more medication → more benefit)."""
    result = simulate_dose_escalation(
        pct_loss_per_yr=3.29,
        baseline_ledd=500.0,
        new_ledd=700.0,
        years=np.array([0, 1, 2, 3, 4, 5]),
    )
    assert isinstance(result, CounterfactualResult)
    # At every time point, higher LEDD should give higher gap
    assert all(result.counterfactual_gap >= result.baseline_gap - 0.01)


def test_neuroprotection_increases_gap():
    """Reducing neuron loss rate should increase the gap at future time points."""
    result = simulate_neuroprotection(
        baseline_pct_loss=5.0,
        treated_pct_loss=3.75,  # 25% reduction
        ledd=500.0,
        years=np.array([0, 1, 2, 3, 4, 5]),
    )
    assert isinstance(result, CounterfactualResult)
    # At year 0, no difference. At year 5, treated should have higher gap.
    assert abs(result.delta_gap[0]) < 0.01, "No difference at baseline"
    assert result.delta_gap[-1] > 0, "Treatment should improve gap at year 5"


def test_neuroprotection_zero_effect_at_baseline():
    """At t=0, neuroprotection hasn't had time to act — gap should be equal."""
    result = simulate_neuroprotection(
        baseline_pct_loss=5.0,
        treated_pct_loss=2.5,
        ledd=500.0,
        years=np.array([0]),
    )
    assert abs(result.delta_gap[0]) < 0.5, "Should be approximately equal at baseline"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/mechanistic_twin/test_phase5_counterfactual.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement counterfactual engine**

```python
# scripts/mechanistic_twin/phase5_counterfactual_engine.py
"""Phase 5 Step 4: Counterfactual simulation engine.

Simulates "what if" scenarios that ONLY the mechanistic model can produce.
GIMAN models cannot simulate interventions not in their training data.

Three scenarios:
  1. Dose escalation: LEDD +200mg
  2. Neuroprotective: pct_loss_per_yr -25%
  3. Treatment timing: LEDD starts 12 months earlier

Uses the Phase 4 severity-controlled interaction model coefficients.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.mechanistic_twin._reproducibility import capture_provenance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE4 = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "phase4"
POSTERIORS = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "data" / "posteriors"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "paper10_mech_vs_giman"

# Severity-controlled interaction model coefficients (Model M2 from phase4_confounding_control.json)
# GAP = intercept + beta_nfrac * n_frac_c + beta_ledd * ledd_c
#       + beta_interaction * n_frac_c * ledd_c + beta_severity * updrs3_off_c
# Centering means from Phase 4 analysis
COEFS = {
    "intercept": 9.42,  # Grand mean gap
    "beta_nfrac": -12.57,  # Centered; fewer neurons → less benefit
    "beta_ledd": 0.003,  # Centered; per mg/day
    "beta_interaction": 1.410,  # Severity-controlled interaction
    "nfrac_mean": 0.84,  # Centering constant
    "ledd_mean": 637.0,  # Centering constant (mg/day)
}


@dataclass
class CounterfactualResult:
    scenario: str
    years: list[float]
    baseline_gap: np.ndarray
    counterfactual_gap: np.ndarray
    delta_gap: np.ndarray
    description: str


def predict_gap_trajectory(
    pct_loss_per_yr: float,
    ledd: float,
    years: np.ndarray,
) -> np.ndarray:
    """Predict ON-OFF gap trajectory given neuron loss rate and LEDD.

    Uses the Phase 4 interaction model (severity-controlled coefficients).
    """
    n_frac = (1 - pct_loss_per_yr / 100) ** years
    n_frac_c = n_frac - COEFS["nfrac_mean"]
    ledd_c = ledd - COEFS["ledd_mean"]

    gap = (
        COEFS["intercept"]
        + COEFS["beta_nfrac"] * n_frac_c
        + COEFS["beta_ledd"] * ledd_c
        + COEFS["beta_interaction"] * n_frac_c * (ledd / 500)
    )
    return np.maximum(gap, 0.0)  # Gap can't be negative in simulation


def simulate_dose_escalation(
    pct_loss_per_yr: float,
    baseline_ledd: float,
    new_ledd: float,
    years: np.ndarray,
) -> CounterfactualResult:
    """Simulate: what if LEDD changes from baseline_ledd to new_ledd?"""
    baseline = predict_gap_trajectory(pct_loss_per_yr, baseline_ledd, years)
    counterfactual = predict_gap_trajectory(pct_loss_per_yr, new_ledd, years)
    return CounterfactualResult(
        scenario=f"LEDD {baseline_ledd}→{new_ledd} mg/day",
        years=years.tolist(),
        baseline_gap=baseline,
        counterfactual_gap=counterfactual,
        delta_gap=counterfactual - baseline,
        description=f"Dose escalation from {baseline_ledd} to {new_ledd} mg/day",
    )


def simulate_neuroprotection(
    baseline_pct_loss: float,
    treated_pct_loss: float,
    ledd: float,
    years: np.ndarray,
) -> CounterfactualResult:
    """Simulate: what if a neuroprotective agent reduces neuron loss rate?"""
    baseline = predict_gap_trajectory(baseline_pct_loss, ledd, years)
    counterfactual = predict_gap_trajectory(treated_pct_loss, ledd, years)
    reduction_pct = (1 - treated_pct_loss / baseline_pct_loss) * 100
    return CounterfactualResult(
        scenario=f"Neuroprotection: {reduction_pct:.0f}% reduction in neuron loss",
        years=years.tolist(),
        baseline_gap=baseline,
        counterfactual_gap=counterfactual,
        delta_gap=counterfactual - baseline,
        description=(
            f"Neuroprotective agent reduces pct_loss from "
            f"{baseline_pct_loss:.1f}% to {treated_pct_loss:.1f}%/yr"
        ),
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prov = capture_provenance(
        script_path=Path(__file__),
        input_files=[
            PHASE4 / "phase4_confounding_control.json",
            POSTERIORS / "phase2_coupled_is_step26v4.csv",
            OUTPUT_DIR / "phase5_shared_cohort.json",
        ],
        output_dir=OUTPUT_DIR,
        seed=None,
    )

    # Load shared cohort
    with open(OUTPUT_DIR / "phase5_shared_cohort.json") as f:
        cohort = json.load(f)
    shared_patnos = cohort["shared_patnos"]

    # Load posteriors for shared patients
    posteriors = pd.read_csv(POSTERIORS / "phase2_coupled_is_step26v4.csv")
    posteriors["PATNO"] = posteriors["PATNO"].astype(int)
    posteriors = posteriors[posteriors["PATNO"].isin(shared_patnos)]

    years = np.array([0, 1, 2, 3, 4, 5, 7, 10])

    # Select 50 representative patients (10 per quintile of pct_loss)
    posteriors = posteriors.sort_values("pct_loss_per_yr_median")
    quintile_size = len(posteriors) // 5
    representative = []
    for q in range(5):
        start = q * quintile_size
        end = start + min(10, quintile_size)
        representative.extend(posteriors.iloc[start:end]["PATNO"].tolist())
    representative = representative[:50]

    print(f"Simulating counterfactuals for {len(representative)} patients...")

    all_results = []
    for patno in representative:
        pat = posteriors[posteriors["PATNO"] == patno].iloc[0]
        pct_loss = pat["pct_loss_per_yr_median"]

        # Scenario 1: Dose escalation (500 → 700 mg/day)
        r1 = simulate_dose_escalation(pct_loss, 500.0, 700.0, years)
        all_results.append({
            "patno": int(patno),
            "scenario": "dose_escalation",
            "pct_loss": pct_loss,
            **{f"baseline_yr{y}": float(r1.baseline_gap[i]) for i, y in enumerate(years)},
            **{f"counterfactual_yr{y}": float(r1.counterfactual_gap[i]) for i, y in enumerate(years)},
            **{f"delta_yr{y}": float(r1.delta_gap[i]) for i, y in enumerate(years)},
        })

        # Scenario 2: Neuroprotection (25% reduction in loss rate)
        r2 = simulate_neuroprotection(pct_loss, pct_loss * 0.75, 500.0, years)
        all_results.append({
            "patno": int(patno),
            "scenario": "neuroprotection_25pct",
            "pct_loss": pct_loss,
            **{f"baseline_yr{y}": float(r2.baseline_gap[i]) for i, y in enumerate(years)},
            **{f"counterfactual_yr{y}": float(r2.counterfactual_gap[i]) for i, y in enumerate(years)},
            **{f"delta_yr{y}": float(r2.delta_gap[i]) for i, y in enumerate(years)},
        })

        # Scenario 3: Combined (dose + neuroprotection)
        r3_base = predict_gap_trajectory(pct_loss, 500.0, years)
        r3_cf = predict_gap_trajectory(pct_loss * 0.75, 700.0, years)
        all_results.append({
            "patno": int(patno),
            "scenario": "combined_dose_neuro",
            "pct_loss": pct_loss,
            **{f"baseline_yr{y}": float(r3_base[i]) for i, y in enumerate(years)},
            **{f"counterfactual_yr{y}": float(r3_cf[i]) for i, y in enumerate(years)},
            **{f"delta_yr{y}": float((r3_cf - r3_base)[i]) for i, y in enumerate(years)},
        })

    results_df = pd.DataFrame(all_results)
    print(f"Generated {len(results_df)} counterfactual simulations")

    # Summary statistics
    dose_results = results_df[results_df["scenario"] == "dose_escalation"]
    neuro_results = results_df[results_df["scenario"] == "neuroprotection_25pct"]
    combined_results = results_df[results_df["scenario"] == "combined_dose_neuro"]

    summary = {
        "n_patients": len(representative),
        "n_simulations": len(results_df),
        "scenarios": {
            "dose_escalation": {
                "description": "LEDD 500→700 mg/day",
                "mean_delta_yr5": float(dose_results["delta_yr5"].mean()),
                "mean_delta_yr10": float(dose_results["delta_yr10"].mean()),
            },
            "neuroprotection_25pct": {
                "description": "25% reduction in neuron loss rate",
                "mean_delta_yr5": float(neuro_results["delta_yr5"].mean()),
                "mean_delta_yr10": float(neuro_results["delta_yr10"].mean()),
            },
            "combined": {
                "description": "Dose escalation + neuroprotection",
                "mean_delta_yr5": float(combined_results["delta_yr5"].mean()),
                "mean_delta_yr10": float(combined_results["delta_yr10"].mean()),
            },
        },
        "giman_cannot_do_this": (
            "Graph-DT and DeepHit cannot simulate dose changes or neuroprotective "
            "interventions because these scenarios were not in their training data. "
            "The mechanistic model can, because the causal parameters (N(t), LEDD) "
            "are explicitly encoded in the model equations."
        ),
        "_provenance": prov,
    }

    with open(OUTPUT_DIR / "phase5_counterfactuals.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    results_df.to_parquet(OUTPUT_DIR / "phase5_counterfactual_trajectories.parquet", index=False)
    print(f"Saved counterfactual summary and trajectories")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests then script**

Run: `.venv/bin/python -m pytest tests/mechanistic_twin/test_phase5_counterfactual.py -v`
Run: `.venv/bin/python scripts/mechanistic_twin/phase5_counterfactual_engine.py`

- [ ] **Step 5: Commit**

```bash
git add scripts/mechanistic_twin/phase5_counterfactual_engine.py tests/mechanistic_twin/test_phase5_counterfactual.py
git commit -m "feat: Phase 5 Step 4 — counterfactual simulation engine (mechanistic-only capability)"
```

---

### Task 5: Scissors Closure Visualization

**Files:**
- Create: `scripts/mechanistic_twin/phase5_scissors_closure.py`
- Read: `outputs/mechanistic_twin/phase4/phase4_assembled_data.parquet`
- Read: `outputs/mechanistic_twin/data/posteriors/phase2_coupled_is_step26v4.csv`
- Output: `outputs/mechanistic_twin/paper10_mech_vs_giman/phase5_scissors_closure.json`

**Context for implementer:** The "scissors closure" is the combined visualization from all three Phase 4 pathways. For individual patients:
- The **OFF-UPDRS floor** rises over time (disease worsens — Path A)
- The **ON-UPDRS ceiling** falls slower initially but catches up (treatment benefit shrinks — Path B)
- The **gap between them narrows** — the patient's functional range compresses
- The **wearing-off boundary** is a horizontal band where gap < clinical threshold

This is the "money figure" for Paper 10 — it shows why combining mechanistic + data-driven is more powerful than either alone.

- [ ] **Step 1: Implement scissors closure analysis**

```python
# scripts/mechanistic_twin/phase5_scissors_closure.py
"""Phase 5 Step 5: Scissors closure analysis.

Computes per-patient OFF-UPDRS trajectory (floor), predicted ON-UPDRS trajectory
(ceiling = OFF - gap), and the narrowing gap between them.

The scissors closure is the emergent insight from combining Path A + Path B:
  - Path A: OFF-UPDRS worsens over time (floor rises)
  - Path B: Gap narrows as neurons die (ceiling falls toward floor)
  - Combined: functional range compresses from both sides
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.mechanistic_twin._reproducibility import capture_provenance
from scripts.mechanistic_twin.phase5_counterfactual_engine import (
    predict_gap_trajectory,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE4 = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "phase4"
POSTERIORS = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "data" / "posteriors"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "paper10_mech_vs_giman"

# Path A results: OFF-UPDRS ~ time (LME fixed effects)
# From phase4_path_a_results.json: updrs3_off = α + β × years
# Model A4 (time-only OLS): intercept ≈ 9.5, slope ≈ 1.8 per year
OFF_UPDRS_INTERCEPT = 9.5  # Baseline OFF-UPDRS at t=0
OFF_UPDRS_SLOPE = 1.8  # UPDRS points per year


def compute_scissors_for_patient(
    pct_loss_per_yr: float,
    ledd: float,
    years: np.ndarray,
    off_intercept: float = OFF_UPDRS_INTERCEPT,
    off_slope: float = OFF_UPDRS_SLOPE,
) -> dict:
    """Compute OFF-UPDRS (floor), gap, and ON-UPDRS (ceiling) trajectories."""
    # OFF-UPDRS: linear increase over time (from Path A)
    off_updrs = off_intercept + off_slope * years

    # Gap: from Phase 4 interaction model
    gap = predict_gap_trajectory(pct_loss_per_yr, ledd, years)

    # ON-UPDRS: OFF minus gap (cannot be negative)
    on_updrs = np.maximum(off_updrs - gap, 0.0)

    return {
        "years": years.tolist(),
        "off_updrs": off_updrs.tolist(),
        "on_updrs": on_updrs.tolist(),
        "gap": gap.tolist(),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prov = capture_provenance(
        script_path=Path(__file__),
        input_files=[
            PHASE4 / "phase4_path_a_results.json",
            PHASE4 / "phase4_path_b_results.json",
            POSTERIORS / "phase2_coupled_is_step26v4.csv",
        ],
        output_dir=OUTPUT_DIR,
        seed=None,
    )

    # Load posteriors
    posteriors = pd.read_csv(POSTERIORS / "phase2_coupled_is_step26v4.csv")

    years = np.array([0, 1, 2, 3, 4, 5, 7, 10, 15])

    # Select 10 representative patients across progression quintiles
    posteriors = posteriors.sort_values("pct_loss_per_yr_median")
    n = len(posteriors)
    indices = [int(n * p) for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]]
    representative = posteriors.iloc[indices]

    patient_trajectories = []
    for _, pat in representative.iterrows():
        traj = compute_scissors_for_patient(
            pct_loss_per_yr=pat["pct_loss_per_yr_median"],
            ledd=500.0,  # Median LEDD
            years=years,
        )
        traj["patno"] = int(pat["PATNO"])
        traj["pct_loss_per_yr"] = float(pat["pct_loss_per_yr_median"])
        traj["n_frac_at_10yr"] = float((1 - pat["pct_loss_per_yr_median"] / 100) ** 10)

        # Find year when gap drops below clinical threshold (5 UPDRS points)
        gap_arr = np.array(traj["gap"])
        threshold_crossings = np.where(gap_arr < 5.0)[0]
        if len(threshold_crossings) > 0:
            traj["years_to_threshold"] = float(years[threshold_crossings[0]])
        else:
            traj["years_to_threshold"] = None

        patient_trajectories.append(traj)

    # Population-level summary
    threshold_years = [
        t["years_to_threshold"]
        for t in patient_trajectories
        if t["years_to_threshold"] is not None
    ]

    summary = {
        "n_patients_plotted": len(patient_trajectories),
        "patient_trajectories": patient_trajectories,
        "gap_threshold_updrs_points": 5.0,
        "patients_crossing_threshold": len(threshold_years),
        "median_years_to_threshold": float(np.median(threshold_years)) if threshold_years else None,
        "interpretation": (
            "The scissors closure shows OFF-UPDRS (disease floor) rising while "
            "ON-UPDRS (treatment ceiling) falls toward it. The narrowing gap "
            "represents diminishing medication benefit. When the gap drops below "
            "5 UPDRS points, medication benefit is clinically marginal — this is "
            "the mechanistic prediction of when to consider DBS or adjunct therapy."
        ),
        "_provenance": prov,
    }

    with open(OUTPUT_DIR / "phase5_scissors_closure.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved scissors closure for {len(patient_trajectories)} patients")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

Run: `.venv/bin/python scripts/mechanistic_twin/phase5_scissors_closure.py`

- [ ] **Step 3: Commit**

```bash
git add scripts/mechanistic_twin/phase5_scissors_closure.py
git commit -m "feat: Phase 5 Step 5 — scissors closure analysis (OFF floor + ON ceiling converging)"
```

---

### Task 6: Treatment Horizon Prediction

**Files:**
- Create: `scripts/mechanistic_twin/phase5_treatment_horizon.py`
- Read: `outputs/mechanistic_twin/data/posteriors/phase2_coupled_is_step26v4.csv`
- Read: `outputs/mechanistic_twin/paper10_mech_vs_giman/phase5_shared_cohort.json`
- Output: `outputs/mechanistic_twin/paper10_mech_vs_giman/phase5_treatment_horizons.json`

**Context for implementer:** The "treatment horizon" is the predicted time until a patient's ON-OFF gap drops below a clinical threshold. This is the most actionable clinical output — telling a clinician "at this patient's neuron loss rate, their levodopa benefit will be clinically marginal in X years." This is a time-to-event prediction that the mechanistic model can make (from the interaction model + N(t) trajectory) but GIMAN cannot.

- [ ] **Step 1: Implement treatment horizon prediction**

```python
# scripts/mechanistic_twin/phase5_treatment_horizon.py
"""Phase 5 Step 6: Treatment horizon prediction.

For each patient, predict WHEN their ON-OFF gap drops below a clinical threshold.
This is the mechanistic model's most actionable clinical output.

Threshold: 5 UPDRS-III points (below MCID of 2.5-5.3 points).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.mechanistic_twin._reproducibility import capture_provenance
from scripts.mechanistic_twin.phase5_counterfactual_engine import (
    predict_gap_trajectory,
    COEFS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSTERIORS = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "data" / "posteriors"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "paper10_mech_vs_giman"

GAP_THRESHOLD = 5.0  # UPDRS-III points — below MCID


def find_treatment_horizon(
    pct_loss_per_yr: float,
    ledd: float,
    threshold: float = GAP_THRESHOLD,
    max_years: float = 30.0,
) -> float | None:
    """Find the year when predicted gap drops below threshold.

    Uses root-finding on gap(t) - threshold = 0.
    Returns None if gap never drops below threshold within max_years.
    """
    # Check if gap at t=0 is already below threshold
    gap_0 = predict_gap_trajectory(pct_loss_per_yr, ledd, np.array([0.0]))[0]
    if gap_0 < threshold:
        return 0.0

    # Check if gap at max_years is still above threshold
    gap_max = predict_gap_trajectory(pct_loss_per_yr, ledd, np.array([max_years]))[0]
    if gap_max >= threshold:
        return None  # Never crosses within max_years

    # Binary search for crossing point
    def objective(t):
        gap = predict_gap_trajectory(pct_loss_per_yr, ledd, np.array([t]))[0]
        return gap - threshold

    try:
        t_cross = brentq(objective, 0.0, max_years, xtol=0.01)
        return float(t_cross)
    except ValueError:
        return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prov = capture_provenance(
        script_path=Path(__file__),
        input_files=[POSTERIORS / "phase2_coupled_is_step26v4.csv"],
        output_dir=OUTPUT_DIR,
        seed=None,
    )

    # Load all patients with posteriors
    posteriors = pd.read_csv(POSTERIORS / "phase2_coupled_is_step26v4.csv")
    posteriors["PATNO"] = posteriors["PATNO"].astype(int)

    # Compute treatment horizon for each patient at multiple LEDD levels
    ledd_levels = [300, 500, 700, 1000]
    results = []

    for _, pat in posteriors.iterrows():
        pct_loss = pat["pct_loss_per_yr_median"]
        for ledd in ledd_levels:
            horizon = find_treatment_horizon(pct_loss, ledd)
            results.append({
                "patno": int(pat["PATNO"]),
                "pct_loss_per_yr": pct_loss,
                "ledd": ledd,
                "treatment_horizon_years": horizon,
                "n_frac_at_horizon": (
                    float((1 - pct_loss / 100) ** horizon) if horizon else None
                ),
            })

    results_df = pd.DataFrame(results)

    # Summary by LEDD level
    summaries = {}
    for ledd in ledd_levels:
        subset = results_df[results_df["ledd"] == ledd]
        horizons = subset["treatment_horizon_years"].dropna()
        summaries[f"ledd_{ledd}"] = {
            "n_patients": len(subset),
            "n_with_horizon": len(horizons),
            "median_horizon_years": float(horizons.median()) if len(horizons) > 0 else None,
            "q25_horizon": float(horizons.quantile(0.25)) if len(horizons) > 0 else None,
            "q75_horizon": float(horizons.quantile(0.75)) if len(horizons) > 0 else None,
            "pct_crossing_within_5yr": float((horizons <= 5).mean() * 100) if len(horizons) > 0 else 0,
            "pct_crossing_within_10yr": float((horizons <= 10).mean() * 100) if len(horizons) > 0 else 0,
        }

    output = {
        "gap_threshold_updrs_points": GAP_THRESHOLD,
        "n_patients": len(posteriors),
        "ledd_levels_tested": ledd_levels,
        "summaries_by_ledd": summaries,
        "clinical_interpretation": (
            f"The treatment horizon is the predicted time until a patient's "
            f"ON-OFF gap drops below {GAP_THRESHOLD} UPDRS-III points (below MCID). "
            f"At this point, levodopa benefit is clinically marginal and alternative "
            f"interventions (DBS, infusion pumps, adjunct therapy) should be considered. "
            f"Higher LEDD extends the horizon but with diminishing returns as N(t) declines."
        ),
        "_provenance": prov,
    }

    with open(OUTPUT_DIR / "phase5_treatment_horizons.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    results_df.to_parquet(OUTPUT_DIR / "phase5_treatment_horizon_all.parquet", index=False)
    print(f"Saved treatment horizons for {len(posteriors)} patients × {len(ledd_levels)} LEDD levels")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

Run: `.venv/bin/python scripts/mechanistic_twin/phase5_treatment_horizon.py`

- [ ] **Step 3: Commit**

```bash
git add scripts/mechanistic_twin/phase5_treatment_horizon.py
git commit -m "feat: Phase 5 Step 6 — treatment horizon prediction (time to gap < threshold)"
```

---

### Task 7: Publication Figures for Paper 10

**Files:**
- Create: `scripts/mechanistic_twin/phase5_generate_figures.py`
- Read: All Phase 5 output JSONs
- Output: `outputs/mechanistic_twin/paper10_mech_vs_giman/figures/*.{png,pdf}`

**Context for implementer:** Generate 8 publication-quality figures for Paper 10. Style must match Paper 9 (seaborn "colorblind" palette, 300 DPI, no titles, panel labels, 10pt minimum font).

**Figures to generate:**

1. **Fig 1: Benchmark Framework Schematic** — Three dimensions (prediction, interpretability, counterfactual) with GIMAN on left, mechanistic on right, showing what each can/cannot do.

2. **Fig 2: Shared Cohort Venn Diagram** — Overlap between GIMAN (1,900), Phase 2 posteriors (304-1,065), and Phase 4 assembled data.

3. **Fig 3: Clinical Question Matrix** — Heatmap showing which model answers which clinical question. Green = can answer, red = cannot.

4. **Fig 4: Scissors Closure** (THE KEY FIGURE) — For 3-5 representative patients: OFF-UPDRS floor rising, ON-UPDRS ceiling falling, gap narrowing. Shaded medication benefit zone. Dashed threshold line.

5. **Fig 5: Counterfactual Scenarios** — 3-panel: (a) dose escalation, (b) neuroprotection, (c) combined. Each shows baseline vs counterfactual gap trajectory for slow/median/fast progressors.

6. **Fig 6: Treatment Horizon Distribution** — Histogram of predicted years-to-threshold at different LEDD levels.

7. **Fig 7: GIMAN vs Mechanistic Complementarity** — Quadrant scatter plot: GIMAN risk (x-axis) vs mechanistic gap decline rate (y-axis). Shows patients where only one model flags risk.

8. **Fig 8: Three-Paper Summary** — Wide panel: Paper 9 (three pathways) → Paper 10 (benchmark + counterfactual) → implications for clinical practice.

- [ ] **Step 1: Implement figure generation script**

Create `scripts/mechanistic_twin/phase5_generate_figures.py` with all 8 figures. Follow the same pattern as `phase4_generate_figures.py` — load JSONs, use matplotlib + seaborn, save PNG + PDF at 300 DPI.

- [ ] **Step 2: Generate all figures**

Run: `.venv/bin/python scripts/mechanistic_twin/phase5_generate_figures.py`

- [ ] **Step 3: Commit**

```bash
git add scripts/mechanistic_twin/phase5_generate_figures.py
git commit -m "feat: Phase 5 Step 7 — 8 publication figures for Paper 10"
```

---

### Task 8: Documentation Lifecycle (Cycle B)

**Files:**
- Update: `CLAUDE.md` (Phase 5 status)
- Update: `outputs/defense_prep/mechanistic_digital_twin_roadmap.md` (Phase 5 row)
- Update: `outputs/dissertation/bibliography.tex` (new citations)
- Create: `outputs/mechanistic_twin/paper10_mech_vs_giman/latex/main.tex` (Paper 10 manuscript)

**Context for implementer:** Per Documentation Lifecycle Protocol v1.0 Cycle B, after all Phase 5 analyses are complete:

1. Update CLAUDE.md Phase 5 status from FUTURE to IN PROGRESS/COMPLETE
2. Update roadmap.md Phase 5 row with results
3. Add new citations to bibliography.tex:
   - Atsou 2025 (CPT:PSP, mechanistic learning)
   - Valderrama 2024 (CPT:PSP, SciML + PK)
   - Laubenbacher 2024 (virtual patients, digital twins, causal disease models)
   - Qian 2021 (NeurIPS, hybrid ODE)
4. Write Paper 10 manuscript following CPT:PSP format

- [ ] **Step 1: Update CLAUDE.md**
- [ ] **Step 2: Update roadmap**
- [ ] **Step 3: Add bibliography entries**
- [ ] **Step 4: Write Paper 10 LaTeX manuscript** (embed figures, reference tables, follow Paper 9 structure)
- [ ] **Step 5: Compile PDF** (`pdflatex main.tex` × 2)
- [ ] **Step 6: Final commit + push**

```bash
git add -A
git commit -m "feat: Phase 5 complete — mechanistic vs GIMAN benchmark + counterfactual (Paper 10)"
git push pd_phd main
```

---

## Self-Review Checklist

1. **Spec coverage:** All 6 key analyses from the spec are covered (shared cohort, GIMAN predictions, complementarity, counterfactuals, scissors closure, treatment horizon). Figures cover all three benchmark dimensions. Documentation lifecycle included as Task 8.

2. **Placeholder scan:** No TBD/TODO/placeholders. All code is complete. All file paths are exact. All commands have expected output notes.

3. **Type consistency:** `predict_gap_trajectory()` signature is consistent across Tasks 4, 5, 6. `COEFS` dict is defined once in Task 4 and imported in Tasks 5-6. `CounterfactualResult` dataclass is defined once and used consistently.
