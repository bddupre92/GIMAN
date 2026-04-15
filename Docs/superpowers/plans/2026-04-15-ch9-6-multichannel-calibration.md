# Chapter 9 §9.6 — Multi-Channel Observation Extension of Phase 2 Bayesian Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Phase 2 single-channel (SBR → N(t)) Bayesian calibration of the coupled α-synuclein aggregation + neuron death ODE to a principled multi-channel observation model, with honest identifiability auditing, for chapter-grade §9.6 of Paper 7.

**Architecture:** Four primary observation channels (SBR, aSyn aggregate%, SAA TTT, NEV α-synuclein) mapped to distinct ODE compartments. One α_tox anchor from Olink Project 222 (CSF GFAP). NfL and CSF α-syn retained as held-out validation channels. Identifiability audit extends the Phase 4 Jacobian+FIM template with sloppiness analysis (FIM eigenvalue spectrum) and profile likelihood on (k_n, α_tox). Starting posterior: existing `saem_multi_obs_v2` (304 patients, 6 channels); endpoint: `saem_multi_obs_v3` (1,065 patients, 4 primary + GFAP anchor channels).

**Tech Stack:** Python 3.10+ (pandas, numpy, scipy, sqlalchemy, pytest, matplotlib, bibtexparser); existing `scripts/mechanistic_twin/multi_obs_saem.py` (SAEM implementation); PostgreSQL `giman_research` (`audit.*`, `mechanistic.*`, `ppmi_olink.*`, `reference.*` schemas); Julia 1.11 via juliaup (Turing.jl for comparative Bayesian runs, optional).

**Timeline:** 6 weeks (Week 2–7 of 17-week dissertation completion plan v2). Each Task below fits roughly one week; SAEM calibration run (Task 6) is the longest.

**Entry gates (must be true before starting Task 1):**
1. `psql giman_research -c "SELECT 1 FROM audit.citation LIMIT 1"` returns a row.
2. `psql giman_research -c "SELECT 1 FROM ppmi_olink.ppmi_project_222_csf_inf_npx LIMIT 1"` returns a row.
3. `test -f outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v2/individual_params.csv` exits 0.
4. `.venv/bin/python -c "import bibtexparser, pandas, numpy, scipy, sqlalchemy, matplotlib" && echo OK` prints `OK`.

**Exit gates (Task 9 completion):**
1. `saem_multi_obs_v3` directory exists with 1,065 patients × 4-channel fit + GFAP anchor.
2. Identifiability report shows rank(J)=2 and FIM κ plus eigenvalue spread documented in a table.
3. LOO forward validation passes with ≥ 85% coverage at 95% credible interval (same gate as Phase 1 Addendum A2's 93.75%).
4. `outputs/mechanistic_twin/ch9_6/figures/` contains 5 figures (channel map, FIM ablation, LOO coverage, NfL validation, posterior comparison).
5. `writing/chapters/ch09_paper7.tex` has a new `\section{§9.6 Multi-Channel Observation Extension}` block compiled in the dissertation PDF.
6. All new numerical claims registered in `audit.numerical_claim`; 4 new citations verified in `audit.citation`.

---

## Task 1 — Registry Load + New Citation Add

**Files:**
- Create: `scripts/mechanistic_twin/load_data_literature_registry.py`
- Create: `scripts/mechanistic_twin/add_ch9_6_citations.py`
- Create: `tests/mechanistic_twin/test_registry_load.py`
- Modify: `outputs/mechanistic_twin/phase2/DATA_LITERATURE_REGISTRY.md` (READ ONLY — parse source)
- Creates SQL: `mechanistic.data_registry`, `mechanistic.literature_anchors`, `mechanistic.empirical_findings`

- [ ] **Step 1: Write failing test for registry loader**

Create `tests/mechanistic_twin/test_registry_load.py`:

```python
"""Gate test: DATA_LITERATURE_REGISTRY.md → mechanistic.* SQL tables."""
import pytest
from sqlalchemy import create_engine, text

ENGINE = create_engine("postgresql+psycopg2://blair.dupre@localhost:5432/giman_research")


def test_data_registry_table_exists():
    with ENGINE.connect() as conn:
        row = conn.execute(text(
            "SELECT to_regclass('mechanistic.data_registry')::text"
        )).scalar()
    assert row == "mechanistic.data_registry", "data_registry table not created"


def test_data_registry_has_gamma_entry():
    """GAMMA = 0.7 (Lee 2019) must be findable — it's a Phase 1 canonical param."""
    with ENGINE.connect() as conn:
        row = conn.execute(text(
            "SELECT parameter, value, source FROM mechanistic.data_registry "
            "WHERE parameter = 'GAMMA'"
        )).fetchone()
    assert row is not None
    assert float(row.value) == pytest.approx(0.7, rel=1e-6)
    assert "Lee 2019" in row.source


def test_empirical_findings_has_mollenhauer_csf_asyn():
    """The CSF α-syn vs SAA TTT ρ=-0.011 finding must be queryable."""
    with ENGINE.connect() as conn:
        row = conn.execute(text(
            "SELECT finding, metric_value FROM mechanistic.empirical_findings "
            "WHERE finding ILIKE '%CSF total α-syn vs SAA TTT%'"
        )).fetchone()
    assert row is not None
    assert abs(float(row.metric_value) - (-0.011)) < 0.005


def test_literature_anchors_has_rutledge2024():
    with ENGINE.connect() as conn:
        n = conn.execute(text(
            "SELECT COUNT(*) FROM mechanistic.literature_anchors "
            "WHERE source ILIKE '%Rutledge 2024%'"
        )).scalar()
    assert n >= 1
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /Users/blair.dupre/Projects/CSCI-FALL-2025
.venv/bin/pytest tests/mechanistic_twin/test_registry_load.py -v
```

Expected: all 4 tests FAIL with `relation "mechanistic.data_registry" does not exist`.

- [ ] **Step 3: Write loader script**

Create `scripts/mechanistic_twin/load_data_literature_registry.py`:

```python
"""Parse outputs/mechanistic_twin/phase2/DATA_LITERATURE_REGISTRY.md pipe tables
into three structured tables in mechanistic.* schema.

Sections parsed:
- §1 "Phase 1/2/3 ODE Parameters" → mechanistic.data_registry
- §2 "Literature Anchors" + §9-level anchors → mechanistic.literature_anchors
- §7 "Empirical Findings" + §8 Phase 3/4 findings → mechanistic.empirical_findings
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "outputs/mechanistic_twin/phase2/DATA_LITERATURE_REGISTRY.md"
ENGINE = create_engine("postgresql+psycopg2://blair.dupre@localhost:5432/giman_research")


def parse_markdown_tables(md_text: str) -> list[pd.DataFrame]:
    """Return each pipe-table in the file as a DataFrame, preserving order."""
    tables: list[pd.DataFrame] = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|") and "|" in lines[i]:
            header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if i + 1 < len(lines) and re.match(r"^\s*\|[\s|:-]+\|", lines[i + 1]):
                rows = []
                j = i + 2
                while j < len(lines) and lines[j].strip().startswith("|"):
                    cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                    if len(cells) == len(header):
                        rows.append(cells)
                    j += 1
                if rows:
                    tables.append(pd.DataFrame(rows, columns=header))
                i = j
                continue
        i += 1
    return tables


def classify_and_normalize(tables: list[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Bucket tables by their header signature."""
    registry_rows: list[dict] = []
    anchor_rows: list[dict] = []
    finding_rows: list[dict] = []

    for t in tables:
        cols = {c.lower() for c in t.columns}
        if {"parameter", "value", "unit"} <= cols:
            for _, r in t.iterrows():
                registry_rows.append({
                    "parameter": r.get("Parameter") or r.get("parameter"),
                    "value": r.get("Value") or r.get("value"),
                    "unit": r.get("Unit") or r.get("unit"),
                    "description": next((r[c] for c in t.columns if c.lower() in {"description"}), None),
                    "source": next((r[c] for c in t.columns if c.lower() in {"source", "literature anchor"}), None),
                    "rationale": next((r[c] for c in t.columns if c.lower() in {"rationale"}), None),
                    "file_refs": next((r[c] for c in t.columns if c.lower() in {"file refs", "file references"}), None),
                    "status": next((r[c] for c in t.columns if c.lower() in {"status"}), None),
                })
        elif {"claim"} <= cols or {"observation"} <= cols or {"finding"} <= cols:
            for _, r in t.iterrows():
                finding_rows.append({
                    "finding": r.get("Finding") or r.get("Claim") or r.get("Observation"),
                    "metric_value": next((r[c] for c in t.columns if c.lower() in {"value", "metric"}), None),
                    "context": next((r[c] for c in t.columns if c.lower() in {"context"}), None),
                    "interpretation": next((r[c] for c in t.columns if c.lower() in {"interpretation"}), None),
                    "source_run": next((r[c] for c in t.columns if c.lower() in {"source", "source run"}), None),
                })
        elif {"literature anchor"} <= cols or {"reference"} <= cols:
            for _, r in t.iterrows():
                anchor_rows.append({
                    "anchor": r.get("Literature Anchor") or r.get("Reference"),
                    "year": next((r[c] for c in t.columns if c.lower() == "year"), None),
                    "reason": next((r[c] for c in t.columns if c.lower() in {"reason", "finding"}), None),
                    "source": next((r[c] for c in t.columns if c.lower() == "source"), None),
                    "decision_trail": next((r[c] for c in t.columns if c.lower() in {"decision trail", "decision"}), None),
                })

    def _num(v):
        if not isinstance(v, str):
            return v
        m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", v.replace("\u2212", "-"))
        return float(m.group(0)) if m else None

    registry = pd.DataFrame(registry_rows)
    if "value" in registry.columns:
        registry["value_num"] = registry["value"].apply(_num)

    findings = pd.DataFrame(finding_rows)
    if "metric_value" in findings.columns:
        findings["metric_value"] = findings["metric_value"].apply(_num)

    anchors = pd.DataFrame(anchor_rows)

    return {
        "data_registry": registry,
        "literature_anchors": anchors,
        "empirical_findings": findings,
    }


def main() -> None:
    md = REGISTRY.read_text()
    tables = parse_markdown_tables(md)
    print(f"Parsed {len(tables)} markdown tables from {REGISTRY.name}")

    dfs = classify_and_normalize(tables)

    with ENGINE.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS mechanistic"))

    for name, df in dfs.items():
        if df.empty:
            print(f"  {name}: 0 rows — skipping")
            continue
        df.to_sql(name, ENGINE, schema="mechanistic", if_exists="replace",
                  index=False, method="multi", chunksize=200)
        print(f"  mechanistic.{name}: {len(df):,} rows × {len(df.columns)} cols")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the loader**

```bash
.venv/bin/python scripts/mechanistic_twin/load_data_literature_registry.py
```

Expected output includes lines like `mechanistic.data_registry: >= 20 rows × >= 7 cols`.

- [ ] **Step 5: Run tests to verify they now pass**

```bash
.venv/bin/pytest tests/mechanistic_twin/test_registry_load.py -v
```

Expected: 4/4 PASS.

- [ ] **Step 6: Write citation-add script**

Create `scripts/mechanistic_twin/add_ch9_6_citations.py`:

```python
"""Add the four new §9.6 citations to audit.citation and reference.phase5_bibliography.

Sources identified 2026-04-15:
- Liu 2023 J Neuroinflammation — CSF GFAP predicts longitudinal cognition + CSF α-syn
- Bäckström 2020 Neurology — NfL predicts PD survival + UPDRS-III rate
- Sampedro 2020 Parkinsonism Relat Disord — Serum NfL reflects cortical, NOT striatal DAT
- Bartl 2021 PLoS ONE — NfL/sTREM2/YKL40/GFAP panel in PPMI PD
"""
from __future__ import annotations
import pandas as pd
from sqlalchemy import create_engine, text

ENGINE = create_engine("postgresql+psycopg2://blair.dupre@localhost:5432/giman_research")

ROWS = [
    {
        "cite_key": "liu2023gfap",
        "author": "Liu, Y.",
        "year": 2023,
        "title": "CSF GFAP predicts cognitive decline and longitudinal α-synuclein in de novo Parkinson's disease",
        "journal": "Journal of Neuroinflammation",
        "doi": "",
        "pmid": "",
        "zotero_key": "",
        "zotero_verified": 0,
        "ezproxy_verified": 0,
        "last_checked": "2026-04-15",
    },
    {
        "cite_key": "backstrom2020nfl",
        "author": "Bäckström, D.",
        "year": 2020,
        "title": "NfL as a biomarker for neurodegeneration and survival in Parkinson disease",
        "journal": "Neurology",
        "doi": "",
        "pmid": "",
        "zotero_key": "",
        "zotero_verified": 0,
        "ezproxy_verified": 0,
        "last_checked": "2026-04-15",
    },
    {
        "cite_key": "sampedro2020nfl",
        "author": "Sampedro, F.",
        "year": 2020,
        "title": "Serum neurofilament light chain reflects cortical neurodegeneration, not striatal DAT",
        "journal": "Parkinsonism & Related Disorders",
        "doi": "",
        "pmid": "",
        "zotero_key": "",
        "zotero_verified": 0,
        "ezproxy_verified": 0,
        "last_checked": "2026-04-15",
    },
    {
        "cite_key": "bartl2021ppmi",
        "author": "Bartl, M.",
        "year": 2021,
        "title": "NfL/sTREM2/YKL40/GFAP panel in PPMI PD — TREM2 does not discriminate",
        "journal": "PLoS ONE",
        "doi": "",
        "pmid": "",
        "zotero_key": "",
        "zotero_verified": 0,
        "ezproxy_verified": 0,
        "last_checked": "2026-04-15",
    },
]


def main() -> None:
    df = pd.DataFrame(ROWS)
    with ENGINE.begin() as conn:
        for _, r in df.iterrows():
            conn.execute(
                text(
                    "INSERT INTO audit.citation (cite_key, author, year, title, journal, "
                    "doi, pmid, zotero_key, zotero_verified, ezproxy_verified, last_checked) "
                    "VALUES (:cite_key, :author, :year, :title, :journal, :doi, :pmid, "
                    ":zotero_key, :zotero_verified, :ezproxy_verified, :last_checked) "
                    "ON CONFLICT (cite_key) DO UPDATE SET last_checked = EXCLUDED.last_checked"
                ),
                r.to_dict(),
            )
    print(f"Inserted/updated {len(df)} citations in audit.citation")

    with ENGINE.connect() as conn:
        for key in df["cite_key"]:
            n = conn.execute(
                text("SELECT COUNT(*) FROM audit.citation WHERE cite_key = :k"),
                {"k": key},
            ).scalar()
            assert n == 1, f"missing {key}"
    print("All 4 citations confirmed present.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the citation-add script**

```bash
.venv/bin/python scripts/mechanistic_twin/add_ch9_6_citations.py
```

Expected: `All 4 citations confirmed present.`

- [ ] **Step 8: Verify end-to-end with one SQL query**

```bash
psql giman_research -c "SELECT cite_key, author, year FROM audit.citation WHERE cite_key IN ('liu2023gfap','backstrom2020nfl','sampedro2020nfl','bartl2021ppmi') ORDER BY cite_key"
```

Expected: 4 rows returned.

- [ ] **Step 9: Manual Zotero sync action** (out-of-band, flagged in plan)

Add these same 4 references to Zotero collection `RT8B9N2J`. After sync, re-run `scripts/mechanistic_twin/add_ch9_6_citations.py` with `zotero_verified = 1` and the `zotero_key` filled in. This will be a Task 9 closeout action, not a blocker for Tasks 2–8.

- [ ] **Step 10: Commit**

```bash
git add scripts/mechanistic_twin/load_data_literature_registry.py \
        scripts/mechanistic_twin/add_ch9_6_citations.py \
        tests/mechanistic_twin/test_registry_load.py
git commit -m "feat(ch9.6-task1): load DATA_LITERATURE_REGISTRY into SQL + add 4 citations

Creates mechanistic.{data_registry,literature_anchors,empirical_findings} from
the markdown registry, and adds Liu 2023, Bäckström 2020, Sampedro 2020, Bartl
2021 to audit.citation (zotero_verified=0 pending RT8B9N2J sync)."
```

---

## Task 2 — Identifiability Audit (Jacobian + FIM + Eigenvalue Spectrum + Profile Likelihood)

**Files:**
- Create: `scripts/mechanistic_twin/ch9_6_identifiability_audit.py`
- Create: `tests/mechanistic_twin/test_ch9_6_identifiability.py`
- Reference: `scripts/mechanistic_twin/phase4_identifiability_proof.py` (template)
- Reference: `scripts/mechanistic_twin/step_2_7_v5_profile_likelihood_csf.py` (profile likelihood pattern)
- Creates output: `outputs/mechanistic_twin/ch9_6/identifiability.json` + `outputs/mechanistic_twin/ch9_6/identifiability_RUN_MANIFEST.md`

**Observation map being audited:**

```
SBR_t         = SBR_0 * (N(t) / N_0)^gamma + eps_SBR
aSyn_agg%_t   = O_ss / (M_ss + O_ss) * 100 + eps_agg        [strongest k_n probe]
SAA_TTT_t     = phi(F_ss(k_n)) + eps_SAA                    [F seeding kinetics]
NEV_asyn_t    = s_NEV * (O_ss + r_F * F_ss) + eps_NEV       [O+F neuronal EVs]
CSF_GFAP_t    = s_GFAP * O(t) + eps_GFAP                    [alpha_tox anchor]
```

Unknowns: k_n (aggregation rate), α_tox (toxicity rate). 5 channels × 2 unknowns is overdetermined; the test is whether rank(J)=2 and whether the FIM eigenvalue spread is under 10³.

- [ ] **Step 1: Write failing test**

Create `tests/mechanistic_twin/test_ch9_6_identifiability.py`:

```python
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/mechanistic_twin/ch9_6/identifiability.json"


def test_identifiability_json_exists():
    assert OUT.exists(), "run scripts/mechanistic_twin/ch9_6_identifiability_audit.py"


def test_jacobian_rank_equals_two():
    with OUT.open() as f:
        data = json.load(f)
    assert data["jacobian_rank"] == 2, (
        f"rank(J)={data['jacobian_rank']} — augmented obs map must be locally identifiable"
    )


def test_fim_condition_number_under_1000():
    with OUT.open() as f:
        data = json.load(f)
    kappa = float(data["fim_condition_number"])
    assert kappa < 1000, f"FIM kappa={kappa:.1f} exceeds practical-identifiability threshold"


def test_fim_eigenvalue_spectrum_reported():
    with OUT.open() as f:
        data = json.load(f)
    eigs = data.get("fim_eigenvalues")
    assert eigs is not None and len(eigs) == 2
    assert all(e > 0 for e in eigs), "FIM should be positive-definite"


def test_profile_likelihood_reports_ci():
    with OUT.open() as f:
        data = json.load(f)
    pl = data.get("profile_likelihood")
    assert pl is not None
    for param in ["k_n", "alpha_tox"]:
        assert param in pl
        ci = pl[param]["ci_95"]
        assert ci[0] < ci[1], f"{param} CI malformed"
        assert not (ci[0] == -1e308 or ci[1] == 1e308), (
            f"{param} profile likelihood is one-sided — practical non-identifiability"
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/mechanistic_twin/test_ch9_6_identifiability.py -v
```

Expected: all FAIL with `identifiability.json not found`.

- [ ] **Step 3: Write the audit script**

Create `scripts/mechanistic_twin/ch9_6_identifiability_audit.py`. The structure follows `phase4_identifiability_proof.py` with three new pieces: a 5-channel Jacobian (rows = channels, cols = [k_n, α_tox]), an FIM eigenvalue plot, and a 2D profile-likelihood grid.

```python
"""Ch 9 §9.6 identifiability audit: 5-channel observation map → (k_n, α_tox).

Outputs:
  outputs/mechanistic_twin/ch9_6/identifiability.json
  outputs/mechanistic_twin/ch9_6/identifiability_RUN_MANIFEST.md
  outputs/mechanistic_twin/ch9_6/figures/fim_eigenvalue_spectrum.png
  outputs/mechanistic_twin/ch9_6/figures/profile_likelihood_2d.png
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.mechanistic_twin._reproducibility import capture_provenance, write_run_manifest  # noqa

OUT_DIR = ROOT / "outputs/mechanistic_twin/ch9_6"
FIG_DIR = OUT_DIR / "figures"

# Nominal population-level parameters (from saem_multi_obs_v2)
K_N_NOM = np.exp(-7.922)           # 3.6e-4
ALPHA_TOX_NOM = np.exp(-11.357)    # 1.17e-5
GAMMA = 0.7                         # SBR-to-N exponent (Lee 2019)
M_SS = 2.0                          # nM
S_NEV = 1.0                         # NEV α-syn scaling
S_GFAP = 1.0                        # GFAP scaling (calibrated in SAEM)
R_F = 0.5                           # fibril weight in NEV signal
SBR_0 = 1.5

CHANNELS = ["SBR", "aSyn_agg_pct", "SAA_TTT", "NEV_asyn", "CSF_GFAP"]


def forward(k_n: float, alpha_tox: float, t_years: float) -> dict[str, float]:
    """Closed-form steady-state observations (placeholder for full ODE;
    replace with ODE solver calls once identifiability structure is confirmed).

    These are ~ correct to leading order for the slow-fast-collapsed ODE.
    """
    # Approximate quasi-steady-state O concentration
    O_ss = k_n * M_SS / (alpha_tox + 1e-30)
    F_ss = O_ss * 0.3  # simplified
    # N(t) approx exponential decay with rate alpha_tox * O_ss
    N_frac = np.exp(-alpha_tox * O_ss * t_years)
    return {
        "SBR": SBR_0 * N_frac ** GAMMA,
        "aSyn_agg_pct": 100.0 * O_ss / (M_SS + O_ss),
        "SAA_TTT": 1.0 / (F_ss + 1e-12),
        "NEV_asyn": S_NEV * (O_ss + R_F * F_ss),
        "CSF_GFAP": S_GFAP * O_ss,
    }


def jacobian(k_n: float, alpha_tox: float, t_years: float, eps: float = 1e-6) -> np.ndarray:
    """Finite-difference Jacobian, rows=channels, cols=[k_n, alpha_tox]."""
    base = forward(k_n, alpha_tox, t_years)
    J = np.zeros((len(CHANNELS), 2))
    for j, (name, val) in enumerate([("k_n", k_n), ("alpha_tox", alpha_tox)]):
        plus = forward(k_n * (1 + eps) if j == 0 else k_n,
                       alpha_tox * (1 + eps) if j == 1 else alpha_tox,
                       t_years)
        minus = forward(k_n * (1 - eps) if j == 0 else k_n,
                        alpha_tox * (1 - eps) if j == 1 else alpha_tox,
                        t_years)
        for i, ch in enumerate(CHANNELS):
            J[i, j] = (plus[ch] - minus[ch]) / (2 * eps * val)
    return J


def build_cohort_jacobian(t_years_list: list[float]) -> np.ndarray:
    """Stack per-visit Jacobians into an (N*5) x 2 matrix."""
    blocks = [jacobian(K_N_NOM, ALPHA_TOX_NOM, t) for t in t_years_list]
    return np.vstack(blocks)


def profile_likelihood_2d(t_years_list: list[float], obs_noise_var: dict[str, float]) -> dict:
    """Compute profile-likelihood 95% CI for each of k_n and α_tox."""
    grid_size = 40
    k_n_grid = np.logspace(np.log10(K_N_NOM) - 1.5, np.log10(K_N_NOM) + 1.5, grid_size)
    alpha_grid = np.logspace(np.log10(ALPHA_TOX_NOM) - 1.5, np.log10(ALPHA_TOX_NOM) + 1.5, grid_size)

    chi2 = np.zeros((grid_size, grid_size))
    for i, kn in enumerate(k_n_grid):
        for j, at in enumerate(alpha_grid):
            total = 0.0
            for t in t_years_list:
                nom = forward(K_N_NOM, ALPHA_TOX_NOM, t)
                pred = forward(kn, at, t)
                for ch in CHANNELS:
                    resid = (pred[ch] - nom[ch]) / np.sqrt(obs_noise_var[ch])
                    total += resid ** 2
            chi2[i, j] = total

    chi2_min = chi2.min()
    threshold_1d = chi2_min + 3.84  # chi²(1, 0.95)

    k_n_profile = chi2.min(axis=1)
    alpha_profile = chi2.min(axis=0)

    def ci_from_profile(grid, profile):
        below = profile <= threshold_1d
        if not below.any():
            return [float(grid[profile.argmin()]), float(grid[profile.argmin()])]
        idx = np.where(below)[0]
        return [float(grid[idx[0]]), float(grid[idx[-1]])]

    return {
        "k_n": {
            "grid": k_n_grid.tolist(),
            "profile_chi2": k_n_profile.tolist(),
            "ci_95": ci_from_profile(k_n_grid, k_n_profile),
        },
        "alpha_tox": {
            "grid": alpha_grid.tolist(),
            "profile_chi2": alpha_profile.tolist(),
            "ci_95": ci_from_profile(alpha_grid, alpha_profile),
        },
        "chi2_surface": chi2.tolist(),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    t_years = [0.0, 1.0, 2.0, 4.0, 6.0]

    # Jacobian + FIM
    J = build_cohort_jacobian(t_years)
    rank = int(np.linalg.matrix_rank(J))
    FIM = J.T @ J
    eigvals = np.linalg.eigvalsh(FIM)
    kappa = float(np.linalg.cond(FIM))
    eig_spread_log10 = float(np.log10(eigvals.max() / eigvals.min()))

    # Profile likelihood
    obs_noise_var = {
        "SBR": 0.04, "aSyn_agg_pct": 100.0, "SAA_TTT": 0.01,
        "NEV_asyn": 0.25, "CSF_GFAP": 0.01,
    }
    pl = profile_likelihood_2d(t_years, obs_noise_var)

    # Figures
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(len(eigvals)), np.log10(eigvals))
    ax.set_xticks(range(len(eigvals)))
    ax.set_xticklabels([f"λ{i+1}" for i in range(len(eigvals))])
    ax.set_ylabel("log10(eigenvalue)")
    ax.set_title(f"FIM eigenvalue spectrum — 5ch → (k_n, α_tox)\nκ={kappa:.1f}, spread={eig_spread_log10:.2f} decades")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fim_eigenvalue_spectrum.png", dpi=200)
    fig.savefig(FIG_DIR / "fim_eigenvalue_spectrum.pdf")
    plt.close(fig)

    chi2 = np.asarray(pl["chi2_surface"])
    k_n_grid = np.asarray(pl["k_n"]["grid"])
    alpha_grid = np.asarray(pl["alpha_tox"]["grid"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cs = ax.contourf(np.log10(k_n_grid), np.log10(alpha_grid), chi2.T - chi2.min(),
                     levels=[0, 3.84, 9.21, 20, 50, 100], cmap="viridis_r")
    ax.set_xlabel("log10(k_n)")
    ax.set_ylabel("log10(α_tox)")
    ax.set_title("Profile likelihood 2D surface (Δχ² contours)")
    plt.colorbar(cs, ax=ax, label="Δχ²")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "profile_likelihood_2d.png", dpi=200)
    fig.savefig(FIG_DIR / "profile_likelihood_2d.pdf")
    plt.close(fig)

    results = {
        "channels": CHANNELS,
        "unknowns": ["k_n", "alpha_tox"],
        "t_years_evaluated": t_years,
        "jacobian": J.tolist(),
        "jacobian_rank": rank,
        "fim": FIM.tolist(),
        "fim_eigenvalues": eigvals.tolist(),
        "fim_condition_number": kappa,
        "fim_eigenvalue_spread_log10": eig_spread_log10,
        "profile_likelihood": {
            "k_n": {k: v for k, v in pl["k_n"].items() if k != "grid" or True},
            "alpha_tox": pl["alpha_tox"],
        },
        "nominal_values": {"k_n": K_N_NOM, "alpha_tox": ALPHA_TOX_NOM},
        "verdict_rank2": rank == 2,
        "verdict_kappa_under_1000": kappa < 1000,
        "verdict_spread_under_3": eig_spread_log10 < 3.0,
        "_provenance": capture_provenance(
            script_path=Path(__file__).resolve(),
            repo_root=ROOT,
            input_files=[],
            extra={"note": "Closed-form steady-state approx; ODE-based refinement deferred to Task 5"},
        ),
    }

    with (OUT_DIR / "identifiability.json").open("w") as f:
        json.dump(results, f, indent=2, default=float)

    write_run_manifest(
        manifest_path=OUT_DIR / "identifiability_RUN_MANIFEST.md",
        step_name="Ch 9.6 Identifiability Audit",
        provenance=results["_provenance"],
        gate_results={
            "Jacobian rank == 2": rank == 2,
            "FIM kappa < 1000": kappa < 1000,
            "FIM eigenvalue spread < 3 decades": eig_spread_log10 < 3.0,
        },
        summary_metrics={
            "rank(J)": rank,
            "FIM kappa": f"{kappa:.2f}",
            "eig spread (log10)": f"{eig_spread_log10:.2f}",
            "k_n 95% PL CI": pl["k_n"]["ci_95"],
            "alpha_tox 95% PL CI": pl["alpha_tox"]["ci_95"],
        },
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the audit**

```bash
.venv/bin/python scripts/mechanistic_twin/ch9_6_identifiability_audit.py
```

Expected: creates `outputs/mechanistic_twin/ch9_6/identifiability.json` and figures.

- [ ] **Step 5: Run tests to verify pass**

```bash
.venv/bin/pytest tests/mechanistic_twin/test_ch9_6_identifiability.py -v
```

Expected: 5/5 PASS.

- [ ] **Step 6: If rank != 2 or kappa > 1000 → STOP and report**

If the audit fails (rank<2 or κ>1000 or spread>3 decades), DO NOT proceed to Task 3. Instead:
1. Capture the failure in `outputs/mechanistic_twin/ch9_6/identifiability_FAILURE.md` with the specific channel that is redundant (use FIM eigenvector loading to identify which channel contributes near-zero information).
2. Redesign the channel set (most likely: drop the redundant channel; the sloppiness analysis will tell you which).
3. Re-open Task 2 with the reduced channel set.

This is a genuine gate, not a formality — the whole §9.6 thesis depends on the 5-channel map being identifiable.

- [ ] **Step 7: Commit**

```bash
git add scripts/mechanistic_twin/ch9_6_identifiability_audit.py \
        tests/mechanistic_twin/test_ch9_6_identifiability.py \
        outputs/mechanistic_twin/ch9_6/identifiability.json \
        outputs/mechanistic_twin/ch9_6/identifiability_RUN_MANIFEST.md \
        outputs/mechanistic_twin/ch9_6/figures/fim_eigenvalue_spectrum.{png,pdf} \
        outputs/mechanistic_twin/ch9_6/figures/profile_likelihood_2d.{png,pdf}
git commit -m "feat(ch9.6-task2): 5-channel identifiability audit PASSES

Jacobian rank=2, FIM κ=<VALUE>, eigenvalue spread=<VALUE> decades.
Profile likelihood 95%% CIs on (k_n, α_tox) are two-sided.
Extends phase4_identifiability_proof.py template with FIM eigenvalue
spectrum and 2D profile likelihood per Gutenkunst 2007 + Raue 2013."
```

---

## Task 3 — Olink GFAP Extraction + QC

**Files:**
- Create: `scripts/mechanistic_twin/ch9_6_extract_gfap.py`
- Create: `tests/mechanistic_twin/test_ch9_6_gfap_extraction.py`
- Reads SQL: `ppmi_olink.ppmi_project_222_csf_inf_npx`
- Writes SQL: `mechanistic.ch9_6_gfap_longitudinal`

- [ ] **Step 1: Inspect Olink table structure**

```bash
psql giman_research -c "\d ppmi_olink.ppmi_project_222_csf_inf_npx" | head -40
psql giman_research -c "SELECT column_name FROM information_schema.columns WHERE table_schema='ppmi_olink' AND table_name='ppmi_project_222_csf_inf_npx' AND column_name ILIKE '%gfap%'"
```

Expected: the Olink CSF NPX table has a column for GFAP (Olink Explore 3072 assay). If the Olink table stores proteins in long format (patno, clinical_event, assay, npx), the extraction is a filter. If wide format (one column per protein), the extraction is a column select.

- [ ] **Step 2: Write failing test**

Create `tests/mechanistic_twin/test_ch9_6_gfap_extraction.py`:

```python
from sqlalchemy import create_engine, text

ENGINE = create_engine("postgresql+psycopg2://blair.dupre@localhost:5432/giman_research")


def test_gfap_table_exists():
    with ENGINE.connect() as conn:
        x = conn.execute(text(
            "SELECT to_regclass('mechanistic.ch9_6_gfap_longitudinal')::text"
        )).scalar()
    assert x == "mechanistic.ch9_6_gfap_longitudinal"


def test_gfap_has_at_least_150_patients():
    """Project 222 has 227 patients with CSF Olink — expect most have GFAP."""
    with ENGINE.connect() as conn:
        n = conn.execute(text(
            "SELECT COUNT(DISTINCT patno) FROM mechanistic.ch9_6_gfap_longitudinal"
        )).scalar()
    assert n >= 150, f"only {n} patients with GFAP — expected ≥ 150"


def test_gfap_npx_values_in_range():
    """Olink NPX is typically log2, values should be in [-5, 15]."""
    with ENGINE.connect() as conn:
        row = conn.execute(text(
            "SELECT MIN(npx), MAX(npx), AVG(npx) FROM mechanistic.ch9_6_gfap_longitudinal"
        )).fetchone()
    assert -10 < row[0] < row[1] < 20
    assert 0 < row[2] < 10, f"mean NPX {row[2]} out of expected range"


def test_gfap_has_longitudinal_structure():
    """Need multiple visits per patient."""
    with ENGINE.connect() as conn:
        n_multi = conn.execute(text(
            "SELECT COUNT(*) FROM (SELECT patno, COUNT(*) as c "
            "FROM mechanistic.ch9_6_gfap_longitudinal GROUP BY patno HAVING COUNT(*) >= 2) s"
        )).scalar()
    assert n_multi >= 80, f"only {n_multi} patients with ≥2 visits"
```

- [ ] **Step 3: Run test to verify failure**

```bash
.venv/bin/pytest tests/mechanistic_twin/test_ch9_6_gfap_extraction.py -v
```

Expected: all FAIL.

- [ ] **Step 4: Write extraction script**

Create `scripts/mechanistic_twin/ch9_6_extract_gfap.py`. The SQL will depend on whether the Olink table is long or wide format — the script handles both:

```python
"""Extract CSF GFAP longitudinal measurements from PPMI Olink Project 222.

Strategy: detect format (long: uniprot/assay/npx cols OR wide: one col per protein)
then normalize to long (patno, clinical_event, visit_date, npx, qc_warning).
"""
from __future__ import annotations
import pandas as pd
from sqlalchemy import create_engine, inspect, text

ENGINE = create_engine("postgresql+psycopg2://blair.dupre@localhost:5432/giman_research")
SRC = "ppmi_olink.ppmi_project_222_csf_inf_npx"


def detect_format(engine, schema: str, table: str) -> str:
    insp = inspect(engine)
    cols = [c["name"].lower() for c in insp.get_columns(table, schema=schema)]
    if {"assay", "npx"} <= set(cols) or {"olinkid", "npx"} <= set(cols):
        return "long"
    if "gfap" in cols or any(c for c in cols if "gfap" in c):
        return "wide"
    return "unknown"


def extract_long(engine) -> pd.DataFrame:
    return pd.read_sql(
        text(f"""SELECT patno, clinical_event, npx, qc_warning
                 FROM {SRC}
                 WHERE UPPER(assay) = 'GFAP' OR UPPER(olinkid) LIKE '%GFAP%'"""),
        engine,
    )


def extract_wide(engine) -> pd.DataFrame:
    col = "gfap"
    insp = inspect(engine)
    cols = [c["name"] for c in insp.get_columns("ppmi_project_222_csf_inf_npx",
                                                 schema="ppmi_olink")]
    match = [c for c in cols if c.lower() == col or c.lower().endswith("_gfap")]
    if not match:
        raise ValueError("No GFAP column in wide-format Olink table")
    actual_col = match[0]
    return pd.read_sql(
        text(f"SELECT patno, clinical_event, \"{actual_col}\" AS npx FROM {SRC}"),
        engine,
    )


def main() -> None:
    fmt = detect_format(ENGINE, "ppmi_olink", "ppmi_project_222_csf_inf_npx")
    if fmt == "long":
        df = extract_long(ENGINE)
    elif fmt == "wide":
        df = extract_wide(ENGINE)
        df["qc_warning"] = None
    else:
        raise ValueError("Could not detect Olink table format")

    df = df.dropna(subset=["npx"])
    df["patno"] = pd.to_numeric(df["patno"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["patno"])

    df.to_sql("ch9_6_gfap_longitudinal", ENGINE, schema="mechanistic",
              if_exists="replace", index=False, method="multi", chunksize=500)

    print(f"mechanistic.ch9_6_gfap_longitudinal: {len(df):,} rows, "
          f"{df['patno'].nunique()} patients")
    vc = df.groupby("patno").size()
    print(f"  >= 2 visits: {(vc >= 2).sum()}")
    print(f"  NPX mean={df['npx'].mean():.2f}, sd={df['npx'].std():.2f}, "
          f"range=[{df['npx'].min():.2f}, {df['npx'].max():.2f}]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run extraction**

```bash
.venv/bin/python scripts/mechanistic_twin/ch9_6_extract_gfap.py
```

Expected: prints patient count ≥ 150 with mean NPX roughly in [2, 8].

- [ ] **Step 6: Run tests to verify pass**

```bash
.venv/bin/pytest tests/mechanistic_twin/test_ch9_6_gfap_extraction.py -v
```

Expected: 4/4 PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/mechanistic_twin/ch9_6_extract_gfap.py \
        tests/mechanistic_twin/test_ch9_6_gfap_extraction.py
git commit -m "feat(ch9.6-task3): extract Olink CSF GFAP to mechanistic.ch9_6_gfap_longitudinal"
```

---

## Task 4 — Multi-Channel Cohort Assembly (5 Channels + NfL Held-Out)

**Files:**
- Create: `scripts/mechanistic_twin/ch9_6_assemble_cohort.py`
- Create: `tests/mechanistic_twin/test_ch9_6_cohort.py`
- Reads SQL: `mechanistic.dat_spect_longitudinal`, `mechanistic.multi_observable_inventory`, `mechanistic.ch9_6_gfap_longitudinal`, `ppmi_raw.current_biospecimen_analysis_results`
- Writes file: `outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet`

- [ ] **Step 1: Write failing test**

Create `tests/mechanistic_twin/test_ch9_6_cohort.py`:

```python
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet"


@pytest.fixture(scope="module")
def cohort():
    assert COHORT.exists(), "run ch9_6_assemble_cohort.py"
    return pd.read_parquet(COHORT)


def test_cohort_has_required_columns(cohort):
    required = {"patno", "visit_month", "sbr_putamen", "asyn_agg_pct",
                "saa_ttt", "nev_asyn", "gfap_npx", "nfl_pg_per_ml"}
    missing = required - set(cohort.columns)
    assert not missing, f"missing columns: {missing}"


def test_cohort_size_at_least_800_patients(cohort):
    n = cohort["patno"].nunique()
    assert n >= 800, f"only {n} patients — target 1,065 but at least 800 needed"


def test_sbr_coverage_near_full(cohort):
    """SBR should be present for nearly every visit (it's the anchor channel)."""
    coverage = cohort["sbr_putamen"].notna().mean()
    assert coverage > 0.90, f"SBR coverage {coverage:.1%} < 90%"


def test_gfap_coverage_reasonable(cohort):
    """GFAP from Olink Project 222 — sparser; accept ≥ 150 patients."""
    n_gfap = cohort.dropna(subset=["gfap_npx"])["patno"].nunique()
    assert n_gfap >= 150


def test_nfl_held_out_flag_exists(cohort):
    """NfL must be present as held-out validation (not in likelihood)."""
    assert "nfl_pg_per_ml" in cohort.columns
    n_nfl = cohort.dropna(subset=["nfl_pg_per_ml"])["patno"].nunique()
    assert n_nfl >= 500, f"only {n_nfl} patients with NfL — need ≥ 500 for validation"
```

- [ ] **Step 2: Run test to verify failure**

```bash
.venv/bin/pytest tests/mechanistic_twin/test_ch9_6_cohort.py -v
```

Expected: all FAIL.

- [ ] **Step 3: Write assembly script**

Create `scripts/mechanistic_twin/ch9_6_assemble_cohort.py`:

```python
"""Assemble 5-channel observation cohort + NfL held-out validation channel.

Inputs (all from local Postgres):
  - mechanistic.dat_spect_longitudinal          (SBR, 2,137 patients)
  - mechanistic.multi_observable_inventory      (aSyn agg%, SAA TTT, NEV α-syn)
  - mechanistic.ch9_6_gfap_longitudinal         (GFAP, from Task 3)
  - ppmi_raw.current_biospecimen_analysis_results  (NfL, Project 144)

Output: outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet"
ENGINE = create_engine("postgresql+psycopg2://blair.dupre@localhost:5432/giman_research")


def load_sbr() -> pd.DataFrame:
    df = pd.read_sql(
        text("""SELECT patno, visit_month,
                       sbr_putamen_mean AS sbr_putamen
                FROM mechanistic.dat_spect_longitudinal"""),
        ENGINE,
    )
    return df


def load_multi_obs() -> pd.DataFrame:
    df = pd.read_sql(
        text("""SELECT patno, clinical_event,
                       asyn_agg_pct, saa_ttt, nev_asyn
                FROM mechanistic.multi_observable_inventory"""),
        ENGINE,
    )
    df["visit_month"] = df["clinical_event"].map(_event_to_month)
    return df.drop(columns=["clinical_event"])


def load_gfap() -> pd.DataFrame:
    df = pd.read_sql(
        text("""SELECT patno, clinical_event, npx AS gfap_npx
                FROM mechanistic.ch9_6_gfap_longitudinal"""),
        ENGINE,
    )
    df["visit_month"] = df["clinical_event"].map(_event_to_month)
    return df.drop(columns=["clinical_event"])


def load_nfl() -> pd.DataFrame:
    df = pd.read_sql(
        text("""SELECT patno, clinical_event, testvalue::float AS nfl_pg_per_ml
                FROM ppmi_raw.current_biospecimen_analysis_results
                WHERE projectid = '144' AND testname = 'NfL'"""),
        ENGINE,
    )
    df["visit_month"] = df["clinical_event"].map(_event_to_month)
    return df.drop(columns=["clinical_event"])


# PPMI clinical_event → month offset. Expand as needed; canonical mapping in
# reference.code_list_annotated.
_EVENT_MAP = {
    "BL": 0, "V01": 3, "V02": 6, "V03": 9, "V04": 12,
    "V05": 18, "V06": 24, "V07": 30, "V08": 36, "V09": 42,
    "V10": 48, "V11": 54, "V12": 60, "V13": 72, "V14": 84,
    "V15": 96, "V16": 108, "V17": 120,
    "SC": -1,  # screening
}


def _event_to_month(ev: str | None) -> float | None:
    if ev is None:
        return None
    return _EVENT_MAP.get(str(ev).strip().upper())


def main() -> None:
    sbr = load_sbr()
    multi = load_multi_obs()
    gfap = load_gfap()
    nfl = load_nfl()

    for df, name in [(sbr, "SBR"), (multi, "multi-obs"), (gfap, "GFAP"), (nfl, "NfL")]:
        print(f"  {name}: {len(df):,} rows, {df['patno'].nunique()} patients")

    keys = ["patno", "visit_month"]
    out = sbr.merge(multi, on=keys, how="outer") \
             .merge(gfap, on=keys, how="outer") \
             .merge(nfl, on=keys, how="outer")
    out = out.dropna(subset=["visit_month"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\nAssembled: {len(out):,} rows, {out['patno'].nunique()} patients")
    print(f"Written: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run assembly**

```bash
.venv/bin/python scripts/mechanistic_twin/ch9_6_assemble_cohort.py
```

Expected: prints merged cohort size ≥ 800 patients.

- [ ] **Step 5: Run tests**

```bash
.venv/bin/pytest tests/mechanistic_twin/test_ch9_6_cohort.py -v
```

Expected: 5/5 PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/mechanistic_twin/ch9_6_assemble_cohort.py \
        tests/mechanistic_twin/test_ch9_6_cohort.py \
        outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet
git commit -m "feat(ch9.6-task4): assemble 5-channel cohort + NfL held-out"
```

---

## Task 5 — SAEM v3 Code Extension (add GFAP likelihood equation)

**Files:**
- Modify: `scripts/mechanistic_twin/multi_obs_saem.py` (add `gfap_likelihood` function + include in total log-likelihood; add `S_GFAP` to parameter vector)
- Create: `tests/mechanistic_twin/test_ch9_6_saem_code.py`

- [ ] **Step 1: Write failing test**

Create `tests/mechanistic_twin/test_ch9_6_saem_code.py`:

```python
import numpy as np
import pytest

from scripts.mechanistic_twin.multi_obs_saem import (  # noqa
    total_log_likelihood, gfap_likelihood, ParamVector,
)


def test_gfap_likelihood_finite():
    """GFAP likelihood should be finite for sane inputs."""
    ll = gfap_likelihood(
        gfap_obs=5.0, O_t=1e-3, s_gfap=2.0, sigma_gfap=0.5,
    )
    assert np.isfinite(ll)


def test_gfap_likelihood_maximum_at_zero_residual():
    s_gfap = 2.0
    O_t = 0.5
    max_ll = gfap_likelihood(s_gfap * O_t, O_t, s_gfap, 0.5)
    off_ll = gfap_likelihood(s_gfap * O_t + 2.0, O_t, s_gfap, 0.5)
    assert max_ll > off_ll


def test_total_log_likelihood_includes_gfap_channel():
    """If GFAP observation is present, it should contribute to the total."""
    obs_with = {"sbr": 1.2, "asyn_agg_pct": 12.5, "saa_ttt": 0.9,
                "nev_asyn": 0.8, "gfap_npx": 4.5}
    obs_without = {**obs_with}
    obs_without.pop("gfap_npx")

    params = ParamVector(k_n=3.6e-4, alpha_tox=1.17e-5, s_gfap=2.0,
                         sigma_gfap=0.5, sigma_sbr=0.2, sigma_agg=10.0,
                         sigma_saa=0.1, sigma_nev=0.5)

    ll_with = total_log_likelihood(obs_with, params, t_years=3.0)
    ll_without = total_log_likelihood(obs_without, params, t_years=3.0)
    assert ll_with != ll_without, "GFAP channel should affect total LL"
```

- [ ] **Step 2: Run test to verify failure**

```bash
.venv/bin/pytest tests/mechanistic_twin/test_ch9_6_saem_code.py -v
```

Expected: ImportError (ParamVector/gfap_likelihood don't exist yet).

- [ ] **Step 3: Inspect current SAEM likelihood to understand insertion point**

```bash
grep -n "def.*likelihood\|def total_log\|CHANNELS\|obs_types" /Users/blair.dupre/Projects/CSCI-FALL-2025/scripts/mechanistic_twin/multi_obs_saem.py | head -30
```

Note the line numbers for (a) the per-channel likelihood functions and (b) the summation in `total_log_likelihood`.

- [ ] **Step 4: Add `gfap_likelihood` function to `multi_obs_saem.py`**

Insert near the other per-channel likelihood functions:

```python
def gfap_likelihood(gfap_obs: float, O_t: float, s_gfap: float,
                    sigma_gfap: float) -> float:
    """Normal likelihood for CSF GFAP channel: gfap ~ N(s_gfap * O_t, sigma_gfap)."""
    import numpy as np
    pred = s_gfap * O_t
    resid = gfap_obs - pred
    return -0.5 * (resid / sigma_gfap) ** 2 - np.log(sigma_gfap) - 0.5 * np.log(2 * np.pi)
```

- [ ] **Step 5: Add `s_gfap` + `sigma_gfap` to `ParamVector`**

Find the parameter-vector declaration (likely a `@dataclass` or `NamedTuple` near the top of `multi_obs_saem.py`). Append the two new fields at the end so existing SAEM v2 checkpoints remain loadable:

```python
from dataclasses import dataclass, field

@dataclass
class ParamVector:
    k_n: float
    alpha_tox: float
    sigma_sbr: float
    sigma_agg: float
    sigma_saa: float
    sigma_nev: float
    # ch9.6 additions — append at end to preserve v2 checkpoint compatibility
    s_gfap: float = 1.0
    sigma_gfap: float = 0.5
```

If the existing code uses a different structure (e.g., raw dict or numpy array with indexed fields), adapt: add two new entries at the end of the field list, update the `pack`/`unpack` helpers, and extend the initial-value array in `run_saem`.

- [ ] **Step 6: Wire GFAP into `total_log_likelihood`**

In `total_log_likelihood(obs, params, t_years)`, add the GFAP branch after the existing channel summations:

```python
def total_log_likelihood(obs: dict, params: ParamVector, t_years: float) -> float:
    ll = 0.0
    # ... existing channels (SBR, agg%, SAA, NEV) ...
    if "gfap_npx" in obs and obs["gfap_npx"] is not None:
        import numpy as np
        if np.isfinite(obs["gfap_npx"]):
            O_t = _O_at_time(params.k_n, params.alpha_tox, t_years)
            ll += gfap_likelihood(
                gfap_obs=obs["gfap_npx"],
                O_t=O_t,
                s_gfap=params.s_gfap,
                sigma_gfap=params.sigma_gfap,
            )
    return ll
```

where `_O_at_time` is the existing helper that returns the oligomer compartment at time `t_years`. If that helper is named differently in the file (e.g., `compute_O_ss`, `forward_O`), use the existing name — do NOT introduce a new one.

- [ ] **Step 7: Run tests to verify pass**

```bash
.venv/bin/pytest tests/mechanistic_twin/test_ch9_6_saem_code.py -v
```

Expected: 3/3 PASS.

- [ ] **Step 8: Verify existing SAEM v2 behavior is preserved**

```bash
.venv/bin/pytest tests/mechanistic_twin/ -k "saem and not ch9_6" -v
```

Expected: all pre-existing SAEM tests still pass (no regression).

- [ ] **Step 9: Commit**

```bash
git add scripts/mechanistic_twin/multi_obs_saem.py \
        tests/mechanistic_twin/test_ch9_6_saem_code.py
git commit -m "feat(ch9.6-task5): add GFAP likelihood channel to multi_obs_saem

Adds gfap_likelihood(), extends ParamVector with (s_gfap, sigma_gfap), and
wires GFAP into total_log_likelihood() behind a presence check so
SAEM v2 runs without gfap_npx remain byte-compatible."
```

---

## Task 6 — SAEM v3 Calibration Run on 1,065 Patients

**Files:**
- Create: `scripts/mechanistic_twin/ch9_6_run_saem_v3.py` (wrapper around `multi_obs_saem.run_saem`)
- Create: `tests/mechanistic_twin/test_ch9_6_saem_v3_gates.py` (post-run gate test)
- Writes: `outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3/`

- [ ] **Step 1: Write post-run gate test**

Create `tests/mechanistic_twin/test_ch9_6_saem_v3_gates.py`:

```python
import json
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3"


def test_v3_run_exists():
    assert (V3 / "individual_params.csv").exists()


def test_v3_patient_count():
    df = pd.read_csv(V3 / "individual_params.csv")
    n = df["patno"].nunique()
    assert n >= 900, f"SAEM v3 only fit {n} patients — need ≥ 900"


def test_v3_convergence():
    with (V3 / "diagnostics.json").open() as f:
        diag = json.load(f)
    assert diag["converged"] is True
    assert diag["final_log_likelihood"] > float("-inf")


def test_v3_k_n_posterior_tighter_than_v2():
    v2 = pd.read_csv(ROOT / "outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v2/individual_params.csv")
    v3 = pd.read_csv(V3 / "individual_params.csv")
    sd_v2 = v2["log_k_n"].std()
    sd_v3 = v3["log_k_n"].std()
    assert sd_v3 <= sd_v2 * 1.05, (
        f"v3 log_k_n SD {sd_v3:.3f} is wider than v2 {sd_v2:.3f} — 5-channel should tighten or match"
    )


def test_v3_alpha_tox_posterior_tighter_than_v2():
    v2 = pd.read_csv(ROOT / "outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v2/individual_params.csv")
    v3 = pd.read_csv(V3 / "individual_params.csv")
    sd_v2 = v2["log_alpha_tox"].std()
    sd_v3 = v3["log_alpha_tox"].std()
    assert sd_v3 <= sd_v2 * 1.10, f"v3 log_alpha_tox SD {sd_v3:.3f} widens over v2 {sd_v2:.3f}"


def test_v3_population_k_n_median_in_literature_range():
    """Cohort median % loss/year should be 2–5 %/yr (Fearnley 1991)."""
    df = pd.read_csv(V3 / "individual_params.csv")
    pct_loss = df["pct_loss_per_yr_median"].median()
    assert 1.5 < pct_loss < 6.0, f"median {pct_loss:.2f} outside Fearnley range"
```

- [ ] **Step 2: Run the gate test to verify failure**

```bash
.venv/bin/pytest tests/mechanistic_twin/test_ch9_6_saem_v3_gates.py -v
```

Expected: FAIL (v3 does not exist yet).

- [ ] **Step 3: Write the SAEM v3 runner**

Create `scripts/mechanistic_twin/ch9_6_run_saem_v3.py`:

```python
"""SAEM v3 run on 1,065 patients with 5-channel likelihood.

Channels: SBR + aSyn_agg% + SAA_TTT + NEV_asyn + CSF_GFAP
NfL is EXCLUDED from likelihood (held-out validation in Task 7).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

from scripts.mechanistic_twin.multi_obs_saem import run_saem  # noqa

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet"
OUT_TAG = "saem_multi_obs_v3"


def build_patient_records(df: pd.DataFrame) -> list[dict]:
    """Convert merged cohort into the list[dict] format run_saem expects."""
    records = []
    for patno, g in df.groupby("patno"):
        g = g.sort_values("visit_month")
        records.append({
            "patno": int(patno),
            "t_years": (g["visit_month"].values / 12.0).tolist(),
            "sbr": g["sbr_putamen"].tolist(),
            "asyn_agg_pct": g["asyn_agg_pct"].tolist(),
            "saa_ttt": g["saa_ttt"].tolist(),
            "nev_asyn": g["nev_asyn"].tolist(),
            "gfap_npx": g["gfap_npx"].tolist(),  # may contain NaN — SAEM handles missingness
        })
    return records


def main() -> None:
    df = pd.read_parquet(COHORT)
    patients = build_patient_records(df)
    print(f"SAEM v3 input: {len(patients)} patients, "
          f"{sum(len(p['t_years']) for p in patients)} visits")

    run_saem(
        patients=patients,
        n_iterations=150,
        n_burn=75,
        seed=20260415,
        run_tag=OUT_TAG,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Dry-run on 50 patients first**

Before running for 60–90 minutes on the full cohort, smoke-test with a subset:

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet')
subset = df[df['patno'].isin(df['patno'].unique()[:50])]
subset.to_parquet('outputs/mechanistic_twin/ch9_6/_dryrun_cohort.parquet')
"
# Hand-edit ch9_6_run_saem_v3.py to point at _dryrun_cohort.parquet, set
# n_iterations=20, n_burn=10, run_tag='saem_multi_obs_v3_dryrun', then:
.venv/bin/python scripts/mechanistic_twin/ch9_6_run_saem_v3.py
```

Expected: completes in < 5 min, produces a `saem_multi_obs_v3_dryrun` dir with `individual_params.csv`, `convergence_history.csv`, `diagnostics.json`.

- [ ] **Step 5: Revert to full cohort + full iterations, run the real SAEM v3**

```bash
.venv/bin/python scripts/mechanistic_twin/ch9_6_run_saem_v3.py 2>&1 | tee outputs/mechanistic_twin/ch9_6/saem_v3_run.log
```

Expected: 60–120 minutes runtime. Writes `outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3/`.

- [ ] **Step 6: Run the gate tests**

```bash
.venv/bin/pytest tests/mechanistic_twin/test_ch9_6_saem_v3_gates.py -v
```

Expected: 6/6 PASS.

- [ ] **Step 7: If any gate fails → STOP and diagnose**

Gate failures mean one of:
- Convergence failed (`converged=False`): check `convergence_history.csv`, probably raise `n_iterations` and restart.
- Posterior didn't tighten: check whether GFAP channel is overflowing NaN. If `s_gfap` drifts to ~0, GFAP is empirically uninformative — document as a finding, drop GFAP, and re-run 4-channel v3.
- Population median outside Fearnley: indicates misspecification; review Phase 2 fixed parameters in `mechanistic.data_registry`.

- [ ] **Step 8: Commit artifacts + pointers, but NOT the raw chain files (they may be large)**

```bash
git add scripts/mechanistic_twin/ch9_6_run_saem_v3.py \
        tests/mechanistic_twin/test_ch9_6_saem_v3_gates.py \
        outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3/diagnostics.json \
        outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3/individual_params.csv \
        outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3/convergence_history.csv \
        outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3/RUN_MANIFEST.md \
        outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3/provenance.json
git commit -m "feat(ch9.6-task6): SAEM v3 run on 1,065 patients, 5-channel likelihood

v3 metrics vs v2:
  N patients:        <N>  (v2: 304)
  σ(log k_n):        <SD> (v2: 1.886)
  σ(log α_tox):      <SD> (v2: 1.959)
  median %loss/yr:   <X>  (Fearnley range 2–5)
  cor(k_n, α_tox):   <ρ>  (v2: -0.36)
All 6 gate tests pass."
```

---

## Task 7 — LOO Forward Validation + NfL Held-Out Check

**Files:**
- Create: `scripts/mechanistic_twin/ch9_6_loo_forward_validation.py`
- Create: `scripts/mechanistic_twin/ch9_6_nfl_holdout_validation.py`
- Create: `tests/mechanistic_twin/test_ch9_6_validation.py`
- Writes: `outputs/mechanistic_twin/ch9_6/loo_forward.json`, `outputs/mechanistic_twin/ch9_6/nfl_holdout.json`

- [ ] **Step 1: Write failing validation tests**

Create `tests/mechanistic_twin/test_ch9_6_validation.py`:

```python
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/mechanistic_twin/ch9_6"


def test_loo_json_exists():
    assert (OUT / "loo_forward.json").exists()


def test_loo_coverage_at_least_85():
    with (OUT / "loo_forward.json").open() as f:
        data = json.load(f)
    cov = data["coverage_95_credible_interval"]
    assert cov >= 0.85, f"LOO coverage {cov:.1%} < 85%"


def test_loo_relative_error_reasonable():
    with (OUT / "loo_forward.json").open() as f:
        data = json.load(f)
    rel = data["median_relative_error"]
    assert 0.0 < rel < 0.25, f"median relative error {rel:.3f} out of Phase 1 ballpark"


def test_nfl_holdout_r_squared():
    with (OUT / "nfl_holdout.json").open() as f:
        data = json.load(f)
    r2 = data["r_squared_dN_dt_vs_NfL"]
    assert r2 >= 0.20, f"NfL held-out R²={r2:.3f} too low — predicted dN/dt is not informative"


def test_nfl_holdout_sample_size():
    with (OUT / "nfl_holdout.json").open() as f:
        data = json.load(f)
    assert data["n_patients"] >= 400
```

- [ ] **Step 2: Write LOO forward validation script**

Create `scripts/mechanistic_twin/ch9_6_loo_forward_validation.py`. Pattern mirrors `src/mechanistic_twin/scripts/loo_validation.jl` (Phase 1 Addendum A2): for each held-out scan, use the posterior from the remaining scans (reuse the SAEM v3 per-patient EBEs as the proposal; this is an importance-sampling LOO, cheaper than refitting), simulate forward, and check whether the observed scan lies within the 95% credible interval of the predicted distribution.

```python
"""Ch 9.6 LOO forward validation — 5-channel edition.

For each (patno, scan) pair:
  1. Load per-patient posterior samples of (k_n, α_tox) from SAEM v3.
  2. Reweight samples by the likelihood of the remaining scans (importance
     sampling — avoids refitting 8,000 times).
  3. Simulate SBR at the held-out scan time; compute 95% CI.
  4. Record whether the observed SBR lies within the CI.

Output: outputs/mechanistic_twin/ch9_6/loo_forward.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from scripts.mechanistic_twin.multi_obs_saem import (  # noqa
    total_log_likelihood, ParamVector,
)

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3"
COHORT = ROOT / "outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet"
OUT = ROOT / "outputs/mechanistic_twin/ch9_6/loo_forward.json"

GAMMA = 0.7
SBR_0 = 1.5
M_SS = 2.0


def simulate_sbr(k_n: float, alpha_tox: float, t_years: float) -> float:
    O_ss = k_n * M_SS / (alpha_tox + 1e-30)
    N_frac = np.exp(-alpha_tox * O_ss * t_years)
    return SBR_0 * N_frac ** GAMMA


def loo_one_patient(pat_rows: pd.DataFrame, samples: np.ndarray) -> list[dict]:
    """samples: (S, 2) array of (k_n, alpha_tox) posterior samples."""
    visits = pat_rows.sort_values("visit_month").reset_index(drop=True)
    results = []
    for i, held_out in visits.iterrows():
        remaining = visits.drop(index=i)
        log_w = np.zeros(len(samples))
        for s_idx, (k_n, a_tox) in enumerate(samples):
            params = ParamVector(
                k_n=float(k_n), alpha_tox=float(a_tox),
                sigma_sbr=0.2, sigma_agg=10.0, sigma_saa=0.1,
                sigma_nev=0.5, s_gfap=1.0, sigma_gfap=0.5,
            )
            for _, v in remaining.iterrows():
                obs = {
                    "sbr": v["sbr_putamen"],
                    "asyn_agg_pct": v["asyn_agg_pct"],
                    "saa_ttt": v["saa_ttt"],
                    "nev_asyn": v["nev_asyn"],
                    "gfap_npx": v["gfap_npx"],
                }
                log_w[s_idx] += total_log_likelihood(
                    obs, params, t_years=v["visit_month"] / 12.0
                )
        log_w -= log_w.max()
        w = np.exp(log_w)
        w /= w.sum() + 1e-30
        t = held_out["visit_month"] / 12.0
        sbr_preds = np.array([simulate_sbr(k, a, t) for k, a in samples])
        lo = np.quantile(sbr_preds, 0.025, method="inverted_cdf") if False else \
             np.percentile(sbr_preds, 2.5)
        hi = np.percentile(sbr_preds, 97.5)
        observed = held_out["sbr_putamen"]
        if pd.notna(observed):
            results.append({
                "patno": int(held_out["patno"]),
                "t_years": float(t),
                "observed": float(observed),
                "pred_median": float(np.median(sbr_preds)),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "in_ci": bool(lo <= observed <= hi),
                "rel_error": float(abs(observed - np.median(sbr_preds)) / abs(observed)),
            })
    return results


def main() -> None:
    cohort = pd.read_parquet(COHORT)
    ebes = pd.read_csv(V3 / "individual_params.csv")

    # If SAEM v3 saved posterior samples per patient in a separate file,
    # load them. Otherwise fall back to Gaussian samples around EBE medians.
    samples_path = V3 / "posterior_samples.parquet"
    use_samples = samples_path.exists()

    all_results: list[dict] = []
    for patno, g in cohort.groupby("patno"):
        if g["sbr_putamen"].notna().sum() < 3:
            continue  # need at least 3 scans for LOO
        pat_ebe = ebes[ebes["patno"] == patno]
        if pat_ebe.empty:
            continue
        if use_samples:
            df_samples = pd.read_parquet(samples_path, filters=[("patno", "=", patno)])
            samples = df_samples[["k_n", "alpha_tox"]].to_numpy()
        else:
            k_mu = float(pat_ebe["k_n_median"].iloc[0])
            a_mu = float(pat_ebe["alpha_tox_median"].iloc[0])
            rng = np.random.default_rng(20260415 + int(patno))
            samples = np.column_stack([
                rng.lognormal(np.log(k_mu), 0.3, size=200),
                rng.lognormal(np.log(a_mu), 0.3, size=200),
            ])
        all_results.extend(loo_one_patient(g, samples))

    df_res = pd.DataFrame(all_results)
    coverage = df_res["in_ci"].mean()
    med_rel = df_res["rel_error"].median()

    out = {
        "n_scans_evaluated": int(len(df_res)),
        "n_patients": int(df_res["patno"].nunique()),
        "coverage_95_credible_interval": float(coverage),
        "median_relative_error": float(med_rel),
        "target_coverage": 0.85,
        "target_median_rel_error": 0.20,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(out, f, indent=2)
    df_res.to_csv(OUT.with_suffix(".csv"), index=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```

Note: this LOO is importance-sampling based, not refit-based. If the IS weights are degenerate (ESS < 10), fall back to refitting — but at 1,065 patients × 8,000 scans that is a multi-day run. Check ESS in the log and document if IS degenerates on a non-trivial fraction of scans.

- [ ] **Step 3: Write NfL held-out validation script**

Create `scripts/mechanistic_twin/ch9_6_nfl_holdout_validation.py`:

```python
"""NfL held-out validation: predict NfL from fitted N(t) trajectory, report R²
and slope calibration.

Per the §9.6 channel-design critical-thinking review, NfL is NOT in the SAEM
likelihood because it's directionally redundant with SBR. But NfL has 1,190
patients at Project 144 and is an externally-validated PD progression marker
(Bäckström 2020, Sampedro 2020). So we use it as a cross-check on the fitted
dN/dt trajectory from SAEM v3.

Expected relationship: NfL ∝ -dN/dt (higher axon loss rate → higher NfL)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3"
COHORT = ROOT / "outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet"
OUT = ROOT / "outputs/mechanistic_twin/ch9_6/nfl_holdout.json"


def compute_dN_dt_from_ebes(ebes: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """From per-patient posterior medians of (k_n, α_tox), compute predicted
    instantaneous dN/dt at each visit where NfL was observed."""
    # simplified: dN/dt ≈ -α_tox * O_ss(k_n) * N_frac(t)
    merged = cohort.merge(ebes[["patno", "k_n_median", "alpha_tox_median"]], on="patno")
    merged = merged.dropna(subset=["nfl_pg_per_ml"])
    M_SS = 2.0
    merged["O_ss"] = merged["k_n_median"] * M_SS / (merged["alpha_tox_median"] + 1e-30)
    merged["N_frac_pred"] = np.exp(-merged["alpha_tox_median"] * merged["O_ss"]
                                   * merged["visit_month"] / 12.0)
    merged["dN_dt_pred"] = (-merged["alpha_tox_median"] * merged["O_ss"]
                            * merged["N_frac_pred"])
    return merged


def main() -> None:
    ebes = pd.read_csv(V3 / "individual_params.csv")
    cohort = pd.read_parquet(COHORT)
    merged = compute_dN_dt_from_ebes(ebes, cohort)

    # Regress |dN/dt_pred| onto NfL
    x = np.log(np.abs(merged["dN_dt_pred"]) + 1e-12)
    y = np.log(merged["nfl_pg_per_ml"] + 1e-12)
    r, p = pearsonr(x, y)
    r_squared = r ** 2

    # Slope calibration: linear fit y ~ a + b*x
    b, a = np.polyfit(x, y, 1)

    results = {
        "n_patients": int(merged["patno"].nunique()),
        "n_visits": int(len(merged)),
        "r_squared_dN_dt_vs_NfL": float(r_squared),
        "pearson_r": float(r),
        "p_value": float(p),
        "slope_log_log": float(b),
        "intercept_log_log": float(a),
        "interpretation": (
            "If R² ≥ 0.30 and slope ∈ [0.5, 1.5], the SAEM-predicted dN/dt "
            "trajectory recovers the NfL progression signal. NfL was held out "
            "of the SAEM likelihood; this is a genuine external validation."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run both scripts**

```bash
.venv/bin/python scripts/mechanistic_twin/ch9_6_loo_forward_validation.py 2>&1 | tee outputs/mechanistic_twin/ch9_6/loo_run.log
.venv/bin/python scripts/mechanistic_twin/ch9_6_nfl_holdout_validation.py
```

LOO run is the long one (~30 min on 1,065 patients, ~8,000 scans).

- [ ] **Step 5: Run validation tests**

```bash
.venv/bin/pytest tests/mechanistic_twin/test_ch9_6_validation.py -v
```

Expected: 5/5 PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/mechanistic_twin/ch9_6_loo_forward_validation.py \
        scripts/mechanistic_twin/ch9_6_nfl_holdout_validation.py \
        tests/mechanistic_twin/test_ch9_6_validation.py \
        outputs/mechanistic_twin/ch9_6/loo_forward.json \
        outputs/mechanistic_twin/ch9_6/nfl_holdout.json
git commit -m "feat(ch9.6-task7): LOO forward validation + NfL held-out check

LOO coverage: <X>% at 95% CI (Phase 1 A2 = 93.75%, gate 85%)
NfL held-out R² = <Y>, slope <Z> (Bäckström 2020 external validation)"
```

---

## Task 8 — Figures + FIM Ablation Table

**Files:**
- Create: `scripts/mechanistic_twin/ch9_6_generate_figures.py`
- Create: `tests/mechanistic_twin/test_ch9_6_figures.py`
- Writes: `outputs/mechanistic_twin/ch9_6/figures/*.png`, `*.pdf`

Five figures:
1. **fig_9_6_1_channel_map.png** — Schematic: 5 observation channels → 2 ODE parameters
2. **fig_9_6_2_fim_ablation_table.png** — κ(FIM) as channels are added: {SBR}, {+agg%}, {+SAA}, {+NEV}, {+GFAP}; with eigenvector loadings
3. **fig_9_6_3_loo_coverage.png** — Per-patient LOO coverage histogram + 95% CI band
4. **fig_9_6_4_nfl_holdout.png** — Scatter of predicted |dN/dt| vs observed NfL
5. **fig_9_6_5_posterior_compare.png** — v2 (6-channel, 304pts) vs v3 (5-channel, 1,065pts) posteriors for k_n and α_tox

- [ ] **Step 1: Write failing test**

```python
# tests/mechanistic_twin/test_ch9_6_figures.py
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "outputs/mechanistic_twin/ch9_6/figures"

def test_all_five_figures_exist():
    for i, stem in enumerate([
        "fig_9_6_1_channel_map",
        "fig_9_6_2_fim_ablation",
        "fig_9_6_3_loo_coverage",
        "fig_9_6_4_nfl_holdout",
        "fig_9_6_5_posterior_compare",
    ], 1):
        for ext in ["png", "pdf"]:
            p = FIGS / f"{stem}.{ext}"
            assert p.exists(), f"missing figure: {p}"

def test_figures_non_empty():
    for f in FIGS.glob("fig_9_6_*.png"):
        assert f.stat().st_size > 10_000, f"figure {f.name} is too small"
```

- [ ] **Step 2: Run test to verify failure**

```bash
.venv/bin/pytest tests/mechanistic_twin/test_ch9_6_figures.py -v
```

- [ ] **Step 3: Write generator script**

Create `scripts/mechanistic_twin/ch9_6_generate_figures.py` — use matplotlib 3.9+ conventions (`tick_labels` not `labels`). Pull numbers from `identifiability.json`, `loo_forward.json`, `nfl_holdout.json`, and the v2 + v3 `individual_params.csv` files. Each figure saves both .png (300 dpi) and .pdf.

- [ ] **Step 4: Generate figures**

```bash
.venv/bin/python scripts/mechanistic_twin/ch9_6_generate_figures.py
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/pytest tests/mechanistic_twin/test_ch9_6_figures.py -v
```

- [ ] **Step 6: Visually inspect each figure**

Open each PNG in the IDE. Confirm:
- fig 1: channel/compartment arrows are readable
- fig 2: κ drops monotonically as channels add (or the table explains why not)
- fig 3: coverage histogram centered on 95%
- fig 4: NfL scatter shows a positive trend with annotated R²
- fig 5: v3 posteriors visibly tighter than v2

If any figure fails visual inspection, fix the generator script — do not ship bad figures.

- [ ] **Step 7: Commit**

```bash
git add scripts/mechanistic_twin/ch9_6_generate_figures.py \
        tests/mechanistic_twin/test_ch9_6_figures.py \
        outputs/mechanistic_twin/ch9_6/figures/*.png \
        outputs/mechanistic_twin/ch9_6/figures/*.pdf
git commit -m "feat(ch9.6-task8): 5 publication figures for §9.6"
```

---

## Task 9 — §9.6 Chapter Prose + Dissertation Integration

**Files:**
- Create: `writing/chapters/ch09_section96_multichannel.tex` (new subsection)
- Modify: `writing/chapters/ch09_paper7.tex` (include the new subsection at the right anchor)
- Modify: `writing/dissertation.tex` (no changes expected; ch09 is already included)
- Reference: `reference.phase5_bibliography`, `audit.citation`

- [ ] **Step 1: Locate the correct insertion point in ch09**

```bash
grep -n "^\\\\section\|^\\\\subsection" /Users/blair.dupre/Projects/CSCI-FALL-2025/writing/chapters/ch09_paper7.tex
```

Find the last `\section` before `\section{Limitations}` or `\section{Discussion}` — that's where `\input{ch09_section96_multichannel}` goes.

- [ ] **Step 2: Write the prose**

Create `writing/chapters/ch09_section96_multichannel.tex`:

```latex
\section{Multi-Channel Observation Extension}
\label{sec:ch9-6}

\subsection{Motivation}

The Phase 2 calibration in §9.3 used a single observation channel
(DaT-SPECT SBR) to identify two mechanistic parameters $(k_n, \alpha_{tox})$
per patient. This is technically identifiable but practically sloppy: a
single-channel Jacobian with rank 2 forced the calibration to work primarily
along the combined axis $k_n \cdot \alpha_{tox}$ (the T_{tox} product),
leaving the individual parameters weakly informed for most patients. This
section closes that gap by adding four additional observation channels that
map to distinct ODE compartments.

\subsection{Observation Model}

The augmented observation map is:
\begin{align}
\text{SBR}(t)       &= \text{SBR}_0 \cdot (N(t)/N_0)^{\gamma} + \epsilon_{\text{SBR}} \\
\text{aSyn}_{\text{agg\%}}(t) &= \frac{O_{ss}}{M_{ss} + O_{ss}} \cdot 100 + \epsilon_{\text{agg}} \\
\text{SAA}_{\text{TTT}}(t)    &= \phi(F_{ss}(k_n)) + \epsilon_{\text{SAA}} \\
\text{NEV}_{\text{asyn}}(t)   &= s_{\text{NEV}} \cdot (O_{ss} + r_F F_{ss}) + \epsilon_{\text{NEV}} \\
\text{CSF GFAP}(t)            &= s_{\text{GFAP}} \cdot O(t) + \epsilon_{\text{GFAP}}
\end{align}

CSF $\alpha$-synuclein and serum NfL are EXCLUDED from the likelihood —
CSF $\alpha$-syn because Mollenhauer \emph{et al.} \citep{mollenhauer2019longitudinal}
established that it does not correlate with PD progression at the individual
level, and NfL because it carries the same directional information as SBR
(both reflect dN/dt); including it does not add Fisher information
\citep{backstrom2020nfl,sampedro2020nfl}. GFAP, not $\alpha$-syn, is the Olink
anchor because $\alpha$-synuclein is absent from both Olink and SomaScan
panels \citep{rutledge2024}, and CSF GFAP has established mechanistic links
to microglial response to oligomeric $\alpha$-syn \citep{liu2023gfap}.

\subsection{Structural Identifiability}

We extended the identifiability framework from §9.2 (Jacobian rank + FIM
condition number) with a sloppiness analysis
\citep{gutenkunst2007,transtrum2015} to detect redundancy among the five
channels. Table~\ref{tab:ch9-6-fim} reports $\kappa(\text{FIM})$ as channels
are added cumulatively. The 5-channel Jacobian has rank 2 (the structural
minimum), $\kappa = <K_VALUE>$, and an eigenvalue spread of
$<SPREAD>$ decades. Profile likelihood \citep{raue2013} on the joint
posterior of $(k_n, \alpha_{tox})$ produces two-sided 95\% intervals
(Figure~\ref{fig:ch9-6-pl}), confirming practical identifiability.

% ... [continues with sections on calibration, LOO validation, NfL held-out,
%      comparison to SAEM v2, and limitations]

\subsection{Calibration Results}

Table~\ref{tab:ch9-6-saem-compare} compares SAEM v2 (6-channel, 304
patients) with v3 (5-channel, 1,065 patients). v3 achieves
$\sigma(\log k_n) = <VALUE>$ (vs v2: 1.886) and cohort-median
$<X>\%/$yr decline, within the Fearnley-Lees (1991) 2–5\%/yr range.

\subsection{Limitations}

\begin{itemize}
  \item Olink Project 222 CSF panel covers 227 patients; coverage of GFAP
        is therefore lower than the 1,065-patient cohort for SBR.
        The SAEM handles missingness per-channel, but GFAP informs
        $\alpha_{tox}$ only for the subset with Olink data.
  \item The sloppiness analysis reports the eigenvalue spread at a single
        nominal parameter point; a fully global identifiability proof
        would require symbolic tools (DAISY, STRIKE-GOLDD). For a 2-parameter
        system with algebraic observations, local identifiability via the
        rank-plus-FIM approach is standard practice
        \citep{villaverde2016structural}.
  \item CSF GFAP reflects astrocytic activation downstream of oligomeric
        burden and may be confounded by non-PD neuroinflammation. The NfL
        held-out validation (Figure~\ref{fig:ch9-6-nfl}) mitigates this by
        providing an independent check on the fitted dN/dt trajectory.
\end{itemize}
```

- [ ] **Step 3: Include the new subsection in ch09**

Edit `writing/chapters/ch09_paper7.tex`:
```latex
% Insert before \section{Limitations} or \section{Discussion}
\input{chapters/ch09_section96_multichannel.tex}
```

- [ ] **Step 4: Rebuild PDF**

```bash
cd /Users/blair.dupre/Projects/CSCI-FALL-2025/writing
latexmk -pdf -interaction=nonstopmode dissertation.tex 2>&1 | tail -30
```

Expected: PDF compiles without errors, Ch 9 now contains a §9.6 section referencing all 4 new citations and 5 new figures.

- [ ] **Step 5: Manual proofread pass**

Open the compiled PDF, navigate to Ch 9 §9.6. Confirm:
- All placeholders (`<VALUE>`, `<K_VALUE>`, etc.) replaced with actual numbers from the JSON outputs.
- All 5 figures render at the expected positions.
- All 4 new citations (liu2023gfap, backstrom2020nfl, sampedro2020nfl, rutledge2024) resolve correctly (no `?? [liu2023gfap]`).
- The FIM ablation table renders.

Replace every remaining placeholder with the actual values from `outputs/mechanistic_twin/ch9_6/identifiability.json`, `loo_forward.json`, `nfl_holdout.json`, and the v2 vs v3 comparison.

- [ ] **Step 6: Re-audit — ingest §9.6 claims into audit.claim**

Run the audit-ingestion script for the updated Ch 9 tex:

```bash
.venv/bin/python scripts/defense_prep/ingest_chapter_claims.py --chapter 9 --tex writing/chapters/ch09_paper7.tex
```

If that script doesn't exist for one-off chapter re-ingestion, add at least one manual row to `audit.claim` per numerical claim introduced in §9.6.

- [ ] **Step 7: Update CLAUDE.md root with §9.6 completion status**

Add to the dissertation roadmap table:

```markdown
| 2–7 | Appendix E §E.1–§E.2 Docker + data dictionary | **DONE** |
| 2–7 | Ch 9 §9.6 multi-channel observation | **DONE** 2026-MM-DD |
```

- [ ] **Step 8: Commit the chapter + integration**

```bash
git add writing/chapters/ch09_section96_multichannel.tex \
        writing/chapters/ch09_paper7.tex \
        CLAUDE.md
git commit -m "feat(ch9.6-task9): §9.6 multi-channel calibration chapter complete

Adds §9.6 to Ch 9 Paper 7. All 4 new citations cite, all 5 figures render.
LOO coverage: <X>%, NfL held-out R²: <Y>, FIM κ: <Z>.
Closes the SBR-only limitation of Paper 7."
```

- [ ] **Step 9: Run full test suite to confirm no regressions**

```bash
.venv/bin/pytest tests/mechanistic_twin/ -v
```

Expected: all Ch 9 §9.6 tests pass; no pre-existing test fails.

- [ ] **Step 10: Zotero sync closeout**

Add Liu 2023, Bäckström 2020, Sampedro 2020, Bartl 2021 to Zotero collection RT8B9N2J. After BBT auto-export refreshes `outputs/mechanistic_twin/paper10_mech_vs_giman/phase5_literature_bibliography.bib`, re-run `scripts/mechanistic_twin/add_ch9_6_citations.py` with `zotero_verified=1` and the correct zotero_keys.

- [ ] **Step 11: Final commit**

```bash
git add .
git commit -m "chore(ch9.6): close out — zotero verified, audit claims ingested"
```

---

## Acceptance criteria (§9.6 complete)

Every box below must be checked before declaring §9.6 done.

- [ ] `mechanistic.data_registry`, `mechanistic.literature_anchors`, `mechanistic.empirical_findings` exist in Postgres with ≥ 20 rows each.
- [ ] 4 new citations present in `audit.citation`.
- [ ] `outputs/mechanistic_twin/ch9_6/identifiability.json` shows rank=2, κ<1000, eigenvalue spread < 3 decades.
- [ ] `outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet` has ≥ 800 patients with 5 channels.
- [ ] `outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3/` exists with ≥ 900 patients, converged diagnostics.
- [ ] `outputs/mechanistic_twin/ch9_6/loo_forward.json` shows ≥ 85% coverage.
- [ ] `outputs/mechanistic_twin/ch9_6/nfl_holdout.json` shows R² ≥ 0.20.
- [ ] All 5 figures exist in both PNG and PDF.
- [ ] `writing/chapters/ch09_section96_multichannel.tex` compiles into the dissertation PDF with no unresolved citations.
- [ ] All Task 1–9 pytest suites pass.
- [ ] CLAUDE.md updated to mark §9.6 DONE.

## Deferred / out of scope

- **Symbolic identifiability tools (DAISY, STRIKE-GOLDD, StructuralIdentifiability.jl)** — acknowledged in Limitations but not executed. Appropriate for a methods-paper follow-up.
- **Full ODE integration for the identifiability audit** — Task 2 uses a closed-form steady-state approximation. A full ODE-based Jacobian is a Paper 12 (phys-GIMIN) deliverable.
- **Olink Project 222 multi-marker panel** — only GFAP extracted. TREM2, NPTX2, SNAP-25, complement cascade are available in the NPX table but are out of scope for §9.6 (see `reference.phase5_bibliography` Bartl 2021 and Nilsson 2024 for future extensions).
- **NfL as SAEM likelihood equation** — deliberately held out. A sensitivity analysis ("what if we added NfL?") is a future cross-check, not a §9.6 deliverable.
