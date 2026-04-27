# Dissertation Completion Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Scope note:** This plan covers 4 subsystems: (A) Appendix E reproducibility, (B) Chapter 16 Paper 11 Cross-Cohort, (C) Chapter 11 §11.7 Regional ROI, (D) Chapter 13 §13.8 Mechanistic Conformal + narrative refreshes. Each subsystem is executable independently; Subsystem A (Docker + data dictionary) is on the critical path and must finish first so committee members can rebuild the environment. For subsystems B-D, this plan defines phase-level tasks; when each phase starts, write a detailed sub-plan per `writing-plans` conventions.

**Goal:** Ship a defendable 16-chapter + 2-appendix PhD dissertation in 17 weeks by closing the PPMI-only training limitation (Ch 16), adding spatial resolution (Ch 11 §11.7), promoting NASEM uncertainty compliance (Ch 13 §13.8), and packaging reproducibility artifacts (Appendix E).

**Architecture:** Build-on from [completion mapping](../../../Docs/research_directions/2026-04-13_dissertation_completion_mapping.md). Single new chapter (Paper 11) bundles 5 Phase C items (GDT-5 + C4-1..4); single new appendix bundles C5-1..3. All other Phase C items are sub-section additions. Leverages existing infrastructure: 10 Paper 3 checkpoints (5 DeepHit + 5 Graph-DT) for inductive cross-cohort deployment, 1,065 Phase 2 mechanistic posteriors for §13.8 conformal bands, local PostgreSQL `giman_research` (146 tables) for one-command rebuild.

**Tech Stack:** Python 3.10+ (PyTorch 2.8, PyTorch Geometric 2.6, MAPIE 1.3, CatBoost), PostgreSQL 17, Docker Compose, LaTeX (IEEEtran), Julia 1.11 (mechanistic_twin only).

**Predecessor plans (build-on):**
- [Phase B detailed audit plan](2026-04-13-phase-b-detailed-audit-plan.md) — defense-prep audit COMPLETE; same tooling stack carries forward
- [Phase B tooling alignment](2026-04-13-phase-b-tooling-alignment.md) — MCP/skill inventory
- [Phase 5 mechanistic vs GIMAN](2026-04-12-phase5-mechanistic-vs-giman-benchmark.md) — Paper 10 Tasks 0-8 COMPLETE

---

## Timeline

| Week | Subsystem | Deliverable | Blocking? |
|---|---|---|---|
| 1 | A (App E §E.1-§E.2) | Docker compose rebuild + data dictionary | **Yes** — committee needs this to verify any other work |
| 2-3 | B (Ch 16 Task 1-2) | 12-feature alignment + inductive Graph-DT | Yes for B |
| 4-5 | B (Ch 16 Task 3) | BioFIND validation (NSD+ subgroup, n=103) | Yes for B |
| 6-7 | B (Ch 16 Task 4) | PDBP transfer (n=893 prevalent PD) | Yes for B |
| 8 | B (Ch 16 Task 5) | HBS subset (n=649, 8/12 features) | Yes for B |
| 9 | B (Ch 16 Task 6-7) | Pooled meta-analysis + Ch 16 draft | No (Paper 11 venue submission is post-defense) |
| 10-13 | C (Ch 11 §11.7) | 6-region ROI split + Phase 8b reanalysis | No |
| 14-15 | D (Ch 13 §13.8) | Mechanistic conformal bands on counterfactuals | No |
| 16 | D (Ch 14/15 refresh) | Discussion + Conclusion narrative refresh | Yes — requires Ch 16 results to cite |
| 17 | All | Final PDF compile, presubmit checks, defense slides | Yes |

---

## Subsystem A: Appendix E — Reproducibility (Week 1) — DETAILED TASKS

**Goal:** Committee members run one command and have a fully-rebuilt environment with PostgreSQL `giman_research`, all Python deps, all checkpoints, all notebooks.

**Why week 1:** Everything else in this plan is invisible to the committee until they can rebuild the environment. One week of Docker work protects 16 weeks of research from the "can't reproduce" veto.

### Task A1: Data dictionary (§E.1)

**Files:**
- Create: `outputs/dissertation/appendix_e/E1_data_dictionary.md`
- Create: `scripts/appendix_e/generate_data_dictionary.py`
- Read from: PostgreSQL `giman_research` (all 10 schemas, 146 tables)

- [ ] **A1.1: Write script that reads PostgreSQL schema**

Create `scripts/appendix_e/generate_data_dictionary.py`:

```python
"""Generate E.1 data dictionary from live giman_research PostgreSQL.

Output: per-schema Markdown tables with column name, type, n_rows, n_nonnull,
row example. One section per schema (10 total), one subsection per table (146).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import text

from giman_pipeline.data.db import get_engine

OUTPUT = Path("outputs/dissertation/appendix_e/E1_data_dictionary.md")
SCHEMAS = [
    "ppmi_raw",
    "biofind_raw",
    "pdbp_raw",
    "hbs_raw",
    "staging",
    "features",
    "longitudinal",
    "paper3",
    "ledd",
    "mechanistic",
]


def describe_table(engine, schema: str, table: str) -> str:
    cols = pd.read_sql(
        text(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = :s AND table_name = :t
            ORDER BY ordinal_position
            """
        ),
        engine,
        params={"s": schema, "t": table},
    )
    n_rows = pd.read_sql(text(f'SELECT count(*) AS n FROM "{schema}"."{table}"'), engine)["n"].iloc[0]
    lines = [f"### `{schema}.{table}` — {n_rows:,} rows\n"]
    lines.append("| Column | Type | Nullable |")
    lines.append("|---|---|---|")
    for _, r in cols.iterrows():
        lines.append(f"| `{r.column_name}` | {r.data_type} | {r.is_nullable} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    engine = get_engine()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as fh:
        fh.write("# Appendix E.1 — Data Dictionary\n\n")
        fh.write("Source: `giman_research` PostgreSQL 17 DB (auto-generated).\n\n")
        for schema in SCHEMAS:
            tables = pd.read_sql(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = :s ORDER BY table_name"
                ),
                engine,
                params={"s": schema},
            )
            fh.write(f"## Schema `{schema}` ({len(tables)} tables)\n\n")
            for t in tables["table_name"]:
                fh.write(describe_table(engine, schema, t))
                fh.write("\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
```

- [ ] **A1.2: Run generator**

Run: `.venv/bin/python scripts/appendix_e/generate_data_dictionary.py`
Expected: writes `outputs/dissertation/appendix_e/E1_data_dictionary.md` (~5000 lines, 10 schemas × 14.6 tables avg).

- [ ] **A1.3: Verify output against known counts**

Run: `grep -c "^### " outputs/dissertation/appendix_e/E1_data_dictionary.md`
Expected: 146 (matches CLAUDE.md table count).

- [ ] **A1.4: Commit**

```bash
git add scripts/appendix_e/generate_data_dictionary.py outputs/dissertation/appendix_e/E1_data_dictionary.md
git commit -m "docs(appendix-e): §E.1 data dictionary auto-generated from giman_research (146 tables)"
```

### Task A2: Docker compose rebuild (§E.2)

**Files:**
- Create: `docker/Dockerfile`
- Create: `docker/docker-compose.yml`
- Create: `docker/entrypoint.sh`
- Create: `outputs/dissertation/appendix_e/E2_docker_rebuild.md`
- Read from: `db_dump/schema_and_data.sql` (190 MB, existing)

- [ ] **A2.1: Write Dockerfile**

Create `docker/Dockerfile`:

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    postgresql-client \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /giman
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv pip install --system -e .

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
```

- [ ] **A2.2: Write docker-compose.yml**

Create `docker/docker-compose.yml`:

```yaml
version: "3.9"

services:
  postgres:
    image: postgres:17
    environment:
      POSTGRES_DB: giman_research
      POSTGRES_USER: giman
      POSTGRES_PASSWORD: giman_local_2026
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ../db_dump/schema_and_data.sql:/docker-entrypoint-initdb.d/01_restore.sql:ro
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U giman -d giman_research"]
      interval: 5s
      timeout: 5s
      retries: 20

  giman:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql+psycopg2://giman:giman_local_2026@postgres:5432/giman_research
    volumes:
      - ../outputs:/giman/outputs
      - ../data:/giman/data:ro
    ports:
      - "8888:8888"
    command: ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

volumes:
  pgdata:
```

- [ ] **A2.3: Write entrypoint that verifies DB connection**

Create `docker/entrypoint.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "Waiting for PostgreSQL at ${DATABASE_URL:-postgres:5432}..."
until PGPASSWORD=giman_local_2026 psql -h postgres -U giman -d giman_research -c 'SELECT 1' >/dev/null 2>&1; do
    sleep 1
done
echo "PostgreSQL ready."

python -c "from giman_pipeline.data.db import read_sql; print(read_sql('SELECT count(*) AS n FROM features.paper1_features_with_targets').iloc[0])"

exec "$@"
```

- [ ] **A2.4: Write E.2 companion doc**

Create `outputs/dissertation/appendix_e/E2_docker_rebuild.md`:

```markdown
# Appendix E.2 — Docker Reproducibility

## Prerequisites

- Docker Desktop ≥ 4.30 (or Linux `docker` + `docker-compose-plugin`)
- 16 GB disk free (PostgreSQL data volume ~1 GB, Python image ~3 GB, working data ~8 GB)
- Optional: NVIDIA GPU + `nvidia-container-toolkit` for CUDA inference (MPS works out-of-box on macOS host)

## One-command rebuild

From the repository root:

\`\`\`bash
cd docker
docker compose up --build
\`\`\`

On first run, PostgreSQL loads `db_dump/schema_and_data.sql` (190 MB, ~3 min), then the `giman` service starts Jupyter Lab on http://localhost:8888 .

## Verification

After the stack is up, in a second terminal:

\`\`\`bash
docker compose exec giman python -c "from giman_pipeline.data.db import read_sql; print(read_sql('SELECT count(*) FROM staging.nsd_iss_staging_results'))"
\`\`\`

Expected: `2201` (PPMI NSD-ISS staging row count).

## Teardown

\`\`\`bash
docker compose down -v  # -v removes the pgdata volume
\`\`\`

## Rebuilding specific papers

All Paper 1-10 scripts live in `scripts/` and read from the mounted `giman_research` DB. Example:

\`\`\`bash
docker compose exec giman .venv/bin/python scripts/paper3/run_benchmark.py
\`\`\`
```

- [ ] **A2.5: Smoke-test the stack**

Run:

```bash
cd docker && docker compose up --build -d
sleep 30  # wait for PG init
docker compose exec giman python -c "from giman_pipeline.data.db import read_sql; print(read_sql('SELECT count(*) FROM features.paper1_features_with_targets'))"
docker compose down -v
```

Expected: prints `2201`.

- [ ] **A2.6: Commit**

```bash
git add docker/ outputs/dissertation/appendix_e/E2_docker_rebuild.md
git commit -m "feat(appendix-e): §E.2 Docker compose one-command rebuild"
```

### Task A3: Appendix E LaTeX integration

**Files:**
- Create: `outputs/dissertation/chapters/appendix_e.tex`
- Modify: `outputs/dissertation/main.tex` (add `\input{chapters/appendix_e}` after existing Appendix D)

- [ ] **A3.1: Write Appendix E TeX skeleton**

Create `outputs/dissertation/chapters/appendix_e.tex`:

```latex
\chapter{Reproducibility and Deployment Pipeline}
\label{app:reproducibility}

This appendix packages the infrastructure required for independent verification of every
claim in Chapters 3-16. Contents: (E.1) data dictionary for the \texttt{giman\_research}
PostgreSQL database; (E.2) Docker Compose specification for one-command rebuild.
Sections E.3-E.5 (FastAPI service, NSD-ISS staging API, deterministic reproduction
protocol) are deferred to post-defense deployment.

\section{Data dictionary}
\label{app:e:data-dict}

The full 146-table data dictionary is provided in the supplementary repository at
\texttt{outputs/dissertation/appendix\_e/E1\_data\_dictionary.md} and is auto-generated
from the live PostgreSQL schema via
\texttt{scripts/appendix\_e/generate\_data\_dictionary.py}. Table~\ref{tab:e1-schemas}
summarizes the 10 schemas.

\begin{table}[h]
  \centering
  \caption{PostgreSQL schemas in \texttt{giman\_research} (290 MB, 146 tables).}
  \label{tab:e1-schemas}
  \begin{tabular}{lrl}
    \toprule
    Schema & Tables & Content \\
    \midrule
    \texttt{ppmi\_raw}       & 25 & PPMI clinical/imaging \\
    \texttt{biofind\_raw}    & 23 & BioFIND external validation \\
    \texttt{pdbp\_raw}       & 52 & PDBP external prediction \\
    \texttt{hbs\_raw}        & 11 & HBS external prediction \\
    \texttt{staging}         & 3  & NSD-ISS staging results \\
    \texttt{features}        & 4  & ML feature sets \\
    \texttt{longitudinal}    & 4  & Paper 3 longitudinal staging \\
    \texttt{paper3}          & 1  & Paper 3 longitudinal features \\
    \texttt{mechanistic}     & 21 & Phase 1-4 mechanistic outputs \\
    \texttt{ledd}            & 2  & LEDD + PD medication use \\
    \bottomrule
  \end{tabular}
\end{table}

\section{Docker reproducibility}
\label{app:e:docker}

\texttt{docker/docker-compose.yml} declares two services: PostgreSQL~17 (auto-restoring
from \texttt{db\_dump/schema\_and\_data.sql}) and the \texttt{giman} Python~3.10
environment with PyTorch~2.8, PyTorch Geometric~2.6, MAPIE~1.3, and CatBoost. Committee
verification protocol:

\begin{enumerate}
  \item Clone the repository.
  \item \texttt{cd docker \&\& docker compose up --build}.
  \item Wait for the PostgreSQL restore to complete (approximately 3~minutes).
  \item In a second terminal, execute
    \texttt{docker compose exec giman python -c "..."} to verify that
    \texttt{features.paper1\_features\_with\_targets} contains 2{,}201 rows.
\end{enumerate}

Per-paper rebuild commands are documented in
\texttt{outputs/dissertation/appendix\_e/E2\_docker\_rebuild.md}.
```

- [ ] **A3.2: Wire into main.tex**

Edit `outputs/dissertation/main.tex`: after the existing `\input{chapters/appendix_d}` line (or wherever appendix D is included), add:

```latex
\input{chapters/appendix_e}
```

- [ ] **A3.3: Compile and verify**

Run: `cd outputs/dissertation && latexmk -pdf main.tex`
Expected: `main.pdf` builds with Appendix E present in TOC and LOT.

- [ ] **A3.4: Commit**

```bash
git add outputs/dissertation/chapters/appendix_e.tex outputs/dissertation/main.tex
git commit -m "docs(dissertation): integrate Appendix E (reproducibility) into main.tex"
```

---

## Subsystem B: Chapter 16 — Paper 11 Cross-Cohort Generalization (Weeks 2-9) — PHASE-LEVEL

**Goal:** Deploy Paper 3 Graph-DT checkpoints inductively on BioFIND (n=118 PD), PDBP (n=893 PD), HBS (n=649 PD) to close the PPMI-only training limitation.

**Why inductive:** GAT is inherently inductive (Veličković 2018). The 5 Graph-DT checkpoints at `outputs/paper3_checkpoints/graph_dt/fold{0-4}_graph_dt.pt` were trained on 1,900 PPMI patients; deploying on external cohorts requires nearest-neighbor graph extension to the training graph, NOT retraining.

**Before starting this subsystem:** Write detailed TDD sub-plan `docs/superpowers/plans/2026-04-14-ch16-paper11-cross-cohort.md` per writing-plans conventions.

### Task B1 (Week 2-3): Feature alignment + inductive graph extension

**Spec:**
- Enumerate the 12 common features across PPMI + BioFIND + PDBP (and the 8/12 subset for HBS — missing UPDRS1, UPDRS2, UPDRS4, MOCA, ESS).
- Extend `src/giman_pipeline/paper3/graph_digital_twin.py` with an `inductive_predict(checkpoint, external_features)` method: given a loaded checkpoint, compute k-NN edges from external patients to `node_baseline` (PPMI training graph), concatenate, run forward pass, return per-patient CIF.
- Add unit test: round-trip a held-out PPMI fold through `inductive_predict` and verify ΔC-td < 0.01 vs the original in-graph prediction.

**Deliverables:**
- `src/giman_pipeline/paper11/__init__.py`
- `src/giman_pipeline/paper11/inductive.py` (`inductive_predict`)
- `src/giman_pipeline/paper11/feature_align.py` (12-feature and 8-feature alignment)
- `tests/paper11/test_inductive.py` (round-trip regression test)

### Task B2 (Week 4-5): BioFIND validation (NSD+ subgroup, n=103)

**Spec:**
- BioFIND has NSD-ISS ground truth (Russo 2025 replication, n=103 S+ PD). Deploy both DeepHit and Graph-DT on this subgroup.
- Primary metric: NSD+ subgroup C-index (Paper 1 AUC = 0.900 was on PPMI NSD+; expected similar order on BioFIND if training generalizes).
- Secondary: transition-timing coverage at 90% CL via Paper 4 conformal module (`src/giman_pipeline/paper4/conformal_survival.py`).

**Deliverables:**
- `scripts/paper11/01_biofind_validation.py`
- `outputs/paper11/biofind/` (JSON metrics, reliability diagrams)

### Task B3 (Week 6-7): PDBP transfer (prevalent PD, n=893)

**Spec:**
- PDBP is a **prevalent** PD cohort (not de novo). This is a domain-shift stress test.
- Deploy Graph-DT 12-feature model. Report C-index, IBS, per-transition C-td.
- Document the domain shift (PPMI de novo → PDBP prevalent) with KS + PSI tests on the 12 features.
- Follow Paper 5 temporal validation protocol (covariate shift detection) from `outputs/paper5/`.

**Deliverables:**
- `scripts/paper11/02_pdbp_validation.py`
- `outputs/paper11/pdbp/`
- `scripts/paper11/02b_covariate_shift_pdbp.py`

### Task B4 (Week 8): HBS subset (n=649, 8/12 features)

**Spec:**
- HBS has only 8/12 common features (missing UPDRS1/2/4, MOCA, ESS).
- Retrain the **8-feature** Graph-DT from PPMI → deploy on HBS. This is a separate 5-fold checkpoint set saved to `outputs/paper11/checkpoints_8feat/`.
- Compare HBS 8-feat C-index against PPMI 12-feat baseline to quantify low-data deployment floor.

**Deliverables:**
- `scripts/paper11/03_hbs_validation.py`
- `outputs/paper11/hbs/`
- `outputs/paper11/checkpoints_8feat/fold{0-4}_graph_dt_8feat.pt`

### Task B5 (Week 9): Pooled meta-analysis + manuscript

**Spec:**
- Pool C-indices across cohorts using random-effects meta-analysis (metafor-style). Report I² heterogeneity.
- Write Chapter 16 as a standalone Paper 11 manuscript (targeting *Brain* or *npj Digital Medicine*). Structure:
  - §16.1 Motivation: PPMI-only training as field-wide limitation
  - §16.2 Methods: 12-feature alignment, inductive deployment, statistical-equivalence framework
  - §16.3-§16.5: BioFIND / PDBP / HBS results
  - §16.6 Pooled meta-analysis
  - §16.7 Discussion + deployment recommendations
- 8-12 figures following Paper 3 / Paper 4 visual language.
- Ingest into `outputs/dissertation/chapters/ch16_paper11.tex` via `scripts/dissertation/ingest_standalone_papers.py`.

**Deliverables:**
- `scripts/paper11/04_pooled_meta_analysis.py`
- `scripts/paper11/05_generate_figures.py`
- `outputs/paper11/manuscript/main.tex`
- `outputs/paper11/figures/` (8-12 PNG + PDF)
- `outputs/dissertation/chapters/ch16_paper11.tex`
- Update `outputs/dissertation/bibliography.tex` with new `\bibitem{}` entries (follow the hand-curated `\bibitem{}` convention, NOT natbib — see CLAUDE.md Gotcha about bibliography format).

---

## Subsystem C: Chapter 11 §11.7 — 6-region ROI split (Weeks 10-13) — PHASE-LEVEL

**Goal:** Close the Phase 8b spatial-resolution limitation by splitting putamen into anterior/posterior subregions (6 ROIs total: bilateral caudate + anterior putamen + posterior putamen) and re-running the Phase 2 coupled ODE calibration.

**Why it matters:** Paper 8b (Chapter 11) reports spatial propagation is NOT detectable with whole-putamen SBR (M1 wins, ΔAIC = 3,856). Reviewer concern: this could be an artifact of coarse ROI definition. Dzialas 2025 reports anterior putamen declines 4%/yr vs posterior 6%/yr — our 3.29%/yr whole-striatum median is consistent with the population-weighted average but should be stratified.

**Before starting this subsystem:** Write detailed TDD sub-plan `docs/superpowers/plans/2026-04-14-ch11-regional-roi-split.md`.

### Task C1 (Week 10): Extract 6-region SBR from raw DaT-SPECT

**Spec:**
- Query `ppmi_raw.datscan_sbr_analysis` for per-ROI SBR (6 regions × left + right × 2,137 patients × 1-5 scans).
- Verify column mapping: `SBR_CAUDATE_R`, `SBR_CAUDATE_L`, `SBR_PUTAMEN_ANTERIOR_R`, `SBR_PUTAMEN_ANTERIOR_L`, `SBR_PUTAMEN_POSTERIOR_R`, `SBR_PUTAMEN_POSTERIOR_L` (check actual column names in DB; LONI IDA naming may differ).
- Write `scripts/paper8b_regional/01_extract_6region_sbr.py` producing `outputs/mechanistic_twin/regional_6roi/sbr_6region_longitudinal.parquet`.

### Task C2 (Week 11-12): Re-run Phase 2 coupled calibration per-region

**Spec:**
- Reuse `src/mechanistic_twin/` Julia infrastructure from Phase 2.
- Fit the coupled α-syn + N(t) ODE **per region** (6 separate fits per patient).
- Output: per-patient × per-region posterior samples (HDF5), per-region decay rate summaries.
- Expected regional decay rates (Dzialas 2025):

| Region | Expected %/yr |
|---|---|
| Caudate (bilateral) | 2.0-3.0 |
| Anterior putamen (bilateral) | 3.5-4.5 |
| Posterior putamen (bilateral) | 5.0-6.5 |

### Task C3 (Week 13): Update §11.7 narrative + figures

**Spec:**
- Write `outputs/dissertation/chapters/ch11_paper8b.tex` §11.7 "Anterior/posterior putamen sub-gradient" (3-4 pages).
- Include 3-4 new figures: per-region decline slopes, sub-gradient boxplot, within-patient anterior-vs-posterior slope correlation, revised spatial propagation test (does anterior→posterior propagation hold when split?).
- **Honest reporting:** if M1 still wins at 6-region resolution, SAY SO — this would strengthen the chapter, not weaken it.
- Add `\bibitem{dzialas2025}` to `outputs/dissertation/bibliography.tex` if not already present.

---

## Subsystem D: Chapter 13 §13.8 + Ch 14/15 refresh (Weeks 14-16) — PHASE-LEVEL

### Task D1 (Weeks 14-15): Ch 13 §13.8 — Mechanistic conformal bands on counterfactuals

**Goal:** Promote Paper 10 NASEM uncertainty quantification score from 3 to "regulatorily MIDD-ready" by applying Paper 4 conformal framework to the Phase 5 Task 6 observational counterfactual predictions.

**Why:** Paper 10 Task 6 showed calibration slope 1.074 [0.88, 1.29] (passes), but the NASEM auditor flagged the absence of formal coverage intervals on counterfactual trajectories. Paper 4 conformal module (`src/giman_pipeline/paper4/conformal_survival.py`) is already validated with 91.1% coverage at 95% CL on Paper 3 survival predictions — reusing it for mechanistic counterfactuals requires only the IPCW weight adaptation for severity-controlled treatment effects.

**Before starting:** Write detailed TDD sub-plan `docs/superpowers/plans/2026-04-14-ch13-mech-conformal.md`.

**Deliverables:**
- `src/giman_pipeline/paper10/conformal_counterfactual.py` (adapts Paper 4 `CauseSpecificConformal` to mechanistic predictions)
- `scripts/paper10/task8_conformal_bands.py`
- `outputs/mechanistic_twin/paper10_mech_vs_giman/conformal_bands/` (per-patient CI trajectories)
- `outputs/dissertation/chapters/ch13_paper10.tex` §13.8 (3-4 pages, 2-3 figures)
- Updated NASEM audit: Uncertainty Quantification criterion moves from 3 → 3+ (substantial+).

### Task D2 (Week 16): Ch 14 Discussion + Ch 15 Conclusion refresh

**Spec:**
- Ch 14 §14.3 "Complementarity not competition" — add GDT-5 inductive-deployment evidence from Ch 16 pooled meta-analysis.
- Ch 14 §14.4 "Limitations" — update to mark PPMI-only training as CLOSED (Ch 16), whole-putamen ROI as CLOSED (§11.7), mechanistic UQ as CLOSED (§13.8).
- Ch 15 §15.3 F1-F15 — mark which are CLOSED in dissertation vs DEFERRED to postdoc. Per completion mapping: F1-F11 CLOSED (Papers 1-10 + 11), F12 (MindMend Phase 6) DEFERRED, F13 (DeNoPa external val) DEFERRED, F14 (prospective trial) DEFERRED, F15 (reproducibility) CLOSED (App E).

**Deliverables:**
- Edited `outputs/dissertation/chapters/ch14_discussion.tex`
- Edited `outputs/dissertation/chapters/ch15_conclusion.tex`

---

## Week 17: Final compile + presubmit

- [ ] **W17.1: Full LaTeX build**

Run: `cd outputs/dissertation && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: `main.pdf` builds without errors (warnings acceptable). TOC includes Ch 1-16 + App A-E.

- [ ] **W17.2: Check refs + critique-manuscript pass**

Run the `check-refs` skill on `outputs/dissertation/main.tex`.
Expected: 0 undefined references, 0 dangling `\cite{}`.

- [ ] **W17.3: Re-run `/deep-review:synthesizer` on full dissertation**

Spawn deep-review synthesizer agent across Chapters 1-16 and Appendices A-E. Target: zero red flags, all yellow flags acknowledged in the discussion.

- [ ] **W17.4: Presubmit defensibility scorer**

Run: `.venv/bin/python scripts/defense_prep/99_defensibility_scorer.py --full`
Expected: every chapter scored ≥ 3.5/5 on claim-coverage, citation-coverage, code-coverage, data-lineage.

- [ ] **W17.5: Build defense slides**

Create `outputs/dissertation/defense/slides.tex` using Beamer. 40 slides, 12-minute walkthrough per Chapter 1-16 + 3 minutes for Q&A prep.

- [ ] **W17.6: Tag release**

```bash
git tag -a dissertation-v1.0 -m "Final dissertation ready for defense"
git push pd_phd dissertation-v1.0
```

---

## Self-Review

**Spec coverage:**
- ✅ Week 1 Appendix E Docker (A1-A3)
- ✅ Weeks 2-9 Chapter 16 Paper 11 Cross-Cohort (B1-B5)
- ✅ Weeks 10-13 Chapter 11 §11.7 Regional ROI (C1-C3)
- ✅ Weeks 14-15 Chapter 13 §13.8 Mechanistic Conformal (D1)
- ✅ Week 16 Ch 14/15 refresh (D2)
- ✅ Week 17 Final compile + presubmit (W17.1-W17.6)

**Subsystem decomposition:** Each of the 4 subsystems (A Docker, B Ch 16, C §11.7, D §13.8+refresh) is independent and testable on its own. Detailed TDD sub-plans will be written for B, C, D when each phase starts.

**Placeholder scan:** No TBDs. Week-level tasks for B/C/D carry explicit specs and deliverables. Sub-plans will add step-level TDD granularity.

**Type consistency:** `inductive_predict` signature defined in Task B1 is referenced consistently in B2/B3/B4.

---

## Execution Handoff

Plan complete and saved to `Docs/superpowers/plans/2026-04-13-dissertation-completion-execution.md`.

Recommend **Inline Execution** for Subsystem A (Week 1 Docker) — 3 tasks × ~5 steps each, reversible, needs committee verification ASAP. Switch to **Subagent-Driven** when starting Subsystem B (Week 2), since each B task is a 1-2 week research block that benefits from fresh subagent context per task.

Next action after plan save: update CLAUDE.md with completion roadmap pointer, then start Task A1.
