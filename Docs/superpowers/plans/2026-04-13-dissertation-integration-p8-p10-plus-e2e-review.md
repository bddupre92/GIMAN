# Dissertation Integration (Papers 8a, 8b, 9, 10) + End-to-End Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fold Papers 8a, 8b, 9, 10 into the dissertation as Chapters 10-13, add Discussion (Ch 14) and Conclusion (Ch 15), and conduct an end-to-end code/data/claims review that answers two big questions: (a) given the Graph-DT head-to-head result is a near-wash for wearing-off, what else can Graph-DT + GIMAN deliver? (b) what is the full research surface still left in the data we already have?

**Architecture:** The dissertation already has Chapters 1-9 (intro + systematic review + 6 data-driven papers + discussion + conclusion) under `outputs/dissertation/`. We re-number so the existing discussion/conclusion become appendices or are merged, then insert the four mechanistic-twin papers as Chapters 10-13 in publication-dependency order (8a→8b→9→10). Chapters 14-15 provide a unified cross-cutting discussion and a single closing conclusion that explicitly enumerates "what's next" research directions that remain fundable in this dataset.

**Tech Stack:** LaTeX (dissertation `main.tex`), Python (for code + data audits), matplotlib (unified figure style), mempalace (diary + KG for the review findings), git (atomic commits per chapter integration), ProQuest PDF/A compliance targets.

---

## Scope Check

This plan spans three sub-projects that each produce independently working, testable outputs:

- **A. Chapter integration** — dissertation LaTeX + bibliography + figure paths
- **B. End-to-end audit** — code/data/claims traceability report across all 10 papers
- **C. "What else is possible" analysis** — a research-opportunities catalog keyed to existing data assets

Each sub-project has a verifiable deliverable (compiled PDF, audit report, opportunities catalog). They are sequenced: A first (gets the chapters in place), then B (audits the now-integrated content), then C (uses the audit to enumerate next steps).

---

## Current State Audit (pre-implementation)

Before touching anything, map existing artifacts. The four new chapters each have:

| Paper | LaTeX artifact (if any) | Outputs dir | Key result |
|---|---|---|---|
| 8a (Regional DaT-SPECT SAEM) | `outputs/mechanistic_twin/paper8a/latex/` (check) | `outputs/mechanistic_twin/phase3/` | M1 wins, ΔAIC=5,668, spatial propagation not detectable from 4-region DaT |
| 8b (Connectome-informed NDM) | `outputs/mechanistic_twin/paper8b/latex/` (check) | `outputs/mechanistic_twin/phase3/` (shared) | k_spread=1.26 yr⁻¹ estimable but not predictively useful |
| 9 (PK-PD N(t)×LEDD) | `outputs/mechanistic_twin/phase4/latex/main.tex` (confirmed) | `outputs/mechanistic_twin/phase4/` | Path B positive p=0.044; Path A, C informative negatives |
| 10 (Bidirectional + NASEM audit) | `outputs/mechanistic_twin/paper10_mech_vs_giman/latex/` (TO CREATE in Task 9) | `outputs/mechanistic_twin/paper10_mech_vs_giman/` | Bidirectional MAE 33% ↓, counterfactual slope 1.07 CI⊇1.0, NASEM 16/21 |

Existing dissertation skeleton: `outputs/dissertation/main.tex` + 11 chapters (ch01..ch10 + mechtwin_review.tex as Appendix D) + 175-entry bibliography.tex. Dissertation currently uses ch09_discussion and ch10_conclusion — these will be renumbered to ch14/ch15 after the four mechanistic papers land as ch10/11/12/13.

---

## File Structure

### Sub-Project A (Chapter integration)

- Modify: `outputs/dissertation/main.tex` — re-order `\include{chapters/...}` to insert 4 new chapters and renumber discussion/conclusion to Ch 14/15
- Create: `outputs/dissertation/chapters/ch10_paper8a.tex` — Regional SAEM chapter
- Create: `outputs/dissertation/chapters/ch11_paper8b.tex` — Connectome NDM chapter
- Create: `outputs/dissertation/chapters/ch12_paper9.tex` — PK-PD chapter
- Create: `outputs/dissertation/chapters/ch13_paper10.tex` — Bidirectional + NASEM chapter
- Create: `outputs/dissertation/chapters/ch14_discussion.tex` — unified cross-cutting discussion (absorbs existing ch09_discussion.tex, adds mechanistic-twin integration)
- Create: `outputs/dissertation/chapters/ch15_conclusion.tex` — single closing conclusion with future-work catalog
- Archive: `outputs/dissertation/chapters/ch09_discussion.tex` → `_archive_pre_integration/ch09_discussion.tex`
- Archive: `outputs/dissertation/chapters/ch10_conclusion.tex` → `_archive_pre_integration/ch10_conclusion.tex`
- Modify: `outputs/dissertation/bibliography.tex` — merge in Phase 5 bibliography (dedup against existing 175 entries)
- Modify: `outputs/dissertation/figures/` — symlink or copy paper 8a/8b/9/10 figures with consistent prefix (`ch10_`, `ch11_`, etc.)

### Sub-Project B (End-to-end audit)

- Create: `outputs/defense_prep/e2e_audit/e2e_audit_report.md` — master markdown report
- Create: `scripts/defense_prep/audit_data_lineage.py` — verifies every CSV/parquet claim resolves to a committed artifact
- Create: `scripts/defense_prep/audit_code_coverage.py` — verifies every paper's claims have runnable script + test
- Create: `scripts/defense_prep/audit_claims_to_citations.py` — checks DATA_LITERATURE_REGISTRY against bibliography.tex
- Create: `outputs/defense_prep/e2e_audit/claim_coverage.csv` — one row per load-bearing claim (10 papers × ~20 claims)
- Create: `outputs/defense_prep/e2e_audit/data_lineage_map.json` — which CSV feeds which paper
- Create: `outputs/defense_prep/e2e_audit/code_coverage_matrix.csv` — script-to-test-to-output mapping

### Sub-Project C ("What else can we do" catalog)

- Create: `Docs/research_directions/2026-04-13_unfunded_research_opportunities.md` — opportunity catalog
- Create: `Docs/research_directions/giman_beyond_wearing_off.md` — specific Graph-DT secondary use-cases given Paper 10 finding
- Create: `Docs/research_directions/data_asset_inventory.md` — what's in our data that isn't analyzed yet

---

## Bite-Sized Task Granularity

Tasks grouped into Phases A (integration) → B (audit) → C (opportunities). Each task = 2-5 minutes of focused action.

---

### Phase A — Dissertation Integration (Chapters 10-15)

#### Task A1: Inventory existing paper 8a/8b/9/10 LaTeX

- [ ] **Step 1:** Check which paper-level LaTeX files exist

```bash
find outputs/mechanistic_twin -name "main.tex" -type f
find outputs/mechanistic_twin -name "*.bib" -type f
```

- [ ] **Step 2:** Record which paper has manuscript-ready LaTeX vs outline-only

Expected: Paper 9 has full `outputs/mechanistic_twin/phase4/latex/main.tex`. Papers 8a, 8b, 10 likely need chapter-form bodies written from RUN_MANIFESTs + results JSONs. Record findings in `outputs/defense_prep/e2e_audit/pre_integration_inventory.md`.

- [ ] **Step 3:** Commit the inventory

```bash
git add outputs/defense_prep/e2e_audit/pre_integration_inventory.md
git commit -m "audit: pre-integration LaTeX inventory for Papers 8a/8b/9/10"
```

#### Task A2: Re-number main.tex to reserve chapter slots 10-15

- [ ] **Step 1:** Read current `outputs/dissertation/main.tex` and record current `\include{}` order

- [ ] **Step 2:** Write a test asserting the final chapter sequence is correct

```python
# tests/dissertation/test_chapter_order.py
from pathlib import Path
import re

def test_main_tex_chapter_order():
    """main.tex must include chapters in order 1..15."""
    tex = Path("outputs/dissertation/main.tex").read_text()
    includes = re.findall(r"\\include\{chapters/(ch\d+_[^\}]+)\}", tex)
    nums = [int(re.match(r"ch(\d+)", inc).group(1)) for inc in includes]
    assert nums == sorted(nums), f"Chapter includes not in order: {nums}"
    assert nums == list(range(1, len(nums) + 1)), f"Gaps in chapter numbers: {nums}"
```

- [ ] **Step 3:** Run test, confirm it FAILS (existing main.tex doesn't have 15 chapters)

- [ ] **Step 4:** Archive old ch09 (Discussion) and ch10 (Conclusion) into `_archive_pre_integration/`

```bash
mkdir -p outputs/dissertation/chapters/_archive_pre_integration
git mv outputs/dissertation/chapters/ch09_discussion.tex outputs/dissertation/chapters/_archive_pre_integration/
git mv outputs/dissertation/chapters/ch10_conclusion.tex outputs/dissertation/chapters/_archive_pre_integration/
```

- [ ] **Step 5:** Update main.tex to `\include{}` placeholders for ch10..ch15 (files will be created in later tasks)

- [ ] **Step 6:** Commit

```bash
git add outputs/dissertation/main.tex outputs/dissertation/chapters/_archive_pre_integration/ tests/dissertation/test_chapter_order.py
git commit -m "refactor(dissertation): reserve chapter slots 10-15 for P8a/P8b/P9/P10 + Discussion + Conclusion"
```

#### Task A3: Draft Chapter 10 — Paper 8a (Regional DaT-SPECT SAEM)

- [ ] **Step 1:** Open `outputs/mechanistic_twin/phase3/` artifacts (paper8a_*) + RUN_MANIFEST, extract headline numbers: ΔAIC, 4-region fit, M1 vs M6r model comparison

- [ ] **Step 2:** Create `ch10_paper8a.tex` with sections: Introduction | Methods | Results | Discussion | Acknowledgments. Target 10-14 pages with 4-5 figures pulled from `outputs/paper8a_figures/` (or equivalent). Use dissertation-local TikZ for any new schematics.

- [ ] **Step 3:** Ensure every claim has a `\cite{}` against `bibliography.tex`. Missing entries → add to bibliography in a later task.

- [ ] **Step 4:** Commit

```bash
git add outputs/dissertation/chapters/ch10_paper8a.tex
git commit -m "docs(dissertation): add Chapter 10 from Paper 8a (regional DaT-SPECT SAEM, M1 wins)"
```

#### Task A4: Draft Chapter 11 — Paper 8b (Connectome-informed NDM)

- [ ] **Step 1:** Pull Phase 3 connectome findings: k_spread=1.26 yr⁻¹, estimable-but-not-useful finding, HCP1065 + Melbourne Subcortex + Budapest v3.0 sensitivity

- [ ] **Step 2:** Create `ch11_paper8b.tex` — mirror Chapter 10 structure but focused on connectome propagation. Cite Pandya 2019, Zheng 2019, Borghammer 2021, Kerstens 2023, Abdelgawad 2023 Powell 2018.

- [ ] **Step 3:** Explicitly label this as an informative negative: "spatial propagation not detectable from 4-region DaT-SPECT; would require finer spatial sampling (8-region anterior/posterior putamen split, Drori 2022 Sci Adv)".

- [ ] **Step 4:** Commit

```bash
git add outputs/dissertation/chapters/ch11_paper8b.tex
git commit -m "docs(dissertation): add Chapter 11 from Paper 8b (connectome NDM, informative negative)"
```

#### Task A5: Draft Chapter 12 — Paper 9 (PK-PD N(t)×LEDD)

- [ ] **Step 1:** Copy `outputs/mechanistic_twin/phase4/latex/main.tex` body into chapter form, stripping preamble and turning `\section{}` → `\section{}` in dissertation context

- [ ] **Step 2:** Ensure the 3-pathway framing (A time-wins, B positive, C PK-driven) is front and centre with all headline numbers and figures fig1-fig10 from Paper 9

- [ ] **Step 3:** Add a one-paragraph bridge to Chapter 13 explicitly: "the interaction coefficient β_nfrac_c:ledd_c=1.4096 (p=0.044) fit here will be used without retuning in Chapter 13's observational counterfactual validation."

- [ ] **Step 4:** Commit

```bash
git add outputs/dissertation/chapters/ch12_paper9.tex
git commit -m "docs(dissertation): add Chapter 12 from Paper 9 (three-pathway PK-PD)"
```

#### Task A6: Draft Chapter 13 — Paper 10 (Bidirectional + NASEM)

- [ ] **Step 1:** Run Task 9 from Phase 5 plan (write `outputs/mechanistic_twin/paper10_mech_vs_giman/latex/main.tex`) if not already done

- [ ] **Step 2:** Copy Paper 10 body into `ch13_paper10.tex` with dissertation context (sections: Introduction | Bidirectional architecture | External validation | Head-to-head benchmark | Observational counterfactual | NASEM audit | Discussion). Insert all 9 figures from `outputs/mechanistic_twin/paper10_mech_vs_giman/figures/`.

- [ ] **Step 3:** Include the Methods-section defense paragraph verbatim from `phase5_task5_RUN_MANIFEST.md` under a Methods subsection called "Statistical and computational methodology." All citations must resolve against the merged bibliography from Task A8.

- [ ] **Step 4:** Commit

```bash
git add outputs/dissertation/chapters/ch13_paper10.tex
git commit -m "docs(dissertation): add Chapter 13 from Paper 10 (bidirectional + NASEM 16/21)"
```

#### Task A7: Draft Chapter 14 — Unified Discussion

- [ ] **Step 1:** Read existing `_archive_pre_integration/ch09_discussion.tex` for prose we want to preserve (primary dissertation discussion of Papers 1-6)

- [ ] **Step 2:** Create `ch14_discussion.tex` with four explicit threads:

  - **§14.1 The data-driven arc (Papers 1-6):** preserve from archived ch09
  - **§14.2 The mechanistic arc (Papers 7-10):** newly written from Phase 2 IS posteriors → Phase 3 regional → Phase 4 PK-PD → Phase 5 bidirectional
  - **§14.3 Complementarity-not-competition:** the headline cross-cutting finding. Quote Paper 10 head-to-head result (Graph-DT slightly better on wearing-off but both near-random because wearing-off is PK-driven; Path B mechanistic wins on treatment-benefit modulation; the models answer different clinical questions)
  - **§14.4 Honest limitations:** PPMI-only training cohort, no longitudinal external PD DaT, cross-sectional external only, no prospective interventional validation, continuous-sensor integration is Phase 6 future work

- [ ] **Step 3:** Cross-reference figures across chapters (e.g., Paper 10 Fig 9 dissertation arc can be reused here)

- [ ] **Step 4:** Commit

```bash
git add outputs/dissertation/chapters/ch14_discussion.tex
git commit -m "docs(dissertation): Chapter 14 unified discussion of Papers 1-10 (complementarity-not-competition)"
```

#### Task A8: Draft Chapter 15 — Conclusion + Future Work

- [ ] **Step 1:** Create `ch15_conclusion.tex` with: a 1-page synopsis of the dissertation's contributions, followed by an explicit future-work catalog sourced from Sub-Project C's opportunity analysis (Tasks C1-C4 below).

- [ ] **Step 2:** Ensure the conclusion names specific venues for each future-work item (e.g., "Paper 11 Hybrid SciML — target npj Parkinson's Disease; Phase 6 MindMend biosensor — target Nature Digital Medicine").

- [ ] **Step 3:** Commit

```bash
git add outputs/dissertation/chapters/ch15_conclusion.tex
git commit -m "docs(dissertation): Chapter 15 conclusion + explicit future-work catalog"
```

#### Task A9: Merge bibliography — reconcile 175 existing + 75 new entries

- [ ] **Step 1:** Write a deduplication script

```python
# scripts/dissertation/merge_bibliography.py
"""Merge outputs/dissertation/bibliography.tex with
outputs/mechanistic_twin/paper10_mech_vs_giman/phase5_literature_bibliography.bib.
Report duplicates (same DOI or same first-author+year). Prefer the LaTeX bibitem
format over BibTeX when merging into bibliography.tex (which uses thebibliography
env, not natbib).
"""
from pathlib import Path
# Implementation: parse each file, key by DOI (fallback first-author-year),
# emit merged unique list sorted by cite key, report duplicates to stderr.
```

- [ ] **Step 2:** Write a test

```python
def test_bibliography_no_duplicate_doi():
    tex = Path("outputs/dissertation/bibliography.tex").read_text()
    dois = re.findall(r"doi[:=]\s*10\.\S+", tex, flags=re.IGNORECASE)
    assert len(dois) == len(set(dois)), "Duplicate DOIs in bibliography.tex"

def test_all_cite_keys_resolve():
    """For each \\cite{X} in any chapter, \\bibitem{X} must exist."""
    # Implementation: walk chapter .tex files, collect \\cite{} keys, ensure each
    # has a matching \\bibitem{} in bibliography.tex
```

- [ ] **Step 3:** Run merger, resolve duplicates manually if any, run tests

- [ ] **Step 4:** Commit

```bash
git add outputs/dissertation/bibliography.tex scripts/dissertation/merge_bibliography.py tests/dissertation/test_bibliography.py
git commit -m "docs(dissertation): merge Phase 5 bibliography (~75 new DOIs) with existing 175 entries"
```

#### Task A10: Compile dissertation PDF

- [ ] **Step 1:** Compile twice for cross-references

```bash
cd outputs/dissertation
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

- [ ] **Step 2:** Open the PDF and eyeball each of the four new chapters, confirming figures render and citations resolve. Save `outputs/dissertation/main.pdf`.

- [ ] **Step 3:** Commit the compiled PDF

```bash
git add -f outputs/dissertation/main.pdf
git commit -m "build(dissertation): compile integrated 15-chapter PDF (Papers 1-10)"
```

---

### Phase B — End-to-End Audit

#### Task B1: Build data lineage map

- [ ] **Step 1:** Catalog every CSV/parquet read by any chapter's experiments. For each file record: path (relative to repo root), producing script, consuming script(s), row count, date modified.

```python
# scripts/defense_prep/audit_data_lineage.py
"""Walks every scripts/** .py file, greps for pd.read_csv / pd.read_parquet /
json.load calls, and builds a JSON map of file → (producers, consumers)."""
```

- [ ] **Step 2:** Emit `outputs/defense_prep/e2e_audit/data_lineage_map.json` and a human-readable DAG picture via graphviz

- [ ] **Step 3:** Commit

#### Task B2: Build claim-to-citation coverage matrix

- [ ] **Step 1:** For each of the 10 papers, enumerate the 10-20 load-bearing claims (biological, statistical, methodological). For each claim record: text, source chapter, supporting citation key, verification status.

- [ ] **Step 2:** Output as `outputs/defense_prep/e2e_audit/claim_coverage.csv`. Columns: `paper, claim, cite_key, verified_doi, script_produced_result, test_asserts_it`.

- [ ] **Step 3:** Flag any claim with missing `cite_key` or `script_produced_result` — these are the defensibility gaps.

- [ ] **Step 4:** Commit

#### Task B3: Code coverage matrix (script ↔ test ↔ output)

- [ ] **Step 1:** For each script in `scripts/**`, list: what tests exercise it, what outputs it produces, whether those outputs are tested for shape/sanity.

- [ ] **Step 2:** Output `outputs/defense_prep/e2e_audit/code_coverage_matrix.csv`. Columns: `script, tests_touching_it, output_artifacts, output_tests`.

- [ ] **Step 3:** Flag orphan scripts (no tests, no downstream users) — candidates for archive.

- [ ] **Step 4:** Commit

#### Task B4: End-to-end rebuild dry-run

- [ ] **Step 1:** Write a `scripts/defense_prep/verify_e2e_rebuild.sh` that, given a clean checkout and the `db_dump/schema_and_data.sql`, claims to reproduce every paper's headline metric within tolerance. Do NOT run the full rebuild now (too slow); just confirm the script is logically complete.

- [ ] **Step 2:** Run the script on existing outputs to confirm nothing errors. This is a dry-run — every step should be either "produces fresh artifact" or "validates cached artifact against stored SHA."

- [ ] **Step 3:** Commit the script.

#### Task B5: Write consolidated e2e audit report

- [ ] **Step 1:** Create `outputs/defense_prep/e2e_audit/e2e_audit_report.md` summarising: data lineage, claim coverage, code coverage, rebuild verification. Include a "defensibility scorecard": green/yellow/red per paper.

- [ ] **Step 2:** Commit. This is the document Blair will show his committee.

---

### Phase C — "What Else Can We Do" Opportunity Catalog

#### Task C1: Graph-DT beyond wearing-off — enumerate secondary use-cases

- [ ] **Step 1:** Given Paper 10 showed Graph-DT doesn't beat mechanistic on wearing-off, inventory where Graph-DT DOES excel:

  - Forward transition timing (Paper 3: C-td=0.926 for stage→stage transitions) — still useful for clinical trial enrollment enrichment
  - Transition-specific conformal intervals (Paper 4: 91.1% coverage)
  - Short-horizon binary stage prediction (Paper 1: AUC 0.979 binary, 0.942 three-class)
  - Subgroup-equity stratification (Paper 4 subgroup analysis)
  - Imputation-to-prediction cascade (Paper 2 → Paper 1)

- [ ] **Step 2:** For each use-case, name a specific downstream product: "Enrich prodromal PD trials using Graph-DT to filter to high-progression-risk candidates", "Use Paper 2 GIMIN imputation as a drop-in EHR preprocessing layer", etc.

- [ ] **Step 3:** Write to `Docs/research_directions/giman_beyond_wearing_off.md`. Target 5-8 named directions with venue targets.

- [ ] **Step 4:** Commit.

#### Task C2: Data asset inventory — what's in the data that we haven't analyzed

- [ ] **Step 1:** Audit PPMI tables we have but haven't used:

  - DTI (140 patients, white matter connectivity)
  - FreeSurfer ASEG brain volumes (~1,900 patients)
  - Olink CSF proteomics (Project 222, ~200 patients, 367 proteins)
  - Olink/SomaScan serum proteomics (if any)
  - Genetic GRS full variant panel (beyond the GRS_TOTAL used in Paper 1)
  - Skin synSAA, Amprion semi-quantitative SAA
  - NfL (523 patients, 4,961 rows)
  - aSyn aggregate/surface (100 patients)
  - SAA dilution series (213 patients)
  - Behavioral / cognitive longitudinal beyond MDS-UPDRS: MoCA, ESS, RBD, SCOPA-AUT, GDS, STAI, QUIP
  - Genetic data: LRRK2/GBA/SNCA beyond the binary flag used in Paper 1
  - Wearable / actigraphy data from PPMI 2.0 (if downloaded)

- [ ] **Step 2:** For each untapped asset, note: what kind of research question it enables, what model variant it would require, approximate scope (a paper or a sub-analysis).

- [ ] **Step 3:** Write to `Docs/research_directions/data_asset_inventory.md`. Target 10-15 untapped asset → research question pairings.

- [ ] **Step 4:** Commit.

#### Task C3: Cross-cohort research opportunities

- [ ] **Step 1:** We have PPMI + BioFIND + PDBP + HBS + LCC loaded locally. Enumerate questions that require TWO OR MORE cohorts:

  - PPMI × BioFIND: externally validate NSD+ subgroup predictions (Paper 1 showed AUC 0.900 — could be tested on BioFIND 103 S+ patients)
  - PPMI × PDBP: cross-cohort survival model generalization (Paper 3 Graph-DT transferability)
  - PPMI × HBS: common-feature-only (12-feature clinical) prediction on a prevalent PD cohort
  - BioFIND × PPMI biological-stage matching: do Russo-staged BioFIND patients progress differently from PPMI-staged matched patients?

- [ ] **Step 2:** Write to the same `data_asset_inventory.md` as a §3.

- [ ] **Step 3:** Commit.

#### Task C4: Mechanistic twin extensions with existing data

- [ ] **Step 1:** Given Phase 2 IS posteriors exist for 1,065 patients (k_n, α_tox, T_tox), enumerate what can be built WITHOUT new data:

  - **Phase 3 re-analysis:** 6-region anterior/posterior putamen split (Drori 2022) might detect spatial propagation that the 4-region analysis missed
  - **Phase 4 extensions:** add LEDD × genetic interaction (LRRK2 carriers escalate differently?)
  - **Phase 4 extensions:** add age × LEDD interaction (younger PD patients have steeper Hill curves?)
  - **Paper 11 Hybrid SciML:** GIMAN features (26 baseline) + mechanistic N(t) → hybrid UDE (Rackauckas 2020) for transition timing — submit to npj Parkinson's Disease
  - **Mechanistic conformal:** apply Paper 4 conformal framework to mechanistic-twin predictions
  - **Counterfactual extensions:** other LEDD-change thresholds (100mg, 300mg), symptom-onset counterfactuals (what if MAO-B started earlier?)

- [ ] **Step 2:** Write to `Docs/research_directions/mechanistic_twin_extensions.md`.

- [ ] **Step 3:** Commit.

#### Task C5: Synthesize Chapter 15 future-work catalog from Tasks C1-C4

- [ ] **Step 1:** Read all four opportunity-catalog files

- [ ] **Step 2:** Distil 10-15 concrete next research directions for Chapter 15, each with:
  - Research question
  - Data/model required (all already in hand)
  - Venue target
  - Time-to-manuscript estimate (months)
  - How it addresses a Paper 10 NASEM gap (if any)

- [ ] **Step 3:** Paste into `ch15_conclusion.tex` under "§15.3 Future work"

- [ ] **Step 4:** Recompile dissertation PDF, commit.

---

## Self-Review

After finishing all 19 tasks, verify:

**1. Spec coverage:** Are all 6 sub-points covered?

- ✅ Papers 8a/8b/9/10 as Ch 10-13 → Tasks A3-A6
- ✅ Ch 14 discussion + Ch 15 conclusion → Tasks A7-A8, C5
- ✅ End-to-end review of dissertation + code + data → Phase B
- ✅ "What else can we do with Graph-DT" → Task C1
- ✅ "What else with the data" → Tasks C2, C3
- ✅ "What else with the mechanistic twin" → Task C4

**2. Placeholder scan:** Are there any "TBD" / "fill in later" in this plan?

- No. Every task lists specific files, specific scripts, specific outputs. Where content is voluminous (e.g., chapter prose) the pointer is to committed RUN_MANIFESTs + results JSONs, not hallucinated content.

**3. Type consistency:** Chapter numbering is consistent (ch10_paper8a, ch11_paper8b, ch12_paper9, ch13_paper10, ch14_discussion, ch15_conclusion) across all tasks.

**4. Executability:** Every task has a concrete exit criterion — a committed file, a passing test, or a rendered PDF.

---

## Execution Handoff

**Plan saved to `Docs/superpowers/plans/2026-04-13-dissertation-integration-p8-p10-plus-e2e-review.md`.**

Two execution options:

1. **Subagent-Driven (recommended for breadth)** — dispatch one subagent per task; main agent reviews after each; lets Phases A/B/C run somewhat in parallel
2. **Inline Execution (recommended for prose quality)** — chapters 10-15 require significant careful prose that benefits from a single sustained context

**Recommended:** Phase A inline (prose work), Phase B subagent-driven (audit scripts are formulaic), Phase C inline (opportunities analysis is synthesis-heavy).

**Predicted scope:** 19 tasks, 2-4 sessions total. The integration (Phase A) is the biggest single lift (~1.5 sessions). The audit (Phase B) is 0.5 session. Opportunity catalog (Phase C) is 0.5-1 session.
