# Project Conventions

Standing conventions for the GIMAN dissertation repository. Established after the 2026-04-22 reality-check pass discovered that Paper 1's "46-feature" claim was prose-only — no SQL table, no CSV, no benchmark — and had propagated through the dissertation chapter, two submission drafts, and the deep dive.

These conventions exist to prevent that failure mode from recurring.

---

## 1. Feature Schema — SQL is the source of truth

**Rule:** Every feature schema used by any paper MUST live in a Postgres table under `features.*` BEFORE any benchmark, figure, or submission claim is produced against it. CSVs are artefacts for distribution or backup; Postgres is authoritative.

### Why

On 2026-04-22 we discovered that Paper 1's submission claimed "46 features across 10 domains" in prose (Table II, §III-D, §IV-E), while the production `scripts/run_paper1_benchmark.py` loaded a 22-column CSV. The "46" had been introduced by a find-replace pass that never touched the benchmark code. Three weeks of prose edits, one full submission PDF, and a deep-dive "alignment" commit later, the mismatch was finally caught. Every benchmark number in the submission (0.979, 0.944, 0.954, 0.913) was computed on 22 features regardless of the label.

The root cause was that nothing in the pipeline forced a prose claim to correspond to a runnable SQL query.

### Protocol for new feature schemas

When introducing a new feature set for Paper N:

1. **Create the Postgres table first.** Name it `features.paper{N}_{description}` — descriptive, not abbreviated.
   - Example: `features.paper1_features_with_targets`, `features.paper1_features_extended_33`, `features.paper2_gimin_cohort`.
2. **Write an idempotent assembler script** at `scripts/paper{N}/create_sql_paper{N}_{description}.py` that reads from raw PPMI CSVs or joins existing features tables and writes to Postgres via SQLAlchemy. Must be re-runnable (`DROP TABLE IF EXISTS` then `CREATE TABLE`).
3. **Emit a feature-metadata JSON** at `outputs/paper{N}_sql/metadata.json` documenting each column: domain, description, literature citation key. This is the traceability layer — every feature must have a citation.
4. **Emit a null-rate report** at `outputs/paper{N}_sql/null_rate_report.json` showing per-column non-null coverage on the target cohort.
5. **Benchmark scripts read from Postgres**, not CSV. Use `giman_pipeline.data.db.read_sql("SELECT * FROM features.paper{N}_{description}")`.
6. **Update the `features` row in `CLAUDE.md`'s Schemas table** in the same commit as the new table.

### What NOT to do

- ❌ Describe a feature schema in LaTeX or a README before assembling the SQL table.
- ❌ Rename features in prose without renaming the columns in Postgres.
- ❌ Run a benchmark on a CSV without first writing it to a Postgres table (unless the CSV itself is assembled by a committed script whose output path is a Postgres table).
- ❌ Claim "N features across M domains" in a manuscript without counting the actual Postgres table columns.

### Verification one-liner

```bash
psql giman_research -Atc "SELECT COUNT(*) FROM information_schema.columns \
  WHERE table_schema='features' AND table_name='paper1_features_extended_33' \
  AND column_name NOT IN ('patno','nsd_iss_stage', ..., 'target_nsd_positive');"
```

Any prose claim of "Paper 1 uses 33 features" should be verifiable by this query returning 33.

---

## 2. Literature-grounded feature selection

**Rule:** Every feature included in any benchmark must have a literature citation tying it to the disease mechanism, cohort protocol, or staging framework. No data-driven filtering (SHAP, mRMR, Boruta, permutation importance) without explicit justification and pre-registration.

### Why

Clinical prediction papers are held to TRIPOD+AI's standard of a priori scientific rationale for predictor selection. Data-driven filters overfit to training-set signal, especially on cohorts with n ~ 2,000. Our canonical methods are:

1. **Literature citation** for each feature (e.g., Simuni 2024 for NSD-ISS anchors; Goetz 2008 for UPDRS; Fischl 2012 for FreeSurfer; Nalls 2019 for GRS).
2. **Circularity audit** (for staging prediction): exclude features that enter the staging definition directly. For Paper 1: NP3TOT and Putamen SBR excluded; subscales retained.
3. **Feature ablation** (for importance interpretation): full-feature vs reduced-feature benchmarks produce the Δ percentages that quantify feature contribution. This is post-hoc, for interpretation — not pre-hoc, for selection.

### When data-driven selection IS allowed

Only when:
- Pre-registered in a locked analysis plan before the results are generated.
- Reported alongside the literature-grounded comparison (both should appear in the manuscript).
- The selection procedure is itself cross-validated (nested CV) to avoid over-optimistic selection bias.

### What to cite per feature domain

| Domain | Citation key | Reference |
|---|---|---|
| NSD-ISS staging anchors | `simuni2024` | Simuni et al. 2024 Lancet Neurol |
| MDS-UPDRS motor scales | `goetz2008` | Goetz et al. 2008 Mov Disord |
| PPMI cohort battery | `marek2018ppmi` | Marek et al. 2018 Ann Clin Transl Neurol |
| FreeSurfer cortical measurements | `fischl2012` | Fischl 2012 NeuroImage |
| PPMI CSF biomarkers | `mollenhauer2017` | Mollenhauer et al. 2017 Neurology |
| Parkinson's polygenic risk score | `nalls2019` | Nalls et al. 2019 Lancet Neurol |
| DaT-SPECT deployment | `seibyl2018dat` | Seibyl et al. 2018 J Nucl Med |
| Tree-vs-DL on tabular data | `grinsztajn2022`, `shwartzziv2022` | Grinsztajn et al. NeurIPS 2022; Shwartz-Ziv & Armon 2022 |

---

## 3. Submission independence (first-submission papers stand alone)

**Rule:** The first journal submission of a paper must not depend on unpublished companion papers. Cross-paper references are permissible only when the cited companion is already published or the authors are deliberately submitting as a bundled package.

### Why

Paper 1 is our first journal submission (IEEE JBHI, 2026). Reviewers cannot assess a claim that depends on "see companion submission (Dupre, in preparation)." Every feature, every method, every result must be defensible against direct-source literature.

### Protocol

- Feature additions (e.g., FreeSurfer cortical thickness, CSF biomarkers, GRS) are cited directly to their original publications, NOT to companion GIMIN (Paper 2) or unified-pipeline (Paper 6) submissions.
- The GIMIN parquet can be used as a *data-loading convenience*, but the manuscript must describe the features as drawn from raw PPMI measurements per the original publications.
- Only after the first submission is accepted may the dissertation chapter add cross-references to companion papers.

---

## 4. Audit-DB freshness

**Rule:** New bibitems, chapter edits, or result JSONs under `outputs/paper*/` require a corresponding refresh of `audit.claim` + `audit.citation` + `audit.citation_use` tables.

**Already documented in CLAUDE.md** under "Audit-DB freshness protocol." This entry exists as a cross-reference so the conventions are consolidated in one place.

**When to refresh:**
1. Adding `\bibitem{}` to `bibliography_extracted.tex` or `outputs/dissertation/bibliography.tex` → run `scripts/defense_prep/01_extract_citations.py` + `03_resolve_citations_to_zotero.py`.
2. Editing chapter `.tex` prose or numerical claims → run `scripts/defense_prep/02_extract_numerical_claims.py` + `07_per_claim_value_verifier.py`.
3. Adding a new model-result JSON/CSV under `outputs/paper*/` → re-run `07_per_claim_value_verifier.py`; flag for potential `verdict='refuted'` updates if results contradict prior claims.

**Enforcement:** `scripts/check_audit_freshness.py` PreToolUse hook warns on `git commit` when any of the three triggers are present. Warning-only (does not block).

---

## 5. Script output discipline

**Rule:** Every benchmark or analysis script must:

- Write to a timestamped or descriptively-named directory under `outputs/paper{N}_*/`.
- Not overwrite prior outputs without explicit `--force-reload` flag.
- Emit a `{script_name}_report.md` summary alongside raw JSONs.
- Log the git commit hash, seed, Python version, and key library versions into a provenance block at the top of the report.

### Known violations (inherited technical debt)

Per `scripts/CLAUDE.md`: Paper 3-6 runners overwrite `OUTPUT_DIR` in place with no timestamping. This predates the convention and is documented as a reproducibility risk. New scripts MUST follow the timestamped pattern.

---

## 6. Reproducibility contract for compute-heavy runs

**Rule:** Any benchmark taking >10 minutes must emit, alongside its results:

- Git commit hash at time of run
- Random seed(s) used
- Python + key library versions (torch, catboost, sklearn, pymc)
- Hardware backend (MPS/CUDA/CPU)
- Wall-time
- Cohort query (SQL or CSV path + row count at run-time)

Minimum structure in the script's header docstring:

```python
"""Script description.

Reproducibility:
  - Seed: 42
  - Source: features.paper{N}_{description}
  - Cohort: N patients verified at runtime
  - Runtime: ~30 min on MPS
  - Git commit: recorded in output JSON at run-time
"""
```

---

## 7. Paper Rigor Rubric (distilled from 2026-04-23 IEEE JBHI Paper 1 reviewer response)

**Rule:** Every paper (Papers 1–12) must satisfy the eight rigor categories below BEFORE submission. Each category has a concrete "PASS" test and a "reviewer-preempt" artifact.

This rubric was established after the 2026-04-23 IEEE JBHI Paper 1 review surfaced a coherent set of reviewer concerns that, taken together, define the minimum bar for clinical-ML benchmark papers in 2026. Applying the rubric retrospectively to Papers 2–11 is mandatory before their next submission pass (see §References below for paper-by-paper checklist).

### 7.1 Leakage audit

**PASS test:** for every model, explicit per-fold fitting of (a) missing-data imputer, (b) scaler/standardizer, (c) k-NN / similarity graphs, (d) feature selection, (e) class-balance resampling. No step from (a–e) may see test-fold samples during training.

**Reviewer-preempt artifact:** a §Leakage Audit paragraph or table in the Methods section naming each leakage vector explicitly (even when the verdict is "clean"). For graph models, state whether the test-fold graph is built on test nodes only, or on train+test nodes with train-only edges — these are different designs with different generalization stories.

**Canonical literature anchors (DOIs verified 2026-04-23):** Kapoor & Narayanan 2023 *Patterns* `10.1016/j.patter.2023.100804` (leakage taxonomy, 8 categories); Bernett 2024 *Nature Methods* `10.1038/s41592-024-02362-y` (graph / biological ML leakage specifically); Shadbahr 2023 *Commun Med* `10.1038/s43856-023-00356-z` (imputation leakage inflates AUC 0.03–0.10).

**Paper 1 post-audit state (2026-04-23):** Enhanced MM-GAT pipeline is fold-clean (imputer + scaler + graph all per-fold; documented at `scripts/run_enhanced_gat_benchmark.py:506`). Simple GAT likewise (`scripts/run_giman_gat_benchmark.py:319`). **Tabular CatBoost benchmark has a mild pre-CV population-median imputation at `scripts/run_paper1_benchmark.py:107`** — expected bias ≤ 0.5pp AUC but requires a fold-local-imputation re-run to close the reviewer's concern.

### 7.2 Hyperparameter optimization

**PASS test:** for any non-default model, a constrained-search nested 5-fold CV with the top 2–3 competing models receiving equal HPO budget. Random-search or Bayesian-optimization over 20–50 points per model. Default-only baselines ARE acceptable IF explicitly stated and IF the top-k models get HPO — i.e., no hidden advantage for the favoured model.

**Reviewer-preempt artifact:** an §HPO Protocol subsection with: search space specification, HPO budget per model, nested-CV fold count, best-hyperparameter-per-fold table, sensitivity-to-HP curve (expected AUC vs HP value). Prefer the recipe from Feurer & Hutter 2019 *Automated Machine Learning* Chapter 1.

**Canonical literature anchors:** Cawley & Talbot 2010 *JMLR* (optimistic HPO bias on small datasets); Varma & Simon 2006 *BMC Bioinformatics* (nested-CV recipe); Vabalas 2019 *PLOS ONE* (small-sample overfit with HPO).

**What NOT to do:** tune CatBoost while leaving competing models (GAT, MLP, TabPFN) at defaults. This is the single most common asymmetric-comparison reviewer trap.

### 7.3 Ordinal-aware modeling (if target is ordinal)

**PASS test:** for any ordinal target with K ≥ 3 classes, at least one ordinal-aware method benchmarked alongside the multiclass baseline. Acceptable methods: CORAL (Cao 2020), CORN (Shi 2023), ordinal logistic, cumulative-link model, ordinal CatBoost (`loss_function="YetiRank"` or similar).

**Reviewer-preempt artifact:** per-class confusion matrix + quadratic-weighted-kappa (QWK) with bootstrap CI + mean absolute ordinal error (MAOE). Report ALL three — QWK can hide per-class failures; per-class accuracy can hide ordinal distance errors; MAOE doesn't say whether you're biased high or low.

**Canonical literature anchors:** Cao / Mirjalili / Raschka 2020 *Pattern Recognition Letters* (CORAL, canonical); Shi 2023 AAAI (CORN, rank-consistent); Niculescu-Mizil 2005 (calibration for ordinal).

### 7.4 Conformal prediction reporting

**PASS test:** a single primary CL locked in the Methods section (we recommend 90% CL for clinical work, matching Diaz-Rincon 2025). In the main text, report marginal coverage, mean set size, and class-conditional coverage AT THAT CL. In Supplementary, sweep CL ∈ {0.80, 0.85, 0.90, 0.95} and present the coverage-efficiency tradeoff curve. For any model that can produce empty prediction sets (abstention), state the abstention frequency AND the clinical triage policy.

**Reviewer-preempt artifact:** a §Conformal Implementation Checklist bullet list in Methods: (i) CL primary, (ii) CL sweep range, (iii) abstention allowed Y/N, (iv) abstention frequency table, (v) whether calibration set is shared across targets or stratified, (vi) external coverage + class-conditional external coverage (if external cohort exists).

**Canonical literature anchors:** Angelopoulos & Bates 2023 *arXiv* Gentle Intro (§abstention); Sadinle 2019 *JASA* (LAC + abstention); Romano 2020 *NeurIPS* (APS ordinal variant); Einbinder 2022 *NeurIPS* (ordinal CP).

**Paper 1 gotcha (from 2026-04-23 review):** current main text mixes "90% CL" and "95% CL" phrasing, and reports mean set size < 1 without explaining that this means empty sets (abstentions) exist. Fix in the next submission pass.

### 7.5 Calibration metrics

**PASS test:** ECE + Brier + reliability diagram reported in main text for the primary model on every target. Per-class calibration (ECE per class) reported when K ≥ 3. External-cohort calibration reported if any external cohort is evaluated.

**Reviewer-preempt artifact:** Figure X "Calibration Diagnostics": 2×K grid of reliability diagrams (internal top row, external bottom row if applicable). ECE + Brier numbers captioned.

**Canonical literature anchors:** Guo 2017 *ICML* (temperature scaling, canonical ECE); Niculescu-Mizil 2005 (Brier for ordinal); Roelofs 2022 AAAI (reliability-of-reliability — the ECE estimator variance story).

### 7.6 Confounder sensitivity package

**PASS test:** for any ML paper predicting a biological/clinical endpoint, a 5-analysis sensitivity package pre-registered BEFORE running:
- A. Age (matched cohort or residual-adjustment)
- B. Sex (stratified + bootstrap interaction test)
- C. Enrollment wave / study era (leave-one-wave-out)
- D. Scanner / acquisition protocol (leave-one-protocol-out if applicable)
- E. Site (leave-one-site-out on the covered subsample, with pre-registered decision rule)

Additional analyses as indicated by the target: medication status (for PD), genetic carrier status (for LRRK2 / GBA cohorts), comorbidities.

**Reviewer-preempt artifact:** a §Confounder Sensitivity paragraph in Discussion + a §S-5 Supplementary section with per-analysis methodology, results, and pre-registered decision rule. FAIL verdicts are reported honestly without post-hoc retuning — see Paper 1 Analysis E at `outputs/paper1_site_loso/` for the template.

**Canonical literature anchors:** Austin 2011 *Pharmaceutical Statistics* (caliper rule); Schmitz-Steinkrüger 2021 *EJNMMI* (DaT-SPECT age/sex variance explanation); Pfohl 2020 ML4H (subgroup fairness).

**Paper 1 post-audit state (2026-04-23):** A+B+C+D PASS; E is a pre-registered FAIL driven by two small-N + extreme-class-imbalance folds — reported honestly in §S-5.6.

### 7.7 Related-work coverage (by paper class)

**PASS test:** for a clinical-ML benchmark paper, the related-work section must cite (a) the canonical target-domain benchmark of the last 5 years, (b) the state-of-the-art tabular foundation models (TabPFNv2, AutoGluon-Tabular) WITH an explanation of whether they were benchmarked or not, (c) the relevant deep-learning diagnostic literature specifically positioned as SIBLING not PARENT of the biological-staging task, (d) the ordinal ML literature if target is ordinal, (e) the conformal prediction literature with specific variant choices justified.

**Reviewer-preempt artifact:** a §Related Work subsection with one paragraph per (a–e), each ending in a one-sentence "how our paper differs from this work" statement.

**Canonical literature anchors vary by paper.** Paper 1's 2026-04-23 reviewer flagged TabPFNv2 (Hollmann 2024 *Nature Machine Intelligence*), AutoGluon-Tabular (Erickson 2020 *AutoML*), and ordinal CP (Romano 2020 / Einbinder 2022) as specific gaps. Add or explicitly waive each.

### 7.8 Reproducibility package

**PASS test:** at submission, the paper's `outputs/paper{N}_submission/{venue}/` directory contains (i) main.tex + chapter_content.tex + bibliography_extracted.tex, (ii) figures/, (iii) cover_letter.md, (iv) supplementary_*.md files, (v) A REPRODUCIBILITY_PACKAGE.md listing: SQL extract commands, fold-assignment JSONs, conformal calibration files (the per-fold calibration sets + quantiles), checkpoints where applicable, exact seeds, exact package versions (`uv.lock` or `requirements-frozen.txt`), a Zenodo DOI placeholder.

**Reviewer-preempt artifact:** a §Data and Code Availability subsection (journals require this; some reject without it). State:
- What is released publicly (code, aggregate results, figures)
- What is gated (raw PPMI data behind DUA at ppmi-info.org)
- Where the Zenodo DOI points (pre-register on submission; upload artifacts on accept)

**Canonical literature anchors:** McDermott 2021 *Science Translational Medicine* (clinical ML reproducibility); Pineau 2020 *Communications of the ACM* (repro checklist).

---

### 7.9 Application to Papers 2–11 (mandatory retrospective sweep)

Each paper's deep-dive document (`outputs/defense_prep/paper{N}_deep_dive.md`) must grow a §Rigor Rubric section reporting PASS/FAIL for each of 7.1–7.8 before that paper's next submission. Expected initial state:

| Paper | Venue | Status | Known gaps vs rubric (pre-sweep estimate) |
|---|---|---|---|
| 1 | IEEE JBHI | submitted, revision pending | 7.1 mild leakage; 7.2 missing; 7.3 missing; 7.4 clarity; 7.5 missing; 7.7 partial; 7.8 partial |
| 2 | IEEE JBHI | submitted | 7.2 missing; 7.3 n/a; 7.5 missing |
| 3+4 | npj Digital Medicine (combined) | submitted | 7.2 partial (DeepHit defaults); 7.5 partial (calibration done); 7.6 partial |
| 5 | (planning) | pre-submission | 7.1–7.8 all open |
| 6 | JAMIA | submitted | 7.2 missing; 7.6 partial; 7.7 TabPFN missing |
| 7 | CPT:PSP | submitted | 7.2 n/a (ODE calibration); 7.4 n/a |
| 8a | PLoS Comput Biol | submitted | 7.1 needs audit; 7.4 partial |
| 8b | Movement Disorders | submitted | — |
| 9 | CPT:PSP | submitted | 7.5 n/a; 7.6 partial |
| 10 | npj Parkinson's Disease | submitted | 7.2 partial; 7.8 partial |
| 11 | npj Parkinson's Disease | submitted | 7.2 partial; 7.5 missing |

Update column 3 after each paper's retrospective sweep. Column 4 becomes the concrete work-list.

---

## References

- `CLAUDE.md` — Project-wide instructions, Schemas registry, Audit DB protocol
- `outputs/mechanistic_twin/CLAUDE.md` — Mechanistic twin artifact inventory
- `scripts/CLAUDE.md` — Script canonical run order + reproducibility risks
- `scripts/defense_prep/` — Audit DB refresh pipeline
- `docs/documentation_lifecycle_protocol.md` — Cycle A/B/C documentation discipline
- `Docs/superpowers/plans/2026-04-23-paper1-reviewer-response.md` — Paper 1 IEEE JBHI R1 response plan (origin of §7)
- `outputs/paper1_site_loso/PRE_REGISTRATION.md` — canonical pre-registration template for §7.6 analyses

---

*Established 2026-04-22 after the Paper 1 feature-schema reality-check pass. Extended 2026-04-23 with §7 Paper Rigor Rubric from the IEEE JBHI Paper 1 reviewer response. Maintainers: update this file whenever a new convention is introduced or a prior convention is superseded.*
