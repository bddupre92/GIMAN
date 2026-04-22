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

## References

- `CLAUDE.md` — Project-wide instructions, Schemas registry, Audit DB protocol
- `outputs/mechanistic_twin/CLAUDE.md` — Mechanistic twin artifact inventory
- `scripts/CLAUDE.md` — Script canonical run order + reproducibility risks
- `scripts/defense_prep/` — Audit DB refresh pipeline
- `docs/documentation_lifecycle_protocol.md` — Cycle A/B/C documentation discipline

---

*Established 2026-04-22 after the Paper 1 feature-schema reality-check pass. Maintainers: update this file whenever a new convention is introduced or a prior convention is superseded.*
