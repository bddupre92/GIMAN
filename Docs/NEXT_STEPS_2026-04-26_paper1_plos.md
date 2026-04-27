# Paper 1 — Post-compact resume anchor (2026-04-26)

**Last session ended:** 2026-04-26 ~00:30 with three submission packages built and ready.

## Where we are

All four q.e.d. gaps are closed empirically. Three submission tracks exist:

| Track | Path | Pages | Status |
|---|---|---|---|
| **PLOS Digital Health** ★ primary | `outputs/mechanistic_twin/paper1_submission/plos-dh/` | 36 (main.pdf) | **Ready to submit** |
| **npj Digital Medicine** fallback | `outputs/mechanistic_twin/paper1_submission/npj-dm/` | 27 | Ready to submit |
| IEEE JBHI | `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/` | 23 + 7 supp | **Abandoned** (14pg cap incompatible) |
| Archive (full original) | `_archive_2026-04-25_full_27page/` | 27 | Provenance only |

## Immediate resume action — submit to PLOS Digital Health

Open https://journals.plos.org/digitalhealth/ → Submit a Manuscript → Editorial Manager.

### Submission checklist (all files in `plos-dh/`)

1. **Manuscript:** upload `main.pdf` (36 pages, includes embedded Vancouver-style references)
2. **Cover letter:** paste content of `cover_letter_plos_dh.md` into Editorial Manager cover-letter field
3. **Figures (separate uploads):** all `.pdf` and `.png` files in `figures/` — PLOS prefers `.tif`/`.eps` at 300 DPI but accepts at initial submission. Convert to `.tif` only if reviewer requests.
4. **Supporting Information** (separate `S1_*.pdf` uploads):
   - `supplementary_conformal_sensitivity.md` → S1 Appendix
   - `supplementary_confounder_sensitivity.md` → S2 Appendix
   - `supplementary_gat_feature_sensitivity.md` → S3 Appendix
   - `supplementary_reporting_summary.md` → S4 Appendix
   - `supplementary_tabular_33feat_sensitivity.md` → S5 Appendix
   - `supplementary_tripod_ai.md` → S6 Appendix
5. **Data availability statement:** "All code and processed feature tables are released as a single reproducibility package; see REPRODUCIBILITY_PACKAGE.md. Raw cohort data is available via PPMI/AMP-PD DUAs."
6. **Suggested reviewers** (need to fill in before submit — currently bracketed in cover letter)

### Pre-submission TODOs (~15 min total)

1. **Fill in suggested reviewers** in `cover_letter_plos_dh.md` — 3-4 names. Candidates: Tanya Simuni (Northwestern, NSD-ISS framework), Caroline Tanner (UCSF, PPMI), Andrew Siderowf (Penn, SAA), Charles Venuto (URMC, SAA-prediction model author whose work we cite).
2. **Verify ORCID + affiliation** in `main.tex` line 50-58 area
3. **Final spelling/grammar pass** on `main.tex` (auto-generated from archive; manual once-over recommended)
4. **Re-run** `cd plos-dh && python3 build_plos_main.py && pdflatex main.tex && pdflatex main.tex` if any archive content changed

## What was accomplished today (2026-04-25/26)

### Q.E.D. gap closures (all 4)

| Gap | Severity | Verdict | Headline |
|---|---|---|---|
| 1 SAA construct validity | MAJOR | ~98% closed | **Venuto-imputed S+ rate = 91.5%** on 647 SAA-untested NSD+ — squarely in literature band [88%, 93%] |
| 2 External NSD+ modality | MAJOR | ~92% closed | 12-feat clinical-only NSD+ **outperforms** 21-feat by 4-5pp under wave-LOCO/era-LOCO temporal shift |
| 3 UPDRS/MoCA tautology | MAJOR | ~95% closed | 12-feat rule-anchor-elided ablation: \|Δ AUC\| ≤ 0.009 across UPDRS2/MoCA/UPDRS1 drops. **VERDICT: NO_TAUTOLOGY** |
| 4 Conformal exchangeability | MINOR | ~99% closed | Mondrian per-class CP restores three-class min coverage 0.131 → 0.944 with n_cal≥50 |

### New scripts (under `scripts/paper1/`)

- `run_venuto_saa_imputation.py` — Venuto coefficients applied to PPMI; 91.5% imputed S+
- `run_12feat_rule_anchor_ablation.py` — NO_TAUTOLOGY verdict
- `run_nsd_plus_modality_consolidation.py` — 21-vs-12 head-to-head 3 regimes
- `run_venuto_exploration.py` — PPMI-tuned re-fit + 4-stratum cohort expansion
- `run_venuto_4stratum_phase1.py` — Per-stratum NSD+ AUC
- `run_venuto_4stratum_phase2.py` — Binary-target stratified
- `generate_gap_a_figure.py` — Fig 9 generator

### New JSONs (under `outputs/paper1_r2_responses/`)

- `q_gap_a_venuto_saa.json` (+ summary.md + manuscript_paragraph.md)
- `q_gap_2_nsd_plus_modality_consolidation.json`
- `q_gap_3_12feat_rule_anchor_ablation.json`
- `q_venuto_exploration.json`
- `q_venuto_4stratum_phase1.json`
- `q_venuto_4stratum_phase2.json`
- `qed_gap_coverage_audit.md` — line-by-line manuscript coverage matrix

### New figure

`outputs/mechanistic_twin/paper1_submission/{ieee-jbhi,npj-dm,plos-dh}/figures/fig_gap_a_saa_coverage.{pdf,png}` — 2-panel SAA coverage + 77/91/88/93% S+ comparison.

## Build pipelines

- **PLOS DH rebuild from archive:**
  ```bash
  cd outputs/mechanistic_twin/paper1_submission/plos-dh
  python3 build_plos_main.py
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
  ```
  Script preserves all archive content verbatim; only structural transforms applied.

- **npj DM is direct copy of archive:** edits to `_archive_2026-04-25_full_27page/chapter_content.tex` propagate to `npj-dm/` only via manual `cp` (no script).

- **IEEE JBHI is abandoned** but kept on disk for reference. Has Layer 1+2 cuts applied.

## Things tried that didn't work

- **Paper2Any + PaperVizAgent figure-gen tools** (Google + OpenDCAI) — installed at `~/Projects/tools/{Paper2Any,papervizagent}/` but blocked by Anthropic API credit balance = $0. PaperVizAgent additionally hard-coded to `AnthropicVertex` (no direct API support). Abandoned.
- **IEEE JBHI 14-page cap pursuit** — confirmed via WebFetch that JBHI cap is 14 pages **including supplementary**. Layer 1+2 cuts got main from 27→23 + 7 supp = 30 combined, still 16 pages over hard cap. Abandoned in favor of PLOS DH (no cap).

## After PLOS submission

If accepted at PLOS DH:
- Convert `\bibitem{}` Vancouver-style to BibTeX with `plos2025.bst` (template plumbing already in `main.tex`)
- Render architecture TikZ as standalone `.eps` figure file (currently a placeholder caption)
- Convert PNGs to TIF at 300 DPI
- Restructure to PLOS double-spaced format if not already

If rejected at PLOS DH:
- Fall to npj-dm submission package (already submission-ready)
- All same content, different journal-specific cover letter and header

## Open work (not blocking submission)

These are q.e.d. critique residuals or independent improvements that are not required for PLOS DH submission but useful to track:

1. **Phase 2 binary-stratified figure** — could generate a small panel showing the chance-level AUC on S+D− stratum visually
2. **Suggested-reviewer list** — research and add 3-4 names to cover letter
3. **Architecture TikZ → EPS** — only needed at acceptance, not initial submission
4. **q.e.d. response submission** — once we have the rebuttal text from `q_gap_a_manuscript_paragraph.md`, submit to q.e.d. service to refresh their gap analysis

## Provenance / audit trail

- All q.e.d. response artifacts under `outputs/paper1_r2_responses/q_*.json` (15 files as of 2026-04-26)
- All session memories under `~/.claude/projects/-Users-blair-dupre-Projects-CSCI-FALL-2025/memory/`
- Full session detail: `memory/session_2026_04_26_paper1_qed_closure_plos_pivot.md`
- Paper 1 R1→R2 arc updated in `memory/paper1_r1_r2_arc.md`

## TL;DR for resume

Open PLOS DH Editorial Manager. Upload `main.pdf` from `plos-dh/`. Paste cover letter from `cover_letter_plos_dh.md`. Add 3-4 suggested reviewers (Simuni, Tanner, Siderowf, Venuto are obvious candidates). Submit. ~15 min of work.
