# Paper 1 Submission — *npj Digital Medicine* Track

**Status as of 2026-04-25:** Initial-submission-ready PDF compile of the full
27-page manuscript. *npj Digital Medicine* allows PDF-format initial submission
with no strict word/page limit (online-only Open Access journal); LaTeX
formatting compliance only required at acceptance stage.

## What's in this directory

- `main.pdf` — 27-page compiled PDF for initial submission
- `main.tex` — wrapper file (npj-DM-flavoured header)
- `chapter_content.tex` — full body matter (756 lines, all R2 reviewer responses + Q.E.D. gap rebuttals + Phase 1/2 + Venuto exploration)
- `bibliography_extracted.tex` — bibliography with Venuto 2025, Schalkamp 2025, Siderowf 2023 added
- `figures/` — full figure set including new `fig_gap_a_saa_coverage.{pdf,png}`
- `cover_letter_npj.md` — npj DM-specific cover letter
- `rebuttal_letter.md` (75 KB) + `review_23Apr26.md` (11 KB) — original IEEE-JBHI W1-W10 reviewer letter + point-by-point response (kept for provenance)
- `supplementary_*.md` (6 files) — pre-existing supplementary write-ups
- `revision_analyses/` — round-2 revision artifact directory
- `DEPLOYMENT_KIT.md` (24 KB) + `REPRODUCIBILITY_PACKAGE.md` (10 KB) — deployment + reproducibility docs

## Submission process

1. **Initial submission:** Upload `main.pdf` + `cover_letter_npj.md`. *npj Digital Medicine* accepts PDF for initial review.
2. **At acceptance:** Convert to npj LaTeX template (Springer Nature `sn-jnl.cls`); add structured abstract format if required by editor.

## Differences from the IEEE JBHI track

- Header: `markboth` updated to `\textit{npj} DIGITAL MEDICINE`
- Graphics path: `figures/` is local
- File path: `paper1_submission/npj-dm/` instead of `ieee-jbhi/`
- All figures, tables, and analyses retained intact (no length cuts)

## Things to do before submission

- [ ] Update `cover_letter_npj.md` with suggested reviewer names
- [ ] Confirm with co-authors (if any) that this is the intended target
- [ ] Verify Zotero RT8B9N2J export is up-to-date for the bibliography
- [ ] Consider restructuring section order to npj convention (Methods at end) — optional at initial submission, mandatory at acceptance
- [ ] Consider adding structured abstract (Background/Methods/Results/Conclusions) — recommended but not required at initial submission
- [ ] Verify ORCIDs and affiliations
- [ ] Run final spelling/grammar pass on chapter_content.tex

## Why npj Digital Medicine

Per Track decision in chat 2026-04-25: IEEE JBHI's 14-page hard cap (including supplementary) is incompatible with the paper's full scope (5 tables, 10 figures, 30+ reviewer-driven sensitivity analyses). *npj Digital Medicine* is online-only Open Access with no strict combined cap, fits the clinical-decision-support + calibrated-uncertainty framing exactly, and routinely publishes PD ML benchmarks.

## Companion track

The IEEE JBHI version (parallel directory `paper1_submission/ieee-jbhi/`) is being aggressively cut to fit the 14-page combined limit as a backup submission target. See `_archive_2026-04-25_full_27page/ARCHIVE_README.md` for the complete provenance trail.
