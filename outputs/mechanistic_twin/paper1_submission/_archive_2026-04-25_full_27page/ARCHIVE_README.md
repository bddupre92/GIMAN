# Paper 1 Submission Archive — Full 27-page Version (2026-04-25)

## Why this archive exists

This snapshot preserves the **27-page full version** of the IEEE JBHI submission as
of 2026-04-25 23:07, before length reductions for the IEEE JBHI 14-page hard cap.

The full version contains substantial content that fits within longer-format
journals' norms but exceeds IEEE JBHI's per-paper limit. We archive it because:

1. **Future re-submission to a longer-format journal** may need this content
   intact (e.g., npj Digital Medicine, JAMIA, Journal of Biomedical Informatics,
   PLOS Computational Biology, Nature Communications).
2. **Reviewer-question detail** (R-question annotations, full Mondrian/Saerens/
   Phase-2 paragraphs) is preserved verbatim — useful for future revision rounds.
3. **All artifact paths and JSON references** are preserved as-is for
   reproducibility cross-references.

## What's in this archive

The complete submission tree as it existed at 27-page state, including:

- `chapter_content.tex` — 756 lines, all R2 reviewer responses + Q.E.D. gap
  rebuttals + Phase 1 + Phase 2 + Venuto exploration text inline.
- `main.tex` — IEEE JBHI two-column wrapper.
- `bibliography_extracted.tex` — bibliography with Venuto 2025, Schalkamp 2025,
  Siderowf 2023 citations added.
- `figures/` — full figure set including new `fig_gap_a_saa_coverage.{pdf,png}`.
- `main.pdf` — compiled 27-page output.
- `rebuttal_letter.md` (75 KB) + `review_23Apr26.md` (11 KB) — original W1-W10
  reviewer letter + point-by-point response.
- `supplementary_*.md` (6 files) — pre-existing supplementary write-ups for
  conformal sensitivity, confounder sensitivity, GAT feature sensitivity,
  reporting summary, tabular 33-feat sensitivity, TRIPOD+AI.
- `revision_analyses/` — round-2 revision artifact directory.
- `DEPLOYMENT_KIT.md` (24 KB) + `REPRODUCIBILITY_PACKAGE.md` (10 KB) — deployment
  + reproducibility documentation.
- `cover_letter.md` — original IEEE JBHI cover letter.

## What was in this version that gets cut for IEEE JBHI 14-page

The following content is in the archive but will be moved to a separate
supplementary file for the IEEE JBHI submission (Layer 1 cuts):

1. **§IV.G "Addressing the Training-label Confound: Four Training Configurations"**
   (lines 520–561) — full 4-arm balanced-BioFIND experiment.
2. **§IV.B Mondrian (R6-Q2) detailed paragraph** (line 433) — restoration of
   per-class coverage from 0.131 to 0.944.
3. **§IV.B Saerens 2002 prior-shift detailed paragraph** (line 435) — informative
   asymmetric finding.
4. **§IV.B External SOTA gap-close (R4-Q4 + R4-Q6)** paragraph (line 429).
5. **§IV.E Site-LOSO failure-mode decomposition (R6-Q4)** nested block (line 619).
6. **§V.E Phase 2 binary-stratified Venuto paragraph** (just added) — to be
   summarised to 2 sentences in main.
7. **§V.E "Methodological extensions deferred (R7-Q1, Q2, Q5, Q6)"** paragraph
   (line 673) — 4-item enumeration.
8. **R-question parenthetical annotations** throughout (~30 occurrences).

## Future use

When submitting to a longer-format journal:

1. Restore from this archive: `cp -R _archive_2026-04-25_full_27page/* <new-journal-dir>/`
2. Update `main.tex` wrapper to the target journal's class file.
3. Update `bibliography_extracted.tex` if the target uses BibTeX rather than
   bibitem.
4. Strip IEEE-specific commands (`\IEEEkeywords`, `\bibitem` style) as needed.

## Provenance

- Compile date: 2026-04-25 23:07
- Page count: 27
- Total file size: 1,132,342 bytes (1.13 MB)
- Compile chain: `pdflatex main.tex` × 2 (cross-references resolved on second pass)
- Underlying analyses: scripts/paper1/run_*.py — see REPRODUCIBILITY_PACKAGE.md

## Companion archives elsewhere

The `outputs/paper1_r2_responses/` directory holds the JSON results these
text passages cite. Those JSONs are NOT duplicated in this archive (they live
under the project's `outputs/` tree and are not specific to this submission).
Path reference (relative to project root):
`outputs/paper1_r2_responses/q_*.json` (15 files as of 2026-04-25).
