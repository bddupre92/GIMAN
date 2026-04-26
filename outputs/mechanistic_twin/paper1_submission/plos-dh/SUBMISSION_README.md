# Paper 1 Submission — *PLOS Digital Health* Track

**Status as of 2026-04-26:** Initial-submission-ready 36-page PDF in PLOS LaTeX format.

## What's in this directory

- `main.pdf` — 36-page compiled PDF for initial submission (PLOS double-spaced single-column with line numbers; this is normal-density formatting for PLOS, not a length flag)
- `main.tex` — single-file PLOS-formatted manuscript (per PLOS requirements no \\input or \\externaldocument)
- `build_plos_main.py` — automated rebuild script that regenerates `main.tex` from the archive's `chapter_content.tex` + `bibliography_extracted.tex` with PLOS-specific transformations (section unumbering, Fig vs Figure, TikZ stripping, cross-ref label remapping)
- `plos2025.bst` — PLOS-supplied BibTeX style file (kept for future BibTeX migration; current main.tex uses inline `\\bibitem{}` Vancouver-format)
- `_template_reference.tex` — original PLOS LaTeX template kept as reference
- `figures/` — 14 figure files (PDF + PNG variants); per PLOS submission guidelines figures should be uploaded as separate `.tif`/`.eps` files at acceptance
- `cover_letter_plos_dh.md` — PLOS Digital Health-specific cover letter
- `supplementary_*.md` (6 files) — pre-existing supplementary write-ups (will become S1–S6 supporting information files)
- `revision_analyses/` — round-2 revision artifact directory
- `DEPLOYMENT_KIT.md` (24 KB) + `REPRODUCIBILITY_PACKAGE.md` (10 KB) — deployment + reproducibility docs

## How to submit

1. **Initial submission** to PLOS Digital Health via Editorial Manager:
   - Upload `main.pdf` as the manuscript
   - Upload `cover_letter_plos_dh.md` content as cover letter
   - Upload each figure as a separate `.tif`/`.eps` file (PLOS requires figures as separate files)
   - Upload supporting information as separate `S1_*.pdf` files
2. **At acceptance:**
   - Convert `\\bibitem{}` references to BibTeX format using `plos2025.bst` (template plumbing already in place at line ~50 of `main.tex`)
   - Convert TikZ architecture diagram into a separate `.eps` figure file (currently stripped as a placeholder caption in main.tex)
   - Confirm figure files are formatted to PLOS specs (sentence-case figure titles, PNG→TIF conversion at 300 DPI)

## Differences from IEEE JBHI track

- **Section ordering**: PLOS standard (Abstract → Author summary → Introduction → Materials and methods → Results → Discussion → Conclusion → Acknowledgments → References → Supporting information)
- **Section numbering**: All `\\section*{}` (unnumbered) per PLOS style
- **Figure citations**: `Fig~\\ref{}` not `Figure~\\ref{}`
- **Title block**: Single-author flushleft format with PLOS symbol macros (\\Yinyang, \\textcurrency, etc.) supported but unused
- **Author summary**: Added (PLOS-specific 150-200 word lay-audience summary)
- **Cross-references**: All `\\ref{p1:sec:*}` patterns converted to plain text section names by build script
- **TikZ figures**: Stripped (PLOS wants figures as separate image files)
- **Single-file requirement**: All content embedded in main.tex; no \\input commands

## Things to do before submission

- [ ] Update `cover_letter_plos_dh.md` with suggested reviewer names
- [ ] Render the architecture TikZ diagram as a standalone `.eps` figure file
- [ ] Verify ORCID and affiliation accuracy
- [ ] Run final spelling/grammar pass on `main.tex`
- [ ] Upload supplementary `.md` files to a public archive (Zenodo / OSF) and add DOIs to the references
- [ ] Confirm that the embedded `\\bibitem{}` Vancouver-style is acceptable at initial submission (PLOS instructions: "Type in your references following Vancouver style and reference formatting instructions" — yes)

## Why PLOS Digital Health was chosen

Per Track decision: PLOS Digital Health is the cleanest scope match for this paper's deployment-readiness + calibrated-uncertainty + clinical-decision-support framing among all venues considered. It has no page limits (online-only Open Access journal), peer review by both clinical and technical reviewers, faster review cycles than higher-impact alternatives, and is the journal whose stated mission most directly matches our two-stage deployment pipeline + Mondrian on-site recalibration protocol contributions.

## Companion tracks

The IEEE JBHI version (parallel directory `paper1_submission/ieee-jbhi/`) was 23 pages combined main+supp at the time of the venue switch — well above the 14-page hard cap including supplementary. The npj Digital Medicine version (parallel `paper1_submission/npj-dm/`) is at 27 pages (no length limit there either) as a higher-prestige fallback target. The full 27-page archive is preserved at `_archive_2026-04-25_full_27page/`.

## Build script notes

The `build_plos_main.py` script regenerates `main.tex` from the archive in ~2 seconds. To rebuild after archive updates:

```bash
cd /path/to/plos-dh
python3 build_plos_main.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The script preserves all text content from the archive verbatim — it only handles structural transformations (section unumbering, cross-ref remapping, TikZ stripping, figure citation reformatting, IEEE→PLOS preamble swap).
