# Documentation Lifecycle Protocol (v1.0)

**Status:** LOCKED 2026-04-10
**Scope:** All GIMAN dissertation + mechanistic twin work
**Authority:** This protocol is MANDATORY after every computational step that produces publishable results. It integrates with the [Closed-Loop Methodology v1.1](closed_loop_methodology_v1.md) as Stage 6.5 — the documentation gate that fires BETWEEN the decision gate (Stage 6) and the next claim's Stage 1.
**Parent:** Root [CLAUDE.md](../CLAUDE.md) "Mandatory Documentation Update Checklist"

---

## 1. Why this exists

During the 2026-04-10 session, Blocks 3-6 were completed in rapid succession (~4 hours). The reproducibility manifest fell behind (only had Steps 2.1-2.2 when Blocks 3-6 were already done), the roadmap was outdated (still referenced Variant A equations), and the deep dive had stale numerical values. Documentation drift is not just cosmetic — it creates:

- **Reproducibility risk:** a future session reads an outdated CLAUDE.md and makes decisions based on stale numbers
- **Defense risk:** committee members see inconsistent numbers across documents
- **Publication risk:** a reviewer finds a discrepancy between the manuscript and the reproducibility manifest

The protocol makes documentation a **concurrent** activity, not a post-hoc cleanup.

---

## 2. The Three Cycles

### Cycle A: Per-Step Documentation (fires after EVERY computational step)

A "step" is any script execution that produces output files (CSVs, JSONs, parquets, figures, manifests).

**Mandatory actions (in order):**

1. **Verify output integrity.** Check that output files exist, have expected row counts, and hashes are deterministic (rerun if needed to confirm bitwise reproducibility).

2. **Update REPRODUCIBILITY_MANIFEST.md.** Add a new claim→producer row for every publishable number produced by the step. Include: claim text, evidence source, output file path, producer command, RNG seed, and hash.

3. **Update the step's RUN_MANIFEST.md.** Confirm gate verdicts, provenance, and output hashes are embedded.

4. **Update relevant CLAUDE.md state headers.** At minimum: root CLAUDE.md block status table. Also update sub-CLAUDE.md files if new artifacts, gotchas, or API changes are produced.

### Cycle B: Per-Block Documentation (fires after EVERY block completion)

A "block" is a logical unit of work (e.g., Block 3 = CSF coupling, Block 4 = counterfactual). Blocks typically contain 2-5 steps.

**Mandatory actions (in order, after all steps in the block pass):**

1. **Run Cycle A for any steps not yet documented.**

2. **Update paper7_phase2_deep_dive.md.** Add/update the relevant §7 subsection with headline numbers, gate verdicts, and interpretation. Add new Q&As if the block produces defense-relevant findings.

3. **Update mechtwin_review.tex.** Add/update the relevant §11 subsection with LaTeX tables and prose. Ensure all `\cite{}` keys resolve to `bibliography.tex`.

4. **Update bibliography.tex + Zotero.** For every new citation:
   - (a) Add DOI to Zotero via `pyzotero` MCP or manual import → collection RT8B9N2J (Mechanistic Digital Twin) → tag with relevant module + `mechanistic-digital-twin`
   - (b) Create `\bibitem{}` entry in `outputs/dissertation/bibliography.tex` (IEEE format, alphabetical by cite key)
   - (c) Verify total count and zero duplicates (`grep -c '\\bibitem' bibliography.tex`)
   - See [memory/reference_zotero.md](../memory note) for Zotero API config (library 13550602, API key in MCP env)

5. **Update bioRxiv manuscript (main.tex).** If headline numbers changed (cohort size, coverage, delays), update Abstract, Results, tables, and figure captions. Recompile PDF.

6. **Update dissertation chapter (ch09_paper7.tex).** Mirror bioRxiv updates in the dissertation version.

7. **Check downstream impact.** Ask: "Does this block's result change anything in earlier blocks?" If yes, document the impact. Example: Block 6 (Wave B) expanded the cohort from 304→1,065, requiring LOO and counterfactual reruns.

### Cycle C: Per-Session Documentation (fires at END of every work session, before compaction)

**Mandatory actions:**

1. **Update root CLAUDE.md "Key files to read on resume" section.** Ensure every new canonical file is listed.

2. **Update the mechanistic_digital_twin_roadmap.md** if any module status changed (e.g., Phase 2 complete, Phase 3 feasibility confirmed).

3. **Run `grep -c '\\bibitem'` on bibliography.tex** and record the count in the root CLAUDE.md.

4. **Write a compaction-safe summary** at the end of root CLAUDE.md's block status section documenting what was accomplished in this session and what the next session should start with.

5. **Save any session-level learnings to memory** (user preferences, feedback, project state changes) using the auto-memory system.

---

## 3. Integration with Closed-Loop Methodology

The closed-loop methodology v1.1 has 6 stages for scientific claims:

```
Stage 1 (Lit grounding) → Stage 2 (Decision) → Stage 3 (Pre-exec sanity) →
Stage 4 (Post-exec review) → Stage 5 (Independent validation) → Stage 6 (Decision gate)
```

This documentation protocol adds **Stage 6.5**:

```
Stage 6 (APPROVE/MODIFY/ABANDON) → Stage 6.5 (Documentation Lifecycle) → Next claim Stage 1
```

Stage 6.5 fires Cycle A (per-step) and, if the step completes a block, also Cycle B (per-block). The claim is NOT considered "done" until Stage 6.5 passes.

**The iron rule:** No claim is reported to the user as "complete" until its documentation trail exists in the reproducibility manifest, the relevant CLAUDE.md files, and (if block-level) the deep dive and manuscript.

---

## 4. Document Inventory and Ownership

| Document | Path | Owner | Update frequency |
|----------|------|-------|-----------------|
| Root CLAUDE.md | `CLAUDE.md` | Session-level | Cycle A (state header), Cycle B (block status), Cycle C (resume checklist) |
| Reproducibility Manifest | `outputs/mechanistic_twin/phase2/REPRODUCIBILITY_MANIFEST.md` | Per-step | Cycle A (every step) |
| Deep Dive | `outputs/defense_prep/paper7_phase2_deep_dive.md` | Per-block | Cycle B (§7 subsections, Q&As) |
| mechtwin_review.tex | `outputs/dissertation/chapters/mechtwin_review.tex` | Per-block | Cycle B (§11 empirical results) |
| bioRxiv main.tex | `outputs/mechanistic_twin/paper7_bioRxiv/main.tex` | Per-block | Cycle B (when headline numbers change) |
| Dissertation ch09 | `outputs/dissertation/chapters/ch09_paper7.tex` | Per-block | Cycle B (mirror bioRxiv) |
| bibliography.tex | `outputs/dissertation/bibliography.tex` | Per-lit-search | Cycle B (new citations) |
| Sub-CLAUDE.md files | `outputs/*/CLAUDE.md`, `src/*/CLAUDE.md`, `scripts/CLAUDE.md` | Per-artifact | Cycle A (new artifacts) |
| Roadmap | `outputs/defense_prep/mechanistic_digital_twin_roadmap.md` | Per-session | Cycle C (module status) |
| RUN_MANIFEST companions | `outputs/mechanistic_twin/phase2/step_*_RUN_MANIFEST.md` | Per-step | Cycle A (auto-generated by `_reproducibility.py`) |
| GAP presentation | `outputs/gap_milestone/*.pptx` | Per-milestone | Manual (committee meetings) |

---

## 5. Validation Integration

Every document update that contains a numerical claim must trace to one of:

1. **A script output** (CSV, JSON, parquet) referenced in the reproducibility manifest
2. **A literature citation** validated via Consensus MCP / PubMed in a closed-loop Stage 1 sweep
3. **An analytical derivation** verified via `verify-math` or SymPy

If a number appears in a document and cannot be traced to one of these three sources, it is **undocumented** and must be either (a) traced and documented, or (b) removed.

**The traceability test:** For any number in any document, you should be able to answer: "What script produced this? What input data did it use? What RNG seed? What was the output hash?" If you can't, the number fails the traceability test.

---

## 6. Impact Assessment Protocol

When a block produces results that change previous assumptions, the following impact assessment fires:

1. **Identify affected documents.** Which documents reference the changed number/assumption?
2. **Classify impact.** Is this a (a) numerical update only (same conclusion, different number), (b) qualitative change (different conclusion), or (c) structural change (new section needed)?
3. **For (a):** Update the number in all affected documents. Log in the reproducibility manifest update log.
4. **For (b):** Re-run closed-loop Stage 4 (post-execution review) on the affected claims. Update or retract as needed.
5. **For (c):** Trigger a new Cycle B pass on the affected block.

**Example from 2026-04-10:** Block 6 expanded the cohort from 304→1,065. This was a type (a) impact on most documents (update cohort size) but type (c) on the bioRxiv manuscript (needed LOO rerun + new headline numbers + title change).

---

## 7. Versioning

This is v1.0, locked 2026-04-10. Updates follow the same closed-loop discipline as the methodology document.

**Change log:**

- **v1.0 (2026-04-10):** Initial lock. Three cycles (per-step, per-block, per-session). Integrated with closed-loop v1.1 as Stage 6.5. Motivated by documentation drift during Blocks 3-6 rapid execution. Document inventory, validation integration, and impact assessment protocol defined.
