# Closed-Loop Methodology System (v1.0)

**Status:** LOCKED 2026-04-08
**Scope:** GIMAN dissertation + post-dissertation Mechanistic Digital Twin (Papers 1–7 and all downstream work)
**Authority:** This specification OVERRIDES ad-hoc workflows. When in doubt, follow the loop.
**Parent:** [src/mechanistic_twin/CLAUDE.md](../src/mechanistic_twin/CLAUDE.md) Validation Discipline Clause

---

## 1. Why this exists

Three real failures in April 2026 motivated this system:

1. **Variant A F → 10⁷⁵ blow-up (Phase 2 Step 2.4).** A single-state adaptation of the Cohen 2013 / Knowles 2009 α-synuclein ODE was launched to NUTS without a local stability analysis. 11,640 seconds of compute were spent before anyone noticed that the Jacobian eigenvalue was in the right half-plane under literature-pinned rate constants (`k_frag − k_clear_F = +0.005 hr⁻¹`). A 30-second Jacobian check would have caught it.
2. **Original "first computational model" novelty overclaim.** The Phase 2 manuscript novelty claim was written confidently before the adversarial literature sweep identified three near-miss competitors (Bakshi 2018 *CPT:PSP*, Ivanova & Karelina 2024 *CPT:PSP*, Geerts 2023 *Sci Rep*). The claim had to be retracted and narrowed after it was already in the draft.
3. **"k_e absorbs fragmentation" unsupported claim.** The Variant B mass-conservation fix was initially framed as "k_e absorbs fragmentation's kinetic effect," a framing that the peer-review skill correctly identified as unsupported because Iljina 2016 measured `k_e` in a quiescent, non-fragmenting TIRF assay that explicitly excluded fragmentation by design.

In all three cases, the problem was visible *before* any claim was committed to a durable document, but the right skill/command did not fire at the right time. The closed-loop system exists to make the right skills fire automatically at the right decision points.

## 2. The core principle

**No scientific claim is committed to any durable document (markdown, LaTeX, commit message, manuscript, committee handout) until all six stages of the loop have passed for that specific claim.**

Durable documents include: anything under `outputs/defense_prep/`, `outputs/dissertation/`, `outputs/mechanistic_twin/phase2/`, `docs/plans/`, and any file that will be seen by the dissertation committee or a peer reviewer.

`/tmp/` files are exempt from the loop because they are explicitly ephemeral and scratch-pad. This is important: the loop does **not** slow down exploration. It slows down *commitment*.

## 3. The six stages

### Stage 1 — Literature grounding

**Fires BEFORE any new quantitative claim, parameter value, model form, or methodological approach is written to a durable document.**

Skills to fire (in parallel when possible):

- **`mcp__claude_ai_Consensus__search`** — ⭐ PRIMARY Stage 1 tool (locked 2026-04-09). Consensus MCP searches academic papers with best paywall coverage. Use for scoop checks, parameter validation, methodology precedent. Install: https://mcpmarket.com/tools/skills/consensus
- `claude-scholar:openalex` — seed-paper mode ONLY (never free-text sorted by citations, per Validation Discipline Clause item 5). Secondary to Consensus for specific seed-paper graph traversal.
- `claude-scholar:doi-bibtex` — for known DOI → BibTeX conversion
- `paper-lookup` — fallback across 10 academic databases (PubMed, bioRxiv, medRxiv, arXiv, Semantic Scholar, CORE, Crossref, Unpaywall)
- `parallel-web` — Parallel Chat API (core model) for multi-source research reports with inline citations, distinct from OpenAlex's seed-paper scope
- `bgpt-paper-search` — BGPT MCP server returning structured data extracted from full-text papers (25+ fields per paper including methods, results, sample sizes, quality scores). Distinct from abstract-level searches.
- `citation-management` — bulk Google Scholar + PubMed searching when the scope is a literature review, not a single claim

**Gate criterion:**

Every numerical value has a DOI OR every methodological choice cites a precedent OR the claim is explicitly marked "novel to this work, not precedented in literature" with justification. If none of these, the claim enters a holding state in `/tmp/` and does NOT move to Stage 2.

**Why parallel-web + bgpt-paper-search are in Stage 1 alongside OpenAlex:**

OpenAlex seed-paper mode is authoritative for finding *specific papers* but weak for *synthesis* and *structured data extraction*. Parallel-web is the opposite — strong for synthesized multi-source answers with citations, weaker for specific DOI lookup. BGPT is the third leg: it extracts quantitative data from full-text papers (sample sizes, effect sizes, methodology details) that are invisible to abstract-only searches. The combination of all three + paper-lookup + doi-bibtex covers the full search surface. Missing any one of them risks the exact class of error that Bakshi 2018 and Ivanova 2024 slipped past in the original novelty claim.

**Exit:** All three parallel searches have returned AND full-text verification has been performed on at least the top 3 hits per search. Stage 2.

### Stage 2 — Decision deliberation

**Fires when the literature (Stage 1) returns multiple plausible directions and a choice must be made between them.**

Skills to fire:

- `consciousness-council` — 6-perspective structured deliberation with at least one devil's-advocate archetype. This is what caught the FK pivot mistake. Must fire **every time** there is more than one plausible direction.
- `what-if-oracle` — 10-year scenario analysis for "what does each path look like under best case / worst case / surprise?" Useful when the decision has long-term consequences beyond the current sub-step.
- `scientific-critical-thinking` — GRADE evidence quality rating per candidate option
- `hypothesis-generation` — only if the decision involves generating new testable hypotheses

**Gate criterion:**

At least 3 perspectives heard AND the devil's-advocate has explicitly argued for rejection AND the synthesis identifies an explicit "core tension" and "blind spot." If any of these is missing, the decision is not committed.

**Exit:** The decision is documented with its rationale, the alternatives that were rejected, and why. Stage 3.

### Stage 3 — Pre-execution sanity check

**Fires BEFORE any long-running computation (Bayesian calibration, full-cohort run, expensive ML training, symbolic search).**

**This is the most important gate. It would have caught the Variant A failure in 30 seconds.**

Skills to fire:

- `claude-scholar:verify-math` — symbolic verification of ODE derivations, steady-state analysis, Jacobian eigenvalues, transform invariants. This is the one that must fire before any ODE calibration.
- `sympy` — escape hatch for symbolic algebra beyond verify-math's built-in capabilities
- `hypothesis-generation` — devil's-advocate: what am I assuming that I haven't tested?
- `scientific-critical-thinking` — GRADE quick-pass on assumptions going into compute

**Gate criterion:**

1. Any ODE-based calculation MUST have a local stability analysis (Jacobian eigenvalues at the expected operating point) verified before the solver is invoked.
2. Any Bayesian calibration MUST have a structural identifiability proof on the fit set (e.g., `StructuralIdentifiability.jl` + `SIAN.jl` cross-check).
3. Any literature synthesis claim MUST have full-text verification of the top-3 cited papers, not abstract-level only.
4. Any compute run > 1 hour in wall-clock MUST have a 1-patient or 1-sample smoke test complete successfully first.

**NO EXCEPTIONS.** The Variant A failure was caused by skipping exactly this gate. The rule is now: if you are about to call `Bash` on a long-running compute, stop and confirm that the corresponding sanity check has been completed and archived.

**Exit:** Sanity check artifacts exist on disk under `/tmp/` or `outputs/mechanistic_twin/phase{N}/validation/`. Stage 4.

### Stage 4 — Post-execution review

**Fires after compute results exist on disk and BEFORE those results get written into any claim in a durable document.**

Skills to fire:

- `peer-review` — structured reviewer persona (the one that caught the "k_e absorbs fragmentation" error)
- `claude-scholar:critique-manuscript` — reporting-guideline compliance (IMRAD, CONSORT, STROBE, PRISMA as applicable)
- `scholar-evaluation` — ScholarEval quantitative scoring on problem formulation / methodology / analysis / writing
- `scientific-critical-thinking` — second-pass GRADE evaluation on what the results actually show

**Gate criterion:**

At least TWO of the four skills must run and pass. If any skill flags a MAJOR objection, that objection must be either (a) addressed by revising the approach, (b) explicitly acknowledged in the target document as a limitation, or (c) rejected in writing with justification.

**The Hidden Assumptions Audit §3.7 of paper7 is an example of this gate firing correctly — but it fired AFTER the deep dive was written, when it should have fired DURING.** Going forward, Stage 4 fires inline with the write, not as a bolt-on afterward.

**Exit:** All MAJOR objections from the review skills have been addressed or acknowledged. Stage 5.

### Stage 5 — Independent validation

**Fires before a claim is committed. The claim must be independently validated against at least two sources.**

Skills to fire:

- `claude-scholar:check-refs` — every citation resolves to `bibliography.tex` OR an in-markdown inline DOI
- `claude-scholar:verify-math` — second-pass math verification on any equations in the target document
- `claude-scholar:presubmit-checks` — parallel run of check-refs + latex-cleanup + build + frontmatter (for LaTeX documents)
- `pyzotero` — DOI → Zotero round-trip for any new references
- **Cross-skill convergence check** — run `scholar-evaluation` AND `peer-review` on the same draft; if they converge on the same top-3 concerns, confidence is high; if they disagree, investigate the disagreement before committing.

**Gate criterion:**

Any claim that will be published MUST have at least two independent validation sources agreeing:

- Numerical claims: source literature + independent reproduction
- Methodological claims: peer-review skill + critique-manuscript skill (must converge)
- Citation claims: DOI resolves + the cited passage actually supports the claim being made (requires full-text check)

**Exit:** Two-source agreement documented. Stage 6.

### Stage 6 — Decision gate (APPROVE / MODIFY / ABANDON)

**The final gate before the claim enters a durable document.**

Skills to fire:

- `compound-engineering:document-review` — final structural review
- `claude-scholar:presubmit-checks` — final parallel sanity sweep
- `compound-engineering:review:code-simplicity-reviewer` (or equivalent) — is this the simplest defensible claim, or are we over-reaching? YAGNI check for claim scope.

**Three outcomes:**

**APPROVE** — All prior stage gates passed AND the claim is the narrowest defensible version of the finding. Write to the document.

**MODIFY** — At least one gate flagged a minor concern. The claim needs rewording, narrowing, or additional context. Return to Stage 1 with the specific modification (e.g., "add prior sensitivity caveat", "reframe as hypothesis", "soften the rhetoric").

**ABANDON** — At least one gate flagged a major concern that cannot be addressed within current scope. The claim does NOT enter the document. A note is written to `/tmp/abandoned_claims.md` explaining the claim, which gate blocked it, and why. This prevents future sessions from re-attempting the same discarded framing.

**Exit:** Document is updated. Loop closes. Next claim starts again at Stage 1.

## 4. The four mandatory behavior changes

These are the concrete operationalizations of the loop:

### Change 1 — Mandatory `claude-scholar:verify-math` before any ODE calibration

**Rule:** For any Bayesian calibration or forward simulation of a new or modified ODE, run `verify-math` on the Jacobian stability analysis BEFORE launching the compute.

**What this prevents:** Variant A class failures where the ODE is launched with an unstable eigenvalue structure and burns compute before the problem is noticed.

**Enforcement:** Before any `Bash` call on a Julia script that invokes a Bayesian sampler or an ODE solver, the agent must have already run verify-math (or equivalent symbolic analysis via sympy) on:
- Steady-state solutions at expected parameter values
- Jacobian eigenvalues at the operating point
- Transform invariants (for state transformations like log-N)

### Change 2 — Mandatory 3-source adversarial literature sweep before any novelty claim

**Rule:** For any sentence containing "first," "novel," "unprecedented," "only," "has never," or equivalent, run adversarial literature sweeps across **three independent sources** AND full-text verification of the top-3 near-misses BEFORE the sentence is committed.

**The three sources are:**

1. **`mcp__claude_ai_Consensus__search`** — ⭐ PRIMARY (best paywall coverage, structured academic search)
2. `claude-scholar:openalex` in seed-paper mode (authoritative for citation-graph traversal)
3. `parallel-web` Chat API or `bgpt-paper-search` (authoritative for multi-source synthesis / structured full-text extraction)

**Plus `paper-lookup` as a 4th fallback** to cover PubMed, bioRxiv, medRxiv, arXiv, Semantic Scholar, CORE, Crossref, Unpaywall.

**What this prevents:** Novelty overclaims that get caught in peer review because a closely related paper exists in a database the original search missed (Bakshi 2018 in the PubMed/CPT:PSP axis, Ivanova 2024 in CPT:PSP, Geerts 2023 in Nature portfolio — all three missed by the initial OpenAlex-only search).

**Enforcement:** Any markdown/LaTeX edit that adds a novelty claim triggers Stage 1 with all four sources (OpenAlex + parallel-web + bgpt-paper-search + paper-lookup) run in parallel. The agent does not commit the claim until the sweeps return and the top-3 hits per source are full-text-verified.

### Change 3 — Mandatory post-write review on any durable document ≥ 3,000 words

**Rule:** For any document under `outputs/defense_prep/`, `outputs/mechanistic_twin/phase{N}/`, `outputs/dissertation/`, or `docs/plans/` that is ≥ 3,000 words, run `peer-review` and `claude-scholar:critique-manuscript` in parallel IMMEDIATELY after the Write call finishes, BEFORE reporting "done" to the user.

**What this prevents:** The §3.7 Hidden Assumptions Audit class of bolt-on fix. The first version of a long deep dive should already be post-audit when it's reported as done, not awaiting a later review pass.

**Enforcement:** The agent builds post-write review into the write workflow itself. If the review surfaces MAJOR objections, the write is not considered complete until they are either addressed or acknowledged.

### Change 4 — `/tmp/abandoned_claims.md` audit trail

**Rule:** When the Stage 6 decision gate produces ABANDON on a candidate claim, the reason is written to `/tmp/abandoned_claims.md` with:

- The exact claim text that was proposed
- Which gate blocked it (Stage 1 literature, Stage 4 review, etc.)
- What specific skill output triggered the rejection
- Whether the claim was abandoned permanently or could be revisited with new data

**What this prevents:** Future sessions (including compaction-resumed sessions) re-attempting the same discarded framing because they don't know we already tried and rejected it.

**Enforcement:** The agent writes an entry to `/tmp/abandoned_claims.md` on every ABANDON verdict. On session start, the agent reads the file if it exists.

## 5. When the loop applies vs when it doesn't

### Loop applies

- Writing to any file under `outputs/defense_prep/`, `outputs/dissertation/`, `outputs/mechanistic_twin/phase{N}/`, `docs/plans/`
- Any bash command that launches a long-running (>1 hour) computation
- Any commit message or PR description that makes a scientific claim
- Any committee-facing or reviewer-facing document
- Any deep dive, manuscript, or defense prep material

### Loop does NOT apply

- Scratch work in `/tmp/`
- Exploratory scripts in `src/mechanistic_twin/test/` that are not yet committed to the main test suite
- Debugging print statements in active code
- Conversation messages to the user that are not written to files
- Code edits that do not change scientific claims (linting, refactoring, typo fixes)

The point is: **exploration is fast and cheap; commitment is slow and deliberate**. The loop only gates commitment.

## 6. How the loop is invoked

### By the agent (default mode)

The AI agent (Claude) invokes the loop **automatically** when it detects that it is about to make a durable claim. This is the default mode and is what the "mandatory behavior changes" section above enforces.

### By the user (explicit override)

The user can explicitly invoke any stage of the loop by naming the skill or command:

- "Run the closed-loop literature check on this claim" → fires Stage 1
- "Run the closed-loop sanity check before I launch this Julia script" → fires Stage 3
- "Run the closed-loop review pass on paper7" → fires Stage 4
- "Run the full loop on this claim" → fires all 6 stages sequentially

### Skipping stages

**No stage can be skipped without explicit user approval.** If the agent encounters a situation where a stage does not apply (e.g., Stage 2 deliberation on a single-option decision), the agent must explicitly note that the stage was skipped and why, in the document where the claim is committed.

## 7. Data & Literature Assumption Registry (Change 5, added 2026-04-10)

**Rule:** Every fixed parameter value, data source, model design decision, and literature-grounded assumption MUST have an entry in the canonical **Data & Literature Assumption Registry** at `outputs/mechanistic_twin/phase2/DATA_LITERATURE_REGISTRY.md`.

**The registry has 6 sections:**
1. **ODE Parameters** — every hardcoded constant with value, units, literature source, codebase location
2. **Prior Distributions** — every Bayesian prior with anchoring evidence
3. **Data Sources** — every PPMI observable with patient counts, DaT overlap, ODE compartment mapping
4. **Model Design Decisions** — every architectural choice with alternatives considered and justification
5. **Empirical Findings** — every result from this project with validation method and supporting literature
6. **External Validation Targets** — every cohort identified for future validation

**Update triggers:**
- **Stage 1 (Literature grounding):** When a new paper validates or refutes an assumption → update the relevant row's "Supporting/Refuting literature" column and "Status"
- **Stage 6.5 (Documentation Lifecycle Cycle A):** When a new parameter is fixed or a data source is discovered → add a new row
- **After every SAEM/IS/HLME run:** When empirical findings change → update "Empirical Findings" section

**The traceability test:** For any value in the registry, you must be able to answer: "What paper says this? What script uses it? What data supports it?" If any column is empty, the entry is INCOMPLETE.

**Motivation (2026-04-10):** During this session, we discovered that Brockmann 2025 (npj PD) validates our slow-fast timescale separation assumption (SD50 stable over time), Mollenhauer 2019 confirms CSF total α-syn doesn't reflect progression, and Powell 2018 shows population-average connectomes are sufficient — but none of these were tracked in any structured document. The registry ensures that literature findings are captured at discovery, not reconstructed later.

---

## 8. Relationship to existing project infrastructure

### Validation Discipline Clause (inherited)

The Validation Discipline Clause in [src/mechanistic_twin/CLAUDE.md](../src/mechanistic_twin/CLAUDE.md) is the *philosophical* foundation of this system. The closed-loop is the *operational* implementation of that philosophy. Where the Validation Discipline Clause says "no rationalized failures, honest verdicts, literature anchors" — the closed-loop specifies *how* those principles get enforced at each decision point.

### Modular CLAUDE.md system

Each sub-directory CLAUDE.md inherits the closed-loop system. The Phase 2 work under `src/mechanistic_twin/` and `outputs/mechanistic_twin/phase2/` is where it was first invented and is most stringent; other directories apply it as appropriate to their scope.

### Skill registry

The skills and commands invoked by the loop are all already available in the user's skill library (cataloged in `commandlist/skills_reference.md` and `commandlist/plugin_commands.md`). The loop does not require any new tools — it is entirely a reorganization of *when* existing tools fire.

## 8. Versioning and updates

This is version 1.0, locked 2026-04-08. The specification is expected to evolve as we learn which gates fire most usefully. Updates follow the same closed-loop discipline:

- Proposed changes are discussed with the user before being committed
- The old version is archived at `docs/closed_loop_methodology_v{N-1}.md`
- The parent CLAUDE.md summary is updated to point to the new version

**Change log:**

- **v1.0 (2026-04-08):** Initial lock. Six stages, four mandatory behavior changes. Motivated by Variant A F → 10⁷⁵ failure, original novelty overclaim, and "k_e absorbs fragmentation" unsupported claim (all April 2026 Phase 2 work).
- **v1.1 (2026-04-10):** Added `mcp__claude_ai_Consensus__search` (Consensus MCP) as PRIMARY Stage 1 tool. Updated Change 2 three-source list.
- **v1.2 (2026-04-10):** Added **Stage 6.5 — Documentation Lifecycle**. Three cycles: Cycle A/B/C. Full spec: [docs/documentation_lifecycle_protocol.md](documentation_lifecycle_protocol.md).
- **v1.3 (2026-04-10):** Added **Change 5 — Mandatory non-circular decisive test before claiming degeneracy-breaking.** When adding observable Y to break parameter degeneracy in a model fit to observable X, compute `ρ = Spearman(posterior_θ_from_X_only, Y_observed)` BEFORE running joint X+Y posterior. If ρ ≈ 0, the joint model tightening is model-internal (not independently verifiable). Motivated by Phase 2.5 SAA integration where SBR-only k_n showed ρ=-0.01 with SAA TTT (p=0.95) despite joint model ρ=-0.891. Full lesson: [docs/lessons/2026-04-10-identifiability-validation-protocol.md](lessons/2026-04-10-identifiability-validation-protocol.md).
- **v1.5 (2026-04-11):** Added **Change 6 — Batch claim audit for rapid-iteration sessions.** When >3 claims are committed to durable documents in a single session, fire a batch Stage 4 review at session end (not per-claim) to catch claims that bypassed the loop during rapid iteration. Motivated by the 2026-04-11 Phase 3 session where 7 claims were committed during a 10+ hour session involving SBC debugging, and Stage 4 (peer-review) was skipped on several claims due to the speed of iteration. The batch audit must: (a) enumerate every claim committed to durable documents during the session, (b) verify each claim's Stage 1-6 compliance, (c) flag any claim that skipped a stage, (d) independently verify the top numerical claims (recompute from saved artifacts). Session audit template at `/tmp/session_YYYY_MM_DD_claim_audit.md`. Also added **Change 7 — Mandatory independent numerical verification.** Any quantitative claim written to a paper (main.tex) must be independently recomputed from the saved output files (JSON, parquet, CSV) before the paper is submitted. The verification must use a separate code path from the producing script. Motivated by the SNR=0.043 claim in Paper 8a: the stat agent reported 0.043 (max per-observation), but independent verification found mean=0.010 and max=0.047 — same order of magnitude but the exact number reported in the paper was imprecise.

## 9. Acknowledgments

The six-stage structure was synthesized from three real failures during Phase 2 of the GIMAN Mechanistic Digital Twin project and from the three-skill devil's-advocate review (`scientific-critical-thinking` + `scholar-evaluation` + `peer-review`) that caught the §3.7 Hidden Assumptions Audit issues in paper7. The specification was locked at the user's direct request after they observed the recurring failure pattern and asked for a systematic prevention mechanism.

The loop owes its core principle — "exploration is fast and cheap; commitment is slow and deliberate" — to the Bayesian workflow position paper (Gelman et al. 2020 arXiv 2011.01808) and to the validation discipline established in Phase 1 of this project.
