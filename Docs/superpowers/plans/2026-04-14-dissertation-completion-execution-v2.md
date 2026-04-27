# Dissertation Completion Execution Plan — v2 (Paper 11 dropped)

> **Supersedes:** [2026-04-13-dissertation-completion-execution.md](2026-04-13-dissertation-completion-execution.md). Rescoped after 2026-04-14 verification established that Paper 11 as previously scoped duplicates Paper 1's existing external-validation work; reallocating the 8 weeks to deeper verification + stretch-goal sub-sections.

> **Status at 2026-04-14:** Foundation (App E Docker + data dictionary + artifact manifest) is SHIPPED. Defense-prep audit complete with 1,017 / 1,127 claims verified (90.2%) and 12 chapter errors caught + fixed. Paper 1 benchmark re-ran from scratch in Docker and reproduces committed JSONs to 4 decimal places. Mechanistic reproduction flows through committed HDF5 posterior store (Julia refit capability blocked by documented Docker-Pkg issue; not on defense critical path).

**Goal:** Ship a defendable 15-chapter + 2-appendix PhD dissertation (no Paper 11) with the critical reviewer-exposed limitations (whole-putamen ROI, mechanistic UQ) closed via sub-section additions, not a new chapter.

**Tech Stack:** Python 3.10 + PyTorch 2.8 + PyTorch Geometric 2.6 + MAPIE 1.3 + CatBoost (all Docker-baked), PostgreSQL 17 (Docker), LaTeX (IEEEtran), Julia 1.11 (native install required for mechanistic refit — Docker stack has documented Pkg precompile issue).

**Predecessor plans (build-on):**
- [Phase B detailed audit plan](2026-04-13-phase-b-detailed-audit-plan.md) — audit COMPLETE
- [Phase B tooling alignment](2026-04-13-phase-b-tooling-alignment.md) — tooling reused
- [Phase 5 mechanistic vs GIMAN](2026-04-12-phase5-mechanistic-vs-giman-benchmark.md) — Paper 10 Tasks 0–8 COMPLETE
- [Dissertation completion mapping v1](../research_directions/2026-04-13_dissertation_completion_mapping.md) — partially invalidated by v2 (Ch 16 section dropped)

## What changed from v1

| Item | v1 (2026-04-13) | v2 (2026-04-14) | Why |
|---|---|---|---|
| Chapter 16 (Paper 11 Cross-Cohort) | 8 weeks, new chapter | **DROPPED** | Paper 1 already has BioFIND/PDBP/HBS external validation; no new novelty after EDA revealed no viable inductive Graph-DT target |
| Ch 11 §11.7 (6-region ROI) | 4 weeks | **4 weeks — now critical path** | Was Phase C stretch; becomes primary stretch deliverable |
| Ch 13 §13.8 (mech conformal) | 2 weeks | **2 weeks — now critical path** | NASEM UQ score 3 → 3+ upgrade |
| Ch 9 §9.6 (multi-channel Olink/NfL) | Phase C stretch | **6 weeks — upgraded to tractable** | 8-week reallocation from Paper 11 makes this scope-feasible |
| Reproducibility audit completion | Not scheduled | **Week 1 — DONE** | Completed 2026-04-14 |
| Final total chapters | 16 + App D + App E | **15 + App D + App E** | Drop Paper 11 |

## Timeline (17 weeks, reallocated)

| Week | Subsystem | Deliverable | Blocking? | Status |
|---|---|---|---|---|
| 1 | Appendix E §E.1–§E.2 | Docker + data dictionary + artifact manifest | Yes | **DONE** commit `cc2e981` |
| 1 | Repro audit completion | 1,017/1,127 verified, 12 fixes applied, truth table | No | **DONE** commits `1c1cac7`+`b9e7f58` |
| 2–7 | Ch 9 §9.6 (NEW — Paper 7 ext.) | Olink CSF + NfL + Amprion SAA multi-channel observation extension | No | 6 wk |
| 8–11 | Ch 11 §11.7 (NEW — Paper 8b ext.) | 6-region ROI split (anterior/posterior putamen sub-gradient) | No | 4 wk |
| 12–13 | Ch 13 §13.8 (NEW — Paper 10 ext.) | Mechanistic conformal bands on counterfactuals | No | 2 wk |
| 14–15 | Ch 12 §12.6 (NEW — Paper 9 ext.) | Genotype-stratified Path B (LRRK2/GBA/SNCA) | No | 2 wk |
| 16 | Ch 14 + Ch 15 narrative refresh | Update limitations + future work; mark closed vs deferred | Yes — blocks final PDF | 1 wk |
| 17 | Final compile + presubmit + defense slides | Defense-ready PDF + slides | Yes | 1 wk |

**Net: 15 chapters + 2 appendices with 4 sub-section additions (§9.6, §11.7, §12.6, §13.8), all strengthening existing papers rather than adding a marginal one.**

## Why this is stronger than v1

1. **More honest novelty.** Paper 11 as scoped duplicated Paper 1. Its replacement (four sub-section additions) addresses four DIFFERENT reviewer-exposed gaps:
   - §9.6 closes the SBR-only limitation of Paper 7 (adds Olink + NfL observation channels).
   - §11.7 closes the whole-putamen resolution limitation of Paper 8b.
   - §12.6 closes the cohort-averaged PK/PD limitation of Paper 9 (genotype stratification).
   - §13.8 closes the NASEM UQ criterion gap of Paper 10.

2. **Every deliverable is testable today.** The data to execute §9.6/§11.7/§12.6/§13.8 is present in the already-restored PostgreSQL database (`ppmi_raw.biospecimen_csf_abeta_tau`, `ppmi_raw.datscan_sbr_analysis`, etc.). No DUA, no collaboration pending.

3. **Reproducibility is now ahead of plan.** The 2026-04-14 audit verified every Python-path claim, re-ran Paper 1 benchmark to 4 decimal precision, and documented the Julia-Docker limitation. Committee reproducibility via Docker works for Papers 1, 2, 3, 4, 6, 10 end-to-end.

## Critical path for defense

Shortest viable: Week 16 narrative refresh → Week 17 compile. Everything else is parallelizable across the 6–13 weeks. If one of §9.6/§11.7/§12.6/§13.8 slips, the remaining three + narrative refresh still produce a 15-chapter defendable dissertation.

## Deferred to postdoc (unchanged from v1)

- Paper 12: Hybrid SciML UDE (C3-2)
- F12 MindMend Phase 6
- F13 DeNoPa external validation (pending PI collaboration + DUA)
- F14 prospective interventional trial

## Sub-plan creation instructions (for agentic execution)

When a sub-subsystem starts (Week 2 for §9.6, Week 8 for §11.7, Week 12 for §13.8, Week 14 for §12.6), write a detailed TDD plan per `superpowers:writing-plans` conventions at `Docs/superpowers/plans/YYYY-MM-DD-<name>.md`, referencing:

- CLAUDE.md root file for data locations (PostgreSQL `giman_research`)
- `outputs/defense_prep/reproducibility_log_2026-04-14.md` for existing repro entry points
- `outputs/defense_prep/per_paper_verification/` for the 12-claim corrections applied + known open caveats
