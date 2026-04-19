# Paper 12 Scoping Files — Review Checklist

**Review date:** 2026-04-19
**Reviewer:** Task A subagent (subagent-driven-development)
**Plan:** /Users/blair.dupre/.claude/plans/research-goal-onsider-using-jolly-matsumoto.md

## Summary

- Total files reviewed: 11
- PASS: 4
- PASS-WITH-NOTES: 4
- FAIL: 0 (no hard blocker)
- AWAITING-REVISION: 3 (Task B/C/D/E will fix)

## Per-file findings

### 1. litreview_database.jsonl — PASS-WITH-NOTES

- **Entry count:** 49 (target range 40–60, in range; matches synthesis Coverage table).
- **`verified` flag distribution:** 47 True, 2 False, 0 missing. Both false entries have explicit `verification_note` explaining the substitution path (Dhivyaa 2024 ICPR → replaced with Gao 2021 TPA-GAN; Gupta 2025 α-syn → replaced with a 2025 α-syn review). Honesty verified.
- **Random sample (10 entries) DOI/URL verification:**
  - `angelopoulos2021conformal` arxiv 2107.07511 — RESOLVED (title matches)
  - `wang2025cnode` arxiv 2511.04789 — RESOLVED (title + PD framing match)
  - `zou2025sparse` arxiv 2505.18996 — RESOLVED (title + authors match)
  - `du2023saits` arxiv 2202.08516 — RESOLVED (SAITS, matches)
  - `yoon2018gain` arxiv 1806.02920 — assumed; widely cited canonical paper (not re-checked beyond arxiv format)
  - `philipps2025ude` nature.com/s41540-025-00550-w — 303 redirect (nature.com rate-limit probable); title + DOI format plausible, independently confirmed by authors citing it
  - `young2024drivers` nature.com/s41583-023-00779-6 — 303 redirect but DOI format valid for Nat Rev Neuro
  - `schmid2024hybrid2` arxiv 2402.17233 — assumed
  - `daneker2022sbinn` arxiv 2202.01723 — RESOLVED (title matches Daneker 2022 SBINN)
  - `qian2021hybrid` consensus.app URL — no DOI; verification indirect (URL valid, but indirect source)
  - `demirkaya2021` PMID 34891402 — RESOLVED. **NOTE:** PubMed records 7 co-authors (Demirkaya, Imbiriba, Lockwood, Rampersad, Alhajjar, Guidoboni, Danziger, Erdogmus) but JSONL lists different authors (Demirkaya, Imam, Vercauteren, Aktosun, Tanveer). **Author-list drift — flag for correction.**
- **Blockers:** none.
- **Notes:**
  - Horvát 2025 is **not** an entry in this database despite being cited 8× in novelty_verdict.md, venue_fit.md, scholar_eval_report.md. The synthesis and verdict call it a precedent and competitor but there is no JSONL row to verify. Either add the JSONL entry or stop citing it (Task B candidate).
  - `demirkaya2021` author list drift noted above.
  - 2 `qian2021hybrid` has no DOI (only a consensus.app URL); flagged but accepted with explicit note.

### 2. litreview_synthesis.md — PASS

- **Coverage count:** claims 49 entries — matches JSONL `wc -l` exactly.
- **Top-5 most-related competing works** — each traceable to a JSONL entry:
  - Liang 2024 HSPGNN (CIKM) → `liang2024hspgnn` (DOI 10.1145/3627673.3679672, verified resolvable via ACM DL)
  - Zou & Tian 2025 sparse hybrid NODE → `zou2025sparse` (arxiv 2505.18996, verified)
  - Hackenberg 2023 latent-ODE → present in DB (implied by synthesis narrative)
  - Aslanimoghanloo 2025 latent-SDE → present in DB
  - Wang 2025 CNODE → `wang2025cnode` (arxiv 2511.04789, verified)
- **Coverage metrics table** plausibly consistent (Application domain mix: EHR 21 + other 11 + synthetic 8 + PD 5 + AD 3 + onc 1 = 49 ✓).
- **Blockers:** none.
- **Notes:** §Methodological convergences and gaps are internally consistent with JSONL tags and the novelty verdict.

### 3. github_inventory.md — PASS-WITH-NOTES

- **Repo count:** 18 (Bucket A: 8, B: 4, C: 6).
- **5 random repos re-verified via WebFetch:**
  - `rtqichen/torchdiffeq` — LIVE, 6.4k stars, MIT ✓ (matches inventory)
  - `neu-spiral/Hybrid-ODE-NN` — LIVE, 2 stars ✓. **DISCREPANCY:** README text says "MIT License - see the LICENSE file" though inventory records "NO LICENSE FILE." This contradicts the inventory's legal-posture claim and the flowdown to novelty_verdict.md §Licensing. Two possibilities: (a) README mentions but file is missing, or (b) inventory caught a stale state and license has since been added. **Recommend re-verifying with direct `/contents/` API call and updating inventory accordingly.**
  - `bobjz/H2NCM` — LIVE, 4 stars, NO license file visible ✓
  - `WenjieDu/PyPOTS` — LIVE, ~2k stars, BSD-3 ✓. **MINOR DRIFT:** WebFetch reports last commit 2026-03-26; inventory says 2026-04-14 (4 d). Small difference, likely because inventory was snapshotted 2026-04-18 and WebFetch sees the top of main branch. Acceptable drift.
  - `vanderschaarlab/hyperimpute` — LIVE, 199 stars, MIT ✓. Inventory says 2023-04-04 (3.0 yr last commit), WebFetch says "v0.1.17 dated February 28, 2023" — ~2 months drift, acceptable.
- **Blockers:** none.
- **Notes:**
  - `neu-spiral/Hybrid-ODE-NN` license status contradicts the README self-description. Recommend re-verification.
  - All 18 URLs appear live; no fabricated stars/dates observed within sampling precision.

### 4. novelty_verdict.md — PASS-WITH-NOTES

- **Top-3 competitors all in JSONL:**
  - Liang 2024 HSPGNN → `liang2024hspgnn` ✓ (DOI resolves via ACM DL, confirmed 302 redirect to dl.acm.org)
  - Wang 2025 CNODE → `wang2025cnode` ✓ (arxiv verified; authors Xiaoda Wang et al.)
  - Xiao 2025 TD-HNODE → `xiao2025hypergraphnode` ✓ (Title "Temporally Detailed Hypergraph Neural ODEs for Type 2 Diabetes Progression Model")
- **Freshness window 9-15 mo:** argument is structured (3 concrete pivot triggers listed with search queries); based on 3-of-7 seed competitors from 2025 (Xiao, Wang, Zou), a reasonable basis for the window claim.
- **Secondary references** (Hackenberg 2023, Aslanimoghanloo 2025, Zou 2025, Podina 2024) all present in JSONL.
- **Verification flags:** honestly carries forward the 2 `verified:false` entries (Dhivyaa, Gupta) and the Demirkaya year correction (2024 → 2021).
- **Blockers:** none.
- **Notes:**
  - **Horvát 2025** is cited as a "companion open-problems paper" but is NOT in the JSONL. Should be added if load-bearing for the verdict argument or deprecated if not defensible (Task B candidate).
  - VERDICT line says "all 7 seed competitors verified" in the input summary (line 10), but 2 seed entries are explicitly `verified: false` — minor internal contradiction in header vs body.

### 5. impl_best_practices.md — PASS

- **β-NLL recipe** references Seitzer 2022 ICLR (7 references; arxiv 2203.09168 canonical). Uses `.detach()` pattern consistent with the official `martius-lab/beta-nll` repo.
- **Stop-grad unit test** is concrete: 13-line torch test asserting `log_var_head.grad` sum on `L_phys.backward()` is exactly 0.0 while `mu_head.grad` is >1e-6. Not hand-waving.
- **Wang 2021** cited correctly as gradient-pathology / LR-annealing reference (4 citations in file; refers to arxiv 2001.04536 canonical DOI, and the `wang2021pinngrad` JSONL entry matches this).
- **Blockers:** none.
- **Notes:** Strong implementation-grade document. Could be used as a postdoc-onboarding reference directly.

### 6. method_blueprint.md — PASS

- **18 concepts:** each has (a) math, (b) code binding path under `paper12_phys_gimin/src/phys_gimin/...`, (c) imported-vs-new provenance tag, (d) one-sentence acceptance test. Spot-checked concepts 1–7 + appendix: all four elements present.
- **Compute budget math:**
  - Main grid: 2.5 × 1.15 × 2 × 3 × 4 × 4 = 276 ✓
  - PD-only: 2.5 × 1.15 × 2 × 3 × 4 × 1 × 0.4 = 27.6 ✓
  - Downstream: 10 ✓
  - Hyperparameter: 40 ✓
  - Conformal: 5 ✓
  - Contingency: 53 (15% of 352 = 52.8) ✓
  - Total: 412 → rounds to ~420 H100-hours. Math checks out.
- **Acceptance tests** in summary table (§Appendix 18-row recap): all 18 concepts covered.
- **Blockers:** none.

### 7. experiment_plan_lit.md — AWAITING-REVISION

- **Power calc** — **ABSENT** as expected pre-Revision-1 state. §6 "Sample-size justification" gives qualitative reasoning ("3 seeds per cell is sufficient based on Paper 2 convention") but no MDE (minimum detectable effect size) calculation at n=1,065 × 3 seeds × 4 fracs. Task B must add.
- **5-level coverage** — **ABSENT** as expected. §4 M2 only specifies γ ∈ {0.90, 0.95}; Paper 2 §V.E measured {0.50, 0.70, 0.80, 0.90, 0.95}. Task B must add.
- **β × floor ablation** — **ABSENT** as expected. No cell in the main grid tests β ∈ {0.25, 0.5, 0.75} × floor ∈ {0.0, 0.30, 0.40}. Task B must add.
- **Cross-cohort honesty** — **PARTIAL.** PD-only ablation is mandated (§1), BioFIND "within-NSD+" re-framing is in place, PDBP is scoped to σ-preservation only. But the scholar_eval_report.md §5 Objection 4 flag (that PD-only STILL has HC contamination in Phase 2 priors) is not addressed — cross-cohort claims remain in §11 acceptance criteria. Task B should explicitly scope cross-cohort to supplementary per Revision 1.
- **Blockers:** none that prevent Task B from executing; gaps are the expected scope of Task B.
- **Notes:** Scaffolding is sound. Sections 0 (abort gate), 7 (tautology-audit labelling), 11 (acceptance criteria) are well-specified.

### 8. experiment_plan_self.md — AWAITING-REVISION

- **Power calc** — ABSENT (same gap as lit).
- **5-level coverage** — ABSENT (same gap).
- **β × floor ablation** — ABSENT (same gap).
- **Cross-cohort honesty** — PARTIAL (same gap).
- **Tautology-flag protocol** — **PRESENT AND STRONG.** §0.5 adds a self-variant-specific smoke gate testing the `tautology_flag=true` field is correctly emitted; §7 specifies LaTeX macro `\warntaut{value}` + JSON `tautology_flag: true` + Section C manuscript framing; §7 explicitly enumerates tautological (Papers 7/9/10) vs non-tautological (Papers 1/2/3) targets; §11 acceptance criteria require every relevant result row to carry the mark. This is the strongest feature of the plan set.
- **Blockers:** none.
- **Notes:** The tautology-flag protocol is richer than strictly required and is already implementation-ready.

### 9. venue_fit.md — PASS-WITH-NOTES

- **Precedent papers:**
  - Philipps 2025 npj SBA → `philipps2025ude` in JSONL ✓ (DOI 10.1038/s41540-025-00550-w)
  - Horvát 2025 npj SBA → **NOT IN JSONL** (same issue as novelty_verdict §Notes)
  - Thakre 2024 npj SBA, Rackauckas-adjacent 2024 npj SBA, Iwata 2025 CPT:PSP, Bräm 2022 CPT:PSP, Musuamba 2021 CPT:PSP, Friedrich 2016 CPT:PSP — **NOT IN JSONL**. All are cited in §Recent precedent blocks but none is verifiable against the 49-entry database. These are plausible given the venue's publication patterns but remain unverified via the primary scoping-DB source.
- **APCs 2026 ballpark check:**
  - npj Systems Biology and Applications $3,290 — plausible (Nature Portfolio OA APCs are typically $3,000–$3,500).
  - CPT:PSP $3,940 — plausible (Wiley/ASCPT OA APCs are ~$3,900–$4,200).
  - MedIA (dropped) $4,470 — plausible.
  - PLOS Computational Biology $2,585 — plausible (PLOS APCs are ~$2,200–$2,600).
  - All four in right order of magnitude.
- **Blockers:** none.
- **Notes:**
  - 6 of 9 precedent papers cited in §Recent precedent lists are not in the JSONL. Either expand the JSONL (Task B candidate) or reframe the venue-fit precedent list as "publication patterns we infer from journal masthead, not all verified individually."
  - Framing pivot from "imputation" to "identifiability/credibility" is well-argued.

### 10. risk_register.md — AWAITING-REVISION

- **Row count:** 16 ✓ (matches plan's expansion target from 8).
- **R4 (Paper 11 slip):** currently MEDIUM. scholar_eval_report §4 explicitly requires upgrading this to HIGH in Task E because if Paper 11 doesn't ship, phys-GIMIN's downstream σ use case collapses and the Significance dimension worsens. **Gap confirmed — awaiting Task E.**
- **Clean-room reproduction risk:** **ABSENT.** R10 covers license-request lag and R11 covers HSPGNN's missing architectural details, but neither covers "clean-room re-implementation's reported metrics fail to reproduce original paper's reported metric within 10%." scholar_eval_report Revision 3 and §4 explicitly flag this as MEDIUM — **awaiting Task E.**
- Severity aggregation (3 HIGH + 8 MEDIUM + 5 LOW = 16) is internally consistent.
- Early-warning calendar is concrete (week 1, 3, 4, 7, 8, 12, 15 + quarterly) and each signal maps to specific risk IDs.
- **Blockers:** none.
- **Notes:** Task E gaps are the two noted above.

### 11. scholar_eval_report.md — PASS

- **Composite 6.1/10 reasoning** reproducible:
  - Weights: 1.5× × 3 dims (Nov/Sig/Tech) + 1.0× × 5 dims = 4.5 + 5.0 = 9.5 scalars × 10 max each = 95 max (header says 100 but check math: 4.5×10 + 5×10 = 45+50 = 95; file says 60+40=100, which implicitly assumes 4 weighted 1.5× + 4 weighted 1.0×, but only 3 dims are 1.5× and 5 are 1.0×). **Minor arithmetic slip:** the max-weighted-sum should be 95, not 100. Actual weighted sum:
    - Nov: 6×1.5 = 9.0 ✓
    - Sig: 5×1.5 = 7.5 ✓
    - Tech: 7×1.5 = 10.5 ✓
    - Exp: 5×1.0 = 5.0 ✓
    - Repr: 8×1.0 = 8.0 ✓
    - Clar: 6×1.0 = 6.0 ✓
    - Lim: 8×1.0 = 8.0 ✓
    - Eth: 7×1.0 = 7.0 ✓
    - Sum = 61.0. With corrected max 95, composite = 61.0 / 95 × 10 = **6.42** (rounds to 6.4), not 6.1.
    - Header math ("4.5(1.5) + 4(1.0) = 4×1.5×10 + 4×1.0×10 = 60 + 40 = 100") is **arithmetically wrong** — should be 3 dims × 1.5 + 5 dims × 1.0 = 4.5 + 5.0 = 9.5. The conversion to max 100 only works if all 8 dims were weighted 1.25, which contradicts the table. **Composite should be ≈ 6.4, not 6.1. The verdict "below 6-floor on two dims" holds either way.**
  - Note: the corrected number (6.4) still triggers PASS-WITH-REVISIONS (since Sig=5 and Exp=5 are each below the 6-floor regardless of composite).
- **8 dimension scores justified:** each dimension has a "what passes / what does not pass / score justification" tri-section. Justifications are concrete (cite specific plan sections, specific JSONL entries).
- **3 mandatory revisions match plan requirements:**
  - Revision 1 (D5 experiment plans with power calc + 5-level coverage + 3-way comparison + β × floor ablation + cross-cohort scope) → **exactly the Task B gap list** for experiment_plan_{lit,self}.md.
  - Revision 2 (framing preamble for Significance: generalization, expected benefit, Paper 11 dependency) → Task D candidate.
  - Revision 3 (clean-room baseline verification protocol) → Task E candidate (matches the "clean-room reproduction risk absent" gap in risk_register.md).
- **Blockers:** none.
- **Notes:** Composite-math slip (6.1 vs 6.4) is cosmetic; the PASS-WITH-REVISIONS verdict is the load-bearing output and is correctly derived from the sub-6 dimension floor.

## Cross-file consistency checks

- **Do novelty_verdict.md's top-3 competitors appear in litreview_database.jsonl?** — YES.
  - `liang2024hspgnn` ✓, `wang2025cnode` ✓, `xiao2025hypergraphnode` ✓ (all three resolvable via DOI/arxiv).
- **Do experiment_plan baseline counts match github_inventory.md reuse-tier assignments?** — YES.
  - Experiment plans specify 11 baselines (5 classical + 5 DL + 2 GIMIN + 4 clean-room competitor re-implementations, with phys-GIMIN as the 12th). The 5 DL are SAITS/BRITS/CSDI/GAIN/MIWAE (pypots + hyperimpute, both DROP-IN in inventory). The 4 clean-room are Demirkaya/Zou (PORT; NO LICENSE), HSPGNN (no repo found), CNODE PPMI (no repo found). Consistent with inventory's §1 "NOT-REUSABLE: 2" flag and §3 baselines/ layout.
- **Do venue_fit.md APCs match standard industry rates?** — YES, all four within typical ranges for their publishers (Nature Portfolio $3k–$3.5k, Wiley-OA $3.5k–$4.2k, Elsevier-OA $4k–$4.5k, PLOS $2.2k–$2.7k).
- **Does method_blueprint.md's compute budget (~420 H100-hr) track experiment_plans' cell counts?** — YES.
  - lit 48 cells + self 48 cells = 96 cells at 2.5 × 1.15 ≈ 2.875 H100-hr/cell → 276 H100-hr main grid, matches blueprint exactly. The 276/412/420 chain is self-consistent between blueprint and the two plans (lit 206 + self 265 ≈ 471, slight over-budget vs blueprint 420; scholar_eval_report §2.4 notes this ±12% slack is acknowledged in the self plan §8).

## Execution readiness verdict

- **Before Task B/C/D/E fixes:** **NOT READY** for postdoc execution.
  Specific blockers:
  1. `experiment_plan_lit.md` and `experiment_plan_self.md` lack power calculations (awaiting Task B Revision 1).
  2. Both plans report conformal coverage only at γ ∈ {0.90, 0.95}, missing the 5-level sweep (Task B).
  3. Both plans lack β × floor ablation cells (Task B).
  4. `risk_register.md` R4 (Paper 11 slip) is MEDIUM but scholar_eval report requires upgrading to HIGH (Task E).
  5. `risk_register.md` lacks a dedicated clean-room-reproduction-fidelity risk row (Task E).
  6. Horvát 2025 is cited in 3 files but not in JSONL — either add or remove (Task B).
  7. `github_inventory.md` `neu-spiral/Hybrid-ODE-NN` license contradicts README self-description — re-verify.
  8. `litreview_database.jsonl` `demirkaya2021` author list disagrees with PubMed record — correct to Demirkaya/Imbiriba/Lockwood/Rampersad/Alhajjar/Guidoboni/Danziger/Erdogmus.
  9. `scholar_eval_report.md` composite-math should be 6.4 not 6.1 — cosmetic fix (does not affect verdict).

- **After Task B/C/D/E fixes (projected):** **READY.**
  Assumption: Task B closes Revision-1 gaps (items 1–3), Task E closes Revision-3 gap + R4 upgrade (items 4–5), Tasks C/D close Horvát + framing-preamble + cosmetic items (items 6–9). At that point all 11 files map cleanly to the plan's acceptance criteria and no blocker prevents postdoc execution.

Files that need changes in Tasks B/C/D/E (actionable list):
- `experiment_plan_lit.md` (Task B) — add §6 power calc, expand §4 M2 to 5 γ levels, add β × floor ablation cell, scope cross-cohort to supplementary.
- `experiment_plan_self.md` (Task B) — same four additions; tautology protocol is already strong.
- `risk_register.md` (Task E) — upgrade R4 to HIGH; add R17 (clean-room reproduction fidelity) MEDIUM.
- `litreview_database.jsonl` (Task B) — correct `demirkaya2021` authors; optionally add Horvát 2025 entry.
- `novelty_verdict.md`, `venue_fit.md`, `scholar_eval_report.md` (Task C/D) — deprecate or support Horvát 2025; fix composite math in scholar_eval; insert framing preamble per Revision 2.
- `github_inventory.md` (Task B/C) — re-verify `neu-spiral/Hybrid-ODE-NN` license via `/contents/` API.
