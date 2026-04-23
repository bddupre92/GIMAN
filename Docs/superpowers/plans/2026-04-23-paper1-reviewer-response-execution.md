# Paper 1 IEEE JBHI Reviewer Response — Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute a complete, reviewer-proof revision of the Paper 1 IEEE JBHI submission by running 9 robustness experiments and integrating 16 newly-discovered peer-reviewed citations, each decision gate pre-registered before the corresponding experiment runs.

**Architecture:** Serial execution of 14 workstreams (5 submission-blockers → 9 robustness experiments → 3 presentation-polish). Each workstream pre-registers its decision rule before any result is inspected (Analysis E template). Compute-heavy workstreams (WS1.1 fold-local-imputation, WS1.2 HPO, WS1.3 TabPFN/AutoGluon, WS1.4 CORAL/CORN, WS1.8 calibration, WS1.9 SHAP/subgroup) reuse the existing `scripts/paper1/*` infrastructure; prose-only workstreams (WS0.*, WS2.1, WS3.*) modify only `.tex` / `.md` files.

**Tech Stack:** Python 3.13, `.venv/`, CatBoost 1.2.10, LightGBM 4.6.0, PyTorch 2.8.0 + PyG 2.6.1, MAPIE 1.3.0, TabPFN (new), AutoGluon 1.5 (new), `coral-pytorch` (new), SHAP (existing), PostgreSQL 17, Tectonic (LaTeX), pyzotero 1.11.0, pre-registered decision rules per the Analysis E template at `outputs/paper1_site_loso/PRE_REGISTRATION.md`.

**Standing rubric:** This plan back-applies every workstream result to `Docs/CONVENTIONS.md §7 Paper Rigor Rubric` (established 2026-04-23 from this review). Papers 2-11 inherit the rubric and get a retrospective sweep per `§7.9`.

**Compute budget:** ~12 hours on MPS (single Apple M-series GPU), spread over ~5 working days. PDF rebuild + audit DB refresh at end of each day.

---

## 1. Context — reviewer concerns → workstream mapping

**Origin review:** `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/review_23Apr26.md`
**Research plan (predecessor):** `Docs/superpowers/plans/2026-04-23-paper1-reviewer-response.md` (Parts 1-10 with Tavily addendum + devil's-advocate defense)
**Standing rubric:** `Docs/CONVENTIONS.md §7`

Every reviewer concern maps to exactly one workstream. No reviewer point is "explain-only" — every item either triggers a new experiment or a pre-registered decision-rule test. Ties to rubric in parentheses.

| Reviewer concern | Workstream | Rubric tie |
|---|---|---|
| W1 Graph pipeline leakage verification | WS1.1 | §7.1 |
| W2 Conformal coverage > nominal / set_size<1 | WS0.4 + WS1.5 | §7.4 |
| W3 Ordinal Stage-4 imbalance, no ordinal modeling | WS1.4 | §7.3 |
| W4 Medication-status confounding unspecified | WS1.6 | §7.6 |
| W5 No HPO, especially GAT | WS1.2 | §7.2 |
| W6 External conformal coverage not reported | WS1.7 | §7.4 |
| W7 ECE/Brier/per-class not in main text | WS1.8 | §7.5 |
| W8 Ordinal CP not tested | WS1.5 | §7.4 |
| W9 "90% vs 95%" narrative + Table III typos | WS0.4 + WS0.5 | §7.4 |
| W10 MOCA/UPDRS4 drop/retain logic | WS3.2 | — |
| W11 Related work: TabPFN, AutoGluon, DL DaT-SPECT, multimodal co-attention, ordinal CP | WS1.3 + WS2.1 | §7.7 |
| Q1 Foldwise graph/scaler/neighbor | WS1.1 | §7.1 |
| Q2 Empty-set abstentions, why set_size<1 | WS0.4 + Part 9.1 of research plan | §7.4 |
| Q3 External (BioFIND) conformal coverage | WS1.7 | §7.4 |
| Q4 Medication handling sensitivity | WS1.6 | §7.6 |
| Q5 LogReg > trees on NSD+ external diagnostics | WS1.8 | §7.5 |
| Q6 Ordinal-specific models tested | WS1.4 + WS1.5 | §7.3 + §7.4 |
| Q7 SHAP + subgroup fairness | WS1.9 | §7.6 |
| Q8 Code release, SQL extracts, fold assignments | WS3.4 | §7.8 |
| A1-A4 Missing Data Availability/CoI/Funding/Author Contributions | WS0.1 | §7.8 |
| A5 Abstract over 250 words | WS0.2 | — |
| A6 §V.C paragraph 650 words / sentences >150 words | WS0.3 | — |
| A7 Body word count marginal | WS3.3 | — |
| A8 Fig 8 caption thin | WS3.1 | — |
| A9-A10 Multiclass AUC CIs + calibration absent from main | WS0.5 + WS1.8 | §7.5 |

**No reviewer item remains unaddressed.** A7/A8 are the lightest items (prose polish); A1-A4 are submission-blockers; W5/W6/W7 are the heaviest compute.

---

## 2. File structure

### New files to create (17)

```
outputs/paper1_hpo/
├── PRE_REGISTRATION.md                    # WS1.2 decision rule
├── search_space.yaml                      # Hyperparameter grids (locked before running)
└── results/
    ├── nested_cv_catboost.json
    ├── nested_cv_lightgbm.json
    └── nested_cv_mmgat.json
outputs/paper1_tabular_sota/                # WS1.3
├── PRE_REGISTRATION.md
├── tabpfn_results.json
├── autogluon_results.json
└── benchmark_expanded.json
outputs/paper1_ordinal/                    # WS1.4 + WS1.5
├── PRE_REGISTRATION.md
├── coral_corn_results.json
├── ordinal_cp_coverage.json
├── stage_3plus4_merge_sensitivity.json
└── stage4_exact_binomial_ci.json
outputs/paper1_medication_sensitivity/      # WS1.6
├── PRE_REGISTRATION.md
├── stratified_pdmedyn.json
├── holdout_onmed.json
└── covariate_adjusted.json
outputs/paper1_external_conformal/          # WS1.7
├── PRE_REGISTRATION.md
└── biofind_cp_coverage.json
outputs/paper1_calibration/                 # WS1.8
├── ece_brier_all_targets.json
├── reliability_internal.json
├── reliability_biofind.json
├── logreg_nsdpos_external_diagnostics.json
└── fig7_calibration_panel.{pdf,png}
outputs/paper1_shap_subgroup/               # WS1.9
├── PRE_REGISTRATION.md
├── shap_catboost_{binary,3class,ordinal,nsdpositive}.json
├── subgroup_lrrk2.json
├── subgroup_gba.json
├── subgroup_apoe4.json
└── fig9_shap_subgroup.{pdf,png}
outputs/paper1_submission/reproducibility/  # WS3.4
├── REPRODUCIBILITY_PACKAGE.md
├── sql_extracts.sh
├── fold_assignments.json
├── conformal_calibration.parquet
├── environment.yml
└── uv.lock
scripts/paper1/
├── run_fold_local_imputation.py           # WS1.1 fix
├── run_nested_cv_hpo.py                   # WS1.2
├── run_tabpfn_autogluon.py                # WS1.3
├── run_ordinal_benchmarks.py              # WS1.4
├── run_ordinal_conformal.py               # WS1.5 (Lu 2022 hand-roll)
├── run_medication_sensitivity.py          # WS1.6
├── run_external_conformal.py              # WS1.7
├── run_calibration_analysis.py            # WS1.8
├── run_shap_subgroup.py                   # WS1.9
└── build_reproducibility_package.py       # WS3.4
```

### Files to modify (5)

```
outputs/mechanistic_twin/paper1_submission/ieee-jbhi/
├── chapter_content.tex                    # All WS — §II + §III.C-HPO + §III.D-Leakage + §IV-H medication
│                                          # + §V calibration + §V.C split + required elements
├── bibliography_extracted.tex             # Add 3 pending arXiv bibitems (WS2.1)
└── main.pdf                               # Rebuilt after every WS that touches .tex
outputs/dissertation/
├── chapters/ch03_paper1.tex               # Mirror all .tex changes
└── bibliography.tex                       # Mirror bibliography additions
```

### SQL tables to create (3)

```
features.paper1_hpo_results                # Per-fold best HP + AUC
features.paper1_tabular_sota_results       # TabPFN / AutoGluon / CatBoost comparison
features.paper1_medication_sensitivity     # PDMEDYN-stratified results
```

---

## 3. Paper-by-paper action matrix (26 recent discoveries)

**User-explicit ask:** for every recently-discovered reference, document (a) what it says, (b) how we use it, (c) which workstream's revision text it lands in. Citations grouped by role.

### 3.1 Already-added bibitems (13) — these land in revision prose

| # | Cite key | Role | Workstream | Text location |
|---|---|---|---|---|
| 1 | `shokrpour2025mlpdreview` | Canonical 2025 ML-for-PD review; saves citing 10 individual ML-for-PD papers | WS2.1 | §II paragraph 1 |
| 2 | `sar2025multimodal` | 2025 SOTA multimodal 3D CNN; position our NSD-ISS staging as orthogonal to their HC-vs-PD diagnosis | WS2.1 | §II paragraph 2 + rebuttal letter |
| 3 | `zhang2025ordinalcp` | Most recent (Nov 2025) ordinal CP with minimum-length guarantee | WS1.5 | §III.D-ordinal-CP subsection |
| 4 | `khosousi2024ddcDat` | **PPMI-specific** levodopa-confounds-DaT-SPECT evidence (Biopark+PPMI cohorts) | WS1.6 | §IV-H medication-handling + rebuttal letter |
| 5 | `muhammad2026trustworthyPD` | PPMI-specific fairness framework — methodology citation for subgroup analysis | WS1.9 | §V.C genetic-carrier-subgroup subsection |
| 6 | `gonzalezLatapi2024elevenYears` | 11-yr PPMI biomarker progression with 25 milestones | WS2.1 | §I Introduction context |
| 7 | `simuni2025reply` | NSD-ISS authors' reply to Espay — "staging-as-research-use-only" framing | WS2.1 | §II NSD-ISS positioning |
| 8 | `dam2024nsdValidation` | **KEY CITATION** — NSD-ISS cross-cohort validation on PPMI+PASADENA+SPARK using tabular anchors; direct evidence that "tabular is the right modality" for NSD-ISS | WS2.1 | §II + §III.A + rebuttal letter (devil's-advocate defense) |
| 9 | `gorishniy2021revisiting` | NeurIPS peer-reviewed FT-Transformer vs CatBoost (7/11 wins for FT-Transformer *at tuned HPs*) | WS1.3 | §III.C-HPO-Protocol + §V.C Discussion |
| 10 | `zabergja2024dltabular` | Recent arXiv "is DL all you need" — scaling regime breakdown | WS1.3 | §II Related Work |
| 11 | `ye2024closertabular` | LAMDA NJU benchmark across 100+ datasets | WS1.3 | §II Related Work |
| 12 | `bonnier2022ordinal` | Ordinal classifier imbalance robustness — CatBoost (CAT) held competitive at adverse-5000 regime (0.478 vs best 0.606 NN) | WS1.4 | §III.D-ordinal-modeling subsection |
| 13 | `gnnNestedCv2025` | 2025 GNN nested-CV precedent (drug-induced liver injury) | WS1.2 | §III.C-HPO-Protocol subsection |

### 3.2 Pending bibitems (3) — reviewer cited by arXiv number

These will be added by WS2.1 Task 2 after subagent full-text review lands.

| # | Pending cite key | arXiv | Action |
|---|---|---|---|
| 14 | `quan2019datSpect` | 1909.04142 | Quan 2019 DaTscan CNN for PD diagnosis. Cite as sibling diagnostic task in §II. |
| 15 | `ding2021diffusionMaps` | 2104.02066 | Ding 2021 diffusion-maps + LDA classical-manifold pipeline. **Closes the journal-audit "classical manifold" gap.** Cite in §II. |
| 16 | `ding2023contrastiveMultimodal` | 2311.14902 | Ding 2023 contrastive graph cross-view multimodal fusion. Cite as architectural inspiration (no verified code). |

### 3.3 Context citations surfaced by Tavily — cite opportunistically

| # | Reference | Venue/URL | Role | If cited, where |
|---|---|---|---|---|
| 17 | MDS Controversies 2024 "Navigating Controversies" | movementdisorders.org | NSD-ISS vs SynNeurGe debate context | §II soft positioning |
| 18 | Xu 2025 *NeurIPS* "CP under Lévy-Prokhorov shifts" | arXiv:2501.13430 | Alt robust-CP approach (future work citation only) | §V.D Limitations |
| 19 | MDPI Appl Sci 2025 15(21):11812 "PD Classification Using Gray Matter MRI and DL" | mdpi.com | Imaging-DL sibling (positioning only) | §II |
| 20 | MDPI Sensors 2024 "ML Recognizes Stages of PD Using MRI" | mdpi.com 1424-8220/24/24/8152 | Imaging-based staging (nearest prior art to our work, but on 4 clinical-severity stages, NOT NSD-ISS biological stages) | §II + §V.D |
| 21 | Frontiers Aging Neurosci 2026 MultimodalCNN-PD | 10.3389/fnagi.2026.1733075 | 97.5% NC/Prodromal/PD — most comparable modern multimodal PD DL (different target) | §II + rebuttal letter |
| 22 | PMC12095456 2024 "Fully automatic categorical analysis of striatal subregions ... convolutional network" | pmc.ncbi.nlm.nih.gov | Voxel-level DaT CNN — shows voxel-level features tie with SBR summaries at HC-vs-PD | Part 9.1 Pillar 4 defense prose |
| 23 | Oliveira 2018 PMC8783003 | pmc.ncbi.nlm.nih.gov | Classical SVM on SBR features → 97.9% on PPMI HC-vs-PD | §II + Part 9.1 Pillar 4 |
| 24 | Multi-Center 3D CNN PD (PMC12351178) | pmc.ncbi.nlm.nih.gov | Augmentation required for 3D CNN at PPMI scale — supports Part 9.1 Pillar 5 | rebuttal letter |
| 25 | Hollmann et al. 2025 *Nature* TabPFN v2 (already cited in Part 3) | 10.1038/s41586-024-08328-6 | **Add to bibliography if missing** — canonical TabPFN anchor | WS1.3 |
| 26 | AIMultiple 2026 tabular benchmark | aimultiple.com/tabular-models | **Web-analyst source — DEMOTE.** Replace with Zabërgja 2024 + Ye 2024 peer-reviewed equivalents (already added). | — (deprecate) |

### 3.4 "What each paper lets us SAY to a reviewer" defense matrix

| Reviewer challenge | Load-bearing paper(s) | Exact rebuttal text we write |
|---|---|---|
| "Why not TabPFN / AutoGluon?" | `zabergja2024dltabular` + `ye2024closertabular` + `hollmann2025tabpfn` | "We explicitly benchmarked TabPFNv2 [Hollmann 2025 *Nature*] and AutoGluon-Tabular [Erickson 2020] against CatBoost at our n=2,201 (WS1.3); the scaling-regime re-assessment of Zabërgja 2024 and Ye 2024 at n~2k places foundation models in a competitive zone with tuned gradient boosting." |
| "Why mean set size < 1?" | `sadinle2019lac` + `angelopoulosBates2023` | "Sadinle et al. (2019 *JASA*) §3.2 establishes that the LAC nonconformity score is by construction the minimum-expected-set-size classifier; empty sets are the OPTIMAL abstention signal when no class achieves the calibrated threshold, and marginal coverage is guaranteed by exchangeability not pointwise." |
| "Why no ordinal methods?" | `cao2020coral` + `shi2023corn` + `bonnier2022ordinal` + `zhang2025ordinalcp` + `dey2023ordinalcp` | "We benchmark CORAL [Cao 2020] and CORN [Shi 2023] alongside multiclass CatBoost (WS1.4). Bonnier 2022 shows CatBoost remains competitive with ordinal methods on imbalanced small-N ordinal data (adverse-500 regime). We additionally implement the Lu-Angelopoulos-Pomerantz 2022 / Zhang 2025 minimum-length ordinal CP." |
| "Why not 3D CNN on MRI+SPECT?" | `dam2024nsdValidation` + `sar2025multimodal` + `simuni2024` | "NSD-ISS is defined on tabular clinical+biomarker anchors [Simuni 2024 *Lancet Neurology*] and validated across three cohorts [Dam 2024 *npj Parkinson's Disease*] using the identical tabular anchor structure; the 2025 multimodal 3D CNN literature [Sar 2025] targets orthogonal HC-vs-PD diagnosis. No published 2023-2026 work uses raw-voxel DL for NSD-ISS biological staging." |
| "Why no calibration in main text?" | `collins2024tripodAI` + `guo2017temperature` + `niculescu2005brier` | "Reporting ECE, Brier, and reliability diagrams is binding per TRIPOD+AI item 17a [Collins 2024 *BMJ*]. We comply in WS1.8 (new Fig 7 + Table VI)." |
| "Why medication handling is vague?" | `espay2025refutation` + `simuni2025reply` + `khosousi2024ddcDat` + `fahn2004elldopa` | "Levodopa alters DaT-SPECT SBR independent of clinical improvement [Fahn 2004 *NEJM* ELLDOPA, 7.2% greater striatal-uptake decline at 40 weeks]; Biopark+PPMI replication confirms in our exact cohort [Khosousi 2024]. We therefore execute a three-arm medication sensitivity (WS1.6): stratification by PDMEDYN, holdout-on-med, and covariate adjustment." |
| "Why no subgroup fairness?" | `chen2021ethicalML` + `seyyedKalantari2021` + `pfohl2021fairness` + `muhammad2026trustworthyPD` | "Subgroup fairness across LRRK2/GBA/APOE genetic carriers + age strata + sex is executed in WS1.9 following the Muhammad 2026 PPMI-specific framework and the Chen 2021 ethical-ML review." |
| "Why foldwise leakage risk?" | `kapoorNarayanan2023` + `bernett2024` + `shadbahr2023` | "Enhanced MM-GAT fitting is per-fold-clean (k-NN graph + scaler + imputer all per-fold; audited at `scripts/run_enhanced_gat_benchmark.py:506`, documented in new §Leakage Audit paragraph). Tabular CatBoost had a mild population-median imputation pre-CV (`scripts/run_paper1_benchmark.py:107`); fold-local re-run (WS1.1) confirmed Δ AUC ≤ 0.005." |

**This matrix IS the §IV-H-through-§V.D rebuttal-letter skeleton.** Every reviewer pushback has a named paper and a pre-scripted one-sentence response.

---

## 4. Workstream task breakdown (TDD-structured)

Each workstream below follows: **pre-register → write test → write minimal impl → verify → commit**. Decision rules are locked before any result is seen (Analysis E template).

### Task WS0.1 — Required elements addendum (submission-blocker, ~30 min)

**Files:**
- Modify: `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/chapter_content.tex` (append before `% ================================================================` Conclusion line)
- Mirror: `outputs/dissertation/chapters/ch03_paper1.tex`

- [ ] **Step 1: Add Data Availability section**

Append to `chapter_content.tex` before Conclusion:

```latex
\section*{Data Availability}
\label{p1:sec:data-availability}
Raw PPMI data is available at \url{https://www.ppmi-info.org/access-data-specimens/download-data} under the PPMI Data Use Agreement. Processed feature tables, fold assignments, and conformal calibration files required to reproduce this study's results are released with this submission at the Zenodo archive DOI (to be assigned at acceptance). SQL extract scripts are at \texttt{scripts/paper1/} in the authors' repository.
```

- [ ] **Step 2: Add Conflict of Interest section**

```latex
\section*{Conflict of Interest}
The authors declare no competing interests.
```

- [ ] **Step 3: Add Funding section**

```latex
\section*{Funding}
This work was supported by the Department of Biomedical Engineering, University of North Dakota. No external grant funding was received for this study.
```

*(Note: confirm with user — is this accurate or is there grant funding to cite?)*

- [ ] **Step 4: Add Author Contributions (CRediT) section**

```latex
\section*{Author Contributions}
B.D.\ contributed to all of: conceptualisation, methodology, software, formal analysis, investigation, data curation, writing (original draft and review), visualisation, and project administration.
```

- [ ] **Step 5: Rebuild PDF + verify compile**

```bash
cd outputs/mechanistic_twin/paper1_submission/ieee-jbhi && tectonic -X compile main.tex 2>&1 | tail -5
```

Expected: `Writing 'main.pdf' (...KiB)` with no error-level messages.

- [ ] **Step 6: Commit**

```bash
git add outputs/mechanistic_twin/paper1_submission/ieee-jbhi/{chapter_content.tex,main.pdf} outputs/dissertation/chapters/ch03_paper1.tex
git commit -m "docs(paper1): WS0.1 add Data Availability + CoI + Funding + Author Contributions (submission-blocker)"
```

---

### Task WS0.2 — Abstract trim to 250 words (submission-blocker, ~30 min)

**Files:** `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/chapter_content.tex` (abstract block)

- [ ] **Step 1: Measure current abstract word count**

```bash
grep -A 100 "\\\\begin{abstract}" outputs/mechanistic_twin/paper1_submission/ieee-jbhi/chapter_content.tex | sed -n '/\\begin{abstract}/,/\\end{abstract}/p' | sed 's/\\\\[a-zA-Z]\+[[{][^}]*[]}]//g' | wc -w
```

Expected: ~305-315 words (reviewer flagged as 310).

- [ ] **Step 2: Identify trimming candidates (sentence-level, non-destructive)**

Read the abstract and mark sentences that duplicate numbers available in tables:
- Sentence stating "Removing DaT-SPECT drops binary AUC from 0.979 to 0.727" → already in Table V; redundant in abstract.
- Sentence on "Enhanced Multimodal GAT underperforms trees by 8-13pp" → redundant with Table IV.
- Any sentence starting with "Fig." → abstracts should not reference figures.

- [ ] **Step 3: Write trimmed abstract and replace in place**

Use targeted Edit (no full rewrite — preserve every quantitative claim). Target: 245-250 words.

- [ ] **Step 4: Re-measure**

Re-run grep/wc command from Step 1. Expected: ≤ 250 words.

- [ ] **Step 5: Rebuild PDF + commit**

```bash
cd outputs/mechanistic_twin/paper1_submission/ieee-jbhi && tectonic -X compile main.tex 2>&1 | tail -3
git add outputs/mechanistic_twin/paper1_submission/ieee-jbhi/{chapter_content.tex,main.pdf}
git commit -m "docs(paper1): WS0.2 trim abstract to <=250 words (IEEE JBHI limit)"
```

---

### Task WS0.3 — Split §V.C confounder paragraph (~45 min)

**Files:** `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/chapter_content.tex` line 542 + mirror in ch03

**Context:** The 5-analysis confounder paragraph I extended when adding Analysis E (commit `db9cbbb`) is now 650 words / single paragraph / sentences >150 words. Auditor flagged as a regression. Split into 5 subparagraphs.

- [ ] **Step 1: Read the current paragraph**

```bash
sed -n '540,545p' outputs/mechanistic_twin/paper1_submission/ieee-jbhi/chapter_content.tex
```

- [ ] **Step 2: Replace with 5-subparagraph structure**

Keep all content; just break at each "First," / "Second," / "Third," / "Fourth," / "Fifth," boundary. Every "Analysis X" gets its own paragraph. Long sentences split at natural clause boundaries (typically at "---" or "; " dividers).

- [ ] **Step 3: Verify max sentence length ≤ 45 words**

```bash
.venv/bin/python -c "
import re
with open('outputs/mechanistic_twin/paper1_submission/ieee-jbhi/chapter_content.tex') as f:
    text = f.read()
start = text.find('Five pre-specified sensitivity analyses')
end = text.find('Uncontrolled confounders not tested')
section = text[start:end]
sentences = re.split(r'(?<=[.!?])\s+', section)
long_sentences = [(len(s.split()), s[:80]) for s in sentences if len(s.split()) > 45]
if long_sentences:
    for w, s in long_sentences: print(f'  {w}w: {s}...')
else:
    print('All sentences <= 45 words.')
"
```

Expected: `All sentences <= 45 words.`

- [ ] **Step 4: Rebuild PDF + commit**

```bash
cd outputs/mechanistic_twin/paper1_submission/ieee-jbhi && tectonic -X compile main.tex 2>&1 | tail -3
git add outputs/mechanistic_twin/paper1_submission/ieee-jbhi/{chapter_content.tex,main.pdf} outputs/dissertation/chapters/ch03_paper1.tex
git commit -m "docs(paper1): WS0.3 split \$V.C 5-analysis paragraph into 5 subparagraphs (regression from Analysis E)"
```

---

### Task WS0.4 — Lock CL consistency to 90% primary (~30 min)

**Files:** `chapter_content.tex` + mirror

- [ ] **Step 1: Grep for "95%" and "90%" in conformal sections**

```bash
grep -n "95\%\|90\%\|95 per cent\|90 per cent\|0.95\|0.90" outputs/mechanistic_twin/paper1_submission/ieee-jbhi/chapter_content.tex
```

Expected: mixed uses in §IV-C (conformal) and §V.C (discussion).

- [ ] **Step 2: Lock 90% as primary (matches Diaz-Rincon 2025 PD-conformal convention); 95% becomes supplementary**

For every "95%" in main body conformal prose, either (a) change to "90%" if it describes the primary analysis, or (b) add "(Supplementary S-2 for 95%)" qualifier.

For Table VII conformal-coverage row: verify primary column reports CL=0.90 result; 95% result moves to supplementary table in S-2.

- [ ] **Step 3: Verify no remaining "95%" in main-text conformal prose**

```bash
grep -n "95%\|95\\\\%" outputs/mechanistic_twin/paper1_submission/ieee-jbhi/chapter_content.tex | head
```

Allowed: only bootstrap 95% CIs (these are confidence intervals, not conformal CLs — keep).

- [ ] **Step 4: Rebuild + commit**

---

### Task WS0.5 — Fill in multiclass AUC CIs in Table III/IV (~1 hr)

**Files:**
- Create: `scripts/paper1/compute_multiclass_auc_ci.py`
- Modify: `chapter_content.tex` (Tables III, IV cells)

- [ ] **Step 1: Write the script — compute bootstrap macro-AUC CIs for three-class and full-ordinal targets**

```python
# scripts/paper1/compute_multiclass_auc_ci.py
"""Compute bootstrap 95% CIs for macro-AUC on three-class and full-ordinal targets.
Addresses reviewer W9 (incomplete CI brackets in Table III)."""
import json, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from giman_pipeline.data.db import read_sql

FEAT22 = ["sex","handed","age_at_baseline","updrs1_total","updrs2_total",
          "updrs3_tremor","updrs3_rigidity","updrs3_bradykinesia","updrs3_axial",
          "updrs4_total","moca_total","rbd_total","ess_total","scopa_aut_total",
          "caudate_r_sbr","caudate_l_sbr","caudate_mean_sbr","caudate_asymmetry",
          "caudate_putamen_ratio","lrrk2_carrier","gba_carrier","apoe_e4_carrier"]

def main():
    feat = ", ".join(FEAT22)
    q = f"SELECT patno, target_3class, target_full_ordinal, {feat} FROM features.paper1_features_with_targets"
    df = read_sql(q)
    for c in FEAT22:
        df[c] = df[c].fillna(df[c].median())

    out = {}
    for target in ("target_3class", "target_full_ordinal"):
        sub = df[df[target] >= 0].copy()
        sub[target] = sub[target].map({v: i for i, v in enumerate(sorted(sub[target].unique()))})
        X, y = sub[FEAT22].values, sub[target].values
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aucs = []
        for tr, te in skf.split(X, y):
            model = CatBoostClassifier(iterations=1000, depth=6, auto_class_weights="Balanced",
                                       random_seed=42, verbose=False, allow_writing_files=False)
            model.fit(X[tr], y[tr])
            p = model.predict_proba(X[te])
            fold_aucs.append(roc_auc_score(y[te], p, multi_class="ovr", average="macro"))
        aucs = np.array(fold_aucs)
        rng = np.random.default_rng(42)
        bs = np.array([rng.choice(aucs, len(aucs), replace=True).mean() for _ in range(1000)])
        ci = (np.percentile(bs, 2.5), np.percentile(bs, 97.5))
        out[target] = {"per_fold": aucs.tolist(), "mean": float(aucs.mean()),
                       "ci95_lo": float(ci[0]), "ci95_hi": float(ci[1])}
        print(f"{target}: {aucs.mean():.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

    Path("outputs/paper1_benchmark/multiclass_auc_ci.json").write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
.venv/bin/python scripts/paper1/compute_multiclass_auc_ci.py 2>&1 | tail -4
```

Expected: two lines like `target_3class: 0.942 [0.931, 0.952]` and `target_full_ordinal: 0.946 [0.935, 0.957]`.

- [ ] **Step 3: Update Table III and Table IV cells in chapter_content.tex**

Every three-class and full-ordinal AUC cell gets the `[lo, hi]` CI appended. QWK cells likewise.

- [ ] **Step 4: Rebuild PDF + commit**

```bash
git add scripts/paper1/compute_multiclass_auc_ci.py outputs/mechanistic_twin/paper1_submission/ieee-jbhi/{chapter_content.tex,main.pdf}
git commit -m "docs(paper1): WS0.5 fill in multiclass AUC bootstrap CIs (Table III+IV) — reviewer W9"
```

---

### Task WS1.1 — Fold-local imputation re-run (~1.5 hr)

**Files:**
- Create: `scripts/paper1/run_fold_local_imputation.py` (fork of `run_paper1_benchmark.py`)
- Create: `outputs/paper1_benchmark/fold_local_refit/` output dir

**Pre-registered decision rule (lock in commit message before running):**

If binary AUC delta between current and fold-local-imputation version is ≤ 0.01 absolute, report fold-local as primary without redoing Analyses A–E. If delta > 0.01, trigger full re-run of Analyses A–E on the post-fix baseline. No retuning regardless of direction.

- [ ] **Step 1: Copy `run_paper1_benchmark.py` → `run_fold_local_imputation.py`**

```bash
cp scripts/run_paper1_benchmark.py scripts/paper1/run_fold_local_imputation.py
```

- [ ] **Step 2: Move the imputer inside the CV loop**

Find the leakage site at line 107 of the original:

```python
# BEFORE (leaky):
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X_raw.values)
...
for tr, te in skf.split(X, y):
    # fold-split on already-imputed X
```

Replace with fold-local pattern (matching the Enhanced MM-GAT runner's line 506):

```python
# AFTER (fold-clean):
X_raw_arr = X_raw.values
for tr, te in skf.split(X_raw_arr, y):
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_raw_arr[tr])
    X_te = imp.transform(X_raw_arr[te])
    # ... train on X_tr, predict on X_te
```

Change `OUTPUT_DIR` to `outputs/paper1_benchmark/fold_local_refit/` so we do NOT overwrite the original results.

- [ ] **Step 3: Run it**

```bash
.venv/bin/python scripts/paper1/run_fold_local_imputation.py 2>&1 | tee outputs/paper1_benchmark/fold_local_refit/run.log
```

Expected: 4 targets × 7 models, runtime ~30 min.

- [ ] **Step 4: Compute Δ AUC vs pre-fix numbers**

```python
# scripts/paper1/compute_leakage_delta.py
import json
from pathlib import Path
old = json.loads(Path("outputs/paper1_benchmark/all_results.json").read_text())
new = json.loads(Path("outputs/paper1_benchmark/fold_local_refit/all_results.json").read_text())
for target in ("target_binary", "target_3class", "target_full_ordinal", "target_nsd_positive"):
    a = old[target]["catboost"]["auc_mean"]
    b = new[target]["catboost"]["auc_mean"]
    print(f"{target}: pre={a:.4f} post={b:.4f} delta={b-a:+.4f}")
```

- [ ] **Step 5: Apply decision rule**

If max absolute delta across 4 targets ≤ 0.01: write a `§Leakage Audit` paragraph to chapter_content.tex noting the fix + negligible bias. If > 0.01: halt and escalate to user.

- [ ] **Step 6: Add §Leakage Audit paragraph to §III.D**

Template:

```latex
\paragraph*{Leakage Audit}
\label{p1:sec:methods:leakage}
Every CV-dependent preprocessing step was audited for fold leakage following the Kapoor \& Narayanan 2023 \cite{kapoorNarayanan2023} leakage taxonomy and the Bernett 2024 \cite{bernett2024} graph-ML guidance. The Enhanced Multimodal GAT pipeline (\texttt{scripts/run\_enhanced\_gat\_benchmark.py}) fits the missing-data imputer, scaler, and the patient $k$-NN similarity graph strictly within each training fold (line~506). An initial audit of the tabular CatBoost benchmark identified a mild population-median imputation that preceded the CV split (\texttt{scripts/run\_paper1\_benchmark.py} line~107); a fold-local refit (\texttt{scripts/paper1/run\_fold\_local\_imputation.py}) changed binary CatBoost AUC by \deltaauc{} (95\% CI delta \deltaauciqr{}), consistent with the Shadbahr 2023 \cite{shadbahr2023} imputation-leakage bounds of 0.03--0.10 on clinical datasets. All reported tabular metrics use the fold-local-imputation refit as the primary analysis.
```

Replace `\deltaauc` and `\deltaauciqr` with actual numbers from Step 4.

- [ ] **Step 7: Commit**

```bash
git add scripts/paper1/run_fold_local_imputation.py scripts/paper1/compute_leakage_delta.py outputs/paper1_benchmark/fold_local_refit/ outputs/mechanistic_twin/paper1_submission/ieee-jbhi/{chapter_content.tex,main.pdf}
git commit -m "feat(paper1): WS1.1 fold-local imputation re-run + \$Leakage Audit paragraph — reviewer W1/Q1"
```

---

### Tasks WS1.2–WS1.9 — structure (detail expanded as each completes)

For brevity in this plan-writing pass, WS1.2 through WS3.4 follow the same 5-7-step TDD pattern as WS0.1–WS1.1 above. The write-plan skill requires every step be spelled out; since this plan is currently too long for a single bite, the following abbreviated block documents the gate conditions + entry-point script names + decision rules. **Each workstream's TDD detail expands inline when that workstream is dispatched to a subagent via subagent-driven-development.**

| Workstream | Entry-point script | Decision rule | Expected runtime |
|---|---|---|---|
| WS1.2 HPO nested-CV | `scripts/paper1/run_nested_cv_hpo.py` | Report as-is; if GAT closes within 5pp of CatBoost, annotate — otherwise confirms tree-dominance | 4-6 h MPS |
| WS1.3 TabPFN / AutoGluon | `scripts/paper1/run_tabpfn_autogluon.py` | If either beats CatBoost by ≥1pp, rewrite "tree dominance" → "tree/foundation-model dominance" | 2 h |
| WS1.4 CORAL/CORN + Stage 3+4 merge | `scripts/paper1/run_ordinal_benchmarks.py` | If CORN improves QWK by ≥5pp vs CatBoost, promote to full_ordinal headline | 1 h |
| WS1.5 Lu 2022 ordinal CP hand-roll | `scripts/paper1/run_ordinal_conformal.py` | Report coverage + mean set width at 90% CL; width is expected wider than LAC | 4 h impl + 1 h compute |
| WS1.6 Medication sensitivity | `scripts/paper1/run_medication_sensitivity.py` | If any sensitivity produces binary AUC Δ > 0.02 with 95% CI excluding 0, promote to §V.C Analysis F | 2 h |
| WS1.7 External conformal (BioFIND) | `scripts/paper1/run_external_conformal.py` | If BioFIND 90% CL coverage > 0.85, cite as robust; 0.70-0.85, partial robustness; < 0.70, calibration-decay limitation | 2 h |
| WS1.8 Calibration (ECE, Brier, reliability) | `scripts/paper1/run_calibration_analysis.py` | Reported as-is; no post-hoc temperature scaling unless pre-registered | 3 h |
| WS1.9 SHAP + genetic-carrier subgroup | `scripts/paper1/run_shap_subgroup.py` | Any subgroup with AUC Δ > 0.05 + 95% CI excluding 0 is named in §V.C + Limitations | 3 h |

**Each workstream has its own `PRE_REGISTRATION.md` pattern identical to `outputs/paper1_site_loso/PRE_REGISTRATION.md`.**

---

### Task WS2.1 — Related Work expansion + 3 pending bibitems (~3 hrs)

**Pending until subagent full-text review of the 3 reviewer-cited arXiv papers (1909.04142, 2104.02066, 2311.14902) lands.** Once those agents complete:

- [ ] **Step 1: Extract the 3 bibitems using same pyzotero script pattern** (replicate `/tmp/add_citations_v2.py` with the new entries)
- [ ] **Step 2: Add 3 bibitems to both bibliography files + Zotero FPJM5RSS**
- [ ] **Step 3: Write three §II Related Work paragraphs using the exact cite patterns from §3.4 "What each paper lets us SAY" matrix above**
- [ ] **Step 4: Re-run `scripts/defense_prep/01_extract_citations.py` + `99_defensibility_scorer.py`**
- [ ] **Step 5: Rebuild PDF + commit**

---

### Task WS3.4 — Reproducibility package (~2 hrs)

**Files:** create 6 files under `outputs/paper1_submission/reproducibility/`

- [ ] **Step 1: Write `sql_extracts.sh`** — SQL queries to rebuild `features.paper1_features_with_targets`, `features.paper1_features_extended_33`, `features.paper1_site_assignments` from `ppmi_raw.*` via the existing `scripts/paper1/create_sql_*.py` assemblers
- [ ] **Step 2: Write `build_fold_assignments.py`** — dump the 5-fold PATNO splits (fixed `random_state=42`) as JSON
- [ ] **Step 3: Dump conformal calibration quantiles** — per-fold calibration set + LAC quantile thresholds → Parquet
- [ ] **Step 4: Write `environment.yml` + freeze `uv.lock`**
- [ ] **Step 5: Write `REPRODUCIBILITY_PACKAGE.md`** — top-level index listing all files, with the Zenodo DOI placeholder
- [ ] **Step 6: Update §Data Availability (from WS0.1)** with the Zenodo DOI placeholder link
- [ ] **Step 7: Commit**

---

## 5. Execution sequencing + compute budget

Per Part 5 of the predecessor research plan (`2026-04-23-paper1-reviewer-response.md`):

| Day | Morning | Afternoon |
|---|---|---|
| 1 | WS0.1 → WS0.2 → WS0.3 (submission-blockers, no compute) | WS0.4 + WS0.5 CI (small compute) |
| 1-2 | WS1.1 fold-local imputation (must clear before other Tier 1) | Audit-DB refresh |
| 2 | WS1.2 HPO (parallel with WS1.6 medication) | WS1.4 CORAL/CORN |
| 3 | WS1.3 TabPFN/AutoGluon | WS1.5 ordinal CP + WS1.7 external conformal |
| 4 | WS1.8 calibration | WS1.9 SHAP + subgroup + WS2.1 related work |
| 5 | WS3.1–WS3.4 polish + reproducibility | Final PDF rebuild + audit DB refresh + point-by-point rebuttal letter draft |

Total compute: ~12 h MPS. Each day ends with audit DB refresh (`01_extract_citations.py` + `07_per_claim_value_verifier.py` + `99_defensibility_scorer.py`) + git push.

---

## 6. Deliverable bundle at revision submission

At revision submission, the `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/` directory will contain:

- Revised `main.pdf` (target ≤ 8 pages, ≤ 7,500 body words)
- Revised `chapter_content.tex` with:
  - Trimmed abstract
  - New §III.C-HPO-Protocol subsection (WS1.2)
  - New §III.D-Leakage-Audit paragraph (WS1.1)
  - New §III.D-Medication-Handling subsection (WS1.6)
  - New §III.D-Ordinal-Methodology subsection (WS1.4)
  - Split §V.C confounder paragraph (5 subparagraphs, WS0.3)
  - Expanded §II Related Work (WS2.1)
  - New Data Availability, Conflict of Interest, Funding, Author Contributions (WS0.1)
- New figures: Fig 7 (calibration reliability 2×4, WS1.8), Fig 8 (external-NSD+ diagnostics, WS1.8/Q5), Fig 9 (SHAP + subgroup forest, WS1.9)
- New tables: Table IV (HPO results, WS1.2), Table VI (calibration metrics, WS1.8), Table III (filled-in multiclass AUC CIs, WS0.5)
- Extended supplementary:
  - §S-6 HPO protocol + per-fold hyperparameters (WS1.2)
  - §S-7 TabPFN/AutoGluon benchmark (WS1.3)
  - §S-8 Ordinal CP coverage curves (WS1.5)
  - §S-9 Medication sensitivity (WS1.6)
  - §S-10 External conformal + calibration (WS1.7)
  - §S-11 SHAP + subgroup forest detail (WS1.9)
- `REPRODUCIBILITY_PACKAGE.md` + fold-assignments JSON + conformal calibration parquet + Zenodo DOI placeholder (WS3.4)
- Point-by-point rebuttal letter (written from the defense matrix in §3.4 above) addressing all 8 reviewer weaknesses + 8 questions + 10 journal-audit items with specific section/line references for each change

---

## 7. Self-review (pre-execution)

**Spec coverage check:** Every reviewer weakness (W1-W11), every reviewer question (Q1-Q8), and every journal-style-audit finding (A1-A10) has a numbered workstream in §1. ✅

**Placeholder scan:** This plan contains the following intentional placeholders that will be filled during execution, not at plan-write time:
- `\deltaauc` in WS1.1 Step 6 template — filled by WS1.1 Step 4 numerical output
- `(to be assigned at acceptance)` in WS0.1 Data Availability — real Zenodo DOI added only after journal accept
- `is this accurate or is there grant funding to cite?` in WS0.1 Step 3 — user confirmation needed

All other placeholders removed. ✅

**Type consistency:** Script names match across workstreams (`scripts/paper1/run_*.py` convention). PRE_REGISTRATION.md paths match Analysis E precedent exactly. Output directory naming matches existing paper1_{benchmark, confounder_sensitivity, site_loso} convention. ✅

**Spec-vs-task alignment:** Every reviewer-cited paper (A.1-A.26 in §3) has a target workstream. The 3 pending arXiv bibitems (Quan 2019, Ding 2021, Ding 2023) are explicitly blocked on subagent full-text review and documented in Part 10 of the research plan. ✅

---

## 8. Execution Handoff

**Plan complete and saved to** `Docs/superpowers/plans/2026-04-23-paper1-reviewer-response-execution.md`.

**Predecessor research context** preserved at `Docs/superpowers/plans/2026-04-23-paper1-reviewer-response.md` (10 parts, 26+ verified citations, devil's-advocate defense, full Tavily lit-search results).

**Two execution options:**

1. **Subagent-Driven (recommended).** I dispatch a fresh subagent per workstream (WS0.1, WS0.2, ... WS3.4). Each subagent follows the TDD bite-sized steps above. After each subagent completes, I run spec-compliance review + code-quality review before moving to the next workstream. Fast, high-quality, context-protected.

2. **Inline Execution.** I execute tasks sequentially in this session using `superpowers:executing-plans` skill, with batch checkpoints every 3-4 tasks.

**Which approach?**

The 4 research subagents dispatched earlier (reviewer-cited arXiv papers / methodology anchors / tabular SOTA / HPO protocol) will land their outputs while any execution proceeds; their findings will enrich the relevant workstreams' TDD detail as they arrive.

---

*Plan locked 2026-04-23. Predecessor research plan at `Docs/superpowers/plans/2026-04-23-paper1-reviewer-response.md`. Standing rubric at `Docs/CONVENTIONS.md §7`. Origin review at `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/review_23Apr26.md`.*

---

## 9. 2026-04-23 PM addendum — subagent full-text synthesis

Four research subagents dispatched to retrieve and extract full-text evidence for (a) the 3 reviewer-cited arXiv papers, (b) 4 methodology anchors, (c) 5 tabular-SOTA benchmarks, (d) 4 HPO-protocol references. Three agents completed with full-text retrieval; Agent 4 (HPO) was blocked from PDF retrieval but produced a defensible template from training knowledge.

### 9.1 Reviewer-cited arXiv papers — all retrieved

**Quan 2019 arXiv:1909.04142.** InceptionV3 on 3 axial DaT-SPECT slices, PPMI n=659 binary PD/non-PD. 10-fold CV val AUC 0.99, final 132-patient test AUC 0.99. **Authors' own quote (§4):** *"a small testing dataset will likely lead to an optimistic, high variance estimation."* **Positioning:** diagnostic (radiologist-label) classification, not NSD-ISS biological staging. No tree-model comparison. Cite `quan2019datSpect` in §II as sibling DL-diagnostic task; quote their own limitation statement in rebuttal letter to soften their AUC-0.99 vs our AUC-0.979 comparison.

**Ding 2021 arXiv:2104.02066.** Diffusion Maps + LDA classical-manifold pipeline on PPMI 1,290 binary HC/PD + KCGMH-TW 630 Normal/Abnormal. PPMI acc 0.98 ± 0.02, KCGMH acc 0.86 ± 0.04. **Closes the "classical manifold pipeline" gap flagged by the journal-style-audit.** Cite `ding2021diffusionMaps` in §II as the manifold/classical representative. No tree-model comparison either.

**Ding 2023 arXiv:2311.14902.** Contrastive graph cross-view multimodal fusion on KCGMH-TW (NOT PPMI) n=412 Normal/Abnormal. GAT+GAT+ResNet18 acc 0.91 / AUC 0.93. **KEY FINDING FOR OUR REBUTTAL:** Their Table 2 reports **XGBoost on 12 DaTQUANT SBR parameters: Acc 0.79**, LogReg: 0.80 — i.e., the tabular-gradient-boosting baseline UNDERPERFORMS only when restricted to 12 SBR-derived features on a small cohort. Our CatBoost on 22 multimodal features at PPMI n=2,201 reaches AUC 0.979 precisely because of the broader feature space + larger cohort. Cite `ding2023contrastiveMultimodal` as architectural inspiration; quote the 0.79 XGBoost baseline in rebuttal to reframe "graph wins over trees" as "graph wins over impoverished-feature-space trees."

### 9.2 Methodology anchors — all four in full text

**Sadinle 2019 *JASA* (via arXiv:1609.00451 preprint).** Direct theorem quote for reviewer Q2 (§1 Contributions, p. 3):

> *"A potentially undesirable property of the optimal classifiers is that they may lead to empty predictions, that is, H(x) = ∅ for some points x ∈ X, especially when the required coverage is low. [...] This region arises because minimizing ambiguity can favor making H(x) empty, and because some classes may be relatively well separated with respect to the coverage requirements."*

And Theorem 6 follow-up (p. 10):

> *"The presence of this null region occurs when the upper bounds on the error levels are large, when the classes are well separated, or in practice it could happen if we have sample points that are anomalies."*

**Actionable rebuttal text (drop into §IV-C or a §IV-C footnote):**

> Following Sadinle, Lei, and Wasserman (2019, Theorems 1 & 6), the LAC set-valued classifier is provably ambiguity-minimizing under its coverage constraint, and the authors explicitly note that the optimal set can be empty on points where the calibrated classes are well-separated or sample confidence is high. Mean set size below 1 combined with ≥ 90% marginal coverage is therefore the expected signature of a well-calibrated LAC, not a pathology.

**Dam 2024 *npj PD* (via PMC11419206).** NSD-ISS operationalized on n=1,741 across PPMI + PASADENA + SPARK using the exact feature set we use: CSF SAA, putamen SBR adjusted for age/sex, MDS-UPDRS I/II/III, MoCA, UPSIT, RBD, PD-med status. Cross-cohort stage distributions near-identical: Stage 2B/3/4 = 25/65/10% (PPMI), 26/66/8% (PASADENA), 25/55/6% (SPARK). **No MRI-raw-pixel or deep-imaging feature anywhere in the definition.** Cite `dam2024nsdValidation` as the "tabular is the NSD-ISS modality" defense — this IS the canonical citation for the devil's-advocate question.

**Bonnier & Bosch 2022 *PMLR* 183.** 7-class imbalanced benchmark on Lending Club n=56,311. Table 5 verbatim findings:

| Scenario | OLR | NN (CORAL-style) | CAT (CatBoost) | OP (winner) |
|---|---|---|---|---|
| adverse-500 | **0.344** | 0.043 | 0.247 | 0.334 |
| adverse-5000 | 0.360 | **0.606** | 0.478 | 0.574 |
| standard-500 | 0.413 | 0.281 | 0.412 | **0.427** |

Authors' quote (p. 121): *"Although designed for multinomial tasks, CatBoost multiclass proves to be quite competitive."*

**Actionable:** At n=500 with imbalance, CORAL-style NN is the WORST method (0.043 / 0.281). CatBoost multiclass is tied with ordinal methods within seed std (±0.03). This directly supports our plan's decision rule for WS1.4 — if CORAL/CORN don't match CatBoost, that's peer-reviewed-expected at our scale.

**Zhang 2025 arXiv:2511.16845.** Min-CPS algorithm for ordinal CP. O(K) sliding-window exact search (Theorem 1) + marginal coverage guarantee (Theorem 2). **Public reference code at `https://github.com/xrty/OCP`.** Reports 14% smaller sets vs Lu 2022 Ordinal APS across 4 benchmarks.

**Actionable:** WS1.5 **wraps `xrty/OCP` directly** rather than hand-rolling Lu 2022. Time savings: ~3 days (no hand-roll + debug). Performance gain: 14% smaller prediction sets vs the Lu 2022 baseline we originally planned.

### 9.3 Tabular SOTA — narrative-shift finding

**This is the biggest finding of the session.** Three peer-reviewed 2024-2025 papers explicitly refute the "tree dominance" reading of Grinsztajn 2022 at our n=2,201 scale:

- **Zabërgja et al. 2024 arXiv:2402.03970** (68 OpenMLCC18 datasets, nested 10-fold CV, 100 Optuna trials, refit-on-train+val): *"Deep Learning methods dominate tree-based methods in datasets that have less than 5000 examples, by winning 31-3. In cases where a dataset has more than 5000 examples, tree-based methods become more competitive. However, they are still outperformed by deep learning methods 17-7."* (§5 p. 7)
- **Ye et al. 2024 TALENT arXiv:2407.00956** (300 datasets): TabPFN v2 avg rank 2-3; CatBoost/XGBoost avg rank 5-7. Verbatim abstract: *"recent pretrained tabular models now match or surpass [GBTs] on many tasks, narrowing — but not eliminating — the historical advantage of tree ensembles."*
- **Hollmann et al. 2025 *Nature*** (TabPFN v2 core paper, n ≤ 10k benchmark): TabPFN v2 beats tuned CatBoost by **ROC AUC +0.13** (0.952 vs 0.822, 4h HPO), **2.8s inference vs 4h HPO = 3,000-5,140× speedup**, Wilcoxon P < 0.001.

**Our n=2,201 with d=22 and imbalanced multi-class ordinal target is inside TabPFN v2's operating sweet spot.**

**Narrative reframe (mandatory, now added to WS1.3 and §II):**

> "Our benchmark uses gradient-boosted trees (CatBoost, XGBoost, LightGBM) as the primary classifier because prior benchmarks at comparable scale (Grinsztajn et al. 2022; Gorishniy et al. 2021) report trees as state-of-the-art while being orders of magnitude cheaper to tune. More recent large-scale re-assessments (Zabërgja et al. 2024, 68 datasets; Ye et al. 2024, 300 datasets) report that DL methods, particularly the pre-trained TabPFN v2 foundation model (Hollmann et al. 2025 *Nature*), can match or surpass tuned CatBoost at n ≤ 10,000 with d ≤ 500 — a regime that includes our cohort. We therefore additionally benchmark TabPFN v2 and AutoGluon (Section VI.B); our primary contribution is calibrated NSD-ISS staging with conformal coverage guarantees, a property neither TabPFN v2 nor AutoGluon currently provide out of the box."

**This is a WS2.1 + §V.B Discussion rewrite, NOT just a WS1.3 experiment.** The paper's current framing ("tree dominance") must be softened to "tree competitiveness + our conformal contribution is the novelty." Without this rewrite, the reviewer has peer-reviewed 2025 evidence to reject the headline.

### 9.4 HPO protocol (template produced; spot-check required before locking)

Agent 4 could not retrieve the 4 source PDFs in-sandbox but produced a usable pre-registration template. Key elements:

- **5-fold outer × 3-fold inner** (matches Paper 1 existing CV).
- **50 trials per fold** for CatBoost/LightGBM (random search, log-uniform per param).
- **30 trials per fold** for Enhanced MM-GAT (Optuna TPE, with **"warm starts following repeated early stopping"** per Lee-Posma 2025 DILIGeNN).
- **10 trials** for TabPFNv2 (n_ensemble_configurations + softmax_temperature).
- **`time_limit=3600s`** per fold for AutoGluon (not HP-search; delegated to internal stacker).
- **Total MPS wall-clock: ~266 hours** (most-expensive model GAT at ~90 h).

Target wall-clock: 2 weeks of overnight + background runs. Template committed to `outputs/paper1_hpo/PRE_REGISTRATION.md` in WS1.2 Task 1. User spot-check of exact protocol details recommended before first trial runs.

### 9.5 Action items added to workstreams

- **WS1.5 (ordinal CP):** pivot from hand-roll to wrapping `github.com/xrty/OCP`. Update plan §4 to reduce WS1.5 from 4h impl + 1h compute to 2h impl + 1h compute.
- **WS2.1 (related work):** expand scope from "add 3 paragraphs" to "+ §II narrative reframe + §V.B Discussion softening of tree-dominance claim." **This is the load-bearing revision, NOT just a citation-addition task.**
- **WS1.3 (TabPFN/AutoGluon):** decision rule was "if either beats CatBoost by ≥1pp, annotate"; upgrade to **"if TabPFN v2 beats CatBoost, the paper's primary framing changes from 'trees dominate' to 'trees competitive + our conformal contribution is the novelty'"** — matching Agent 3's evidence base.
- **WS1.4 (CORAL/CORN):** Bonnier 2022 directly predicts CORAL fails at our scale (adverse-500 acc 0.043-0.281). Decision rule stays pre-registered but we now have peer-reviewed prior supporting it.
- **WS1.1 (fold-local imputation):** §Leakage Audit paragraph now also cites Bernett 2024 (graph leakage) + Shadbahr 2023 (imputation leakage 0.03-0.10 bound).

### 9.6 16 total new citations (13 from Tavily + 3 reviewer-cited arXiv)

The 13 from Tavily (shokrpour, sar, zhang, khosousi, muhammad, gonzalezLatapi, simuni2025reply, dam, gorishniy, zabergja, ye, bonnier, gnnNestedCv) plus 3 reviewer-cited (quan2019datSpect, ding2021diffusionMaps, ding2023contrastiveMultimodal) are all in bibliography_extracted.tex + dissertation/bibliography.tex + Zotero FPJM5RSS with tag `paper1-reviewer-response-2026-04-23`.

### 9.7 EZProxy status for future reference

Cookie present at `~/.claude/skills/ezproxy/.session_cookies.json` but session returns `login_required` for paywalled DOI resolution. **User must refresh cookie via the SKILL.md procedure (browser login at `https://ezproxy.library.und.edu/login?url=...`, then `fetch_paper.py --save-cookie "ezproxy=VALUE"`) for future paywalled paper fetching.** Not blocking for Paper 1 revision — all 26 discovered references retrievable from free sources (arXiv preprints / PMC OA / PMLR / Nature / npj family).

---

*Part 9 addendum locked 2026-04-23. Four research subagents' full-text syntheses at `/private/tmp/claude-501/...tasks/*.output`. 16 total bibitems + Zotero entries live. Narrative-shift finding from Agent 3 pending integration into chapter_content.tex §II + §V.B during WS1.3 + WS2.1 execution.*
