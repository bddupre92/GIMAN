# Paper-Quality Remediation Plan (all 9 papers)

**Created:** 2026-04-18 (post-peer-review-simulation session)
**Trigger:** 9 parallel peer-review agent memos flagged cross-cutting robustness gaps
**Status:** Tier 1 in execution; Tier 2 agent-spawned; Tier 3 deferred to future work

## Peer-review verdict summary

| Paper | Venue | Verdict | Primary blocker |
|---|---|---|---|
| P1 | IEEE JBHI | Minor revision | PD-only retrain AUC 0.561 on n=118 BioFIND; no CIs on Table II/III |
| P2 | IEEE JBHI | **Major revision** | Stage-label circularity + no CIs on downstream claim |
| P3 | IEEE JBHI | **Major revision** | "Digital twin" overclaim; 39% regression-rate interpretation ambiguous |
| P3+4 | npj Digital Medicine | **Major revision / downgrade** | Same DT overclaim + "per-patient" conformal is cohort-invariant |
| P5 | JAMIA | Minor revision | No CIs on window C-td; W4 conflates shift + halved training |
| P6 | JAMIA | **Major revision** | 42.5% accuracy needs baselines; cohort-invariant conformal mislabeled; novelty vs Lian 2024 |
| P8a | PLoS Comp Biol | Minor revision | MLE-SBC vs Talts-style Bayesian SBC terminology |
| P8b | Mov Disord | Minor revision | ΔAIC inconsistency (5,668 vs 3,856); no Wave B replication |
| P9 | CPT:PSP | **Major revision** | p=0.044 fragile; "Level 2.5" w/o plasma PK; confounding under-controlled |

## Cross-paper themes

1. **"Digital twin" overclaim** — flagged P3, P3+4. Reserve for Paper 10 mechanistic work.
2. **Cohort-invariant conformal mislabeled** — flagged P3+4, P6. Either implement Mondrian OR relabel.
3. **Missing CIs / significance tests** — flagged P1, P2, P3, P5.
4. **External-validity honesty** — flagged P1, P3+4, P6, P8b.
5. **Borderline p=0.044 as headline** — P9. Reframe around ΔAIC.
6. **Novelty defense missing** — P6 vs recent PD-CDS literature.

## Tier 1 — Inline quick wins (~4 hours this session, no new experiments)

| # | Task | File | Status |
|---|---|---|---|
| T1.1 | Retitle P3/P3+4 "Digital Twin" → "Graph-Regularized Survival" | `paper3_submission/ieee-jbhi/main.tex` + `paper3plus4_submission/npj-dm/main.tex` | pending |
| T1.2 | Relabel "per-patient" conformal → "marginal cause-specific" | P3+4 + P6 bodies | pending |
| T1.3 | P6 add accuracy baselines (random=25%, majority=~60%), top-2, ordinal MAE | P6 chapter_content.tex §3.2 | pending |
| T1.4 | Fix numeric inconsistencies (P6 22.1/22.8, P8b ΔAIC, P9 β) | multiple | pending |
| T1.5 | P8b Wilcoxon statistic + CI for putamen-caudate gradient | P8b mov-disord/main.tex | pending |
| T1.6 | P9 reconcile β values abstract/highlights/body | P9 cpt-psp/main.tex | pending |
| T1.7 | P6 novelty defense (1 paragraph vs Lian 2024, iPrognosis, etc.) | P6 chapter_content.tex §Background | pending |
| T1.8 | P9 rename "Level 2.5" → honest PD-only framing | P9 cpt-psp/main.tex | pending |
| T1.9 | P3 fold-selection deployment subsection | P3 chapter_content.tex | pending |

## Tier 2 — Substantive analyses (agent-spawned, ~1-2 weeks compute)

All use existing checkpoints / data — no retraining. Deliverables stored to `outputs/paper*_submission/*/revision_analyses/`.

| # | Task | Data source | Est. runtime | Status |
|---|---|---|---|---|
| T2.1 | **P2 stage-label circularity ablation** — recompute NSD-ISS stages from non-imputed features (demographics, genetics, raw SAA/DaT only), rerun downstream experiment, compare to original +2.9% claim | `data/05_features/paper1_features_with_targets.csv` + 48 GIMIN checkpoints | 2-4 h | pending |
| T2.2 | **P1 bootstrap CIs** — (a) 1,000-resample paired-bootstrap AUC for CatBoost/XGBoost/LightGBM; (b) BioFIND PD-only n=118 bootstrap on balanced accuracy + AUC; (c) conformal conditional coverage stratified by NSD-ISS stage | `outputs/paper1_benchmark/`, `outputs/external_validation/`, `outputs/paper1_conformal/` | 2-3 h | pending |
| T2.3 | **P5 window-level bootstrap CIs + W4 control** — (a) 1,000-resample bootstrap over test-set episodes per window for C-td; (b) n-matched random-split control (train=950 / test=950 randomly) to isolate shift vs sample-size contributions to W4 collapse | `outputs/paper3_checkpoints/{deephit,graph_dt}/fold0_*.pt` + `outputs/paper5/` | 3-4 h | pending |
| T2.4 | **P3 paired-bootstrap + ON/OFF stratification + fold-selection doc** — (a) 1,000-resample paired-bootstrap C-td (Graph-DT − DeepHit) to resolve p=0.108 vs t=0.03 footnote discrepancy; (b) stratify 2,859 observed transitions by PDSTATE at assessment; (c) fraction of regressions reversed at next visit; (d) add §Deployment subsection specifying which fold ships | `outputs/paper3_checkpoints/` + `data/06_longitudinal_staging/transition_events.csv` + raw UPDRS CSV | 2-3 h | pending |
| T2.5 | **P6 full-cohort aggregate** — replace n=5 vignette framing with full 1,900-cohort aggregate statistics: accuracy by true stage, per-stage confusion, Top-2 accuracy, ordinal MAE, CONSORT-style diagram for 1,065 (Phase-2 coverage) vs 835 (not covered) subgroups (demographics, visit count, stage distribution comparison) | `outputs/paper6/pipeline_results/v2_full_cohort/pipeline_summary.json` (already exists) | 1-2 h | pending |
| T2.6 | **P8b Wave B sensitivity** — fit M1 (independent per-region decays) on Wave B patients (≥3 scans, n=605), compare population rates to Wave A, report stability | `data/00_raw/GIMAN/ppmi_data_csv/DaTScan_SBR_Analysis.csv` + longitudinal | 2-3 h | pending |
| T2.7 | **P9 Bayesian posterior for β₃** — run PyMC/Stan NUTS on the mixed-effects interaction model, report full posterior for N(t)×LEDD coefficient, 95% credible interval, and posterior probability P(β₃ < 0); reframe headline around ΔAIC=-72 + posterior CI rather than p=0.044 | `outputs/mechanistic_twin/phase4/phase4_assembled_data.parquet` | 3-4 h | pending |

## Tier 3 — Deferred / future work (documented honestly in each paper)

- **Mondrian / locally-weighted conformal** — requires retraining; scope as Paper 11 or supplementary follow-up
- **External validation on LCC/DeNoPa** — blocked on data access; document as "future work" in P1, P3+4, P6, P8b
- **P6 prospective deployment data** — not required for JAMIA Research & Applications track; document as "prospective audit out of scope"
- **P9 plasma PK instantiation** — requires PPMI supplement data access; Musuamba 2021 credibility framework allows Level-2 without plasma if label is honest (Tier 1.8 addresses by renaming)

## Success criteria

**Per-paper acceptance bar:**
- All numeric inconsistencies resolved
- All CIs reported on headline claims
- All overclaimed framing softened to match evidence
- All major-revision blockers converted to addressed-in-revision-letter items

**Cross-paper consistency:**
- "Digital twin" label reserved for P10 only
- Conformal framing consistent ("marginal cause-specific" everywhere unless Mondrian implemented)
- Companion cross-references use `\cite{dupre2026paperN}` keys, not "Paper N" text

## Artifact organization

Each paper's T2 analysis writes to:
```
outputs/mechanistic_twin/paper{N}_submission/{venue}/revision_analyses/
  ├── bootstrap_cis.json          # CI data
  ├── non_leaky_ablation.json     # (P2 only)
  ├── paired_bootstrap.json       # (P3, P5)
  ├── stage_0_1_circularity.md    # (P2 write-up)
  ├── README.md                   # pointer to script that produced results
  └── figures/
      └── *.pdf                    # any new figures for revision
```

## Updates log

- 2026-04-18 initial: plan written; Tier 1 fixes queued; 7 Tier 2 agents to be spawned in parallel.
