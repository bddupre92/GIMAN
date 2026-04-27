# Reporting Summary — Paper 6

**Manuscript:** A Reference-Implementation Clinical Decision-Support Pipeline for NSD-ISS Parkinson's Disease Staging
**Corresponding author:** Blair D. Dupre (blair.dupre@und.edu)
**Target journal:** Journal of the American Medical Informatics Association (JAMIA), Research and Applications track
**Date prepared:** 2026-04-18

JAMIA does not use the Nature-Portfolio Reporting Summary instrument directly; this document is provided for internal records and mirrors its structure. The primary compliance artefact for this submission is the TRIPOD+AI checklist (`supplementary_tripod_ai.md`), scoped to the pipeline-integration perspective with a focus on items 9 (missing data), 16 (calibration), 19 (interpretation), and 21 (implications for model use) — the four items where the integrated pipeline exceeds upstream component-level disclosure.

---

## Statistics

**Applicable sections**
☒ Experimental design ☒ Statistical parameters ☐ Quantification of statistical significance (descriptive reference implementation) ☒ Data replication ☐ Randomisation (deterministic pipeline) ☐ Blinding (deterministic pipeline)

### Experimental design

**Q1. Sample size.**
Reference-implementation deployment study. Full longitudinal cohort: 1,900 PPMI patients with ≥2 NSD-ISS-staged longitudinal visits; 22,800 CatBoost feature slots (1,900 × 12 features); 16,699 staged observations total. The mechanistic $N(t)/N_0$ layer is available for the 1,065 patients (56.1%) with a Phase-2 calibrated posterior. Within-NSD-positive evaluable subset: 1,108 patients at true stage $\in \{1, 2B, 3, 4\}$.

**Q2. Data exclusions.**
No post-hoc exclusions. 792 Stage-0/5/6 patients fall outside the NSD-positive training scope by design and are flagged as out-of-scope in the per-patient JSON output rather than removed from the deployment run. The 835 patients lacking a Phase-2 calibrated posterior are flagged `mechanistic.available: false` in their output records; their staging and transition-timing predictions are unaffected.

**Q3. Replication.**
Deterministic pipeline: same inputs → same outputs (no stochasticity in CatBoost inference with fixed checkpoint; MC-dropout in GIMIN uses a fixed seed of 42 with $T = 20$ samples). The reference implementation is a single-pass deployment on the full cohort; replication is byte-level by re-running `scripts/paper6/unified_pipeline_demo_v2.py`. Component-level cross-validation estimates (Paper 1 CatBoost 5-fold, Paper 3 Graph-DT 5-fold, Paper 4 5-fold) are inherited from companion submissions.

**Q4. Randomisation.**
Pipeline is deterministic (fixed seeds). PD+Prodromal training restriction (Espay correction) is deterministic from `COHORT_DEFINITION` in `paper1_features_with_targets.csv`.

**Q5. Blinding.**
Not applicable (deterministic reference implementation; no comparator evaluation in this paper).

### Statistical parameters

**Q6. Reported statistical parameters.**
- Sample size: 1,900 total; 1,108 within-NSD-positive evaluable; 1,065 with mechanistic posterior; 22,800 feature slots.
- Throughput: mean 12 ms/patient, 83 patients/s aggregate (single Apple M-series MPS device).
- Accuracy: within-NSD+ staging 471/1108 (42.5%); reported as a descriptive scope metric, not a performance claim.
- Provenance distribution: GIMIN-imputed 33.3% / Paper-1 raw 54.9% / median fallback 11.8%.
- Uncertainty: per-feature temperature scaler median $T = 0.87$ (IQR across 33 features); Paper-4 IPCW conformal band width 0.037 at 90% CL (cohort-invariant marginal wrapper).
- Directional concordance: forward 31.6%, hold 41.0%, backward 27.0% for the 1,074 patients with a non-self top-1 destination.
- Mechanistic layer: cohort-median $N(t)/N_0$ = 0.93 (IQR [0.74, 0.97]) for all 1,065 calibrated; long-follow-up ($\geq 10$ yr) median 0.72 (IQR [0.35, 0.91]) for $n = 241$.
- No hypothesis-test statistics (no inferential claims); reported numbers are sample statistics on a fixed deployment cohort.

### Software

**Q7. Software used for analysis.**
Python 3.12; PyTorch 2.8.0 + PyTorch Geometric 2.6.1 (GIMIN, Graph-DT, DeepHit); CatBoost 1.2.10 (staging); MAPIE 1.3.0 (conformal wrapper at Paper 4 layer); scikit-learn 1.5; matplotlib 3.8; numpy 1.26; pandas 2.2. Custom code at `scripts/paper6/unified_pipeline_demo_v2.py`, `scripts/paper6/generate_v2_aggregate_figures.py`, `scripts/paper6/generate_n_frac_distribution_figure.py`, and `src/giman_pipeline/imputation/temperature_scaling.py`.

---

## Materials and reagents

Not applicable (computational study).

---

## Human research participants

**Q8. Human research participants.**
Yes. PPMI participants (n=1,900, same longitudinal cohort as Papers 3, 4, 5). Informed consent and IRB approval obtained by PPMI at the 33 participating-site level. Secondary analysis exempt from additional IRB review per University of North Dakota institutional policy.

**Q9. Ethics oversight.**
- Primary data: IRBs of 33 PPMI participating sites (PPMI central protocol).
- Secondary analysis: University of North Dakota IRB (exempt, de-identified data).

**Q10. Population characteristics.**
Median age at baseline 64 years (IQR 57–70); 64% male / 36% female; ~92% non-Hispanic white. Cohort distribution across the NSD-ISS stages: Stage 0 n=716, Stage 1 n=65, Stage 2B n=198, Stage 3 n=681, Stage 4 n=164, Stage 5 n=70, Stage 6 n=6. The NSD-positive CatBoost target ($\{1, 2B, 3, 4\}$) covers 1,108 of 1,900 (58.3%) of the evaluable cohort.

**Q11. Recruitment.**
Not applicable (secondary analysis of an existing cohort).

---

## Data availability

**Q12. Data availability.**
- PPMI data: Data Use Agreement at https://www.ppmi-info.org/access-data-specimens/download-data.
- Paper-1 feature file (`data/05_features/paper1_features_with_targets.csv`) and Paper-3 longitudinal features (`data/07_paper3_features/longitudinal_features.csv`) archived at https://github.com/bddupre92/PD_PHD.
- GIMIN checkpoint (Paper 2): `outputs/paper2_benchmark/runs/full_benchmark_20260222_160247/checkpoints/frac0.1_run0_GIMIN_StageDecoderOnly.pt`.
- DeepHit and Graph-DT fold-0 checkpoints (Paper 3): `outputs/paper3_checkpoints/{deephit,graph_dt}/fold0_*.pt`.
- Paper-4 IPCW conformal aggregate: `outputs/paper4/conformal/aggregate_summary.json`.
- Phase-2 mechanistic posteriors (Paper 7/9): `outputs/mechanistic_twin/data/posteriors/phase2_combined_1065.csv` (1,065 patients with per-patient percent-loss-per-year posteriors).
- Full 1,900-patient per-patient pipeline outputs: `outputs/paper6/pipeline_results/v2_full_cohort/` (pipeline_summary.json + per-vignette JSONs).

**Q13. Code availability.**
Code openly archived at https://github.com/bddupre92/PD_PHD under MIT license. Key scripts: `scripts/paper6/unified_pipeline_demo_v2.py` (end-to-end pipeline), `scripts/paper6/generate_v2_aggregate_figures.py` (aggregate figs), `scripts/paper6/generate_n_frac_distribution_figure.py` (Fig. 6 $N(t)/N_0$ histogram), `src/giman_pipeline/imputation/temperature_scaling.py` (per-feature temperature scaler module).

---

## Deployment-monitoring framework

**FDA SaMD PCCP mapping.**
Table 1 of the main manuscript specifies four pre-registered metrics (M1 directional concordance, M2 conformal coverage, M3 NSD+ recall, M4 component failure rate) with acceptance thresholds designed to translate directly into pre-specified retraining triggers that a Software-as-a-Medical-Device sponsor would register as part of a Predetermined Change Control Plan (FDA 2024). Figure 7 (deployment-monitoring workflow) visualises the closed-loop control flow from CDS output to PCCP-gated retrain/continue decisions. For temporal-drift instrumentation, this pipeline inherits the companion-Paper-5 KS + PSI + MMD monitoring triad.

**Companion mechanistic layer.**
The $N(t)/N_0$ output is derived from companion Paper-9 Phase-2 importance-sampled posteriors and is surfaced without rerunning the mechanistic calibration at deployment time. Any site conducting an $n \geq 100$ deployment audit can additionally log the cohort $N(t)/N_0$ distribution as a fifth monitored signal, cross-referenced to Paper-9 Path-B's N(t)×LEDD interaction finding.

---

## Declaration of reproducibility

The 1,900-patient end-to-end run is deterministic and reproducible byte-for-byte from `scripts/paper6/unified_pipeline_demo_v2.py` given the checkpoint paths above. Per-feature provenance, directional concordance, and $N(t)/N_0$ statistics in the Results and Figures 4–6 are derived entirely from `outputs/paper6/pipeline_results/v2_full_cohort/pipeline_summary.json`. All figures regenerate from `scripts/paper6/generate_v2_aggregate_figures.py` and `scripts/paper6/generate_n_frac_distribution_figure.py`.

---

## End of Reporting Summary
