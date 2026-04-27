Blair D. Dupre
Department of Biomedical Engineering
University of North Dakota
Grand Forks, ND 58202 USA
blair.dupre@und.edu

[Date of submission]

The Editors
*npj Digital Medicine*
Nature Portfolio

Dear Editors,

I am pleased to submit the enclosed manuscript, **"Graph-Informed NSD-ISS Transition Prediction with Conformalized Uncertainty and Subgroup Equity in Parkinson's Disease,"** for consideration as an Article in *npj Digital Medicine*.

## Why this paper belongs at npj Digital Medicine

This work sits squarely in your stated scope: distribution-free uncertainty quantification for clinical machine learning applied to a biologically-grounded staging framework, with explicit subgroup-equity analysis and a biological-plausibility cross-reference. It advances digital-medicine practice in three concrete ways that build on themes the journal has recently prioritised:

1. **Temporal prediction on a biological staging system.** The Neuronal alpha-Synuclein Disease Integrated Staging System (NSD-ISS, Simuni *et al.* 2024 *Lancet Neurology*) has been validated cross-sectionally and longitudinally, but no computational model has reported per-patient transition-timing predictions. We develop a Graph-Informed Digital Twin that combines a gated recurrent temporal encoder with graph attention over a $k$-nearest-neighbour patient-similarity graph, and benchmark it against Dynamic-DeepHit and a multi-state Markov baseline on 1,900 Parkinson's Progression Markers Initiative patients (16,699 observations, 2,859 transitions).

2. **Cause-specific conformal prediction with IPCW for competing-risks survival.** We develop the first cause-specific IPCW conformal prediction bands for competing-risks cumulative incidence functions on NSD-ISS transitions, achieving 95.1% (DeepHit) / 95.3% (Graph-DT) marginal coverage at the 95% confidence level and 90.2% / 90.5% at 90%. The framework addresses the distribution-free-uncertainty imperative articulated in recent TRIPOD+AI reporting guidance (Collins *et al.* 2024 *BMJ*) and includes the methodologically-correct Candès–Lei–Ren (2023) IPCW weight formula, validated against four conformal-baseline ablations.

3. **A novel directional analysis and equity audit.** We present the first directional analysis of conformal coverage for NSD-ISS transitions, revealing a coverage asymmetry between forward progressions (91.0%) and backward regressions (85.4%) — a finding with direct clinical implications for treatment-aware modelling. Subgroup analysis across sex, age, and three genetic strata (LRRK2, GBA, APOE ε4) finds no significant model-by-subgroup interactions after Benjamini–Hochberg FDR correction with pooled-out-of-fold permutation testing (smallest p_FDR = 0.72). Conditional conformal coverage stays within 0.87–0.93 of the 0.90 nominal target across all eight (model × stratum) cells, and a Mondrian per-stratum recalibration confirms equity at the deployment-grade per-subgroup coverage guarantee level.

## Relationship to our recent work and the current debate on NSD-ISS

The manuscript explicitly engages with two parallel critiques published in *Movement Disorders* this year: Espay *et al.* 2025 (DOI:10.1002/mds.30269) argue that medication-status confounds undermine the NSD-ISS clinical sub-staging, and Simuni *et al.* 2025 (DOI:10.1002/mds.30272) reply. We adopt the methodological prescription of both sides (PD-only reference-class training, treatment-aware prose) throughout the paper. The companion paper from our group (Paper 1, under review at IEEE JBHI) addresses the cross-sectional prediction problem on the same cohort with a PD-only retraining experiment; the present combined paper addresses the temporal prediction and uncertainty quantification problem. Our analysis shows that transition prediction on the full PPMI cohort is insensitive to the healthy-control confound because HC participants contribute only 0.9% of observed transitions, and we report this sensitivity analysis in the Methods.

## Originality and significance for the digital-medicine readership

To our knowledge this is the first paper to report (i) graph-informed temporal survival modelling for NSD-ISS, (ii) cause-specific IPCW conformal bands on a biological PD staging system, (iii) directional conformal coverage analysis, and (iv) per-subgroup conditional conformal coverage on a Parkinson's cohort. The current revision additionally contributes (v) a deletion-shift faithfulness analysis of the graph-attention mechanism (load-bearing negative result that tightens the interpretability claim), (vi) a Mondrian per-stratum conformal recalibration sensitivity check, (vii) a subject-level cluster-bootstrap CI methodology, (viii) a hidden semi-Markov model sensitivity check on the CTMC's Exponential sojourn assumption, and (ix) a Fine–Gray subdistribution-hazard competing-risks parametric baseline. All contributions are methodologically transferable to other progressive neurological diseases with validated biological staging (Alzheimer's disease under the AT(N) framework, multiple sclerosis, and Huntington's disease) and to any competing-risks clinical prediction setting where guaranteed coverage matters for deployment.

## Fit with recent *npj Digital Medicine* content

Our approach extends the trajectory set by Sreenivasan *et al.* 2025 (*npj Digital Medicine* 8:224) — the closest conceptual precedent in your pages — on three dimensions: (1) we move from binary disease-course classification to competing-risks time-to-event prediction with cause-specific CIF bands; (2) we add graph-informed population context rather than purely per-patient temporal features; and (3) we extend coverage guarantees to per-subgroup conditional coverage with explicit genetic-stratum equity analysis.

## Authorship, data, and reproducibility

I am the sole author of this manuscript. All analysis code is publicly archived at https://github.com/bddupre92/PD_PHD; pre-trained model checkpoints for all 10 cross-validation folds (5 DeepHit + 5 Graph-DT) accompany the submission and will be deposited to Zenodo with a DOI upon acceptance. A completed TRIPOD+AI reporting checklist and a Nature Portfolio Life Sciences Reporting Summary are included as Supplementary Information. The PPMI dataset underlying the analysis is available to qualified researchers under a standard Data Use Agreement.

## Suggested reviewers and exclusions

I respectfully suggest the following reviewers, selected for methodological expertise rather than cohort or institutional overlap:

- **Andrew Gordon Wilson** (NYU) — graph-informed probabilistic modelling and uncertainty quantification
- **Rich Caruana** (Microsoft Research) — calibration and interpretability for clinical ML
- **Tatiana Foroud** or **Caroline Tanner** (PPMI) — NSD-ISS biological anchors and longitudinal staging

I would respectfully ask that the following be excluded as conflicts of interest: members of my dissertation committee at the University of North Dakota and members of the PPMI steering committee with whom I have had direct correspondence in the past 12 months.

This work has not been published elsewhere and is not under consideration at another journal. A preprint will be posted to arXiv immediately after submission.

Thank you for considering this manuscript. I am happy to address any editorial questions or make the pre-trained model checkpoints available to the review panel in a pre-registration format.

Sincerely,

Blair D. Dupre
Department of Biomedical Engineering, University of North Dakota

---

*Submission materials enclosed:*
- Main manuscript (32 pages, PDF) — `main.pdf`
- Supplementary Information File (22 pages, PDF) — `supplementary.pdf`, comprising:
  - §S-1 / §S-2: Reserved
  - §S-3: Carrier subgroup analysis (LRRK2, GBA, APOE ε4) with H1/H2/H3 verdicts and Mondrian recalibration
  - §S-CRIT-A: IPCW formula correction (Candès–Lei–Ren 2023; +8.5pp coverage at 90% CL)
  - §S-CRIT-B: Pooled-OOF permutation test replacing Fisher's combination
  - §S-WS-P3-2: Subject-level (cluster) bootstrap C-td CIs
  - §S-WS-P3-7c: Fine–Gray subdistribution-hazard competing-risks baseline
  - §S-WS-P3-10: Hidden semi-Markov model sensitivity on the CTMC sojourn assumption
  - §S-WS-P3-15: Graph-DT attention faithfulness via deletion-shift experiment (load-bearing negative result)
  - §S-WS-P3-16: External-validation data-availability disclosure (data-blocked, not compute-blocked)
- Completed TRIPOD+AI reporting checklist (`supplementary_tripod_ai.md`, 27 + 10 AI/ML items)
- Nature Portfolio Life Sciences Reporting Summary (`supplementary_reporting_summary.md`)
- 13 manuscript figures + supplementary figures embedded in the supplementary PDF
- Code archive at https://github.com/bddupre92/PD_PHD (Zenodo DOI reserved upon acceptance); pre-trained checkpoints for all 10 cross-validation folds (5 DeepHit + 5 Graph-DT) included.
