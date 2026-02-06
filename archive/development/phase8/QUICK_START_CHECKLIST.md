# Phase 8 Quick Start Checklist

**Project:** GIMAN Model Frontier Development  
**Timeline:** November 2025 - August 2026  
**Current Status:** Planning Phase

---

## Pre-Launch Checklist (Before Nov 4, 2025)

### Data Access & Permissions
- [ ] Verify PPMI data access current (expires: _______)
- [ ] Submit PDBP data application (4-6 week approval)
- [ ] Join PPMI SAA working group
- [ ] Request access to CSF SAA assay data
- [ ] Download latest PPMI data freeze (version: _______)

### Computational Resources
- [ ] Secure GPU cluster access (4× A100 or equivalent)
- [ ] Set up cloud storage for large model checkpoints (500GB)
- [ ] Configure Weights & Biases (wandb) account for experiment tracking
- [ ] Install CUDA 11.8+ and PyTorch 2.1+
- [ ] Verify PyTorch Geometric installation

### Environment Setup
- [ ] Create `phase8_env` virtual environment
- [ ] Install all dependencies from `requirements.txt`
- [ ] Configure environment variables:
  ```bash
  GIMAN_DATA_ROOT="E:/My Drive/CSCI FALL 2025/data"
  GIMAN_RESULTS_ROOT="E:/My Drive/CSCI FALL 2025/results/phase8"
  WANDB_API_KEY="your_wandb_key"
  ```
- [ ] Clone/pull latest Phase 1-7 code
- [ ] Run environment validation script

### Documentation Review
- [ ] Read `PHASE8_STRATEGIC_ROADMAP.md` (10,200 words)
- [ ] Read `SUBPHASE_8_1_DETAILED_PLAN.md` (4,800 words)
- [ ] Review Phase 4-6 manuscripts for context
- [ ] Familiarize with PPMI data dictionary
- [ ] Review survival analysis literature (lifelines, pycox)

---

## Subphase 8.1: Foundational Re-scoping (Nov 4-25, 2025)

### Week 1: Nov 4-8 (Architecture Definition)

#### Day 1-2: Dual-Model Architecture
- [ ] **Task 1.1:** Define `giman_progression.py` architecture
  - [ ] Copy base GIMAN code from Phase 4-6
  - [ ] Modify output head for survival prediction (hazard ratios)
  - [ ] Add Cox loss function
  - [ ] Document architecture in docstrings
  - [ ] Create unit tests for forward pass

- [ ] **Task 1.1 (continued):** Define `giman_conversion.py` architecture
  - [ ] Adapt architecture for prodromal cohort
  - [ ] Add phenoconversion-specific features
  - [ ] Create initialization from pretrained GIMAN-Progression
  - [ ] Test on synthetic data

#### Day 3-5: Prodromal Inclusion Criteria
- [ ] **Task 1.2:** Define prodromal inclusion criteria
  - [ ] Review PPMI study documentation for prodromal definitions
  - [ ] Create `prodromal_inclusion_criteria.yaml`:
    - [ ] Genetic risk: LRRK2, GBA, SNCA mutations
    - [ ] RBD criteria: RBD-SQ score ≥5 or PSG-confirmed
    - [ ] Olfactory: UPSIT <25th percentile for age/sex
    - [ ] DAT-SPECT: Putamen SBR <65% age-expected mean
  - [ ] Define exclusion criteria (e.g., dementia, severe comorbidities)
  - [ ] Validate against PPMI published prodromal cohorts

### Week 2: Nov 11-15 (Data Curation)

#### Day 1-3: Prodromal Data Extraction
- [ ] **Task 1.3:** Extract prodromal data from PPMI
  - [ ] Query Demographics for prodromal `COHORT_DEFINITION`
  - [ ] Query Genetic data: `iu_genetic_consensus_20250515.csv`
  - [ ] Query RBD scores: `REM_Sleep_Behavior_Disorder.csv`
  - [ ] Query UPSIT: `University_of_Pennsylvania_Smell.csv`
  - [ ] Query DAT-SPECT: `Xing_Core_Lab_Quant_SBR.csv`
  - [ ] Merge on `PATNO` and `EVENT_ID`

- [ ] **Task 1.3 (continued):** Apply inclusion/exclusion
  - [ ] Implement filtering logic in `prodromal_cohort_curation.py`
  - [ ] Generate baseline cohort CSV (n≥150 target)
  - [ ] Extract longitudinal follow-up data
  - [ ] Identify phenoconversion events (progression to PD diagnosis)

#### Day 4-5: Quality Control & Characterization
- [ ] **Task 1.3 (QC):** Data quality control
  - [ ] Check for missing data: require >85% completeness
  - [ ] Validate data ranges (e.g., SBR values 0-5)
  - [ ] Flag outliers for manual review
  - [ ] Generate QC report

- [ ] **Task 1.5:** Cohort characterization report
  - [ ] Demographics: age, sex, race distribution
  - [ ] Risk factor prevalence:
    - [ ] % with genetic mutations
    - [ ] % with RBD
    - [ ] % with olfactory dysfunction
    - [ ] % with DAT deficit
  - [ ] Phenoconversion statistics:
    - [ ] Number converted to PD
    - [ ] Median time to conversion
    - [ ] Conversion rate (target 20-30%)
  - [ ] Baseline clinical features: UPDRS, cognition
  - [ ] Write `prodromal_cohort_characterization.md`

### Week 3: Nov 18-22 (Configuration System)

#### Day 1-2: Unified Configuration
- [ ] **Task 1.4:** Create `dual_model_config.yaml`
  - [ ] Define paths section (data, models, results)
  - [ ] Define model hyperparameters:
    - [ ] GIMAN-Progression config
    - [ ] GIMAN-Conversion config
  - [ ] Define training parameters:
    - [ ] Learning rate, batch size, epochs
    - [ ] Loss weights (if multi-task)
    - [ ] Early stopping criteria
  - [ ] Define survival analysis parameters:
    - [ ] Time bins for discrete hazard
    - [ ] Censoring handling
  - [ ] Add validation and test split parameters

- [ ] **Task 1.4 (continued):** Configuration loader
  - [ ] Create `utils/config_loader.py`
  - [ ] Add YAML validation (required fields)
  - [ ] Add environment variable substitution
  - [ ] Test loading and merging configs

#### Day 3-5: Integration Testing
- [ ] Test dual-model loading with config
- [ ] Test prodromal data pipeline end-to-end
- [ ] Validate configuration overrides work correctly
- [ ] Run integration tests on small data subset
- [ ] Fix any bugs discovered

### Week 4: Nov 25 (Documentation & Handoff)

#### Final Day: Deliverables & Transition
- [ ] **Finalize Subphase 8.1 deliverables:**
  - [ ] `giman_progression.py` (functional, tested)
  - [ ] `giman_conversion.py` (functional, tested)
  - [ ] `prodromal_cohort_baseline.csv` (n≥150)
  - [ ] `prodromal_cohort_longitudinal.csv` (with conversion events)
  - [ ] `dual_model_config.yaml` (complete, documented)
  - [ ] `prodromal_cohort_characterization.md` (comprehensive report)

- [ ] **Code quality:**
  - [ ] All code passes `ruff` linting
  - [ ] Google-style docstrings on all functions/classes
  - [ ] Unit tests achieve >80% coverage
  - [ ] Integration tests pass

- [ ] **Git operations:**
  - [ ] Commit all changes with conventional commit messages
  - [ ] Create tag: `v8.1.0-foundational`
  - [ ] Push to `phase8-development` branch
  - [ ] Merge to `main` after review

- [ ] **Transition to Subphase 8.2:**
  - [ ] Review `SUBPHASE_8_2_DETAILED_PLAN.md` (to be created)
  - [ ] Prepare for dynamic endpoint modeling
  - [ ] Schedule kick-off meeting

---

## Subphase 8.2: Dynamic Endpoints (Nov 25 - Dec 23)

### Key Milestones
- [ ] **Week 1 (Nov 25 - Dec 1):** Operationalize 25 disability milestones
  - [ ] Define endpoints from UPDRS, ADL, cognitive assessments
  - [ ] Extract time-to-event data for each endpoint
  - [ ] Handle censoring (patients without event)

- [ ] **Week 2 (Dec 2-8):** Define phenoconversion endpoint
  - [ ] Criteria for prodromal→PD diagnosis
  - [ ] Extract conversion times from PPMI
  - [ ] Validate against published conversion rates

- [ ] **Week 3 (Dec 9-15):** Survival data engineering
  - [ ] Format data for Cox models and DeepSurv
  - [ ] Create time-varying covariates if needed
  - [ ] Generate survival curves for exploratory analysis

- [ ] **Week 4 (Dec 16-23):** Cox baseline models
  - [ ] Fit Cox proportional hazards models
  - [ ] Assess proportional hazards assumption
  - [ ] Calculate C-index (target >0.70)
  - [ ] Generate baseline performance report

---

## Subphase 8.3: SAA Integration (Dec 23 - Jan 20)

### Key Milestones
- [ ] **Week 1 (Dec 23-29):** SAA data curation
  - [ ] Access CSF SAA assay data from PPMI
  - [ ] Merge with multimodal features
  - [ ] Stratify by SAA+ vs SAA- (seed aggregation positive)

- [ ] **Week 2 (Dec 30 - Jan 5):** GIMAN-SAA model development
  - [ ] Adapt GIMAN for binary SAA classification
  - [ ] Train on multimodal features (exclude CSF)
  - [ ] Optimize hyperparameters

- [ ] **Week 3 (Jan 6-12):** Model validation
  - [ ] Evaluate on hold-out test set (AUC target >0.85)
  - [ ] Compare to clinical predictors alone
  - [ ] Assess feature importance (SHAP)

- [ ] **Week 4 (Jan 13-20):** SAA explainability analysis
  - [ ] Which modalities contribute most?
  - [ ] Are genetic features (GBA) important?
  - [ ] Does DAT-SPECT correlate with SAA status?

---

## Subphase 8.4: VAE Heterogeneity (Jan 20 - Feb 10)

### Key Milestones
- [ ] **Week 1 (Jan 20-26):** Extract GIMAN embeddings
  - [ ] Run GIMAN-Progression on full PD cohort
  - [ ] Extract 64-dim node embeddings from final GAT layer
  - [ ] Save embeddings with patient metadata

- [ ] **Week 2 (Jan 27 - Feb 2):** Train VAE
  - [ ] Architecture: 64→32→16→8 (encoder), reverse (decoder)
  - [ ] Optimize beta-VAE loss (reconstruction + KL divergence)
  - [ ] Target reconstruction loss <0.15

- [ ] **Week 3 (Feb 3-9):** Latent space analysis
  - [ ] Visualize 8D latent space (t-SNE/UMAP to 2D)
  - [ ] Correlate latent dimensions with clinical features
  - [ ] Identify interpretable axes (e.g., motor vs cognitive)

- [ ] **Week 4 (Feb 10):** Continuous subtyping validation
  - [ ] Compare to discrete Phase 4 subtypes
  - [ ] Assess clinical utility (prognostic value)
  - [ ] Generate interactive visualization

---

## Subphase 8.5: Multi-Task Architecture (Feb 10 - Mar 10)

### Key Milestones
- [ ] **Week 1 (Feb 10-16):** Design multi-task GIMAN
  - [ ] Shared GAT encoder (3 layers, 4 heads, 64 dims)
  - [ ] Task-specific heads:
    - [ ] Progression head (survival output)
    - [ ] Conversion head (survival output)
    - [ ] SAA head (binary classification)
    - [ ] Diagnostic head (PD vs Control)

- [ ] **Week 2 (Feb 17-23):** Implement composite loss
  - [ ] Weighted sum of task losses
  - [ ] Dynamic task weighting (uncertainty-based)
  - [ ] Gradient normalization

- [ ] **Week 3 (Feb 24 - Mar 2):** Training & optimization
  - [ ] Train multi-task model
  - [ ] Monitor per-task performance
  - [ ] Ensure no performance degradation vs single-task

- [ ] **Week 4 (Mar 3-9):** Multi-task evaluation
  - [ ] Compare to single-task baselines
  - [ ] Assess shared representation quality
  - [ ] Analyze task relationships (positive/negative transfer)

---

## Subphase 8.6: Enhanced Explainability (Mar 10-31)

### Key Milestones
- [ ] **Week 1 (Mar 10-16):** SHAP for survival
  - [ ] Adapt SHAP for time-dependent hazards
  - [ ] Compute SHAP values for top features
  - [ ] Temporal feature importance curves

- [ ] **Week 2 (Mar 17-23):** GNNExplainer for prodromal
  - [ ] Identify important subgraphs for conversion
  - [ ] Visualize neighbor influence
  - [ ] Quantify graph structure importance

- [ ] **Week 3 (Mar 24-30):** Cross-method consensus
  - [ ] Compare SHAP, GNNExplainer, Grad-CAM, attention
  - [ ] Calculate cross-method agreement (target >85%)
  - [ ] Generate consensus report

- [ ] **Week 4 (Mar 31):** Counterfactual analysis
  - [ ] Generate counterfactuals for high-risk predictions
  - [ ] What feature changes prevent conversion?
  - [ ] Validate clinical plausibility with experts

---

## Subphase 8.7: External Validation (Mar 31 - Apr 28)

### Key Milestones
- [ ] **Week 1 (Mar 31 - Apr 6):** PDBP data integration
  - [ ] Download PDBP data (assume approval received)
  - [ ] Harmonize with PPMI preprocessing
  - [ ] Match feature set (87 features)

- [ ] **Week 2 (Apr 7-13):** External validation pipeline
  - [ ] Run GIMAN-Progression on PDBP
  - [ ] Run GIMAN-Conversion on PDBP prodromal (if available)
  - [ ] Calculate C-index (target >0.73, <5% drop)

- [ ] **Week 3 (Apr 14-20):** SOTA benchmarking
  - [ ] Compare to: Random Survival Forest, Cox, DeepSurv
  - [ ] Quantify improvement (ΔC-index, p-values)
  - [ ] Generate benchmark table

- [ ] **Week 4 (Apr 21-28):** Validation report
  - [ ] Write comprehensive external validation report
  - [ ] Discuss generalizability
  - [ ] Identify failure modes (where does GIMAN struggle?)

---

## Subphase 8.8: Dissemination (Apr 28 - Jul 21)

### Manuscript Writing (8 weeks)

#### Week 1-3 (Apr 28 - May 18): Paper 1 (Multi-Task Framework)
- [ ] Draft introduction
- [ ] Draft methods section
- [ ] Generate all figures (GIMAN architecture, performance plots)
- [ ] Draft results section
- [ ] Draft discussion

#### Week 4-6 (May 19 - Jun 8): Paper 2 (SAA Prediction)
- [ ] Draft introduction (SAA clinical importance)
- [ ] Draft methods (GIMAN-SAA architecture)
- [ ] Generate SAA prediction figures
- [ ] Draft results (AUC >0.85, feature importance)
- [ ] Draft discussion (clinical utility)

#### Week 7-8 (Jun 9-22): Paper 3 (Continuous Heterogeneity)
- [ ] Draft introduction (subtyping landscape)
- [ ] Draft methods (VAE architecture)
- [ ] Generate latent space visualizations
- [ ] Draft results (interpretable axes)
- [ ] Draft discussion (implications for trials)

### Open-Source Release (4 weeks)

#### Week 9-10 (Jun 23 - Jul 6): Code packaging
- [ ] Clean and document all code
- [ ] Create `setup.py` and `pyproject.toml`
- [ ] Write comprehensive README.md
- [ ] Create tutorials:
  - [ ] 01_quickstart.ipynb (basic usage)
  - [ ] 02_custom_data.ipynb (adapt to new dataset)
  - [ ] 03_explainability.ipynb (interpret predictions)
- [ ] Add LICENSE (MIT or Apache 2.0)

#### Week 11-12 (Jul 7-21): Release & promotion
- [ ] Create GitHub repository (public)
- [ ] Upload pretrained models (Zenodo or HuggingFace)
- [ ] Write blog post announcing release
- [ ] Submit to Papers with Code
- [ ] Announce on Twitter, LinkedIn, relevant forums
- [ ] Organize workshop/webinar (date: _______)

---

## Success Criteria Summary

### Subphase 8.1 (Foundational)
- [x] Dual-model architecture defined and implemented
- [x] Prodromal cohort curated (n≥150, >85% complete)
- [x] Configuration system operational
- [x] Cohort characterization report complete

### Subphase 8.2 (Dynamic Endpoints)
- [ ] 25 disability milestones operationalized
- [ ] Phenoconversion endpoint defined
- [ ] Cox baseline C-index >0.70

### Subphase 8.3 (SAA Integration)
- [ ] GIMAN-SAA AUC >0.85
- [ ] SAA prediction validated on hold-out set
- [ ] Feature importance analyzed

### Subphase 8.4 (VAE Heterogeneity)
- [ ] VAE trained (reconstruction loss <0.15)
- [ ] Latent space interpretable (≥3 axes correlate with biology)
- [ ] Interactive visualization created

### Subphase 8.5 (Multi-Task)
- [ ] Multi-task model implemented
- [ ] Performance matches single-task baselines
- [ ] Composite loss optimized

### Subphase 8.6 (Explainability)
- [ ] SHAP for survival implemented
- [ ] Cross-method consensus >85%
- [ ] Counterfactual analysis complete

### Subphase 8.7 (Validation)
- [ ] PDBP validation C-index >0.73
- [ ] Benchmarking vs SOTA complete
- [ ] External validation report written

### Subphase 8.8 (Dissemination)
- [ ] 3 manuscripts drafted and submitted
- [ ] Open-source package released
- [ ] Tutorials and documentation complete
- [ ] Workshop organized

---

## Monthly Progress Reviews

### November 2025 Review (Nov 29)
- [ ] Subphase 8.1 complete
- [ ] Subphase 8.2 50% complete
- [ ] Prodromal cohort validated
- [ ] No major blockers

### December 2025 Review (Dec 27)
- [ ] Subphase 8.2 complete
- [ ] Subphase 8.3 50% complete
- [ ] Cox baseline established (C-index: _____)
- [ ] SAA data accessed

### January 2026 Review (Jan 30)
- [ ] Subphase 8.3 complete
- [ ] Subphase 8.4 50% complete
- [ ] GIMAN-SAA validated (AUC: _____)
- [ ] VAE training initiated

### February 2026 Review (Feb 27)
- [ ] Subphase 8.4 complete
- [ ] Subphase 8.5 60% complete
- [ ] Latent space interpretable
- [ ] Multi-task training in progress

### March 2026 Review (Mar 31)
- [ ] Subphase 8.5 complete
- [ ] Subphase 8.6 complete
- [ ] Multi-task model optimized
- [ ] Explainability framework adapted

### April 2026 Review (Apr 28)
- [ ] Subphase 8.7 complete
- [ ] External validation successful (PDBP C-index: _____)
- [ ] Benchmarking shows improvement: Δ_____ over baselines
- [ ] Ready to begin dissemination

### May-July 2026 Review (Jul 21)
- [ ] Subphase 8.8 complete
- [ ] Manuscripts submitted:
  - [ ] Paper 1: ______ (journal)
  - [ ] Paper 2: ______ (journal)
  - [ ] Paper 3: ______ (journal)
- [ ] Open-source release complete
- [ ] Workshop held (date: ______, attendees: _____)

---

## Emergency Contacts & Resources

### Technical Issues
- **PyTorch/CUDA:** [PyTorch Forum](https://discuss.pytorch.org/)
- **PyTorch Geometric:** [GitHub Issues](https://github.com/pyg-team/pytorch_geometric/issues)
- **Survival Analysis:** Lifelines Gitter chat

### Data Issues
- **PPMI Support:** support@ppmi-info.org
- **PDBP Support:** pdbp@ninds.nih.gov

### Compute Issues
- **GPU Cluster:** [Your HPC helpdesk]
- **Cloud Provider:** [AWS/GCP/Azure support]

### Project Management
- **PI:** [PI Email]
- **Lab Manager:** [Manager Email]
- **Collaborators:** [Clinical partner emails]

---

*Document Version: 1.0*  
*Last Updated: October 6, 2025*  
*Next Review: November 4, 2025* (Subphase 8.1 kickoff)
