# Subphase 8.1: Foundational Integration & Prodromal Cohort Enhancement

**Duration:** 1-2 weeks (November 4-15, 2025) **[REVISED from 3-4 weeks]**  
**Status:** Planning  
**Priority:** CRITICAL  
**Code Reusability:** 70-75% of infrastructure exists from Phases 4-6

---

## Objective

**REVISED OBJECTIVE:** Integrate existing Phase 4-6 infrastructure (GAT architecture, survival models, prodromal cohort) and enhance prodromal data with missing multimodal features (genetic, imaging, biomarkers).

**Key Change from Original Plan:** This is now an **integration and enhancement** task, not building from scratch. The dual-model architecture, survival analysis framework, and prodromal cohort base all exist. Focus shifts to data extraction and merging.

---

## Reusable Assets from Phases 4-6

### Phase 5: Survival Analysis (✅ COMPLETE - C-index 0.86)
**Location:** `archive/development/phase5/`

- ✅ **Cox Proportional Hazards** (`task_5_3`) - Baseline C-index 0.86, time-varying C-index 0.997
- ✅ **DeepSurv Neural Survival** (`task_5_4`) - Complete implementation with `cox_partial_likelihood_loss()`
- ✅ **Prodromal Cohort Base** (`task_5_1`) - n=382 with phenoconversion events identified
- ✅ **Survival Data** - `data/prodromal_cohort/prodromal_survival_data.csv` (time-to-event format)
- ✅ **Trained Model** - `visualizations/phase4_5_results/phase5_results/deepsurv_model.pth`

**Reuse Strategy:** 
- DeepSurv architecture becomes the survival prediction head
- `cox_partial_likelihood_loss()` is the training loss function
- Prodromal survival DataFrame is the target format (enhance with genetic/imaging features)

### Phase 6: GAT Architecture (✅ COMPLETE - Trained Models Available)
**Location:** `archive/development/phase6/`

- ✅ **GIMANBackboneGAT** (`task_6_0_1_gat_upgrade.py`) - 3-layer GAT, 4 heads, dims [64, 128, 64]
- ✅ **Training Pipeline** (`train_giman_gat.py`) - Early stopping, learning rate scheduling
- ✅ **Graph Construction** (`prepare_prognostic_graph_data.py`) - K-NN similarity graphs
- ✅ **Trained Models** - `models/giman_gat_phase5/best_model.pth` (prodromal task)

**Reuse Strategy:**
- GIMANBackboneGAT is the shared encoder for both dual models
- Replace classification head with DeepSurv survival head
- Graph construction logic applies directly to Phase 8

### Phase 4: VAE & Clustering (✅ COMPLETE - For Later Use)
**Location:** `archive/development/phase4/`

- ✅ **VaDER Encoder** (`task_4_3_trajectory_clustering.py`) - BiLSTM+VAE, 16-dim latent
- ✅ **Discrete Subtypes** - 3 clusters (Fast/Moderate/Slow progressors)

**Reuse Strategy (Subphase 8.4):**
- Adapt VaDER architecture for 64-dim GAT embeddings (not trajectory sequences)
- Use reparameterization trick and loss function directly

### Existing Data Assets

**Location:** `data/01_processed/giman_enhanced_with_alpha_syn.csv`

- ✅ **Genetic Data:** LRRK2, GBA (85.6% completeness - 477/557 rows)
- ✅ **Alpha-Synuclein CSF:** 223 samples (~40% of cohort) with multiple columns
- ✅ **Olfactory:** UPSIT_TOTAL scores
- ✅ **Tau Biomarkers:** PTAU, TTAU columns
- ✅ **Clinical:** UPDRS Part III (NP3TOT), Hoehn & Yahr (NHY)
- ✅ **Demographics:** Age, sex, cohort definition

**Data Gaps (Need Extraction):**
- ❌ DAT-SPECT SBR values (quantitative striatal binding ratios)
- ❌ RBD scores (RBDSQ)
- ❌ SNCA genetic variants
- ❌ Genetic/imaging integration into Phase 5 prodromal cohort

---

## Tasks & Deliverables (REVISED)

### Task 1.1: Adapt Dual-Model Architecture from Phase 6
**Owner:** Lead Developer  
**Duration:** 1-2 days **[REVISED from 2 days]**  
**Dependencies:** None  
**Priority:** 🟡 MEDIUM (Code already exists, just needs adaptation)

#### Deliverables
- [ ] `giman_progression.py` - **ADAPT** GIMANBackboneGAT + DeepSurv head
- [ ] `giman_conversion.py` - **ADAPT** same architecture for prodromal cohort
- [ ] Architecture comparison document (minimal, mostly links to Phase 5/6 code)

#### Implementation Notes (REVISED)
```python
# giman_progression.py
"""
GIMAN-Progression: Adapted from Phase 6 GAT + Phase 5 DeepSurv.

**Reused Components:**
- Encoder: GIMANBackboneGAT from phase6/task_6_0_1_gat_upgrade.py
- Survival Head: DeepSurv from phase5/task_5_4_deepsurv_neural_survival.py
- Loss: cox_partial_likelihood_loss() from Phase 5

**Adaptation Work:**
1. Replace GIMANBackboneGAT's classification head with DeepSurv survival head
2. Change output_dim from num_classes to num_milestones (25 for Subphase 8.2)
3. Add survival prediction methods (predict_survival_curve, predict_risk_score)
"""

from phase6.task_6_0_1_gat_upgrade import GIMANBackboneGAT
from phase5.task_5_4_deepsurv_neural_survival import DeepSurv, cox_partial_likelihood_loss

class GIMANProgression(nn.Module):
    """Unified progression model: GAT encoder + survival head."""
    
    def __init__(self, input_dim=87, gat_hidden=[64, 128, 64], 
                 survival_hidden=[32, 16], num_milestones=25):
        super().__init__()
        
        # Shared GAT encoder (from Phase 6)
        self.encoder = GIMANBackboneGAT(
            input_dim=input_dim,
            hidden_dims=gat_hidden,
            num_heads=4,
            output_dim=gat_hidden[-1],  # 64-dim embeddings
            dropout=0.3
        )
        
        # Survival prediction head (from Phase 5 DeepSurv)
        self.survival_head = DeepSurv(
            input_dim=gat_hidden[-1],  # 64
            hidden_dims=survival_hidden,  # [32, 16]
            output_dim=num_milestones,  # 25 log-hazards
            dropout=0.3
        )
    
    def forward(self, x, edge_index, batch=None):
        # Encode via GAT
        embeddings, attn_weights = self.encoder(x, edge_index, return_embeddings=True)
        
        # Predict survival via DeepSurv head
        log_hazards = self.survival_head(embeddings)
        
        return log_hazards, embeddings, attn_weights
    
    def predict_survival(self, x, edge_index, time_points):
        """Predict survival curves S(t) = exp(-cumulative_hazard)."""
        with torch.no_grad():
            log_hazards, _, _ = self.forward(x, edge_index)
            hazards = torch.exp(log_hazards)
            
            # Compute survival function at time_points
            # (This uses Phase 5's survival prediction logic)
            from phase5.task_5_4_deepsurv_neural_survival import compute_survival_curve
            survival_curves = compute_survival_curve(hazards, time_points)
            
        return survival_curves


# giman_conversion.py  
"""
GIMAN-Conversion: Same architecture as GIMAN-Progression.

**Key Difference:** Trained on prodromal cohort instead of manifest PD.
- Input features: Baseline-heavy (limited longitudinal pre-diagnosis)
- Target: Time-to-phenoconversion (prodromal → PD diagnosis)
- Emphasis: Genetic risk, RBD, hyposmia, DaTSCAN abnormalities
"""

class GIMANConversion(GIMANProgression):
    """Identical architecture, different cohort/task."""
    
    def __init__(self, input_dim=87, gat_hidden=[64, 128, 64], 
                 survival_hidden=[32, 16], num_milestones=1):
        # Single milestone: phenoconversion event
        super().__init__(input_dim, gat_hidden, survival_hidden, num_milestones)
```

**Acceptance Criteria:**
- ✅ Both models import and reuse existing Phase 5/6 code
- ✅ Forward pass runs without errors on synthetic data
- ✅ Survival prediction methods work (S(t) curves generated)
- ✅ Documentation clearly references source code from Phases 5-6

---

### Task 1.2: Define Prodromal Inclusion Criteria
**Owner:** Clinical Collaborator + Lead Developer  
**Duration:** 3 days  
**Dependencies:** None

#### Deliverables
- [ ] `prodromal_inclusion_criteria.yaml` - Formal inclusion/exclusion rules
- [ ] Clinical justification document
- [ ] Expected cohort size estimates

#### Inclusion Criteria (PPMI Prodromal Cohort)

**Mandatory Base Criteria:**
- Enrolled in PPMI Prodromal cohort
- Age ≥30 years at baseline
- No clinical PD diagnosis at enrollment
- ≥1 year follow-up data available

**Risk Factor Criteria** (must meet ≥1):

1. **Genetic Risk** (High Priority)
   - Pathogenic *LRRK2* mutation (G2019S, R1441C/G/H, I2020T, Y1699C)
   - Pathogenic *GBA* mutation (N370S, L444P, E326K, T369M, IVS2+1)
   - Pathogenic *SNCA* variant (A53T, A30P, E46K, multiplications)

2. **REM Sleep Behavior Disorder** (High Priority)
   - RBD Screening Questionnaire (RBDSQ) positive (score ≥5)
   - OR polysomnography-confirmed RBD

3. **Olfactory Dysfunction** (Moderate Priority)
   - University of Pennsylvania Smell Identification Test (UPSIT) score <25th percentile for age/sex
   - OR self-reported severe hyposmia/anosmia

4. **Imaging Abnormality** (Emerging Priority)
   - DaTSCAN striatal binding ratio (SBR) <80% age-matched controls
   - OR MRI substantia nigra hyperechogenicity

**Exclusion Criteria:**
- Clinical PD diagnosis at any time (use for validation, not training)
- Atypical parkinsonism (MSA, PSP, CBD, DLB)
- Secondary parkinsonism (drug-induced, vascular, toxic)
- Dementia (MoCA <21) at baseline
- Other neurodegenerative disease (AD, ALS, HD)
- Insufficient data: >50% missing across any single modality

#### Expected Cohort Size
- PPMI Prodromal total: n=194
- After inclusion/exclusion: ~150-170 (estimate)
- After data quality filters (>85% complete): ~140-160 (target ≥150)

**Acceptance Criteria:**
- Criteria reviewed and approved by movement disorder specialist
- Alignment with PPMI prodromal cohort definition
- Clear justification for each criterion based on literature

---

### Task 1.3: Prodromal Data Extraction & Curation
**Owner:** Lead Developer  
**Duration:** 5 days  
**Dependencies:** Task 1.2 (criteria definition)

#### Deliverables
- [ ] `prodromal_cohort_curation.py` - Automated data extraction pipeline
- [ ] `prodromal_cohort_baseline.csv` - Baseline features for all qualifying participants
- [ ] `prodromal_cohort_longitudinal.csv` - Follow-up assessments
- [ ] `prodromal_data_dictionary.md` - Complete feature documentation

#### Data Extraction Strategy

**1. Genetic Data** (Priority: Critical)
```python
genetic_features = {
    'LRRK2_mutation': 'binary',
    'GBA_mutation': 'binary', 
    'SNCA_variant': 'binary',
    'APOE_e4_count': 'ordinal (0/1/2)',
    'MAPT_H1H1': 'binary',
    'PD_polygenic_risk_score': 'continuous',
    'AD_polygenic_risk_score': 'continuous',
    'genetic_risk_category': 'categorical (none/single/multiple)'
}
```

**2. Clinical Risk Factors** (Priority: Critical)
```python
clinical_risk_features = {
    # RBD
    'RBDSQ_total': 'continuous (0-13)',
    'RBD_positive': 'binary (≥5)',
    'PSG_RBD_confirmed': 'binary (if available)',
    
    # Olfaction
    'UPSIT_total': 'continuous (0-40)',
    'UPSIT_percentile': 'continuous (0-100)',
    'hyposmia': 'binary (<25th percentile)',
    
    # Motor
    'UPDRS_I_total': 'continuous',
    'UPDRS_II_total': 'continuous', 
    'UPDRS_III_total': 'continuous',
    
    # Cognitive
    'MoCA_total': 'continuous',
    'semantic_fluency': 'continuous',
    'HVLT_delayed_recall': 'continuous',
    
    # Other
    'constipation': 'binary',
    'orthostatic_hypotension': 'binary',
    'depression_BDI': 'continuous',
    'anxiety_STAI': 'continuous'
}
```

**3. Imaging Biomarkers** (Priority: High)
```python
imaging_features = {
    # DaTSCAN (most critical)
    'DaTSCAN_caudate_L': 'continuous',
    'DaTSCAN_caudate_R': 'continuous',
    'DaTSCAN_putamen_L': 'continuous',
    'DaTSCAN_putamen_R': 'continuous',
    'DaTSCAN_SBR_mean': 'continuous',
    'DaTSCAN_asymmetry': 'continuous',
    'DaTSCAN_abnormal': 'binary (<80% age-matched)',
    
    # MRI (FreeSurfer)
    'substantia_nigra_volume': 'continuous',
    'hippocampus_volume': 'continuous',
    'entorhinal_thickness': 'continuous',
    # ... + 30 cortical thickness measures
}
```

**4. CSF Biomarkers** (Priority: Medium - for SAA validation)
```python
csf_features = {
    'CSF_alpha_synuclein': 'continuous (pg/mL)',
    'CSF_Abeta42': 'continuous (pg/mL)',
    'CSF_total_tau': 'continuous (pg/mL)',
    'CSF_phospho_tau': 'continuous (pg/mL)',
    'CSF_SAA_status': 'binary (positive/negative)',  # KEY for Phase 8.3
    'CSF_SAA_RT50': 'continuous (hours)'  # Seed amplification kinetics
}
```

**5. Longitudinal Endpoints** (Priority: Critical)
```python
endpoints = {
    'phenoconversion_occurred': 'binary',
    'time_to_conversion': 'continuous (years)',
    'censoring_time': 'continuous (years)',
    'conversion_status': 'categorical (converted/prodromal/censored)',
    'final_UPDRS_III': 'continuous (at conversion or last visit)',
    'final_MoCA': 'continuous (at conversion or last visit)'
}
```

#### Quality Control Thresholds
- **Missing Data**: Exclude participants with >50% missing in any modality
- **Outliers**: Flag values >5 SD from mean for manual review
- **Follow-up**: Require ≥1 year minimum follow-up
- **Visit Frequency**: Prefer participants with ≥3 visits

**Acceptance Criteria:**
- Prodromal cohort: n≥150 with complete baseline data
- Data completeness: >85% across all modalities
- QC report generated with missing data patterns
- Feature distributions visualized and validated

---

### Task 1.4: Create Unified Configuration System
**Owner:** Lead Developer  
**Duration:** 2 days  
**Dependencies:** Task 1.1 (models defined)

#### Deliverables
- [ ] `dual_model_config.yaml` - Master configuration for both models
- [ ] `config_loader.py` - Python configuration management
- [ ] Configuration documentation

#### Configuration Structure
```yaml
# dual_model_config.yaml

project:
  name: "GIMAN Phase 8 - Model Frontier"
  version: "8.0.0"
  data_root: "data/"
  results_root: "results/"
  models_root: "models/"

cohorts:
  progression:
    name: "PPMI de novo PD"
    file: "data/longitudinal_cohort/giman_dataset_final.csv"
    n_subjects: 536
    primary_task: "disability_milestones"
    secondary_tasks: ["subtype_classification", "diagnostic"]
    
  conversion:
    name: "PPMI Prodromal"
    file: "data/prodromal_cohort/prodromal_baseline_complete.csv"
    n_subjects: 150  # target
    primary_task: "phenoconversion"
    secondary_tasks: ["risk_stratification", "saa_prediction"]

models:
  giman_progression:
    architecture: "GAT"
    n_layers: 3
    n_heads: 4
    hidden_dim: 64
    dropout: 0.2
    graph_k: 10
    output_type: "survival"  # NEW: replaces "classification"
    n_milestones: 25  # 25 disability endpoints
    
  giman_conversion:
    architecture: "GAT"
    n_layers: 3
    n_heads: 4
    hidden_dim: 64
    dropout: 0.3  # Higher dropout for smaller cohort
    graph_k: 8  # Smaller k for smaller cohort
    output_type: "survival"
    n_milestones: 1  # Single endpoint: phenoconversion

features:
  clinical:
    count: 32
    scaling: "standardize"
    imputation: "knn"
  imaging:
    count: 42
    scaling: "standardize"
    imputation: "knn"
  genetic:
    count: 8
    scaling: "none"  # Binary/categorical
    imputation: "mode"
  csf:
    count: 5
    scaling: "standardize"
    imputation: "knn"

training:
  optimizer: "Adam"
  learning_rate: 0.001
  weight_decay: 0.0001
  batch_size: 64
  max_epochs: 300
  early_stopping_patience: 50
  loss_function: "cox_partial_likelihood"  # NEW: survival loss
  
validation:
  strategy: "5-fold-cv"
  stratify_by: "event_status"  # Ensure balanced event rates
  test_size: 0.2
  random_seed: 42

explainability:
  methods: ["shap", "gnnexplainer", "gradcam", "integrated_gradients", 
            "counterfactuals", "attention_visualization"]
  consensus_threshold: 0.85
```

**Acceptance Criteria:**
- Configuration loads successfully in Python
- All parameters documented with inline comments
- Versioning system in place for tracking changes

---

### Task 1.5: Prodromal Cohort Characterization Report
**Owner:** Lead Developer + Clinical Collaborator  
**Duration:** 3 days  
**Dependencies:** Task 1.3 (data extracted)

#### Deliverables
- [ ] `prodromal_cohort_characterization.md` - Comprehensive descriptive report
- [ ] `prodromal_demographics_table1.csv` - Table 1 for publication
- [ ] `prodromal_visualizations.pdf` - Figure panel

#### Report Contents

**1. Demographics**
- Age, sex, education, race/ethnicity
- Family history of PD
- Comparison to de novo PD cohort (Table 1)

**2. Risk Factor Prevalence**
- Genetic: % LRRK2, GBA, SNCA, polygenic risk distribution
- Clinical: % RBD+, hyposmia, constipation, etc.
- Imaging: % with abnormal DaTSCAN, MRI findings
- Multiple risk factors: distribution histogram

**3. Baseline Clinical Characteristics**
- Motor: UPDRS-III distribution (should be minimal)
- Cognitive: MoCA distribution (should be normal)
- Non-motor: RBD, olfaction, autonomic scores
- Comparison to healthy controls (if available)

**4. Conversion Outcomes**
- Conversion rate: % phenoconverted
- Time to conversion: median, IQR, Kaplan-Meier curve
- Conversion by risk group: genetic vs. clinical vs. imaging
- Censoring analysis: reasons and patterns

**5. Data Quality**
- Missing data heatmap by feature and participant
- Completeness by modality
- Outlier detection results
- Follow-up duration distribution

**6. Comparison to Literature**
- How does this cohort compare to other prodromal studies?
- Benchmark conversion rate: literature reports 10-30% over 3-5 years
- Expected: ~24% (47/194) in full PPMI prodromal cohort

**Acceptance Criteria:**
- Report reviewed by clinical collaborator
- Cohort characteristics align with PPMI published descriptions
- Clear rationale for any exclusions
- Data quality meets >85% completeness threshold

---

## Success Metrics

### Quantitative
- [ ] Prodromal cohort: n≥150 participants
- [ ] Data completeness: >85% across all modalities
- [ ] Conversion rate: 20-30% (validate against PPMI reports)
- [ ] Follow-up: mean >3 years, median >2.5 years

### Qualitative
- [ ] Dual-model architecture approved by supervisor
- [ ] Configuration system flexible and well-documented
- [ ] Cohort characterization report publication-ready
- [ ] All code follows project style guidelines (PEP 8, docstrings)

---

## Timeline (Detailed Gantt Chart)

| Week | Days | Tasks | Owner | Deliverables |
|------|------|-------|-------|--------------|
| **Week 1** | Nov 4-8 | Task 1.1, 1.2 | Dev + Clinical | Models defined, Criteria approved |
| **Week 2** | Nov 11-15 | Task 1.3 (start), 1.4 | Dev | Data extraction pipeline, Config |
| **Week 3** | Nov 18-22 | Task 1.3 (finish), 1.5 | Dev + Clinical | Complete dataset, Draft report |
| **Week 4** | Nov 25 | Task 1.5 (finish), Review | All | Final report, Subphase review meeting |

---

## Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Prodromal cohort too small (n<120) | Medium | High | Relax inclusion criteria (e.g., accept 40% missing in CSF), consider augmentation with external data |
| Genetic data missing for many | Low | Medium | Prioritize RBD+hyposmia cases if genetic data limited |
| Conversion rate very low (<15%) | Low | Medium | Extend follow-up period, use intermediate endpoints (e.g., worsening UPDRS) |
| Configuration system too rigid | Low | Low | Build in modular overrides, extensive documentation |

---

## Next Steps (Subphase 8.2)

Upon completion of Subphase 8.1, immediately proceed to:
1. Define 25 disability milestone endpoints for GIMAN-Progression
2. Extract time-to-event data for both cohorts
3. Implement Cox baseline models for benchmarking
4. Develop survival analysis prediction head

---

## Approval Checklist

- [ ] Clinical collaborator approves inclusion criteria
- [ ] Supervisor approves dual-model architecture
- [ ] Data quality meets >85% threshold
- [ ] All deliverables complete and documented
- [ ] Code review passed
- [ ] Ready to proceed to Subphase 8.2

---

*Document Version: 1.0*  
*Last Updated: October 6, 2025*  
*Next Review: November 11, 2025*
