# Phase 8: GIMAN Model Frontier Development - Strategic Roadmap

**Project:** Graph-Informed Multimodal Attention Networks (GIMAN)  
**Phase:** 8 - Model Frontier & Framework Expansion  
**Timeline:** November 2025 - August 2026  
**Status:** Planning  
**Priority:** HIGH - Next-generation model development

---

## Executive Summary

Phase 8 represents a comprehensive revision and expansion of the GIMAN framework to align with PPMI roadmap priorities and address critical scientific frontiers. This phase transitions GIMAN from a proof-of-concept to a production-ready, clinically deployable framework spanning the full Parkinson's disease spectrum from prodromal risk through manifest disease progression.

### Key Strategic Objectives

1. **Dual-Model Architecture**: Separate specialized models for progression (manifest PD) and conversion (prodromal cohort)
2. **Dynamic Endpoints**: Replace static predictions with time-to-event survival analysis
3. **SAA Integration**: Incorporate alpha-synuclein Seed Amplification Assay as biological ground truth
4. **Continuous Heterogeneity**: Model PD heterogeneity as continuous landscape rather than discrete subtypes
5. **Multi-Task Learning**: Simultaneous prediction of multiple clinical endpoints
6. **Enhanced Explainability**: Adapt XAI framework to new dynamic models
7. **External Validation**: Validate on independent cohorts (PDBP)
8. **Open-Source Release**: Package framework for community adoption

---

## Code Reusability Assessment (Phase 4-6 → Phase 8)

**Assessment Date:** October 8, 2025  
**Foundation Status:** 70-75% of Phase 8 infrastructure already exists  
**Timeline Impact:** 6-8 weeks saved through code reuse

### Existing Assets Inventory

#### Phase 5: Survival Analysis Infrastructure (✅ COMPLETE)
| Asset | File | Status | Reusability |
|-------|------|--------|-------------|
| Cox Proportional Hazards | `task_5_3_cox_proportional_hazards.py` | ✅ Trained (C-index 0.86) | Direct reuse for multi-endpoint |
| DeepSurv Neural Survival | `task_5_4_deepsurv_neural_survival.py` | ✅ Trained | Use as survival head |
| Prodromal Cohort | `task_5_1_prodromal_cohort_identification.py` | ✅ n=382 identified | Enhance with genetics/imaging |
| Survival Data Engineering | Time-to-event DataFrames | ✅ Phenoconversion | Adapt for 25 milestones |
| Cox Partial Likelihood Loss | `cox_partial_likelihood_loss()` | ✅ Validated | Direct reuse |

#### Phase 6: GAT Architecture (✅ COMPLETE)
| Asset | File | Status | Reusability |
|-------|------|--------|-------------|
| GIMANBackboneGAT | `task_6_0_1_gat_upgrade.py` | ✅ Trained | Replace classifier → survival head |
| Multi-head Attention | 3 layers, 4 heads, dims [64,128,64] | ✅ Production | Direct reuse as shared encoder |
| Training Pipeline | `train_giman_gat.py` | ✅ Complete | Adapt for dual models |
| Graph Construction | `prepare_prognostic_graph_data.py` | ✅ K-NN (k=10) | Direct reuse |
| Model Weights | `models/giman_gat_phase5/best_model.pth` | ✅ Saved | Transfer learning |

#### Phase 4: VAE & Trajectory Clustering (✅ COMPLETE)
| Asset | File | Status | Reusability |
|-------|------|--------|-------------|
| VaDER Encoder | `task_4_3_trajectory_clustering.py` | ✅ BiLSTM+VAE | Adapt for GAT embeddings |
| Latent Space (16-dim) | Reparameterization trick | ✅ Validated | Adapt to 8-16 dims |
| Discrete Subtypes | 3 clusters (Fast/Moderate/Slow) | ✅ Characterized | Replace with continuous |

#### Phase 1: Data Extraction Framework (✅ COMPLETE)
| Asset | File | Status | Reusability |
|-------|------|--------|-------------|
| LongitudinalCohortExtractor | `task_1_2_longitudinal_cohort_extraction.py` | ✅ Complete | Template for new extractions |
| Demographics Loader | Wide format pivoting | ✅ Validated | Direct reuse |

### Existing Data Assets

| Data Type | File/Location | Completeness | Phase 8 Need |
|-----------|---------------|--------------|--------------|
| **Genetic (LRRK2, GBA)** | `giman_enhanced_with_alpha_syn.csv` | ✅ 85.6% (477/557) | Enhance with SNCA |
| **Alpha-Synuclein CSF** | Same file, ALPHA_SYN columns | ✅ 40% (223/557) | Direct use for SAA model |
| **Olfactory (UPSIT)** | Same file, UPSIT_TOTAL | ✅ Available | Direct use |
| **Tau Biomarkers** | Same file, PTAU, TTAU | ✅ Available | Direct use |
| **DATScan NIfTI Files** | `data/02_nifti/*.nii.gz` | ✅ Available | Extract SBR values |
| **Phase 5 Survival Data** | `prodromal_cohort/prodromal_survival_data.csv` | ✅ n=382 | Merge multimodal features |
| **Trained GAT Models** | `models/giman_gat_phase5/best_model.pth` | ✅ Saved | Transfer learning |

### Critical Data Gaps (Need Extraction)

| Data Type | Priority | Estimated Effort | Source |
|-----------|----------|------------------|--------|
| DAT-SPECT SBR Values | 🔴 CRITICAL | 2-3 days | PPMI DaTSCAN Quantification |
| RBD Scores (RBDSQ) | 🔴 CRITICAL | 1-2 days | PPMI REM_Behavior_Disorder.csv |
| SNCA Variants | 🟡 HIGH | 1 day | PPMI Genetic Consensus |
| 25 Disability Milestones | 🔴 CRITICAL | 3-4 days | PPMI UPDRS, ADL, MoCA, milestones tables |

### Revised Timeline Impact

| Subphase | Original Estimate | Revised Estimate | Time Saved | Reason |
|----------|-------------------|------------------|------------|--------|
| **8.1 Foundational** | 3-4 weeks | **1-2 weeks** | 2 weeks | Reuse Phase 5/6 infrastructure |
| **8.2 Dynamic Endpoints** | 3-4 weeks | **2 weeks** | 1-2 weeks | Adapt Phase 5 survival analysis |
| **8.3 SAA** | 4 weeks | **4 weeks** | 0 | Genuinely new (but CSF data exists) |
| **8.4 VAE** | 3 weeks | **1-2 weeks** | 1-2 weeks | Adapt Phase 4 VaDER |
| **8.5 Multi-Task** | 4 weeks | **4 weeks** | 0 | Core innovation (but uses Phase 6 encoder) |
| **8.6 Explainability** | 3 weeks | **2-3 weeks** | 0-1 weeks | Adapt Phase 7 XAI |
| **8.7 Validation** | 4 weeks | **4 weeks** | 0 | New validation work |
| **8.8 Dissemination** | 8-12 weeks | **8-12 weeks** | 0 | Writing/packaging |
| **TOTAL** | **32-38 weeks** | **26-33 weeks** | **6-8 weeks** | **~23% acceleration** |

---

## Phase 8 Subphase Breakdown

### Subphase 8.1: Foundational Integration & Prodromal Cohort Enhancement
**Duration:** 1-2 weeks (Nov 4-15, 2025) **[REVISED from 3-4 weeks]**  
**Priority:** CRITICAL  
**Goal:** Integrate existing Phase 4-6 infrastructure and enhance prodromal data

**Reusable Assets from Phases 4-6:**
- ✅ GIMANBackboneGAT (phase6/task_6_0_1_gat_upgrade.py) - Production GAT architecture
- ✅ DeepSurv + cox_partial_likelihood_loss (phase5/task_5_4) - Neural survival model
- ✅ Prodromal cohort base (phase5/task_5_1) - n=382 with phenoconversion data
- ✅ Genetic data (LRRK2, GBA) - 85.6% completeness in giman_enhanced_with_alpha_syn.csv
- ✅ Alpha-synuclein CSF data - 223 samples available
- ✅ Cox models trained - Baseline C-index 0.86 (exceeds 0.70 target)

#### Key Deliverables
- [ ] `giman_progression.py` - **ADAPT** Phase 6 GAT + Phase 5 survival head
- [ ] `giman_conversion.py` - **ADAPT** same architecture for prodromal cohort
- [ ] `prodromal_cohort_enhancement.py` - **MERGE** genetic/imaging into Phase 5 cohort
- [ ] `dual_model_config.yaml` - Unified configuration system
- [ ] Enhanced prodromal cohort (genetics, UPSIT, RBD, DaTSCAN SBR, CSF biomarkers)

#### Success Metrics
- Prodromal cohort enhanced: n≥150 with complete multimodal data (from Phase 5 base of n=382)
- Inclusion criteria: LRRK2/GBA carriers, RBD+, and/or olfactory loss, DaTSCAN abnormality
- Data completeness: >85% across all modalities (genetic: 85.6% ✅, need SBR + RBD)
- Documentation: Enhanced cohort characterization report
- Dual models implemented: GAT encoder + survival head (adapted from Phases 5-6)

#### Critical Data Extraction Tasks (Week 1 Priority)
- [ ] `extract_dat_spect_sbr.py` - Quantitative SBR values from DaTSCAN files
- [ ] `extract_rbd_data.py` - RBDSQ scores from PPMI clinical assessments
- [ ] `extract_snca_variants.py` - SNCA genetic variants (LRRK2/GBA already exist ✅)
- [ ] `merge_multimodal_prodromal.py` - Integrate all data sources into Phase 5 cohort

#### Technical Specifications
```python
# Prodromal Inclusion Criteria
inclusion_criteria = {
    'genetic_risk': ['LRRK2_mutation', 'GBA_mutation', 'SNCA_variant'],
    'clinical_risk': ['RBD_positive', 'UPSIT_score < 25th_percentile'],
    'imaging_risk': ['DaTSCAN_SBR < 80% age-matched controls'],
    'exclusion': ['existing_PD_diagnosis', 'atypical_parkinsonism', 'dementia']
}
```

---

### Subphase 8.2: Dynamic Endpoint Expansion (Multi-Milestone Survival)
**Duration:** 2 weeks (Nov 18 - Dec 2, 2025) **[REVISED from 3-4 weeks]**  
**Priority:** CRITICAL  
**Goal:** Expand Phase 5 single-endpoint survival to 25 disability milestones

**Reusable Assets from Phase 5:**
- ✅ Cox proportional hazards framework (task_5_3) - Baseline C-index 0.86
- ✅ DeepSurv architecture (task_5_4) - Neural survival model with cox_partial_likelihood_loss
- ✅ Time-to-event data engineering (task_5_1) - Phenoconversion survival data
- ✅ Survival visualization pipeline - KM curves, risk stratification

#### Key Deliverables
- [ ] `disability_milestones.py` - **OPERATIONALIZE** 25 PPMI milestones from UPDRS, ADL, MoCA
- [ ] `phenoconversion_endpoints.py` - **REUSE** Phase 5 definition (UPDRS-III ≥15)
- [ ] `survival_data_engineering.py` - **ADAPT** Phase 5 engineering for multiple endpoints
- [ ] `cox_baseline_multitask.py` - **EXTEND** Phase 5 Cox for 25 endpoints
- [ ] `deepsurv_integration.py` - **ADAPT** Phase 5 DeepSurv for multi-endpoint prediction

#### PPMI 25 Disability Milestones (Examples)
1. Requiring walking aid
2. Wheelchair dependence
3. MoCA score < 21 (cognitive impairment)
4. Loss of independence in ADLs
5. Nursing home placement
6. Need for caregiver assistance
7. Falls requiring medical attention
8. Freezing of gait episodes
9. Dyskinesia interfering with function
10. [... 15 additional milestones]

#### Time-to-Event Data Structure
```python
survival_data_schema = {
    'PATNO': int,
    'EVENT_TYPE': str,  # milestone identifier
    'TIME_TO_EVENT': float,  # years from baseline
    'EVENT_OCCURRED': bool,  # True if event observed, False if censored
    'CENSORING_TIME': float,  # last follow-up if censored
    'BASELINE_FEATURES': dict  # multimodal features at t=0
}
```

#### Success Metrics
- 25 milestone endpoints operationalized with >80% data completeness
- Phenoconversion endpoint: **REUSE** Phase 5 (conversion rate 20-30% validated ✅)
- Baseline Cox C-index per endpoint: >0.70 (Phase 5 achieved 0.86 as baseline)
- Multi-endpoint survival data: (time, event) × 25 per patient
- Median follow-up time: >3 years (Phase 5 data already meets this ✅)

---

### Subphase 8.3: SAA Biomarker Integration & Proxy Model
**Duration:** 4 weeks (Dec 2 - Dec 30, 2025)  
**Priority:** HIGH  
**Goal:** Train GIMAN to predict SAA status from non-invasive data

**Existing Assets:**
- ✅ Alpha-synuclein CSF data - 223 samples in giman_enhanced_with_alpha_syn.csv
- ✅ Phase 6 GAT architecture - Adaptable for SAA classification
- ✅ Multimodal feature set - Clinical, imaging, genetic, CSF biomarkers

#### Key Deliverables
- [ ] `giman_saa.py` - SAA prediction model (non-CSF → SAA status)
- [ ] `saa_data_curation.py` - Extract and align SAA results
- [ ] `saa_proxy_validation.py` - Validate SAA prediction performance
- [ ] `saa_feature_importance.py` - Identify SAA-predictive features
- [ ] SAA prediction explainability analysis

#### SAA Model Architecture
```python
class GIMAN_SAA(nn.Module):
    """
    Predict CSF SAA status from non-invasive modalities.
    
    Input: MRI (FreeSurfer), DTI, DaTSCAN, genetics, clinical
    Output: Binary SAA status (positive/negative synucleinopathy)
    
    Ground Truth: CSF SAA results (PMCA assay)
    """
    def __init__(self):
        self.feature_encoder = MultimodalEncoder(
            mri_dim=68, dti_dim=20, datscan_dim=6,
            genetic_dim=8, clinical_dim=32
        )
        self.graph_layers = GAT(layers=3, heads=4, hidden=64)
        self.saa_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)  # binary SAA classification
        )
```

#### Success Metrics
- SAA prediction AUC: >0.85 (surrogate for invasive CSF test)
- Sensitivity: >80% (minimize false negatives)
- Specificity: >75%
- SAA data availability: n≥200 across PD + prodromal cohorts

---

### Subphase 8.4: Continuous Heterogeneity Analysis (VAE on Embeddings)
**Duration:** 1-2 weeks (Dec 30, 2025 - Jan 13, 2026) **[REVISED from 3 weeks]**  
**Priority:** HIGH  
**Goal:** Model PD heterogeneity as continuous landscape

**Reusable Assets from Phase 4:**
- ✅ VaDER architecture (task_4_3_trajectory_clustering.py) - BiLSTM encoder + VAE
- ✅ 16-dim latent space implementation with reparameterization trick
- ✅ K-means clustering on embeddings (3 discrete subtypes: Fast/Moderate/Slow)
- ✅ Visualization pipeline for latent space

#### Key Deliverables
- [ ] `giman_vae_heterogeneity.py` - **ADAPT** VaDER encoder for 64-dim GAT embeddings (not trajectories)
- [ ] `embedding_extraction.py` - Extract GAT embeddings from trained GIMAN-Progression
- [ ] `latent_space_analysis.py` - Correlate latent axes with biological drivers
- [ ] `continuous_subtyping.py` - Replace discrete Phase 4 subtypes with continuous gradients
- [ ] Interactive latent space visualization dashboard

#### VAE Architecture Adaptation
```python
class HeterogeneityVAE(nn.Module):
    """
    Adapted from Phase 4 VaDER (task_4_3_trajectory_clustering.py).
    
    Key changes:
    - Input: 64-dim GAT embeddings (not LSTM trajectory features)
    - Latent: 8-16 dims (vs Phase 4's 16 dims)
    - Architecture: MLP encoder/decoder (not BiLSTM)
    """
    def __init__(self, embedding_dim=64, latent_dim=16, hidden_dims=[128, 64, 32]):
        super().__init__()
        # Encoder: GAT embedding → latent space
        encoder_layers = []
        in_dim = embedding_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space projections (reuse VaDER's reparameterization trick)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder: latent → reconstructed embedding
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], embedding_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick from VaDER."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, gat_embeddings):
        # Encode
        h = self.encoder(gat_embeddings)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Sample latent code
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstruction = self.decoder(z)
        
        return reconstruction, mu, logvar, z
```
    VAE trained on GIMAN patient embeddings to discover
    continuous heterogeneity structure.
    
    Input: 64-dim GIMAN embeddings (post-GAT)
    Latent: 8-16 dim continuous disease signature
    Output: Reconstructed embeddings
    """
    def __init__(self, input_dim=64, latent_dim=12):
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 24), nn.ReLU()
        )
        self.mu_layer = nn.Linear(24, latent_dim)
        self.logvar_layer = nn.Linear(24, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24), nn.ReLU(),
            nn.Linear(24, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )
```

#### Latent Space Interpretation Strategy
1. **Axis 1-2**: Correlate with genetic risk (GBA, LRRK2, polygenic risk score)
2. **Axis 3-4**: Correlate with motor vs. cognitive phenotype
3. **Axis 5-6**: Correlate with progression rate (fast vs. slow)
4. **Axis 7-8**: Correlate with imaging patterns (dopaminergic vs. cortical)

#### Success Metrics
- VAE reconstruction loss: <0.1 (embeddings preserved)
- Latent space axes correlate (|r| > 0.4) with known biological drivers
- Continuous signatures predict outcomes **better than Phase 4 discrete subtypes**
- Visualization: Clear gradients in latent space (not discrete Phase 4 clusters)

---

### Subphase 8.5: Architectural Refinement & Multi-Task Implementation
**Duration:** 4 weeks (Jan 13 - Feb 10, 2026)  
**Priority:** CRITICAL  
**Goal:** Unified architecture for simultaneous multi-task prediction

**Reusable Assets from Phase 6:**
- ✅ GIMANBackboneGAT - Shared encoder foundation (3-layer GAT, 4 heads, dims [64, 128, 64])
- ✅ Attention mechanisms with weight extraction
- ✅ Task-specific head concept (Phase 6 uses classification head)
- ✅ Training pipeline with early stopping, learning rate scheduling

#### Key Deliverables
- [ ] `giman_multitask.py` - **EXTEND** Phase 6 GAT with 4 parallel task heads
- [ ] `survival_head.py` - **REUSE** Phase 5 DeepSurv as survival prediction head
- [ ] `multitask_loss.py` - Composite loss with task weighting (NEW)
- [ ] `shared_encoder.py` - **ADAPT** GIMANBackboneGAT as shared encoder
- [ ] `task_balancing.py` - Dynamic task weight optimization (NEW)

#### Multi-Task GIMAN Architecture
```python
class GIMAN_MultiTask(nn.Module):
    """
    Unified GIMAN with multiple prediction heads.
    
    Shared Encoder: 3-layer GAT (64 dim, 4 heads)
    
    Task Heads:
    1. GIMAN-Progression: Time-to-disability milestones (25 endpoints)
    2. GIMAN-Conversion: Time-to-phenoconversion (1 endpoint)
    3. GIMAN-SAA: Binary SAA prediction
    4. GIMAN-Diagnostic: 3-class diagnosis (PD/Prodromal/Control)
    """
    def __init__(self):
        # Shared graph encoder
        self.shared_gat = GAT(layers=3, heads=4, hidden=64)
        
        # Task-specific heads
        self.progression_head = SurvivalHead(
            input_dim=64, n_milestones=25
        )
        self.conversion_head = SurvivalHead(
            input_dim=64, n_milestones=1
        )
        self.saa_head = nn.Linear(64, 2)
        self.diagnostic_head = nn.Linear(64, 3)
        
    def forward(self, x, edge_index, task='all'):
        # Shared encoding
        embeddings = self.shared_gat(x, edge_index)
        
        # Task-specific predictions
        outputs = {}
        if task in ['all', 'progression']:
            outputs['progression'] = self.progression_head(embeddings)
        if task in ['all', 'conversion']:
            outputs['conversion'] = self.conversion_head(embeddings)
        if task in ['all', 'saa']:
            outputs['saa'] = self.saa_head(embeddings)
        if task in ['all', 'diagnostic']:
            outputs['diagnostic'] = self.diagnostic_head(embeddings)
            
        return outputs
```

#### Composite Loss Function
```python
def multitask_loss(outputs, targets, weights):
    """
    L_total = w1*L_survival_prog + w2*L_survival_conv + 
              w3*L_cross_entropy_saa + w4*L_cross_entropy_diag
    """
    loss_prog = cox_partial_likelihood(
        outputs['progression'], targets['progression']
    )
    loss_conv = cox_partial_likelihood(
        outputs['conversion'], targets['conversion']
    )
    loss_saa = F.cross_entropy(
        outputs['saa'], targets['saa']
    )
    loss_diag = F.cross_entropy(
        outputs['diagnostic'], targets['diagnostic']
    )
    
    total_loss = (
        weights['progression'] * loss_prog +
        weights['conversion'] * loss_conv +
        weights['saa'] * loss_saa +
        weights['diagnostic'] * loss_diag
    )
    
    return total_loss, {
        'progression': loss_prog.item(),
        'conversion': loss_conv.item(),
        'saa': loss_saa.item(),
        'diagnostic': loss_diag.item()
    }
```

#### Success Metrics
- Multi-task model matches or exceeds single-task performance
- Task weights learned via gradient-based optimization (uncertainty weighting)
- Training stability: no task collapse or catastrophic forgetting
- Inference speed: <100ms per patient (batch size 32)

---

### Subphase 8.6: Multi-Scale Explainability for Dynamic Models
**Duration:** 3 weeks (Mar 10 - Mar 31, 2026)  
**Priority:** HIGH  
**Goal:** Adapt XAI framework to time-to-event predictions

#### Key Deliverables
- [ ] `shap_survival_analysis.py` - SHAP for time-dependent predictions
- [ ] `gnnexplainer_prodromal.py` - Subgraphs predicting conversion
- [ ] `gradcam_saa_mapping.py` - Brain region saliency for SAA
- [ ] `temporal_feature_importance.py` - How importance changes over time
- [ ] `counterfactual_survival.py` - "What if" for delaying events

#### Survival-Specific Explainability
```python
class SurvivalSHAP:
    """
    Adapt SHAP to explain time-to-event predictions.
    
    Instead of explaining a single prediction, explain:
    1. Predicted hazard at specific time t
    2. Predicted survival probability S(t)
    3. Relative risk vs. population
    """
    def explain_hazard_at_time(self, model, patient, time_point):
        """
        SHAP values for hazard function h(t) at specific time.
        Positive SHAP = feature increases hazard (accelerates event)
        Negative SHAP = feature decreases hazard (delays event)
        """
        pass
        
    def explain_survival_curve(self, model, patient):
        """
        SHAP values for entire survival curve S(t).
        Shows which features contribute to overall survival trajectory.
        """
        pass
```

#### Prodromal Conversion Subgraphs
- Identify patient neighborhoods predictive of rapid conversion
- Extract common patterns: genetic + imaging + clinical profiles
- Validate: Do identified subgraphs share known conversion risk factors?

#### Success Metrics
- Cross-method consensus (SHAP, GNNExplainer, GradCAM): >85% on top-10 features
- Temporal stability: Feature importance rankings stable across time horizons
- Clinical validity: Top features align with known PD biology
- Counterfactual realism: Suggested interventions are biologically plausible

---

### Subphase 8.7: External Validation & Comprehensive Benchmarking
**Duration:** 4 weeks (Mar 31 - Apr 28, 2026)  
**Priority:** CRITICAL  
**Goal:** Validate on PDBP and benchmark against SOTA

#### Key Deliverables
- [ ] `pdbp_data_integration.py` - Harmonize PDBP with PPMI
- [ ] `external_validation_pipeline.py` - Zero-shot PDBP validation
- [ ] `sota_benchmarking.py` - Compare vs. RSF, DeepSurv, Cox
- [ ] `improvement_quantification.py` - Static → Dynamic benefit analysis
- [ ] Validation report with calibration, discrimination, clinical utility

#### External Validation Strategy (PDBP)
```python
# NO retraining on PDBP - pure external validation
validation_protocol = {
    'data_source': 'PDBP (Parkinson's Disease Biomarkers Program)',
    'sample_size': 'n≥200 (target)',
    'preprocessing': 'Match PPMI pipeline exactly',
    'model': 'Use trained PPMI models without modification',
    'metrics': ['C-index', 'Brier score', 'calibration', 'NRI', 'IDI']
}

# Expected performance degradation: 5-10% C-index drop is acceptable
# If C-index drop >15%: investigate PDBP-PPMI differences
```

#### Benchmark Comparisons
| Model | Type | Tasks | Expected C-Index |
|-------|------|-------|------------------|
| **GIMAN-MultiTask** | GNN + Survival | All 4 | **0.78-0.82** |
| GIMAN-Static (Phase 4-6) | GNN + Classification | Diagnostic only | 0.75-0.79 |
| Random Survival Forest | Tree-based | Survival | 0.72-0.76 |
| DeepSurv | MLP + Survival | Survival | 0.74-0.78 |
| Cox PH | Linear | Survival | 0.68-0.72 |

#### Success Metrics
- PDBP validation C-index: >0.73 (within 5% of PPMI performance)
- Calibration: Observed vs. predicted event rates within 10%
- Net reclassification improvement (NRI): >10% vs. clinical model
- Integrated discrimination improvement (IDI): >5% vs. Cox

---

### Subphase 8.8: Synthesis, Dissemination & Open-Source Release
**Duration:** 8-12 weeks (Apr 28 - Jul 21, 2026)  
**Priority:** HIGH  
**Goal:** Package, publish, and release to community

#### Key Deliverables

**Publications (3 manuscripts)**
- [ ] **Paper 1**: "GIMAN-MultiTask: A Unified Graph Neural Network Framework for Parkinson's Disease Across the Clinical Spectrum" → *Nature Machine Intelligence* (target)
- [ ] **Paper 2**: "Predicting Alpha-Synuclein Pathology from Non-Invasive Biomarkers Using Graph-Informed Deep Learning" → *Movement Disorders* (target)
- [ ] **Paper 3**: "Continuous Heterogeneity Mapping in Parkinson's Disease: Beyond Discrete Subtypes" → *Brain* (target)

**Open-Source Releases**
- [ ] `giman_framework/` - Complete GIMAN codebase on GitHub
- [ ] `giman_pretrained_models/` - Trained model weights (HuggingFace)
- [ ] `giman_tutorials/` - Jupyter notebooks for common use cases
- [ ] `giman_documentation/` - Comprehensive API documentation
- [ ] Docker container for reproducible environment

**Community Engagement**
- [ ] GIMAN website with interactive demos
- [ ] Workshop at International Parkinson's Congress 2026
- [ ] Webinar series for PPMI investigators
- [ ] Kaggle competition: "Predict Parkinson's Progression with GIMAN"

#### Open-Source Package Structure
```
giman-framework/
├── README.md
├── LICENSE (MIT or Apache 2.0)
├── setup.py
├── requirements.txt
├── environment.yml
├── docs/
│   ├── api_reference.md
│   ├── quickstart.md
│   ├── advanced_usage.md
│   └── model_architecture.md
├── giman/
│   ├── __init__.py
│   ├── models/
│   │   ├── giman_multitask.py
│   │   ├── giman_progression.py
│   │   ├── giman_conversion.py
│   │   └── giman_saa.py
│   ├── data/
│   │   ├── preprocessing.py
│   │   ├── graph_construction.py
│   │   └── data_loaders.py
│   ├── explainability/
│   │   ├── shap_survival.py
│   │   ├── gnnexplainer.py
│   │   └── counterfactuals.py
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   └── config.py
│   └── pretrained/
│       ├── giman_ppmi_multitask.pth
│       ├── vae_heterogeneity.pth
│       └── model_metadata.json
├── tutorials/
│   ├── 01_quickstart.ipynb
│   ├── 02_custom_data.ipynb
│   ├── 03_explainability.ipynb
│   └── 04_advanced_customization.ipynb
├── tests/
│   ├── test_models.py
│   ├── test_data.py
│   └── test_explainability.py
└── examples/
    ├── train_from_scratch.py
    ├── fine_tune_pretrained.py
    └── generate_predictions.py
```

#### Future Research Directions (Framed in Paper 3)

**1. Causal Inference**
- Use learned graph structure as prior for causal discovery algorithms
- Apply PC algorithm, GES, or LiNGAM to identify causal relationships
- Test interventional predictions: "If we modify feature X, how does Y change?"

**2. Therapeutic Target Identification**
- Identify modifiable features in counterfactual analysis
- Rank features by: (1) causal impact, (2) modifiability, (3) intervention feasibility
- Propose drug targets for slowing progression in specific subgroups

**3. Personalized Medicine**
- Use VAE latent coordinates to match patients to optimal treatments
- Clinical trial enrichment: recruit patients with latent signatures matching responders
- Adaptive treatment strategies based on predicted trajectory

**4. Multi-Disease Extension**
- Apply framework to Alzheimer's, ALS, MSA
- Cross-disease graph: identify shared pathogenic mechanisms
- Transfer learning: leverage PD knowledge for rarer diseases

**5. Real-World Deployment**
- Integrate GIMAN into electronic health records (EHR)
- Clinical decision support system (CDSS) prototype
- FDA regulatory pathway as Software as a Medical Device (SaMD)

#### Success Metrics
- At least 2/3 manuscripts accepted in target journals
- GitHub stars: >500 within 6 months
- Downloads: >1000 installations within first year
- Community contributions: ≥5 external pull requests
- Citation impact: ≥20 citations within first year

---

## Integration with Existing Phases

### Relationship to Phase 1-7
```
Phase 1-3: Data Preprocessing & Initial GIMAN
    ↓ (Static diagnostic classification)
Phase 4-5: Longitudinal Subtyping & Prodromal Prediction
    ↓ (Discrete subtypes, static risk scores)
Phase 6: Explainability Framework
    ↓ (6-method XAI validation)
Phase 7: Manuscript Preparation
    ↓ (3 individual papers + 1 comprehensive manuscript)
════════════════════════════════════════════════════
Phase 8: Model Frontier (This Phase)
    ↓ (Dynamic survival analysis, continuous heterogeneity)
    → Production-ready framework
    → Open-source release
    → Clinical deployment pathway
```

### Key Innovations Beyond Phase 1-7
1. **Survival Analysis**: Replaces static classification with time-to-event modeling
2. **Prodromal Focus**: Dedicated model for pre-diagnosis cohort (Phase 5 expanded)
3. **SAA Integration**: Biological validation via synuclein seed amplification
4. **Continuous Heterogeneity**: VAE latent space replaces discrete subtypes (Phase 4 evolution)
5. **Multi-Task Learning**: Unified architecture for all prediction tasks
6. **External Validation**: PDBP validation proves generalizability
7. **Open Source**: Community-accessible framework (vs. internal research tool)

---

## Resource Requirements

### Personnel
- **Lead**: PhD student/postdoc (100% FTE, 10 months)
- **Supervisor**: Principal investigator (20% FTE, 10 months)
- **Collaborators**: 
  - Neurologist (clinical endpoint definition) - 10% FTE
  - Biostatistician (survival analysis) - 20% FTE
  - Software engineer (open-source packaging) - 30% FTE

### Computational
- **Training**: NVIDIA A100 GPU × 4 (or equivalent cloud compute)
- **Storage**: 500 GB for expanded datasets (SAA, PDBP)
- **Cloud costs**: ~$3,000 for model training + validation

### Data Access
- **PPMI**: Existing access (renewal if needed)
- **PDBP**: Apply for access (2-3 month approval timeline)
- **SAA data**: Coordinate with PPMI SAA working group

### Timeline Buffer
- Add 2-4 weeks contingency for data access delays
- Add 2-4 weeks for manuscript revisions (assume 1 round per paper)
- Total project duration: **10-11 months** (Nov 2025 - Aug/Sep 2026)

---

## Risk Mitigation

### Technical Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| SAA data insufficient (n<100) | Medium | High | Fall back to published SAA cohorts, external data |
| Multi-task training unstable | Medium | Medium | Use uncertainty weighting, task balancing algorithms |
| PDBP access denied | Low | Medium | Use alternative validation cohort (PPMI holdout) |
| VAE fails to discover structure | Low | Low | Revert to supervised dimensionality reduction (PCA, UMAP) |

### Scientific Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Survival models don't improve over static | Low | High | Extensive hyperparameter tuning, alternative architectures |
| Continuous heterogeneity rejected by reviewers | Low | Medium | Frame as complementary to subtypes, not replacement |
| External validation performance poor | Medium | High | Thorough domain adaptation, PDBP preprocessing alignment |

### Timeline Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Data curation delays | Medium | Medium | Start PDBP access application immediately (Nov 2025) |
| Manuscript review delays | High | Low | Submit papers to multiple journals simultaneously |
| Resource constraints | Low | Medium | Prioritize core deliverables, defer nice-to-have analyses |

---

## Success Criteria

### Scientific Success
- [ ] Multi-task C-index: >0.78 on PPMI, >0.73 on PDBP
- [ ] SAA prediction AUC: >0.85 (non-invasive proxy validated)
- [ ] VAE latent space: ≥3 axes correlate (|r|>0.4) with biology
- [ ] External validation: Performance degradation <10%
- [ ] Explainability: Cross-method consensus >85%

### Dissemination Success
- [ ] ≥2 papers accepted in high-impact journals (IF >10)
- [ ] Open-source package: >500 GitHub stars, >1000 downloads
- [ ] Community adoption: ≥3 independent research groups use GIMAN
- [ ] Workshop: >50 attendees at IPC 2026

### Clinical Translation Success
- [ ] Clinician feedback: >80% find tool potentially useful
- [ ] Clinical validation study initiated (prospective cohort)
- [ ] FDA pre-submission meeting completed (SaMD pathway)
- [ ] Licensing interest from industry partner (pharma or medtech)

---

## Alignment with PPMI Roadmap

This phase directly addresses **5 of 6 PPMI Strategic Priorities**:

1. ✅ **Prodromal Frontier**: Dedicated GIMAN-Conversion model (Subphase 8.1)
2. ✅ **Dynamic Endpoints**: Time-to-event survival analysis (Subphase 8.2)
3. ✅ **SAA Integration**: Biological validation (Subphase 8.3)
4. ✅ **Continuous Heterogeneity**: VAE latent space modeling (Subphase 8.4)
5. ✅ **External Validation**: PDBP generalizability testing (Subphase 8.7)
6. ⏭️ **Causal Inference**: Framed as future work (Paper 3 discussion)

### Novel Contributions Beyond PPMI Roadmap
- **Multi-Task Learning**: Simultaneous prediction across disease spectrum
- **Graph-Based Framework**: Patient similarity networks for information propagation
- **Comprehensive XAI**: 6-method validation for trustworthy AI
- **Open-Source Tools**: Community-accessible implementation

---

## Next Steps (Immediate Actions)

### Week 1 (Nov 4-8, 2025)
1. [ ] Secure PPMI data access renewal (if needed)
2. [ ] Submit PDBP data access application
3. [ ] Set up Phase 8 directory structure
4. [ ] Create `dual_model_config.yaml`
5. [ ] Begin prodromal cohort curation script

### Week 2 (Nov 11-15, 2025)
6. [ ] Complete prodromal inclusion/exclusion criteria
7. [ ] Extract prodromal baseline features
8. [ ] Create `giman_progression.py` and `giman_conversion.py` skeletons
9. [ ] Document Phase 8 Git branching strategy
10. [ ] Schedule weekly progress meetings

### Week 3-4 (Nov 18 - Dec 1, 2025)
11. [ ] Finalize prodromal cohort (n≥150)
12. [ ] Validate data quality (>85% completeness)
13. [ ] Generate cohort characterization report
14. [ ] Begin disability milestone extraction (Subphase 8.2)
15. [ ] Create comprehensive Phase 8 Gantt chart

---

## Conclusion

Phase 8 represents a transformative evolution of GIMAN from research prototype to production-ready clinical tool. By addressing the PPMI roadmap's strategic priorities—particularly the prodromal frontier, dynamic endpoints, and biological validation via SAA—this phase positions GIMAN as a **flagship application of AI in precision neurology**.

The planned open-source release ensures broad community impact, while the framing of causal inference as future work establishes a clear path from prediction to mechanistic understanding. Upon completion, GIMAN will be the **first comprehensive, interpretable, externally validated graph neural network framework spanning the full Parkinson's disease spectrum from at-risk individuals through advanced disability**.

**Timeline Summary**: 10 months (Nov 2025 - Aug 2026)  
**Budget Estimate**: $50K-$75K (personnel + compute + data access)  
**Expected Output**: 3 high-impact papers + open-source framework  
**Clinical Impact**: Direct pathway to FDA SaMD submission and clinical deployment

---

*Document Version: 1.0*  
*Last Updated: October 6, 2025*  
*Author: GIMAN Development Team*  
*Next Review: November 11, 2025*
