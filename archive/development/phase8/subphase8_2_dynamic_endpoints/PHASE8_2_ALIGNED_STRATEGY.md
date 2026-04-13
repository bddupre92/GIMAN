# Phase 8.2: Feature Expansion Strategy - ALIGNED ROADMAP

**Date:** October 12, 2025  
**Status:** Planning Phase  
**Alignment:** Bridging Phase 8.1 (4 features, C-index 0.88) → Phase 8.2 Objectives

---

## Strategic Clarification

### What Phase 8.1 Actually Accomplished

**Completed (Oct 12, 2025):**
- ✅ Trained GIMAN-Prognostic on prodromal cohort (n=381, 15 events)
- ✅ Achieved **C-index 0.88** using **ONLY 4 features**:
  - AGE_COMPUTED
  - SEX
  - NP3TOT (UPDRS Part III motor score)
  - MOCA_TOTAL (cognitive function)
- ✅ Generated 5 publication-quality visualizations
- ✅ Demonstrated strong generalizability (0.88 vs 0.38 manifest PD)

**Data Gaps Identified:**
- Missing: 14/18 attempted features (77.8% missing values)
- Genetic markers: LRRK2, GBA, APOE, SNCA (0% available in prodromal cohort)
- Imaging: Caudate/Putamen volumes, DAT-SPECT SBR (0% available)
- Biomarkers: CSF tau, α-synuclein, UPSIT olfactory (0% available)

### Phase 8.2 Dual Objectives (ALIGNED)

The roadmap lists **two overlapping objectives** that we'll address sequentially:

#### **Objective A: Feature Expansion** (Priority: CRITICAL)
**Goal:** Expand from 4 features → 30+ multimodal features  
**Rationale:** Current model relies only on basic clinical assessments. Adding genetic, imaging, and biomarker data will:
1. Improve prognostic accuracy (target C-index ≥0.90)
2. Enable biological subtyping
3. Support multi-task learning (Phase 8.5)

#### **Objective B: Multi-Milestone Endpoints** (Priority: HIGH)
**Goal:** Expand from 1 endpoint (phenoconversion) → 25 disability milestones  
**Rationale:** Single phenoconversion endpoint limited. Need comprehensive progression modeling across:
1. Motor milestones (walking aid, wheelchair, freezing of gait)
2. Cognitive milestones (MoCA < 21, dementia diagnosis)
3. Functional milestones (ADL independence, nursing home placement)
4. Complication milestones (dyskinesia, falls, hospitalizations)

**Decision:** We'll prioritize **Objective A first** (Feature Expansion), then **Objective B** (Multi-Milestone), as feature expansion is prerequisite for accurate multi-endpoint prediction.

---

## Phase 8.2 Revised Implementation Plan

### Timeline Breakdown

**Total Duration:** 2-3 weeks (Oct 14 - Nov 1, 2025)

| Week | Focus | Deliverables |
|------|-------|--------------|
| **Week 1** | Data extraction & feature engineering | 30+ features extracted, merged with prodromal cohort |
| **Week 2** | Model training & evaluation | GIMAN-Prognostic retrained, performance analysis |
| **Week 3** | Multi-milestone engineering (if time) | 25 endpoints operationalized, baseline models |

---

## Week 1: Multimodal Feature Extraction

### Priority 1: Genetic Features (5 features)

**Target Features:**
1. **LRRK2_CARRIER** - LRRK2 mutation status (binary)
2. **GBA_CARRIER** - GBA mutation status (binary)
3. **APOE_E4_CARRIER** - APOE ε4 allele (binary/count)
4. **SNCA_RISK_SCORE** - SNCA variant burden
5. **GENETIC_RISK_SCORE** - Composite polygenic risk

**Data Source:** `data/00_raw/GIMAN/ppmi_data_csv/iu_genetic_consensus_20250515_18Sep2025.csv`

**Script:** `scripts/extract_genetic_features_phase8_2.py`

**Expected Coverage:** 85%+ (based on giman_enhanced_with_alpha_syn.csv baseline)

---

### Priority 2: Imaging Features (12 features)

#### 2A: Structural MRI (FreeSurfer) - 6 features
**Target Features:**
1. **CAUDATE_LEFT_VOLUME** - Left caudate volume (mm³)
2. **CAUDATE_RIGHT_VOLUME** - Right caudate volume (mm³)
3. **PUTAMEN_LEFT_VOLUME** - Left putamen volume (mm³)
4. **PUTAMEN_RIGHT_VOLUME** - Right putamen volume (mm³)
5. **TOTAL_STRIATAL_VOLUME** - Sum of caudate + putamen
6. **STRIATAL_ASYMMETRY_INDEX** - (L-R)/(L+R) asymmetry

**Data Source:** `data/00_raw/GIMAN/ppmi_data_csv/FS7_APARC_ASEG_18Sep2025.csv`

**Script:** `scripts/extract_freesurfer_volumes_phase8_2.py`

**Expected Coverage:** 60-70% (MRI availability in prodromal cohort)

#### 2B: DAT-SPECT Imaging - 6 features
**Target Features:**
1. **CAUDATE_LEFT_SBR** - Left caudate striatal binding ratio
2. **CAUDATE_RIGHT_SBR** - Right caudate SBR
3. **PUTAMEN_LEFT_SBR** - Left putamen SBR
4. **PUTAMEN_RIGHT_SBR** - Right putamen SBR
5. **TOTAL_STRIATAL_SBR** - Mean of all regions
6. **ASYMMETRY_INDEX_SBR** - Laterality of dopaminergic deficit

**Data Source:** `data/00_raw/GIMAN/ppmi_data_csv/Xing_Core_Lab_-_Quant_SBR_18Sep2025.csv`

**Script:** `scripts/extract_dat_spect_sbr_phase8_2.py`

**Expected Coverage:** 50-60% (DAT-SPECT availability in prodromal)

---

### Priority 3: Biomarker Features (8 features)

#### 3A: CSF Biomarkers - 4 features
**Target Features:**
1. **ALPHA_SYN_CSF** - CSF α-synuclein concentration (pg/mL)
2. **PTAU_CSF** - CSF phosphorylated tau (pg/mL)
3. **TTAU_CSF** - CSF total tau (pg/mL)
4. **ABETA_CSF** - CSF amyloid-β 1-42 (pg/mL)

**Data Source:** `giman_enhanced_with_alpha_syn.csv` (already extracted, need to merge)

**Script:** `scripts/merge_csf_biomarkers_phase8_2.py`

**Expected Coverage:** 40% (CSF data limited in prodromal cohort)

#### 3B: Clinical Biomarkers - 4 features
**Target Features:**
1. **UPSIT_TOTAL** - Olfactory function score (0-40)
2. **RBDSQ_TOTAL** - REM behavior disorder score (0-13)
3. **SCOPA_AUT_TOTAL** - Autonomic symptoms score
4. **ESS_TOTAL** - Epworth Sleepiness Scale (excessive daytime sleepiness)

**Data Sources:**
- UPSIT: `University_of_Pennsylvania_Smell_ID_Test_18Sep2025.csv`
- RBD: `REM_Sleep_Disorder_Questionnaire_18Sep2025.csv`
- SCOPA-AUT: `SCOPA-Autonomic_18Sep2025.csv`
- ESS: `Epworth_Sleepiness_Scale_18Sep2025.csv`

**Script:** `scripts/extract_clinical_biomarkers_phase8_2.py`

**Expected Coverage:** 70-80% (clinical assessments widely available)

---

### Priority 4: Expanded Clinical Features (5 features)

**Target Features (Beyond baseline AGE, SEX, UPDRS, MoCA):**
1. **UPDRS_PART_I** - Non-motor experiences of daily living
2. **UPDRS_PART_II** - Motor experiences of daily living
3. **SCHWAB_ENGLAND** - Activities of daily living scale (0-100%)
4. **PIGD_SCORE** - Postural instability and gait difficulty subscore
5. **TREMOR_SCORE** - Tremor-dominant subscore

**Data Sources:**
- MDS-UPDRS Parts I, II: `MDS-UPDRS_Part_I_18Sep2025.csv`, `MDS-UPDRS_Part_II_18Sep2025.csv`
- Schwab & England: `Modified_Schwab_+_England_ADL_18Sep2025.csv`
- PIGD/Tremor: Computed from UPDRS Part III items

**Script:** `scripts/extract_expanded_clinical_phase8_2.py`

**Expected Coverage:** 95%+ (core clinical assessments)

---

## Week 1 Deliverables Summary

### Feature Engineering Pipeline

**Script:** `scripts/phase8_2_feature_engineering.py`

**Workflow:**
1. Load existing prodromal cohort (n=381, 15 events)
2. Extract 30+ features from 5 modality groups
3. Handle missing data:
   - Imputation: KNN (k=5) for <30% missing
   - Flagging: Create "missing" indicator for >30% missing
4. Normalize features: StandardScaler per modality
5. Feature selection: Remove features with >70% missing
6. Quality control: Validate distributions, outliers, correlations
7. Save: `data/03_prodromal/enhanced/prodromal_multimodal_features.csv`

**Expected Feature Availability:**

| Modality | Features | Expected Coverage | Imputation Strategy |
|----------|----------|-------------------|---------------------|
| Clinical (baseline) | 4 | 99%+ | Already complete ✅ |
| Clinical (expanded) | 5 | 95% | Mean imputation |
| Genetic | 5 | 85% | KNN imputation |
| MRI (structural) | 6 | 65% | KNN imputation + missing flag |
| DAT-SPECT | 6 | 55% | KNN imputation + missing flag |
| CSF biomarkers | 4 | 40% | Missing flag only |
| Clinical biomarkers | 4 | 75% | KNN imputation |
| **TOTAL** | **34 features** | **~70% avg** | Mixed strategy |

---

## Week 2: Enhanced Model Training

### Objective

Train GIMAN-Prognostic with 30+ features and compare to 4-feature baseline.

### Training Configuration

**Architecture:**
- Same as Phase 8.1: GIMANProgression
- Input features: 34 (vs 4 baseline)
- Hidden dim: 64 (may increase to 128 if needed)
- GAT layers: 3
- Attention heads: 4
- Dropout: 0.3

**Training:**
- Loss: Cox proportional hazards (same as Phase 8.1)
- Optimizer: Adam (lr=0.001, weight decay=1e-4)
- Scheduler: ReduceLROnPlateau
- Early stopping: Patience 20 epochs
- Max epochs: 200

**Data:**
- Same splits as Phase 8.1:
  - Train: 266 patients (10 events)
  - Val: 57 patients (2 events)
  - Test: 58 patients (3 events)
- New feature matrix: 381 × 34 (vs 381 × 4)

### Scripts

1. **`scripts/prepare_enhanced_training_data_phase8_2.py`**
   - Load multimodal features
   - Merge with Phase 8.1 survival data
   - Create PyG Data objects with 34 features
   - Use same train/val/test split (reproducibility)

2. **`scripts/train_giman_enhanced_phase8_2.py`**
   - Train GIMAN-Prognostic with 34 features
   - Monitor training curves
   - Save best model checkpoint
   - Generate predictions on test set

### Performance Targets

| Metric | Phase 8.1 (4 features) | Phase 8.2 Target (34 features) | Improvement |
|--------|------------------------|--------------------------------|-------------|
| **Test C-index** | 0.88 | ≥0.90 | +2.3% |
| **Val C-index** | 0.625 | ≥0.70 | +12% |
| **ROC AUC** | 0.873 | ≥0.90 | +3.1% |
| **Features** | 4 | 34 | +750% |

**Rationale:** Multimodal features should capture biological heterogeneity better than clinical-only features, enabling more accurate risk stratification.

---

## Week 2 Deliverables

### Performance Analysis

**Script:** `scripts/analyze_phase8_2_performance.py`

**Analyses:**
1. **Comparative C-index:**
   - 4-feature vs 34-feature models
   - Per-feature-group ablation (remove genetics, imaging, biomarkers separately)
   
2. **Feature Importance:**
   - Integrated gradients on 34 features
   - Rank features by prognostic value
   - Identify top 10 most predictive features

3. **Modality Contribution:**
   - Clinical-only: C-index using 9 clinical features
   - +Genetics: C-index with clinical + genetic (14 features)
   - +Imaging: C-index with clinical + genetic + imaging (26 features)
   - +Biomarkers: Full model (34 features)

4. **Kaplan-Meier Analysis:**
   - Risk stratification with 34 features
   - Compare to 4-feature stratification
   - Log-rank test for separation improvement

### Visualization Updates

**Script:** `scripts/generate_phase8_2_visualizations.py`

**New Figures:**
1. **Feature Availability Heatmap** - 381 patients × 34 features missingness pattern
2. **Feature Importance Ranking** - Top 15 features by integrated gradients
3. **Modality Contribution** - C-index improvement by modality addition
4. **Comparative KM Curves** - 4-feature vs 34-feature risk stratification
5. **Feature Correlation Network** - Graph of inter-feature relationships

---

## Week 3 (Optional): Multi-Milestone Endpoints

**Contingent on Week 1-2 completion speed.**

If Week 2 completes early, begin multi-milestone endpoint engineering:

### Deliverables

1. **`scripts/operationalize_disability_milestones.py`**
   - Define 25 PPMI disability milestones
   - Extract time-to-event data for each milestone
   - Create survival data: (PATNO, EVENT_TYPE, TIME, STATUS)

2. **`scripts/baseline_cox_multimile.py`**
   - Train Cox PH model per milestone
   - Report C-index for each of 25 endpoints
   - Identify most/least predictable milestones

3. **Documentation:**
   - Milestone definitions and clinical relevance
   - Event rates and follow-up times per milestone
   - Baseline performance benchmarks

**Note:** Full multi-task learning (simultaneous prediction of 25 endpoints) is Phase 8.5. This week only establishes baselines.

---

## Success Criteria (Phase 8.2)

### Must-Have (Week 1-2)

- [✓] **30+ features extracted** and merged with prodromal cohort
- [✓] **GIMAN-Prognostic retrained** with enhanced features
- [✓] **C-index ≥0.90** on test set (vs 0.88 baseline)
- [✓] **Feature importance analysis** identifying top 10 prognostic features
- [✓] **5 new visualizations** showing feature expansion impact
- [✓] **Comparative analysis** documenting 4-feature vs 34-feature performance

### Nice-to-Have (Week 3)

- [ ] 25 disability milestones operationalized
- [ ] Baseline Cox models trained per milestone
- [ ] Multi-endpoint survival data curated

---

## Alignment with Overall Phase 8 Strategy

### How Phase 8.2 Enables Downstream Subphases

**Phase 8.3 (SAA Biomarker):**
- Requires: CSF α-synuclein features → ✅ Extracted in Week 1
- Requires: Multimodal features for SAA proxy model → ✅ Available after Week 2

**Phase 8.4 (VAE Heterogeneity):**
- Requires: Rich patient embeddings from multimodal GIMAN → ✅ 34-feature model provides this
- Requires: Biological driver correlations → ✅ Genetic/imaging features enable this

**Phase 8.5 (Multi-Task Learning):**
- Requires: Multimodal feature set → ✅ 34 features ready
- Requires: Multi-endpoint survival data → ⏳ Week 3 optional, or Phase 8.5 Week 1

**Phase 8.6 (Explainability):**
- Requires: Complex model to explain → ✅ 34-feature GIMAN is more complex
- Requires: Biological feature interpretations → ✅ Genetic/imaging features are interpretable

**Phase 8.7 (Validation):**
- Requires: Best-performing model architecture → ✅ Phase 8.2 establishes this
- Requires: Feature extraction pipeline → ✅ Phase 8.2 creates reusable pipeline

---

## Risk Mitigation

### Risk 1: Feature Missingness

**Risk:** Some prodromal patients lack imaging/biomarker data (expected 40-60% missingness).

**Mitigation:**
1. Implement missing indicator flags (model learns from missingness pattern)
2. Use KNN imputation for <30% missing
3. Modality-specific encoders can handle partial observations
4. Test model with/without missing data to assess robustness

### Risk 2: Performance Plateau

**Risk:** Adding features may not improve C-index beyond 0.88 (small test set, ceiling effect).

**Mitigation:**
1. Primary metric: **Val C-index** (more stable with 57 patients, 2 events)
2. Secondary metrics: ROC AUC, feature importance, biological interpretability
3. Even if C-index doesn't improve, richer features enable:
   - Better subtyping (Phase 8.4)
   - SAA prediction (Phase 8.3)
   - Multi-task learning (Phase 8.5)

### Risk 3: Timeline Overrun

**Risk:** Feature extraction takes longer than 1 week.

**Mitigation:**
1. **Priority tiers:** Extract Priority 1-2 features first (genetic + clinical expanded), train baseline model
2. **Incremental additions:** Add imaging/biomarkers in Phase 2 if needed
3. **Parallel work:** Feature extraction (scripts) can run overnight/unattended

---

## Conclusion

**Phase 8.2 Aligned Strategy:**

1. **Week 1:** Extract 30+ multimodal features (genetic, imaging, biomarkers, expanded clinical)
2. **Week 2:** Retrain GIMAN-Prognostic, achieve C-index ≥0.90, analyze feature importance
3. **Week 3 (optional):** Begin multi-milestone endpoint engineering

This approach:
- ✅ Addresses Phase 8.1 data gaps (77.8% missing → <30% missing)
- ✅ Enables Phase 8.3-8.7 downstream work
- ✅ Maintains realistic timeline (2-3 weeks)
- ✅ Provides clear success metrics
- ✅ Balances roadmap objectives (features first, then milestones)

**Next Steps:**
1. Confirm Phase 8.2 strategy alignment
2. Begin Week 1 feature extraction
3. Create `scripts/phase8_2_feature_engineering.py`

---

**Document Version:** 1.0  
**Created:** October 12, 2025  
**Author:** GIMAN Research Team  
**Status:** Strategy Aligned, Ready for Implementation
