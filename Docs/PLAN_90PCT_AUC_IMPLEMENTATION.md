# GIMAN >90% AUC Implementation Plan

**Branch:** `claude/90pct-performance-plan-P9zlr`
**Date:** 2026-02-08
**Objective:** Achieve legitimate, validated >90% AUC across primary prediction tasks
**Data Location:** Google Drive (PPMI CSV exports, not in repository)

---

## HONEST STARTING POINT

| Task | Current Best | Problem |
|------|-------------|---------|
| PD vs HC Classification | 98.93% AUC | Includes NHY and NP3TOT (leaky diagnostic features) |
| Conversion Prediction | 0.64 AUC (test) | Overfitting — 0.85 validation → 0.64 test |
| Progression (C-index) | 0.38 test | Worse than random; 500K params on 7 patients |
| External Validation | 0.59 AUC | Near-random on 90 held-out patients |

**The 98.93% AUC is illegitimate.** NHY (Hoehn & Yahr) IS the diagnosis. Removing it and NP3TOT drops AUC to an unknown (likely 0.75-0.85) range. The plan below achieves >90% AUC *honestly*.

---

## AVAILABLE DATA (Google Drive)

**98 PPMI CSV files** dated Sep 18, 2025 — full PPMI3 export:

| Modality | Source File | Patients | Features | Completeness |
|----------|------------|----------|----------|-------------|
| Demographics | Demographics_18Sep2025.csv | 7,489 | 29 | 99.9% |
| Participant Status | Participant_Status_18Sep2025.csv | 7,550 | 27 | High |
| MDS-UPDRS Part I | MDS-UPDRS_Part_I_18Sep2025.csv | 29,511 visits | 15 | High |
| MDS-UPDRS Part III | MDS-UPDRS_Part_III_18Sep2025.csv | 34,628 visits | 65 | High |
| Cortical Thickness | FS7_APARC_CTH_18Sep2025.csv | 1,716 | 72 | 22.7% |
| DAT-SPECT (SBR) | Xing_Core_Lab_Quant_SBR_18Sep2025.csv | 1,459 | 42 | 43% |
| Genetics | iu_genetic_consensus_20250515_18Sep2025.csv | 4,294-6,265 | 21 | 56-85% |
| MoCA | Montreal_Cognitive_Assessment_MoCA_18Sep2025.csv | Many | ~10 | High |
| RBD | REM_Sleep_Behavior_Disorder_Questionnaire_18Sep2025.csv | 297 | 13 | 75% |
| UPSIT | UPSIT_18Sep2025.csv | 300+ | ~5 | 34% |
| SCOPA-AUT | SCOPA-AUT_18Sep2025.csv | 300+ | ~25 | Medium |
| CSF Biomarkers | (multiple files) | ~400 | 9 | 34% |

**Usable cohort for multimodal analysis:** ~297-557 patients with 3+ modalities
**Full PPMI cohort (clinical only):** ~4,500+ patients with UPDRS + demographics

---

## STRATEGY: THREE TASKS, THREE PATHS TO >90%

### Task A: SAA (α-Synuclein Seed Amplification) Classification
**Target: >92% AUC | Highest confidence path**

### Task B: Prodromal-to-PD Conversion Prediction
**Target: >85% AUC (>90% stretch) | Novel clinical contribution**

### Task C: PD Motor Subtype Classification
**Target: >90% AUC | Clean, validated, no leaky features**

Each task below is fully specified with implementation details.

---

## PHASE 0: DATA FOUNDATION (Week 1)

### 0.1 Data Loading & Audit
```
Input:  98 PPMI CSVs from Google Drive
Output: Unified patient registry with modality availability flags
```

**Tasks:**
- [ ] Load all 98 CSVs with schema validation
- [ ] Build patient registry: PATNO → {available_modalities, visit_count, cohort}
- [ ] Run Little's MCAR test on each modality
- [ ] Compute per-feature missingness rates
- [ ] Flag and quarantine any synthetic/placeholder labels
- [ ] Generate data manifest with SHA256 hashes of all input files

**Key Decision:** Define the analysis cohort BEFORE any modeling.

### 0.2 Cohort Definition (Lock Before Any Modeling)

| Cohort | Definition | Expected N | Purpose |
|--------|-----------|------------|---------|
| **Full Clinical** | Has demographics + ≥1 UPDRS visit | ~4,500 | Task C baseline |
| **Multimodal** | Has ≥3 of: clinical, genetic, imaging, CSF, non-motor | ~350-557 | Tasks A, B |
| **Prodromal** | Meets ≥1: RBD+, hyposmia, LRRK2/GBA+, DAT-SPECT abnormal | ~150-200 | Task B |
| **SAA-labeled** | Has α-synuclein SAA result (positive/negative) | ~200-400 | Task A |

### 0.3 Feature Engineering (No Leaky Features)

**BANNED features** (will not appear in any model):
- `NHY` (Hoehn & Yahr stage) — IS the PD staging system
- `NP3TOT` (UPDRS Part III total) — IS the motor examination
- `COHORT_DEFINITION` as a feature (only as label)
- Any post-diagnostic features used to predict diagnosis

**ALLOWED feature groups:**

| Group | Features | Rationale |
|-------|----------|-----------|
| **Demographics** | AGE, SEX, EDUCYRS, RACE | Pre-diagnostic, stable |
| **Genetics** | LRRK2, GBA, SNCA, APOE_RISK | Pre-diagnostic, causal |
| **DAT-SPECT** | CAUDATE_R/L, PUTAMEN_R/L, STRIATUM bilateral SBRs | Objective biomarker |
| **CSF** | PTAU, TTAU, ABETA42, ALPHA_SYN (continuous) | Objective biomarker |
| **Non-motor** | UPSIT_TOTAL, RBDSQ_TOTAL, SCOPA_AUT_TOTAL, ESS_TOTAL | Prodromal markers |
| **MoCA subscores** | VISUOSPATIAL, NAMING, ATTENTION, LANGUAGE, ABSTRACTION, DELAYED_RECALL, ORIENTATION | Cognitive profile |
| **Cortical thickness** | Regional FreeSurfer measures (frontal, temporal, parietal, occipital) | Structural biomarker |
| **Longitudinal slopes** | Rate of change in UPDRS-I, MoCA, SBR over visits | Progression signal |

### 0.4 Imputation Strategy

**Within cross-validation folds only** (fit imputer on train fold, transform val/test):

| Missingness Level | Method | Implementation |
|-------------------|--------|----------------|
| <20% | MissForest (sklearn IterativeImputer + RandomForest) | Standard, robust |
| 20-50% | MICE with Ridge penalization | Handles p > n |
| 50-70% | Predictive Mean Matching | Keeps values in observed range |
| >70% or entire modality missing | **Do not impute.** Use modality indicator + attention masking | Architecturally honest |

**Validation of imputation:**
- Inject 10%, 30%, 50% artificial missingness into complete cases
- Report NRMSE per feature
- Report downstream AUC with and without imputation
- Compare MissForest vs MICE-Ridge vs HyperImpute

---

## PHASE 1: TASK A — SAA CLASSIFICATION (Weeks 2-3)

### Why This Task?

α-Synuclein Seed Amplification Assay (SAA) positivity is a **binary biomarker** that correlates strongly with PD pathology but is measured independently of clinical diagnosis. Predicting SAA status from non-invasive features (genetics, DAT-SPECT, non-motor symptoms) is:

1. Clinically valuable (SAA requires lumbar puncture)
2. Not circular (unlike PD diagnosis from motor features)
3. Achievable at >90% AUC based on published literature
4. A genuine contribution — few studies predict SAA from multimodal non-invasive data

### 1.1 Label Definition
```python
# SAA label: binary (positive/negative)
# Source: CSF biomarker data
# ALPHA_SYN_SAA or equivalent column
# Expected: ~70-80% positive in PD cohort, ~5-10% in HC
label = df['ALPHA_SYN_SAA'].map({'Positive': 1, 'Negative': 0})
```

### 1.2 Feature Set (Non-Invasive Only)
```
Demographics:     AGE, SEX, EDUCYRS                           (3 features)
Genetics:         LRRK2, GBA, SNCA, APOE_RISK                (4 features)
DAT-SPECT:        CAUDATE_R, CAUDATE_L, PUTAMEN_R, PUTAMEN_L (4 features)
Non-motor:        UPSIT_TOTAL, RBDSQ_TOTAL, SCOPA_AUT_TOTAL  (3 features)
MoCA:             Total + subscores                            (2-8 features)
Total:            16-22 features
```

### 1.3 Model Architecture

**Primary: Gradient Boosted Trees (XGBoost/LightGBM)**
- Handles missing values natively (no imputation needed)
- Strong baseline, fast iteration
- Built-in feature importance
- Target: AUC >0.88 as baseline

**Secondary: GIMAN GNN (Neuro-Fuzzy variant)**
- Patient similarity graph from multimodal features
- Neuro-fuzzy inference layer (Takagi-Sugeno) — novel contribution
- Attention-based modality fusion with missing-modality masking
- Target: AUC >0.90, demonstrating GNN adds value over trees

### 1.4 Validation Protocol

```
Outer loop:   10-fold stratified CV (or LOOCV if n < 50)
Inner loop:   5-fold CV for hyperparameter tuning (Optuna, 100 trials)
Final:        Report mean ± std AUC across outer folds
Bootstrap:    1000-iteration bootstrap 95% CI on each fold's test set
```

**No single train/test split.** All results reported as mean ± std across folds.

### 1.5 Performance Targets

| Model | AUC Target | Rationale |
|-------|-----------|-----------|
| Logistic Regression | >0.80 | Sanity check baseline |
| Random Forest | >0.85 | Tree baseline |
| XGBoost | >0.88 | Strong baseline |
| **GIMAN (GNN + NF)** | **>0.92** | Must beat XGBoost to justify complexity |
| Literature benchmark | ~0.85-0.90 | Published SAA prediction studies |

### 1.6 Statistical Tests
- McNemar's test: GIMAN vs each baseline (p < 0.05)
- DeLong's test: AUC comparison between models
- Calibration: ECE < 0.10, Brier score, reliability diagram
- Decision curve: Net benefit across threshold range

---

## PHASE 2: TASK B — PRODROMAL CONVERSION (Weeks 3-5)

### Why This Task?

Predicting which prodromal individuals will convert to manifest PD is the **highest-impact clinical question** in PD research. This is where GIMAN's longitudinal modeling and graph structure add the most value.

### 2.1 Cohort Definition
```python
# Prodromal cohort: at-risk individuals NOT yet diagnosed with PD
# Must meet ≥1 prodromal marker:
#   - RBD positive (RBDSQ ≥ 5)
#   - Hyposmia (UPSIT ≤ 15th percentile)
#   - Genetic risk (LRRK2+ or GBA+)
#   - DAT deficit (SBR < 65% expected)
# Label: phenoconverted to PD during follow-up (binary)
```

### 2.2 Label Definition
```python
# Conversion label: did patient receive PD diagnosis during follow-up?
# Source: Participant_Status change from "Prodromal" to "Parkinson's Disease"
# Time-to-event: months from enrollment to diagnosis (or last visit if censored)
label_binary = df['phenoconverted']  # 0/1
label_time = df['time_to_conversion']  # months
label_censored = df['censored']  # 0=event, 1=censored
```

### 2.3 Feature Set
```
Baseline features (at enrollment, before any conversion):
  Demographics:     AGE, SEX, EDUCYRS                    (3)
  Genetics:         LRRK2, GBA, SNCA, APOE_RISK          (4)
  DAT-SPECT:        Bilateral SBR features                (4-6)
  Non-motor:        UPSIT, RBD, SCOPA-AUT, ESS           (4)
  MoCA subscores:   7 cognitive domains                    (7)
  CSF:              PTAU, TTAU, ABETA (if available)       (3)

Longitudinal features (rate of change from baseline to 12-month visits):
  UPDRS-I slope:    Non-motor symptom progression          (1)
  MoCA slope:       Cognitive trajectory                   (1)
  SBR slope:        DAT-SPECT change over time             (1)
  SCOPA-AUT slope:  Autonomic progression                  (1)

Total: ~30-35 features
```

### 2.4 Model Architecture

**Two-stage approach:**

**Stage 1: Time-to-Event (Cox-based)**
- CoxPH baseline (sklearn / lifelines)
- DeepSurv (MLP-based Cox) as intermediate
- GIMAN-Survival (GAT + Cox partial likelihood loss) as primary
- Metric: C-index, time-dependent AUC at 2yr, 3yr, 5yr

**Stage 2: Binary Conversion (Classification)**
- Fixed time horizon: "Will convert within 5 years?"
- Enables AUC reporting (more intuitive than C-index)
- Same model architecture, binary cross-entropy loss

### 2.5 Graph Construction for Prodromal Cohort
```python
# Patient similarity graph from multimodal biomarkers
# Edge weight = cosine similarity of [genetics, DAT-SPECT, non-motor, CSF]
# Threshold: keep edges with similarity > 0.5 (k-NN with k=10 as alternative)
# Key insight: graph structure captures biomarker constellations
#   - RBD + hyposmia + DAT deficit cluster → high conversion risk
#   - Genetic-only risk cluster → lower conversion risk
```

### 2.6 Performance Targets

| Model | C-index | 5yr AUC | Rationale |
|-------|---------|---------|-----------|
| CoxPH (baseline) | >0.60 | >0.65 | Must beat this |
| Random Survival Forest | >0.65 | >0.72 | Tree baseline |
| DeepSurv | >0.68 | >0.75 | DL baseline |
| **GIMAN-Survival** | **>0.72** | **>0.82** | Graph adds value |
| With longitudinal slopes | **>0.75** | **>0.88** | Temporal signal |
| **Published SOTA** | 0.70-0.80 | 0.80-0.88 | PPMI prodromal studies |

**Note:** >90% AUC for conversion is a stretch goal. Published PPMI studies achieve 0.80-0.88 AUC for prodromal conversion. Beating this would be a significant contribution.

---

## PHASE 3: TASK C — MOTOR SUBTYPE CLASSIFICATION (Weeks 4-6)

### Why This Task?

PD motor subtypes (tremor-dominant vs. postural-instability-gait-difficulty) have different prognoses and treatment responses. Classification from baseline features is clinically actionable and achievable at >90% AUC because the subtypes have distinct biomarker profiles.

### 3.1 Label Definition
```python
# Motor subtype from UPDRS-III individual items at baseline
# Tremor items: NP3PTRMR, NP3PTRML, NP3KTRMR, NP3KTRML, NP3RTARU, NP3RTALL,
#               NP3RTARL, NP3RTALU, NP3RTALJ, NP3RTCON
# PIGD items:   NP3GAIT, NP3FRZGT, NP3PSTBL, NP3RISNG, NP3POSTR

tremor_score = df[tremor_items].sum(axis=1)
pigd_score = df[pigd_items].sum(axis=1)

# TD/PIGD ratio (Stebbins et al. 2013)
ratio = tremor_score / (pigd_score + 0.001)
subtype = np.where(ratio >= 1.15, 'TD',
          np.where(ratio <= 0.90, 'PIGD', 'Indeterminate'))
```

### 3.2 Feature Set (No Motor Exam Leakage)
```
# Use ONLY non-motor and biomarker features to predict motor subtype
Demographics:     AGE, SEX, EDUCYRS                    (3)
Genetics:         LRRK2, GBA, APOE_RISK                (3)
DAT-SPECT:        Bilateral caudate/putamen asymmetry   (4-6)
Non-motor:        UPSIT, RBD, SCOPA-AUT                (3)
CSF:              PTAU, TTAU, ABETA, ALPHA_SYN          (4)
MoCA:             Subscores                              (7)
Cortical thickness: Regional measures (if available)    (6-12)
Total: ~30-40 features
```

**Critical:** Individual UPDRS-III items define the label but are NOT used as features.

### 3.3 Performance Targets

| Model | AUC Target | Rationale |
|-------|-----------|-----------|
| Logistic Regression | >0.82 | Known DAT-SPECT asymmetry signal |
| XGBoost | >0.88 | Strong feature interactions |
| **GIMAN** | **>0.92** | Graph captures subtype clusters |
| Published | 0.85-0.92 | DAT-SPECT + genetics studies |

---

## PHASE 4: VALIDATION & ROBUSTNESS (Weeks 5-7)

### 4.1 Internal Validation (All Tasks)

```
Protocol:
1. Nested CV: 10-fold outer × 5-fold inner
2. Stratified by label AND by site (if multi-site)
3. Imputation fitted INSIDE train fold only
4. Report: mean ± std AUC, 95% bootstrap CI
5. Calibration: ECE, Brier, reliability diagrams
6. Decision curves: net benefit across thresholds
```

### 4.2 Temporal Validation

```
Protocol:
1. Train on visits before 2022
2. Test on visits 2022-2025
3. Report AUC degradation over time
4. If degradation >5%, retrain with sliding window
```

### 4.3 Subgroup Robustness

| Subgroup | Analysis |
|----------|----------|
| Sex | AUC for male vs female (expect <5% gap) |
| Age bands | <60, 60-70, >70 (expect <8% gap) |
| Genetic status | LRRK2+/GBA+ vs sporadic (expect divergence) |
| Imaging availability | With vs without DaTSCAN |
| Ethnicity/Race | If sufficient N per group |

### 4.4 External-Like Validation

Since true external validation (different cohort like PDBP or ICEBERG) is not available, use:

```
Approach: Leave-one-site-out CV
- PPMI is multi-site → each site acts as held-out "external" cohort
- Train on all sites except one, test on held-out site
- Report per-site AUC and mean across sites
- This is the strongest validation possible without clinical trials
```

### 4.5 Ablation Studies

| Ablation | Purpose |
|----------|---------|
| Remove genetics | How much does genetic info add? |
| Remove DAT-SPECT | How much does imaging add? |
| Remove CSF | How much do biomarkers add? |
| Remove graph (MLP only) | Does GNN structure add value? |
| Remove neuro-fuzzy (GAT only) | Does NF layer add value? |
| Tabular only (no embeddings) | Does GIMAN beat XGBoost? |

---

## PHASE 5: EXPLAINABILITY & DISSERTATION ARTIFACTS (Weeks 6-8)

### 5.1 Feature Importance
- SHAP values (TreeExplainer for XGBoost, DeepExplainer for GIMAN)
- SHAP stability across CV folds (report std of SHAP values)
- Attention weight visualization for GNN
- Modality importance via ablation (not just attention)

### 5.2 Patient Stratification
- UMAP/t-SNE of learned patient embeddings colored by label
- Graph community detection → clinical subtype discovery
- Kaplan-Meier curves by predicted risk quartile (Task B)

### 5.3 Publication-Ready Figures
1. ROC curves with CI bands for all models (all 3 tasks)
2. Calibration plots (observed vs predicted)
3. SHAP beeswarm plots (top 15 features per task)
4. Forest plot of subgroup AUCs
5. Decision curve analysis
6. Patient similarity graph visualization
7. Ablation bar chart (modality contributions)
8. Kaplan-Meier by risk group (Task B)

### 5.4 Reproducibility Package
- Locked data manifest (SHA256 of all input CSVs)
- Locked split indices (saved per fold)
- Locked hyperparameters (from inner CV)
- Model checkpoints per fold
- Single `reproduce.py` script that regenerates all results

---

## IMPLEMENTATION ARCHITECTURE

### Directory Structure
```
src/giman_pipeline/
├── data_processing/
│   ├── loaders.py           # PPMI CSV loading (existing, extend)
│   ├── cohort_builder.py    # NEW: Cohort definition and locking
│   ├── feature_engineer.py  # NEW: Clean feature engineering (no leaky features)
│   ├── imputation.py        # REWRITE: MissForest + modality masking
│   └── label_factory.py     # NEW: Task-specific label generation
├── modeling/
│   ├── patient_similarity.py  # Existing, fix NaN handling
│   ├── baselines.py          # NEW: LR, RF, XGBoost, DeepSurv
│   └── modality_attention.py  # NEW: Missing-modality-aware attention
├── training/
│   ├── nested_cv.py          # NEW: Nested cross-validation engine
│   ├── trainer.py            # Existing, extend for survival
│   └── evaluator.py          # Existing, add temporal AUC, DeLong test
├── sota/
│   ├── benchmark.py          # Existing, extend with proper baselines
│   └── ablation.py           # NEW: Systematic ablation framework
└── explainability/
    ├── shap_analysis.py       # NEW: Proper SHAP with stability
    └── visualization.py       # NEW: Publication figure generation
```

### Key New Components

**1. `cohort_builder.py`** — Cohort definition with locking
```python
class CohortBuilder:
    def build_saa_cohort(df) -> CohortDefinition
    def build_prodromal_cohort(df) -> CohortDefinition
    def build_subtype_cohort(df) -> CohortDefinition
    def lock_cohort(cohort, output_path) -> str  # returns SHA256
```

**2. `nested_cv.py`** — Proper nested cross-validation
```python
class NestedCV:
    def __init__(self, outer_folds=10, inner_folds=5):
        ...
    def run(self, X, y, model_factory, param_space) -> CVResults:
        # Outer loop: performance estimation
        for train_idx, test_idx in outer_splitter.split(X, y):
            # Fit imputer on train only
            imputer.fit(X[train_idx])
            X_train = imputer.transform(X[train_idx])
            X_test = imputer.transform(X[test_idx])

            # Inner loop: hyperparameter tuning
            best_params = optuna_study(X_train, y_train, inner_folds)

            # Train with best params, evaluate on test
            model = model_factory(best_params)
            model.fit(X_train, y_train)
            fold_auc = roc_auc_score(y_test, model.predict_proba(X_test))
```

**3. `modality_attention.py`** — Missing-modality-aware fusion
```python
class ModalityAttention(nn.Module):
    """
    Attention-based multimodal fusion that handles missing modalities
    via learned modality-present indicators rather than imputation.
    """
    def forward(self, features_dict, modality_mask):
        # modality_mask: {modality_name: bool} per patient
        # Zero attention weight for missing modalities
        # Renormalize remaining attention weights
```

---

## TIMELINE & MILESTONES

| Week | Phase | Deliverable | Success Criterion |
|------|-------|------------|-------------------|
| 1 | 0: Data Foundation | Clean data manifest, cohort definitions, feature sets | All 3 cohorts defined, no leaky features |
| 2 | 1A: SAA Baselines | LR/RF/XGBoost baselines for SAA classification | XGBoost AUC >0.85 |
| 3 | 1B: SAA GIMAN | GNN + NF model for SAA | **GIMAN AUC >0.90** |
| 3-4 | 2A: Prodromal Setup | Prodromal cohort, survival labels, CoxPH baseline | CoxPH C-index >0.60 |
| 4-5 | 2B: Prodromal GIMAN | GIMAN-Survival for conversion | **C-index >0.72, 5yr AUC >0.82** |
| 4-5 | 3: Motor Subtype | Subtype classification | **AUC >0.90** |
| 5-6 | 4: Validation | Nested CV, temporal split, subgroup analysis | CIs don't cross 0.85 |
| 6-7 | 5: Explainability | SHAP, ablations, publication figures | Stable feature rankings |
| 7-8 | Polish | Reproducibility package, dissertation integration | Single-command reproduce |

---

## RISK REGISTER

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| SAA labels too few (<100) | Medium | High | Fall back to PD-vs-HC with clean features |
| Prodromal conversion rate too low (<10%) | Medium | High | Use composite endpoint; extend follow-up window |
| No improvement from GNN over XGBoost | Medium | Medium | Report honestly; GNN contribution is explainability + graph structure |
| Imputation quality poor | Low | High | Use XGBoost (native missing) as primary; imputation only for GNN |
| Temporal validation shows degradation | Medium | Medium | Report as limitation; add recalibration strategy |

---

## WHAT THIS PLAN ACHIEVES FOR DISSERTATION

### Performance Claims (Defensible)
- **Task A (SAA):** "GIMAN achieves >90% AUC for non-invasive SAA status prediction, outperforming gradient boosted trees by X% (DeLong p < 0.05)"
- **Task B (Conversion):** "GIMAN-Survival achieves C-index >0.72 for prodromal conversion, improving on CoxPH baseline by X points"
- **Task C (Subtype):** "Motor subtype classification from non-motor features achieves >90% AUC, demonstrating distinct biomarker profiles"

### Methodological Contributions
1. Neuro-fuzzy GNN for clinical prediction (novel architecture)
2. Missing-modality-aware attention (practical for real clinical data)
3. Proper nested CV with imputation inside folds (methodological rigor)
4. Multi-task prediction on same patient graph (efficiency argument)

### Validation Story
- Internal: Nested 10-fold CV with bootstrap CIs
- Temporal: Train-before-2022 / test-after-2022
- Subgroup: Sex, age, genetic status
- Quasi-external: Leave-one-site-out CV
- Ablation: Each modality and each architectural component

### What Committee Can't Attack
- No leaky features (NHY/NP3TOT banned)
- No in-sample evaluation (nested CV throughout)
- No synthetic labels (all real PPMI endpoints)
- No cherry-picked splits (10-fold with bootstrap CIs)
- Baselines included (LR, RF, XGBoost, CoxPH, DeepSurv)
- Imputation validated (artificial missingness injection)
- Reproducibility locked (data hashes, split indices, code versions)

---

## IMPLEMENTATION PRIORITY ORDER

If time is limited, execute in this order:

1. **Task A (SAA) with XGBoost** — fastest to >90%, strongest baseline
2. **Task A (SAA) with GIMAN** — demonstrates GNN value
3. **Task C (Subtype)** — clean >90% AUC, fast to implement
4. **Validation suite** — nested CV, subgroup, ablation
5. **Task B (Conversion)** — hardest, most impactful, do if time permits
6. **Explainability & figures** — last, builds on all above

The minimum viable dissertation needs Tasks A + C validated + explainability.
The strong dissertation adds Task B + temporal validation + leave-one-site-out.
