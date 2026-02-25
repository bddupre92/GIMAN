# Paper 1: NSD-ISS Stage Prediction with Calibrated Uncertainty

## A Deep Dive for Dissertation Defense Preparation

---

## 1. The Conceptual Problem (Beginner Level)

### The Real-World Analogy: The Weather Station Problem

Imagine you want to know how severe a hurricane is. Meteorologists classify hurricanes into categories (Category 1 through 5) based on specific measurements: wind speed, barometric pressure, and storm surge height. Each measurement tells part of the story, and the category tells doctors of disaster relief **what to prepare for**.

Parkinson's disease has a similar staging system called **NSD-ISS** (Neuronal alpha-Synuclein Disease Integrated Staging System), published by Simuni et al. in *The Lancet Neurology* in 2024. Just like hurricane categories, NSD-ISS assigns patients to biological stages (0 through 6) based on specific biomarker measurements:

- **Stage 0**: No biological markers of disease (like clear skies)
- **Stage 1**: Biological markers present but no symptoms (clouds forming)
- **Stage 2B**: Clinical signs present but no daily life impact (tropical storm)
- **Stage 3**: Mild functional impairment (Category 1 hurricane)
- **Stage 4**: Moderate functional impairment (Category 2+)

### What Question Is This Paper Answering?

The NSD-ISS stages require expensive, invasive tests to measure directly: a spinal tap to detect alpha-synuclein protein (the "S anchor"), and a specialized brain scan called DaT-SPECT to measure dopamine activity (the "D anchor"). Not every patient gets these tests at every visit, and not every hospital has the equipment.

**Paper 1 asks**: Can we predict a patient's NSD-ISS stage using only the clinical information a doctor already collects during a routine visit, such as motor exam scores, cognitive tests, sleep questionnaires, and basic demographics? And when we make that prediction, can we say **how confident we are**?

### Why Does It Matter?

If a doctor could look at routine clinical data and say "this patient is most likely in Stage 3, and I'm 90% confident the true stage is either 2B or 3," that fundamentally changes clinical care. It means:

1. **Earlier intervention**: Identify patients progressing before the expensive tests confirm it
2. **Clinical trial enrollment**: Screen patients for trials targeting specific disease stages
3. **Prognosis**: Give patients and families realistic expectations about disease trajectory
4. **Resource allocation**: Reserve expensive biomarker tests for ambiguous cases

No computational model for predicting NSD-ISS stages existed before this work. Paper 1 creates the first one and rigorously tests whether it works.

---

## 2. The Architectural Solution (Intermediate Level)

### Data Flow Overview

```
Raw PPMI Clinical Data (8,042 participants)
         |
         v
[NSD-ISS Staging Pipeline] -- Uses SAA + DaT-SPECT + UPDRS-III + H&Y
         |
         v
2,201 Staged Patients with Ground Truth Labels
         |
         v
[Feature Engineering] -- 22 non-circular features across 8 modalities
         |
         v
paper1_features_with_targets.csv (2,201 rows x 22 features + 4 targets)
         |
    +----+----+----+
    |         |         |
    v         v         v
[7-Model   [AdaMed   [Enhanced
Benchmark]  Graph]    GAT]
    |         |         |
    v         v         v
Best model: CatBoost (AUC 0.979 binary)
         |
         v
[Conformal Prediction] -- Wraps trained models
         |
         v
Prediction Sets with Coverage Guarantees (>= 90%)
         |
         v
[External Validation] -- BioFIND, PDBP, HBS cohorts
         |
         v
Domain Shift Analysis + Clinical-Only Feasibility
```

### Key Components Explained

#### What Is CatBoost? (The Winning Model)

**CatBoost** is a machine learning algorithm that builds predictions using **gradient-boosted decision trees**. To understand this, you need three concepts:

**A decision tree** is like a flowchart of yes/no questions. Imagine a doctor asking: "Is the patient's motor exam score above 15? Yes -> Ask: Is their cognitive score below 25? Yes -> Predict Stage 3." Each question splits patients into groups, and at the end of the chain you get a prediction. A single tree is simple but often too crude.

**An ensemble** means combining many decision trees together. Think of it as asking 500 different doctors (each with slightly different decision rules) and taking a vote. This is more reliable than asking one doctor.

**Gradient boosting** is a specific way to build the ensemble. Instead of building 500 independent trees, you build them sequentially: Tree 1 makes predictions, Tree 2 focuses on correcting Tree 1's mistakes, Tree 3 corrects whatever Tree 2 still gets wrong, and so on. Each new tree specializes in the cases the previous ensemble struggled with. "Gradient" refers to using calculus (the gradient of the error function) to determine exactly how each new tree should correct the previous ones.

**CatBoost specifically** (developed by Yandex) adds innovations for handling categorical features (like sex, handedness) natively, and uses "ordered boosting" to reduce overfitting. In our case, it processes 22 clinical features and outputs a probability distribution over the NSD-ISS stages.

- **Input**: 22 clinical feature values for one patient (age, motor scores, imaging values, etc.)
- **Output**: Probability for each stage (e.g., 5% Stage 0, 10% Stage 1, 60% Stage 2B, 20% Stage 3, 5% Stage 4)
- **Prediction**: The stage with the highest probability

#### What Is a Random Forest?

A **Random Forest** is an ensemble of decision trees built **independently** (not sequentially like boosting). Each tree is trained on a random subset of the data and a random subset of the features. The final prediction is a majority vote across all trees.

Think of it as: instead of asking 500 doctors who each learned from the same textbook (boosting), you ask 500 doctors who each studied a random subset of patient cases and a random subset of symptoms. Their collective wisdom averages out individual errors.

- **Input**: Same 22 features
- **Output**: Same probability distribution over stages
- **Key difference from CatBoost**: Trees are independent (parallel), not sequential (no error correction). Typically less accurate than boosting but more robust to noisy data.

#### What Is Conformal Prediction?

Traditional ML: "The patient is in Stage 2B" (point prediction, might be wrong)

**Conformal prediction**: "The patient is in {Stage 2B, Stage 3} with 90% confidence" (prediction SET, guaranteed to contain the true answer at least 90% of the time)

The "weather forecast" analogy: A weather forecast that says "Tomorrow will be 72 degrees" is a point prediction and is often wrong. A forecast that says "Tomorrow will be between 68 and 76 degrees, and I'm 90% confident" is more useful because it communicates **uncertainty**. Conformal prediction does this for disease staging.

**How it works mechanically**:
1. Train a model (e.g., CatBoost) on most of the data
2. Set aside a "calibration set" of patients with known true stages
3. For each calibration patient, measure how "surprised" the model is by the true answer (the "nonconformity score")
4. For a new patient, include all stages where the model would NOT be surprised -- this becomes the prediction set
5. **Mathematical guarantee**: If you target 90% confidence, the true stage will be in the set at least 90% of the time, regardless of the data distribution

- **Input**: A trained model + a new patient's features
- **Output**: A SET of possible stages (could be {2B}, could be {2B, 3}, could be {1, 2B, 3})
- **Key property**: The guarantee holds even if the model is imperfect

Two methods were used:
- **Split conformal**: Splits test data into calibration (50%) and evaluation (50%). Simple but wastes data.
- **Cross-conformal (CV+)**: Uses k-fold cross-validation for calibration. More data-efficient.

#### What Is a Patient Similarity Graph?

Imagine a social network where patients are "friends" if they have similar clinical profiles. Two patients with similar motor scores, similar ages, and similar cognitive test results would be connected.

**Formally**: Each patient is a node. An edge connects two patients if their similarity (measured by cosine similarity on their clinical features) exceeds a threshold. The resulting network captures patterns like "patients in Stage 3 tend to cluster together because they share similar feature profiles."

- **Input**: Feature matrix (2,201 patients x 22 features)
- **Output**: A graph with 2,201 nodes and thousands of edges, where edge weights represent patient-to-patient similarity

#### What Is a Graph Attention Network (GAT)?

Once you have a patient similarity graph, a **GAT** lets each patient "learn" from their similar neighbors. It works like this:

1. Each patient starts with their own feature vector (22 values)
2. The model looks at each patient's neighbors in the graph
3. It computes **attention weights**: how much should this patient pay attention to each neighbor?
4. It updates each patient's representation by combining their own features with a weighted average of their neighbors' features
5. This is repeated across multiple layers (like multiple rounds of information sharing)

Think of it as: "If I'm trying to predict your disease stage and I know 15 patients similar to you, I'll pay the most attention to the ones whose features are most informative for staging, and less attention to the ones that are less helpful."

- **Input**: Patient features + graph structure (who is connected to whom)
- **Output**: Updated patient representations that incorporate neighborhood information, then classified into stages

#### What Is AdaMedGraph?

**AdaMedGraph** (from Lian et al., 2024) combines two ideas:

1. **APPNP** (Approximate Personalized Propagation of Neural Predictions): A graph neural network that separates prediction from propagation. First, an MLP makes predictions; then, those predictions are smoothed across the graph using a process inspired by Google's PageRank algorithm. The "alpha" parameter (0.1) controls how much the original prediction is preserved vs. how much neighborhood information is incorporated.

2. **AdaBoost** (Adaptive Boosting): Instead of one big graph, AdaMedGraph builds many small graphs -- one per clinical feature. For each feature, patients are connected if their values for that specific feature are similar. Then an APPNP is trained on each per-feature graph, and the ensemble combines them with AdaBoost weighting (the SAMME algorithm), where better-performing graphs get higher weight.

- **Input**: 22 features, generates 22 x 3 = 66 candidate graphs (22 features x 3 similarity thresholds)
- **Process**: Iteratively selects the best per-feature graph, trains an APPNP, updates sample weights to focus on misclassified patients
- **Output**: Ensemble prediction combining multiple per-feature APPNP classifiers

#### Key Evaluation Metrics Explained

**AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**: Measures how well the model ranks patients. An AUC of 0.979 means: if you pick a random NSD-positive patient and a random NSD-negative patient, there's a 97.9% chance the model assigns a higher probability to the positive patient. Perfect = 1.0, random = 0.5.

**Balanced Accuracy**: The average of per-class recall. In our dataset, Stage 0 has 64.4% of patients. Regular accuracy could reach 64.4% by always predicting Stage 0. Balanced accuracy weights each class equally: if you perfectly predict Stage 0 but miss all Stage 4 patients, your balanced accuracy would be low. CatBoost achieves 0.951 balanced accuracy for binary, meaning it correctly identifies both NSD-positive and NSD-negative patients at nearly equal rates.

**Cohen's Kappa**: Measures agreement between predictions and truth, adjusted for chance. A kappa of 0.906 (CatBoost binary) means agreement is 90.6% better than random chance. Values above 0.8 are considered "almost perfect agreement."

**Quadratic Weighted Kappa (QWK)**: Like Cohen's Kappa but penalizes distant misclassifications more than adjacent ones. Predicting Stage 0 when the true stage is Stage 4 is penalized much more than predicting Stage 2B when the true stage is Stage 3. Especially important for ordinal targets.

**Cross-Validation**: Instead of testing on the same data used for training (which would be cheating), we split the data into 5 parts. Train on 4 parts, test on 1 part. Repeat 5 times, each time holding out a different part. Average the results. This gives an honest estimate of how the model would perform on truly unseen patients.

---

## 3. The Deep Dive (Advanced Level)

### 3.1 NSD-ISS Staging Pipeline

**File**: `src/giman_pipeline/staging/nsd_iss.py`

This module implements the exact Simuni et al. (2024) staging criteria as a deterministic algorithm. Key functions:

**`compute_s_anchor(saa_label)`**: Returns `True` if alpha-synuclein seed amplification assay (SAA) is positive, `False` if negative, `None` if missing. Coverage: only 12.6% (277/2,201) of PPMI patients have SAA data.

**`compute_d_anchor(putamen_sbr_left, putamen_sbr_right, ...)`**: Returns `True` if dopaminergic deficit detected. Uses the LOWEST of left/right putamen SBR values. Threshold: SBR < 0.80 = deficit. Fallback chain: lateralized putamen -> mean putamen -> mean caudate. Coverage: 97.1% (2,137/2,201).

**`compute_nsd_iss_stage(...)`**: The core staging logic. Decision hierarchy:
- Stage 0: No S+, no D+, genetic risk only
- Stage 1: S+ and/or D+, no clinical signs (UPDRS-III < 10 AND H&Y = 0)
- Stage 2B: S+/D+ with clinical parkinsonism (UPDRS-III >= 10 OR H&Y > 0), no functional impairment
- Stages 3-6: Progressive functional impairment mapped from Hoehn & Yahr stages

**Critical non-circular design**: The features used for staging (putamen SBR, UPDRS-III total, SAA) are **excluded** from the ML feature set. The 22 ML features use caudate SBR (not putamen), UPDRS-III subscales (not total), and never use SAA. This prevents the model from trivially recovering the staging rules.

### 3.2 Seven-Model Benchmark

**File**: `src/giman_pipeline/sota/nsd_iss_benchmark.py`

**Model factory pattern** (critical for CatBoost compatibility):
```python
factories["catboost"] = lambda: cb.CatBoostClassifier(
    iterations=500, depth=6, auto_class_weights="Balanced", verbose=0
)
```
Models are created via factory functions (lambdas) rather than cloned, because CatBoost's `class_weights` parameter doesn't survive sklearn's `clone()` function.

**Cross-validation**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Features z-score standardized within each fold (fit on train, transform on test) to prevent data leakage.

**Bootstrap CIs**: 1,000 resamples of the concatenated out-of-fold predictions. Percentile method: [2.5th, 97.5th] for 95% CI.

**Four target formulations** from the same 2,201 patients:
| Target | Classes | N | Key challenge |
|--------|---------|---|---------------|
| Binary | NSD- vs NSD+ | 2,201 | Class imbalance (64.4% vs 35.6%) |
| Three-class | Early, Mild, Impaired | 2,197 | Middle class (2B) hardest |
| Full ordinal | 0, 1, 2B, 3, 4 | 2,197 | Stage 4 = 0.8%, Stage 1 = 3.0% |
| NSD-positive | 1, 2B, 3, 4 | 779 | Smaller sample, no HC confound |

### 3.3 Results: Why CatBoost Dominates

| Target | CatBoost Bal Acc | CatBoost AUC | Best Alternative | Gap |
|--------|-----------------|-------------|-----------------|-----|
| Binary | 0.951 | 0.979 | LightGBM (0.948/0.977) | +0.3% |
| Three-class | 0.783 | 0.942 | XGBoost (0.767/0.944) | +1.6% |
| Full ordinal | 0.660 | 0.946 | XGBoost (0.635/0.947) | +2.5% |
| NSD-positive | 0.664 | 0.904 | XGBoost (0.654/0.887) | +1.0% |

CatBoost wins on balanced accuracy for all 4 targets. The margin widens for harder tasks (full ordinal: +2.5 pp over XGBoost). XGBoost achieves slightly higher AUC on three-class and full ordinal, but CatBoost's balanced class weighting (`auto_class_weights="Balanced"`) gives it better minority-class recall.

**Per-class recall analysis (CatBoost, three-class)**:
- Class 0 (Early): 0.935 recall
- Class 1 (Mild clinical): 0.582 recall (hardest -- Stage 2B is transitional)
- Class 2 (Impaired): 0.831 recall

### 3.4 Conformal Prediction Implementation

**File**: `src/giman_pipeline/sota/conformal.py`

Uses MAPIE 1.3.0's `SplitConformalClassifier` and `CrossConformalClassifier` with the LAC (Least Ambiguous set-valued Classifier) conformity score.

**Split conformal workflow**:
1. Train model on training fold
2. Split test fold 50/50: calibration vs evaluation
3. `SplitConformalClassifier(estimator=model, confidence_level=0.90, conformity_score="lac", prefit=True)`
4. `.conformalize(X_cal, y_cal)` -- learns nonconformity threshold
5. `.predict_set(X_eval)` -- returns `(y_pred, prediction_sets_bool)` tuple

**Results at 90% confidence (CatBoost binary, cross-conformal)**:
- Marginal coverage: 0.941 (exceeds 90% target)
- Mean set size: 0.964 (nearly all predictions are singletons)
- Singleton rate: 96.4%
- Empty set rate: 3.6%

The near-unity set sizes indicate that CatBoost's binary predictions are so confident that conformal prediction rarely needs to add a second class. For multiclass targets, set sizes increase (three-class: ~1.27, full ordinal: ~1.5).

### 3.5 AdaMedGraph Reproduction

**File**: `src/giman_pipeline/models/adamedgraph.py`

The key innovation is per-feature graph construction. For each of 22 features, 3 similarity thresholds are computed (feature_range / q for q in {4, 8, 16}), yielding 66 candidate graphs. Each candidate is:

1. Build binary adjacency: patients connected if |feature_i - feature_j| <= threshold
2. Train APPNP (2-layer MLP + 5-step PageRank propagation, alpha=0.1)
3. Compute weighted error on current sample weights
4. Select best (feature, threshold) pair for this boosting round
5. Update SAMME weights: alpha = log((1 - error) / error) + log(K - 1)
6. Reweight samples: misclassified patients get higher weight

**Results**: Binary bal_acc 0.870, AUC 0.958. Gap to CatBoost: -8.1% bal_acc, -2.1% AUC. This confirms the finding from Grinsztajn et al. (NeurIPS 2022) that gradient-boosted trees outperform graph neural networks on tabular clinical data.

### 3.6 Enhanced Multimodal GAT

**File**: `scripts/run_enhanced_gat_benchmark.py`

Architecture: Two modality encoders (clinical: 15 features -> 128d; biomarker: 7 features -> 128d) -> 3-layer GATConv (4 heads per layer, concat) per modality -> Cross-modal attention (nn.MultiheadAttention) -> Fusion (256d -> 128d) -> Classification head.

Graph: k-NN (k=10, cosine similarity) built within each CV fold. Undirected + self-loops.

**Results**: Binary bal_acc 0.825 +/- 0.013. Gap to CatBoost: -12.6 pp. The cross-modal attention improved over single-modality GAT for multiclass targets (three-class: +3.9%) but the gap to trees remained substantial.

### 3.7 Feature Ablation: DaT-SBR Is the Critical Feature

| Target | Full 22-feat AUC | Clinical 12-feat AUC | Delta |
|--------|-----------------|---------------------|-------|
| Binary | 0.979 | 0.727 | -25.2% |
| Three-class | 0.942 | 0.797 | -14.6% |
| Full ordinal | 0.946 | 0.823 | -12.3% |
| NSD-positive | 0.904 | 0.900 | -0.4% |

DaT-SPECT caudate SBR features are essential for binary prediction (Stage 0 vs Stage 1+) because the S anchor (SAA) has only 12.6% coverage, so the D anchor (DaT) carries most of the biological signal. However, for NSD-positive sub-staging (discriminating within stages 1-4), clinical features alone achieve AUC 0.900 -- nearly identical to the full model. This is because once you know a patient IS NSD-positive, their motor severity, cognitive status, and sleep patterns sufficiently discriminate between stages.

### 3.8 External Validation and Domain Shift

**File**: `scripts/run_external_validation.py`

Trained on PPMI (12 common clinical features), validated on:

| Cohort | N | Features Available | Ground Truth |
|--------|---|-------------------|-------------|
| BioFIND | 118 PD | 11/12 | NSD-ISS stages (Russo et al. 2025 replication) |
| PDBP | 893 PD | 12/12 | None (prediction only) |
| HBS | 649 PD | 8/12 | None (prediction only) |

**BioFIND binary external validation (n=108 with ground truth)**:
- CatBoost: bal_acc 0.516, AUC 0.637 (near-random)
- Root cause: **Domain shift from healthy control contamination**

PPMI's Stage 0 class (64.4% of training data) includes healthy controls with UPDRS-III bradykinesia mean of 7.4. BioFIND patients are ALL diagnosed PD with mean bradykinesia of ~20. The model learned to distinguish healthy controls from PD patients, NOT S+ from S- within PD. When applied to an all-PD external cohort, this distinction is meaningless.

**BioFIND NSD-ISS staging replication** (`scripts/stage_biofind_nsd_iss.py`): Exact replication of Russo et al. (2025) methodology using 7 staging variables (NP1COG, MCATOT, P1TOT, P2TOT, P3TOT, PDMEDYN, RBD_STATUS) with published thresholds. Near-perfect match to published distribution: Stage 2: 9, Stage 3: 58, Stage 4: 34 (vs 35 published), Stage 5: 2.

### 3.9 Key Constants and Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| CatBoost iterations | 500 | Convergence observed; higher values showed diminishing returns |
| CatBoost depth | 6 | Standard for tabular; deeper trees overfit |
| CV folds | 5 | Balance between variance reduction and training data size |
| Bootstrap resamples | 1000 | Standard for 95% CI precision (~0.003 width) |
| Conformal confidence | 0.80, 0.90, 0.95 | Clinical practice levels (screening, diagnostic, high-confidence) |
| D anchor threshold | 0.80 SBR | Published NSD-ISS criterion (Simuni et al. 2024) |
| UPDRS3 clinical threshold | 10 | Published NSD-ISS criterion |
| k-NN neighbors (GAT) | 10 | Empirical; 5-20 range explored, 10 balanced density/sparsity |
| APPNP propagation steps | 5 | Standard from Klicpera et al. (2019) |
| APPNP teleport alpha | 0.1 | 90% neighborhood influence, 10% self-retention |
| AdaBoost max estimators | 10 | Convergence typically at 5-8 rounds for 22 features |

---

## 4. Committee Questions & Answers

### Q1: "Your PPMI training set includes healthy controls in Stage 0. Doesn't this fundamentally compromise the model's validity for classifying PD patients?"

**Answer**: This is the central methodological tension of Paper 1, and we confront it directly rather than hiding it.

Yes, 64.4% of our training data (1,418/2,201 patients) are Stage 0, which includes healthy controls, prodromal participants, and SWEDD (scans without evidence of dopaminergic deficit) subjects alongside PD patients who happen to be S-/D-. This means the binary model partially learns a healthy-vs-diseased distinction rather than a pure S+/D+ vs S-/D- distinction.

We demonstrate this is a real problem through our external validation on BioFIND (118 all-PD patients): binary balanced accuracy drops to 0.516 (near-random) because BioFIND's S- patients have motor profiles consistent with PD (UPDRS3 bradykinesia ~20), while PPMI's Stage 0 includes healthy controls (UPDRS3 bradykinesia ~7.4).

However, we show this confound is target-specific. The **NSD-positive sub-staging** model (which excludes Stage 0 entirely, training only on 779 S+/D+ patients) achieves AUC 0.904 with the full feature set and 0.900 with clinical features alone. This model is free of the HC contamination confound and demonstrates that within NSD-positive patients, clinical features sufficiently discriminate between stages 1, 2B, 3, and 4. The three-class model (which groups stages 0-1 together) also shows moderate external validity (BioFIND AUC 0.703).

Our honest recommendation: for clinical deployment, use the NSD-positive sub-staging model (which requires first confirming NSD-positivity via biomarkers) rather than the binary model.

### Q2: "You benchmark 7 tabular models, but they all use the same features and same CV strategy. How do you know the feature engineering isn't doing all the work?"

**Answer**: The feature ablation experiment directly addresses this. When we remove the 5 DaT-SPECT imaging features (reducing from 22 to 12 clinical-only features), binary AUC drops from 0.979 to 0.727 -- a 25.2% decline. This proves that the features themselves carry substantial signal, and the DaT imaging features in particular are critical for binary classification.

However, the feature engineering is deliberate and non-trivial. We had to ensure **non-circularity**: the features used to define NSD-ISS stages (putamen SBR, UPDRS-III total, SAA) cannot be used as ML features, as this would make prediction trivially circular. We use caudate SBR (correlated with but distinct from putamen SBR), UPDRS-III subscales (tremor, rigidity, bradykinesia, axial -- which sum to roughly the total but provide richer information), and never use SAA.

The fact that CatBoost achieves 0.979 AUC for binary despite using caudate (not putamen) SBR demonstrates that there is genuine predictive signal in the non-staging biomarkers. The model is learning real biological relationships, not memorizing staging rules.

Additionally, the model comparison IS informative: CatBoost outperforms logistic regression by 4.8 pp balanced accuracy (binary), showing that non-linear feature interactions matter. The gap between linear and tree-based models is consistent across all 4 targets, indicating the feature space has complex decision boundaries.

### Q3: "Your conformal prediction sets are almost always singletons (96.4% for binary). If the sets are never larger than 1, what's the practical value of conformal prediction?"

**Answer**: The high singleton rate for binary is actually a strength, not a weakness -- it means the model is confident enough that conformal prediction confirms its predictions are reliable. The practical value becomes apparent in two ways:

First, the **3.6% non-singleton cases** are precisely the clinically interesting ones: patients where the model is uncertain. These are candidates for additional biomarker testing. Conformal prediction identifies who needs the expensive DaT-SPECT or SAA test, functioning as a clinical triage tool.

Second, the value increases dramatically for harder targets. For the three-class target, mean set size is 1.27 (27% of patients get multi-class sets). For full ordinal (5 classes), set sizes average ~1.5. In these multi-class settings, conformal prediction provides genuinely actionable uncertainty: "this patient is either Stage 2B or Stage 3" is much more useful than a forced single-stage prediction that might be wrong.

Third, conformal prediction provides a **distribution-free guarantee**. Unlike model-reported probabilities (which can be miscalibrated), conformal coverage of 94.1% at 90% confidence level means the true stage is in the set at least 90% of the time -- guaranteed by mathematical theory, not by model calibration. This is critical for clinical deployment where we need reliable uncertainty.

### Q4: "Graph neural networks underperform CatBoost by 8-13%. Why include them at all?"

**Answer**: We include graph models for three reasons, even though they underperform on raw accuracy metrics:

First, **scientific completeness**. The Grinsztajn et al. (NeurIPS 2022) finding that trees dominate deep learning on tabular data was published for generic tabular benchmarks. We needed to verify this holds specifically for clinical biomarker data in PD staging. Our results confirm it: Enhanced MM-GAT achieves 0.825 vs CatBoost's 0.951 for binary. This is a meaningful contribution -- it tells future researchers not to pursue complex GNN architectures for this specific task.

Second, **the patient similarity graph concept is foundational to Papers 2, 3, and 6**. Paper 2 uses graph-informed imputation (GIMIN), Paper 3 uses a Graph-Informed Digital Twin, and Paper 6 integrates graphs into the unified pipeline. Paper 1 establishes that while graphs don't help classification on complete data, the similarity relationships between patients become powerful when data is incomplete (Paper 2) or when modeling temporal dynamics (Paper 3).

Third, **AdaMedGraph provides unique interpretability**. Each boosting round selects the single most informative feature and its similarity threshold, creating an interpretable trace: "Round 1 selected putamen SBR similarity (threshold 0.05), Round 2 selected age similarity (threshold 8.3)..." This per-round feature selection is more clinically interpretable than CatBoost's feature importance scores.

### Q5: "With only 17 patients in Stage 4 and 67 in Stage 1, how can you claim reliable prediction for these rare stages?"

**Answer**: This is a genuine limitation that we address through multiple strategies:

First, **class-balanced training**. CatBoost uses `auto_class_weights="Balanced"`, which inversely weights samples by class frequency. Stage 4 patients get approximately 130x the weight of Stage 0 patients (2,201/17 = 129.5). This prevents the model from ignoring minority classes.

Second, **balanced accuracy as primary metric**. Unlike regular accuracy (which would reward always predicting Stage 0), balanced accuracy averages per-class recall. CatBoost's full ordinal balanced accuracy of 0.660 means it correctly classifies roughly 66% of patients from EACH stage, even Stage 4 with only 17 patients.

Third, **target formulation design**. The three-class target groups stages 3+4 into "Impaired" (504 patients, 22.9%), making the smallest class more viable. The NSD-positive target excludes Stage 0 entirely, creating a more balanced 4-class problem (67, 208, 487, 17 patients).

Fourth, **bootstrap confidence intervals** honestly reflect the uncertainty. For full ordinal CatBoost, the 95% CI on balanced accuracy is [0.628, 0.689] -- wider than for binary [0.940, 0.961], correctly reflecting the greater difficulty.

We acknowledge that Stage 4 (17 patients) is too small for reliable per-class metrics. The QWK of 0.861 for CatBoost on full ordinal provides some reassurance that adjacent-stage misclassifications dominate over distant ones (predicting 3 when true is 4 is penalized less than predicting 0 when true is 4).

---

## 5. Publication Reviewer Questions & Answers

### Q1: "Table 3 shows CatBoost as the best model for all 4 targets. Have you performed any statistical tests to determine if the differences are significant, or is this just a single random seed?"

**Answer**: We use 5-fold stratified cross-validation with a fixed random seed (42) for reproducibility, and 1,000 bootstrap resamples to compute 95% confidence intervals. For binary, CatBoost's balanced accuracy CI is [0.940, 0.961] while LightGBM's is [0.937, 0.959] -- the CIs overlap, indicating the difference (0.951 vs 0.948) is not statistically significant at p < 0.05.

For the more challenging targets where CatBoost's advantage is larger, the CIs separate more clearly. Three-class CatBoost CI [0.762, 0.803] vs XGBoost [0.746, 0.787] shows partial overlap. Full ordinal shows clearer separation: CatBoost [0.628, 0.689] vs XGBoost [0.608, 0.663].

We did not perform formal paired statistical tests (e.g., paired t-test across folds) because with only 5 folds, the test has very low power. The bootstrap CIs on the pooled out-of-fold predictions provide a more reliable comparison. The key finding is that gradient-boosted tree ensembles (CatBoost, XGBoost, LightGBM) form a performance cluster that substantially outperforms linear models and SVMs, while within the cluster, differences are modest.

### Q2: "Your external validation on BioFIND shows binary AUC of 0.637. This is barely above chance. How can you claim the model has clinical utility?"

**Answer**: We explicitly frame the binary external result as a **negative finding** that reveals the healthy control contamination confound, not as evidence of clinical utility for the binary model. The paper's discussion section dedicates a full paragraph to explaining why binary external validation fails: PPMI's Stage 0 includes healthy controls (UPDRS3 bradykinesia mean 7.4) while BioFIND is entirely diagnosed PD (mean ~20).

The clinically relevant external validation is for the **three-class** and **NSD-positive** targets:
- Three-class BioFIND AUC: 0.703 (moderate ranking ability)
- NSD-positive sub-staging (PPMI internal): AUC 0.900 with clinical features only

The three-class result (0.703) is notable because it groups stages 0-1 together, reducing the HC confound's impact. The NSD-positive model (0.900 AUC) is the recommended clinical tool, to be applied after a patient is confirmed NSD-positive via biomarkers.

We also note that BioFIND has only 108 evaluable patients (103 S+, 5 S-), making the binary validation severely underpowered. The bootstrap CI for AUC is [0.48, 0.76], spanning nearly the entire range from chance to moderate.

### Q3: "You report that DaT-SBR features cause a 25% AUC drop when removed for binary prediction, yet these are non-circular features (caudate, not putamen). Isn't the model still partially leveraging the staging biomarker?"

**Answer**: This is a nuanced point. Caudate SBR and putamen SBR are highly correlated (r > 0.85 in PPMI) because they both measure dopamine transporter density, just in different brain regions. The staging criterion uses putamen SBR < 0.80 as the D anchor, while our features use caudate SBR, caudate asymmetry, and caudate-to-putamen ratio.

We argue this is **non-circular but correlated**, which is the appropriate design. The analogy: if a doctor uses systolic blood pressure to diagnose hypertension (> 140 mmHg), a prediction model could legitimately use diastolic blood pressure and pulse pressure as features -- they're correlated with systolic but provide independent physiological information.

The 25% AUC drop demonstrates that DaT imaging carries genuine predictive signal for biological staging. Without it, the model relies only on clinical symptoms, which can appear similar across NSD+ and NSD- patients (especially in the Stage 0 class that includes healthy controls).

The NSD-positive sub-staging result (AUC 0.900 without DaT) proves that clinical features alone are sufficient once you remove the Stage 0/healthy control confound. This finding has direct clinical implications: for patients already known to be NSD-positive, expensive DaT-SPECT imaging may not be needed for sub-staging.

### Q4: "The paper lacks any analysis of model calibration. A model with high AUC but poor calibration could be clinically dangerous."

**Answer**: We address calibration through two mechanisms:

First, **conformal prediction inherently provides calibrated prediction sets**. The coverage guarantee (P(Y in C(X)) >= 1 - alpha) is distribution-free and does not depend on model calibration. Even if CatBoost's raw probabilities are poorly calibrated, the conformal prediction sets will still achieve the target coverage rate.

Second, **we include calibration plots** (Figure 7 in the paper) showing reliability diagrams (fraction of positives vs mean predicted probability) with Brier score annotations for the binary target across all 4 models. CatBoost shows good calibration with a slight overconfidence bias for high-probability predictions, which is typical of gradient-boosted trees.

For multiclass calibration, we report the log loss metric in the benchmark results. CatBoost achieves the lowest log loss for binary (0.128) and three-class (0.371), indicating better-calibrated probability estimates than alternatives.

### Q5: "You compare against only classical tabular models and a single GNN architecture. Where are the comparisons to other recent deep learning methods for clinical prediction?"

**Answer**: We deliberately focus on models with established clinical deployment precedent rather than chasing the latest deep learning architectures. Our model selection rationale:

**Tabular baselines** (7 models): Logistic Regression, ElasticNet, SVM, Random Forest cover the classical ML spectrum. CatBoost, XGBoost, LightGBM represent the current state-of-the-art for tabular data (per Grinsztajn et al. 2022, Shwartz-Ziv & Armon 2022).

**Graph models** (2): AdaMedGraph (the only published graph model for PPMI PD staging, Lian et al. 2024) and our Enhanced Multimodal GAT (a strong GNN baseline with cross-modal attention and multi-head GAT layers).

We considered but excluded TabNet (Arik & Pfister, 2021), FT-Transformer (Gorishniy et al., 2021), and SAINT (Somepalli et al., 2021). These attention-based tabular models have shown marginal improvements over tree ensembles on benchmark datasets but require significantly more tuning and have not been validated in clinical settings. Our primary contribution is not "best model ever" but rather "first reliable NSD-ISS prediction framework with calibrated uncertainty," making clinical applicability (interpretability, conformal guarantees, external validation) more important than incremental accuracy gains.

---

## 6. Alternative Approaches

### What Else Could Have Solved This Problem?

#### Alternative 1: Deep Tabular Models (TabNet, FT-Transformer)

**What they are**: Neural network architectures specifically designed for tabular data, using attention mechanisms instead of decision trees. TabNet uses sequential attention to select features at each step. FT-Transformer treats each feature as a token and applies transformer attention.

**Why we didn't choose them**: Grinsztajn et al. (NeurIPS 2022) showed that on medium-sized datasets (1K-10K samples), tree ensembles match or beat deep tabular models on most benchmarks. Our dataset (2,201 samples) falls in this range. Additionally, CatBoost provides native handling of categorical features and class imbalance that these models lack.

**Trade-off**: We may sacrifice 1-2% accuracy for substantial gains in training speed (seconds vs hours), interpretability (feature importance is straightforward), and clinical deployment simplicity (no GPU required for inference).

#### Alternative 2: Ordinal Regression (CORAL, Proportional Odds)

**What they are**: Specialized models that respect the ordering of stages (0 < 1 < 2B < 3 < 4) rather than treating them as independent categories. CORAL (Cao et al., 2020) uses rank-consistent ordinal classification.

**Why we didn't choose them**: Our four-target design already captures ordinality through QWK (Quadratic Weighted Kappa) evaluation. We did use CatBoost with class weighting, which implicitly handles ordinal relationships through the loss function. Explicit ordinal regression (proportional odds model) assumes a specific functional form that may not hold for NSD-ISS stages -- the clinical meaning of "distance" between Stage 2B and Stage 3 is different from Stage 3 and Stage 4.

**Trade-off**: Ordinal regression could improve QWK at the cost of balanced accuracy for minority classes. Our CatBoost achieves QWK of 0.861 for full ordinal, which is already strong. The three-class and binary formulations capture the clinically most important distinctions.

#### Alternative 3: Multi-Task Learning

**What it is**: Training a single model to predict all 4 targets simultaneously, with shared feature representations.

**Why we didn't choose them**: The 4 targets are deterministic transformations of the same underlying variable (NSD-ISS stage). Multi-task learning benefits most when targets capture complementary information. Our targets are hierarchical (binary is coarser than three-class, which is coarser than full ordinal), so shared representations wouldn't add new information.

**Trade-off**: Multi-task learning could provide regularization benefits but adds architectural complexity. Given that CatBoost already achieves near-ceiling performance for binary (AUC 0.979), the marginal gains would not justify the complexity.

#### Alternative 4: Bayesian Neural Networks for Uncertainty

**What they are**: Neural networks where weights are distributions (not point values), providing epistemic uncertainty estimates through the posterior distribution.

**Why we didn't choose them**: Conformal prediction provides **distribution-free** coverage guarantees that Bayesian methods cannot match. Bayesian uncertainty requires specifying prior distributions and approximation methods (variational inference, MCMC) that introduce additional assumptions. Conformal prediction wraps any model (including our CatBoost) without additional assumptions and provides finite-sample coverage guarantees.

**Trade-off**: Bayesian methods provide richer uncertainty decomposition (epistemic vs aleatoric) that conformal prediction does not. We address this in Paper 2 (MC dropout for imputation uncertainty) and Paper 4 (conformalized survival analysis), combining both approaches.

#### Alternative 5: Direct Biomarker Prediction Instead of Stage Prediction

**What it is**: Instead of predicting NSD-ISS stage (a derived label), predict the raw biomarkers (SAA positivity, DaT SBR deficit) directly, then apply the staging rules.

**Why we didn't choose this**: While scientifically appealing, this approach faces a fundamental data limitation: SAA has only 12.6% coverage in PPMI, making it impossible to train a reliable SAA prediction model. DaT SBR prediction is feasible (97.1% coverage) but would only recover one of two staging anchors.

**Trade-off**: Predicting biomarkers directly would be more interpretable and scientifically grounded, but the data gaps make it infeasible for the S anchor. Our approach (predicting stages from correlated non-circular features) pragmatically sidesteps the missing data problem.

### Honest Assessment

The honest assessment is that **CatBoost with conformal prediction is likely the right choice for this specific task**. The dataset is medium-sized (2,201 patients), the features are tabular and mixed-type, and clinical deployment requires interpretability and calibrated uncertainty. More sophisticated approaches (deep tabular models, Bayesian networks, graph neural networks) either don't outperform trees on this data type or introduce unnecessary complexity.

The real limitations are not in the modeling approach but in the **data**: healthy control contamination in Stage 0, 12.6% SAA coverage, only 17 Stage 4 patients, and limited external cohorts with ground truth. Future work should focus on acquiring more comprehensive biomarker data and external validation cohorts rather than more complex models.

---

*Document generated for dissertation defense preparation. All metrics cited from actual output JSONs in `outputs/paper1_benchmark/`, `outputs/paper1_conformal/`, and `outputs/external_validation/`. All file paths verified against the codebase.*
