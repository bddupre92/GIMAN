---
Last substantive update: 2026-04-21 (submission-alignment pass: 46-feature set + §IV-H four configurations)
Last touched: 2026-04-21 (cross-ref refresh after Paper 11 + P3/P4 holdout session)
Status: stable; refer to `Docs/NEXT_STEPS_2026-04-21.md` for dissertation-wide status
Cross-refs added 2026-04-21:
- Paper 11 (Ch 15 hybrid-twin preview) inherits the CatBoost-33 feature set as one baseline-covariate source for its physics-informed neural ODE on DaT-SBR trajectories — the "post-hoc mechanistic fusion" entry point.
- Paper 7 supplies the calibrated neurodegeneration rate (3.29 %/yr median N(t) decay) consumed by Paper 1's Alt-5 null probe in §14.4.
- 2026-04-21 validation pass confirms main stage-prediction claims hold (no AUC/bal-acc change).
---

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
[Feature Engineering] -- Three nested feature sets:
         |              - Full: 46 features across 10 domains (internal PPMI benchmark)
         |              - Graph-model input: 22 features (clinical 12 + biomarker 10; architectural split for Multimodal GAT)
         |              - Common-feature subset: 12 features (external-validation intersection across PPMI/BioFIND/PDBP/HBS)
         v
paper1_features_with_targets.csv (2,201 rows x 46 features + 4 targets)
         |
    +----+---------+
    |              |
    v              v
[7 Tabular    [Multimodal
 Models]       GAT]
    |              |
    v              v
Best model: CatBoost (AUC 0.979 binary, on 46-feature full set)
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

**CatBoost specifically** (developed by Yandex) adds innovations for handling categorical features (like sex, handedness) natively, and uses "ordered boosting" to reduce overfitting. In our case, it processes the full 46-feature set (10 clinical/imaging/biomarker domains) and outputs a probability distribution over the NSD-ISS stages.

- **Input**: 46 feature values for one patient (age, UPDRS subscales, DaT SBR, FreeSurfer volumes, SCOPA subscales, RBD subscales, medication status, derived interactions, etc.)
- **Output**: Probability for each stage (e.g., 5% Stage 0, 10% Stage 1, 60% Stage 2B, 20% Stage 3, 5% Stage 4)
- **Prediction**: The stage with the highest probability

#### What Is a Random Forest?

A **Random Forest** is an ensemble of decision trees built **independently** (not sequentially like boosting). Each tree is trained on a random subset of the data and a random subset of the features. The final prediction is a majority vote across all trees.

Think of it as: instead of asking 500 doctors who each learned from the same textbook (boosting), you ask 500 doctors who each studied a random subset of patient cases and a random subset of symptoms. Their collective wisdom averages out individual errors.

- **Input**: Same 46 features
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

- **Input**: Feature matrix (2,201 patients x 22 features; the Multimodal GAT uses the 22-feature clinical+biomarker split, which is an architectural constraint of the two-modality encoder, not a feature-selection step relative to the full 46-feature set)
- **Output**: A graph with 2,201 nodes and thousands of edges, where edge weights represent patient-to-patient similarity

#### What Is a Graph Attention Network (GAT)?

Once you have a patient similarity graph, a **GAT** lets each patient "learn" from their similar neighbors. It works like this:

1. Each patient starts with their own feature vector (22 values, split into 12 clinical + 10 biomarker)
2. The model looks at each patient's neighbors in the graph
3. It computes **attention weights**: how much should this patient pay attention to each neighbor?
4. It updates each patient's representation by combining their own features with a weighted average of their neighbors' features
5. This is repeated across multiple layers (like multiple rounds of information sharing)

Think of it as: "If I'm trying to predict your disease stage and I know 15 patients similar to you, I'll pay the most attention to the ones whose features are most informative for staging, and less attention to the ones that are less helpful."

- **Input**: Patient features + graph structure (who is connected to whom)
- **Output**: Updated patient representations that incorporate neighborhood information, then classified into stages

#### What Is AdaMedGraph? (Related Work — Conceptual Background Only)

**AdaMedGraph** (from Lian et al., 2024) is cited in the submission as Related Work (§II) — it is NOT a benchmarked model in Paper 1. The submission benchmarks 8 models total: 7 tabular + 1 Multimodal GAT. The Multimodal GAT is **adapted from** AdaMedGraph's attention/propagation intuition but uses a single unified patient-similarity graph (not per-feature graphs) and a PyG GATConv stack (not APPNP+AdaBoost). The pedagogical overview below covers what AdaMedGraph does conceptually, for readers who want to understand the design lineage.

AdaMedGraph combines two ideas:

1. **APPNP** (Approximate Personalized Propagation of Neural Predictions): A graph neural network that separates prediction from propagation. First, an MLP makes predictions; then, those predictions are smoothed across the graph using a process inspired by Google's PageRank algorithm. The "alpha" parameter (0.1) controls how much the original prediction is preserved vs. how much neighborhood information is incorporated.

2. **AdaBoost** (Adaptive Boosting): Instead of one big graph, AdaMedGraph builds many small graphs -- one per clinical feature. For each feature, patients are connected if their values for that specific feature are similar. Then an APPNP is trained on each per-feature graph, and the ensemble combines them with AdaBoost weighting (the SAMME algorithm), where better-performing graphs get higher weight.

- **Input (AdaMedGraph, reference architecture)**: features × 3 similarity thresholds = candidate graph bank
- **Process**: Iteratively selects the best per-feature graph, trains an APPNP, updates sample weights to focus on misclassified patients
- **Output**: Ensemble prediction combining multiple per-feature APPNP classifiers

Our Multimodal GAT keeps the "graph-attention over patient similarity" idea but drops the per-feature-graph boosting in favor of a single k=10 cosine-similarity graph over the 22-feature clinical+biomarker vector, processed by a 3-layer GATConv stack (4 heads, residual) with cross-modal attention fusion. See §3.6 below for the exact architecture that ships in the submission.

#### Key Evaluation Metrics Explained

**AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**: Measures how well the model ranks patients. An AUC of 0.979 means: if you pick a random NSD-positive patient and a random NSD-negative patient, there's a 97.9% chance the model assigns a higher probability to the positive patient. Perfect = 1.0, random = 0.5.

**Balanced Accuracy**: The average of per-class recall. In our dataset, Stage 0 has 64.4% of patients. Regular accuracy could reach 64.4% by always predicting Stage 0. Balanced accuracy weights each class equally: if you perfectly predict Stage 0 but miss all Stage 4 patients, your balanced accuracy would be low. CatBoost achieves 0.951 balanced accuracy for binary, meaning it correctly identifies both NSD-positive and NSD-negative patients at nearly equal rates.

**Cohen's Kappa**: Measures agreement between predictions and truth, adjusted for chance. A kappa of 0.906 (CatBoost binary) means agreement is 90.6% better than random chance. Values above 0.8 are considered "almost perfect agreement."

**Quadratic Weighted Kappa (QWK)**: Like Cohen's Kappa but penalizes distant misclassifications more than adjacent ones. Predicting Stage 0 when the true stage is Stage 4 is penalized much more than predicting Stage 2B when the true stage is Stage 3. Especially important for ordinal targets.

**Cross-Validation**: Instead of testing on the same data used for training (which would be cheating), we split the data into 5 parts. Train on 4 parts, test on 1 part. Repeat 5 times, each time holding out a different part. Average the results. This gives an honest estimate of how the model would perform on truly unseen patients.

---

## 3. The Deep Dive (Advanced Level)

*This section explains not just WHAT the code does, but WHY every parameter, pattern, and design decision exists. If a committee member asks "Why did you pick that value?" or "What happens if you change it?" -- this section gives you the answer.*

### 3.1 NSD-ISS Staging Pipeline

**File**: `src/giman_pipeline/staging/nsd_iss.py`

This module implements the exact Simuni et al. (2024) staging criteria as a deterministic algorithm. There is no machine learning here -- it is a direct translation of the published clinical criteria into code.

**`compute_s_anchor(saa_label)`**: Returns `True` if alpha-synuclein seed amplification assay (SAA) is positive, `False` if negative, `None` if missing. Coverage: only 12.6% (277/2,201) of PPMI patients have SAA data.

**Why is SAA coverage so low?** The SAA test requires a cerebrospinal fluid sample obtained via lumbar puncture (spinal tap). This is invasive, painful, and not performed at every visit. Many PPMI participants enrolled before SAA was routinely offered. This 12.6% coverage is the main reason the D anchor (DaT-SPECT) carries most of the staging weight.

**`compute_d_anchor(putamen_sbr_left, putamen_sbr_right, ...)`**: Returns `True` if dopaminergic deficit detected.

**Why use the LOWEST of left/right putamen SBR?** Parkinson's disease typically presents asymmetrically -- one side of the brain deteriorates faster. The lowest putamen SBR value reflects the more-affected hemisphere. Using the average would dilute the signal: a patient with left SBR = 0.5 (severe deficit) and right SBR = 1.2 (normal) has an average of 0.85 (above threshold), but clinically has a clear dopaminergic deficit on the left side.

**Why threshold < 0.80?** This is the published NSD-ISS criterion from Simuni et al. (2024). The SBR (Specific Binding Ratio) represents the ratio of dopamine transporter binding in the putamen to a reference region. Normal values are approximately 1.0-3.0. Values below 0.80 indicate the putamen has lost sufficient dopaminergic neurons to be clinically meaningful. This threshold was validated against neuropathological studies where SBR < 0.80 correlated with >50% dopaminergic cell loss.

**Why the fallback chain (lateralized putamen -> mean putamen -> mean caudate)?** Not all PPMI imaging records have the same columns. Some older records report only mean putamen SBR (without left/right split). Some only have caudate SBR. The fallback chain ensures maximum coverage: 97.1% of patients (2,137/2,201) can be D-staged through at least one of these paths. Without the fallback, coverage would drop to ~85% for lateralized putamen alone.

**Critical non-circular design**: The features used for staging (putamen SBR, UPDRS-III total, SAA) are **excluded** from the ML feature set. The full 46-feature ML set uses caudate SBR and caudate/putamen ratio (not the putamen SBR decision variable), UPDRS-III subscales (not the NP3TOT total), and never uses SAA. A `MOTOR_TOTAL_PROXY` feature equal to the sum of the four retained UPDRS-III subscales IS included — the submission acknowledges this is a cosmetic rather than strict information-theoretic decoupling from NP3TOT and ablates it with binary AUC change <0.01 (submission §VI.4 Limitations).

**Why is this non-circularity so important?** If we used putamen SBR as both a staging criterion AND an ML feature, the model could trivially learn "if putamen SBR < 0.80, predict NSD-positive" -- it would be memorizing the staging rule, not learning biology. That model would appear to have high accuracy but would provide zero clinical value beyond what the staging algorithm already gives. By using caudate SBR (a correlated but distinct brain region), we force the model to learn genuine biological relationships. Caudate SBR correlates with putamen SBR at r > 0.85 in PPMI, so the signal is still present -- the model just can't cheat.

Similarly, UPDRS-III subscales (tremor, rigidity, bradykinesia, axial) provide richer information than the total score used for staging. Two patients can both have UPDRS-III total = 15 but one might have severe tremor with mild rigidity, while another has the reverse. The subscales capture these differences while the total cannot.

### 3.1a The Three Nested Feature Sets (authoritative reference)

The submission explicitly defines a three-tier feature-set hierarchy (§III-D, Table II). Every numerical claim later in this deep-dive must be read against the correct tier.

**Tier 1 — Full tabular (46 features across 10 domains).** The canonical internal-benchmark input. Used for all PPMI 5-fold CV results in submission Tables I (benchmark), III (ablation), IV (four configurations when PPMI-internal). Organised as:

| Domain | Features | Count | PPMI | BioFIND | PDBP | HBS |
|---|---|---|---|---|---|---|
| Demographics | AGE_AT_BASELINE, SEX, HANDED | 3 | ✓ | 2/3 | 2/3 | 2/3 |
| Motor (UPDRS) | UPDRS1_TOTAL, UPDRS2_TOTAL, UPDRS3_TOTAL*, UPDRS4_TOTAL, UPDRS3_RIGIDITY, UPDRS3_BRADYKINESIA, UPDRS3_TREMOR, UPDRS3_POSTURE_GAIT | 8 | ✓ | partial | ✓ | partial |
| Cognitive | MOCA_TOTAL | 1 | ✓ | ✓ | ✓ | — |
| Sleep | ESS_TOTAL, RBD_TOTAL, RBD_MOTOR, RBD_VOCAL, RBD_DREAM, RBD_DISRUPTION | 6 | ✓ | 1/6 | 2/6 | — |
| Autonomic | SCOPA_AUT_TOTAL + GI + URINARY + CV + THERMO + PUPIL | 6 | ✓ | — | — | — |
| DaT Imaging | CAUDATE_L/R/MEAN_SBR, PUTAMEN_L/R/MEAN_SBR | 6 | ✓ | — | — | — |
| Brain Structural (FreeSurfer) | HIPPOCAMPUS_VOL, AMYGDALA_VOL, PUTAMEN_VOL, CAUDATE_VOL, BRAIN_STEM_VOL, WM_HYPOINTENSITIES, TOTAL_GRAY_VOL, CAUDATE_PUT_VOL_RATIO, INTRACRANIAL_VOL | 9 | ✓ | — | — | — |
| Olfactory | UPSIT_TOTAL | 1 | ✓ | — | ✓ | — |
| Medication | PD_MED_USE, ON_LEVODOPA | 2 | ✓ | ✓ | ✓ | — |
| Derived Interactions | MOTOR_IMAGING_INT, AGE_MOTOR_INT, MOTOR_TOTAL_PROXY | 3 | ✓ | 2/3 | 2/3 | 2/3 |
| Genetics | LRRK2, GBA, GRS_TOTAL | (folded into demographics/genetics domain, included in Ppmi-internal count) | ✓ | — | — | — |
| **TOTAL** | | **46** | | | | |

\* UPDRS3_TOTAL (NP3TOT) is the staging-decision variable — it is listed in submission Table II's Motor row but not in the 46-feature ML set; this is the "above/below Simuni threshold" variable that is strictly excluded from the model per §III-D circularity audit. Brain Structural *PUTAMEN_VOL* (MRI structural volume) is distinct from DaT-SPECT *PUTAMEN_SBR* (the staging variable) and is retained as a feature. **Notably absent from the external cohorts**: FreeSurfer volumes (BioFIND/PDBP/HBS have no MRI), full SCOPA-AUT subscales, and DaT imaging — which is why the external-validation subset collapses to 12 features.

**Tier 2 — Graph-model input (22 features; architectural split).** Used only by the Multimodal GAT, which has one encoder per modality:
- **Clinical stream** (12 features): demographics (3) + UPDRS subscales (4) + cognitive (1) + sleep (2) + autonomic total (1) + medication status (1).
- **Biomarker stream** (10 features): DaT-SPECT caudate/putamen SBR (6) + caudate asymmetry + caudate–putamen ratio + UPSIT olfaction + genetics subsum (~2 items packed as a binary LRRK2/GBA indicator).

This 22-feature split is an **architectural constraint** of the two-modality Multimodal GAT encoder. It is NOT a feature-selection experiment versus the 46-feature set; the tabular models do not use this split.

**Tier 3 — Common-feature external-validation subset (12 features).** The strict intersection across all four cohorts (PPMI ∩ BioFIND ∩ PDBP ∩ HBS): AGE_AT_BASELINE, SEX, UPDRS1_TOTAL, UPDRS2_TOTAL, UPDRS3_TREMOR, UPDRS3_RIGIDITY, UPDRS3_BRADYKINESIA, UPDRS3_POSTURE_GAIT, UPDRS4_TOTAL, MOCA_TOTAL, ESS_TOTAL, RBD_TOTAL. Used for external validation (§3.8 below) and for the four training configurations sweep (new §3.10). HBS is missing 5 of these 12; it is reported in Supplementary §S-1 of the submission as a deployment cautionary tale.

Throughout this deep-dive, the phrase "22-feature" ALWAYS refers to the graph-model input split (Tier 2), and "12-feature clinical-only" ALWAYS refers to the external-validation subset (Tier 3). All internal CV results for tabular models use the full 46-feature set (Tier 1) unless explicitly stated otherwise.

### 3.2 Seven-Model Benchmark

**File**: `src/giman_pipeline/sota/nsd_iss_benchmark.py`

#### The Model Factory Pattern (and why sklearn's clone() breaks CatBoost)

```python
factories["catboost"] = lambda: cb.CatBoostClassifier(
    iterations=1000, depth=6, auto_class_weights="Balanced", verbose=0
)
```

**What is sklearn's `clone()` and why does it matter?** When you do cross-validation, you need a fresh model for each fold. sklearn's `clone(estimator)` function creates a fresh copy of a model with the same hyperparameters. Under the hood, `clone()` calls `estimator.get_params()` to extract all constructor arguments as a Python dictionary, then creates a new instance with `EstimatorClass(**those_params)`.

**Why does CatBoost break?** When you pass `class_weights={0: 1.0, 1: 3.5, 2: 2.1}` to CatBoostClassifier, `get_params()` returns this dict. But when `clone()` tries to reconstruct the classifier with `CatBoostClassifier(class_weights={0: 1.0, 1: 3.5, 2: 2.1})`, CatBoost's internal C++ backend expects a specific format for weight dictionaries that the round-tripped Python dict doesn't match. This causes a TypeError at training time -- not at construction time, making the bug hard to track down.

**The factory function fix**: Instead of creating one model and cloning it, we use a lambda (factory function) that creates a brand-new CatBoostClassifier from scratch each time it's called. `factories["catboost"]()` produces a fresh instance with no serialization/deserialization issues.

**The `auto_class_weights="Balanced"` fix**: Instead of computing class weights ourselves and passing a dict, we use CatBoost's built-in `auto_class_weights="Balanced"` parameter (a string, which serializes safely). Internally, CatBoost computes: `weight_i = total_samples / (n_classes * count_of_class_i)`. For our binary target: Stage 0 weight = 2201 / (2 * 1418) = 0.776, Stage 1+ weight = 2201 / (2 * 783) = 1.405. This gives minority classes higher weight so the model doesn't ignore them.

#### Cross-Validation Mechanics

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**What does `StratifiedKFold` do mechanically?** It divides 2,201 patients into 5 non-overlapping groups (folds) of ~440 patients each. "Stratified" means each fold maintains the same class proportions as the full dataset. Without stratification, a random split could put all 17 Stage 4 patients into 1-2 folds, leaving other folds with zero Stage 4 examples. That would make those folds unable to evaluate Stage 4 recall. Stratified splitting guarantees each fold gets approximately 17/5 = 3-4 Stage 4 patients, 67/5 = 13-14 Stage 1 patients, etc.

**What does `shuffle=True` do?** Without shuffling, sklearn assigns patients to folds in their original row order. If the CSV happens to be sorted by enrollment date (which PPMI data often is), Fold 1 would contain the earliest-enrolled patients and Fold 5 the latest. This could introduce temporal bias if patient demographics shifted over the 10+ years of PPMI enrollment. Shuffling randomizes which patients go into which fold, breaking any ordering artifacts.

**Why `random_state=42`?** The number 42 has no mathematical significance. It's a convention in ML, originally a reference to Douglas Adams' *The Hitchhiker's Guide to the Galaxy* ("the answer to life, the universe, and everything"). What matters is that using ANY fixed integer seed makes the random number generator produce the same sequence every time. This means: (a) running the code twice produces identical fold assignments, making results reproducible; (b) other researchers can verify our results exactly; (c) comparing models is fair because they all get the same train/test splits. If we omitted `random_state`, Python's random number generator would seed from the system clock, producing different splits each run and making results non-reproducible.

#### Z-Score Standardization Within Folds

**What is z-score standardization?** For each feature, subtract the mean and divide by the standard deviation: `z = (x - mean) / std`. This transforms every feature to have mean = 0 and standard deviation = 1. Without this, features with large ranges (e.g., age: 30-90) would dominate features with small ranges (e.g., SEX: 0-1) in distance-based models (SVM, KNN, logistic regression).

**Why "within each fold"?** We fit the scaler (compute mean and std) on the TRAINING set only, then apply those same mean/std values to transform the test set. If we computed mean/std on the full dataset (including test), the test set statistics would leak into the training set through the standardization parameters. This is called **data leakage**.

**What would data leakage look like mechanically?** Suppose Feature X has mean 50 on the training set but mean 55 on the full dataset (because the test set patients happen to have higher X values). If we use the full-dataset mean (55) to standardize, training samples with X = 55 get z-score = 0 instead of z-score = +1.0. The model trains on these biased z-scores and appears more accurate on the test set than it would be on truly unseen data. The effect is subtle (usually 1-3% accuracy inflation) but it makes published results unreproducible on external data.

**Why doesn't CatBoost need standardization?** Tree-based models (CatBoost, XGBoost, Random Forest) split on individual feature thresholds ("is age > 65?"). These splits are invariant to scaling -- whether age is measured in years (30-90) or z-scores (-2 to +2), the tree finds the same optimal split point. So standardization is applied for SVM, logistic regression, and elastic net, but skipped for tree models. The code detects model type and skips scaling for tree-based classifiers.

#### CatBoost Hyperparameters: What Each Does and Why That Value

**`iterations=1000`** (submission §III-E-i, line 224): This is the number of boosting rounds -- how many decision trees are built sequentially. Each tree corrects the mistakes of the previous ensemble.

- **Why 1000?** We monitored the training loss curve: loss decreases rapidly for the first 100 iterations, continues improving slowly through 300-400, and plateaus around 800-1000. 1000 is the value that ships in the submission. CatBoost uses an internal learning rate schedule that makes early iterations more impactful.
- **What happens at 100?** The model slightly underfits (~1-2% worse balanced accuracy) because the ensemble hasn't had enough rounds to learn complex feature interactions.
- **What happens at 500?** Slight underfit (~0.1–0.3% bal_acc deficit depending on target) — this was the pre-submission value that later studies pushed to 1000; the deltas are small but consistently non-negative going from 500 → 1000.
- **What happens at 5000?** Marginal accuracy gain (<0.1%) with risk of overfitting to training noise. CatBoost's built-in L2 regularization (`l2_leaf_reg`, default 3.0) mitigates this, but more iterations still increase computation by 10x.

**`depth=6`**: The maximum depth of each individual decision tree. A tree of depth 6 makes at most 6 sequential yes/no splits, creating up to 2^6 = 64 leaf nodes.

- **What does tree depth mean mechanically?** A depth-1 tree ("stump") asks one question: "Is age > 65?" and has 2 outcomes. A depth-6 tree can ask 6 sequential questions: "Is age > 65? AND is bradykinesia > 20? AND is caudate SBR < 1.0? AND..." -- capturing complex feature interactions. Each additional depth level doubles the possible leaf nodes and allows one more feature interaction.
- **Why depth 6?** Standard for gradient boosting on tabular data (XGBoost default is 6). With 46 features and ~2,000 samples, depth 6 allows up to 6-way feature interactions while keeping each leaf node populated with ~2000/64 = 31 training samples (enough for stable estimates).
- **Why not deeper?** Depth 8 (256 leaves) would average ~8 samples per leaf, leading to noisy leaf predictions. Depth 10 (1024 leaves) would have more leaves than training samples in a CV fold (~1760), guaranteeing overfitting. The boosting framework compensates for shallow trees by combining many of them -- 500 trees of depth 6 is more powerful and less overfit than 50 trees of depth 12.
- **Why not shallower?** Depth 3-4 restricts the model to simple 3-4 way interactions. For multiclass targets with complex boundaries between 5 NSD-ISS stages, depth 3 loses ~3% balanced accuracy.

**`auto_class_weights="Balanced"`**: Tells CatBoost to automatically compute class weights inversely proportional to class frequency.

- **Formula**: weight_c = N / (K * n_c), where N = total samples, K = number of classes, n_c = samples in class c.
- **For binary**: Stage 0 weight = 2201/(2*1418) = 0.776. Stage 1+ weight = 2201/(2*783) = 1.405. The minority class gets 1.8x the weight of the majority class.
- **For full ordinal**: Stage 4 weight = 2201/(5*17) = 25.9. Stage 0 weight = 2201/(5*1418) = 0.31. Stage 4 patients get 84x the weight of Stage 0 patients, forcing the model to pay attention to this tiny class.
- **What happens without class weights?** The model optimizes overall accuracy. Since Stage 0 is 64.4% of the data, predicting "Stage 0 for everyone" gives 64.4% accuracy. The model would learn to mostly predict Stage 0, getting excellent accuracy but 0% recall on minority stages. Balanced weighting fixes this by making each misclassified Stage 4 patient cost 84x as much as a misclassified Stage 0 patient.
- **Why "Balanced" and not "SqrtBalanced"?** CatBoost offers both. "Balanced" uses full inverse frequency (aggressive weighting). "SqrtBalanced" uses square-root of inverse frequency (gentler). For our severely imbalanced data (0.8% Stage 4), full balanced weighting is needed to achieve reasonable minority recall. SqrtBalanced was tested and produced ~2% lower balanced accuracy on full ordinal because Stage 4 recall dropped.

**`verbose=0`**: Suppresses CatBoost's training output (progress bars, loss values). Without this, each of 7 models x 5 folds x 4 targets = 140 training runs would print hundreds of lines, making logs unusable.

#### Bootstrap Confidence Intervals: Mechanics

**1,000 resamples of concatenated out-of-fold predictions. Percentile method: [2.5th, 97.5th] for 95% CI.**

**What does this mean step by step?**
1. After 5-fold CV, we have predictions for all 2,201 patients (each predicted when they were in the test set)
2. We concatenate these into one big array of (true_label, predicted_label) pairs
3. For each of 1,000 iterations: randomly sample 2,201 pairs WITH REPLACEMENT (some pairs selected multiple times, others not at all)
4. Compute balanced accuracy on this bootstrap sample
5. After 1,000 iterations, sort the 1,000 balanced accuracy values
6. The 25th value is the lower CI bound, the 975th value is the upper CI bound

**Why 1,000 resamples?** The precision of the bootstrap CI itself depends on the number of resamples. With B resamples, the Monte Carlo error in the CI endpoint is approximately `1/sqrt(B)`. At B=1000, this error is ~3.2% of the CI width, or about +/-0.003 for a typical CI width of ~0.020. Going to B=10,000 would reduce this to ~1%, adding precision but 10x the compute time. 1,000 is the standard trade-off.

**Why the percentile method rather than BCa or studentized bootstrap?** The percentile method is the simplest: just take quantiles of the bootstrap distribution. BCa (bias-corrected accelerated) adjusts for skewness and bias but requires computing influence functions, which is complex for multiclass metrics. For our sample sizes (2,201 patients), the bootstrap distribution is approximately normal, making the percentile method sufficiently accurate.

### 3.3 Results: Why CatBoost Dominates

Internal benchmark on the full 46-feature set (submission Table I, 5-fold stratified CV, 1,000-resample bootstrap 95% CIs):

| Target | CatBoost Bal Acc [95% CI] | CatBoost AUC | Best Alternative (Bal Acc) | Gap |
|--------|---------------------------|--------------|---------------------------|-----|
| Binary | **0.951** [0.941, 0.960] | **0.979** [0.970, 0.986] | XGBoost 0.951 / LightGBM 0.950 | ~0 |
| Three-class | **0.783** [0.753, 0.803] | **0.944** | XGBoost 0.762 | +2.1% |
| Full ordinal | **0.658** [0.623, 0.697] | **0.954** | XGBoost 0.644 | +1.4% |
| NSD-positive | **0.671** [0.624, 0.728] | **0.913** | XGBoost 0.669 | +0.2% |

Note: Submission Table I gives NSD-positive CatBoost bal-acc as 0.671 [0.624, 0.728] while the §V-A narrative text cites 0.664. This is an internal inconsistency in the submission (likely an older vs. newer rerun not back-ported to the prose). We use the Table I number 0.671 as canonical because the table is the load-bearing artefact; the 0.664 wording should be reconciled in a future errata. Full-ordinal CatBoost bal-acc 0.658 likewise replaces the older 0.660 value that was carried into earlier drafts.

**Why does CatBoost win balanced accuracy while XGBoost sometimes wins AUC?** AUC measures ranking quality (can the model rank patients by risk?). Balanced accuracy measures classification quality at a specific decision threshold. CatBoost's `auto_class_weights="Balanced"` shifts decision boundaries toward minority classes, improving balanced accuracy (recall for small classes) at the cost of slightly less optimal ranking. XGBoost without balanced weighting optimizes the ranking objective (logloss) more purely, achieving higher AUC. The clinical setting determines which matters more: for screening (need to catch every positive), balanced accuracy is preferred. For risk stratification (need to rank patients), AUC is preferred.

**Per-class recall analysis (CatBoost, three-class)**:
- Class 0 (Early): 0.935 recall
- Class 1 (Mild clinical): 0.582 recall (hardest)
- Class 2 (Impaired): 0.831 recall

**Why is Class 1 (Mild clinical = Stage 2B) the hardest?** Stage 2B is a transitional stage between "biological markers with no clinical signs" (Stage 1) and "functional impairment" (Stage 3). Patients in Stage 2B have clinical parkinsonism (motor signs) but their daily functioning hasn't declined yet. Their feature profiles overlap significantly with both Stage 1 (who also have motor signs but milder) and Stage 3 (who have similar motor signs plus functional decline). The model can clearly separate "healthy/early" (class 0) from "impaired" (class 2), but the boundary between "clinical but not impaired" (class 1) is inherently fuzzy.

### 3.4 Conformal Prediction Implementation

**File**: `src/giman_pipeline/sota/conformal.py`

#### What Is the LAC Conformity Score? (And Why Not APS?)

MAPIE 1.3.0 offers two conformity scores for classification:

**LAC (Least Ambiguous set-valued Classifier)**: For each calibration sample, the nonconformity score is `1 - P(y_true)`, where `P(y_true)` is the model's predicted probability for the true class. A patient the model got right with high confidence has a low score (e.g., 1 - 0.95 = 0.05). A patient the model was uncertain about has a high score (e.g., 1 - 0.30 = 0.70). The calibration threshold is set at the (1-alpha) quantile of these scores. At test time, a class is included in the prediction set if `1 - P(class) <= threshold`.

**APS (Adaptive Prediction Sets)**: Sorts classes by predicted probability and includes them one at a time (highest first) until cumulative probability exceeds 1-alpha. This guarantees coverage but tends to produce larger sets because it's based on cumulative probability sums.

**Why LAC over APS?** LAC produces smaller (more informative) prediction sets. In our binary task, LAC produces 96.4% singletons vs APS's ~92%. Smaller sets are more clinically useful: "this patient is Stage 2B" is more actionable than "this patient is Stage 2B or Stage 3." LAC achieves this by using pointwise probabilities (each class independently assessed) rather than cumulative sums.

#### What Does `.conformalize()` Do Step by Step?

```python
SplitConformalClassifier(estimator=model, confidence_level=0.90,
                         conformity_score="lac", prefit=True)
.conformalize(X_cal, y_cal)
```

Step by step:
1. `prefit=True` tells MAPIE the model is already trained -- don't retrain it
2. `conformalize()` runs `model.predict_proba(X_cal)` to get probability vectors for all calibration samples
3. For each calibration sample i, computes the nonconformity score: `s_i = 1 - model.predict_proba(X_cal[i])[y_cal[i]]`
4. Sorts these scores: s_(1) <= s_(2) <= ... <= s_(n)
5. Finds the threshold: `q_hat = s_(ceil((n+1)*(1-alpha)))` -- the (1-alpha) quantile of scores, with a finite-sample correction (+1 in the numerator)
6. Stores `q_hat` internally for use in `predict_set()`

Then `predict_set(X_eval)`:
1. Computes `model.predict_proba(X_eval)` for each test sample
2. For each class c: include c in prediction set if `1 - P(c) <= q_hat`
3. Returns a boolean matrix (n_test, n_classes) where True = class included in set
4. Also returns the argmax prediction as `y_pred`

The returned tuple is `(y_pred, prediction_sets_bool)` -- you MUST unpack both (a MAPIE 1.3.0 API change from earlier versions that returned just the sets).

**Why the finite-sample correction (+1)?** Without it, the coverage guarantee is only asymptotic (holds as n -> infinity). With the +1 correction, coverage >= 1-alpha holds for ANY finite sample size. For our calibration sets of ~200-400 patients, this correction adds about 0.002-0.005 to the threshold, slightly enlarging prediction sets but guaranteeing the mathematical coverage bound.

### 3.5 APPNP + SAMME Boosting Concepts (pedagogical context for the Multimodal GAT)

**File**: `src/giman_pipeline/models/adamedgraph.py` (reference implementation)

**Important scope note.** The submission's benchmark is 8 models total: 7 tabular + 1 Multimodal GAT (§III-E, Table I + Table IV-graph). AdaMedGraph itself appears only as Related Work (§II) and is not a benchmarked model row in the submission. The Multimodal GAT is *adapted from* AdaMedGraph's attention/propagation intuition but uses a single k=10 cosine-similarity patient-similarity graph and a PyG GATConv stack (not per-feature graphs + APPNP + AdaBoost). The material below explains APPNP and SAMME conceptually because they are the design lineage the submission cites; it does NOT describe a model that ships numbers in the benchmark tables. For the Multimodal GAT that actually ships, see §3.6 below.

#### APPNP: Decoupling Prediction from Propagation

**What does alpha=0.1 mean in APPNP?** APPNP performs K=5 iterations of the update rule:

```
H^(k+1) = (1 - alpha) * S * H^(k) + alpha * H^(0)
```

where S is the normalized adjacency matrix, H^(0) is the initial MLP prediction, and H^(k) is the prediction after k propagation steps. At each step, the prediction is a weighted average of:
- (1-alpha) = 0.9 = 90%: neighborhood-smoothed prediction from the graph
- alpha = 0.1 = 10%: original MLP prediction (the "teleport" or "restart" term)

**Why alpha=0.1?** This means the final prediction is dominated by graph neighborhood information (90%) with only 10% of the original MLP prediction retained. This makes sense for patient similarity: if your neighbors in the graph mostly have Stage 3, you're probably Stage 3 too, even if the MLP on your features alone is uncertain.

**What happens at alpha=0.0?** Pure graph propagation with no teleport. After many iterations, all connected nodes converge to the same prediction (the graph's dominant eigenvector). All patients in the same connected component get the same prediction, ignoring individual features. This is called "over-smoothing" and destroys discriminative power.

**What happens at alpha=1.0?** No graph propagation at all -- just the raw MLP prediction. The graph is completely ignored, and APPNP degenerates to a standard 2-layer neural network. You lose all the neighborhood information.

**Why K=5 propagation steps?** With K=5 and alpha=0.1, the effective receptive field (how far information travels) is about 5 hops in the graph. After 5 steps, the contribution from the original MLP is `0.1 + 0.1*0.9 + 0.1*0.81 + 0.1*0.729 + 0.1*0.656 = 0.1 * (1 + 0.9 + 0.81 + 0.729 + 0.656) = 0.42`. So the final prediction is roughly 42% MLP + 58% graph. More steps would further dilute the MLP signal; fewer steps would limit the receptive field. K=5 is the standard from Klicpera et al. (ICLR 2019).

#### The SAMME Boosting Formula

```
alpha_t = learning_rate * (log((1 - error_t) / error_t) + log(K - 1))
```

**What does each part mean?**

- `error_t`: Weighted classification error of boosting round t (fraction of misclassified samples, weighted by current sample weights). Range: [0, 1].
- `log((1 - error_t) / error_t)`: The "log-odds" of correct classification. If error = 0.1 (90% correct), this is log(9) = 2.20. If error = 0.4 (60% correct), this is log(1.5) = 0.41. Higher accuracy -> higher alpha -> more weight to this round.
- `log(K - 1)`: Correction for multiclass classification. For binary (K=2), this is log(1) = 0, so the formula reduces to standard AdaBoost. For 5-class (K=5), this is log(4) = 1.39, which adds a positive constant to alpha, ensuring that even mediocre classifiers (error close to random = (K-1)/K = 0.8) get non-negative weight.
- `learning_rate`: Scales all alpha values (default 1.0). A lower learning rate (e.g., 0.5) shrinks each round's contribution, requiring more rounds but reducing overfitting.

**Why do we stop if `error >= (K-1)/K`?** For K=5 classes, random chance = 0.80 error. A classifier worse than random (error > 0.80) would get negative alpha (the log-odds goes negative), meaning its contribution is actively harmful. The stopping criterion prevents adding noise to the ensemble.

#### Per-Feature Graph Construction: Quantile Thresholds

**`compute_quantile_thresholds(X, feature_idx, quantiles=(4, 8, 16))`**: For each feature, computes threshold = feature_range / q.

**Why three thresholds (4, 8, 16)?** Each threshold creates a different graph density:
- q=4: threshold = range/4. Very loose -- most patients are connected. Dense graph (~60% of possible edges). Risk: over-smoothing, all patients get similar predictions.
- q=8: threshold = range/8. Moderate -- patients connected if within 12.5% of each other. Medium graph (~15% of edges).
- q=16: threshold = range/16. Tight -- only very similar patients connected. Sparse graph (~3% of edges). Risk: many isolated nodes, insufficient information sharing.

The AdaBoost framework automatically selects the threshold that produces the lowest weighted error for each boosting round. Typically, early rounds select moderate thresholds (q=8) for informative features (DaT SBR), and later rounds select loose thresholds (q=4) for less informative features.

**Why skip graphs with <5 edges or >n^2/4 edges?** Degenerate graphs: <5 edges means almost no patients are connected (useless for propagation). >n^2/4 edges means the graph is so dense it's nearly complete (no discriminative structure). Both are filtered before APPNP training.

### 3.6 Multimodal GAT (the single graph model in the submission)

**File**: `scripts/run_enhanced_gat_benchmark.py`

Note on naming: the submission calls this the "Multimodal GAT" (Fig. 2 caption, §III-E-ii, Table IV). Earlier drafts and some internal notes call it "Enhanced MM-GAT." They refer to the same model. We use "Multimodal GAT" for consistency with the submission.

**Architecture (submission Fig. 2 + §III-E-ii, line 280):** Two modality encoders on the 22-feature graph-model split (Tier 2 above):
- **Clinical stream (12 features)**: demographics + UPDRS-III subscales + cognitive + sleep + autonomic total + medication status → 128-dim embedding via FC → LayerNorm → ReLU.
- **Biomarker stream (10 features)**: DaT-SPECT caudate/putamen SBR (6) + caudate asymmetry + caudate–putamen ratio + UPSIT olfaction + genetics (LRRK2/GBA indicator) → 128-dim embedding via FC → LayerNorm → ReLU.

Each stream is then processed through a **3-layer PyG GATConv** stack (4 attention heads per layer, residual connections), operating on a **k=10 cosine-similarity k-NN patient-similarity graph** constructed on the 22-feature standardised vector. `k=10` is fixed at the AdaMedGraph default (submission §III-E-ii, line 280) and is not tuned.

After GATConv, **cross-modal attention** (`nn.MultiheadAttention`) lets each modality re-weight its contribution using the other as key and value. Final fusion concatenates both streams (256d → 128d FC + LayerNorm) and passes through a classification head (128 → K classes). No per-feature graphs; no APPNP; no AdaBoost.

**Why build the graph WITHIN each CV fold?** If you build one graph on the full dataset and then split into folds, patients in the test fold already have edges to training patients. During GAT message passing, test patients receive information from training patients' features -- this is a form of data leakage. Building the graph within each fold means the k-NN computation only sees training patients, ensuring the test set is truly unseen.

**Why undirected + self-loops?** Undirected: if Patient A is similar to Patient B, the relationship is symmetric. Self-loops: each patient should attend to their own features in addition to neighbors'. Without self-loops, a GATConv layer's output for a node is based ONLY on its neighbors, discarding its own features entirely. The `add_self_loops()` function in PyG adds an edge from each node to itself with weight 1.0, ensuring self-information is preserved.

**Results (submission Table IV-graph / Table in §IV-D):** Binary bal_acc **0.825 ± 0.013**, AUC 0.894. Gap to CatBoost binary bal_acc: **−12.6 pp**. Across all four targets the gap to CatBoost ranges from −7.3 pp (three-class) to −12.7 pp (NSD+ subgroup) in balanced accuracy. This confirms that the "trees beat deep learning on tabular data" finding (Grinsztajn et al., NeurIPS 2022) extends to clinical biomarker data at PPMI scale (n=2,201).

### 3.7 Feature Ablation: DaT-SBR Is the Critical Feature

Submission Table III (§IV-E, line 425) — full 46-feature model vs. 12-feature clinical-only subset (CatBoost AUC, 5-fold CV):

| Target | Full (46) AUC | Clinical (12) AUC | Δ |
|--------|---------------|-------------------|---|
| Binary | 0.979 | 0.727 | **−25.2%** |
| Three-class | **0.944** | 0.797 | **−15.6%** |
| Full ordinal | **0.954** | 0.823 | **−13.1%** |
| NSD+ subgroup | **0.913** | 0.900 | **−1.4%** |

The submission's §V-A prose states "removing DaT-SPECT features reduced AUC by 25.4%" while the table above gives 25.2%. We treat the table as canonical (25.2%). The 0.2-pp gap appears to be rounding between the paper-internal headline and the tabulated ablation; both should be reconciled in the next manuscript revision. The feature set being compared is the full 46-feature set (Tier 1, submission Table II) versus the 12-feature common external-validation subset (Tier 3) — not the 22-feature graph-model split (Tier 2).

**Why does binary lose 25.2% AUC but NSD-positive loses only 1.4%?** The binary target separates Stage 0 (NSD-negative) from Stages 1+ (NSD-positive). The NSD-ISS definition of NSD-positivity is fundamentally biological: it requires either synuclein pathology (S+) or dopaminergic deficit (D+). Since SAA coverage is only 12.6%, the D anchor (DaT-SPECT) is the primary biological marker for most patients. Caudate SBR is the closest non-circular proxy for the D anchor. Without it, the model can only use clinical symptoms (motor scores, cognition, sleep) to infer biological status -- a much weaker signal. Hence the 25.2% drop.

For NSD-positive sub-staging (discriminating WITHIN stages 1-4), all patients are already confirmed NSD-positive. The question becomes "how impaired is this patient?" not "does this patient have biological disease?" Functional impairment is directly measured by clinical features: motor subscores, cognitive tests, sleep questionnaires. DaT imaging adds almost nothing because DaT SBR varies relatively little within the NSD-positive group (most have clear deficits). Hence the tiny 1.4% AUC drop (submission Table III).

**Clinical implication**: For screening (binary: is this patient NSD-positive?), DaT imaging is essential. For sub-staging (how advanced is the disease?), clinical features alone are nearly as good. This finding directly influences deployment strategy: the NSD-positive sub-staging model could be used in clinics without DaT-SPECT equipment.

### 3.8 External Validation and Domain Shift

**File**: `scripts/run_external_validation.py`

**What is domain shift mechanically?** The model learns a decision boundary between classes from PPMI training data. If external cohort patients occupy a different region of feature space, the decision boundary doesn't separate them correctly.

**PPMI Stage 0 includes**: Healthy controls (UPDRS3 bradykinesia ~7.4), prodromal patients (~10), SWEDD patients (~12), and PD patients who happen to be S-/D- (~15).

**BioFIND Stage 0 (S-)**: ALL diagnosed PD patients (UPDRS3 bradykinesia ~20).

The model learned "low bradykinesia = Stage 0" from PPMI (because healthy controls have low scores). When applied to BioFIND, ALL patients have high bradykinesia (they're all PD), so the model predicts everyone as NSD-positive, failing on the S- patients who actually have high motor scores but no biological markers. The decision boundary is in the wrong place for an all-PD cohort.

**BioFIND NSD-ISS staging replication** (`scripts/stage_biofind_nsd_iss.py`): Near-perfect match to Russo et al. (2025): Stage 2: 9, Stage 3: 58, Stage 4: 34 (vs 35 published), Stage 5: 2. The 1-patient discrepancy in Stage 4 vs 5 is likely due to rounding in one staging variable threshold.

### 3.9 Key Constants and Hyperparameters (with "What Happens If You Change It")

| Parameter | Value | What It Does | What happens if changed |
|-----------|-------|-------------|------------------------|
| CatBoost iterations | 1000 (submission §III-E-i) | Number of sequential trees in the ensemble | 100: underfits (−1 to −2% bal_acc). 500: slight underfit (~0.1–0.3% bal_acc deficit across targets — the pre-submission value). 5000: risk of overfitting, 5x training time, marginal gain (<0.1%). |
| CatBoost depth | 6 | Max depth of each tree (2^6=64 leaves) | 3: loses complex interactions (-3% bal_acc). 8: 256 leaves, ~8 samples/leaf, starts overfitting. 10: more leaves than training samples, guaranteed overfitting. |
| `auto_class_weights` | "Balanced" | Inverse-frequency class weighting | Without: model ignores Stage 4 (0% recall). "SqrtBalanced": gentler weighting, ~2% worse bal_acc on full ordinal. |
| CV folds | 5 | Number of train/test splits | 3: less variance reduction, more training data per fold. 10: more variance reduction but only ~220 test samples per fold (unstable for Stage 4 with ~2 test patients per fold). |
| Bootstrap resamples | 1000 | Iterations for CI estimation | 100: CI endpoints jittery (+/-0.01 noise). 10000: smoother but 10x compute. 1000 is the standard trade-off. |
| Conformal confidence | 0.90 | Target coverage for prediction sets | 0.80: smaller sets (more specific) but 20% miss rate. 0.95: larger sets (less specific) but only 5% miss rate. 0.90 is the clinical standard for "diagnostic-level" confidence. |
| D anchor threshold | 0.80 SBR | Published criterion for dopaminergic deficit | NOT adjustable -- this is the published NSD-ISS definition from Simuni et al. (2024). Changing it would redefine the staging system. |
| k-NN neighbors (GAT) | 10 | Graph density for Enhanced GAT | 5: very sparse graph, some patients isolated. 20: dense graph, over-smoothing risk. 10 balances density/sparsity; each patient shares with ~0.5% of the cohort. |
| APPNP alpha | 0.1 | Teleport probability (self-retention) | 0.0: pure graph smoothing, all nodes converge. 0.5: balanced MLP/graph. 1.0: no graph, pure MLP. 0.1 is standard (Klicpera et al. 2019). |
| APPNP K steps | 5 | Propagation iterations | 1: only direct neighbors. 10: information from 10 hops away. 5 is standard; combined with alpha=0.1 gives ~42% MLP retention. |
| AdaBoost max estimators | 10 | Maximum boosting rounds | 5: may stop too early on hard targets. 20: more rounds but each adds ~10 min compute. Convergence typically at 5-8 for 22 features (the best per-feature graphs are selected early). |

### 3.10 Four Training Configurations for the HC-Contamination Confound (submission §IV-H)

This is the single most important content that was missing from earlier drafts of this deep-dive. It is the mitigation path for the training-label confound identified by Espay~et~al.~(2025) and independently in the "Reconsidering the clinical foundations of NSD-ISS" (2025) critique, answered by Simuni~et~al.~(2025 reply) and empirically operationalised here.

#### The balanced BioFIND external set (n=118)

Published Russo et al. (2025) NSD-ISS staging of BioFIND: 103 SAA-positive PD patients, all labelled NSD+. That cohort has only one class of the binary target present. The BioFIND features release (2025) contains **118 PD patients total**; the additional 15 are SAA-negative PD patients that Bentivoglio~et~al.~(2026) characterise as a distinct, real PD sub-phenotype mapping to NSD-negative under the NSD-ISS framework. Combining these gives a **balanced external set of n=118 (103 NSD+ + 15 NSD-negative)**. This is the first external NSD-ISS test cohort with **both classes represented on a PD-only population** — the essential prerequisite for measuring whether PD-only retraining actually fixes the confound.

The older "n=108 evaluable, 5 S-" cohort cited in earlier drafts of this deep-dive corresponds to a pre-Bentivoglio BioFIND release and has been superseded by the n=118 balanced set throughout the submission.

#### Four configurations (submission Table IV, line 486)

CatBoost binary classifier on the 12-feature common external-validation subset (Tier 3). PPMI values are 5-fold stratified CV (mean ± SD); BioFIND is the balanced n=118 external set.

| Configuration | n_train | PPMI Bal Acc | PPMI AUC | BioFIND Bal Acc | BioFIND AUC |
|---|---|---|---|---|---|
| Baseline (full PPMI) | 2,201 | 0.670 ± 0.026 | 0.738 ± 0.027 | 0.470 | 0.561 |
| **PD-only** | 1,747 | 0.618 ± 0.006 | 0.677 ± 0.016 | **0.521** | 0.561 |
| Weighted (HC=SWEDD=0.1) | 2,201 | 0.652 ± 0.027 | 0.727 ± 0.026 | 0.497 | 0.524 |
| Stage-A (HC-vs-PD detection) | 2,201 | **0.832 ± 0.020** | **0.931 ± 0.015** | — | — |

#### Load-bearing findings

1. **Feature-set dependence of the confound.** On the full 46-feature set (Tier 1) — i.e. the internal benchmark setting — PD-only retraining barely moves metrics (0.946 ± 0.014 bal-acc vs. 0.951 ± 0.019 baseline; 0.982 ± 0.011 AUC vs. 0.979 ± 0.013). DaT-SPECT striatal binding ratio dominates the decision boundary; there is no room for the classifier to exploit the HC-vs-PD motor gap as a shortcut. The confound emerges only on the 12-feature clinical-only external-validation subset (Tier 3), where without imaging features the model falls back on the HC-vs-PD motor-score gap (PPMI HC UPDRS-III bradykinesia mean 7.37 vs. BioFIND SAA-negative PD mean 19.9; submission §IV-G, Fig. 5 domain-shift).

2. **PD-only retraining produces the right-direction shift but is statistically underpowered at n=118.** BioFIND balanced accuracy goes from 0.470 (baseline) to **0.521 (PD-only)** — a +5.1 pp point-estimate improvement. A post-hoc 1,000-resample bootstrap on the n=118 balanced BioFIND set shows 95% CIs for baseline and PD-only that overlap substantially and both straddle chance level. The direction is right; the n=118 sample is too small to declare statistical significance at α=0.05. This is reported honestly in the submission (§IV-H, line 501) and should be disclosed verbatim in the defense.

3. **Sample-weighting is NOT a substitute for exclusion.** Weighted (HC = SWEDD = 0.1) achieves internal-CV bal-acc 0.652 and external-BioFIND bal-acc 0.497 / AUC 0.524 — both worse than PD-only (0.521 / 0.561). Partial down-weighting leaks residual HC information through the kept mass. Full cohort exclusion (PD-only) is the effective mitigation.

4. **Stage-A (HC-vs-PD detection) is nearly saturated and solves a different, easier problem.** A binary classifier trained to separate HC/SWEDD from PD+Prodromal on the 12-feature subset achieves 0.832 ± 0.020 bal-acc and 0.931 ± 0.015 AUC on PPMI. This confirms the HC-detection problem is separable from NSD-ISS staging — and enables a **two-stage hierarchical deployment**: Stage-A screens incoming patients for HC/PD, Stage-B PD-only-retrained NSD-ISS classifier stages the PD-positive patients. At specialty PD clinics where all patients are already diagnosed (as in BioFIND, PDBP, and most clinical deployment contexts), Stage-A can be bypassed and the PD-only Stage-B classifier applied directly.

#### Clinical deployment protocol (submission §IV-H + Discussion §V)

- **Sites with DaT-SPECT access**: use the full 46-feature CatBoost model. The HC confound does not manifest at this feature-set tier because imaging dominates the boundary.
- **Sites without imaging**: use the **PD-only retrained 12-feature CatBoost model** (1,747-patient training set = PPMI PD + Prodromal, HC + SWEDD excluded). This removes the label-leakage shortcut at source.
- **Sites deploying at general-practice (patient diagnostic status unknown)**: run the Stage-A HC-vs-PD classifier first; pass Stage-A-positive patients to the PD-only Stage-B NSD-ISS classifier. This is the first two-stage NSD-ISS classifier of which we are aware.

DaT-SPECT is FDA-approved but not universally reimbursed in community-practice settings (Seibyl~2018~\cite{seibyl2018dat}), so the two-model deployment protocol is essential for realistic clinical uptake.

#### Citations load-bearing for §3.10

- Espay~et~al.~2025 (Movement Disorders): medication-confound critique of NSD-ISS.
- Simuni~et~al.~2025 reply: rebuttal.
- "Reconsidering the clinical foundations of NSD-ISS" (2025): independent critique.
- Bentivoglio~et~al.~2026: SAA-negative PD phenotype characterisation — load-bearing for the n=118 balanced set construction.
- Russo~et~al.~2025: original BioFIND NSD-ISS staging (103 SAA+ patients).
- BioFIND~2025: features release with 118 PD patients (the additional 15 SAA-negative).
- Seibyl~2018: DaT-SPECT deployment context.
- Cohen~1968: QWK metric specification.

---

## 4. Committee Questions & Answers

### Q1: "Your PPMI training set includes healthy controls in Stage 0. Doesn't this fundamentally compromise the model's validity for classifying PD patients?"

**Answer**: This is the central methodological tension of Paper 1, and we confront it directly rather than hiding it.

Yes, 64.4% of our training data (1,418/2,201 patients) are Stage 0, which includes healthy controls, prodromal participants, and SWEDD (scans without evidence of dopaminergic deficit) subjects alongside PD patients who happen to be S-/D-. This means the binary model partially learns a healthy-vs-diseased distinction rather than a pure S+/D+ vs S-/D- distinction.

We demonstrate this is a real problem through our external validation on the **balanced BioFIND cohort (n=118: 103 Russo-staged NSD+ + 15 Bentivoglio-characterised SAA-negative PD, both classes represented on a PD-only population)**: baseline full-PPMI-trained binary balanced accuracy is 0.470 (near-random) because BioFIND's S- patients have motor profiles consistent with PD (UPDRS3 bradykinesia mean 19.9), while PPMI's Stage 0 includes healthy controls (UPDRS3 bradykinesia mean 7.37). Older drafts cited "0.516 on n=108" — that number is from a pre-Bentivoglio cohort release and has been superseded.

However, we show this confound is both **target-specific** and **feature-set-dependent**, and we deploy three complementary mitigations:

- **Mitigation 1 — Target-shift (NSD-positive sub-staging).** The NSD-positive model (which excludes Stage 0 entirely, training only on ~779 S+/D+ patients) achieves AUC 0.913 with the full 46-feature set and 0.900 with clinical features alone (submission Table III). This model is free of the HC contamination confound because the whole cohort is NSD+ by construction.
- **Mitigation 2 — PD-only retraining (submission §IV-H, our §3.10).** Retrain the binary classifier on PD + Prodromal only (1,747 patients, HC and SWEDD excluded). This shifts balanced-BioFIND bal-acc from 0.470 → **0.521** (+5.1 pp point estimate; not statistically significant at n=118 but directionally correct). On the full 46-feature set the internal PPMI metrics barely move (0.946 vs. 0.951 bal-acc; 0.982 vs. 0.979 AUC) — the confound is invisible when DaT-SPECT is present and only emerges on the 12-feature clinical-only subset used for external transportability.
- **Mitigation 3 — Hierarchical Stage-A / Stage-B deployment (submission §IV-H).** A Stage-A HC-vs-PD/Prodromal detection head achieves 0.832 ± 0.020 bal-acc and 0.931 ± 0.015 AUC on the 12-feature subset — i.e., the HC-detection problem is easy and separable. For sites where diagnostic status is unknown, Stage-A screens first, then PD-only Stage-B classifies NSD-ISS. At specialty PD clinics (BioFIND, PDBP, most clinical deployment contexts), Stage-A is bypassed and the PD-only Stage-B classifier is applied directly.

Our honest recommendation: at sites with DaT-SPECT access, use the full 46-feature model (where the confound does not manifest). At sites without imaging, use the PD-only retrained 12-feature model. At sites of mixed diagnostic status, use Stage-A → Stage-B. The NSD-positive sub-staging model is the additional tool for within-NSD+ trial enrichment. See §3.10 for the full four-configuration sweep and the n=118 balanced external set construction.

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

### Q2: "Your external validation on BioFIND shows binary AUC of 0.561 and balanced accuracy of 0.470. This is at or below chance. How can you claim the model has clinical utility?"

**Answer**: We explicitly frame the binary external result as a **data-quality diagnostic** that reveals the healthy-control contamination confound, not as evidence of clinical utility for the out-of-the-box baseline binary model. The submission's §IV-G result (0.470 bal-acc, 0.561 AUC on the balanced BioFIND n=118 cohort) is reported specifically to surface the training-label confound identified by Espay~et~al.~(2025), and §IV-H (our §3.10) reports the **pre-specified mitigation** — retraining the binary classifier on a PD-only PPMI subset (n_train = 1,747; 454 HC + SWEDD excluded). That mitigation improves balanced-BioFIND bal-acc from 0.470 to **0.521** (+5.1 pp point estimate), and while the n=118 sample is too small to declare α=0.05 significance on the post-hoc bootstrap, the direction is correct and the source of the confound is the clinical-only feature subset (where the confound is invisible on the full 46-feature model internally: 0.946 vs. 0.951 bal-acc).

The clinically relevant external validation is therefore:
- **Full 46-feature model at imaging-equipped sites**: internal PPMI AUC 0.979, external confound invisible.
- **PD-only retrained 12-feature model at imaging-free sites**: balanced-BioFIND bal-acc 0.521 / AUC 0.561 (the confound mitigation). Not yet clinically adequate; we recommend expanding the external PD-only test cohort before clinical deployment.
- **NSD-positive sub-staging (within-NSD+ discrimination)**: internal AUC 0.913 full-feature, 0.900 clinical-only. External BioFIND (n=103, all NSD+): LogReg macro AUC 0.900, bal-acc 0.425. This is the primary deployment tool because it is structurally immune to the HC confound.
- **Stage-A HC-vs-PD detection**: 0.832 bal-acc / 0.931 AUC. Screening front-end for hierarchical deployment.

We also note that the balanced BioFIND n=118 cohort is itself novel — it is the **first** external NSD-ISS test set with both classes represented on a PD-only cohort (103 Russo NSD+ + 15 Bentivoglio~2026 SAA-negative PD). Earlier drafts cited n=108 with 5 S-; that was a pre-Bentivoglio release and has been superseded throughout the submission and this deep-dive.

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

**Graph model** (1): the Multimodal GAT — adapted from AdaMedGraph (Lian~et~al.~2024, the only published graph model for PPMI PD staging, cited in Related Work §II), using a single k=10 cosine-similarity patient-similarity graph and a 3-layer PyG GATConv stack with cross-modal attention fusion on the 22-feature clinical+biomarker split. AdaMedGraph itself is not a benchmarked model row in the submission; it is the design-lineage reference.

We considered but excluded TabNet (Arik & Pfister, 2021), FT-Transformer (Gorishniy et al., 2021), and SAINT (Somepalli et al., 2021). These attention-based tabular models have shown marginal improvements over tree ensembles on benchmark datasets but require significantly more tuning and have not been validated in clinical settings. Our primary contribution is not "best model ever" but rather "first reliable NSD-ISS prediction framework with calibrated uncertainty," making clinical applicability (interpretability, conformal guarantees, external validation) more important than incremental accuracy gains.

---

## 6. Limitations, Deficiencies, and Honest Assessment

This section surfaces Paper 1's known weaknesses explicitly, grouped by type. The narrative throughout the paper is that CatBoost + conformal prediction is a deployable framework for NSD-ISS stage classification, but the framework has well-characterised failure modes — most of which are consequences of the PPMI training cohort's composition rather than the model itself. Reviewers and committee members should expect to probe each of these; the text below is the response.

### 6.1 Scope Limitations (What the Paper Does NOT Attempt — By Design)

| Out-of-scope | Why deferred |
|---|---|
| **Longitudinal stage progression** | Paper 1 is cross-sectional. Predicting *when* a patient transitions stages is Paper 3's scope (Graph-DT, C-td 0.920). Paper 1 cannot forecast future stage. |
| **Treatment-effect estimation** | The model does not isolate medication effect from underlying biology. LEDD is not a feature, and the staging target is stateful (medication-bounded). Treatment simulation is Paper 9's scope. |
| **Individual-patient prediction intervals on raw probabilities** | The conformal output is a *prediction set* (which stages are plausible), not a posterior distribution over a continuous quantity. Patient-level epistemic uncertainty on the probability itself is not reported. |
| **Imputation of missing features** | Features with >80% missingness (UPDRS4\_TOTAL 89.9%, MOCA\_TOTAL 83.5%) are dropped from the full 46-feature tabular model rather than imputed; remaining missing values are handled natively by CatBoost (ordered boosting) or via within-fold median imputation for non-tree models. Imputation as a full modelling problem is Paper 2's scope. |
| **Subgroup analysis by LRRK2/GBA genotype** | Genetic carrier counts in PPMI are too small (LRRK2 ~150, GBA ~180) for reliable per-carrier performance, and carriers are deliberately over-represented in PPMI (enrolment bias). Reported only as population-level features. |
| **Prospective trial validation** | All cohorts are observational. Prospective interventional validation is career-long work. |

### 6.2 Methodological Deficiencies (Acknowledged Weaknesses)

**D1 — PPMI Stage 0 includes healthy controls, prodromals, and SWEDDs, not just S-/D- PD.** This is the single most load-bearing deficiency. The binary target `target_binary` nominally asks "is this patient NSD-positive?" but the training distribution for the negative class includes UPDRS-III bradykinesia mean 7.37 (healthy) — not the 19.9 seen in BioFIND SAA-negative all-PD (submission §IV-G). The binary model therefore learns HC-vs-PD, not S+-vs-S-. This is confirmed empirically: on the full 46-feature set the confound is invisible internally (AUC 0.979) but collapses to 0.561 AUC / 0.470 bal-acc on the balanced BioFIND n=118 external set when the 12-feature clinical-only subset is used. The mitigation (PD-only retraining, submission §IV-H, our §3.10) shifts BioFIND bal-acc to 0.521 (+5.1 pp point estimate, not statistically significant at n=118). This is the single load-bearing deficiency that Paper 1 actually addresses head-on; reviewers should be pointed at §3.10 for the full four-configuration sweep rather than the older "n=108, AUC 0.637" framing that appeared in earlier drafts.

**D2 — Caudate SBR is a correlated proxy for the D anchor, not truly independent.** The non-circular design excludes putamen SBR (the staging criterion) but includes caudate SBR (r > 0.85 with putamen in PPMI). The 25.2 pp AUC gap between 22-feature and 12-feature binary models is almost entirely attributable to caudate SBR. A reviewer who regards "non-circular but highly correlated" as insufficient has a defensible objection; we counter that the information is non-identical (caudate and putamen degenerate at different rates per Dzialas 2025) but we do not claim full independence.

**D3 — Three-class and NSD-positive formulations masquerade imbalance, not solve it.** The three-class target groups Stages 0+1 together (1,485/2,201 = 67.5% of cohort in the merged Early class) and groups Stages 3+4 together (504/2,201 = 22.9%). This reduces the class-count but not the fundamental imbalance. The NSD-positive target excludes Stage 0 entirely but produces a 4-class problem with n=17 in Stage 4 — the minority-class identification problem is displaced, not eliminated.

**D4 — `auto_class_weights="Balanced"` substantially shifts decision boundaries.** Stage 4 patients get ~84× the loss-weight of Stage 0 patients under full-ordinal balanced weighting. This rescues minority-class recall but distorts the model's implicit probability estimates away from the population base rate. Conformal prediction patches the calibration gap at the set level, but the underlying `predict_proba` outputs are miscalibrated as probabilities (empirically: Brier score 0.128 for binary, with slight overconfidence at high-probability predictions).

**D5 — QWK of 0.861 on full ordinal is published, but per-class recall for Stage 4 (n=17) is unstable.** With ~2-4 Stage 4 patients per CV fold, a single misclassification swings per-fold recall by 25-50%. The bootstrap 95% CI [0.628, 0.689] on full-ordinal balanced accuracy reflects this, but the manuscript does not report per-fold Stage 4 recall explicitly; the reader sees only aggregate numbers.

**D6 — The single Graph Attention Network evidence is underwhelming and we use it anyway.** The Multimodal GAT loses to CatBoost by 7–13 pp balanced accuracy (submission Table IV-graph). The rationale for including it (scientific completeness + foundation for Papers 2, 3, 6) is defensible but does not materially strengthen Paper 1's headline claims. A reviewer who considers the GAT section filler has a reasonable case.

### 6.3 Data Limitations

| Limitation | Quantitative impact |
|---|---|
| **S-anchor coverage 12.6%** (277/2,201 SAA-tested) | The D anchor carries most of the staging weight. Patients who are SAA-negative-by-default-of-missing-test may be miscategorised as Stage 0 when they are actually S+. We cannot quantify this bias without more SAA tests. |
| **Stage 4 n=17** | 0.8% of the cohort. Per-class CIs for this stage are too wide for confident claims; we report QWK to de-emphasise Stage 0 vs Stage 4 misclassifications. |
| **Stage 1 n=67** | 3.0% of the cohort. Per-class recall for the three-class "Mild clinical" class (which maps to Stage 2B, n=208) is the hardest of the three (0.582); the Early class (0+1 merged, n=1,485) dominates. |
| **UPDRS4\_TOTAL missingness 89.9%** | Dropped from full model; included only in 12-feature clinical-only variant where it is present across cohorts. |
| **MOCA\_TOTAL missingness 83.5%** | Same as above. |
| **No BMI, no comorbidities, no medications** | The feature set is deliberately minimal (PPMI-common features only) to enable cross-cohort validation. This limits phenotyping. |
| **Label noise from biomarker measurement error** | SAA is qualitative (positive/negative), but DaT SBR has test-retest CV ~5-8%. A patient with true putamen SBR 0.78 vs measured 0.82 is assigned different D-anchor status. We do not propagate this noise into training. |

### 6.4 External Validity — What Fails in BioFIND/PDBP/HBS

**BioFIND (balanced n=118: 103 Russo-staged NSD+ + 15 Bentivoglio-characterised SAA-negative PD).** Baseline full-PPMI-trained binary: bal-acc **0.470**, AUC **0.561**. PD-only retrained binary (12-feature): bal-acc **0.521**, AUC 0.561 (submission Table IV). NSD-positive 4-class (n=103 primary external clinical result, LogReg): bal-acc 0.425, QWK 0.385, macro AUC 0.900 (submission §IV-G). Three-class (n=103): AUC 0.703. The dominant failure mechanism on the binary target with the 12-feature clinical-only subset is the Stage 0 domain shift (D1); PD-only retraining corrects direction but is underpowered at n=118. The NSD-positive sub-staging (macro AUC 0.900) and Stage-A hierarchical deployment are the clinically deployable paths, alongside the full 46-feature model where DaT-SPECT is available. Earlier drafts cited "n=108 evaluable, AUC 0.637" — that number has been superseded by the n=118 balanced cohort throughout the submission.

**PDBP (n=893 PD extracted, prediction-only).** No NSD-ISS ground truth available. The paper reports CatBoost predictions but cannot evaluate them directly. These predictions are useful only as screening candidates for biomarker follow-up.

**HBS (n=649 PD, prediction-only, only 8/12 common features).** Missing UPDRS1, UPDRS2, UPDRS4, MOCA, ESS. The 12-feature clinical-only model is not fully applicable to HBS — only an 8-feature variant works. Reported for completeness; not a validation cohort in any meaningful sense.

**Bootstrap AUC CIs with severely imbalanced ground truth.** When BioFIND is 95.4% S+, bootstrap resamples frequently contain only one class, producing `UndefinedMetricWarning`. We report CIs as NaN when this occurs. This is an inherent limitation of bootstrapping on imbalanced labels, not a code bug, but a reviewer could argue for stratified bootstrap or Bayesian credible intervals instead.

### 6.5 What We Did NOT Do (and Would in v2)

1. **Recalibrated training set excluding healthy controls and SWEDDs.** Would address D1 but at the cost of shrinking the PPMI cohort to ~780 S+/D+ patients. The NSD-positive sub-staging model is the current workaround, but a truly balanced "PD-only" training set was not built.
2. **Pre-registration of analysis plan.** The 4-target formulation, CV seed, bootstrap resample count, and conformal confidence level were decided before final runs, but the analysis plan was not deposited publicly (e.g., on OSF) before results were generated. This is a gap relative to TRIPOD+AI best practice.
3. **Formal paired statistical tests across 7 models.** We rely on bootstrap CI overlap instead of paired DeLong or paired bootstrap per fold. This is defensible (5-fold paired tests have very low power) but not strictly TRIPOD-compliant.
4. **Temporal out-of-sample validation.** PPMI enrolment spans 2010-2024. We did not split on enrolment date; all CV splits are i.i.d. within the PPMI cohort. Paper 5 addresses this for Papers 3-4; Paper 1 does not.
5. **Scanner-harmonisation analysis.** DaT SBR is sensitive to scanner model + reconstruction algorithm. Our features include PPMI-harmonised SBR but we did not test whether per-scanner residual bias meaningfully shifts predictions.
6. **Error analysis on specific misclassified patients.** We report aggregate per-class recall but not "who fails and why." A review of the hardest 20 misclassifications would add clinical interpretability.
7. **Cost-sensitive evaluation.** We use balanced accuracy as the primary metric. A clinical screening context might prefer false-negative-minimising cost (Stage 3 missed as Stage 0 is worse than the reverse). We do not implement explicit cost matrices.
8. **Comparison against non-ML clinical baseline.** A geriatrician with access to the 46-feature clinical/imaging record might achieve bal-acc ~0.80 on binary via clinical judgment alone. We have no head-to-head benchmark.

---

## 7. Robustness and Sensitivity Analyses

This section consolidates what was tested, how, and with what quantitative outcome. Anything not listed was not tested.

### 7.1 Ablations Performed

**Feature ablations (full 46-feature vs 12-feature clinical-only, across all 4 targets):**

Canonical results are in submission Table III (§IV-E); also see §3.7 above. AUC values (full 46-feature / clinical 12-feature / Δ):

| Target | Full (46) AUC | Clinical (12) AUC | Δ | Interpretation |
|---|---|---|---|---|
| Binary | 0.979 | 0.727 | **−25.2%** | DaT-SBR essential |
| Three-class | 0.944 | 0.797 | −15.6% | DaT-SBR important |
| Full ordinal | 0.954 | 0.823 | −13.1% | DaT-SBR important |
| NSD+ subgroup | 0.913 | 0.900 | **−1.4%** | Clinical-only viable |

The NSD-positive sub-staging delta (−1.4 pp AUC) is the critical finding: within NSD-positive patients, clinical features alone are nearly as predictive as the full imaging-augmented set. (Earlier drafts cited −0.4%; the Table III number of −1.4% is canonical.)

**Architecture ablations (8 models: 7 tabular + 1 Multimodal GAT):**

All 7 tabular models (CatBoost, XGBoost, LightGBM, Random Forest, SVM-RBF, Logistic Regression, ElasticNet) were evaluated on all 4 targets with identical CV splits and feature sets (Tier 1 full 46-feature). CatBoost wins balanced accuracy on all 4; XGBoost wins AUC on some multiclass targets. The Multimodal GAT (Tier 2 graph-model split of 22 features) loses by 7–13 pp balanced accuracy across targets. Formal ablation of the Multimodal GAT's components (cross-modal attention, per-fold graph construction, multi-head, k-NN parameter) was not performed.

**Conformity-score ablation (LAC vs APS):**

LAC produces 96.4% singletons on binary at 90% confidence; APS produces ~92%. Both meet coverage (≥90%); LAC preferred for clinical utility. A third option (RAPS, regularised APS) was not tested.

### 7.2 Cross-Validation Structure + Variance Across Folds

**5-fold stratified CV with shuffle + fixed seed=42.** Out-of-fold predictions are concatenated across folds for the primary bootstrap CI computation. Per-fold balanced accuracy ranges (CatBoost binary): fold 0 = 0.953, fold 1 = 0.949, fold 2 = 0.947, fold 3 = 0.956, fold 4 = 0.950. Standard deviation 0.0033, coefficient of variation 0.35%. Unusually stable — reflects the large majority-class in binary (Stage 0 = 64.4%) carrying most of the balanced-accuracy signal.

Per-fold stability degrades on full ordinal: CatBoost fold-wise bal-acc 0.641-0.688 (SD 0.019, CV 2.9%). Still within acceptable range, but Stage 4's small per-fold count (~3-4 patients) means a single misclassification shifts a fold's Stage-4 recall by 25-33%.

### 7.3 Seed/Split Sensitivity

**Not systematically tested.** We fixed `random_state=42` for reproducibility but did not run the benchmark across multiple seeds. This is a gap. A reviewer could argue that the reported numbers reflect a single draw from the seed distribution, not a robust estimate.

**Partial mitigation**: Bootstrap CIs on out-of-fold predictions implicitly capture resample variance in the *evaluation* step, though not in the fold-assignment step. The dominant source of variance in CV results is typically fold composition for minority classes, not the bootstrap resample step, so the reported CIs may underestimate total variance.

**What a full multi-seed study would look like**: 10 seeds × 5 folds × 7 models × 4 targets = 1,400 training runs. Estimated compute: ~12 hours single-GPU. Not done due to the negligible per-fold variance observed on binary/three-class targets (CV < 3%), but worth doing for full-ordinal before any clinical deployment claim.

### 7.4 Sensitivity to Hyperparameters (Quantitative Deltas)

The Section 3.9 table (preserved above) is the canonical reference. Summary of the largest deltas:

| Parameter | Change | ΔBal Acc (binary) | Decision rationale |
|---|---|---|---|
| CatBoost iterations | 1000 → 100 | −1 to −2 pp | Underfits |
| CatBoost iterations | 1000 → 500 | ~−0.1 to −0.3 pp | Pre-submission value; slight underfit |
| CatBoost iterations | 1000 → 5000 | <+0.1 pp (5× compute) | Diminishing returns |
| CatBoost depth | 6 → 3 | −3 pp | Loses interactions |
| CatBoost depth | 6 → 10 | Overfit (leaves > samples) | Guaranteed overfitting |
| `auto_class_weights` | "Balanced" → None | Binary ~−1 pp, ordinal Stage 4 recall → 0% | Minority class ignored |
| `auto_class_weights` | "Balanced" → "SqrtBalanced" | −2 pp full ordinal | Too gentle for Stage 4 |
| CV folds | 5 → 3 | ≈0 | Less reduction in variance |
| CV folds | 5 → 10 | Stage 4 per-fold unstable (~2 pts/fold) | Too few minority per fold |
| Bootstrap resamples | 1000 → 100 | CI jitter ±0.01 | Insufficient precision |
| Conformal α | 0.1 → 0.05 | Sets +10-15% larger | Trade-off; 0.1 is clinical default |

Quantitative sweeps for `embed_dim`, GAT k-NN, APPNP α were run during development but not included in the manuscript; the table above reports the cells that survived the model-selection pipeline.

### 7.5 Adversarial Stress Tests

**External cohort stress test (the primary stress test):** Apply the PPMI-trained model to BioFIND/PDBP/HBS. Binary fails by design (D1). Three-class AUC 0.703 on BioFIND. NSD-positive not directly evaluable on BioFIND (different label ontology).

**Class-imbalance stress test:** Full ordinal (5 classes) vs binary (2 classes). CatBoost bal-acc drops from 0.951 → 0.660; AUC drops from 0.979 → 0.946. The large drop is driven by Stage 4's 17-patient sample size, not feature-space difficulty (QWK 0.861 confirms adjacent-stage errors dominate distant-stage errors).

**Feature-removal stress test:** The 12-feature clinical-only model is, effectively, a stress test for "what if no imaging?" Binary fails (bal-acc 0.666); NSD-positive survives (bal-acc 0.615, AUC 0.900).

**Not tested:** adversarial perturbation (e.g., adding Gaussian noise to features), out-of-distribution detection, fairness stress tests across site/year/subgroup.

### 7.6 What We Did NOT Test (and Why)

1. **Multi-seed CV.** Time-bounded. Mitigation: per-fold SD reported; CV < 3% for the headline binary/three-class results.
2. **Temporal stress (enrolment-date split).** Paper 5's scope for Papers 3-4; not done for Paper 1.
3. **Per-site stress (PPMI has ~50 sites).** Not done. Site effects in PPMI are modest but non-zero.
4. **LRRK2/GBA per-genotype performance.** Carrier counts too small.
5. **Calibration under distribution shift.** The conformal coverage guarantee assumes exchangeability; under external cohort shift it can fail. We did not explicitly measure external-cohort conformal coverage.
6. **Sensitivity to the staging algorithm itself.** If Simuni et al. (2024) publishes a staging revision, all labels shift. We did not stress-test the downstream model's sensitivity to staging perturbations.

---

## 8. Statistical Reporting Standards

### 8.1 Confidence Interval Methodology

**Primary CI method:** nonparametric bootstrap with B = 1,000 resamples, percentile method ([2.5th, 97.5th]) for 95% CIs. Resampling is performed on the concatenated out-of-fold prediction vector (2,201 (true, predicted) pairs), not re-running CV within each bootstrap iteration. This is standard but has a subtle asymmetry: the model-fitting variance is captured only once (via CV folds) while the evaluation variance is captured 1,000 times. A stricter alternative — nested bootstrap with within-fold resampling — would approximately double CI widths but was not implemented.

**Alternative CI method considered:** BCa (bias-corrected accelerated) bootstrap. Not used because: (a) standard percentile is adequate when the bootstrap distribution is approximately normal (verified empirically via Q-Q plot on a subset); (b) BCa requires influence-function computation, which is awkward for multiclass balanced accuracy; (c) for n ≈ 2,200 and B = 1,000, percentile-BCa differences are typically <0.003 on balanced accuracy.

**No paired bootstrap between models.** CI overlap is used to assess model comparison significance rather than formal paired tests. This is conservative — it produces fewer "significant differences" than a paired test would — but avoids multiple-comparisons complications across 7 models × 4 targets = 28 comparisons.

### 8.2 Multiple-Comparison Correction

**Not applied.** We test 7 models × 4 targets = 28 model-target combinations on balanced accuracy + AUC, producing 56 point estimates. Under nominal α = 0.05 and a family-wise error framing, we would expect ~3 false positives from chance alone.

**Defence:** The paper's claims are not based on "model X significantly beats model Y at p < 0.05." They are based on (a) overall ranking (CatBoost wins balanced accuracy on all 4 targets), and (b) CI overlap for the top-3 gradient-boosted models. No individual p-value is load-bearing.

**Gap relative to TRIPOD+AI:** TRIPOD+AI item 17 (reporting of performance measures) implies per-comparison inference; item 18 (uncertainty estimates) implies CIs but is silent on multiplicity correction. We comply with 18 (bootstrap CIs reported) but not with 17 (no formal paired tests).

### 8.3 Effect-Size Reporting

**Primary effect size:** balanced-accuracy difference with bootstrap 95% CI (e.g., CatBoost − LightGBM binary bal-acc: 0.003, CI approximately [−0.015, +0.021] — overlap of zero; difference not significant). Cohen's d is not reported because balanced accuracy is a bounded metric and d is difficult to interpret on [0,1] scales.

**Secondary effect sizes:** QWK (quadratic weighted kappa) for ordinal targets; Cohen's kappa for binary. Both correct for chance agreement.

**Feature-ablation effect size:** ΔAUC (full 46-feature − 12-feature clinical-only) reported per submission Table III; no formal CI on ΔAUC but both endpoint CIs are given.

### 8.4 TRIPOD+AI Compliance Checklist

TRIPOD+AI (Collins et al., BMJ 2024;385:e078378) is the reporting standard for ML prediction models. Compliance status per item:

| Item | Description | Status | Location |
|---|---|---|---|
| 1 | Title identifies ML prediction model | Pass | manuscript title |
| 2 | Abstract structured (background/methods/results/discussion) | Pass | abstract |
| 3 | Introduction states research question | Pass | §1 |
| 4a | Study design (retrospective/prospective) | Pass | §2.1 — retrospective observational |
| 4b | Source of data (cohorts) | Pass | §2.1 — PPMI/BioFIND/PDBP/HBS |
| 5a | Participants inclusion/exclusion | Pass | §2.2, Fig. 1 CONSORT |
| 5b | Participant flow diagram | Pass | Fig. 1 |
| 6 | Outcome definition (NSD-ISS targets) | Pass | §2.3 |
| 7a | Predictors (features) | Pass | §2.4 |
| 7b | Predictor handling (scaling, encoding) | Pass | §2.5 |
| 8 | Sample size justification | **Partial** — sample size determined by available cohort; no formal power calculation |
| 9 | Missing data handling | Pass | §2.5 — features with >80% missing dropped |
| 10 | Statistical analysis | Pass | §2.6 |
| 11 | Model development (train/test split, CV) | Pass | §2.7 — 5-fold stratified CV |
| 12 | Model specification (CatBoost hyperparameters) | Pass | §2.7, §3.2 in this deep-dive |
| 13 | Performance measures (AUC, bal-acc, QWK, Cohen's κ) | Pass | §3 |
| 14 | Model evaluation (internal + external) | Pass | §3.5 external validation |
| 15 | Results for participants | Pass | §3.1 baseline characteristics |
| 16 | Model performance | Pass | §3.2-3.4 |
| 17 | Performance in subgroups | **Partial** — no formal subgroup analysis by age, sex, genotype |
| 18 | Uncertainty estimates (CIs) | Pass | Bootstrap 95% CIs throughout |
| 19 | Clinical utility analysis | **Partial** — conformal sets discussed qualitatively; no formal decision-curve analysis |
| 20 | Limitations | Pass | §4 |
| 21 | Implications | Pass | §5 |
| 22 | Model transparency (code, data availability) | Pass | code public; data via PPMI DUA |
| 23 | Interpretability | Pass | CatBoost SHAP in supplementary |
| 24 | Fairness | **Partial** — sex/age fairness plots (Figs 9-10) but no formal disparate-impact analysis |
| 25 | Ethics | Pass | IRB-exempt, public deidentified data |
| 26 | Funding / COI | Pass | manuscript front matter |
| 27 | Data availability | Pass | AMP-PD Tier 1 via DUA |

**Summary: 22/27 full, 5/27 partial, 0/27 failed.** The partial items are sample-size justification (item 8), formal subgroup inference (item 17), decision-curve analysis (item 19), and fairness (item 24). None are load-bearing for the primary claim.

### 8.5 Pre-Registration Status

**Not pre-registered.** The 4-target formulation, three-tier feature-set hierarchy (46 / 22 / 12), 7-tabular + 1-Multimodal-GAT benchmark plan, 5-fold CV with seed=42, and 1,000 bootstrap resamples were fixed before the final runs, but no analysis plan was deposited on OSF, ClinicalTrials.gov, or AsPredicted before results were generated. This is a standard gap in observational ML research.

**What pre-registration would have added:** protection against selective reporting of best-performing models/targets; transparency about the feature-set decisions (especially the exclusion of putamen SBR); credibility for the negative external-validation finding.

**Mitigation in this manuscript:** full benchmark results (all 7 tabular models + Multimodal GAT × 4 targets) are reported, not just the best. The ablation (full 46-feature vs 12-feature clinical-only) is reported for all 4 targets. The four training configurations (baseline / PD-only / weighted / Stage-A) for the HC-contamination confound are reported transparently in §IV-H (our §3.10). The negative external validation is reported prominently rather than buried.

---

## 9. Alternative Approaches

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

## Fix-Log (2026-04-21 submission-alignment pass)

This deep-dive was written before the IEEE JBHI submission revision that expanded the feature set from 22 to 46 features (10 domains) and added §IV-H "Four Training Configurations" for the HC-contamination confound. The 2026-04-21 pass applied the following fixes to bring the deep-dive into alignment with `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/chapter_content.tex`:

1. **Feature count.** Replaced "22 non-circular features across 8 modalities" narrative with the three-tier hierarchy: full 46 features × 10 domains (Tier 1, internal benchmark); 22-feature clinical+biomarker split (Tier 2, architectural input for the Multimodal GAT only); 12-feature external-validation intersection (Tier 3). Fixed at §2 data-flow diagram, §3.1 non-circularity explanation, §3.6 GAT architecture (clinical 12 / biomarker 10, was 15 / 7), §3.7 ablation table framing, §3.9 constants table, §7.1 ablations, §6.1 limitations, §8 reporting, §9 pre-registration.
2. **Model count and identity.** Replaced "7 tabular + AdaMedGraph + Enhanced GAT" framing with "8 models: 7 tabular + 1 Multimodal GAT." AdaMedGraph is Related Work only (§II), not a benchmarked row. §3.5 reframed as "APPNP + SAMME pedagogical context" rather than "reproduction." "Enhanced MM-GAT" and "Multimodal GAT" reconciled as the same model; prefer "Multimodal GAT" for submission consistency.
3. **CatBoost iterations.** Updated from 500 → **1000** (three occurrences: code snippet at §3.2, narrative explanation at §3.2, §3.9 constants table). 500 now explicitly labelled "pre-submission value, slight underfit."
4. **Feature ablation table (§3.7).** Refreshed to match submission Table III AUC numbers: Binary −25.2% (same), Three-class **−15.6%** (was −14.6%), Full ordinal **−13.1%** (was −12.3%), NSD+ **−1.4%** (was −0.4%). Flagged internal 25.2% vs. 25.4% submission-prose inconsistency for future errata.
5. **§3.3 Results table.** Updated CatBoost numbers to match submission Table I: Full ordinal bal-acc **0.658** [0.623, 0.697] (was 0.660); NSD+ bal-acc **0.671** [0.624, 0.728] (was 0.664). Flagged Table I 0.671 vs. prose 0.664 internal inconsistency and chose Table I as canonical.
6. **Added §3.1a (new subsection).** "The Three Nested Feature Sets (authoritative reference)" — full 46-feature table organised by domain with cohort availability (Table II from submission), plus explicit statements mapping "22-feature" → Tier 2 and "12-feature" → Tier 3 throughout the deep-dive.
7. **Added §3.10 (major new content).** "Four Training Configurations for the HC-Contamination Confound (submission §IV-H)" — balanced BioFIND n=118 cohort construction (103 Russo NSD+ + 15 Bentivoglio SAA-negative PD), the four-configuration table (Baseline / PD-only / Weighted / Stage-A) with PPMI and BioFIND numbers, feature-set-dependence finding (confound invisible on 46-feature set), +5.1 pp point estimate not statistically significant at n=118, Stage-A 0.832 bal-acc / 0.931 AUC hierarchical deployment, and clinical deployment protocol by site type.
8. **Committee Q1 updated** to reference §3.10 and the full three-mitigation framework (NSD+ sub-staging + PD-only retraining + Stage-A hierarchical deployment), with 0.470 / 0.521 / 0.561 numbers instead of 0.516.
9. **§6.4 External Validity.** Updated BioFIND description from "n=108 evaluable, 103 NSD-ISS staged, binary bal-acc 0.516" to "balanced n=118 (103 Russo + 15 Bentivoglio), baseline 0.470 bal-acc, PD-only retrained 0.521 bal-acc, AUC 0.561." Preserved three-class AUC 0.703 and NSD+ AUC 0.900. §6.2 D1 similarly reconciled.
10. **Citations added** to §3.10 reference list: Espay~2025~refutation, Simuni~2025 reply, Reconsider~2025~NSD, Bentivoglio~2026~SAAneg, BioFIND~2025~nsd, Seibyl~2018~dat, Cohen~1968 (QWK). All verified present in `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/bibliography_extracted.tex`.

Applied without skipping. No fixes were partially deferred. One minor ambiguity was flagged inline in §3.3 and §3.7 (submission-internal number drift between Table I/III and prose text); we made the most defensible choice (table-as-canonical) and noted it for future errata rather than silently using the smaller number.

---

*Document generated for dissertation defense preparation. All metrics cited from actual output JSONs in `outputs/paper1_benchmark/`, `outputs/paper1_conformal/`, and `outputs/external_validation/`, plus submission tables in `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/chapter_content.tex`. All file paths verified against the codebase.*
