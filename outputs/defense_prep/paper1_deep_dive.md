---
Last substantive update: 2026-04-12
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

**Critical non-circular design**: The features used for staging (putamen SBR, UPDRS-III total, SAA) are **excluded** from the ML feature set. The 22 ML features use caudate SBR (not putamen), UPDRS-III subscales (not total), and never use SAA.

**Why is this non-circularity so important?** If we used putamen SBR as both a staging criterion AND an ML feature, the model could trivially learn "if putamen SBR < 0.80, predict NSD-positive" -- it would be memorizing the staging rule, not learning biology. That model would appear to have high accuracy but would provide zero clinical value beyond what the staging algorithm already gives. By using caudate SBR (a correlated but distinct brain region), we force the model to learn genuine biological relationships. Caudate SBR correlates with putamen SBR at r > 0.85 in PPMI, so the signal is still present -- the model just can't cheat.

Similarly, UPDRS-III subscales (tremor, rigidity, bradykinesia, axial) provide richer information than the total score used for staging. Two patients can both have UPDRS-III total = 15 but one might have severe tremor with mild rigidity, while another has the reverse. The subscales capture these differences while the total cannot.

### 3.2 Seven-Model Benchmark

**File**: `src/giman_pipeline/sota/nsd_iss_benchmark.py`

#### The Model Factory Pattern (and why sklearn's clone() breaks CatBoost)

```python
factories["catboost"] = lambda: cb.CatBoostClassifier(
    iterations=500, depth=6, auto_class_weights="Balanced", verbose=0
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

**`iterations=500`**: This is the number of boosting rounds -- how many decision trees are built sequentially. Each tree corrects the mistakes of the previous ensemble.

- **Why 500?** We monitored the training loss curve: loss decreases rapidly for the first 100 iterations, continues improving slowly through 300-400, and plateaus by 500. Going to 1000 or 5000 would add training time (~2x and ~10x respectively) with negligible accuracy improvement (<0.1% balanced accuracy change). CatBoost uses an internal learning rate schedule that makes early iterations more impactful.
- **What happens at 100?** The model slightly underfits (~1-2% worse balanced accuracy) because the ensemble hasn't had enough rounds to learn complex feature interactions.
- **What happens at 5000?** Marginal accuracy gain (<0.1%) with risk of overfitting to training noise. CatBoost's built-in L2 regularization (`l2_leaf_reg`, default 3.0) mitigates this, but more iterations still increase computation by 10x.

**`depth=6`**: The maximum depth of each individual decision tree. A tree of depth 6 makes at most 6 sequential yes/no splits, creating up to 2^6 = 64 leaf nodes.

- **What does tree depth mean mechanically?** A depth-1 tree ("stump") asks one question: "Is age > 65?" and has 2 outcomes. A depth-6 tree can ask 6 sequential questions: "Is age > 65? AND is bradykinesia > 20? AND is caudate SBR < 1.0? AND..." -- capturing complex feature interactions. Each additional depth level doubles the possible leaf nodes and allows one more feature interaction.
- **Why depth 6?** Standard for gradient boosting on tabular data (XGBoost default is 6). With 22 features and ~2,000 samples, depth 6 allows up to 6-way feature interactions while keeping each leaf node populated with ~2000/64 = 31 training samples (enough for stable estimates).
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

| Target | CatBoost Bal Acc | CatBoost AUC | Best Alternative | Gap |
|--------|-----------------|-------------|-----------------|-----|
| Binary | 0.951 | 0.979 | LightGBM (0.948/0.977) | +0.3% |
| Three-class | 0.783 | 0.942 | XGBoost (0.767/0.944) | +1.6% |
| Full ordinal | 0.660 | 0.946 | XGBoost (0.635/0.947) | +2.5% |
| NSD-positive | 0.664 | 0.904 | XGBoost (0.654/0.887) | +1.0% |

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

### 3.5 AdaMedGraph Reproduction

**File**: `src/giman_pipeline/models/adamedgraph.py`

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

### 3.6 Enhanced Multimodal GAT

**File**: `scripts/run_enhanced_gat_benchmark.py`

Architecture: Two modality encoders (clinical: 15 features -> 128d; biomarker: 7 features -> 128d) -> 3-layer GATConv (4 heads per layer, concat) per modality -> Cross-modal attention (nn.MultiheadAttention) -> Fusion (256d -> 128d) -> Classification head.

**Why build the graph WITHIN each CV fold?** If you build one graph on the full dataset and then split into folds, patients in the test fold already have edges to training patients. During GAT message passing, test patients receive information from training patients' features -- this is a form of data leakage. Building the graph within each fold means the k-NN computation only sees training patients, ensuring the test set is truly unseen.

**Why undirected + self-loops?** Undirected: if Patient A is similar to Patient B, the relationship is symmetric. Self-loops: each patient should attend to their own features in addition to neighbors'. Without self-loops, a GATConv layer's output for a node is based ONLY on its neighbors, discarding its own features entirely. The `add_self_loops()` function in PyG adds an edge from each node to itself with weight 1.0, ensuring self-information is preserved.

**Results**: Binary bal_acc 0.825 +/- 0.013. Gap to CatBoost: -12.6 pp. This 12.6 pp gap confirms the "trees beat deep learning on tabular data" finding (Grinsztajn et al., NeurIPS 2022) extends to clinical biomarker data.

### 3.7 Feature Ablation: DaT-SBR Is the Critical Feature

| Target | Full 22-feat AUC | Clinical 12-feat AUC | Delta |
|--------|-----------------|---------------------|-------|
| Binary | 0.979 | 0.727 | -25.2% |
| Three-class | 0.942 | 0.797 | -14.6% |
| Full ordinal | 0.946 | 0.823 | -12.3% |
| NSD-positive | 0.904 | 0.900 | -0.4% |

**Why does binary lose 25.2% AUC but NSD-positive loses only 0.4%?** The binary target separates Stage 0 (NSD-negative) from Stages 1+ (NSD-positive). The NSD-ISS definition of NSD-positivity is fundamentally biological: it requires either synuclein pathology (S+) or dopaminergic deficit (D+). Since SAA coverage is only 12.6%, the D anchor (DaT-SPECT) is the primary biological marker for most patients. Caudate SBR is the closest non-circular proxy for the D anchor. Without it, the model can only use clinical symptoms (motor scores, cognition, sleep) to infer biological status -- a much weaker signal. Hence the 25.2% drop.

For NSD-positive sub-staging (discriminating WITHIN stages 1-4), all patients are already confirmed NSD-positive. The question becomes "how impaired is this patient?" not "does this patient have biological disease?" Functional impairment is directly measured by clinical features: motor subscores, cognitive tests, sleep questionnaires. DaT imaging adds almost nothing because DaT SBR varies relatively little within the NSD-positive group (most have clear deficits). Hence the tiny 0.4% drop.

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
| CatBoost iterations | 500 | Number of sequential trees in the ensemble | 100: underfits (-1-2% bal_acc). 1000: marginal gain (<0.1%), 2x training time. 5000: risk of overfitting, 10x training time. |
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
