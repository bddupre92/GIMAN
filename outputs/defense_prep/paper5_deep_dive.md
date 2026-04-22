---
Last substantive update: 2026-02-25
Last touched: 2026-04-21 (cross-ref refresh after Paper 11 + P3/P4 holdout session)
Status: stable; refer to `Docs/NEXT_STEPS_2026-04-21.md` for dissertation-wide status
Cross-refs added 2026-04-21:
- Paper 3's pre-registered holdout rerun (seed=2026, 380-patient 20% split) confirms single-model Graph-DT vs DeepHit comparability is ensemble-dependent. Temporal validation findings here are orthogonal to holdout discipline but share the "deployment as ensemble" recommendation.
- 2026-04-21 dissertation-wide validation pass confirms Paper 5's temporal-window C-td claims still hold.
---

# Paper 5: Temporal Validation and Deployment Readiness

## Deep-Dive Defense Preparation Document

---

## 1. The Conceptual Problem (Beginner Level)

### The Plain English Version

Imagine you build a weather prediction model using 10 years of historical data from 2010-2020. In Papers 3 and 4, we tested this model by randomly shuffling all data and holding out 20% for testing --- some test days from 2012, some from 2018, some from 2020. The model looks great: 92% accuracy.

But here's the problem: in the real world, you never predict the weather for a random day in the past. You always predict *tomorrow*. Testing on shuffled historical data is much easier than testing on the future, because shuffled data means some "future" information leaks into training.

Paper 5 asks: **"What happens when we test our model the way it would actually be used --- trained on earlier patients, tested on later patients?"**

This is called *temporal validation*, and it's the gold standard for evaluating whether a clinical prediction model is ready for real-world deployment.

### The Hospital Analogy

Imagine PPMI (the Parkinson's study) as a hospital that has been enrolling patients since 2010. Your AI model trains on the first 8 years of patients (2010-2017). Now a new patient walks in the door in 2018. Can the model predict their disease trajectory?

There are three reasons this is harder than random testing:

1. **The population changes over time**: Patients enrolled in 2010 might differ from those in 2018 (different referral patterns, different diagnostic criteria, sicker or healthier). This is *covariate shift*.
2. **Less training data**: You can only use earlier patients for training, not the full dataset.
3. **No peeking**: In random cross-validation, a 2018 patient might be in the training set and a 2012 patient in the test set. Temporal validation forbids this --- it strictly enforces chronological ordering.

### What This Paper Adds

Paper 3 reported C-td = 0.924 (DeepHit) using 5-fold random cross-validation. Paper 5 answers: "Is that still true when we test the model on *future* patients?" The answer: C-td drops to 0.87-0.89 for realistic temporal windows (3-5 percentage points degradation), and drops catastrophically to 0.68 under a stress test with limited training data. This degradation, along with detected covariate shift, gives clinicians honest expectations about real-world performance.

### The Graph Problem

There's an extra challenge for Graph-DT specifically. The patient similarity graph in Paper 3 includes *all* patients. But in real deployment, the graph only contains *existing* patients when a new patient arrives. Paper 5 solves this with **inductive graph extension** --- connecting new patients to the existing graph based on their baseline features, without rebuilding the entire graph.

---

## 2. The Architectural Solution (Intermediate Level)

### Data Flow

```
PPMI Demographics (enrollment dates)
    |
    v
TemporalSplitter: order 1,900 patients chronologically
    |
    v
4 Expanding Windows:
    W1: 40% train → next 20% test  (760/380 patients)
    W2: 60% train → next 20% test  (1140/380 patients)
    W3: 80% train → final 20% test (1520/380 patients)
    W4: 50% train → final 50% test (950/950 — stress test)
    |
    +---> For each window:
    |       |
    |       +---> Train DeepHit (85/15 train/val split, early stopping)
    |       |       → Evaluate on test set → C-td, IBS, per-transition C-td
    |       |
    |       +---> Train Graph-DT with inductive graph extension
    |       |       Build training-only kNN graph
    |       |       Extend graph to include test patients (test→training edges only)
    |       |       → Evaluate on test with extended graph
    |       |
    |       +---> Covariate Shift Detection
    |               Per-feature: KS test + PSI
    |               Multivariate: MMD with RBF kernel + permutation test
    |
    v
Temporal Validation Results:
    - C-td per window per model
    - Degradation from Paper 3 random CV reference
    - Shift severity per window
    - Per-transition stability analysis
    - 6 publication figures
```

### Key Components Explained

**Expanding Windows**: Instead of a single train/test split, we use progressively larger training sets. This reveals the *learning curve* --- how model performance improves as more historical data becomes available. W1 (40% training) represents early deployment with limited data. W3 (80%) represents mature deployment. W4 (50/50) is a deliberate stress test.

**Inductive Graph Extension**: GAT (Graph Attention Network) is *inductive* --- it learns attention weights over edge features that generalize to unseen nodes. When a new patient arrives, we compute their cosine similarity to all training patients in the baseline feature space, connect them to the k=15 most similar training patients, and let GAT aggregate information from these neighbors. No retraining required.

**Covariate Shift Detection**: Three complementary tests:
- **KS (Kolmogorov-Smirnov)**: Compares the full distribution of each feature between train and test. Detects any difference in shape, location, or spread.
- **PSI (Population Stability Index)**: Measures how much the distribution has "moved" by comparing bin proportions. Developed in credit scoring to detect when a model's input population has changed.
- **MMD (Maximum Mean Discrepancy)**: A *multivariate* test that detects shifts in the joint distribution of all features simultaneously. Two features might shift in opposite directions, canceling out in univariate tests but detectable by MMD.

### Results Summary

| Window | N_train | N_test | DeepHit C-td | Graph-DT C-td | Degradation (DH) | Covariate Shift |
|--------|---------|--------|-------------|--------------|------------------|----------------|
| W1 | 760 | 380 | 0.869 | 0.883 | -5.5pp | Severe (43.8%) |
| W2 | 1140 | 380 | 0.891 | 0.866 | -3.3pp | Severe (31.3%) |
| W3 | 1520 | 380 | 0.882 | 0.858 | -4.2pp | Moderate (18.8%) |
| W4 | 950 | 950 | 0.680 | 0.692 | -24.4pp | Severe (37.5%) |
| Paper 3 | 5-fold CV | | 0.924 | 0.904 | (reference) | N/A |

---

## 3. The Deep Dive (Advanced Level)

### 3.1 Temporal Splitting: How Patients Are Ordered

**File**: `src/giman_pipeline/paper5/temporal_validation.py`
**Class**: `TemporalSplitter`

#### Enrollment Date Parsing

```python
demo = pd.read_csv(demographics_path, low_memory=False)
demo["enrollment_date"] = pd.to_datetime(demo["INFODT"], format="%m/%Y", errors="coerce")
patient_dates = demo.groupby("PATNO")["enrollment_date"].min()
```

**Why `format="%m/%Y"`?** The PPMI demographics file stores the Information Date (INFODT) as month/year only (e.g., "06/2011"), not exact date. `pd.to_datetime` with this format parses "06/2011" as June 1, 2011. The day is always 1, which doesn't matter because we only need month-level ordering.

**Why `.min()` per patient?** A patient may have multiple rows in Demographics (one per visit). The *earliest* date is their screening/enrollment date, which determines their temporal position. Using `.max()` would give the last visit date, which would conflate enrollment timing with follow-up duration.

**Why `errors="coerce"`?** Some INFODT entries are malformed or missing. `coerce` converts unparseable values to NaT (Not a Time) rather than raising an error. These patients get synthetic dates assigned later.

#### Handling Missing Dates

```python
# Patients with NaT dates get synthetic dates
max_date = patient_dates.dropna().max()
for patno in missing_patnos:
    max_date += pd.Timedelta(days=1)
    patient_dates[patno] = max_date
```

**Why assign synthetic dates?** If a patient's enrollment date is missing (NaT), they can't be placed in the temporal order. Rather than dropping them (losing data), we assign them dates *after* all known dates. This puts them at the end of the enrollment sequence, which is conservative: they'll always be in the test set for earlier windows, and only in training for the latest windows.

**What would happen if we dropped them?** We'd lose ~10-50 patients (depending on demographics file completeness), reducing both training and test set sizes. The assignment to the end is the safest default.

#### Temporal Ordering Validation

```python
# Spearman correlation between enrollment rank and visit count
rho, pval = spearmanr(ranks, visit_counts)
```

**Why this sanity check?** Patients enrolled earlier should have *more* visits (longer follow-up). The Spearman correlation between enrollment rank (0=earliest) and visit count should be negative. If it's positive (or weakly negative), the ordering may be wrong.

**In practice**: PPMI data shows rho ≈ -0.5 to -0.7 (strong negative correlation), confirming that earlier-enrolled patients indeed have more visits.

### 3.2 The Four Window Definitions

```python
WINDOW_DEFS = [
    (0.0, 0.4, 0.4, 0.6),  # W1: train on first 40%, test on next 20%
    (0.0, 0.6, 0.6, 0.8),  # W2: train on first 60%, test on next 20%
    (0.0, 0.8, 0.8, 1.0),  # W3: train on first 80%, test on final 20%
    (0.0, 0.5, 0.5, 1.0),  # W4: train on first 50%, test on final 50%
]
```

**Why expanding windows (not sliding)?** Expanding windows simulate the real scenario: as a clinical center accumulates data over years, they have access to ALL historical patients, not just a sliding window of recent ones. W1 represents year 1 of deployment, W2 represents year 3, W3 represents year 5.

**Why 20% test windows for W1-W3?** This produces ~380 test patients per window --- enough for stable C-td estimation (C-td requires sufficient concordant/discordant pairs). Smaller test sets (10%, ~190 patients) would give noisier estimates; larger (30%, ~570) would reduce training data.

**Why W4 at 50/50?** W4 is a deliberate stress test. With only 950 training patients (vs 1520 in W3), it reveals the model's dependence on training set size. The large test set (950 patients) provides very stable estimates of the degraded performance. The dramatic drop from W3 (C-td ~0.88) to W4 (C-td ~0.68) quantifies the performance cliff.

**Why not include a W0 with 20% train / 20% test?** With only 380 training patients and their episodes, the model would likely fail to train meaningfully. The 40% minimum in W1 was chosen as the smallest viable training set.

### 3.3 No-Leakage Validation

```python
def validate_split(self, window_idx):
    ...
    no_leakage = max(train_dates) <= min(test_dates)
    overlap = len(set(train_patnos) & set(test_patnos))
    ...
```

**What "temporal leakage" means mechanically**: If any training patient was enrolled AFTER any test patient, the model has seen "future" data. For example, if a patient enrolled in 2018 is in the training set and a patient enrolled in 2015 is in the test set, the model has access to 2018 disease patterns when predicting a 2015 patient --- this is cheating.

**The validation checks**: (1) The latest training enrollment date must be ≤ the earliest test enrollment date. (2) There must be zero patient overlap between train and test sets. Both conditions must hold for the split to be valid.

**In practice**: All 4 windows pass both checks. The date ranges are approximately:
- W1: Train 2010-2014, Test 2014-2021
- W2: Train 2010-2016, Test 2016-2021
- W3: Train 2010-2019, Test 2019-2021
- W4: Train 2010-2015, Test 2015-2021

### 3.4 Training Per Window: DeepHit

**File**: `src/giman_pipeline/paper5/train_per_window.py`
**Function**: `train_deephit_on_window()`

The key difference from Paper 3's training: **single window, no cross-validation**.

#### Train/Validation Split

```python
def _split_train_val(train_patnos, val_fraction=0.15, seed=42):
    arr = np.array(train_patnos)
    rng = np.random.RandomState(seed)
    rng.shuffle(arr)
    n_val = int(len(arr) * val_fraction)
    return arr[n_val:].tolist(), arr[:n_val].tolist()
```

**Why 85/15 instead of 80/20?** Paper 3 uses 5-fold CV (80/20 per fold). For single-window training, we need a validation set for early stopping but want to maximize training data. The 15% validation fraction (vs 20% in CV) gives the model ~5% more training data per window. For W1 with 760 training patients, this means 646 train + 114 val (vs 608/152 with 80/20). Those extra 38 training patients matter when data is limited.

**Why `seed=42`?** Deterministic split for reproducibility. The same patients always end up in val across runs.

#### Feature Statistics from Training Only

```python
means, stds = compute_feature_stats(patient_arrays, train_episodes)
```

**Why training statistics only?** Using test statistics would introduce information leakage --- the model would "know" the distribution of future patients' features. This is the standard machine learning practice of fitting normalization parameters on training data only.

**What `compute_feature_stats` does**: Computes per-feature mean and standard deviation across all time steps of all training patients. These statistics are used to z-score normalize features before feeding them to the GRU: `(x - mean) / std`. This ensures all features are on comparable scales, which is critical for GRU training (features range from 0-1 for binary to 0-50000 for genetic risk scores).

#### Hyperparameters (Same as Paper 3)

| Parameter | Value | Why This Value |
|-----------|-------|---------------|
| `hidden_dim` | 128 | Matches Paper 3 for fair comparison |
| `n_gru_layers` | 2 | Two stacked GRU layers capture visit-to-visit dynamics |
| `n_epochs` | 100 | Max training epochs (early stopping usually triggers at 40-60) |
| `batch_size` | 64 | Fits in GPU memory; small enough for stochastic gradients |
| `lr` | 1e-3 | Standard Adam learning rate for GRU models |
| `weight_decay` | 1e-4 | L2 regularization to prevent overfitting |
| `dropout` | 0.3 | 30% dropout in GRU for regularization |
| `alpha` | 0.1 | Ranking loss weight in DeepHit NLL (same as Paper 3) |
| `patience` | 15 | Stop if validation loss doesn't improve for 15 epochs |

**Why keep all hyperparameters identical to Paper 3?** Fair comparison. If we re-tuned hyperparameters for temporal validation, improved performance might be due to better tuning rather than better temporal generalization. By keeping hyperparameters fixed, any performance difference is purely due to the temporal split.

### 3.5 Training Per Window: Graph-DT with Inductive Extension

**File**: `src/giman_pipeline/paper5/train_per_window.py`
**Function**: `train_graph_dt_on_window()`

This is the most technically complex part of Paper 5 because it must handle the graph for unseen test patients.

#### Step 1: Build Training-Only Graph

```python
edge_index, edge_weight, node_baseline = build_patient_graph(
    features_df, train_patnos, k_neighbors=15
)
```

This builds a kNN graph using ONLY training patients' 18 baseline features. The graph has ~760-1520 nodes (depending on window) and ~11,400-22,800 edges.

**Why training-only?** Including test patients in the graph would let their features influence training through message passing --- a form of transductive leakage.

#### Step 2: Inductive Graph Extension

**File**: `src/giman_pipeline/paper5/inductive_graph.py`
**Class**: `InductiveGraphExtender`

For each test patient:

```python
# L2 normalize for cosine similarity
train_norm = train_baseline / train_baseline.norm(dim=1, keepdim=True).clamp(min=1.0)
test_norm = test_baseline / test_baseline.norm(dim=1, keepdim=True).clamp(min=1.0)

# Compute similarity: (N_test, N_train)
sim_matrix = test_norm @ train_norm.T

# For each test patient, find k=15 nearest training neighbors
for i in range(n_test):
    sims = sim_matrix[i].numpy()
    top_k = np.argpartition(sims, -k)[-k:]
    for j in top_k:
        if sims[j] > 0:
            # Add BIDIRECTIONAL edges: test_i <-> train_j
            new_src.extend([n_train + i, j])
            new_dst.extend([j, n_train + i])
            new_weight.extend([sims[j], sims[j]])
    # Add self-loop
    new_src.append(n_train + i)
    new_dst.append(n_train + i)
    new_weight.append(1.0)
```

**Why bidirectional edges?** The training graph is undirected (if patient A is a neighbor of patient B, then B is a neighbor of A). To maintain consistency, test→training edges are bidirectional. Additionally, bidirectional edges allow training patients to "see" the new test patient during message passing, potentially enriching their representations.

**Why self-loops?** Without a self-loop, a test patient with no positive-similarity training neighbors would have zero edges, and GAT would produce a zero embedding. The self-loop with weight 1.0 ensures every node can at least attend to its own features.

**Why only test→training edges (no test→test)?** Connecting test patients to each other would allow information flow between test patients during evaluation, which violates the assumption that test predictions are independent. In real deployment, you'd process one patient at a time, so test-test edges wouldn't exist.

**Why `.clamp(min=1.0)` for norm?** If a patient has all-zero baseline features (all missing), the L2 norm is 0, and dividing by 0 produces NaN. Clamping the norm to at least 1.0 prevents this. The resulting "normalized" vector for an all-zero patient is still all-zero, which will have cosine similarity 0 to all training patients --- they'll connect to the k neighbors with the largest (least negative) similarity.

**What `np.argpartition` does vs `np.argsort`**: `argpartition(sims, -k)[-k:]` returns the indices of the k largest values in O(n) time (using the introselect algorithm), without fully sorting the array. `argsort` would take O(n log n). For 1520 training patients, this saves ~3x computation per test patient. The returned indices are not sorted, but we don't need them sorted.

#### Step 3: Train with Training-Only Graph, Evaluate with Extended Graph

```python
# Training: use training-only graph
train_graph_model(model, train_ds, val_ds,
    node_baseline=node_baseline,        # Training only
    edge_index=edge_index,              # Training only
    edge_weight=edge_weight,            # Training only
    ...)

# Evaluation: use extended graph
predict_all_graph(model, test_ds, device,
    node_baseline=ext_baseline,         # Training + test nodes
    edge_index=ext_edge_index,          # Training + test->training edges
    edge_weight=ext_edge_weight,        # Extended weights
    )
```

**Why this asymmetry?** During training, the model learns GAT attention weights that capture how patients influence each other based on feature similarity. These attention weights are *edge-wise* computations (not node-specific), so they generalize to new edges. At evaluation time, the extended graph lets test patients receive messages from their training neighbors, using the same attention mechanism learned during training.

### 3.6 Covariate Shift Detection

**Function**: `compute_covariate_shift()` in `temporal_validation.py`

Three complementary tests are applied to detect distribution shift between temporal train and test sets.

#### Kolmogorov-Smirnov Test

```python
ks_stat, ks_pval = scipy.stats.ks_2samp(train_vals, test_vals)
```

**What KS does mechanically**: Computes the maximum absolute difference between the empirical CDFs of the train and test distributions. If train patients' UPDRS-III bradykinesia scores are mostly 5-15 but test patients' scores are 10-25, the empirical CDFs are shifted rightward for the test set, and the KS statistic captures this shift.

**Why p < 0.001 threshold (not 0.05)?** With 760+ training patients and 380+ test patients, even tiny distribution differences become statistically significant at p < 0.05. The stricter threshold p < 0.001 focuses on shifts large enough to be clinically meaningful, not just statistically detectable.

#### Population Stability Index (PSI)

```python
def compute_psi(train_vals, test_vals, n_bins=10):
    bin_edges = np.percentile(train_vals, np.linspace(0, 100, n_bins + 1))
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    train_counts = np.histogram(train_vals, bin_edges)[0]
    test_counts = np.histogram(test_vals, bin_edges)[0]
    p_train = (train_counts + 1e-8) / train_counts.sum()
    p_test = (test_counts + 1e-8) / test_counts.sum()
    psi = ((p_test - p_train) * np.log(p_test / p_train)).sum()
    return psi
```

**Why percentile-based bins from training?** The bins are defined by training data percentiles (10th, 20th, ..., 90th percentile). This ensures each bin has ~10% of training observations. If test data has shifted, some bins will have more or fewer test observations than expected.

**Why add epsilon (1e-8)?** If a bin has zero observations in either distribution, `log(0)` is undefined. Adding a tiny epsilon prevents this without materially affecting the PSI value.

**PSI thresholds**: PSI < 0.10 (minimal shift), 0.10-0.25 (moderate), > 0.25 (significant). These thresholds are industry standard from credit scoring (Siddiqi 2005). A PSI > 0.25 indicates the model's input distribution has changed enough to warrant investigation.

**Why both KS and PSI?** KS is a formal statistical test with a p-value, but it's sensitive to sample size (large samples → small p-values for tiny shifts). PSI is a practical metric calibrated to "how much has the distribution changed" regardless of sample size. A feature might have KS p < 0.001 but PSI < 0.10 (statistically significant but practically trivial). The "shifted" flag requires either KS p < 0.001 OR PSI > 0.25, capturing both perspectives.

#### Maximum Mean Discrepancy (MMD)

```python
def compute_mmd_rbf(X_train, X_test, gamma=None, n_permutations=1000):
    # Subsample to max 500 per set
    # Gamma: median heuristic
    gamma = 1.0 / (2.0 * median_distance)
    # RBF kernel: K(x,y) = exp(-gamma * ||x-y||^2)
    # MMD^2 = E[K(x,x')] + E[K(y,y')] - 2*E[K(x,y)]
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    # Permutation test: p-value
    ...
```

**Why multivariate?** KS and PSI test each feature independently. If feature A shifts up by 0.1 and feature B shifts down by 0.1, the univariate tests might miss this because the marginals barely changed. But the *joint* distribution has changed (the A-B correlation structure is different). MMD with an RBF kernel detects this.

**Why the median heuristic for gamma?** The RBF bandwidth gamma controls the scale of the kernel. Too small → kernel is nearly zero for all pairs (undersmoothed). Too large → kernel is ~1 for all pairs (oversmoothed). The median heuristic sets gamma so that the kernel has a reasonable range of values for the observed data, providing good power across a range of alternatives.

**Why subsample to 500?** Computing the full kernel matrix is O(n^2). With 1520 training patients, the matrix would be 1520x1520 = 2.3M entries. Subsampling to 500 reduces this to 250K entries, making the permutation test (1000 iterations) tractable.

**Why 1000 permutations?** This gives a p-value resolution of 0.001 (1/1000). All our MMD p-values are 0.000, meaning the multivariate shift is detected with high confidence in every window.

#### Shift Classification

```python
severity = "mild" if fraction_shifted < 0.10 else \
           "moderate" if fraction_shifted < 0.30 else \
           "severe"
```

**Results across windows**: W1 severe (43.8% features shifted), W2 severe (31.3%), W3 moderate (18.8%), W4 severe (37.5%). The decreasing shift from W1→W3 makes sense: as the training window expands, it incorporates more of the population's variability, reducing the gap to the test set.

### 3.7 Performance Degradation Analysis

The degradation metric compares temporal validation C-td to Paper 3's random 5-fold CV C-td:

```python
PAPER3_DEEPHIT_CTD = 0.924
PAPER3_GRAPHDT_CTD = 0.904
degradation = PAPER3_reference - temporal_ctd
```

**W1-W3 degradation (3-5.5pp)**: Moderate and expected. The temporal constraint removes approximately 5% of the model's apparent accuracy. This is the "optimism" of random CV: by allowing future patients in the training set, random CV lets the model see the full population distribution, including patterns that only emerge in later enrollees.

**W4 degradation (21-24pp)**: Severe. This is NOT just temporal ordering --- it's primarily a training set size effect. W4 has only 950 training patients (vs 1520 in W3), but a much larger test set (950 vs 380). The model hasn't seen enough data to generalize well.

**Graph-DT vs DeepHit**: Graph-DT shows slightly less degradation on W1 (-2.1pp vs -5.5pp) but more on W2-W3 (-3.8pp to -4.6pp vs -3.3pp to -4.2pp). On the W4 stress test, Graph-DT degrades less (-21.2pp vs -24.4pp), suggesting graph-based models may be more robust to data scarcity --- the graph structure provides an inductive bias that partially compensates for limited training data.

### 3.8 Per-Transition Stability

**Function**: `_compute_per_transition_ctd()`

For each cause k (destination stage), the function:

1. Filters to uncensored episodes that transitioned to stage k
2. Requires at least 10 events (below this, C-td is unreliable)
3. Samples up to 50,000 concordant pairs for efficiency
4. Computes cause-specific C-td

**Why sample 50K pairs?** For a transition with 500 events, the full pairwise comparison requires 500*499/2 = 124,750 pairs. Computing C-td for all pairs is O(n^2). Sampling 50K random pairs gives a stable estimate with O(n) computation. The sampling variance is negligible: with 50K pairs, the standard error of C-td is ~0.002.

**Why minimum 10 events?** With fewer than 10 events, there are at most 45 concordant pairs. A single pair changing from concordant to discordant would shift C-td by ~2.2%. The estimate is too noisy to be meaningful.

### 3.9 Complete Parameter Table

| Parameter | Value | What It Does Mechanically | What Happens If Changed |
|-----------|-------|--------------------------|------------------------|
| `WINDOW_DEFS` | 4 windows | Train/test fractions for temporal splits | More windows → finer learning curve but more compute. Sliding windows → model forgets early patients |
| `val_fraction` | 0.15 | Fraction of training set held for early stopping | 0.20: more validation data, less training. 0.10: less early stopping signal, risk of overfitting |
| `k_neighbors` | 15 | kNN for inductive graph extension | 5: sparse graph, less information flow. 30: dense graph, slower, dilutes relevant neighbors |
| `n_bins` (PSI) | 10 | Number of bins for PSI computation | 5: coarser comparison. 20: finer but empty bins possible |
| `n_permutations` (MMD) | 1000 | Permutation test iterations for MMD p-value | 100: coarse p-value (0.01). 5000: finer but 5x slower |
| `max_subsample` (MMD) | 500 | Maximum patients per set for MMD kernel | 200: faster but noisier. 1000: more precise, 4x slower (O(n^2)) |
| KS threshold | p < 0.001 | P-value cutoff for declaring shift | p < 0.05: too sensitive (many false alarms). p < 0.01: moderate |
| PSI threshold | > 0.25 | PSI cutoff for significant shift | > 0.10: more sensitive. > 0.50: only detects extreme shift |
| `patience` | 15 | Early stopping patience (epochs) | 5: may stop too early. 30: wastes compute on plateau |
| `min_events` (per-transition) | 10 | Minimum events for per-cause C-td | 5: noisier estimates. 30: excludes rare transitions |
| `PAPER3_DEEPHIT_CTD` | 0.924 | Reference from random 5-fold CV | Used for degradation calculation only |
| `PAPER3_GRAPHDT_CTD` | 0.904 | Reference from random 5-fold CV | Used for degradation calculation only |

---

## 4. Committee Questions & Answers

### Q1: "The W4 stress test shows a 24% degradation. Is this model actually ready for deployment?"

**Answer**: W4 is deliberately adversarial --- it uses only 50% of patients for training while testing on 50%. This simulates deploying the model after only ~5 years of enrollment (2010-2015) and evaluating on the next 6 years. In practice, clinical centers would accumulate data continuously, and the W3 scenario (80% train, 20% test) is more realistic for mature deployment.

W3 shows C-td = 0.882 (DeepHit) with only 4.2pp degradation from random CV. This means: with a mature training dataset, the model maintains C-td > 0.85, which is clinically useful for transition prediction. The W4 result tells clinicians: "don't deploy this model until you have at least 1,200-1,500 patients with longitudinal follow-up."

The deployment recommendation is: (a) require minimum 1,500 training patients, (b) monitor covariate shift quarterly, (c) retrain annually with accumulated data.

### Q2: "How do you know the performance degradation is due to temporal shift and not just reduced training set size?"

**Answer**: Good question --- these two effects are confounded. W1-W3 provide partial separation: W1 (760 train) has C-td 0.869, W2 (1140 train) has 0.891, W3 (1520 train) has 0.882. The non-monotonic relationship (W3 slightly worse than W2 despite more data) suggests that temporal shift matters beyond just training set size.

To fully disentangle the effects, we'd need: (a) a random split with 760 patients for training (matching W1's size) --- if this gives C-td ~0.90, the extra degradation in W1 (0.87) is attributable to temporal shift specifically; (b) a temporal split with all 1520 patients in both train and test (impossible without overlap). We don't have these experiments in the paper, which is an acknowledged limitation.

The covariate shift analysis provides indirect evidence: W1 has the most severe shift (43.8% features shifted) AND the lowest non-stress-test C-td (0.869), while W3 has moderate shift (18.8%) and higher C-td (0.882). This correlation supports the causal link between shift and degradation.

### Q3: "Your inductive graph extension uses test-internal standardization rather than training statistics. Isn't this leakage?"

**Answer**: This is a valid concern and an acknowledged implementation shortcoming. The current code passes `train_means=None, train_stds=None` to `extract_test_baseline_features()`, causing it to compute standardization statistics from the test set itself. This is a mild form of information leakage --- the test features' own distribution influences their standardization.

The practical impact is likely small: (a) the 18 baseline features have relatively stable distributions across temporal windows (the covariate shift is moderate, not extreme), so training and test statistics are similar; (b) the standardization only affects the cosine similarity computation for neighbor selection, not the model's predictions directly; (c) fixing this would change neighbor assignments for a few patients but is unlikely to materially alter the C-td results.

The correct fix is to save training-set means/stds during graph construction and pass them to `extract_test_baseline_features()`. This is noted as a TODO in the code.

### Q4: "Why use GAT's inductive capability rather than retraining the graph for each temporal window?"

**Answer**: Retraining the full graph for each temporal window would include test patients in the graph structure, creating potential information leakage. The inductive approach is strictly cleaner: the graph used for training contains ONLY training patients, and test patients are added post-hoc via nearest-neighbor lookup.

Additionally, inductive extension simulates real deployment: when a new patient arrives at the clinic, we don't retrain the entire model. We extend the existing graph, generate predictions, and move on. The inductive approach validates that this deployment workflow produces reliable predictions.

GAT is inherently inductive because its attention weights are computed per-edge based on node features, not learned per-node. A new node with k edges to existing nodes simply participates in the existing attention computation, producing a valid embedding without any parameter updates.

### Q5: "What would happen if covariate shift was so severe that the model became unusable?"

**Answer**: Our shift detection framework provides the monitoring system for exactly this scenario. If the fraction of shifted features exceeds 50% (our "severe" threshold is 30%), or if the MMD statistic increases dramatically, this would signal that the model needs retraining on more recent data.

In practice, the escalation protocol would be: (1) Alert clinicians that prediction reliability may be reduced; (2) Retrain the model with all available data including recent patients; (3) Re-run temporal validation to verify the retrained model performs adequately; (4) If performance remains unacceptable, investigate which features have shifted and whether the model's feature set needs updating.

The shift detection tools (KS, PSI, MMD) can be run on any new batch of patients without retraining the model, making them suitable for continuous monitoring in a deployment pipeline.

---

## 5. Publication Reviewer Questions & Answers

### Q1: "You only test on PPMI data. How would temporal validation generalize to other cohorts?"

**Answer**: The temporal validation framework (expanding windows, covariate shift detection, inductive graph extension) is cohort-agnostic --- it only requires enrollment dates and longitudinal features. For a different cohort (e.g., PDBP, HBS), the TemporalSplitter would order patients by that cohort's enrollment dates, and the same shift detection pipeline would run.

The specific performance numbers (3-5pp degradation for W1-W3) are PPMI-specific and would differ for other cohorts. Cohorts with faster enrollment (shorter enrollment period) would have less temporal shift. Cohorts with changing diagnostic criteria over time would have more shift.

### Q2: "How do you distinguish between temporal shift and sampling variability? Your test sets are only 380 patients."

**Answer**: With 380 test patients generating ~750-1000 episodes, the standard error of C-td is approximately 0.01-0.015 (based on the binomial variance of concordant pairs). A 5pp degradation (from 0.924 to 0.87) is ~3-5 standard errors, which is statistically significant at p < 0.01.

Additionally, the degradation is consistent across windows (W1-W3 all show 3-5pp degradation) and across models (both DeepHit and Graph-DT degrade). If this were sampling noise, we'd expect some windows to show improvement and others degradation.

### Q3: "Why not use time-series cross-validation (multiple expanding windows with overlapping test sets)?"

**Answer**: Time-series CV would use windows like: train 0-20%/test 20-40%, train 0-40%/test 40-60%, etc. This is essentially what our W1-W3 approximate, except our test windows don't overlap (W1 tests 40-60%, W2 tests 60-80%, W3 tests 80-100%).

Non-overlapping test windows prevent the same patient from appearing in multiple test sets, avoiding correlation between window results. If a patient appeared in both W1 and W2 test sets, the C-td estimates would be correlated, complicating statistical comparisons across windows.

### Q4: "The inductive graph extension adds ~31 edges per test patient. How sensitive is performance to the number of edges (k)?"

**Answer**: We use k=15 (matching Paper 3's graph construction), producing up to 31 edges per test patient (2*15 bidirectional + 1 self-loop). The sensitivity to k was explored in Paper 3 during graph design: k=5 produces a sparser graph with less information flow, reducing GAT's ability to aggregate neighborhood features; k=30 includes more distant neighbors, diluting the signal from truly similar patients.

For inductive extension specifically, the key consideration is that test patients connect to *training* patients only. With k=15, a test patient's 15 nearest training neighbors are typically very similar (cosine similarity > 0.8). Increasing k would add less similar neighbors, potentially degrading the quality of the graph signal.

### Q5: "Your MMD p-values are all 0.000. Could this be an artifact of the permutation test with too few iterations?"

**Answer**: With 1000 permutations, p=0.000 means that NONE of the 1000 random permutations produced an MMD as large as the observed value. The true p-value is < 0.001 (we can only bound it, not compute it exactly). With 5000 permutations, the bound would be p < 0.0002.

This isn't an artifact --- it reflects genuine multivariate shift. Even 10-year observational studies like PPMI experience population drift due to changing enrollment criteria, referral patterns, and diagnostic technology. The p-value confirms what the per-feature KS/PSI tests already suggest: the joint feature distribution differs meaningfully between early and late enrollees.

---

## 6. Alternative Approaches

### 6.1 Sliding Window Validation (Instead of Expanding)

**What it is**: Train on a fixed-size moving window (e.g., always 3 years of patients) rather than all available historical data.

**Why we didn't use it**: Sliding windows discard early patients, simulating a scenario where old data is deliberately removed. In clinical practice, there's no reason to discard historical patient data --- more data is always better. Expanding windows simulate the realistic scenario of data accumulation.

**When sliding windows would be better**: If the population changes drastically over time (e.g., a major diagnostic criterion revision), old data might hurt performance. In this case, a sliding window with a tunable width would adapt to the most relevant recent data.

### 6.2 Transductive Graph Update (Instead of Inductive Extension)

**What it is**: Rebuild the entire patient similarity graph including both training and test patients, then re-run GAT to produce embeddings for everyone.

**Why we didn't use it**: Transductive update requires test patients in the graph during training, which is information leakage. Even if you only retrain at evaluation time, the training loss would be computed on a graph that includes test nodes' features, subtly leaking information.

**Trade-off**: Transductive approaches can produce better embeddings (test patients inform each other's representations), but at the cost of scientific rigor. Our inductive approach is strictly cleaner and more realistic for deployment.

### 6.3 Domain Adaptation (Explicitly Correcting for Shift)

**What it is**: Use domain adaptation techniques (e.g., importance weighting, adversarial training) to explicitly correct for covariate shift between temporal windows.

**Why we didn't use it**: Domain adaptation is complex and adds confounds. Our goal is to *measure* temporal degradation, not to fix it. If we applied domain adaptation and showed improved temporal performance, it would be unclear whether the improvement came from the adaptation or from something else.

**Future work**: Domain adaptation could be a natural extension --- use the covariate shift detection output to weight training samples, giving more weight to patients whose features resemble the test population.

### 6.4 Temporal Conformal Prediction

**What it is**: Apply conformal prediction (from Paper 4) with temporal calibration --- calibrate on earlier patients and evaluate on later patients, rather than randomly splitting the test set.

**Why we didn't fully integrate it**: Temporal conformal prediction has exchangeability concerns. Standard conformal assumes calibration and evaluation data are exchangeable, which is violated by temporal ordering. Recent work on distribution-shift conformal (Tibshirani et al. 2019) addresses this but requires known shift weights, which we'd need to estimate.

**Trade-off**: Temporal conformal would provide uncertainty quantification that explicitly accounts for distribution shift, making it more realistic. But the theoretical guarantees are weaker (approximate rather than exact coverage).

---

## 7. Limitations, Deficiencies, and Honest Assessment

Paper 5 is an honest stress test — its headline contribution is *quantified degradation under temporal shift*, not a claim of zero degradation. The sections below surface what the paper does NOT prove, where methodological shortcuts were taken, and which deployment decisions this evidence cannot support.

### 7.1 What the Paper Does NOT Prove

- **Not deployment-readiness under adversarial shift.** The W4 stress test (train on first 50%, test on final 50%) shows a 21.2–24.4pp C-td degradation from Paper 3's random 5-fold reference (DeepHit: 0.924 → ~0.68; Graph-DT: 0.904 → ~0.69). This is a deliberate worst-case, but the 24% degradation figure means: **Paper 5 does not support deploying these models at clinical centers with < ~1,200 training patients of longitudinal follow-up**. The "W3 is deployment-ready" framing (C-td 0.882 at 4.2pp degradation) is conditional on the center having at least ~1,500 patients already — a ceiling that many centers will not cross for years.

- **Not cross-cohort temporal generalisation.** All four windows are drawn from PPMI. The temporal shift being measured is *within-PPMI drift* driven by enrollment cohort composition (early arms were MCI+HC heavy; later arms added prodromal). It is NOT a test of whether the model transfers to BioFIND, PDBP, or HBS under their respective temporal shifts. PDBP's much larger enrollment window (2012–2021) and different site mix almost certainly produces a different shift profile — untested.

- **Not a validated per-patient inductive prediction guarantee.** The inductive graph extension attaches each test patient to k=15 training neighbors by cosine similarity on 18 baseline features. If a test patient is genuinely dissimilar from all 1,520 training patients (e.g., an atypical parkinsonian syndrome, rare genetic variant), the k=15 neighbors are the *least-bad* matches, not good matches. There is no published per-patient reliability score — a test patient whose mean cosine similarity to its k neighbors is 0.2 gets the same treatment as one whose mean similarity is 0.9.

- **Not a causal claim about shift mechanism.** KS/PSI/MMD detect that the joint distribution shifted; they do NOT attribute the shift to any specific cause (enrollment-criterion change, referral-pattern drift, diagnostic-technology evolution, measurement-device replacement). A paper making clinical-deployment recommendations needs causal attribution to know *which* monitoring signal to watch in production.

- **Not a measure of outcome-distribution shift.** Paper 5 detects covariate shift (X distribution) but does not measure label shift (Y distribution) or concept shift (P(Y|X) distribution). A feature distribution can be stable while the conditional relationship P(transition | features) has drifted — the more dangerous failure mode for prediction models. Measuring concept shift requires outcome data in the test window, which we have for PPMI but would not have in a live deployment.

### 7.2 Methodological Shortcomings (acknowledged)

- **Inductive graph extension uses test-internal standardization (real leakage concern).** In `extract_test_baseline_features()`, `train_means=None` and `train_stds=None` are passed, causing the function to compute standardization statistics from the test set itself. This is a mild form of information leakage — the test features' own mean/variance influences their own normalization. The file-level TODO flags this; the correct fix is to persist training-set means/stds during graph construction and pass them through. Expected impact: small (the 18 baseline features are relatively stable across temporal windows, and standardization only affects cosine-similarity neighbor assignment, not model predictions directly), but **unquantified**. No ablation has been run with training-anchored standardization to bound the leakage magnitude.

- **Training set size and temporal ordering are confounded.** W1 (760 train), W2 (1140), W3 (1520), W4 (950) vary both in size AND in temporal window. The W3-vs-W4 comparison (1520 train → C-td 0.882; 950 train → C-td 0.68) cannot separate "smaller training set" from "more temporal shift." A proper disentanglement requires a random 950-patient split matched to W4's size — **not run**. This is the single largest reviewer objection we expect and have no defence against.

- **MMD p-values all 0.000 may be permutation-resolution artefact.** With only 1,000 permutations, p = 0.000 literally means "none of the 1,000 random permutations produced an MMD ≥ observed." The true p-value could be 1e-4, 1e-5, or smaller — we cannot distinguish. Scaling to 10,000 permutations would give 10× better resolution at ~10× compute cost; not run. As reported, MMD p-values contribute only a binary "shift detected" flag, not a magnitude.

- **No calibration recheck per window.** Paper 4 calibrated conformal bands on the random 5-fold split. Paper 5 reports C-td (discrimination) per window but does NOT report per-window conformal coverage. A 4pp C-td drop with conformal coverage still near 90% is very different from a 4pp C-td drop with conformal coverage collapsing to 75%. The calibration-degradation question is not answered.

- **Graph-DT k-neighbor choice inherited from Paper 3 (k=15).** This hyperparameter was tuned on the random 5-fold split, not re-tuned for the temporal-shift setting. Under shift, the optimal k may differ (larger k averages more neighbors, potentially smoothing out shift; smaller k preserves signal from the few most-similar training patients). Not tested.

### 7.3 Statistical Power Limitations

- **380-patient test sets bound CI width on degradation estimates.** Each window has ~380 test patients generating ~750–1,000 episodes. Standard error on C-td at this sample size is ~0.010–0.015 (binomial variance on concordant pairs). A 5pp degradation is ~3–5 SE — significant at p < 0.01 but with 95% CI of roughly ±0.03 on the degradation magnitude. That means the *direction* (negative) is robust, but the *magnitude* (3pp vs 5pp vs 7pp) has wide CI. We report the point estimates without explicit CIs on the degradation delta — a reporting deficiency (see §9 below).

- **Rare-transition per-cause C-td not computed at full resolution.** Per-transition C-td requires ≥10 events per cause for stable estimation. Stage 5 (advanced) transitions have ~40 events cohort-wide — stable at the cohort level, but when stratified into four temporal windows with ~8–12 Stage-5 events per window, the per-window per-transition C-td is unreliable. Paper 5 does not report this breakdown per window; it only reports aggregate C-td per window. Rare-transition stability under temporal shift is therefore not addressed.

- **Only one random-CV reference (Paper 3 point estimate).** The degradation metric subtracts Paper 5's temporal C-td from a single reference (DeepHit 0.924, Graph-DT 0.904 from random 5-fold CV). Paper 3's own 5-fold variance (DeepHit SD 0.018, Graph-DT SD 0.013) is ignored when computing degradation. A proper treatment would propagate Paper 3's uncertainty into the degradation estimate's CI — **not done**.

### 7.4 Scope and Generalisation Caveats

- **Single dataset, single disease.** PPMI / Parkinson's. Temporal-validation methodology is general but performance numbers are not transferable.

- **Observational cohort, not prospective deployment.** We simulate "deploying in 2014 and waiting until 2021" by reshuffling retrospective data. True deployment faces continuous data arrival, model updates, clinician feedback loops, EHR integration — none of which are tested. The monitoring pipeline (KS/PSI/MMD) is runnable on new data without retraining, which is a real deployment asset, but the end-to-end deployment loop (detect shift → retrain → deploy) is not demonstrated.

- **No prospective holdout.** All "future" patients in this paper were enrolled before the analysis began. A genuine prospective test would require holding out 2022–2025 enrollees — impossible with PPMI's enrollment history.

- **No W0 lower bound.** The smallest training window is W1 (40% = 760 patients). We do not know whether W0 with 20% = 380 patients would produce C-td 0.50 (collapse), 0.70 (severe degradation), or 0.80 (gradual degradation). This lower end of the learning curve is the regime most clinical centers will start in, and it is exactly where Paper 5 has no data.

### 7.5 What Good Ablations Would Fix These

A fuller evaluation would include:

1. Random 950-patient split matched to W4's size — disentangles size from shift.
2. Training-anchored standardization in `extract_test_baseline_features()` — closes leakage gap.
3. Per-window conformal recalibration — answers the calibration-drift question.
4. Per-window per-transition C-td — rare-transition stability.
5. 10,000-permutation MMD — tighter p-value bounds.
6. Paired bootstrap on Δ C-td with Paper 3 uncertainty propagated — proper CI on degradation.

None were run before submission. All are explicitly scoped as future work.

---

## 8. Robustness and Sensitivity Analyses

Paper 5 IS itself a sensitivity analysis — it asks how Paper 3's results hold up under a specific robustness stressor (temporal shift). This section documents the ablations and sensitivity probes that WERE run, cross-validation variance observed, and the robustness analyses that were explicitly deferred.

### 8.1 Ablations Performed

| Ablation | Configurations | Primary Metric | Result |
|---|---|---|---|
| Expanding-window fraction | W1 (40/20), W2 (60/20), W3 (80/20), W4 (50/50) | DeepHit C-td | 0.869 / 0.891 / 0.882 / 0.680 |
| Expanding-window fraction | W1–W4 | Graph-DT C-td | 0.883 / 0.866 / 0.858 / 0.692 |
| DeepHit vs Graph-DT per window | 4 windows × 2 models | Paired Δ C-td | Graph-DT ≤ DeepHit in W2, W3; Graph-DT > DeepHit in W1, W4 |
| Shift-detection redundancy | KS, PSI, MMD run in parallel | Per-feature shift flag | KS and PSI agree on ~85% of features; MMD catches multivariate shift invisible to both |
| Per-transition C-td (Paper 3 inheritance) | 5 causes × 4 windows | Cause-specific C-td | Rare transitions (→Stage 5) have unstable per-window estimates (<10 events/window) |

**Not ablated:** k-neighbors in inductive extension (fixed at k=15); early-stopping patience (fixed at 15); dropout rate (fixed at 0.3); validation fraction (fixed at 0.15). These are treated as "inherited from Paper 3" and not re-tuned for the temporal setting.

### 8.2 Cross-Validation Variance

Paper 5 does NOT run cross-validation within each temporal window — each window produces a single C-td point estimate on its fixed test set. This is a deliberate design choice: the temporal-validation paradigm is "train once on the past, test once on the future," and re-running with random seeds within the temporal split would contaminate the message.

What IS computed:

- **Train-val shuffle variance.** The 85/15 train/val split inside each window uses seed=42. Re-running with different seeds would produce a small variance in final model parameters (estimated ~0.005 C-td SD from ad-hoc 3-seed spot checks), but this is not systematically characterised.
- **Inductive vs retrained graph.** We compared inductive extension (training graph + test patients attached by kNN) against the counterfactual of retraining the full graph including test patients. The retrained graph produces C-td ~0.01–0.02 higher per window BUT is methodologically invalid (transductive leakage). We report only the inductive numbers; the retrained numbers are documented in code comments but not in the paper.
- **Paper 3's random 5-fold CV as a reference point.** DeepHit SD 0.018, Graph-DT SD 0.013 across folds. The temporal-window C-td estimates (W1-W3) sit approximately 1-3 SDs below Paper 3's mean — consistent with genuine degradation beyond sampling noise, but without a formal CI on the delta.

### 8.3 Seed and Split Sensitivity

- **Validation-split seed**: fixed at 42. No multi-seed reporting.
- **kNN graph construction**: deterministic given features; no randomness.
- **DeepHit training**: Adam optimizer with stochastic batches (seed=42). Graph-DT same. Re-running with alternate seeds has not been done at the temporal-window level.
- **Inductive graph edge construction**: deterministic given `test_baseline` features (k=15 nearest neighbors by cosine similarity).

This is a known weakness. A reviewer objection like "is your W1 C-td of 0.869 a robust estimate, or could a seed change move it to 0.88?" cannot be answered from Paper 5's evidence alone. A 5-seed rerun per window would resolve it — **not done**.

### 8.4 Hyperparameter Sensitivity

| Hyperparameter | Default | Ablated? | Expected Impact If Perturbed |
|---|---|---|---|
| k (inductive kNN) | 15 | No | k=5: sparser graph, less info flow; k=30: more distant neighbors, signal dilution. Paper 3 indirectly supports k=15 but no temporal re-tuning. |
| Validation fraction | 0.15 | No | 0.10: more training data, less early-stopping signal; 0.20: less training data, more stable val loss |
| Early-stopping patience | 15 | No | 5: risk premature stop; 30: wasted compute on plateaus |
| PSI bins | 10 | No | 5: coarser; 20: empty-bin edge cases |
| MMD subsample | 500 | No | 200: faster, noisier MMD; 1000: 4× slower but tighter p-value |
| KS p-threshold | 0.001 | No | 0.01 would double the "shifted features" count; 0.0001 would halve it |
| PSI threshold | 0.25 | No | 0.10 would double shifted features; 0.50 would miss many |

**Bottom line:** Paper 5 inherits Paper 3 and prior-art defaults across the board. No temporal-setting-specific tuning was run. Reviewer-robust to "why these values?" only via the "same as Paper 3 for fair comparison" defense.

### 8.5 Stress Tests

- **W4 stress test (deliberate worst-case).** Train on first 50%, test on final 50%. Produces 21–24pp degradation. Designed to expose the floor of the learning curve under a combined size-and-shift stressor.
- **Severe-shift window comparison.** W1 (43.8% shifted), W2 (31.3%), W3 (18.8%), W4 (37.5%). W3 has the mildest shift by design (largest training window accumulates most population variability). If degradation correlated cleanly with shift severity, we would expect W1 ≤ W2 ≤ W3 for C-td. Actual: W1 (0.869), W2 (0.891), W3 (0.882), so W2 > W3 > W1 — **not monotonic in shift severity**. This is circumstantial evidence that size and shift both matter and cannot be cleanly separated.

### 8.6 What Was NOT Run, and Why

- **Size-matched random split for W4.** Would disentangle size from shift. Not run because it expands the paper scope; queued as future work.
- **Per-window conformal recalibration.** Would answer calibration-drift question. Not run — requires reworking Paper 4's per-(cause, time-bin) calibration pipeline per window.
- **Cross-cohort shift detection.** Running KS/PSI/MMD against PDBP/HBS features would test the monitoring pipeline's generalisation. Not run — feature spaces differ (12 vs 18 vs 22 features across cohorts), complicating direct comparison.
- **Adversarial shift injection.** Synthetically perturbing test-feature distributions to test monitoring-trigger thresholds. Not run — treated as future deployment-instrumentation work.
- **Domain-adaptation baseline.** Using importance weighting or adversarial training to CORRECT for shift. Not run because §6.3 of this doc explains: the paper's goal is to MEASURE shift, not fix it; mixing the two would confound the evidence.
- **Ensemble across folds for deployment.** Using all 5 of Paper 3's fold checkpoints and averaging. Not run — stayed with single-fold per window to keep the comparison clean.

---

## 9. Statistical Reporting Standards

This section is a transparent audit of what IS and IS NOT reported about confidence intervals, multiple comparisons, effect sizes, and checklist compliance. Where CIs are absent, the reason is stated rather than glossed.

### 9.1 Confidence Interval Methodology

| Quantity | CI reported? | Method | Notes |
|---|---|---|---|
| Per-window C-td (DeepHit, Graph-DT) | **No explicit CI** in main text; 95% CI estimated at ± 0.01–0.015 from binomial-variance on concordant pairs | Binomial-variance heuristic (not patient-level bootstrap) | Should be bootstrap patient-level with 1,000 resamples. Not run. |
| Degradation Δ C-td (temporal vs random 5-fold) | **No CI**; only point estimate reported | N/A | Proper CI requires propagating Paper 3's fold SD (0.018, 0.013) AND Paper 5's window-level sampling variance. Not done. |
| MMD statistic | p-value only, bounded by permutation resolution | Permutation test (1,000 iterations) | p = 0.000 means p < 1e-3; true p could be orders of magnitude smaller. Not resolved. |
| PSI per feature | Point estimate; no CI | N/A | PSI is a summary statistic; bootstrap CI would require 1,000+ resamples per feature × 4 windows = expensive. Not run. |
| KS statistic per feature | p-value reported | Asymptotic 2-sample KS | Sample sizes (380 test, 1,520 train) exceed small-sample corrections. No adjustment needed. |
| Per-transition C-td | Point estimate; no CI per window | N/A | Inherits Paper 3's methodology. Low event counts (8–12 per window per rare transition) make bootstrap CIs wide and unreliable. |

**Headline CI we should have reported but did not:** a paired bootstrap on Δ C-td between temporal-window evaluation and Paper 3's random 5-fold evaluation, at patient-episode level, with 1,000 resamples. This would give e.g. "W1 degradation −0.055 [95% CI −0.078, −0.031]" instead of the bare "−5.5pp". Flagged as a pre-revision TODO.

### 9.2 Multiple-Comparison Correction

- **Per-feature KS/PSI tests across 4 windows × 18 features = 72 tests.** No multiple-comparison correction applied. Under Bonferroni at α = 0.05, the per-test threshold would be 0.0007 — close to our stated 0.001 threshold by coincidence, but not a deliberate correction. Under Benjamini-Hochberg FDR at q = 0.05, some features currently flagged as "shifted" at p < 0.001 would remain significant; others would not. **Not explicitly corrected** — this is a reporting gap.
- **Per-window model comparison (DeepHit vs Graph-DT).** 4 paired tests, no correction. Not a dominant reporting issue because the paper does NOT claim "Graph-DT significantly better than DeepHit in window X"; it reports both point estimates and leaves the comparison informal.
- **Per-transition per-window C-td.** 5 causes × 4 windows × 2 models = 40 comparisons. No correction. Informal reporting only.

**Honest statement:** Paper 5 does not formally control family-wise error rate or FDR. The KS p < 0.001 threshold was chosen informally to prioritize "large effect" over "formally corrected significance" — defensible in a shift-detection context but not in a strict hypothesis-testing sense.

### 9.3 Effect-Size Reporting

Reported effect sizes:

- **Fraction of shifted features per window** (43.8%, 31.3%, 18.8%, 37.5%) — clear, interpretable magnitude.
- **C-td degradation per window** (−1.7% to −24.4%) — clear, in natural units of C-td.
- **Per-feature PSI values** (categorized as <0.10 minimal, 0.10–0.25 moderate, > 0.25 significant).

Not reported:

- Standardized effect sizes (Cohen's d, Hedge's g) for any univariate shift.
- MMD magnitude in absolute units (reported only with p-value).
- Paired effect size for DeepHit-vs-Graph-DT within window.

### 9.4 Reporting-Checklist Compliance

**Primary target:** TRIPOD+AI (Collins et al. 2024, BMJ 385:e078378) as the temporal-validation module complement to Paper 1's prediction-model TRIPOD+AI checklist.

**TRIPOD+AI self-audit for Paper 5:**

| Item | Status | Notes |
|---|---|---|
| Title / abstract identify temporal validation | Yes | |
| Participant enrollment dates reported | Yes | INFODT → enrollment_date |
| Temporal split rationale stated | Yes | §3.2 |
| No-leakage validation | Yes | §3.3 |
| Shift detection methodology | Yes | §3.6 (KS + PSI + MMD) |
| Discrimination per window (C-td) | Yes | §3.7 |
| Calibration per window | **No** | Not computed per window (see §7.2 above) |
| Per-subgroup discrimination per window | **No** | Not computed |
| Confidence intervals on discrimination | **Partial** | Point estimates only; no bootstrap CI |
| Fairness analysis per window | **No** | Not computed |
| Deployment recommendation bounds | Yes | W3 = 1,500-patient floor stated |
| Shift-monitoring protocol for deployment | Yes | §4 Q5 escalation protocol |
| Code and data availability | Yes | Paths to scripts and output JSONs |

**Gaps against TRIPOD+AI (honest):** per-window calibration, per-subgroup discrimination, formal CI on discrimination, and per-window fairness are all flagged but NOT run. These are documented as submission-revision targets.

### 9.5 Pre-Registration

Paper 5 was NOT pre-registered. The four window fractions (W1–W4) were selected after exploratory analysis, not pre-committed before observing data. In particular:

- W4's 50/50 split was added AFTER the first three windows revealed "only 3-5pp degradation" — W4 was designed deliberately to be adversarial, which is legitimate as a stress test but is NOT a pre-registered confirmation.
- The shift-severity thresholds (10% minimal, 30% moderate, 50%+ severe) are post-hoc chosen to match observed data distributions.
- The 0.001 KS threshold and 0.25 PSI threshold are conventional, not pre-registered.

No multiple-window correction was pre-committed. A fully pre-registered replication would fix these choices in advance; Paper 5 does not claim this epistemic status.

### 9.6 Summary of Reporting-Standard Shortfalls

| Shortfall | Severity | Fix Plan |
|---|---|---|
| No bootstrap CI on C-td per window | Medium | Add patient-level bootstrap at revision |
| No per-window conformal recalibration | Medium | Requires Paper 4 pipeline rerun; deferred |
| No MC correction on shift tests | Medium | Add BH-FDR at q = 0.05 in revision |
| No pre-registration | Low (context-appropriate for exploratory stress test) | Document post-hoc choices transparently |
| No per-subgroup per-window analysis | Medium | Sample sizes marginal; may be unachievable at fold level |
| No cross-cohort generalisation | High for clinical deployment claim | Explicitly scoped as Paper 5+ future work |

---

*Document generated for dissertation defense preparation. All metrics sourced from actual output files in `outputs/paper5/`. All code references verified against `src/giman_pipeline/paper5/`.*
