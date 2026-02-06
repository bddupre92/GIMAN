# Phase 4: Progression Subtype Discovery - Implementation Plan

## Executive Summary

**Goal**: Discover distinct progression subtypes in PPMI cohort using data-driven machine learning approaches, predict subtype membership from baseline features, and demonstrate clinical trial enrichment potential.

**Expected Impact**:
- Discover 2-3 progression subtypes (fast, moderate, slow progressors)
- Achieve AUC ~0.75-0.80 for baseline subtype prediction
- Demonstrate 30-43% reduction in required clinical trial sample size
- First study to combine GNN patient similarity with trajectory clustering

**Timeline**: 6-8 weeks for full implementation + validation

---

## Phase 4 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 4: SUBTYPE DISCOVERY                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────┐
    │  Task 4.1: Longitudinal Data Preparation           │
    │  - Extract BL, V04, V06, V08, V12 time points     │
    │  - Calculate individual trajectories               │
    │  - Quality control (min 3 visits)                  │
    └────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────┐
    │  Task 4.2: Latent Time Alignment (LTJMM)          │
    │  - Align patients on common disease timeline      │
    │  - Account for variable disease durations          │
    │  - Output: Disease progression scores              │
    └────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────┐
    │  Task 4.3: Trajectory Clustering                   │
    │  - Method 1: K-means on aligned trajectories      │
    │  - Method 2: Hierarchical clustering               │
    │  - Method 3: Gaussian Mixture Models               │
    │  - Validation: Silhouette, Davies-Bouldin         │
    └────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────┐
    │  Task 4.4: Subtype Characterization                │
    │  - Clinical profiles (motor, cognitive, autonomic) │
    │  - Imaging signatures (DAT-SPECT, FreeSurfer)     │
    │  - Demographic factors (age, sex, education)       │
    │  - Statistical validation (Kruskal-Wallis, χ²)    │
    └────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────┐
    │  Task 4.5: Baseline Subtype Prediction             │
    │  - Train GNN classifier (extend GIMAN)             │
    │  - Use only baseline + 1-year follow-up            │
    │  - Target AUC: 0.75-0.80                          │
    │  - 5-fold cross-validation                         │
    └────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────┐
    │  Task 4.6: Clinical Trial Enrichment Simulation    │
    │  - Simulate trials enriched with fast progressors  │
    │  - Calculate sample size reduction                 │
    │  - Power analysis for different scenarios          │
    │  - Cost-benefit analysis                           │
    └────────────────────────────────────────────────────┘
```

---

## Task 4.1: Longitudinal Data Preparation

### Objective
Extract multi-visit longitudinal trajectories for motor and cognitive outcomes from Phase 1 cohort.

### Data Requirements

**Source**: Your existing Phase 1 dataset (`giman_expanded_cohort_final.csv`)

**Key Variables**:
```python
longitudinal_features = {
    'identifiers': ['PATNO', 'EVENT_ID'],
    'motor_outcomes': ['UPDRS_III_TOTAL', 'motor_slope_pts_per_year'],
    'cognitive_outcomes': ['MOCA_TOTAL', 'cognitive_decline_binary'],
    'time_variables': ['disease_duration_years', 'months_since_baseline'],
    'baseline_features': [
        'age_at_baseline', 'sex', 'education_years',
        'baseline_updrs_iii', 'baseline_moca'
    ]
}
```

**Visit Coverage Analysis**:
```
Expected visits: BL, V04, V06, V08, V12
Minimum required: 3 visits per patient
Target N: ~1,200-1,500 patients with sufficient follow-up
```

### Implementation Steps

**Step 1.1**: Create `task_4_1_longitudinal_data_prep.py`
```python
class LongitudinalDataPreparation:
    """
    Prepare longitudinal trajectories for subtype discovery.
    """
    
    def __init__(self, phase1_path: str):
        self.phase1_df = pd.read_csv(phase1_path)
        
    def extract_longitudinal_cohort(
        self, 
        min_visits: int = 3,
        required_variables: List[str] = ['UPDRS_III_TOTAL', 'MOCA_TOTAL']
    ) -> pd.DataFrame:
        """
        Extract patients with sufficient longitudinal data.
        
        Returns:
            DataFrame with columns: PATNO, EVENT_ID, visit_number, 
            months_since_baseline, UPDRS_III, MOCA, [other features]
        """
        pass
    
    def calculate_individual_trajectories(self) -> pd.DataFrame:
        """
        Calculate motor and cognitive trajectories for each patient.
        
        Returns:
            DataFrame with slopes, intercepts, R² values per patient
        """
        pass
    
    def quality_control_analysis(self) -> Dict:
        """
        Analyze data quality and coverage.
        
        Returns:
            {
                'total_patients': int,
                'patients_with_3_visits': int,
                'patients_with_4_visits': int,
                'visit_distribution': pd.Series,
                'mean_follow_up_months': float
            }
        """
        pass
```

**Step 1.2**: Run quality control report
```python
# Expected output
quality_report = {
    'total_patients': 2046,
    'patients_with_3plus_visits': 1234,  # Estimate
    'mean_visits_per_patient': 3.8,
    'median_follow_up_months': 24.0,
    'motor_trajectory_completeness': 0.85,
    'cognitive_trajectory_completeness': 0.82
}
```

**Deliverables**:
- ✅ `task_4_1_longitudinal_data_prep.py` (400 lines)
- ✅ `data/longitudinal_cohort/longitudinal_trajectories.csv`
- ✅ `data/longitudinal_cohort/quality_control_report.json`
- ✅ Visualization: Visit distribution histogram, trajectory plots

---

## Task 4.2: Latent Time Alignment (LTJMM)

### Objective
Align patients on a common disease progression timeline to account for heterogeneity in disease duration and progression rates.

### Method: Latent Time Joint Mixed-Effects Model

**Mathematical Formulation**:
```
y_i(t) = α_i + β_i * f(t - τ_i) + ε_i(t)

Where:
- y_i(t): Observed outcome for patient i at time t
- α_i: Patient-specific intercept (random effect)
- β_i: Patient-specific slope (random effect)
- f(t - τ_i): Warping function with patient-specific time shift τ_i
- ε_i(t): Observation noise

Goal: Estimate τ_i (latent disease time) for each patient
```

### Implementation Approaches

**Approach A: Simplified Linear Mixed-Effects Model** (Recommended first)
```python
from sklearn.linear_model import HuberRegressor
from scipy.optimize import minimize

class LatentTimeAlignment:
    """
    Align patient trajectories using latent time model.
    """
    
    def __init__(self, n_iterations: int = 100):
        self.n_iterations = n_iterations
        self.time_shifts = {}  # τ_i for each patient
        
    def fit(self, patient_trajectories: pd.DataFrame) -> Dict:
        """
        Estimate latent time shifts for each patient.
        
        Algorithm:
        1. Initialize τ_i = 0 for all patients
        2. Iteratively:
           a. Fix τ_i, estimate population trajectory f(t)
           b. Fix f(t), optimize τ_i for each patient
        3. Repeat until convergence
        
        Returns:
            {
                'time_shifts': Dict[int, float],
                'aligned_trajectories': pd.DataFrame,
                'population_trajectory': np.ndarray
            }
        """
        pass
    
    def align_patient_trajectory(
        self, 
        patient_data: pd.DataFrame,
        population_curve: callable
    ) -> float:
        """
        Find optimal time shift for single patient.
        
        Minimize: ||y_i(t) - f(t + τ_i)||²
        """
        pass
```

**Approach B: Use Existing Package** (If needed)
```python
# Option 1: Use pymc3 for Bayesian hierarchical model
import pymc3 as pm

# Option 2: Use nlme-like functionality from statsmodels
from statsmodels.regression.mixed_linear_model import MixedLM
```

### Implementation Steps

**Step 2.1**: Implement simplified LTJMM
```python
# task_4_2_latent_time_alignment.py

class SimplifiedLTJMM:
    def fit_population_curve(self, aligned_data):
        """Fit smooth population trajectory (cubic spline or polynomial)"""
        pass
    
    def estimate_time_shift(self, patient_data, pop_curve):
        """Optimize τ_i for individual patient"""
        pass
    
    def iterate_em_algorithm(self):
        """Expectation-Maximization for convergence"""
        pass
```

**Step 2.2**: Validate alignment quality
```python
def validate_alignment(original_data, aligned_data):
    """
    Metrics:
    - Reduction in between-patient variance
    - Improvement in population curve R²
    - Visual inspection of aligned trajectories
    """
    return {
        'variance_reduction': 0.35,  # Target: 30-40%
        'population_r2': 0.68,       # Target: >0.65
        'convergence_iterations': 47
    }
```

**Deliverables**:
- ✅ `task_4_2_latent_time_alignment.py` (500 lines)
- ✅ `data/longitudinal_cohort/aligned_trajectories.csv`
- ✅ `data/longitudinal_cohort/latent_time_shifts.json`
- ✅ Visualization: Before/after alignment trajectory plots

---

## Task 4.3: Trajectory Clustering

### Objective
Discover distinct progression subtypes by clustering aligned patient trajectories.

### Clustering Methods

**Method 1: K-Means on Trajectory Features** (Primary)
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class TrajectoryClusteringKMeans:
    """
    Cluster patients based on trajectory features.
    """
    
    def extract_trajectory_features(self, aligned_data: pd.DataFrame) -> np.ndarray:
        """
        Features per patient:
        - Motor slope (aligned)
        - Motor intercept (baseline)
        - Motor acceleration (2nd derivative)
        - Cognitive slope
        - Cognitive intercept
        - Trajectory variability (residual variance)
        
        Returns: (N_patients, 6) feature matrix
        """
        features = []
        for patient_id in aligned_data['PATNO'].unique():
            patient_data = aligned_data[aligned_data['PATNO'] == patient_id]
            
            # Fit linear regression to aligned trajectory
            motor_slope, motor_intercept = fit_trajectory(patient_data, 'UPDRS_III')
            cognitive_slope, cognitive_intercept = fit_trajectory(patient_data, 'MOCA')
            
            # Calculate acceleration (curvature)
            motor_accel = calculate_acceleration(patient_data, 'UPDRS_III')
            
            # Trajectory variability
            motor_variance = calculate_residual_variance(patient_data, 'UPDRS_III')
            
            features.append([
                motor_slope, motor_intercept, motor_accel,
                cognitive_slope, cognitive_intercept, motor_variance
            ])
        
        return np.array(features)
    
    def determine_optimal_k(self, features: np.ndarray) -> int:
        """
        Use silhouette score, elbow method, Davies-Bouldin index.
        
        Test k = 2, 3, 4, 5
        Expected optimal: k = 2 or 3
        """
        pass
    
    def fit_kmeans(self, features: np.ndarray, k: int = 3) -> np.ndarray:
        """
        Fit K-means clustering.
        
        Returns: cluster labels (N_patients,)
        """
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = kmeans.fit_predict(features_scaled)
        
        return labels
```

**Method 2: Hierarchical Clustering** (Secondary)
```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

def hierarchical_clustering(trajectory_features: np.ndarray) -> np.ndarray:
    """
    Agglomerative hierarchical clustering.
    
    Advantage: Dendrogram visualization shows natural groupings
    """
    # Compute pairwise distances
    distances = pdist(trajectory_features, metric='euclidean')
    
    # Linkage (Ward's method minimizes within-cluster variance)
    Z = linkage(distances, method='ward')
    
    # Cut dendrogram at optimal height
    labels = fcluster(Z, t=3, criterion='maxclust')
    
    return labels, Z
```

**Method 3: Gaussian Mixture Models** (Tertiary)
```python
from sklearn.mixture import GaussianMixture

def gmm_clustering(trajectory_features: np.ndarray, n_components: int = 3):
    """
    Probabilistic clustering - provides soft cluster assignments.
    
    Advantage: Can identify patients with ambiguous subtype membership
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=42
    )
    
    labels = gmm.fit_predict(trajectory_features)
    probabilities = gmm.predict_proba(trajectory_features)
    
    return labels, probabilities
```

### Cluster Validation Metrics

```python
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score,
    calinski_harabasz_score
)

def validate_clustering(features: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Comprehensive cluster validation.
    """
    return {
        'silhouette_score': silhouette_score(features, labels),  # Target: >0.3
        'davies_bouldin_score': davies_bouldin_score(features, labels),  # Target: <1.5
        'calinski_harabasz_score': calinski_harabasz_score(features, labels),  # Target: >100
        'cluster_sizes': np.bincount(labels),
        'cluster_proportions': np.bincount(labels) / len(labels)
    }
```

### Implementation Steps

**Step 3.1**: Create `task_4_3_trajectory_clustering.py`

**Step 3.2**: Run all three methods and compare

**Step 3.3**: Select optimal k (expected: 2-3 subtypes)

**Deliverables**:
- ✅ `task_4_3_trajectory_clustering.py` (600 lines)
- ✅ `results/phase4/cluster_labels.csv` (PATNO, subtype_label)
- ✅ `results/phase4/clustering_validation_metrics.json`
- ✅ Visualization: Dendrogram, silhouette plots, trajectory plots by subtype

---

## Task 4.4: Subtype Characterization

### Objective
Comprehensively characterize each discovered subtype using clinical, imaging, demographic, and outcome features.

### Characterization Dimensions

**1. Clinical Profiles**
```python
clinical_features = {
    'motor': [
        'baseline_updrs_iii', 'motor_slope', 'motor_acceleration',
        'tremor_score', 'rigidity_score', 'bradykinesia_score',
        'PIGD_score', 'TD_ratio'
    ],
    'cognitive': [
        'baseline_moca', 'cognitive_slope', 'mci_conversion_rate',
        'executive_function', 'memory', 'attention'
    ],
    'non_motor': [
        'rbd_score', 'constipation', 'orthostatic_hypotension',
        'depression_score', 'anxiety_score', 'apathy_score'
    ]
}
```

**2. Imaging Signatures**
```python
imaging_features = {
    'datscan': [
        'caudate_sbr_mean', 'putamen_sbr_mean',
        'asymmetry_index', 'sbr_decline_rate'
    ],
    'structural_mri': [
        'total_grey_matter_volume',
        'hippocampal_volume',
        'frontal_cortex_thickness',
        'substantia_nigra_volume'  # If available
    ]
}
```

**3. Demographics & Genetics**
```python
demographic_features = {
    'demographics': ['age_at_onset', 'sex', 'education_years', 'race'],
    'genetics': ['LRRK2_status', 'GBA_status', 'APOE_e4_carrier'],
    'family_history': ['family_history_pd']
}
```

**4. Outcomes**
```python
outcome_features = {
    'progression': [
        'time_to_hoehn_yahr_3',
        'time_to_mci',
        'time_to_dementia',
        'survival_months'
    ],
    'treatment': [
        'levodopa_equivalent_dose',
        'medication_response',
        'time_to_motor_complications'
    ]
}
```

### Statistical Comparison Framework

```python
class SubtypeCharacterization:
    """
    Compare subtypes across all dimensions.
    """
    
    def __init__(self, cohort_data: pd.DataFrame, cluster_labels: np.ndarray):
        self.data = cohort_data
        self.labels = cluster_labels
        self.n_subtypes = len(np.unique(cluster_labels))
        
    def compare_continuous_features(
        self, 
        features: List[str]
    ) -> pd.DataFrame:
        """
        Compare continuous features across subtypes.
        
        Statistical tests:
        - Kruskal-Wallis H-test (non-parametric ANOVA)
        - Post-hoc Dunn's test with Bonferroni correction
        
        Returns:
            DataFrame with columns: feature, subtype_0_mean, subtype_1_mean, 
            subtype_2_mean, p_value, effect_size
        """
        from scipy.stats import kruskal
        from scikit_posthocs import posthoc_dunn
        
        results = []
        for feature in features:
            groups = [
                self.data[self.labels == i][feature].dropna() 
                for i in range(self.n_subtypes)
            ]
            
            # Kruskal-Wallis test
            h_stat, p_value = kruskal(*groups)
            
            # Effect size (epsilon-squared)
            effect_size = self.calculate_epsilon_squared(groups)
            
            # Post-hoc tests
            posthoc_df = posthoc_dunn(groups)
            
            results.append({
                'feature': feature,
                **{f'subtype_{i}_mean': np.mean(groups[i]) for i in range(self.n_subtypes)},
                **{f'subtype_{i}_std': np.std(groups[i]) for i in range(self.n_subtypes)},
                'p_value': p_value,
                'effect_size': effect_size,
                'significant': p_value < 0.05
            })
        
        return pd.DataFrame(results)
    
    def compare_categorical_features(
        self, 
        features: List[str]
    ) -> pd.DataFrame:
        """
        Compare categorical features across subtypes.
        
        Statistical test: Chi-square test of independence
        """
        from scipy.stats import chi2_contingency
        
        results = []
        for feature in features:
            contingency_table = pd.crosstab(self.labels, self.data[feature])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Cramér's V for effect size
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
            results.append({
                'feature': feature,
                'chi2': chi2,
                'p_value': p_value,
                'cramers_v': cramers_v,
                'significant': p_value < 0.05
            })
        
        return pd.DataFrame(results)
    
    def survival_analysis(self, event: str, time: str) -> Dict:
        """
        Compare survival curves across subtypes.
        
        Uses Kaplan-Meier estimator + log-rank test
        """
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import multivariate_logrank_test
        
        kmf = KaplanMeierFitter()
        results = {}
        
        for subtype in range(self.n_subtypes):
            mask = self.labels == subtype
            kmf.fit(
                self.data.loc[mask, time],
                self.data.loc[mask, event],
                label=f'Subtype {subtype}'
            )
            results[f'subtype_{subtype}_median_survival'] = kmf.median_survival_time_
        
        # Log-rank test
        results['logrank_test'] = multivariate_logrank_test(
            self.data[time],
            self.labels,
            self.data[event]
        )
        
        return results
```

### Visualization Suite

```python
def create_characterization_dashboard(
    characterization_results: Dict,
    output_dir: Path
):
    """
    Create comprehensive visualization dashboard.
    
    Plots:
    1. Trajectory comparison (motor & cognitive by subtype)
    2. Radar plot of clinical profiles
    3. Kaplan-Meier survival curves
    4. Violin plots for key continuous features
    5. Heatmap of imaging signatures
    6. Forest plot of effect sizes
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # Plot 1: Motor trajectories by subtype
    plot_trajectories(axes[0, 0], 'UPDRS_III', characterization_results)
    
    # Plot 2: Cognitive trajectories by subtype
    plot_trajectories(axes[0, 1], 'MOCA', characterization_results)
    
    # Plot 3: Radar plot of clinical profiles
    plot_radar_chart(axes[1, 0], characterization_results['clinical_profiles'])
    
    # Plot 4: Survival curves
    plot_kaplan_meier(axes[1, 1], characterization_results['survival'])
    
    # Plot 5: Imaging heatmap
    plot_imaging_heatmap(axes[2, 0], characterization_results['imaging'])
    
    # Plot 6: Effect size forest plot
    plot_forest_plot(axes[2, 1], characterization_results['effect_sizes'])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'subtype_characterization_dashboard.png', dpi=300)
```

### Implementation Steps

**Step 4.1**: Create `task_4_4_subtype_characterization.py`

**Step 4.2**: Run comprehensive statistical comparisons

**Step 4.3**: Generate characterization report

**Deliverables**:
- ✅ `task_4_4_subtype_characterization.py` (700 lines)
- ✅ `results/phase4/subtype_clinical_profiles.csv`
- ✅ `results/phase4/subtype_imaging_signatures.csv`
- ✅ `results/phase4/statistical_comparison_report.csv`
- ✅ `Docs/PHASE4_SUBTYPE_CHARACTERIZATION_REPORT.md`
- ✅ Visualization: 6-panel characterization dashboard

---

## Task 4.5: Baseline Subtype Prediction

### Objective
Build a classifier that predicts subtype membership using only baseline + 1-year follow-up data, enabling early personalized prognosis.

### Model Architecture: GNN Subtype Classifier

**Extend existing GIMAN architecture**:
```python
class GIMANSubtypeClassifier(nn.Module):
    """
    Graph neural network for subtype prediction.
    
    Key difference from prognostic model:
    - Output: Subtype probabilities (softmax over K subtypes)
    - Loss: Cross-entropy loss
    - Graph: Patient similarity based on baseline features
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        num_subtypes: int = 3,
        num_gat_layers: int = 3,
        num_attention_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GAT layers (reuse from Phase 2)
        self.gat_layers = nn.ModuleList([
            GATConv(
                hidden_dim,
                hidden_dim // num_attention_heads,
                heads=num_attention_heads,
                dropout=dropout,
                add_self_loops=True
            )
            for _ in range(num_gat_layers)
        ])
        
        # Subtype classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_subtypes)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (N_patients, input_dim)
            edge_index: Graph edges (2, N_edges)
            edge_weights: Edge weights (N_edges,)
            
        Returns:
            Subtype logits (N_patients, num_subtypes)
        """
        # Project to hidden dimension
        h = self.input_projection(x)
        
        # GAT layers with residual connections
        for gat_layer in self.gat_layers:
            h_new = gat_layer(h, edge_index)
            h = h + h_new  # Residual connection
        
        # Classify
        logits = self.classifier(h)
        
        return logits
```

### Training Strategy

```python
class SubtypePredictor:
    """
    Train and evaluate subtype prediction model.
    """
    
    def __init__(
        self,
        model: GIMANSubtypeClassifier,
        num_subtypes: int = 3
    ):
        self.model = model
        self.num_subtypes = num_subtypes
        
        # Handle class imbalance
        self.criterion = nn.CrossEntropyLoss(weight=self.calculate_class_weights())
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )
    
    def calculate_class_weights(self) -> torch.Tensor:
        """
        Handle imbalanced subtypes (e.g., 40% fast, 45% moderate, 15% slow).
        """
        class_counts = np.bincount(self.train_labels)
        weights = 1.0 / class_counts
        weights = weights / weights.sum()
        return torch.FloatTensor(weights)
    
    def train_epoch(
        self,
        train_data: Data,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor
    ) -> float:
        """Single training epoch."""
        self.model.train()
        
        logits = self.model(train_data.x, edge_index, edge_weights)
        loss = self.criterion(logits, train_data.y)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(
        self,
        val_data: Data,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor
    ) -> Dict:
        """
        Comprehensive evaluation metrics.
        """
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(val_data.x, edge_index, edge_weights)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        from sklearn.metrics import (
            roc_auc_score,
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            classification_report,
            confusion_matrix
        )
        
        y_true = val_data.y.cpu().numpy()
        y_pred = preds.cpu().numpy()
        y_probs = probs.cpu().numpy()
        
        # Multi-class AUC (one-vs-rest)
        if self.num_subtypes == 2:
            auc = roc_auc_score(y_true, y_probs[:, 1])
        else:
            auc = roc_auc_score(
                y_true, 
                y_probs, 
                multi_class='ovr',
                average='weighted'
            )
        
        return {
            'auc': auc,
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred)
        }
```

### Cross-Validation Strategy

```python
def stratified_kfold_evaluation(
    data: pd.DataFrame,
    labels: np.ndarray,
    n_splits: int = 5
) -> Dict:
    """
    5-fold stratified cross-validation.
    
    Ensures each fold has proportional representation of all subtypes.
    """
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
        print(f"\n=== Fold {fold_idx + 1}/{n_splits} ===")
        
        # Split data
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        
        # Create patient similarity graphs
        train_edge_index, train_edge_weights = create_patient_similarity_graph(
            train_data, k=6
        )
        val_edge_index, val_edge_weights = create_patient_similarity_graph(
            val_data, k=6
        )
        
        # Train model
        model = GIMANSubtypeClassifier(
            input_dim=train_data.shape[1],
            num_subtypes=len(np.unique(labels))
        )
        
        trainer = SubtypePredictor(model, num_subtypes=len(np.unique(labels)))
        
        # Train for 100 epochs with early stopping
        best_val_auc = 0
        patience_counter = 0
        
        for epoch in range(100):
            train_loss = trainer.train_epoch(train_data, train_edge_index, train_edge_weights)
            val_metrics = trainer.evaluate(val_data, val_edge_index, val_edge_weights)
            
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch}")
                break
        
        fold_results.append(val_metrics)
    
    # Aggregate results
    return {
        'mean_auc': np.mean([r['auc'] for r in fold_results]),
        'std_auc': np.std([r['auc'] for r in fold_results]),
        'mean_accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'mean_f1': np.mean([r['f1_weighted'] for r in fold_results]),
        'fold_results': fold_results
    }
```

### Feature Importance Analysis

```python
def analyze_feature_importance(
    model: GIMANSubtypeClassifier,
    data: pd.DataFrame,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Permutation feature importance for subtype prediction.
    """
    from sklearn.inspection import permutation_importance
    
    baseline_auc = evaluate_model(model, data)
    
    importances = []
    for feature_idx, feature_name in enumerate(feature_names):
        # Permute feature
        data_permuted = data.copy()
        data_permuted.iloc[:, feature_idx] = np.random.permutation(
            data_permuted.iloc[:, feature_idx]
        )
        
        # Evaluate with permuted feature
        permuted_auc = evaluate_model(model, data_permuted)
        
        # Importance = drop in performance
        importance = baseline_auc - permuted_auc
        
        importances.append({
            'feature': feature_name,
            'importance': importance,
            'importance_pct': (importance / baseline_auc) * 100
        })
    
    return pd.DataFrame(importances).sort_values('importance', ascending=False)
```

### Implementation Steps

**Step 5.1**: Create `task_4_5_baseline_subtype_prediction.py`

**Step 5.2**: Prepare baseline + 1-year features
```python
baseline_features = [
    'age_at_baseline', 'sex', 'education_years',
    'baseline_updrs_iii', 'baseline_moca',
    'motor_slope_1year', 'cognitive_change_1year'
]
```

**Step 5.3**: Train GNN classifier with 5-fold CV

**Step 5.4**: Analyze feature importance

**Deliverables**:
- ✅ `task_4_5_baseline_subtype_prediction.py` (650 lines)
- ✅ `models/giman_subtype_classifier_best.pth`
- ✅ `results/phase4/subtype_prediction_cv_results.json`
- ✅ `results/phase4/feature_importance.csv`
- ✅ Visualization: ROC curves, confusion matrix, feature importance plot

**Target Performance**:
```
Expected Results:
- AUC: 0.75-0.80 (target: >0.75)
- Balanced Accuracy: 0.70-0.75
- F1-score (weighted): 0.68-0.73
```

---

## Task 4.6: Clinical Trial Enrichment Simulation

### Objective
Demonstrate how subtype-informed enrollment can reduce required sample sizes for clinical trials detecting treatment effects.

### Simulation Framework

**Scenario**: Clinical trial testing disease-modifying therapy
```python
trial_parameters = {
    'treatment_effect': 0.30,  # 30% reduction in progression rate
    'alpha': 0.05,             # Type I error rate
    'beta': 0.20,              # Type II error rate (80% power)
    'baseline_progression_rate': 5.0,  # UPDRS-III points/year
    'trial_duration': 2.0,     # years
    'dropout_rate': 0.15       # 15% attrition
}
```

### Power Analysis Functions

```python
class ClinicalTrialSimulator:
    """
    Simulate clinical trials with subtype enrichment.
    """
    
    def __init__(
        self,
        cohort_data: pd.DataFrame,
        subtype_labels: np.ndarray,
        subtype_progression_rates: Dict[int, float]
    ):
        """
        Args:
            subtype_progression_rates: {0: 3.2, 1: 5.8, 2: 8.1}
                Fast progressors have rate ~8 pts/year
                Moderate progressors ~6 pts/year
                Slow progressors ~3 pts/year
        """
        self.data = cohort_data
        self.labels = subtype_labels
        self.rates = subtype_progression_rates
        
    def simulate_standard_trial(
        self,
        n_per_arm: int,
        treatment_effect: float = 0.30,
        duration_years: float = 2.0
    ) -> Dict:
        """
        Simulate standard trial with random enrollment.
        
        Returns: statistical power to detect treatment effect
        """
        # Sample patients randomly
        trial_cohort = np.random.choice(len(self.data), size=n_per_arm * 2, replace=False)
        
        # Assign to treatment/placebo
        treatment_arm = trial_cohort[:n_per_arm]
        placebo_arm = trial_cohort[n_per_arm:]
        
        # Simulate progression
        placebo_progression = self._simulate_progression(
            placebo_arm, 
            duration_years, 
            treatment_effect=0.0
        )
        treatment_progression = self._simulate_progression(
            treatment_arm,
            duration_years,
            treatment_effect=treatment_effect
        )
        
        # Statistical test
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(treatment_progression, placebo_progression)
        
        return {
            'n_per_arm': n_per_arm,
            'mean_placebo_progression': np.mean(placebo_progression),
            'mean_treatment_progression': np.mean(treatment_progression),
            'effect_size': np.mean(placebo_progression) - np.mean(treatment_progression),
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def simulate_enriched_trial(
        self,
        n_per_arm: int,
        enrichment_strategy: str = 'fast_only',
        treatment_effect: float = 0.30,
        duration_years: float = 2.0
    ) -> Dict:
        """
        Simulate trial enriched with fast progressors.
        
        Enrichment strategies:
        - 'fast_only': Only enroll fast progressors (subtype 2)
        - 'fast_moderate': 70% fast, 30% moderate
        - 'predicted_fast': Use baseline prediction model
        """
        if enrichment_strategy == 'fast_only':
            eligible_patients = np.where(self.labels == 2)[0]
        elif enrichment_strategy == 'fast_moderate':
            fast_patients = np.where(self.labels == 2)[0]
            moderate_patients = np.where(self.labels == 1)[0]
            
            n_fast = int(n_per_arm * 2 * 0.7)
            n_moderate = n_per_arm * 2 - n_fast
            
            eligible_patients = np.concatenate([
                np.random.choice(fast_patients, n_fast),
                np.random.choice(moderate_patients, n_moderate)
            ])
        
        # Rest of simulation same as standard_trial
        trial_cohort = np.random.choice(eligible_patients, size=n_per_arm * 2, replace=False)
        
        treatment_arm = trial_cohort[:n_per_arm]
        placebo_arm = trial_cohort[n_per_arm:]
        
        placebo_progression = self._simulate_progression(placebo_arm, duration_years, 0.0)
        treatment_progression = self._simulate_progression(
            treatment_arm, duration_years, treatment_effect
        )
        
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(treatment_progression, placebo_progression)
        
        return {
            'n_per_arm': n_per_arm,
            'enrichment_strategy': enrichment_strategy,
            'mean_placebo_progression': np.mean(placebo_progression),
            'mean_treatment_progression': np.mean(treatment_progression),
            'effect_size': np.mean(placebo_progression) - np.mean(treatment_progression),
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def _simulate_progression(
        self,
        patient_indices: np.ndarray,
        duration: float,
        treatment_effect: float
    ) -> np.ndarray:
        """
        Simulate disease progression for given patients.
        
        progression = baseline_rate * (1 - treatment_effect) * duration + noise
        """
        progression_values = []
        
        for idx in patient_indices:
            subtype = self.labels[idx]
            baseline_rate = self.rates[subtype]
            
            # Apply treatment effect
            effective_rate = baseline_rate * (1 - treatment_effect)
            
            # Add individual variability (CV ~20%)
            noise = np.random.normal(0, baseline_rate * 0.2)
            
            progression = effective_rate * duration + noise
            progression_values.append(progression)
        
        return np.array(progression_values)
    
    def calculate_required_sample_size(
        self,
        treatment_effect: float = 0.30,
        alpha: float = 0.05,
        power: float = 0.80,
        enrichment_strategy: str = None
    ) -> int:
        """
        Calculate required N per arm for 80% power.
        
        Uses iterative simulation to find minimum N.
        """
        from statsmodels.stats.power import tt_ind_solve_power
        
        # Estimate effect size in each group
        if enrichment_strategy == 'fast_only':
            baseline_mean = self.rates[2]  # Fast progressors
            baseline_std = self.rates[2] * 0.25
        else:
            # Mixed population
            baseline_mean = np.mean(list(self.rates.values()))
            baseline_std = np.std(list(self.rates.values()))
        
        treatment_mean = baseline_mean * (1 - treatment_effect)
        
        # Cohen's d effect size
        cohens_d = (baseline_mean - treatment_mean) / baseline_std
        
        # Calculate required N
        n_required = tt_ind_solve_power(
            effect_size=cohens_d,
            alpha=alpha,
            power=power,
            alternative='two-sided'
        )
        
        return int(np.ceil(n_required))
```

### Sample Size Reduction Analysis

```python
def compare_enrollment_strategies(simulator: ClinicalTrialSimulator) -> pd.DataFrame:
    """
    Compare sample size requirements across strategies.
    """
    strategies = {
        'standard': None,
        'fast_only': 'fast_only',
        'fast_moderate_70_30': 'fast_moderate',
        'predicted_fast': 'predicted_fast'
    }
    
    results = []
    
    for strategy_name, enrichment in strategies.items():
        n_required = simulator.calculate_required_sample_size(
            treatment_effect=0.30,
            alpha=0.05,
            power=0.80,
            enrichment_strategy=enrichment
        )
        
        results.append({
            'strategy': strategy_name,
            'n_per_arm': n_required,
            'total_n': n_required * 2,
            'reduction_vs_standard': None  # Calculate after
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate reduction
    standard_n = results_df.loc[results_df['strategy'] == 'standard', 'total_n'].values[0]
    results_df['reduction_vs_standard'] = (
        (standard_n - results_df['total_n']) / standard_n * 100
    )
    
    return results_df

# Expected output:
"""
      strategy          n_per_arm  total_n  reduction_vs_standard
0     standard              250      500           0.0%
1     fast_only             140      280          44.0%
2     fast_moderate_70_30   175      350          30.0%
3     predicted_fast        160      320          36.0%
"""
```

### Cost-Benefit Analysis

```python
def calculate_trial_cost_savings(
    sample_size_comparison: pd.DataFrame,
    cost_per_patient: float = 50000,  # $50K per patient over 2 years
    additional_screening_cost: float = 2000  # Cost of subtype prediction test
) -> pd.DataFrame:
    """
    Calculate financial impact of enrichment strategies.
    """
    results = sample_size_comparison.copy()
    
    # Standard trial cost
    standard_cost = results.loc[results['strategy'] == 'standard', 'total_n'].values[0] * cost_per_patient
    
    for idx, row in results.iterrows():
        if row['strategy'] == 'standard':
            total_cost = standard_cost
            savings = 0
        else:
            # Enrollment cost
            enrollment_cost = row['total_n'] * cost_per_patient
            
            # Screening cost (need to screen more patients to find fast progressors)
            # If 30% are fast progressors, need to screen 3.3x to find required N
            screening_multiplier = 1.0 / 0.30 if 'fast' in row['strategy'] else 1.0
            screening_cost = row['total_n'] * screening_multiplier * additional_screening_cost
            
            total_cost = enrollment_cost + screening_cost
            savings = standard_cost - total_cost
        
        results.loc[idx, 'total_cost_usd'] = total_cost
        results.loc[idx, 'savings_vs_standard_usd'] = savings
        results.loc[idx, 'roi_percent'] = (savings / standard_cost) * 100
    
    return results

# Expected output:
"""
      strategy          total_n  total_cost_usd  savings_vs_standard_usd  roi_percent
0     standard              500      25,000,000                  0        0.0%
1     fast_only             280      14,186,667             10,813,333      43.3%
2     fast_moderate_70_30   350      17,583,333              7,416,667      29.7%
"""
```

### Implementation Steps

**Step 6.1**: Create `task_4_6_clinical_trial_simulation.py`

**Step 6.2**: Run power simulations (1000 iterations per scenario)

**Step 6.3**: Generate sample size comparison report

**Step 6.4**: Calculate cost-benefit analysis

**Deliverables**:
- ✅ `task_4_6_clinical_trial_simulation.py` (550 lines)
- ✅ `results/phase4/sample_size_comparison.csv`
- ✅ `results/phase4/cost_benefit_analysis.csv`
- ✅ `Docs/PHASE4_CLINICAL_TRIAL_ENRICHMENT_REPORT.md`
- ✅ Visualization: Power curves, sample size comparison bar chart

**Expected Results**:
```
Sample Size Reduction: 30-43% for fast-progressor enrichment
Cost Savings: $7-11 million per trial
Power: Maintained at 80% with smaller samples
```

---

## Phase 4 Summary & Integration

### Deliverables Checklist

**Code Files** (6 tasks):
- [ ] `task_4_1_longitudinal_data_prep.py` (400 lines)
- [ ] `task_4_2_latent_time_alignment.py` (500 lines)
- [ ] `task_4_3_trajectory_clustering.py` (600 lines)
- [ ] `task_4_4_subtype_characterization.py` (700 lines)
- [ ] `task_4_5_baseline_subtype_prediction.py` (650 lines)
- [ ] `task_4_6_clinical_trial_simulation.py` (550 lines)

**Total Code**: ~3,400 lines

**Data Outputs**:
- [ ] `data/longitudinal_cohort/longitudinal_trajectories.csv`
- [ ] `data/longitudinal_cohort/aligned_trajectories.csv`
- [ ] `results/phase4/cluster_labels.csv`
- [ ] `results/phase4/subtype_clinical_profiles.csv`
- [ ] `results/phase4/subtype_prediction_cv_results.json`
- [ ] `results/phase4/sample_size_comparison.csv`

**Documentation**:
- [ ] `Docs/PHASE4_SUBTYPE_CHARACTERIZATION_REPORT.md`
- [ ] `Docs/PHASE4_CLINICAL_TRIAL_ENRICHMENT_REPORT.md`
- [ ] `PHASE4_COMPLETION_SUMMARY.md`

**Visualizations**:
- [ ] Trajectory alignment before/after plots
- [ ] Dendrogram and silhouette plots
- [ ] 6-panel subtype characterization dashboard
- [ ] ROC curves and confusion matrix
- [ ] Power curves and cost savings chart

### Expected Scientific Contributions

1. **Novel Methodology**: First application of GNN-based patient similarity to PD subtype discovery

2. **Clinical Impact**: 
   - Personalized prognosis (subtype-specific trajectories)
   - Clinical trial efficiency (30-43% sample size reduction)

3. **Validation**: 
   - Internal: 5-fold cross-validation
   - External: Test on PDBP cohort (if available)

4. **Publication Target**: 
   - **npj Parkinson's Disease** (Nature portfolio)
   - **Movement Disorders** (high impact clinical journal)

### Integration with Phases 2-3

**Reuse from Phase 2**:
- ✅ GATConv layers and attention mechanisms
- ✅ Patient similarity graph construction
- ✅ Cross-validation framework

**Extend from Phase 3**:
- ✅ Multimodal features (for complete-case subtype characterization)
- ✅ Imaging signatures (DAT-SPECT, FreeSurfer in subtype profiles)

**New Contributions**:
- ✅ Longitudinal trajectory analysis
- ✅ Latent time alignment
- ✅ Subtype discovery algorithms
- ✅ Clinical trial simulation framework

---

## Timeline & Milestones

**Week 1-2**: Tasks 4.1-4.2 (Data prep + Time alignment)
- Milestone: Aligned trajectory dataset ready

**Week 3-4**: Task 4.3 (Trajectory clustering)
- Milestone: Subtypes discovered and validated

**Week 5**: Task 4.4 (Subtype characterization)
- Milestone: Comprehensive characterization report

**Week 6**: Task 4.5 (Baseline prediction model)
- Milestone: AUC ≥ 0.75 achieved

**Week 7**: Task 4.6 (Clinical trial simulation)
- Milestone: Sample size reduction demonstrated

**Week 8**: Integration, documentation, manuscript draft

---

## Next Steps

1. **Immediate**: Start with Task 4.1 (Longitudinal Data Preparation)
2. **Parallel**: Review literature on LTJMM implementations (Task 4.2)
3. **Future**: Plan Phase 5 validation and Phase 6 explainability

---

## Questions to Address During Implementation

1. **Optimal number of subtypes** (k=2 or k=3)?
   - Will be determined by clustering validation metrics

2. **Trajectory features** (linear slopes vs polynomial fits)?
   - Start with linear, explore nonlinear if needed

3. **Patient similarity metric** for graphs (cosine vs Euclidean)?
   - Reuse cosine similarity from Phase 2 (worked well)

4. **Handling missing longitudinal data**?
   - Require minimum 3 visits (balances coverage vs quality)

5. **External validation dataset**?
   - Explore PDBP (Parkinson's Disease Biomarkers Program) if accessible

---

**Ready to start with Task 4.1?** Let me know and I'll create the first implementation file! 🚀
