# Phase 5: Prodromal-to-Clinical PD Transition Modeling - Implementation Plan

## Executive Summary

**Goal**: Predict when and which prodromal individuals will convert to clinical Parkinson's Disease using survival analysis and time-varying biomarkers.

**Expected Impact**:
- Risk scores for prodromal → clinical PD conversion (0-5 years)
- Identify critical biomarker thresholds for diagnosis
- Enable early intervention trials in highest-risk individuals
- Define therapeutic windows for disease-modifying therapies

**Timeline**: 5-6 weeks for full implementation

---

## Scientific Background

### Prodromal PD Definition

**Prodromal markers** (MDS Research Criteria 2015):
- REM sleep behavior disorder (RBD)
- Hyposmia (olfactory dysfunction)
- DAT deficit on imaging
- Genetic risk (LRRK2, GBA, SNCA mutations)
- Constipation
- Depression

**PPMI Prodromal Cohorts**:
1. **Prodromal RBD**: RBD + DAT deficit (~200 patients)
2. **Prodromal Hyposmia**: Hyposmia + DAT deficit (~150 patients)
3. **Genetic Risk**: LRRK2/GBA carriers without manifest PD (~800 patients)

### Research Questions

1. **When**: Time-to-conversion prediction (months/years)
2. **Who**: Which prodromal individuals are highest risk
3. **Biomarker thresholds**: What values predict imminent conversion
4. **Trajectories**: How do biomarkers evolve pre-diagnosis

---

## Phase 5 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│           PHASE 5: PRODROMAL TRANSITION MODELING                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────┐
    │  Task 5.1: Prodromal Cohort Identification         │
    │  - Define prodromal criteria (RBD, hyposmia, etc.) │
    │  - Extract converters vs non-converters            │
    │  - Calculate time-to-conversion                    │
    └────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────┐
    │  Task 5.2: Time-Varying Biomarker Extraction       │
    │  - Longitudinal DAT-SPECT decline                  │
    │  - Motor symptom emergence (UPDRS)                 │
    │  - Cognitive changes (MoCA trajectory)             │
    │  - Non-motor symptom accumulation                  │
    └────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────┐
    │  Task 5.3: Cox Proportional Hazards Model          │
    │  - Baseline predictor model                        │
    │  - Time-varying covariate extensions               │
    │  - Competing risks (PD vs other parkinsonism)      │
    └────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────┐
    │  Task 5.4: Neural Survival Model (DeepSurv)        │
    │  - Deep learning for complex interactions          │
    │  - Personalized survival curves                    │
    │  - Compare to Cox model                            │
    └────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────┐
    │  Task 5.5: Biomarker Threshold Identification      │
    │  - Critical DAT-SPECT SBR values                   │
    │  - UPDRS score thresholds                          │
    │  - Non-motor symptom count cutoffs                 │
    └────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────┐
    │  Task 5.6: Risk Stratification Tool                │
    │  - Individual risk scores (0-5 year horizon)       │
    │  - Interactive nomogram                            │
    │  - Clinical decision support dashboard             │
    └────────────────────────────────────────────────────┘
```

---

## Task 5.1: Prodromal Cohort Identification

### Objective
Identify and characterize prodromal individuals in PPMI dataset, classify converters vs non-converters.

### Data Requirements

**Source Files**:
```python
prodromal_data_files = {
    'cohort_definition': 'Participant_Status_18Sep2025.csv',
    'demographics': 'Demographics_18Sep2025.csv',
    'rbd': 'REM_Sleep_Disorder_Questionnaire_18Sep2025.csv',
    'smell': 'University_of_Pennsylvania_Smell_ID_Test_18Sep2025.csv',
    'datscan': 'Xing_Core_Lab_-_Quant_SBR_18Sep2025.csv',
    'genetics': 'iu_genetic_consensus_20250515_18Sep2025.csv',
    'motor': 'MDS-UPDRS_Part_III_18Sep2025.csv',
    'cognitive': 'Montreal_Cognitive_Assessment__MoCA__18Sep2025.csv'
}
```

### Prodromal Criteria

**MDS Research Criteria for Prodromal PD** (adapted for PPMI):
```python
prodromal_criteria = {
    'rbd_positive': {
        'description': 'REM sleep behavior disorder',
        'variable': 'RBD_positive',  # PPMI RBD questionnaire
        'threshold': 'score >= 5 on RBD screening questionnaire'
    },
    'hyposmia': {
        'description': 'Olfactory dysfunction',
        'variable': 'UPSIT_total',  # University of Pennsylvania Smell Test
        'threshold': 'UPSIT < 25th percentile for age/sex'
    },
    'dat_deficit': {
        'description': 'Abnormal DAT imaging',
        'variable': 'caudate_putamen_sbr_mean',
        'threshold': 'SBR < 1.5 (or < 2 SD below age-matched controls)'
    },
    'genetic_risk': {
        'description': 'Pathogenic mutation carrier',
        'variable': 'genetic_status',
        'threshold': 'LRRK2_positive or GBA_positive'
    },
    'motor_subtle': {
        'description': 'Subtle motor signs (not meeting PD diagnosis)',
        'variable': 'UPDRS_III_total',
        'threshold': 'UPDRS-III > 0 but < diagnostic threshold'
    }
}
```

### Conversion Definition

**Clinical PD Diagnosis** (PPMI criteria):
```python
def determine_conversion_status(participant_df: pd.DataFrame) -> Dict:
    """
    Identify conversion to clinical PD.
    
    Conversion criteria:
    1. COHORT_DEFINITION changes from "Prodromal" → "Parkinson's Disease"
    2. OR UPDRS-III >= 15 with clinician diagnosis
    3. OR new dopaminergic medication prescription
    
    Returns:
        {
            'converter': bool,
            'conversion_event_id': str,  # Visit when conversion occurred
            'time_to_conversion_months': float,
            'conversion_date': datetime
        }
    """
    pass
```

### Implementation

**Step 1.1**: Create `task_5_1_prodromal_cohort_identification.py`

```python
class ProdromalCohortIdentifier:
    """
    Identify and classify prodromal participants in PPMI.
    """
    
    def __init__(self, ppmi_data_dir: Path):
        self.data_dir = ppmi_data_dir
        self.cohort_df = None
        self.converters = None
        self.non_converters = None
        
    def load_ppmi_prodromal_data(self) -> pd.DataFrame:
        """
        Load all relevant files for prodromal classification.
        
        Returns:
            DataFrame with PATNO, baseline prodromal markers, 
            longitudinal follow-up, conversion status
        """
        # Load cohort definitions
        cohort = pd.read_csv(self.data_dir / 'Participant_Status_18Sep2025.csv')
        
        # Filter to prodromal cohorts
        prodromal_cohorts = [
            'Prodromal',
            'Prodromal RBD',
            'Prodromal Hyposmia',
            'LRRK2',
            'GBA'
        ]
        
        prodromal_df = cohort[
            cohort['COHORT_DEFINITION'].isin(prodromal_cohorts)
        ]
        
        return prodromal_df
    
    def classify_prodromal_markers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary indicators for each prodromal marker.
        
        Returns:
            DataFrame with columns: PATNO, rbd_positive, hyposmia, 
            dat_deficit, genetic_risk, motor_subtle, prodromal_score
        """
        markers = df.copy()
        
        # RBD
        markers['rbd_positive'] = markers['RBD_score'] >= 5
        
        # Hyposmia (age/sex adjusted percentiles)
        markers['hyposmia'] = self._classify_hyposmia(markers)
        
        # DAT deficit
        markers['dat_deficit'] = markers['caudate_sbr_mean'] < 1.5
        
        # Genetic risk
        markers['genetic_risk'] = (
            (markers['LRRK2'] == 'Positive') | 
            (markers['GBA'] == 'Positive')
        )
        
        # Subtle motor signs
        markers['motor_subtle'] = (
            (markers['UPDRS_III_total'] > 0) & 
            (markers['UPDRS_III_total'] < 15)
        )
        
        # Prodromal likelihood score (count of markers)
        marker_cols = ['rbd_positive', 'hyposmia', 'dat_deficit', 
                      'genetic_risk', 'motor_subtle']
        markers['prodromal_score'] = markers[marker_cols].sum(axis=1)
        
        return markers
    
    def identify_converters(
        self, 
        longitudinal_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Classify participants as converters vs non-converters.
        
        Returns:
            converters_df: PATNO, conversion_event, time_to_conversion_months
            non_converters_df: PATNO, last_follow_up_event, follow_up_months, censored=True
        """
        converters = []
        non_converters = []
        
        for patno in longitudinal_df['PATNO'].unique():
            patient_data = longitudinal_df[longitudinal_df['PATNO'] == patno].sort_values('EVENT_ID')
            
            # Check for conversion
            conversion_row = self._detect_conversion(patient_data)
            
            if conversion_row is not None:
                converters.append({
                    'PATNO': patno,
                    'conversion_event': conversion_row['EVENT_ID'],
                    'time_to_conversion_months': conversion_row['months_since_baseline'],
                    'censored': False
                })
            else:
                # Non-converter (censored observation)
                last_visit = patient_data.iloc[-1]
                non_converters.append({
                    'PATNO': patno,
                    'last_event': last_visit['EVENT_ID'],
                    'follow_up_months': last_visit['months_since_baseline'],
                    'censored': True
                })
        
        return pd.DataFrame(converters), pd.DataFrame(non_converters)
    
    def _detect_conversion(self, patient_data: pd.DataFrame) -> Optional[pd.Series]:
        """
        Detect conversion event for single patient.
        
        Conversion criteria (any of):
        1. COHORT changes to "Parkinson's Disease"
        2. UPDRS-III >= 15 with clinician diagnosis
        3. Dopaminergic medication started
        """
        for idx, row in patient_data.iterrows():
            if row['COHORT_DEFINITION'] == "Parkinson's Disease":
                return row
            
            if row['UPDRS_III_total'] >= 15 and row['PD_diagnosis'] == 1:
                return row
            
            if row['dopaminergic_medication'] == 1:
                return row
        
        return None
    
    def generate_cohort_report(self) -> Dict:
        """
        Summary statistics for prodromal cohort.
        
        Returns:
            {
                'total_prodromal': int,
                'converters': int,
                'non_converters': int,
                'conversion_rate': float,
                'median_time_to_conversion_months': float,
                'median_follow_up_months': float,
                'prodromal_marker_distribution': Dict
            }
        """
        pass
```

**Expected Output**:
```python
prodromal_cohort_summary = {
    'total_prodromal': 1150,
    'converters': 187,  # ~16% conversion rate
    'non_converters': 963,
    'conversion_rate': 0.163,
    'median_time_to_conversion_months': 36.0,
    'median_follow_up_censored_months': 48.0,
    'prodromal_markers': {
        'rbd_positive': 412,
        'hyposmia': 389,
        'dat_deficit': 654,
        'genetic_risk': 802,
        'motor_subtle': 267
    }
}
```

**Deliverables**:
- ✅ `task_5_1_prodromal_cohort_identification.py` (500 lines)
- ✅ `data/prodromal/prodromal_cohort_baseline.csv`
- ✅ `data/prodromal/converters.csv`
- ✅ `data/prodromal/non_converters.csv`
- ✅ `data/prodromal/prodromal_cohort_report.json`

---

## Task 5.2: Time-Varying Biomarker Extraction

### Objective
Extract longitudinal biomarker trajectories leading up to conversion (or censoring).

### Key Biomarkers

**Imaging biomarkers** (primary):
```python
imaging_biomarkers = {
    'dat_spect': [
        'caudate_sbr_left', 'caudate_sbr_right',
        'putamen_sbr_left', 'putamen_sbr_right',
        'caudate_putamen_sbr_mean',
        'asymmetry_index',
        'sbr_slope'  # Rate of decline
    ]
}
```

**Motor biomarkers**:
```python
motor_biomarkers = {
    'updrs_iii_total': 'Total motor score',
    'tremor_score': 'Tremor subscore',
    'rigidity_score': 'Rigidity subscore',
    'bradykinesia_score': 'Bradykinesia subscore',
    'pigd_score': 'Postural instability and gait difficulty'
}
```

**Cognitive biomarkers**:
```python
cognitive_biomarkers = {
    'moca_total': 'Montreal Cognitive Assessment',
    'executive_function': 'Executive function domain',
    'memory': 'Memory domain',
    'attention': 'Attention domain'
}
```

**Non-motor biomarkers**:
```python
non_motor_biomarkers = {
    'rbd_severity': 'RBD symptom severity',
    'constipation_score': 'Constipation severity',
    'orthostatic_hypotension': 'OH presence',
    'depression_score': 'Depression scale',
    'anxiety_score': 'Anxiety scale',
    'upsit_score': 'Smell test score'
}
```

### Implementation

**Step 2.1**: Create `task_5_2_time_varying_biomarkers.py`

```python
class TimeVaryingBiomarkerExtractor:
    """
    Extract longitudinal biomarker trajectories for survival analysis.
    """
    
    def __init__(self, prodromal_cohort: pd.DataFrame):
        self.cohort = prodromal_cohort
        
    def extract_longitudinal_biomarkers(
        self, 
        ppmi_data_dir: Path
    ) -> pd.DataFrame:
        """
        Extract all biomarker measurements over time.
        
        Returns:
            Long-format DataFrame:
            PATNO | EVENT_ID | months_since_baseline | biomarker_name | biomarker_value
        """
        pass
    
    def calculate_biomarker_slopes(
        self, 
        longitudinal_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate rate of change for each biomarker.
        
        For each patient:
        - Fit linear regression: biomarker ~ time
        - Extract slope (rate of decline/increase)
        
        Returns:
            PATNO | biomarker_name | slope | intercept | r_squared
        """
        from scipy.stats import linregress
        
        slopes = []
        
        for patno in longitudinal_data['PATNO'].unique():
            patient_data = longitudinal_data[longitudinal_data['PATNO'] == patno]
            
            for biomarker in patient_data['biomarker_name'].unique():
                biomarker_data = patient_data[
                    patient_data['biomarker_name'] == biomarker
                ]
                
                if len(biomarker_data) >= 2:  # Need at least 2 points
                    result = linregress(
                        biomarker_data['months_since_baseline'],
                        biomarker_data['biomarker_value']
                    )
                    
                    slopes.append({
                        'PATNO': patno,
                        'biomarker': biomarker,
                        'slope': result.slope,
                        'intercept': result.intercept,
                        'r_squared': result.rvalue ** 2,
                        'p_value': result.pvalue
                    })
        
        return pd.DataFrame(slopes)
    
    def create_time_varying_dataset(
        self,
        longitudinal_data: pd.DataFrame,
        conversion_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Format data for Cox regression with time-varying covariates.
        
        Returns:
            Wide-format DataFrame suitable for survival analysis:
            PATNO | start_time | stop_time | event | dat_sbr | updrs_iii | moca | ...
        """
        # Reshape from long to wide format
        # Create time intervals for each measurement update
        pass
```

**Deliverables**:
- ✅ `task_5_2_time_varying_biomarkers.py` (450 lines)
- ✅ `data/prodromal/longitudinal_biomarkers.csv`
- ✅ `data/prodromal/biomarker_slopes.csv`
- ✅ `data/prodromal/time_varying_dataset.csv`

---

## Task 5.3: Cox Proportional Hazards Model

### Objective
Build survival model predicting time-to-conversion using Cox regression.

### Cox Model Formulation

**Hazard function**:
```
h(t | X) = h₀(t) * exp(β₁X₁ + β₂X₂ + ... + βₚXₚ)

Where:
- h(t | X): Hazard of conversion at time t given covariates X
- h₀(t): Baseline hazard function
- β: Regression coefficients (log hazard ratios)
- X: Covariate values (biomarkers)
```

**Interpretation**:
- Hazard ratio (HR) = exp(β)
- HR > 1: Increased risk of conversion
- HR < 1: Decreased risk (protective)

### Implementation

**Step 3.1**: Create `task_5_3_cox_survival_model.py`

```python
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

class CoxProdromalModel:
    """
    Cox proportional hazards model for prodromal PD conversion.
    """
    
    def __init__(self):
        self.model = CoxPHFitter()
        self.results = None
        
    def prepare_survival_data(
        self,
        cohort_df: pd.DataFrame,
        conversion_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare data for Cox regression.
        
        Required columns:
        - duration: Time to event or censoring (months)
        - event: 1 if converted, 0 if censored
        - covariates: All predictor variables
        """
        survival_df = cohort_df.merge(
            conversion_df[['PATNO', 'time_to_conversion_months', 'censored']],
            on='PATNO'
        )
        
        # Rename for lifelines
        survival_df['duration'] = survival_df['time_to_conversion_months']
        survival_df['event'] = ~survival_df['censored']
        
        return survival_df
    
    def fit_baseline_model(
        self,
        survival_data: pd.DataFrame,
        covariates: List[str]
    ) -> Dict:
        """
        Fit Cox model with baseline predictors only.
        
        Covariates:
        - age_at_baseline
        - sex
        - genetic_risk (LRRK2/GBA)
        - rbd_positive
        - hyposmia
        - baseline_dat_sbr
        - baseline_updrs_iii
        - baseline_moca
        - prodromal_score (count of markers)
        
        Returns:
            {
                'coefficients': pd.Series,  # β values
                'hazard_ratios': pd.Series,  # exp(β)
                'p_values': pd.Series,
                'concordance_index': float,  # C-index (AUC analog)
                'log_likelihood': float
            }
        """
        self.model.fit(
            survival_data,
            duration_col='duration',
            event_col='event',
            formula=' + '.join(covariates)
        )
        
        c_index = concordance_index(
            survival_data['duration'],
            -self.model.predict_partial_hazard(survival_data),
            survival_data['event']
        )
        
        return {
            'coefficients': self.model.params_,
            'hazard_ratios': np.exp(self.model.params_),
            'p_values': self.model.summary['p'],
            'confidence_intervals': self.model.confidence_intervals_,
            'concordance_index': c_index,
            'log_likelihood': self.model.log_likelihood_
        }
    
    def fit_time_varying_model(
        self,
        time_varying_data: pd.DataFrame
    ) -> Dict:
        """
        Fit Cox model with time-varying covariates.
        
        Accounts for changes in biomarkers over time.
        
        Data format:
        PATNO | start | stop | event | dat_sbr | updrs_iii | ...
        """
        from lifelines import CoxTimeVaryingFitter
        
        ctv_model = CoxTimeVaryingFitter()
        
        ctv_model.fit(
            time_varying_data,
            id_col='PATNO',
            event_col='event',
            start_col='start',
            stop_col='stop'
        )
        
        return {
            'coefficients': ctv_model.params_,
            'hazard_ratios': np.exp(ctv_model.params_),
            'p_values': ctv_model.summary['p']
        }
    
    def test_proportional_hazards_assumption(
        self,
        survival_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Test if proportional hazards assumption holds.
        
        Uses Schoenfeld residuals test.
        
        Returns:
            DataFrame with p-values for each covariate
            (p < 0.05 indicates violation)
        """
        from lifelines.statistics import proportional_hazard_test
        
        results = proportional_hazard_test(
            self.model,
            survival_data,
            time_transform='rank'
        )
        
        return results.summary
    
    def stratified_analysis(
        self,
        survival_data: pd.DataFrame,
        stratify_by: str = 'genetic_risk'
    ) -> Dict:
        """
        Stratified analysis by genetic risk, RBD, etc.
        
        Compares conversion hazards across subgroups.
        """
        from lifelines.statistics import multivariate_logrank_test
        
        # Log-rank test
        results = multivariate_logrank_test(
            survival_data['duration'],
            survival_data[stratify_by],
            survival_data['event']
        )
        
        return {
            'test_statistic': results.test_statistic,
            'p_value': results.p_value,
            'summary': results.summary
        }
```

**Expected Results**:
```python
cox_model_results = {
    'baseline_model': {
        'concordance_index': 0.72,  # Target: 0.70-0.75
        'significant_predictors': [
            {'variable': 'baseline_dat_sbr', 'HR': 0.42, 'p': 0.001},
            {'variable': 'rbd_positive', 'HR': 2.31, 'p': 0.003},
            {'variable': 'prodromal_score', 'HR': 1.58, 'p': 0.012},
            {'variable': 'genetic_risk', 'HR': 1.87, 'p': 0.024}
        ]
    },
    'time_varying_model': {
        'concordance_index': 0.76,  # Better with time-varying covariates
        'significant_time_varying': [
            {'variable': 'dat_sbr_slope', 'HR': 2.15, 'p': 0.002},
            {'variable': 'updrs_iii_change', 'HR': 1.12, 'p': 0.018}
        ]
    }
}
```

**Deliverables**:
- ✅ `task_5_3_cox_survival_model.py` (550 lines)
- ✅ `results/phase5/cox_model_results.json`
- ✅ `results/phase5/hazard_ratios.csv`
- ✅ Visualization: Kaplan-Meier curves by risk groups, Forest plot of HRs

---

## Task 5.4: Neural Survival Model (DeepSurv)

### Objective
Implement deep learning survival model to capture non-linear interactions between biomarkers.

### DeepSurv Architecture

```python
import torch
import torch.nn as nn
from pycox.models import CoxPH

class DeepSurvModel(nn.Module):
    """
    Neural network for survival analysis.
    
    Extends Cox model with deep neural network.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        activation: str = 'relu'
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU() if activation == 'relu' else nn.Tanh(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (log hazard)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, input_dim)
            
        Returns:
            Log hazard predictions (batch_size, 1)
        """
        return self.network(x)


class DeepSurvTrainer:
    """
    Train and evaluate DeepSurv model.
    """
    
    def __init__(self, model: DeepSurvModel):
        self.model = model
        
    def fit(
        self,
        X_train: np.ndarray,
        durations_train: np.ndarray,
        events_train: np.ndarray,
        X_val: np.ndarray,
        durations_val: np.ndarray,
        events_val: np.ndarray,
        num_epochs: int = 100,
        learning_rate: float = 0.001
    ) -> Dict:
        """
        Train DeepSurv model using Cox partial likelihood loss.
        """
        from pycox.models import CoxPH
        from pycox.evaluation import EvalSurv
        
        # Use pycox wrapper
        net = self.model
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        
        cox_model = CoxPH(net, optimizer)
        
        # Prepare data
        train_data = (X_train, (durations_train, events_train))
        val_data = (X_val, (durations_val, events_val))
        
        # Train
        log = cox_model.fit(*train_data, val_data=val_data, epochs=num_epochs)
        
        # Evaluate
        surv_funcs = cox_model.predict_surv_df(X_val)
        ev = EvalSurv(surv_funcs, durations_val, events_val, censor_surv='km')
        
        c_index = ev.concordance_td()
        integrated_brier_score = ev.integrated_brier_score(
            np.linspace(0, durations_val.max(), 100)
        )
        
        return {
            'concordance_index': c_index,
            'integrated_brier_score': integrated_brier_score,
            'training_log': log
        }
    
    def predict_individual_risk(
        self,
        X: np.ndarray,
        time_points: np.ndarray = np.array([12, 24, 36, 48, 60])
    ) -> pd.DataFrame:
        """
        Predict individual survival probabilities at specific time points.
        
        Args:
            X: Patient features (N, input_dim)
            time_points: Months at which to predict survival (e.g., 1, 2, 3, 4, 5 years)
            
        Returns:
            DataFrame with columns: PATNO, 12_month_survival, 24_month_survival, ...
        """
        surv_funcs = self.model.predict_surv_df(X)
        
        predictions = []
        for idx, surv_func in surv_funcs.iterrows():
            pred = {'PATNO': idx}
            for t in time_points:
                # Survival probability at time t
                pred[f'{t}_month_survival'] = surv_func.loc[t] if t in surv_func.index else np.nan
                # Risk = 1 - survival
                pred[f'{t}_month_risk'] = 1 - pred[f'{t}_month_survival']
            predictions.append(pred)
        
        return pd.DataFrame(predictions)
```

**Deliverables**:
- ✅ `task_5_4_deep_surv_model.py` (500 lines)
- ✅ `models/deepsurv_prodromal_best.pth`
- ✅ `results/phase5/deepsurv_results.json`
- ✅ `results/phase5/individual_risk_scores.csv`

---

## Task 5.5: Biomarker Threshold Identification

### Objective
Identify critical biomarker values that predict imminent conversion (within 6-12 months).

### Threshold Analysis Methods

**Method 1: Time-dependent ROC curves**
```python
from lifelines.utils import concordance_index
from sklearn.metrics import roc_curve, auc

def time_dependent_roc(
    biomarker_values: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    time_point: float = 12.0  # months
) -> Dict:
    """
    ROC curve for predicting conversion within specified time.
    
    Returns:
        {
            'fpr': np.ndarray,
            'tpr': np.ndarray,
            'auc': float,
            'optimal_threshold': float,
            'sensitivity_at_threshold': float,
            'specificity_at_threshold': float
        }
    """
    # Define outcome: converted within time_point?
    outcome = (durations <= time_point) & (events == 1)
    
    fpr, tpr, thresholds = roc_curve(outcome, biomarker_values)
    roc_auc = auc(fpr, tpr)
    
    # Optimal threshold (Youden's index)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'sensitivity': tpr[optimal_idx],
        'specificity': 1 - fpr[optimal_idx]
    }
```

**Method 2: Recursive partitioning (survival trees)**
```python
from sksurv.tree import SurvivalTree

def identify_critical_thresholds(
    X: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray
) -> Dict:
    """
    Use survival trees to identify biomarker cutpoints.
    
    Returns:
        {
            'biomarker': str,
            'threshold': float,
            'log_rank_p': float,
            'high_risk_hr': float,  # Hazard ratio above vs below threshold
            'median_survival_low': float,
            'median_survival_high': float
        }
    """
    survival_tree = SurvivalTree(max_depth=3, min_samples_split=50)
    
    # Prepare survival data
    y = np.array(
        [(e, d) for e, d in zip(events, durations)],
        dtype=[('event', bool), ('time', float)]
    )
    
    survival_tree.fit(X, y)
    
    # Extract split points from tree
    thresholds = extract_tree_splits(survival_tree)
    
    return thresholds
```

**Method 3: Maximally selected rank statistics**
```python
def maximal_chi_square_test(
    biomarker: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray
) -> Dict:
    """
    Find cutpoint that maximizes separation in survival curves.
    
    Tests all possible cutpoints and selects optimal based on log-rank test.
    """
    from lifelines.statistics import logrank_test
    
    best_p = 1.0
    best_threshold = None
    best_chi2 = 0
    
    # Try all unique values as cutpoints
    for threshold in np.percentile(biomarker, range(10, 91, 5)):
        group = (biomarker <= threshold).astype(int)
        
        # Log-rank test
        result = logrank_test(
            durations[group == 0], durations[group == 1],
            events[group == 0], events[group == 1]
        )
        
        if result.test_statistic > best_chi2:
            best_chi2 = result.test_statistic
            best_p = result.p_value
            best_threshold = threshold
    
    return {
        'threshold': best_threshold,
        'chi2': best_chi2,
        'p_value': best_p
    }
```

**Expected Thresholds**:
```python
critical_thresholds = {
    'dat_sbr_caudate_putamen': {
        'threshold': 1.2,
        'interpretation': 'SBR < 1.2 predicts conversion within 18 months',
        'sensitivity': 0.78,
        'specificity': 0.71,
        'auc': 0.81
    },
    'updrs_iii_total': {
        'threshold': 8.5,
        'interpretation': 'UPDRS-III > 8.5 indicates imminent clinical diagnosis',
        'sensitivity': 0.82,
        'specificity': 0.69,
        'auc': 0.76
    },
    'prodromal_score': {
        'threshold': 3,
        'interpretation': '3+ prodromal markers = high conversion risk',
        'median_time_to_conversion': 24,  # months
        'hazard_ratio': 3.2
    }
}
```

**Deliverables**:
- ✅ `task_5_5_biomarker_thresholds.py` (400 lines)
- ✅ `results/phase5/biomarker_thresholds.csv`
- ✅ `results/phase5/time_dependent_roc_curves.png`

---

## Task 5.6: Risk Stratification Tool

### Objective
Create clinical decision support tool for risk stratification and patient counseling.

### Risk Score Calculator

```python
class ProdromalRiskCalculator:
    """
    Calculate individual conversion risk scores.
    """
    
    def __init__(
        self,
        cox_model: CoxPHFitter,
        deepsurv_model: DeepSurvModel
    ):
        self.cox_model = cox_model
        self.deepsurv_model = deepsurv_model
        
    def calculate_risk_score(
        self,
        patient_features: pd.Series,
        time_horizons: List[int] = [12, 24, 36, 48, 60]
    ) -> Dict:
        """
        Calculate personalized conversion risk.
        
        Returns:
            {
                '12_month_risk': 0.08,
                '24_month_risk': 0.18,
                '36_month_risk': 0.32,
                '48_month_risk': 0.45,
                '60_month_risk': 0.56,
                'risk_category': 'High',  # Low/Moderate/High
                'median_survival': 42  # months
            }
        """
        # Cox model predictions
        surv_func_cox = self.cox_model.predict_survival_function(patient_features)
        
        # DeepSurv predictions
        surv_func_deep = self.deepsurv_model.predict_survival_function(patient_features)
        
        # Ensemble (average)
        risks = {}
        for t in time_horizons:
            risk_cox = 1 - surv_func_cox.loc[t]
            risk_deep = 1 - surv_func_deep.loc[t]
            risks[f'{t}_month_risk'] = (risk_cox + risk_deep) / 2
        
        # Risk category
        risk_3year = risks['36_month_risk']
        if risk_3year < 0.20:
            category = 'Low'
        elif risk_3year < 0.40:
            category = 'Moderate'
        else:
            category = 'High'
        
        return {
            **risks,
            'risk_category': category,
            'median_survival': self._estimate_median_survival(surv_func_cox)
        }
    
    def create_interactive_nomogram(self) -> None:
        """
        Generate interactive nomogram for clinicians.
        
        Allows adjusting biomarker values and seeing updated risk predictions.
        """
        import plotly.graph_objects as go
        
        # Create Plotly dashboard with sliders for each biomarker
        # Real-time risk calculation as sliders move
        pass


class ClinicalDecisionSupport:
    """
    Clinical decision support dashboard.
    """
    
    def generate_patient_report(
        self,
        patient_id: str,
        risk_scores: Dict,
        biomarker_values: pd.Series,
        critical_thresholds: Dict
    ) -> str:
        """
        Generate clinical report for patient counseling.
        
        Returns:
            Markdown formatted report with:
            - Current risk stratification
            - Trajectory visualization
            - Comparison to population
            - Recommended follow-up interval
            - Clinical trial eligibility
        """
        report = f"""
        # Prodromal PD Risk Assessment Report
        **Patient ID**: {patient_id}
        **Assessment Date**: {datetime.now().strftime('%Y-%m-%d')}
        
        ## Risk Stratification
        - **Risk Category**: {risk_scores['risk_category']}
        - **3-Year Conversion Risk**: {risk_scores['36_month_risk']:.1%}
        - **5-Year Conversion Risk**: {risk_scores['60_month_risk']:.1%}
        - **Estimated Time to Conversion**: {risk_scores['median_survival']} months
        
        ## Key Biomarkers
        | Biomarker | Current Value | Critical Threshold | Status |
        |-----------|--------------|-------------------|---------|
        | DAT SBR | {biomarker_values['dat_sbr']:.2f} | {critical_thresholds['dat_sbr']:.2f} | {'⚠️ Below' if biomarker_values['dat_sbr'] < critical_thresholds['dat_sbr'] else '✓ Normal'} |
        | UPDRS-III | {biomarker_values['updrs_iii']:.0f} | {critical_thresholds['updrs_iii']:.0f} | {'⚠️ Elevated' if biomarker_values['updrs_iii'] > critical_thresholds['updrs_iii'] else '✓ Normal'} |
        | MoCA | {biomarker_values['moca']:.0f} | {critical_thresholds['moca']:.0f} | {'⚠️ Low' if biomarker_values['moca'] < critical_thresholds['moca'] else '✓ Normal'} |
        
        ## Recommendations
        - **Follow-up Interval**: {self._recommend_followup_interval(risk_scores)}
        - **Consider Clinical Trial Enrollment**: {'Yes' if risk_scores['risk_category'] == 'High' else 'Discuss with patient'}
        - **Lifestyle Interventions**: Exercise, Mediterranean diet
        
        ## Interpretation
        This patient's biomarker profile suggests {risk_scores['risk_category'].lower()} risk 
        of converting to clinical PD within the next 3-5 years. 
        {self._generate_personalized_interpretation(risk_scores, biomarker_values)}
        """
        
        return report
```

**Interactive Dashboard** (Streamlit or Gradio):
```python
import streamlit as st

def create_risk_calculator_app():
    """
    Interactive web app for risk calculation.
    """
    st.title("Prodromal PD Risk Calculator")
    
    # Input sliders
    age = st.slider("Age", 40, 85, 65)
    dat_sbr = st.slider("DAT-SPECT SBR", 0.5, 3.0, 1.5)
    updrs_iii = st.slider("UPDRS-III Total", 0, 30, 5)
    moca = st.slider("MoCA Score", 15, 30, 27)
    rbd = st.checkbox("RBD Present")
    genetic_risk = st.checkbox("LRRK2/GBA Mutation")
    
    # Calculate risk
    calculator = ProdromalRiskCalculator(cox_model, deepsurv_model)
    risk_scores = calculator.calculate_risk_score(
        pd.Series({
            'age': age,
            'dat_sbr': dat_sbr,
            'updrs_iii': updrs_iii,
            'moca': moca,
            'rbd': int(rbd),
            'genetic_risk': int(genetic_risk)
        })
    )
    
    # Display results
    st.header("Risk Assessment")
    st.metric("3-Year Risk", f"{risk_scores['36_month_risk']:.1%}")
    st.metric("Risk Category", risk_scores['risk_category'])
    
    # Visualization
    time_points = [12, 24, 36, 48, 60]
    risks = [risk_scores[f'{t}_month_risk'] for t in time_points]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_points, 
        y=risks,
        mode='lines+markers',
        name='Conversion Risk'
    ))
    fig.update_layout(
        xaxis_title="Months from Baseline",
        yaxis_title="Cumulative Conversion Risk",
        yaxis=dict(tickformat='.0%')
    )
    st.plotly_chart(fig)
```

**Deliverables**:
- ✅ `task_5_6_risk_stratification_tool.py` (600 lines)
- ✅ `app/prodromal_risk_calculator.py` (Streamlit app)
- ✅ `results/phase5/patient_risk_reports/` (directory with individual reports)
- ✅ Interactive nomogram HTML file

---

## Phase 5 Summary

### Deliverables Checklist

**Code Files** (6 tasks):
- [ ] `task_5_1_prodromal_cohort_identification.py` (500 lines)
- [ ] `task_5_2_time_varying_biomarkers.py` (450 lines)
- [ ] `task_5_3_cox_survival_model.py` (550 lines)
- [ ] `task_5_4_deep_surv_model.py` (500 lines)
- [ ] `task_5_5_biomarker_thresholds.py` (400 lines)
- [ ] `task_5_6_risk_stratification_tool.py` (600 lines)

**Total Code**: ~3,000 lines

**Data Outputs**:
- [ ] Prodromal cohort datasets (converters, non-converters)
- [ ] Longitudinal biomarker trajectories
- [ ] Survival model results (Cox + DeepSurv)
- [ ] Biomarker threshold tables
- [ ] Individual risk scores

**Documentation**:
- [ ] `Docs/PHASE5_PRODROMAL_TRANSITION_REPORT.md`
- [ ] `PHASE5_COMPLETION_SUMMARY.md`

**Clinical Tools**:
- [ ] Interactive risk calculator (Streamlit app)
- [ ] Patient report generator
- [ ] Clinical trial eligibility screener

### Expected Performance

```python
model_performance = {
    'cox_baseline_model': {
        'c_index': 0.72,  # 0.70-0.75 expected
        'top_predictors': ['dat_sbr', 'rbd', 'genetic_risk']
    },
    'cox_time_varying_model': {
        'c_index': 0.76,  # 0.74-0.78 expected
        'improvement': '+5.6% over baseline'
    },
    'deepsurv_model': {
        'c_index': 0.78,  # 0.76-0.80 expected
        'integrated_brier_score': 0.15
    },
    'biomarker_thresholds': {
        'dat_sbr_12month_prediction': {'auc': 0.81, 'threshold': 1.2},
        'updrs_iii_12month_prediction': {'auc': 0.76, 'threshold': 8.5}
    }
}
```

### Clinical Impact

1. **Personalized Risk Assessment**: Individual conversion probability curves

2. **Optimal Screening Intervals**: Risk-based follow-up schedules

3. **Clinical Trial Enrichment**: Identify high-risk individuals for prevention trials

4. **Therapeutic Window**: Define optimal timing for disease-modifying interventions

### Publication Target

- **Journal**: *Lancet Neurology* or *JAMA Neurology*
- **Impact**: High (prodromal prediction is critical unmet need)
- **Novelty**: First deep learning survival model for prodromal PD

---

## Timeline

**Week 1**: Task 5.1 (Cohort identification)
**Week 2**: Task 5.2 (Biomarker extraction)
**Week 3**: Task 5.3 (Cox models)
**Week 4**: Task 5.4 (DeepSurv)
**Week 5**: Task 5.5 (Thresholds) + Task 5.6 (Risk tool)
**Week 6**: Integration, validation, manuscript draft

---

Ready to proceed with Phase 5 after completing Phase 4! 🚀
