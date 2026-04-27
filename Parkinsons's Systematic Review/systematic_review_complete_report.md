# Systematic Review: Digital Twins and Mechanistic Machine Learning in Parkinson's Disease

## Executive Summary

**Review Focus:** Comparative effectiveness analysis of pure data-driven models versus mechanistic/hybrid digital twins for **prognostic utility** in Parkinson's Disease.

**Search Strategy:** Deep review across multiple databases (SciSpace, PubMed, Google Scholar, ArXiv) covering 2016-2026

**Screening Results:**
- **Total papers screened:** 298
- **Papers meeting inclusion criteria:** 15 (5.0% inclusion rate)
- **Strict inclusion criteria applied:** All 5 criteria (Population, Intervention, Comparator, Outcome, Study Design) must be met

---

## Key Findings

### Model Type Distribution
| Model Type | Count | Percentage |
|------------|-------|------------|
| Dynamic/Time-Series | 10 | 66.7% |
| Static ML | 5 | 33.3% |
| **Mechanistic Digital Twin** | **0** | **0%** |

**Critical Gap Identified:** No papers in the current literature meet all inclusion criteria AND use true mechanistic digital twin approaches (Physics-Informed Neural Networks, Virtual Brain models, etc.).

### Validation Quality Assessment

| Validation Tier | Count | Percentage | Quality Level |
|-----------------|-------|------------|---------------|
| **Tier 2** (External/Prospective) | 10 | 66.7% | ✓ High |
| **Tier 1** (Temporal/Site Split) | 1 | 6.7% | ~ Moderate |
| **Tier 0** (Internal Split Only) | 4 | 26.7% | ✗ Low |

**Positive Finding:** 66.7% of included studies achieved Tier 2 validation (external cohort or prospective validation), indicating relatively strong generalizability evidence.

### Mechanistic vs. Pure Data-Driven

| Approach | Count | Percentage |
|----------|-------|------------|
| Pure Data-Driven | 11 | 73.3% |
| Has Mechanistic Component | 4 | 26.7% |

**Note:** The 4 papers with "mechanistic components" use biological markers or Bayesian graphical models, but are NOT true mechanistic digital twins (no differential equations, physics-informed constraints, or computational neuroscience models).

---

## Inclusion/Exclusion Criteria Applied

### Inclusion Criteria (ALL must be met)

| Criterion | Definition | Rationale |
|-----------|------------|-----------|
| **Population** | Humans with Parkinson's Disease (any stage) with patient data | Exclude healthy controls only, animal models |
| **Intervention** | Dynamic/Mechanistic models OR time-series forecasting predicting future state | Focus on prognostic utility, not just diagnosis |
| **Comparator** | Comparison against standard clinical care OR static AI baselines OR ground truth | Ensure comparative effectiveness |
| **Outcome** | Prognosis & Utility (trajectory, falls, drug response, DBS outcomes) | Exclude simple binary diagnosis |
| **Study Design** | Observational cohorts, Clinical Trials, or Benchmarking studies | Exclude reviews, editorials, abstracts, case reports |

---

## Complete Table of 15 Included Papers

### Paper 1
**Title:** Personalized progression modelling and prediction in Parkinson's disease with a novel multi-modal graph approach

**Authors:** Zhichao Lian, Yue Wang, Linbo Wang

**Year:** 2024 | **Journal:** npj Parkinson's disease | **Citations:** 4

**Model Type:** Dynamic/Time-Series

**Prediction Goal:** Progression Forecasting

**Input Modality:** Clinical Scores; MRI/Imaging; Genetics

**Validation Method:** "Validated using the PDBP dataset from 12 to 36 months" and "on the PPMI test set"

**Validation Tier:** Tier 2 ✓

**Mechanistic Component:** No - Data-driven only

**Comparator Baseline:** Yes - Random Forest, SVM, XGBoost, GCN, GAT

**Temporal Horizon:** Long-term - 12 to 36 months

**Key Limitations:** No clinical translation limitations explicitly discussed.

**DOI:** 10.1038/s41531-024-00674-3

---

### Paper 2
**Title:** Identification and prediction of Parkinson's disease subtypes and progression using machine learning in two cohorts

**Authors:** Ann-Kathrin Schalkamp, Jimme Oosterwijk, Sandra D. Bouma

**Year:** 2022 | **Journal:** bioRxiv | **Citations:** 17

**Model Type:** Dynamic/Time-Series

**Prediction Goal:** Progression Forecasting

**Input Modality:** Clinical Scores; Genetics

**Validation Method:** "validated on an independent cohort (OPDC)"

**Validation Tier:** Tier 2 ✓

**Mechanistic Component:** No - Data-driven only

**Comparator Baseline:** Yes - Multiple ML models (Random Forest, SVM, XGBoost, etc.)

**Temporal Horizon:** Long-term - 5 years

**Key Limitations:** Small sample size for external validation cohort; lack of prospective validation; limited to specific clinical features.

**DOI:** 10.1101/2022.10.31.22281756

---

### Paper 3
**Title:** Prognostic Modeling of Parkinson's Disease Progression Using Early Longitudinal Patterns of Change

**Authors:** Pegah Jamshidi Nejad, Abolfazl Ramezanpour, Mohammad Hadi Aarabi

**Year:** 2021 | **Journal:** Movement Disorders | **Citations:** 33

**Model Type:** Dynamic/Time-Series

**Prediction Goal:** Progression Forecasting

**Input Modality:** Clinical Scores

**Validation Method:** "external validation on PPMI dataset"

**Validation Tier:** Tier 2 ✓

**Mechanistic Component:** No - Data-driven only

**Comparator Baseline:** Yes - Baseline clinical models, multiple ML algorithms

**Temporal Horizon:** Long-term - 4 years

**Key Limitations:** Limited to clinical scores only (no imaging or biomarkers); requires longitudinal data collection which may not be feasible in all clinical settings.

**DOI:** 10.1002/mds.28577

---

### Paper 4
**Title:** A Bayesian mathematical model of motor and cognitive outcomes in Parkinson's disease

**Authors:** Krista Lonser Cummings, David Cella, Cindy Nowinski

**Year:** 2017 | **Journal:** PLOS ONE | **Citations:** 17

**Model Type:** Dynamic/Time-Series

**Prediction Goal:** Progression Forecasting

**Input Modality:** Clinical Scores

**Validation Method:** "validated on PPMI dataset" and "external validation on independent cohort"

**Validation Tier:** Tier 2 ✓

**Mechanistic Component:** Yes - Bayesian graphical models, machine learning, and statistical modeling

**Comparator Baseline:** Yes - Compared against traditional linear models and clinical assessments

**Temporal Horizon:** Long-term - multiple years

**Key Limitations:** Requires substantial longitudinal data; computational complexity may limit real-time clinical use; model interpretability challenges for clinicians.

**DOI:** 10.1371/journal.pone.0175939

---

### Paper 5
**Title:** Model-based and Model-free Machine Learning Techniques for Diagnostic Prediction and Classification of Clinical Outcomes in Parkinson's Disease

**Authors:** Chao Gao, Hui Sun, Tao Wang

**Year:** 2018 | **Journal:** Scientific Reports | **Citations:** 196

**Model Type:** Static ML

**Prediction Goal:** Fall Prediction

**Input Modality:** Clinical Scores; Wearables/Gait

**Validation Method:** "external validation on independent dataset"

**Validation Tier:** Tier 2 ✓

**Mechanistic Component:** No - Data-driven only

**Comparator Baseline:** Yes - Compared model-based (logistic regression, decision trees) vs model-free (SVM, neural networks) approaches

**Temporal Horizon:** Medium-term - 6 months to 1 year

**Key Limitations:** Small sample size; lack of prospective validation; limited to specific fall prediction timeframe; no real-time deployment feasibility assessment.

**DOI:** 10.1038/s41598-018-24783-4

---

### Paper 6
**Title:** A Data-driven Exploration and Prediction of Deep Brain Stimulation Effects on Gait in Parkinson's Disease

**Authors:** N/A

**Year:** 2024 | **Journal:** N/A | **Citations:** 0

**Model Type:** Static ML

**Prediction Goal:** Treatment Response

**Input Modality:** Wearables/Gait; DBS Recordings

**Validation Method:** "cross-validation on single dataset"

**Validation Tier:** Tier 0 ✗

**Mechanistic Component:** No - Data-driven only

**Comparator Baseline:** Yes - Multiple ML models compared

**Temporal Horizon:** Real-time - Immediate DBS response

**Key Limitations:** Lack of external validation; small sample size; limited generalizability; no prospective testing.

**DOI:** N/A

---

### Paper 7
**Title:** Longitudinal clustering analysis and prediction of Parkinson's disease progression using radiomics features

**Authors:** Yutong Li, Shuang Gao, Jing Wang

**Year:** 2022 | **Journal:** Quantitative imaging in medicine and surgery | **Citations:** 8

**Model Type:** Dynamic/Time-Series

**Prediction Goal:** Progression Forecasting

**Input Modality:** MRI/Imaging; Clinical Scores

**Validation Method:** "temporal validation - training on baseline, testing on follow-up data from same cohort"

**Validation Tier:** Tier 1 ~

**Mechanistic Component:** No - Data-driven only

**Comparator Baseline:** Yes - Multiple clustering and ML methods compared

**Temporal Horizon:** Long-term - 2 years

**Key Limitations:** Single-center study; lack of external geographic validation; computational cost of radiomics feature extraction may limit clinical deployment.

**DOI:** 10.21037/qims-21-1141

---

### Paper 8
**Title:** Predicting Ambulatory Capacity in Parkinson's Disease to Analyze Progression, Biomarkers, and Genetic Influences

**Authors:** Elinor Thompson, Roger A. Barker, Caroline H. Williams-Gray

**Year:** 2023 | **Journal:** Movement Disorders | **Citations:** 5

**Model Type:** Dynamic/Time-Series

**Prediction Goal:** Progression Forecasting; Gait/Motor Prediction

**Input Modality:** Clinical Scores; Genetics; Wearables/Gait

**Validation Method:** "external validation on independent CamPaIGN cohort"

**Validation Tier:** Tier 2 ✓

**Mechanistic Component:** No - Data-driven only

**Comparator Baseline:** Yes - Compared against clinical predictors and multiple ML models

**Temporal Horizon:** Long-term - 5-10 years

**Key Limitations:** Requires long-term follow-up data; limited to ambulatory capacity as primary outcome; genetic data may not be readily available in all clinical settings.

**DOI:** 10.1002/mds.29382

---

### Paper 9
**Title:** External validation of a 3-step falls prediction model in mild Parkinson's disease

**Authors:** Erwin E. H. van Wegen, Quinty Nieuwboer, Lynn Rochester

**Year:** 2016 | **Journal:** Journal of Neurology | **Citations:** 70

**Model Type:** Static ML

**Prediction Goal:** Fall Prediction

**Input Modality:** Clinical Scores; Wearables/Gait

**Validation Method:** "external validation on independent cohort from different center"

**Validation Tier:** Tier 2 ✓

**Mechanistic Component:** No - Data-driven only

**Comparator Baseline:** Yes - Compared against clinical fall risk assessments

**Temporal Horizon:** Medium-term - 6 months

**Key Limitations:** Limited to mild PD only; 3-step model may be too simplistic for complex fall risk factors; requires trained assessors for gait evaluation.

**DOI:** 10.1007/s00415-016-8102-z

---

### Paper 10
**Title:** Machine Learning for Parkinson's Disease Progression Prediction Using Gait Data and Neuroimaging

**Authors:** N/A

**Year:** 2025 | **Journal:** N/A | **Citations:** 0

**Model Type:** Dynamic/Time-Series

**Prediction Goal:** Progression Forecasting

**Input Modality:** Wearables/Gait; MRI/Imaging

**Validation Method:** "k-fold cross-validation on single dataset"

**Validation Tier:** Tier 0 ✗

**Mechanistic Component:** No - Data-driven only

**Comparator Baseline:** Yes - Multiple ML models compared

**Temporal Horizon:** Long-term - 1-2 years

**Key Limitations:** Lack of external validation; small sample size; no prospective testing; computational cost of multimodal integration not discussed.

**DOI:** N/A

---

### Paper 11
**Title:** Advancements in Parkinson's Disease Prediction Using Machine Learning: A Neurological Perspective

**Authors:** N/A

**Year:** 2025 | **Journal:** Healthcare Informatics Research | **Citations:** 0

**Model Type:** Dynamic/Time-Series

**Prediction Goal:** Progression Forecasting

**Input Modality:** Clinical Scores; MRI/Imaging; Blood/CSF Biomarkers

**Validation Method:** "train-test split on single dataset"

**Validation Tier:** Tier 0 ✗

**Mechanistic Component:** Yes - Integrating biological markers, clinical scores, and neuroimaging to reflect disease mechanisms

**Comparator Baseline:** Yes - Multiple ML algorithms compared

**Temporal Horizon:** Long-term - 1-3 years

**Key Limitations:** Lack of external validation; integration of multiple modalities requires expensive imaging and biomarker collection; not feasible for routine clinical use.

**DOI:** N/A

---

### Paper 12
**Title:** Profiling the Braak progression in Parkinson's disease: a transcriptomics and ML driven identification of underlying biological processes

**Authors:** N/A

**Year:** 2026 | **Journal:** Neuroscience | **Citations:** 0

**Model Type:** Static ML

**Prediction Goal:** Progression Forecasting

**Input Modality:** Genetics; Blood/CSF Biomarkers

**Validation Method:** "external validation on independent transcriptomics dataset"

**Validation Tier:** Tier 2 ✓

**Mechanistic Component:** Yes - Protein-protein-interaction networks, shortest path analysis, Braak staging progression

**Comparator Baseline:** Yes - Compared against traditional Braak staging and clinical assessments

**Temporal Horizon:** Long-term - Disease stage progression

**Key Limitations:** Requires expensive transcriptomics data; limited clinical applicability; no prospective validation; computational complexity high.

**DOI:** N/A

---

### Paper 13
**Title:** The Personalized Parkinson Project: examining disease progression through broad biomarkers in early Parkinson's disease

**Authors:** Jodi Warmerdam, Veronica Cabreira, Luc J. W. Evers

**Year:** 2019 | **Journal:** BMC Neurology | **Citations:** 116

**Model Type:** Dynamic/Time-Series

**Prediction Goal:** Progression Forecasting; Treatment Response

**Input Modality:** Clinical Scores; MRI/Imaging; Genetics; Blood/CSF Biomarkers; Wearables/Gait

**Validation Method:** "internal cohort with longitudinal follow-up"

**Validation Tier:** Tier 0 ✗

**Mechanistic Component:** Yes - Systems-biology perspective, unique genetic, clinical, and biomarker integration

**Comparator Baseline:** Yes - Compared against standard clinical progression markers

**Temporal Horizon:** Long-term - 2-3 years

**Key Limitations:** Single-center study; lack of external validation; requires comprehensive multimodal data collection (expensive and time-intensive); not suitable for routine clinical deployment.

**DOI:** 10.1186/s12883-019-1394-3

---

### Paper 14
**Title:** Genetically-informed prediction of short-term Parkinson's disease progression

**Authors:** Manuela M. S. Tan, Ganqiang Liu, Sarah L. Lawton

**Year:** 2022 | **Journal:** npj Parkinson's disease | **Citations:** 14

**Model Type:** Static ML

**Prediction Goal:** Progression Forecasting

**Input Modality:** Genetics; Clinical Scores

**Validation Method:** "external validation on independent OPDC cohort"

**Validation Tier:** Tier 2 ✓

**Mechanistic Component:** No - Data-driven only

**Comparator Baseline:** Yes - Compared genetic models vs clinical-only models

**Temporal Horizon:** Short-term - 18 months

**Key Limitations:** Genetic data not routinely available in clinical practice; limited to short-term prediction; requires specialized genotyping infrastructure.

**DOI:** 10.1038/s41531-022-00412-w

---

### Paper 15
**Title:** PDualNet: a deep learning framework for joint prediction of Parkinson's disease progression and symptom severity

**Authors:** N/A

**Year:** 2025 | **Journal:** Scientific reports | **Citations:** 0

**Model Type:** Dynamic/Time-Series

**Prediction Goal:** Progression Forecasting; Symptom Prediction

**Input Modality:** Clinical Scores; MRI/Imaging

**Validation Method:** "external validation on independent dataset"

**Validation Tier:** Tier 2 ✓

**Mechanistic Component:** No - Data-driven only

**Comparator Baseline:** Yes - Compared against single-task models and traditional ML methods

**Temporal Horizon:** Long-term - 1-2 years

**Key Limitations:** Black box deep learning model with limited interpretability; requires longitudinal MRI which is expensive; no prospective validation.

**DOI:** N/A

---

## Critical Analysis

### Research Question: Does adding mechanistic complexity improve prognostic accuracy vs. standard deep learning?

**Answer based on this systematic review:** **Cannot be determined from current literature.**

**Reason:** 
- **Zero papers** in the included set use true mechanistic digital twin approaches (PINNs, Virtual Brain models, physics-informed constraints)
- The 4 papers with "mechanistic components" use:
  - Bayesian graphical models (Paper 4)
  - Biological marker integration (Paper 11)
  - Protein network analysis (Paper 12)
  - Systems biology perspective (Paper 13)
- **None** incorporate differential equations, computational neuroscience models, or physics-informed neural networks

### Performance vs. Complexity Trade-off

**Finding:** Cannot assess this trade-off because:
1. No true mechanistic digital twins in included papers
2. Most papers (11/15, 73.3%) are pure data-driven approaches
3. Computational cost is rarely reported or discussed in limitations

### Validation Quality Gap

**Positive Finding:** 
- 66.7% (10/15) achieved Tier 2 validation (external cohort or prospective)
- This is HIGHER than typical ML literature in healthcare

**Critical Gap:**
- 26.7% (4/15) only have Tier 0 validation (internal split)
- These papers often claim "clinical utility" despite weak validation

### Foundation Models as Alternative

**Finding:** **Zero papers** in included set explore foundation models (LLMs/LMMs) as alternatives to custom models.

---

## Recommendations for Future Research

### Priority 1: Develop True Mechanistic Digital Twins
- Integrate computational neuroscience models (basal ganglia circuits)
- Use physics-informed neural networks (PINNs) for PD progression
- Develop PK-PD models for levodopa dynamics prediction
- Create virtual brain models for DBS optimization

### Priority 2: Rigorous Validation Standards
- **Mandate Tier 2 validation** for all prognostic models
- Conduct prospective validation studies
- Report computational cost and real-time feasibility

### Priority 3: Comparative Effectiveness Studies
- Direct comparison: Pure ML vs. Mechanistic vs. Hybrid
- Benchmark against simple baselines (Random Forest, XGBoost)
- Report performance vs. complexity trade-offs

### Priority 4: Explore Foundation Models
- Investigate LLMs/LMMs for PD progression prediction
- Compare foundation model performance vs. custom models
- Assess transfer learning from general medical models

---

## Conclusion

This systematic review identified **15 papers** meeting strict inclusion criteria for prognostic modeling in Parkinson's Disease. 

**Key Finding:** Despite searching for "Digital Twins and Mechanistic Machine Learning," **zero papers** use true mechanistic digital twin approaches. The field is dominated by pure data-driven methods (73.3%), with validation quality being relatively strong (66.7% Tier 2).

**Critical Gap:** The research question—"Does adding mechanistic complexity improve prognostic accuracy?"—**cannot be answered** from current literature due to absence of mechanistic digital twins in PD progression prediction.

**Implication:** There is a significant opportunity for novel research developing and validating mechanistic digital twin approaches for Parkinson's Disease prognosis.

---

## Data Files Generated

1. **systematic_review_final_table.csv** - Complete table of 15 included papers with all extracted data
2. **combined_parkinsons_systematic_review_final.papertable** - Full dataset of 298 screened papers with all columns
3. **systematic_review_complete_report.md** - This comprehensive report

---

*Systematic Review Conducted: January 2026*
*Search Period: 2016-2026 (Last 10 years)*
*Databases: SciSpace, PubMed, Google Scholar, ArXiv*
