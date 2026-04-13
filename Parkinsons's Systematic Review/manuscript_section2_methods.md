# 2. Methods

## 2.1 Protocol and Registration

This systematic review was conducted in accordance with the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) 2020 guidelines [45]. The review protocol was not pre-registered in a public registry (e.g., PROSPERO), which is acknowledged as a limitation. However, a pre-specified protocol was developed internally prior to literature search, defining eligibility criteria, search strategy, data extraction procedures, and risk-of-bias assessment methods.

## 2.2 Eligibility Criteria

Studies were included if they met all five of the following pre-specified criteria:

### 2.2.1 Inclusion Criteria

**Criterion 1 – Population:** Studies must have enrolled human participants with clinically diagnosed Parkinson's disease at any disease stage (early, moderate, or advanced) and any subtype (tremor-dominant, postural instability and gait difficulty, or mixed). Studies were required to use real patient data; purely theoretical models or simulations without patient validation were excluded.

**Criterion 2 – Intervention:** Studies must have implemented models designed to predict future disease states or clinical outcomes. Eligible model types included:
- Dynamic or temporal models (long short-term memory networks, recurrent neural networks, time-series forecasting models)
- Mechanistic models (differential equations, physics-informed neural networks, computational neuroscience models)
- Digital twin frameworks (virtual patient models integrating longitudinal data and physiological constraints)
- Hybrid mechanistic-machine learning approaches

Static diagnostic or classification models that distinguished Parkinson's disease from healthy controls or atypical parkinsonism without a prognostic component were excluded.

**Criterion 3 – Comparator:** Studies must have included at least one of the following comparisons:
- Direct comparison to a static machine learning baseline (Random Forest, XGBoost, Logistic Regression, Support Vector Machine)
- Comparison to established clinical prediction tools or scoring systems
- Comparison to ground truth clinical outcomes with quantitative performance metrics
- Head-to-head comparison of multiple models evaluated on the same test set

**Criterion 4 – Outcome:** Studies must have reported prognostic utility measures for predicting future clinical events or disease trajectories. Eligible outcomes included:
- Disease progression forecasting (Movement Disorder Society-Unified Parkinson's Disease Rating Scale trajectory prediction, Hoehn and Yahr stage advancement)
- Treatment response prediction (levodopa response, deep brain stimulation outcomes)
- Clinical event prediction (falls, freezing of gait, cognitive decline, dyskinesia)
- Time-to-event outcomes (time to disability milestones, nursing home placement, or death)

Studies reporting only diagnostic accuracy (distinguishing Parkinson's disease from controls or atypical parkinsonism) without prognostic endpoints were excluded.

**Criterion 5 – Study Design:** Eligible study designs included:
- Observational studies with real patient data (cohort studies, registry analyses)
- Clinical trials with prognostic endpoints
- Prospective or retrospective cohort studies with longitudinal follow-up

Excluded study designs included systematic reviews, meta-analyses, editorials, commentaries, conference abstracts without full-text availability, case reports, and simulation studies without validation on real patient data.

### 2.2.2 Exclusion Criteria

Studies were excluded if they met any of the following criteria:
- Non-human studies (animal models, in vitro experiments)
- Purely diagnostic models without prognostic component
- Absence of comparator or baseline model
- Theoretical or simulation-only studies without patient data validation
- Non-English language publications
- Conference abstracts without available full-text manuscript

## 2.3 Information Sources and Search Strategy

### 2.3.1 Databases and Search Period

A comprehensive literature search was conducted across four databases to capture both peer-reviewed publications and grey literature:
- **SciSpace** (comprehensive academic database covering multidisciplinary sources)
- **PubMed/MEDLINE** (biomedical and life sciences literature)
- **Google Scholar** (grey literature, preprints, and institutional repositories)
- **ArXiv** (machine learning and artificial intelligence preprints)

The search period spanned January 1, 2018 to January 20, 2026, representing an 8-year window designed to capture recent advances in machine learning and artificial intelligence applications to Parkinson's disease prognosis.

### 2.3.2 Search Strategy

The search strategy employed a comprehensive combination of terms covering disease terminology, model architectures, and clinical applications. No language restrictions were applied during the initial search, though non-English papers were excluded during screening. The search terms included:

**Core disease terms:**
- "Parkinson's disease" OR "Parkinson disease" OR "PD"

**Model architecture terms:**
- "digital twin" OR "mechanistic model" OR "physics-informed neural network" OR "PINN" OR "scientific machine learning" OR "computational neuroscience" OR "virtual brain" OR "dynamic model" OR "temporal model" OR "time-series" OR "LSTM" OR "recurrent neural network" OR "RNN" OR "transformer" OR "attention mechanism"

**Clinical application terms:**
- "progression prediction" OR "prognosis" OR "forecasting" OR "trajectory" OR "DBS optimization" OR "deep brain stimulation" OR "gait prediction" OR "fall prediction" OR "drug dosing" OR "treatment response"

Search strategies were adapted to the syntax requirements of each database while maintaining semantic equivalence. The full search strings for each database are provided in Supplementary Material S1.

## 2.4 Selection Process

### 2.4.1 Screening Workflow

Records retrieved from all databases were imported into a reference management system and deduplicated using DOI and title matching algorithms. Two independent reviewers (initials blinded for peer review) screened titles and abstracts against the eligibility criteria. Studies deemed potentially eligible by either reviewer advanced to full-text review.

Full-text articles were independently assessed by both reviewers using a standardized eligibility checklist that operationalized each of the five inclusion criteria. Disagreements were resolved through consensus discussion. When consensus could not be reached, a third senior reviewer adjudicated the decision.

### 2.4.2 Documentation of Selection Process

The study selection process was documented in accordance with PRISMA 2020 guidelines. A PRISMA flow diagram was constructed to illustrate the number of records identified, screened, excluded (with reasons), and ultimately included in the review. Reasons for exclusion at the full-text stage were recorded and categorized by which inclusion criterion was not met.

## 2.5 Data Collection Process

### 2.5.1 Data Extraction Instrument

Data extraction was performed using a structured electronic form developed and pilot-tested on five randomly selected included studies. The extraction form captured the following domains:

**Study Characteristics:**
- First author, publication year, digital object identifier (DOI)
- Study design (prospective cohort, retrospective cohort, randomized controlled trial, registry analysis)
- Sample size (total enrolled, training set, validation set, test set reported separately)
- Population characteristics (disease stage, Hoehn and Yahr stage, disease duration, medication status)

**Model Characteristics:**
- Model type classification (Dynamic/Time-Series, Static Machine Learning, Mechanistic Digital Twin, Hybrid)
- Specific architecture (LSTM, Random Forest, XGBoost, Convolutional Neural Network, etc.)
- Input data modalities (clinical rating scales, magnetic resonance imaging, electroencephalography, wearable sensors, voice recordings, biomarkers)
- Temporal modeling approach (prognostic vs. diagnostic)
- Mechanistic principles incorporated (if any)

**Validation Methodology:**
- Validation approach (internal cross-validation, temporal split, external cohort validation)
- Validation tier classification:
  - **Tier 0:** Internal split only (80/20 split, k-fold cross-validation on single dataset) – LOW generalizability
  - **Tier 1:** Temporal or site split (same institution, different time periods or clinical sites) – MODERATE generalizability
  - **Tier 2:** External validation or prospective cohort (independent dataset, different institution, or prospective enrollment) – HIGH generalizability
- External cohort names (Parkinson's Progression Markers Initiative, Netherlands Parkinson's Disease Registry, UK Biobank, etc.)
- Test sample size (N)

**Comparative Performance Data:**
- Primary performance metric (area under the receiver operating characteristic curve, accuracy, mean absolute error, root mean squared error, hazard ratio, concordance index, etc.)
- Intervention model performance score
- Comparator or baseline model performance score
- 95% confidence intervals or standard deviations
- Statistical significance (p-values, confidence intervals)
- Presence of direct model comparison (Yes/No/Partial)

**Clinical Context:**
- Prediction goal (progression forecasting, fall prediction, deep brain stimulation response, cognitive decline, etc.)
- Prediction horizon (timeframe: 6 months, 1 year, 2 years, etc.)
- Disease stage of enrolled patients (early, moderate, advanced, mixed)
- Medication status during outcome assessment (ON-medication, OFF-medication, mixed, not specified)

**Quality and Limitations:**
- Key limitations for clinical translation as stated by study authors
- Risk-of-bias indicators (data leakage, heterogeneous cohort composition, unclear train-test splitting)

### 2.5.2 Data Extraction Procedure

Data extraction was performed independently by two reviewers. Discrepancies were identified through automated comparison of extracted data fields and resolved through consensus discussion with reference to the original manuscript. When critical data were missing or unclear, study authors were not contacted due to resource constraints, and the missing data were recorded as "not reported."

## 2.6 Risk of Bias Assessment

### 2.6.1 PROBAST Framework

Risk of bias in included studies was assessed using the Prediction model Risk Of Bias ASsessment Tool (PROBAST) [46], adapted for machine learning prediction models. PROBAST evaluates risk of bias across four domains:

**Domain 1 – Participants:** Were participants representative of the target population? Were inclusion and exclusion criteria appropriate? Was there potential for selection bias?

**Domain 2 – Predictors:** Were predictors measured appropriately and consistently? Were predictors available at the time predictions would be made in clinical practice? Was there risk of data leakage (future information incorporated into predictors)?

**Domain 3 – Outcome:** Was the outcome defined appropriately and consistently? Was outcome assessment blinded to predictor information? Was there potential for ascertainment bias?

**Domain 4 – Analysis:** Were appropriate modeling techniques used? Were measures taken to prevent overfitting (regularization, cross-validation, external validation)? Was model performance evaluated on an independent test set? Were performance metrics appropriate for the clinical question?

Each domain was rated as low, moderate, or high risk of bias. An overall risk-of-bias judgment was made for each study based on the highest risk rating across domains.

### 2.6.2 Validation Tier System

As a primary quality metric, studies were classified into a three-tier validation hierarchy:

- **Tier 2 (External/Prospective):** Models validated on an external cohort from a different institution or geographic region, or validated prospectively on newly enrolled patients. This tier represents LOW risk of bias due to high generalizability.

- **Tier 1 (Temporal/Site Split):** Models validated using temporal splits (training on earlier time periods, testing on later periods) or site-based splits within the same institution. This tier represents MODERATE risk of bias due to moderate generalizability.

- **Tier 0 (Internal Cross-Validation Only):** Models validated only through internal data splitting (random 80/20 split, k-fold cross-validation) on a single dataset. This tier represents HIGH risk of bias due to low generalizability and potential overfitting.

### 2.6.3 Additional Risk-of-Bias Indicators

Beyond PROBAST domains, studies were flagged for the following methodological concerns:
- **Data leakage:** Incorporation of future information (e.g., outcomes measured after the prediction timepoint) into predictor variables
- **"Frankenstein cohorts":** Inappropriate merging of heterogeneous datasets with incompatible measurement protocols or patient populations
- **Lack of baseline comparator:** Absence of comparison to simpler models or clinical standards
- **Missing variance estimates:** Failure to report confidence intervals, standard deviations, or p-values for performance metrics
- **Unclear train-test methodology:** Insufficient description of data splitting procedures, raising concerns about potential data leakage

## 2.7 Data Synthesis and Analysis

### 2.7.1 Meta-Analysis Feasibility Assessment

The feasibility of quantitative meta-analysis was assessed by evaluating:
- **Outcome metric homogeneity:** Whether studies reported comparable performance metrics (e.g., area under the curve, accuracy, mean absolute error)
- **Availability of variance estimates:** Whether studies reported 95% confidence intervals, standard deviations, or standard errors enabling calculation of pooled effect sizes
- **Clinical homogeneity:** Whether studies addressed sufficiently similar clinical questions (prediction goals, patient populations, prediction horizons) to justify pooling

Based on this assessment, quantitative meta-analysis was deemed infeasible due to substantial heterogeneity in outcome metrics (integrated area under the curve, area under the curve, symmetric mean absolute percentage error, F-measure, accuracy), absence of reported variance estimates (0 of 6 comparative studies reported 95% confidence intervals for intervention models), and clinical heterogeneity in prediction goals and timeframes.

### 2.7.2 Narrative Synthesis Approach

Given the infeasibility of quantitative meta-analysis, a structured narrative synthesis was conducted. The synthesis approach included:

**Vote-counting of comparative studies:** For studies directly comparing dynamic or mechanistic models to static baselines, the direction of effect (favoring intervention vs. favoring comparator) was tabulated. Effect sizes were calculated as raw differences and relative percentage improvements.

**Harvest plot visualization:** A graphical display was constructed to visualize study findings according to validation tier, prediction goal, and direction of effect.

**Subgroup analysis:** Studies were stratified by:
- Prediction goal (progression forecasting, fall prediction, cognitive decline, deep brain stimulation response)
- Validation tier (Tier 0, Tier 1, Tier 2)
- Model type (Dynamic data-driven vs. Mechanistic vs. Static)
- Prediction horizon (short-term ≤12 months vs. long-term >12 months)

**Effect size reporting:** For each comparative study, raw performance differences and relative percentage improvements were calculated and reported. When multiple metrics were reported, the primary metric as designated by study authors was used.

### 2.7.3 Heterogeneity Assessment

Although quantitative meta-analysis was not performed, heterogeneity was assessed qualitatively by examining:
- Diversity in outcome metrics and their measurement properties
- Variation in patient populations (disease stage, medication status, comorbidities)
- Differences in prediction horizons (6 months to 36 months)
- Variation in model architectures and input data modalities

Had meta-analysis been feasible, the I² statistic would have been calculated to quantify statistical heterogeneity, with I² > 75% indicating substantial heterogeneity potentially precluding pooling.

### 2.7.4 Reporting Standards

The systematic review adhered to the following reporting standards:
- **PRISMA 2020 checklist:** All items from the PRISMA 2020 statement were addressed and documented in Supplementary Material S2.
- **PRISMA flow diagram:** A flow diagram illustrating the study selection process was constructed and included in the Results section.
- **Summary of findings tables:** Structured tables summarizing key characteristics and findings of included studies were prepared.
- **Certainty of evidence assessment:** The overall certainty of evidence was assessed using principles adapted from the Grading of Recommendations Assessment, Development and Evaluation (GRADE) framework, considering risk of bias, inconsistency, indirectness, imprecision, and publication bias.

---
