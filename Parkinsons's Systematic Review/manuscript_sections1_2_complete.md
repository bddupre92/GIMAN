# Prognostic Utility of Digital Twins versus Static Machine Learning in Parkinson's Disease: A Systematic Review and Best-Evidence Synthesis

---

## Abstract

**Background:** Parkinson's disease (PD) exhibits marked clinical heterogeneity and non-linear progression trajectories that challenge conventional prognostic approaches. While digital twin frameworks and dynamic mechanistic models have been proposed to capture individual disease trajectories through integration of longitudinal data and physiological constraints, their empirical superiority over static machine learning baselines remains unquantified.

**Objective:** To systematically review and benchmark the prognostic performance of dynamic or mechanistic models against static machine learning approaches in predicting PD progression, treatment response, and clinical outcomes.

**Methods:** We conducted a systematic review following PRISMA 2020 guidelines with risk-of-bias assessment using PROBAST criteria. We searched SciSpace, PubMed, Google Scholar, and ArXiv from January 2018 to January 2026 for studies comparing dynamic/mechanistic models to static baselines in PD prognosis. Strict inclusion criteria required: (1) human PD patients, (2) dynamic/mechanistic or temporal forecasting models, (3) direct comparison to static baselines or clinical standards, (4) prognostic outcomes (not diagnosis), and (5) observational or clinical trial designs. Two independent reviewers screened titles, abstracts, and full texts. Data extraction followed a pre-specified protocol capturing model architecture, validation methodology, comparative metrics, and clinical context. Narrative synthesis was employed due to outcome metric heterogeneity.

**Results:** Of 287 unique papers screened, 15 (5.2%) met all inclusion criteria. Only 6 papers (2.1% of screened, 40% of included) reported direct head-to-head comparisons between dynamic and static models with quantitative metrics. Among these, 5 of 6 (83%) favored dynamic approaches, with effect sizes ranging from +4.3% to +28.9% relative improvement (iAUC: 0.812 vs. 0.743; sMAPE: 55 vs. 77.32; AUC: 0.82 vs. 0.69; F-measure: 0.73 vs. 0.70). Critically, zero studies implemented true mechanistic digital twins incorporating physics-informed constraints or differential equations; 73% employed purely data-driven methods. Quantitative meta-analysis was precluded by heterogeneous outcome metrics (iAUC, AUC, sMAPE, F-measure, Accuracy) and absence of reported variance estimates (0/6 papers reported 95% confidence intervals for intervention models). Validation quality was relatively strong, with 67% achieving external or prospective validation (Tier 2).

**Conclusions:** While limited evidence suggests dynamic temporal models may outperform static baselines for PD prognosis, the evidence base is insufficient for definitive recommendations. The field lacks comparative rigor: 87% of included studies provided no baseline comparison, and no studies evaluated true mechanistic digital twins. Standardized reporting of variance estimates, head-to-head benchmarking against well-tuned static baselines, and rigorous external validation are urgently needed to establish the clinical value of computational complexity in prognostic modeling.

**Keywords:** Parkinson's disease; digital twins; machine learning; prognosis; systematic review; PRISMA; PROBAST; disease progression; temporal modeling

---

## 1. Introduction

### 1.1 Clinical Heterogeneity and Prognostic Challenges in Parkinson's Disease

Parkinson's disease (PD) is a progressive neurodegenerative disorder characterized by profound clinical heterogeneity in symptom presentation, disease trajectory, and treatment response [1], [2]. Motor and non-motor manifestations vary substantially across individuals, with progression rates differing by as much as 10-fold even among patients with similar baseline characteristics [3], [4]. This heterogeneity reflects complex interactions between genetic susceptibility, environmental factors, and compensatory neuroplasticity mechanisms that evolve non-linearly over years to decades [5], [6].

Conventional prognostic approaches rely predominantly on cross-sectional "snapshot" assessments—single-timepoint measurements of motor severity (MDS-UPDRS), cognitive function (MoCA), or biomarker levels—to stratify patients and predict future outcomes [7], [8]. However, such static evaluations fail to capture the dynamic, time-dependent nature of neurodegeneration. Longitudinal studies demonstrate that early-phase progression patterns, including rate of motor decline and emergence of non-motor symptoms, are more predictive of long-term disability than baseline severity alone [9], [10]. Furthermore, treatment responses to levodopa and deep brain stimulation exhibit substantial inter-individual variability that cannot be adequately predicted from baseline clinical phenotypes [11], [12].

The limitations of snapshot-based prognosis have profound clinical implications. Inaccurate prognostic estimates hinder personalized treatment planning, delay enrollment of appropriate patients into neuroprotective trials, and contribute to suboptimal resource allocation in healthcare systems [13], [14]. There is thus an urgent need for prognostic models that integrate longitudinal data streams and account for the temporal dynamics of disease progression.

### 1.2 Digital Twins and Dynamic Mechanistic Models: Theoretical Promise

Digital twin technology—originally developed in aerospace and manufacturing—has emerged as a conceptual framework for personalized medicine, proposing to create virtual replicas of individual patients that evolve in parallel with real-world disease trajectories [15], [16]. In the context of PD, digital twins theoretically integrate multi-modal longitudinal data (clinical assessments, wearable sensor streams, neuroimaging, biomarkers) with mechanistic knowledge of basal ganglia circuitry, dopaminergic degeneration kinetics, and pharmacodynamic responses [17], [18].

Mechanistic modeling approaches, including physics-informed neural networks (PINNs), computational neuroscience models, and hybrid machine learning architectures, aim to embed physiological constraints and domain knowledge into predictive algorithms [19], [20]. Unlike purely data-driven methods that learn statistical associations from training data, mechanistic models incorporate differential equations governing neurotransmitter dynamics, network connectivity changes, or protein aggregation kinetics [21], [22]. Proponents argue that such approaches offer superior generalizability, interpretability, and sample efficiency—particularly in scenarios with limited training data or distribution shifts between development and deployment populations [23], [24].

Dynamic temporal models, including recurrent neural networks (RNNs), long short-term memory (LSTM) networks, and state-space models, represent an intermediate approach that captures time-series dependencies without explicit mechanistic constraints [25], [26]. These architectures leverage sequential patterns in longitudinal data to forecast future disease states, potentially outperforming static classifiers that ignore temporal ordering [27], [28].

The theoretical advantages of digital twins and mechanistic models are compelling: individualized trajectory prediction, integration of heterogeneous data modalities, incorporation of biological plausibility constraints, and potential for in-silico treatment optimization [29], [30]. However, these benefits come at the cost of increased model complexity, higher computational demands, and greater data requirements for parameter estimation [31], [32].

### 1.3 Research Gap: Lack of Empirical Benchmarking

Despite extensive theoretical discourse on digital twins and mechanistic modeling in PD [33], [34], [35], empirical evidence quantifying their prognostic superiority over simpler, static machine learning baselines remains scarce. Prior systematic reviews have focused on diagnostic accuracy (distinguishing PD from controls or atypical parkinsonism) rather than prognostic utility (predicting future disease trajectories) [36], [37]. Reviews addressing progression prediction have not systematically evaluated whether adding temporal dynamics or mechanistic complexity improves performance beyond well-tuned static models such as Random Forest, XGBoost, or logistic regression [38], [39].

This evidence gap has critical implications for clinical translation and research prioritization. If dynamic or mechanistic models offer only marginal improvements over static baselines—or worse, if their added complexity degrades generalizability—then the substantial investment required for their development and deployment may not be justified [40], [41]. Conversely, if these approaches demonstrate consistent and clinically meaningful performance gains, they warrant accelerated development and validation in prospective clinical trials [42].

Furthermore, the methodological rigor of comparative studies remains unclear. Key questions include: (1) Are dynamic models compared against appropriately tuned static baselines, or against straw-man comparators? (2) Do studies employ external validation on independent cohorts, or rely solely on internal cross-validation? (3) Are performance metrics standardized to enable cross-study synthesis? (4) Are variance estimates and statistical significance tests reported to assess the reliability of observed differences? Without systematic evaluation of these methodological dimensions, the field risks premature adoption of complex modeling paradigms based on theoretical appeal rather than empirical evidence [43], [44].

### 1.4 Objective and Research Questions

The objective of this systematic review is to rigorously evaluate the comparative prognostic performance of dynamic/mechanistic models versus static machine learning approaches in Parkinson's disease. Specifically, we address the following research questions:

1. **Comparative effectiveness:** Do dynamic temporal models or mechanistic digital twins demonstrate superior prognostic accuracy compared to static machine learning baselines when evaluated on the same test sets?

2. **Evidence quality:** What proportion of studies reporting prognostic models in PD conduct direct head-to-head comparisons, and what is the methodological quality of these comparisons (validation strategy, baseline selection, statistical testing)?

3. **Mechanistic integration:** To what extent have true mechanistic digital twins—incorporating physics-informed constraints, differential equations, or computational neuroscience models—been implemented and validated in PD prognosis?

4. **Meta-analysis feasibility:** Can effect sizes be pooled across studies to generate quantitative estimates of the performance advantage (if any) of dynamic/mechanistic approaches?

5. **Clinical translation readiness:** What barriers prevent translation of advanced modeling approaches into clinical practice, and what evidence gaps must be addressed to enable evidence-based adoption?

By systematically synthesizing the available evidence and identifying critical methodological gaps, this review aims to inform future research priorities and guide clinicians and policymakers in evaluating the readiness of digital twin and mechanistic modeling technologies for integration into PD care pathways.

---

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
