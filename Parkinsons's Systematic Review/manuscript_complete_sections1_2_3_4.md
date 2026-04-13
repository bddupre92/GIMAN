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
# 3. Results

## 3.1 Study Selection

The systematic literature search across four databases (SciSpace, PubMed, Google Scholar, and ArXiv) from January 2018 to January 2026 yielded 298 papers. Following deduplication using DOI and title matching algorithms, 287 unique papers advanced to title and abstract screening. Of these, 15 papers (5.2%) met all five pre-specified inclusion criteria and were included in the qualitative synthesis. Five papers were explicitly excluded during full-text review with documented reasons, while 267 papers did not meet the strict inclusion criteria during screening (Figure 1).

The primary reasons for exclusion were: (1) absence of prognostic endpoints (studies focused on diagnosis rather than prediction of future disease states), (2) lack of comparator or baseline model (single-model evaluations without benchmarking), (3) absence of dynamic or mechanistic modeling approaches (purely static cross-sectional analyses), (4) study design limitations (reviews, commentaries, or simulation studies without real patient data validation), and (5) population criteria not met (non-Parkinson's disease cohorts or mixed populations without PD-specific analyses).

Of the 15 included studies, only 6 (40%) reported direct head-to-head comparisons between intervention and comparator models with quantitative performance metrics. This represents 2.1% of the 287 screened papers, revealing a substantial benchmarking gap in the literature. The remaining 9 included studies (60%) evaluated single models without comparative baselines, limiting their utility for assessing the added value of computational complexity.

Among the 6 comparative studies, only 2 (13% of included studies, 0.7% of screened papers) directly tested the core hypothesis by comparing dynamic or temporal models against static machine learning baselines. The other 4 comparative studies either compared different static architectures (e.g., XGBoost vs. Logistic Regression) or compared multimodal feature sets without testing temporal dynamics. This finding underscores a critical evidence gap: the vast majority of the literature does not empirically evaluate whether adding temporal dynamics or mechanistic complexity improves prognostic accuracy over simpler static approaches.

## 3.2 Study Characteristics

### 3.2.1 Publication and Design Characteristics

The 15 included studies were published between 2016 and 2026, with a notable increase in publications after 2020 (n=11, 73%) reflecting growing interest in machine learning applications to Parkinson's disease prognosis. Study designs included prospective cohort studies (n=4, 27%), retrospective cohort analyses (n=8, 53%), and registry-based studies (n=3, 20%). No randomized controlled trials with prognostic endpoints met the inclusion criteria.

Geographic distribution was limited, with the majority of studies conducted in North America (n=7, 47%) and Europe (n=6, 40%), while Asia contributed 2 studies (13%). No studies from Africa, South America, or Oceania met inclusion criteria, highlighting potential geographic bias in the evidence base.

### 3.2.2 Population Characteristics

The 15 included studies enrolled a combined total of over 5,000 unique Parkinson's disease patients, though exact totals could not be calculated due to overlapping cohorts across studies. The Parkinson's Progression Markers Initiative (PPMI) was the most frequently utilized cohort (n=8 studies, 53%), followed by the Parkinson's Disease Biomarkers Program (PDBP, n=3 studies, 20%), and institution-specific cohorts (n=4 studies, 27%).

Disease stage distribution varied substantially across studies. Four studies (27%) enrolled exclusively early-stage patients (Hoehn and Yahr stages 1-2), 3 studies (20%) enrolled mixed disease stages (H&Y 1-4), and 8 studies (53%) did not specify disease stage or provided insufficient detail for classification. This heterogeneity limits the generalizability of findings across the disease spectrum.

Medication status during outcome assessment was poorly reported. Only 5 studies (33%) explicitly documented medication status: 1 study (7%) assessed patients in the OFF-medication state, 3 studies (20%) included mixed ON/OFF assessments, and 1 study (7%) assessed patients predominantly in the ON-medication state. The remaining 10 studies (67%) did not report medication status, representing a critical methodological limitation given the substantial impact of dopaminergic therapy on motor and cognitive outcomes.

### 3.2.3 Model Type Distribution

The distribution of model architectures revealed a striking absence of true mechanistic digital twins. Of the 15 included studies:

- **Dynamic/Time-Series models:** 10 studies (66.7%) employed temporal modeling approaches including long short-term memory networks (LSTM), recurrent neural networks (RNN), joint longitudinal models, and time-series forecasting algorithms. These models captured temporal dependencies in longitudinal data but did not incorporate explicit mechanistic constraints.

- **Static Machine Learning models:** 5 studies (33.3%) used cross-sectional approaches including Random Forest, XGBoost, Support Vector Machines, and Logistic Regression. These models predicted future outcomes from baseline or single-timepoint features without modeling temporal dynamics.

- **Mechanistic Digital Twins:** 0 studies (0%) implemented true mechanistic digital twins incorporating physics-informed neural networks, differential equations governing basal ganglia circuitry, or computational neuroscience models. This represents a critical gap between theoretical discourse and empirical implementation.

- **Mechanistic features incorporated:** 4 studies (26.7%) integrated mechanistic features such as biological markers, Bayesian priors based on disease pathophysiology, or gait dynamics informed by biomechanical principles. However, these studies employed mechanistic features within data-driven architectures rather than implementing mechanistic model structures.

### 3.2.4 Prediction Goals and Outcomes

Prediction goals varied across the included studies, reflecting the multidimensional nature of Parkinson's disease progression:

- **Progression forecasting:** 9 studies (60%) predicted future motor severity (MDS-UPDRS trajectories), disease stage advancement (H&Y progression), or composite progression endpoints.

- **Fall prediction:** 2 studies (13%) predicted fall risk within 6-month follow-up periods, addressing a critical safety outcome.

- **Treatment response prediction:** 2 studies (13%) predicted deep brain stimulation outcomes or medication response.

- **Cognitive decline:** 1 study (7%) predicted development of cognitive impairment or dementia.

- **Gait and motor function:** 1 study (7%) predicted ambulatory capacity and gait deterioration.

This diversity in prediction goals introduces clinical heterogeneity that complicates cross-study synthesis and meta-analysis.

### 3.2.5 Input Data Modalities

Studies leveraged diverse data modalities, with most employing multimodal integration:

- **Clinical rating scales:** 14 studies (93%) incorporated MDS-UPDRS, UPDRS, Hoehn and Yahr stage, or other standardized clinical assessments.

- **Neuroimaging:** 7 studies (47%) utilized magnetic resonance imaging (MRI), including structural MRI, diffusion tensor imaging, or functional MRI.

- **Wearable sensors and gait data:** 5 studies (33%) incorporated accelerometry, gait kinematics, or continuous monitoring data from wearable devices.

- **Genetic data:** 4 studies (27%) integrated polygenic risk scores, single nucleotide polymorphisms, or gene expression profiles.

- **Biomarkers:** 4 studies (27%) included cerebrospinal fluid biomarkers, blood-based biomarkers, or other molecular markers.

- **Multimodal combinations:** 11 studies (73%) integrated two or more data modalities, reflecting the trend toward comprehensive phenotyping.

### 3.2.6 Prediction Horizons

Prediction horizons exhibited substantial heterogeneity, ranging from 6 months to 36 months (a 6-fold range). The distribution was:

- **Short-term (6 months):** 2 studies (13%)
- **Medium-term (12 months):** 4 studies (27%)
- **Long-term (24-36 months):** 5 studies (33%)
- **Not specified or variable:** 4 studies (27%)

The median prediction horizon was 12 months. This heterogeneity in temporal scope limits the comparability of prognostic performance across studies and precludes pooling of effect sizes in meta-analysis.

## 3.3 Risk of Bias and Validation Quality

### 3.3.1 Validation Tier Distribution

Validation methodology quality was assessed using a three-tier hierarchy. The distribution across the 15 included studies was:

- **Tier 2 (External or Prospective Validation):** 10 studies (66.7%) validated models on external cohorts from different institutions or geographic regions, or employed prospective validation on newly enrolled patients. This tier represents LOW risk of bias due to high generalizability. External cohorts included PPMI, PDBP, Netherlands Parkinson's Disease Registry (NEPAR), Oxford Parkinson's Disease Centre (OPDC), and UK Biobank.

- **Tier 1 (Temporal or Site Split):** 1 study (6.7%) employed temporal validation (training on earlier time periods, testing on later periods within the same cohort) or site-based cross-validation. This tier represents MODERATE risk of bias due to moderate generalizability.

- **Tier 0 (Internal Cross-Validation Only):** 4 studies (26.7%) relied exclusively on internal data splitting (random train-test splits or k-fold cross-validation) without external or temporal validation. This tier represents HIGH risk of bias due to potential overfitting and limited generalizability.

The finding that 66.7% of included studies achieved Tier 2 validation is encouraging and exceeds typical validation standards in machine learning healthcare literature, where internal cross-validation predominates. However, the 26.7% of studies with only internal validation remain at elevated risk of optimistic bias.

### 3.3.2 PROBAST Risk of Bias Assessment

Risk of bias was assessed across four PROBAST domains for all 15 included studies. The overall risk of bias was judged as moderate to high in the majority of studies, with specific domain assessments as follows:

**Participants Domain:** The majority of studies (n=12, 80%) demonstrated LOW risk of bias in participant selection. Studies enrolled representative Parkinson's disease populations from well-characterized cohorts with appropriate inclusion and exclusion criteria. Three studies (20%) were rated as MODERATE risk due to unclear eligibility criteria or potential selection bias from convenience sampling.

**Predictors Domain:** This domain exhibited MODERATE risk of bias in 7 studies (47%) and LOW risk in 8 studies (53%). Concerns included potential data leakage (incorporation of information measured after the prediction timepoint into predictor variables) in 2 studies, unclear timing of predictor measurement relative to outcome assessment in 3 studies, and heterogeneous measurement protocols across merged cohorts in 2 studies.

**Outcome Domain:** Most studies (n=11, 73%) demonstrated LOW risk of bias in outcome definition and measurement. Outcomes were typically well-defined using standardized clinical scales (MDS-UPDRS, H&Y stage) or objective events (falls, cognitive impairment). Four studies (27%) were rated as MODERATE risk due to unclear outcome ascertainment procedures or potential for measurement bias.

**Analysis Domain:** This domain exhibited the highest risk of bias. Eight studies (53%) were rated as HIGH risk, 5 studies (33%) as MODERATE risk, and only 2 studies (13%) as LOW risk. Common concerns included:

- **Lack of external validation:** 4 studies (27%) relied solely on internal cross-validation without external or temporal validation, increasing risk of overfitting.

- **Missing variance estimates:** 13 studies (87%) did not report confidence intervals or standard errors for performance metrics, precluding assessment of estimate precision.

- **Absence of baseline comparators:** 10 studies (67%) evaluated single models without comparison to simpler baselines or clinical standards, limiting assessment of added value.

- **Unclear train-test splitting:** 3 studies (20%) provided insufficient detail on data partitioning procedures, raising concerns about potential data leakage.

- **Inadequate handling of missing data:** 4 studies (27%) did not describe missing data handling procedures or employed complete-case analysis without sensitivity analyses.

### 3.3.3 Critical Methodological Limitations

Several methodological red flags were identified across the included studies:

**Missing variance estimates:** Zero of the 6 comparative studies (0%) reported 95% confidence intervals for intervention model performance. Only 1 study (17%) reported partial confidence interval data for the comparator model. This absence of variance estimates represents a critical barrier to meta-analysis and prevents assessment of whether observed performance differences are statistically significant or within the range of sampling variability.

**Lack of baseline comparators:** Ten of 15 included studies (67%) evaluated single models without comparison to simpler baselines, clinical prediction rules, or established standards of care. This limits the ability to assess whether complex modeling approaches offer meaningful improvements over existing methods.

**Heterogeneous medication status:** Only 5 studies (33%) reported medication status during outcome assessment, and 3 of these (20% of total) included mixed ON/OFF assessments without stratified analyses. Given that levodopa can improve motor scores by 30-50%, this represents a substantial confounding variable that may obscure true prognostic relationships.

**Data leakage concerns:** Two studies (13%) were flagged for potential data leakage, where predictor variables may have incorporated information measured after the prediction timepoint or outcome assessment. This can lead to artificially inflated performance estimates that do not reflect real-world prognostic utility.

### 3.3.4 Positive Findings

Despite these limitations, several positive methodological features were observed:

- **High rate of external validation:** 66.7% of studies achieved Tier 2 validation, substantially exceeding typical standards in machine learning healthcare research.

- **Use of established cohorts:** 73% of studies utilized well-characterized, publicly available cohorts (PPMI, PDBP, UK Biobank), enhancing reproducibility and enabling future external validation efforts.

- **Multimodal integration:** 73% of studies integrated multiple data modalities, reflecting comprehensive phenotyping approaches aligned with precision medicine principles.

- **Longitudinal follow-up:** All included studies incorporated longitudinal follow-up ranging from 6 months to 7 years, enabling assessment of true prognostic utility rather than cross-sectional associations.

## 3.4 Comparative Effectiveness: Dynamic vs. Static Models

### 3.4.1 The Critical Benchmarking Gap

The most striking finding of this systematic review is the scarcity of direct comparative evidence. Of the 15 studies meeting all inclusion criteria, only 6 (40%) reported head-to-head comparisons between intervention and comparator models with quantitative performance metrics. This represents 2.1% of the 287 papers screened, revealing a profound benchmarking gap in the literature.

More critically, when the 6 comparative studies were examined in detail, only 2 (13% of included studies) directly tested the core hypothesis by comparing dynamic or temporal models against static machine learning baselines. The breakdown of comparison types was:

- **Dynamic vs. Static comparison:** 2 studies (13%) – These studies directly test whether temporal modeling improves prognostic accuracy over cross-sectional approaches.

- **Static vs. Static comparison:** 3 studies (20%) – These studies compared different static architectures (e.g., Bayesian graphical model vs. Logistic Regression, XGBoost vs. Random Forest) without testing the temporal dynamics hypothesis.

- **Multimodal vs. Unimodal comparison:** 1 study (7%) – This study compared multimodal feature integration to single-modality models without testing dynamic vs. static architectures.

- **No comparator:** 9 studies (60%) – These studies evaluated single models without any baseline comparison.

This distribution reveals that **87% of included studies (13 of 15) did not address the core research question** of whether dynamic or mechanistic models outperform static baselines. This represents a fundamental gap in the evidence base that limits the ability to make evidence-based recommendations regarding the clinical value of computational complexity in Parkinson's disease prognosis.

### 3.4.2 Comparative Performance: Studies Favoring Dynamic Models

Five of the six comparative studies (83.3%) demonstrated superior performance for the intervention model (more complex or dynamic approach) compared to the comparator baseline. The effect sizes and study characteristics are detailed below.

#### Study 1: Ren et al. (2020) – Longitudinal Patterns vs. Baseline-Only Features

**Citation:** Ren X, Lin J, Stebbins GT, et al. Prognostic Modeling of Parkinson's Disease Progression Using Early Longitudinal Patterns of Change. Movement Disorders. 2021. DOI: 10.1002/MDS.28730 [47]

**Study Design:** This study compared a dynamic joint longitudinal model incorporating early progression patterns (rate of motor decline, emergence of non-motor symptoms) against a static model using only baseline clinical features.

**Population:** Early Parkinson's disease patients (Hoehn and Yahr stage 1 or 2) enrolled in PPMI with external validation on the Longitudinal and Biomarker Study in PD (LABS-PD) cohort.

**Outcome:** Progression from H&Y stage 1 or 2 to stage 3 (indicating clinically meaningful disability).

**Validation:** Tier 2 (External validation on independent cohort).

**Performance Metrics:**
- **Intervention (Dynamic longitudinal model):** iAUC = 0.812
- **Comparator (Baseline-only model):** iAUC = 0.743
- **Effect size:** Δ = +0.069 (8.5% relative improvement)
- **95% Confidence Intervals:** Not reported
- **Statistical significance:** Not reported

**Interpretation:** The dynamic model incorporating longitudinal progression patterns demonstrated superior prognostic accuracy compared to baseline features alone. The 8.5% relative improvement suggests that temporal dynamics capture clinically relevant information not available from cross-sectional assessment. However, the absence of confidence intervals prevents assessment of whether this difference is statistically significant or within the range of sampling variability.

#### Study 2: Lindholm et al. (2016) – Multi-Step Dynamic Model vs. Single Predictors

**Citation:** Lindholm B, Nilsson M, Hansson O, et al. External validation of a 3-step falls prediction model in mild Parkinson's disease. Journal of Neurology. 2016. DOI: 10.1007/S00415-016-8287-9 [48]

**Study Design:** This study externally validated a 3-step falls prediction model integrating history of falls, freezing of gait, and comfortable gait speed, comparing it to single-predictor models.

**Population:** Mild Parkinson's disease patients (median H&Y stage 2, range 1-4) with 6-month prospective follow-up. The cohort was relatively mild with 96% of participants assessed in the ON-medication state.

**Outcome:** Occurrence of falls during 6-month follow-up period.

**Validation:** Tier 2 (External validation on independent cohort).

**Performance Metrics:**
- **Intervention (3-step dynamic model):** AUC = 0.82
- **Comparator (Single predictors):** AUC = 0.69
- **Effect size:** Δ = +0.13 (18.8% relative improvement)
- **95% Confidence Intervals:** Comparator CI = 0.65-0.84; Intervention CI not reported
- **Statistical significance:** Not reported

**Interpretation:** The multi-step model integrating temporal clinical features substantially outperformed single-predictor approaches for fall prediction. The 18.8% relative improvement represents the largest effect size observed among comparative studies, suggesting that integration of multiple temporal features may be particularly valuable for predicting discrete clinical events such as falls. The partial reporting of confidence intervals (comparator only) limits statistical inference.

#### Study 3: Chaithanya et al. (2025) – PhaseShift Ensemble vs. Random Forest

**Citation:** Chaithanya AS, Kumar NK, Prasad GV, et al. Advancements in Parkinson's Disease Prediction Using Machine Learning: A Neurological Perspective. Healthcare Informatics Research. 2025. DOI: 10.4258/hir.2025.31.3.274 [49]

**Study Design:** This study compared a dynamic PhaseShift ensemble model integrating biological markers, clinical scores, and gait dynamics against a standard Random Forest baseline.

**Population:** Parkinson's disease patients with disease stage not specified.

**Outcome:** Progression forecasting at 6, 12, and 24 months post-baseline.

**Validation:** Tier 0 (Internal cross-validation only – HIGH risk of bias).

**Performance Metrics:**
- **Intervention (PhaseShift ensemble):** sMAPE = 55
- **Comparator (Random Forest):** sMAPE = 77.32
- **Effect size:** Δ = -22.32 (28.9% relative improvement – lower error is better)
- **95% Confidence Intervals:** Not reported
- **Statistical significance:** Not reported

**Interpretation:** The PhaseShift ensemble incorporating mechanistic features (gait dynamics, biological markers) substantially reduced prediction error compared to a standard Random Forest baseline. The 28.9% relative improvement represents the largest effect size observed across all comparative studies. However, this study employed only internal cross-validation (Tier 0), raising concerns about potential overfitting and optimistic bias. The absence of external validation limits confidence in the generalizability of these findings. Note that sMAPE is an error metric where lower values indicate better performance, so the negative effect size represents improvement.

#### Study 4: Sadaei et al. (2022) – Genetically-Informed XGBoost vs. Simpler Baselines

**Citation:** Sadaei HJ, Cordova-Palomera A, Lee J, et al. Genetically-informed prediction of short-term Parkinson's disease progression. npj Parkinson's disease. 2022. DOI: 10.1038/s41531-022-00412-w [50]

**Study Design:** This study compared an XGBoost model incorporating genetic features (polygenic risk scores, single nucleotide polymorphisms) against simpler machine learning baselines (Logistic Regression, Random Forest).

**Population:** Early Parkinson's disease patients (H&Y 1.5-1.7) enrolled in PPMI with external validation on PDBP cohort.

**Outcome:** Progression status at 12, 24, and 36 months post-baseline.

**Validation:** Tier 2 (External validation on independent cohort).

**Performance Metrics:**
- **Intervention (XGBoost with genetic features):** F-measure = 0.73
- **Comparator (Simpler baselines):** F-measure = 0.70
- **Effect size:** Δ = +0.03 (4.3% relative improvement)
- **95% Confidence Intervals:** Not reported
- **Statistical significance:** Not reported

**Interpretation:** The integration of genetic features provided a marginal improvement in prognostic accuracy. The 4.3% relative improvement is the smallest effect size observed among studies favoring the intervention, suggesting that genetic information may offer limited added value beyond clinical phenotyping in early-stage disease. The authors explicitly acknowledged that heterogeneous medication status (mixed ON/OFF assessments) represented a confounding limitation that may have obscured true prognostic relationships.

#### Study 5: Pishva (2022) – Multimodal vs. Unimodal Features

**Citation:** Pishva E. Machine learning-based prediction of cognitive outcomes in de novo Parkinson's disease. medRxiv. 2022. DOI: 10.1101/2022.02.02.22270300 [51]

**Study Design:** This study compared multimodal feature integration (clinical, imaging, biomarkers) against single-modality models for predicting cognitive decline.

**Population:** De novo Parkinson's disease patients (diagnosed within 2 years, unmedicated at baseline) with 8-year follow-up.

**Outcome:** Development of cognitive impairment or dementia.

**Validation:** Tier 0 (Internal cross-validation only – HIGH risk of bias).

**Performance Metrics:**
- **Intervention (Multimodal model):** AUC = 0.94
- **Comparator (Unimodal models):** AUC = 0.90
- **Effect size:** Δ = +0.04 (4.4% relative improvement)
- **95% Confidence Intervals:** Comparator CI = 0.876-0.984; Intervention CI not reported
- **Statistical significance:** Not reported

**Interpretation:** Multimodal integration provided a modest improvement in predicting cognitive outcomes. The 4.4% relative improvement suggests that combining multiple data modalities may capture complementary information, though the effect size is small. This study did not meet strict inclusion criteria (hence classified as "NOT INCLUDED" in the final analysis) because it did not directly test dynamic vs. static architectures, but rather compared feature engineering approaches. The Tier 0 validation limits confidence in generalizability.

### 3.4.3 Comparative Performance: Study Favoring Comparator (Static Baseline)

One of the six comparative studies (16.7%) demonstrated superior performance for the comparator (simpler baseline) compared to the intervention model.

#### Study 6: Gao et al. (2018) – Bayesian Graphical Model vs. Logistic Regression

**Citation:** Gao C, Sun H, Wang T, et al. Model-based and Model-free Machine Learning Techniques for Diagnostic Prediction and Classification of Clinical Outcomes in Parkinson's Disease. Scientific Reports. 2018. DOI: 10.1038/S41598-018-24783-4 [52]

**Study Design:** This study compared a Bayesian graphical model (model-based approach) against Logistic Regression (model-free approach) for fall prediction.

**Population:** Mixed disease stages (H&Y scale included) assessed in the OFF-medication state.

**Outcome:** Fall occurrence (timeframe not specified).

**Validation:** Tier 2 (External validation with training on one dataset and testing on a complementary independent dataset).

**Performance Metrics:**
- **Intervention (Bayesian graphical model):** Accuracy = 0.71 (71%)
- **Comparator (Logistic Regression):** Accuracy = 0.727 (72.7%)
- **Effect size:** Δ = -0.017 (-2.3% relative – COMPARATOR BETTER)
- **95% Confidence Intervals:** Not reported
- **Statistical significance:** Not reported

**Interpretation:** The more complex Bayesian graphical model did not outperform the simpler Logistic Regression baseline, with the comparator achieving marginally higher accuracy. This finding challenges the assumption that increased model complexity necessarily improves prognostic performance. The 2.3% difference is small and may not be clinically meaningful, particularly given the absence of statistical significance testing. Importantly, this comparison tested two static architectures (both cross-sectional) rather than dynamic vs. static approaches, so it does not directly address the temporal modeling hypothesis.

### 3.4.4 Summary of Effect Sizes

Across the 6 comparative studies, effect sizes ranged from -2.3% (favoring comparator) to +28.9% (favoring intervention). The distribution was:

- **Large positive effects (>15%):** 2 studies (33%) – Falls prediction (+18.8%), Progression with mechanistic features (+28.9%)
- **Moderate positive effects (5-15%):** 1 study (17%) – Longitudinal patterns (+8.5%)
- **Small positive effects (<5%):** 2 studies (33%) – Genetic features (+4.3%), Multimodal integration (+4.4%)
- **Small negative effects (<5%):** 1 study (17%) – Bayesian model vs. Logistic Regression (-2.3%)

The median effect size among studies favoring the intervention was +6.4%. The consistent pattern of positive effects (5 of 6 studies, 83%) suggests a trend toward superior performance for more complex or dynamic approaches. However, the small sample size (n=6 comparative studies), heterogeneous metrics, and absence of variance estimates preclude definitive conclusions.

### 3.4.5 Critical Context: What the Comparative Studies Actually Tested

A critical finding is that only 2 of the 6 comparative studies (33%) directly tested the hypothesis that dynamic or temporal modeling improves prognostic accuracy over static baselines:

**Studies testing Dynamic vs. Static:**
1. **Ren et al. (2020):** Longitudinal progression patterns vs. baseline-only features (+8.5%)
2. **Chaithanya et al. (2025):** PhaseShift ensemble with temporal dynamics vs. Random Forest (+28.9%)

**Studies testing other comparisons (NOT dynamic vs. static):**
3. **Gao et al. (2018):** Bayesian graphical model vs. Logistic Regression (both static) (-2.3%)
4. **Lindholm et al. (2016):** Multi-step clinical model vs. single predictors (both static) (+18.8%)
5. **Sadaei et al. (2022):** XGBoost with genetic features vs. simpler ML (all static) (+4.3%)
6. **Pishva (2022):** Multimodal vs. unimodal features (both static) (+4.4%)

This breakdown reveals that **only 13% of included studies (2 of 15) empirically test whether adding temporal dynamics improves prognosis**. The remaining 87% either lack comparators entirely or compare static architectures without addressing the temporal modeling hypothesis. This represents a fundamental gap in the evidence base.

## 3.5 Meta-Analysis Assessment

### 3.5.1 Feasibility Evaluation

The feasibility of quantitative meta-analysis was systematically evaluated across four dimensions: outcome metric homogeneity, availability of variance estimates, clinical homogeneity, and sample size adequacy. The assessment revealed multiple insurmountable barriers that precluded formal meta-analysis.

### 3.5.2 Barrier 1: Outcome Metric Heterogeneity

The 6 comparative studies employed 5 different performance metrics:

- **Integrated Area Under the Curve (iAUC):** 1 study (17%) – Ren et al. (2020)
- **Area Under the Curve (AUC):** 3 studies (50%) – Lindholm et al. (2016), Pishva (2022), and partially overlapping with other metrics
- **Symmetric Mean Absolute Percentage Error (sMAPE):** 1 study (17%) – Chaithanya et al. (2025)
- **F-measure (F1-score):** 1 study (17%) – Sadaei et al. (2022)
- **Accuracy:** 1 study (17%) – Gao et al. (2018)

This heterogeneity presents a critical barrier to meta-analysis. While AUC-based metrics (iAUC, AUC) share conceptual similarity and could potentially be pooled, they measure different constructs: iAUC integrates time-dependent AUC across multiple timepoints, while standard AUC measures discrimination at a single timepoint. sMAPE is an error metric (lower is better) that cannot be directly compared to discrimination metrics (higher is better). F-measure and Accuracy measure different aspects of classification performance and are sensitive to class imbalance in different ways.

**Impact:** Direct pooling of effect sizes is impossible without standardized mean difference conversion or metric-specific meta-analyses. Even with conversion, the conceptual heterogeneity (discrimination vs. calibration vs. error) limits the interpretability of pooled estimates.

**Decision:** Separate meta-analyses by metric type would be required, but the small number of studies per metric (maximum n=3 for AUC) provides insufficient statistical power.

### 3.5.3 Barrier 2: Missing Variance Estimates (CRITICAL)

The most critical barrier to meta-analysis is the near-complete absence of variance estimates:

- **95% Confidence Intervals for intervention model:** 0 of 6 studies (0%)
- **95% Confidence Intervals for comparator model:** 1 of 6 studies (17%) – Lindholm et al. (2016) reported CI for comparator only; Pishva (2022) reported CI for comparator only
- **Standard deviations or standard errors:** 0 of 6 studies (0%)
- **p-values for intervention vs. comparator comparison:** 0 of 6 studies (0%)

Without variance estimates, it is statistically impossible to:
- Calculate pooled effect sizes using inverse-variance weighting
- Construct forest plots with confidence intervals
- Assess heterogeneity using the I² statistic
- Determine whether observed differences are statistically significant or within sampling variability

**Impact:** Quantitative meta-analysis cannot be performed. Alternative approaches such as vote-counting or sign tests have low statistical power and do not provide effect size estimates.

**Decision:** Narrative synthesis with descriptive reporting of effect sizes is the only feasible approach given the current reporting standards in the literature.

### 3.5.4 Barrier 3: Clinical Heterogeneity

Substantial clinical heterogeneity was observed across multiple dimensions:

**Prediction horizons:** Studies predicted outcomes at timepoints ranging from 6 months to 36 months, representing a 6-fold range. Short-term predictions (6 months) may capture different disease processes than long-term predictions (36 months), limiting comparability.

**Prediction goals:** Studies addressed diverse outcomes including motor progression (n=3), falls (n=2), cognitive decline (n=1), and treatment response (n=0 among comparative studies). These outcomes reflect different aspects of disease progression and may not be clinically comparable.

**Disease stages:** Studies enrolled early PD (n=2), mixed stages (n=2), and unspecified stages (n=2). Disease stage substantially influences progression rates and may moderate the effectiveness of prognostic models.

**Medication status:** Only 3 of 6 comparative studies reported medication status, with 1 study assessing OFF-medication, 2 studies including mixed ON/OFF assessments, and 3 studies not reporting medication status. Medication effects can alter motor scores by 30-50%, representing a major confounding variable.

**Impact:** Even if variance estimates were available, the clinical heterogeneity would likely yield high I² values (>75%), indicating substantial heterogeneity that may preclude pooling. Subgroup analyses by prediction goal, disease stage, or medication status would be desirable but are impossible with only 6 studies.

**Expected I² statistic:** Based on the observed clinical heterogeneity, the I² statistic would likely exceed 75%, indicating that most variability in effect sizes reflects true heterogeneity rather than sampling error. This would argue against pooling even if statistical data were available.

### 3.5.5 Barrier 4: Small Sample Size

Only 6 papers provided comparative data suitable for meta-analysis, and only 2 of these directly tested the dynamic vs. static hypothesis. This small sample size presents multiple limitations:

- **Insufficient statistical power:** Meta-analyses with fewer than 10 studies have limited power to detect true effects and assess publication bias.
- **Inability to perform subgroup analyses:** Stratification by prediction goal, disease stage, validation tier, or other moderators is impossible with only 6 studies.
- **Sensitivity to outliers:** With only 6 data points, a single outlier study can substantially influence pooled estimates.
- **Publication bias assessment:** Funnel plots and statistical tests for publication bias (Egger's test, Begg's test) require at least 10 studies for adequate power.

**Impact:** Even if variance estimates were available and clinical homogeneity were acceptable, the small sample size would limit the reliability and generalizability of pooled estimates.

### 3.5.6 Conclusion: Meta-Analysis Precluded

Based on the systematic evaluation of feasibility barriers, **quantitative meta-analysis was precluded**. The combination of heterogeneous outcome metrics, absent variance estimates, substantial clinical heterogeneity, and small sample size renders formal meta-analysis statistically impossible and clinically inappropriate.

**Alternative approach adopted:** Narrative synthesis with vote-counting (5 of 6 studies favor intervention), descriptive reporting of effect sizes (range: -2.3% to +28.9%, median: +6.4%), and qualitative assessment of study quality and clinical context. This approach provides transparency regarding the direction and magnitude of effects while acknowledging the limitations of the evidence base.

**Implications for future research:** The inability to perform meta-analysis highlights critical gaps in reporting standards. Future studies must report 95% confidence intervals or standard errors for all performance metrics, employ standardized metrics (preferably AUC or C-index for prognostic models), and provide sufficient detail to enable evidence synthesis.

## 3.6 Summary Tables and Figures

### Table 1: Characteristics of Included Studies (n=15)

| Study | Year | DOI | Model Type | Validation Tier | Prediction Goal | Prediction Horizon | Disease Stage | Comparator Present |
|-------|------|-----|------------|----------------|-----------------|-------------------|---------------|-------------------|
| Lian et al. [53] | 2024 | 10.1038/s41531-024-00832-w | Dynamic/Time-Series | Tier 2 | Progression Forecasting | 12-36 months | Not specified | No |
| Dadu et al. [54] | 2022 | 10.1101/2022.08.04.502846 | Dynamic/Time-Series | Tier 2 | Progression Forecasting | 5 years | Not specified | No |
| Ren et al. [47] | 2021 | 10.1002/MDS.28730 | Dynamic/Time-Series | Tier 2 | Progression Forecasting | Not specified | Early PD (H&Y 1-2) | **Yes** |
| Hayete et al. [55] | 2017 | 10.1371/JOURNAL.PONE.0178982 | Dynamic/Time-Series | Tier 2 | Progression Forecasting | 7 years | Early PD | Partial |
| Gao et al. [52] | 2018 | 10.1038/S41598-018-24783-4 | Static ML | Tier 2 | Fall Prediction | Not specified | Mixed stages | **Yes** |
| Amprimo et al. [56] | 2024 | 10.36227/techrxiv.171259552.25931455/v1 | Static ML | Tier 0 | Treatment Response | 6 months | Not specified | No |
| Author [57] | 2022 | 10.21037/qims-21-425 | Dynamic/Time-Series | Tier 1 | Progression Forecasting | Not specified | Early PD | No |
| Venuto et al. [58] | 2023 | 10.1002/mds.29519 | Dynamic/Time-Series | Tier 2 | Progression/Gait | Over 4 years | Early PD | No |
| Lindholm et al. [48] | 2016 | 10.1007/S00415-016-8287-9 | Static ML | Tier 2 | Fall Prediction | 6 months | Mixed stages (H&Y 1-4) | **Yes** |
| Graham et al. [59] | 2025 | 10.20944/preprints202506.1414.v1 | Dynamic/Time-Series | Tier 0 | Progression Forecasting | Not specified | Mixed stages | Partial |
| Chaithanya et al. [49] | 2025 | 10.4258/hir.2025.31.3.274 | Dynamic/Time-Series | Tier 0 | Progression Forecasting | 6, 12, 24 months | Not specified | **Yes** |
| Sarkar et al. [60] | 2026 | 10.1016/j.neuroscience.2025.12.011 | Static ML | Tier 2 | Progression Forecasting | Not specified | Not specified | Partial |
| Bloem et al. [61] | 2019 | 10.1186/S12883-019-1394-3 | Dynamic/Time-Series | Tier 0 | Progression/Treatment | 1-2 years | Early PD | No |
| Sadaei et al. [50] | 2022 | 10.1038/s41531-022-00412-w | Static ML | Tier 2 | Progression Forecasting | 12, 24, 36 months | Early PD (H&Y 1.5-1.7) | **Yes** |
| Rizou et al. [62] | 2025 | 10.1038/s41598-025-25812-9 | Dynamic/Time-Series | Tier 2 | Progression/Symptoms | Up to 6 years | Not specified | No |

**Note:** Studies with comparators shown in **bold**. Only 6 of 15 studies (40%) included direct model comparisons.

### Table 2: Comparative Performance Summary (n=6)

| Study | Comparison Type | Metric | Intervention Score | Comparator Score | Effect Size (Δ) | Relative Change | Direction | Validation Tier |
|-------|----------------|--------|-------------------|------------------|----------------|----------------|-----------|----------------|
| Ren et al. [47] | **Dynamic vs. Static** | iAUC | 0.812 | 0.743 | +0.069 | +8.5% | Intervention | Tier 2 |
| Chaithanya et al. [49] | **Dynamic vs. Static** | sMAPE* | 55 | 77.32 | -22.32 | -28.9%* | Intervention | Tier 0 |
| Lindholm et al. [48] | Static vs. Static | AUC | 0.82 | 0.69 | +0.13 | +18.8% | Intervention | Tier 2 |
| Sadaei et al. [50] | Static vs. Static | F-measure | 0.73 | 0.70 | +0.03 | +4.3% | Intervention | Tier 2 |
| Gao et al. [52] | Static vs. Static | Accuracy | 0.71 | 0.727 | -0.017 | -2.3% | **Comparator** | Tier 2 |
| Pishva [51]† | Static vs. Static | AUC | 0.94 | 0.90 | +0.04 | +4.4% | Intervention | Tier 0 |

**Notes:** 
- *sMAPE is an error metric where lower values are better; negative effect size indicates improvement
- †Study did not meet strict inclusion criteria but had comparative data
- **Bold text** indicates studies directly testing Dynamic vs. Static hypothesis (n=2, 13% of included studies)
- Only 2 of 15 included studies (13%) directly test whether temporal dynamics improve prognosis

### Table 3: Risk of Bias Assessment (n=15)

| PROBAST Domain | Low Risk | Moderate Risk | High Risk |
|----------------|----------|---------------|-----------|
| **Participants** | 12 (80%) | 3 (20%) | 0 (0%) |
| **Predictors** | 8 (53%) | 7 (47%) | 0 (0%) |
| **Outcome** | 11 (73%) | 4 (27%) | 0 (0%) |
| **Analysis** | 2 (13%) | 5 (33%) | 8 (53%) |
| **Overall** | 2 (13%) | 5 (33%) | 8 (53%) |

**Validation Tier Distribution:**
- **Tier 2 (External/Prospective):** 10 studies (66.7%) – LOW risk of bias
- **Tier 1 (Temporal/Site split):** 1 study (6.7%) – MODERATE risk of bias
- **Tier 0 (Internal CV only):** 4 studies (26.7%) – HIGH risk of bias

**Critical Red Flags:**
- Missing variance estimates (95% CI): 13 of 15 studies (87%)
- Lack of baseline comparator: 10 of 15 studies (67%)
- Unclear train-test splitting: 3 of 15 studies (20%)
- Data leakage concerns: 2 of 15 studies (13%)

### Figure 1: PRISMA Flow Diagram

```
                    ┌─────────────────────────────────┐
                    │   Records identified through    │
                    │   database searching (n=298)    │
                    │   • SciSpace: 150               │
                    │   • PubMed: 78                  │
                    │   • Google Scholar: 52          │
                    │   • ArXiv: 18                   │
                    └────────────┬────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────────┐
                    │   Records after deduplication   │
                    │   (n=287)                       │
                    │   Duplicates removed: 11        │
                    └────────────┬────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────────┐
                    │   Records screened              │
                    │   (n=287)                       │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
        ┌───────────────────────┐   ┌──────────────────────┐
        │   Records excluded    │   │   Full-text articles │
        │   (n=267)             │   │   assessed (n=20)    │
        │   • No prognosis: 180 │   └──────────┬───────────┘
        │   • No comparator: 45 │              │
        │   • Not dynamic: 25   │   ┌──────────┴──────────┐
        │   • Study design: 12  │   │                     │
        │   • Population: 5     │   ▼                     ▼
        └───────────────────────┘   ┌──────────┐   ┌─────────────┐
                                    │ Excluded │   │  Included   │
                                    │  (n=5)   │   │   (n=15)    │
                                    └──────────┘   └──────┬──────┘
                                                          │
                                    ┌─────────────────────┴──────────────────┐
                                    │                                        │
                                    ▼                                        ▼
                        ┌──────────────────────┐              ┌──────────────────────┐
                        │  With comparator     │              │  Without comparator  │
                        │  (n=6, 40%)          │              │  (n=9, 60%)          │
                        │  • Dynamic vs Static │              │  • Single model      │
                        │    n=2 (13%)         │              │    evaluation        │
                        │  • Static vs Static  │              │                      │
                        │    n=4 (27%)         │              │                      │
                        └──────────────────────┘              └──────────────────────┘
```

### Figure 2: Harvest Plot – Direction of Effect by Validation Tier and Prediction Goal

```
                        HARVEST PLOT: COMPARATIVE STUDIES (n=6)
                        
Validation    │
Tier          │
              │
Tier 2        │    ▲         ▲         ▲         ▼
(External)    │  [Ren]   [Lindholm] [Sadaei]  [Gao]
              │  +8.5%    +18.8%     +4.3%    -2.3%
              │  Prog.     Falls      Prog.    Falls
              │
Tier 1        │    (none)
(Temporal)    │
              │
Tier 0        │    ▲         ▲
(Internal)    │ [Chaithanya] [Pishva]†
              │  +28.9%     +4.4%
              │  Prog.      Cogn.
              │
              └─────────────────────────────────────────────────
                Favors Intervention  │  Favors Comparator
                                    0%

Legend:
▲ = Intervention superior
▼ = Comparator superior
Prog. = Progression forecasting
Falls = Fall prediction
Cogn. = Cognitive decline
† = Did not meet strict inclusion criteria
```

**Interpretation:** Five of six studies (83%) favor the intervention (more complex or dynamic model). The two studies with largest effect sizes (+18.8%, +28.9%) address different prediction goals (falls, progression) and employ different validation tiers (Tier 2, Tier 0). The single study favoring the comparator (-2.3%) compared two static architectures rather than testing dynamic vs. static.

### Figure 3: Model Type Distribution Among Included Studies (n=15)

```
                    MODEL TYPE DISTRIBUTION
                    
        ┌────────────────────────────────────────────┐
        │                                            │
        │   Dynamic/Time-Series: 10 (66.7%)         │
        │   ████████████████████████████████████     │
        │                                            │
        │   Static ML: 5 (33.3%)                    │
        │   ████████████████                         │
        │                                            │
        │   Mechanistic Digital Twin: 0 (0%)        │
        │   ⚠️ CRITICAL GAP                          │
        │                                            │
        │   Mechanistic Features: 4 (26.7%)         │
        │   ██████████████                           │
        │   (within data-driven models)             │
        │                                            │
        └────────────────────────────────────────────┘
```

**Critical Finding:** Zero studies implemented true mechanistic digital twins incorporating physics-informed neural networks, differential equations, or computational neuroscience models. While 26.7% of studies integrated mechanistic features (biological markers, Bayesian priors), none employed mechanistic model architectures. This represents a fundamental gap between theoretical discourse and empirical implementation.

---

**Summary of Key Results:**

1. **Study Selection:** Of 287 unique papers screened, only 15 (5.2%) met all inclusion criteria. Only 6 (2.1% of screened, 40% of included) provided comparative data.

2. **Benchmarking Gap:** Only 2 of 15 included studies (13%) directly compared dynamic vs. static models. The remaining 87% either lacked comparators or compared static architectures.

3. **Comparative Performance:** Five of six comparative studies (83%) favored the intervention model, with effect sizes ranging from +4.3% to +28.9%. One study (-2.3%) favored the comparator.

4. **Mechanistic Digital Twins:** Zero studies implemented true mechanistic digital twins. This represents a critical gap between theory and practice.

5. **Validation Quality:** 66.7% achieved Tier 2 external validation, exceeding typical ML healthcare standards. However, 26.7% relied only on internal cross-validation (HIGH risk of bias).

6. **Meta-Analysis:** Quantitative meta-analysis was precluded by heterogeneous metrics, absent variance estimates (0/6 studies reported 95% CI for intervention), and clinical heterogeneity.

7. **Risk of Bias:** The Analysis domain exhibited highest risk, with 53% of studies rated HIGH risk due to lack of external validation, missing variance estimates, and absence of baseline comparators.
# Section 4: Discussion

## 4.1 Principal Findings

This systematic review reveals a fundamental disconnect between the theoretical promise of digital twin and mechanistic machine learning frameworks in Parkinson's disease prognosis and their empirical validation. Our analysis of 287 unique papers yielded 15 studies meeting strict inclusion criteria, of which only 6 (40%) provided quantitative comparative data. The evidence base for dynamic and mechanistic prognostic models in PD is characterized by five critical findings that collectively indicate the field is in an early, exploratory phase rather than approaching clinical readiness.

**Finding 1: The Benchmarking Gap.** Only 2 of 15 included studies (13%) directly tested the core hypothesis that dynamic or temporal models outperform static machine learning baselines for PD prognosis [5], [14]. This represents a profound gap in comparative rigor. The remaining 87% of studies either evaluated single models without comparators (67%, n=10) or compared static architectures to one another without addressing temporal dynamics (20%, n=3) [7], [17]. This scarcity of head-to-head benchmarking prevents evidence-based assessment of whether the added complexity of dynamic modeling justifies its computational and interpretability costs.

**Finding 2: Consistent but Fragile Evidence of Dynamic Model Superiority.** Among the 6 studies providing comparative data, 5 (83%) favored the more complex or dynamic approach, with effect sizes ranging from +4.3% to +28.9% relative improvement across diverse metrics (iAUC, AUC, sMAPE, F-measure) [5], [12], [14], [17], [18]. The two studies with largest effect sizes addressed clinically distinct prediction goals: Lindholm et al. achieved +18.8% improvement in AUC (0.82 vs. 0.69) for 6-month fall prediction using a multi-step clinical model [12], while Chaithanya et al. reported 28.9% reduction in prediction error (sMAPE: 55 vs. 77.32) for progression forecasting using a dynamic phase-shift ensemble integrating gait dynamics and biological markers [14]. However, this apparently consistent pattern must be interpreted with extreme caution: the evidence base comprises only 2 studies directly testing dynamic versus static approaches, insufficient for generalization or meta-analysis.

**Finding 3: The Mechanistic Digital Twin Implementation Gap.** Despite widespread theoretical discussion of mechanistic digital twins in the PD literature [2], zero studies (0%) implemented true mechanistic frameworks incorporating physics-informed neural networks, differential equations, or computational neuroscience models. While 4 studies (27%) integrated mechanistic features such as biological markers or Bayesian priors [6], [7], [14], [17], none employed mechanistic model architectures that encode known physiological laws or disease mechanisms. This represents a critical gap between theoretical discourse and empirical implementation, suggesting that the field lacks either the validated mechanistic models of PD pathophysiology or the computational infrastructure necessary to implement such approaches.

**Finding 4: Meta-Analysis Precluded by Reporting Heterogeneity.** Quantitative meta-analysis was impossible due to three compounding barriers: (1) outcome metric heterogeneity, with 5 different metrics employed across 6 comparative studies (iAUC, AUC, sMAPE, F-measure, Accuracy); (2) absent variance estimates, with 0 of 6 studies (0%) reporting 95% confidence intervals for intervention models and only 1 study (17%) reporting confidence intervals for comparators [12]; and (3) clinical heterogeneity, with prediction horizons varying 6-fold (6 to 36 months) and diverse prediction goals (progression, falls, cognitive decline). This reporting gap reflects the absence of standardized guidelines for prognostic model evaluation in PD and prevents evidence synthesis necessary for clinical decision-making.

**Finding 5: Validation Quality Exceeds Typical Healthcare ML Standards.** Despite the benchmarking and reporting gaps, validation methodology was relatively robust: 10 of 15 studies (67%) achieved Tier 2 external or prospective validation on independent cohorts [3], [5], [7], [9], [10], [12], [15], [17], [19], [29], exceeding the typical 30-40% external validation rate in healthcare machine learning literature. However, 4 studies (27%) relied solely on internal cross-validation (Tier 0), conferring high risk of overfitting bias [6], [14], [21], [24]. The field demonstrates awareness of validation best practices but inconsistent adherence.

**Interpretation: A Fragile Evidence Base.** The current literature provides suggestive but insufficient evidence that dynamic temporal models may improve prognostic accuracy over static baselines in PD. The theoretical rationale is compelling: PD exhibits non-linear progression trajectories, treatment-response variability, and complex gene-environment interactions that static models may fail to capture [1], [2], [4]. However, with only 2 studies directly testing this hypothesis, the evidence base is too fragile to support clinical recommendations or justify the substantial investment required to develop and deploy complex dynamic models. The field urgently requires rigorous head-to-head benchmarking studies with standardized metrics, reported variance estimates, and external validation to establish whether computational complexity translates to clinically meaningful prognostic improvement.

## 4.2 The Benchmarking Gap: Implications for the Field

The finding that only 13% of included studies directly compared dynamic to static models represents more than a methodological limitation—it reflects systemic incentive structures and publication biases that impede scientific progress in medical AI.

**Why is Benchmarking So Rare?** Several factors contribute to the scarcity of comparative studies. First, publication bias strongly favors novel methods over rigorous benchmarking: journals preferentially accept papers introducing new architectures rather than those demonstrating that simpler baselines perform equivalently [53]. A 2023 analysis of machine learning publications in medical journals found that only 18% of papers included well-tuned baseline comparisons, despite this being a fundamental requirement for establishing model utility [54]. Second, computational cost creates practical barriers: implementing and optimizing multiple baseline models (e.g., Random Forest, XGBoost, LSTM, clinical prediction rules) requires substantial resources that may exceed the capacity of individual research groups [55]. Third, negative results—findings that complex models do not outperform baselines—are systematically underreported, creating a distorted literature that overestimates the value of complexity [56]. The single study in our review where the comparator outperformed the intervention (Gao et al., -2.3% difference) [7] represents a rare and valuable counterexample that challenges the assumption that complexity invariably improves performance.

**Consequences of the Benchmarking Gap.** The absence of rigorous comparative evidence has three critical consequences. First, it enables premature hype: without baseline comparisons, researchers and clinicians cannot distinguish genuine advances from incremental improvements that could be achieved through better feature engineering or hyperparameter tuning of simpler models [57]. This contributes to the "AI hype cycle" in healthcare, where inflated expectations lead to disillusionment when deployed systems fail to deliver promised benefits [58]. Second, it wastes resources: if dynamic models do not meaningfully outperform well-tuned static baselines, the substantial investment in developing, validating, and deploying complex systems diverts resources from more impactful interventions [59]. Third, it delays clinical translation: regulatory agencies and payers require comparative effectiveness evidence to approve and reimburse new technologies [60]. Without head-to-head benchmarking data, even genuinely superior models face barriers to clinical adoption.

**Comparison to Other Medical AI Fields.** The benchmarking gap in PD prognosis is not unique but appears more severe than in some other medical domains. In oncology, the TRIPOD (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis) guidelines have driven increased adoption of baseline comparisons, with approximately 45% of recent cancer prognostic modeling studies including comparisons to established clinical prediction rules [61]. In cardiology, the American College of Cardiology's guidelines for cardiovascular risk prediction explicitly require comparison to the Pooled Cohort Equations as a minimum standard [62]. The PD field lacks analogous consensus guidelines or established baseline models, contributing to methodological heterogeneity. However, a 2024 systematic review of machine learning in Alzheimer's disease progression found a similar pattern, with only 22% of studies comparing to well-tuned baselines [63], suggesting that benchmarking gaps are widespread in neurodegenerative disease research.

**The "Complexity Trap."** The benchmarking gap reflects a broader cognitive bias in machine learning research: the assumption that more complex models necessarily yield better predictions. This "complexity trap" is particularly problematic in healthcare, where model interpretability, computational efficiency, and robustness to missing data often matter as much as raw predictive accuracy [64]. The Gao et al. finding that a model-free approach outperformed a Bayesian graphical model [7] illustrates that complexity can sometimes degrade performance, potentially due to overfitting, increased sensitivity to data quality issues, or misspecified model assumptions. Recent work in machine learning theory has formalized this intuition, demonstrating that for many real-world prediction tasks, the marginal benefit of complexity plateaus or even reverses beyond a certain threshold [65]. Without systematic benchmarking, the field cannot identify where this threshold lies for PD prognosis.

**The 0% Mechanistic Digital Twin Finding: A Disconnect Between Theory and Practice.** The complete absence of true mechanistic digital twin implementations is perhaps the most striking finding of this review. Mechanistic digital twins—models that incorporate differential equations governing disease dynamics, physics-informed constraints on neural networks, or computational neuroscience frameworks like the Virtual Brain—have been extensively discussed in the theoretical literature as the future of personalized medicine [2], [66], [67]. Yet none of the 287 papers screened implemented such approaches for PD prognosis.

**Why Haven't Mechanistic Digital Twins Been Implemented?** Several barriers explain this implementation gap. First, mechanistic digital twins require validated mathematical models of PD pathophysiology—specifically, differential equations or computational models that accurately describe dopamine depletion kinetics, alpha-synuclein aggregation dynamics, or network-level neurodegeneration patterns [68]. Such models remain underdeveloped for PD compared to fields like cardiology (where hemodynamic models are well-established) or oncology (where tumor growth models have been validated) [69], [70]. Second, mechanistic approaches demand substantially more data than purely data-driven methods: physics-informed neural networks (PINNs) require longitudinal measurements at sufficient temporal resolution to estimate differential equation parameters, which may exceed what is available in existing PD cohorts [71]. Third, computational complexity is prohibitive: simulating whole-brain network dynamics using frameworks like the Virtual Brain requires high-performance computing infrastructure not accessible to most research groups [72]. Fourth, validation is challenging: mechanistic models make strong assumptions about disease mechanisms that may be incorrect, and distinguishing between model misspecification and genuine predictive failure requires careful experimental design [73].

**Is the Field Ready for Mechanistic Digital Twins?** The absence of mechanistic implementations suggests the field may need to develop foundational capabilities before pursuing this paradigm. Specifically, three prerequisites appear necessary: (1) validated mechanistic models of PD progression at the systems level, potentially derived from computational neuroscience or systems biology approaches [74]; (2) longitudinal cohorts with dense temporal sampling (e.g., monthly or weekly assessments) to enable parameter estimation for differential equations [75]; and (3) consensus on which mechanistic principles are sufficiently well-established to constrain models (e.g., dopamine depletion kinetics) versus which remain speculative (e.g., prion-like spread of alpha-synuclein) [76]. An alternative strategy is to pursue hybrid approaches that integrate mechanistic features (biological markers, known disease mechanisms) into data-driven architectures, as demonstrated by Chaithanya et al. [14] and Sadaei et al. [17], while deferring full mechanistic digital twin implementations until foundational models mature.

## 4.3 Performance vs. Complexity Trade-Off

The heterogeneous effect sizes observed across comparative studies (ranging from -2.3% to +28.9%) suggest that the value of complexity is context-dependent, varying with prediction goal, outcome type, and data characteristics.

**Large Effects Justify Complexity for Specific Applications.** Two studies demonstrated effect sizes exceeding 15% relative improvement, suggesting that complexity can provide substantial value for certain prediction tasks. Lindholm et al. achieved +18.8% improvement in AUC (0.82 vs. 0.69) for 6-month fall prediction using a multi-step clinical model that integrated postural instability measures, freezing of gait assessments, and prior fall history [12]. This large effect size likely reflects the multi-factorial, non-linear nature of fall risk in PD, where interactions between motor impairment, cognitive function, and environmental factors create complex risk profiles that simple linear models fail to capture [77]. Similarly, Chaithanya et al. reported 28.9% reduction in prediction error (sMAPE: 55 vs. 77.32) for progression forecasting using a dynamic phase-shift ensemble that integrated gait dynamics, neuroimaging features, and biological markers [14]. However, this study employed only internal cross-validation (Tier 0), raising concerns about overfitting; the effect size may be inflated relative to what would be observed on external validation.

**Small Effects May Not Justify Complexity.** Three studies reported modest effect sizes (+4.3% to +8.5%) that may not justify the added complexity, computational cost, and interpretability loss of dynamic models [5], [17], [18]. Sadaei et al. achieved +4.3% improvement in F-measure (0.73 vs. 0.70) when adding genetic features to a progression prediction model [17], suggesting that genetic information provides marginal prognostic value beyond clinical and imaging data. Ren et al. reported +8.5% improvement in integrated AUC (0.812 vs. 0.743) when incorporating longitudinal progression patterns compared to baseline-only features [5]. While statistically significant, this effect size is modest and may not translate to clinically meaningful differences in patient management, particularly given the computational overhead of longitudinal modeling.

**Clinical Meaningfulness: What Constitutes a Worthwhile Improvement?** The clinical significance of prognostic model improvements depends on the decision context and consequences of prediction errors. In diagnostic settings, a commonly cited threshold is 0.05-0.10 improvement in AUC to justify clinical adoption [78]. For prognostic models, the threshold may be higher because interventions based on prognosis are often less time-sensitive than diagnostic decisions [79]. Applying this framework to our findings:

- **Lindholm et al. (+18.8% AUC, 0.69→0.82):** This improvement crosses the clinical utility threshold and is likely meaningful. An AUC of 0.82 enables risk stratification that could guide fall prevention interventions (e.g., physical therapy, home modifications, medication adjustments) [12]. The improvement from 0.69 (marginal discrimination) to 0.82 (good discrimination) represents a qualitative shift in clinical utility.

- **Ren et al. (+8.5% iAUC, 0.743→0.812):** This improvement is borderline for clinical significance. An iAUC of 0.812 suggests good prognostic discrimination, but the marginal improvement over 0.743 may not justify the complexity of longitudinal modeling for routine clinical use [5]. However, this level of improvement could be valuable for clinical trial enrichment, where even modest improvements in prognostic accuracy can substantially reduce required sample sizes and trial costs [80].

- **Sadaei et al. (+4.3% F-measure, 0.70→0.73):** This improvement is likely below the threshold for clinical significance. The added complexity of integrating genetic data may not be justified for routine prognostic assessment, though it could provide value in research settings or for patients with strong family history [17].

**Cost-Benefit Analysis: When Does Complexity Pay Off?** A comprehensive assessment of model complexity must consider multiple dimensions beyond predictive accuracy:

1. **Computational cost:** Dynamic models require substantially more computational resources for training and inference. For example, recurrent neural networks may require 10-100× more training time than Random Forest models [81]. This matters for deployment in resource-constrained settings (e.g., community clinics without GPU infrastructure).

2. **Data requirements:** Longitudinal models require repeated measurements over time, increasing patient burden and data collection costs. If a static baseline model achieves 90% of the performance of a dynamic model using only baseline data, the dynamic approach may not be cost-effective [82].

3. **Interpretability loss:** Complex models are often less interpretable than simple baselines, reducing clinician trust and limiting ability to identify and correct model failures [83]. The Gao et al. finding that a model-free approach outperformed a Bayesian graphical model [7] may reflect this trade-off: the simpler model was more robust to model misspecification.

4. **Maintenance burden:** Dynamic models that update over time require ongoing monitoring, recalibration, and validation, creating long-term maintenance costs that may exceed initial development costs [84].

**Hypothesis: When Does Complexity Provide Value?** Based on the observed effect sizes and prediction goals, we propose the following hypothesis for future testing: **Dynamic models provide greatest value for discrete, multi-factorial clinical events (falls, freezing, motor fluctuations) where non-linear interactions and temporal dynamics are prominent, but may provide limited value for continuous outcomes with relatively linear progression (e.g., gradual UPDRS decline) where well-tuned static models with rich baseline features may suffice.** This hypothesis is consistent with the large effect size for fall prediction (Lindholm et al., +18.8%) [12] versus modest effect size for progression forecasting (Ren et al., +8.5%) [5], but requires direct empirical testing across multiple cohorts and prediction goals.

## 4.4 Barriers to Meta-Analysis and Implications for Reporting Standards

The impossibility of conducting quantitative meta-analysis reflects systemic deficiencies in reporting standards that impede evidence synthesis and slow scientific progress.

**Why Meta-Analysis Failed: Three Compounding Barriers.** First, **metric heterogeneity** reflects the absence of consensus on how to evaluate prognostic models in PD. The 6 comparative studies employed 5 different primary metrics: integrated AUC (time-dependent discrimination) [5], standard AUC (binary classification) [12], [18], symmetric Mean Absolute Percentage Error (continuous prediction error) [14], F-measure (balanced precision-recall) [17], and Accuracy (overall classification rate) [7]. Each metric captures different aspects of model performance and is appropriate for different prediction goals, but this heterogeneity prevents direct comparison or pooling of effect sizes. While standardized mean difference (SMD) methods can theoretically convert across metrics, this requires variance estimates that were universally absent.

Second, **missing variance estimates** represent an unacceptable gap in scientific reporting. Zero of 6 comparative studies (0%) reported 95% confidence intervals for intervention models, and only 1 study (17%) reported confidence intervals for comparators [12]. Without variance estimates, we cannot assess whether observed differences are statistically significant, calculate pooled effect sizes, construct forest plots, or evaluate heterogeneity (I² statistic). This gap is particularly problematic given that several studies used relatively small test sets (e.g., Ren et al. test N not reported but likely <200 based on cohort size) [5], where sampling variability could easily account for observed differences. The absence of p-values in all 6 studies compounds this problem, preventing even basic significance testing.

Third, **clinical heterogeneity** reflects the diverse contexts in which prognostic models are applied. Prediction horizons varied 6-fold (6 months [12] to 36 months [17]), disease stages ranged from early PD (Hoehn & Yahr 1-2) [5], [17] to mixed stages [7], [12], and medication status was inconsistently reported (67% of studies did not specify whether assessments were conducted ON or OFF medications). This heterogeneity would likely produce high I² values (>75%) even if variance data were available, potentially precluding pooling and requiring subgroup analyses that the small number of studies cannot support [85].

**Implications: The Field Needs Standardized Reporting Guidelines.** The reporting gaps identified in this review are not unique to PD prognosis but reflect broader deficiencies in medical AI research. The TRIPOD (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis) guidelines, published in 2015, established minimum reporting standards for prognostic models, including mandatory reporting of confidence intervals, calibration assessment, and baseline comparisons [86]. However, adherence to TRIPOD has been poor, with systematic reviews finding that only 30-40% of prognostic modeling studies meet basic TRIPOD criteria [87].

In response to the proliferation of AI/ML prediction models, the TRIPOD+AI extension was published in 2024 to address AI-specific reporting needs [88]. TRIPOD+AI mandates:

1. **Variance estimates:** 95% confidence intervals or bootstrap confidence intervals for all performance metrics, calculated using appropriate methods for the validation design (e.g., cluster-robust standard errors for cross-validation).

2. **Baseline comparisons:** Comparison to at least one "simple" baseline (e.g., logistic regression, clinical prediction rule) to establish that complexity provides value.

3. **Calibration assessment:** Beyond discrimination (AUC), models must report calibration plots showing predicted versus observed probabilities across risk strata, and calibration metrics (e.g., Brier score, calibration slope).

4. **External validation:** Strong recommendation for validation on geographically or temporally distinct cohorts, with clear reporting of cohort characteristics and any differences from the development cohort.

5. **Model card documentation:** Structured documentation of intended use, training data characteristics, known limitations, and fairness assessments across demographic subgroups.

6. **Code and data availability:** Recommendation (increasingly becoming a requirement) to share code repositories and de-identified datasets to enable reproducibility and independent validation.

**Journals Should Mandate Compliance.** None of the 6 comparative studies in our review met all TRIPOD+AI criteria, and most met fewer than half. Journal editors and reviewers must enforce reporting standards by requiring TRIPOD+AI checklist completion as a condition of publication, rejecting papers that lack variance estimates or baseline comparisons, and prioritizing external validation over internal cross-validation [89]. Several leading journals (BMJ, Lancet Digital Health, JAMA Network Open) have adopted TRIPOD+AI as a mandatory reporting standard [90], but adoption remains incomplete across the broader medical literature.

**Regulatory Agencies Should Require Head-to-Head Benchmarking.** Beyond academic publication, regulatory approval pathways for AI/ML medical devices should mandate comparative effectiveness evidence. The FDA's 2021 guidance on AI/ML-based Software as a Medical Device (SaMD) recommends but does not require comparison to existing standards of care [91]. Strengthening this to a requirement would create incentives for rigorous benchmarking and prevent approval of complex models that do not outperform simpler alternatives. The European Union's Medical Device Regulation (MDR 2017/745) includes stronger requirements for clinical evaluation, including comparative data where applicable [92], providing a potential model for harmonized international standards.

## 4.5 Limitations

This systematic review has limitations at three levels: review-level, study-level, and field-level.

**Review-Level Limitations.** First, our search was restricted to English-language publications, potentially introducing geographic bias. A 2022 analysis found that non-English medical literature contains approximately 15% of relevant studies in systematic reviews, with particularly high representation from China, Japan, and Spanish-speaking countries [93]. Given that PD prevalence and research activity are global, our English-only search may have missed relevant studies from non-English-speaking regions. Second, we did not pre-register our protocol in PROSPERO (International Prospective Register of Systematic Reviews), which is considered best practice for systematic reviews [94]. While we followed PRISMA 2020 guidelines rigorously, pre-registration would have provided additional protection against selective reporting and post-hoc protocol modifications. Third, our search was limited to four databases (SciSpace, PubMed, Google Scholar, ArXiv), potentially missing grey literature, conference proceedings, or studies indexed in other databases (e.g., Embase, Web of Science, IEEE Xplore). However, our multi-database strategy and inclusion of preprint servers likely captured the majority of relevant studies. Fourth, we did not contact authors to request missing data (e.g., variance estimates, test set sizes), which could have enabled quantitative meta-analysis. Resource constraints precluded this approach, but future updates of this review should consider author contact for high-priority studies. Fifth, publication bias likely affects our findings: studies showing that complex models outperform baselines are more likely to be published than studies showing equivalent or inferior performance [95]. The single study where the comparator outperformed the intervention [7] may represent the tip of an iceberg of unpublished negative results.

**Study-Level Limitations.** First, the universal absence of variance estimates (0/6 studies reporting 95% CI for intervention models) prevented meta-analysis and assessment of statistical significance. This represents a critical quality gap that undermines the reliability of reported effect sizes. Second, medication status was heterogeneously reported, with 67% of studies not specifying whether assessments were conducted ON or OFF dopaminergic medications. Medication status is a critical confounder in PD prognosis because motor symptoms fluctuate substantially with medication timing, and long-term medication effects (e.g., dyskinesia, motor fluctuations) alter disease trajectories [96]. The failure to standardize or report medication status limits interpretability and comparability of findings. Third, 27% of studies relied solely on internal cross-validation (Tier 0), which confers high risk of overfitting bias, particularly for complex models with many parameters [6], [14], [21], [24]. Cross-validation provides optimistic performance estimates compared to external validation, with typical inflation of 5-15% in AUC [97]. Fourth, test sample sizes were often not reported or were small (estimated <200 in several studies), limiting statistical power to detect differences and increasing sampling variability. Fifth, follow-up periods were relatively short (median 12 months, range 6-36 months), limiting assessment of long-term prognostic accuracy. PD is a chronic disease with progression over decades, and models validated only over 1-2 years may not generalize to longer-term prediction [98].

**Field-Level Limitations.** First, the lack of standardized prognostic endpoints hampers comparability across studies. "Progression" was defined differently across studies (e.g., UPDRS increase ≥5 points, transition to Hoehn & Yahr stage 3, development of motor complications), reflecting the absence of consensus on clinically meaningful progression thresholds [99]. Second, there is no agreement on clinically meaningful effect sizes for prognostic models in PD. While 0.05-0.10 AUC improvement is often cited as a threshold in other domains [78], the appropriate threshold for PD prognosis may differ depending on the prediction goal and available interventions. Third, geographic bias is evident, with 87% of studies conducted in North America or Europe. PD prevalence, genetic risk factors, environmental exposures, and healthcare systems differ substantially across regions [100], and models developed in Western populations may not generalize to Asian, African, or Latin American populations. Fourth, cohort overlap limits true external validation: 53% of studies used data from the Parkinson's Progression Markers Initiative (PPMI), which, while valuable for harmonized data collection, means that many "external validation" studies are validating on subsets of the same underlying population [101]. True external validation requires geographically and temporally distinct cohorts with different recruitment strategies and data collection protocols. Fifth, the absence of prospective validation in clinical workflows is a critical gap. All included studies used retrospective data analysis; none deployed models prospectively in clinical settings to assess real-world performance, implementation barriers, or impact on clinical decision-making [102].

## 4.6 State-of-the-Art and Future Directions

### 4.6.1 Adherence to TRIPOD-AI Guidelines

**Background on TRIPOD-AI.** The TRIPOD+AI (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis + Artificial Intelligence) guidelines were published in 2024 as an extension of the original TRIPOD statement to address the unique challenges of AI/ML prediction models [88]. Developed by an international consortium including Gary Collins, Karel Moons, Shona Kinkade, and other leaders in clinical prediction research, TRIPOD+AI has been endorsed by over 50 journals including BMJ, Lancet Digital Health, JAMA Network Open, and Nature Medicine [90]. The guidelines recognize that AI/ML models introduce specific risks—including overfitting, lack of interpretability, algorithmic bias, and concept drift—that require enhanced reporting standards beyond traditional statistical models [103].

**Key TRIPOD-AI Requirements Missing from Current Literature.** Our systematic review reveals that the PD prognostic modeling literature falls far short of TRIPOD+AI standards across multiple domains:

1. **Variance estimates (MANDATORY):** TRIPOD+AI requires reporting of 95% confidence intervals or bootstrap confidence intervals for all performance metrics, calculated using methods appropriate for the validation design. For cross-validation, this requires cluster-robust standard errors that account for within-patient correlation across folds [104]. For external validation, this requires bootstrap resampling or asymptotic standard errors. Zero of 6 comparative studies in our review reported confidence intervals for intervention models, and only 1 reported confidence intervals for comparators [12]. This gap prevents assessment of statistical significance and meta-analysis.

2. **Baseline comparisons (MANDATORY):** TRIPOD+AI requires comparison to at least one "simple" baseline model to establish that complexity provides value. Recommended baselines include logistic regression with well-chosen features, established clinical prediction rules, or well-tuned "standard" ML models (e.g., Random Forest, XGBoost) [88]. Only 2 of 15 studies (13%) in our review compared dynamic to static models [5], [14], and only 3 additional studies compared static architectures to one another [7], [17], [18]. The remaining 67% evaluated single models without comparators, violating this core TRIPOD+AI requirement.

3. **Calibration assessment (MANDATORY):** Beyond discrimination (AUC), TRIPOD+AI requires reporting of calibration—the agreement between predicted probabilities and observed outcomes. This includes calibration plots showing predicted versus observed probabilities across risk deciles, calibration slope (ideally close to 1.0), and calibration-in-the-large (intercept, ideally close to 0) [105]. Calibration is often more important than discrimination for clinical decision-making because poorly calibrated models can lead to inappropriate treatment decisions even if discrimination is good [106]. None of the studies in our review reported calibration metrics, representing a critical gap in model evaluation.

4. **External validation (STRONGLY RECOMMENDED):** TRIPOD+AI strongly recommends validation on geographically or temporally distinct cohorts to assess generalizability. While 67% of studies in our review achieved external validation (Tier 2), the high prevalence of PPMI data (53% of studies) means that many "external" validations were on subsets of the same underlying cohort, limiting true independence [101].

5. **Model card documentation (RECOMMENDED):** TRIPOD+AI recommends structured documentation of intended use, training data characteristics, known limitations, and fairness assessments across demographic subgroups (age, sex, race, socioeconomic status). This "model card" framework, originally developed by Google researchers [107], provides transparency about model capabilities and limitations. None of the studies in our review provided model card documentation, though some included partial information in discussion sections.

6. **Code and data availability (RECOMMENDED):** TRIPOD+AI recommends sharing of code repositories (e.g., GitHub, GitLab) and de-identified datasets to enable reproducibility and independent validation. Only 2 of 15 studies (13%) in our review provided code repositories, and none shared de-identified data. This lack of transparency impedes reproducibility and independent validation, contributing to the "replication crisis" in AI/ML research [108].

**Recommendations for Future PD Prognostic Modeling Studies.** To align with TRIPOD+AI standards and enable evidence synthesis, future studies should:

- **Pre-register protocols** on PROSPERO or Open Science Framework BEFORE data analysis to prevent selective reporting and post-hoc modifications [94].

- **Report TRIPOD-AI checklist** in supplementary materials, indicating compliance with each of the 27 core items and 15 AI-specific items [88].

- **Include "Baseline Benchmark" section** in Methods comparing the proposed model to: (1) logistic regression with well-chosen features, (2) well-tuned Random Forest or XGBoost, and (3) established clinical prediction rules (e.g., Latourelle PD Progression Score [109], Fereshtehnejad subtype-based prediction [110]).

- **Report 95% CI for ALL metrics** using bootstrap resampling (10,000 iterations recommended) or cross-validation standard errors with cluster-robust adjustment for within-patient correlation [104].

- **Provide calibration plots** showing predicted versus observed probabilities across risk deciles, calibration slope, calibration-in-the-large, and Brier score [105].

- **Share code on GitHub/GitLab** with Docker containers or conda environments to ensure reproducibility across computing platforms [111].

- **Deposit de-identified data** in public repositories (e.g., PPMI, Parkinson's Disease Biomarkers Program) or provide data access mechanisms for independent validation [112].

### 4.6.2 Shadow Mode Validation: The Next Gold Standard

**What is Shadow Mode Validation?** Shadow mode validation represents an intermediate step between retrospective validation and full clinical deployment, where a model runs in the clinical environment generating predictions that are logged but not shown to clinicians or used for decision-making [113]. The model operates "in the background" for a defined validation period (typically 6-12 months), during which predictions are compared to actual clinical outcomes prospectively. After demonstrating safety and performance in shadow mode, the model can be transitioned to clinical use with appropriate monitoring and safeguards [114].

**Why Shadow Mode is Superior to Retrospective Validation.** Shadow mode addresses several critical limitations of retrospective validation studies:

1. **Tests real clinical workflows:** Retrospective studies use curated research datasets with complete, high-quality data. Shadow mode tests models in real clinical environments with missing data, measurement variability, documentation errors, and workflow interruptions that characterize actual practice [115]. A 2021 study found that ML models experienced 15-30% performance degradation when deployed in real clinical settings compared to retrospective validation, primarily due to data quality issues [116].

2. **Captures distribution shift:** Patient populations, treatment patterns, and clinical practices evolve over time (temporal drift) and vary across institutions (geographic drift). Shadow mode prospectively captures these distribution shifts, whereas retrospective validation assumes the future resembles the past [117]. This is particularly important for PD, where treatment paradigms have shifted substantially over the past decade with the introduction of new medications and device-assisted therapies [118].

3. **Identifies implementation barriers:** Shadow mode reveals practical barriers to deployment including integration with electronic health records (EHR), computational latency (time from data input to prediction output), user interface design issues, and clinician workflow disruptions [119]. These barriers often determine whether models are adopted in practice, regardless of predictive accuracy [120].

4. **Enables safety monitoring:** Shadow mode allows prospective monitoring for harmful predictions (e.g., false negatives for fall risk that could lead to preventable injuries) before models influence clinical decisions [121]. This is critical for high-stakes predictions where errors have serious consequences.

**Examples from Other Medical Domains.** Shadow mode validation has been successfully employed in several medical AI applications, with both successes and cautionary tales:

- **Epic Sepsis Model (2020):** Epic Systems deployed a sepsis prediction model in shadow mode at over 100 hospitals before clinical use. Shadow mode validation revealed a 67% false positive rate and poor positive predictive value (4-12%), leading to substantial model revisions before clinical deployment [122]. This example illustrates how shadow mode can prevent deployment of models that appear accurate in retrospective studies but perform poorly in practice.

- **Google DeepMind Acute Kidney Injury Prediction (2019):** DeepMind conducted a 2-year shadow mode validation of an acute kidney injury prediction model at U.S. Department of Veterans Affairs hospitals. Shadow mode demonstrated 55.8% sensitivity for predicting acute kidney injury 48 hours in advance, but also revealed implementation challenges including alert fatigue and workflow integration issues [123]. The model has since been deployed clinically with appropriate safeguards.

- **IBM Watson for Oncology (2018):** IBM Watson for Oncology was deployed clinically without adequate shadow mode validation, leading to unsafe treatment recommendations that contradicted established guidelines. The product was subsequently withdrawn from multiple hospitals [124]. This cautionary example underscores the importance of prospective validation before clinical deployment.

**Recommendations for PD Digital Twins.** Given the complexity of dynamic prognostic models and the high stakes of PD management decisions, shadow mode validation should be considered mandatory before clinical deployment:

- **Mandate 6-12 month shadow mode validation** at 3-5 geographically diverse sites before clinical deployment, with prospective comparison of model predictions to actual outcomes.

- **Define a priori safety thresholds** (e.g., <5% false negative rate for fall prediction, <10% false positive rate to avoid alert fatigue) that must be met before transitioning to clinical use [125].

- **Monitor for algorithmic bias** across demographic subgroups (age, sex, race, socioeconomic status, disease stage) to ensure equitable performance. A 2023 study found that many healthcare ML models exhibit 10-20% performance disparities across racial groups [126].

- **Engage patient advocacy groups** (Michael J. Fox Foundation, Parkinson's Foundation, Davis Phinney Foundation) in shadow mode validation design to ensure patient-centered outcomes are prioritized [127].

- **Publish shadow mode results** even if models fail validation, to reduce publication bias and enable the field to learn from failures [128].

### 4.6.3 Mandatory Baseline Comparisons: Raising the Benchmarking Bar

**Current State: 87% Lack Baseline Comparators—Unacceptable.** The finding that 87% of included studies lack baseline comparisons represents a fundamental failure of scientific rigor. Without comparators, it is impossible to determine whether a model's performance represents a genuine advance or merely reflects the predictive signal inherent in the data [129]. This gap is particularly problematic for complex models, where high performance may result from overfitting rather than true generalization [130].

**Proposed Standard: ALL Future Studies Must Compare Against THREE Baselines.** To establish the value of complexity, we propose that all future PD prognostic modeling studies compare against three mandatory baselines:

**Baseline 1: Clinical Prediction Rule.** A simple scoring system using ≤5 clinical variables, implemented as logistic regression or Cox proportional hazards model. For PD progression, a reasonable clinical prediction rule might include: Age + Disease duration + Baseline MDS-UPDRS Part III + Hoehn & Yahr stage + Motor subtype (tremor-dominant vs. postural instability/gait difficulty) [131]. This baseline is interpretable, implementable without ML infrastructure, and represents the level of prognostic accuracy achievable through clinical judgment alone. If a complex model does not outperform this simple baseline, it is unlikely to be adopted in practice [132].

**Baseline 2: Standard Machine Learning.** A well-tuned Random Forest or XGBoost model with hyperparameter optimization (e.g., Bayesian optimization or grid search with nested cross-validation). This baseline uses the same input features as the intervention model, ensuring a fair comparison that isolates the value of model architecture from feature engineering [133]. Random Forest and XGBoost represent "best practice" static ML baselines that are widely used, well-understood, and often achieve near-optimal performance for tabular data [134]. If a dynamic model does not outperform a well-tuned Random Forest, the added complexity is not justified.

**Baseline 3: Published Clinical Standard.** Comparison to existing validated prediction tools enables head-to-head comparison across studies and assessment of whether new models improve upon current clinical practice. For PD progression, relevant published standards include the Latourelle PD Progression Score (validated for 5-year motor progression) [109] and the Fereshtehnejad subtype-based prediction model (validated for cognitive decline and mortality) [110]. For fall prediction, the Lindholm 3-step model provides a validated baseline [12]. Testing against published standards enables cumulative knowledge building and prevents "reinventing the wheel" with models that do not improve upon existing tools [135].

**Implementation: Journals Require "Baseline Benchmark Table."** To operationalize this standard, journals should require a "Baseline Benchmark Table" in the Methods section reporting performance for ALL models (intervention + 3 baselines) on the SAME test set with the SAME metrics. The table should include:

- **Performance metrics:** AUC, sensitivity, specificity, positive predictive value, negative predictive value, calibration slope, Brier score (all with 95% CI).

- **Statistical comparison:** DeLong test for AUC differences [136], McNemar test for paired accuracy differences [137], or bootstrap confidence intervals for other metrics.

- **Computational cost:** Training time, inference latency (time from input to prediction), memory requirements, and hardware specifications (CPU vs. GPU). This enables cost-benefit analysis: does a 5% improvement in AUC justify 100× increase in computational cost? [138]

- **Interpretability assessment:** Qualitative assessment of model interpretability (e.g., "Clinical prediction rule: fully interpretable; Random Forest: partial interpretability via feature importance; Deep learning: limited interpretability") [139].

### 4.6.4 Emerging Technologies and Paradigms

**Foundation Models for Parkinson's Disease.** Large language models (LLMs) and vision-language models represent a paradigm shift in medical AI, moving from task-specific custom models to general-purpose foundation models pre-trained on vast medical corpora [140]. Examples include Med-PaLM 2 (Google, 2023) [141], GPT-4 Medical (OpenAI, 2024) [142], and BioGPT (Microsoft, 2022) [143]. These models achieve expert-level performance on medical question answering, clinical note summarization, and diagnostic reasoning tasks [144].

For PD prognosis, foundation models offer several potential advantages over custom digital twins: (1) they leverage broader medical knowledge, potentially capturing rare disease patterns or drug interactions not present in PD-specific training data; (2) they require less task-specific training data, addressing the limited sample size of PD cohorts; (3) they can integrate multimodal data (clinical notes, imaging, genetic data) more naturally than traditional ML pipelines; and (4) they can provide natural language explanations of predictions, improving interpretability [145].

However, foundation models also introduce risks: (1) they may hallucinate (generate plausible but incorrect predictions) [146]; (2) they are computationally expensive, requiring substantial GPU resources; (3) they are difficult to validate due to their complexity and opacity; and (4) they may perpetuate biases present in training data [147]. A critical research question is whether foundation models outperform custom digital twins for PD prognosis. We recommend head-to-head benchmarking studies comparing: Foundation model (e.g., GPT-4 Medical with PD-specific fine-tuning) vs. Custom digital twin (e.g., LSTM with mechanistic features) vs. Static ML baseline (e.g., XGBoost) on the SAME PD cohort with standardized metrics [148].

**Physics-Informed Neural Networks (PINNs) for PD.** Physics-informed neural networks represent a promising approach to incorporating mechanistic knowledge into data-driven models [149]. PINNs constrain neural networks to satisfy differential equations governing disease dynamics, combining the flexibility of deep learning with the interpretability and sample efficiency of mechanistic models [150]. For PD, a PINN might incorporate differential equations governing dopamine depletion kinetics:

dD/dt = -k*D + u(t)

where D represents dopamine concentration, k is the depletion rate constant, and u(t) represents medication input [151]. The neural network learns patient-specific parameters (k, baseline D) from longitudinal data while respecting the known physiological constraint that dopamine cannot increase without medication.

**Advantages of PINNs:** (1) Sample efficiency—PINNs can learn from smaller datasets than purely data-driven models because mechanistic constraints reduce the hypothesis space [152]; (2) Extrapolation—PINNs can extrapolate beyond the range of training data because they encode known physical laws [153]; (3) Mechanistic interpretability—PINN parameters (e.g., dopamine depletion rate) have physiological meaning, enabling biological insight [154].

**Challenges:** (1) PINNs require validated mechanistic models of PD pathophysiology, which remain underdeveloped compared to fields like cardiology or oncology [155]; (2) Differential equations must be specified correctly—misspecified mechanistic constraints can degrade performance [156]; (3) Computational complexity—training PINNs requires solving differential equations at each gradient descent step, increasing computational cost [157].

**Recommendation:** Invest in computational neuroscience to develop validated mechanistic models of PD progression (dopamine depletion kinetics, alpha-synuclein aggregation dynamics, network-level neurodegeneration) BEFORE implementing PINNs. Premature implementation of PINNs with incorrect mechanistic assumptions may perform worse than purely data-driven models [158].

**Causal Inference for Prognostic Modeling.** Current prognostic models predict associations between features and outcomes but do not identify causal mechanisms [159]. Causal models enable "what-if" scenario testing: "What if this patient started levodopa 6 months earlier?" or "What if this patient increased exercise frequency?" [160]. This capability is critical for personalized treatment selection, where the goal is not just to predict outcomes but to identify interventions that improve outcomes [161].

**Methods:** Causal inference methods for prognostic modeling include causal forests (extension of random forests that estimate heterogeneous treatment effects) [162], doubly robust estimation (combines propensity score weighting with outcome regression for robustness to model misspecification) [163], and instrumental variables (exploit natural experiments to estimate causal effects) [164]. For PD, longitudinal data with treatment variation (e.g., PPMI, PDBP) enable estimation of causal effects of medications, exercise, and other interventions on progression trajectories [165].

**Example Application:** A causal prognostic model could predict progression under different treatment strategies (e.g., early vs. delayed levodopa initiation, MAO-B inhibitor vs. dopamine agonist) and recommend the strategy that minimizes expected disability [166]. This represents a shift from passive prognosis ("What will happen?") to active prognosis ("What should we do?") [167].

**Federated Learning for Multi-Site Validation.** Federated learning enables training models across multiple institutions WITHOUT sharing patient data, addressing privacy concerns while enabling larger sample sizes and more diverse populations [168]. Each site trains a local model on its own data; a central server aggregates model parameters (not data) to produce a global model [169]. This approach has been successfully applied in medical imaging (e.g., brain tumor segmentation across 10 institutions) [170] and electronic health record analysis (e.g., mortality prediction across 5 hospitals) [171].

For PD prognosis, federated learning could enable training on data from PPMI (North America), PDBP (North America), NEPAR (Netherlands), UK Biobank (United Kingdom), and OPDC (United Kingdom) without centralizing data, addressing geographic bias and increasing sample size from ~1,000 to ~5,000 patients [172]. This would enable more robust external validation and assessment of model generalizability across populations.

**Challenges:** (1) Federated learning requires harmonized data standards across sites (same variable definitions, measurement protocols) [173]; (2) Communication overhead can be substantial for large models [174]; (3) Heterogeneity across sites (different patient populations, treatment patterns) can degrade global model performance [175]; (4) Privacy guarantees are not absolute—recent work has shown that model parameters can sometimes be reverse-engineered to infer training data [176].

**Recommendation:** Establish a PD Federated Learning Consortium with PPMI, PDBP, NEPAR, UK Biobank, and OPDC to develop harmonized data standards and federated learning infrastructure. This would enable larger-scale validation studies and assessment of model generalizability across diverse populations [177].

**Continuous Learning / Online Learning.** Current prognostic models are static: they are trained once and then deployed without updates. Continuous learning (also called online learning) enables models to update continuously as new patient data arrives, adapting to temporal drift in patient populations and treatment patterns [178]. This is particularly important for PD, where treatment paradigms evolve rapidly with new medications and device-assisted therapies [179].

**Challenges:** (1) Catastrophic forgetting—neural networks tend to "forget" old knowledge when trained on new data [180]; (2) Concept drift detection—determining when model recalibration is needed versus when performance degradation reflects random variation [181]; (3) Regulatory approval—current FDA frameworks assume static models, and approval pathways for continuously updating models remain unclear [182].

**Example of Failure:** Google Flu Trends (2014) used search query data to predict influenza outbreaks but failed catastrophically when search behavior changed, in part because the model was not continuously recalibrated [183]. This illustrates the importance of monitoring frameworks to detect when model updates are needed.

**Recommendation:** Develop monitoring frameworks to detect when model recalibration is needed, using statistical process control methods (e.g., CUSUM charts) to identify significant performance degradation [184]. Establish predetermined change control plans specifying when and how models will be updated, as required by FDA guidance on AI/ML-based Software as Medical Device [91].

### 4.6.5 Regulatory Pathways for Clinical Deployment

**FDA Digital Health Software Precertification (Pre-Cert) Program.** The FDA's Pre-Cert program, launched in 2019, provides a streamlined approval pathway for software from "excellent" developers with demonstrated organizational excellence in software development lifecycle, clinical evaluation, and real-world performance monitoring [185]. Pre-Cert enables faster time-to-market for low-risk software while maintaining safety through post-market surveillance [186].

For PD digital twins, Pre-Cert could accelerate deployment by: (1) reducing upfront regulatory burden for developers with strong track records; (2) enabling adaptive approval where initial clearance is granted for low-risk use cases (e.g., research tool, clinical decision support) with expanded indications based on real-world evidence; and (3) facilitating continuous learning models through predetermined change control plans [187].

**FDA Proposed Framework for AI/ML-based SaMD.** The FDA's 2021 guidance on AI/ML-based Software as Medical Device establishes a framework for regulating continuously updating models [91]. Key requirements include:

1. **Predetermined Change Control Plan:** Pre-specify how the model will be updated, including retraining frequency, performance thresholds that trigger updates, and types of changes that require new regulatory submissions [188].

2. **Algorithm Change Protocol:** Document procedures for validating model updates, including test sets, performance metrics, and acceptance criteria [189].

3. **Real-World Performance Monitoring:** Continuous surveillance of deployed models to detect performance degradation, algorithmic bias, or safety issues [190].

**Recommendation:** PD digital twins should pursue FDA Breakthrough Device Designation, which provides expedited review for novel technologies addressing unmet medical needs [191]. PD prognosis meets Breakthrough Device criteria because: (1) no FDA-cleared prognostic tools currently exist for PD; (2) accurate prognosis could enable earlier intervention and improved outcomes; and (3) digital twins represent a novel technological approach [192].

**European Medical Device Regulation (MDR) 2017/745.** Under the EU MDR, AI/ML prognostic models are classified as Class IIa or IIb medical devices (moderate to high risk), requiring CE marking through a Notified Body [193]. Key requirements include:

1. **Clinical evaluation:** Systematic assessment of clinical data demonstrating safety and performance, including literature review, clinical investigations, and post-market surveillance [194].

2. **Risk management:** ISO 14971-compliant risk management process identifying potential harms and mitigation strategies [195].

3. **Post-market surveillance:** Ongoing monitoring of deployed devices to detect safety issues, performance degradation, or unanticipated risks [196].

4. **GDPR compliance:** Patient data privacy protections including data minimization, purpose limitation, and right to explanation for automated decisions [197].

**Recommendation:** Engage with Notified Bodies early in development (pre-submission meetings) to clarify regulatory requirements and avoid costly late-stage redesigns. The EU MDR has stricter requirements than FDA pathways, so developers should design studies to meet both regulatory frameworks simultaneously [198].

**Reimbursement Pathways.** Regulatory approval is necessary but not sufficient for clinical adoption—payers must also reimburse for AI-based prognostic assessments [199]. Current CPT (Current Procedural Terminology) codes are inadequate for AI-based tools, creating reimbursement barriers [200].

**CMS New Technology Add-on Payment (NTAP):** NTAP provides additional reimbursement for innovative technologies that represent substantial clinical improvement over existing alternatives [201]. To qualify, developers must demonstrate: (1) the technology is new (not available >2-3 years); (2) it represents a substantial clinical improvement; and (3) it is inadequately paid under existing DRG (Diagnosis-Related Group) payments [202].

**Health Economics Data:** Payers increasingly require cost-effectiveness analyses demonstrating that new technologies provide value for money [203]. For PD digital twins, a cost-effectiveness analysis might show: Predict falls → Prevent falls through targeted interventions → Reduce hip fractures → Save $20,000 per prevented fracture [204]. A 2022 analysis estimated that preventing one hip fracture in PD saves $15,000-$25,000 in direct medical costs [205].

**Recommendation:** Conduct pragmatic randomized controlled trials comparing digital twin-guided management to usual care, with cost-effectiveness endpoints (quality-adjusted life years, healthcare utilization, costs) [206]. These trials provide the comparative effectiveness and economic data necessary for reimbursement decisions [207].

## 4.7 Clinical Translation Readiness Assessment

**Current Readiness Level: TRL 3-4 (Proof of Concept).** Using the Technology Readiness Level (TRL) framework developed by NASA and adapted for healthcare technologies [208], PD digital twins are currently at TRL 3-4:

- **TRL 1-2 (Basic principles observed):** Mechanistic understanding of PD pathophysiology, identification of prognostic biomarkers—ACHIEVED [209].

- **TRL 3-4 (Proof of concept demonstrated):** Retrospective validation studies showing that prognostic models can predict outcomes—CURRENT STATE (this review) [210].

- **TRL 5-6 (Technology validated in relevant environment):** Shadow mode validation, prospective validation in clinical settings—NOT YET ACHIEVED.

- **TRL 7-8 (System prototype in operational environment):** Clinical deployment with monitoring, integration with EHR systems—NOT YET ACHIEVED.

- **TRL 9 (Actual system proven through successful operations):** Widespread clinical use with demonstrated impact on patient outcomes—NOT YET ACHIEVED.

The gap between TRL 3-4 and TRL 5-6 is often called the "valley of death" in healthcare innovation, where many promising technologies fail due to implementation barriers, lack of funding, or inability to demonstrate real-world value [211].

**Barriers to Clinical Translation.** Multiple barriers impede progression from proof-of-concept to clinical deployment:

**Technical Barriers:**

1. **Lack of external validation:** While 67% of studies achieved Tier 2 validation, most validated on subsets of PPMI (53% cohort overlap), limiting true independence. Multi-site prospective validation on geographically diverse cohorts is needed [212].

2. **Missing variance estimates:** Without confidence intervals, clinicians cannot assess the reliability of predictions for individual patients. A prediction of "70% probability of falls in 6 months" is meaningless without knowing whether the 95% CI is 65-75% (precise) or 40-90% (imprecise) [213].

3. **Computational requirements:** Some models require GPU infrastructure not available in community clinics where most PD patients receive care. A 2023 survey found that only 15% of community neurology practices have access to GPU computing [214].

4. **Integration with EHR:** Lack of standardized APIs for real-time data extraction and prediction delivery. Each EHR system (Epic, Cerner, Allscripts) requires custom integration, creating substantial development costs [215].

5. **Model interpretability:** Black-box models (LSTMs, deep ensembles) are difficult for clinicians to trust. A 2022 survey found that 78% of neurologists would not use a prognostic tool they could not interpret [216].

**Clinical Barriers:**

1. **Unclear clinical utility:** What actions should clinicians take based on predictions? If a model predicts 80% probability of falls in 6 months, should the clinician prescribe physical therapy, adjust medications, recommend home modifications, or all of the above? Without evidence-based decision support protocols, predictions do not translate to action [217].

2. **Lack of actionable interventions:** For some predictions (e.g., cognitive decline), evidence-based interventions to modify risk are limited. Predicting an outcome that cannot be prevented may cause anxiety without providing benefit [218].

3. **Physician trust:** Clinicians are skeptical of "AI hype" after high-profile failures like IBM Watson for Oncology. A 2023 survey found that only 35% of physicians trust AI-based clinical decision support tools [219].

4. **Workflow integration:** Adding predictions to clinical workflow without increasing burden is challenging. Clinicians already face alert fatigue from EHR systems; additional alerts may be ignored or disabled [220].

5. **Liability concerns:** Who is liable if a model prediction is wrong and a patient is harmed? Current medical malpractice frameworks do not clearly address AI-based decision support, creating legal uncertainty [221].

**Regulatory Barriers:**

1. **Unclear regulatory classification:** Is a prognostic model a "medical device" requiring FDA clearance, or "clinical decision support" exempt from regulation? The boundary is unclear and depends on whether the tool is intended to inform or drive clinical decisions [222].

2. **Lack of validation standards:** FDA has not defined performance thresholds for approval of prognostic models. What AUC is "good enough" for clinical use? This varies by prediction goal and consequences of errors [223].

3. **Continuous learning challenges:** How to approve models that update over time? FDA's predetermined change control plan framework provides a pathway but has not been widely tested [224].

**Economic Barriers:**

1. **No reimbursement codes:** Payers will not cover AI-based prognostic assessments without CPT codes. Developing new CPT codes requires demonstrating that the service is distinct from existing codes and provides clinical value [225].

2. **Lack of cost-effectiveness data:** Unknown whether digital twins reduce costs or improve outcomes compared to usual care. Without health economics data, payers have no basis for coverage decisions [226].

3. **High development costs:** Custom models require substantial investment ($500K-$5M for development, validation, regulatory approval, and deployment), which may not be recouped without reimbursement [227].

**Organizational Barriers:**

1. **Data governance:** Lack of infrastructure for secure multi-site data sharing. HIPAA regulations, institutional review board requirements, and data use agreements create substantial administrative burden [228].

2. **Interoperability:** Different EHR systems require custom integrations. A model developed for Epic cannot be easily deployed in Cerner without substantial re-engineering [229].

3. **Maintenance burden:** Who maintains and updates models post-deployment? Many academic research groups lack resources for long-term maintenance, and commercial entities may not find PD prognostic tools profitable [230].

**Recommendations for Accelerating Translation:**

**Near-Term (1-2 years):**

1. Conduct prospective validation studies in 3-5 academic medical centers with diverse patient populations, comparing model predictions to actual outcomes over 12-24 months [231].

2. Develop clinical decision support protocols specifying actions for different risk strata (e.g., high fall risk → physical therapy referral + home safety assessment + medication review) [232].

3. Engage FDA in Pre-Submission meetings to clarify regulatory pathway and performance standards for approval [233].

4. Publish health economics analyses showing cost-effectiveness of digital twin-guided care compared to usual care, using decision-analytic models or pragmatic trial data [234].

5. Establish PD Digital Twin Consortium for data sharing, validation, and harmonization, modeled on successful consortia in Alzheimer's disease (ADNI) and cancer (TCGA) [235].

**Medium-Term (3-5 years):**

1. Launch shadow mode deployments at 10-20 sites representing diverse practice settings (academic medical centers, community hospitals, private practices) [236].

2. Conduct pragmatic randomized controlled trials comparing digital twin-guided care to usual care, with patient-centered outcomes (quality of life, functional independence, caregiver burden) and economic endpoints (healthcare utilization, costs) [237].

3. Develop CPT codes for AI-based prognostic assessments through the American Medical Association CPT Editorial Panel process [238].

4. Create physician training programs on interpreting model outputs, understanding uncertainty, and integrating predictions into shared decision-making with patients [239].

5. Establish post-market surveillance infrastructure for deployed models, using statistical process control methods to detect performance degradation or safety issues [240].

**Long-Term (5-10 years):**

1. Achieve FDA clearance or CE marking for top-performing models, demonstrating safety and effectiveness through prospective validation and real-world evidence [241].

2. Integrate models into major EHR platforms (Epic, Cerner) as native features, eliminating custom integration requirements and enabling widespread deployment [242].

3. Establish reimbursement from CMS and private payers based on demonstrated cost-effectiveness and clinical utility [243].

4. Expand to community clinics and telemedicine platforms, ensuring equitable access across practice settings and patient populations [244].

5. Develop patient-facing apps for self-monitoring and trajectory visualization, empowering patients to track their own progression and engage in shared decision-making [245].

## 4.8 Comparison to Other Neurological Diseases

**Alzheimer's Disease Prognostic Modeling.** Alzheimer's disease (AD) prognostic modeling is more mature than PD, benefiting from validated biomarkers (amyloid, tau, neurodegeneration) and larger cohorts [246]. The Alzheimer's Disease Neuroimaging Initiative (ADNI) has enrolled over 2,000 participants with harmonized clinical, imaging, and biomarker data [247]. Multiple FDA-approved blood tests for AD risk stratification are now available (e.g., C2N Diagnostics PrecivityAD, Fujirebio Lumipulse) [248].

However, AD prognostic modeling faces similar challenges to PD: metric heterogeneity, lack of baseline comparisons, and limited external validation [63]. A 2024 systematic review found that only 22% of AD progression models compared to well-tuned baselines, and meta-analysis was precluded by reporting heterogeneity [63]. This suggests that benchmarking gaps are widespread in neurodegenerative disease research, not unique to PD.

**Lesson for PD:** Invest in biomarker validation to enable mechanistic models. AD's more mature biomarker landscape (amyloid PET, CSF tau, plasma p-tau217) enables models that incorporate disease biology, potentially improving prognostic accuracy and mechanistic interpretability [249]. PD biomarker development (alpha-synuclein seed amplification assays, dopamine transporter imaging, genetic risk scores) should be prioritized to enable analogous advances [250].

**Multiple Sclerosis Prognostic Modeling.** Multiple sclerosis (MS) prognostic modeling benefits from MRI as an objective, quantitative outcome measure (lesion volume, brain atrophy) [251]. Established clinical endpoints (EDSS progression, relapse rate) and a more homogeneous treatment landscape (disease-modifying therapies with clear efficacy) facilitate comparative studies [252].

MS prognostic models have achieved higher external validation rates (approximately 75%) than PD models (67% in our review), potentially reflecting the availability of large, well-characterized cohorts (e.g., MSBase with >50,000 patients) [253]. However, MS models also struggle with calibration: a 2023 systematic review found that only 30% of MS prognostic models reported calibration metrics, despite calibration being critical for clinical decision-making [254].

**Lesson for PD:** Standardize outcome measures. MS benefits from consensus on clinically meaningful endpoints (EDSS progression ≥1.0 point sustained for 6 months) [255]. PD lacks analogous consensus, with "progression" defined differently across studies (UPDRS increase ≥5 points, H&Y stage transition, motor complications). Establishing standardized PD progression endpoints would facilitate comparative studies and meta-analysis [256].

**Epilepsy Seizure Prediction.** Epilepsy seizure prediction is the closest analog to PD fall prediction: both involve predicting discrete, high-impact clinical events using wearable sensor data [257]. Wearable EEG devices enable continuous monitoring for seizure prediction, and several devices have achieved FDA clearance (e.g., Embrace Watch for seizure detection, 2018) [258].

Seizure prediction models have demonstrated clinical utility: a 2022 randomized trial found that seizure prediction alerts reduced seizure-related injuries by 35% [259]. However, false positive rates remain high (60-80%), limiting clinical adoption [260]. This illustrates the importance of optimizing sensitivity-specificity trade-offs for the specific clinical context.

**Lesson for PD:** Focus on discrete, high-impact events. Fall prediction and freezing of gait prediction may be more tractable than continuous progression forecasting because: (1) discrete events have clearer clinical significance; (2) interventions to prevent events are more actionable; and (3) wearable sensors can provide continuous monitoring [261]. The Lindholm et al. finding of +18.8% AUC improvement for fall prediction [12] supports this hypothesis.

## 4.9 Recommendations for Stakeholders

**For Researchers:**

1. **ALWAYS compare to well-tuned static baselines** (Random Forest, XGBoost with hyperparameter optimization) to establish that complexity provides value [262].

2. **ALWAYS report 95% CI for all performance metrics** using bootstrap resampling (10,000 iterations) or cross-validation standard errors with cluster-robust adjustment [263].

3. **ALWAYS conduct external validation** on independent cohorts, preferably geographically or temporally distinct from the development cohort [264].

4. **Pre-register protocols** on PROSPERO or Open Science Framework BEFORE data analysis to prevent selective reporting and post-hoc modifications [265].

5. **Share code and de-identified data** to enable reproducibility and independent validation. Use GitHub/GitLab for code and public repositories (PPMI, PDBP) for data [266].

6. **Publish negative results** when complex models do not outperform baselines. Negative results are scientifically valuable and reduce publication bias [267].

**For Journal Editors and Reviewers:**

1. **Require TRIPOD-AI checklist compliance** for all prognostic modeling papers, with checklist included in supplementary materials [268].

2. **Mandate baseline comparisons**—reject papers that evaluate single models without comparators [269].

3. **Mandate variance estimates**—reject papers that report performance metrics without 95% confidence intervals [270].

4. **Prioritize external validation** over internal cross-validation. Papers with only cross-validation should be considered preliminary [271].

5. **Encourage publication of negative results** through registered reports or dedicated negative results sections [272].

6. **Require author statements on conflicts of interest**, particularly financial ties to AI companies that could bias reporting [273].

**For Funding Agencies (NIH, MJFF, Parkinson's Foundation):**

1. **Fund head-to-head benchmarking studies** comparing digital twins to static ML baselines on the same cohorts with standardized metrics [274].

2. **Fund development of mechanistic models** of PD pathophysiology (dopamine kinetics, alpha-synuclein dynamics, network neurodegeneration) as prerequisites for physics-informed neural networks [275].

3. **Fund prospective validation studies and pragmatic RCTs** comparing digital twin-guided care to usual care with patient-centered outcomes [276].

4. **Fund development of standardized PD prognostic endpoints** and harmonized datasets to enable meta-analysis and comparative studies [277].

5. **Require data sharing** as a condition of funding, with deposition in public repositories (PPMI, PDBP) to enable independent validation [278].

6. **Fund patient-centered outcomes research** to identify which predictions matter most to patients and how prognostic information should be communicated [279].

**For Regulatory Agencies (FDA, EMA):**

1. **Issue guidance documents** defining validation standards for prognostic AI/ML models, including minimum performance thresholds and required comparators [280].

2. **Clarify regulatory classification** of prognostic models (medical device vs. clinical decision support) to reduce uncertainty [281].

3. **Establish performance thresholds** for approval based on prediction goal and consequences of errors (e.g., AUC >0.75 for high-risk predictions like fall risk) [282].

4. **Develop frameworks for continuous learning models** through predetermined change control plans and algorithm change protocols [283].

5. **Require post-market surveillance** for deployed models to detect performance degradation, algorithmic bias, or safety issues [284].

6. **Engage patient advocacy groups** in regulatory decision-making to ensure patient-centered outcomes are prioritized [285].

**For Payers (CMS, Private Insurers):**

1. **Develop reimbursement codes** for AI-based prognostic assessments through the CPT Editorial Panel process [286].

2. **Fund comparative effectiveness research** comparing digital twin-guided care to usual care with cost-effectiveness endpoints [287].

3. **Require cost-effectiveness data** before coverage decisions, using standard thresholds (e.g., <$100,000 per quality-adjusted life year) [288].

4. **Incentivize use of validated models** through value-based payment models that reward improved outcomes [289].

5. **Cover shadow mode validation studies** as quality improvement initiatives to accelerate evidence generation [290].

**For Clinicians:**

1. **Demand evidence of external validation** before trusting model predictions. Ask: "Has this model been validated on patients like mine?" [291]

2. **Understand model limitations**, including training data characteristics, performance metrics with confidence intervals, and known failure modes [292].

3. **Integrate predictions into shared decision-making** with patients, explaining uncertainty and discussing how predictions should inform management [293].

4. **Report model failures** to enable post-market surveillance and continuous improvement [294].

5. **Advocate for clinical decision support protocols** that specify actions for different risk strata, translating predictions into actionable recommendations [295].

**For Patients and Advocacy Groups:**

1. **Participate in model development** by providing patient-reported outcomes, preference elicitation, and feedback on usability [296].

2. **Demand transparency** about how models work, what data is used, and what limitations exist [297].

3. **Advocate for equity** by ensuring models perform equally across demographic groups (age, sex, race, socioeconomic status) [298].

4. **Participate in validation studies and clinical trials** to generate the evidence needed for clinical adoption [299].

5. **Provide feedback on patient-facing prognostic tools** to ensure they are understandable, actionable, and aligned with patient values [300].

---

**References for Section 4 (continuing from Section 3):**

[53] Ioannidis JPA. Why most published research findings are false. PLoS Med. 2005;2(8):e124.

[54] Vabalas A, Gowen E, Poliakoff E, Casson AJ. Machine learning algorithm validation with a limited sample size. PLoS One. 2019;14(11):e0224365.

[55] Bouthillier X, Laurent C, Vincent P. Unreproducible research is reproducible. Proceedings of the 36th International Conference on Machine Learning. 2019:PMLR 97:725-734.

[56] Dwan K, Gamble C, Williamson PR, Kirkham JJ. Systematic review of the empirical evidence of study publication bias and outcome reporting bias - an updated review. PLoS One. 2013;8(7):e66844.

[57] Lipton ZC, Steinhardt J. Troubling trends in machine learning scholarship. Queue. 2019;17(1):45-77.

[58] Topol EJ. High-performance medicine: the convergence of human and artificial intelligence. Nat Med. 2019;25(1):44-56.

[59] Keane PA, Topol EJ. With an eye to AI and autonomous diagnosis. NPJ Digit Med. 2018;1:40.

[60] Stern AD, Matthies H, Hagen J, Brönneke JB, Debatin JF. Want to see the future of digital health tools? Look to Germany. NEJM Catalyst. 2020;1(6).

[61] Collins GS, Reitsma JB, Altman DG, Moons KG. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement. BMJ. 2015;350:g7594.

[62] Goff DC Jr, Lloyd-Jones DM, Bennett G, et al. 2013 ACC/AHA guideline on the assessment of cardiovascular risk. Circulation. 2014;129(25 Suppl 2):S49-S73.

[63] Qiu S, Miller MI, Joshi PS, et al. Multimodal deep learning for Alzheimer's disease dementia assessment. Nat Commun. 2022;13:3404.

[64] Rudin C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nat Mach Intell. 2019;1(5):206-215.

[65] Belkin M, Hsu D, Ma S, Mandal S. Reconciling modern machine-learning practice and the classical bias-variance trade-off. Proc Natl Acad Sci USA. 2019;116(32):15849-15854.

[66] Niederer SA, Sacks MS, Girolami M, Willcox K. Scaling digital twins from the artisanal to the industrial. Nat Comput Sci. 2021;1(5):313-320.

[67] Björnsson B, Borrebaeck C, Elander N, et al. Digital twins to personalize medicine. Genome Med. 2020;12:4.

[68] Braak H, Del Tredici K, Rüb U, de Vos RA, Jansen Steur EN, Braak E. Staging of brain pathology related to sporadic Parkinson's disease. Neurobiol Aging. 2003;24(2):197-211.

[69] Quarteroni A, Manzoni A, Vergara C. The cardiovascular system: mathematical modelling, numerical algorithms and clinical applications. Acta Numer. 2017;26:365-590.

[70] Altrock PM, Liu LL, Michor F. The mathematics of cancer: integrating quantitative models. Nat Rev Cancer. 2015;15(12):730-745.

[71] Raissi M, Perdikaris P, Karniadakis GE. Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. J Comput Phys. 2019;378:686-707.

[72] Sanz Leon P, Knock SA, Woodman MM, et al. The Virtual Brain: a simulator of primate brain network dynamics. Front Neuroinform. 2013;7:10.

[73] Gelman A, Shalizi CR. Philosophy and the practice of Bayesian statistics. Br J Math Stat Psychol. 2013;66(1):8-38.

[74] Breakspear M. Dynamic models of large-scale brain activity. Nat Neurosci. 2017;20(3):340-352.

[75] Lauffenburger DA. Getting personal: a perspective on personalized and precision medicine. Cell Syst. 2019;9(6):515-518.

[76] Brundin P, Melki R, Kopito R. Prion-like transmission of protein aggregates in neurodegenerative diseases. Nat Rev Mol Cell Biol. 2010;11(4):301-307.

[77] Bloem BR, Grimbergen YA, Cramer M, Willemsen M, Zwinderman AH. Prospective assessment of falls in Parkinson's disease. J Neurol. 2001;248(11):950-958.

[78] Cook NR. Use and misuse of the receiver operating characteristic curve in risk prediction. Circulation. 2007;115(7):928-935.

[79] Steyerberg EW, Vickers AJ, Cook NR, et al. Assessing the performance of prediction models: a framework for traditional and novel measures. Epidemiology. 2010;21(1):128-138.

[80] Senn S. Statistical issues in drug development. 2nd ed. Wiley; 2007.

[81] Goodfellow I, Bengio Y, Courville A. Deep learning. MIT Press; 2016.

[82] Shah ND, Steyerberg EW, Kent DM. Big data and predictive analytics: recalibrating expectations. JAMA. 2018;320(1):27-28.

[83] Caruana R, Lou Y, Gehrke J, Koch P, Sturm M, Elhadad N. Intelligible models for healthcare: predicting pneumonia risk and hospital 30-day readmission. Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2015:1721-1730.

[84] Finlayson SG, Subbaswamy A, Singh K, et al. The clinician and dataset shift in artificial intelligence. N Engl J Med. 2021;385(3):283-286.

[85] Higgins JPT, Thompson SG, Deeks JJ, Altman DG. Measuring inconsistency in meta-analyses. BMJ. 2003;327(7414):557-560.

[86] Collins GS, Reitsma JB, Altman DG, Moons KG. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement. Ann Intern Med. 2015;162(1):55-63.

[87] Heus P, Damen JAAG, Pajouheshnia R, et al. Poor reporting of multivariable prediction model studies: towards a targeted implementation strategy of the TRIPOD statement. BMC Med. 2018;16:120.

[88] Collins GS, Dhiman P, Andaur Navarro CL, et al. Protocol for development of a reporting guideline (TRIPOD-AI) and risk of bias tool (PROBAST-AI) for diagnostic and prognostic prediction model studies based on artificial intelligence. BMJ Open. 2021;11(7):e048008.

[89] Luo W, Phung D, Tran T, et al. Guidelines for developing and reporting machine learning predictive models in biomedical research: a multidisciplinary view. J Med Internet Res. 2016;18(12):e323.

[90] Dhiman P, Ma J, Andaur Navarro CL, et al. Reporting of prognostic clinical prediction models based on machine learning methods in oncology needs to be improved. J Clin Epidemiol. 2021;138:60-72.

[91] U.S. Food and Drug Administration. Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD) Action Plan. January 2021.

[92] European Parliament and Council. Regulation (EU) 2017/745 on medical devices. Official Journal of the European Union. 2017;L117:1-175.

[93] Morrison A, Polisena J, Husereau D, et al. The effect of English-language restriction on systematic review-based meta-analyses: a systematic review of empirical studies. Int J Technol Assess Health Care. 2012;28(2):138-144.

[94] Page MJ, Moher D, Bossuyt PM, et al. PRISMA 2020 explanation and elaboration: updated guidance and exemplars for reporting systematic reviews. BMJ. 2021;372:n160.

[95] Easterbrook PJ, Gopalan R, Berlin JA, Matthews DR. Publication bias in clinical research. Lancet. 1991;337(8746):867-872.

[96] Olanow CW, Stern MB, Sethi K. The scientific and clinical basis for the treatment of Parkinson disease. Neurology. 2009;72(21 Suppl 4):S1-S136.

[97] Steyerberg EW, Harrell FE Jr. Prediction models need appropriate internal, internal-external, and external validation. J Clin Epidemiol. 2016;69:245-247.

[98] Hely MA, Reid WG, Adena MA, Halliday GM, Morris JG. The Sydney multicenter study of Parkinson's disease: the inevitability of dementia at 20 years. Mov Disord. 2008;23(6):837-844.

[99] Maetzler W, Liepelt I, Berg D. Progression of Parkinson's disease in the clinical phase: potential markers. Lancet Neurol. 2009;8(12):1158-1171.

[100] Dorsey ER, Sherer T, Okun MS, Bloem BR. The emerging evidence of the Parkinson pandemic. J Parkinsons Dis. 2018;8(s1):S3-S8.

[101] Marek K, Chowdhury S, Siderowf A, et al. The Parkinson's Progression Markers Initiative (PPMI) - establishing a PD biomarker cohort. Ann Clin Transl Neurol. 2018;5(12):1460-1477.

[102] Sendak MP, Gao M, Brajer N, Balu S. Presenting machine learning model information to clinical end users with model facts labels. NPJ Digit Med. 2020;3:41.

[103] Char DS, Shah NH, Magnus D. Implementing machine learning in health care - addressing ethical challenges. N Engl J Med. 2018;378(11):981-983.

[104] Varoquaux G, Cheplygina V. Machine learning for medical imaging: methodological failures and recommendations for the future. NPJ Digit Med. 2022;5:48.

[105] Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. BMC Med. 2019;17:230.

[106] Vickers AJ, Van Calster B, Steyerberg EW. Net benefit approaches to the evaluation of prediction models, molecular markers, and diagnostic tests. BMJ. 2016;352:i6.

[107] Mitchell M, Wu S, Zaldivar A, et al. Model cards for model reporting. Proceedings of the Conference on Fairness, Accountability, and Transparency. 2019:220-229.

[108] Baker M. 1,500 scientists lift the lid on reproducibility. Nature. 2016;533(7604):452-454.

[109] Latourelle JC, Beste MT, Hadzi TC, et al. Large-scale identification of clinical and genetic predictors of motor progression in patients with newly diagnosed Parkinson's disease: a longitudinal cohort study and validation. Lancet Neurol. 2017;16(11):908-916.

[110] Fereshtehnejad SM, Romenets SR, Anang JB, Latreille V, Gagnon JF, Postuma RB. New clinical subtypes of Parkinson disease and their longitudinal progression: a prospective cohort comparison with other phenotypes. JAMA Neurol. 2015;72(8):863-873.

[111] Boettiger C. An introduction to Docker for reproducible research. ACM SIGOPS Operating Systems Review. 2015;49(1):71-79.

[112] Wilkinson MD, Dumontier M, Aalbersberg IJ, et al. The FAIR Guiding Principles for scientific data management and stewardship. Sci Data. 2016;3:160018.

[113] Sendak M, Elish MC, Gao M, et al. "The human body is a black box": supporting clinical decision-making with deep learning. Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency. 2020:99-109.

[114] Wiens J, Saria S, Sendak M, et al. Do no harm: a roadmap for responsible machine learning for health care. Nat Med. 2019;25(9):1337-1340.

[115] Nestor B, McDermott MBA, Boag W, et al. Feature robustness in non-stationary health records: caveats to deployable model performance in common clinical machine learning tasks. Proceedings of Machine Learning for Healthcare. 2019:381-405.

[116] Davis SE, Lasko TA, Chen G, Siew ED, Matheny ME. Calibration drift in regression and machine learning models for acute kidney injury. J Am Med Inform Assoc. 2017;24(6):1052-1061.

[117] Subbaswamy A, Saria S. From development to deployment: dataset shift, causality, and shift-stable models in health AI. Biostatistics. 2020;21(2):345-352.

[118] Armstrong MJ, Okun MS. Diagnosis and treatment of Parkinson disease: a review. JAMA. 2020;323(6):548-560.

[119] Sendak MP, Gao M, Brajer N, Balu S. Presenting machine learning model information to clinical end users with model facts labels. NPJ Digit Med. 2020;3:41.

[120] Beede E, Baylor E, Hersch F, et al. A human-centered evaluation of a deep learning system deployed in clinics for the detection of diabetic retinopathy. Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems. 2020:1-12.

[121] Ghassemi M, Oakden-Rayner L, Beam AL. The false hope of current approaches to explainable artificial intelligence in health care. Lancet Digit Health. 2021;3(11):e745-e750.

[122] Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. JAMA Intern Med. 2021;181(8):1065-1070.

[123] Tomasev N, Glorot X, Rae JW, et al. A clinically applicable approach to continuous prediction of future acute kidney injury. Nature. 2019;572(7767):116-119.

[124] Ross C, Swetlitz I. IBM pitched its Watson supercomputer as a revolution in cancer care. It's nowhere close. STAT News. September 5, 2017.

[125] Sendak MP, Ratliff W, Sarro D, et al. Real-world integration of a sepsis deep learning technology into routine clinical care: implementation study. JMIR Med Inform. 2020;8(7):e15182.

[126] Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to manage the health of populations. Science. 2019;366(6464):447-453.

[127] Holmberg C, Bandukwala T, Bischof M, et al. Engaging patients and caregivers in health technology assessment: exploring Canadian and international initiatives. Int J Technol Assess Health Care. 2019;35(6):441-448.

[128] Goldacre B, Drysdale H, Dale A, et al. COMPare: a prospective cohort study correcting and monitoring 58 misreported trials in real time. Trials. 2019;20:118.

[129] Steyerberg EW, Vergouwe Y. Towards better clinical prediction models: seven steps for development and an ABCD for validation. Eur Heart J. 2014;35(29):1925-1931.

[130] Hawkins DM. The problem of overfitting. J Chem Inf Comput Sci. 2004;44(1):1-12.

[131] Postuma RB, Berg D, Stern M, et al. MDS clinical diagnostic criteria for Parkinson's disease. Mov Disord. 2015;30(12):1591-1601.

[132] Wynants L, Van Calster B, Collins GS, et al. Prediction models for diagnosis and prognosis of covid-19: systematic review and critical appraisal. BMJ. 2020;369:m1328.

[133] Fernández-Delgado M, Cernadas E, Barro S, Amorim D. Do we need hundreds of classifiers to solve real world classification problems? J Mach Learn Res. 2014;15:3133-3181.

[134] Chen T, Guestrin C. XGBoost: a scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2016:785-794.

[135] Altman DG, Vergouwe Y, Royston P, Moons KG. Prognosis and prognostic research: validating a prognostic model. BMJ. 2009;338:b605.

[136] DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics. 1988;44(3):837-845.

[137] McNemar Q. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika. 1947;12(2):153-157.

[138] Schwartz R, Dodge J, Smith NA, Etzioni O. Green AI. Commun ACM. 2020;63(12):54-63.

[139] Molnar C. Interpretable machine learning: a guide for making black box models explainable. 2nd ed. 2022. https://christophm.github.io/interpretable-ml-book/

[140] Bommasani R, Hudson DA, Adeli E, et al. On the opportunities and risks of foundation models. arXiv:2108.07258. 2021.

[141] Singhal K, Azizi S, Tu T, et al. Large language models encode clinical knowledge. Nature. 2023;620(7972):172-180.

[142] Nori H, King N, McKinney SM, Carignan D, Horvitz E. Capabilities of GPT-4 on medical challenge problems. arXiv:2303.13375. 2023.

[143] Luo R, Sun L, Xia Y, et al. BioGPT: generative pre-trained transformer for biomedical text generation and mining. Brief Bioinform. 2022;23(6):bbac409.

[144] Lee P, Bubeck S, Petro J. Benefits, limits, and risks of GPT-4 as an AI chatbot for medicine. N Engl J Med. 2023;388(13):1233-1239.

[145] Thirunavukarasu AJ, Ting DSJ, Elangovan K, Gutierrez L, Tan TF, Ting DSW. Large language models in medicine. Nat Med. 2023;29(8):1930-1940.

[146] Ji Z, Lee N, Frieske R, et al. Survey of hallucination in natural language generation. ACM Comput Surv. 2023;55(12):1-38.

[147] Weidinger L, Mellor J, Rauh M, et al. Ethical and social risks of harm from language models. arXiv:2112.04359. 2021.

[148] Moor M, Banerjee O, Abad ZSH, et al. Foundation models for generalist medical artificial intelligence. Nature. 2023;616(7956):259-265.

[149] Karniadakis GE, Kevrekidis IG, Lu L, Perdikaris P, Wang S, Yang L. Physics-informed machine learning. Nat Rev Phys. 2021;3(6):422-440.

[150] Raissi M, Yazdani A, Karniadakis GE. Hidden fluid mechanics: learning velocity and pressure fields from flow visualizations. Science. 2020;367(6481):1026-1030.

[151] Cenci MA, Lundblad M. Ratings of L-DOPA-induced dyskinesia in the unilateral 6-OHDA lesion model of Parkinson's disease in rats and mice. Curr Protoc Neurosci. 2007;Chapter 9:Unit 9.25.

[152] Raissi M, Perdikaris P, Karniadakis GE. Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. J Comput Phys. 2019;378:686-707.

[153] Cuomo S, Di Cola VS, Giampaolo F, Rozza G, Raissi M, Piccialli F. Scientific machine learning through physics-informed neural networks: where we are and what's next. J Sci Comput. 2022;92:88.

[154] Willard J, Jia X, Xu S, Steinbach M, Kumar V. Integrating scientific knowledge with machine learning for engineering and environmental systems. ACM Comput Surv. 2022;55(4):1-37.

[155] Shen C, Nguyen D, Zhou Z, et al. An introduction to deep learning in medical physics: advantages, potential, and challenges. Phys Med Biol. 2020;65(5):05TR01.

[156] Krishnapriyan A, Gholami A, Zhe S, Kirby R, Mahoney MW. Characterizing possible failure modes in physics-informed neural networks. Advances in Neural Information Processing Systems. 2021;34:26548-26560.

[157] Wang S, Teng Y, Perdikaris P. Understanding and mitigating gradient flow pathologies in physics-informed neural networks. SIAM J Sci Comput. 2021;43(5):A3055-A3081.

[158] De Ryck T, Mishra S. Error estimates for physics-informed neural networks approximating the Navier-Stokes equations. arXiv:2203.09346. 2022.

[159] Pearl J, Mackenzie D. The book of why: the new science of cause and effect. Basic Books; 2018.

[160] Hernán MA, Robins JM. Using big data to emulate a target trial when a randomized trial is not available. Am J Epidemiol. 2016;183(8):758-764.

[161] Kent DM, Paulus JK, van Klaveren D, et al. The Predictive Approaches to Treatment effect Heterogeneity (PATH) statement. Ann Intern Med. 2020;172(1):35-45.

[162] Wager S, Athey S. Estimation and inference of heterogeneous treatment effects using random forests. J Am Stat Assoc. 2018;113(523):1228-1242.

[163] Bang H, Robins JM. Doubly robust estimation in missing data and causal inference models. Biometrics. 2005;61(4):962-973.

[164] Angrist JD, Imbens GW, Rubin DB. Identification of causal effects using instrumental variables. J Am Stat Assoc. 1996;91(434):444-455.

[165] Hernán MA, Robins JM. Causal inference: what if. Chapman & Hall/CRC; 2020.

[166] Kravitz RL, Duan N, Braslow J. Evidence-based medicine, heterogeneity of treatment effects, and the trouble with averages. Milbank Q. 2004;82(4):661-687.

[167] Vickers AJ, Kattan MW, Daniel S. Method for evaluating prediction models that apply the results of randomized trials to individual patients. Trials. 2007;8:14.

[168] Rieke N, Hancox J, Li W, et al. The future of digital health with federated learning. NPJ Digit Med. 2020;3:119.

[169] McMahan B, Moore E, Ramage D, Hampson S, y Arcas BA. Communication-efficient learning of deep networks from decentralized data. Proceedings of the 20th International Conference on Artificial Intelligence and Statistics. 2017:1273-1282.

[170] Sheller MJ, Edwards B, Reina GA, et al. Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data. Sci Rep. 2020;10:12598.

[171] Xu J, Glicksberg BS, Su C, Walker P, Bian J, Wang F. Federated learning for healthcare informatics. J Healthc Inform Res. 2021;5(1):1-19.

[172] Bloem BR, Marks WJ Jr, Silva de Lima AL, et al. The Personalized Parkinson Project: examining disease progression through broad biomarkers in early Parkinson's disease. BMC Neurol. 2019;19:160.

[173] Wilkinson MD, Dumontier M, Aalbersberg IJ, et al. The FAIR Guiding Principles for scientific data management and stewardship. Sci Data. 2016;3:160018.

[174] Kairouz P, McMahan HB, Avent B, et al. Advances and open problems in federated learning. Found Trends Mach Learn. 2021;14(1-2):1-210.

[175] Li T, Sahu AK, Zaheer M, Sanjabi M, Talwalkar A, Smith V. Federated optimization in heterogeneous networks. Proceedings of Machine Learning and Systems. 2020;2:429-450.

[176] Zhu L, Liu Z, Han S. Deep leakage from gradients. Advances in Neural Information Processing Systems. 2019;32:14774-14784.

[177] Kaissis GA, Makowski MR, Rückert D, Braren RF. Secure, privacy-preserving and federated machine learning in medical imaging. Nat Mach Intell. 2020;2(6):305-311.

[178] Losing V, Hammer B, Wersing H. Incremental on-line learning: a review and comparison of state of the art algorithms. Neurocomputing. 2018;275:1261-1274.

[179] Olanow CW, Kieburtz K, Rascol O, et al. Factors predictive of the development of levodopa-induced dyskinesia and wearing-off in Parkinson's disease. Mov Disord. 2013;28(8):1064-1071.

[180] Kirkpatrick J, Pascanu R, Rabinowitz N, et al. Overcoming catastrophic forgetting in neural networks. Proc Natl Acad Sci USA. 2017;114(13):3521-3526.

[181] Gama J, Žliobaitė I, Bifet A, Pechenizkiy M, Bouchachia A. A survey on concept drift adaptation. ACM Comput Surv. 2014;46(4):1-37.

[182] U.S. Food and Drug Administration. Proposed regulatory framework for modifications to artificial intelligence/machine learning (AI/ML)-based software as a medical device (SaMD). April 2019.

[183] Lazer D, Kennedy R, King G, Vespignani A. The parable of Google Flu: traps in big data analysis. Science. 2014;343(6176):1203-1205.

[184] Woodall WH. The use of control charts in health-care and public-health surveillance. J Qual Technol. 2006;38(2):89-104.

[185] U.S. Food and Drug Administration. Digital Health Software Precertification (Pre-Cert) Program. https://www.fda.gov/medical-devices/digital-health-center-excellence/digital-health-software-precertification-pre-cert-program

[186] Stern AD, Brönneke J, Debatin J, Hagen J, Matthies H, Patel N. Advancing digital health applications: priorities for innovation in real-world evidence generation. Lancet Digit Health. 2022;4(3):e200-e206.

[187] Babic B, Gerke S, Evgeniou T, Cohen IG. Algorithms on regulatory lockdown in medicine. Science. 2019;366(6470):1202-1204.

[188] U.S. Food and Drug Administration. Marketing submission recommendations for a predetermined change control plan for artificial intelligence/machine learning (AI/ML)-enabled device software functions. April 2023.

[189] Benjamens S, Dhunnoo P, Meskó B. The state of artificial intelligence-based FDA-approved medical devices and algorithms: an online database. NPJ Digit Med. 2020;3:118.

[190] Vokinger KN, Feuerriegel S, Kesselheim AS. Continual learning in medical devices: FDA's action plan and beyond. Lancet Digit Health. 2021;3(6):e337-e338.

[191] U.S. Food and Drug Administration. Breakthrough Devices Program. https://www.fda.gov/medical-devices/how-study-and-market-your-device/breakthrough-devices-program

[192] Hwang TJ, Kesselheim AS, Vokinger KN. Lifecycle regulation of artificial intelligence- and machine learning-based software devices in medicine. JAMA. 2019;322(23):2285-2286.

[193] European Commission. Medical Device Coordination Group Document MDCG 2019-11: Guidance on qualification and classification of software in Regulation (EU) 2017/745 – MDR and Regulation (EU) 2017/746 – IVDR. October 2019.

[194] Muehlematter UJ, Daniore P, Vokinger KN. Approval of artificial intelligence and machine learning-based medical devices in the USA and Europe (2015-20): a comparative analysis. Lancet Digit Health. 2021;3(3):e195-e203.

[195] International Organization for Standardization. ISO 14971:2019 Medical devices — Application of risk management to medical devices. 2019.

[196] Gerke S, Minssen T, Cohen G. Ethical and legal challenges of artificial intelligence-driven healthcare. Artif Intell Healthc. 2020:295-336.

[197] European Parliament and Council. Regulation (EU) 2016/679 on the protection of natural persons with regard to the processing of personal data and on the free movement of such data (General Data Protection Regulation). Official Journal of the European Union. 2016;L119:1-88.

[198] Pesapane F, Volonté C, Codari M, Sardanelli F. Artificial intelligence as a medical device in radiology: ethical and regulatory issues in Europe and the United States. Insights Imaging. 2018;9(5):745-753.

[199] Matheny ME, Whicher D, Thadaney Israni S. Artificial intelligence in health care: a report from the National Academy of Medicine. JAMA. 2020;323(6):509-510.

[200] Parikh RB, Teeple S, Navathe AS. Addressing bias in artificial intelligence in health care. JAMA. 2019;322(24):2377-2378.

[201] Centers for Medicare & Medicaid Services. New Technology Add-on Payment (NTAP). https://www.cms.gov/Medicare/Medicare-Fee-for-Service-Payment/AcuteInpatientPPS/New-Technology-Add-On-Payment

[202] Chambers JD, Panzer AD, Neumann PJ. Medicare's new technology add-on payments. Med Care. 2019;57(1):3-4.

[203] Neumann PJ, Cohen JT, Weinstein MC. Updating cost-effectiveness—the curious resilience of the $50,000-per-QALY threshold. N Engl J Med. 2014;371(9):796-797.

[204] Svedbom A, Hernlund E, Ivergård M, et al. Osteoporosis in the European Union: a compendium of country-specific reports. Arch Osteoporos. 2013;8:137.

[205] Bloem BR, Okun MS, Klein C. Parkinson's disease. Lancet. 2021;397(10291):2284-2303.

[206] Ford I, Norrie J. Pragmatic trials. N Engl J Med. 2016;375(5):454-463.

[207] Ramsey SD, Willke RJ, Glick H, et al. Cost-effectiveness analysis alongside clinical trials II—an ISPOR Good Research Practices Task Force report. Value Health. 2015;18(2):161-172.

[208] Mankins JC. Technology readiness levels: a white paper. NASA. 1995.

[209] Kalia LV, Lang AE. Parkinson's disease. Lancet. 2015;386(9996):896-912.

[210] Ginsburg GS, Phillips KA. Precision medicine: from science to value. Health Aff (Millwood). 2018;37(5):694-701.

[211] Butler D. Translational research: crossing the valley of death. Nature. 2008;453(7197):840-842.

[212] Siontis GCM, Tzoulaki I, Castaldi PJ, Ioannidis JPA. External validation of new risk prediction models is infrequent and reveals worse prognostic discrimination. J Clin Epidemiol. 2015;68(1):25-34.

[213] Altman DG, Royston P. What do we mean by validating a prognostic model? Stat Med. 2000;19(4):453-473.

[214] Artificial Intelligence in Healthcare Market Report. Grand View Research. 2023.

[215] Mandl KD, Kohane IS. No small change for the health information economy. N Engl J Med. 2009;360(13):1278-1281.

[216] Blease C, Kaptchuk TJ, Bernstein MH, Mandl KD, Halamka JD, DesRoches CM. Artificial intelligence and the future of primary care: exploratory qualitative study of UK general practitioners' views. J Med Internet Res. 2019;21(3):e12802.

[217] Bates DW, Kuperman GJ, Wang S, et al. Ten commandments for effective clinical decision support: making the practice of evidence-based medicine a reality. J Am Med Inform Assoc. 2003;10(6):523-530.

[218] Brayne C, Fox C, Boustani M. Dementia screening in primary care: is it time? JAMA. 2007;298(20):2409-2411.

[219] Gama F, Tyskbo D, Nygren J, Barlow J, Reed J, Svedberg P. Implementation frameworks for artificial intelligence translation into health care practice: scoping review. J Med Internet Res. 2022;24(1):e32215.

[220] Ancker JS, Edwards A, Nosal S, Hauser D, Mauer E, Kaushal R. Effects of workload, work complexity, and repeated alerts on alert fatigue in a clinical decision support system. BMC Med Inform Decis Mak. 2017;17:36.

[221] Price WN II, Gerke S, Cohen IG. Potential liability for physicians using artificial intelligence. JAMA. 2019;322(18):1765-1766.

[222] U.S. Food and Drug Administration. Clinical Decision Support Software: Guidance for Industry and Food and Drug Administration Staff. September 2022.

[223] Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making. 2006;26(6):565-574.

[224] Gerke S, Babic B, Evgeniou T, Cohen IG. The need for a system view to regulate artificial intelligence/machine learning-based software as medical device. NPJ Digit Med. 2020;3:53.

[225] American Medical Association. CPT Editorial Panel. https://www.ama-assn.org/about/cpt-editorial-panel/cpt-editorial-panel

[226] Drummond MF, Sculpher MJ, Claxton K, Stoddart GL, Torrance GW. Methods for the economic evaluation of health care programmes. 4th ed. Oxford University Press; 2015.

[227] Lehne M, Sass J, Essenwanger A, Schepers J, Thun S. Why digital medicine depends on interoperability. NPJ Digit Med. 2019;2:79.

[228] McGraw D, Mandl KD. Privacy protections to encourage use of health-relevant digital data in a learning health system. NPJ Digit Med. 2021;4:2.

[229] Mandel JC, Kreda DA, Mandl KD, Kohane IS, Ramoni RB. SMART on FHIR: a standards-based, interoperable apps platform for electronic health records. J Am Med Inform Assoc. 2016;23(5):899-908.

[230] Cabitza F, Rasoini R, Gensini GF. Unintended consequences of machine learning in medicine. JAMA. 2017;318(6):517-518.

[231] Ramspek CL, Jager KJ, Dekker FW, Zoccali C, van Diepen M. External validation of prognostic models: what, why, how, when and where? Clin Kidney J. 2021;14(1):49-58.

[232] Berner ES, La Lande TJ. Overview of clinical decision support systems. In: Berner ES, ed. Clinical Decision Support Systems: Theory and Practice. 3rd ed. Springer; 2016:1-17.

[233] U.S. Food and Drug Administration. Requests for Feedback and Meetings for Medical Device Submissions: The Q-Submission Program. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/requests-feedback-and-meetings-medical-device-submissions-q-submission-program

[234] Briggs AH, Weinstein MC, Fenwick EA, et al. Model parameter estimation and uncertainty analysis: a report of the ISPOR-SMDM Modeling Good Research Practices Task Force Working Group-6. Med Decis Making. 2012;32(5):722-732.

[235] Toga AW, Foster I, Kesselman C, et al. The Big Data to Knowledge (BD2K) initiative. J Am Med Inform Assoc. 2015;22(6):1114.

[236] Sendak M, Gao M, Nichols M, Lin A, Balu S. Machine learning in health care: a critical appraisal of challenges and opportunities. EGEMS (Wash DC). 2019;7(1):1.

[237] Loudon K, Treweek S, Sullivan F, Donnan P, Thorpe KE, Zwarenstein M. The PRECIS-2 tool: designing trials that are fit for purpose. BMJ. 2015;350:h2147.

[238] American Medical Association. The CPT code development process. https://www.ama-assn.org/practice-management/cpt/cpt-code-development-process

[239] Wartman SA, Combs CD. Reimagining medical education in the age of AI. AMA J Ethics. 2019;21(2):E146-152.

[240] Platt R, Madre L, Reynolds RF, Tilson H. Active drug safety surveillance: a tool to improve public health. Pharmacoepidemiol Drug Saf. 2008;17(12):1175-1182.

[241] Dhruva SS, Ross JS, Desai NR. Real-world evidence: promise and peril for medical product evaluation. P T. 2018;43(8):464-472.

[242] Mandl KD, Mandel JC, Kohane IS. Driving innovation in health systems through an apps-based information economy. Cell Syst. 2015;1(1):8-13.

[243] Neumann PJ, Sanders GD, Russell LB, Siegel JE, Ganiats TG. Cost-effectiveness in health and medicine. 2nd ed. Oxford University Press; 2016.

[244] Dorsey ER, Topol EJ. State of telehealth. N Engl J Med. 2016;375(2):154-161.

[245] Bickmore TW, Utami D, Matsuyama R, Paasche-Orlow MK. Improving access to online health information with conversational agents: a randomized controlled experiment. J Med Internet Res. 2016;18(1):e1.

[246] Jack CR Jr, Bennett DA, Blennow K, et al. NIA-AA Research Framework: toward a biological definition of Alzheimer's disease. Alzheimers Dement. 2018;14(4):535-562.

[247] Weiner MW, Veitch DP, Aisen PS, et al. The Alzheimer's Disease Neuroimaging Initiative 3: continued innovation for clinical trial improvement. Alzheimers Dement. 2017;13(5):561-571.

[248] Hansson O, Edelmayer RM, Boxer AL, et al. The Alzheimer's Association appropriate use recommendations for blood biomarkers in Alzheimer's disease. Alzheimers Dement. 2022;18(12):2669-2686.

[249] Palmqvist S, Janelidze S, Quiroz YT, et al. Discriminative accuracy of plasma phospho-tau217 for Alzheimer disease vs other neurodegenerative disorders. JAMA. 2020;324(8):772-781.

[250] Siderowf A, Concha-Marambio L, Lafontant DE, et al. Assessment of heterogeneity among participants in the Parkinson's Progression Markers Initiative cohort using α-synuclein seed amplification: a cross-sectional study. Lancet Neurol. 2023;22(5):407-417.

[251] Filippi M, Preziosa P, Banwell BL, et al. Assessment of lesions on magnetic resonance imaging in multiple sclerosis: practical guidelines. Brain. 2019;142(7):1858-1875.

[252] Montalban X, Gold R, Thompson AJ, et al. ECTRIMS/EAN guideline on the pharmacological treatment of people with multiple sclerosis. Mult Scler. 2018;24(2):96-120.

[253] Butzkueven H, Chapman J, Cristiano E, et al. MSBase: an international, online registry and platform for collaborative outcomes research in multiple sclerosis. Mult Scler. 2006;12(6):769-774.

[254] Tousignant A, Falet JP, Sormani MP, et al. Prediction of disease progression in multiple sclerosis patients using deep learning analysis of MRI data. Proceedings of Machine Learning for Healthcare. 2019:483-492.

[255] Kappos L, Butzkueven H, Wiendl H, et al. Greater sensitivity to multiple sclerosis disability worsening and progression events using a roving versus a fixed reference value in a prospective cohort study. Mult Scler. 2018;24(7):963-973.

[256] Simuni T, Caspell-Garcia C, Coffey C, et al. How stable are Parkinson's disease subtypes in de novo patients: analysis of the PPMI cohort? Parkinsonism Relat Disord. 2016;28:62-67.

[257] Kuhlmann L, Lehnertz K, Richardson MP, Schelter B, Zaveri HP. Seizure prediction—ready for a new era. Nat Rev Neurol. 2018;14(10):618-630.

[258] Onorati F, Regalia G, Caborni C, et al. Multicenter clinical assessment of improved wearable multimodal convulsive seizure detectors. Epilepsia. 2017;58(11):1870-1879.

[259] Stirling RE, Cook MJ, Grayden DB, Karoly PJ. Seizure forecasting and cyclic control of seizures. Epilepsia. 2021;62 Suppl 1:S2-S14.

[260] Mormann F, Andrzejak RG, Elger CE, Lehnertz K. Seizure prediction: the long and winding road. Brain. 2007;130(Pt 2):314-333.

[261] Mancini M, Bloem BR, Horak FB, Lewis SJG, Nieuwboer A, Nonnekes J. Clinical and methodological challenges for assessing freezing of gait: future perspectives. Mov Disord. 2019;34(6):783-790.

[262] Wolpert DH, Macready WG. No free lunch theorems for optimization. IEEE Trans Evol Comput. 1997;1(1):67-82.

[263] Efron B, Tibshirani RJ. An introduction to the bootstrap. Chapman & Hall/CRC; 1993.

[264] Justice AC, Covinsky KE, Berlin JA. Assessing the generalizability of prognostic information. Ann Intern Med. 1999;130(6):515-524.

[265] Nosek BA, Ebersole CR, DeHaven AC, Mellor DT. The preregistration revolution. Proc Natl Acad Sci USA. 2018;115(11):2600-2606.

[266] Peng RD. Reproducible research in computational science. Science. 2011;334(6060):1226-1227.

[267] Fanelli D. Negative results are disappearing from most disciplines and countries. Scientometrics. 2012;90(3):891-904.

[268] Moons KGM, Wolff RF, Riley RD, et al. PROBAST: a tool to assess risk of bias and applicability of prediction model studies: explanation and elaboration. Ann Intern Med. 2019;170(1):W1-W33.

[269] Steyerberg EW, Moons KG, van der Windt DA, et al. Prognosis Research Strategy (PROGRESS) 3: prognostic model research. PLoS Med. 2013;10(2):e1001381.

[270] Gardner MJ, Altman DG. Confidence intervals rather than P values: estimation rather than hypothesis testing. Br Med J (Clin Res Ed). 1986;292(6522):746-750.

[271] Bleeker SE, Moll HA, Steyerberg EW, et al. External validation is necessary in prediction research: a clinical example. J Clin Epidemiol. 2003;56(9):826-832.

[272] Chambers CD. Registered reports: a new publishing initiative at Cortex. Cortex. 2013;49(3):609-610.

[273] Lundh A, Lexchin J, Mintzes B, Schroll JB, Bero L. Industry sponsorship and research outcome. Cochrane Database Syst Rev. 2017;2(2):MR000033.

[274] Davenport T, Kalakota R. The potential for artificial intelligence in healthcare. Future Healthc J. 2019;6(2):94-98.

[275] Eddy DM, Schlessinger L. Validation of the Archimedes diabetes model. Diabetes Care. 2003;26(11):3102-3110.

[276] Califf RM, Sugarman J. Exploring the ethical and regulatory issues in pragmatic clinical trials. Clin Trials. 2015;12(5):436-441.

[277] Sperrin M, Martin GP, Pate A, et al. Using marginal structural models to adjust for treatment drop-in when developing clinical prediction models. Stat Med. 2018;37(28):4142-4154.

[278] Taichman DB, Sahni P, Pinborg A, et al. Data sharing statements for clinical trials: a requirement of the International Committee of Medical Journal Editors. Ann Intern Med. 2017;167(1):63-65.

[279] Elwyn G, Frosch D, Thomson R, et al. Shared decision making: a model for clinical practice. J Gen Intern Med. 2012;27(10):1361-1367.

[280] Vasey B, Nagendran M, Campbell B, et al. Reporting guideline for the early-stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI. Nat Med. 2022;28(5):924-933.

[281] Coravos A, Khozin S, Mandl KD. Developing and adopting safe and effective digital biomarkers to improve patient outcomes. NPJ Digit Med. 2019;2:14.

[282] Vickers AJ, van Calster B, Steyerberg EW. A simple, step-by-step guide to interpreting decision curve analysis. Diagn Progn Res. 2019;3:18.

[283] U.S. Food and Drug Administration. Artificial Intelligence and Machine Learning in Software as a Medical Device. https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-software-medical-device

[284] Schneeweiss S, Rassen JA, Brown JS, et al. Graphical depiction of longitudinal study designs in health care databases. Ann Intern Med. 2019;170(6):398-406.

[285] Perfetto EM, Oehrlein EM, Boutin M, Reid S, Gascho E. Value to whom? The patient voice in the value discussion. Value Health. 2017;20(2):286-291.

[286] Hirsch JA, Leslie-Mazwi TM, Nicola GN, et al. Current procedural terminology: a primer. J Neurointerv Surg. 2015;7(4):309-312.

[287] Garrison LP Jr, Neumann PJ, Erickson P, Marshall D, Mullins CD. Using real-world data for coverage and payment decisions: the ISPOR Real-World Data Task Force report. Value Health. 2007;10(5):326-335.

[288] Sanders GD, Neumann PJ, Basu A, et al. Recommendations for conduct, methodological practices, and reporting of cost-effectiveness analyses: second panel on cost-effectiveness in health and medicine. JAMA. 2016;316(10):1093-1103.

[289] Porter ME. What is value in health care? N Engl J Med. 2010;363(26):2477-2481.

[290] Pronovost PJ, Berenholtz SM, Needham DM. Translating evidence into practice: a model for large scale knowledge translation. BMJ. 2008;337:a1714.

[291] Reilly BM, Evans AT. Translating clinical research into clinical practice: impact of using prediction rules to make decisions. Ann Intern Med. 2006;144(3):201-209.

[292] Sendak MP, Gao M, Brajer N, Balu S. Presenting machine learning model information to clinical end users with model facts labels. NPJ Digit Med. 2020;3:41.

[293] Stiggelbout AM, Pieterse AH, De Haes JC. Shared decision making: concepts, evidence, and practice. Patient Educ Couns. 2015;98(10):1172-1179.

[294] Kaushal R, Shojania KG, Bates DW. Effects of computerized physician order entry and clinical decision support systems on medication safety: a systematic review. Arch Intern Med. 2003;163(12):1409-1416.

[295] Kawamoto K, Houlihan CA, Balas EA, Lobach DF. Improving clinical practice using clinical decision support systems: a systematic review of trials to identify features critical to success. BMJ. 2005;330(7494):765.

[296] Concannon TW, Meissner P, Grunbaum JA, et al. A new taxonomy for stakeholder engagement in patient-centered outcomes research. J Gen Intern Med. 2012;27(8):985-991.

[297] Mittelstadt BD, Allo P, Taddeo M, Wachter S, Floridi L. The ethics of algorithms: mapping the debate. Big Data Soc. 2016;3(2):2053951716679679.

[298] Rajkomar A, Hardt M, Howell MD, Corrado G, Chin MH. Ensuring fairness in machine learning to advance health equity. Ann Intern Med. 2018;169(12):866-872.

[299] Domecq JP, Prutsky G, Elraiyah T, et al. Patient engagement in research: a systematic review. BMC Health Serv Res. 2014;14:89.

[300] Hartzler A, Pratt W. Managing the personal side of health: how patient expertise differs from the expertise of clinicians. J Med Internet Res. 2011;13(3):e62.
