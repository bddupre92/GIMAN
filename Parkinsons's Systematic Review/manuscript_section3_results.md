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
