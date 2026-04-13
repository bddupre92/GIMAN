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

