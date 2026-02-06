# GIMAN Manuscript Preparation Action Plan

**Created**: October 5, 2025  
**Purpose**: Step-by-step action items for manuscript preparation  
**Timeline**: October 2025 - February 2026

---

## 🚨 URGENT: Phase 6 Manuscript (Due: October 31, 2025)

### Week 1 (October 6-12, 2025) - CURRENT WEEK

#### Monday-Tuesday: Data Review & Figure Selection
- [ ] Review all Phase 6 visualization directories (`visualizations/phase6_task6_1_attention/` through `task6_6_dashboard/`)
- [ ] Select best examples from each task for main figures
- [ ] Create figure legends for each selected visualization
- [ ] List: 4 main figures needed (attention, GNNExplainer, attribution, dashboard)

#### Wednesday-Thursday: Results Section Draft
- [ ] **Section 4.1: Attention Weight Analysis**
  - Describe attention patterns observed in Phase 4 and Phase 5
  - Report quantitative metrics (e.g., attention entropy, top-k connections)
  - Reference Figure 1 (attention networks)
  
- [ ] **Section 4.2: GNNExplainer Subgraph Identification**
  - Report subgraph sizes and importance scores
  - Describe clinical relevance of identified connections
  - Reference Figure 2 (explanatory subgraphs)
  
- [ ] **Section 4.3: Feature Attribution Analysis**
  - Report top features across Phase 4 and Phase 5 applications
  - Compare attribution patterns between tasks
  - Reference Figure 3 (feature importance)
  
- [ ] **Section 4.4: Clinical Validation**
  - Describe patient clustering findings ("progression twins")
  - Report counterfactual scenario examples
  - Reference Figure 4 (clinical dashboard)

#### Friday: Methods Section Draft
- [ ] **GNN Architecture Recap** (300 words)
  - Brief description of GIMAN architecture
  - Graph construction methodology
  - Training procedure reference
  
- [ ] **Explainability Methods** (900 words)
  - Attention weight extraction and analysis
  - GNNExplainer algorithm and parameters
  - Integrated gradients for feature attribution
  - Clustering methodology for patient similarity
  - Counterfactual generation approach
  
- [ ] **Validation Framework** (300 words)
  - How explanations were validated
  - Clinical expert review process (if applicable)
  - Quantitative metrics for explanation quality

### Week 2 (October 13-19, 2025)

#### Monday-Tuesday: Abstract & Introduction
- [ ] **Abstract** (150 words, Nature format)
  - Background: Black box problem in medical AI
  - Methods: Multi-method explainability framework for GNNs
  - Results: Key findings (specific numbers if available)
  - Conclusion: Impact on clinical trust and adoption
  
- [ ] **Introduction** (600-800 words)
  - Paragraph 1: GNNs in medicine - growing use, black box problem
  - Paragraph 2: Prior explainability work - limitations for graphs
  - Paragraph 3: Clinical need for interpretability
  - Paragraph 4: Study objectives and contributions

#### Wednesday-Thursday: Discussion & Figures
- [ ] **Discussion** (800-1000 words)
  - Key insight 1: Attention reveals meaningful patient connections
  - Key insight 2: GNNExplainer identifies clinically relevant subgraphs
  - Key insight 3: Feature attribution aligns with domain knowledge
  - Comparison to other explainability methods
  - Limitations (e.g., explanation stability, clinical validation needs)
  - Future directions (prospective validation, user studies)
  
- [ ] **Create Figure Panel Files**
  - Figure 1: Attention weight networks (multi-panel)
  - Figure 2: GNNExplainer subgraphs (examples from Phase 4 and 5)
  - Figure 3: Feature attribution bar charts (comparative)
  - Figure 4: Clinical dashboard screenshot (annotated)

#### Friday: Supplementary Materials
- [ ] **Extended Methods** (additional technical details)
- [ ] **Supplementary Figures** (additional examples, validation plots)
- [ ] **Supplementary Tables** (quantitative metrics for all explanations)
- [ ] **Code Availability Statement** (GitHub repository info)

### Week 3 (October 20-26, 2025)

#### Monday-Tuesday: Internal Review
- [ ] Self-review complete draft
- [ ] Check all references are formatted correctly
- [ ] Verify all figures are cited in text
- [ ] Word count check (Nature MI has limits)
- [ ] Send to co-authors for review (if applicable)

#### Wednesday-Friday: Revisions & Formatting
- [ ] Address co-author feedback
- [ ] Format according to Nature Machine Intelligence guidelines
- [ ] Prepare cover letter
- [ ] Complete submission forms (author contributions, conflicts of interest)
- [ ] Prepare graphical abstract (if required)

### Week 4 (October 27-31, 2025)

#### Monday-Wednesday: Final Preparations
- [ ] Final proofread
- [ ] Check all supplementary files are ready
- [ ] Verify figure quality (300 DPI minimum)
- [ ] Confirm all co-author approvals

#### Thursday-Friday: SUBMISSION
- [ ] **SUBMIT to Nature Machine Intelligence by October 31, 2025**
- [ ] Post preprint to bioRxiv
- [ ] Tweet/announce preprint
- [ ] Email notification to collaborators

---

## 📊 Phase 4 Manuscript (Due: Early January 2026)

### November Week 1-2: Data Extraction

#### Task 1.1: Extract Cohort Characteristics
```python
# Script to run: extract_phase4_cohort_stats.py
import pandas as pd
import json

# Load longitudinal observations
df = pd.read_csv('data/longitudinal_cohort/longitudinal_observations.csv')

# Calculate Table 1 statistics
cohort_stats = {
    'n_patients': df['PATNO'].nunique(),
    'n_observations': len(df),
    'mean_age': df.groupby('PATNO')['AGE'].first().mean(),
    'sex_distribution': df.groupby('PATNO')['SEX'].first().value_counts(),
    'mean_visits': df.groupby('PATNO').size().mean(),
    # ... add more
}

# Save as CSV for Table 1
pd.DataFrame([cohort_stats]).to_csv('tables/phase4_table1_cohort.csv')
```

**Action Items**:
- [ ] Create `extract_phase4_cohort_stats.py` script
- [ ] Run script to generate Table 1 CSV
- [ ] Verify all metrics match JSON reports
- [ ] Calculate confidence intervals where needed

#### Task 1.2: Extract Clustering Metrics
```python
# Load clustering report
with open('data/longitudinal_cohort/clustering_report.json') as f:
    clustering = json.load(f)

# Extract for Table 2
subtype_metrics = {
    'n_subtypes': clustering['n_subtypes'],
    'silhouette_score': clustering['silhouette_score'],
    'ch_index': clustering['calinski_harabasz_index'],
    'db_index': clustering['davies_bouldin_index'],
    # ... add more
}
```

**Action Items**:
- [ ] Extract all clustering metrics from JSON
- [ ] Load `patient_trajectories_clustered.csv` for subtype counts
- [ ] Calculate subtype-specific progression rates
- [ ] Generate Table 2: Clustering Metrics

#### Task 1.3: Extract Subtype Profiles
**Action Items**:
- [ ] Load `subtype_characterization_report.json`
- [ ] Extract clinical profiles per subtype
- [ ] Load `subtype_statistical_tests.csv` for p-values
- [ ] Generate Table 3: Subtype Clinical Profiles

#### Task 1.4: Extract Prediction Performance
**Action Items**:
- [ ] Load `baseline_prediction_report.json`
- [ ] Extract AUC, accuracy, F1 scores
- [ ] Get confusion matrix data
- [ ] Generate Table 4: Baseline Prediction Performance

#### Task 1.5: Extract Trial Enrichment Results
**Action Items**:
- [ ] Load `trial_enrichment_report.json`
- [ ] Extract sample size reduction percentages
- [ ] Get power calculations
- [ ] Calculate cost savings estimates
- [ ] Generate Table 5: Trial Enrichment Impact

### November Week 3-4: Methods Section

#### Methods Section 2.1: Study Design
**Word Target**: 300 words

**Action Items**:
- [ ] Describe PPMI dataset
- [ ] Inclusion/exclusion criteria
- [ ] Longitudinal data structure
- [ ] Outcome definitions (motor progression, cognitive decline)

**Template**:
```
We utilized data from the Parkinson's Progression Markers Initiative (PPMI), 
a multicenter longitudinal observational study... [describe cohort]

Inclusion criteria: [list criteria from quality_control_report.json]
- Minimum 3 longitudinal visits
- Complete motor (UPDRS-III) and cognitive (MoCA) assessments
- [additional criteria]

The primary outcomes were:
1. Motor progression rate (UPDRS-III points per year)
2. Cognitive decline rate (MoCA points per year)
```

#### Methods Section 2.2: Latent Time Alignment
**Word Target**: 400 words

**Action Items**:
- [ ] Describe LTJMM mathematical formulation
- [ ] Explain why alignment is needed
- [ ] Detail implementation approach
- [ ] Reference `task_4_2_latent_time_alignment.py` for technical details

**Template**:
```
To account for heterogeneity in disease duration and progression rates, 
we applied a Latent Time Joint Mixed-Effects Model (LTJMM) to align 
patient trajectories on a common disease timeline...

Mathematical Formulation:
y_i(t) = α_i + β_i * f(t - τ_i) + ε_i(t)

where y_i(t) represents the observed outcome for patient i at time t, 
α_i and β_i are patient-specific random effects, f(·) is a warping 
function, and τ_i is the patient-specific latent time shift...

[Describe algorithm, convergence criteria, validation]
```

#### Methods Section 2.3: Trajectory Clustering
**Word Target**: 400 words

**Action Items**:
- [ ] Describe VaDER architecture
- [ ] Explain trajectory encoding
- [ ] Detail clustering algorithms tested (K-means, hierarchical, GMM)
- [ ] Describe validation metrics (silhouette, CH, DB indices)

#### Methods Section 2.4: GNN Baseline Prediction
**Word Target**: 400 words

**Action Items**:
- [ ] Describe graph construction (patient similarity)
- [ ] Detail GNN architecture (GAT layers, dimensions)
- [ ] Explain training procedure
- [ ] Describe cross-validation approach

#### Methods Section 2.5: Statistical Analysis
**Word Target**: 200 words

**Action Items**:
- [ ] List all statistical tests used
- [ ] Describe significance thresholds
- [ ] Detail multiple comparison corrections
- [ ] Software packages and versions

### December Week 1-2: Results Section

#### Results Section 3.1: Cohort Characteristics
**Word Target**: 300 words

**Action Items**:
- [ ] Describe final cohort (Table 1)
- [ ] Report visit distribution
- [ ] Describe longitudinal follow-up
- [ ] Reference Figure 1 (flowchart)

#### Results Section 3.2: Trajectory Alignment
**Word Target**: 300 words

**Action Items**:
- [ ] Report LTJMM convergence and performance
- [ ] Describe alignment quality (R² from latent_time_model_report.json)
- [ ] Reference Figure 2 (alignment visualization)

#### Results Section 3.3: Subtype Discovery
**Word Target**: 400 words

**Action Items**:
- [ ] Report number of subtypes identified
- [ ] Present clustering metrics (Table 2)
- [ ] Describe trajectory patterns (fast, moderate, slow progressors)
- [ ] Reference Figure 3 (clustering visualization)

#### Results Section 3.4: Subtype Characterization
**Word Target**: 500 words

**Action Items**:
- [ ] Present clinical profiles (Table 3)
- [ ] Report statistical comparisons between subtypes
- [ ] Describe demographic and clinical differences
- [ ] Reference Figure 4 (subtype profiles)

#### Results Section 3.5: Baseline Prediction
**Word Target**: 300 words

**Action Items**:
- [ ] Report prediction performance (Table 4)
- [ ] Compare models (GNN vs traditional ML)
- [ ] Describe feature importance
- [ ] Reference Figure 5 (ROC curves)

#### Results Section 3.6: Trial Enrichment
**Word Target**: 300 words

**Action Items**:
- [ ] Report sample size reduction (Table 5)
- [ ] Present power calculations
- [ ] Estimate cost savings
- [ ] Reference Figure 6 (enrichment simulation)

### December Week 3-4: Abstract, Introduction, Discussion

#### Abstract
**Word Target**: 250 words

**Structure**:
- Background (50 words): PD heterogeneity, need for subtyping
- Methods (80 words): LTJMM, VaDER, GNN, trial simulation
- Results (80 words): Key numbers (n_subtypes, AUC, sample size reduction)
- Conclusion (40 words): Clinical implications

**Action Items**:
- [ ] Extract key numbers from Tables 2-5
- [ ] Write each section according to word target
- [ ] Ensure clear, impactful statements

#### Introduction
**Word Target**: 800-1000 words

**Structure**:
- Paragraph 1 (200w): PD heterogeneity problem
- Paragraph 2 (200w): Previous subtyping approaches and limitations
- Paragraph 3 (200w): Latent time modeling and trajectory analysis
- Paragraph 4 (200w): Clinical trial implications
- Paragraph 5 (200w): Study objectives and innovation

**Action Items**:
- [ ] Literature search for PD subtyping studies
- [ ] Cite key papers on LTJMM, trajectory clustering
- [ ] Emphasize clinical trial enrichment novelty
- [ ] Clear statement of objectives

#### Discussion
**Word Target**: 1200-1500 words

**Structure**:
- Summary of key findings (200w)
- Clinical implications of subtypes (300w)
- Comparison to previous studies (300w)
- Trial design implications (200w)
- Limitations (300w)
- Future directions (200w)

**Action Items**:
- [ ] Interpret each major finding
- [ ] Compare to literature (cite specific studies)
- [ ] Address limitations honestly
- [ ] Propose next steps (external validation, prospective trials)

### January 1-7: Finalization

**Action Items**:
- [ ] Compile all sections into single document
- [ ] Format according to npj Parkinson's Disease guidelines
- [ ] Prepare all 6 main figures (300 DPI)
- [ ] Create supplementary materials
- [ ] Write cover letter
- [ ] **SUBMIT by January 7, 2026**

---

## 🧬 Phase 5 Manuscript (Due: End of February 2026)

### January Week 1-2: Survival Analysis Extraction

#### Task 2.1: Generate Kaplan-Meier Curves
```python
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Load survival data
df = pd.read_csv('data/prodromal_cohort/prodromal_survival_data.csv')

# Fit KM curve
kmf = KaplanMeierFitter()
kmf.fit(df['time'], df['event'], label='Overall')

# Plot
ax = kmf.plot()
plt.title('Prodromal to Clinical PD Conversion')
plt.xlabel('Time (months)')
plt.ylabel('Conversion-free probability')
plt.savefig('figures/phase5_km_curve_overall.png', dpi=300)
```

**Action Items**:
- [ ] Generate overall KM curve
- [ ] Generate risk-stratified KM curves
- [ ] Calculate median survival times
- [ ] Perform log-rank tests between groups
- [ ] Save all curves as 300 DPI PNGs

#### Task 2.2: Extract Cox Model Results
**Action Items**:
- [ ] Load `cox_model_results.json`
- [ ] Extract C-indices with 95% CIs
- [ ] Get hazard ratios for significant predictors
- [ ] Generate Table 3: Cox Model Performance
- [ ] Create forest plot of hazard ratios

#### Task 2.3: Extract DeepSurv Results
**Action Items**:
- [ ] Load `deepsurv_results.json`
- [ ] Extract C-index and integrated Brier score
- [ ] Get training history for learning curves
- [ ] Generate calibration curves
- [ ] Generate Table 4: DeepSurv Performance

#### Task 2.4: Extract Biomarker Thresholds
**Action Items**:
- [ ] Load `biomarker_thresholds.json`
- [ ] Extract cutpoints with sensitivity/specificity
- [ ] Get Youden indices
- [ ] Generate Table 5: Biomarker Thresholds
- [ ] Create threshold visualization plots

### January Week 3-4: Methods & Results Drafting

#### Methods Section (Comprehensive)
**Total Word Target**: 2000-2500 words

**Sections**:
1. Prodromal Cohort Definition (400w)
2. Time-Varying Biomarker Extraction (400w)
3. Cox Proportional Hazards Models (500w)
4. DeepSurv Neural Survival Model (500w)
5. Biomarker Threshold Identification (400w)
6. Statistical Analyses (300w)

**Action Items per section**:
- [ ] Draft each section using implementation code as reference
- [ ] Include mathematical formulations where appropriate
- [ ] Reference specific Python packages used
- [ ] Describe validation procedures

#### Results Section (Comprehensive)
**Total Word Target**: 2000-2500 words

**Sections**:
1. Prodromal Cohort Characteristics (400w, Table 1)
2. Conversion Events (300w, Table 2)
3. Cox Model Performance (500w, Table 3)
4. DeepSurv Performance (500w, Table 4)
5. Biomarker Thresholds (500w, Table 5)
6. Risk Stratification Validation (300w, Table 6)

**Action Items per section**:
- [ ] Write results referring to specific tables
- [ ] Include all statistical test results with p-values
- [ ] Reference figures appropriately
- [ ] Report confidence intervals for all estimates

### February Week 1-2: Abstract, Introduction, Discussion

#### Abstract (Lancet Format)
**Word Target**: 300 words

**Structure** (Lancet Neurology format):
- **Background** (75w): Prodromal PD, need for conversion prediction
- **Methods** (100w): Cohort, survival analysis approaches, validation
- **Findings** (75w): Key results with specific numbers
- **Interpretation** (50w): Clinical implications

**Action Items**:
- [ ] Extract key statistics from all tables
- [ ] Write according to Lancet structured abstract format
- [ ] Emphasize clinical actionability

#### Introduction
**Word Target**: 1000-1200 words

**Structure**:
- Paragraph 1 (250w): Prodromal PD definition and prevalence
- Paragraph 2 (250w): Importance of conversion prediction
- Paragraph 3 (250w): Current approaches and limitations
- Paragraph 4 (250w): Study objectives, innovation, and expected impact

**Action Items**:
- [ ] Comprehensive literature review of prodromal PD studies
- [ ] Cite key MDS Research Criteria papers
- [ ] Emphasize survival analysis innovation
- [ ] State clear, specific aims

#### Discussion
**Word Target**: 1500-1800 words

**Structure**:
- Summary of findings (250w)
- Clinical implications (400w)
- Comparison to existing tools (400w)
- Integration into practice (300w)
- Limitations (350w)
- Future directions (300w)

**Action Items**:
- [ ] Interpret survival curves clinically
- [ ] Discuss threshold selection rationale
- [ ] Compare C-indices to literature benchmarks
- [ ] Address DeepSurv vs Cox tradeoffs
- [ ] Propose prospective validation study
- [ ] Discuss early intervention trial implications

### February Week 3-4: Finalization & Submission

**Action Items**:
- [ ] Compile complete manuscript
- [ ] Format for Lancet Neurology guidelines
- [ ] Prepare 6 main figures (all 300 DPI)
- [ ] Create comprehensive supplementary materials
- [ ] Prepare model architecture diagram
- [ ] Write detailed cover letter
- [ ] Complete author contribution statements
- [ ] Verify all references formatted correctly
- [ ] **SUBMIT by February 28, 2026**

---

## ✅ Cross-Manuscript Tasks

### Reproducibility Package (November-December)

**Action Items**:
- [ ] Create GitHub repository with all code
- [ ] Write comprehensive README
- [ ] Create `environment.yml` for conda
- [ ] Create `requirements.txt` for pip
- [ ] Document all preprocessing steps
- [ ] Create tutorial Jupyter notebooks
- [ ] Register on Open Science Framework
- [ ] Request Zenodo DOI

### Data Sharing Compliance (December-January)

**Action Items**:
- [ ] Verify IRB approval for data sharing
- [ ] Confirm PPMI data use agreement compliance
- [ ] Prepare data sharing statement for each manuscript
- [ ] Create data access instructions
- [ ] Prepare de-identified datasets (if allowed)

### Preprint Posting (As manuscripts are completed)

**Action Items**:
- [ ] Post Phase 6 preprint to bioRxiv (immediately after submission)
- [ ] Post Phase 4 preprint to medRxiv (January 2026)
- [ ] Post Phase 5 preprint to medRxiv (February 2026)
- [ ] Create Twitter threads for each preprint
- [ ] Share on LinkedIn, ResearchGate
- [ ] Notify collaborators and PPMI community

---

## 📅 Master Timeline Summary

| Date Range | Milestone | Status |
|------------|-----------|--------|
| Oct 6-12 | Phase 6 Results & Methods | 🟡 In Progress |
| Oct 13-19 | Phase 6 Abstract & Discussion | ⏳ Upcoming |
| Oct 20-26 | Phase 6 Internal Review | ⏳ Upcoming |
| Oct 27-31 | **Phase 6 SUBMISSION** | ⏳ Upcoming |
| Nov 1-14 | Phase 4 Data Extraction | ⏳ Upcoming |
| Nov 15-30 | Phase 4 Methods & Results | ⏳ Upcoming |
| Dec 1-31 | Phase 4 Complete & SUBMIT | ⏳ Upcoming |
| Jan 1-14 | Phase 5 Data Extraction | ⏳ Upcoming |
| Jan 15-31 | Phase 5 Methods & Results | ⏳ Upcoming |
| Feb 1-28 | Phase 5 Complete & SUBMIT | ⏳ Upcoming |

---

## 💡 Pro Tips

### Writing Efficiency
1. **Use templates**: Create section templates based on journal requirements
2. **Extract first, write second**: Get all metrics into tables before writing
3. **Reference as you write**: Add citations immediately, don't leave for later
4. **Version control**: Use Git or track changes for all drafts

### Figure Preparation
1. **Create figure panels early**: Assemble multi-panel figures first
2. **Consistent styling**: Use same color schemes across all figures
3. **High resolution**: Always save as 300+ DPI, even during drafting
4. **Legend completeness**: Write full legends with all abbreviations defined

### Co-author Management
1. **Early engagement**: Share preliminary results as soon as available
2. **Clear deadlines**: Give co-authors specific review deadlines
3. **Track changes**: Use tools that show who contributed what
4. **Authorship order**: Discuss and agree on authorship early

### Statistical Rigor
1. **Report everything**: Include all statistical tests, even non-significant
2. **Effect sizes matter**: Report effect sizes, not just p-values
3. **Confidence intervals**: Always report CIs with point estimates
4. **Multiple comparisons**: Document correction methods used

---

**Document Purpose**: Step-by-step execution guide for manuscript preparation  
**Last Updated**: October 5, 2025  
**Owner**: [Your Name]  
**Next Update**: Weekly during active writing phases
