# Supplementary Materials for Systematic Review

## Overview

This directory contains all supplementary materials for the systematic review:

**"Prognostic Utility of Digital Twins versus Static Machine Learning in Parkinson's Disease: A Systematic Review and Best-Evidence Synthesis"**

---

## File Inventory

### 1. Supplementary Table S1: Search Strategies for Each Database
**File:** `supplementary_table_s1_search_strategies.md`

**Format:** Markdown table

**Content:**
- Complete search strings for all 4 databases (SciSpace, PubMed, Google Scholar, ArXiv)
- Filters applied to each search
- Date ranges (January 1, 2018 - January 20, 2026)
- Number of results retrieved per database
- Boolean operators and field tags used
- Deduplication methodology
- Reproducibility notes

**Purpose:** Enables complete reproducibility of the systematic search strategy

---

### 2. Supplementary Table S2: Characteristics of All 287 Screened Papers
**File:** `supplementary_table_s2_all_papers_characteristics.csv`

**Format:** CSV (comma-separated values)

**Columns:**
- `Paper_ID`: Sequential identifier (1-287)
- `First_Author`: First author surname
- `Year`: Publication year
- `Title`: Full paper title
- `DOI`: Digital Object Identifier
- `Model_Type`: Model category (Dynamic/Time-Series, Static ML, Mechanistic Digital Twin, etc.)
- `Prediction_Goal`: Primary prediction target (Progression Forecasting, Fall Prediction, Treatment Response, etc.)
- `Validation_Tier`: Validation quality level (Tier 0, 1, or 2)
- `Inclusion_Status`: INCLUDED or EXCLUDED
- `Exclusion_Reason`: Primary reason for exclusion (for excluded papers only)

**Purpose:** Provides complete transparency of screening decisions for all 287 unique papers

**Statistics:**
- Total papers: 287
- Included: 15 (5.2%)
- Excluded: 272 (94.8%)

---

### 3. Supplementary Table S3: Risk of Bias Assessment Details (PROBAST)
**File:** `supplementary_table_s3_probast_assessment.csv`

**Format:** CSV (comma-separated values)

**Columns:**
- `Paper_ID`: Links to Supplementary Table S2
- `First_Author`: First author surname
- `Year`: Publication year
- `Domain_1_Participants`: Risk of bias for participant selection (Low/Moderate/High)
- `Domain_2_Predictors`: Risk of bias for predictor assessment (Low/Moderate/High)
- `Domain_3_Outcome`: Risk of bias for outcome assessment (Low/Moderate/High)
- `Domain_4_Analysis`: Risk of bias for statistical analysis (Low/Moderate/High)
- `Overall_Risk`: Overall risk of bias rating (Low/Moderate/High)
- `Validation_Tier`: Validation quality (Tier 0/1/2)
- `Key_Concerns`: Specific methodological concerns and limitations

**Purpose:** Transparent risk of bias assessment using PROBAST criteria for all 15 included studies

**Assessment Framework:** Prediction model Risk Of Bias ASsessment Tool (PROBAST)

---

### 4. Supplementary Table S4: Excluded Studies with Reasons (n=272)
**File:** `supplementary_table_s4_excluded_papers.csv`

**Format:** CSV (comma-separated values)

**Columns:**
- `Paper_ID`: Sequential identifier (1-287, excluding included papers)
- `First_Author`: First author surname
- `Year`: Publication year
- `Title`: Full paper title
- `DOI`: Digital Object Identifier
- `Primary_Exclusion_Reason`: Main reason for exclusion
- `Failed_Criteria`: Specific inclusion criteria not met

**Exclusion Categories:**
1. Diagnostic only - no prognostic endpoint (Criterion 4)
2. No comparator or baseline comparison (Criterion 3)
3. Static ML only - no temporal/dynamic component (Criterion 2)
4. Review/commentary - not original research (Criterion 5)
5. No real patient data (Criteria 1 & 5)
6. Does not predict future clinical state (Criterion 4)
7. Multiple inclusion criteria not met (2+ criteria)

**Purpose:** Complete documentation of exclusion decisions for transparency and reproducibility

---

### 5. Supplementary Figure S1: Harvest Plot Visualization
**File:** `supplementary_figure_s1_harvest_plot.md`

**Format:** Markdown with ASCII visualization and Python code

**Content:**
- Harvest plot showing direction and magnitude of effect for 6 comparative studies
- X-axis: Effect size (relative improvement %)
- Y-axis: Validation tier (0, 1, 2)
- Symbols: Different prediction goals (Progression, Falls, Cognitive)
- Data table with exact coordinates
- Python/Matplotlib code for publication-quality figure generation

**Key Findings:**
- 5 of 6 studies (83%) favored dynamic models
- Effect sizes: +4.3% to +28.9% relative improvement
- 1 study showed slight underperformance (-2.3%)

**Purpose:** Visual synthesis of comparative evidence showing heterogeneity in outcomes and validation quality

---

### 6. Supplementary Data S1: Complete Data Extraction Forms
**File:** `supplementary_data_s1_extraction_forms.csv`

**Format:** CSV (comma-separated values)

**Content:** Complete data extraction for all 15 included papers with 24 extracted fields:

**Study Characteristics:**
- Rank, Title, Authors, Year, Journal, DOI, Citations

**Model Characteristics:**
- Model Type, Prediction Goal, Input Modality, Mechanistic Component

**Validation Methodology:**
- Validation Method, Validation Tier, Comparator Baseline

**Performance Metrics:**
- Primary Metric Type, Intervention Performance Score, Comparator Performance Score
- Intervention 95% CI, Comparator 95% CI, Statistical Significance (p-value)

**Clinical Context:**
- Temporal Horizon, Disease Stage, Medication Status, Test Sample Size

**Quality Assessment:**
- Key Limitations, Direct Model Comparison Present

**Purpose:** Provides complete raw data for all included studies to enable independent verification and future meta-analyses

---

### 7. PRISMA 2020 Checklist Completed
**File:** `supplementary_prisma_2020_checklist.csv`

**Format:** CSV (comma-separated values)

**Columns:**
- `Item_Number`: PRISMA item number (1-27 with subitems)
- `Section`: PRISMA section (Title, Abstract, Methods, Results, Discussion, Other)
- `Item`: Full description of reporting item
- `Location_in_Manuscript`: Where the item is addressed (section number)
- `Page_Number`: Page number in manuscript
- `Reported`: Yes/No/Partial/NA

**Content:** Complete PRISMA 2020 checklist with 27 main items and 14 subitems (41 total items)

**Purpose:** Demonstrates adherence to PRISMA 2020 reporting guidelines for systematic reviews

**Compliance:** 40/41 items fully reported (97.6%); 1 item partially reported (95% CIs not available for all intervention models)

---

## Data Availability and Reproducibility

### Source Data Files
All supplementary materials were generated from:
1. `deduplicated_parkinsons_papers.papertable` (287 screened papers)
2. `systematic_review_final_table.csv` (15 included papers with full extraction)
3. `meta_analysis_ready_papers_all.csv` (6 comparative papers)
4. `included_paper_indices.json` (indices of 15 included papers)

### Reproducibility
- All search strings are provided exactly as executed
- All screening decisions are documented with reasons
- All data extraction is available in structured format
- All risk of bias assessments are transparent
- Python code is provided for figure generation

### Data Sharing
All supplementary materials are provided in open, machine-readable formats (CSV, Markdown) to facilitate:
- Independent verification of results
- Future systematic reviews and meta-analyses
- Integration with other evidence synthesis efforts
- Methodological research on prognostic modeling

---

## Citation

If you use these supplementary materials, please cite the main manuscript:

[Authors]. Prognostic Utility of Digital Twins versus Static Machine Learning in Parkinson's Disease: A Systematic Review and Best-Evidence Synthesis. [Journal]. [Year]. DOI: [DOI]

---

## Contact

For questions about the supplementary materials or to request additional data, please contact:

[Corresponding Author Name]
[Email]
[Institution]

---

## Version History

- **Version 1.0** (January 21, 2026): Initial release with manuscript submission
  - 7 supplementary files
  - 287 screened papers
  - 15 included papers
  - 6 comparative studies
  - Complete PROBAST and PRISMA assessments

---

## File Checksums (MD5)

For verification of file integrity:

```
[To be generated upon final submission]
```

---

## License

These supplementary materials are provided under [License Type] to facilitate open science and reproducible research.

---

**Date of Preparation:** January 21, 2026

**Systematic Review Registration:** Not pre-registered

**Protocol Availability:** Protocol available upon request from corresponding author
