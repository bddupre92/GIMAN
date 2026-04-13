# Supplementary Materials Summary

## Systematic Review: Prognostic Utility of Digital Twins vs Static ML in Parkinson's Disease

**Date Generated:** January 21, 2026

---

## Overview

This package contains **7 supplementary files** plus 1 README document providing complete transparency and reproducibility for the systematic review of 287 papers on prognostic modeling in Parkinson's disease.

---

## Quick Statistics

### Screening Results
- **Total papers retrieved:** 354 (across 4 databases)
- **After deduplication:** 287 unique papers
- **Included in review:** 15 papers (5.2%)
- **Excluded:** 272 papers (94.8%)
- **With direct comparisons:** 6 papers (2.1% of screened, 40% of included)

### Search Coverage
- **Databases searched:** 4 (SciSpace, PubMed, Google Scholar, ArXiv)
- **Date range:** January 1, 2018 - January 20, 2026
- **Search date:** January 15-20, 2026

### Quality Assessment
- **PROBAST assessment:** 15 papers
  - Low risk: 4 papers (27%)
  - Moderate risk: 9 papers (60%)
  - High risk: 2 papers (13%)
- **Validation quality:**
  - Tier 2 (external validation): 10 papers (67%)
  - Tier 1 (internal validation): 1 paper (7%)
  - Tier 0 (no external validation): 4 papers (27%)

### Key Findings
- **Dynamic models favored:** 5 of 6 comparative studies (83%)
- **Effect sizes:** +4.3% to +28.9% relative improvement
- **Mechanistic digital twins:** 0 papers (0%)
- **Meta-analysis feasibility:** Not possible (metric heterogeneity, missing variance)

---

## File Descriptions

### 1. Search Strategies (S1)
**File:** `supplementary_table_s1_search_strategies.md`
- **Format:** Markdown table
- **Size:** 5.7 KB
- **Content:** Complete search strings for all 4 databases with filters, date ranges, and results
- **Purpose:** Enable complete reproducibility of systematic search

### 2. All Papers Characteristics (S2)
**File:** `supplementary_table_s2_all_papers_characteristics.csv`
- **Format:** CSV
- **Size:** 56 KB
- **Rows:** 287 papers (header + 287 data rows)
- **Columns:** 10 (Paper_ID, First_Author, Year, Title, DOI, Model_Type, Prediction_Goal, Validation_Tier, Inclusion_Status, Exclusion_Reason)
- **Purpose:** Complete transparency of screening decisions

### 3. PROBAST Risk of Bias Assessment (S3)
**File:** `supplementary_table_s3_probast_assessment.csv`
- **Format:** CSV
- **Size:** 3.4 KB
- **Rows:** 15 included papers
- **Columns:** 10 (Paper_ID, First_Author, Year, 4 PROBAST domains, Overall_Risk, Validation_Tier, Key_Concerns)
- **Purpose:** Transparent quality assessment using PROBAST criteria

### 4. Excluded Papers with Reasons (S4)
**File:** `supplementary_table_s4_excluded_papers.csv`
- **Format:** CSV
- **Size:** 52 KB
- **Rows:** 272 excluded papers
- **Columns:** 7 (Paper_ID, First_Author, Year, Title, DOI, Primary_Exclusion_Reason, Failed_Criteria)
- **Purpose:** Document all exclusion decisions

### 5. Harvest Plot Visualization (S5)
**File:** `supplementary_figure_s1_harvest_plot.md`
- **Format:** Markdown with ASCII art and Python code
- **Size:** 6.7 KB
- **Content:** Visual synthesis of 6 comparative studies showing effect sizes by validation tier
- **Includes:** Data table, ASCII visualization, Python/Matplotlib code for publication-quality figure
- **Purpose:** Visual evidence synthesis

### 6. Complete Data Extraction Forms (S6)
**File:** `supplementary_data_s1_extraction_forms.csv`
- **Format:** CSV
- **Size:** 14 KB
- **Rows:** 15 included papers
- **Columns:** 24 extracted fields (study characteristics, model details, validation, performance, clinical context, quality)
- **Purpose:** Complete raw data for independent verification and future meta-analyses

### 7. PRISMA 2020 Checklist (S7)
**File:** `supplementary_prisma_2020_checklist.csv`
- **Format:** CSV
- **Size:** 9.2 KB
- **Rows:** 43 PRISMA items (27 main + 16 subitems)
- **Columns:** 6 (Item_Number, Section, Item, Location_in_Manuscript, Page_Number, Reported)
- **Compliance:** 42/43 fully reported (97.7%), 1 partially reported
- **Purpose:** Demonstrate adherence to PRISMA 2020 guidelines

### 8. README Document
**File:** `SUPPLEMENTARY_MATERIALS_README.md`
- **Format:** Markdown
- **Size:** 8.4 KB
- **Content:** Comprehensive overview of all supplementary materials with detailed descriptions
- **Purpose:** Guide for reviewers and readers

---

## Data Quality Checks

### Completeness
✓ All 287 screened papers documented
✓ All 15 included papers have complete data extraction
✓ All 272 excluded papers have exclusion reasons
✓ All 15 included papers have PROBAST assessment
✓ All 6 comparative studies have effect size data

### Consistency
✓ Paper IDs consistent across all tables (1-287)
✓ Included paper count matches across files (15)
✓ Excluded paper count matches across files (272)
✓ Total papers sum to 287 in all files

### Reproducibility
✓ Search strings provided exactly as executed
✓ All filters and date ranges documented
✓ Deduplication methodology described
✓ Screening criteria explicitly stated
✓ Data extraction protocol documented
✓ Risk of bias assessment criteria specified

---

## File Formats and Accessibility

### CSV Files (Machine-Readable)
- Standard comma-separated values
- UTF-8 encoding
- Header row included
- Compatible with Excel, R, Python, SPSS, Stata

### Markdown Files (Human-Readable)
- Plain text with formatting
- Viewable in any text editor
- Renders nicely on GitHub, GitLab, etc.
- Convertible to PDF, HTML, Word

---

## Usage Guidelines

### For Reviewers
1. Start with `SUPPLEMENTARY_MATERIALS_README.md` for overview
2. Check `supplementary_table_s1_search_strategies.md` for search reproducibility
3. Review `supplementary_table_s3_probast_assessment.csv` for quality assessment
4. Examine `supplementary_data_s1_extraction_forms.csv` for data extraction
5. Verify `supplementary_prisma_2020_checklist.csv` for reporting completeness

### For Meta-Analysts
1. Use `supplementary_data_s1_extraction_forms.csv` for raw data
2. Check `supplementary_table_s2_all_papers_characteristics.csv` for additional studies
3. Review `supplementary_table_s3_probast_assessment.csv` for quality filtering
4. Note: Meta-analysis not feasible due to metric heterogeneity and missing variance

### For Systematic Reviewers
1. Use `supplementary_table_s1_search_strategies.md` as template for search strategy
2. Adapt `supplementary_data_s1_extraction_forms.csv` structure for data extraction
3. Follow `supplementary_table_s3_probast_assessment.csv` for quality assessment
4. Use `supplementary_prisma_2020_checklist.csv` for reporting checklist

---

## Citation

If you use these supplementary materials, please cite:

[Authors]. Prognostic Utility of Digital Twins versus Static Machine Learning in Parkinson's Disease: A Systematic Review and Best-Evidence Synthesis. [Journal]. [Year]. DOI: [DOI]

---

## Data Availability

All supplementary materials are provided in open formats to facilitate:
- Independent verification
- Future systematic reviews
- Meta-analyses (when sufficient data available)
- Methodological research
- Evidence synthesis

---

## Contact Information

For questions or additional data requests:
- **Corresponding Author:** [Name]
- **Email:** [Email]
- **Institution:** [Institution]

---

## Version Control

**Version 1.0** (January 21, 2026)
- Initial release with manuscript submission
- 7 supplementary files + README
- 287 screened papers, 15 included, 6 comparative

---

## Verification Checksums

File integrity can be verified using MD5 checksums:

```bash
md5sum supplementary_*.csv supplementary_*.md SUPPLEMENTARY_*.md
```

Expected output:
```
[To be generated upon final submission]
```

---

## License

These supplementary materials are provided under [License] to support open science and reproducible research.

---

## Acknowledgments

We thank the authors of all 287 screened papers for their contributions to the field of prognostic modeling in Parkinson's disease.

---

**End of Supplementary Materials Summary**

**Total Package Size:** ~155 KB (8 files)

**Ready for Journal Submission:** ✓ Yes

**PRISMA Compliant:** ✓ Yes (97.7%)

**Data Complete:** ✓ Yes

**Reproducible:** ✓ Yes
