# Supplementary Table S1: Search Strategies for Each Database

## Complete Search Strings and Filters Applied

This table provides the complete, reproducible search strategies used for each database in the systematic review. All searches were conducted between January 1, 2018 and January 20, 2026.

---

| **Database** | **Search String** | **Filters Applied** | **Date Range** | **Results Retrieved** |
|--------------|-------------------|---------------------|----------------|----------------------|
| **SciSpace** | ("Parkinson's disease" OR "Parkinson disease" OR "PD") AND ("digital twin" OR "digital twins" OR "mechanistic model" OR "mechanistic modeling" OR "physics-informed" OR "physiological model" OR "computational model" OR "dynamic model" OR "temporal model" OR "longitudinal model" OR "time-series model" OR "progression model" OR "trajectory model") AND ("prognosis" OR "prognostic" OR "prediction" OR "forecasting" OR "progression" OR "outcome" OR "treatment response") | • Publication date: 2018-2026<br>• Language: English<br>• Document type: Journal articles, preprints, conference papers | January 1, 2018 - January 20, 2026 | 87 |
| **PubMed** | (("Parkinson Disease"[Mesh] OR "Parkinson's disease"[tiab] OR "Parkinson disease"[tiab] OR "PD"[tiab]) AND ("digital twin"[tiab] OR "digital twins"[tiab] OR "mechanistic model"[tiab] OR "mechanistic modeling"[tiab] OR "physics-informed"[tiab] OR "physiological model"[tiab] OR "computational model"[tiab] OR "dynamic model"[tiab] OR "temporal model"[tiab] OR "longitudinal model"[tiab] OR "time-series"[tiab] OR "progression model"[tiab] OR "trajectory model"[tiab] OR "machine learning"[tiab] OR "deep learning"[tiab] OR "artificial intelligence"[tiab]) AND ("prognosis"[tiab] OR "prognostic"[tiab] OR "prediction"[tiab] OR "forecasting"[tiab] OR "progression"[tiab] OR "outcome"[tiab] OR "treatment response"[tiab])) | • Publication date: 2018/01/01 - 2026/01/20<br>• Language: English<br>• Species: Humans<br>• Article types: Journal Article, Clinical Trial, Observational Study | January 1, 2018 - January 20, 2026 | 124 |
| **Google Scholar** | "Parkinson's disease" OR "Parkinson disease" ("digital twin" OR "mechanistic model" OR "dynamic model" OR "temporal model" OR "machine learning" OR "deep learning") (prognosis OR prediction OR progression OR forecasting) -diagnosis -detection -classification | • Date range: 2018-2026<br>• Sorted by relevance<br>• Excluded patents<br>• First 100 results reviewed | January 1, 2018 - January 20, 2026 | 100 (top results) |
| **ArXiv** | all:("Parkinson's disease" OR "Parkinson disease") AND all:("digital twin" OR "mechanistic model" OR "dynamic model" OR "temporal model" OR "progression model" OR "machine learning" OR "deep learning") AND all:(prognosis OR prediction OR progression OR forecasting) | • Submission date: 2018-01-01 to 2026-01-20<br>• Categories: cs.LG, cs.AI, q-bio.QM, stat.ML<br>• Document type: preprints | January 1, 2018 - January 20, 2026 | 43 |

---

## Search Strategy Notes

### Boolean Operators and Field Tags

- **AND**: Required all terms to be present
- **OR**: At least one term from the group must be present
- **[Mesh]**: Medical Subject Heading (PubMed only)
- **[tiab]**: Title or abstract (PubMed only)
- **all:**: All fields (ArXiv only)

### Inclusion of Broader Terms

The search strategy intentionally included broader terms ("machine learning", "deep learning", "artificial intelligence") alongside specific terms ("digital twin", "mechanistic model") to ensure comprehensive capture of relevant studies. This approach was necessary because:

1. Many dynamic/temporal models do not explicitly use the term "digital twin"
2. Mechanistic modeling approaches may be described using various terminologies
3. The field lacks standardized nomenclature for prognostic modeling approaches

### Deduplication Process

After retrieving results from all four databases:
- **Total retrieved**: 354 records
- **After deduplication**: 287 unique papers
- **Deduplication method**: DOI matching, followed by title and author matching for records without DOIs
- **Deduplication tool**: Custom Python script using fuzzy string matching (threshold: 90% similarity)

### Search Date and Update Strategy

- **Initial search date**: January 15-20, 2026
- **Search update**: Not applicable (single time-point search)
- **Database versions**: 
  - PubMed: Accessed January 18, 2026
  - SciSpace: Accessed January 17, 2026
  - Google Scholar: Accessed January 19, 2026
  - ArXiv: Accessed January 16, 2026

### Limitations of Search Strategy

1. **Language restriction**: English-only papers may have excluded relevant non-English studies
2. **Database coverage**: Did not search Embase, Web of Science, or Cochrane Library due to access limitations
3. **Grey literature**: Limited capture of conference abstracts, dissertations, and unpublished studies beyond ArXiv
4. **Publication bias**: Positive results more likely to be published and indexed

### Reproducibility Statement

All search strings are provided exactly as executed. The search can be reproduced by:
1. Accessing each database on the specified dates (or later, with appropriate date filters)
2. Copying the exact search strings provided
3. Applying the filters as specified
4. Following the deduplication protocol described above

---

**Corresponding Author Contact**: For questions about search strategy or to request search results files, please contact the corresponding author.

**Search Protocol Registration**: This systematic review was not pre-registered. The search strategy was developed iteratively based on pilot searches and expert consultation.

**Date of Table Preparation**: January 21, 2026
