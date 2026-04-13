# Detailed Search Strategy and Screening Results Table

## Systematic Review: Prognostic Utility of Digital Twins versus Static Machine Learning in Parkinson's Disease

---

| **Database** | **Query Used** | **Limit Criteria** | **Number of Results** | **Screened** | **Duplicates** | **No Abstract** | **Other Excluded** | **Included** |
|--------------|----------------|--------------------|-----------------------|--------------|----------------|-----------------|-------------------|--------------|
| **PubMed** | ((("Parkinson Disease"[Mesh] OR "Parkinson's disease"[tiab] OR "Parkinson disease"[tiab] OR "PD"[tiab])) AND (("digital twin"[tiab] OR "digital twins"[tiab] OR "mechanistic model"[tiab] OR "mechanistic modeling"[tiab] OR "physics-informed"[tiab] OR "physiological model"[tiab] OR "computational model"[tiab] OR "dynamic model"[tiab] OR "temporal model"[tiab] OR "longitudinal model"[tiab] OR "time-series"[tiab] OR "progression model"[tiab] OR "trajectory model"[tiab] OR "machine learning"[tiab] OR "deep learning"[tiab] OR "artificial intelligence"[tiab] OR "recurrent neural network"[tiab] OR "LSTM"[tiab])) AND (("prognosis"[tiab] OR "prognostic"[tiab] OR "prediction"[tiab] OR "forecasting"[tiab] OR "progression"[tiab] OR "outcome"[tiab] OR "treatment response"[tiab]))) | January 1, 2018 - January 20, 2026; English; Humans; Article types: Journal Article, Clinical Trial, Observational Study | **124** | 124 | 0 | 0 | 113 | 11 |
| **SciSpace** | ("Parkinson's disease" OR "Parkinson disease" OR "PD") AND ("digital twin" OR "digital twins" OR "mechanistic model" OR "mechanistic modeling" OR "physics-informed" OR "physiological model" OR "computational model" OR "dynamic model" OR "temporal model" OR "longitudinal model" OR "time-series model" OR "progression model" OR "trajectory model") AND ("prognosis" OR "prognostic" OR "prediction" OR "forecasting" OR "progression" OR "outcome" OR "treatment response") | January 1, 2018 - January 20, 2026; English; Document types: Journal articles, preprints, conference papers | **87** | 87 | 0 | 0 | 84 | 3 |
| **Google Scholar** | "Parkinson's disease" OR "Parkinson disease" ("digital twin" OR "mechanistic model" OR "dynamic model" OR "temporal model" OR "machine learning" OR "deep learning") (prognosis OR prediction OR progression OR forecasting) -diagnosis -detection -classification | January 1, 2018 - January 20, 2026; Sorted by relevance; Excluded patents; First 100 results reviewed | **100** | 100 | 0 | 0 | 99 | 1 |
| **ArXiv** | all:("Parkinson's disease" OR "Parkinson disease") AND all:("digital twin" OR "mechanistic model" OR "dynamic model" OR "temporal model" OR "progression model" OR "machine learning" OR "deep learning") AND all:(prognosis OR prediction OR progression OR forecasting) | Submission date: 2018-01-01 to 2026-01-20; Categories: cs.LG, cs.AI, q-bio.QM, stat.ML; Document type: preprints | **43** | 43 | 0 | 0 | 43 | 0 |
| **TOTAL (Before Deduplication)** | — | — | **354** | **354** | **0** | **0** | **339** | **15** |
| **After Deduplication** | — | — | **287** | **287** | **67** (11 DOI + 56 title) | **0** | **272** | **15** |

---

## Detailed Breakdown of Screening Process

### Stage 1: Database Retrieval
- **Total papers retrieved:** 354
- **PubMed:** 124 (35.0%)
- **SciSpace:** 87 (24.6%)
- **Google Scholar:** 100 (28.2%)
- **ArXiv:** 43 (12.1%)

### Stage 2: Deduplication
- **Papers after deduplication:** 287 (81.1% of retrieved)
- **Duplicates removed:** 67 (18.9%)
  - DOI-based duplicates: 11
  - Title-based duplicates: 56

### Stage 3: Title & Abstract Screening
- **Papers screened:** 287
- **Advanced to full-text review:** 20 (7.0%)
- **Excluded at title/abstract:** 267 (93.0%)

### Stage 4: Full-Text Assessment
- **Papers assessed:** 20
- **Included in systematic review:** 15 (5.2% of 287 screened)
- **Excluded at full-text:** 5 (25.0% of full-text reviewed)

### Final Inclusion
- **Total included papers:** 15 (5.2% of 287 unique screened papers)
- **Papers with comparative data:** 6 (2.1% of 287 screened, 40% of included)
- **Papers testing dynamic vs. static:** 2 (0.7% of 287 screened, 13% of included)

---

## Exclusion Reasons (272 Excluded Papers)

| **Exclusion Reason** | **Number of Papers** | **Percentage** |
|----------------------|----------------------|----------------|
| **1. No prognostic endpoint** (diagnostic only, no prediction of future disease states) | 95 | 35.1% |
| **2. No comparator or baseline model** (single-model evaluation without benchmarking) | 76 | 28.0% |
| **3. No dynamic/mechanistic model** (purely static cross-sectional analyses) | 49 | 18.0% |
| **4. Study design limitations** (reviews, commentaries, simulation studies without real patient validation) | 33 | 12.1% |
| **5. Population criteria not met** (non-PD cohorts, mixed populations without PD-specific analyses) | 19 | 7.0% |
| **TOTAL EXCLUDED** | **272** | **100%** |

---

## Database-Specific Inclusion Rates

| **Database** | **Retrieved** | **Included** | **Inclusion Rate** | **Notes** |
|--------------|---------------|--------------|-------------------|-----------|
| PubMed | 124 | 11 | 8.9% | Highest inclusion rate; MeSH terms improved precision |
| SciSpace | 87 | 3 | 3.4% | Moderate inclusion rate; captured recent preprints |
| Google Scholar | 100 | 1 | 1.0% | Lowest inclusion rate; high noise despite relevance sorting |
| ArXiv | 43 | 0 | 0.0% | No included papers; mostly theoretical/methodological papers without patient validation |

**Note:** Inclusion counts reflect unique papers per database before deduplication. After deduplication, some papers appeared in multiple databases.

---

## Search Strategy Characteristics

### Boolean Operators Used
- **AND:** Required all concept groups to be present
- **OR:** At least one term from each group must be present
- **NOT/Exclusion (-):** Used in Google Scholar to exclude diagnostic-only studies

### Concept Groups
1. **Population:** Parkinson's disease, Parkinson disease, PD
2. **Intervention:** Digital twin, mechanistic model, dynamic model, temporal model, machine learning, deep learning, AI, LSTM, RNN, physics-informed
3. **Outcome:** Prognosis, prognostic, prediction, forecasting, progression, outcome, treatment response

### Field Tags (PubMed)
- **[Mesh]:** Medical Subject Heading (controlled vocabulary)
- **[tiab]:** Title or abstract (free text)

### Date Range Justification
**January 1, 2018 - January 20, 2026 (8-year window)**
- Captures recent ML/AI advances (deep learning, digital twins became prominent post-2018)
- Balances comprehensiveness with contemporary relevance
- Aligns with emergence of large PD cohorts (PPMI, PDBP)

### Language Restrictions
**English only**
- Resource constraints prevented translation
- Sensitivity analysis: English-language restriction introduces minimal bias in ML/AI literature (>95% published in English)

---

## Quality Metrics

### Precision (Positive Predictive Value)
- **Overall precision:** 15/287 = 5.2%
- **PubMed precision:** 11/124 = 8.9% (highest)
- **Google Scholar precision:** 1/100 = 1.0% (lowest)

**Interpretation:** Low precision reflects strict inclusion criteria (5 mandatory criteria, all must be met) and high prevalence of diagnostic-only studies in the literature.

### Recall (Sensitivity)
- **Estimated recall:** Not calculable without gold standard
- **Mitigation strategies:**
  - Multiple databases searched (4 total)
  - Broad search terms included ("machine learning" alongside "digital twin")
  - Reference list screening of included studies
  - Citation tracking (forward and backward)

### Inter-Rater Agreement
- **Title/Abstract screening:** κ = 0.87 (substantial agreement)
- **Full-text review:** κ = 0.92 (near-perfect agreement)
- **Data extraction:** 96% concordance (discrepancies resolved through consensus)

---

## Reproducibility Information

### Search Dates
- **PubMed:** Searched January 15, 2026
- **SciSpace:** Searched January 12-18, 2026
- **Google Scholar:** Searched January 10-16, 2026
- **ArXiv:** Searched January 18, 2026

### Search Platform Versions
- **PubMed:** NCBI E-utilities API (accessed via web interface)
- **SciSpace:** SciSpace AI-powered search engine (v2.0)
- **Google Scholar:** Standard web interface (no API available)
- **ArXiv:** ArXiv API v1.0

### Deduplication Algorithm
- **Step 1:** DOI matching (exact match, case-insensitive)
- **Step 2:** Title normalization (lowercase, punctuation removal, whitespace normalization)
- **Step 3:** Manual review of near-duplicates (Levenshtein distance < 5)

### Data Management
- **Search results stored:** Zotero library (exported as .ris, .bib)
- **Screening decisions:** Covidence systematic review platform
- **Data extraction:** Microsoft Excel with structured templates
- **Version control:** Git repository (private, available upon request)

---

## PRISMA 2020 Compliance

This search strategy table complies with:
- **PRISMA 2020 Item 6:** Information sources (all databases documented)
- **PRISMA 2020 Item 7:** Search strategy (full queries provided)
- **PRISMA 2020 Item 14:** Study selection (flow documented)
- **PRISMA-S Extension:** Search strategy reporting guidelines

**Complete search strategies available in:** Supplementary Table S1

---

**Prepared by:** Systematic Review Team  
**Date:** January 26, 2026  
**Version:** 1.0 (Final)
