# NPJ Parkinson's Disease - Manuscript Restructuring Plan

## Journal: NPJ Parkinson's Disease (Nature Partner Journal)
**Submission Type:** Article (Systematic Review)  
**Impact Factor:** 7.8  
**Open Access:** Yes

---

## CRITICAL REQUIREMENTS SUMMARY

| Element | NPJ Requirement | Current Status | Action Required |
|---------|----------------|----------------|-----------------|
| **Title** | ≤15 words, no punctuation | 18 words | ✗ REDUCE by 3 words |
| **Abstract** | ≤150 words, no subheadings | 289 words, structured | ✗ REDUCE by 139 words (48% cut) |
| **Introduction** | No subheadings | 4 subsections | ✗ REMOVE all subheadings |
| **Results** | Subheadings required | Has subheadings | ✓ COMPLIANT |
| **Discussion** | No subheadings, no Limitations/Conclusions | 9 subsections + Conclusion | ✗ REMOVE subheadings, MERGE Conclusion |
| **Methods** | Subheadings required | Has subheadings | ✓ COMPLIANT |
| **References** | 60 (flexible to ~80) | 300 | ✗ REDUCE to 60-80 |
| **Figure Legends** | ≤350 words each | TBD | Check after figures created |

---

## STRATEGIC RESTRUCTURING PLAN

### PHASE 1: TITLE OPTIMIZATION (15 words max)

**Current (18 words):**
> "Prognostic Utility of Digital Twins versus Static Machine Learning in Parkinson's Disease: A Systematic Review and Best-Evidence Synthesis"

**Option 1 (14 words):**
> "Digital Twins versus Static Machine Learning for Parkinson's Disease Prognosis: A Systematic Review"

**Option 2 (13 words):**
> "Benchmarking Dynamic Models Against Static Machine Learning in Parkinson's Disease Prognosis"

**Option 3 (15 words):**
> "Prognostic Performance of Digital Twins Compared to Static Machine Learning in Parkinson's Disease: Systematic Review"

**RECOMMENDED: Option 1** (clear, concise, includes key terms)

---

### PHASE 2: ABSTRACT CONDENSATION (150 words max)

**Current:** 289 words (structured: Background, Objective, Methods, Results, Conclusions)  
**Target:** 150 words (unstructured, single paragraph)  
**Reduction needed:** 48%

**Strategy:**
- Remove structure labels (Background, Objective, etc.)
- Condense to 3-4 sentences:
  1. **Sentence 1 (30 words):** Rationale - Digital twins proposed for PD prognosis but lack empirical benchmarking
  2. **Sentence 2 (40 words):** Methods - Systematic review, 4 databases, 287 papers screened, 15 included, PROBAST assessment
  3. **Sentence 3 (50 words):** Results - 5.2% inclusion, only 13% tested hypothesis, 83% favored dynamic models (+4% to +29%), 0% true digital twins, meta-analysis impossible
  4. **Sentence 4 (30 words):** Conclusions - Limited evidence suggests promise but insufficient for recommendations; 87% lack comparators; standardized reporting needed

**Key cuts:**
- Remove detailed database names
- Remove specific effect sizes (keep range)
- Remove validation tier details
- Remove specific percentages for exclusion reasons

---

### PHASE 3: INTRODUCTION RESTRUCTURING (No subheadings)

**Current structure (4 subsections):**
1. Clinical Heterogeneity and Prognostic Challenges
2. Digital Twins and Dynamic Mechanistic Models  
3. Research Gap: Lack of Empirical Benchmarking
4. Objective and Research Questions

**New structure (flowing narrative, no breaks):**
- Paragraph 1: PD heterogeneity and prognostic challenges (keep concise)
- Paragraph 2: Rise of ML in PD, introduction of digital twins concept
- Paragraph 3: Critical gap - no systematic benchmarking evidence
- Paragraph 4: Study objectives and research questions (PICO format)

**Target length:** ~800-1000 words (reduce from 1,800)

**What to cut:**
- Detailed mechanistic modeling explanations → Move to Supplementary
- Extensive literature context → Cite reviews instead
- Theoretical advantages of digital twins → Brief mention only
- Multiple examples of heterogeneity → Keep 2-3 key points

---

### PHASE 4: RESULTS OPTIMIZATION (Subheadings allowed)

**Current structure (6 subsections):**
1. Study Selection
2. Study Characteristics
3. Risk of Bias and Validation Quality
4. Comparative Effectiveness: Dynamic vs Static Models
5. Meta-Analysis Assessment
6. Summary Tables and Figures

**Keep all subsections but streamline:**

**3.1 Study Selection** (KEEP)
- PRISMA flow: 287→15 (5.2%)
- 6 comparative studies (2.1%)
- 2 testing hypothesis (13%)

**3.2 Study Characteristics** (CONDENSE)
- Merge 3.2.1-3.2.3 into single section
- Table 1: Study characteristics (keep)
- Remove extensive narrative, let table speak

**3.3 Risk of Bias Assessment** (KEEP)
- PROBAST results: 27% low, 60% moderate, 13% high
- Validation tiers: 67% Tier 2
- Table 2: PROBAST summary (keep)

**3.4 Comparative Effectiveness** (KEEP - CORE FINDING)
- 6 papers with comparisons
- Effect sizes: -2.3% to +28.9%
- 5/6 favor dynamic models
- Table 3: Comparative performance (keep)

**3.5 Meta-Analysis Feasibility** (CONDENSE)
- One paragraph: impossible due to heterogeneity
- Refer to Supplementary for detailed assessment

**3.6 Critical Gaps** (NEW - MERGE FROM DISCUSSION)
- 87% no comparators
- 0% mechanistic digital twins
- 0% variance reporting

**Target length:** ~2,500-3,000 words (reduce from 6,400)

---

### PHASE 5: DISCUSSION RESTRUCTURING (No subheadings, merge Conclusion)

**Current structure (9 subsections + separate Conclusion):**
1. Principal Findings
2. Benchmarking Gap Analysis
3. Performance vs Complexity Trade-Off
4. Barriers to Meta-Analysis
5. Limitations (3 levels)
6. State-of-the-Art and Future Directions
7. Clinical Translation Readiness
8. Comparison to Other Diseases
9. Recommendations for Stakeholders
+ Separate Conclusion section

**New structure (flowing narrative, NO subheadings):**

**Paragraph flow:**
1. **Opening (Principal findings):** Restate key results - only 13% test hypothesis, 83% favor dynamic, evidence fragile
2. **Interpretation:** Why benchmarking gap exists (publication bias, computational cost, negative result suppression)
3. **Clinical implications:** What current evidence means for practice
4. **Mechanistic digital twin gap:** Why 0% implementation (data requirements, lack of validated models)
5. **Methodological issues:** Metric heterogeneity, missing variance, prevents meta-analysis
6. **Limitations:** Three-level (review, study, field) - condensed to 1-2 paragraphs
7. **Future directions:** TRIPOD-AI, shadow mode, mandatory comparisons (brief)
8. **Stakeholder recommendations:** Condensed to 2-3 key actions per group
9. **Conclusion (integrated):** Final synthesis, call to action, optimistic closing

**Target length:** ~3,000-3,500 words (reduce from 12,400 total Discussion+Conclusion)

**What to MOVE to Supplementary:**
- Detailed TRIPOD-AI requirements → Supplementary Discussion S1
- Shadow mode validation examples → Supplementary Discussion S2
- Emerging technologies (foundation models, PINNs) → Supplementary Discussion S3
- Comparison to other diseases → Supplementary Discussion S4
- Detailed TRL framework → Supplementary Discussion S5
- Extensive stakeholder recommendations → Supplementary Table S5

---

### PHASE 6: METHODS STREAMLINING (Subheadings allowed)

**Current structure (7 subsections):**
1. Protocol and Registration
2. Eligibility Criteria
3. Information Sources and Search Strategy
4. Selection Process
5. Data Collection Process
6. Risk of Bias Assessment
7. Data Synthesis and Analysis

**Keep all subsections but condense:**
- Move full search strings to Supplementary Table S1
- Move detailed PROBAST criteria to Supplementary Methods S1
- Move data extraction form to Supplementary Methods S2
- Keep core methodology in main text

**Target length:** ~2,000-2,500 words (reduce from 3,500)

---

### PHASE 7: REFERENCE REDUCTION (60-80 total)

**Current:** 300 references  
**Target:** 60-80 references  
**Reduction:** 73-80%

**Strategy:**

**KEEP (Priority 1 - ~30 refs):**
- All 15 included studies (MANDATORY)
- PRISMA 2020 statement
- PROBAST tool paper
- TRIPOD-AI guideline
- Key PD cohort papers (PPMI, PDBP)
- Landmark PD clinical papers (MDS criteria, heterogeneity)

**KEEP (Priority 2 - ~20 refs):**
- Key methodological papers (meta-analysis methods, validation frameworks)
- Representative digital twin papers (1-2 examples)
- Key ML in healthcare papers (2-3 seminal)
- Shadow mode validation examples (1-2)
- Foundation model papers (1-2)

**KEEP (Priority 3 - ~15 refs):**
- Supporting evidence for Discussion points
- Regulatory framework papers (FDA AI/ML guidance)
- Health economics papers (1-2)
- Comparison disease papers (1-2)

**CUT (Priority 4 - ~235 refs):**
- Extensive SOTA references → Cite review papers instead
- Multiple examples of same concept → Keep 1-2 representative
- Detailed mechanistic modeling → Move to Supplementary
- Stakeholder recommendation citations → Reduce to essentials

**Reference management:**
- Cite systematic reviews instead of individual studies where possible
- Combine related concepts under single review citation
- Use "et al." citations for multi-author papers

---

## FIGURES & TABLES PLAN

### MAIN TEXT (Maximum 4-6 figures/tables)

**Figure 1: PRISMA Flow Diagram** (MANDATORY)
- 354 → 287 → 15 papers
- Exclusion reasons at each stage
- Legend: ≤350 words

**Table 1: Characteristics of 15 Included Studies**
- Study, Year, Design, N, Model Type, Prediction Goal, Validation Tier, Risk of Bias
- Condensed from current detailed table

**Figure 2: Harvest Plot - Comparative Effectiveness**
- 6 studies with effect sizes
- X-axis: Effect size (-5% to +30%)
- Y-axis: Validation tier
- Color/shape: Prediction goal
- Legend: ≤350 words

**Table 2: Summary of Comparative Performance**
- 6 papers with quantitative comparisons
- Intervention vs. Comparator performance
- Effect sizes and direction

**Figure 3: Risk of Bias Summary (PROBAST)**
- Stacked bar chart: 4 domains
- Traffic light plot: 15 studies × 4 domains
- Legend: ≤350 words

**Figure 4: Critical Evidence Gaps**
- Visual showing:
  - 87% no comparators
  - 0% digital twins
  - 0% variance reporting
  - Meta-analysis barriers
- Legend: ≤350 words

---

### SUPPLEMENTARY INFORMATION

**Supplementary Tables:**
- **Table S1:** Complete search strategies (4 databases)
- **Table S2:** All 287 screened papers with characteristics
- **Table S3:** PROBAST detailed assessments (15 papers)
- **Table S4:** Excluded papers with reasons (272 papers)
- **Table S5:** Stakeholder recommendations (detailed)

**Supplementary Figures:**
- **Figure S1:** Publication timeline (2016-2026)
- **Figure S2:** Geographic distribution of studies
- **Figure S3:** Model architecture distribution

**Supplementary Methods:**
- **Methods S1:** Detailed PROBAST assessment criteria
- **Methods S2:** Data extraction form template
- **Methods S3:** Search strategy development process

**Supplementary Discussion:**
- **Discussion S1:** Detailed TRIPOD-AI requirements and compliance
- **Discussion S2:** Shadow mode validation framework and examples
- **Discussion S3:** Emerging technologies (foundation models, PINNs, federated learning)
- **Discussion S4:** Comparison to other neurodegenerative diseases
- **Discussion S5:** Clinical translation readiness (TRL framework)

**Supplementary Data:**
- **Data S1:** Complete data extraction spreadsheet (Excel)
- **Data S2:** PRISMA 2020 checklist (completed)

---

## ESTIMATED FINAL WORD COUNTS

| Section | Current | Target | Reduction |
|---------|---------|--------|-----------|
| Title | 18 words | 14 words | -22% |
| Abstract | 289 words | 150 words | -48% |
| Introduction | 1,800 words | 1,000 words | -44% |
| Methods | 3,500 words | 2,200 words | -37% |
| Results | 6,400 words | 3,000 words | -53% |
| Discussion | 12,400 words | 3,500 words | -72% |
| **TOTAL** | **~24,400 words** | **~10,000 words** | **-59%** |
| References | 300 | 70 | -77% |

**Final manuscript:** ~10,000 words + 4-6 figures/tables + extensive Supplementary Information

---

## WRITING PRIORITIES

### HIGH PRIORITY (Core findings - MUST include)
1. Only 13% of studies test dynamic vs. static hypothesis
2. 87% lack baseline comparators (benchmarking gap)
3. 0% implement true mechanistic digital twins
4. 5/6 comparative studies favor dynamic models
5. Effect sizes range -2.3% to +28.9%
6. Meta-analysis impossible (heterogeneity, missing variance)
7. 67% achieved Tier 2 external validation (positive)

### MEDIUM PRIORITY (Important context)
1. Search strategy and screening results
2. PROBAST risk of bias findings
3. Validation quality assessment
4. Clinical heterogeneity issues
5. Reporting quality gaps (TRIPOD-AI)
6. Limitations (three levels)

### LOW PRIORITY (Move to Supplementary)
1. Detailed TRIPOD-AI requirements
2. Shadow mode validation framework
3. Emerging technologies discussion
4. Comparison to other diseases
5. Extensive stakeholder recommendations
6. Detailed TRL framework
7. State-of-the-art technology review

---

## NEXT STEPS

### Step 1: Create NPJ-Compliant Manuscript Structure
- New title (14 words)
- Condensed abstract (150 words)
- Restructured Introduction (no subheadings, ~1,000 words)
- Streamlined Results (keep subheadings, ~3,000 words)
- Unified Discussion (no subheadings, merge Conclusion, ~3,500 words)
- Condensed Methods (keep subheadings, ~2,200 words)

### Step 2: Create Main Figures (4-6 total)
- Figure 1: PRISMA flow diagram
- Figure 2: Harvest plot
- Figure 3: PROBAST summary
- Figure 4: Evidence gaps visualization
- Table 1: Study characteristics
- Table 2: Comparative performance

### Step 3: Organize Supplementary Information
- Move detailed content from Discussion
- Create supplementary tables (S1-S5)
- Create supplementary figures (S1-S3)
- Create supplementary methods (S1-S3)
- Create supplementary discussion (S1-S5)

### Step 4: Reference Curation
- Identify 60-70 essential references
- Remove redundant citations
- Cite reviews instead of multiple individual papers
- Ensure all 15 included studies cited

### Step 5: Format for NPJ Submission
- Title page with affiliations
- Author contributions statement
- Competing interests declaration
- Data/code availability statements
- Acknowledgments with funding

---

## TIMELINE

- **Day 1-2:** Restructure manuscript (title, abstract, sections)
- **Day 3:** Create main figures and tables
- **Day 4:** Organize supplementary information
- **Day 5:** Curate references (300 → 70)
- **Day 6:** Format for NPJ submission
- **Day 7:** Final review and quality check

---

**Status:** Ready to begin restructuring  
**Target journal:** NPJ Parkinson's Disease  
**Submission type:** Article (Systematic Review)  
**Estimated completion:** 7 days
