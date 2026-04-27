# Feature Attribution Analysis: Phase4_Progression_Subtypes

## Overview
- **Task**: phase4_subtype_classification
- **Features analyzed**: 9
- **Attribution methods**: IntegratedGradients, GradientSHAP

## Global Feature Importance

### Cross-Method Consensus

**Consensus Important Features** (identified by ≥50% of methods):

- **updrs_slope**: Identified by 2/2 methods
- **SEX**: Identified by 2/2 methods
- **MOCA_BL**: Identified by 2/2 methods
- **UPDRS_III_BL**: Identified by 2/2 methods
- **UPDRS_III_V06**: Identified by 2/2 methods

### Method-Specific Insights


**IntegratedGradients** Top 5 Features:

- **updrs_slope**: Importance = 5.7976 (Rank #1)
- **SEX**: Importance = 0.6969 (Rank #2)
- **MOCA_BL**: Importance = 0.6290 (Rank #3)
- **UPDRS_III_BL**: Importance = 0.5496 (Rank #4)
- **UPDRS_III_V06**: Importance = 0.5127 (Rank #5)

**GradientSHAP** Top 5 Features:

- **updrs_slope**: Importance = 3.3569 (Rank #1)
- **SEX**: Importance = 0.7597 (Rank #2)
- **UPDRS_III_V06**: Importance = 0.5719 (Rank #3)
- **MOCA_BL**: Importance = 0.4385 (Rank #4)
- **UPDRS_III_BL**: Importance = 0.4232 (Rank #5)

## Clinical Implications

1. **Focus on consensus features** for clinical decision-making:
   - updrs_slope
   - SEX
   - MOCA_BL
   - UPDRS_III_BL
   - UPDRS_III_V06

2. **These features are robustly important** across different attribution methods, suggesting high reliability

3. **Method diversity provides complementary insights**:
   - Gradient-based methods capture local importance
   - SHAP-based methods provide global context

## Recommendations

1. Prioritize consensus features in clinical assessments
2. Use feature attributions to guide targeted biomarker development
3. Validate feature importance in prospective cohorts
4. Consider feature interactions when interpreting individual attributions
