# Feature Attribution Analysis: Phase5_Prodromal_Conversion

## Overview
- **Task**: phase5_prodromal_conversion
- **Features analyzed**: 7
- **Attribution methods**: IntegratedGradients, GradientSHAP

## Global Feature Importance

### Cross-Method Consensus

**Consensus Important Features** (identified by ≥50% of methods):

- **baseline_updrs**: Identified by 2/2 methods
- **handed**: Identified by 2/2 methods
- **sex**: Identified by 2/2 methods
- **baseline_moca**: Identified by 2/2 methods
- **time_to_event**: Identified by 1/2 methods
- **age_approx**: Identified by 1/2 methods

### Method-Specific Insights


**IntegratedGradients** Top 5 Features:

- **baseline_updrs**: Importance = 0.1970 (Rank #1)
- **handed**: Importance = 0.0961 (Rank #2)
- **time_to_event**: Importance = 0.0889 (Rank #3)
- **sex**: Importance = 0.0782 (Rank #4)
- **baseline_moca**: Importance = 0.0733 (Rank #5)

**GradientSHAP** Top 5 Features:

- **baseline_updrs**: Importance = 0.1643 (Rank #1)
- **baseline_moca**: Importance = 0.0949 (Rank #2)
- **sex**: Importance = 0.0889 (Rank #3)
- **handed**: Importance = 0.0736 (Rank #4)
- **age_approx**: Importance = 0.0667 (Rank #5)

## Clinical Implications

1. **Focus on consensus features** for clinical decision-making:
   - baseline_updrs
   - handed
   - sex
   - baseline_moca
   - time_to_event

2. **These features are robustly important** across different attribution methods, suggesting high reliability

3. **Method diversity provides complementary insights**:
   - Gradient-based methods capture local importance
   - SHAP-based methods provide global context

## Recommendations

1. Prioritize consensus features in clinical assessments
2. Use feature attributions to guide targeted biomarker development
3. Validate feature importance in prospective cohorts
4. Consider feature interactions when interpreting individual attributions
