# GAT Attention Analysis: Phase5_Prodromal_Conversion

## Overview
- **Task**: phase5_prodromal_conversion
- **Patients**: 381
- **High-importance edges analyzed**: 50

## Key Findings

### 1. Attention Pattern Summary
- **Same-label connections**: 88.0% of high-attention edges connect patients with the same diagnosis
- **Same-prediction connections**: 84.0% of high-attention edges connect patients with the same predicted class
- **Mean attention weight**: 0.0974 ± 0.0029

### 2. Class-Specific Attention Patterns

**Class 0**:
- 49 high-attention edges (98.0%)
- Mean attention: 0.0974
- Intra-class edges: 43 (87.8%)
- Inter-class edges: 6 (12.2%)

**Class 1**:
- 7 high-attention edges (14.0%)
- Mean attention: 0.0973
- Intra-class edges: 1 (14.3%)
- Inter-class edges: 6 (85.7%)


### 3. Clinical Implications

- **Strong diagnostic coherence**: The model primarily attends to clinically similar patients, suggesting it has learned meaningful disease patterns.

- **High-confidence connections**: 2 edges (4.0%) connect patients where model is >80% confident
  - These represent the model's most reliable similarity assessments

### 4. Recommendations for Clinical Application

1. **Patient stratification**: High-attention patient pairs could be grouped for targeted treatment strategies
2. **Clinical trial design**: Use attention patterns to identify homogeneous patient subgroups
3. **Prognostic refinement**: Patients with high attention to different diagnostic groups may warrant closer monitoring
