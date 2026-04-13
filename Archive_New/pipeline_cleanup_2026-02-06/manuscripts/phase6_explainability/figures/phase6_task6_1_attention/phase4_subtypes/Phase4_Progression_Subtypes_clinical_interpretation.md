# GAT Attention Analysis: Phase4_Progression_Subtypes

## Overview
- **Task**: phase4_subtype_classification
- **Patients**: 364
- **High-importance edges analyzed**: 50

## Key Findings

### 1. Attention Pattern Summary
- **Same-label connections**: 68.0% of high-attention edges connect patients with the same diagnosis
- **Same-prediction connections**: 78.0% of high-attention edges connect patients with the same predicted class
- **Mean attention weight**: 0.1741 ± 0.0246

### 2. Class-Specific Attention Patterns

**Class 0**:
- 30 high-attention edges (60.0%)
- Mean attention: 0.1711
- Intra-class edges: 18 (60.0%)
- Inter-class edges: 12 (40.0%)

**Class 1**:
- 20 high-attention edges (40.0%)
- Mean attention: 0.1704
- Intra-class edges: 5 (25.0%)
- Inter-class edges: 15 (75.0%)

**Class 2**:
- 16 high-attention edges (32.0%)
- Mean attention: 0.1726
- Intra-class edges: 11 (68.8%)
- Inter-class edges: 5 (31.2%)


### 3. Clinical Implications

- **Cross-diagnostic attention**: The model shows significant attention to patients with different diagnoses, which may indicate:
  - Shared clinical features across diagnostic groups
  - Patients in transition states between diagnoses
  - Need for refined diagnostic criteria

- **High-confidence connections**: 27 edges (54.0%) connect patients where model is >80% confident
  - These represent the model's most reliable similarity assessments

### 4. Recommendations for Clinical Application

1. **Patient stratification**: High-attention patient pairs could be grouped for targeted treatment strategies
2. **Clinical trial design**: Use attention patterns to identify homogeneous patient subgroups
3. **Prognostic refinement**: Patients with high attention to different diagnostic groups may warrant closer monitoring
