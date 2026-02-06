# Patient Similarity Clustering Analysis: Phase5_Prodromal_Conversion

## Overview
- **Task**: phase5_prodromal_conversion
- **Patients**: 381
- **Optimal number of clusters**: 6
- **Clustering methods**: Hierarchical, KMeans

## Cluster Quality Metrics

### Hierarchical

**Cluster Size Distribution**:

- Cluster 0: 89 patients (23.4%)
- Cluster 1: 53 patients (13.9%)
- Cluster 2: 99 patients (26.0%)
- Cluster 3: 27 patients (7.1%)
- Cluster 4: 58 patients (15.2%)
- Cluster 5: 55 patients (14.4%)

**Label Purity** (dominant class proportion):

- Cluster 0: 97.8% (dominant class: 0)
- Cluster 1: 100.0% (dominant class: 0)
- Cluster 2: 99.0% (dominant class: 0)
- Cluster 3: 96.3% (dominant class: 0)
- Cluster 4: 98.3% (dominant class: 0)
- Cluster 5: 81.8% (dominant class: 0)

**Average Prediction Confidence**:

- Cluster 0: 0.552
- Cluster 1: 0.561
- Cluster 2: 0.538
- Cluster 3: 0.535
- Cluster 4: 0.573
- Cluster 5: 0.751

### KMeans

**Cluster Size Distribution**:

- Cluster 0: 43 patients (11.3%)
- Cluster 1: 94 patients (24.7%)
- Cluster 2: 108 patients (28.3%)
- Cluster 3: 27 patients (7.1%)
- Cluster 4: 58 patients (15.2%)
- Cluster 5: 51 patients (13.4%)

**Label Purity** (dominant class proportion):

- Cluster 0: 100.0% (dominant class: 0)
- Cluster 1: 96.8% (dominant class: 0)
- Cluster 2: 100.0% (dominant class: 0)
- Cluster 3: 96.3% (dominant class: 0)
- Cluster 4: 98.3% (dominant class: 0)
- Cluster 5: 80.4% (dominant class: 0)

**Average Prediction Confidence**:

- Cluster 0: 0.558
- Cluster 1: 0.553
- Cluster 2: 0.538
- Cluster 3: 0.535
- Cluster 4: 0.573
- Cluster 5: 0.771

## Clinical Implications

### Cluster Homogeneity
- **Optimal k=6** suggests 6 distinct patient subgroups
- High label purity indicates clusters align with clinical diagnoses
- Low label purity suggests novel subgroups cutting across traditional categories

### Clinical Trial Enrichment
- Homogeneous clusters can be used for:
  1. **Patient stratification** in clinical trials
  2. **Targeted recruitment** of similar patients
  3. **Subgroup-specific treatment strategies**

### Precision Medicine Applications
- Clusters represent patients with similar:
  - Clinical trajectories
  - Treatment responses (hypothesized)
  - Prognostic outcomes

## Recommendations

1. **Validate clusters** in independent cohorts
2. **Investigate cluster-specific biomarkers** from feature analysis
3. **Design cluster-targeted interventions**
4. **Use for trial enrichment** to reduce sample size requirements
5. **Monitor patients** who fall between clusters (transition states)

## Next Steps

- Compare clustering results with clinical subtypes
- Investigate features driving cluster separation
- Assess cluster stability across different time points
