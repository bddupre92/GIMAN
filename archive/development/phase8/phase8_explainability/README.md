# Phase 8.6: Explainability Framework for GIMAN Models

**Project:** Graph-Informed Multimodal Attention Network (GIMAN)  
**Phase:** 8.6 - Multi-Scale Explainability for Phase 8 Models  
**Date:** October 19, 2025  
**Status:** 🚧 IN PROGRESS

---

## Overview

Phase 8.6 adapts the Phase 7 explainability framework to the three production-ready Phase 8 models, with special emphasis on **survival-specific explanations** for time-to-event predictions.

### Models to Explain

| Model | Type | Performance | Priority |
|-------|------|-------------|----------|
| **Phase 8.2 Progression** | Survival (Cox PH) | C-index 0.998 | 🔴 CRITICAL |
| **Phase 8.3 SAA** | Binary classification | AUC 0.623 | 🟡 HIGH |
| **Phase 8.4 Diagnostic** | Multi-class (3-way) | Accuracy 0.998 | 🟡 HIGH |

**Note:** Phase 8.5 Multi-Task model excluded (failed approach per completion report).

---

## XAI Methods

### 1. SHAP (SHapley Additive exPlanations)

**Standard Implementation:**
- Binary/Multi-class: Feature importance for class predictions
- Cohort-level: Mean absolute SHAP values

**Survival-Specific Adaptations (NEW):**
- **Hazard SHAP:** Explain predicted hazard at specific time points (1-year, 3-year, 5-year)
- **Survival Curve Decomposition:** Show how features shift survival probability S(t)
- **Relative Risk SHAP:** Explain hazard ratio vs. population baseline
- **Temporal Stability:** Track feature importance evolution over time horizons

### 2. GNNExplainer

**Standard Implementation:**
- Identify important subgraphs (patient neighborhoods)
- Extract edge and node importance

**Phase 8 Adaptations:**
- **Progression:** Subgraphs predicting rapid vs. slow decline
- **SAA:** Neighborhoods sharing synucleinopathy biomarkers
- **Diagnostic:** Subgraphs distinguishing PD/Prodromal/Control
- **Validation:** Do identified patterns match known clinical risk factors?

### 3. GradCAM (Gradient-weighted Class Activation Mapping)

**Standard Implementation:**
- Saliency maps for feature importance
- Gradient-based attribution

**Imaging-Specific Adaptations:**
- **Brain Region Saliency:** Map attention to FreeSurfer cortical parcellations
- **Subcortical ROIs:** Highlight striatum, substantia nigra, hippocampus
- **Visualization:** Overlay saliency on standardized brain templates

---

## Directory Structure

```
phase8_explainability/
├── README.md                          # This file
├── config/
│   ├── shap_config.yaml              # SHAP hyperparameters
│   ├── gnnexplainer_config.yaml      # GNNExplainer settings
│   └── gradcam_config.yaml           # GradCAM parameters
├── scripts/
│   ├── shap_survival_analysis.py     # SHAP for Cox PH models
│   ├── shap_classification.py        # SHAP for SAA/Diagnostic
│   ├── gnnexplainer_subgraphs.py     # Subgraph extraction
│   ├── gradcam_imaging.py            # Brain region saliency
│   └── consensus_analysis.py         # Cross-method comparison
├── models/
│   ├── phase8_2_progression.pth      # Symlink to Phase 8.2 model
│   ├── phase8_3_saa.pth              # Symlink to Phase 8.3 model
│   └── phase8_4_diagnostic.pth       # Symlink to Phase 8.4 model
├── outputs/
│   ├── shap/
│   │   ├── progression/              # Time-dependent SHAP values
│   │   ├── saa/                      # SAA feature importance
│   │   └── diagnostic/               # Diagnostic explanations
│   ├── gnnexplainer/
│   │   ├── progression_subgraphs/
│   │   ├── saa_subgraphs/
│   │   └── diagnostic_subgraphs/
│   ├── gradcam/
│   │   ├── progression_saliency/
│   │   ├── saa_saliency/
│   │   └── diagnostic_saliency/
│   └── consensus/
│       └── cross_method_agreement.json
├── reports/
│   ├── per_patient/                  # Individual prediction reports
│   ├── cohort_level/                 # Aggregated feature importance
│   └── clinical_insights/            # Clinician-friendly summaries
└── visualizations/
    ├── survival_curves/              # SHAP-decomposed S(t)
    ├── brain_maps/                   # GradCAM overlays on brain
    └── subgraph_networks/            # GNNExplainer graph viz
```

---

## Survival-Specific Explainability (Phase 8.2 Focus)

### Challenge

Traditional XAI methods explain **static predictions** (class probabilities). Survival models predict **dynamic outcomes** (time-to-event).

**Question:** How do we explain a survival curve S(t) or hazard function h(t)?

### Solution: Time-Conditioned SHAP

**Approach:**
1. Select time horizons of interest: t ∈ {1 year, 3 years, 5 years}
2. For each t, compute SHAP values explaining:
   - Predicted survival probability: S(t|X)
   - Predicted hazard: h(t|X)
   - Predicted cumulative hazard: H(t|X)
3. Visualize feature importance **trajectories** over time

**Implementation Pseudocode:**

```python
def explain_survival_at_time(model, patient, time_point):
    """
    Explain survival prediction at specific time horizon.
    
    Args:
        model: GIMANSurvivalGAT (Phase 8.2)
        patient: PyG Data object (single patient)
        time_point: Time horizon (e.g., 1, 3, 5 years)
    
    Returns:
        shap_values: Feature importance for S(time_point)
        base_value: Population baseline survival
        expected_value: Patient predicted survival
    """
    # Define prediction function for SHAP
    def predict_survival_wrapper(X):
        # Forward pass through model
        risk_scores = model(X)  # Cox risk scores
        
        # Convert to survival probability at time_point
        # Using Breslow estimator or Kaplan-Meier baseline
        baseline_hazard = model.baseline_hazard[time_point]
        survival_probs = np.exp(-baseline_hazard * np.exp(risk_scores))
        
        return survival_probs
    
    # Create SHAP explainer
    explainer = shap.KernelExplainer(
        predict_survival_wrapper,
        background_data  # Representative sample
    )
    
    # Compute SHAP values
    shap_values = explainer.shap_values(patient.x)
    
    return shap_values

def explain_survival_curve(model, patient, time_points=[1, 3, 5]):
    """
    Explain entire survival curve by decomposing into feature contributions.
    
    Returns:
        feature_trajectories: Dict mapping feature -> importance over time
    """
    feature_trajectories = {}
    
    for t in time_points:
        shap_vals = explain_survival_at_time(model, patient, t)
        
        for feature_idx, importance in enumerate(shap_vals):
            if feature_idx not in feature_trajectories:
                feature_trajectories[feature_idx] = []
            feature_trajectories[feature_idx].append((t, importance))
    
    return feature_trajectories
```

### Visualization: SHAP Force Plots for Survival

**Standard SHAP Force Plot:**
```
E[S(t)] = 0.60 ---[Feature1 +0.15]--[Feature2 -0.08]--[Feature3 +0.03]--> S(t) = 0.70
          ↑                                                                    ↑
     Population baseline                                            Patient prediction
```

**Temporal Evolution:**
```
t=1 year:  S(1) = 0.95  [UPDRS +0.02] [Age -0.01] [Genetics +0.01]
t=3 years: S(3) = 0.75  [UPDRS +0.08] [Age -0.05] [Genetics +0.03]
t=5 years: S(5) = 0.50  [UPDRS +0.15] [Age -0.10] [Genetics +0.05]
                         ↑ Increasing importance over time
```

---

## GNNExplainer for Patient Neighborhoods

### Objective

Identify which **patient neighborhoods** (subgraphs) are most informative for predictions.

**Research Questions:**
1. Do patients with similar predictions share local graph structure?
2. What clinical profiles characterize high-risk vs. low-risk neighborhoods?
3. Can we identify "archetype" patients representing distinct patterns?

### Implementation Strategy

```python
class GNNExplainerPipeline:
    """
    Extract and analyze important subgraphs for Phase 8 models.
    """
    
    def explain_patient_prediction(self, model, patient_idx, graph_data):
        """
        Explain prediction for a single patient.
        
        Returns:
            edge_mask: Importance of each edge in neighborhood
            node_mask: Importance of each neighbor node
            subgraph: Extracted k-hop neighborhood
        """
        explainer = GNNExplainer(model, epochs=200)
        
        node_feat_mask, edge_mask = explainer.explain_node(
            patient_idx, 
            graph_data.x, 
            graph_data.edge_index
        )
        
        # Extract k-hop subgraph
        subgraph_nodes, subgraph_edges, mapping = k_hop_subgraph(
            patient_idx, 
            num_hops=2,
            edge_index=graph_data.edge_index,
            relabel_nodes=True
        )
        
        return {
            'edge_mask': edge_mask,
            'node_mask': node_feat_mask,
            'subgraph_nodes': subgraph_nodes,
            'subgraph_edges': subgraph_edges
        }
    
    def identify_common_patterns(self, explanations):
        """
        Cluster explanations to find recurring neighborhood patterns.
        
        Args:
            explanations: List of GNNExplainer outputs
        
        Returns:
            pattern_clusters: Grouped patients with similar explanations
            archetype_patients: Representative examples per cluster
        """
        # Extract edge importance vectors
        edge_vectors = [exp['edge_mask'] for exp in explanations]
        
        # Cluster using hierarchical clustering
        linkage = scipy.cluster.hierarchy.linkage(edge_vectors, method='ward')
        clusters = scipy.cluster.hierarchy.fcluster(linkage, t=5, criterion='maxclust')
        
        # Identify archetype (medoid) per cluster
        archetypes = []
        for cluster_id in np.unique(clusters):
            cluster_members = np.where(clusters == cluster_id)[0]
            centroid = np.mean([edge_vectors[i] for i in cluster_members], axis=0)
            medoid_idx = cluster_members[
                np.argmin([np.linalg.norm(edge_vectors[i] - centroid) 
                          for i in cluster_members])
            ]
            archetypes.append(medoid_idx)
        
        return clusters, archetypes
```

---

## GradCAM for Brain Region Saliency

### Objective

Identify which **brain regions** (FreeSurfer parcellations) are most important for predictions.

**Applications:**
- **Progression:** Cortical thinning in motor cortex predicts faster decline?
- **SAA:** Hippocampal atrophy correlates with synucleinopathy?
- **Diagnostic:** Striatal features distinguish PD from prodromal?

### Implementation Strategy

```python
class BrainGradCAM:
    """
    GradCAM adapted for graph neural networks with FreeSurfer features.
    """
    
    def __init__(self, model, target_layer='gat_layers.2'):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_module = dict(self.model.named_modules())[self.target_layer]
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
    
    def generate_saliency_map(self, patient_data, class_idx=None):
        """
        Generate saliency map for patient prediction.
        
        Args:
            patient_data: PyG Data object
            class_idx: Target class (for multi-class) or None (for survival)
        
        Returns:
            saliency_map: Feature-level importance (dim: num_features)
        """
        # Forward pass
        output = self.model(patient_data.x, patient_data.edge_index)
        
        # Backward pass
        if class_idx is not None:
            # Multi-class: backprop from specific class
            self.model.zero_grad()
            output[0, class_idx].backward(retain_graph=True)
        else:
            # Survival: backprop from risk score
            self.model.zero_grad()
            output.sum().backward()
        
        # Compute weighted activation
        weights = self.gradients.mean(dim=0)  # Global average pooling
        saliency = (weights * self.activations).sum(dim=1)
        saliency = F.relu(saliency)  # ReLU to keep positive contributions
        
        return saliency.cpu().numpy()
    
    def map_to_brain_regions(self, saliency_map, feature_names):
        """
        Map feature-level saliency to anatomical brain regions.
        
        Args:
            saliency_map: Feature importance scores
            feature_names: List of FreeSurfer ROI names
        
        Returns:
            region_importance: Dict mapping ROI -> importance score
        """
        region_importance = {}
        
        # Assuming features are ordered by FreeSurfer parcellations
        freesurfer_start = 0  # Adjust based on actual feature ordering
        freesurfer_end = 68   # 68 cortical parcellations
        
        for i, roi_name in enumerate(feature_names[freesurfer_start:freesurfer_end]):
            region_importance[roi_name] = saliency_map[freesurfer_start + i]
        
        return region_importance
```

---

## Cross-Method Consensus Analysis

### Objective

Ensure explanations are **robust** across different XAI methods. High consensus = reliable insights.

**Consensus Metrics:**

1. **Top-K Overlap:**
   - Identify top-10 features from SHAP, GNNExplainer, GradCAM
   - Compute intersection: consensus = |F_SHAP ∩ F_GNN ∩ F_GradCAM| / 10
   - **Target:** >85% consensus (≥8/10 features agree)

2. **Rank Correlation:**
   - Spearman correlation between feature rankings
   - **Target:** ρ > 0.7 for pairwise method comparisons

3. **Sign Agreement:**
   - For SHAP (has sign), check if GNN/GradCAM rank same features highly
   - Binary agreement: Do methods agree on importance direction?

**Implementation:**

```python
def compute_consensus(shap_importance, gnn_importance, gradcam_importance, k=10):
    """
    Compute cross-method consensus on top-k features.
    
    Returns:
        consensus_features: Features appearing in ≥2 methods
        consensus_score: Fraction of features with agreement
        rank_correlations: Pairwise Spearman correlations
    """
    # Get top-k features per method
    shap_top_k = set(np.argsort(np.abs(shap_importance))[-k:])
    gnn_top_k = set(np.argsort(gnn_importance)[-k:])
    gradcam_top_k = set(np.argsort(gradcam_importance)[-k:])
    
    # Compute intersection
    full_consensus = shap_top_k & gnn_top_k & gradcam_top_k
    partial_consensus = (shap_top_k & gnn_top_k) | (shap_top_k & gradcam_top_k) | (gnn_top_k & gradcam_top_k)
    
    # Rank correlations
    from scipy.stats import spearmanr
    corr_shap_gnn = spearmanr(shap_importance, gnn_importance).correlation
    corr_shap_grad = spearmanr(shap_importance, gradcam_importance).correlation
    corr_gnn_grad = spearmanr(gnn_importance, gradcam_importance).correlation
    
    return {
        'full_consensus': full_consensus,
        'full_consensus_score': len(full_consensus) / k,
        'partial_consensus': partial_consensus,
        'partial_consensus_score': len(partial_consensus) / k,
        'rank_correlations': {
            'SHAP_vs_GNN': corr_shap_gnn,
            'SHAP_vs_GradCAM': corr_shap_grad,
            'GNN_vs_GradCAM': corr_gnn_grad,
            'mean': np.mean([corr_shap_gnn, corr_shap_grad, corr_gnn_grad])
        }
    }
```

---

## Clinical Validation

### Objective

Ensure explanations align with **known PD biology** and **clinical expert knowledge**.

**Validation Checklist:**

| Task | Expected Important Features | Rationale |
|------|----------------------------|-----------|
| **Progression** | UPDRS-III, MoCA, Age, Genetics (GBA/LRRK2), Striatal SBR | Established progression markers |
| **SAA** | Genetics (SNCA/LRRK2), CSF Aβ42/Tau, Hippocampal volume | Synucleinopathy biomarkers |
| **Diagnostic** | UPDRS-III, Striatal DAT binding, Bradykinesia items | Core PD diagnostic criteria |

**Validation Process:**

1. **Literature Review:** Identify features from published PD prediction models
2. **Expert Annotation:** Clinician reviews top-10 features, marks as plausible/implausible
3. **Biological Plausibility:** Check if features have mechanistic explanation
4. **Counterfactual Testing:** "What if UPDRS was lower?" → Should reduce risk

---

## Success Metrics

| Metric | Target | Phase 8.2 | Phase 8.3 | Phase 8.4 |
|--------|--------|-----------|-----------|-----------|
| **Cross-method consensus (top-10)** | >85% | TBD | TBD | TBD |
| **Rank correlation (mean)** | >0.70 | TBD | TBD | TBD |
| **Clinical validity** | >90% expert-rated plausible | TBD | TBD | TBD |
| **Temporal stability (survival)** | Feature ranks stable across horizons | TBD | N/A | N/A |
| **Counterfactual realism** | Suggested changes biologically plausible | TBD | TBD | TBD |

---

## Timeline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| **Week 1** | Framework setup, SHAP for Phase 8.2 | `shap_survival_analysis.py`, survival SHAP outputs |
| **Week 2** | SHAP for 8.3/8.4, GNNExplainer all models | Classification SHAP, subgraph patterns |
| **Week 3** | GradCAM, consensus analysis | Brain saliency maps, consensus report |
| **Week 4** | Clinical validation, final report | XAI completion report, patient-level explanations |

---

## Dependencies

**Code:**
- SHAP library (`shap>=0.41.0`)
- PyTorch Geometric (`torch-geometric>=2.0`)
- Captum (optional, for alternative GradCAM)
- NetworkX (for graph visualization)
- Matplotlib, Seaborn (visualizations)

**Data:**
- Phase 8.2 trained model weights
- Phase 8.3 trained model weights
- Phase 8.4 trained model weights
- Test set data (PyG Data objects)
- Feature names and metadata

**Phase 7 Assets to Reuse:**
- SHAP wrapper classes (adapt for survival)
- GNNExplainer pipeline (adapt for new models)
- Visualization templates

---

## References

1. **SHAP:** Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.
2. **GNNExplainer:** Ying et al. (2019). "GNNExplainer: Generating Explanations for Graph Neural Networks." *NeurIPS*.
3. **Survival SHAP:** Kovalev et al. (2020). "SurvSHAP: Time-dependent SHAP values for survival models." *arXiv*.
4. **GradCAM:** Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." *ICCV*.

---

**Status:** Framework design complete. Ready for implementation.  
**Next:** Create `shap_survival_analysis.py` for Phase 8.2 Progression model.
