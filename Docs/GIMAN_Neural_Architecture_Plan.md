# GIMAN Neural Architecture Plan
**Graph-Informed Multimodal Attention Network for Parkinson's Disease Classification**

---

## Executive Summary

This document outlines the comprehensive architecture plan for the Graph-Informed Multimodal Attention Network (GIMAN), designed to leverage patient similarity graphs and multimodal biomarker data for Parkinson's Disease classification. Building upon our completed preprocessing pipeline with 557 patients and robust similarity graph structure, GIMAN will implement state-of-the-art Graph Neural Network techniques combined with attention mechanisms for interpretable and accurate PD classification.

**Key Architecture Components:**
- Graph Neural Network backbone for patient similarity propagation
- Multimodal attention mechanisms for biomarker feature weighting
- Graph-level and node-level representation learning
- Interpretable classification with biomarker importance analysis

---

## 1. Input Data Architecture

### 1.1 Patient Similarity Graph
- **Nodes**: 557 patients (PD and Healthy Controls)
- **Edges**: 44,274 similarity connections (density = 0.2859)
- **Node Features**: 7 standardized biomarker features per patient
- **Graph Properties**: Strong community structure (Q = 0.512, 3 communities)

### 1.2 Biomarker Feature Matrix
```
Feature Matrix: [557 patients × 7 biomarkers]
- LRRK2: Genetic risk variant (binary)
- GBA: Genetic risk variant (binary)  
- APOE_RISK: APOE risk score (continuous)
- PTAU: CSF phosphorylated tau (continuous)
- TTAU: CSF total tau (continuous)
- UPSIT_TOTAL: Smell test score (continuous)
- ALPHA_SYN: CSF alpha-synuclein (continuous)
```

### 1.3 Target Labels
- **Binary Classification**: PD (1) vs Healthy Control (0)
- **Cohort Distribution**: Balanced representation across communities
- **Evaluation Strategy**: Stratified splits preserving cohort balance

---

## 2. GIMAN Neural Architecture

### 2.1 Overall Architecture Flow
```
Input Graph + Features → Graph Embedding → Multimodal Attention → Classification
```

### 2.2 Core Components

#### A. Graph Embedding Layer (GNN Backbone)
```python
class GIMANGraphEmbedding(nn.Module):
    """
    Graph Neural Network backbone for patient similarity propagation
    """
    - Input: Patient similarity graph + node features [557 × 7]
    - GraphConv Layers: 2-3 layers with residual connections
    - Hidden Dimensions: [7 → 64 → 128 → 64]
    - Activation: ReLU with dropout (0.2)
    - Output: Node embeddings [557 × 64]
```

#### B. Multimodal Attention Module
```python
class MultimodalAttention(nn.Module):
    """
    Attention mechanism for biomarker feature importance weighting
    """
    - Input: Node embeddings [557 × 64] + Original features [557 × 7]
    - Attention Types:
      * Self-attention across biomarker features
      * Cross-attention between graph embeddings and raw features
      * Temporal attention (if longitudinal data available)
    - Output: Attended feature representations [557 × 64]
```

#### C. Graph-Level Aggregation
```python
class GraphLevelPooling(nn.Module):
    """
    Aggregate node-level representations to graph-level
    """
    - Pooling Strategies:
      * Global attention pooling (primary)
      * Global mean/max pooling (auxiliary)
      * Graph-level readout with learned aggregation
    - Output: Graph-level representation [1 × 64]
```

#### D. Classification Head
```python
class GIMANClassifier(nn.Module):
    """
    Final classification with interpretability features
    """
    - Input: Graph-level representation [1 × 64]
    - Architecture: [64 → 32 → 16 → 1]
    - Output: PD probability + attention weights for interpretation
```

---

## 3. Detailed Layer Specifications

### 3.1 Graph Convolutional Layers

#### Layer 1: Input Feature Transformation
- **Input**: Raw biomarker features [557 × 7]
- **GraphConv**: GCNConv(7, 64) 
- **Activation**: ReLU
- **Normalization**: BatchNorm1d
- **Dropout**: 0.2

#### Layer 2: Graph Information Propagation
- **Input**: First layer embeddings [557 × 64]
- **GraphConv**: GCNConv(64, 128)
- **Residual Connection**: Skip connection from input
- **Activation**: ReLU
- **Normalization**: BatchNorm1d
- **Dropout**: 0.3

#### Layer 3: Feature Refinement
- **Input**: Second layer embeddings [557 × 128]
- **GraphConv**: GCNConv(128, 64)
- **Residual Connection**: Skip connection to Layer 1
- **Activation**: ReLU
- **Output**: Final node embeddings [557 × 64]

### 3.2 Attention Mechanism Details

#### Self-Attention for Biomarker Features
```python
# Attention across biomarker dimensions
Q = node_features @ W_q  # [557 × 64]
K = node_features @ W_k  # [557 × 64] 
V = node_features @ W_v  # [557 × 64]

attention_weights = softmax(Q @ K.T / sqrt(d_k))
attended_features = attention_weights @ V
```

#### Cross-Modal Attention
```python
# Attention between graph embeddings and raw features
graph_query = graph_embeddings @ W_gq
feature_key = raw_features @ W_fk
feature_value = raw_features @ W_fv

cross_attention = softmax(graph_query @ feature_key.T)
enhanced_embeddings = cross_attention @ feature_value
```

### 3.3 Aggregation Strategy

#### Global Attention Pooling
```python
# Learn importance weights for each patient node
node_scores = MLP(node_embeddings)  # [557 × 1]
attention_weights = softmax(node_scores)
graph_representation = sum(attention_weights * node_embeddings)  # [64]
```

---

## 4. Training Strategy

### 4.1 Loss Function Design
```python
# Multi-component loss function
total_loss = classification_loss + attention_regularization + graph_regularization

# Primary: Binary cross-entropy for PD classification
classification_loss = BCEWithLogitsLoss(predictions, labels)

# Secondary: Attention sparsity regularization
attention_regularization = λ₁ * L1_penalty(attention_weights)

# Tertiary: Graph smoothness regularization
graph_regularization = λ₂ * graph_laplacian_regularization(embeddings, adjacency)
```

### 4.2 Training Configuration
- **Optimizer**: Adam with learning rate scheduling
- **Learning Rate**: Initial 0.001, ReduceLROnPlateau
- **Batch Size**: Full-graph (557 patients) - graph-level training
- **Epochs**: 200 with early stopping (patience=20)
- **Regularization**: L2 weight decay (0.0001)

### 4.3 Cross-Validation Strategy
```python
# Stratified K-fold preserving cohort and community structure
n_folds = 5
stratify_by = ['cohort', 'community_id']  # Ensure balanced splits
validation_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

## 5. Evaluation Metrics

### 5.1 Classification Performance
- **Primary**: AUC-ROC, AUC-PR
- **Secondary**: Accuracy, Precision, Recall, F1-Score
- **Statistical**: 95% confidence intervals, permutation tests

### 5.2 Model Interpretation
- **Attention Weights**: Biomarker importance ranking
- **Node Importance**: Patient similarity contribution analysis
- **Community Analysis**: Performance across graph communities
- **Feature Attribution**: SHAP values for model explanation

### 5.3 Graph-Specific Metrics
- **Graph Classification Accuracy**: Performance on graph-level task
- **Node Embedding Quality**: Silhouette score for community separation
- **Attention Consistency**: Reproducibility of attention patterns

---

## 6. Implementation Plan

### 6.1 Development Phases

#### Phase 1: Core GNN Implementation (Week 1)
- [ ] Implement basic GraphConv layers with PyTorch Geometric
- [ ] Create node feature embedding pipeline
- [ ] Implement graph data loading and batching
- [ ] Basic forward pass and gradient flow testing

#### Phase 2: Attention Mechanisms (Week 2)  
- [ ] Implement self-attention for biomarker features
- [ ] Add cross-modal attention between graph and raw features
- [ ] Global attention pooling for graph-level representation
- [ ] Attention weight visualization and interpretation

#### Phase 3: Training Infrastructure (Week 3)
- [ ] Training loop with multi-component loss function
- [ ] Cross-validation framework with stratified splits
- [ ] Model checkpointing and early stopping
- [ ] Hyperparameter optimization with Optuna

#### Phase 4: Evaluation and Interpretation (Week 4)
- [ ] Comprehensive evaluation metrics calculation
- [ ] Attention pattern analysis and visualization
- [ ] Model interpretation with SHAP and feature attribution
- [ ] Performance comparison with baseline models

### 6.2 File Structure
```
src/giman_pipeline/modeling/
├── giman_model.py           # Main GIMAN model implementation
├── graph_layers.py          # Custom graph convolution layers  
├── attention_modules.py     # Multi-modal attention mechanisms
├── pooling.py              # Graph-level pooling strategies
└── utils.py                # Model utilities and helpers

src/giman_pipeline/training/
├── training_pipeline.py    # Main training orchestration
├── data_loaders.py         # Graph data loading for PyTorch Geometric
├── loss_functions.py       # Multi-component loss implementations
├── metrics.py              # Evaluation metrics and analysis
└── interpretation.py       # Model explanation and visualization
```

---

## 7. Expected Outcomes

### 7.1 Model Performance Targets
- **AUC-ROC**: Target ≥ 0.85 (significantly better than random 0.5)
- **Accuracy**: Target ≥ 80% with balanced precision/recall
- **Interpretability**: Clear biomarker importance ranking
- **Robustness**: Consistent performance across CV folds

### 7.2 Scientific Contributions
- **Graph-Based PD Classification**: Novel application of GNNs to patient similarity
- **Multimodal Biomarker Integration**: Attention-based feature fusion
- **Interpretable AI**: Explainable biomarker importance for clinical insight
- **Community-Aware Learning**: Leveraging patient similarity communities

### 7.3 Technical Innovations
- **Patient Similarity GNN**: Graph construction from biomarker profiles
- **Multi-Scale Attention**: Node-level and graph-level attention mechanisms  
- **Biomarker Cross-Attention**: Integration of raw features with graph embeddings
- **Community-Stratified Validation**: Evaluation respecting graph structure

---

## 8. Risk Mitigation

### 8.1 Technical Risks
- **Overfitting**: Addressed through dropout, regularization, and cross-validation
- **Graph Quality**: Validated through community detection and modularity analysis
- **Attention Collapse**: Prevented through attention regularization and monitoring
- **Computational Complexity**: Optimized through efficient PyTorch Geometric operations

### 8.2 Data Risks  
- **Sample Size**: 557 patients provides adequate power for deep learning
- **Class Imbalance**: Stratified sampling ensures balanced training
- **Feature Quality**: Comprehensive imputation and validation completed
- **Graph Connectivity**: Dense graph (28.6%) ensures good information propagation

---

## 9. Success Metrics

### 9.1 Technical Success Criteria
- [ ] Model converges stably during training
- [ ] Cross-validation performance exceeds baseline methods
- [ ] Attention weights provide interpretable biomarker rankings
- [ ] Graph communities show distinct classification patterns

### 9.2 Scientific Success Criteria  
- [ ] Identified biomarkers align with clinical PD knowledge
- [ ] Model provides novel insights into PD progression patterns
- [ ] Graph structure reveals meaningful patient similarities
- [ ] Interpretability enables clinical decision support

---

## 10. Future Extensions

### 10.1 Model Enhancements
- **Temporal GNNs**: Incorporate longitudinal patient visits
- **Hierarchical Attention**: Multi-level attention across biomarker types
- **Graph Transformers**: Replace GCN with graph transformer architecture
- **Multi-Task Learning**: Joint prediction of PD subtypes and progression

### 10.2 Data Integration
- **Imaging Modalities**: Integration of MRI and DaTscan imaging features
- **Clinical Notes**: Natural language processing of clinical assessments
- **Genetic Variants**: Expanded genetic risk profiling
- **Environmental Factors**: Non-motor symptom and lifestyle integration

---

## Conclusion

The GIMAN architecture represents a novel and comprehensive approach to Parkinson's Disease classification that leverages the power of Graph Neural Networks combined with attention mechanisms for interpretable, accurate, and clinically relevant predictions. Built upon our robust preprocessing pipeline with 557 patients and strong similarity graph structure, GIMAN is positioned to make significant contributions to both the technical ML community and clinical PD research.

The modular design ensures extensibility while the attention mechanisms provide the interpretability crucial for clinical adoption. With careful implementation following this architectural plan, GIMAN has the potential to advance the state-of-the-art in graph-based biomedical machine learning and provide valuable insights for Parkinson's Disease research and diagnosis.

---

**Document Version**: 1.0  
**Date**: September 22, 2025  
**Authors**: GIMAN Development Team  
**Status**: Architecture Design Complete - Ready for Implementation