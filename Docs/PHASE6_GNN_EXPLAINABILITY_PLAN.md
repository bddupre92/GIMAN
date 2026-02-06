# Phase 6: GNN Explainability for Clinical Insights - Implementation Plan

## Executive Summary

**Goal**: Make GIMAN's graph neural network predictions interpretable for clinicians by explaining which patient connections, features, and graph structures drive prognostic predictions.

**Expected Impact**:
- Identify "progression twins" (similar patients with shared trajectories)
- Explain WHY specific patients are predicted to progress faster/slower
- Extract clinically actionable insights from GNN attention patterns
- Build trust in AI predictions through transparent explanations

**Timeline**: 4-5 weeks for full implementation

---

## Scientific Background

### The Black Box Problem

**Challenge**: Graph neural networks learn complex patient similarity patterns, but:
- Clinicians can't see WHY a prediction was made
- Can't identify which features drive similarity
- Can't explain which patient connections matter most

**Solution**: Explainability methods that reveal:
1. **Feature importance**: Which clinical/imaging features matter most?
2. **Edge importance**: Which patient connections influence predictions?
3. **Subgraph discovery**: Which patient clusters share progression patterns?
4. **Counterfactual analysis**: "What if this biomarker value changed?"

### Explainability Methods for GNNs

1. **GNNExplainer** (Ying et al., NeurIPS 2019)
   - Identifies important subgraph + feature subset
   - Optimization-based approach

2. **Attention Visualization**
   - Interpret GAT attention weights
   - Shows which patients attend to each other

3. **Integrated Gradients**
   - Attribution method from image domain
   - Adapted for graphs

4. **Counterfactual Explanations**
   - "Minimal change to flip prediction"
   - Clinically actionable recommendations

---

## Phase 6 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 6: GNN EXPLAINABILITY                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────┐
    │  Task 6.1: Attention Weight Visualization          │
    │  - Extract GAT attention patterns                  │
    │  - Identify high-attention patient pairs           │
    │  - Visualize attention heatmaps                    │
    └────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────┐
    │  Task 6.2: GNNExplainer Integration                │
    │  - Explain individual predictions                  │
    │  - Extract important subgraphs                     │
    │  - Identify critical features                      │
    └────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────┐
    │  Task 6.3: Feature Attribution Analysis            │
    │  - Integrated Gradients for feature importance     │
    │  - Permutation feature importance                  │
    │  - SHAP values for GNNs                           │
    └────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────┐
    │  Task 6.4: Patient Similarity Cluster Analysis     │
    │  - Community detection in patient graph            │
    │  - Characterize progression clusters               │
    │  - "Progression twin" identification               │
    └────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────┐
    │  Task 6.5: Counterfactual Explanations             │
    │  - "What if" scenario analysis                     │
    │  - Modifiable risk factor identification           │
    │  - Actionable clinical recommendations             │
    └────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────┐
    │  Task 6.6: Clinical Explanation Dashboard          │
    │  - Interactive visualization tool                  │
    │  - Per-patient explanation reports                 │
    │  - Clinician-friendly summaries                    │
    └────────────────────────────────────────────────────┘
```

---

## Task 6.1: Attention Weight Visualization

### Objective
Extract and visualize GAT attention weights to understand which patient connections the model deems important.

### Attention Mechanism Recap

From GIMAN architecture (Phase 2):
```python
# In GATConv layer
α_ij = attention_weight(patient_i, patient_j)

# Aggregation
h_i = Σ_j α_ij * h_j

# α_ij tells us: "How much does patient i attend to patient j?"
```

### Implementation

**Step 1.1**: Create `task_6_1_attention_visualization.py`

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionAnalyzer:
    """
    Analyze and visualize GAT attention patterns.
    """
    
    def __init__(
        self,
        model: nn.Module,
        data: Data,
        edge_index: torch.Tensor,
        patient_ids: np.ndarray
    ):
        self.model = model
        self.data = data
        self.edge_index = edge_index
        self.patient_ids = patient_ids
        
        # Storage for attention weights
        self.attention_weights = {}
        
        # Register hooks to capture attention
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        """
        Register forward hooks on GAT layers to capture attention weights.
        """
        def get_attention_hook(layer_name):
            def hook(module, input, output):
                # GATConv returns (output, attention_weights) if return_attention_weights=True
                if isinstance(output, tuple) and len(output) == 2:
                    self.attention_weights[layer_name] = output[1].detach().cpu()
            return hook
        
        for idx, layer in enumerate(self.model.gat_layers):
            layer.register_forward_hook(get_attention_hook(f'gat_layer_{idx}'))
    
    def extract_attention_weights(self) -> Dict[str, torch.Tensor]:
        """
        Run forward pass and extract attention weights from all GAT layers.
        
        Returns:
            {
                'gat_layer_0': attention_weights (num_edges, num_heads),
                'gat_layer_1': attention_weights (num_edges, num_heads),
                ...
            }
        """
        self.model.eval()
        with torch.no_grad():
            # Forward pass (with return_attention_weights=True)
            _ = self.model(self.data.x, self.edge_index, return_attention=True)
        
        return self.attention_weights
    
    def create_attention_matrix(
        self, 
        layer_name: str = 'gat_layer_0',
        head_idx: int = 0
    ) -> np.ndarray:
        """
        Convert attention weights to patient × patient matrix.
        
        Args:
            layer_name: Which GAT layer to analyze
            head_idx: Which attention head (if multi-head)
            
        Returns:
            (N_patients, N_patients) attention matrix
        """
        attention = self.attention_weights[layer_name][:, head_idx].numpy()
        edge_index_np = self.edge_index.cpu().numpy()
        
        N = len(self.patient_ids)
        attention_matrix = np.zeros((N, N))
        
        for idx, (src, dst) in enumerate(edge_index_np.T):
            attention_matrix[src, dst] = attention[idx]
        
        return attention_matrix
    
    def plot_attention_heatmap(
        self,
        attention_matrix: np.ndarray,
        top_k: int = 50,
        output_path: Path = None
    ):
        """
        Visualize attention matrix as heatmap.
        
        Args:
            attention_matrix: (N, N) attention weights
            top_k: Show only top K patients (for readability)
        """
        # Select top K patients with highest average attention
        avg_attention = attention_matrix.sum(axis=1)
        top_indices = np.argsort(avg_attention)[-top_k:]
        
        subset_matrix = attention_matrix[top_indices][:, top_indices]
        subset_ids = self.patient_ids[top_indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            subset_matrix,
            xticklabels=subset_ids,
            yticklabels=subset_ids,
            cmap='YlOrRd',
            cbar_kws={'label': 'Attention Weight'},
            ax=ax
        )
        ax.set_title(f'GAT Attention Weights (Top {top_k} Patients)')
        ax.set_xlabel('Target Patient')
        ax.set_ylabel('Source Patient')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def identify_high_attention_pairs(
        self,
        attention_matrix: np.ndarray,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Find patient pairs with high mutual attention.
        
        Returns:
            DataFrame with columns: patient_i, patient_j, attention_i_to_j, 
            attention_j_to_i, bidirectional_score
        """
        high_attention_pairs = []
        
        N = attention_matrix.shape[0]
        for i in range(N):
            for j in range(i + 1, N):  # Upper triangle only
                attn_i_to_j = attention_matrix[i, j]
                attn_j_to_i = attention_matrix[j, i]
                
                # Bidirectional attention score
                bidirectional = (attn_i_to_j + attn_j_to_i) / 2
                
                if bidirectional > threshold:
                    high_attention_pairs.append({
                        'patient_i': self.patient_ids[i],
                        'patient_j': self.patient_ids[j],
                        'attention_i_to_j': attn_i_to_j,
                        'attention_j_to_i': attn_j_to_i,
                        'bidirectional_score': bidirectional
                    })
        
        return pd.DataFrame(high_attention_pairs).sort_values(
            'bidirectional_score', ascending=False
        )
    
    def visualize_patient_neighborhood(
        self,
        patient_id: int,
        k_neighbors: int = 10
    ):
        """
        Visualize attention-weighted neighborhood for specific patient.
        
        Shows which patients this patient attends to most strongly.
        """
        patient_idx = np.where(self.patient_ids == patient_id)[0][0]
        
        # Get attention from this patient to all others
        attention_from_patient = self.create_attention_matrix()[patient_idx]
        
        # Top K neighbors
        top_neighbor_indices = np.argsort(attention_from_patient)[-k_neighbors:]
        top_neighbor_ids = self.patient_ids[top_neighbor_indices]
        top_attention_values = attention_from_patient[top_neighbor_indices]
        
        # Create networkx graph
        G = nx.DiGraph()
        
        # Add central patient
        G.add_node(patient_id, node_type='central')
        
        # Add neighbors
        for neighbor_id, attention in zip(top_neighbor_ids, top_attention_values):
            G.add_node(neighbor_id, node_type='neighbor')
            G.add_edge(patient_id, neighbor_id, weight=attention)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Central node (larger, different color)
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[patient_id],
            node_size=1000,
            node_color='red',
            alpha=0.9,
            ax=ax
        )
        
        # Neighbor nodes
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=top_neighbor_ids,
            node_size=500,
            node_color='lightblue',
            alpha=0.7,
            ax=ax
        )
        
        # Edges (width proportional to attention)
        edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            alpha=0.6,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            ax=ax
        )
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        
        ax.set_title(f'Attention Neighborhood for Patient {patient_id}')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
```

**Deliverables**:
- ✅ `task_6_1_attention_visualization.py` (500 lines)
- ✅ `visualizations/phase6/attention_heatmaps/`
- ✅ `visualizations/phase6/patient_neighborhoods/`
- ✅ `results/phase6/high_attention_pairs.csv`

---

## Task 6.2: GNNExplainer Integration

### Objective
Use GNNExplainer to identify the most important subgraph and features for individual predictions.

### GNNExplainer Method

**Goal**: For a specific patient's prediction, find:
1. **Subgraph**: Which patient connections matter most?
2. **Feature mask**: Which features drive the prediction?

**Optimization problem**:
```
maximize: mutual_information(Y, (G_S, X_F))
subject to: |G_S| ≤ k_edges, |X_F| ≤ k_features

Where:
- Y: Model prediction
- G_S: Explanatory subgraph
- X_F: Explanatory feature subset
```

### Implementation

**Step 2.1**: Create `task_6_2_gnn_explainer.py`

```python
from torch_geometric.explain import Explainer, GNNExplainer

class GIMANExplainer:
    """
    Explain GIMAN predictions using GNNExplainer.
    """
    
    def __init__(
        self,
        model: nn.Module,
        data: Data,
        edge_index: torch.Tensor
    ):
        self.model = model
        self.data = data
        self.edge_index = edge_index
        
        # Initialize GNNExplainer
        self.explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='regression',  # or 'classification' for cognitive task
                task_level='node',
                return_type='raw'
            )
        )
    
    def explain_node_prediction(
        self,
        node_idx: int,
        task: str = 'motor'  # 'motor' or 'cognitive'
    ) -> Dict:
        """
        Explain prediction for single patient.
        
        Returns:
            {
                'node_feat_mask': Tensor of shape (num_features,),
                'edge_mask': Tensor of shape (num_edges,),
                'important_neighbors': List[int],
                'important_features': List[str],
                'prediction': float
            }
        """
        # Get explanation
        explanation = self.explainer(
            x=self.data.x,
            edge_index=self.edge_index,
            index=node_idx
        )
        
        # Extract masks
        node_feat_mask = explanation.node_feat_mask[node_idx].cpu().numpy()
        edge_mask = explanation.edge_mask.cpu().numpy()
        
        # Identify important features (top 5)
        important_feat_indices = np.argsort(node_feat_mask)[-5:]
        important_features = [
            self.data.feature_names[idx] for idx in important_feat_indices
        ]
        
        # Identify important neighbors (edges with high mask values)
        edge_index_np = self.edge_index.cpu().numpy()
        edges_from_node = edge_index_np[0] == node_idx
        neighbor_edge_indices = np.where(edges_from_node)[0]
        
        important_neighbor_edges = neighbor_edge_indices[
            np.argsort(edge_mask[neighbor_edge_indices])[-5:]
        ]
        important_neighbors = edge_index_np[1][important_neighbor_edges].tolist()
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            if task == 'motor':
                prediction = self.model(self.data.x, self.edge_index)[0][node_idx].item()
            else:
                prediction = self.model(self.data.x, self.edge_index)[1][node_idx].item()
        
        return {
            'node_feat_mask': node_feat_mask,
            'edge_mask': edge_mask,
            'important_neighbors': important_neighbors,
            'important_features': important_features,
            'prediction': prediction
        }
    
    def visualize_explanation(
        self,
        node_idx: int,
        explanation: Dict,
        cohort_df: pd.DataFrame,
        output_path: Path = None
    ):
        """
        Visualize GNNExplainer results.
        
        Creates multi-panel figure:
        1. Feature importance bar chart
        2. Explanatory subgraph
        3. Comparison of patient features to important neighbors
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Panel 1: Feature importance
        feature_importance = explanation['node_feat_mask']
        top_features = explanation['important_features']
        top_feature_indices = [
            i for i, name in enumerate(self.data.feature_names) 
            if name in top_features
        ]
        
        axes[0].barh(
            range(len(top_features)),
            feature_importance[top_feature_indices],
            color='steelblue'
        )
        axes[0].set_yticks(range(len(top_features)))
        axes[0].set_yticklabels(top_features)
        axes[0].set_xlabel('Feature Importance')
        axes[0].set_title('Top Contributing Features')
        
        # Panel 2: Explanatory subgraph
        self._plot_explanatory_subgraph(
            axes[1],
            node_idx,
            explanation['important_neighbors'],
            explanation['edge_mask']
        )
        
        # Panel 3: Feature value comparison
        self._plot_feature_comparison(
            axes[2],
            node_idx,
            explanation['important_neighbors'],
            top_features,
            cohort_df
        )
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_text_explanation(
        self,
        node_idx: int,
        explanation: Dict,
        cohort_df: pd.DataFrame
    ) -> str:
        """
        Generate human-readable explanation.
        
        Returns:
            Natural language explanation of prediction
        """
        patient_id = self.data.patient_ids[node_idx]
        prediction = explanation['prediction']
        important_features = explanation['important_features']
        important_neighbors = explanation['important_neighbors']
        
        # Get patient data
        patient_data = cohort_df[cohort_df['PATNO'] == patient_id].iloc[0]
        
        # Get neighbor data
        neighbor_ids = [self.data.patient_ids[idx] for idx in important_neighbors]
        neighbor_data = cohort_df[cohort_df['PATNO'].isin(neighbor_ids)]
        
        # Generate explanation
        explanation_text = f"""
        ## Prediction Explanation for Patient {patient_id}
        
        **Predicted Motor Progression**: {prediction:.2f} UPDRS-III points/year
        
        ### Key Contributing Factors
        
        The model's prediction is primarily driven by:
        
        """
        
        for idx, feature in enumerate(important_features[:3]):
            feature_value = patient_data[feature]
            neighbor_avg = neighbor_data[feature].mean()
            
            comparison = "higher" if feature_value > neighbor_avg else "lower"
            
            explanation_text += f"""
        {idx + 1}. **{feature}**: {feature_value:.2f}
           - This value is {comparison} than similar patients (avg: {neighbor_avg:.2f})
           - {'Indicates faster progression risk' if feature in ['baseline_updrs_iii', 'age'] else 'May indicate slower progression'}
        """
        
        explanation_text += f"""
        
        ### Similar Patients ("Progression Twins")
        
        This patient's progression trajectory closely resembles {len(important_neighbors)} other patients:
        """
        
        for neighbor_idx in important_neighbors[:3]:
            neighbor_id = self.data.patient_ids[neighbor_idx]
            neighbor_info = cohort_df[cohort_df['PATNO'] == neighbor_id].iloc[0]
            
            explanation_text += f"""
        - **Patient {neighbor_id}**: 
          Age {neighbor_info['age_at_baseline']:.0f}, 
          Baseline UPDRS-III {neighbor_info['baseline_updrs_iii']:.1f},
          Actual progression {neighbor_info['motor_slope_pts_per_year']:.2f} pts/yr
        """
        
        return explanation_text
    
    def batch_explain(
        self,
        node_indices: List[int],
        output_dir: Path
    ) -> pd.DataFrame:
        """
        Generate explanations for multiple patients.
        
        Returns:
            DataFrame with aggregated explanation statistics
        """
        explanations = []
        
        for node_idx in tqdm(node_indices, desc="Generating explanations"):
            explanation = self.explain_node_prediction(node_idx)
            
            explanations.append({
                'patient_id': self.data.patient_ids[node_idx],
                'prediction': explanation['prediction'],
                'top_feature_1': explanation['important_features'][0],
                'top_feature_2': explanation['important_features'][1],
                'top_feature_3': explanation['important_features'][2],
                'num_important_neighbors': len(explanation['important_neighbors']),
                'avg_edge_importance': np.mean(explanation['edge_mask'])
            })
        
        return pd.DataFrame(explanations)
```

**Deliverables**:
- ✅ `task_6_2_gnn_explainer.py` (600 lines)
- ✅ `results/phase6/patient_explanations/` (individual explanation reports)
- ✅ `results/phase6/explanation_summary.csv`
- ✅ `visualizations/phase6/explanation_visualizations/`

---

## Task 6.3: Feature Attribution Analysis

### Objective
Quantify the contribution of each feature to model predictions using gradient-based methods.

### Methods

**1. Integrated Gradients**
```python
from captum.attr import IntegratedGradients

def compute_integrated_gradients(
    model: nn.Module,
    input_features: torch.Tensor,
    edge_index: torch.Tensor,
    baseline: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute integrated gradients for GNN.
    
    IG measures importance by integrating gradients along path 
    from baseline to input.
    """
    if baseline is None:
        baseline = torch.zeros_like(input_features)
    
    ig = IntegratedGradients(model)
    
    attributions = ig.attribute(
        input_features,
        baselines=baseline,
        additional_forward_args=(edge_index,),
        n_steps=50
    )
    
    return attributions
```

**2. SHAP for GNNs**
```python
import shap

class GNN_SHAP:
    """
    SHAP (SHapley Additive exPlanations) adapted for GNNs.
    """
    
    def __init__(self, model: nn.Module, background_data: Data):
        self.model = model
        self.background = background_data
        
    def compute_shap_values(
        self,
        patient_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> np.ndarray:
        """
        Compute SHAP values for patient features.
        
        Returns:
            (num_patients, num_features) SHAP value matrix
        """
        # Define prediction function
        def predict_fn(features):
            features_tensor = torch.FloatTensor(features)
            with torch.no_grad():
                predictions = self.model(features_tensor, edge_index)[0]
            return predictions.numpy()
        
        # Initialize SHAP explainer
        explainer = shap.KernelExplainer(
            predict_fn,
            self.background.x.numpy()
        )
        
        # Compute SHAP values
        shap_values = explainer.shap_values(patient_features.numpy())
        
        return shap_values
    
    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        output_path: Path = None
    ):
        """
        Create SHAP summary plot.
        
        Shows feature importance and impact on predictions.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        shap.summary_plot(
            shap_values,
            feature_names=feature_names,
            show=False
        )
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
```

**Implementation**: Create `task_6_3_feature_attribution.py`

**Deliverables**:
- ✅ `task_6_3_feature_attribution.py` (450 lines)
- ✅ `results/phase6/feature_attributions.csv`
- ✅ `visualizations/phase6/shap_summary_plot.png`
- ✅ `visualizations/phase6/integrated_gradients_heatmap.png`

---

## Task 6.4: Patient Similarity Cluster Analysis

### Objective
Discover patient communities in the graph and characterize their progression patterns.

### Community Detection Methods

```python
import networkx as nx
from sklearn.cluster import SpectralClustering
import community as community_louvain  # python-louvain

class PatientCommunityAnalyzer:
    """
    Detect and analyze patient communities in similarity graph.
    """
    
    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
        patient_ids: np.ndarray,
        cohort_df: pd.DataFrame
    ):
        self.edge_index = edge_index
        self.edge_weights = edge_weights
        self.patient_ids = patient_ids
        self.cohort_df = cohort_df
        
        # Create NetworkX graph
        self.G = self._create_networkx_graph()
    
    def _create_networkx_graph(self) -> nx.Graph:
        """Convert PyG graph to NetworkX."""
        G = nx.Graph()
        
        edge_index_np = self.edge_index.cpu().numpy()
        edge_weights_np = self.edge_weights.cpu().numpy()
        
        for idx, (src, dst) in enumerate(edge_index_np.T):
            G.add_edge(
                self.patient_ids[src],
                self.patient_ids[dst],
                weight=edge_weights_np[idx]
            )
        
        return G
    
    def detect_communities_louvain(self) -> Dict[int, int]:
        """
        Detect communities using Louvain algorithm.
        
        Returns:
            {patient_id: community_id}
        """
        partition = community_louvain.best_partition(self.G, weight='weight')
        return partition
    
    def detect_communities_spectral(self, n_clusters: int = 5) -> Dict[int, int]:
        """
        Spectral clustering on graph Laplacian.
        """
        # Get adjacency matrix
        adj_matrix = nx.to_scipy_sparse_array(self.G, weight='weight')
        
        # Spectral clustering
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        
        labels = clustering.fit_predict(adj_matrix)
        
        partition = {
            patient_id: label 
            for patient_id, label in zip(self.patient_ids, labels)
        }
        
        return partition
    
    def characterize_communities(
        self,
        partition: Dict[int, int]
    ) -> pd.DataFrame:
        """
        Compare clinical/imaging features across communities.
        
        Returns:
            DataFrame with community profiles
        """
        # Add community labels to cohort data
        cohort_with_communities = self.cohort_df.copy()
        cohort_with_communities['community'] = cohort_with_communities['PATNO'].map(partition)
        
        # Aggregate statistics per community
        community_profiles = []
        
        for community_id in sorted(partition.values()):
            community_data = cohort_with_communities[
                cohort_with_communities['community'] == community_id
            ]
            
            profile = {
                'community_id': community_id,
                'n_patients': len(community_data),
                'mean_motor_slope': community_data['motor_slope_pts_per_year'].mean(),
                'std_motor_slope': community_data['motor_slope_pts_per_year'].std(),
                'mean_baseline_updrs': community_data['baseline_updrs_iii'].mean(),
                'mean_age': community_data['age_at_baseline'].mean(),
                'pct_cognitive_decline': community_data['cognitive_decline_binary'].mean() * 100,
                'mean_disease_duration': community_data['disease_duration_years'].mean()
            }
            
            community_profiles.append(profile)
        
        return pd.DataFrame(community_profiles)
    
    def find_progression_twins(
        self,
        patient_id: int,
        same_community: bool = True,
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Find patients with most similar progression to target patient.
        
        "Progression twins" = similar baseline + similar trajectory
        
        Returns:
            DataFrame with top K similar patients
        """
        # Get target patient data
        target_data = self.cohort_df[self.cohort_df['PATNO'] == patient_id].iloc[0]
        
        # Get community if filtering
        if same_community:
            partition = self.detect_communities_louvain()
            target_community = partition[patient_id]
            candidate_patients = [
                pid for pid, comm in partition.items() 
                if comm == target_community and pid != patient_id
            ]
        else:
            candidate_patients = [
                pid for pid in self.patient_ids if pid != patient_id
            ]
        
        # Calculate similarity scores
        similarities = []
        
        for candidate_id in candidate_patients:
            candidate_data = self.cohort_df[self.cohort_df['PATNO'] == candidate_id].iloc[0]
            
            # Feature-wise similarity
            baseline_updrs_diff = abs(
                target_data['baseline_updrs_iii'] - candidate_data['baseline_updrs_iii']
            )
            motor_slope_diff = abs(
                target_data['motor_slope_pts_per_year'] - candidate_data['motor_slope_pts_per_year']
            )
            age_diff = abs(
                target_data['age_at_baseline'] - candidate_data['age_at_baseline']
            )
            
            # Composite similarity score (lower = more similar)
            similarity_score = (
                baseline_updrs_diff / 10 +  # Normalize
                motor_slope_diff / 2 +
                age_diff / 20
            )
            
            similarities.append({
                'patient_id': candidate_id,
                'similarity_score': similarity_score,
                'baseline_updrs_iii': candidate_data['baseline_updrs_iii'],
                'motor_slope': candidate_data['motor_slope_pts_per_year'],
                'age': candidate_data['age_at_baseline']
            })
        
        # Sort and return top K
        similarities_df = pd.DataFrame(similarities).sort_values('similarity_score')
        
        return similarities_df.head(top_k)
    
    def visualize_community_graph(
        self,
        partition: Dict[int, int],
        output_path: Path = None
    ):
        """
        Visualize patient graph colored by community.
        """
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Layout
        pos = nx.spring_layout(self.G, k=0.5, iterations=50)
        
        # Color by community
        unique_communities = sorted(set(partition.values()))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
        
        for community_id, color in zip(unique_communities, colors):
            community_nodes = [
                node for node, comm in partition.items() 
                if comm == community_id
            ]
            
            nx.draw_networkx_nodes(
                self.G, pos,
                nodelist=community_nodes,
                node_size=100,
                node_color=[color],
                alpha=0.8,
                label=f'Community {community_id}',
                ax=ax
            )
        
        # Draw edges (light gray)
        nx.draw_networkx_edges(
            self.G, pos,
            alpha=0.1,
            edge_color='gray',
            ax=ax
        )
        
        ax.set_title('Patient Similarity Graph with Communities')
        ax.legend()
        ax.axis('off')
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
```

**Deliverables**:
- ✅ `task_6_4_patient_communities.py` (500 lines)
- ✅ `results/phase6/community_assignments.csv`
- ✅ `results/phase6/community_profiles.csv`
- ✅ `results/phase6/progression_twins.csv`
- ✅ `visualizations/phase6/community_graph.png`

---

## Task 6.5: Counterfactual Explanations

### Objective
Answer "What if?" questions: What minimal changes to features would alter the prediction?

### Counterfactual Generation

```python
class CounterfactualGenerator:
    """
    Generate counterfactual explanations for GNN predictions.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def generate_counterfactual(
        self,
        patient_features: torch.Tensor,
        edge_index: torch.Tensor,
        target_change: float = -1.0,  # Reduce progression by 1 pt/yr
        max_features_to_change: int = 3,
        feature_constraints: Dict = None
    ) -> Dict:
        """
        Find minimal feature changes to achieve target prediction change.
        
        Args:
            patient_features: Original features (1, num_features)
            target_change: Desired change in prediction
            max_features_to_change: Limit changes to K features
            feature_constraints: Which features are modifiable
                e.g., {'baseline_updrs_iii': (0, 30), 'age': None}  # None = non-modifiable
        
        Returns:
            {
                'original_prediction': float,
                'counterfactual_prediction': float,
                'feature_changes': Dict[str, Tuple[float, float]],  # {feature: (original, new)}
                'feasibility_score': float  # How realistic are the changes?
            }
        """
        original_pred = self.model(patient_features, edge_index)[0].item()
        target_pred = original_pred + target_change
        
        # Initialize counterfactual (copy of original)
        counterfactual = patient_features.clone().detach().requires_grad_(True)
        
        # Optimization to find counterfactual
        optimizer = torch.optim.Adam([counterfactual], lr=0.01)
        
        for iteration in range(500):
            optimizer.zero_grad()
            
            # Prediction with counterfactual features
            cf_pred = self.model(counterfactual, edge_index)[0]
            
            # Loss: distance to target + L1 penalty for changes
            prediction_loss = (cf_pred - target_pred) ** 2
            sparsity_loss = torch.sum(torch.abs(counterfactual - patient_features))
            
            loss = prediction_loss + 0.1 * sparsity_loss
            
            loss.backward()
            optimizer.step()
            
            # Apply constraints
            with torch.no_grad():
                for feat_idx, (feat_name, constraint) in enumerate(feature_constraints.items()):
                    if constraint is None:
                        # Non-modifiable: revert to original
                        counterfactual[0, feat_idx] = patient_features[0, feat_idx]
                    else:
                        # Clip to valid range
                        min_val, max_val = constraint
                        counterfactual[0, feat_idx] = torch.clamp(
                            counterfactual[0, feat_idx], min_val, max_val
                        )
            
            # Check convergence
            if iteration % 100 == 0:
                with torch.no_grad():
                    current_pred = self.model(counterfactual, edge_index)[0].item()
                    print(f"Iter {iteration}: Pred = {current_pred:.2f} (target: {target_pred:.2f})")
        
        # Extract changes
        changes = {}
        for feat_idx, feat_name in enumerate(self.model.feature_names):
            original_val = patient_features[0, feat_idx].item()
            new_val = counterfactual[0, feat_idx].item()
            
            if abs(new_val - original_val) > 0.01:
                changes[feat_name] = (original_val, new_val)
        
        # Feasibility score (based on magnitude of changes)
        feasibility = self._calculate_feasibility(changes)
        
        return {
            'original_prediction': original_pred,
            'counterfactual_prediction': self.model(counterfactual, edge_index)[0].item(),
            'feature_changes': changes,
            'feasibility_score': feasibility
        }
    
    def _calculate_feasibility(self, changes: Dict) -> float:
        """
        Score feasibility of counterfactual changes.
        
        Returns:
            0.0-1.0, where 1.0 = highly feasible
        """
        # Example: Large changes are less feasible
        total_change = sum([
            abs(new - orig) / orig if orig != 0 else abs(new - orig)
            for orig, new in changes.values()
        ])
        
        # Sigmoid to map to [0, 1]
        feasibility = 1 / (1 + np.exp(total_change - 2))
        
        return feasibility
    
    def generate_actionable_recommendations(
        self,
        counterfactual_result: Dict,
        modifiable_features: List[str] = ['baseline_updrs_iii', 'exercise_frequency']
    ) -> str:
        """
        Convert counterfactual to actionable clinical recommendations.
        
        Returns:
            Human-readable recommendations
        """
        recommendations = []
        
        for feature, (original, new) in counterfactual_result['feature_changes'].items():
            if feature not in modifiable_features:
                continue
            
            change = new - original
            change_pct = (change / original * 100) if original != 0 else 0
            
            if feature == 'baseline_updrs_iii':
                if change < 0:
                    recommendations.append(
                        f"Reducing motor symptoms by {abs(change):.1f} points "
                        f"could slow progression by {abs(counterfactual_result['counterfactual_prediction'] - counterfactual_result['original_prediction']):.2f} pts/year"
                    )
            elif feature == 'exercise_frequency':
                if change > 0:
                    recommendations.append(
                        f"Increasing exercise frequency by {abs(change_pct):.0f}% "
                        f"may reduce progression rate"
                    )
        
        if not recommendations:
            return "No actionable modifiable factors identified."
        
        return "\n".join([f"• {rec}" for rec in recommendations])
```

**Deliverables**:
- ✅ `task_6_5_counterfactual_explanations.py` (450 lines)
- ✅ `results/phase6/counterfactual_scenarios.csv`
- ✅ `results/phase6/actionable_recommendations.txt`

---

## Task 6.6: Clinical Explanation Dashboard

### Objective
Create interactive tool for clinicians to explore model explanations.

### Dashboard Components

```python
import streamlit as st
import plotly.graph_objects as go

def create_explanation_dashboard():
    """
    Interactive Streamlit dashboard for model explanations.
    """
    st.set_page_config(page_title="GIMAN Explainability", layout="wide")
    
    st.title("🧠 GIMAN Model Explanations")
    st.markdown("Interactive tool to understand PD progression predictions")
    
    # Sidebar: Patient selection
    st.sidebar.header("Patient Selection")
    patient_id = st.sidebar.selectbox(
        "Select Patient ID",
        options=patient_ids
    )
    
    # Load patient data
    patient_data = load_patient_data(patient_id)
    explanation = generate_explanation(patient_id)
    
    # Main dashboard (3 columns)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("📊 Prediction")
        st.metric(
            "Motor Progression",
            f"{explanation['prediction']:.2f} pts/year",
            delta=f"{explanation['prediction'] - cohort_mean:.2f} vs cohort"
        )
        st.metric(
            "Cognitive Decline Risk",
            f"{explanation['cognitive_risk']:.1%}"
        )
    
    with col2:
        st.header("🔑 Key Features")
        # Feature importance chart
        fig_features = create_feature_importance_chart(explanation)
        st.plotly_chart(fig_features, use_container_width=True)
    
    with col3:
        st.header("👥 Progression Twins")
        twins_df = explanation['progression_twins']
        st.dataframe(twins_df)
    
    # Tabs for detailed explanations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Trajectory",
        "🕸️ Similarity Network",
        "💡 Counterfactuals",
        "📄 Clinical Report"
    ])
    
    with tab1:
        st.subheader("Predicted Trajectory vs Similar Patients")
        fig_trajectory = create_trajectory_plot(patient_id, explanation)
        st.plotly_chart(fig_trajectory, use_container_width=True)
    
    with tab2:
        st.subheader("Patient Similarity Network")
        fig_network = create_network_visualization(patient_id, explanation)
        st.plotly_chart(fig_network, use_container_width=True)
    
    with tab3:
        st.subheader("What-If Scenarios")
        
        # Interactive counterfactual generator
        st.markdown("**Adjust features to see impact on prediction:**")
        
        baseline_updrs = st.slider(
            "Baseline UPDRS-III",
            0, 50,
            int(patient_data['baseline_updrs_iii'])
        )
        
        age = st.slider(
            "Age at Baseline",
            40, 85,
            int(patient_data['age_at_baseline'])
        )
        
        if st.button("Generate Counterfactual"):
            counterfactual = generate_counterfactual(
                patient_id,
                modified_features={'baseline_updrs_iii': baseline_updrs, 'age': age}
            )
            
            st.write(f"**New Prediction:** {counterfactual['new_prediction']:.2f} pts/year")
            st.write(f"**Change:** {counterfactual['new_prediction'] - explanation['prediction']:.2f}")
    
    with tab4:
        st.subheader("Clinical Report")
        
        # Generate PDF report
        if st.button("Generate PDF Report"):
            report_pdf = generate_clinical_report_pdf(patient_id, explanation)
            st.download_button(
                "Download Report",
                data=report_pdf,
                file_name=f"patient_{patient_id}_explanation_report.pdf",
                mime="application/pdf"
            )
        
        # Display text report
        st.markdown(explanation['text_report'])
```

**Deliverables**:
- ✅ `task_6_6_explanation_dashboard.py` (700 lines)
- ✅ `app/giman_explainability_dashboard.py` (Streamlit app)
- ✅ `results/phase6/clinical_reports/` (PDF reports)

---

## Phase 6 Summary

### Deliverables Checklist

**Code Files** (6 tasks):
- [ ] `task_6_1_attention_visualization.py` (500 lines)
- [ ] `task_6_2_gnn_explainer.py` (600 lines)
- [ ] `task_6_3_feature_attribution.py` (450 lines)
- [ ] `task_6_4_patient_communities.py` (500 lines)
- [ ] `task_6_5_counterfactual_explanations.py` (450 lines)
- [ ] `task_6_6_explanation_dashboard.py` (700 lines)

**Total Code**: ~3,200 lines

**Visualizations**:
- Attention heatmaps
- GNNExplainer subgraphs
- SHAP summary plots
- Community graphs
- Counterfactual scenarios
- Interactive dashboard

**Documentation**:
- `Docs/PHASE6_EXPLAINABILITY_REPORT.md`
- `PHASE6_COMPLETION_SUMMARY.md`

### Clinical Impact

1. **Trust & Adoption**: Transparent predictions clinicians can understand

2. **Personalized Insights**: "Progression twins" for patient counseling

3. **Actionable Recommendations**: Modifiable risk factors identified

4. **Educational Tool**: Helps clinicians learn PD progression patterns

### Publication Target

- **Journal**: *Nature Machine Intelligence* or *npj Digital Medicine*
- **Focus**: Novel explainability methods for medical GNNs
- **Impact**: Bridge AI and clinical practice

---

## Timeline

**Week 1**: Task 6.1 (Attention) + Task 6.2 (GNNExplainer)
**Week 2**: Task 6.3 (Attribution) + Task 6.4 (Communities)
**Week 3**: Task 6.5 (Counterfactuals)
**Week 4**: Task 6.6 (Dashboard) + Integration
**Week 5**: User testing with clinicians + refinements

---

## Integration with Phases 4-5

**Phase 4 Synergy**: Explain subtype predictions
- "Why is this patient classified as fast progressor?"
- "Which features distinguish subtypes?"

**Phase 5 Synergy**: Explain prodromal conversion predictions
- "What drives this patient's high conversion risk?"
- "Which biomarkers need monitoring?"

**Complete Pipeline**:
```
Phase 2: Prognostic Prediction
  ↓
Phase 4: Subtype Discovery
  ↓
Phase 5: Prodromal Conversion
  ↓
Phase 6: Explain Everything! 🎉
```

---

**All three research directions planned! Ready to start with Phase 4, Task 4.1?** 🚀
