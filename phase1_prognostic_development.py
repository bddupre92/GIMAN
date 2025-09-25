#!/usr/bin/env python3
"""
Phase 1 Prognostic GIMAN Development - Motor Progression & Cognitive Decline
===========================================================================

This script implements Phase 1 of the GIMAN prognostic development plan:
1. Motor progression regression (UPDRS slope prediction)
2. Cognitive decline classification (MCI/dementia conversion)

Author: Blair Dupre
Date: September 24, 2025
Enhanced GIMAN v1.1.0 → Prognostic Phase 1
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, classification_report

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

print("🚀 Phase 1 Prognostic GIMAN Development - Starting Analysis")
print("=" * 70)

# =============================================================================
# 1. DATA LOADING AND EXPLORATION
# =============================================================================

def load_enhanced_giman_data() -> pd.DataFrame:
    """Load the enhanced GIMAN dataset with 12 features."""
    try:
        enhanced_path = "data/enhanced/enhanced_giman_12features_v1.1.0_20250924_075919.csv"
        df = pd.read_csv(enhanced_path)
        print(f"✅ Enhanced GIMAN data loaded: {df.shape[0]} records, {df.shape[1]} features")
        print(f"   Unique patients: {df['PATNO'].nunique()}")
        return df
    except FileNotFoundError:
        print("❌ Enhanced GIMAN file not found. Please verify path.")
        return pd.DataFrame()

def load_longitudinal_data() -> pd.DataFrame:
    """Load the full longitudinal PPMI dataset."""
    try:
        longitudinal_path = "data/01_processed/giman_corrected_longitudinal_dataset.csv"
        df = pd.read_csv(longitudinal_path)
        print(f"✅ Longitudinal data loaded: {df.shape[0]} records, {df.shape[1]} features")
        print(f"   Unique patients: {df['PATNO'].nunique()}")
        return df
    except FileNotFoundError:
        print("❌ Longitudinal file not found. Please verify path.")
        return pd.DataFrame()

def explore_progression_data(enhanced_df: pd.DataFrame, longitudinal_df: pd.DataFrame) -> None:
    """Explore motor and cognitive progression patterns."""
    print("\n📊 PROGRESSION DATA EXPLORATION")
    print("-" * 50)
    
    # Get enhanced model patients
    enhanced_patients = set(enhanced_df['PATNO'].unique())
    
    # Filter longitudinal data to enhanced patients
    enhanced_longitudinal = longitudinal_df[
        longitudinal_df['PATNO'].isin(enhanced_patients)
    ].copy()
    
    # Motor progression analysis
    motor_data = enhanced_longitudinal[enhanced_longitudinal['NP3TOT'].notna()]
    motor_patients = motor_data.groupby('PATNO').size()
    motor_sufficient = motor_patients[motor_patients >= 3]
    
    print(f"Motor Progression Analysis:")
    print(f"  • Enhanced patients in longitudinal data: {enhanced_longitudinal['PATNO'].nunique()}")
    print(f"  • Patients with motor data: {motor_data['PATNO'].nunique()}")
    print(f"  • Patients with ≥3 motor visits: {len(motor_sufficient)}")
    print(f"  • Average motor visits per patient: {motor_patients.mean():.1f}")
    
    # Cognitive progression analysis
    cognitive_data = enhanced_longitudinal[enhanced_longitudinal['MCATOT'].notna()]
    cognitive_patients = cognitive_data.groupby('PATNO').size()
    cognitive_sufficient = cognitive_patients[cognitive_patients >= 3]
    
    print(f"\nCognitive Progression Analysis:")
    print(f"  • Patients with cognitive data: {cognitive_data['PATNO'].nunique()}")
    print(f"  • Patients with ≥3 cognitive visits: {len(cognitive_sufficient)}")
    print(f"  • Average cognitive visits per patient: {cognitive_patients.mean():.1f}")
    
    # Show sample progression patterns
    print(f"\n📈 Sample Motor Progression Patterns:")
    sample_patients = motor_sufficient.head(5).index
    for patient in sample_patients:
        patient_data = motor_data[motor_data['PATNO'] == patient].sort_values('EVENT_ID')
        scores = patient_data['NP3TOT'].tolist()
        visits = len(scores)
        cohort = enhanced_df[enhanced_df['PATNO'] == patient]['COHORT_DEFINITION'].iloc[0]
        print(f"  Patient {patient} ({cohort}): {visits} visits → {scores}")

# =============================================================================
# 2. PROGRESSION TARGET CALCULATION
# =============================================================================

def calculate_motor_progression_slopes(longitudinal_df: pd.DataFrame, 
                                     enhanced_patients: List[int],
                                     time_window_months: int = 36,
                                     min_visits: int = 3) -> pd.DataFrame:
    """
    Calculate motor progression slopes for enhanced model patients.
    
    Args:
        longitudinal_df: Full longitudinal dataset
        enhanced_patients: List of enhanced model patient IDs
        time_window_months: Analysis window in months
        min_visits: Minimum visits required for slope calculation
    
    Returns:
        DataFrame with patient IDs and motor progression slopes
    """
    print(f"\n🔢 CALCULATING MOTOR PROGRESSION SLOPES")
    print("-" * 50)
    
    # Filter to enhanced patients with motor data
    motor_data = longitudinal_df[
        (longitudinal_df['PATNO'].isin(enhanced_patients)) &
        (longitudinal_df['NP3TOT'].notna())
    ].copy()
    
    # Convert EVENT_ID to months (assuming standard PPMI visit schedule)
    event_to_months = {
        'BL': 0, 'SC': 1, 'V01': 3, 'V02': 6, 'V03': 9, 'V04': 12,
        'V05': 15, 'V06': 18, 'V07': 21, 'V08': 24, 'V09': 27, 'V10': 30,
        'V11': 33, 'V12': 36, 'V13': 39, 'V14': 42, 'V15': 45, 'V16': 48
    }
    
    motor_data['MONTHS'] = motor_data['EVENT_ID'].map(event_to_months)
    motor_data = motor_data[motor_data['MONTHS'].notna()]
    
    slopes = []
    
    for patient in enhanced_patients:
        patient_data = motor_data[
            (motor_data['PATNO'] == patient) &
            (motor_data['MONTHS'] <= time_window_months)
        ].sort_values('MONTHS')
        
        if len(patient_data) >= min_visits:
            # Calculate linear slope using least squares
            months = patient_data['MONTHS'].values
            scores = patient_data['NP3TOT'].values
            
            if len(months) > 1 and np.std(months) > 0:
                slope, intercept, r_value, p_value, std_err = stats.linregress(months, scores)
                
                slopes.append({
                    'PATNO': patient,
                    'motor_slope': slope,
                    'baseline_updrs': scores[0],
                    'final_updrs': scores[-1],
                    'total_change': scores[-1] - scores[0],
                    'months_observed': months[-1] - months[0],
                    'n_visits': len(scores),
                    'r_squared': r_value**2,
                    'p_value': p_value
                })
    
    slopes_df = pd.DataFrame(slopes)
    
    print(f"✅ Motor progression slopes calculated:")
    print(f"   • Patients with slopes: {len(slopes_df)}")
    print(f"   • Mean slope: {slopes_df['motor_slope'].mean():.3f} ± {slopes_df['motor_slope'].std():.3f} points/month")
    print(f"   • Slope range: {slopes_df['motor_slope'].min():.3f} to {slopes_df['motor_slope'].max():.3f}")
    print(f"   • Mean R²: {slopes_df['r_squared'].mean():.3f}")
    
    return slopes_df

def calculate_cognitive_conversion_labels(longitudinal_df: pd.DataFrame,
                                        enhanced_patients: List[int],
                                        time_window_months: int = 36,
                                        min_visits: int = 3,
                                        mci_threshold: float = 3.0,
                                        dementia_threshold: float = 5.0) -> pd.DataFrame:
    """
    Calculate cognitive decline conversion labels.
    
    Args:
        longitudinal_df: Full longitudinal dataset  
        enhanced_patients: List of enhanced model patient IDs
        time_window_months: Analysis window in months
        min_visits: Minimum visits required
        mci_threshold: MoCA decline threshold for MCI
        dementia_threshold: MoCA decline threshold for dementia
    
    Returns:
        DataFrame with patient IDs and cognitive conversion labels
    """
    print(f"\n🧠 CALCULATING COGNITIVE CONVERSION LABELS")
    print("-" * 50)
    
    # Filter to enhanced patients with cognitive data
    cognitive_data = longitudinal_df[
        (longitudinal_df['PATNO'].isin(enhanced_patients)) &
        (longitudinal_df['MCATOT'].notna())
    ].copy()
    
    # Convert EVENT_ID to months
    event_to_months = {
        'BL': 0, 'SC': 1, 'V01': 3, 'V02': 6, 'V03': 9, 'V04': 12,
        'V05': 15, 'V06': 18, 'V07': 21, 'V08': 24, 'V09': 27, 'V10': 30,
        'V11': 33, 'V12': 36, 'V13': 39, 'V14': 42, 'V15': 45, 'V16': 48
    }
    
    cognitive_data['MONTHS'] = cognitive_data['EVENT_ID'].map(event_to_months)
    cognitive_data = cognitive_data[cognitive_data['MONTHS'].notna()]
    
    conversions = []
    
    for patient in enhanced_patients:
        patient_data = cognitive_data[
            (cognitive_data['PATNO'] == patient) &
            (cognitive_data['MONTHS'] <= time_window_months)
        ].sort_values('MONTHS')
        
        if len(patient_data) >= min_visits:
            baseline_moca = patient_data['MCATOT'].iloc[0]
            final_moca = patient_data['MCATOT'].iloc[-1]
            max_decline = baseline_moca - patient_data['MCATOT'].min()
            
            # Define conversion criteria
            mci_conversion = (max_decline >= mci_threshold) and (patient_data['MCATOT'].min() < 26)
            dementia_conversion = (max_decline >= dementia_threshold) and (patient_data['MCATOT'].min() < 24)
            
            conversions.append({
                'PATNO': patient,
                'cognitive_conversion': int(mci_conversion or dementia_conversion),
                'mci_conversion': int(mci_conversion),
                'dementia_conversion': int(dementia_conversion),
                'baseline_moca': baseline_moca,
                'final_moca': final_moca,
                'max_decline': max_decline,
                'months_observed': patient_data['MONTHS'].iloc[-1] - patient_data['MONTHS'].iloc[0],
                'n_visits': len(patient_data)
            })
    
    conversions_df = pd.DataFrame(conversions)
    
    print(f"✅ Cognitive conversion labels calculated:")
    print(f"   • Patients with labels: {len(conversions_df)}")
    print(f"   • Any conversion: {conversions_df['cognitive_conversion'].sum()} ({conversions_df['cognitive_conversion'].mean()*100:.1f}%)")
    print(f"   • MCI conversion: {conversions_df['mci_conversion'].sum()} ({conversions_df['mci_conversion'].mean()*100:.1f}%)")
    print(f"   • Dementia conversion: {conversions_df['dementia_conversion'].sum()} ({conversions_df['dementia_conversion'].mean()*100:.1f}%)")
    print(f"   • Mean max decline: {conversions_df['max_decline'].mean():.2f} ± {conversions_df['max_decline'].std():.2f} points")
    
    return conversions_df

# =============================================================================
# 3. MULTI-TASK PROGNOSTIC MODEL ARCHITECTURE
# =============================================================================

class PrognosticGIMAN(nn.Module):
    """
    Multi-task prognostic GIMAN model for motor progression and cognitive decline.
    """
    
    def __init__(self, 
                 input_dim: int = 12,
                 hidden_dims: List[int] = [96, 256, 64],
                 dropout: float = 0.3):
        """
        Initialize prognostic GIMAN model.
        
        Args:
            input_dim: Number of input features (12 for enhanced model)
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate for regularization
        """
        super(PrognosticGIMAN, self).__init__()
        
        # Shared GNN backbone (preserving enhanced model architecture)
        self.conv1 = GCNConv(input_dim, hidden_dims[0])
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.conv3 = GCNConv(hidden_dims[1], hidden_dims[2])
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dims[0])
        self.batch_norm2 = nn.BatchNorm1d(hidden_dims[1])
        self.batch_norm3 = nn.BatchNorm1d(hidden_dims[2])
        
        # Task-specific prediction heads
        self.motor_head = nn.Sequential(
            nn.Linear(hidden_dims[2], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # Regression output
        )
        
        self.cognitive_head = nn.Sequential(
            nn.Linear(hidden_dims[2], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),  # Classification output
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through prognostic GIMAN.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for graph pooling
        
        Returns:
            Tuple of (motor_predictions, cognitive_predictions)
        """
        # Shared GNN feature extraction
        h1 = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        h1 = self.dropout(h1)
        
        h2 = F.relu(self.batch_norm2(self.conv2(h1, edge_index)))
        h2 = self.dropout(h2)
        
        h3 = F.relu(self.batch_norm3(self.conv3(h2, edge_index)))
        shared_features = self.dropout(h3)
        
        # Node-level predictions (no pooling - predict for each patient)
        # Task-specific predictions
        motor_pred = self.motor_head(shared_features).squeeze()
        cognitive_pred = self.cognitive_head(shared_features).squeeze()
        
        return motor_pred, cognitive_pred

def create_prognostic_dataset(enhanced_df: pd.DataFrame,
                            motor_slopes: pd.DataFrame,
                            cognitive_labels: pd.DataFrame) -> Data:
    """
    Create PyTorch Geometric dataset for prognostic training.
    
    Args:
        enhanced_df: Enhanced GIMAN dataset with features
        motor_slopes: Motor progression slopes
        cognitive_labels: Cognitive conversion labels
    
    Returns:
        PyTorch Geometric Data object
    """
    print(f"\n🔧 CREATING PROGNOSTIC DATASET")
    print("-" * 50)
    
    # Get baseline features for each patient (use first record per patient)
    patient_features = enhanced_df.groupby('PATNO').first().reset_index()
    
    # Merge with progression targets
    dataset = patient_features.merge(motor_slopes[['PATNO', 'motor_slope']], on='PATNO', how='left')
    dataset = dataset.merge(cognitive_labels[['PATNO', 'cognitive_conversion']], on='PATNO', how='left')
    
    # Select the 12 enhanced model features (using actual column names)
    feature_cols = ['LRRK2', 'GBA', 'APOE_RISK', 'PTAU', 'TTAU', 'UPSIT_TOTAL', 
                   'ALPHA_SYN', 'AGE_COMPUTED', 'NHY', 'SEX', 'NP3TOT', 'HAS_DATSCAN']
    
    # Prepare node features
    X = torch.tensor(dataset[feature_cols].fillna(0).values, dtype=torch.float32)
    
    # Prepare targets (handle missing values)
    motor_targets = torch.tensor(dataset['motor_slope'].fillna(0).values, dtype=torch.float32)
    cognitive_targets = torch.tensor(dataset['cognitive_conversion'].fillna(0).values, dtype=torch.float32)
    
    # Create masks for available targets
    motor_mask = torch.tensor(~dataset['motor_slope'].isna().values, dtype=torch.bool)
    cognitive_mask = torch.tensor(~dataset['cognitive_conversion'].isna().values, dtype=torch.bool)
    
    # Create graph structure (k-nearest neighbors as in enhanced model)
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    
    # Standardize features for distance calculation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.numpy())
    
    # Find k-nearest neighbors (k=6 as in enhanced model)
    nbrs = NearestNeighbors(n_neighbors=7, metric='cosine').fit(X_scaled)  # 7 to include self
    distances, indices = nbrs.kneighbors(X_scaled)
    
    # Create edge index (exclude self-connections)
    edge_list = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:  # Skip self (index 0)
            edge_list.append([i, neighbor])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    print(f"✅ Prognostic dataset created:")
    print(f"   • Nodes (patients): {X.shape[0]}")
    print(f"   • Features per node: {X.shape[1]}")
    print(f"   • Edges: {edge_index.shape[1]}")
    print(f"   • Motor targets available: {motor_mask.sum().item()}")
    print(f"   • Cognitive targets available: {cognitive_mask.sum().item()}")
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=X,
        edge_index=edge_index,
        motor_y=motor_targets,
        cognitive_y=cognitive_targets,
        motor_mask=motor_mask,
        cognitive_mask=cognitive_mask,
        patient_ids=torch.tensor(dataset['PATNO'].values, dtype=torch.long)
    )
    
    return data

# =============================================================================
# 4. TRAINING PIPELINE
# =============================================================================

def train_prognostic_model(data: Data, 
                         epochs: int = 200,
                         lr: float = 0.001,
                         weight_decay: float = 1e-4,
                         motor_weight: float = 0.5) -> PrognosticGIMAN:
    """
    Train the multi-task prognostic GIMAN model.
    
    Args:
        data: PyTorch Geometric Data object
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        motor_weight: Weight balance between motor and cognitive tasks
    
    Returns:
        Trained model
    """
    print(f"\n🏋️ TRAINING PROGNOSTIC GIMAN MODEL")
    print("-" * 50)
    
    # Initialize model
    model = PrognosticGIMAN(input_dim=data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    model.train()
    train_losses = []
    motor_losses = []
    cognitive_losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        motor_pred, cognitive_pred = model(data.x, data.edge_index)
            
        # Calculate task-specific losses (only for available targets)
        if data.motor_mask.sum() > 0:
            motor_loss = F.mse_loss(motor_pred[data.motor_mask], data.motor_y[data.motor_mask])
        else:
            motor_loss = torch.tensor(0.0, requires_grad=True)
            
        if data.cognitive_mask.sum() > 0:
            cognitive_loss = F.binary_cross_entropy(cognitive_pred[data.cognitive_mask], 
                                                   data.cognitive_y[data.cognitive_mask])
        else:
            cognitive_loss = torch.tensor(0.0, requires_grad=True)
        
        # Multi-task loss
        total_loss = motor_weight * motor_loss + (1 - motor_weight) * cognitive_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Store losses
        train_losses.append(total_loss.item())
        motor_losses.append(motor_loss.item())
        cognitive_losses.append(cognitive_loss.item())
        
        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch+1:3d}: Total={total_loss:.6f}, Motor={motor_loss:.6f}, Cognitive={cognitive_loss:.6f}")
    
    print(f"✅ Training completed!")
    print(f"   • Final total loss: {train_losses[-1]:.6f}")
    print(f"   • Final motor loss: {motor_losses[-1]:.6f}")
    print(f"   • Final cognitive loss: {cognitive_losses[-1]:.6f}")
    
    return model

def evaluate_prognostic_model(model: PrognosticGIMAN, data: Data) -> Dict:
    """
    Evaluate the trained prognostic model.
    
    Args:
        model: Trained prognostic GIMAN model
        data: PyTorch Geometric Data object
    
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n📊 EVALUATING PROGNOSTIC MODEL")
    print("-" * 50)
    
    model.eval()
    with torch.no_grad():
        motor_pred, cognitive_pred = model(data.x, data.edge_index)
    
    results = {}
    
    # Motor progression evaluation
    if data.motor_mask.sum() > 0:
        motor_true = data.motor_y[data.motor_mask].numpy()
        motor_predictions = motor_pred[data.motor_mask].numpy()
        
        motor_mse = mean_squared_error(motor_true, motor_predictions)
        motor_r2 = r2_score(motor_true, motor_predictions)
        motor_corr = np.corrcoef(motor_true, motor_predictions)[0, 1]
        
        results['motor'] = {
            'mse': motor_mse,
            'rmse': np.sqrt(motor_mse),
            'r2': motor_r2,
            'correlation': motor_corr,
            'n_samples': len(motor_true)
        }
        
        print(f"Motor Progression Regression:")
        print(f"   • R² Score: {motor_r2:.4f}")
        print(f"   • RMSE: {np.sqrt(motor_mse):.6f}")
        print(f"   • Correlation: {motor_corr:.4f}")
        print(f"   • Samples: {len(motor_true)}")
    
    # Cognitive decline evaluation
    if data.cognitive_mask.sum() > 0:
        cognitive_true = data.cognitive_y[data.cognitive_mask].numpy()
        cognitive_predictions = cognitive_pred[data.cognitive_mask].numpy()
        
        # Handle edge case where all labels are the same
        if len(np.unique(cognitive_true)) > 1:
            cognitive_auc = roc_auc_score(cognitive_true, cognitive_predictions)
        else:
            cognitive_auc = 0.5  # Random performance for single class
        
        cognitive_binary = (cognitive_predictions > 0.5).astype(int)
        accuracy = (cognitive_binary == cognitive_true).mean()
        
        results['cognitive'] = {
            'auc_roc': cognitive_auc,
            'accuracy': accuracy,
            'n_samples': len(cognitive_true),
            'n_positive': cognitive_true.sum(),
            'n_negative': len(cognitive_true) - cognitive_true.sum()
        }
        
        print(f"\nCognitive Decline Classification:")
        print(f"   • AUC-ROC: {cognitive_auc:.4f}")
        print(f"   • Accuracy: {accuracy:.4f}")
        print(f"   • Samples: {len(cognitive_true)}")
        print(f"   • Positive cases: {cognitive_true.sum():.0f}")
    
    return results

# =============================================================================
# 5. VISUALIZATION AND ANALYSIS
# =============================================================================

def create_prognostic_visualizations(data: Data, 
                                   model: PrognosticGIMAN,
                                   motor_slopes: pd.DataFrame,
                                   cognitive_labels: pd.DataFrame) -> None:
    """Create comprehensive visualizations for prognostic analysis."""
    print(f"\n📈 CREATING PROGNOSTIC VISUALIZATIONS")
    print("-" * 50)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('GIMAN Phase 1 Prognostic Development - Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. Motor progression distribution
    axes[0, 0].hist(motor_slopes['motor_slope'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(motor_slopes['motor_slope'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {motor_slopes["motor_slope"].mean():.3f}')
    axes[0, 0].set_xlabel('Motor Progression Slope (UPDRS points/month)')
    axes[0, 0].set_ylabel('Number of Patients')
    axes[0, 0].set_title('Distribution of Motor Progression Rates')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Cognitive conversion rates
    conv_counts = cognitive_labels['cognitive_conversion'].value_counts()
    axes[0, 1].pie(conv_counts.values, labels=['Stable', 'Conversion'], autopct='%1.1f%%', 
                  colors=['lightgreen', 'lightcoral'])
    axes[0, 1].set_title('Cognitive Conversion Distribution')
    
    # 3. Motor slope vs baseline UPDRS
    axes[0, 2].scatter(motor_slopes['baseline_updrs'], motor_slopes['motor_slope'], 
                      alpha=0.6, color='purple')
    axes[0, 2].set_xlabel('Baseline UPDRS III Score')
    axes[0, 2].set_ylabel('Motor Progression Slope')
    axes[0, 2].set_title('Baseline UPDRS vs Progression Rate')
    axes[0, 2].grid(alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(motor_slopes['baseline_updrs'], motor_slopes['motor_slope'])[0, 1]
    axes[0, 2].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0, 2].transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Model predictions vs actual (if model is provided)
    model.eval()
    with torch.no_grad():
        motor_pred, cognitive_pred = model(data.x, data.edge_index)
    
    if data.motor_mask.sum() > 0:
        motor_true = data.motor_y[data.motor_mask].numpy()
        motor_predictions = motor_pred[data.motor_mask].numpy()
        
        axes[1, 0].scatter(motor_true, motor_predictions, alpha=0.6, color='orange')
        
        # Add perfect prediction line
        min_val, max_val = min(motor_true.min(), motor_predictions.min()), max(motor_true.max(), motor_predictions.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        axes[1, 0].set_xlabel('True Motor Slope')
        axes[1, 0].set_ylabel('Predicted Motor Slope')
        axes[1, 0].set_title('Motor Progression: Predicted vs Actual')
        axes[1, 0].grid(alpha=0.3)
        
        # Add R² score
        r2 = r2_score(motor_true, motor_predictions)
        axes[1, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[1, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 5. Temporal progression patterns
    sample_patients = motor_slopes.head(10)['PATNO'].tolist()
    for i, patient in enumerate(sample_patients):
        patient_data = motor_slopes[motor_slopes['PATNO'] == patient].iloc[0]
        months = np.linspace(0, patient_data['months_observed'], int(patient_data['n_visits']))
        predicted_progression = patient_data['baseline_updrs'] + patient_data['motor_slope'] * months
        
        axes[1, 1].plot(months, predicted_progression, alpha=0.6, linewidth=1)
    
    axes[1, 1].set_xlabel('Months from Baseline')
    axes[1, 1].set_ylabel('Predicted UPDRS III Score')
    axes[1, 1].set_title('Sample Motor Progression Trajectories')
    axes[1, 1].grid(alpha=0.3)
    
    # 6. Feature importance visualization (using model weights)
    feature_names = ['LRRK2', 'GBA', 'APOE_RISK', 'PTAU', 'TTAU', 'UPSIT_TOTAL', 
                    'ALPHA_SYN', 'AGE_COMPUTED', 'NHY', 'SEX', 'NP3TOT', 'HAS_DATSCAN']
    
    # Get first layer weights as proxy for feature importance
    # GCNConv stores weights in lin.weight
    first_layer_weights = model.conv1.lin.weight.data.abs().mean(dim=0).numpy()
    
    # Create horizontal bar plot
    y_pos = np.arange(len(feature_names))
    axes[1, 2].barh(y_pos, first_layer_weights, color='teal', alpha=0.7)
    axes[1, 2].set_yticks(y_pos)
    axes[1, 2].set_yticklabels(feature_names)
    axes[1, 2].set_xlabel('Feature Importance (Weight Magnitude)')
    axes[1, 2].set_title('Model Feature Importance')
    axes[1, 2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = "visualizations/enhanced_progression/phase1_prognostic_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualizations saved to: {output_path}")
    plt.show()

# =============================================================================
# 6. MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    """Main execution pipeline for Phase 1 prognostic development."""
    print("🚀 GIMAN Phase 1 Prognostic Development")
    print("=" * 70)
    print("Objectives:")
    print("  1. Motor progression regression (UPDRS slope prediction)")
    print("  2. Cognitive decline classification (MCI/dementia conversion)")
    print("  3. Multi-task GNN architecture validation")
    print("=" * 70)
    
    # Step 1: Load data
    enhanced_df = load_enhanced_giman_data()
    longitudinal_df = load_longitudinal_data()
    
    if enhanced_df.empty or longitudinal_df.empty:
        print("❌ Required data files not found. Please check file paths.")
        return
    
    # Step 2: Explore progression patterns
    explore_progression_data(enhanced_df, longitudinal_df)
    
    # Step 3: Calculate progression targets
    enhanced_patients = enhanced_df['PATNO'].unique().tolist()
    
    motor_slopes = calculate_motor_progression_slopes(
        longitudinal_df, enhanced_patients, time_window_months=36, min_visits=3
    )
    
    cognitive_labels = calculate_cognitive_conversion_labels(
        longitudinal_df, enhanced_patients, time_window_months=36, min_visits=3
    )
    
    # Step 4: Create prognostic dataset
    data = create_prognostic_dataset(enhanced_df, motor_slopes, cognitive_labels)
    
    # Step 5: Train prognostic model
    model = train_prognostic_model(data, epochs=200, lr=0.001, motor_weight=0.6)
    
    # Step 6: Evaluate model performance
    results = evaluate_prognostic_model(model, data)
    
    # Step 7: Create visualizations
    create_prognostic_visualizations(data, model, motor_slopes, cognitive_labels)
    
    # Step 8: Save results and model
    print(f"\n💾 SAVING RESULTS")
    print("-" * 50)
    
    # Save progression targets
    motor_slopes.to_csv("data/prognostic/motor_progression_targets.csv", index=False)
    cognitive_labels.to_csv("data/prognostic/cognitive_conversion_labels.csv", index=False)
    
    # Save model
    torch.save(model.state_dict(), "models/prognostic_giman_phase1.pth")
    
    print("✅ Phase 1 prognostic development completed!")
    print(f"   • Motor progression model R²: {results.get('motor', {}).get('r2', 'N/A')}")
    print(f"   • Cognitive decline model AUC: {results.get('cognitive', {}).get('auc_roc', 'N/A')}")
    print(f"   • Results saved to: data/prognostic/")
    print(f"   • Model saved to: models/prognostic_giman_phase1.pth")
    
    print("\n🎯 Next Steps:")
    print("   1. Analyze feature importances and clinical interpretability")
    print("   2. Implement temporal cross-validation for robust evaluation")  
    print("   3. Begin Phase 2: Advanced modality-specific encoders")
    print("   4. Validate prognostic utility with external cohorts")

if __name__ == "__main__":
    main()