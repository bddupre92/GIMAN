#!/usr/bin/env python3
"""GIMAN Phase 4: Unified System with Research Analysis & Counterfactual Generation

This script implements the unified GIMAN system that combines:
- Phase 3.1: Basic GAT reliability
- Phase 3.2: Optimized cross-modal attention
- Phase 3.3: Temporal modeling capabilities
- Advanced research analysis tools
- Counterfactual generation for causal inference

Author: GIMAN Development Team
Date: September 24, 2025
Phase: 4.0 - Unified System with Research Analytics
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

from collections import defaultdict

from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

# Optional imports for interpretability (may not be available due to version conflicts)
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Import our previous phase models
archive_phase3_path = Path(__file__).parent.parent / "phase3"
sys.path.append(str(archive_phase3_path))
from phase3_1_real_data_integration import RealDataPhase3Integration

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UnifiedAttentionModule(nn.Module):
    """Unified attention combining cross-modal and temporal mechanisms."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.embed_dim = embed_dim

        # Cross-modal attention (from Phase 3.2)
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Temporal attention (from Phase 3.3)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Unified fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

        # Attention importance weights
        self.attention_importance = nn.Parameter(torch.ones(2) / 2.0)

    def forward(
        self,
        spatial_emb: torch.Tensor,
        genomic_emb: torch.Tensor,
        temporal_emb: torch.Tensor | None = None,
    ):
        """Unified attention forward pass."""
        # Ensure proper dimensions
        if spatial_emb.dim() == 2:
            spatial_emb = spatial_emb.unsqueeze(1)
        if genomic_emb.dim() == 2:
            genomic_emb = genomic_emb.unsqueeze(1)

        # Cross-modal attention
        cross_modal_output, cross_weights = self.cross_modal_attention(
            spatial_emb, genomic_emb, genomic_emb
        )

        # Temporal attention (if available)
        if temporal_emb is not None:
            if temporal_emb.dim() == 2:
                temporal_emb = temporal_emb.unsqueeze(1)
            temporal_output, temporal_weights = self.temporal_attention(
                spatial_emb, temporal_emb, temporal_emb
            )
        else:
            temporal_output = spatial_emb
            temporal_weights = None

        # Weighted combination
        attention_weights = F.softmax(self.attention_importance, dim=0)
        combined_features = torch.cat(
            [
                cross_modal_output * attention_weights[0],
                temporal_output * attention_weights[1],
            ],
            dim=-1,
        )

        # Fusion
        unified_output = self.fusion_layer(combined_features)

        return {
            "unified_features": unified_output.squeeze(1),
            "cross_modal_weights": cross_weights,
            "temporal_weights": temporal_weights,
            "attention_importance": attention_weights,
        }


class EnsemblePredictor(nn.Module):
    """Ensemble predictor combining multiple prediction strategies."""

    def __init__(self, embed_dim: int, dropout: float = 0.15):
        super().__init__()

        # Multiple prediction heads with different architectures
        self.predictor_heads = nn.ModuleList(
            [
                # Head 1: Simple linear
                nn.Sequential(nn.Linear(embed_dim, 1), nn.Dropout(dropout * 0.5)),
                # Head 2: Deep network
                nn.Sequential(
                    nn.Linear(embed_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                ),
                # Head 3: Residual approach
                nn.Sequential(
                    nn.Linear(embed_dim, 32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, 1),
                ),
            ]
        )

        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3.0)

    def forward(self, features: torch.Tensor):
        """Ensemble prediction forward pass."""
        predictions = []

        for head in self.predictor_heads:
            pred = head(features)
            predictions.append(pred)

        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_pred = sum(
            w * pred for w, pred in zip(weights, predictions, strict=False)
        )

        return {
            "ensemble_prediction": ensemble_pred,
            "individual_predictions": predictions,
            "ensemble_weights": weights,
        }


class UnifiedGIMANSystem(nn.Module):
    """Unified GIMAN system combining all phase capabilities."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.embed_dim = embed_dim

        # Unified attention module
        self.unified_attention = UnifiedAttentionModule(embed_dim, num_heads, dropout)

        # Feature processor (from Phase 3.2 enhanced)
        self.feature_processor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Ensemble predictors for different tasks
        self.motor_predictor = EnsemblePredictor(embed_dim, dropout)
        self.cognitive_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, spatial_emb, genomic_emb, temporal_emb):
        """Forward pass for the unified system.

        Args:
            spatial_emb: Spatiotemporal embeddings.
            genomic_emb: Genomic embeddings.
            temporal_emb: Temporal embeddings.

        Returns:
            Tuple containing motor prediction, cognitive prediction, and attention weights.
        """
        # Get unified features and attention weights
        attention_output = self.unified_attention(
            spatial_emb, genomic_emb, temporal_emb
        )
        unified_features = attention_output["unified_features"]
        attention_weights = attention_output["attention_importance"]

        # Process features
        processed_features = self.feature_processor(unified_features)

        # Make predictions
        motor_pred_output = self.motor_predictor(processed_features)
        motor_pred = motor_pred_output["ensemble_prediction"]
        cognitive_pred = self.cognitive_predictor(processed_features)

        return motor_pred, cognitive_pred, attention_weights

    def _compute_loss(
        self,
        motor_pred,
        motor_true,
        cognitive_pred,
        cognitive_true,
        attention_weights,
        pos_weight=None,
    ):
        """Computes the combined loss for the model with class balancing."""
        motor_loss = F.huber_loss(motor_pred, motor_true.unsqueeze(1))

        # Use weighted binary cross entropy for class imbalance
        if pos_weight is not None:
            cognitive_loss = F.binary_cross_entropy_with_logits(
                cognitive_pred, cognitive_true.unsqueeze(1), pos_weight=pos_weight
            )
        else:
            cognitive_loss = F.binary_cross_entropy_with_logits(
                cognitive_pred, cognitive_true.unsqueeze(1)
            )

        # Attention regularization to encourage balanced attention
        attention_loss = torch.std(attention_weights) * 0.01

        total_loss = motor_loss + cognitive_loss + attention_loss
        return total_loss, motor_loss, cognitive_loss


def run_phase4_experiment(
    data_integrator,
    epochs: int = 150,
    lr: float = 5e-5,
    weight_decay: float = 1e-5,
    patience: int = 20,
):
    """Runs a full training and evaluation cycle for the Phase 4 system."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running Phase 4 experiment on {device}")

    # Load data
    spatial_emb = data_integrator.spatiotemporal_embeddings
    genomic_emb = data_integrator.genomic_embeddings
    temporal_emb = data_integrator.temporal_embeddings
    motor_targets = data_integrator.prognostic_targets[:, 0]
    cognitive_targets = data_integrator.prognostic_targets[:, 1]

    # Enhanced data preprocessing and validation
    logging.info("🔧 Preprocessing and validating input data...")

    # Validate input data dimensions
    n_patients = len(data_integrator.patient_ids)
    assert spatial_emb.shape[0] == n_patients, (
        f"Spatial embedding dimension mismatch: {spatial_emb.shape[0]} vs {n_patients}"
    )
    assert genomic_emb.shape[0] == n_patients, (
        f"Genomic embedding dimension mismatch: {genomic_emb.shape[0]} vs {n_patients}"
    )
    assert temporal_emb.shape[0] == n_patients, (
        f"Temporal embedding dimension mismatch: {temporal_emb.shape[0]} vs {n_patients}"
    )

    # Data scaling with robust preprocessing
    scaler_motor = StandardScaler()
    scaler_spatial = StandardScaler()
    scaler_genomic = StandardScaler()
    scaler_temporal = StandardScaler()

    motor_targets_scaled = scaler_motor.fit_transform(
        motor_targets.reshape(-1, 1)
    ).flatten()
    spatial_emb_scaled = scaler_spatial.fit_transform(spatial_emb)
    genomic_emb_scaled = scaler_genomic.fit_transform(genomic_emb)
    temporal_emb_scaled = scaler_temporal.fit_transform(temporal_emb)

    # Convert to tensors with validation
    def safe_tensor_conversion(data, name):
        # Handle NaN/Inf before tensor conversion
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        tensor = torch.tensor(data, dtype=torch.float32)
        logging.info(
            f"✅ {name}: {tensor.shape}, range=[{tensor.min():.3f}, {tensor.max():.3f}]"
        )
        return tensor

    X_spatial = safe_tensor_conversion(spatial_emb_scaled, "Spatial embeddings")
    X_genomic = safe_tensor_conversion(genomic_emb_scaled, "Genomic embeddings")
    X_temporal = safe_tensor_conversion(temporal_emb_scaled, "Temporal embeddings")
    y_motor = safe_tensor_conversion(motor_targets_scaled, "Motor targets")
    y_cognitive = safe_tensor_conversion(cognitive_targets, "Cognitive targets")

    # Analyze class balance for cognitive task
    cognitive_balance = cognitive_targets.mean()
    logging.info(f"📊 Cognitive conversion balance: {cognitive_balance:.1%} positive")

    # Calculate class weights for imbalanced classification
    pos_weight = (
        torch.tensor((1 - cognitive_balance) / cognitive_balance)
        if cognitive_balance > 0
        else torch.tensor(1.0)
    )
    logging.info(
        f"🔧 Using pos_weight={pos_weight.item():.2f} for cognitive classification"
    )

    # Create dataset and dataloaders
    dataset = TensorDataset(X_spatial, X_genomic, X_temporal, y_motor, y_cognitive)

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model, optimizer, and scheduler
    model = UnifiedGIMANSystem().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=patience // 2, factor=0.5
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = defaultdict(list)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = [b.to(device) for b in batch]
            s_emb, g_emb, t_emb, y_m, y_c = batch

            optimizer.zero_grad()
            motor_pred, cog_pred, att_weights = model(s_emb, g_emb, t_emb)
            loss, _, _ = model._compute_loss(
                motor_pred, y_m, cog_pred, y_c, att_weights, pos_weight.to(device)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        # Validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(device) for b in batch]
                s_emb, g_emb, t_emb, y_m, y_c = batch
                motor_pred, cog_pred, att_weights = model(s_emb, g_emb, t_emb)
                loss, _, _ = model._compute_loss(
                    motor_pred, y_m, cog_pred, y_c, att_weights, pos_weight.to(device)
                )
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        history["val_loss"].append(avg_val_loss)
        scheduler.step(avg_val_loss)

        logging.info(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "phase4_best_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    # Evaluation
    model.load_state_dict(torch.load("phase4_best_model.pth"))
    model.eval()

    all_motor_preds, all_motor_true = [], []
    all_cog_preds, all_cog_true = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = [b.to(device) for b in batch]
            s_emb, g_emb, t_emb, y_m, y_c = batch
            motor_pred, cog_pred, _ = model(s_emb, g_emb, t_emb)

            all_motor_preds.extend(motor_pred.cpu().numpy())
            all_motor_true.extend(y_m.cpu().numpy())
            all_cog_preds.extend(torch.sigmoid(cog_pred).cpu().numpy())
            all_cog_true.extend(y_c.cpu().numpy())

    motor_r2 = r2_score(all_motor_true, all_motor_preds)
    cognitive_auc = roc_auc_score(all_cog_true, all_cog_preds)

    logging.info(
        f"Final Validation -> Motor R²: {motor_r2:.4f}, Cognitive AUC: {cognitive_auc:.4f}"
    )

    return {
        "model": model,
        "history": history,
        "motor_r2": motor_r2,
        "cognitive_auc": cognitive_auc,
        "scaler_motor": scaler_motor,
        "scaler_spatial": scaler_spatial,
        "scaler_genomic": scaler_genomic,
        "scaler_temporal": scaler_temporal,
    }


class GIMANResearchAnalyzer:
    """Comprehensive research analysis and counterfactual generation system."""

    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.results_dir = Path("visualizations/phase4_unified_system")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data containers
        self.enhanced_df = None
        self.longitudinal_df = None
        self.motor_targets_df = None
        self.cognitive_targets_df = None

        # Processed data
        self.patient_ids = None
        self.spatiotemporal_embeddings = None
        self.genomic_embeddings = None
        self.temporal_embeddings = None
        self.prognostic_targets = None

        # Models
        self.unified_model = None
        self.phase_models = {}  # Store individual phase models for comparison

        logger.info(f"🚀 GIMAN Phase 4 Unified System initialized on {self.device}")

    def load_multimodal_data(self):
        """Load and prepare all multimodal PPMI data."""
        logger.info("📊 Loading comprehensive multimodal PPMI data...")

        # Load datasets
        self.enhanced_df = pd.read_csv("data/enhanced/enhanced_dataset_latest.csv")
        self.longitudinal_df = pd.read_csv(
            "data/01_processed/giman_corrected_longitudinal_dataset.csv",
            low_memory=False,
        )
        self.motor_targets_df = pd.read_csv(
            "data/prognostic/motor_progression_targets.csv"
        )
        self.cognitive_targets_df = pd.read_csv(
            "data/prognostic/cognitive_conversion_labels.csv"
        )

        # Find patients with complete data
        enhanced_patients = set(self.enhanced_df.PATNO.unique())
        longitudinal_patients = set(self.longitudinal_df.PATNO.unique())
        motor_patients = set(self.motor_targets_df.PATNO.unique())
        cognitive_patients = set(self.cognitive_targets_df.PATNO.unique())

        complete_patients = enhanced_patients.intersection(
            longitudinal_patients, motor_patients, cognitive_patients
        )

        self.patient_ids = sorted(list(complete_patients))
        logger.info(
            f"👥 Patients with complete multimodal data: {len(self.patient_ids)}"
        )

    def create_unified_embeddings(self):
        """Create unified embeddings combining all modalities."""
        logger.info("🧠 Creating unified multimodal embeddings...")

        # Core features for different modalities
        imaging_features = [
            "PUTAMEN_REF_CWM",
            "PUTAMEN_L_REF_CWM",
            "PUTAMEN_R_REF_CWM",
            "CAUDATE_REF_CWM",
            "CAUDATE_L_REF_CWM",
            "CAUDATE_R_REF_CWM",
        ]
        genetic_features = ["LRRK2", "GBA", "APOE_RISK"]

        spatial_embeddings = []
        genomic_embeddings = []
        temporal_embeddings = []
        prognostic_targets = []

        valid_patients = []

        for patno in self.patient_ids:
            # Spatial embeddings (current approach)
            patient_longitudinal = self.longitudinal_df[
                (patno == self.longitudinal_df.PATNO)
                & (self.longitudinal_df[imaging_features].notna().all(axis=1))
            ].sort_values("EVENT_ID")

            if len(patient_longitudinal) == 0:
                continue

            # Create spatial embedding
            imaging_data = patient_longitudinal[imaging_features].values
            spatial_emb = self._create_spatial_embedding(imaging_data)

            # Genomic embedding
            patient_genetic = self.enhanced_df[patno == self.enhanced_df.PATNO].iloc[0]
            genomic_emb = self._create_genomic_embedding(
                patient_genetic[genetic_features].values
            )

            # Temporal embedding (for Phase 3.3 compatibility)
            temporal_emb = self._create_temporal_embedding(imaging_data)

            # Prognostic targets
            motor_data = self.motor_targets_df[patno == self.motor_targets_df.PATNO]
            cognitive_data = self.cognitive_targets_df[
                patno == self.cognitive_targets_df.PATNO
            ]

            if len(motor_data) == 0 or len(cognitive_data) == 0:
                continue

            motor_slope = motor_data["motor_slope"].iloc[0]
            cognitive_conversion = cognitive_data["cognitive_conversion"].iloc[0]

            # Normalize motor slope
            motor_norm = max(0, min(10, motor_slope)) / 10.0

            spatial_embeddings.append(spatial_emb)
            genomic_embeddings.append(genomic_emb)
            temporal_embeddings.append(temporal_emb)
            prognostic_targets.append([motor_norm, float(cognitive_conversion)])
            valid_patients.append(patno)

        # Convert to arrays
        self.spatiotemporal_embeddings = np.array(spatial_embeddings, dtype=np.float32)
        self.genomic_embeddings = np.array(genomic_embeddings, dtype=np.float32)
        self.temporal_embeddings = np.array(temporal_embeddings, dtype=np.float32)
        self.prognostic_targets = np.array(prognostic_targets, dtype=np.float32)
        self.patient_ids = valid_patients

        # Normalize embeddings
        for emb in [
            self.spatiotemporal_embeddings,
            self.genomic_embeddings,
            self.temporal_embeddings,
        ]:
            emb[:] = np.nan_to_num(emb)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1
            emb[:] = emb / norms

        logger.info(f"✅ Unified embeddings created: {len(self.patient_ids)} patients")
        logger.info(f"📐 Spatial: {self.spatiotemporal_embeddings.shape}")
        logger.info(f"🧬 Genomic: {self.genomic_embeddings.shape}")
        logger.info(f"⏰ Temporal: {self.temporal_embeddings.shape}")

    def _create_spatial_embedding(
        self, imaging_data: np.ndarray, target_dim: int = 256
    ) -> np.ndarray:
        """Create spatial embedding from imaging data."""
        if len(imaging_data) == 0:
            return np.zeros(target_dim)

        # Statistical features
        mean_vals = np.mean(imaging_data, axis=0)
        std_vals = np.std(imaging_data, axis=0)

        # Progression features
        if len(imaging_data) > 1:
            slopes = []
            for i in range(imaging_data.shape[1]):
                vals = imaging_data[:, i]
                if np.std(vals) > 1e-6:
                    slope = np.polyfit(np.arange(len(vals)), vals, 1)[0]
                else:
                    slope = 0.0
                slopes.append(slope)
            slopes = np.array(slopes)
        else:
            slopes = np.zeros(imaging_data.shape[1])

        # Recent values
        recent_vals = imaging_data[-1] if len(imaging_data) > 0 else mean_vals

        # Combine features
        features = np.concatenate([mean_vals, std_vals, slopes, recent_vals])

        # Expand to target dimension
        return self._expand_to_target_dim(features, target_dim)

    def _create_genomic_embedding(
        self, genetic_data: np.ndarray, target_dim: int = 256
    ) -> np.ndarray:
        """Create genomic embedding with interaction terms."""
        base_features = genetic_data

        # Interaction terms
        interactions = []
        for i in range(len(base_features)):
            for j in range(i + 1, len(base_features)):
                interactions.append(base_features[i] * base_features[j])

        # Risk combinations
        total_risk = np.sum(base_features)
        risk_combinations = [
            base_features[0] + base_features[1] if len(base_features) > 1 else 0,
            base_features[0] + base_features[2] if len(base_features) > 2 else 0,
            base_features[1] + base_features[2] if len(base_features) > 2 else 0,
        ]

        # Combine all features
        full_features = np.concatenate(
            [base_features, interactions, [total_risk], risk_combinations]
        )

        return self._expand_to_target_dim(full_features, target_dim)

    def _create_temporal_embedding(
        self, imaging_sequence: np.ndarray, target_dim: int = 256
    ) -> np.ndarray:
        """Create temporal embedding from imaging sequence."""
        if len(imaging_sequence) <= 1:
            return np.zeros(target_dim)

        # Temporal difference features
        diffs = np.diff(imaging_sequence, axis=0)

        # Temporal statistics
        mean_diffs = np.mean(diffs, axis=0)
        std_diffs = np.std(diffs, axis=0)

        # Acceleration (second derivatives)
        if len(diffs) > 1:
            accel = np.diff(diffs, axis=0)
            mean_accel = np.mean(accel, axis=0)
        else:
            mean_accel = np.zeros(imaging_sequence.shape[1])

        # Trend features
        trend_features = []
        for i in range(imaging_sequence.shape[1]):
            vals = imaging_sequence[:, i]
            # Linear trend
            linear_trend = np.polyfit(np.arange(len(vals)), vals, 1)[0]
            # Curvature
            if len(vals) >= 3:
                curvature = np.polyfit(np.arange(len(vals)), vals, 2)[0]
            else:
                curvature = 0
            trend_features.extend([linear_trend, curvature])

        # Combine temporal features
        temporal_features = np.concatenate(
            [mean_diffs, std_diffs, mean_accel, trend_features]
        )

        return self._expand_to_target_dim(temporal_features, target_dim)

    def _expand_to_target_dim(
        self, features: np.ndarray, target_dim: int
    ) -> np.ndarray:
        """Expand feature vector to target dimension."""
        current_dim = len(features)

        if current_dim >= target_dim:
            return features[:target_dim]

        # Repeat and pad
        repeat_factor = target_dim // current_dim
        remainder = target_dim % current_dim

        expanded = np.tile(features, repeat_factor)
        if remainder > 0:
            expanded = np.concatenate([expanded, features[:remainder]])

        return expanded

    def train_unified_system(self, num_epochs: int = 100) -> dict:
        """Train the unified GIMAN system."""
        logger.info(f"🚂 Training Unified GIMAN System for {num_epochs} epochs...")

        # Create unified model
        self.unified_model = UnifiedGIMANSystem(embed_dim=256, num_heads=4, dropout=0.4)
        self.unified_model.to(self.device)

        # Prepare data
        spatial_emb = torch.tensor(self.spatiotemporal_embeddings, dtype=torch.float32)
        genomic_emb = torch.tensor(self.genomic_embeddings, dtype=torch.float32)
        temporal_emb = torch.tensor(self.temporal_embeddings, dtype=torch.float32)
        targets = torch.tensor(self.prognostic_targets, dtype=torch.float32)

        # Normalize motor targets
        motor_targets = targets[:, 0]
        motor_mean = motor_targets.mean()
        motor_std = motor_targets.std() + 1e-8
        targets[:, 0] = (motor_targets - motor_mean) / motor_std

        # Stratified data splits
        n_patients = len(self.patient_ids)
        indices = np.arange(n_patients)

        cognitive_labels = targets[:, 1].numpy()
        if len(np.unique(cognitive_labels)) > 1:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
            train_idx, temp_idx = next(sss.split(indices, cognitive_labels))

            temp_cognitive = cognitive_labels[temp_idx]
            if len(np.unique(temp_cognitive)) > 1:
                sss_temp = StratifiedShuffleSplit(
                    n_splits=1, test_size=0.5, random_state=42
                )
                val_temp_idx, test_temp_idx = next(
                    sss_temp.split(temp_idx, temp_cognitive)
                )
                val_idx = temp_idx[val_temp_idx]
                test_idx = temp_idx[test_temp_idx]
            else:
                val_idx, test_idx = train_test_split(
                    temp_idx, test_size=0.5, random_state=42
                )
        else:
            train_idx, temp_idx = train_test_split(
                indices, test_size=0.4, random_state=42
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, random_state=42
            )

        # Move to device
        spatial_emb = spatial_emb.to(self.device)
        genomic_emb = genomic_emb.to(self.device)
        temporal_emb = temporal_emb.to(self.device)
        targets = targets.to(self.device)

        # Optimizer and loss functions
        optimizer = torch.optim.AdamW(
            self.unified_model.parameters(), lr=5e-4, weight_decay=1e-3
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=15, factor=0.5, min_lr=1e-6
        )

        huber_loss = nn.HuberLoss(delta=1.0)
        bce_loss = nn.BCELoss()

        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        patience = 25
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training
            self.unified_model.train()
            optimizer.zero_grad()

            train_outputs = self.unified_model(
                spatial_emb[train_idx], genomic_emb[train_idx], temporal_emb[train_idx]
            )

            # Loss calculation
            motor_loss = huber_loss(
                train_outputs["motor_prediction"].squeeze(), targets[train_idx, 0]
            )
            cognitive_loss = bce_loss(
                train_outputs["cognitive_prediction"].squeeze(), targets[train_idx, 1]
            )

            # Regularization terms
            attention_reg = torch.mean(train_outputs["feature_importance"] ** 2) * 0.005
            ensemble_reg = 0.01 * (
                torch.mean(
                    train_outputs["motor_ensemble_info"]["ensemble_weights"] ** 2
                )
                + torch.mean(
                    train_outputs["cognitive_ensemble_info"]["ensemble_weights"] ** 2
                )
            )

            train_loss = (
                1.5 * motor_loss + cognitive_loss + attention_reg + ensemble_reg
            )
            train_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.unified_model.parameters(), max_norm=1.0
            )

            optimizer.step()

            # Validation
            self.unified_model.eval()
            with torch.no_grad():
                val_outputs = self.unified_model(
                    spatial_emb[val_idx], genomic_emb[val_idx], temporal_emb[val_idx]
                )

                val_motor_loss = huber_loss(
                    val_outputs["motor_prediction"].squeeze(), targets[val_idx, 0]
                )
                val_cognitive_loss = bce_loss(
                    val_outputs["cognitive_prediction"].squeeze(), targets[val_idx, 1]
                )

                val_loss = 1.5 * val_motor_loss + val_cognitive_loss

            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                best_model_state = self.unified_model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch:3d}: Train = {train_loss:.6f}, Val = {val_loss:.6f}"
                )

        # Restore best model
        if "best_model_state" in locals():
            self.unified_model.load_state_dict(best_model_state)

        # Final evaluation
        self.unified_model.eval()
        with torch.no_grad():
            test_outputs = self.unified_model(
                spatial_emb[test_idx], genomic_emb[test_idx], temporal_emb[test_idx]
            )

            motor_pred_norm = test_outputs["motor_prediction"].squeeze().cpu().numpy()
            cognitive_pred = (
                test_outputs["cognitive_prediction"].squeeze().cpu().numpy()
            )

            motor_true_norm = targets[test_idx, 0].cpu().numpy()
            cognitive_true = targets[test_idx, 1].cpu().numpy()

            # Denormalize motor predictions
            motor_pred = (
                motor_pred_norm * motor_std.cpu().numpy() + motor_mean.cpu().numpy()
            )
            motor_true = (
                motor_true_norm * motor_std.cpu().numpy() + motor_mean.cpu().numpy()
            )

            # Calculate metrics
            motor_r2 = r2_score(motor_true, motor_pred)
            motor_corr = (
                np.corrcoef(motor_true, motor_pred)[0, 1]
                if not np.any(np.isnan([motor_true, motor_pred]))
                else 0.0
            )

            cognitive_acc = accuracy_score(
                cognitive_true, (cognitive_pred > 0.5).astype(int)
            )

            if len(np.unique(cognitive_true)) > 1:
                cognitive_auc = roc_auc_score(cognitive_true, cognitive_pred)
            else:
                cognitive_auc = 0.5

        results = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "normalization_params": {
                "motor_mean": motor_mean.cpu().numpy(),
                "motor_std": motor_std.cpu().numpy(),
            },
            "test_metrics": {
                "motor_r2": motor_r2,
                "motor_correlation": motor_corr,
                "cognitive_accuracy": cognitive_acc,
                "cognitive_auc": cognitive_auc,
            },
            "test_predictions": {
                "motor": motor_pred,
                "cognitive": cognitive_pred,
                "motor_true": motor_true,
                "cognitive_true": cognitive_true,
            },
            "test_indices": test_idx,
            "train_indices": train_idx,
            "val_indices": val_idx,
        }

        logger.info("✅ Unified training completed!")
        logger.info(f"📈 Motor R²: {motor_r2:.4f}, Correlation: {motor_corr:.4f}")
        logger.info(
            f"🧠 Cognitive AUC: {cognitive_auc:.4f}, Accuracy: {cognitive_acc:.4f}"
        )

        return results

    def run_unified_system(self):
        """Run the complete unified GIMAN system."""
        logger.info("🎬 Running complete Unified GIMAN System...")

        # Load data and train
        self.load_multimodal_data()
        self.create_unified_embeddings()
        training_results = self.train_unified_system(num_epochs=100)

        return training_results


def main():
    """Main function for Phase 4 unified system."""
    logger.info("🎬 GIMAN Phase 4: Unified System with Research Analytics")

    # Initialize and run system
    data_integrator = RealDataPhase3Integration()
    data_integrator.load_and_prepare_data()

    results = run_phase4_experiment(data_integrator)

    # Summary
    print("" + "=" * 80)
    print("🎉 GIMAN Phase 4 Unified System Results")
    print("=" * 80)
    print(f"📊 PPMI patients: {len(data_integrator.patient_ids)}")
    print("🧠 Unified architecture: Cross-modal + Temporal + Ensemble")
    print(f"📈 Motor progression R²: {results['motor_r2']:.4f}")
    print(f"🧠 Cognitive conversion AUC: {results['cognitive_auc']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
