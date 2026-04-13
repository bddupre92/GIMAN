#!/usr/bin/env python3
"""Phase 2.1: Spatiotemporal Imaging Encoder for GIMAN Prognostic Architecture

This module implements a hybrid 3D CNN + GRU encoder for longitudinal neuroimaging data,
specifically designed for DAT-SPECT and structural MRI progression modeling.

Architecture:
- 3D CNN spatial encoder for multi-region neuroimaging features
- GRU temporal encoder for longitudinal evolution modeling
- Attention-based fusion for variable-length sequences
- Prognostic embedding output for downstream hub integration

Author: GitHub Copilot
Date: 2025-01-26
Version: 1.0.0 - Initial spatiotemporal encoder implementation
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


class SpatiotemporalImagingDataset(Dataset):
    """Dataset class for longitudinal neuroimaging data.

    Handles variable-length sequences and creates proper 3D spatial tensors
    from multi-region DAT-SPECT and structural MRI features.
    """

    def __init__(
        self,
        longitudinal_data: pd.DataFrame,
        imaging_features: list[str],
        max_sequence_length: int = 8,
        spatial_dims: tuple[int, int, int] = (
            2,
            3,
            1,
        ),  # (bilateral, regions, modalities)
    ):
        """Initialize spatiotemporal imaging dataset.

        Args:
            longitudinal_data: DataFrame with PATNO, EVENT_ID, and imaging features
            imaging_features: List of imaging feature column names
            max_sequence_length: Maximum number of time points to include
            spatial_dims: Dimensions for 3D spatial tensor construction
        """
        self.data = longitudinal_data
        self.imaging_features = imaging_features
        self.max_sequence_length = max_sequence_length
        self.spatial_dims = spatial_dims

        # Create sequences for each patient
        self.sequences = self._create_sequences()

        # Fit scaler on all data
        self.scaler = StandardScaler()
        all_features = []
        for seq in self.sequences:
            for timepoint in seq["features"]:
                all_features.append(timepoint)
        self.scaler.fit(all_features)

        print(f"Created {len(self.sequences)} spatiotemporal sequences")
        print(f"Spatial tensor dimensions: {spatial_dims}")

    def _create_sequences(self) -> list[dict]:
        """Create patient-level longitudinal sequences."""
        sequences = []

        for patno in self.data.PATNO.unique():
            patient_data = self.data[patno == self.data.PATNO].copy()

            # Filter to have complete imaging data
            complete_mask = patient_data[self.imaging_features].notna().all(axis=1)
            patient_data = patient_data[complete_mask]

            if len(patient_data) < 2:  # Need at least 2 timepoints
                continue

            # Sort by visit order (using EVENT_ID as proxy)
            visit_order = {
                "BL": 0,
                "SC": 0,
                "V01": 1,
                "V02": 2,
                "V04": 4,
                "V06": 6,
                "V08": 8,
                "V10": 10,
                "V12": 12,
            }
            patient_data["visit_order"] = patient_data["EVENT_ID"].map(
                lambda x: visit_order.get(x, 99)
            )
            patient_data = patient_data.sort_values("visit_order")

            # Limit sequence length
            if len(patient_data) > self.max_sequence_length:
                patient_data = patient_data.head(self.max_sequence_length)

            # Extract features and create sequence
            features = patient_data[self.imaging_features].values
            event_ids = patient_data["EVENT_ID"].tolist()

            sequences.append(
                {
                    "patno": patno,
                    "features": features,
                    "event_ids": event_ids,
                    "sequence_length": len(features),
                }
            )

        return sequences

    def _create_spatial_tensor(self, features: np.ndarray) -> torch.Tensor:
        """Convert flat feature vector to 3D spatial tensor.

        For DAT-SPECT: organizes bilateral (L/R) x regions (putamen/caudate) x modalities
        """
        # Assume features are ordered: [overall, left, right] for each region
        # Shape: (bilateral=2, regions=3, modalities=1)

        if len(features) == 6:  # Putamen and Caudate, bilateral
            # [PUTAMEN_REF_CWM, PUTAMEN_L_REF_CWM, PUTAMEN_R_REF_CWM,
            #  CAUDATE_REF_CWM, CAUDATE_L_REF_CWM, CAUDATE_R_REF_CWM]
            spatial_tensor = np.zeros((2, 3, 1))  # (bilateral, regions, modalities)

            # Putamen
            spatial_tensor[0, 0, 0] = features[1]  # Left putamen
            spatial_tensor[1, 0, 0] = features[2]  # Right putamen
            spatial_tensor[0, 1, 0] = features[0]  # Overall putamen (as central)

            # Caudate
            spatial_tensor[0, 2, 0] = features[4]  # Left caudate
            spatial_tensor[1, 2, 0] = features[5]  # Right caudate
            spatial_tensor[1, 1, 0] = features[3]  # Overall caudate (as central)

        else:
            # Fallback: reshape into spatial dimensions
            spatial_tensor = features.reshape(self.spatial_dims)

        return torch.FloatTensor(spatial_tensor)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sequence = self.sequences[idx]

        # Scale features
        scaled_features = self.scaler.transform(sequence["features"])

        # Create spatial tensors for each timepoint
        spatial_sequence = []
        for t in range(len(scaled_features)):
            spatial_tensor = self._create_spatial_tensor(scaled_features[t])
            spatial_sequence.append(spatial_tensor.unsqueeze(0))  # Add time dimension

        # Pad sequence to max length
        while len(spatial_sequence) < self.max_sequence_length:
            # Pad with zeros
            spatial_sequence.append(torch.zeros_like(spatial_sequence[0]))

        # Stack into 4D tensor: (time, channels=1, height, width, depth)
        spatial_sequence = torch.stack(spatial_sequence[: self.max_sequence_length])

        return {
            "spatial_sequence": spatial_sequence,
            "sequence_length": torch.tensor(
                sequence["sequence_length"], dtype=torch.long
            ),
            "patno": sequence["patno"],
        }


class SpatialCNNEncoder(nn.Module):
    """3D CNN for spatial feature extraction from neuroimaging data.

    Processes multi-region, bilateral neuroimaging features to create
    rich spatial representations for temporal modeling.
    """

    def __init__(
        self,
        input_dims: tuple[int, int, int] = (2, 3, 1),
        hidden_dim: int = 64,
        output_dim: int = 128,
    ):
        """Initialize 3D CNN spatial encoder.

        Args:
            input_dims: Input spatial dimensions (bilateral, regions, modalities)
            hidden_dim: Hidden layer dimensionality
            output_dim: Output embedding dimensionality
        """
        super(SpatialCNNEncoder, self).__init__()

        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(1, hidden_dim // 2, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv3d(
            hidden_dim // 2, hidden_dim, kernel_size=2, stride=1, padding=0
        )

        # Batch normalization and dropout
        self.bn1 = nn.BatchNorm3d(hidden_dim // 2)
        self.bn2 = nn.BatchNorm3d(hidden_dim)
        self.dropout = nn.Dropout3d(0.2)

        # Calculate flattened size after convolutions
        self.flattened_size = self._calculate_flattened_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to prevent gradient issues."""
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _calculate_flattened_size(self) -> int:
        """Calculate size after convolutions for FC layer."""
        with torch.no_grad():
            x = torch.zeros(1, 1, *self.input_dims)
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            return x.view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spatial CNN.

        Args:
            x: Input tensor (batch_size, 1, bilateral, regions, modalities)

        Returns:
            Spatial embeddings (batch_size, output_dim)
        """
        # 3D convolutions with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        # Flatten for FC layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class TemporalGRUEncoder(nn.Module):
    """GRU-based temporal encoder for longitudinal progression modeling.

    Processes sequences of spatial embeddings to capture disease progression
    dynamics over time.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        output_dim: int = 256,
        bidirectional: bool = True,
    ):
        """Initialize temporal GRU encoder.

        Args:
            input_dim: Dimensionality of spatial embeddings
            hidden_dim: GRU hidden state dimensionality
            num_layers: Number of GRU layers
            output_dim: Output embedding dimensionality
            bidirectional: Whether to use bidirectional GRU
        """
        super(TemporalGRUEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional

        # GRU layer
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if num_layers > 1 else 0,
        )

        # Attention mechanism for variable-length sequences
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.MultiheadAttention(
            gru_output_dim, num_heads=8, dropout=0.1, batch_first=True
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to prevent gradient issues."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(
        self, spatial_embeddings: torch.Tensor, sequence_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through temporal GRU.

        Args:
            spatial_embeddings: Spatial embeddings (batch_size, max_seq_len, input_dim)
            sequence_lengths: Actual sequence lengths (batch_size,)

        Returns:
            Temporal embeddings (batch_size, output_dim)
        """
        batch_size, max_seq_len = spatial_embeddings.shape[:2]

        # Pack sequences for efficient processing
        packed_input = nn.utils.rnn.pack_padded_sequence(
            spatial_embeddings,
            sequence_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # GRU forward pass
        packed_output, hidden = self.gru(packed_input)

        # Unpack sequences
        gru_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )

        # Create attention mask for padding (only for actual sequence length)
        actual_seq_len = gru_output.shape[1]  # Use actual GRU output length
        attention_mask = torch.zeros(
            batch_size, actual_seq_len, dtype=torch.bool, device=gru_output.device
        )
        for i, length in enumerate(sequence_lengths):
            if length < actual_seq_len:
                attention_mask[i, length:] = True

        # Self-attention over temporal sequence
        attended_output, _ = self.attention(
            gru_output, gru_output, gru_output, key_padding_mask=attention_mask
        )

        # Global temporal representation (mean of attended outputs)
        temporal_embedding = []
        for i, length in enumerate(sequence_lengths):
            # Take mean over actual sequence length, limited by attended output length
            actual_length = min(length.item(), attended_output.shape[1])
            seq_embedding = attended_output[i, :actual_length].mean(dim=0)
            temporal_embedding.append(seq_embedding)

        temporal_embedding = torch.stack(temporal_embedding)

        # Output projection
        output = self.output_projection(temporal_embedding)

        return output


class SpatiotemporalImagingEncoder(nn.Module):
    """Complete spatiotemporal imaging encoder combining 3D CNN and GRU.

    This encoder serves as the first "spoke" in the multimodal hub-and-spoke
    architecture, processing longitudinal neuroimaging data into prognostic
    embeddings for downstream fusion.
    """

    def __init__(
        self,
        spatial_dims: tuple[int, int, int] = (2, 3, 1),
        spatial_hidden_dim: int = 64,
        spatial_output_dim: int = 128,
        temporal_hidden_dim: int = 256,
        temporal_num_layers: int = 2,
        final_output_dim: int = 256,
    ):
        """Initialize complete spatiotemporal encoder.

        Args:
            spatial_dims: Input spatial dimensions
            spatial_hidden_dim: CNN hidden dimensionality
            spatial_output_dim: CNN output dimensionality
            temporal_hidden_dim: GRU hidden dimensionality
            temporal_num_layers: Number of GRU layers
            final_output_dim: Final embedding dimensionality
        """
        super(SpatiotemporalImagingEncoder, self).__init__()

        self.spatial_encoder = SpatialCNNEncoder(
            input_dims=spatial_dims,
            hidden_dim=spatial_hidden_dim,
            output_dim=spatial_output_dim,
        )

        self.temporal_encoder = TemporalGRUEncoder(
            input_dim=spatial_output_dim,
            hidden_dim=temporal_hidden_dim,
            num_layers=temporal_num_layers,
            output_dim=final_output_dim,
        )

        self.final_output_dim = final_output_dim

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through complete spatiotemporal encoder.

        Args:
            batch: Dictionary containing spatial_sequence and sequence_lengths

        Returns:
            Spatiotemporal embeddings (batch_size, final_output_dim)
        """
        spatial_sequence = batch[
            "spatial_sequence"
        ]  # (batch_size, max_seq_len, 1, h, w, d)
        sequence_lengths = batch["sequence_length"]  # (batch_size,)

        batch_size, max_seq_len = spatial_sequence.shape[:2]

        # Process each timepoint through spatial CNN
        spatial_embeddings = []
        for t in range(max_seq_len):
            timepoint_data = spatial_sequence[:, t]  # (batch_size, 1, h, w, d)
            spatial_embedding = self.spatial_encoder(timepoint_data)
            spatial_embeddings.append(spatial_embedding)

        # Stack spatial embeddings into temporal sequence
        spatial_embeddings = torch.stack(
            spatial_embeddings, dim=1
        )  # (batch_size, max_seq_len, spatial_dim)

        # Process through temporal encoder
        spatiotemporal_embedding = self.temporal_encoder(
            spatial_embeddings, sequence_lengths
        )

        return spatiotemporal_embedding


def load_spatiotemporal_data(
    longitudinal_path: str = "data/01_processed/giman_corrected_longitudinal_dataset.csv",
    enhanced_path: str = "data/enhanced/enhanced_giman_12features_v1.1.0_20250924_075919.csv",
) -> tuple[pd.DataFrame, list[str]]:
    """Load and prepare spatiotemporal imaging data for encoder training.

    Returns:
        Longitudinal imaging data and list of core imaging features
    """
    print("Loading spatiotemporal imaging data...")

    # Load datasets
    df_long = pd.read_csv(longitudinal_path, low_memory=False)
    df_enhanced = pd.read_csv(enhanced_path)

    # Filter to enhanced model patients
    enhanced_patients = set(df_enhanced.PATNO.unique())
    df_imaging = df_long[df_long.PATNO.isin(enhanced_patients)].copy()

    # Define core imaging features for spatiotemporal modeling
    core_imaging_features = [
        "PUTAMEN_REF_CWM",
        "PUTAMEN_L_REF_CWM",
        "PUTAMEN_R_REF_CWM",
        "CAUDATE_REF_CWM",
        "CAUDATE_L_REF_CWM",
        "CAUDATE_R_REF_CWM",
    ]

    # Filter to rows with complete imaging data
    imaging_mask = df_imaging[core_imaging_features].notna().all(axis=1)
    df_imaging = df_imaging[imaging_mask].copy()

    print(f"Loaded data for {df_imaging.PATNO.nunique()} patients")
    print(f"Total imaging observations: {len(df_imaging):,}")
    print(f"Core imaging features: {len(core_imaging_features)}")

    return df_imaging, core_imaging_features


def train_spatiotemporal_encoder(
    dataset: SpatiotemporalImagingDataset,
    model: SpatiotemporalImagingEncoder,
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
) -> dict[str, list[float]]:
    """Train the spatiotemporal imaging encoder using self-supervised learning.

    Uses next-timepoint prediction as the pretext task for learning
    meaningful spatiotemporal representations.
    """
    print(f"Training spatiotemporal encoder for {num_epochs} epochs...")

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    # Training loop
    training_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            optimizer.zero_grad()

            # Forward pass
            embeddings = model(batch)

            # Contrastive self-supervised loss
            # Normalize embeddings to unit sphere
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)

            # Compute pairwise similarities
            batch_size = embeddings_norm.shape[0]
            if batch_size > 1:
                similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())

                # Create mask to exclude self-similarities
                mask = ~torch.eye(
                    batch_size, dtype=torch.bool, device=embeddings.device
                )

                # Contrastive loss: minimize average pairwise similarity (encourage diversity)
                contrastive_loss = similarity_matrix[mask].mean()

                # Regularization: prevent collapse by ensuring non-zero norms
                magnitude_reg = (
                    torch.clamp(embeddings.norm(dim=1), min=1e-8).log().mean()
                )

                loss = contrastive_loss - 0.1 * magnitude_reg
            else:
                # Fallback for batch_size = 1
                loss = embeddings.norm(dim=1).mean()

            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(
                    f"Warning: NaN/Inf loss detected at epoch {epoch + 1}, batch {batch_idx}"
                )
                print(
                    f"Embedding stats: mean={embeddings.mean().item():.6f}, std={embeddings.std().item():.6f}"
                )
                continue

            # Backward pass
            loss.backward()

            # Check for NaN gradients
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN gradient detected in {name}")
                    has_nan_grad = True
                    break

            if has_nan_grad:
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        training_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}/{num_epochs} - Loss: {avg_loss:.6f}")

    return {"training_loss": training_losses}


def evaluate_spatiotemporal_encoder(
    model: SpatiotemporalImagingEncoder, dataset: SpatiotemporalImagingDataset
) -> dict[str, np.ndarray]:
    """Evaluate the trained spatiotemporal encoder.

    Returns embeddings and analysis of learned representations.
    """
    print("Evaluating spatiotemporal encoder...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Extract embeddings
    all_embeddings = []
    all_patnos = []
    all_seq_lengths = []

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Get embeddings
            embeddings = model(batch)

            all_embeddings.append(embeddings.cpu().numpy())
            all_patnos.extend(batch["patno"])
            all_seq_lengths.extend(batch["sequence_length"].cpu().numpy())

    # Concatenate results
    embeddings = np.vstack(all_embeddings)
    patnos = np.array(all_patnos)
    seq_lengths = np.array(all_seq_lengths)

    print(f"Generated embeddings for {len(embeddings)} patients")
    print(f"Embedding dimensionality: {embeddings.shape[1]}")

    return {"embeddings": embeddings, "patnos": patnos, "sequence_lengths": seq_lengths}


def visualize_spatiotemporal_results(
    training_history: dict[str, list[float]],
    evaluation_results: dict[str, np.ndarray],
    save_path: str = "visualizations/enhanced_progression/phase2_1_spatiotemporal_encoder.png",
) -> None:
    """Create comprehensive visualization of spatiotemporal encoder results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Phase 2.1: Spatiotemporal Imaging Encoder Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # Training loss curve
    axes[0, 0].plot(training_history["training_loss"], "b-", linewidth=2)
    axes[0, 0].set_title("Training Loss Curve", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Self-Supervised Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # Embedding dimensionality distribution
    embeddings = evaluation_results["embeddings"]
    if not np.isnan(embeddings).any():
        axes[0, 1].hist(np.std(embeddings, axis=0), bins=30, alpha=0.7, color="green")
        axes[0, 1].set_title("Embedding Feature Variance", fontweight="bold")
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "NaN values detected\nin embeddings",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )
        axes[0, 1].set_title("Embedding Feature Variance (NaN)", fontweight="bold")
    axes[0, 1].set_xlabel("Standard Deviation")
    axes[0, 1].set_ylabel("Number of Features")
    axes[0, 1].grid(True, alpha=0.3)

    # Sequence length distribution
    seq_lengths = evaluation_results["sequence_lengths"]
    axes[1, 0].hist(
        seq_lengths,
        bins=np.arange(0.5, max(seq_lengths) + 1.5, 1),
        alpha=0.7,
        color="orange",
    )
    axes[1, 0].set_title("Temporal Sequence Lengths", fontweight="bold")
    axes[1, 0].set_xlabel("Number of Timepoints")
    axes[1, 0].set_ylabel("Number of Patients")
    axes[1, 0].grid(True, alpha=0.3)

    # Embedding space visualization (first 2 PCs)
    if not np.isnan(embeddings).any():
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        scatter = axes[1, 1].scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=seq_lengths,
            cmap="viridis",
            alpha=0.6,
        )
        axes[1, 1].set_title("Spatiotemporal Embedding Space (PCA)", fontweight="bold")
        axes[1, 1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        axes[1, 1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label("Sequence Length")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Cannot visualize\nNaN embeddings",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_title("Embedding Space (NaN)", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {save_path}")
    plt.close()


def main():
    """Main execution function for Phase 2.1 spatiotemporal encoder development."""
    print("=== PHASE 2.1: SPATIOTEMPORAL IMAGING ENCODER ===")
    print("Implementing 3D CNN + GRU hybrid architecture for longitudinal neuroimaging")

    # Load spatiotemporal data
    df_imaging, core_features = load_spatiotemporal_data()

    # Create spatiotemporal dataset
    print("\nCreating spatiotemporal dataset...")
    dataset = SpatiotemporalImagingDataset(
        longitudinal_data=df_imaging,
        imaging_features=core_features,
        max_sequence_length=8,
        spatial_dims=(2, 3, 1),  # bilateral, regions, modalities
    )

    # Initialize spatiotemporal encoder
    print("\nInitializing spatiotemporal encoder...")
    model = SpatiotemporalImagingEncoder(
        spatial_dims=(2, 3, 1),
        spatial_hidden_dim=64,
        spatial_output_dim=128,
        temporal_hidden_dim=256,
        temporal_num_layers=2,
        final_output_dim=256,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Train the encoder
    print("\nTraining spatiotemporal encoder...")
    training_history = train_spatiotemporal_encoder(
        dataset=dataset, model=model, num_epochs=50, batch_size=16, learning_rate=1e-4
    )

    # Evaluate the encoder
    print("\nEvaluating spatiotemporal encoder...")
    evaluation_results = evaluate_spatiotemporal_encoder(model, dataset)

    # Create visualizations
    print("\nCreating visualizations...")
    visualize_spatiotemporal_results(training_history, evaluation_results)

    # Save the trained model
    model_path = "models/spatiotemporal_imaging_encoder_phase2_1.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "spatial_dims": (2, 3, 1),
                "spatial_hidden_dim": 64,
                "spatial_output_dim": 128,
                "temporal_hidden_dim": 256,
                "temporal_num_layers": 2,
                "final_output_dim": 256,
            },
            "training_history": training_history,
            "evaluation_results": evaluation_results,
            "core_features": core_features,
        },
        model_path,
    )

    print(f"\nModel saved to: {model_path}")

    print("\n=== PHASE 2.1 COMPLETION SUMMARY ===")
    print(f"✅ Spatiotemporal encoder trained on {len(dataset)} longitudinal sequences")
    print(f"✅ 3D CNN spatial encoder: {(2, 3, 1)} → 128 dimensions")
    print("✅ GRU temporal encoder: 128 → 256 dimensions")
    print(f"✅ Total model parameters: {total_params:,}")
    print("✅ Final embedding dimensionality: 256")
    print("✅ Trained on DAT-SPECT longitudinal progression data")
    print("\n🚀 Ready for Phase 2.2: Genomic Sequence Encoder development!")


if __name__ == "__main__":
    main()
