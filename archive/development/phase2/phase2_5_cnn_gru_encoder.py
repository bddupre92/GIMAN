#!/usr/bin/env python3
"""Phase 2.5: 3D CNN + GRU Spatiotemporal Encoder Architecture

This script implements the hybrid 3D CNN + GRU encoder as specified in Stage II
of the GIMAN research plan. The model processes longitudinal 3D brain scans to
generate comprehensive spatiotemporal embeddings.

Architecture:
1. 3D CNN Feature Extractor: Learns spatial patterns from individual 3D scans
2. GRU Temporal Encoder: Models temporal evolution of spatial features
3. Output: 256-dimensional spatiotemporal embedding per patient

Key Features:
- Multi-modal input (sMRI + DAT-SPECT)
- Grad-CAM compatible for interpretability
- Memory-efficient implementation
- Flexible architecture configuration
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class CNNConfig:
    """Configuration for 3D CNN feature extractor."""

    # Input dimensions
    input_channels: int = 2  # sMRI + DAT-SPECT
    input_shape: tuple[int, int, int] = (96, 96, 96)

    # CNN architecture
    base_filters: int = 32
    num_blocks: int = 4
    growth_factor: int = 2
    kernel_size: int = 3
    pool_size: int = 2

    # Regularization
    dropout_rate: float = 0.3
    batch_norm: bool = True

    # Output
    feature_dim: int = 256


@dataclass
class GRUConfig:
    """Configuration for GRU temporal encoder."""

    input_size: int = 256  # From CNN feature extractor
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False
    output_size: int = 256


@dataclass
class SpatiotemporalConfig:
    """Complete configuration for spatiotemporal encoder."""

    cnn_config: CNNConfig
    gru_config: GRUConfig
    max_timepoints: int = 5

    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        return {
            "cnn_config": asdict(self.cnn_config),
            "gru_config": asdict(self.gru_config),
            "max_timepoints": self.max_timepoints,
        }


class ResidualBlock3D(nn.Module):
    """3D Residual block for feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batch_norm,
        )
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            bias=not use_batch_norm,
        )

        self.batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = nn.BatchNorm3d(out_channels)
            self.bn2 = nn.BatchNorm3d(out_channels)

        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv3d(
                in_channels, out_channels, 1, stride=stride, bias=False
            )
            if use_batch_norm:
                self.skip_bn = nn.BatchNorm3d(out_channels)
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # First convolution
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = F.relu(out)

        # Second convolution
        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)

        # Skip connection
        if self.skip is not None:
            identity = self.skip(identity)
            if hasattr(self, "skip_bn"):
                identity = self.skip_bn(identity)

        out += identity
        out = F.relu(out)

        return out


class CNN3DFeatureExtractor(nn.Module):
    """3D CNN for extracting spatial features from brain scans."""

    def __init__(self, config: CNNConfig):
        super().__init__()
        self.config = config

        # Calculate channel progression
        channels = [config.input_channels]
        for i in range(config.num_blocks):
            channels.append(config.base_filters * (config.growth_factor**i))

        # Build encoder blocks
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        for i in range(config.num_blocks):
            # Residual block
            block = ResidualBlock3D(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=config.kernel_size,
                stride=1,
                use_batch_norm=config.batch_norm,
            )
            self.blocks.append(block)

            # Pooling layer
            if i < config.num_blocks - 1:  # No pooling after last block
                pool = nn.MaxPool3d(config.pool_size, stride=config.pool_size)
                self.pools.append(pool)

        # Calculate final spatial dimensions
        final_spatial_dim = self._calculate_final_spatial_dim()
        final_channels = channels[-1]

        # Global average pooling and final layers
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(final_channels, config.feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.feature_dim * 2, config.feature_dim),
        )

        logger.info(
            f"CNN3D initialized: {channels[0]} -> {channels[-1]} channels, "
            f"output dim: {config.feature_dim}"
        )

    def _calculate_final_spatial_dim(self) -> int:
        """Calculate final spatial dimensions after all pooling operations."""
        dim = self.config.input_shape[0]  # Assume cubic input
        num_pools = len(self.pools)

        for _ in range(num_pools):
            dim = dim // self.config.pool_size

        return dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through 3D CNN.

        Args:
            x: Input tensor of shape (batch_size, channels, depth, height, width)

        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        batch_size = x.size(0)

        # Pass through convolutional blocks
        for i, block in enumerate(self.blocks):
            x = block(x)

            # Apply pooling (except after last block)
            if i < len(self.pools):
                x = self.pools[i](x)

        # Global average pooling
        x = self.global_avg_pool(x)  # Shape: (batch_size, channels, 1, 1, 1)
        x = x.view(batch_size, -1)  # Shape: (batch_size, channels)

        # Feature projection
        features = self.feature_projection(x)  # Shape: (batch_size, feature_dim)

        return features


class GRUTemporalEncoder(nn.Module):
    """GRU for modeling temporal evolution of spatial features."""

    def __init__(self, config: GRUConfig):
        super().__init__()
        self.config = config

        self.gru = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

        # Calculate GRU output size
        gru_output_size = config.hidden_size * (2 if config.bidirectional else 1)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(gru_output_size, config.output_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_size, config.output_size),
        )

        logger.info(
            f"GRU initialized: {config.input_size} -> {gru_output_size} -> {config.output_size}"
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass through GRU.

        Args:
            x: Input tensor of shape (batch_size, max_seq_len, input_size)
            lengths: Actual sequence lengths for each batch item

        Returns:
            Spatiotemporal embedding of shape (batch_size, output_size)
        """
        batch_size, max_seq_len, input_size = x.shape

        # Pack sequences if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # Pass through GRU
        gru_out, hidden = self.gru(x)

        # Unpack if necessary
        if lengths is not None:
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)

        # Use final hidden state (last layer)
        if self.config.bidirectional:
            # Concatenate forward and backward final states
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            final_hidden = hidden[-1]  # Last layer

        # Project to output size
        spatiotemporal_embedding = self.output_projection(final_hidden)

        return spatiotemporal_embedding


class SpatiotemporalEncoder(nn.Module):
    """Complete spatiotemporal encoder combining 3D CNN and GRU."""

    def __init__(self, config: SpatiotemporalConfig):
        super().__init__()
        self.config = config

        # Initialize components
        self.cnn_feature_extractor = CNN3DFeatureExtractor(config.cnn_config)
        self.gru_temporal_encoder = GRUTemporalEncoder(config.gru_config)

        logger.info("SpatiotemporalEncoder initialized successfully")

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through complete spatiotemporal encoder.

        Args:
            batch: Dictionary containing:
                - 'imaging_data': (batch_size, max_timepoints, channels, D, H, W)
                - 'num_timepoints': (batch_size,) actual number of timepoints

        Returns:
            Spatiotemporal embeddings of shape (batch_size, output_size)
        """
        imaging_data = batch["imaging_data"]  # (B, T, C, D, H, W)
        num_timepoints = batch["num_timepoints"]  # (B,)

        batch_size, max_timepoints, channels, depth, height, width = imaging_data.shape

        # Reshape for CNN processing: (B*T, C, D, H, W)
        reshaped_data = imaging_data.view(
            batch_size * max_timepoints, channels, depth, height, width
        )

        # Extract spatial features using CNN
        spatial_features = self.cnn_feature_extractor(
            reshaped_data
        )  # (B*T, feature_dim)

        # Reshape back to sequences: (B, T, feature_dim)
        feature_dim = spatial_features.size(1)
        sequence_features = spatial_features.view(
            batch_size, max_timepoints, feature_dim
        )

        # Process temporal sequences with GRU
        spatiotemporal_embeddings = self.gru_temporal_encoder(
            sequence_features, lengths=num_timepoints
        )

        return spatiotemporal_embeddings

    def get_cnn_features(self, single_scan: torch.Tensor) -> torch.Tensor:
        """Extract CNN features from a single scan (for Grad-CAM).

        Args:
            single_scan: Single 3D scan of shape (channels, depth, height, width)

        Returns:
            CNN features of shape (feature_dim,)
        """
        # Add batch dimension
        scan_batch = single_scan.unsqueeze(0)  # (1, C, D, H, W)

        # Extract features
        features = self.cnn_feature_extractor(scan_batch)  # (1, feature_dim)

        return features.squeeze(0)  # (feature_dim,)

    def save_config(self, filepath: Path) -> None:
        """Save model configuration."""
        config_dict = self.config.to_dict()
        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load_config(cls, filepath: Path) -> "SpatiotemporalConfig":
        """Load model configuration."""
        with open(filepath) as f:
            config_dict = json.load(f)

        cnn_config = CNNConfig(**config_dict["cnn_config"])
        gru_config = GRUConfig(**config_dict["gru_config"])

        return SpatiotemporalConfig(
            cnn_config=cnn_config,
            gru_config=gru_config,
            max_timepoints=config_dict["max_timepoints"],
        )


def create_spatiotemporal_encoder(
    input_shape: tuple[int, int, int] = (96, 96, 96),
    feature_dim: int = 256,
    max_timepoints: int = 5,
) -> SpatiotemporalEncoder:
    """Create a spatiotemporal encoder with default configuration."""
    # CNN configuration
    cnn_config = CNNConfig(
        input_channels=2,  # sMRI + DAT-SPECT
        input_shape=input_shape,
        base_filters=32,
        num_blocks=4,
        feature_dim=feature_dim,
    )

    # GRU configuration
    gru_config = GRUConfig(
        input_size=feature_dim,
        hidden_size=feature_dim,
        num_layers=2,
        output_size=feature_dim,
    )

    # Complete configuration
    config = SpatiotemporalConfig(
        cnn_config=cnn_config, gru_config=gru_config, max_timepoints=max_timepoints
    )

    return SpatiotemporalEncoder(config)


def test_model_architecture():
    """Test the spatiotemporal encoder architecture."""
    logger.info("Testing spatiotemporal encoder architecture...")

    # Create model
    model = create_spatiotemporal_encoder(
        input_shape=(64, 64, 64),  # Smaller for testing
        feature_dim=128,
        max_timepoints=3,
    )

    # Create dummy batch
    batch_size = 2
    max_timepoints = 3
    channels = 2
    depth, height, width = 64, 64, 64

    dummy_batch = {
        "imaging_data": torch.randn(
            batch_size, max_timepoints, channels, depth, height, width
        ),
        "num_timepoints": torch.tensor([3, 2]),  # Variable sequence lengths
    }

    # Test forward pass
    model.eval()
    with torch.no_grad():
        embeddings = model(dummy_batch)

    logger.info(f"Input shape: {dummy_batch['imaging_data'].shape}")
    logger.info(f"Output shape: {embeddings.shape}")
    logger.info(f"Expected output shape: ({batch_size}, 128)")

    # Test single scan feature extraction
    single_scan = torch.randn(channels, depth, height, width)
    cnn_features = model.get_cnn_features(single_scan)
    logger.info(f"Single scan CNN features shape: {cnn_features.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    logger.info("✅ Architecture test completed successfully!")

    return model


def main():
    """Main function for testing."""
    test_model_architecture()


if __name__ == "__main__":
    main()
