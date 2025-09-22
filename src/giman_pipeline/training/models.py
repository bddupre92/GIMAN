"""GIMAN Core GNN Backbone Implementation.

This module implements the Graph-Informed Multimodal Attention Network (GIMAN)
backbone architecture using PyTorch Geometric. The architecture follows a
3-layer GraphConv design with residual connections and multimodal integration.

Architecture Overview:
- Input Layer: 7 biomarker features per patient node
- Hidden Layers: 64 → 128 → 64 dimensional embeddings
- GraphConv layers with ReLU activation and dropout
- Residual connections for gradient flow
- Graph-level pooling for classification
- Binary classification (PD vs Healthy Control)
"""

from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, global_max_pool, global_mean_pool


class GIMANBackbone(nn.Module):
    """Core GIMAN backbone GNN architecture.

    This class implements the fundamental Graph Convolutional Network backbone
    for processing patient similarity graphs with biomarker features.

    Architecture:
    - Layer 1: GraphConv(7 → 64) + ReLU + Dropout(0.3)
    - Layer 2: GraphConv(64 → 128) + ReLU + Dropout(0.3)
    - Layer 3: GraphConv(128 → 64) + ReLU + Dropout(0.3)
    - Residual connection from Layer 1 to Layer 3
    - Graph pooling: Global mean + max pooling
    - Classification head: FC(128 → 2) for binary classification

    Attributes:
        input_dim (int): Number of input biomarker features (7)
        hidden_dims (List[int]): Hidden layer dimensions [64, 128, 64]
        output_dim (int): Number of output classes (2 for binary)
        dropout_rate (float): Dropout probability for regularization
        pooling_method (str): Graph pooling strategy ('mean', 'max', 'concat')
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dims: list[int] | None = None,
        output_dim: int = 2,
        dropout_rate: float = 0.3,
        pooling_method: str = "concat",
        use_residual: bool = True,
    ):
        """Initialize the GIMAN backbone architecture.

        Args:
            input_dim: Number of input biomarker features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output classes
            dropout_rate: Dropout probability for regularization
            pooling_method: Graph pooling method ('mean', 'max', 'concat')
            use_residual: Whether to use residual connections
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 64]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.pooling_method = pooling_method
        self.use_residual = use_residual

        # Validate architecture parameters
        if len(hidden_dims) != 3:
            raise ValueError("GIMAN backbone requires exactly 3 hidden layers")

        # Graph Convolutional Layers
        self.conv1 = GraphConv(input_dim, hidden_dims[0])
        self.conv2 = GraphConv(hidden_dims[0], hidden_dims[1])
        self.conv3 = GraphConv(hidden_dims[1], hidden_dims[2])

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])

        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)

        # Residual connection projection (if dimensions don't match)
        if use_residual and hidden_dims[0] != hidden_dims[2]:
            self.residual_proj = nn.Linear(hidden_dims[0], hidden_dims[2])
        else:
            self.residual_proj = None

        # Graph pooling and classification head
        pooled_dim = self._get_pooled_dimension(hidden_dims[2])
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, pooled_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(pooled_dim // 2, output_dim),
        )

        # Initialize weights
        self._initialize_weights()

    def _get_pooled_dimension(self, node_embed_dim: int) -> int:
        """Calculate pooled feature dimension based on pooling method.

        Args:
            node_embed_dim: Node embedding dimension

        Returns:
            Dimension after graph pooling
        """
        if self.pooling_method == "mean" or self.pooling_method == "max":
            return node_embed_dim
        elif self.pooling_method == "concat":
            return node_embed_dim * 2  # Mean + Max concatenation
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through GIMAN backbone.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges] (optional)
            batch: Batch assignment for multiple graphs (optional)

        Returns:
            Dictionary containing:
            - 'logits': Classification logits [batch_size, output_dim]
            - 'node_embeddings': Final node embeddings [num_nodes, hidden_dims[-1]]
            - 'graph_embedding': Graph-level embedding [batch_size, pooled_dim]
            - 'layer_embeddings': Embeddings from each layer
        """
        # Store intermediate embeddings for analysis
        layer_embeddings = {}

        # Layer 1: Input → 64
        h1 = self.conv1(x, edge_index, edge_weight)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)
        layer_embeddings["layer_1"] = h1

        # Layer 2: 64 → 128
        h2 = self.conv2(h1, edge_index, edge_weight)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = self.dropout(h2)
        layer_embeddings["layer_2"] = h2

        # Layer 3: 128 → 64
        h3 = self.conv3(h2, edge_index, edge_weight)
        h3 = self.bn3(h3)

        # Residual connection (Layer 1 → Layer 3)
        if self.use_residual:
            residual = self.residual_proj(h1) if self.residual_proj is not None else h1
            h3 = h3 + residual

        h3 = F.relu(h3)
        h3 = self.dropout(h3)
        layer_embeddings["layer_3"] = h3

        # Final node embeddings
        node_embeddings = h3

        # Graph-level pooling
        if batch is None:
            # Single graph case - create dummy batch
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        graph_embedding = self._pool_graph_features(node_embeddings, batch)

        # Classification
        logits = self.classifier(graph_embedding)

        return {
            "logits": logits,
            "node_embeddings": node_embeddings,
            "graph_embedding": graph_embedding,
            "layer_embeddings": layer_embeddings,
        }

    def _pool_graph_features(
        self, node_embeddings: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Apply graph-level pooling to aggregate node embeddings.

        Args:
            node_embeddings: Node embeddings [num_nodes, embed_dim]
            batch: Batch assignment [num_nodes]

        Returns:
            Graph-level embeddings [batch_size, pooled_dim]
        """
        if self.pooling_method == "mean":
            return global_mean_pool(node_embeddings, batch)
        elif self.pooling_method == "max":
            return global_max_pool(node_embeddings, batch)
        elif self.pooling_method == "concat":
            mean_pool = global_mean_pool(node_embeddings, batch)
            max_pool = global_max_pool(node_embeddings, batch)
            return torch.cat([mean_pool, max_pool], dim=1)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")

    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract node embeddings without classification.

        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_weight: Edge weights (optional)

        Returns:
            Node embeddings from final layer
        """
        with torch.no_grad():
            output = self.forward(x, edge_index, edge_weight)
            return output["node_embeddings"]

    def get_layer_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract embeddings from all layers for analysis.

        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_weight: Edge weights (optional)

        Returns:
            Dictionary of layer embeddings
        """
        with torch.no_grad():
            output = self.forward(x, edge_index, edge_weight)
            return output["layer_embeddings"]


class GIMANClassifier(nn.Module):
    """Complete GIMAN classifier combining backbone with additional components.

    This is the main model class that users should instantiate for training
    and inference. It wraps the GIMANBackbone with additional utilities.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dims: list[int] | None = None,
        output_dim: int = 2,
        dropout_rate: float = 0.3,
        pooling_method: str = "concat",
    ):
        """Initialize GIMAN classifier.

        Args:
            input_dim: Number of input biomarker features
            hidden_dims: Hidden layer dimensions
            output_dim: Number of output classes
            dropout_rate: Dropout probability
            pooling_method: Graph pooling method
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 64]

        self.backbone = GIMANBackbone(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            pooling_method=pooling_method,
        )

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, data: Data) -> dict[str, torch.Tensor]:
        """Forward pass using PyG Data object.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Dictionary with model outputs
        """
        return self.backbone(
            x=data.x,
            edge_index=data.edge_index,
            edge_weight=getattr(data, "edge_attr", None),
            batch=getattr(data, "batch", None),
        )

    def predict_proba(self, data: Data) -> torch.Tensor:
        """Get prediction probabilities.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Class probabilities [batch_size, num_classes]
        """
        with torch.no_grad():
            logits = self.forward(data)["logits"]
            return F.softmax(logits, dim=1)

    def predict(self, data: Data) -> torch.Tensor:
        """Get class predictions.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Predicted class indices [batch_size]
        """
        with torch.no_grad():
            logits = self.forward(data)["logits"]
            return torch.argmax(logits, dim=1)

    def get_model_info(self) -> dict:
        """Get model architecture information.

        Returns:
            Dictionary with model metadata
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": "GIMAN",
            "backbone_type": "GraphConv",
            "input_dim": self.input_dim,
            "hidden_dims": self.backbone.hidden_dims,
            "output_dim": self.output_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "pooling_method": self.backbone.pooling_method,
            "dropout_rate": self.backbone.dropout_rate,
            "use_residual": self.backbone.use_residual,
        }


def create_giman_model(
    model_type: str = "backbone",
    input_dim: int = 7,
    hidden_dims: list[int] | None = None,
    output_dim: int = 2,
    dropout_rate: float = 0.3,
    pooling_method: str = "max",
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Create GIMAN model instance with specified configuration.

    Args:
        model_type: Type of model to create ('backbone' or 'classifier')
        input_dim: Dimension of input features
        hidden_dims: List of hidden layer dimensions
        output_dim: Dimension of output (2 for binary classification)
        dropout_rate: Dropout rate for regularization
        pooling_method: Graph pooling method
        device: Device to place model on ('cpu' or 'cuda')

    Returns:
        Tuple of (model, config_dict) where config_dict contains model metadata
    """
    if hidden_dims is None:
        hidden_dims = [64, 128, 64]

    # Create model configuration
    if model_type == "backbone":
        model = GIMANBackbone(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
        )
        config = {
            "model_type": "backbone",
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "dropout_rate": dropout_rate,
            "parameters": sum(p.numel() for p in model.parameters()),
        }
    else:
        model = GIMANClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            pooling_method=pooling_method,
        )
        config = {
            "model_type": "classifier",
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "output_dim": output_dim,
            "dropout_rate": dropout_rate,
            "pooling_method": pooling_method,
            "parameters": sum(p.numel() for p in model.parameters()),
        }

    model = model.to(device)

    print(f"🔧 Created GIMAN {model_type} model:")
    print(f"   - Parameters: {config['parameters']:,}")
    print(f"   - Architecture: {input_dim} → {' → '.join(map(str, hidden_dims))} → {output_dim if model_type == 'classifier' else 'embeddings'}")

    return model, config
