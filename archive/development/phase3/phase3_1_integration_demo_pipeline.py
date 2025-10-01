#!/usr/bin/env python3
"""GIMAN Phase 3.1: Graph Attention Network Integration

This script demonstrates the integration of Phase 3.1 Graph Attention Network
with the existing GIMAN pipeline infrastructure, including:
- Phase 2 encoder integration (spatiotemporal + genomic)
- Patient similarity graph utilization
- Multimodal fusion and prognostic prediction

Author: GIMAN Development Team
Date: September 24, 2025
Phase: 3.1 Integration & Demonstration
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import GIMAN components
from src.giman_pipeline.modeling.patient_similarity import PatientSimilarityGraph
from src.giman_pipeline.models.graph_attention_network import (
    GATTrainer,
    MultiModalGraphAttention,
    Phase3DataIntegrator,
    create_phase3_gat_model,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Phase3IntegrationDemo:
    """Complete integration demonstration for Phase 3.1 Graph Attention Network.

    Demonstrates:
    1. Integration with existing patient similarity infrastructure
    2. Phase 2 encoder output utilization
    3. GAT training and validation
    4. Prognostic prediction evaluation
    5. Visualization and analysis
    """

    def __init__(self, device: torch.device | None = None):
        """Initialize Phase 3 integration demonstration."""
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"🚀 Phase 3.1 Integration Demo initialized on {self.device}")

        # Initialize components
        self.similarity_constructor = PatientSimilarityGraph()
        self.data_integrator = Phase3DataIntegrator(
            similarity_graph_constructor=self.similarity_constructor, device=self.device
        )

        # Data storage
        self.spatiotemporal_embeddings = None
        self.genomic_embeddings = None
        self.prognostic_targets = None
        self.patient_data = None

        # Model and training
        self.gat_model = None
        self.trainer = None
        self.training_history = None

    def generate_phase2_compatible_embeddings(
        self,
        num_patients: int = 300,
        embedding_dim: int = 256,
        add_realistic_structure: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate Phase 2 compatible embeddings for demonstration.

        In actual implementation, these would come from:
        - Phase 2.1: Spatiotemporal Vision Transformer outputs
        - Phase 2.2: Genomic Transformer outputs

        Args:
            num_patients: Number of patients to simulate
            embedding_dim: Embedding dimension (Phase 2 standard: 256)
            add_realistic_structure: Whether to add realistic patient clustering

        Returns:
            Tuple of (spatiotemporal_embeddings, genomic_embeddings)
        """
        logger.info(
            f"📊 Generating Phase 2 compatible embeddings for {num_patients} patients"
        )

        # Set seed for reproducibility
        np.random.seed(42)

        if add_realistic_structure:
            # Create realistic patient subgroups
            n_controls = num_patients // 3
            n_pd_early = num_patients // 3
            n_pd_advanced = num_patients - n_controls - n_pd_early

            # Spatiotemporal embeddings (imaging-based)
            # Controls: normal patterns
            controls_spatial = np.random.multivariate_normal(
                mean=np.zeros(embedding_dim),
                cov=0.5 * np.eye(embedding_dim),
                size=n_controls,
            )

            # Early PD: mild changes
            early_pd_spatial = np.random.multivariate_normal(
                mean=0.3 * np.ones(embedding_dim),
                cov=0.7 * np.eye(embedding_dim),
                size=n_pd_early,
            )

            # Advanced PD: significant changes
            advanced_pd_spatial = np.random.multivariate_normal(
                mean=0.8 * np.ones(embedding_dim),
                cov=0.9 * np.eye(embedding_dim),
                size=n_pd_advanced,
            )

            spatiotemporal_embeddings = np.vstack(
                [controls_spatial, early_pd_spatial, advanced_pd_spatial]
            )

            # Genomic embeddings (genetic risk factors)
            # Controls: low genetic risk
            controls_genomic = np.random.multivariate_normal(
                mean=-0.2 * np.ones(embedding_dim),
                cov=0.4 * np.eye(embedding_dim),
                size=n_controls,
            )

            # Early PD: moderate genetic risk
            early_pd_genomic = np.random.multivariate_normal(
                mean=0.1 * np.ones(embedding_dim),
                cov=0.6 * np.eye(embedding_dim),
                size=n_pd_early,
            )

            # Advanced PD: high genetic risk
            advanced_pd_genomic = np.random.multivariate_normal(
                mean=0.5 * np.ones(embedding_dim),
                cov=0.8 * np.eye(embedding_dim),
                size=n_pd_advanced,
            )

            genomic_embeddings = np.vstack(
                [controls_genomic, early_pd_genomic, advanced_pd_genomic]
            )

            # Create patient labels
            labels = np.concatenate(
                [
                    np.zeros(n_controls),  # 0: Control
                    np.ones(n_pd_early),  # 1: Early PD
                    np.full(n_pd_advanced, 2),  # 2: Advanced PD
                ]
            )

            self.patient_labels = labels

        else:
            # Simple random embeddings
            spatiotemporal_embeddings = np.random.randn(num_patients, embedding_dim)
            genomic_embeddings = np.random.randn(num_patients, embedding_dim)

        # Normalize embeddings (as Phase 2 encoders would)
        spatiotemporal_embeddings = spatiotemporal_embeddings / np.linalg.norm(
            spatiotemporal_embeddings, axis=1, keepdims=True
        )
        genomic_embeddings = genomic_embeddings / np.linalg.norm(
            genomic_embeddings, axis=1, keepdims=True
        )

        logger.info(
            f"✅ Generated embeddings - Spatial: {spatiotemporal_embeddings.shape}, "
            f"Genomic: {genomic_embeddings.shape}"
        )

        self.spatiotemporal_embeddings = spatiotemporal_embeddings
        self.genomic_embeddings = genomic_embeddings

        return spatiotemporal_embeddings, genomic_embeddings

    def generate_prognostic_targets(
        self, num_patients: int, add_realistic_progression: bool = True
    ) -> np.ndarray:
        """Generate prognostic targets for training.

        Args:
            num_patients: Number of patients
            add_realistic_progression: Whether to simulate realistic progression patterns

        Returns:
            Prognostic targets [num_patients, 2] (motor, cognitive)
        """
        logger.info(f"🎯 Generating prognostic targets for {num_patients} patients")

        if add_realistic_progression and hasattr(self, "patient_labels"):
            # Realistic progression based on patient groups
            motor_scores = []
            cognitive_scores = []

            for label in self.patient_labels:
                if label == 0:  # Control
                    motor_scores.append(
                        np.random.normal(1.0, 0.2)
                    )  # Minimal progression
                    cognitive_scores.append(
                        np.random.normal(0.1, 0.1)
                    )  # No cognitive decline
                elif label == 1:  # Early PD
                    motor_scores.append(
                        np.random.normal(2.5, 0.5)
                    )  # Moderate motor progression
                    cognitive_scores.append(
                        np.random.normal(0.3, 0.2)
                    )  # Mild cognitive changes
                else:  # Advanced PD
                    motor_scores.append(
                        np.random.normal(4.2, 0.8)
                    )  # Significant motor progression
                    cognitive_scores.append(
                        np.random.normal(0.8, 0.3)
                    )  # Cognitive decline

            prognostic_targets = np.column_stack([motor_scores, cognitive_scores])
        else:
            # Random targets
            prognostic_targets = np.random.randn(num_patients, 2)

        self.prognostic_targets = prognostic_targets
        logger.info(f"✅ Generated prognostic targets: {prognostic_targets.shape}")

        return prognostic_targets

    def setup_gat_model(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
    ) -> MultiModalGraphAttention:
        """Setup Graph Attention Network model.

        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            num_heads: Number of attention heads
            num_layers: Number of GAT layers

        Returns:
            Initialized GAT model
        """
        logger.info("🏗️ Setting up Graph Attention Network model")

        self.gat_model = create_phase3_gat_model(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=256,
            num_heads=num_heads,
            num_layers=num_layers,
            use_pytorch_geometric=True,
        )

        # Initialize trainer
        self.trainer = GATTrainer(
            model=self.gat_model,
            device=self.device,
            learning_rate=1e-4,
            weight_decay=1e-5,
            patience=20,
        )

        logger.info(
            f"✅ GAT model setup complete - Parameters: {sum(p.numel() for p in self.gat_model.parameters()):,}"
        )

        return self.gat_model

    def prepare_training_data(self) -> tuple[Data, Data, Data]:
        """Prepare training, validation, and test data.

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        logger.info("📋 Preparing training data with patient similarity integration")

        # First, create synthetic patient data for the similarity graph
        self._create_synthetic_patient_data()

        # Create multimodal graph data
        graph_data = self.data_integrator.prepare_multimodal_graph_data(
            spatiotemporal_embeddings=self.spatiotemporal_embeddings,
            genomic_embeddings=self.genomic_embeddings,
            prognostic_targets=self.prognostic_targets,
        )

        # Split data (normally would be done by patient, here we simulate)
        num_patients = self.spatiotemporal_embeddings.shape[0]
        train_idx, temp_idx = train_test_split(
            range(num_patients), test_size=0.4, random_state=42
        )
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

        # Create data splits with proper edge remapping
        def create_subset(indices):
            subset_data = Data()
            subset_data.x_spatiotemporal = graph_data.x_spatiotemporal[indices]
            subset_data.x_genomic = graph_data.x_genomic[indices]

            # Create mapping from original indices to new indices
            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}

            # Filter edges to only include those between nodes in the subset
            edge_mask = []
            new_edges = []

            for i in range(graph_data.edge_index.size(1)):
                src, dst = (
                    graph_data.edge_index[0, i].item(),
                    graph_data.edge_index[1, i].item(),
                )
                if src in old_to_new and dst in old_to_new:
                    edge_mask.append(i)
                    new_edges.append([old_to_new[src], old_to_new[dst]])

            if new_edges:
                subset_data.edge_index = (
                    torch.tensor(new_edges, dtype=torch.long).t().contiguous()
                )
                if (
                    hasattr(graph_data, "edge_attr")
                    and graph_data.edge_attr is not None
                ):
                    subset_data.edge_attr = graph_data.edge_attr[edge_mask]
            else:
                # No edges in subset - create empty edge tensors
                subset_data.edge_index = torch.empty((2, 0), dtype=torch.long)
                if hasattr(graph_data, "edge_attr"):
                    subset_data.edge_attr = torch.empty((0, 1), dtype=torch.float)

            if hasattr(graph_data, "prognostic_targets"):
                subset_data.prognostic_targets = graph_data.prognostic_targets[indices]

            if hasattr(graph_data, "similarity_matrix"):
                subset_data.similarity_matrix = graph_data.similarity_matrix[
                    np.ix_(indices, indices)
                ]

            return subset_data

        train_data = create_subset(train_idx)
        val_data = create_subset(val_idx)
        test_data = create_subset(test_idx)

        logger.info(
            f"✅ Data splits prepared - Train: {len(train_idx)}, "
            f"Val: {len(val_idx)}, Test: {len(test_idx)}"
        )

        return train_data, val_data, test_data

    def _create_synthetic_patient_data(self):
        """Create synthetic patient data compatible with PatientSimilarityGraph."""
        logger.info("🧬 Creating synthetic patient data for similarity graph")

        num_patients = self.spatiotemporal_embeddings.shape[0]

        # Create synthetic biomarker data
        np.random.seed(42)

        # Generate patient IDs (integers for PyTorch compatibility)
        patient_ids = list(
            range(1000, 1000 + num_patients)
        )  # Start from 1000 to avoid conflicts

        # Generate cohort definitions based on our patient labels
        if hasattr(self, "patient_labels"):
            cohort_map = {
                0: "Healthy Control",
                1: "Parkinson's Disease",
                2: "Parkinson's Disease",
            }
            cohorts = [cohort_map[int(label)] for label in self.patient_labels]
        else:
            # Random assignment
            cohorts = np.random.choice(
                ["Healthy Control", "Parkinson's Disease"],
                size=num_patients,
                p=[0.3, 0.7],
            )

        # Generate synthetic biomarker features
        # These match the features expected by PatientSimilarityGraph
        biomarker_features = {
            "LRRK2": np.random.choice(
                [0, 1], num_patients, p=[0.85, 0.15]
            ),  # 15% positive
            "GBA": np.random.choice(
                [0, 1], num_patients, p=[0.90, 0.10]
            ),  # 10% positive
            "APOE_RISK": np.random.choice(
                [0, 1], num_patients, p=[0.75, 0.25]
            ),  # 25% high risk
            "PTAU": np.random.normal(35.0, 15.0, num_patients),  # CSF phospho-tau
            "TTAU": np.random.normal(280.0, 120.0, num_patients),  # CSF total tau
            "UPSIT_TOTAL": np.random.normal(28.5, 8.2, num_patients),  # Smell test
            "ALPHA_SYN": np.random.normal(
                1.8, 0.8, num_patients
            ),  # CSF alpha-synuclein
        }

        # Adjust biomarkers based on cohort (make them more realistic)
        for i, cohort in enumerate(cohorts):
            if cohort == "Parkinson's Disease":
                # PD patients have different biomarker profiles
                biomarker_features["PTAU"][i] *= 1.2  # Higher phospho-tau
                biomarker_features["TTAU"][i] *= 1.3  # Higher total tau
                biomarker_features["UPSIT_TOTAL"][i] *= 0.7  # Lower smell scores
                biomarker_features["ALPHA_SYN"][i] *= 0.8  # Lower alpha-synuclein

        # Create patient DataFrame
        patient_data = pd.DataFrame(
            {
                "PATNO": patient_ids,
                "COHORT_DEFINITION": cohorts,
                "EVENT_ID": ["BL"] * num_patients,  # Baseline visit
                **biomarker_features,
            }
        )

        # Set up the similarity graph with this synthetic data
        self.similarity_constructor.patient_data = patient_data
        self.similarity_constructor.biomarker_features = list(biomarker_features.keys())

        # Calculate similarity matrix
        similarity_matrix = self.similarity_constructor.calculate_patient_similarity()

        # Create similarity graph
        similarity_graph = self.similarity_constructor.create_similarity_graph()

        logger.info(
            f"✅ Created synthetic patient data: {num_patients} patients, "
            f"{len(similarity_graph.edges())} similarity edges"
        )

        self.patient_data = patient_data

    def train_gat_model(
        self, train_data: Data, val_data: Data, num_epochs: int = 100
    ) -> dict:
        """Train the Graph Attention Network.

        Args:
            train_data: Training data
            val_data: Validation data
            num_epochs: Number of training epochs

        Returns:
            Training history
        """
        logger.info("🔥 Starting Graph Attention Network training")

        # Ensure model and trainer are set up
        if self.trainer is None:
            self.setup_gat_model()

        # Train model
        self.training_history = self.trainer.train(
            train_data=train_data,
            val_data=val_data,
            num_epochs=num_epochs,
            save_path="src/giman_pipeline/models/checkpoints/gat_phase3_1.pth",
        )

        return self.training_history

    def evaluate_model(self, test_data: Data) -> dict:
        """Evaluate trained GAT model on test data.

        Args:
            test_data: Test data

        Returns:
            Evaluation metrics
        """
        logger.info("📊 Evaluating Graph Attention Network performance")

        self.gat_model.eval()

        with torch.no_grad():
            test_data = test_data.to(self.device)

            # Forward pass
            modality_embeddings = [test_data.x_spatiotemporal, test_data.x_genomic]
            outputs = self.gat_model(
                modality_embeddings, test_data.edge_index, test_data.edge_attr
            )

            # Extract predictions and targets
            prognostic_predictions = outputs["prognostic_predictions"]

            if hasattr(test_data, "prognostic_targets"):
                targets = test_data.prognostic_targets.cpu().numpy()

                # Calculate metrics for each target
                metrics = {}
                target_names = ["Motor Progression", "Cognitive Conversion"]

                for i, (pred, name) in enumerate(
                    zip(prognostic_predictions, target_names, strict=False)
                ):
                    pred_np = pred.cpu().numpy().flatten()
                    target_np = targets[:, i]

                    r2 = r2_score(target_np, pred_np)
                    mse = mean_squared_error(target_np, pred_np)

                    metrics[f"{name}_R2"] = r2
                    metrics[f"{name}_MSE"] = mse

                    logger.info(f"{name} - R²: {r2:.4f}, MSE: {mse:.4f}")

                return metrics
            else:
                logger.warning("No prognostic targets available for evaluation")
                return {}

    def visualize_results(
        self, test_data: Data, save_dir: str = "visualizations/phase3_1_visualization"
    ):
        """Create comprehensive visualization of results.

        Args:
            test_data: Test data for visualization
            save_dir: Directory to save visualizations
        """
        logger.info("📈 Creating Phase 3.1 result visualizations")

        # Create save directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Set up visualization style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # 1. Training History
        if self.training_history:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            epochs = range(len(self.training_history["train_losses"]))

            ax1.plot(
                epochs,
                self.training_history["train_losses"],
                label="Training Loss",
                linewidth=2,
            )
            ax1.plot(
                epochs,
                self.training_history["val_losses"],
                label="Validation Loss",
                linewidth=2,
            )
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("GIMAN Phase 3.1: GAT Training Progress")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Loss distribution
            ax2.hist(
                self.training_history["train_losses"],
                alpha=0.7,
                label="Training",
                bins=20,
            )
            ax2.hist(
                self.training_history["val_losses"],
                alpha=0.7,
                label="Validation",
                bins=20,
            )
            ax2.set_xlabel("Loss Value")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Loss Distribution")
            ax2.legend()

            plt.tight_layout()
            plt.savefig(
                f"{save_dir}/training_history.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        # 2. Model Architecture Visualization
        self._visualize_model_architecture(save_dir)

        # 3. Attention Weights Analysis
        self._visualize_attention_weights(test_data, save_dir)

        # 4. Embedding Space Analysis
        self._visualize_embedding_space(test_data, save_dir)

        logger.info(f"✅ Visualizations saved to {save_dir}")

    def _visualize_model_architecture(self, save_dir: str):
        """Visualize GAT model architecture."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create architecture diagram
        architecture_info = [
            f"Input Dimension: {self.gat_model.input_dim}",
            f"Hidden Dimension: {self.gat_model.hidden_dim}",
            f"Output Dimension: {self.gat_model.output_dim}",
            f"Attention Heads: {self.gat_model.num_heads}",
            f"GAT Layers: {self.gat_model.num_layers}",
            f"Modalities: {self.gat_model.num_modalities}",
            f"Total Parameters: {sum(p.numel() for p in self.gat_model.parameters()):,}",
        ]

        ax.text(
            0.5,
            0.7,
            "GIMAN Phase 3.1\nGraph Attention Network",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
        )

        for i, info in enumerate(architecture_info):
            ax.text(0.5, 0.6 - i * 0.06, info, ha="center", va="center", fontsize=12)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Model Architecture Overview", fontsize=16, pad=20)

        plt.savefig(f"{save_dir}/model_architecture.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _visualize_attention_weights(self, test_data: Data, save_dir: str):
        """Visualize attention weights from the model."""
        self.gat_model.eval()

        with torch.no_grad():
            test_data = test_data.to(self.device)
            modality_embeddings = [test_data.x_spatiotemporal, test_data.x_genomic]
            outputs = self.gat_model(modality_embeddings, test_data.edge_index)

            if "attention_weights" in outputs:
                attention_weights = outputs["attention_weights"].cpu().numpy()

                # Plot attention weight heatmap
                fig, ax = plt.subplots(figsize=(10, 8))

                # Average attention weights across heads and layers
                avg_attention = np.mean(attention_weights, axis=1)  # Average over heads

                sns.heatmap(avg_attention[:20, :20], annot=False, cmap="viridis", ax=ax)
                ax.set_title(
                    "Cross-Modal Attention Weights\n(Sample of 20x20 patients)"
                )
                ax.set_xlabel("Target Patients")
                ax.set_ylabel("Source Patients")

                plt.savefig(
                    f"{save_dir}/attention_weights.png", dpi=300, bbox_inches="tight"
                )
                plt.close()

    def _visualize_embedding_space(self, test_data: Data, save_dir: str):
        """Visualize learned embedding space."""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        self.gat_model.eval()

        with torch.no_grad():
            test_data = test_data.to(self.device)
            modality_embeddings = [test_data.x_spatiotemporal, test_data.x_genomic]
            outputs = self.gat_model(modality_embeddings, test_data.edge_index)

            fused_embeddings = outputs["fused_embeddings"].cpu().numpy()

            # PCA visualization
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(fused_embeddings)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Color by patient labels if available
            if hasattr(self, "patient_labels"):
                test_labels = self.patient_labels[: len(fused_embeddings)]
                scatter = ax1.scatter(
                    pca_embeddings[:, 0],
                    pca_embeddings[:, 1],
                    c=test_labels,
                    cmap="viridis",
                    alpha=0.7,
                )
                plt.colorbar(scatter, ax=ax1)
            else:
                ax1.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], alpha=0.7)

            ax1.set_title("PCA: Fused Embeddings")
            ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")

            # t-SNE visualization (sample for performance)
            if len(fused_embeddings) > 100:
                sample_idx = np.random.choice(len(fused_embeddings), 100, replace=False)
                sample_embeddings = fused_embeddings[sample_idx]
                sample_labels = (
                    test_labels[sample_idx] if hasattr(self, "patient_labels") else None
                )
            else:
                sample_embeddings = fused_embeddings
                sample_labels = test_labels if hasattr(self, "patient_labels") else None

            tsne = TSNE(n_components=2, random_state=42)
            tsne_embeddings = tsne.fit_transform(sample_embeddings)

            if sample_labels is not None:
                scatter = ax2.scatter(
                    tsne_embeddings[:, 0],
                    tsne_embeddings[:, 1],
                    c=sample_labels,
                    cmap="viridis",
                    alpha=0.7,
                )
                plt.colorbar(scatter, ax=ax2)
            else:
                ax2.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], alpha=0.7)

            ax2.set_title("t-SNE: Fused Embeddings")
            ax2.set_xlabel("t-SNE 1")
            ax2.set_ylabel("t-SNE 2")

            plt.tight_layout()
            plt.savefig(f"{save_dir}/embedding_space.png", dpi=300, bbox_inches="tight")
            plt.close()

    def run_complete_demonstration(
        self, num_patients: int = 300, num_epochs: int = 50
    ) -> dict:
        """Run complete Phase 3.1 integration demonstration.

        Args:
            num_patients: Number of patients to simulate
            num_epochs: Number of training epochs

        Returns:
            Complete results dictionary
        """
        logger.info("🚀 Starting complete Phase 3.1 integration demonstration")

        # Step 1: Generate Phase 2 compatible data
        self.generate_phase2_compatible_embeddings(num_patients=num_patients)
        self.generate_prognostic_targets(num_patients=num_patients)

        # Step 2: Setup GAT model
        self.setup_gat_model()

        # Step 3: Prepare training data
        train_data, val_data, test_data = self.prepare_training_data()

        # Step 4: Train model
        training_history = self.train_gat_model(
            train_data=train_data, val_data=val_data, num_epochs=num_epochs
        )

        # Step 5: Evaluate model
        evaluation_metrics = self.evaluate_model(test_data)

        # Step 6: Create visualizations
        self.visualize_results(test_data)

        # Compile results
        results = {
            "training_history": training_history,
            "evaluation_metrics": evaluation_metrics,
            "model_parameters": sum(p.numel() for p in self.gat_model.parameters()),
            "data_shapes": {
                "spatiotemporal": self.spatiotemporal_embeddings.shape,
                "genomic": self.genomic_embeddings.shape,
                "prognostic": self.prognostic_targets.shape,
            },
        }

        logger.info("✅ Phase 3.1 integration demonstration completed successfully!")

        return results


def main():
    """Main demonstration function."""
    logger.info("🎬 GIMAN Phase 3.1: Graph Attention Network Integration Demo")

    # Create demonstration instance
    demo = Phase3IntegrationDemo()

    # Run complete demonstration
    results = demo.run_complete_demonstration(num_patients=300, num_epochs=50)

    # Print summary
    print("\n" + "=" * 80)
    print("🎉 GIMAN Phase 3.1 Integration Demo Results")
    print("=" * 80)
    print(f"Model Parameters: {results['model_parameters']:,}")
    print(
        f"Training completed with best validation loss: {results['training_history']['best_val_loss']:.6f}"
    )

    if results["evaluation_metrics"]:
        print("\n📊 Evaluation Metrics:")
        for metric, value in results["evaluation_metrics"].items():
            print(f"  {metric}: {value:.4f}")

    print("\n📈 Visualizations saved to: visualizations/phase3_1_visualization/")
    print("=" * 80)


if __name__ == "__main__":
    main()
