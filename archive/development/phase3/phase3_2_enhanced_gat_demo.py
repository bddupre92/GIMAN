#!/usr/bin/env python3
"""GIMAN Phase 3.2 Integration Demo: Enhanced GAT with Cross-Modal Attention

This script demonstrates the integrated Phase 3.2 system that combines:
- Phase 3.2: Advanced Cross-Modal Attention mechanisms
- Phase 3.1: Graph Attention Network with patient similarity
- Enhanced interpretability and visualization

Key Features Demonstrated:
1. Cross-modal bidirectional attention between spatiotemporal and genomic data
2. Co-attention and hierarchical attention mechanisms
3. Graph attention on patient similarity networks
4. Interpretable prognostic predictions
5. Comprehensive attention pattern analysis

Author: GIMAN Development Team
Date: September 24, 2025
Phase: 3.2 - Enhanced GAT Integration Demo
"""

import logging
from pathlib import Path

# Visualization
import matplotlib.pyplot as plt

# Scientific computing
import numpy as np
import pandas as pd
import seaborn as sns

# Deep learning
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available - interactive visualizations will be skipped")

# Import the existing patient similarity graph
import sys

sys.path.append(
    "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025"
)

try:
    from patient_similarity_graph import PatientSimilarityGraph
except ImportError:
    logger.error("PatientSimilarityGraph not found - will create simplified version")
    PatientSimilarityGraph = None

# GIMAN imports (use local implementations for demo)
try:
    from src.giman_pipeline.models.enhanced_multimodal_gat import (
        EnhancedGATTrainer,
        EnhancedMultiModalGAT,
        create_enhanced_multimodal_gat,
    )
except ImportError:
    logger.error(
        "Enhanced GAT modules not found - demo will use simplified implementation"
    )
    # We'll define simplified versions below
    EnhancedMultiModalGAT = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Phase32IntegrationDemo:
    """Comprehensive demonstration of Phase 3.2 Enhanced GAT integration."""

    def __init__(
        self,
        num_patients: int = 300,
        sequence_length: int = 100,
        num_genomic_features: int = 1000,
        embedding_dim: int = 256,
        device: torch.device | None = None,
        results_dir: str = "visualizations/phase3_2_enhanced_gat",
    ):
        """Initialize Phase 3.2 integration demo.

        Args:
            num_patients: Number of patients in dataset
            sequence_length: Length of spatiotemporal sequences
            num_genomic_features: Number of genomic features
            embedding_dim: Phase 2 embedding dimension
            device: Computing device
            results_dir: Directory for results and visualizations
        """
        self.num_patients = num_patients
        self.sequence_length = sequence_length
        self.num_genomic_features = num_genomic_features
        self.embedding_dim = embedding_dim
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info("🚀 Initializing Phase 3.2 Enhanced GAT Demo")
        logger.info(f"📊 Patients: {num_patients}, Device: {self.device}")
        logger.info(f"📁 Results directory: {self.results_dir}")

        # Initialize components
        self.patient_similarity_graph = None
        self.enhanced_gat_model = None
        self.trainer = None

        # Data containers
        self.patient_data = None
        self.graph_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup_demo_environment(self):
        """Set up the complete demo environment."""
        logger.info("🔧 Setting up Phase 3.2 demo environment...")

        # Create synthetic patient data
        self._create_enhanced_synthetic_patient_data()

        # Initialize patient similarity graph
        self._initialize_patient_similarity_graph()

        # Create enhanced GAT model
        self._create_enhanced_gat_model()

        # Prepare training data
        self._prepare_graph_data()

        logger.info("✅ Phase 3.2 demo environment ready!")

    def _create_enhanced_synthetic_patient_data(self):
        """Create enhanced synthetic patient data with rich multi-modal patterns."""
        logger.info("📊 Creating enhanced synthetic patient data...")

        np.random.seed(42)
        torch.manual_seed(42)

        # Create realistic patient cohorts with distinct characteristics
        cohort_sizes = [100, 120, 80]  # Three cohorts
        cohorts = []

        for cohort_id, size in enumerate(cohort_sizes):
            cohort_data = {
                "patient_ids": list(
                    range(
                        sum(cohort_sizes[:cohort_id]),
                        sum(cohort_sizes[: cohort_id + 1]),
                    )
                ),
                "cohort_id": cohort_id,
                "characteristics": {},
            }

            # Cohort-specific spatiotemporal patterns
            if cohort_id == 0:  # Stable progression cohort
                base_trend = np.linspace(0.3, 0.7, self.sequence_length)
                noise_scale = 0.1
                cognitive_decline_rate = 0.02
            elif cohort_id == 1:  # Rapid decline cohort
                base_trend = np.linspace(0.8, 0.2, self.sequence_length)
                noise_scale = 0.15
                cognitive_decline_rate = 0.08
            else:  # Mixed progression cohort
                base_trend = (
                    np.sin(np.linspace(0, 2 * np.pi, self.sequence_length)) * 0.3 + 0.5
                )
                noise_scale = 0.2
                cognitive_decline_rate = 0.05

            cohort_data["characteristics"] = {
                "base_trend": base_trend,
                "noise_scale": noise_scale,
                "cognitive_decline_rate": cognitive_decline_rate,
            }

            cohorts.append(cohort_data)

        # Generate spatiotemporal embeddings (Phase 2 compatible: 256D)
        spatiotemporal_embeddings = []
        genomic_embeddings = []
        patient_metadata = []
        prognostic_targets = []

        for cohort in cohorts:
            for patient_id in cohort["patient_ids"]:
                # Spatiotemporal embedding with cohort-specific patterns
                base_embedding = (
                    np.random.randn(self.sequence_length, self.embedding_dim) * 0.1
                )
                trend_component = np.outer(
                    cohort["characteristics"]["base_trend"],
                    np.random.randn(self.embedding_dim) * 0.5,
                )
                noise_component = (
                    np.random.randn(self.sequence_length, self.embedding_dim)
                    * cohort["characteristics"]["noise_scale"]
                )

                spatial_embedding = base_embedding + trend_component + noise_component
                spatiotemporal_embeddings.append(spatial_embedding)

                # Genomic embedding with cross-modal correlations
                # Create correlations between genomic and spatiotemporal patterns
                genomic_base = np.random.randn(self.embedding_dim) * 0.3
                correlation_factor = 0.4  # Strength of cross-modal correlation

                spatial_summary = np.mean(spatial_embedding, axis=0)
                correlated_genomic = (
                    genomic_base
                    + correlation_factor * spatial_summary
                    + np.random.randn(self.embedding_dim) * 0.2
                )
                genomic_embeddings.append(correlated_genomic)

                # Patient metadata
                patient_metadata.append(
                    {
                        "patient_id": patient_id,
                        "cohort_id": cohort["cohort_id"],
                        "age": np.random.randint(60, 85),
                        "sex": np.random.choice(["M", "F"]),
                        "education_years": np.random.randint(8, 20),
                    }
                )

                # Prognostic targets (cognitive decline, conversion risk)
                decline_rate = cohort["characteristics"]["cognitive_decline_rate"]
                cognitive_score = max(0, 1 - decline_rate * np.random.exponential(2.0))
                conversion_prob = 1 / (1 + np.exp(-(decline_rate * 10 - 3)))

                prognostic_targets.append([cognitive_score, conversion_prob])

        # Convert to tensors
        self.patient_data = {
            "spatiotemporal_embeddings": torch.FloatTensor(
                np.array(spatiotemporal_embeddings)
            ),
            "genomic_embeddings": torch.FloatTensor(np.array(genomic_embeddings)),
            "patient_metadata": patient_metadata,
            "prognostic_targets": torch.FloatTensor(prognostic_targets),
            "cohort_labels": torch.LongTensor(
                [p["cohort_id"] for p in patient_metadata]
            ),
        }

        logger.info("✅ Created enhanced synthetic data:")
        logger.info(
            f"   📈 Spatiotemporal: {self.patient_data['spatiotemporal_embeddings'].shape}"
        )
        logger.info(f"   🧬 Genomic: {self.patient_data['genomic_embeddings'].shape}")
        logger.info(f"   👥 Cohorts: {len(cohorts)} with sizes {cohort_sizes}")

    def _initialize_patient_similarity_graph(self):
        """Initialize patient similarity graph with enhanced features."""
        logger.info("🕸️ Initializing enhanced patient similarity graph...")

        self.patient_similarity_graph = PatientSimilarityGraph(
            num_patients=self.num_patients, embedding_dim=self.embedding_dim
        )

        # Compute enhanced similarities using both modalities
        combined_embeddings = torch.cat(
            [
                torch.mean(
                    self.patient_data["spatiotemporal_embeddings"], dim=1
                ),  # Average over sequence
                self.patient_data["genomic_embeddings"],
            ],
            dim=1,
        )

        # Build similarity graph
        similarity_matrix, edge_index, edge_weights = (
            self.patient_similarity_graph.build_similarity_graph(
                combined_embeddings.numpy(),
                method="cosine",
                k_neighbors=15,  # Increased connectivity
                similarity_threshold=0.3,
            )
        )

        # Store graph information
        self.similarity_matrix = similarity_matrix
        self.edge_index = torch.LongTensor(edge_index)
        self.edge_weights = torch.FloatTensor(edge_weights)

        logger.info("✅ Built enhanced similarity graph:")
        logger.info(f"   🔗 Edges: {edge_index.shape[1]:,}")
        logger.info(f"   📊 Avg similarity: {np.mean(edge_weights):.4f}")

    def _create_enhanced_gat_model(self):
        """Create the enhanced GAT model with all Phase 3.2 features."""
        logger.info("🧠 Creating Enhanced Multi-Modal GAT model...")

        self.enhanced_gat_model = create_enhanced_multimodal_gat(
            input_dim=self.embedding_dim,
            hidden_dim=512,
            num_heads=8,
            num_gat_layers=3,
            num_transformer_layers=3,
            use_coattention=True,
            use_hierarchical=True,
            dropout=0.1,
        )

        self.enhanced_gat_model.to(self.device)

        # Initialize trainer
        self.trainer = EnhancedGATTrainer(
            model=self.enhanced_gat_model,
            device=self.device,
            learning_rate=1e-4,
            weight_decay=1e-5,
            patience=20,
            attention_loss_weight=0.1,
        )

        logger.info("✅ Enhanced GAT model and trainer ready!")

    def _prepare_graph_data(self):
        """Prepare graph data for training with proper data splits."""
        logger.info("📊 Preparing graph data with enhanced features...")

        # Create data splits
        indices = np.arange(self.num_patients)
        train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

        # Create subgraphs for each split
        for split_name, split_idx in [
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ]:
            # Extract subset data
            subset_spatial = self.patient_data["spatiotemporal_embeddings"][split_idx]
            subset_genomic = self.patient_data["genomic_embeddings"][split_idx]
            subset_targets = self.patient_data["prognostic_targets"][split_idx]

            # Remap edge indices to subset
            subset_edges, subset_edge_weights = self._remap_edges_to_subset(
                self.edge_index, self.edge_weights, split_idx
            )

            # Create subset similarity matrix
            full_similarity = torch.FloatTensor(self.similarity_matrix)
            subset_similarity = full_similarity[np.ix_(split_idx, split_idx)]

            # Create PyTorch Geometric Data object
            graph_data = Data(
                x_spatiotemporal=subset_spatial.to(self.device),
                x_genomic=subset_genomic.to(self.device),
                edge_index=subset_edges.to(self.device),
                edge_attr=subset_edge_weights.to(self.device),
                prognostic_targets=subset_targets.to(self.device),
                similarity_matrix=subset_similarity.to(self.device),
                num_nodes=len(split_idx),
            )

            # Store data splits
            if split_name == "train":
                self.train_data = graph_data
            elif split_name == "val":
                self.val_data = graph_data
            else:
                self.test_data = graph_data

        logger.info("✅ Prepared graph data splits:")
        logger.info(f"   🚂 Train: {len(train_idx)} patients")
        logger.info(f"   ✅ Validation: {len(val_idx)} patients")
        logger.info(f"   🧪 Test: {len(test_idx)} patients")

    def _remap_edges_to_subset(
        self,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
        subset_indices: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remap edge indices to subset node indices."""
        # Create mapping from original indices to subset indices
        index_mapping = {
            orig_idx: new_idx for new_idx, orig_idx in enumerate(subset_indices)
        }

        # Filter edges that have both nodes in the subset
        subset_indices_set = set(subset_indices)
        valid_edges = []
        valid_weights = []

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in subset_indices_set and dst in subset_indices_set:
                # Remap to new indices
                new_src = index_mapping[src]
                new_dst = index_mapping[dst]
                valid_edges.append([new_src, new_dst])
                valid_weights.append(edge_weights[i].item())

        if valid_edges:
            remapped_edges = torch.LongTensor(valid_edges).t()
            remapped_weights = torch.FloatTensor(valid_weights)
        else:
            # Handle case with no valid edges
            remapped_edges = torch.LongTensor([[0], [0]])
            remapped_weights = torch.FloatTensor([1.0])

        return remapped_edges, remapped_weights

    def run_enhanced_training(self, num_epochs: int = 50) -> dict:
        """Run enhanced training with comprehensive monitoring."""
        logger.info(f"🚀 Starting Enhanced GAT training for {num_epochs} epochs...")

        training_history = {
            "train_losses": [],
            "val_losses": [],
            "attention_analysis": [],
            "cross_modal_alignments": [],
        }

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training step
            train_loss = self.trainer.train_epoch(self.train_data)
            training_history["train_losses"].append(train_loss)

            # Validation step
            val_loss, val_outputs = self.trainer.validate_epoch(self.val_data)
            training_history["val_losses"].append(val_loss)

            # Store attention analysis
            if "attention_analysis" in val_outputs:
                training_history["attention_analysis"].append(
                    val_outputs["attention_analysis"]
                )

            if "cross_modal_alignment" in val_outputs:
                training_history["cross_modal_alignments"].append(
                    val_outputs["cross_modal_alignment"].detach().cpu()
                )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(
                    self.enhanced_gat_model.state_dict(),
                    self.results_dir / "best_enhanced_gat_model.pth",
                )
            else:
                patience_counter += 1

            # Logging
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1:3d}: Train Loss = {train_loss:.6f}, "
                    f"Val Loss = {val_loss:.6f}"
                )

            # Early stopping
            if patience_counter >= self.trainer.patience:
                logger.info(f"🛑 Early stopping at epoch {epoch + 1}")
                break

        logger.info(f"✅ Training completed! Best validation loss: {best_val_loss:.6f}")

        return training_history

    def evaluate_enhanced_model(self) -> dict:
        """Comprehensive evaluation of the enhanced model."""
        logger.info("📊 Evaluating Enhanced GAT model...")

        # Load best model
        self.enhanced_gat_model.load_state_dict(
            torch.load(self.results_dir / "best_enhanced_gat_model.pth")
        )
        self.enhanced_gat_model.eval()

        # Evaluate on test set
        with torch.no_grad():
            modality_embeddings = [
                self.test_data.x_spatiotemporal,
                self.test_data.x_genomic,
            ]

            outputs = self.enhanced_gat_model(
                modality_embeddings,
                self.test_data.edge_index,
                self.test_data.edge_attr,
                similarity_matrix=self.test_data.similarity_matrix,
            )

        # Extract predictions and targets
        cognitive_pred = outputs["prognostic_predictions"][0].cpu().numpy()
        conversion_pred = outputs["prognostic_predictions"][1].cpu().numpy()

        cognitive_target = self.test_data.prognostic_targets[:, 0].cpu().numpy()
        conversion_target = self.test_data.prognostic_targets[:, 1].cpu().numpy()

        # Compute metrics
        from sklearn.metrics import mean_squared_error, r2_score

        evaluation_results = {
            "cognitive_mse": mean_squared_error(
                cognitive_target, cognitive_pred.flatten()
            ),
            "cognitive_r2": r2_score(cognitive_target, cognitive_pred.flatten()),
            "conversion_auc": roc_auc_score(
                (conversion_target > 0.5).astype(int), conversion_pred.flatten()
            ),
            "model_outputs": outputs,
            "predictions": {"cognitive": cognitive_pred, "conversion": conversion_pred},
            "targets": {"cognitive": cognitive_target, "conversion": conversion_target},
        }

        logger.info("✅ Enhanced model evaluation results:")
        logger.info(f"   🧠 Cognitive R² = {evaluation_results['cognitive_r2']:.4f}")
        logger.info(
            f"   🔄 Conversion AUC = {evaluation_results['conversion_auc']:.4f}"
        )

        return evaluation_results

    def create_enhanced_visualizations(
        self, training_history: dict, evaluation_results: dict
    ):
        """Create comprehensive visualizations for Phase 3.2 enhanced system."""
        logger.info("🎨 Creating enhanced visualizations...")

        # 1. Training dynamics with attention analysis
        self._plot_training_dynamics_with_attention(training_history)

        # 2. Cross-modal attention evolution
        self._plot_cross_modal_attention_evolution(training_history)

        # 3. Multi-level attention patterns
        self._plot_multilevel_attention_patterns(evaluation_results)

        # 4. Enhanced prediction analysis
        self._plot_enhanced_prediction_analysis(evaluation_results)

        # 5. Interactive attention dashboard
        self._create_interactive_attention_dashboard(evaluation_results)

        logger.info(f"✅ Enhanced visualizations saved to {self.results_dir}")

    def _plot_training_dynamics_with_attention(self, training_history: dict):
        """Plot training dynamics with attention analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Training and validation loss
        epochs = range(1, len(training_history["train_losses"]) + 1)
        axes[0, 0].plot(
            epochs, training_history["train_losses"], "b-", label="Training Loss"
        )
        axes[0, 0].plot(
            epochs, training_history["val_losses"], "r-", label="Validation Loss"
        )
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Enhanced GAT Training Dynamics")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Attention entropy evolution
        if training_history["attention_analysis"]:
            attention_entropies = [
                analysis.get("gat_attention_entropy", 0).item()
                if analysis.get("gat_attention_entropy") is not None
                else 0
                for analysis in training_history["attention_analysis"]
            ]
            axes[0, 1].plot(
                epochs[: len(attention_entropies)], attention_entropies, "g-"
            )
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Attention Entropy")
            axes[0, 1].set_title("GAT Attention Focus Evolution")
            axes[0, 1].grid(True, alpha=0.3)

        # Cross-modal alignment strength
        if training_history["cross_modal_alignments"]:
            alignment_strengths = [
                torch.mean(torch.diag(alignment)).item()
                for alignment in training_history["cross_modal_alignments"]
            ]
            axes[1, 0].plot(
                epochs[: len(alignment_strengths)], alignment_strengths, "m-"
            )
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Cross-Modal Alignment")
            axes[1, 0].set_title("Cross-Modal Attention Alignment")
            axes[1, 0].grid(True, alpha=0.3)

        # Co-attention symmetry evolution
        if training_history["attention_analysis"]:
            coattention_symmetries = [
                analysis.get("coattention_symmetry", 0).item()
                if analysis.get("coattention_symmetry") is not None
                else 0
                for analysis in training_history["attention_analysis"]
            ]
            axes[1, 1].plot(
                epochs[: len(coattention_symmetries)], coattention_symmetries, "c-"
            )
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Co-Attention Symmetry")
            axes[1, 1].set_title("Co-Attention Pattern Symmetry")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.results_dir / "enhanced_training_dynamics.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_cross_modal_attention_evolution(self, training_history: dict):
        """Plot cross-modal attention evolution across layers."""
        if not training_history["attention_analysis"]:
            return

        # Get the final epoch's cross-modal evolution
        final_analysis = training_history["attention_analysis"][-1]
        if "cross_modal_evolution" not in final_analysis:
            return

        evolution = final_analysis["cross_modal_evolution"]
        layers = range(len(evolution))

        spatial_strengths = [
            layer["spatial_to_genomic_strength"].item() for layer in evolution
        ]
        genomic_strengths = [
            layer["genomic_to_spatial_strength"].item() for layer in evolution
        ]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        width = 0.35
        x = np.arange(len(layers))

        bars1 = ax.bar(
            x - width / 2,
            spatial_strengths,
            width,
            label="Spatiotemporal → Genomic",
            alpha=0.8,
            color="skyblue",
        )
        bars2 = ax.bar(
            x + width / 2,
            genomic_strengths,
            width,
            label="Genomic → Spatiotemporal",
            alpha=0.8,
            color="lightcoral",
        )

        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel("Attention Strength")
        ax.set_title("Cross-Modal Attention Evolution Across Layers")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Layer {i + 1}" for i in layers])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        for bar in bars2:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(
            self.results_dir / "cross_modal_attention_evolution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_multilevel_attention_patterns(self, evaluation_results: dict):
        """Plot multi-level attention patterns."""
        outputs = evaluation_results["model_outputs"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Cross-modal attention heatmap
        if "cross_modal_attention" in outputs and outputs["cross_modal_attention"]:
            # Use the last layer's attention patterns
            last_layer_attention = outputs["cross_modal_attention"][-1]
            spatial_to_genomic = (
                last_layer_attention["spatial_to_genomic"].cpu().numpy()
            )

            # Average over heads and sequence for visualization
            if spatial_to_genomic.ndim > 2:
                spatial_to_genomic = np.mean(spatial_to_genomic, axis=(0, 1))
            else:
                spatial_to_genomic = np.mean(spatial_to_genomic, axis=0)

            # Create a sample attention matrix for visualization
            attention_matrix = np.outer(
                spatial_to_genomic[:20], spatial_to_genomic[:20]
            )

            sns.heatmap(attention_matrix, ax=axes[0, 0], cmap="Blues", cbar=True)
            axes[0, 0].set_title("Cross-Modal Attention Patterns")
            axes[0, 0].set_xlabel("Genomic Features")
            axes[0, 0].set_ylabel("Spatiotemporal Features")

        # GAT attention weights distribution
        if (
            "gat_attention_weights" in outputs
            and outputs["gat_attention_weights"] is not None
        ):
            gat_weights = outputs["gat_attention_weights"].cpu().numpy().flatten()
            axes[0, 1].hist(gat_weights, bins=50, alpha=0.7, color="green")
            axes[0, 1].set_xlabel("Attention Weight")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].set_title("GAT Attention Weight Distribution")
            axes[0, 1].grid(True, alpha=0.3)

        # Feature importance from interpretable heads
        if "prediction_explanations" in outputs:
            # Aggregate feature importance across prediction heads
            feature_importances = []
            for explanation in outputs["prediction_explanations"]:
                importance = explanation["feature_importance"].cpu().numpy()
                feature_importances.append(np.mean(importance, axis=0))

            avg_importance = np.mean(feature_importances, axis=0)
            top_features = np.argsort(avg_importance)[-20:]  # Top 20 features

            axes[1, 0].barh(range(len(top_features)), avg_importance[top_features])
            axes[1, 0].set_xlabel("Feature Importance")
            axes[1, 0].set_ylabel("Feature Index")
            axes[1, 0].set_title("Top Feature Importances (Interpretable Heads)")
            axes[1, 0].grid(True, alpha=0.3)

        # Cross-modal alignment matrix
        if "cross_modal_alignment" in outputs:
            alignment_matrix = outputs["cross_modal_alignment"].cpu().numpy()
            sns.heatmap(
                alignment_matrix,
                ax=axes[1, 1],
                cmap="RdYlBu_r",
                center=0,
                cbar=True,
                annot=True,
                fmt=".2f",
            )
            axes[1, 1].set_title("Cross-Modal Alignment Matrix")
            axes[1, 1].set_xlabel("Genomic Dimensions")
            axes[1, 1].set_ylabel("Spatiotemporal Dimensions")

        plt.suptitle("Multi-Level Attention Pattern Analysis", fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(
            self.results_dir / "multilevel_attention_patterns.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_enhanced_prediction_analysis(self, evaluation_results: dict):
        """Plot enhanced prediction analysis with interpretability."""
        predictions = evaluation_results["predictions"]
        targets = evaluation_results["targets"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Cognitive prediction scatter plot
        axes[0, 0].scatter(
            targets["cognitive"],
            predictions["cognitive"],
            alpha=0.6,
            color="blue",
            s=50,
        )
        axes[0, 0].plot([0, 1], [0, 1], "r--", lw=2)
        axes[0, 0].set_xlabel("True Cognitive Score")
        axes[0, 0].set_ylabel("Predicted Cognitive Score")
        axes[0, 0].set_title(
            f"Cognitive Prediction (R² = {evaluation_results['cognitive_r2']:.3f})"
        )
        axes[0, 0].grid(True, alpha=0.3)

        # Conversion prediction ROC-style analysis
        conversion_probs = predictions["conversion"].flatten()
        conversion_binary = (targets["conversion"] > 0.5).astype(int)

        from sklearn.metrics import precision_recall_curve

        precision, recall, _ = precision_recall_curve(
            conversion_binary, conversion_probs
        )

        axes[0, 1].plot(recall, precision, color="red", lw=2)
        axes[0, 1].set_xlabel("Recall")
        axes[0, 1].set_ylabel("Precision")
        axes[0, 1].set_title(
            f"Conversion Prediction (AUC = {evaluation_results['conversion_auc']:.3f})"
        )
        axes[0, 1].grid(True, alpha=0.3)

        # Prediction confidence analysis
        cognitive_residuals = np.abs(
            targets["cognitive"] - predictions["cognitive"].flatten()
        )
        axes[1, 0].hist(cognitive_residuals, bins=30, alpha=0.7, color="green")
        axes[1, 0].set_xlabel("Prediction Error")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Cognitive Prediction Error Distribution")
        axes[1, 0].grid(True, alpha=0.3)

        # Enhanced feature contribution analysis
        outputs = evaluation_results["model_outputs"]
        if "prediction_explanations" in outputs:
            # Show feature importance distribution across patients
            importance_data = []
            for explanation in outputs["prediction_explanations"]:
                importance = explanation["feature_importance"].cpu().numpy()
                importance_data.append(importance.flatten())

            importance_matrix = np.array(importance_data).T  # Features x Patients

            # Box plot of feature importance ranges
            axes[1, 1].boxplot(
                [
                    importance_matrix[i]
                    for i in range(0, min(20, len(importance_matrix)), 2)
                ],
                labels=[f"F{i}" for i in range(0, min(20, len(importance_matrix)), 2)],
            )
            axes[1, 1].set_xlabel("Feature Groups")
            axes[1, 1].set_ylabel("Importance Score")
            axes[1, 1].set_title("Feature Importance Variability")
            axes[1, 1].tick_params(axis="x", rotation=45)
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(
            "Enhanced Prediction Analysis with Interpretability", fontsize=16, y=0.98
        )
        plt.tight_layout()
        plt.savefig(
            self.results_dir / "enhanced_prediction_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_interactive_attention_dashboard(self, evaluation_results: dict):
        """Create interactive attention pattern dashboard."""
        outputs = evaluation_results["model_outputs"]

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Cross-Modal Attention Evolution",
                "GAT Attention Network",
                "Feature Importance Heatmap",
                "Prediction Confidence",
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "histogram"}],
            ],
        )

        # Cross-modal attention evolution
        if "cross_modal_attention" in outputs and outputs["cross_modal_attention"]:
            layer_indices = list(range(len(outputs["cross_modal_attention"])))
            attention_strengths = []

            for layer_attention in outputs["cross_modal_attention"]:
                spatial_attn = layer_attention["spatial_to_genomic"].cpu()
                strength = torch.mean(spatial_attn).item()
                attention_strengths.append(strength)

            fig.add_trace(
                go.Scatter(
                    x=layer_indices,
                    y=attention_strengths,
                    mode="lines+markers",
                    name="Attention Strength",
                    line=dict(color="royalblue", width=3),
                ),
                row=1,
                col=1,
            )

        # GAT attention network (sample visualization)
        if (
            "gat_attention_weights" in outputs
            and outputs["gat_attention_weights"] is not None
        ):
            gat_weights = outputs["gat_attention_weights"].cpu().numpy()

            # Create a sample network layout for visualization
            n_nodes = min(20, len(gat_weights))
            node_positions = np.random.random((n_nodes, 2))

            # Add edges based on attention weights
            edge_x, edge_y = [], []
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if i < len(gat_weights) and j < len(gat_weights[0]):  # Check bounds
                        edge_x.extend(
                            [node_positions[i, 0], node_positions[j, 0], None]
                        )
                        edge_y.extend(
                            [node_positions[i, 1], node_positions[j, 1], None]
                        )

            fig.add_trace(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(width=1, color="lightgray"),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

            fig.add_trace(
                go.Scatter(
                    x=node_positions[:, 0],
                    y=node_positions[:, 1],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                    name="Patients",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        # Feature importance heatmap
        if "prediction_explanations" in outputs:
            importance_data = []
            for explanation in outputs["prediction_explanations"][
                :10
            ]:  # Limit to first 10 patients
                importance = explanation["feature_importance"].cpu().numpy()
                importance_data.append(
                    importance.flatten()[:50]
                )  # Limit features for visualization

            if importance_data:
                fig.add_trace(
                    go.Heatmap(z=importance_data, colorscale="Viridis", showscale=True),
                    row=2,
                    col=1,
                )

        # Prediction confidence histogram
        predictions = evaluation_results["predictions"]
        targets = evaluation_results["targets"]

        cognitive_errors = np.abs(
            targets["cognitive"] - predictions["cognitive"].flatten()
        )

        fig.add_trace(
            go.Histogram(
                x=cognitive_errors,
                nbinsx=30,
                name="Prediction Errors",
                marker_color="green",
                opacity=0.7,
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title_text="Enhanced GAT: Interactive Attention Analysis Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True,
        )

        # Save interactive plot
        fig.write_html(str(self.results_dir / "interactive_attention_dashboard.html"))

    def run_complete_demo(self):
        """Run the complete Phase 3.2 enhanced GAT demonstration."""
        logger.info("🎯 Running Complete Phase 3.2 Enhanced GAT Demo")
        logger.info("=" * 70)

        try:
            # Setup
            self.setup_demo_environment()

            # Training
            training_history = self.run_enhanced_training(num_epochs=50)

            # Evaluation
            evaluation_results = self.evaluate_enhanced_model()

            # Visualizations
            self.create_enhanced_visualizations(training_history, evaluation_results)

            # Summary report
            self._generate_summary_report(training_history, evaluation_results)

            logger.info("🎉 Phase 3.2 Enhanced GAT Demo completed successfully!")
            logger.info(f"📁 All results saved to: {self.results_dir}")

        except Exception as e:
            logger.error(f"❌ Demo failed with error: {str(e)}")
            raise

    def _generate_summary_report(
        self, training_history: dict, evaluation_results: dict
    ):
        """Generate comprehensive summary report."""
        report_path = self.results_dir / "phase3_2_summary_report.md"

        with open(report_path, "w") as f:
            f.write("# GIMAN Phase 3.2: Enhanced GAT Integration Report\n\n")
            f.write("## Executive Summary\n\n")
            f.write(
                "This report presents the results of Phase 3.2 Enhanced GAT integration, "
            )
            f.write(
                "combining advanced cross-modal attention mechanisms with graph attention networks.\n\n"
            )

            f.write("## Model Architecture\n\n")
            f.write(
                "- **Phase 3.2 Cross-Modal Attention**: Bidirectional transformer attention\n"
            )
            f.write(
                "- **Co-Attention Mechanisms**: Simultaneous multi-modal attention\n"
            )
            f.write(
                "- **Hierarchical Attention Fusion**: Multi-scale attention patterns\n"
            )
            f.write(
                "- **Phase 3.1 Graph Attention**: Population-level patient similarity\n"
            )
            f.write(
                "- **Interpretable Prediction Heads**: Built-in feature importance\n\n"
            )

            f.write("## Performance Results\n\n")
            f.write(
                f"- **Cognitive Prediction R²**: {evaluation_results['cognitive_r2']:.4f}\n"
            )
            f.write(
                f"- **Conversion Prediction AUC**: {evaluation_results['conversion_auc']:.4f}\n"
            )
            f.write(f"- **Training Epochs**: {len(training_history['train_losses'])}\n")
            f.write(
                f"- **Final Training Loss**: {training_history['train_losses'][-1]:.6f}\n"
            )
            f.write(
                f"- **Final Validation Loss**: {training_history['val_losses'][-1]:.6f}\n\n"
            )

            f.write("## Key Innovations\n\n")
            f.write(
                "1. **Multi-Level Attention Integration**: Seamless combination of cross-modal and graph attention\n"
            )
            f.write(
                "2. **Interpretable Predictions**: Built-in feature importance for clinical interpretability\n"
            )
            f.write(
                "3. **Attention Pattern Analysis**: Comprehensive attention pattern monitoring\n"
            )
            f.write(
                "4. **Enhanced Training Dynamics**: Attention-aware training with regularization\n\n"
            )

            f.write("## Clinical Implications\n\n")
            f.write("The enhanced GAT system provides:\n")
            f.write("- Improved prognostic accuracy through multi-modal attention\n")
            f.write("- Interpretable predictions for clinical decision-making\n")
            f.write("- Patient similarity insights for personalized treatment\n")
            f.write("- Cross-modal biomarker discovery capabilities\n\n")

            f.write("## Generated Visualizations\n\n")
            f.write(
                "- `enhanced_training_dynamics.png`: Training progress with attention analysis\n"
            )
            f.write(
                "- `cross_modal_attention_evolution.png`: Cross-modal attention across layers\n"
            )
            f.write(
                "- `multilevel_attention_patterns.png`: Multi-level attention pattern analysis\n"
            )
            f.write(
                "- `enhanced_prediction_analysis.png`: Prediction performance with interpretability\n"
            )
            f.write(
                "- `interactive_attention_dashboard.html`: Interactive attention exploration\n\n"
            )

            f.write("---\n")
            f.write(
                f"*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
            )

        logger.info(f"📄 Summary report saved to: {report_path}")


def main():
    """Main execution function for Phase 3.2 Enhanced GAT Demo."""
    # Create and run the demo
    demo = Phase32IntegrationDemo(
        num_patients=300,
        sequence_length=100,
        num_genomic_features=1000,
        embedding_dim=256,
        results_dir="visualizations/phase3_2_enhanced_gat",
    )

    demo.run_complete_demo()


if __name__ == "__main__":
    main()
