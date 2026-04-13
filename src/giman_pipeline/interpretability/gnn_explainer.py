"""GIMAN GNN Explainability and Interpretability Module

Provides comprehensive interpretation tools for understanding how the GIMAN model
makes predictions, including attention weights, node importance, edge contributions,
and patient similarity analysis.

Author: GIMAN Team
Date: 2024-09-23
"""

import warnings
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")


class GIMANExplainer:
    """Comprehensive explainability toolkit for GIMAN Graph Neural Network.

    Provides multiple interpretation methods:
    - Node importance analysis
    - Edge contribution analysis
    - Attention weight visualization
    - Patient similarity exploration
    - Feature importance ranking
    - Decision boundary analysis
    """

    def __init__(self, model, graph_data, feature_names: list[str]):
        """Initialize the GIMAN explainer.

        Args:
            model: Trained GIMAN model
            graph_data: PyTorch Geometric Data object
            feature_names: List of feature column names
        """
        self.model = model
        self.graph_data = graph_data
        self.feature_names = feature_names
        self.model.eval()

        # Extract graph components
        self.x = graph_data.x
        self.edge_index = graph_data.edge_index
        self.y = graph_data.y if hasattr(graph_data, "y") else None

        print("🔍 GIMAN Explainer initialized:")
        print(f"   - Nodes: {self.x.shape[0]}")
        print(f"   - Features: {self.x.shape[1]} ({len(feature_names)} named)")
        print(f"   - Edges: {self.edge_index.shape[1]}")
        print(
            f"   - Classes: {len(torch.unique(self.y)) if self.y is not None else 'Unknown'}"
        )

    def get_node_importance(self, method: str = "gradient") -> dict[str, np.ndarray]:
        """Calculate importance scores for each node.

        Args:
            method: 'gradient', 'integrated_gradients', or 'attention'

        Returns:
            Dictionary with importance scores and metadata
        """
        self.model.eval()

        if method == "gradient":
            return self._gradient_based_importance()
        elif method == "integrated_gradients":
            return self._integrated_gradients_importance()
        elif method == "attention":
            return self._attention_based_importance()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _gradient_based_importance(self) -> dict[str, np.ndarray]:
        """Calculate node importance using gradient magnitudes."""
        self.x.requires_grad_(True)

        # Forward pass
        logits = self.model(self.x, self.edge_index)

        # Calculate gradients for each class
        importance_scores = {}

        if logits.shape[1] == 1:  # Binary classification
            # For binary, take gradient w.r.t. the single output
            loss = logits.sum()
            loss.backward(retain_graph=True)

            grad_magnitudes = torch.abs(self.x.grad).sum(dim=1).detach().cpu().numpy()
            importance_scores["binary"] = grad_magnitudes

        else:  # Multi-class
            for class_idx in range(logits.shape[1]):
                # Zero previous gradients
                if self.x.grad is not None:
                    self.x.grad.zero_()

                # Calculate gradient for this class
                class_logits = logits[:, class_idx].sum()
                class_logits.backward(retain_graph=True)

                grad_magnitudes = (
                    torch.abs(self.x.grad).sum(dim=1).detach().cpu().numpy()
                )
                importance_scores[f"class_{class_idx}"] = grad_magnitudes

        return {
            "importance_scores": importance_scores,
            "method": "gradient",
            "feature_names": self.feature_names,
        }

    def _integrated_gradients_importance(
        self, steps: int = 50
    ) -> dict[str, np.ndarray]:
        """Calculate node importance using integrated gradients."""
        # Create baseline (zero features)
        baseline = torch.zeros_like(self.x)

        importance_scores = {}

        # Calculate integrated gradients
        for step in range(steps):
            alpha = step / (steps - 1)
            interpolated_x = baseline + alpha * (self.x - baseline)
            interpolated_x.requires_grad_(True)

            logits = self.model(interpolated_x, self.edge_index)

            if step == 0:
                if logits.shape[1] == 1:
                    importance_scores["binary"] = torch.zeros(self.x.shape[0])
                else:
                    for class_idx in range(logits.shape[1]):
                        importance_scores[f"class_{class_idx}"] = torch.zeros(
                            self.x.shape[0]
                        )

            if logits.shape[1] == 1:  # Binary
                loss = logits.sum()
                loss.backward(retain_graph=True)
                grads = torch.abs(interpolated_x.grad).sum(dim=1)
                importance_scores["binary"] += grads.detach()

            else:  # Multi-class
                for class_idx in range(logits.shape[1]):
                    if interpolated_x.grad is not None:
                        interpolated_x.grad.zero_()

                    class_logits = logits[:, class_idx].sum()
                    class_logits.backward(retain_graph=True)
                    grads = torch.abs(interpolated_x.grad).sum(dim=1)
                    importance_scores[f"class_{class_idx}"] += grads.detach()

        # Average over steps and convert to numpy
        for key in importance_scores:
            importance_scores[key] = (importance_scores[key] / steps).cpu().numpy()

        return {
            "importance_scores": importance_scores,
            "method": "integrated_gradients",
            "feature_names": self.feature_names,
        }

    def _attention_based_importance(self) -> dict[str, np.ndarray]:
        """Extract attention weights from GNN layers if available."""
        importance_scores = {}

        # Check if model has attention mechanisms
        attention_weights = []

        def hook_fn(module, input, output):
            if hasattr(output, "attention_weights"):
                attention_weights.append(output.attention_weights.detach())

        # Register hooks on attention layers
        hooks = []
        for name, module in self.model.named_modules():
            if "attention" in name.lower() or hasattr(module, "attention"):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            logits = self.model(self.x, self.edge_index)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        if attention_weights:
            # Aggregate attention weights
            avg_attention = torch.stack(attention_weights).mean(dim=0)
            importance_scores["attention"] = avg_attention.cpu().numpy()
        else:
            # Fallback: use node degrees as proxy for importance
            edge_index_np = self.edge_index.cpu().numpy()
            degrees = np.bincount(edge_index_np[0], minlength=self.x.shape[0])
            importance_scores["degree_based"] = degrees

        return {
            "importance_scores": importance_scores,
            "method": "attention",
            "feature_names": self.feature_names,
        }

    def get_feature_importance(self) -> dict[str, np.ndarray]:
        """Calculate importance of each feature across all nodes.

        Returns:
            Feature importance scores and rankings
        """
        self.x.requires_grad_(True)

        # Forward pass
        logits = self.model(self.x, self.edge_index)

        # Calculate gradients w.r.t. input features
        if logits.shape[1] == 1:  # Binary
            loss = logits.sum()
            loss.backward()

            # Feature importance = mean absolute gradient across all nodes
            feature_importance = (
                torch.abs(self.x.grad).mean(dim=0).detach().cpu().numpy()
            )

        else:  # Multi-class - use gradient magnitude for predicted class
            preds = torch.argmax(logits, dim=1)
            loss = F.cross_entropy(logits, preds)
            loss.backward()

            feature_importance = (
                torch.abs(self.x.grad).mean(dim=0).detach().cpu().numpy()
            )

        # Create importance ranking
        importance_ranking = np.argsort(feature_importance)[::-1]

        return {
            "feature_importance": feature_importance,
            "importance_ranking": importance_ranking,
            "feature_names": self.feature_names,
            "ranked_features": [self.feature_names[i] for i in importance_ranking],
        }

    def get_edge_contributions(self) -> dict[str, Any]:
        """Analyze contribution of each edge to the model's predictions.

        Returns:
            Edge contribution analysis
        """
        self.model.eval()

        # Get original predictions
        with torch.no_grad():
            original_logits = self.model(self.x, self.edge_index)

        edge_contributions = []

        # Test removing each edge
        num_edges = self.edge_index.shape[1]
        for edge_idx in range(
            min(num_edges, 100)
        ):  # Limit for computational efficiency
            # Create new edge index without current edge
            mask = torch.ones(num_edges, dtype=torch.bool)
            mask[edge_idx] = False
            modified_edge_index = self.edge_index[:, mask]

            # Get predictions without this edge
            with torch.no_grad():
                modified_logits = self.model(self.x, modified_edge_index)

            # Calculate change in predictions
            logit_change = torch.abs(original_logits - modified_logits).sum().item()

            edge_contributions.append(
                {
                    "edge_idx": edge_idx,
                    "source": self.edge_index[0, edge_idx].item(),
                    "target": self.edge_index[1, edge_idx].item(),
                    "contribution": logit_change,
                }
            )

        # Sort by contribution
        edge_contributions.sort(key=lambda x: x["contribution"], reverse=True)

        return {
            "edge_contributions": edge_contributions[:20],  # Top 20
            "total_edges_analyzed": min(num_edges, 100),
        }

    def visualize_node_importance(
        self, importance_results: dict, save_path: str | None = None
    ):
        """Create visualizations for node importance analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("GIMAN Node Importance Analysis", fontsize=16, fontweight="bold")

        importance_scores = importance_results["importance_scores"]

        # Get first importance score for visualization
        first_key = list(importance_scores.keys())[0]
        scores = importance_scores[first_key]

        # 1. Histogram of importance scores
        axes[0, 0].hist(scores, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        axes[0, 0].set_title(
            f"Distribution of Node Importance ({importance_results['method']})"
        )
        axes[0, 0].set_xlabel("Importance Score")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Top important nodes
        top_indices = np.argsort(scores)[-10:]
        top_scores = scores[top_indices]

        axes[0, 1].barh(range(len(top_scores)), top_scores, color="coral")
        axes[0, 1].set_title("Top 10 Most Important Nodes")
        axes[0, 1].set_xlabel("Importance Score")
        axes[0, 1].set_yticks(range(len(top_scores)))
        axes[0, 1].set_yticklabels([f"Node {i}" for i in top_indices])
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Importance vs node degree (if we can calculate degree)
        degrees = np.bincount(
            self.edge_index[0].cpu().numpy(), minlength=self.x.shape[0]
        )

        axes[1, 0].scatter(degrees, scores, alpha=0.6, color="green")
        axes[1, 0].set_title("Node Importance vs Degree")
        axes[1, 0].set_xlabel("Node Degree")
        axes[1, 0].set_ylabel("Importance Score")
        axes[1, 0].grid(True, alpha=0.3)

        # Calculate correlation
        correlation = np.corrcoef(degrees, scores)[0, 1]
        axes[1, 0].text(
            0.05,
            0.95,
            f"Correlation: {correlation:.3f}",
            transform=axes[1, 0].transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # 4. Class-wise importance (if labels available)
        if self.y is not None:
            unique_classes = torch.unique(self.y).cpu().numpy()
            class_importance = []
            class_labels = []

            for class_val in unique_classes:
                class_mask = self.y.cpu().numpy() == class_val
                class_scores = scores[class_mask]
                class_importance.extend(class_scores)
                class_labels.extend([f"Class {class_val}"] * len(class_scores))

            # Create boxplot
            import pandas as pd

            df_importance = pd.DataFrame(
                {"Importance": class_importance, "Class": class_labels}
            )

            sns.boxplot(data=df_importance, x="Class", y="Importance", ax=axes[1, 1])
            axes[1, 1].set_title("Importance Distribution by Class")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No class labels available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
                fontsize=12,
            )
            axes[1, 1].set_title("Class Analysis")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 Node importance visualization saved to {save_path}")

        plt.show()

    def visualize_feature_importance(
        self, feature_results: dict, save_path: str | None = None
    ):
        """Create feature importance visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            "GIMAN Feature Importance Analysis", fontsize=16, fontweight="bold"
        )

        importance = feature_results["feature_importance"]
        feature_names = feature_results["feature_names"]

        # 1. Feature importance bar plot
        sorted_idx = np.argsort(importance)
        axes[0].barh(range(len(importance)), importance[sorted_idx], color="lightcoral")
        axes[0].set_title("Feature Importance Ranking")
        axes[0].set_xlabel("Importance Score")
        axes[0].set_yticks(range(len(importance)))
        axes[0].set_yticklabels([feature_names[i] for i in sorted_idx])
        axes[0].grid(True, alpha=0.3)

        # 2. Cumulative importance
        cumsum_importance = np.cumsum(importance[sorted_idx[::-1]])
        cumsum_importance = cumsum_importance / cumsum_importance[-1]  # Normalize

        axes[1].plot(
            range(1, len(cumsum_importance) + 1),
            cumsum_importance,
            marker="o",
            linewidth=2,
            markersize=6,
            color="navy",
        )
        axes[1].axhline(
            y=0.8, color="red", linestyle="--", alpha=0.7, label="80% Threshold"
        )
        axes[1].axhline(
            y=0.95, color="orange", linestyle="--", alpha=0.7, label="95% Threshold"
        )
        axes[1].set_title("Cumulative Feature Importance")
        axes[1].set_xlabel("Number of Top Features")
        axes[1].set_ylabel("Cumulative Importance (Normalized)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # Add feature count annotations
        features_80 = np.where(cumsum_importance >= 0.8)[0][0] + 1
        features_95 = np.where(cumsum_importance >= 0.95)[0][0] + 1

        axes[1].annotate(
            f"{features_80} features\nfor 80%",
            xy=(features_80, 0.8),
            xytext=(features_80 + 1, 0.6),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
            fontsize=10,
            ha="center",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 Feature importance visualization saved to {save_path}")

        plt.show()

    def create_interactive_graph_visualization(
        self, importance_results: dict, save_path: str | None = None
    ) -> go.Figure:
        """Create an interactive graph visualization with node importance.

        Args:
            importance_results: Results from get_node_importance()
            save_path: Optional path to save HTML file

        Returns:
            Plotly Figure object
        """
        # Get importance scores
        importance_scores = importance_results["importance_scores"]
        first_key = list(importance_scores.keys())[0]
        scores = importance_scores[first_key]

        # Create NetworkX graph
        G = nx.Graph()

        # Add nodes
        for i in range(self.x.shape[0]):
            G.add_node(i)

        # Add edges
        edge_list = self.edge_index.t().cpu().numpy()
        G.add_edges_from(edge_list)

        # Calculate layout
        print("🎨 Calculating graph layout... (this may take a moment)")
        pos = nx.spring_layout(G, k=1 / np.sqrt(len(G.nodes())), iterations=50)

        # Extract positions
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]

        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="lightgray"),
            hoverinfo="none",
            mode="lines",
        )

        # Create node trace with importance coloring
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            marker=dict(
                size=10,
                color=scores,
                colorscale="Viridis",
                colorbar=dict(
                    thickness=15, len=0.7, x=1.02, title="Node<br>Importance"
                ),
                line=dict(width=1, color="white"),
            ),
        )

        # Add hover information
        node_adjacencies = []
        node_text = []
        for node in G.nodes():
            adjacencies = list(G.neighbors(node))
            node_adjacencies.append(len(adjacencies))

            # Create hover text
            hover_text = f"Node: {node}<br>"
            hover_text += f"Connections: {len(adjacencies)}<br>"
            hover_text += f"Importance: {scores[node]:.4f}<br>"

            if self.y is not None:
                hover_text += f"True Label: {self.y[node].item()}<br>"

            # Add top features for this node
            node_features = self.x[node].cpu().numpy()
            top_feature_idx = np.argsort(np.abs(node_features))[-3:][::-1]
            hover_text += "Top Features:<br>"
            for idx in top_feature_idx:
                hover_text += (
                    f"  {self.feature_names[idx]}: {node_features[idx]:.3f}<br>"
                )

            node_text.append(hover_text)

        node_trace.text = node_text

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f"GIMAN Graph Visualization<br>Node Importance ({importance_results['method']})",
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text=f"Nodes: {len(G.nodes())} | Edges: {len(G.edges())}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(color="gray", size=12),
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        if save_path:
            fig.write_html(save_path)
            print(f"📊 Interactive graph visualization saved to {save_path}")

        return fig

    def generate_interpretation_report(self, save_path: str = None) -> dict[str, Any]:
        """Generate comprehensive interpretation report.

        Args:
            save_path: Optional path to save report

        Returns:
            Complete interpretation analysis
        """
        print("🔍 Generating comprehensive GIMAN interpretation report...")

        report = {
            "model_info": {
                "num_nodes": self.x.shape[0],
                "num_features": self.x.shape[1],
                "num_edges": self.edge_index.shape[1],
                "feature_names": self.feature_names,
            },
            "analyses": {},
        }

        # 1. Node importance analysis
        print("   📊 Calculating node importance...")
        node_importance = self.get_node_importance(method="gradient")
        report["analyses"]["node_importance"] = node_importance

        # 2. Feature importance analysis
        print("   📈 Calculating feature importance...")
        feature_importance = self.get_feature_importance()
        report["analyses"]["feature_importance"] = feature_importance

        # 3. Edge contribution analysis
        print("   🔗 Analyzing edge contributions...")
        edge_contributions = self.get_edge_contributions()
        report["analyses"]["edge_contributions"] = edge_contributions

        # 4. Graph statistics
        print("   📉 Computing graph statistics...")
        degrees = np.bincount(
            self.edge_index[0].cpu().numpy(), minlength=self.x.shape[0]
        )

        report["graph_statistics"] = {
            "degree_stats": {
                "mean": float(np.mean(degrees)),
                "std": float(np.std(degrees)),
                "min": int(np.min(degrees)),
                "max": int(np.max(degrees)),
            },
            "clustering_coefficient": float(
                nx.average_clustering(
                    nx.Graph(self.edge_index.t().cpu().numpy().tolist())
                )
            ),
            "density": float(
                2 * self.edge_index.shape[1] / (self.x.shape[0] * (self.x.shape[0] - 1))
            ),
        }

        # 5. Class distribution analysis (if available)
        if self.y is not None:
            unique, counts = torch.unique(self.y, return_counts=True)
            report["class_distribution"] = {
                int(cls.item()): int(count.item())
                for cls, count in zip(unique, counts, strict=False)
            }

        # 6. Summary insights
        first_importance_key = list(node_importance["importance_scores"].keys())[0]
        node_scores = node_importance["importance_scores"][first_importance_key]

        report["insights"] = {
            "most_important_node": int(np.argmax(node_scores)),
            "least_important_node": int(np.argmin(node_scores)),
            "importance_concentration": float(
                np.std(node_scores) / np.mean(node_scores)
            ),
            "top_features": feature_importance["ranked_features"][:3],
            "most_influential_edges": edge_contributions["edge_contributions"][:3],
        }

        print("✅ Interpretation report completed!")

        if save_path:
            import json

            # Convert numpy arrays to lists for JSON serialization
            json_report = self._make_json_serializable(report)
            with open(save_path, "w") as f:
                json.dump(json_report, f, indent=2)
            print(f"📄 Report saved to {save_path}")

        return report

    def _make_json_serializable(self, obj):
        """Convert numpy arrays and torch tensors to JSON-serializable format."""
        if isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif torch.is_tensor(obj):
            return obj.cpu().numpy().tolist()
        else:
            return obj

    def compare_predictions_with_without_edges(
        self, target_node: int, num_edges_to_remove: int = 5
    ) -> dict[str, Any]:
        """Analyze how removing specific edges affects predictions for a target node.

        Args:
            target_node: Node to focus analysis on
            num_edges_to_remove: Number of top edges to analyze

        Returns:
            Detailed edge removal analysis
        """
        # Get original prediction
        with torch.no_grad():
            original_logits = self.model(self.x, self.edge_index)
            original_pred = torch.softmax(original_logits[target_node], dim=0)

        # Find edges connected to target node
        target_edges = []
        for i in range(self.edge_index.shape[1]):
            if (
                self.edge_index[0, i] == target_node
                or self.edge_index[1, i] == target_node
            ):
                target_edges.append(i)

        edge_removal_results = []

        # Test removing each edge
        for edge_idx in target_edges[:num_edges_to_remove]:
            # Create mask to remove this edge
            mask = torch.ones(self.edge_index.shape[1], dtype=torch.bool)
            mask[edge_idx] = False
            modified_edge_index = self.edge_index[:, mask]

            # Get prediction without this edge
            with torch.no_grad():
                modified_logits = self.model(self.x, modified_edge_index)
                modified_pred = torch.softmax(modified_logits[target_node], dim=0)

            # Calculate change
            pred_change = torch.abs(original_pred - modified_pred).sum().item()

            # Get edge information
            source = self.edge_index[0, edge_idx].item()
            target = self.edge_index[1, edge_idx].item()

            edge_removal_results.append(
                {
                    "edge_idx": edge_idx,
                    "source_node": source,
                    "target_node_in_edge": target,
                    "prediction_change": pred_change,
                    "original_prediction": original_pred.cpu().numpy(),
                    "modified_prediction": modified_pred.cpu().numpy(),
                }
            )

        # Sort by prediction change
        edge_removal_results.sort(key=lambda x: x["prediction_change"], reverse=True)

        return {
            "target_node": target_node,
            "original_prediction": original_pred.cpu().numpy(),
            "edge_removal_analysis": edge_removal_results,
            "total_connected_edges": len(target_edges),
        }
