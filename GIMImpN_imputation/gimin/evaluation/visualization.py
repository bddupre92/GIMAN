"""
Publication-quality figure generation for the GIMIN paper.

Generates 10 figures using matplotlib + seaborn with IEEE-compatible styling.
All figures are saved at 300 DPI as PNG and use consistent method colors.

Figures
-------
1. Accuracy heatmap (methods x modalities, RMSE values)
2. RMSE vs. mask fraction (line plot with error bars)
3. Distribution violin plots (observed vs. imputed per feature)
4. Calibration reliability diagram (expected vs. observed coverage)
5. Uncertainty vs. error scatter (with Spearman correlation)
6. Ablation study bar chart (component removal impact)
7. Patient similarity graph (NetworkX spring layout)
8. Training curves (3-panel loss plots)
9. Missingness heatmap (binary observed/missing matrix)
10. Scalability plot (time and memory vs. patient count)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


class GIMINVisualizer:
    """Publication-quality figure generator for GIMIN experiments."""

    IEEE_STYLE = {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    }

    SINGLE_COL_WIDTH = 3.5  # inches
    DOUBLE_COL_WIDTH = 7.16  # inches

    METHOD_COLORS = {
        "GIMIN": "#2196F3",
        "MICE": "#FF9800",
        "KNN": "#4CAF50",
        "Mean": "#9E9E9E",
        "Median": "#795548",
        "MissForest": "#E91E63",
        "GAIN": "#9C27B0",
        "SoftImpute": "#00BCD4",
    }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lazy_imports():
        """Lazily import heavy plotting libraries and return them."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        return plt, sns

    @staticmethod
    def _lazy_networkx():
        """Lazily import networkx."""
        import networkx as nx

        return nx

    def _apply_style(self, plt):
        """Apply IEEE-compatible rcParams."""
        plt.rcParams.update(self.IEEE_STYLE)

    def _color_for(self, method: str) -> str:
        """Return the colour for *method*, falling back to grey."""
        return self.METHOD_COLORS.get(method, "#607D8B")

    # ------------------------------------------------------------------
    # Figure 1 -- Accuracy Heatmap
    # ------------------------------------------------------------------

    def plot_accuracy_heatmap(
        self,
        results_dict: dict[str, dict[str, float]],
        output_path: str | Path,
    ) -> str:
        """Create a heatmap of RMSE values (rows=methods, cols=modalities).

        Args:
            results_dict: ``{method_name: {modality_name: rmse_value, ...}, ...}``.
            output_path: Destination PNG path.

        Returns:
            Absolute path of the saved figure.
        """
        plt, sns = self._lazy_imports()
        self._apply_style(plt)

        import pandas as pd

        df = pd.DataFrame(results_dict).T  # rows=methods, cols=modalities
        df = df.sort_index()

        n_methods = len(df.index)
        n_modalities = len(df.columns)
        fig_height = max(2.5, 0.45 * n_methods + 1.0)

        fig, ax = plt.subplots(
            figsize=(self.DOUBLE_COL_WIDTH, fig_height),
        )
        sns.heatmap(
            df.astype(float),
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            linewidths=0.5,
            linecolor="white",
            ax=ax,
            cbar_kws={"label": "RMSE"},
        )
        ax.set_xlabel("Modality")
        ax.set_ylabel("Method")
        ax.set_title("Imputation Accuracy (RMSE) by Method and Modality")
        plt.tight_layout()

        output_path = str(Path(output_path).resolve())
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Figure 2 -- RMSE vs Mask Fraction
    # ------------------------------------------------------------------

    def plot_rmse_vs_fraction(
        self,
        results_dict: dict[str, dict[float, dict[str, float]]],
        output_path: str | Path,
    ) -> str:
        """Line plot of RMSE vs. mask fraction for each method.

        Args:
            results_dict: ``{method_name: {fraction: {'mean': float, 'std': float}, ...}, ...}``.
            output_path: Destination PNG path.

        Returns:
            Absolute path of the saved figure.
        """
        plt, sns = self._lazy_imports()
        self._apply_style(plt)

        fig, ax = plt.subplots(figsize=(self.SINGLE_COL_WIDTH, 2.8))

        markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

        for idx, (method, frac_data) in enumerate(sorted(results_dict.items())):
            fractions = sorted(frac_data.keys())
            means = [frac_data[f]["mean"] for f in fractions]
            stds = [frac_data[f].get("std", 0.0) for f in fractions]
            marker = markers[idx % len(markers)]

            ax.errorbar(
                fractions,
                means,
                yerr=stds,
                label=method,
                color=self._color_for(method),
                marker=marker,
                markersize=5,
                linewidth=1.4,
                capsize=3,
                capthick=1,
            )

        ax.set_xlabel("Mask Fraction")
        ax.set_ylabel("RMSE")
        ax.set_title("Imputation RMSE vs. Missing Fraction")
        ax.legend(
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            borderaxespad=0,
            frameon=True,
            edgecolor="#cccccc",
        )
        ax.set_xticks([0.1, 0.2, 0.3, 0.5])
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()

        output_path = str(Path(output_path).resolve())
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Figure 3 -- Distribution Violin Plots
    # ------------------------------------------------------------------

    def plot_distribution_violins(
        self,
        observed_values: dict[str, np.ndarray],
        imputed_values: dict[str, np.ndarray],
        feature_names: list[str],
        output_path: str | Path,
    ) -> str:
        """Paired violin plots (observed vs. imputed) per feature.

        Args:
            observed_values: ``{feature_name: array_of_observed_values}``.
            imputed_values: ``{feature_name: array_of_imputed_values}``.
            feature_names: Features to plot.
            output_path: Destination PNG path.

        Returns:
            Absolute path of the saved figure.
        """
        from scipy import stats as sp_stats

        plt, sns = self._lazy_imports()
        self._apply_style(plt)

        n = len(feature_names)
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(self.DOUBLE_COL_WIDTH, 2.2 * nrows),
            squeeze=False,
        )

        for idx, feat in enumerate(feature_names):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]

            obs = np.asarray(observed_values[feat]).ravel()
            imp = np.asarray(imputed_values[feat]).ravel()

            parts_obs = ax.violinplot(obs, positions=[0], showmedians=True)
            parts_imp = ax.violinplot(imp, positions=[1], showmedians=True)

            # colour bodies
            for body in parts_obs.get("bodies", []):
                body.set_facecolor("#2196F3")
                body.set_alpha(0.7)
            for body in parts_imp.get("bodies", []):
                body.set_facecolor("#FF9800")
                body.set_alpha(0.7)

            # KS test
            ks_stat, ks_p = sp_stats.ks_2samp(obs, imp)
            p_label = f"p={ks_p:.2e}" if ks_p < 0.01 else f"p={ks_p:.3f}"
            ax.annotate(
                f"KS {p_label}",
                xy=(0.5, 0.97),
                xycoords="axes fraction",
                ha="center",
                va="top",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
            )

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Obs", "Imp"], fontsize=8)
            ax.set_title(feat, fontsize=9, pad=3)

        # hide unused axes
        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle(
            "Observed vs. Imputed Distributions",
            fontsize=12,
            y=1.01,
        )
        plt.tight_layout()

        output_path = str(Path(output_path).resolve())
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Figure 4 -- Calibration Reliability Diagram
    # ------------------------------------------------------------------

    def plot_calibration_reliability(
        self,
        expected_coverage: np.ndarray,
        observed_coverage: np.ndarray,
        output_path: str | Path,
    ) -> str:
        """Reliability diagram comparing expected vs. observed coverage.

        Args:
            expected_coverage: Nominal confidence levels (e.g., [0.5, 0.8, 0.9, 0.95]). Array-like, shape (n_levels,).
            observed_coverage: Empirically observed fraction of values within the interval. Array-like, shape (n_levels,).
            output_path: Destination PNG path.

        Returns:
            Absolute path of the saved figure.
        """
        plt, sns = self._lazy_imports()
        self._apply_style(plt)

        expected = np.asarray(expected_coverage)
        observed = np.asarray(observed_coverage)

        fig, ax = plt.subplots(figsize=(self.SINGLE_COL_WIDTH, 3.2))

        # perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect")

        # shade gap
        ax.fill_between(
            expected,
            expected,
            observed,
            alpha=0.20,
            color="#E91E63",
            label="Calibration gap",
        )

        # observed curve
        ax.plot(
            expected,
            observed,
            "o-",
            color="#2196F3",
            markersize=5,
            linewidth=1.5,
            label="GIMIN",
        )

        # ECE (Expected Calibration Error)
        ece = float(np.mean(np.abs(expected - observed)))
        ax.annotate(
            f"ECE = {ece:.4f}",
            xy=(0.05, 0.92),
            xycoords="axes fraction",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"),
        )

        ax.set_xlabel("Expected Confidence Level")
        ax.set_ylabel("Observed Coverage Fraction")
        ax.set_title("Calibration Reliability Diagram")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(loc="lower right", frameon=True, edgecolor="#cccccc")
        ax.grid(linestyle="--", alpha=0.3)
        plt.tight_layout()

        output_path = str(Path(output_path).resolve())
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Figure 5 -- Uncertainty vs Error Scatter
    # ------------------------------------------------------------------

    def plot_uncertainty_vs_error(
        self,
        pred_uncertainty: np.ndarray,
        actual_error: np.ndarray,
        output_path: str | Path,
    ) -> str:
        """Scatter / hex-bin plot of predicted uncertainty vs. actual error.

        Args:
            pred_uncertainty: Model-predicted uncertainty (e.g., std. dev.). Array-like, shape (n,).
            actual_error: Absolute imputation error. Array-like, shape (n,).
            output_path: Destination PNG path.

        Returns:
            Absolute path of the saved figure.
        """
        from scipy import stats as sp_stats

        plt, sns = self._lazy_imports()
        self._apply_style(plt)

        unc = np.asarray(pred_uncertainty).ravel()
        err = np.asarray(actual_error).ravel()

        fig, ax = plt.subplots(figsize=(self.SINGLE_COL_WIDTH, 3.2))

        # hex-bin for density
        hb = ax.hexbin(
            unc,
            err,
            gridsize=30,
            cmap="Blues",
            mincnt=1,
            linewidths=0.2,
        )
        cb = fig.colorbar(hb, ax=ax, pad=0.02)
        cb.set_label("Count", fontsize=8)

        # overlay transparent scatter for visibility at edges
        ax.scatter(unc, err, s=4, alpha=0.15, color="#2196F3", rasterized=True)

        # diagonal reference
        lims = [
            min(unc.min(), err.min()),
            max(unc.max(), err.max()),
        ]
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.6, label="y = x")

        # Spearman correlation
        rho, p_val = sp_stats.spearmanr(unc, err)
        p_str = f"p={p_val:.2e}" if p_val < 0.01 else f"p={p_val:.3f}"
        ax.annotate(
            f"Spearman $\\rho$={rho:.3f}\n({p_str})",
            xy=(0.05, 0.92),
            xycoords="axes fraction",
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"),
        )

        ax.set_xlabel("Predicted Uncertainty")
        ax.set_ylabel("Absolute Imputation Error")
        ax.set_title("Uncertainty vs. Error")
        ax.grid(linestyle="--", alpha=0.3)
        plt.tight_layout()

        output_path = str(Path(output_path).resolve())
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Figure 6 -- Ablation Bar Chart
    # ------------------------------------------------------------------

    def plot_ablation_bar(
        self,
        ablation_results: dict[str, dict[str, float]],
        output_path: str | Path,
    ) -> str:
        """Grouped bar chart for ablation study.

        Args:
            ablation_results: ``{component_label: {'mean': float, 'std': float, 'significant': bool}, ...}``.
                Must include a ``'Full Model'`` key used as the baseline.
            output_path: Destination PNG path.

        Returns:
            Absolute path of the saved figure.
        """
        plt, sns = self._lazy_imports()
        self._apply_style(plt)

        components = list(ablation_results.keys())
        means = [ablation_results[c]["mean"] for c in components]
        stds = [ablation_results[c].get("std", 0.0) for c in components]
        significants = [
            ablation_results[c].get("significant", False) for c in components
        ]

        # colour: full model gets distinct colour, others grey-ish
        colors = []
        for c in components:
            if c == "Full Model":
                colors.append(self.METHOD_COLORS["GIMIN"])
            else:
                colors.append("#B0BEC5")

        fig, ax = plt.subplots(figsize=(self.SINGLE_COL_WIDTH, 3.0))

        x = np.arange(len(components))
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            capsize=3,
            error_kw={"linewidth": 1},
        )

        # baseline horizontal line for Full Model
        if "Full Model" in ablation_results:
            baseline = ablation_results["Full Model"]["mean"]
            ax.axhline(
                y=baseline,
                color=self.METHOD_COLORS["GIMIN"],
                linestyle="--",
                linewidth=0.9,
                alpha=0.7,
                label="Full model baseline",
            )

        # significance markers
        for i, (sig, mean_val, std_val) in enumerate(zip(significants, means, stds)):
            if sig and components[i] != "Full Model":
                ax.annotate(
                    "*",
                    xy=(i, mean_val + std_val + 0.005),
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    fontweight="bold",
                    color="#D32F2F",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("RMSE")
        ax.set_title("Ablation Study")
        ax.legend(loc="upper right", fontsize=8, frameon=True, edgecolor="#cccccc")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()

        output_path = str(Path(output_path).resolve())
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Figure 7 -- Patient Similarity Graph
    # ------------------------------------------------------------------

    def plot_patient_graph(
        self,
        edge_index: np.ndarray,
        node_features: np.ndarray,
        node_labels: np.ndarray,
        missingness_frac: np.ndarray,
        output_path: str | Path,
        edge_weights: np.ndarray | None = None,
        label_names: dict[int, str] | None = None,
    ) -> str:
        """Visualise the patient similarity graph with spring layout.

        Args:
            edge_index: COO-format edge indices. Array-like, shape (2, E).
            node_features: Node feature matrix (used only for layout seed). Array-like, shape (N, D).
            node_labels: Integer diagnosis labels for node colouring. Array-like, shape (N,).
            missingness_frac: Fraction of missing values per patient (0 = fully observed). Array-like, shape (N,).
            output_path: Destination PNG path.
            edge_weights: Edge weights; used for edge alpha. Array-like, shape (E,). Optional.
            label_names: ``{label_int: human_readable_name}``. Used for the legend. Optional.

        Returns:
            Absolute path of the saved figure.
        """
        plt, sns = self._lazy_imports()
        nx = self._lazy_networkx()
        self._apply_style(plt)

        edge_index = np.asarray(edge_index)
        node_labels = np.asarray(node_labels)
        missingness_frac = np.asarray(missingness_frac).ravel()

        N = node_labels.shape[0]
        E = edge_index.shape[1]

        G = nx.Graph()
        G.add_nodes_from(range(N))
        for e in range(E):
            src, dst = int(edge_index[0, e]), int(edge_index[1, e])
            w = float(edge_weights[e]) if edge_weights is not None else 1.0
            if G.has_edge(src, dst):
                continue
            G.add_edge(src, dst, weight=w)

        # layout
        pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(N + 1))

        fig, ax = plt.subplots(figsize=(self.DOUBLE_COL_WIDTH, 5.0))

        # edges with alpha proportional to weight
        if edge_weights is not None:
            ew = np.asarray(edge_weights).ravel()
            ew_norm = (ew - ew.min()) / (ew.max() - ew.min() + 1e-9)
        else:
            ew_norm = np.ones(E) * 0.3

        edge_list = list(G.edges())
        edge_alpha_map: dict[tuple[int, int], float] = {}
        for e in range(E):
            src, dst = int(edge_index[0, e]), int(edge_index[1, e])
            key = (min(src, dst), max(src, dst))
            edge_alpha_map[key] = float(np.clip(ew_norm[e] * 0.6 + 0.05, 0.05, 0.65))

        for u, v in edge_list:
            key = (min(u, v), max(u, v))
            alpha = edge_alpha_map.get(key, 0.15)
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                "-",
                color="#9E9E9E",
                alpha=alpha,
                linewidth=0.5,
                zorder=1,
            )

        # nodes: colour by label, size by completeness
        unique_labels = np.unique(node_labels)
        cmap = plt.cm.get_cmap("tab10", len(unique_labels))
        label_to_color = {int(lab): cmap(i) for i, lab in enumerate(unique_labels)}

        completeness = 1.0 - missingness_frac
        node_sizes = 20 + 180 * completeness  # range ~20..200

        for lab in unique_labels:
            mask = node_labels == lab
            indices = np.where(mask)[0]
            xs = [pos[int(i)][0] for i in indices]
            ys = [pos[int(i)][1] for i in indices]
            sizes = node_sizes[mask]
            display_label = (
                label_names[int(lab)]
                if label_names and int(lab) in label_names
                else f"Label {lab}"
            )
            ax.scatter(
                xs,
                ys,
                s=sizes,
                c=[label_to_color[int(lab)]],
                label=display_label,
                edgecolors="white",
                linewidths=0.3,
                zorder=2,
            )

        ax.set_title("Patient Similarity Graph")
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            edgecolor="#cccccc",
            title="Diagnosis",
            title_fontsize=9,
        )
        ax.axis("off")
        plt.tight_layout()

        output_path = str(Path(output_path).resolve())
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Figure 8 -- Training Curves
    # ------------------------------------------------------------------

    def plot_training_curves(
        self,
        history: dict[str, Any],
        output_path: str | Path,
    ) -> str:
        """Three-panel training-curve plot (total, reconstruction, distribution).

        Args:
            history: Must contain keys ``'total_loss'``, ``'recon_loss'``,
                ``'dist_loss'`` (each a list of per-epoch floats).
                Optional keys:
                - ``'graph_refinement_epochs'``: list of epoch indices where the
                  graph was refined.
                - ``'best_epoch'``: int, epoch with the best validation loss.
            output_path: Destination PNG path.

        Returns:
            Absolute path of the saved figure.
        """
        plt, sns = self._lazy_imports()
        self._apply_style(plt)

        loss_keys = ["total_loss", "recon_loss", "dist_loss"]
        titles = ["Total Loss", "Reconstruction Loss", "Distribution Loss"]

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(self.DOUBLE_COL_WIDTH, 2.6),
            sharex=True,
        )

        for ax, key, title in zip(axes, loss_keys, titles):
            values = np.asarray(history[key])
            epochs = np.arange(1, len(values) + 1)

            ax.semilogy(epochs, values, linewidth=1.2, color="#2196F3")
            ax.set_title(title, fontsize=10, pad=4)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(linestyle="--", alpha=0.3)

            # graph refinement boundaries
            if "graph_refinement_epochs" in history:
                for ge in history["graph_refinement_epochs"]:
                    ax.axvline(
                        x=ge,
                        color="#E91E63",
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.6,
                    )

            # best epoch marker
            if "best_epoch" in history:
                be = history["best_epoch"]
                if 0 <= be - 1 < len(values):
                    ax.plot(
                        be,
                        values[be - 1],
                        marker="*",
                        markersize=10,
                        color="#FF5722",
                        zorder=5,
                        label=f"Best (epoch {be})",
                    )
                    ax.legend(loc="upper right", fontsize=7, frameon=True)

        fig.suptitle("Training Curves", fontsize=12, y=1.03)
        plt.tight_layout()

        output_path = str(Path(output_path).resolve())
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Figure 9 -- Missingness Heatmap
    # ------------------------------------------------------------------

    def plot_missingness_heatmap(
        self,
        mask_matrix: np.ndarray,
        feature_names: list[str],
        output_path: str | Path,
        modality_groups: dict[str, list[str]] | None = None,
        max_patients: int = 200,
    ) -> str:
        """Binary heatmap of the missingness pattern.

        Args:
            mask_matrix: Binary matrix where 1 = observed, 0 = missing. Array-like, shape (N, D).
            feature_names: Column names corresponding to *mask_matrix* columns.
            output_path: Destination PNG path.
            modality_groups: ``{modality_name: [feature_name, ...]}``. Used to draw vertical
                separators and group features. Optional.
            max_patients: Subsample rows when *N* exceeds this value.

        Returns:
            Absolute path of the saved figure.
        """
        plt, sns = self._lazy_imports()
        self._apply_style(plt)

        mask = np.asarray(mask_matrix)
        N, D = mask.shape

        # subsample if needed
        if N > max_patients:
            rng = np.random.default_rng(42)
            idx = rng.choice(N, size=max_patients, replace=False)
            idx.sort()
            mask = mask[idx]
            N = max_patients

        # reorder columns by modality if provided
        col_order = list(range(D))
        group_boundaries: list[int] = []
        group_labels: list[str] = []
        if modality_groups is not None:
            col_order = []
            feat_to_idx = {f: i for i, f in enumerate(feature_names)}
            for mod_name, feats in modality_groups.items():
                group_boundaries.append(len(col_order))
                group_labels.append(mod_name)
                for f in feats:
                    if f in feat_to_idx:
                        col_order.append(feat_to_idx[f])
            # append any features not in groups
            remaining = [i for i in range(D) if i not in col_order]
            if remaining:
                group_boundaries.append(len(col_order))
                group_labels.append("Other")
                col_order.extend(remaining)

        mask = mask[:, col_order]
        ordered_names = [feature_names[i] for i in col_order]

        # per-feature missingness rate (on full reordered mask)
        miss_rate = 1.0 - mask.mean(axis=0)

        fig_height = max(3.0, 0.015 * N + 1.5)
        fig = plt.figure(figsize=(self.DOUBLE_COL_WIDTH, fig_height))

        # GridSpec: top bar + main heatmap
        gs = fig.add_gridspec(
            2,
            1,
            height_ratios=[1, max(6, int(N / 30))],
            hspace=0.05,
        )
        ax_bar = fig.add_subplot(gs[0])
        ax_hm = fig.add_subplot(gs[1])

        # top bar: per-feature missingness
        ax_bar.bar(
            np.arange(D),
            miss_rate,
            color="#E91E63",
            width=1.0,
            edgecolor="none",
        )
        ax_bar.set_xlim(-0.5, D - 0.5)
        ax_bar.set_ylabel("Miss %", fontsize=8)
        ax_bar.set_xticks([])
        ax_bar.set_ylim(0, min(1.0, miss_rate.max() * 1.2 + 0.05))
        ax_bar.tick_params(axis="y", labelsize=7)

        # main heatmap: black=observed, white=missing
        ax_hm.imshow(
            mask,
            aspect="auto",
            cmap="gray",
            interpolation="none",
            vmin=0,
            vmax=1,
        )
        ax_hm.set_xlabel("Feature")
        ax_hm.set_ylabel("Patient")

        # feature labels
        if D <= 40:
            ax_hm.set_xticks(np.arange(D))
            ax_hm.set_xticklabels(ordered_names, rotation=90, fontsize=6)
        else:
            ax_hm.set_xticks([])

        # modality separators
        for boundary in group_boundaries[1:]:
            ax_hm.axvline(x=boundary - 0.5, color="red", linewidth=0.8)
            ax_bar.axvline(x=boundary - 0.5, color="red", linewidth=0.8)

        ax_hm.set_title(
            "Missingness Pattern (black=observed, white=missing)", fontsize=10
        )
        plt.tight_layout()

        output_path = str(Path(output_path).resolve())
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Figure 10 -- Scalability Plot
    # ------------------------------------------------------------------

    def plot_scalability(
        self,
        patient_counts: Sequence[int],
        times: Sequence[float],
        memory_usage: Sequence[float],
        incremental_times: Sequence[float],
        output_path: str | Path,
    ) -> str:
        """Dual-axis scalability plot (time and memory vs. patient count).

        Args:
            patient_counts: Number of patients for each measurement.
            times: Total training time in seconds.
            memory_usage: Peak memory in MB.
            incremental_times: Time to incorporate a single new patient (seconds).
            output_path: Destination PNG path.

        Returns:
            Absolute path of the saved figure.
        """
        plt, sns = self._lazy_imports()
        self._apply_style(plt)

        counts = np.asarray(patient_counts)
        t = np.asarray(times)
        mem = np.asarray(memory_usage)
        inc = np.asarray(incremental_times)

        fig, ax1 = plt.subplots(figsize=(self.SINGLE_COL_WIDTH, 3.0))

        # left axis: time
        color_time = "#2196F3"
        color_inc = "#4CAF50"
        ax1.set_xlabel("Number of Patients")
        ax1.set_ylabel("Time (s)", color=color_time)
        (line1,) = ax1.plot(
            counts,
            t,
            "o-",
            color=color_time,
            markersize=4,
            linewidth=1.3,
            label="Total training time",
        )
        (line3,) = ax1.plot(
            counts,
            inc,
            "s--",
            color=color_inc,
            markersize=4,
            linewidth=1.1,
            label="Incremental (1 patient)",
        )
        ax1.tick_params(axis="y", labelcolor=color_time)

        # right axis: memory
        color_mem = "#E91E63"
        ax2 = ax1.twinx()
        ax2.set_ylabel("Memory (MB)", color=color_mem)
        (line2,) = ax2.plot(
            counts,
            mem,
            "^-",
            color=color_mem,
            markersize=4,
            linewidth=1.3,
            label="Peak memory",
        )
        ax2.tick_params(axis="y", labelcolor=color_mem)

        # combined legend
        lines = [line1, line3, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(
            lines,
            labels,
            loc="upper left",
            fontsize=8,
            frameon=True,
            edgecolor="#cccccc",
        )

        ax1.set_title("Scalability")
        ax1.grid(linestyle="--", alpha=0.3)
        fig.tight_layout()

        output_path = str(Path(output_path).resolve())
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Convenience -- Generate All Figures
    # ------------------------------------------------------------------

    def generate_all_figures(
        self,
        results_dir: str | Path,
        output_dir: str | Path,
    ) -> dict[str, str]:
        """Load result files from *results_dir* and generate all 10 figures.

        Expected files inside *results_dir* (JSON or .npy):

        - ``accuracy_heatmap.json``   -> Figure 1
        - ``rmse_vs_fraction.json``   -> Figure 2
        - ``distributions.npz``       -> Figure 3
        - ``calibration.npz``         -> Figure 4
        - ``uncertainty.npz``         -> Figure 5
        - ``ablation.json``           -> Figure 6
        - ``graph.npz``               -> Figure 7
        - ``training_history.json``   -> Figure 8
        - ``missingness.npz``         -> Figure 9
        - ``scalability.json``        -> Figure 10

        Args:
            results_dir: Directory containing the result files listed above.
            output_dir: Directory where figures will be saved.

        Returns:
            ``{figure_name: output_path}`` for every figure that was
            successfully generated.
        """
        results_dir = Path(results_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated: dict[str, str] = {}

        # -- Figure 1: accuracy heatmap -----------------------------------
        path = results_dir / "accuracy_heatmap.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            try:
                out = self.plot_accuracy_heatmap(
                    data,
                    output_dir / "fig01_accuracy_heatmap.png",
                )
                generated["accuracy_heatmap"] = out
            except Exception as exc:
                print(f"[WARN] Figure 1 failed: {exc}")

        # -- Figure 2: RMSE vs fraction -----------------------------------
        path = results_dir / "rmse_vs_fraction.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            # JSON keys are strings; convert fraction keys to float
            converted = {}
            for method, frac_data in data.items():
                converted[method] = {float(k): v for k, v in frac_data.items()}
            try:
                out = self.plot_rmse_vs_fraction(
                    converted,
                    output_dir / "fig02_rmse_vs_fraction.png",
                )
                generated["rmse_vs_fraction"] = out
            except Exception as exc:
                print(f"[WARN] Figure 2 failed: {exc}")

        # -- Figure 3: distribution violins --------------------------------
        path = results_dir / "distributions.npz"
        if path.exists():
            npz = np.load(path, allow_pickle=True)
            feature_names = list(npz["feature_names"])
            observed = {f: npz[f"obs_{f}"] for f in feature_names}
            imputed = {f: npz[f"imp_{f}"] for f in feature_names}
            try:
                out = self.plot_distribution_violins(
                    observed,
                    imputed,
                    feature_names,
                    output_dir / "fig03_distribution_violins.png",
                )
                generated["distribution_violins"] = out
            except Exception as exc:
                print(f"[WARN] Figure 3 failed: {exc}")

        # -- Figure 4: calibration reliability -----------------------------
        path = results_dir / "calibration.npz"
        if path.exists():
            npz = np.load(path)
            try:
                out = self.plot_calibration_reliability(
                    npz["expected"],
                    npz["observed"],
                    output_dir / "fig04_calibration.png",
                )
                generated["calibration"] = out
            except Exception as exc:
                print(f"[WARN] Figure 4 failed: {exc}")

        # -- Figure 5: uncertainty vs error --------------------------------
        path = results_dir / "uncertainty.npz"
        if path.exists():
            npz = np.load(path)
            try:
                out = self.plot_uncertainty_vs_error(
                    npz["pred_uncertainty"],
                    npz["actual_error"],
                    output_dir / "fig05_uncertainty_vs_error.png",
                )
                generated["uncertainty_vs_error"] = out
            except Exception as exc:
                print(f"[WARN] Figure 5 failed: {exc}")

        # -- Figure 6: ablation bar chart ----------------------------------
        path = results_dir / "ablation.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            try:
                out = self.plot_ablation_bar(
                    data,
                    output_dir / "fig06_ablation.png",
                )
                generated["ablation"] = out
            except Exception as exc:
                print(f"[WARN] Figure 6 failed: {exc}")

        # -- Figure 7: patient graph --------------------------------------
        path = results_dir / "graph.npz"
        if path.exists():
            npz = np.load(path, allow_pickle=True)
            edge_weights = npz.get("edge_weights", None)
            label_names = None
            if "label_names" in npz:
                label_names = npz["label_names"].item()  # dict stored as 0-d object
            try:
                out = self.plot_patient_graph(
                    edge_index=npz["edge_index"],
                    node_features=npz["node_features"],
                    node_labels=npz["node_labels"],
                    missingness_frac=npz["missingness_frac"],
                    output_path=output_dir / "fig07_patient_graph.png",
                    edge_weights=edge_weights,
                    label_names=label_names,
                )
                generated["patient_graph"] = out
            except Exception as exc:
                print(f"[WARN] Figure 7 failed: {exc}")

        # -- Figure 8: training curves ------------------------------------
        path = results_dir / "training_history.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            try:
                out = self.plot_training_curves(
                    data,
                    output_dir / "fig08_training_curves.png",
                )
                generated["training_curves"] = out
            except Exception as exc:
                print(f"[WARN] Figure 8 failed: {exc}")

        # -- Figure 9: missingness heatmap --------------------------------
        path = results_dir / "missingness.npz"
        if path.exists():
            npz = np.load(path, allow_pickle=True)
            feature_names = list(npz["feature_names"])
            modality_groups = None
            if "modality_groups" in npz:
                modality_groups = npz["modality_groups"].item()
            try:
                out = self.plot_missingness_heatmap(
                    mask_matrix=npz["mask_matrix"],
                    feature_names=feature_names,
                    output_path=output_dir / "fig09_missingness.png",
                    modality_groups=modality_groups,
                )
                generated["missingness"] = out
            except Exception as exc:
                print(f"[WARN] Figure 9 failed: {exc}")

        # -- Figure 10: scalability ----------------------------------------
        path = results_dir / "scalability.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            try:
                out = self.plot_scalability(
                    patient_counts=data["patient_counts"],
                    times=data["times"],
                    memory_usage=data["memory_usage"],
                    incremental_times=data["incremental_times"],
                    output_path=output_dir / "fig10_scalability.png",
                )
                generated["scalability"] = out
            except Exception as exc:
                print(f"[WARN] Figure 10 failed: {exc}")

        return generated
