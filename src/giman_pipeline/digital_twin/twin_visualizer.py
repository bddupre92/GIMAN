"""Publication-quality visualization for digital twin trajectories.

Generates figures for:
- Patient risk trajectories with ensemble confidence bands
- Counterfactual comparisons (baseline vs perturbed)
- Update history (trajectory evolution across visits)
- Cohort risk distributions
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .temporal_twin_state import TemporalCounterfactualResult, TemporalTwinState

# Publication-quality style
COLORS = {
    "baseline": "#4E79A7",
    "updated": "#E15759",
    "ci_fill": "#4E79A7",
    "counterfactual": [
        "#F28E2B",
        "#59A14F",
        "#EDC948",
        "#B07AA1",
        "#76B7B2",
        "#FF9DA7",
    ],
}


class TwinVisualizer:
    """Publication-quality visualization of digital twin trajectories."""

    @staticmethod
    def plot_patient_trajectory(
        state: TemporalTwinState,
        output_path: Path,
        title: str | None = None,
        figsize: tuple[float, float] = (10, 6),
        dpi: int = 300,
    ) -> None:
        """Plot risk trajectory with ensemble confidence band.

        Args:
            state: Patient's twin state with trajectory and CI.
            output_path: Path to save the figure.
            title: Optional custom title.
            figsize: Figure dimensions.
            dpi: Output resolution.
        """
        t = np.array(state.time_months)
        risk = np.array(state.risk_trajectory)
        ci_low = np.array(state.risk_trajectory_ci_low)
        ci_high = np.array(state.risk_trajectory_ci_high)

        fig, ax = plt.subplots(figsize=figsize)

        # Confidence band
        ax.fill_between(
            t, ci_low, ci_high, alpha=0.2, color=COLORS["ci_fill"],
            label="95% CI (ensemble)",
        )

        # Mean trajectory
        ax.plot(
            t, risk, "o-", color=COLORS["baseline"], linewidth=2,
            markersize=8, label="Risk trajectory (mean)",
        )

        # Visit labels
        for i, vid in enumerate(state.visit_ids):
            ax.annotate(
                vid, (t[i], risk[i]),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=8, color="gray",
            )

        # Final risk annotation
        ax.annotate(
            f"Risk = {state.risk_score:.3f}\n"
            f"CI: [{state.risk_score_ci[0]:.3f}, {state.risk_score_ci[1]:.3f}]",
            xy=(t[-1], risk[-1]),
            textcoords="offset points", xytext=(15, -5),
            fontsize=9, color=COLORS["baseline"],
            arrowprops={"arrowstyle": "->", "color": "gray", "lw": 0.8},
        )

        ax.set_xlabel("Time from Baseline (months)", fontsize=12)
        ax.set_ylabel("Predicted Risk (monotone)", fontsize=12)
        ax.set_title(
            title or f"Patient {state.patno}: Digital Twin Risk Trajectory",
            fontsize=13,
        )
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.set_ylim(bottom=min(0, min(ci_low) - 0.05))

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)

    @staticmethod
    def plot_counterfactual_comparison(
        result: TemporalCounterfactualResult,
        output_path: Path,
        title: str | None = None,
        figsize: tuple[float, float] = (10, 6),
        dpi: int = 300,
    ) -> None:
        """Plot baseline vs counterfactual trajectories.

        Args:
            result: Counterfactual simulation result.
            output_path: Path to save the figure.
            title: Optional custom title.
            figsize: Figure dimensions.
            dpi: Output resolution.
        """
        baseline = result.baseline_state
        t = np.array(baseline.time_months)

        fig, ax = plt.subplots(figsize=figsize)

        # Baseline with CI
        risk = np.array(baseline.risk_trajectory)
        ci_low = np.array(baseline.risk_trajectory_ci_low)
        ci_high = np.array(baseline.risk_trajectory_ci_high)
        ax.fill_between(t, ci_low, ci_high, alpha=0.15, color=COLORS["ci_fill"])
        ax.plot(
            t, risk, "o-", color=COLORS["baseline"], linewidth=2,
            markersize=7, label="Baseline",
        )

        # Counterfactual trajectories
        cf_colors = COLORS["counterfactual"]
        for i, (key, traj) in enumerate(
            result.counterfactual_trajectories.items()
        ):
            color = cf_colors[i % len(cf_colors)]
            delta = result.delta_risk.get(key, 0.0)
            sign = "+" if delta >= 0 else ""
            ax.plot(
                t, traj, "s--", color=color, linewidth=1.5,
                markersize=5, alpha=0.85,
                label=f"{key} ({sign}{delta:.4f})",
            )

        ax.set_xlabel("Time from Baseline (months)", fontsize=12)
        ax.set_ylabel("Predicted Risk", fontsize=12)
        ax.set_title(
            title
            or f"Patient {baseline.patno}: Counterfactual Analysis",
            fontsize=13,
        )
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.25)

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)

    @staticmethod
    def plot_update_history(
        states: list[TemporalTwinState],
        output_path: Path,
        title: str | None = None,
        figsize: tuple[float, float] = (10, 6),
        dpi: int = 300,
    ) -> None:
        """Plot trajectory evolution across twin updates.

        Each state is shown as a separate trajectory curve to visualize
        how risk predictions evolve as more visits arrive.

        Args:
            states: List of twin states (chronological order).
            output_path: Path to save the figure.
            title: Optional custom title.
            figsize: Figure dimensions.
            dpi: Output resolution.
        """
        if not states:
            return

        fig, ax = plt.subplots(figsize=figsize)
        patno = states[0].patno

        cmap = plt.cm.Blues
        n = len(states)

        for i, state in enumerate(states):
            t = np.array(state.time_months)
            risk = np.array(state.risk_trajectory)
            alpha = 0.4 + 0.6 * (i / max(n - 1, 1))
            color = cmap(0.3 + 0.7 * (i / max(n - 1, 1)))

            label = f"{state.n_visits} visits (risk={state.risk_score:.3f})"
            ax.plot(
                t, risk, "o-", color=color, linewidth=1.5 + i * 0.3,
                markersize=6, alpha=alpha, label=label,
            )

        # CI band for the final (most recent) state
        final = states[-1]
        t_final = np.array(final.time_months)
        ax.fill_between(
            t_final,
            np.array(final.risk_trajectory_ci_low),
            np.array(final.risk_trajectory_ci_high),
            alpha=0.15, color=COLORS["ci_fill"],
            label="95% CI (latest)",
        )

        ax.set_xlabel("Time from Baseline (months)", fontsize=12)
        ax.set_ylabel("Predicted Risk", fontsize=12)
        ax.set_title(
            title
            or f"Patient {patno}: Trajectory Evolution Across Updates",
            fontsize=13,
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.25)

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)

    @staticmethod
    def plot_cohort_risk_distribution(
        states: list[TemporalTwinState],
        output_path: Path,
        title: str | None = None,
        figsize: tuple[float, float] = (10, 5),
        dpi: int = 300,
    ) -> None:
        """Plot distribution of final risk scores across a cohort.

        Args:
            states: List of twin states for the cohort.
            output_path: Path to save the figure.
            title: Optional custom title.
            figsize: Figure dimensions.
            dpi: Output resolution.
        """
        if not states:
            return

        scores = [s.risk_score for s in states]

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(
            scores, bins=40, color=COLORS["baseline"], alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )
        ax.axvline(
            np.median(scores), color=COLORS["updated"], linestyle="--",
            linewidth=1.5, label=f"Median = {np.median(scores):.3f}",
        )

        ax.set_xlabel("Final Risk Score", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(
            title or f"Cohort Risk Distribution (N={len(states)})",
            fontsize=13,
        )
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
