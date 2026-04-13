"""Temporal Validation — Expanding-Window Splits for NSD-ISS Transition Models.

Implements enrollment-date-ordered temporal splits to assess deployment
readiness of Dynamic-DeepHit and Graph-DT survival models.

Key concepts:
    - Patients are ordered by enrollment date (INFODT from PPMI Demographics)
    - 4 expanding training windows test how performance scales with data size
    - Test sets are ALWAYS chronologically after training sets (no leakage)
    - Covariate shift detection via KS, PSI, and MMD tests

Window definitions:
    W1: first 40% train, next 20% test  (~760 / ~380 patients)
    W2: first 60% train, next 20% test  (~1140 / ~380)
    W3: first 80% train, final 20% test (~1520 / ~380)
    W4: first 50% train, final 50% test (~950 / ~950)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Window Definitions ────────────────────────────────────────────────

WINDOW_DEFS = [
    # (train_start_frac, train_end_frac, test_start_frac, test_end_frac)
    (0.0, 0.4, 0.4, 0.6),  # W1: 40% train, next 20% test
    (0.0, 0.6, 0.6, 0.8),  # W2: 60% train, next 20% test
    (0.0, 0.8, 0.8, 1.0),  # W3: 80% train, final 20% test
    (0.0, 0.5, 0.5, 1.0),  # W4: 50% train, final 50% test
]


@dataclass
class WindowSplit:
    """Result of a temporal window split."""

    window_idx: int
    train_patnos: list[int]
    test_patnos: list[int]
    train_enrollment_range: tuple[str, str]  # (earliest, latest) dates
    test_enrollment_range: tuple[str, str]
    n_train: int
    n_test: int


@dataclass
class ShiftResult:
    """Covariate shift test results for one feature in one window."""

    feature: str
    window_idx: int
    ks_statistic: float
    ks_pvalue: float
    psi: float
    shifted: bool  # p < 0.001 or PSI > 0.25


# ── Temporal Splitter ─────────────────────────────────────────────────


class TemporalSplitter:
    """Enrollment-date-ordered temporal splits for PPMI patients.

    Extracts enrollment dates from the PPMI Demographics CSV (INFODT column,
    MM/YYYY format), sorts all patients by enrollment date, and provides
    expanding-window train/test splits for temporal validation.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        demographics_path: Path,
        verbose: bool = True,
    ):
        self.verbose = verbose

        # Get unique patients from features
        feature_patnos = set(features_df["PATNO"].unique())

        # Parse enrollment dates
        enrollment = self._parse_enrollment_dates(demographics_path, feature_patnos)

        # Sort patients by enrollment date
        enrollment = enrollment.sort_values("enrollment_date")
        self.ordered_patnos = enrollment["PATNO"].tolist()
        self.enrollment_dates = enrollment.set_index("PATNO")["enrollment_date"]

        if verbose:
            n = len(self.ordered_patnos)
            date_range = (
                self.enrollment_dates.min().strftime("%Y-%m"),
                self.enrollment_dates.max().strftime("%Y-%m"),
            )
            print(f"  TemporalSplitter: {n} patients ordered by enrollment date")
            print(f"  Date range: {date_range[0]} to {date_range[1]}")

        # Validate: earlier patients should have more visits
        self._validate_ordering(features_df)

    def _parse_enrollment_dates(
        self,
        demographics_path: Path,
        feature_patnos: set[int],
    ) -> pd.DataFrame:
        """Extract earliest INFODT per patient from Demographics CSV."""
        demo = pd.read_csv(demographics_path)

        # INFODT is MM/YYYY format (e.g., "01/2011")
        demo["enrollment_date"] = pd.to_datetime(
            demo["INFODT"],
            format="%m/%Y",
            errors="coerce",
        )

        # Take earliest date per patient (screening or transformed visit)
        earliest = demo.groupby("PATNO")["enrollment_date"].min().reset_index()

        # Filter to patients in features data
        earliest = earliest[earliest["PATNO"].isin(feature_patnos)]
        n_missing = earliest["enrollment_date"].isna().sum()

        if n_missing > 0:
            warnings.warn(
                f"{n_missing} patients missing enrollment dates; "
                f"using PATNO ordering as fallback for those."
            )
            # For missing dates, assign a date based on PATNO order
            # (PPMI PATNOs roughly correlate with enrollment order)
            missing_mask = earliest["enrollment_date"].isna()
            missing_patnos = earliest.loc[missing_mask, "PATNO"].sort_values()
            # Place them at the very end with synthetic dates
            max_date = earliest["enrollment_date"].max()
            if pd.isna(max_date):
                max_date = pd.Timestamp("2020-01-01")
            for i, patno in enumerate(missing_patnos):
                earliest.loc[earliest["PATNO"] == patno, "enrollment_date"] = (
                    max_date + pd.DateOffset(days=i + 1)
                )

        return earliest.dropna(subset=["enrollment_date"])

    def _validate_ordering(self, features_df: pd.DataFrame) -> None:
        """Sanity check: earlier patients should have more follow-up visits."""
        visit_counts = features_df.groupby("PATNO").size()
        ranks = []
        counts = []
        for rank, patno in enumerate(self.ordered_patnos):
            if patno in visit_counts.index:
                ranks.append(rank)
                counts.append(visit_counts[patno])

        if len(ranks) > 10:
            corr, pval = stats.spearmanr(ranks, counts)
            if self.verbose:
                print(f"  Enrollment rank vs visit count: rho={corr:.3f}, p={pval:.3e}")
            if corr > 0.1:
                warnings.warn(
                    f"Positive correlation ({corr:.3f}) between enrollment rank "
                    f"and visit count — expected negative (earlier patients should "
                    f"have more visits). Check enrollment date parsing."
                )

    @property
    def n_patients(self) -> int:
        return len(self.ordered_patnos)

    @property
    def n_windows(self) -> int:
        return len(WINDOW_DEFS)

    def get_window(self, window_idx: int) -> WindowSplit:
        """Get train/test patient IDs for an expanding temporal window.

        Args:
            window_idx: 0-3 (W1 through W4)

        Returns:
            WindowSplit with train and test patient ID lists.
        """
        if window_idx < 0 or window_idx >= len(WINDOW_DEFS):
            raise ValueError(
                f"window_idx must be 0-{len(WINDOW_DEFS) - 1}, got {window_idx}"
            )

        tr_start, tr_end, te_start, te_end = WINDOW_DEFS[window_idx]
        n = self.n_patients

        train_slice = self.ordered_patnos[int(tr_start * n) : int(tr_end * n)]
        test_slice = self.ordered_patnos[int(te_start * n) : int(te_end * n)]

        train_dates = self.enrollment_dates[train_slice]
        test_dates = self.enrollment_dates[test_slice]

        return WindowSplit(
            window_idx=window_idx,
            train_patnos=train_slice,
            test_patnos=test_slice,
            train_enrollment_range=(
                train_dates.min().strftime("%Y-%m"),
                train_dates.max().strftime("%Y-%m"),
            ),
            test_enrollment_range=(
                test_dates.min().strftime("%Y-%m"),
                test_dates.max().strftime("%Y-%m"),
            ),
            n_train=len(train_slice),
            n_test=len(test_slice),
        )

    def validate_split(self, window_idx: int) -> dict:
        """Validate no temporal leakage in a window split.

        Returns dict with validation metrics.
        """
        ws = self.get_window(window_idx)
        train_dates = self.enrollment_dates[ws.train_patnos]
        test_dates = self.enrollment_dates[ws.test_patnos]

        train_max = train_dates.max()
        test_min = test_dates.min()
        no_leakage = train_max <= test_min

        overlap_count = len(set(ws.train_patnos) & set(ws.test_patnos))

        return {
            "window_idx": window_idx,
            "n_train": ws.n_train,
            "n_test": ws.n_test,
            "train_date_range": ws.train_enrollment_range,
            "test_date_range": ws.test_enrollment_range,
            "no_temporal_leakage": bool(no_leakage),
            "train_max_date": str(train_max),
            "test_min_date": str(test_min),
            "patient_overlap": overlap_count,
        }


# ── Covariate Shift Detection ────────────────────────────────────────


def compute_psi(
    train_values: np.ndarray,
    test_values: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Population Stability Index (PSI).

    PSI > 0.25 indicates significant distribution shift.
    PSI > 0.10 indicates moderate shift requiring investigation.
    """
    # Remove NaN
    train_values = train_values[~np.isnan(train_values)]
    test_values = test_values[~np.isnan(test_values)]

    if len(train_values) < 10 or len(test_values) < 10:
        return float("nan")

    # Compute bin edges from training distribution
    edges = np.percentile(train_values, np.linspace(0, 100, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    # Remove duplicate edges
    edges = np.unique(edges)

    train_counts = np.histogram(train_values, bins=edges)[0]
    test_counts = np.histogram(test_values, bins=edges)[0]

    # Convert to proportions (add small epsilon to avoid division by zero)
    eps = 1e-8
    train_pct = train_counts / train_counts.sum() + eps
    test_pct = test_counts / test_counts.sum() + eps

    psi = np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))
    return float(psi)


def compute_mmd_rbf(
    X_train: np.ndarray,
    X_test: np.ndarray,
    gamma: float | None = None,
    n_permutations: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Maximum Mean Discrepancy with RBF kernel and permutation p-value.

    Args:
        X_train: (n_train, d) feature matrix
        X_test: (n_test, d) feature matrix
        gamma: RBF kernel bandwidth. If None, uses median heuristic.
        n_permutations: Number of bootstrap permutations for p-value.

    Returns:
        (mmd_statistic, p_value)
    """
    # Remove rows with any NaN
    X_train = X_train[~np.isnan(X_train).any(axis=1)]
    X_test = X_test[~np.isnan(X_test).any(axis=1)]

    if len(X_train) < 10 or len(X_test) < 10:
        return float("nan"), float("nan")

    # Subsample for efficiency
    rng = np.random.RandomState(seed)
    max_n = 500
    if len(X_train) > max_n:
        X_train = X_train[rng.choice(len(X_train), max_n, replace=False)]
    if len(X_test) > max_n:
        X_test = X_test[rng.choice(len(X_test), max_n, replace=False)]

    n_tr, n_te = len(X_train), len(X_test)

    if gamma is None:
        # Median heuristic
        combined = np.vstack([X_train, X_test])
        dists = np.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=-1)
        gamma = 1.0 / (2.0 * max(np.median(dists[dists > 0]), 1e-8))

    def _rbf_mmd(X, Y):
        XX = np.exp(-gamma * np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1))
        YY = np.exp(-gamma * np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=-1))
        XY = np.exp(-gamma * np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1))
        return XX.mean() + YY.mean() - 2 * XY.mean()

    observed = _rbf_mmd(X_train, X_test)

    # Permutation test
    combined = np.vstack([X_train, X_test])
    perm_stats = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(len(combined))
        perm_stats[i] = _rbf_mmd(combined[perm[:n_tr]], combined[perm[n_tr:]])

    p_value = (perm_stats >= observed).mean()
    return float(observed), float(p_value)


def compute_covariate_shift(
    features_df: pd.DataFrame,
    train_patnos: list[int],
    test_patnos: list[int],
    feature_columns: list[str] | None = None,
    verbose: bool = True,
) -> tuple[list[ShiftResult], dict]:
    """Full covariate shift analysis between train and test windows.

    Uses baseline (first visit) features per patient. Computes per-feature
    KS test + PSI, and multivariate MMD on joint distribution.

    Args:
        features_df: Longitudinal features DataFrame.
        train_patnos: Training patient IDs.
        test_patnos: Test patient IDs.
        feature_columns: Columns to test. If None, uses standard set.

    Returns:
        (per_feature_results, summary_dict)
    """
    from giman_pipeline.paper3.dynamic_deephit import ALL_FEATURES

    if feature_columns is None:
        feature_columns = [c for c in ALL_FEATURES if c in features_df.columns]

    # Extract baseline features per patient (first visit)
    baseline = features_df.sort_values("months_from_baseline").groupby("PATNO").first()

    train_base = baseline.loc[baseline.index.isin(train_patnos), feature_columns]
    test_base = baseline.loc[baseline.index.isin(test_patnos), feature_columns]

    if verbose:
        print(
            f"  Shift analysis: {len(train_base)} train, {len(test_base)} test, "
            f"{len(feature_columns)} features"
        )

    # Per-feature KS + PSI
    per_feature = []
    n_shifted = 0
    for col in feature_columns:
        tr_vals = train_base[col].dropna().values
        te_vals = test_base[col].dropna().values

        if len(tr_vals) < 5 or len(te_vals) < 5:
            per_feature.append(
                ShiftResult(
                    feature=col,
                    window_idx=-1,
                    ks_statistic=float("nan"),
                    ks_pvalue=float("nan"),
                    psi=float("nan"),
                    shifted=False,
                )
            )
            continue

        ks_stat, ks_pval = stats.ks_2samp(tr_vals, te_vals)
        psi = compute_psi(tr_vals, te_vals)
        shifted = ks_pval < 0.001 or psi > 0.25

        per_feature.append(
            ShiftResult(
                feature=col,
                window_idx=-1,
                ks_statistic=float(ks_stat),
                ks_pvalue=float(ks_pval),
                psi=float(psi),
                shifted=shifted,
            )
        )
        if shifted:
            n_shifted += 1

    # Multivariate MMD
    train_mat = train_base.values.astype(np.float64)
    test_mat = test_base.values.astype(np.float64)
    mmd_stat, mmd_pval = compute_mmd_rbf(train_mat, test_mat)

    # Shift severity
    n_features = len(feature_columns)
    frac_shifted = n_shifted / max(n_features, 1)
    if frac_shifted < 0.10:
        severity = "mild"
    elif frac_shifted < 0.30:
        severity = "moderate"
    else:
        severity = "severe"

    summary = {
        "n_features_tested": n_features,
        "n_shifted": n_shifted,
        "fraction_shifted": float(frac_shifted),
        "severity": severity,
        "mmd_statistic": mmd_stat,
        "mmd_pvalue": mmd_pval,
    }

    if verbose:
        print(
            f"  Shift severity: {severity} ({n_shifted}/{n_features} features shifted)"
        )
        print(f"  MMD: {mmd_stat:.6f} (p={mmd_pval:.4f})")

    return per_feature, summary
