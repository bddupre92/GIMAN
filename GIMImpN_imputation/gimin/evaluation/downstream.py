"""Downstream task evaluation for GIMIN-imputed data.

Evaluates how well GIMIN-imputed multimodal clinical data supports
downstream classification tasks -- specifically, the SAA (Sleep Apnea
Assessment) classification task used in the companion GIMAN neuro-fuzzy
framework.

The evaluator trains a lightweight classifier on imputed data and
compares its performance against a MICE-imputed control, using:

- Accuracy, balanced accuracy, F1, AUROC
- Cohen's kappa
- Confidence-interval estimation via bootstrap

This provides an extrinsic measure of imputation quality beyond the
intrinsic point- and distribution-based metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import ArrayLike
from ..utils import to_numpy as _to_numpy

logger = logging.getLogger(__name__)


class DownstreamEvaluator:
    """Evaluate impact of GIMIN imputation on downstream SAA classification.

    The evaluator supports two modes:

    1. **Internal classifier** -- trains a gradient-boosted classifier
       (XGBoost or sklearn GBM) on the imputed data and reports
       standard classification metrics.
    2. **External GIMAN model** -- loads a pre-trained GIMAN neuro-fuzzy
       model and evaluates its performance on GIMIN-imputed vs.
       MICE-imputed data.

    Args:
        n_splits: Number of cross-validation folds.  Default: 5.
        random_state: Random seed for reproducibility.  Default: 42.
        n_bootstrap: Number of bootstrap iterations for confidence
            intervals.  Default: 1000.
    """

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        n_bootstrap: int = 1000,
    ) -> None:
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_bootstrap = n_bootstrap

    # ------------------------------------------------------------------
    # Internal classifier evaluation
    # ------------------------------------------------------------------

    def _train_evaluate_classifier(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, float]:
        """Train and cross-validate a gradient-boosted classifier.

        Args:
            features: Imputed feature matrix, shape ``(N, F)``.
            labels: Classification labels, shape ``(N,)``.

        Returns:
            Dictionary of classification metrics (mean across folds).
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            cohen_kappa_score,
            f1_score,
            roc_auc_score,
        )
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        fold_metrics: dict[str, list[float]] = {
            "accuracy": [],
            "balanced_accuracy": [],
            "f1_weighted": [],
            "cohen_kappa": [],
        }
        auroc_scores: list[float] = []

        num_classes = len(np.unique(labels))

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            clf = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=self.random_state,
                subsample=0.8,
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            fold_metrics["accuracy"].append(accuracy_score(y_test, y_pred))
            fold_metrics["balanced_accuracy"].append(
                balanced_accuracy_score(y_test, y_pred)
            )
            fold_metrics["f1_weighted"].append(
                f1_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            fold_metrics["cohen_kappa"].append(cohen_kappa_score(y_test, y_pred))

            # AUROC (binary or multi-class).
            if hasattr(clf, "predict_proba"):
                y_proba = clf.predict_proba(X_test)
                try:
                    if num_classes == 2:
                        auroc = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        auroc = roc_auc_score(
                            y_test, y_proba, multi_class="ovr", average="weighted"
                        )
                    auroc_scores.append(auroc)
                except ValueError:
                    logger.debug("AUROC computation failed on fold %d.", fold_idx)

        result: dict[str, float] = {}
        for metric_name, values in fold_metrics.items():
            result[f"{metric_name}_mean"] = float(np.mean(values))
            result[f"{metric_name}_std"] = float(np.std(values))

        if auroc_scores:
            result["auroc_mean"] = float(np.mean(auroc_scores))
            result["auroc_std"] = float(np.std(auroc_scores))

        return result

    # ------------------------------------------------------------------
    # Bootstrap confidence intervals
    # ------------------------------------------------------------------

    def _bootstrap_ci(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        metric_name: str = "accuracy",
        alpha: float = 0.05,
    ) -> dict[str, float]:
        """Compute bootstrap confidence intervals for a classification metric.

        Args:
            features: Imputed feature matrix.
            labels: Classification labels.
            metric_name: Which metric to bootstrap.  Default: ``"accuracy"``.
            alpha: Significance level for the CI.  Default: 0.05 (95% CI).

        Returns:
            Dictionary with ``"lower"``, ``"upper"``, ``"mean"`` keys.
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, balanced_accuracy_score

        metric_fn_map = {
            "accuracy": accuracy_score,
            "balanced_accuracy": balanced_accuracy_score,
        }
        metric_fn = metric_fn_map.get(metric_name, accuracy_score)

        rng = np.random.default_rng(self.random_state)
        n = len(labels)
        boot_values: list[float] = []

        for _ in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            X_boot, y_boot = features[idx], labels[idx]

            # Use a small held-out set from within the bootstrap sample.
            split = int(0.8 * n)
            X_tr, X_te = X_boot[:split], X_boot[split:]
            y_tr, y_te = y_boot[:split], y_boot[split:]

            if len(np.unique(y_tr)) < 2 or len(y_te) == 0:
                continue

            clf = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                random_state=self.random_state,
            )
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            boot_values.append(float(metric_fn(y_te, y_pred)))

        if not boot_values:
            return {"lower": float("nan"), "upper": float("nan"), "mean": float("nan")}

        boot_arr = np.array(boot_values)
        lower = float(np.percentile(boot_arr, 100 * alpha / 2))
        upper = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))
        mean = float(boot_arr.mean())

        return {"lower": lower, "upper": upper, "mean": mean}

    # ------------------------------------------------------------------
    # External GIMAN model evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_with_giman(
        imputed_data: np.ndarray,
        saa_labels: np.ndarray,
        giman_model_path: str,
    ) -> dict[str, float]:
        """Evaluate a pre-trained GIMAN model on imputed data.

        This is a placeholder for integration with the GIMAN neuro-fuzzy
        classifier.  When GIMAN is available, it loads the model from
        ``giman_model_path`` and evaluates on the provided imputed data.

        Args:
            imputed_data: Feature matrix, shape ``(N, F)``.
            saa_labels: SAA classification labels, shape ``(N,)``.
            giman_model_path: Path to the saved GIMAN model.

        Returns:
            Dictionary of classification metrics.

        Raises:
            FileNotFoundError: If the GIMAN model path does not exist.
            ImportError: If the GIMAN package is not available.
        """
        model_path = Path(giman_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"GIMAN model not found at: {model_path}")

        try:
            # Attempt to import the GIMAN evaluation module.
            # This import path will depend on the actual GIMAN package
            # structure; adjust as needed.
            from giman.evaluation import (
                evaluate_model,  # type: ignore[import-not-found]
            )

            results = evaluate_model(
                model_path=str(model_path),
                features=imputed_data,
                labels=saa_labels,
            )
            return results
        except ImportError:
            logger.warning("GIMAN package not available.  Returning empty results.")
            return {"error": "GIMAN package not installed"}

    # ------------------------------------------------------------------
    # Main evaluation entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        imputed_data: ArrayLike,
        saa_labels: ArrayLike,
        giman_model_path: str | None = None,
        mice_imputed_data: ArrayLike | None = None,
    ) -> dict[str, Any]:
        """Train/evaluate downstream SAA classifier with GIMIN-imputed data.

        Optionally compares against MICE-imputed data and/or a pre-trained
        GIMAN neuro-fuzzy model.

        Args:
            imputed_data: GIMIN-imputed feature matrix, shape ``(N, F)``.
            saa_labels: SAA classification labels, shape ``(N,)``.
            giman_model_path: Optional path to a pre-trained GIMAN model.
                If provided, the GIMAN model is also evaluated.
            mice_imputed_data: Optional MICE-imputed feature matrix for
                comparison.

        Returns:
            Results dictionary with structure::

                {
                    "gimin_classifier": { metrics },
                    "gimin_bootstrap_ci": { metric: {lower, upper, mean} },
                    "mice_classifier": { metrics },   # if mice_imputed_data
                    "giman_gimin": { metrics },        # if giman_model_path
                    "giman_mice": { metrics },         # if both
                    "comparison": { delta metrics },
                }
        """
        imputed_np = _to_numpy(imputed_data)
        labels_np = _to_numpy(saa_labels).astype(int)

        results: dict[str, Any] = {}

        # -- GIMIN-imputed classifier --
        logger.info("Evaluating downstream classifier on GIMIN-imputed data.")
        results["gimin_classifier"] = self._train_evaluate_classifier(
            imputed_np, labels_np
        )

        # Bootstrap CI for accuracy.
        results["gimin_bootstrap_ci"] = self._bootstrap_ci(
            imputed_np, labels_np, metric_name="accuracy"
        )

        # -- MICE-imputed classifier (if provided) --
        if mice_imputed_data is not None:
            mice_np = _to_numpy(mice_imputed_data)
            logger.info("Evaluating downstream classifier on MICE-imputed data.")
            results["mice_classifier"] = self._train_evaluate_classifier(
                mice_np, labels_np
            )

            # Comparison (GIMIN minus MICE).
            comparison: dict[str, float] = {}
            for key in results["gimin_classifier"]:
                if key.endswith("_mean"):
                    gimin_val = results["gimin_classifier"][key]
                    mice_val = results["mice_classifier"].get(key, 0.0)
                    comparison[f"delta_{key}"] = gimin_val - mice_val
            results["comparison"] = comparison

        # -- GIMAN model (if provided) --
        if giman_model_path is not None:
            logger.info("Evaluating GIMAN model on GIMIN-imputed data.")
            results["giman_gimin"] = self._evaluate_with_giman(
                imputed_np, labels_np, giman_model_path
            )

            if mice_imputed_data is not None:
                mice_np = _to_numpy(mice_imputed_data)
                logger.info("Evaluating GIMAN model on MICE-imputed data.")
                results["giman_mice"] = self._evaluate_with_giman(
                    mice_np, labels_np, giman_model_path
                )

        return results
