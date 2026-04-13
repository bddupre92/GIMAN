"""GIMAN Experiment Tracking - Phase 2.

This module provides comprehensive experiment tracking and management capabilities
using MLflow for systematic hyperparameter optimization, model comparison,
and reproducible research for the GIMAN Parkinson's Disease classification pipeline.

Features:
- MLflow experiment tracking and logging
- Hyperparameter optimization with Optuna
- Model artifact management
- Automated experiment comparison
- Reproducible experiment configuration
- Performance visualization and analysis
"""

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import optuna
import pandas as pd
from optuna.integration.mlflow import MLflowCallback
from torch_geometric.data import DataLoader

from .evaluator import GIMANEvaluator
from .models import GIMANClassifier
from .trainer import GIMANTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GIMANExperimentTracker:
    """Comprehensive experiment tracking system for GIMAN models.

    This class integrates MLflow for experiment tracking, Optuna for hyperparameter
    optimization, and provides utilities for reproducible research and model
    comparison in Parkinson's Disease classification tasks.

    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking server URI (default: local file store)
        artifact_root: Root directory for MLflow artifacts
    """

    def __init__(
        self,
        experiment_name: str = "giman_parkinson_classification",
        tracking_uri: str | None = None,
        artifact_root: str | None = None,
    ):
        """Initialize GIMAN experiment tracker."""
        self.experiment_name = experiment_name

        # Set up MLflow tracking
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Default to local file store
            mlflow_dir = Path("mlruns")
            mlflow_dir.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")

        # Set experiment
        mlflow.set_experiment(experiment_name)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)

        logger.info("🧪 GIMAN Experiment Tracker initialized")
        logger.info(f"   - Experiment: {experiment_name}")
        logger.info(f"   - Tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"   - Experiment ID: {self.experiment.experiment_id}")

    def log_experiment(
        self,
        trainer: GIMANTrainer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
        config: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        run_name: str | None = None,
    ) -> str:
        """Log a complete GIMAN training experiment."""
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logger.info(f"🚀 Starting experiment run: {run_id}")

            # Log tags
            if tags:
                mlflow.set_tags(tags)

            # Log model configuration
            model_info = trainer.model.get_model_info()
            mlflow.log_params(model_info)

            # Log training configuration
            training_config = {
                "learning_rate": trainer.learning_rate,
                "weight_decay": trainer.weight_decay,
                "max_epochs": trainer.max_epochs,
                "patience": trainer.patience,
                "device": str(trainer.device),
                "optimizer": type(trainer.optimizer).__name__,
                "scheduler": type(trainer.scheduler).__name__
                if trainer.scheduler
                else "None",
            }
            mlflow.log_params(training_config)

            # Log additional config
            if config:
                mlflow.log_params(config)

            # Log dataset info
            dataset_info = {
                "train_size": len(train_loader.dataset),
                "val_size": len(val_loader.dataset),
                "batch_size": train_loader.batch_size,
            }
            if test_loader:
                dataset_info["test_size"] = len(test_loader.dataset)
            mlflow.log_params(dataset_info)

            # Train model with MLflow logging
            history = self._train_with_logging(trainer, train_loader, val_loader)

            # Evaluate and log results
            evaluator = GIMANEvaluator(trainer.model, device=trainer.device)

            # Validation results
            val_results = evaluator.evaluate_single(val_loader, "validation")
            self._log_evaluation_results(val_results, prefix="val")

            # Test results (if available)
            if test_loader:
                test_results = evaluator.evaluate_single(test_loader, "test")
                self._log_evaluation_results(test_results, prefix="test")

            # Log model artifact
            mlflow.pytorch.log_model(
                trainer.model,
                "model",
                extra_files=[str(Path(__file__).parent / "models.py")],
            )

            # Log training history
            history_df = pd.DataFrame(history)
            history_df.to_csv("training_history.csv", index=False)
            mlflow.log_artifact("training_history.csv")

            logger.info(f"✅ Experiment logged successfully: {run_id}")
            return run_id

    def _train_with_logging(
        self, trainer: GIMANTrainer, train_loader: DataLoader, val_loader: DataLoader
    ) -> list[dict[str, float]]:
        """Train model with MLflow metric logging."""
        history = []

        for epoch in range(trainer.max_epochs):
            # Training step
            train_metrics = trainer.train_epoch(train_loader)

            # Validation step
            val_metrics = trainer.validate_epoch(val_loader)

            # Combine metrics
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_auc_roc": val_metrics["auc_roc"],
            }

            # Log to MLflow
            mlflow.log_metrics(epoch_metrics, step=epoch)

            # Store in history
            history.append(epoch_metrics)

            # Early stopping check
            if trainer.early_stopping:
                if trainer.early_stopping.early_stop:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        return history

    def _log_evaluation_results(self, results: dict[str, Any], prefix: str = ""):
        """Log evaluation results to MLflow."""
        metrics_to_log = {
            f"{prefix}_accuracy": results["accuracy"],
            f"{prefix}_precision": results["precision"],
            f"{prefix}_recall": results["recall"],
            f"{prefix}_f1": results["f1"],
            f"{prefix}_auc_roc": results["auc_roc"],
            f"{prefix}_n_samples": results["n_samples"],
        }
        mlflow.log_metrics(metrics_to_log)

        # Log confusion matrix as artifact
        cm = results["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            columns=["Predicted_HC", "Predicted_PD"],
            index=["Actual_HC", "Actual_PD"],
        )
        cm_path = f"{prefix}_confusion_matrix.csv"
        cm_df.to_csv(cm_path)
        mlflow.log_artifact(cm_path)

    def hyperparameter_optimization(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
        n_trials: int = 50,
        timeout: int | None = None,
        study_name: str | None = None,
        optimization_metric: str = "val_f1",
    ) -> tuple[dict[str, Any], optuna.Study]:
        """Perform hyperparameter optimization with Optuna and MLflow."""
        study_name = (
            study_name
            or f"giman_optimization_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Set up MLflow callback for Optuna
        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(), metric_name=optimization_metric
        )

        # Create study
        study = optuna.create_study(direction="maximize", study_name=study_name)
        study.set_user_attr("experiment_name", self.experiment_name)

        def objective(trial):
            """Objective function for hyperparameter optimization."""
            # Sample hyperparameters
            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1e-2, log=True
                ),
                "weight_decay": trial.suggest_float(
                    "weight_decay", 1e-6, 1e-3, log=True
                ),
                "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.7),
                "hidden_dims": trial.suggest_categorical(
                    "hidden_dims",
                    [[32, 64, 32], [64, 128, 64], [128, 256, 128], [64, 128, 256, 128]],
                ),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                "max_epochs": trial.suggest_int("max_epochs", 50, 200),
                "patience": trial.suggest_int("patience", 10, 30),
            }

            # Create model with suggested hyperparameters
            input_dim = next(iter(train_loader)).x.size(1)
            model = GIMANClassifier(
                input_dim=input_dim,
                hidden_dims=params["hidden_dims"],
                output_dim=2,
                dropout_rate=params["dropout_rate"],
            )

            # Create trainer with suggested hyperparameters
            trainer = GIMANTrainer(
                model=model,
                learning_rate=params["learning_rate"],
                weight_decay=params["weight_decay"],
                max_epochs=params["max_epochs"],
                patience=params["patience"],
            )

            # Create data loaders with suggested batch size
            trial_train_loader = DataLoader(
                train_loader.dataset, batch_size=params["batch_size"], shuffle=True
            )
            trial_val_loader = DataLoader(
                val_loader.dataset, batch_size=params["batch_size"], shuffle=False
            )

            # Train model
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.set_tag("trial_number", trial.number)

                try:
                    history = self._train_with_logging(
                        trainer, trial_train_loader, trial_val_loader
                    )

                    # Evaluate
                    evaluator = GIMANEvaluator(trainer.model)
                    val_results = evaluator.evaluate_single(
                        trial_val_loader, "validation"
                    )

                    self._log_evaluation_results(val_results, prefix="val")

                    # Return optimization metric
                    metric_value = val_results[optimization_metric.replace("val_", "")]
                    mlflow.log_metric("optimization_metric", metric_value)

                    return metric_value

                except Exception as e:
                    logger.error(f"Trial {trial.number} failed: {str(e)}")
                    return 0.0  # Return worst possible score for failed trials

        # Run optimization
        logger.info("🔍 Starting hyperparameter optimization")
        logger.info(f"   - Study: {study_name}")
        logger.info(f"   - Trials: {n_trials}")
        logger.info(f"   - Optimization metric: {optimization_metric}")

        study.optimize(
            objective, n_trials=n_trials, timeout=timeout, callbacks=[mlflow_callback]
        )

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value

        logger.info("🏆 Optimization complete!")
        logger.info(f"   - Best {optimization_metric}: {best_value:.4f}")
        logger.info(f"   - Best parameters: {best_params}")

        # Train final model with best parameters
        logger.info("🎯 Training final model with best parameters")
        final_model = self._train_final_model(
            best_params, train_loader, val_loader, test_loader
        )

        return best_params, study

    def _train_final_model(
        self,
        best_params: dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
    ) -> GIMANClassifier:
        """Train final model with best hyperparameters."""
        with mlflow.start_run(run_name="best_model_final_training"):
            # Create model with best parameters
            input_dim = next(iter(train_loader)).x.size(1)
            model = GIMANClassifier(
                input_dim=input_dim,
                hidden_dims=best_params["hidden_dims"],
                output_dim=2,
                dropout_rate=best_params["dropout_rate"],
            )

            # Create trainer with best parameters
            trainer = GIMANTrainer(
                model=model,
                learning_rate=best_params["learning_rate"],
                weight_decay=best_params["weight_decay"],
                max_epochs=best_params["max_epochs"],
                patience=best_params["patience"],
            )

            # Log best parameters
            mlflow.log_params(best_params)
            mlflow.set_tag("model_type", "best_hyperparameters")

            # Create data loaders with best batch size
            final_train_loader = DataLoader(
                train_loader.dataset, batch_size=best_params["batch_size"], shuffle=True
            )
            final_val_loader = DataLoader(
                val_loader.dataset, batch_size=best_params["batch_size"], shuffle=False
            )

            # Train final model
            history = self._train_with_logging(
                trainer, final_train_loader, final_val_loader
            )

            # Comprehensive evaluation
            evaluator = GIMANEvaluator(trainer.model)

            val_results = evaluator.evaluate_single(final_val_loader, "validation")
            self._log_evaluation_results(val_results, prefix="val")

            if test_loader:
                final_test_loader = DataLoader(
                    test_loader.dataset,
                    batch_size=best_params["batch_size"],
                    shuffle=False,
                )
                test_results = evaluator.evaluate_single(final_test_loader, "test")
                self._log_evaluation_results(test_results, prefix="test")

            # Log final model
            mlflow.pytorch.log_model(trainer.model, "best_model")

            return trainer.model

    def compare_experiments(
        self, experiment_names: list[str] | None = None, metric: str = "val_f1"
    ) -> pd.DataFrame:
        """Compare experiments and return results DataFrame."""
        if experiment_names is None:
            experiment_names = [self.experiment_name]

        all_runs = []

        for exp_name in experiment_names:
            experiment = mlflow.get_experiment_by_name(exp_name)
            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=[f"metrics.{metric} DESC"],
                )
                runs["experiment_name"] = exp_name
                all_runs.append(runs)

        if all_runs:
            comparison_df = pd.concat(all_runs, ignore_index=True)
            logger.info(f"📊 Experiment comparison complete: {len(comparison_df)} runs")
            return comparison_df
        else:
            logger.warning("No experiments found for comparison")
            return pd.DataFrame()

    def export_best_model(
        self, output_dir: Path, metric: str = "val_f1", export_format: str = "pytorch"
    ) -> Path:
        """Export the best performing model."""
        # Find best run
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )

        if runs.empty:
            raise ValueError("No runs found in experiment")

        best_run = runs.iloc[0]
        run_id = best_run.run_id

        logger.info(f"🏆 Exporting best model (Run ID: {run_id})")
        logger.info(f"   - Best {metric}: {best_run[f'metrics.{metric}']:.4f}")

        # Download model artifact
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="model", dst_path=str(output_dir)
        )

        logger.info(f"💾 Model exported to: {model_path}")

        # Export run metadata
        metadata = {
            "run_id": run_id,
            "experiment_name": self.experiment_name,
            "best_metric": metric,
            "best_value": float(best_run[f"metrics.{metric}"]),
            "parameters": {
                k.replace("params.", ""): v
                for k, v in best_run.items()
                if k.startswith("params.")
            },
            "metrics": {
                k.replace("metrics.", ""): v
                for k, v in best_run.items()
                if k.startswith("metrics.")
            },
            "export_timestamp": pd.Timestamp.now().isoformat(),
        }

        metadata_path = output_dir / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"📄 Metadata exported to: {metadata_path}")

        return Path(model_path)
