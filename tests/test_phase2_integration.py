"""Integration tests for GIMAN Phase 2 components.

This module tests the integration of trainer, evaluator, and experiment tracker
components with real PPMI data to ensure the complete training pipeline works correctly.
"""

import tempfile

import pytest
from torch_geometric.data import DataLoader

from src.giman_pipeline.training.data_loaders import GIMANDataLoader
from src.giman_pipeline.training.evaluator import GIMANEvaluator
from src.giman_pipeline.training.experiment_tracker import GIMANExperimentTracker
from src.giman_pipeline.training.models import GIMANClassifier
from src.giman_pipeline.training.trainer import GIMANTrainer


class TestPhase2Integration:
    """Test suite for Phase 2 GIMAN component integration."""

    @pytest.fixture
    def ppmi_data(self):
        """Load real PPMI data for testing."""
        data_loader = GIMANDataLoader()
        data_dict = data_loader.load_ppmi_data()
        return data_dict

    @pytest.fixture
    def train_test_split(self, ppmi_data):
        """Create train/val/test split for integration testing."""
        data_loader = GIMANDataLoader()

        # Create graph data
        graph_data = data_loader.create_graph_data(ppmi_data, similarity_threshold=0.7)

        # Split data
        train_data, val_data, test_data = data_loader.create_train_test_split(
            graph_data, test_size=0.2, val_size=0.2
        )

        return train_data, val_data, test_data

    @pytest.fixture
    def data_loaders(self, train_test_split):
        """Create PyTorch DataLoaders."""
        train_data, val_data, test_data = train_test_split

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

        return train_loader, val_loader, test_loader

    @pytest.fixture
    def model(self, data_loaders):
        """Create GIMAN model for testing."""
        train_loader, _, _ = data_loaders
        input_dim = next(iter(train_loader)).x.size(1)

        return GIMANClassifier(
            input_dim=input_dim,
            hidden_dims=[32, 64, 32],
            output_dim=2,
            dropout_rate=0.3,
        )

    def test_trainer_integration(self, model, data_loaders):
        """Test GIMANTrainer integration with real data."""
        train_loader, val_loader, test_loader = data_loaders

        # Create trainer
        trainer = GIMANTrainer(
            model=model,
            learning_rate=0.001,
            max_epochs=5,  # Short for testing
            patience=3,
        )

        # Test training
        history = trainer.train(train_loader, val_loader, verbose=False)

        # Verify training completed
        assert len(history) > 0
        assert all("train_loss" in epoch for epoch in history)
        assert all("val_loss" in epoch for epoch in history)
        assert all("val_accuracy" in epoch for epoch in history)

        # Test single epoch methods
        train_metrics = trainer.train_epoch(train_loader)
        assert "loss" in train_metrics
        assert "accuracy" in train_metrics

        val_metrics = trainer.validate_epoch(val_loader)
        assert "loss" in val_metrics
        assert "accuracy" in val_metrics
        assert "f1" in val_metrics
        assert "auc_roc" in val_metrics

    def test_evaluator_integration(self, model, data_loaders):
        """Test GIMANEvaluator integration with real data."""
        train_loader, val_loader, test_loader = data_loaders

        # Quick train to get meaningful evaluation
        trainer = GIMANTrainer(model=model, max_epochs=3)
        trainer.train(train_loader, val_loader, verbose=False)

        # Create evaluator
        evaluator = GIMANEvaluator(model=trainer.model)

        # Test single evaluation
        results = evaluator.evaluate_single(test_loader, "test")

        required_metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        for metric in required_metrics:
            assert metric in results
            assert 0 <= results[metric] <= 1

        assert "confusion_matrix" in results
        assert "classification_report" in results
        assert results["n_samples"] > 0

        # Test report generation
        report = evaluator.generate_report(results)
        assert "GIMAN MODEL EVALUATION REPORT" in report
        assert "CLASSIFICATION METRICS" in report
        assert "CONFUSION MATRIX" in report

    def test_cross_validation_integration(self, data_loaders):
        """Test cross-validation with real data."""
        train_loader, val_loader, test_loader = data_loaders

        # Combine train and val for CV
        all_data = list(train_loader.dataset) + list(val_loader.dataset)

        # Create model
        input_dim = next(iter(train_loader)).x.size(1)
        model = GIMANClassifier(
            input_dim=input_dim, hidden_dims=[32, 64, 32], output_dim=2
        )

        evaluator = GIMANEvaluator(model=model)

        # Test cross-validation (small n_splits for speed)
        cv_results = evaluator.cross_validate(
            dataset=all_data,
            n_splits=3,
            max_epochs=3,  # Short for testing
            verbose=False,
        )

        # Verify CV results structure
        assert "fold_results" in cv_results
        assert "metrics_summary" in cv_results
        assert len(cv_results["fold_results"]) == 3

        # Check fold results
        for fold_result in cv_results["fold_results"]:
            assert "accuracy" in fold_result
            assert "f1" in fold_result
            assert "fold" in fold_result

        # Check summary statistics
        metrics_summary = cv_results["metrics_summary"]
        for metric in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
            assert metric in metrics_summary
            assert "mean" in metrics_summary[metric]
            assert "std" in metrics_summary[metric]

    def test_experiment_tracker_integration(self, model, data_loaders):
        """Test GIMANExperimentTracker integration."""
        train_loader, val_loader, test_loader = data_loaders

        # Create temporary directory for MLflow
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = GIMANExperimentTracker(
                experiment_name="test_integration",
                tracking_uri=f"file://{temp_dir}/mlruns",
            )

            # Create trainer
            trainer = GIMANTrainer(
                model=model,
                max_epochs=3,  # Short for testing
                patience=2,
            )

            # Test experiment logging
            run_id = tracker.log_experiment(
                trainer=trainer,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config={"test_config": "integration_test"},
                tags={"test": "phase2_integration"},
            )

            assert run_id is not None
            assert len(run_id) > 0

            # Test experiment comparison
            comparison_df = tracker.compare_experiments()
            assert not comparison_df.empty
            assert "run_id" in comparison_df.columns

    def test_hyperparameter_optimization_integration(self, data_loaders):
        """Test hyperparameter optimization integration (minimal)."""
        train_loader, val_loader, test_loader = data_loaders

        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = GIMANExperimentTracker(
                experiment_name="test_hpo", tracking_uri=f"file://{temp_dir}/mlruns"
            )

            # Test minimal hyperparameter optimization
            best_params, study = tracker.hyperparameter_optimization(
                train_loader=train_loader,
                val_loader=val_loader,
                n_trials=2,  # Very small for testing
                timeout=60,  # 1 minute limit
                optimization_metric="val_accuracy",
            )

            assert best_params is not None
            assert study is not None
            assert len(study.trials) > 0

            # Check that best_params contains expected keys
            expected_keys = [
                "learning_rate",
                "weight_decay",
                "dropout_rate",
                "hidden_dims",
            ]
            for key in expected_keys:
                assert key in best_params

    def test_end_to_end_pipeline(self, ppmi_data):
        """Test complete end-to-end Phase 2 pipeline."""
        # Data loading and preprocessing
        data_loader = GIMANDataLoader()
        graph_data = data_loader.create_graph_data(ppmi_data, similarity_threshold=0.7)
        train_data, val_data, test_data = data_loader.create_train_test_split(
            graph_data
        )

        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

        # Model creation
        input_dim = next(iter(train_loader)).x.size(1)
        model = GIMANClassifier(
            input_dim=input_dim,
            hidden_dims=[64, 128, 64],
            output_dim=2,
            dropout_rate=0.3,
        )

        # Training
        trainer = GIMANTrainer(
            model=model, learning_rate=0.001, max_epochs=10, patience=5
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Experiment tracking
            tracker = GIMANExperimentTracker(
                experiment_name="test_e2e", tracking_uri=f"file://{temp_dir}/mlruns"
            )

            # Log complete experiment
            run_id = tracker.log_experiment(
                trainer=trainer,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config={"pipeline": "end_to_end_test"},
                run_name="e2e_test",
            )

            # Evaluation
            evaluator = GIMANEvaluator(trainer.model)
            final_results = evaluator.evaluate_single(test_loader, "final_test")

            # Verify final results
            assert final_results["accuracy"] >= 0
            assert final_results["f1"] >= 0
            assert final_results["auc_roc"] >= 0

            # Generate and verify report
            report = evaluator.generate_report(final_results)
            assert "GIMAN MODEL EVALUATION REPORT" in report

            print("✅ End-to-end test completed successfully!")
            print(f"   - Run ID: {run_id}")
            print(f"   - Final accuracy: {final_results['accuracy']:.4f}")
            print(f"   - Final F1: {final_results['f1']:.4f}")
            print(f"   - Final AUC-ROC: {final_results['auc_roc']:.4f}")


if __name__ == "__main__":
    """Run integration tests manually."""
    import sys

    # Create test instance
    test_suite = TestPhase2Integration()

    try:
        print("🚀 Running Phase 2 integration tests...")

        # Load data fixtures
        ppmi_data = test_suite.ppmi_data()
        train_test_split = test_suite.train_test_split(ppmi_data)
        data_loaders = test_suite.data_loaders(train_test_split)
        model = test_suite.model(data_loaders)

        print("✅ Test fixtures loaded successfully")

        # Run individual tests
        test_suite.test_trainer_integration(model, data_loaders)
        print("✅ Trainer integration test passed")

        test_suite.test_evaluator_integration(model, data_loaders)
        print("✅ Evaluator integration test passed")

        test_suite.test_experiment_tracker_integration(model, data_loaders)
        print("✅ Experiment tracker integration test passed")

        # Run end-to-end test
        test_suite.test_end_to_end_pipeline(ppmi_data)
        print("✅ End-to-end pipeline test passed")

        print("🎉 All Phase 2 integration tests passed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        sys.exit(1)
