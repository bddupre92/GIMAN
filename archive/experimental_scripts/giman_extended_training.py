#!/usr/bin/env python3
"""Extended Epoch Training for GIMAN Phase 4
=========================================

This script modifies the standard GIMAN training to use more epochs
for potentially better convergence and performance analysis.

Author: AI Assistant
Date: September 24, 2025
"""

import logging
import time

import torch

from phase3_1_real_data_integration import RealDataPhase3Integration
from phase4_unified_giman_system import run_phase4_experiment


def run_extended_epoch_experiment(
    data_integrator: RealDataPhase3Integration,
    epochs: int = 300,
    patience: int = 30,
    learning_rate: float = 1e-4,
):
    """Run GIMAN experiment with extended epochs."""
    logging.info(f"🚀 Starting extended epoch experiment with {epochs} max epochs")

    # Run the experiment with extended parameters
    start_time = time.time()

    results = run_phase4_experiment(
        data_integrator=data_integrator,
        epochs=epochs,  # Extended epochs
        lr=learning_rate,  # Learning rate
        patience=patience,  # Extended patience
    )

    training_time = time.time() - start_time

    logging.info(f"⏱️ Training completed in {training_time:.1f} seconds")
    logging.info(f"📈 Final Motor R²: {results['motor_r2']:.4f}")
    logging.info(f"🧠 Final Cognitive AUC: {results['cognitive_auc']:.4f}")

    # Add metadata to results
    results["training_time_seconds"] = training_time
    results["max_epochs"] = epochs
    results["patience"] = patience
    results["learning_rate"] = learning_rate
    results["device"] = str(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    return results


def main():
    """Main execution function for extended epoch training."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)
    logger.info("🎬 GIMAN Phase 4: Extended Epoch Training")

    # Initialize data integrator
    logger.info("🔧 Setting up data integrator...")
    data_integrator = RealDataPhase3Integration()
    data_integrator.load_and_prepare_data()
    logger.info(f"✅ Data loaded: {len(data_integrator.patient_ids)} patients")

    # Run extended training
    results = run_extended_epoch_experiment(
        data_integrator=data_integrator,
        epochs=300,  # 2x the normal epochs
        patience=30,  # Extended patience for convergence
        learning_rate=1e-4,  # Slightly lower LR for extended training
    )

    # Summary
    print("\n" + "=" * 80)
    print("🎉 GIMAN Phase 4: Extended Epoch Training Results")
    print("=" * 80)
    print(f"📊 PPMI patients: {len(data_integrator.patient_ids)}")
    print(f"⚙️ Max epochs: {results['max_epochs']} (patience: {results['patience']})")
    print(f"⏱️ Training time: {results['training_time_seconds']:.1f}s")
    print(f"📈 Motor progression R²: {results['motor_r2']:.4f}")
    print(f"🧠 Cognitive conversion AUC: {results['cognitive_auc']:.4f}")
    print(f"🔧 Learning rate: {results['learning_rate']}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
