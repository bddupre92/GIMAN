#!/usr/bin/env python3
"""Phase 4 Enhanced System Analysis & Debugging
============================================

Debug and analyze the enhanced Phase 4 system to understand performance
patterns and optimize for better generalization.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_enhanced_results():
    """Analyze the enhanced Phase 4 results."""
    logger.info("🔍 Analyzing Enhanced Phase 4 Results")
    logger.info("=" * 50)

    # Load results
    results = torch.load("enhanced_phase4_results.pth", weights_only=False)

    # Extract training history
    history = results["training_history"]
    eval_results = results["evaluation_results"]
    config = results["config"]

    logger.info("📊 Training Summary:")
    logger.info(f"   Total epochs: {history['total_epochs']}")
    logger.info(f"   Best validation loss: {history['best_val_loss']:.4f}")
    logger.info(f"   Final train loss: {history['train_losses'][-1]:.4f}")
    logger.info(f"   Final val loss: {history['val_losses'][-1]:.4f}")

    logger.info("\n🎯 Final Performance:")
    logger.info(f"   Motor R²: {eval_results['motor_r2']:.4f}")
    logger.info(f"   Motor RMSE: {eval_results['motor_rmse']:.4f}")
    logger.info(f"   Cognitive AUC: {eval_results['cognitive_auc']:.4f}")

    # Analyze training curves
    logger.info("\n📈 Training Analysis:")

    # Check for overfitting
    train_loss_final = history["train_losses"][-1]
    val_loss_final = history["val_losses"][-1]
    overfitting_ratio = val_loss_final / train_loss_final

    logger.info(f"   Overfitting ratio (val/train loss): {overfitting_ratio:.2f}")

    if overfitting_ratio > 3.0:
        logger.info("   ⚠️  Significant overfitting detected!")
        logger.info("   💡 Recommendations:")
        logger.info("      - Increase dropout rate")
        logger.info("      - Reduce model complexity")
        logger.info("      - Add more regularization")
        logger.info("      - Increase training data")
    elif overfitting_ratio > 1.5:
        logger.info("   ⚠️  Moderate overfitting detected")
        logger.info("   💡 Consider increasing regularization")
    else:
        logger.info("   ✅ Training appears well-balanced")

    # Learning rate analysis
    lr_changes = []
    for i in range(1, len(history["learning_rates"])):
        if history["learning_rates"][i] != history["learning_rates"][i - 1]:
            lr_changes.append((i, history["learning_rates"][i]))

    logger.info("\n📊 Learning Rate Schedule:")
    logger.info(f"   Initial LR: {history['learning_rates'][0]:.6f}")
    logger.info(f"   Final LR: {history['learning_rates'][-1]:.6f}")
    logger.info(f"   LR changes: {len(lr_changes)}")

    if lr_changes:
        logger.info("   LR schedule events:")
        for epoch, lr in lr_changes:
            logger.info(f"      Epoch {epoch}: {lr:.6f}")

    # Create visualization
    create_training_plots(history)

    return results, history, eval_results


def create_training_plots(history):
    """Create comprehensive training plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    epochs = range(len(history["train_losses"]))

    # Loss curves
    axes[0, 0].plot(epochs, history["train_losses"], label="Train", alpha=0.8)
    axes[0, 0].plot(epochs, history["val_losses"], label="Validation", alpha=0.8)
    axes[0, 0].set_title("Total Loss Curves")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Motor loss curves
    axes[0, 1].plot(
        epochs, history["train_motor_losses"], label="Train Motor", alpha=0.8
    )
    axes[0, 1].plot(epochs, history["val_motor_losses"], label="Val Motor", alpha=0.8)
    axes[0, 1].set_title("Motor Loss Curves")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Motor Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Cognitive loss curves
    axes[0, 2].plot(
        epochs, history["train_cognitive_losses"], label="Train Cognitive", alpha=0.8
    )
    axes[0, 2].plot(
        epochs, history["val_cognitive_losses"], label="Val Cognitive", alpha=0.8
    )
    axes[0, 2].set_title("Cognitive Loss Curves")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Cognitive Loss")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Learning rate schedule
    axes[1, 0].plot(epochs, history["learning_rates"], alpha=0.8, color="orange")
    axes[1, 0].set_title("Learning Rate Schedule")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3)

    # Overfitting analysis
    overfitting_ratio = [
        v / t if t > 0 else 1
        for v, t in zip(history["val_losses"], history["train_losses"], strict=False)
    ]
    axes[1, 1].plot(epochs, overfitting_ratio, alpha=0.8, color="red")
    axes[1, 1].axhline(
        y=1.0, color="black", linestyle="--", alpha=0.5, label="Perfect Balance"
    )
    axes[1, 1].axhline(
        y=1.5, color="orange", linestyle="--", alpha=0.5, label="Moderate Overfitting"
    )
    axes[1, 1].axhline(
        y=3.0, color="red", linestyle="--", alpha=0.5, label="Severe Overfitting"
    )
    axes[1, 1].set_title("Overfitting Analysis (Val/Train Loss Ratio)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Val/Train Loss Ratio")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Loss difference analysis
    loss_diff = [
        v - t
        for v, t in zip(history["val_losses"], history["train_losses"], strict=False)
    ]
    axes[1, 2].plot(epochs, loss_diff, alpha=0.8, color="purple")
    axes[1, 2].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[1, 2].set_title("Generalization Gap (Val - Train Loss)")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Loss Difference")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("enhanced_phase4_training_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("📊 Training plots saved as 'enhanced_phase4_training_analysis.png'")


def recommend_improvements(history, eval_results):
    """Provide specific recommendations for improvement."""
    logger.info("\n💡 Improvement Recommendations:")
    logger.info("=" * 40)

    # Analyze final performance
    final_motor_r2 = eval_results["motor_r2"]
    final_cognitive_auc = eval_results["cognitive_auc"]

    # Training stability analysis
    train_loss_stability = np.std(history["train_losses"][-10:])
    val_loss_stability = np.std(history["val_losses"][-10:])

    logger.info("📊 Performance Analysis:")
    logger.info(f"   Motor R² = {final_motor_r2:.4f}")
    logger.info(f"   Cognitive AUC = {final_cognitive_auc:.4f}")

    # Motor prediction recommendations
    if final_motor_r2 < -50:
        logger.info(f"\n🎯 Motor Prediction Issues (R² = {final_motor_r2:.4f}):")
        logger.info("   ❌ Severely negative R² indicates predictions worse than mean")
        logger.info("   💡 Recommendations:")
        logger.info("      1. Check target normalization/scaling")
        logger.info("      2. Reduce model complexity to prevent overfitting")
        logger.info("      3. Add stronger regularization")
        logger.info("      4. Consider simpler baseline models first")
        logger.info("      5. Verify data quality and preprocessing")
    elif final_motor_r2 < 0:
        logger.info(
            f"\n⚠️  Motor Prediction Below Baseline (R² = {final_motor_r2:.4f}):"
        )
        logger.info("   💡 Model predictions worse than using mean")
        logger.info("   💡 Try: Simpler architecture, more regularization")

    # Cognitive prediction recommendations
    if final_cognitive_auc < 0.6:
        logger.info(
            f"\n🧠 Cognitive Prediction Issues (AUC = {final_cognitive_auc:.4f}):"
        )
        logger.info("   ❌ Below acceptable threshold (0.6)")
        logger.info("   💡 Recommendations:")
        logger.info("      1. Check class balance and sampling")
        logger.info("      2. Try focal loss for imbalanced classes")
        logger.info("      3. Adjust decision threshold")
        logger.info("      4. Feature selection/engineering")

    # Training stability recommendations
    logger.info("\n📈 Training Stability:")
    logger.info(f"   Train loss stability (std last 10): {train_loss_stability:.4f}")
    logger.info(f"   Val loss stability (std last 10): {val_loss_stability:.4f}")

    if val_loss_stability > 0.1:
        logger.info("   ⚠️  High validation loss instability")
        logger.info("   💡 Consider: Lower learning rate, more epochs")

    # Overfitting recommendations
    final_overfitting = history["val_losses"][-1] / history["train_losses"][-1]
    if final_overfitting > 3.0:
        logger.info("\n🔄 Overfitting Mitigation:")
        logger.info("   💡 Strong regularization needed:")
        logger.info("      - Increase dropout to 0.5-0.7")
        logger.info("      - Add L2 weight decay (1e-3 to 1e-2)")
        logger.info("      - Reduce model capacity")
        logger.info("      - Early stopping with smaller patience")
        logger.info("      - Data augmentation if applicable")

    # Architecture recommendations
    logger.info("\n🏗️  Architecture Suggestions:")
    logger.info("   For next iteration:")
    logger.info("   1. Start with simpler baseline (linear model)")
    logger.info("   2. Gradually add complexity")
    logger.info("   3. Focus on one task at a time")
    logger.info("   4. Use cross-validation for better evaluation")
    logger.info("   5. Implement ensemble of simpler models")


def create_optimized_config():
    """Create an optimized configuration based on analysis."""
    logger.info("\n⚙️  Optimized Configuration Recommendation:")

    optimized_config = {
        "embed_dim": 128,  # Reduced complexity
        "num_heads": 4,  # Fewer attention heads
        "dropout_rate": 0.5,  # Higher dropout
        "learning_rate": 0.0001,  # Lower learning rate
        "weight_decay": 1e-3,  # Stronger regularization
        "gradient_clip_value": 0.5,  # Tighter gradient clipping
        "lr_scheduler_patience": 5,  # More aggressive LR reduction
        "lr_scheduler_factor": 0.3,  # Stronger LR reduction
        "early_stopping_patience": 15,  # Earlier stopping
        "warmup_epochs": 3,  # Shorter warmup
        "batch_size": 32,  # Larger batches for stability
        "label_smoothing": 0.0,  # Remove label smoothing initially
    }

    logger.info("   Key changes for better generalization:")
    for key, value in optimized_config.items():
        logger.info(f"   {key}: {value}")

    return optimized_config


def main():
    """Main analysis function."""
    try:
        results, history, eval_results = analyze_enhanced_results()
        recommend_improvements(history, eval_results)
        optimized_config = create_optimized_config()

        logger.info("\n✅ Analysis complete!")
        logger.info(
            "📊 Check 'enhanced_phase4_training_analysis.png' for visualizations"
        )

        return results, optimized_config

    except FileNotFoundError:
        logger.error("❌ Results file not found. Run the enhanced system first.")
        return None, None


if __name__ == "__main__":
    results, config = main()
