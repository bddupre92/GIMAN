"""
Hyperparameter Tuning for GIMAN-SAA Model

This script performs systematic hyperparameter tuning to improve SAA prediction
performance from current AUC 0.52 to target AUC 0.85.

Tuning Strategy:
1. Grid search over focal loss alpha (0.65, 0.75, 0.85)
2. Grid search over k-NN k values (5, 10, 15)
3. Grid search over learning rates (0.0005, 0.001, 0.002)
4. Increase max epochs to 200

Author: GIMAN Research Team
Date: October 13, 2025
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import itertools
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from archive.development.phase8.subphase8_3_saa_integration.scripts.train_giman_saa import SAATrainer
from archive.development.phase8.subphase8_3_saa_integration.configs.saa_config import SAAConfig


class SAAHyperparameterTuner:
    """
    Systematic hyperparameter tuning for GIMAN-SAA.
    """
    
    def __init__(self):
        """Initialize tuner."""
        self.base_config = SAAConfig()
        self.results = []
        
    def tune(self):
        """
        Run grid search over hyperparameters.
        """
        print("=" * 80)
        print("GIMAN-SAA HYPERPARAMETER TUNING")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Define hyperparameter grid
        param_grid = {
            'focal_alpha': [0.65, 0.75, 0.85],
            'knn_k': [5, 10, 15],
            'learning_rate': [0.0005, 0.001, 0.002]
        }
        
        # Calculate total combinations
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"\nTotal configurations to test: {total_combinations}")
        print(f"Estimated time: {total_combinations * 15} minutes (~{total_combinations * 15 / 60:.1f} hours)")
        
        # Grid search
        best_val_auc = 0.0
        best_config = None
        best_test_auc = 0.0
        
        config_num = 0
        for focal_alpha in param_grid['focal_alpha']:
            for knn_k in param_grid['knn_k']:
                for lr in param_grid['learning_rate']:
                    config_num += 1
                    
                    print("\n" + "=" * 80)
                    print(f"Configuration {config_num}/{total_combinations}")
                    print("=" * 80)
                    print(f"Focal Alpha: {focal_alpha}")
                    print(f"k-NN k: {knn_k}")
                    print(f"Learning Rate: {lr}")
                    
                    # Create modified config
                    config = self._create_config(focal_alpha, knn_k, lr)
                    
                    # Train model
                    try:
                        trainer = SAATrainer(config)
                        trainer.load_data()
                        trainer.prepare_features()
                        trainer.split_data()
                        trainer.initialize_model()
                        
                        # Train with modified config
                        trainer.train()
                        
                        # Evaluate
                        test_results = trainer.evaluate(trainer.test_data, split_name="Test")
                        
                        # Record results
                        result = {
                            'config_num': config_num,
                            'focal_alpha': focal_alpha,
                            'knn_k': knn_k,
                            'learning_rate': lr,
                            'val_auc': trainer.best_val_auc,
                            'test_auc': test_results['auc'],
                            'test_acc': test_results['accuracy'],
                            'test_balanced_acc': test_results['balanced_accuracy'],
                            'saa_pos_recall': test_results.get('saa_pos_recall', 0.0),
                            'saa_neg_recall': test_results.get('saa_neg_recall', 0.0),
                            'best_epoch': trainer.best_epoch
                        }
                        
                        self.results.append(result)
                        
                        # Check if best so far
                        if result['val_auc'] > best_val_auc:
                            best_val_auc = result['val_auc']
                            best_test_auc = result['test_auc']
                            best_config = result.copy()
                            
                            print(f"\n🎉 NEW BEST MODEL!")
                            print(f"   Val AUC: {best_val_auc:.4f}")
                            print(f"   Test AUC: {best_test_auc:.4f}")
                        
                        # Save intermediate results
                        self._save_results()
                        
                    except Exception as e:
                        print(f"\n❌ Configuration failed: {e}")
                        continue
        
        # Print final summary
        self._print_summary(best_config)
        
        return best_config
    
    def _create_config(self, focal_alpha, knn_k, learning_rate):
        """
        Create modified config with specific hyperparameters.
        """
        config = SAAConfig()
        
        # Modify config
        config.FOCAL_ALPHA = focal_alpha
        config.KNN_K = knn_k
        config.LEARNING_RATE = learning_rate
        config.MAX_EPOCHS = 200  # Increase from 100
        config.PATIENCE = 30  # Increase patience for longer training
        
        return config
    
    def _save_results(self):
        """Save tuning results to CSV."""
        output_dir = self.base_config.RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "hyperparameter_tuning_results.csv"
        
        df = pd.DataFrame(self.results)
        df = df.sort_values('val_auc', ascending=False)
        df.to_csv(results_file, index=False)
        
        print(f"\nResults saved to: {results_file}")
    
    def _print_summary(self, best_config):
        """Print tuning summary."""
        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING COMPLETE")
        print("=" * 80)
        
        if best_config:
            print("\n🏆 BEST CONFIGURATION:")
            print(f"   Focal Alpha: {best_config['focal_alpha']}")
            print(f"   k-NN k: {best_config['knn_k']}")
            print(f"   Learning Rate: {best_config['learning_rate']}")
            print(f"\n📊 BEST PERFORMANCE:")
            print(f"   Validation AUC: {best_config['val_auc']:.4f}")
            print(f"   Test AUC: {best_config['test_auc']:.4f}")
            print(f"   Test Accuracy: {best_config['test_acc']:.4f}")
            print(f"   Balanced Accuracy: {best_config['test_balanced_acc']:.4f}")
            print(f"   SAA+ Recall: {best_config['saa_pos_recall']:.4f}")
            print(f"   SAA- Recall: {best_config['saa_neg_recall']:.4f}")
            print(f"   Best Epoch: {best_config['best_epoch']}")
            
            # Compare to baseline
            baseline_auc = 0.5169  # From current best model
            improvement = best_config['test_auc'] - baseline_auc
            print(f"\n📈 IMPROVEMENT:")
            print(f"   Baseline Test AUC: {baseline_auc:.4f}")
            print(f"   Improved Test AUC: {best_config['test_auc']:.4f}")
            print(f"   Absolute Improvement: {improvement:+.4f}")
            print(f"   Relative Improvement: {improvement/baseline_auc*100:+.1f}%")
            
            if best_config['test_auc'] >= 0.85:
                print("\n✅ TARGET ACHIEVED! Test AUC ≥ 0.85")
            elif best_config['test_auc'] >= 0.75:
                print("\n⚠️  CLOSE TO TARGET. Test AUC ≥ 0.75 (target 0.85)")
            else:
                print("\n❌ BELOW TARGET. Test AUC < 0.75 (target 0.85)")
                print("   Consider: (1) More sophisticated architecture, (2) Feature engineering")
                print("   (3) External validation data, (4) Ensemble methods")
        else:
            print("\n❌ No successful configurations found.")
        
        # Print top 5 configurations
        print("\n" + "=" * 80)
        print("TOP 5 CONFIGURATIONS BY VALIDATION AUC")
        print("=" * 80)
        
        df = pd.DataFrame(self.results)
        df = df.sort_values('val_auc', ascending=False).head(5)
        
        for idx, row in df.iterrows():
            print(f"\n{row['config_num']}. Val AUC: {row['val_auc']:.4f}, Test AUC: {row['test_auc']:.4f}")
            print(f"   α={row['focal_alpha']}, k={row['knn_k']}, lr={row['learning_rate']}")


def main():
    """Main execution."""
    tuner = SAAHyperparameterTuner()
    best_config = tuner.tune()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    
    if best_config and best_config['test_auc'] >= 0.75:
        print("\n1. Re-train final model with best hyperparameters")
        print("2. Perform 5-fold cross-validation for robust evaluation")
        print("3. Feature importance analysis via attention weights")
        print("4. Clinical subgroup analyses")
        print("5. Generate Phase 8.3 completion report")
    elif best_config:
        print("\n1. Consider more sophisticated approaches:")
        print("   - Ensemble methods (multiple models)")
        print("   - Different graph construction (e.g., learned similarity)")
        print("   - Feature engineering (interaction terms)")
        print("   - Transfer learning from pre-trained models")
        print("\n2. OR proceed to Subphase 8.4 (VAE Heterogeneity Analysis)")
        print("   - Current SAA model can be used for biological validation")
        print("   - VAE may provide insights for improving SAA prediction")
    else:
        print("\n⚠️  Tuning failed. Debug issues and retry.")


if __name__ == "__main__":
    main()
