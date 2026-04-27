"""
Quick hyperparameter tuning for GIMAN-SAA (reduced grid for faster testing).

Author: GIMAN Research Team  
Date: October 13, 2025
"""

import sys
from pathlib import Path

# Add project root and package to path
script_path = Path(__file__).resolve()
scripts_dir = script_path.parent
subphase_dir = scripts_dir.parent
phase8_dir = subphase_dir.parent
project_root = phase8_dir.parent.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(subphase_dir))

from scripts.train_giman_saa import SAATrainer
from configs.saa_config import SAAConfig
import pandas as pd
import numpy as np
from datetime import datetime


def main():
    """Quick hyperparameter tuning."""
    print("=" * 80)
    print("GIMAN-SAA QUICK HYPERPARAMETER TUNING")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test configurations
    configs_to_test = [
        {'focal_alpha': 0.70, 'knn_k': 10, 'lr': 0.001},
        {'focal_alpha': 0.70, 'knn_k': 15, 'lr': 0.001},
        {'focal_alpha': 0.80, 'knn_k': 10, 'lr': 0.001},
        {'focal_alpha': 0.80, 'knn_k': 15, 'lr': 0.001},
        {'focal_alpha': 0.75, 'knn_k': 10, 'lr': 0.002},  # Higher LR
        {'focal_alpha': 0.75, 'knn_k': 15, 'lr': 0.002},  # Higher LR
    ]
    
    results = []
    best_val_auc = 0.0
    best_config = None
    
    for idx, params in enumerate(configs_to_test, 1):
        print("\n" + "=" * 80)
        print(f"Configuration {idx}/{len(configs_to_test)}")
        print("=" * 80)
        print(f"Focal Alpha: {params['focal_alpha']}")
        print(f"k-NN k: {params['knn_k']}")
        print(f"Learning Rate: {params['lr']}")
        
        # Create config
        config = SAAConfig()
        config.FOCAL_ALPHA = params['focal_alpha']
        config.KNN_K = params['knn_k']
        config.LEARNING_RATE = params['lr']
        config.MAX_EPOCHS = 100
        config.PATIENCE = 20
        
        try:
            # Train
            trainer = SAATrainer(config)
            trainer.load_data()
            X, y, patient_ids, feature_cols = trainer.prepare_features()
            trainer.train_data, trainer.val_data, trainer.test_data = trainer.split_data(X, y, patient_ids)
            trainer.initialize_model(num_features=X.shape[1])
            trainer.initialize_optimizer()
            trainer.train()
            
            # Evaluate (returns tuple: results, probs, labels)
            test_results, _, _ = trainer.evaluate(trainer.test_data, split_name="Test")
            
            result = {
                **params,
                'val_auc': trainer.best_val_auc,
                'test_auc': test_results['auc'],
                'test_acc': test_results['accuracy'],
                'saa_pos_recall': test_results.get('saa_pos_recall', 0.0),
                'saa_neg_recall': test_results.get('saa_neg_recall', 0.0),
            }
            
            results.append(result)
            
            if result['val_auc'] > best_val_auc:
                best_val_auc = result['val_auc']
                best_config = result.copy()
                print(f"\n🎉 NEW BEST! Val AUC: {best_val_auc:.4f}")
                
        except Exception as e:
            print(f"\n❌ Failed: {e}")
            continue
    
    # Print results
    print("\n" + "=" * 80)
    print("TUNING COMPLETE")
    print("=" * 80)
    
    if best_config:
        print("\n🏆 BEST CONFIGURATION:")
        print(f"   Focal Alpha: {best_config['focal_alpha']}")
        print(f"   k-NN k: {best_config['knn_k']}")
        print(f"   Learning Rate: {best_config['lr']}")
        print(f"   Val AUC: {best_config['val_auc']:.4f}")
        print(f"   Test AUC: {best_config['test_auc']:.4f}")
        print(f"   SAA+ Recall: {best_config['saa_pos_recall']:.4f}")
        
        # Save results
        df = pd.DataFrame(results)
        output_path = config.RESULTS_DIR / "quick_tuning_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    return best_config


if __name__ == "__main__":
    main()
