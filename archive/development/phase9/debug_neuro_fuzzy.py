import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

sys.path.append(str(project_root / "archive/development/phase8/subphase8_2_dynamic_endpoints"))
from train_final_giman_survival import GIMANSurvivalGAT
from archive.development.phase9.neuro_fuzzy import NeuroFuzzyGIMAN

logging.basicConfig(level=logging.INFO, format='%(message)s')

def debug_model():
    logging.info("🔍 Debugging Neuro-Fuzzy Model Internals")
    
    # 1. Load Data
    data_path = project_root / "data/03_prodromal/final_pyg_data/train_data.pt"
    dataset = torch.load(data_path, weights_only=False)
    
    device = torch.device('cpu') # Debug on CPU
    dataset = dataset.to(device)
    
    # 2. Initialize Model
    in_features = dataset.x.shape[1]
    gat_encoder = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128)
    model = NeuroFuzzyGIMAN(gat_encoder, num_classes=2, num_rules=16).to(device)
    
    # Load weights
    checkpoint_path = project_root / "outputs/phase8_2_final_training/giman_survival_final.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.gat_encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logging.info("Loaded GAT weights.")
    
    # 3. Inspect Forward Pass
    model.train()
    
    # Hook to capture intermediate features
    features_out = []
    def hook_fn(module, input, output):
        features_out.append(output)
    
    # Register hook on the last GAT layer (before fuzzy)
    # In NeuroFuzzyGIMAN.forward, we do: features = F.elu(x)
    # We can't easily hook F.elu. But we can hook the fuzzy layer input.
    model.fuzzy_layer.register_forward_hook(lambda m, i, o: logging.info(f"Fuzzy Layer Input Stats: Mean={i[0].mean():.4f}, Std={i[0].std():.4f}, Min={i[0].min():.4f}, Max={i[0].max():.4f}"))
    model.fuzzy_layer.register_forward_hook(lambda m, i, o: logging.info(f"Fuzzy Membership Stats: Mean={o.mean():.4f}, Std={o.std():.4f}, Min={o.min():.4f}, Max={o.max():.4f}"))
    
    logging.info("\n--- Forward Pass ---")
    logits, weights = model(dataset)
    
    logging.info(f"\nLogits Stats: Mean={logits.mean():.4f}, Std={logits.std():.4f}")
    logging.info(f"Weights Stats: Mean={weights.mean():.4f}, Std={weights.std():.4f}")
    
    # 4. Check Gradients
    logging.info("\n--- Backward Pass ---")
    targets = dataset.event.long()
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    loss.backward()
    
    logging.info("Gradient Stats:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            logging.info(f"{name}: Grad Mean={param.grad.mean():.4e}, Std={param.grad.std():.4e}")
        else:
            logging.info(f"{name}: No Gradient")
            
    # 5. Check Initialization of Fuzzy Parameters
    logging.info("\n--- Parameter Stats ---")
    logging.info(f"Mu: Mean={model.fuzzy_layer.mu.mean():.4f}, Std={model.fuzzy_layer.mu.std():.4f}")
    logging.info(f"Sigma: Mean={model.fuzzy_layer.sigma.mean():.4f}, Std={model.fuzzy_layer.sigma.std():.4f}")

if __name__ == "__main__":
    debug_model()
