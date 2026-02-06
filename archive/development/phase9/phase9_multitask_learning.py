import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import logging
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

sys.path.append(str(project_root / "archive/development/phase8/subphase8_2_dynamic_endpoints"))
from train_final_giman_survival import GIMANSurvivalGAT, cox_partial_likelihood_loss, concordance_index

from archive.development.phase9.neuro_fuzzy import MultiTaskNeuroFuzzyGIMAN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_pretrained_gat(model, checkpoint_path):
    if not checkpoint_path.exists():
        logging.warning(f"Checkpoint not found at {checkpoint_path}. Training from scratch.")
        return model
    
    logging.info(f"Loading pre-trained GAT from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    gat_state_dict = checkpoint['model_state_dict']
    try:
        model.gat_encoder.load_state_dict(gat_state_dict, strict=False)
        logging.info("Successfully loaded GAT weights.")
    except Exception as e:
        logging.error(f"Error loading weights: {e}")
    return model

def train_multitask():
    logging.info("🚀 Starting Phase 9: Multi-Task Learning Verification")
    
    # 1. Load Data
    data_path = project_root / "data/03_prodromal/final_pyg_data/train_data.pt"
    dataset = torch.load(data_path, weights_only=False)
    
    # Create Train/Test Masks
    num_nodes = dataset.num_nodes
    indices = torch.randperm(num_nodes)
    train_size = int(0.8 * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[indices[train_size:]] = True
    dataset.train_mask = train_mask
    dataset.test_mask = test_mask
    
    # 2. Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = dataset.to(device)
    
    in_features = dataset.x.shape[1]
    gat_encoder = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128)
    
    model = MultiTaskNeuroFuzzyGIMAN(gat_encoder, num_classes=2, num_rules=32).to(device)
    
    # Load Pre-trained Weights (Optional, but good for stability)
    checkpoint_path = project_root / "outputs/phase8_2_final_training/giman_survival_final.pth"
    model = load_pretrained_gat(model, checkpoint_path)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    # Class weights for SAA
    targets = dataset.event.long()
    train_targets = targets[train_mask]
    pos_count = (train_targets == 1).sum().item()
    neg_count = (train_targets == 0).sum().item()
    class_weights = torch.tensor([1.0, neg_count/pos_count], device=device)
    classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 3. Training Loop
    EPOCHS = 100
    logging.info(f"Training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Forward
        logits, risk_scores, weights = model(dataset)
        
        # Task A: Survival Loss (Cox)
        # Cox loss needs to be computed on the *batch* (or full train set here)
        # We only compute loss on training nodes
        # Cox loss function expects sorted input usually, but our implementation handles it?
        # Let's check `cox_partial_likelihood_loss` implementation in `train_final_giman_survival.py`.
        # It sorts inside. So we just pass the masked tensors.
        
        surv_loss = cox_partial_likelihood_loss(
            risk_scores[train_mask], 
            dataset.time[train_mask], 
            dataset.event[train_mask]
        )
        
        # Task B: Classification Loss (SAA)
        class_loss = classification_criterion(logits[train_mask], targets[train_mask])
        
        # Combined Loss
        # We weight them. Survival is primary (Phase 8 baseline).
        # SAA is auxiliary.
        loss = surv_loss + 1.0 * class_loss
        
        loss.backward()
        optimizer.step()
        
        # Evaluation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits, risk_scores, _ = model(dataset)
                
                # SAA AUC
                probs = F.softmax(logits, dim=1)[:, 1]
                test_probs = probs[test_mask].cpu().numpy()
                test_targets = targets[test_mask].cpu().numpy()
                try:
                    auc = roc_auc_score(test_targets, test_probs)
                except:
                    auc = 0.5
                
                # Survival C-index
                risk_np = risk_scores[test_mask].cpu().numpy()
                time_np = dataset.time[test_mask].cpu().numpy()
                event_np = dataset.event[test_mask].cpu().numpy()
                c_index = concordance_index(risk_np, time_np, event_np)
                
            logging.info(f"Epoch {epoch+1}: Total Loss={loss.item():.4f} | Surv Loss={surv_loss.item():.4f} | Class Loss={class_loss.item():.4f}")
            logging.info(f"          Test SAA AUC={auc:.4f} | Test C-index={c_index:.4f}")

    logging.info("\n✅ Multi-Task Verification Complete")
    logging.info(f"Final SAA AUC: {auc:.4f}")
    logging.info(f"Final C-index: {c_index:.4f}")

if __name__ == "__main__":
    train_multitask()
