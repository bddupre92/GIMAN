import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import logging
import json

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Import GIMAN components
sys.path.append(str(project_root / "archive/development/phase8/subphase8_2_dynamic_endpoints"))
from train_final_giman_survival import GIMANSurvivalGAT

from archive.development.phase9.neuro_fuzzy import NeuroFuzzyGIMAN

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_pretrained_gat(model, checkpoint_path):
    if not checkpoint_path.exists():
        logging.warning(f"Checkpoint not found at {checkpoint_path}. Training from scratch.")
        return model
    
    logging.info(f"Loading pre-trained GAT from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # The checkpoint has 'model_state_dict' which contains keys like 'convs.0.att_src', 'risk_head.0.weight'
    # We need to filter out 'risk_head' as NeuroFuzzyGIMAN uses 'gat_encoder' which is the GIMANSurvivalGAT instance.
    # If 'model' is NeuroFuzzyGIMAN, model.gat_encoder is the GAT.
    
    gat_state_dict = checkpoint['model_state_dict']
    
    # Load into the gat_encoder submodule
    # We might need to handle strict=False if we modified the GAT class, but it should be same.
    try:
        model.gat_encoder.load_state_dict(gat_state_dict, strict=False)
        logging.info("Successfully loaded GAT weights.")
    except Exception as e:
        logging.error(f"Error loading weights: {e}")
        
    return model

def train_neuro_fuzzy_full():
    logging.info("🚀 Starting Phase 9: Full Scale Training & Tuning")
    
    # 1. Load Data
    data_path = project_root / "data/03_prodromal/final_pyg_data/train_data.pt"
    if not data_path.exists():
        logging.error(f"Data not found at {data_path}")
        return
        
    dataset = torch.load(data_path, weights_only=False)
    logging.info(f"Loaded patient graph: {dataset.num_nodes} nodes")
    
    # Create Train/Test Masks (Stratified if possible, but random for now)
    num_nodes = dataset.num_nodes
    indices = torch.randperm(num_nodes)
    train_size = int(0.8 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[indices[train_size:]] = True
    
    dataset.train_mask = train_mask
    dataset.test_mask = test_mask
    
    # 2. Initialize Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = dataset.to(device)
    
    # Check Class Balance
    targets = dataset.event.long()
    train_targets = targets[train_mask]
    pos_count = (train_targets == 1).sum().item()
    neg_count = (train_targets == 0).sum().item()
    logging.info(f"Class Balance (Train): Positive={pos_count}, Negative={neg_count}, Ratio={pos_count/(pos_count+neg_count):.2f}")
    
    # Calculate class weights for loss
    class_weights = torch.tensor([1.0, neg_count/pos_count], device=device)
    logging.info(f"Using Class Weights: {class_weights}")
    
    in_features = dataset.x.shape[1]
    gat_encoder = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128)
    
    # Initialize Neuro-Fuzzy Model
    # Hyperparameters
    NUM_RULES = 32 
    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 1e-4
    EPOCHS = 200
    
    model = NeuroFuzzyGIMAN(gat_encoder, num_classes=2, num_rules=NUM_RULES).to(device)
    
    # Load Pre-trained Phase 8 Weights
    checkpoint_path = project_root / "outputs/phase8_2_final_training/giman_survival_final.pth"
    model = load_pretrained_gat(model, checkpoint_path)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 3. Training Loop
    best_auc = 0.0
    best_model_state = None
    
    logging.info(f"Training for {EPOCHS} epochs on {device}...")
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass on full graph
        logits, weights = model(dataset)
        
        # Target: SAA status (proxy using 'event' for now)
        targets = dataset.event.long()
        
        # Compute loss only on training nodes
        loss = criterion(logits[train_mask], targets[train_mask])
        
        # Entropy regularization
        entropy = -torch.sum(weights[train_mask] * torch.log(weights[train_mask] + 1e-6), dim=1).mean()
        loss += 0.01 * entropy
        
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            logits, _ = model(dataset)
            probs = F.softmax(logits, dim=1)[:, 1]
            
            # Evaluate on test nodes
            test_probs = probs[test_mask].cpu().numpy()
            test_targets = targets[test_mask].cpu().numpy()
            
            try:
                auc = roc_auc_score(test_targets, test_probs)
            except ValueError:
                auc = 0.5 # Handle single class case
                
        scheduler.step(auc)
        
        if auc > best_auc:
            best_auc = auc
            best_model_state = model.state_dict().copy()
            
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}, Test AUC: {auc:.4f}")
            
    # 4. Final Evaluation
    logging.info(f"\n🏆 Training Complete. Best AUC: {best_auc:.4f}")
    
    # Save Best Model
    save_dir = project_root / "outputs/phase9_neuro_fuzzy"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_model_state, save_dir / "neuro_fuzzy_best.pth")
    logging.info(f"Saved best model to {save_dir / 'neuro_fuzzy_best.pth'}")
    
    # 5. Rule Extraction (Interpretability)
    model.load_state_dict(best_model_state)
    logging.info("\n🔍 Extracting Top Fuzzy Rules...")
    
    # Analyze consequent weights to see which rules contribute most to positive class
    # consequent_weight: (num_rules, hidden_dim) -> we need to see effect on output_layer
    # output_layer: (num_rules, 2)
    
    # Rule importance = |weight_to_positive_class - weight_to_negative_class|
    rule_importance = model.output_layer.weight[1] - model.output_layer.weight[0] # (num_rules,)
    top_rules = torch.argsort(rule_importance, descending=True)[:5]
    
    for i, rule_idx in enumerate(top_rules):
        logging.info(f"   Rule {rule_idx.item()}: Importance Score {rule_importance[rule_idx].item():.4f}")
        # We could further analyze the antecedent centers (mu) for this rule to say "High Age", etc.

if __name__ == "__main__":
    train_neuro_fuzzy_full()
