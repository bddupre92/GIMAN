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

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Import GIMAN components
# We need the GAT model definition. 
# Assuming it's in archive/development/phase8/subphase8_2_dynamic_endpoints/train_final_giman_survival.py
# We might need to import it dynamically or copy the class definition if it's not in a shared module.
# Let's try to import it from the file directly using importlib or adding the path.
sys.path.append(str(project_root / "archive/development/phase8/subphase8_2_dynamic_endpoints"))
from train_final_giman_survival import GIMANSurvivalGAT

from archive.development.phase9.neuro_fuzzy import NeuroFuzzyGIMAN

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_neuro_fuzzy():
    logging.info("🚀 Starting Phase 9: Neuro-Fuzzy Enhancement")
    
    # 1. Load Data
    # We'll use the same data as Phase 8
    data_path = project_root / "data/03_prodromal/final_pyg_data/train_data.pt"
    if not data_path.exists():
        logging.error(f"Data not found at {data_path}")
        return
        
    # 1. Load Data
    # The dataset is a single large graph where nodes are patients
    dataset = torch.load(data_path, weights_only=False)
    logging.info(f"Loaded patient graph: {dataset}")
    logging.info(f"Nodes: {dataset.num_nodes}, Edges: {dataset.num_edges}")
    
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
    
    # 2. Initialize Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = dataset.to(device)
    
    in_features = dataset.x.shape[1]
    gat_encoder = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128)
    
    # Initialize Neuro-Fuzzy Model
    model = NeuroFuzzyGIMAN(gat_encoder, num_classes=2, num_rules=16).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training Loop
    epochs = 50
    logging.info(f"Training for {epochs} epochs on {device}...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass on full graph
        logits, weights = model(dataset)
        
        # Target: SAA status (proxy using 'event' for now)
        # dataset.event is (num_nodes,)
        targets = dataset.event.long()
        
        # Compute loss only on training nodes
        loss = criterion(logits[train_mask], targets[train_mask])
        
        # Entropy regularization
        entropy = -torch.sum(weights[train_mask] * torch.log(weights[train_mask] + 1e-6), dim=1).mean()
        loss += 0.01 * entropy
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
    # 4. Evaluation
    model.eval()
    with torch.no_grad():
        logits, _ = model(dataset)
        probs = F.softmax(logits, dim=1)[:, 1]
        targets = dataset.event.long()
        
        # Evaluate on test nodes
        test_probs = probs[test_mask].cpu().numpy()
        test_targets = targets[test_mask].cpu().numpy()
            
    auc = roc_auc_score(test_targets, test_probs)
    acc = accuracy_score(test_targets, test_probs > 0.5)
    
    logging.info(f"✅ Phase 9 Verification Complete")
    logging.info(f"   - SAA AUC: {auc:.4f}")
    logging.info(f"   - Accuracy: {acc:.4f}")
    
    # 5. Rule Extraction (Interpretability)
    logging.info("🔍 Extracting Fuzzy Rules...")
    logging.info(f"   - Learned {model.rule_layer.num_rules} fuzzy rules.")
    logging.info(f"   - Example Rule 1 Weight mean: {model.consequent_weight[0].mean().item():.4f}")

if __name__ == "__main__":
    train_neuro_fuzzy()
