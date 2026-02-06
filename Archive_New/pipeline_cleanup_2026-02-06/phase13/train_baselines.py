"""
Baseline Model Training for Benchmarking
Trains 6 baseline models on real PPMI data for Figure 3 comparison
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import json

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Import existing models
sys.path.append(str(project_root / "archive/development/phase8/subphase8_2_dynamic_endpoints"))
from train_final_giman_survival import GIMANSurvivalGAT

# Output directory
output_dir = project_root / "outputs/phase13_baselines"
output_dir.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load real PPMI data"""
    data_path = project_root / "data/03_prodromal/final_pyg_data/train_data.pt"
    dataset = torch.load(data_path, weights_only=False)
    
    # Create train/test split (80/20)
    n = dataset.num_nodes
    indices = np.random.permutation(n)
    train_size = int(0.8 * n)
    
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    return dataset, train_idx, test_idx

def extract_features(dataset, indices):
    """Extract features for sklearn models"""
    X = dataset.x[indices].cpu().numpy()
    y_survival = dataset.event[indices].cpu().numpy()
    time = dataset.time[indices].cpu().numpy()
    
    return X, y_survival, time

class SimpleMLPSurvival(nn.Module):
    """Simple MLP baseline"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def train_baseline_models(dataset, train_idx, test_idx):
    """Train all baseline models"""
    results = {}
    
    # Extract features
    X_train, y_train, time_train = extract_features(dataset, train_idx)
    X_test, y_test, time_test = extract_features(dataset, test_idx)
    
    print("="*80)
    print("BASELINE MODEL TRAINING")
    print("="*80)
    print(f"Training set: {len(train_idx)} patients")
    print(f"Test set: {len(test_idx)} patients")
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    # 1. Linear Cox Proportional Hazards
    print("\n1. Training Linear CPH...")
    df_train = pd.DataFrame(X_train[:, :10])  # Use first 10 features for CPH
    df_train['time'] = time_train
    df_train['event'] = y_train
    
    cph = CoxPHFitter()
    cph.fit(df_train, duration_col='time', event_col='event')
    
    df_test = pd.DataFrame(X_test[:, :10])
    df_test['time'] = time_test
    df_test['event'] = y_test
    
    risk_scores = cph.predict_partial_hazard(df_test).values
    c_index_cph = concordance_index(time_test, -risk_scores, y_test)
    results['Linear_CPH'] = {'c_index': c_index_cph, 'auc': None}
    print(f"   C-Index: {c_index_cph:.4f}")
    
    # 2. Random Forest
    print("\n2. Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_prob = rf.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, y_pred_prob)
    c_index_rf = concordance_index(time_test, y_pred_prob, y_test)
    results['Random_Forest'] = {'c_index': c_index_rf, 'auc': auc_rf}
    print(f"   AUC: {auc_rf:.4f}, C-Index: {c_index_rf:.4f}")
    
    # 3. SVM
    print("\n3. Training SVM...")
    svm = SVC(probability=True, random_state=42)
    # Subsample for speed
    subsample = min(1000, len(X_train))
    svm.fit(X_train[:subsample], y_train[:subsample])
    y_pred_prob = svm.predict_proba(X_test)[:, 1]
    auc_svm = roc_auc_score(y_test, y_pred_prob)
    c_index_svm = concordance_index(time_test, y_pred_prob, y_test)
    results['SVM'] = {'c_index': c_index_svm, 'auc': auc_svm}
    print(f"   AUC: {auc_svm:.4f}, C-Index: {c_index_svm:.4f}")
    
    # 4. Simple MLP
    print("\n4. Training Simple MLP...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp = SimpleMLPSurvival(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    for epoch in range(50):
        mlp.train()
        optimizer.zero_grad()
        outputs = mlp(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    
    mlp.eval()
    with torch.no_grad():
        y_pred_prob = torch.sigmoid(mlp(X_test_t)).cpu().numpy().flatten()
    
    auc_mlp = roc_auc_score(y_test, y_pred_prob)
    c_index_mlp = concordance_index(time_test, y_pred_prob, y_test)
    results['Simple_MLP'] = {'c_index': c_index_mlp, 'auc': auc_mlp}
    print(f"   AUC: {auc_mlp:.4f}, C-Index: {c_index_mlp:.4f}")
    
    # 5. Logistic Regression (fast baseline)
    print("\n5. Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_prob = lr.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, y_pred_prob)
    c_index_lr = concordance_index(time_test, y_pred_prob, y_test)
    results['Logistic_Regression'] = {'c_index': c_index_lr, 'auc': auc_lr}
    print(f"   AUC: {auc_lr:.4f}, C-Index: {c_index_lr:.4f}")
    
    # Save results
    results_path = output_dir / "baseline_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {results_path}")
    
    return results

def main():
    print("Loading real PPMI data...")
    dataset, train_idx, test_idx = load_data()
    
    print(f"Dataset: {dataset.num_nodes} patients, {dataset.x.shape[1]} features")
    
    results = train_baseline_models(dataset, train_idx, test_idx)
    
    print("\n" + "="*80)
    print("BASELINE TRAINING COMPLETE")
    print("="*80)
    for model, metrics in results.items():
        print(f"{model:25s} | C-Index: {metrics['c_index']:.4f} | AUC: {metrics['auc'] if metrics['auc'] else 'N/A'}")

if __name__ == "__main__":
    main()
