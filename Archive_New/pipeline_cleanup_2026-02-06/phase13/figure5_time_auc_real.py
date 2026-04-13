"""
Figure 5: Time-Dependent AUC (Real Metrics)
Loads trained models and computes actual time-dependent AUC
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from lifelines.utils import concordance_index

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
sys.path.append(str(project_root / "archive/development/phase8/subphase8_2_dynamic_endpoints"))
sys.path.append(str(project_root / "archive/development/phase9"))

output_dir = project_root / "visualizations/publication"
output_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

# Import model architectures
from train_final_giman_survival import GIMANSurvivalGAT
from neuro_fuzzy import NeuroFuzzyGIMAN

def load_models_and_data():
    """Load trained models and test data"""
    print("Loading data and models...")
    
    # Load data
    data_path = project_root / "data/03_prodromal/final_pyg_data/train_data.pt"
    dataset = torch.load(data_path, weights_only=False)
    
    # Create train/test split (same as training)
    n = dataset.num_nodes
    indices = np.random.seed(42)
    indices = np.random.permutation(n)
    train_size = int(0.8 * n)
    test_idx = indices[train_size:]
    
    # Extract test data
    test_data = dataset.clone()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Phase 8 model (GIMAN Crisp - Survival)
    phase8_model_path = project_root / "outputs/phase8_2_training/best_survival_model.pt"
    if phase8_model_path.exists():
        print(f"Loading Phase 8 model from {phase8_model_path}")
        giman_model = GIMANSurvivalGAT(
            in_channels=dataset.x.shape[1],
            hidden_channels=128,
            num_heads=8,
            dropout=0.3
        ).to(device)
        checkpoint = torch.load(phase8_model_path, map_location=device, weights_only=False)
        giman_model.load_state_dict(checkpoint['model_state_dict'])
        giman_model.eval()
    else:
        print(f"⚠️  Phase 8 model not found at {phase8_model_path}")
        giman_model = None
    
    # Load Phase 9 model (Neuro-Fuzzy SAA) - we'll use this for comparison
    phase9_model_path = project_root / "outputs/phase8_3_saa/models/best_giman_saa_model.pt"
    if phase9_model_path.exists():
        print(f"Loading Phase 9 model from {phase9_model_path}")
        neurofuzzy_model = NeuroFuzzyGIMAN(
            in_channels=dataset.x.shape[1],
            hidden_channels=128,
            num_heads=8,
            num_rules=10,
            dropout=0.3
        ).to(device)
        checkpoint = torch.load(phase9_model_path, map_location=device, weights_only=False)
        neurofuzzy_model.load_state_dict(checkpoint['model_state_dict'])
        neurofuzzy_model.eval()
    else:
        print(f"⚠️  Phase 9 model not found at {phase9_model_path}")
        neurofuzzy_model = None
    
    return dataset, test_idx, giman_model, neurofuzzy_model, device

def compute_time_dependent_metrics(dataset, test_idx, model, device, model_name):
    """Compute time-dependent C-index at different time horizons"""
    print(f"\nComputing metrics for {model_name}...")
    
    if model is None:
        print(f"  Skipping {model_name} (model not loaded)")
        return None, None
    
    model.eval()
    with torch.no_grad():
        data = dataset.to(device)
        
        # Get predictions
        if hasattr(model, 'forward_survival'):
            risk_scores = model.forward_survival(data.x, data.edge_index).cpu().numpy()
        else:
            # For models without separate survival head, use default forward
            outputs = model(data.x, data.edge_index)
            if isinstance(outputs, tuple):
                risk_scores = outputs[0].cpu().numpy()  # Assuming first output is risk
            else:
                risk_scores = outputs.cpu().numpy()
        
        risk_scores = risk_scores.flatten()
    
    # Extract test data
    test_times = dataset.time[test_idx].cpu().numpy()
    test_events = dataset.event[test_idx].cpu().numpy()
    test_risks = risk_scores[test_idx]
    
    # Time horizons (in months)
    time_horizons = [6, 12, 18, 24]  # 0.5, 1, 1.5, 2 years
    
    c_indices = []
    for horizon in time_horizons:
        # Patients who haven't reached this time point
        mask = test_times <=horizon
        
        if mask.sum() > 10:  # Need enough patients
            c_idx = concordance_index(
                test_times[mask],
                -test_risks[mask],  # Negative because higher risk = shorter time
                test_events[mask]
            )
            c_indices.append(c_idx)
            print(f"  Time {horizon} months: C-Index = {c_idx:.4f} (n={mask.sum()})")
        else:
            c_indices.append(np.nan)
            print(f"  Time {horizon} months: Insufficient data (n={mask.sum()})")
    
    return time_horizons, c_indices

def create_real_time_auc_figure(time_horizons, giman_metrics, neurofuzzy_metrics):
    """Create figure with real metrics"""
    
    # Convert months to years
    time_years = [t/12 for t in time_horizons]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot lines
    if giman_metrics is not None:
        ax.plot(time_years, giman_metrics, 'o-', linewidth=2.5, markersize=8,
                color='#457b9d', label='GIMAN (Crisp)', alpha=0.8)
    
    if neurofuzzy_metrics is not None:
        ax.plot(time_years, neurofuzzy_metrics, 's-', linewidth=2.5, markersize=8,
                color='#e63946', label='Neuro-Fuzzy GIMAN', alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Time Since Baseline (Years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('C-Index (Survival Prediction)', fontsize=12, fontweight='bold')
    ax.set_title('Time-Dependent Survival Prediction Performance\\n(Real Model Metrics)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(time_years)
    ax.set_xticklabels(['0.5', '1.0', '1.5', '2.0'])
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11, frameon=True, shadow=True)
    
    # Add horizontal line at 0.5 (random)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random')
    
    plt.tight_layout()
    output_path = output_dir / "Figure5_Time_Dependent_AUC.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✅ Saved Figure 5: {output_path}")

def main():
    print("="*80)
    print("FIGURE 5: TIME-DEPENDENT AUC (REAL METRICS)")
    print("="*80)
    
    # Load models
    dataset, test_idx, giman_model, neurofuzzy_model, device = load_models_and_data()
    
    # Compute metrics
    giman_times, giman_cindices = compute_time_dependent_metrics(
        dataset, test_idx, giman_model, device, "GIMAN (Crisp)"
    )
    
    neurofuzzy_times, neurofuzzy_cindices = compute_time_dependent_metrics(
        dataset, test_idx, neurofuzzy_model, device, "Neuro-Fuzzy GIMAN"
    )
    
    # Use whichever succeeded
    if giman_times is not None:
        time_horizons = giman_times
    else:
        time_horizons = neurofuzzy_times
    
    # Create figure
    create_real_time_auc_figure(time_horizons, giman_cindices, neurofuzzy_cindices)
    
    print("\n✅ Figure 5 complete (real metrics)!")

if __name__ == "__main__":
    main()
