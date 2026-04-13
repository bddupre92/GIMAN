# 🚀 QUICK MODEL ACCESS GUIDE
*Essential commands for accessing your saved GIMAN models*

---

## 🎯 **EMERGENCY MODEL RESTORATION** (30 seconds)
```bash
cd "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025"
python scripts/restore_production_model.py
```
**Result:** Production model v1.0.0 (98.93% AUC-ROC) instantly restored

---

## 📊 **CURRENT PRODUCTION MODEL SPECS**
- **Version:** v1.0.0 
- **Performance:** 98.93% AUC-ROC, 76.84% Accuracy
- **Architecture:** Binary classifier (Healthy vs Disease)
- **Features:** 7 biomarkers (Age, Education, MoCA, UPDRS I/III, Caudate/Putamen SBR)
- **Parameters:** 92,866 total parameters
- **Graph:** k=6 cosine similarity, 557 nodes

---

## 🔧 **MODEL LOADING CODE**
```python
import torch
from torch_geometric.data import Data
import sys
sys.path.append('scripts')
from giman_models import GIMAN

# Load the production model
def load_production_model():
    model = GIMAN(
        input_dim=7,
        hidden_dims=[96, 256, 64],
        output_dim=2,
        dropout=0.41
    )
    
    # Load saved weights
    checkpoint = torch.load('models/registry/giman_binary_classifier_v1.0.0/model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load graph data
    graph_data = torch.load('models/registry/giman_binary_classifier_v1.0.0/graph_data.pth')
    
    return model, graph_data

# Quick prediction
model, graph_data = load_production_model()
with torch.no_grad():
    predictions = model(graph_data)
    probabilities = torch.softmax(predictions, dim=1)
```

---

## 📂 **FILE LOCATIONS**
```
models/registry/giman_binary_classifier_v1.0.0/
├── model.pth                 # Model weights & optimizer
├── graph_data.pth           # Graph structure & features  
├── config.json              # Model configuration
└── metadata.json           # Performance metrics & info
```

---

## 🛠️ **MODEL REGISTRY OPERATIONS**
```python
from scripts.create_model_backup_system import GIMANModelRegistry

registry = GIMANModelRegistry()

# List all models
registry.list_models()

# Get model info
info = registry.get_model_info('giman_binary_classifier', 'v1.0.0')

# Load specific version
model, graph_data, config = registry.restore_model('giman_binary_classifier', 'v1.0.0')

# Compare models
registry.compare_models('giman_binary_classifier', 'v1.0.0', 'v1.1.0')
```

---

## ⚡ **QUICK COMMANDS REFERENCE**

### Restore Production Model
```bash
python scripts/restore_production_model.py
```

### Validate Model Integrity  
```bash
python scripts/validate_production_model.py models/restored_production_giman_binary_classifier_v1.0.0
```

### Check Model Registry
```bash
python -c "from scripts.create_model_backup_system import GIMANModelRegistry; GIMANModelRegistry().list_models()"
```

### Backup Current Model (before experiments)
```bash
python scripts/create_model_backup_system.py --model-path YOUR_MODEL_PATH --version v1.X.X
```

---

## 🔍 **TROUBLESHOOTING**

**Issue:** Model files not found
**Solution:** Run restoration command - all files are safely backed up

**Issue:** Different performance than expected
**Solution:** Check if testing on training vs validation data (normal variance)

**Issue:** Graph data mismatch
**Solution:** Use graph_data.pth from same model version directory

**Issue:** Import errors  
**Solution:** Ensure you're in project root directory and scripts are in path

---

## 📈 **ENHANCEMENT TRACKING**

| Version | Features | AUC-ROC | Status | Notes |
|---------|----------|---------|--------|-------|
| v1.0.0  | 7 biomarkers | 98.93% | ✅ Production | Current baseline |
| v1.1.0  | 12 biomarkers | TBD | 🚧 In Progress | +genetics +CSF |

---

## 🎯 **NEXT: Phase 1.5 Enhanced Features**
- **Goal:** Add 5 biomarkers (LRRK2, GBA, APOE_RISK, ALPHA_SYN, NHY)
- **Target:** >99% AUC-ROC  
- **Safety:** v1.0.0 preserved as fallback
- **Timeline:** 1-2 weeks

---
*Updated: September 23, 2025*
*Production Model: v1.0.0 (98.93% AUC-ROC)*