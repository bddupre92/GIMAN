import sys
from pathlib import Path
models_dir = Path(__file__).resolve().parent.parent / "models"
sys.path.insert(0, str(models_dir))
import torch
from giman_multitask import create_multitask_giman
from multitask_loss import MultiTaskLoss
print(" Imports successful")
project_root = Path(__file__).resolve().parents[5]
data_dir = project_root / "archive" / "development" / "phase8" / "subphase8_5_multitask_architecture" / "data"
train_data_list = torch.load(data_dir / "multitask_train_data.pt", weights_only=False)
train_data = train_data_list[0] if isinstance(train_data_list, list) else train_data_list
print(f" Loaded data: {train_data.num_nodes} nodes")
model = create_multitask_giman(input_dim=49)
print(f" Model created: {sum(p.numel() for p in model.parameters()):,} params")
model.eval()
with torch.no_grad():
    outputs = model(train_data)
print(f" Forward pass successful")
diagnostic_weights = torch.tensor([0.132, 0.868])
loss_fn = MultiTaskLoss(task_weights={'progression': 1.0, 'conversion': 1.0, 'saa': 1.0, 'diagnostic': 1.0}, saa_pos_weight=4.63, diagnostic_class_weights=diagnostic_weights)
total_loss, task_losses = loss_fn(outputs, train_data)
print(f" Loss computed: {total_loss.item():.4f}")
for task, loss in task_losses.items():
    print(f"  - {task}: {loss.item():.4f}")
print("="*60)
print(" ALL TESTS PASSED - Ready for full training!")
print("="*60)
