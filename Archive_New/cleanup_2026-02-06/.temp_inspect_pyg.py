import torch
from pathlib import Path
from torch_geometric.data import Data
p=Path(r"E:\My Drive\CSCI FALL 2025\data\03_prodromal\enhanced_training_ready\train_data.pt")
obj=torch.load(p)
print('type:', type(obj))
if isinstance(obj, Data):
    print('x shape:', obj.x.shape)
    print('time:', getattr(obj, 'time', None))
    print('event:', getattr(obj, 'event', None))
    print('edge_index shape:', obj.edge_index.shape if hasattr(obj, 'edge_index') else None)
else:
    print(repr(obj))
