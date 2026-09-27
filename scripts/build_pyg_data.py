from pathlib import Path

import torch
from torch_geometric.data import Data

PROCESSED_DIR = Path("processed")

x = torch.load(PROCESSED_DIR / "x_full.pt", weights_only=True)
y = torch.load(PROCESSED_DIR / "y_full.pt", weights_only=True)
edge_index = torch.load(PROCESSED_DIR / "edge_index.pt", weights_only=True)
edge_type = torch.load(PROCESSED_DIR / "edge_type.pt", weights_only=True)
train_mask = torch.load(PROCESSED_DIR / "train_mask.pt", weights_only=True)
val_mask = torch.load(PROCESSED_DIR / "val_mask.pt", weights_only=True)
test_mask = torch.load(PROCESSED_DIR / "test_mask.pt", weights_only=True)

data = Data(
    x=x,
    y=y,
    edge_index=edge_index,
    edge_type=edge_type,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
)

print(data)
print("节点数:", data.num_nodes)
print("边数:", data.num_edges)
print("特征维度:", data.num_node_features)
print("edge_index dtype:", data.edge_index.dtype)
print("edge_type dtype:", data.edge_type.dtype)
print("标签 dtype:", data.y.dtype)
print("train 节点:", int(data.train_mask.sum()))
print("validation 节点:", int(data.val_mask.sum()))
print("test 节点:", int(data.test_mask.sum()))
print("support 节点:", int((data.y == -1).sum()))

assert data.x.shape == (229580, 10)
assert data.y.shape == (229580,)
assert data.edge_index.shape[0] == 2
assert data.edge_index.shape[1] == data.edge_type.shape[0]
assert data.train_mask.sum() == 8278
assert data.val_mask.sum() == 2365
assert data.test_mask.sum() == 1183
assert not torch.isnan(data.x).any()
assert not torch.isinf(data.x).any()

torch.save(data, PROCESSED_DIR / "twibot20_full_data.pt")
print("Data 对象已保存")