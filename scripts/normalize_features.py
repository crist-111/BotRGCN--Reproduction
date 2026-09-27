import torch
from pathlib import Path

PROCESSED_DIR = Path("processed")

x = torch.load(PROCESSED_DIR / "x_full_raw.pt", weights_only=True)
train_mask = torch.load(PROCESSED_DIR / "train_mask.pt", weights_only=True)

numeric_dim = 5

x_processed = x.clone()

# 1. 对数值特征做 log1p
x_processed[:, :numeric_dim] = torch.log1p(
    torch.clamp(x_processed[:, :numeric_dim], min=0)
)

# 2. 只使用训练集统计量
train_numeric = x_processed[train_mask, :numeric_dim]
mean = train_numeric.mean(dim=0)
std = train_numeric.std(dim=0).clamp_min(1e-6)

# 3. 标准化全部节点
x_processed[:, :numeric_dim] = (
    x_processed[:, :numeric_dim] - mean
) / std

torch.save(x_processed, PROCESSED_DIR / "x_full.pt")
torch.save(mean, PROCESSED_DIR / "feature_mean.pt")
torch.save(std, PROCESSED_DIR / "feature_std.pt")

print("原始特征形状:", tuple(x.shape))
print("处理后特征形状:", tuple(x_processed.shape))
print("训练集均值:", mean)
print("训练集标准差:", std)
print("处理后 NaN:", bool(torch.isnan(x_processed).any()))
print("处理后 Inf:", bool(torch.isinf(x_processed).any()))
print("训练集数值列均值:", x_processed[train_mask, :numeric_dim].mean(dim=0))
print("训练集数值列标准差:", x_processed[train_mask, :numeric_dim].std(dim=0))
print("布尔特征唯一值:")

for i in range(numeric_dim, x.shape[1]):
    print(i, torch.unique(x_processed[:, i]))