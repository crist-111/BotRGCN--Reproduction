from pathlib import Path

import torch

PROCESSED_DIR = Path("processed")
x_profile = torch.load(PROCESSED_DIR / "x_full.pt", weights_only=True)
x_behavior = torch.load(PROCESSED_DIR / "behavior_raw.pt", weights_only=True)
train_mask = torch.load(PROCESSED_DIR / "train_mask.pt", weights_only=True)

# 保留 7 个有变化的行为特征，去掉恒定的 empty_tweet_ratio（第 7 列）。
behavior = x_behavior[:, :7].clone()
behavior[:, :6] = torch.log1p(torch.clamp(behavior[:, :6], min=0))

train_behavior = behavior[train_mask]
mean = train_behavior.mean(dim=0)
std = train_behavior.std(dim=0).clamp_min(1e-6)
behavior = (behavior - mean) / std

x_selected = torch.cat([x_profile, behavior], dim=1)
torch.save(x_selected, PROCESSED_DIR / "x_profile_behavior_selected.pt")

print("selected behavior shape:", tuple(behavior.shape))
print("combined shape:", tuple(x_selected.shape))
print("nan:", bool(torch.isnan(x_selected).any()))
print("inf:", bool(torch.isinf(x_selected).any()))
