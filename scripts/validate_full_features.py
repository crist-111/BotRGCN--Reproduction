import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "processed"

def load(name):
    path = P / name
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")
    return torch.load(path, weights_only=True)

with (P / "full_node_index.json").open(encoding="utf-8") as f:
    node_index = json.load(f)

n = len(node_index)
x_prop = load("x_paper_properties.pt")
desc = load("description_roberta_full.pt")
tweets = load("tweet_roberta_full.pt")
y = load("y_full.pt")
edge_index = load("edge_index.pt")
edge_type = load("edge_type.pt")
train = load("train_mask.pt")
val = load("val_mask.pt")
test = load("test_mask.pt")

print("=== shapes ===")
print("nodes:", n)
print("properties:", tuple(x_prop.shape))
print("description:", tuple(desc.shape))
print("tweets:", tuple(tweets.shape))
print("y:", tuple(y.shape))
print("edge_index:", tuple(edge_index.shape))
print("edge_type:", tuple(edge_type.shape))

assert x_prop.shape == (n, 17)
assert desc.shape == (n, 768)
assert tweets.shape == (n, 768)
assert y.shape == (n,)
assert edge_index.shape[0] == 2
assert edge_index.shape[1] == edge_type.shape[0]
assert train.shape == val.shape == test.shape == (n,)

for name, tensor in [("properties", x_prop), ("description", desc), ("tweets", tweets)]:
    ok = bool(torch.isfinite(tensor).all())
    print(f"{name} finite:", ok)
    assert ok

print("train:", int(train.sum()))
print("validation:", int(val.sum()))
print("test:", int(test.sum()))
print("support labels:", int((y == -1).sum()))
assert int(train.sum()) == 8278
assert int(val.sum()) == 2365
assert int(test.sum()) == 1183
assert int((train & val).sum()) == 0
assert int((train & test).sum()) == 0
assert int((val & test).sum()) == 0
assert int(((y >= 0) & ~(train | val | test)).sum()) == 0
assert torch.all(y[~(y >= 0)] == -1)

assert int(edge_index.min()) >= 0
assert int(edge_index.max()) < n
assert set(torch.unique(edge_type).tolist()) == {0, 1}

support_idx = y == -1
zero_desc = int((desc[support_idx].abs().sum(1) == 0).sum())
zero_tweets = int((tweets[support_idx].abs().sum(1) == 0).sum())
print("support zero description:", zero_desc)
print("support zero tweets:", zero_tweets)
assert zero_desc == 0
print("说明: zero tweets 允许存在，表示原始记录没有可用推文；不等于编码缺失。")
print("完整特征对齐检查通过。")
