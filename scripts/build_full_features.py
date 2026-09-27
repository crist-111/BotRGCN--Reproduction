import json
from pathlib import Path

import ijson
import torch

DATA_DIR = Path("Twibot-20")
PROCESSED_DIR = Path("processed")
INDEX_PATH = PROCESSED_DIR / "full_node_index.json"

NUMERIC_FIELDS = [
    "followers_count",
    "friends_count",
    "listed_count",
    "favourites_count",
    "statuses_count",
]
BOOLEAN_FIELDS = [
    "protected",
    "geo_enabled",
    "verified",
    "default_profile",
    "default_profile_image",
]


def parse_number(value):
    if value is None:
        return 0.0
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def parse_bool(value):
    return 1.0 if str(value).strip().lower() == "true" else 0.0


def profile_features(record):
    profile = record.get("profile") or {}
    values = [parse_number(profile.get(field)) for field in NUMERIC_FIELDS]
    values.extend(parse_bool(profile.get(field)) for field in BOOLEAN_FIELDS)
    return values


with INDEX_PATH.open("r", encoding="utf-8") as f:
    node_to_idx = json.load(f)

num_nodes = len(node_to_idx)
num_features = len(NUMERIC_FIELDS) + len(BOOLEAN_FIELDS)
x = torch.zeros((num_nodes, num_features), dtype=torch.float32)
y = torch.full((num_nodes,), -1, dtype=torch.long)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)


def write_labeled_split(split, mask):
    path = DATA_DIR / f"{split}.json"
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    for record in records:
        user_id = str(record["ID"]).strip()
        idx = node_to_idx[user_id]
        x[idx] = torch.tensor(profile_features(record), dtype=torch.float32)
        y[idx] = int(record["label"])
        mask[idx] = True


write_labeled_split("train", train_mask)
write_labeled_split("dev", val_mask)
write_labeled_split("test", test_mask)

support_count = 0
with (DATA_DIR / "support.json").open("rb") as f:
    for record in ijson.items(f, "item"):
        user_id = str(record["ID"]).strip()
        idx = node_to_idx[user_id]
        x[idx] = torch.tensor(profile_features(record), dtype=torch.float32)
        support_count += 1

torch.save(x, PROCESSED_DIR / "x_full_raw.pt")
torch.save(y, PROCESSED_DIR / "y_full.pt")
torch.save(train_mask, PROCESSED_DIR / "train_mask.pt")
torch.save(val_mask, PROCESSED_DIR / "val_mask.pt")
torch.save(test_mask, PROCESSED_DIR / "test_mask.pt")

all_mask = train_mask | val_mask | test_mask
print("x shape:", tuple(x.shape))
print("y shape:", tuple(y.shape))
print("train_mask 数量:", int(train_mask.sum()))
print("val_mask 数量:", int(val_mask.sum()))
print("test_mask 数量:", int(test_mask.sum()))
print("support 记录数:", support_count)
print("support 无标签节点数量:", int((y == -1).sum()))
print("mask 交集数量:", int((train_mask & val_mask).sum() + (train_mask & test_mask).sum() + (val_mask & test_mask).sum()))
print("未覆盖节点数量:", int((~all_mask & (y != -1)).sum()))
print("NaN:", bool(torch.isnan(x).any()))
print("Inf:", bool(torch.isinf(x).any()))
print("标签分布:", {int(k): int(v) for k, v in zip(*torch.unique(y[y >= 0], return_counts=True))})
