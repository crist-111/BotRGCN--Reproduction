"""Build the structured property features specified in BotRGCN.

Numerical properties (paper Table I): followers, followings, favorites,
statuses, active_days, screen_name_length.
Categorical properties (paper Table II): eleven profile properties.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import ijson
import torch

DATA_DIR = Path("Twibot-20")
OUT = Path("processed")
with (OUT / "full_node_index.json").open("r", encoding="utf-8") as f:
    node_to_idx = json.load(f)

# TwiBot-20 was collected around 2020; this reference date is used only to
# reproduce the paper's active_days feature from the available created_at text.
REFERENCE_DATE = datetime(2020, 8, 1, tzinfo=timezone.utc)
NUM_NAMES = ["followers", "followings", "favorites", "statuses", "active_days", "screen_name_length"]
CAT_NAMES = [
    "protected", "geo_enabled", "verified", "contributors_enabled",
    "is_translator", "is_translation_enabled", "profile_background_tile",
    "profile_user_background_image", "has_extended_profile", "default_profile",
    "default_profile_image",
]

def number(profile, key):
    try:
        return float(str(profile.get(key, "0")).strip())
    except (TypeError, ValueError):
        return 0.0

def boolean(profile, key):
    # Dataset uses profile_use_background_image; paper calls it
    # profile_user_background_image.
    if key == "profile_user_background_image":
        value = profile.get("profile_user_background_image", profile.get("profile_use_background_image"))
    else:
        value = profile.get(key)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return float(value)
    return 1.0 if str(value).strip().lower() == "true" else 0.0

def active_days(profile):
    text = str(profile.get("created_at", "")).strip()
    try:
        created = datetime.strptime(text, "%a %b %d %H:%M:%S %z %Y")
        return max(0.0, (REFERENCE_DATE - created).total_seconds() / 86400.0)
    except (ValueError, TypeError):
        return 0.0

def vector(record):
    p = record.get("profile") or {}
    values = [
        number(p, "followers_count"), number(p, "friends_count"),
        number(p, "favourites_count"), number(p, "statuses_count"),
        active_days(p), float(len(str(p.get("screen_name", "")).strip())),
    ]
    values.extend(boolean(p, name) for name in CAT_NAMES)
    return values

x = torch.zeros((len(node_to_idx), len(NUM_NAMES) + len(CAT_NAMES)), dtype=torch.float32)
y = torch.full((len(node_to_idx),), -1, dtype=torch.long)
train_mask = torch.zeros(len(node_to_idx), dtype=torch.bool)
val_mask = torch.zeros(len(node_to_idx), dtype=torch.bool)
test_mask = torch.zeros(len(node_to_idx), dtype=torch.bool)

for split, mask in [("train", train_mask), ("dev", val_mask), ("test", test_mask)]:
    with (DATA_DIR / f"{split}.json").open("r", encoding="utf-8") as f:
        for record in json.load(f):
            idx = node_to_idx[str(record["ID"]).strip()]
            x[idx] = torch.tensor(vector(record))
            y[idx] = int(record["label"])
            mask[idx] = True

with (DATA_DIR / "support.json").open("rb") as f:
    for record in ijson.items(f, "item"):
        idx = node_to_idx[str(record["ID"]).strip()]
        x[idx] = torch.tensor(vector(record))

# Paper's normalization details are not fully specified in the available method
# description. Keep the established train-only z-score preprocessing explicit;
# this is a reproduction choice, not a claimed exact paper hyperparameter.
train_num = x[train_mask, :len(NUM_NAMES)]
mean = train_num.mean(0)
std = train_num.std(0).clamp_min(1e-6)
x[:, :len(NUM_NAMES)] = (x[:, :len(NUM_NAMES)] - mean) / std

torch.save(x, OUT / "x_paper_properties.pt")
torch.save(mean, OUT / "paper_property_mean.pt")
torch.save(std, OUT / "paper_property_std.pt")
print("shape:", tuple(x.shape))
print("numerical dims:", len(NUM_NAMES), NUM_NAMES)
print("categorical dims:", len(CAT_NAMES), CAT_NAMES)
print("train numeric mean:", x[train_mask, :6].mean(0))
print("train numeric std:", x[train_mask, :6].std(0))
print("nan:", bool(torch.isnan(x).any()), "inf:", bool(torch.isinf(x).any()))
