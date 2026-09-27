import json
from pathlib import Path

import torch

PROCESSED = Path("processed")

with (PROCESSED / "full_node_index.json").open("r", encoding="utf-8") as f:
    node_to_idx = json.load(f)

with (PROCESSED / "roberta_labeled_ids.json").open("r", encoding="utf-8") as f:
    labeled_ids = json.load(f)

description = torch.load(
    PROCESSED / "description_roberta_labeled.pt",
    weights_only=True,
)

tweets = torch.load(
    PROCESSED / "tweet_roberta_labeled.pt",
    weights_only=True,
)

num_nodes = len(node_to_idx)
hidden_dim = description.shape[1]

description_full = torch.zeros(
    (num_nodes, hidden_dim),
    dtype=description.dtype,
)

tweets_full = torch.zeros(
    (num_nodes, hidden_dim),
    dtype=tweets.dtype,
)

for row, user_id in enumerate(labeled_ids):
    node_idx = node_to_idx[str(user_id).strip()]
    description_full[node_idx] = description[row]
    tweets_full[node_idx] = tweets[row]

torch.save(description_full, PROCESSED / "description_roberta_full_labeled.pt")
torch.save(tweets_full, PROCESSED / "tweet_roberta_full_labeled.pt")

print("description full:", tuple(description_full.shape))
print("tweets full:", tuple(tweets_full.shape))
print("labeled text nodes:", len(labeled_ids))
print("support text nodes: 0")
print("description finite:", bool(torch.isfinite(description_full).all()))
print("tweets finite:", bool(torch.isfinite(tweets_full).all()))