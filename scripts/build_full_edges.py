import json
from collections import Counter
from pathlib import Path

import ijson
import torch

DATA_DIR = Path("Twibot-20")
PROCESSED_DIR = Path("processed")
INDEX_PATH = PROCESSED_DIR / "full_node_index.json"

with INDEX_PATH.open("r", encoding="utf-8") as f:
    node_to_idx = json.load(f)

relation_to_type = {"following": 0, "follower": 1}
edges = set()
raw_counts = Counter()
kept_counts = Counter()
missing_target_counts = Counter()
records_by_split = Counter()

def process_record(record, source_split):
    source_id = str(record.get("ID", "")).strip()
    source_idx = node_to_idx[source_id]
    neighbor = record.get("neighbor")
    if not isinstance(neighbor, dict):
        return

    for relation, targets in neighbor.items():
        if relation not in relation_to_type or not isinstance(targets, list):
            continue
        relation_type = relation_to_type[relation]
        for target in targets:
            raw_counts[relation] += 1
            target_id = str(target).strip()
            target_idx = node_to_idx.get(target_id)
            if target_idx is None:
                missing_target_counts[relation] += 1
                continue
            edges.add((source_idx, target_idx, relation_type))
            kept_counts[relation] += 1

for split in ["train", "dev", "test"]:
    with (DATA_DIR / f"{split}.json").open("r", encoding="utf-8") as f:
        records = json.load(f)
    for record in records:
        records_by_split[split] += 1
        process_record(record, split)

with (DATA_DIR / "support.json").open("rb") as f:
    for record in ijson.items(f, "item"):
        records_by_split["support"] += 1
        process_record(record, "support")

edge_list = sorted(edges)
edge_index = torch.tensor(
    [[src for src, _, _ in edge_list], [dst for _, dst, _ in edge_list]],
    dtype=torch.long,
)
edge_type = torch.tensor([rel for _, _, rel in edge_list], dtype=torch.long)

torch.save(edge_index, PROCESSED_DIR / "edge_index.pt")
torch.save(edge_type, PROCESSED_DIR / "edge_type.pt")

print("节点数:", len(node_to_idx))
print("扫描记录:", dict(records_by_split))
print("原始边:", dict(raw_counts), "总数:", sum(raw_counts.values()))
print("保留边:", dict(kept_counts), "总数:", len(edge_list))
print("未找到目标节点:", dict(missing_target_counts))
print("edge_index 形状:", tuple(edge_index.shape))
print("edge_type 形状:", tuple(edge_type.shape))
