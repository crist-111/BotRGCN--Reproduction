import json
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT / "processed"
CHUNK_DIR = PROCESSED / "support_roberta_chunks"


def load_index():
    with (PROCESSED / "full_node_index.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    node_to_idx = load_index()
    num_nodes = len(node_to_idx)

    labeled_desc = torch.load(
        PROCESSED / "description_roberta_full_labeled.pt",
        weights_only=True,
    )
    labeled_tweets = torch.load(
        PROCESSED / "tweet_roberta_full_labeled.pt",
        weights_only=True,
    )
    with (PROCESSED / "roberta_labeled_ids.json").open("r", encoding="utf-8") as f:
        labeled_ids = json.load(f)

    hidden = labeled_desc.shape[1]
    description = torch.zeros((num_nodes, hidden), dtype=torch.float32)
    tweets = torch.zeros((num_nodes, hidden), dtype=torch.float32)
    filled = torch.zeros(num_nodes, dtype=torch.bool)

    for row, user_id in enumerate(labeled_ids):
        idx = node_to_idx[str(user_id).strip()]
        description[idx] = labeled_desc[row]
        tweets[idx] = labeled_tweets[row]
        filled[idx] = True

    chunks = sorted(CHUNK_DIR.glob("chunk_*.pt"))
    if not chunks:
        raise FileNotFoundError(f"没有找到 support chunk: {CHUNK_DIR}")

    support_count = 0
    for chunk_path in chunks:
        chunk = torch.load(chunk_path, weights_only=False)
        ids = chunk["ids"]
        desc = chunk["description"]
        tw = chunk["tweets"]
        if len(ids) != desc.shape[0] or len(ids) != tw.shape[0]:
            raise ValueError(f"chunk 形状不一致: {chunk_path.name}")
        for row, user_id in enumerate(ids):
            idx = node_to_idx[str(user_id).strip()]
            if filled[idx]:
                raise ValueError(f"节点重复填充: {user_id}")
            description[idx] = desc[row]
            tweets[idx] = tw[row]
            filled[idx] = True
            support_count += 1

    missing = int((~filled).sum())
    print("节点总数:", num_nodes)
    print("标签节点文本数:", len(labeled_ids))
    print("support chunk 文本数:", support_count)
    print("已填充节点:", int(filled.sum()))
    print("缺失节点:", missing)

    if missing:
        raise RuntimeError("仍有节点缺少文本向量，不能生成完整矩阵")
    if not torch.isfinite(description).all() or not torch.isfinite(tweets).all():
        raise RuntimeError("文本矩阵包含 NaN 或 Inf")

    torch.save(description, PROCESSED / "description_roberta_full.pt")
    torch.save(tweets, PROCESSED / "tweet_roberta_full.pt")
    print("description shape:", tuple(description.shape))
    print("tweets shape:", tuple(tweets.shape))
    print("finite: True")


if __name__ == "__main__":
    main()
