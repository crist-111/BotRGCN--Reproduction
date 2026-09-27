import argparse
import json
import os
from pathlib import Path

# Keep the model cache in the project, where the current user can write.
PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT / "hf_cache"))

import torch
from transformers import AutoModel, AutoTokenizer

DATA_DIR = PROJECT / "Twibot-20"
OUT_DIR = PROJECT / "processed"
MODEL_NAME = "roberta-base"


def text(value):
    return "" if value is None else str(value).strip()


def encode(tokenizer, model, texts, device, max_length):
    if not texts:
        return torch.empty((0, model.config.hidden_size), dtype=torch.float32)
    safe = [item if item else "[EMPTY]" for item in texts]
    batch = tokenizer(
        safe, padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state[:, 0, :]
    return hidden.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-tweets", type=int, default=200)
    parser.add_argument("--max-users", type=int, default=0, help="0 means all labeled users")
    args = parser.parse_args()

    records = []
    for split in ["train", "dev", "test"]:
        with (DATA_DIR / f"{split}.json").open("r", encoding="utf-8") as f:
            records.extend(json.load(f))
    if args.max_users > 0:
        records = records[:args.max_users]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
).to(device)
    model.eval()
    hidden = model.config.hidden_size

    description = []
    tweet_vectors = []
    ids = []
    for start in range(0, len(records), args.batch_size):
        chunk = records[start:start + args.batch_size]
        ids.extend(str(r["ID"]).strip() for r in chunk)
        description.extend(encode(
            tokenizer, model,
            [text((r.get("profile") or {}).get("description")) for r in chunk],
            device, args.max_length,
        ))
        for record in chunk:
            tweets = [text(t) for t in (record.get("tweet") or [])[:args.max_tweets]]
            if not tweets:
                tweet_vectors.append(torch.zeros(hidden))
                continue
            vectors = []
            for tstart in range(0, len(tweets), args.batch_size):
                vectors.append(encode(tokenizer, model, tweets[tstart:tstart + args.batch_size], device, args.max_length))
            tweet_vectors.append(torch.cat(vectors, dim=0).mean(dim=0))
        print(f"processed {min(start + args.batch_size, len(records))}/{len(records)}")

    desc = torch.stack(description)
    tweets = torch.stack(tweet_vectors)
    OUT_DIR.mkdir(exist_ok=True)
    suffix = "labeled_sample" if args.max_users else "labeled"
    torch.save(desc, OUT_DIR / f"description_roberta_{suffix}.pt")
    torch.save(tweets, OUT_DIR / f"tweet_roberta_{suffix}.pt")
    with (OUT_DIR / f"roberta_{suffix}_ids.json").open("w", encoding="utf-8") as f:
        json.dump(ids, f)
    print("description:", tuple(desc.shape))
    print("tweets:", tuple(tweets.shape))
    print("finite:", bool(torch.isfinite(desc).all() and torch.isfinite(tweets).all()))


if __name__ == "__main__":
    main()
