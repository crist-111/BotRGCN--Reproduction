import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "Twibot-20" / "support.json"
CACHE_DIR = PROJECT_DIR / "hf_cache"
OUTPUT_DIR = PROJECT_DIR / "processed" / "support_roberta_chunks"
MODEL_NAME = "roberta-base"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-tweets", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def clean_text(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


def get_description(record):
    profile = record.get("profile") or {}
    return clean_text(profile.get("description"))


def get_tweets(record, max_tweets):
    tweets = record.get("tweet") or []

    if not isinstance(tweets, list):
        return []

    result = []

    for tweet in tweets[:max_tweets]:
        text = ""

        if isinstance(tweet, str):
            text = tweet
        elif isinstance(tweet, dict):
            text = (
                tweet.get("full_text")
                or tweet.get("text")
                or tweet.get("content")
                or ""
            )

        text = clean_text(text)

        if text:
            result.append(text)

    return result


def encode_texts(texts, tokenizer, model, device, max_length, batch_size):
    outputs = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            result = model(**encoded).last_hidden_state[:, 0, :]

        outputs.append(result.cpu())

    if not outputs:
        return torch.empty((0, 768), dtype=torch.float32)

    return torch.cat(outputs, dim=0)


def encode_user_texts(
    descriptions,
    tweets,
    tokenizer,
    model,
    device,
    max_length,
    batch_size,
):
    description_texts = [
        text if text else "[EMPTY]"
        for text in descriptions
    ]

    description_embeddings = encode_texts(
        description_texts,
        tokenizer,
        model,
        device,
        max_length,
        batch_size,
    )

    tweet_texts = []
    tweet_owner = []

    for user_index, user_tweets in enumerate(tweets):
        for text in user_tweets:
            tweet_texts.append(text)
            tweet_owner.append(user_index)

    tweet_embeddings = torch.zeros(
        (len(tweets), 768),
        dtype=torch.float32,
    )

    if tweet_texts:
        encoded_tweets = encode_texts(
            tweet_texts,
            tokenizer,
            model,
            device,
            max_length,
            batch_size,
        )

        counts = torch.zeros(len(tweets), dtype=torch.long)

        for row, owner in enumerate(tweet_owner):
            tweet_embeddings[owner] += encoded_tweets[row]
            counts[owner] += 1

        nonempty = counts > 0
        tweet_embeddings[nonempty] /= counts[nonempty].unsqueeze(1)

    return description_embeddings, tweet_embeddings


def load_model(device):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    ).to(device)

    model.eval()
    return tokenizer, model


def main():
    args = parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("device:", device)
    print("data:", DATA_PATH)
    print("output:", OUTPUT_DIR)

    tokenizer, model = load_model(device)

    ids = []
    descriptions = []
    tweets = []

    processed = 0
    chunk_index = 0

    with DATA_PATH.open("rb") as file:
        records = __import__("ijson").items(file, "item")

        for record in records:
            if args.max_users is not None and processed >= args.max_users:
                break

            user_id = str(record["ID"]).strip()

            ids.append(user_id)
            descriptions.append(get_description(record))
            tweets.append(get_tweets(record, args.max_tweets))

            processed += 1

            if len(ids) < args.chunk_size:
                continue

            output_path = OUTPUT_DIR / f"chunk_{chunk_index:06d}.pt"

            if output_path.exists() and not args.overwrite:
                print("skip", output_path.name)
            else:
                description_embeddings, tweet_embeddings = encode_user_texts(
                    descriptions,
                    tweets,
                    tokenizer,
                    model,
                    device,
                    args.max_length,
                    args.batch_size,
                )

                torch.save(
                    {
                        "ids": ids,
                        "description": description_embeddings,
                        "tweets": tweet_embeddings,
                    },
                    output_path,
                )

                print("saved", output_path.name)

            chunk_index += 1
            ids = []
            descriptions = []
            tweets = []

            print(f"processed {processed}")

    if ids:
        output_path = OUTPUT_DIR / f"chunk_{chunk_index:06d}.pt"

        if output_path.exists() and not args.overwrite:
            print("skip", output_path.name)
        else:
            description_embeddings, tweet_embeddings = encode_user_texts(
                descriptions,
                tweets,
                tokenizer,
                model,
                device,
                args.max_length,
                args.batch_size,
            )

            torch.save(
                {
                    "ids": ids,
                    "description": description_embeddings,
                    "tweets": tweet_embeddings,
                },
                output_path,
            )

            print("saved", output_path.name)

        processed += 0

    manifest = {
        "model": MODEL_NAME,
        "description_dim": 768,
        "tweet_dim": 768,
        "chunk_size": args.chunk_size,
        "max_length": args.max_length,
        "max_tweets": args.max_tweets,
        "processed_users": processed,
    }

    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print("done:", processed)


if __name__ == "__main__":
    main()    