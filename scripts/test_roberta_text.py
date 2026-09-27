import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

DATA_PATH = Path("Twibot-20/train.json")
MODEL_NAME = "roberta-base"
MAX_LENGTH = 128
MAX_USERS = 2
MAX_TWEETS_PER_USER = 5


def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def encode_texts(texts, tokenizer, model, device):
    texts = [text if text else "[EMPTY]" for text in texts]
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**batch)
    # RoBERTa 的 <s> token 表示作为句子向量。
    return outputs.last_hidden_state[:, 0, :]


with DATA_PATH.open("r", encoding="utf-8") as f:
    users = json.load(f)[:MAX_USERS]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("loading:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

for user in users:
    profile = user.get("profile") or {}
    description = clean(profile.get("description"))
    tweets = [clean(t) for t in (user.get("tweet") or [])][:MAX_TWEETS_PER_USER]

    description_embedding = encode_texts([description], tokenizer, model, device)
    tweet_embeddings = encode_texts(tweets or [""], tokenizer, model, device)
    tweet_embedding = tweet_embeddings.mean(dim=0, keepdim=True)

    print("ID:", user["ID"])
    print("description chars:", len(description))
    print("tweets encoded:", len(tweets))
    print("description shape:", tuple(description_embedding.shape))
    print("tweet shape:", tuple(tweet_embedding.shape))
    print("finite:", bool(torch.isfinite(description_embedding).all() and torch.isfinite(tweet_embedding).all()))
