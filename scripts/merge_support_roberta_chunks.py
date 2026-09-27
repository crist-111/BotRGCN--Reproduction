import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'processed'
CHUNK_DIR = P / 'support_roberta_chunks'
H = 768

with (P / 'full_node_index.json').open(encoding='utf-8') as f:
    node_to_idx = json.load(f)
with (P / 'roberta_labeled_ids.json').open(encoding='utf-8') as f:
    labeled = {str(x).strip() for x in json.load(f)}

chunks = sorted(CHUNK_DIR.glob('chunk_*.pt'))
if not chunks:
    raise FileNotFoundError('No support chunks found')

seen = set()
total = 0
for path in chunks:
    d = torch.load(path, weights_only=True)
    ids = [str(x).strip() for x in d['ids']]
    desc, tweet = d['description'], d['tweets']
    if desc.shape != (len(ids), H) or tweet.shape != (len(ids), H):
        raise ValueError(f'{path.name}: shape mismatch')
    if not torch.isfinite(desc).all() or not torch.isfinite(tweet).all():
        raise ValueError(f'{path.name}: NaN/Inf')
    for uid in ids:
        if uid in seen: raise ValueError(f'duplicate support ID: {uid}')
        if uid in labeled: raise ValueError(f'ID overlaps labeled set: {uid}')
        if uid not in node_to_idx: raise ValueError(f'ID missing index: {uid}')
        seen.add(uid)
    total += len(ids)
    print(f'{path.name}: {len(ids)}')

expected = len(node_to_idx) - len(labeled)
print('chunks:', len(chunks), 'support:', total, 'expected:', expected)
if total != expected:
    raise ValueError(f'incomplete support: {total} != {expected}')

desc_full = torch.zeros((len(node_to_idx), H), dtype=torch.float32)
tweet_full = torch.zeros((len(node_to_idx), H), dtype=torch.float32)
for path in chunks:
    d = torch.load(path, weights_only=True)
    for i, uid in enumerate(d['ids']):
        j = node_to_idx[str(uid).strip()]
        desc_full[j] = d['description'][i]
        tweet_full[j] = d['tweets'][i]

torch.save(desc_full, P / 'description_roberta_full.pt')
torch.save(tweet_full, P / 'tweet_roberta_full.pt')
print('description:', tuple(desc_full.shape))
print('tweets:', tuple(tweet_full.shape))
print('finite:', bool(torch.isfinite(desc_full).all() and torch.isfinite(tweet_full).all()))
