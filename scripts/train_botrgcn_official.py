import random
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data

P = Path(__file__).resolve().parents[1] / "processed"
SEED = 42
D = 64
LR = 1e-3
WEIGHT_DECAY = 5e-4
EPOCHS = 100
PATIENCE = 20

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

class BotRGCN(nn.Module):
    def __init__(self, text_dim=768, num_dim=6, cat_dim=11, hidden_dim=D, relations=2):
        super().__init__()
        part = hidden_dim // 4
        self.description = nn.Linear(text_dim, part)
        self.tweets = nn.Linear(text_dim, part)
        self.numeric = nn.Linear(num_dim, part)
        self.categorical = nn.Linear(cat_dim, part)
        self.input = nn.Linear(hidden_dim, hidden_dim)
        self.rgcn1 = RGCNConv(hidden_dim, hidden_dim, num_relations=relations)
        self.rgcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations=relations)
        # Paper Eq. (4): an MLP projection after the final R-GCN layer.
        self.post_rgcn = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, description, tweets, properties, edge_index, edge_type):
        numeric = properties[:, :6]
        categorical = properties[:, 6:]
        parts = [
            F.leaky_relu(self.description(description)),
            F.leaky_relu(self.tweets(tweets)),
            F.leaky_relu(self.numeric(numeric)),
            F.leaky_relu(self.categorical(categorical)),
        ]
        x = F.leaky_relu(self.input(torch.cat(parts, dim=1)))
        x = F.leaky_relu(self.rgcn1(x, edge_index, edge_type))
        x = F.leaky_relu(self.rgcn2(x, edge_index, edge_type))
        x = F.leaky_relu(self.post_rgcn(x))
        return self.output(x).squeeze(-1)

def metrics(logits, y, mask):
    pred = (torch.sigmoid(logits[mask]) >= 0.5).long().detach().cpu().numpy()
    true = y[mask].detach().cpu().numpy()
    return {
        'accuracy': accuracy_score(true, pred),
        'precision': precision_score(true, pred, zero_division=0),
        'recall': recall_score(true, pred, zero_division=0),
        'f1': f1_score(true, pred, zero_division=0),
        'mcc': matthews_corrcoef(true, pred),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--save-result', action='store_true')
    parser.add_argument('--result-dir', type=Path, default=P.parent / 'results')
    args = parser.parse_args()
    seed_everything(args.seed)
    properties = torch.load(P / 'x_paper_properties.pt', weights_only=True)
    description = torch.load(P / 'description_roberta_full.pt', weights_only=True)
    tweets = torch.load(P / 'tweet_roberta_full.pt', weights_only=True)
    data = Data(
        x=properties,
        y=torch.load(P / 'y_full.pt', weights_only=True),
        edge_index=torch.load(P / 'edge_index.pt', weights_only=True),
        edge_type=torch.load(P / 'edge_type.pt', weights_only=True),
        train_mask=torch.load(P / 'train_mask.pt', weights_only=True),
        val_mask=torch.load(P / 'val_mask.pt', weights_only=True),
        test_mask=torch.load(P / 'test_mask.pt', weights_only=True),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device); properties = properties.to(device)
    description = description.to(device); tweets = tweets.to(device)
    print('device:', device, 'nodes:', data.num_nodes, 'edges:', data.num_edges)
    model = BotRGCN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best, state, bad = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train(); opt.zero_grad()
        logits = model(description, tweets, properties, data.edge_index, data.edge_type)
        loss = F.binary_cross_entropy_with_logits(logits[data.train_mask], data.y[data.train_mask].float())
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(description, tweets, properties, data.edge_index, data.edge_type)
        val = metrics(out, data.y, data.val_mask); train = metrics(out, data.y, data.train_mask)
        if val['f1'] > best:
            best = val['f1']; bad = 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: bad += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Train F1 {train['f1']:.4f} | Val F1 {val['f1']:.4f}")
        if bad >= PATIENCE: print('Early stopping at epoch:', epoch); break
    model.load_state_dict(state); model.to(device); model.eval()
    with torch.no_grad(): out = model(description, tweets, properties, data.edge_index, data.edge_type)
    result = metrics(out, data.y, data.test_mask)
    print(f'=== Official BotRGCN Test Result (seed={args.seed}) ===')
    for k, v in result.items(): print(f'{k}: {v:.4f}')
    if args.save_result:
        args.result_dir.mkdir(parents=True, exist_ok=True)
        out = args.result_dir / f'official_result_seed_{args.seed}.json'
        out.write_text(json.dumps({'seed': args.seed, **result}, indent=2), encoding='utf-8')
        print('saved:', out)

if __name__ == '__main__': main()
