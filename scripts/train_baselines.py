import argparse, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch_geometric.nn import GCNConv, GATConv

P=Path(__file__).resolve().parents[1]/'processed'
def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
class Net(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.kind=kind
        self.d=nn.Linear(768,16); self.t=nn.Linear(768,16); self.n=nn.Linear(6,16); self.c=nn.Linear(11,16); self.fc=nn.Linear(64,64)
        if kind=='mlp': self.g1=self.g2=None
        elif kind=='gcn': self.g1=GCNConv(64,64); self.g2=GCNConv(64,64)
        else: self.g1=GATConv(64,64,heads=1); self.g2=GATConv(64,64,heads=1)
        self.post=nn.Linear(64,64); self.out=nn.Linear(64,1)
    def forward(self,d,t,p,e):
        z=torch.cat([F.leaky_relu(self.d(d)),F.leaky_relu(self.t(t)),F.leaky_relu(self.n(p[:,:6])),F.leaky_relu(self.c(p[:,6:]))],1); z=F.leaky_relu(self.fc(z))
        if self.g1 is not None: z=F.leaky_relu(self.g1(z,e)); z=F.leaky_relu(self.g2(z,e))
        return self.out(F.leaky_relu(self.post(z))).squeeze(-1)
def score(o,y,m):
    a=(torch.sigmoid(o[m])>=.5).long().cpu().numpy(); b=y[m].cpu().numpy()
    return accuracy_score(b,a),f1_score(b,a),matthews_corrcoef(b,a)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--kind',choices=['mlp','gcn','gat'],required=True); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args(); seed(args.seed)
    d=torch.load(P/'description_roberta_full.pt',weights_only=True); t=torch.load(P/'tweet_roberta_full.pt',weights_only=True); p=torch.load(P/'x_paper_properties.pt',weights_only=True); y=torch.load(P/'y_full.pt',weights_only=True); e=torch.load(P/'edge_index.pt',weights_only=True); tr=torch.load(P/'train_mask.pt',weights_only=True); va=torch.load(P/'val_mask.pt',weights_only=True); te=torch.load(P/'test_mask.pt',weights_only=True)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); d,t,p,y,e,tr,va,te=[x.to(dev) for x in (d,t,p,y,e,tr,va,te)]; model=Net(args.kind).to(dev); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=5e-4); best=-1; state=None; bad=0
    for ep in range(1,101):
        model.train(); opt.zero_grad(); o=model(d,t,p,e); loss=F.binary_cross_entropy_with_logits(o[tr],y[tr].float()); loss.backward(); opt.step(); model.eval()
        with torch.no_grad(): oo=model(d,t,p,e)
        v=score(oo,y,va)
        if v[1]>best: best=v[1]; state={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=20: break
    model.load_state_dict(state); model.to(dev); model.eval()
    with torch.no_grad(): r=score(model(d,t,p,e),y,te)
    print(f'=== baseline={args.kind}, seed={args.seed} ==='); print(f'accuracy: {r[0]:.4f}'); print(f'f1: {r[1]:.4f}'); print(f'mcc: {r[2]:.4f}')
if __name__=='__main__': main()
