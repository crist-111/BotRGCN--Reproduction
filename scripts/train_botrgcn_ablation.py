import argparse, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from torch_geometric.nn import RGCNConv

P = Path(__file__).resolve().parents[1] / 'processed'

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class Model(nn.Module):
    def __init__(self, rels):
        super().__init__(); q=16
        self.d=nn.Linear(768,q); self.t=nn.Linear(768,q); self.n=nn.Linear(6,q); self.c=nn.Linear(11,q)
        self.fc=nn.Linear(64,64); self.g1=RGCNConv(64,64,num_relations=rels); self.g2=RGCNConv(64,64,num_relations=rels); self.post=nn.Linear(64,64); self.out=nn.Linear(64,1)
    def forward(self,d,t,p,e,r):
        z=torch.cat([F.leaky_relu(self.d(d)),F.leaky_relu(self.t(t)),F.leaky_relu(self.n(p[:,:6])),F.leaky_relu(self.c(p[:,6:]))],1)
        z=F.leaky_relu(self.fc(z)); z=F.leaky_relu(self.g1(z,e,r)); z=F.leaky_relu(self.g2(z,e,r)); z=F.leaky_relu(self.post(z)); return self.out(z).squeeze(-1)

def score(o,y,m):
    a=(torch.sigmoid(o[m])>=0.5).long().cpu().numpy(); b=y[m].cpu().numpy()
    return [accuracy_score(b,a),precision_score(b,a,zero_division=0),recall_score(b,a,zero_division=0),f1_score(b,a,zero_division=0),matthews_corrcoef(b,a)]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--ablation',choices=['full','no-description','no-tweets','no-numeric','no-categorical','properties-only'],default='full'); ap.add_argument('--relation',choices=['all','following','follower'],default='all'); args=ap.parse_args(); seed_all(args.seed)
    prop=torch.load(P/'x_paper_properties.pt',weights_only=True); d=torch.load(P/'description_roberta_full.pt',weights_only=True); t=torch.load(P/'tweet_roberta_full.pt',weights_only=True)
    y=torch.load(P/'y_full.pt',weights_only=True); ei=torch.load(P/'edge_index.pt',weights_only=True); et=torch.load(P/'edge_type.pt',weights_only=True); tr=torch.load(P/'train_mask.pt',weights_only=True); va=torch.load(P/'val_mask.pt',weights_only=True); te=torch.load(P/'test_mask.pt',weights_only=True)
    if args.ablation in ('no-description','properties-only'): d.zero_()
    if args.ablation in ('no-tweets','properties-only'): t.zero_()
    if args.ablation in ('no-numeric','properties-only'): prop[:, :6]=0
    if args.ablation in ('no-categorical','properties-only'): prop[:, 6:]=0
    if args.relation != 'all':
        k=0 if args.relation=='following' else 1; keep=et==k; ei=ei[:,keep]; et=torch.zeros(int(keep.sum()),dtype=torch.long)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); d,t,prop,y,ei,et,tr,va,te=[x.to(dev) for x in (d,t,prop,y,ei,et,tr,va,te)]
    model=Model(2 if args.relation=='all' else 1).to(dev); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=5e-4); best=-1; state=None; bad=0
    for ep in range(1,101):
        model.train(); opt.zero_grad(); o=model(d,t,prop,ei,et); loss=F.binary_cross_entropy_with_logits(o[tr],y[tr].float()); loss.backward(); opt.step(); model.eval()
        with torch.no_grad(): oo=model(d,t,prop,ei,et)
        v=score(oo,y,va); q=score(oo,y,tr)
        if v[3]>best: best=v[3]; state={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}; bad=0
        else: bad+=1
        if ep==1 or ep%10==0: print(f'Epoch {ep:03d} | Loss {loss.item():.4f} | Train F1 {q[3]:.4f} | Val F1 {v[3]:.4f}')
        if bad>=20: break
    model.load_state_dict(state); model.to(dev); model.eval()
    with torch.no_grad(): r=score(model(d,t,prop,ei,et),y,te)
    print(f'=== Ablation={args.ablation}, relation={args.relation}, seed={args.seed} ==='); print('accuracy precision recall f1 mcc'); print(' '.join(f'{x:.4f}' for x in r))
if __name__=='__main__': main()
