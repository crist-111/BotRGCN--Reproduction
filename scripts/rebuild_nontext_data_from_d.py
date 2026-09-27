import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import torch

DATA = Path(r'D:/Twibot-20/Twibot-20')
OUT = Path(r'D:/processed')
OUT.mkdir(parents=True, exist_ok=True)
for n in ['train.json','dev.json','test.json','support.json']:
    if not (DATA/n).exists(): raise FileNotFoundError(DATA/n)

def records(split):
    if split == 'support':
        # Streaming parser for the large top-level JSON array.
        dec=json.JSONDecoder(); buf=''; started=False
        with (DATA/'support.json').open('r',encoding='utf-8') as f:
            while True:
                chunk=f.read(4*1024*1024)
                if not chunk: break
                buf += chunk; pos=0
                while True:
                    if not started:
                        k=buf.find('[')
                        if k<0: break
                        pos=k+1; started=True
                    while pos < len(buf) and buf[pos].isspace(): pos+=1
                    if pos < len(buf) and buf[pos]==']': return
                    try: obj,end=dec.raw_decode(buf,pos)
                    except ValueError: break
                    yield obj; pos=end
                    while pos < len(buf) and buf[pos].isspace(): pos+=1
                    if pos < len(buf) and buf[pos]==',': pos+=1
                buf=buf[pos:]
    else:
        yield from json.loads((DATA/f'{split}.json').read_text(encoding='utf-8'))

ids=[]; seen=set()
for split in ['train','dev','test','support']:
    for r in records(split):
        uid=str(r.get('ID','')).strip()
        if uid and uid not in seen: seen.add(uid); ids.append(uid)
node_to_idx={u:i for i,u in enumerate(ids)}
(OUT/'full_node_index.json').write_text(json.dumps(node_to_idx,ensure_ascii=False),encoding='utf-8')
(OUT/'full_node_ids.txt').write_text('\n'.join(ids)+'\n',encoding='utf-8')
print('nodes:',len(ids))

rel={'following':0,'follower':1}; edges=set(); counts=Counter()
for split in ['train','dev','test','support']:
    for r in records(split):
        s=node_to_idx.get(str(r.get('ID','')).strip()); nb=r.get('neighbor')
        if s is None or not isinstance(nb,dict): continue
        for name,targets in nb.items():
            if name not in rel or not isinstance(targets,list): continue
            for target in targets:
                t=node_to_idx.get(str(target).strip()); counts[name]+=1
                if t is not None: edges.add((s,t,rel[name]))
edge_list=sorted(edges)
torch.save(torch.tensor([[a for a,b,c in edge_list],[b for a,b,c in edge_list]],dtype=torch.long),OUT/'edge_index.pt')
torch.save(torch.tensor([c for a,b,c in edge_list],dtype=torch.long),OUT/'edge_type.pt')
print('edges:',len(edge_list),dict(counts))

cats=['protected','geo_enabled','verified','contributors_enabled','is_translator','is_translation_enabled','profile_background_tile','profile_user_background_image','has_extended_profile','default_profile','default_profile_image']
ref=datetime(2020,8,1,tzinfo=timezone.utc)
def num(p,k):
    try:return float(str(p.get(k,'0')).strip())
    except:return 0.0
def boolean(p,k):
    v=p.get('profile_use_background_image') if k=='profile_user_background_image' and 'profile_user_background_image' not in p else p.get(k)
    if isinstance(v,bool): return float(v)
    if isinstance(v,(int,float)) and v in (0,1): return float(v)
    return float(str(v).strip().lower()=='true')
def active(p):
    try:return max(0.,(ref-datetime.strptime(str(p.get('created_at','')).strip(),'%a %b %d %H:%M:%S %z %Y')).total_seconds()/86400.)
    except:return 0.0
def vec(r):
    p=r.get('profile') or {}
    v=[num(p,'followers_count'),num(p,'friends_count'),num(p,'favourites_count'),num(p,'statuses_count'),active(p),float(len(str(p.get('screen_name','')).strip()))]
    return v+[boolean(p,k) for k in cats]
x=torch.zeros((len(ids),17),dtype=torch.float32); y=torch.full((len(ids),),-1,dtype=torch.long)
masks={k:torch.zeros(len(ids),dtype=torch.bool) for k in ['train','dev','test']}
for split in ['train','dev','test']:
    for r in records(split):
        i=node_to_idx[str(r['ID']).strip()]; x[i]=torch.tensor(vec(r)); y[i]=int(r['label']); masks[split][i]=True
for r in records('support'):
    x[node_to_idx[str(r['ID']).strip()]]=torch.tensor(vec(r))
tr=x[masks['train'],:6]; mean=tr.mean(0); std=tr.std(0).clamp_min(1e-6); x[:,:6]=(x[:,:6]-mean)/std
torch.save(x,OUT/'x_paper_properties.pt'); torch.save(mean,OUT/'paper_property_mean.pt'); torch.save(std,OUT/'paper_property_std.pt')
torch.save(y,OUT/'y_full.pt'); torch.save(masks['train'],OUT/'train_mask.pt'); torch.save(masks['dev'],OUT/'val_mask.pt'); torch.save(masks['test'],OUT/'test_mask.pt')
print('properties:',tuple(x.shape),'labels:',Counter(y[y>=0].tolist()),'masks:',[int(masks[k].sum()) for k in ['train','dev','test']])
print('finite:',bool(torch.isfinite(x).all()),'unlabeled:',int((y<0).sum()))
