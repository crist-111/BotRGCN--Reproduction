import json
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TRAIN = ROOT / 'scripts' / 'train_botrgcn_official.py'
seeds = [42, 43, 44]
results = []
for seed in seeds:
    print('\n===== seed {} ====='.format(seed), flush=True)
    subprocess.run([PY, str(TRAIN), '--seed', str(seed), '--save-result'], check=True)
    p = ROOT / 'official_result_seed_{}.json'.format(seed)
    results.append(json.loads(p.read_text(encoding='utf-8')))
metrics = ['accuracy', 'precision', 'recall', 'f1']
summary = {'seeds': seeds, 'runs': results, 'mean': {}, 'std': {}}
for m in metrics:
    values = np.array([r[m] for r in results], dtype=float)
    summary['mean'][m] = float(values.mean())
    summary['std'][m] = float(values.std(ddof=1))
out = ROOT / 'official_repeated_summary.json'
out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('\n=== repeated summary ===')
for m in metrics:
    print('{}: {:.4f} +/- {:.4f}'.format(m, summary['mean'][m], summary['std'][m]))
print('saved:', out)
