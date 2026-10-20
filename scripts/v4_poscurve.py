"""NLL by grid-chunk index from a cold start, on the 24 non-pg19 v4_eval val windows (127+1024 ids, same rng):
does the loss keep falling as the carry accumulates context (uses info >128 ids back), or rise (carry drift)?
Chunk 0 = ids 1..127 (first chunk, in-chunk context only), chunk c = ids 128c..128c+127."""
import os, sys, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fractus.fixed_generation import load_checkpoint, FixedGenerator
torch.set_num_threads(int(os.environ.get('THREADS', '7')))
CK, TAG = os.environ['CKPT'], os.environ['TAG']; OUT = os.environ.get('OUT', '/workspace/scratch/extern/poscurve.jsonl')
eng, cfg = load_checkpoint(CK, torch.float32); G = FixedGenerator(eng)
rng = np.random.default_rng(0); rows = []
for s in ['fineweb', 'wiki', 'tinystories', 'pg19']:
    a = np.fromfile(f'/workspace/v4build/{s}_val.i16', dtype=np.int16).astype(np.int64)
    for st in rng.integers(0, len(a) - 1152, 8):
        if s == 'pg19': continue
        ctx = a[st:st + 1151].tolist(); lg = G.tf_logits(ctx)[:-1]
        nll = F.cross_entropy(lg, torch.tensor(ctx[1:]), reduction='none')
        rows.append([float(nll[max(0, 128 * c - 1):128 * c + 127].mean()) for c in range(9)])
m = np.mean(rows, 0).tolist()
r = dict(tag=TAG, n_windows=len(rows), nll_by_chunk=m, per_window=rows)
open(OUT, 'a').write(json.dumps(r) + '\n'); print(TAG, [round(x, 3) for x in m])
