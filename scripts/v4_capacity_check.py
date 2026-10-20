"""Does B=1 inference compute the training function? PhaseRoutedMoE capacity dispatch keeps C = max(8, 4*B*L*K/E)
slots per expert per chunk call. In training B=16 (C=128 for 2,048 tokens), at inference B=1 (C=8 for 128 tokens):
same 4x average load, but a single row has more variance, so more (token, expert) pairs can overflow and be dropped.
Measures, on the 16 first v4_eval val windows (127+1024 ids, same rng), the dropped fraction and the TF NLL with
B=1 (one window at a time) vs B=16 (all 16 windows as one batch, as the trainer/val sees chunks)."""
import os, sys, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fractus.fixed_generation import load_checkpoint, SEQ
import fractus.nn.moe as moe
torch.set_num_threads(int(os.environ.get('THREADS', '3')))
CK, TAG = os.environ['CKPT'], os.environ['TAG']
OUT = os.environ.get('OUT', '/workspace/scratch/extern/capacity.jsonl')
eng, cfg = load_checkpoint(CK, torch.float32)
stat = {'kept': 0.0, 'all': 0}
cls = [c for c in vars(moe).values() if isinstance(c, type) and hasattr(c, '_capacity_expert_forward')][0]
orig = cls._capacity_expert_forward
def wrapped(self, h, topk_idx, cf=4.0):
    B, L, _ = h.shape; K = topk_idx.shape[-1]; E = self.n_experts; A = B * L * K
    e = topk_idx.reshape(-1); oh = F.one_hot(e, E).float(); pos = (oh.cumsum(0) * oh).sum(1) - 1
    C = max(8, int((cf * A / E + 7) // 8 * 8)); stat['kept'] += float((pos < C).float().sum()); stat['all'] += A
    return orig(self, h, topk_idx, cf)
cls._capacity_expert_forward = wrapped
VB = '/workspace/v4build'; rng = np.random.default_rng(0); VW = []
for s in ['fineweb', 'wiki', 'tinystories', 'pg19']:
    a = np.fromfile(f'{VB}/{s}_val.i16', dtype=np.int16).astype(np.int64)
    for st in rng.integers(0, len(a) - 1152, 8): VW.append(a[st:st + 1151])
X = torch.tensor(np.stack([VW[i] for i in [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]]))  # 4 per source
def tf(ids):
    B = ids.shape[0]; eng.reset_thought(B); out = []
    with torch.inference_mode():
        for j in range(0, 1024 + 127, SEQ):
            x = ids[:, j:j + SEQ]; h = eng.observe(x).clone(); h[:, 0] += eng.thought_state[:, 0]
            for blk in eng.blocks: h, _ = blk.tick_chunk_core(h)
            eng.thought_state = h[:, -1:].detach(); out.append(h @ eng.output_head.weight.t())
    lg = torch.cat(out, 1)[:, :-1]; nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction='none').view(B, -1)
    return nll[:, 126:].mean(1)
res = {}
stat.update(kept=0.0, all=0); n1 = torch.stack([tf(X[i:i + 1])[0] for i in range(16)]); res['B1'] = dict(dropped=1 - stat['kept'] / stat['all'], nll=n1.tolist())
stat.update(kept=0.0, all=0); n16 = tf(X); res['B16'] = dict(dropped=1 - stat['kept'] / stat['all'], nll=n16.tolist())
r = dict(tag=TAG, dropped_B1=res['B1']['dropped'], dropped_B16=res['B16']['dropped'], nll_B1_median=float(n1.median()),
         nll_B16_median=float(n16.median()), nll_B1_mean_ok=float(n1[n1 < 6].mean()), nll_B16_mean_ok=float(n16[n16 < 6].mean()),
         per_window_B1=res['B1']['nll'], per_window_B16=res['B16']['nll'])
open(OUT, 'a').write(json.dumps(r) + '\n'); print(json.dumps(r)[:400])
