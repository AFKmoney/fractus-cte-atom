"""Mechanism check for fractus/nn/copy_head.py alone: embedding -> CopyHead (gate init 0) -> linear readout, no
Fractus body. Same copy rows as scripts/copyhead_tiny.py. If this learns to copy, the head math (mask, shifted
keys, K/V window across chunks) is right and any failure inside Fractus is optimisation, not plumbing."""
import os, sys, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fractus.nn.copy_head import CopyHead
torch.set_num_threads(2); torch.manual_seed(0)
os.environ.setdefault('ARM', 'base')
txt = np.fromfile('/workspace/v4build/tinystories.i16', dtype=np.int16, count=5_000_000).astype(np.int64)
if os.environ.get('FILLER') == 'letters':   # diagnostic: random letters instead of text around the copies
    txt = np.random.default_rng(5).integers(97, 123, 5_000_000)
LET = np.arange(97, 123); L = 256
def row(kind, rng):
    s = rng.integers(0, len(txt) - L - 1); r = txt[s:s + L + 1].copy(); sc = np.zeros(L, bool); S = rng.choice(LET, 16)
    a = rng.integers(130, 160) if kind == 'same' else rng.integers(5, 40)
    b = min(a + 18 + (rng.integers(40, 60) if kind == 'same' else rng.integers(130, 170)), L - 17)
    r[a] = 124; r[a + 1:a + 17] = S; r[a + 17] = 124; r[b] = 124; r[b + 1:b + 17] = S; sc[b:b + 16] = True
    return r, sc
def batch(rng, kinds):
    rs = [row(k, rng) for k in kinds]; return torch.tensor(np.stack([a for a, _ in rs])), torch.tensor(np.stack([b for _, b in rs]))
D = 64; emb = nn.Embedding(266, D); nn.init.normal_(emb.weight, std=0.1); head = CopyHead(D, 2, 32, 128, ngram=int(os.environ.get('NGRAM', 1))); out_l = nn.Linear(D, 266)
head.ghost = float(os.environ.get('GHOST', 0))
with torch.no_grad(): head.gate.fill_(float(os.environ.get('GATE_INIT', 0)))   # diagnostic only; default 0
opt = torch.optim.AdamW(list(emb.parameters()) + list(head.parameters()) + list(out_l.parameters()), lr=3e-3)
def fwd(ids):
    head.reset_state(ids.shape[0]); out = []
    for j in range(0, L, 128):
        h = emb(ids[:, j:j + 128]); out.append(out_l(head.attend(h)))
    return torch.cat(out, 1)
rng = np.random.default_rng(0); er = np.random.default_rng(9)
EV = {k: batch(er, [k] * 64) for k in ('same', 'cross')}; res = []
for step in range(1, int(os.environ.get('STEPS', 600)) + 1):
    ids, sc = batch(rng, ['same', 'cross'] * 8)
    lg = fwd(ids); loss = F.cross_entropy(lg.reshape(-1, 266), ids[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0:
        with torch.no_grad():
            r = dict(step=step, loss=float(loss), gate_rms=float(head.gate.pow(2).mean().sqrt()))
            for k, (i, s) in EV.items(): r[k] = float(((fwd(i).argmax(-1) == i[:, 1:]) & s).sum() / s.sum())
        res.append(r); print(r, flush=True)
json.dump(res, open(f"/workspace/scratch/copyhead/mech_check_ng{os.environ.get('NGRAM', 1)}_{os.environ.get('FILLER', 'text')}_{os.environ.get('STEPS', 600)}_g{os.environ.get('GATE_INIT', 0)}_ghost{os.environ.get('GHOST', 0)}.json", 'w'), indent=1)
