"""Copy head, test 2: tiny Fractus (same engine code, small dims), trained from scratch on CPU, with vs without the
readout copy head (fractus/nn/copy_head.py). Same seed, data and steps for both arms.

Batch of 256-id rows = 2 grid chunks (carry detached between chunks, as in the trainer):
  copy_same  : |S| ... |S  with S = 16 random letters, both copies in chunk 2 (gap 40-60 ids)
  copy_cross : first |S| in chunk 1, second in chunk 2 (gap 130-190 ids: only the carry / the K/V window reach it)
  ar         : 16 pairs "KQ=ab;" in chunk 2, then 4 queries "KQ=" -> ab (associative recall)
  text       : TinyStories train ids (filler around the synthetic parts too)
Scores: top-1 on the 16 letters of the second copy, on the 2 value letters of each query; text val NLL.
"""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.kuramoto_fix import apply_kuramoto_routing_fix
from fractus.nn.copy_head import attach_copy_head

torch.set_num_threads(int(os.environ.get('THREADS', '2')))
ARM = os.environ.get('ARM', 'base')            # base | copy
STEPS = int(os.environ.get('STEPS', 600)); B = 16; L = 256; SEQ = 128
OUT = os.environ.get('OUT', f'/workspace/scratch/copyhead/tiny_{ARM}.jsonl')
CFG = dict(d_model=128, n_heads=4, d_head=32, n_levels=2, n_oscillators=8, coupling_rank=4,
           n_experts=8, top_k=2, expert_d_ff=256, siren_rank=8, n_layers=4)
txt = np.fromfile('/workspace/v4build/tinystories.i16', dtype=np.int16, count=20_000_000).astype(np.int64)
txv = np.fromfile('/workspace/v4build/tinystories_val.i16', dtype=np.int16).astype(np.int64)
LET = np.arange(97, 123); UP = np.arange(65, 91)

def sample(kind, rng):
    s = rng.integers(0, len(txt) - L - 1); row = txt[s:s + L + 1].copy(); score = np.zeros(L, bool)  # score[t]: target row[t+1]
    if kind in ('copy_same', 'copy_cross'):
        S = rng.choice(LET, 16)
        a = rng.integers(130, 160) if kind == 'copy_same' else rng.integers(5, 40)
        b = a + 18 + (rng.integers(40, 60) if kind == 'copy_same' else rng.integers(130, 170))
        b = min(b, L - 17)
        row[a] = 124; row[a + 1:a + 17] = S; row[a + 17] = 124
        row[b] = 124; row[b + 1:b + 17] = S
        score[b:b + 16] = True
    elif kind == 'ar':
        keys = set()
        while len(keys) < 16: keys.add(tuple(rng.choice(UP, 2)))
        keys = list(keys); vals = [tuple(rng.choice(LET, 2)) for _ in keys]
        p = 129
        for k, v in zip(keys, vals):
            row[p:p + 6] = [k[0], k[1], 61, v[0], v[1], 59]; p += 6
        for qi in rng.choice(16, 4, replace=False):
            k, v = keys[qi], vals[qi]; row[p:p + 6] = [k[0], k[1], 61, v[0], v[1], 59]
            score[p + 2:p + 4] = True; p += 6
    return row, score

MIX = os.environ.get('MIX', 'default')          # default: 6 copy / 4 ar / 6 text rows; synth: 6 / 6 / 4
KINDS = (['copy_same'] * 3 + ['copy_cross'] * 3 + ['ar'] * 4 + ['text'] * 6 if MIX == 'default'
         else ['copy_same'] * 3 + ['copy_cross'] * 3 + ['ar'] * 6 + ['text'] * 4)
def batch(rng, kinds=KINDS):
    rs = [sample(k, rng) for k in kinds]
    return torch.from_numpy(np.stack([r for r, _ in rs])), torch.from_numpy(np.stack([s for _, s in rs]))

torch.manual_seed(0)
eng = ContinuousThoughtEngine(vocab_size=266, **CFG)
apply_kuramoto_routing_fix(eng, log=lambda *a, **k: None)
if ARM == 'copy':
    attach_copy_head(eng, n_heads=2, d_head=32, mem=128, ngram=int(os.environ.get('NGRAM', 1)))
    eng.copy_head.ghost = float(os.environ.get('GHOST', 0))
    with torch.no_grad(): eng.copy_head.gate.fill_(float(os.environ.get('GATE_INIT', 0)))
nparam = sum(p.numel() for p in eng.parameters()); nhead = sum(p.numel() for p in eng.copy_head.parameters()) if ARM == 'copy' else 0
opt = torch.optim.AdamW(eng.parameters(), lr=float(os.environ.get('LR', 2e-3)), weight_decay=0.0)

def run_rows(ids, train):
    eng.reset_thought(ids.shape[0]); ces = []; hs = []
    for j in range(0, L, SEQ):
        x, y = ids[:, j:j + SEQ], ids[:, j + 1:j + SEQ + 1]
        ce, lb, h = eng.tick_chunk_train_ce(x, y, return_hidden=True)
        if train:
            (ce + 0.01 * lb).backward()
        ces.append(float(ce)); hs.append(h.detach())
    return ces, torch.cat(hs, 1)

erng = np.random.default_rng(123)
EV = {k: batch(erng, [k] * 64) for k in ('copy_same', 'copy_cross', 'ar')}
TV = torch.from_numpy(np.stack([txv[i * (L + 1):(i + 1) * (L + 1)] for i in range(48)]))

@torch.no_grad()
def evaluate():
    eng.eval(); r = {}
    for k, (ids, sc) in EV.items():
        _, h = run_rows(ids, False); lg = eng.output_head(h)
        r[k] = float(((lg.argmax(-1) == ids[:, 1:]) & sc).sum() / sc.sum())
    ces, _ = run_rows(TV, False); r['text_val_nll'] = sum(ces) / len(ces)
    eng.train(); return r

rng = np.random.default_rng(7); t0 = time.time(); f = open(OUT, 'w')
f.write(json.dumps(dict(arm=ARM, gate_init=float(os.environ.get('GATE_INIT', 0)), ngram=int(os.environ.get('NGRAM', 1)), mix=MIX, lr=float(os.environ.get('LR', 2e-3)), cfg=CFG, params=nparam, head_params=nhead, steps=STEPS)) + '\n'); f.flush()
for step in range(1, STEPS + 1):
    ids, _ = batch(rng); opt.zero_grad(set_to_none=True)
    ces, _ = run_rows(ids, True)
    torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0); opt.step()
    if step % 50 == 0 or step == STEPS:
        r = dict(step=step, train_ce=sum(ces) / 2, secs=round(time.time() - t0), **evaluate())
        if ARM == 'copy': r['gate_rms'] = float(eng.copy_head.gate.pow(2).mean().sqrt())
        f.write(json.dumps(r) + '\n'); f.flush(); print(r, flush=True)
