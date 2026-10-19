"""Copy head, test 3: train ONLY the readout copy head (+ gate) of the real v4 checkpoint, on cached body states.

Exact: the head sits after the last CTEBlock and the body never sees it (fractus/nn/copy_head.py), so
logits = (h_body + gate * head(h_body)) @ output_head.T with h_body cached by scripts/copyhead_1b_prep.py.
Train: cache/train (512-id windows of atom_corpus_v4 train_006, 4 grid chunks each, window cache carried across
chunks and detached, as in a trainer). 10% of the train windows are held out for model selection.
Eval: the 32 v4_eval val windows (127+1024 ids, NLL on the last 1024), overall / per source, and the copy probe
(word classes first / repeat_same_chunk / repeat_earlier_chunk, NLL on bytes 2..n). Baseline = gate 0 = v4.
Checkpointed: head state + eval rows every EVAL steps (resumable).
"""
import os, sys, json, time, glob, re
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fractus.nn.copy_head import CopyHead

torch.set_num_threads(int(os.environ.get('THREADS', '4')))
D = os.environ.get('DIR', '/workspace/scratch/copyhead')
TAG = os.environ.get('TAG', 'h8d64m128')
NH, DH, MEM = int(os.environ.get('NH', 8)), int(os.environ.get('DH', 64)), int(os.environ.get('MEM', 128))
STEPS, EVAL, LR = int(os.environ.get('STEPS', 600)), int(os.environ.get('EVAL', 50)), float(os.environ.get('LR', 1e-3))
CK = os.environ.get('CKPT', '/workspace/ckpt_v4_good/fractus_atom_1b_gpu0.pt')
SEQ = 128
OUT = f'{D}/train_{TAG}.jsonl'; HEAD = f'{D}/head_{TAG}.pt'
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)

sd = torch.load(CK, map_location='cpu', weights_only=False, mmap=True)['model_state']
W = sd[[k for k in sd if k.endswith('output_head.weight')][0]].float().clone(); del sd
files = sorted(glob.glob(f'{D}/cache/train/b*.pt'))
TR = [torch.load(f) for f in files]
n_hold = max(1, len(TR) // 10); HOLD, TR = TR[:n_hold], TR[n_hold:]
VAL = [torch.load(f) for f in sorted(glob.glob(f'{D}/cache/val/*.pt'))]
log('train batches', len(TR), 'holdout', len(HOLD), 'val windows', len(VAL))

torch.manual_seed(0)
NG = int(os.environ.get('NGRAM', 1))
head = CopyHead(W.shape[1], NH, DH, MEM, ngram=NG)
head.diag_only = os.environ.get('DIAG', '0') == '1'
head.ghost = float(os.environ.get('GHOST', 0))   # see CopyHead.ghost
GATE_INIT = float(os.environ.get('GATE_INIT', 0))   # 0 = exact v4 at init; >0 = live head from step 0
with torch.no_grad(): head.gate.fill_(GATE_INIT)
opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.0)
step0 = 0
if os.path.exists(HEAD):
    s = torch.load(HEAD); head.load_state_dict(s['head']); opt.load_state_dict(s['opt']); step0 = s['step']

def run(h, ids, grad=True):
    """h (B, L, D), ids (B, L+1) or (B, L) -> per-position NLL (B, L) or (B, L-1)."""
    head.reset_state(h.shape[0]); out = []
    for j in range(0, h.shape[1], SEQ):
        h2 = head.attend(h[:, j:j + SEQ])
        tgt = ids[:, j + 1:j + SEQ + 1]; lg = (h2 @ W.t())[:, :tgt.shape[1]]   # val: last row has no target
        out.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                                   reduction='none').view(h.shape[0], -1))
    return torch.cat(out, 1)

def word_classes(ctx, start):
    pos = [i for i, t in enumerate(ctx) if t < 256]; b = bytes(ctx[i] for i in pos); last = {}
    for m in re.finditer(rb'[A-Za-z]{5,}', b):
        w = m.group(); ids = [pos[k] for k in range(m.start(), m.end())]
        prev = last.get(w); last[w] = ids[0]
        if ids[0] < start or ids[0] == 0: continue
        c = 'first' if prev is None else ('repeat_same_chunk' if prev // SEQ == ids[0] // SEQ else 'repeat_earlier_chunk')
        yield c, ids

CLS = {}
for v in VAL:
    ctx = v['ids'].tolist(); CLS[v['key']] = [(c, [i - 1 for i in ids][1:]) for c, ids in word_classes(ctx, 127)]

@torch.no_grad()
def evaluate():
    r = {'hold_nll': float(torch.cat([run(b['h'], b['ids'], False).flatten() for b in HOLD]).mean())}
    per_src, cls = {}, {}
    for v in VAL:
        nll = run(v['h'][None], v['ids'][None, :], False)[0]          # rows 0..L-2 valid (last row has no target)
        src = v['key'].split(':')[0]
        per_src.setdefault(src, []).append(float(nll[126:].mean()))      # rows predicting ctx[127:], as v4_eval
        for c, rows in CLS[v['key']]:
            e = cls.setdefault(c, [0.0, 0]); e[0] += float(nll[rows].sum()); e[1] += len(rows)
    r['val_by_src'] = {s: float(np.mean(x)) for s, x in per_src.items()}
    allw = [x for s in per_src.values() for x in s]
    r['val_mean'] = float(np.mean(allw)); r['val_median'] = float(np.median(allw))
    r['val_mean_no_outliers'] = float(np.mean([x for x in allw if x < 6]))
    r['copy_rest_nll'] = {c: e[0] / max(e[1], 1) for c, e in cls.items()}
    r['copy_rest_n'] = {c: e[1] for c, e in cls.items()}
    r['gate_rms'] = float(head.gate.pow(2).mean().sqrt())
    return r

f = open(OUT, 'a')
if step0 == 0:
    r = dict(tag=TAG, step=0, diag_only=head.diag_only, ghost=head.ghost, gate_init=GATE_INIT, ngram=NG, nh=NH, dh=DH, mem=MEM, lr=LR, head_params=sum(p.numel() for p in head.parameters()), **evaluate())
    f.write(json.dumps(r) + '\n'); f.flush(); log(r)
rng = np.random.default_rng(step0); t0 = time.time(); tl = []
for step in range(step0 + 1, STEPS + 1):
    b = TR[rng.integers(len(TR))]
    loss = run(b['h'], b['ids']).mean()
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); tl.append(float(loss))
    if step % EVAL == 0 or step == STEPS:
        r = dict(tag=TAG, step=step, train_nll=float(np.mean(tl)), secs=round(time.time() - t0), **evaluate()); tl = []
        f.write(json.dumps(r) + '\n'); f.flush(); log(r)
        torch.save(dict(head=head.state_dict(), opt=opt.state_dict(), step=step), HEAD + '.tmp'); os.replace(HEAD + '.tmp', HEAD)
log('done')
