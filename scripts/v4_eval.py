"""v4 evaluation on real-text val data (inference only; fp32, fixed generation path; trainer untouched).

Per checkpoint (CKPT, TAG), resumable (records already in OUT are skipped), checkpointed per record:
  * val windows: 8 random windows of 127+1024 ids from each per-source val file; TF NLL/top-1 on the last 1024 ids.
    The same TF pass feeds the copy probe: words (ASCII letters, >=5) in the scored region are classed as
    first mention / repeat whose previous occurrence is in the SAME 128-id grid chunk / repeat whose previous occurrence
    is only in an EARLIER chunk (reachable only through the carry); NLL/top-1 over the word's bytes, and over bytes 2..n.
  * generation: greedy, beam 8, min-p T0.7 p0.1 no-repeat-4 on 3 CONT val docs (256-id prefix, 40 ids) and 3 short
    prompts, warm start = real 127-id tail of a val doc (ending <B6><EOS>).
"""
import os, sys, json, time, re, difflib
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fractus.fixed_generation import load_checkpoint, FixedGenerator, SEQ, show
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_best import Decoder

torch.set_num_threads(int(os.environ.get('THREADS', '7')))
CK, TAG = os.environ['CKPT'], os.environ['TAG']
OUT = os.environ.get('OUT', '/workspace/scratch/dec/v4_eval.jsonl')
VB = '/workspace/v4build'
SRC = ['fineweb', 'wiki', 'tinystories', 'pg19']
NW, W, L = int(os.environ.get('NWIN', 8)), 127, 1024
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: d = json.loads(l); done.add((d['tag'], d['kind'], d['key']))
        except Exception: pass
def emit(r):
    open(OUT, 'a').write(json.dumps(r, ensure_ascii=False) + '\n'); print(json.dumps(r, ensure_ascii=False)[:300], flush=True)
eng, cfg = load_checkpoint(CK, torch.float32); G = FixedGenerator(eng, warm=W); dec = Decoder(G)
words_ok = set(open('/workspace/words_alpha.txt').read().split())
val = {s: np.fromfile(f'{VB}/{s}_val.i16', dtype=np.int16).astype(np.int64) for s in SRC}

def word_classes(ctx, start):
    """yield (class, [id positions of the word bytes]) for words whose first byte index >= start."""
    pos = [i for i, t in enumerate(ctx) if t < 256]; b = bytes(ctx[i] for i in pos)
    last = {}
    for m in re.finditer(rb'[A-Za-z]{5,}', b):
        w = m.group(); ids = [pos[k] for k in range(m.start(), m.end())]
        prev = last.get(w); last[w] = ids[0]
        if ids[0] < start or ids[0] == 0: continue
        if prev is None: c = 'first'
        elif prev // SEQ == ids[0] // SEQ: c = 'repeat_same_chunk'
        else: c = 'repeat_earlier_chunk'
        yield c, ids, (ids[0] - prev) if prev is not None else None

rng = np.random.default_rng(0)
for s in SRC:
    a = val[s]; starts = rng.integers(0, len(a) - (W + L + 1), NW)
    for wi, st in enumerate(starts):
        key = f'{s}:{wi}'
        if (TAG, 'valwin', key) in done: continue
        t = time.time(); ctx = a[st:st + W + L].tolist()
        lg = G.tf_logits(ctx)[:-1]; tgt = torch.tensor(ctx[1:])
        nll = F.cross_entropy(lg, tgt, reduction='none'); t1 = (lg.argmax(-1) == tgt).float()
        sc = slice(W - 1, None)   # rows predicting ctx[W:]
        acc = {}
        for c, ids, dist in word_classes(ctx, W):
            rows = [i - 1 for i in ids]; rest = rows[1:]
            e = acc.setdefault(c, dict(n=0, nll=0.0, top1=0.0, n_rest=0, nll_rest=0.0, top1_rest=0.0, n_first=0, nll_first=0.0, top1_first=0.0, dists=[]))
            e['n'] += len(rows); e['nll'] += float(nll[rows].sum()); e['top1'] += float(t1[rows].sum())
            e['n_rest'] += len(rest); e['nll_rest'] += float(nll[rest].sum()); e['top1_rest'] += float(t1[rest].sum())
            e['n_first'] += 1; e['nll_first'] += float(nll[rows[0]]); e['top1_first'] += float(t1[rows[0]])
            if dist is not None: e['dists'].append(dist)
        emit(dict(tag=TAG, kind='valwin', key=key, src=s, start=int(st), nll=float(nll[sc].mean()), top1=float(t1[sc].mean()),
                  bytes=int(sum(1 for x in ctx[W:] if x < 256)), ids=L, words=acc, secs=round(time.time() - t, 1)))

# ---- generation
fw = val['fineweb']; bos = np.flatnonzero(fw == 257)
WARM = fw[bos[1] - W:bos[1]].tolist()       # real tail of val doc 0 (ends <B6><EOS>)
cont = []
for src in ('fineweb', 'wiki', 'tinystories'):
    a = val[src]; b = np.flatnonzero(a == 257); lens = np.diff(np.append(b, len(a)))
    k = int(np.flatnonzero(lens > 400)[3])
    cont.append((f'{src}_doc{k}', a[b[k] - W:b[k]].tolist() + a[b[k]:b[k] + 256].tolist(), a[b[k] + 256:b[k] + 296].tolist()))
prompts = [(p, WARM + G.encode_prompt(p), None) for p in ['The capital of France is', 'Once upon a time', 'Hello']]
DECS = [('greedy', 'greedy', {}), ('minp_T0.7_p0.1_ng4', 'sample', dict(T=0.7, min_p=0.1, ngram=4)), ('beam8', 'beam', dict(W=8))]
for name, fn, kw in DECS:
    for label, ctx, true in cont + prompts:
        key = f'{name}:{label}'
        if (TAG, 'gen', key) in done: continue
        t = time.time(); gen = getattr(dec, fn)(ctx, n=40, **kw)
        txt = bytes(i for i in gen if i < 256).decode('utf-8', 'replace'); ws = re.findall(r'[A-Za-z]{2,}', txt)
        r = dict(tag=TAG, kind='gen', key=key, decoder=name, case=label, unique40=len(set(gen[:40])), generated=show(gen), text=txt,
                 real_word_rate=(sum(w.lower() in words_ok for w in ws) / len(ws)) if ws else None, secs=round(time.time() - t, 1))
        if true is not None:
            r.update(exact=sum(p == q for p, q in zip(gen, true)), seqmatch=difflib.SequenceMatcher(None, gen, true).ratio(), true=show(true))
        emit(r)
print('ALLDONE', TAG, flush=True)
