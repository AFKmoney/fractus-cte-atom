"""Externalization test (inference only; fp32 fixed generation path; trainer untouched). Per checkpoint (CKPT, TAG),
resumable per record into OUT.

  tf     : the 32 v4_eval val windows (127 warm + 1,024 scored ids, same rng). Teacher-forced top-1 / top-5 on the scored
           ids. Long range: NLL on the ids of grid chunk 7 (rows predicting ids 897..1024) with the full carry (7 chunks
           of real context before) vs the carry reset at chunk 6 (128-255 ids of context) vs at chunk 7 (0-127 ids, in-chunk
           only). A mid-document zero carry is a state training only sees at BOS, so 'reset' is an upper bound on the loss of
           forgetting, not a clean ablation.
  compl  : word completion. Real words (letters, len>=5, in words_alpha, starting after a non-letter) in the scored
           region of the first 8 windows of each source (up to NCOMP per window). For k=1,2,3 given bytes, greedy decode from
           the true prefix (committed carry + open chunk, exactly as generation) until a non-letter; exact = the word.
           Consistency: the first greedy step's logits vs the TF row at the same position (must be equal: same function).
  free   : 6 val contexts (2 per source fineweb/wiki/tinystories, 256-id prefix after a 127-id doc tail) x 64 new ids,
           greedy and min-p (T 0.7, p 0.1, seed 0). Real-word rate of the generated words vs the true continuation, and
           consistency: per-step generation logits vs a teacher-forced pass over (context + generated ids) on the fixed grid.
"""
import os, sys, json, time, re
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fractus.fixed_generation import load_checkpoint, FixedGenerator, SEQ, show

torch.set_num_threads(int(os.environ.get('THREADS', '7')))
CK, TAG = os.environ['CKPT'], os.environ['TAG']
OUT = os.environ.get('OUT', '/workspace/scratch/extern/extern.jsonl')
NCOMP = int(os.environ.get('NCOMP', 6))
VB = '/workspace/v4build'; SRC = ['fineweb', 'wiki', 'tinystories', 'pg19']; W, L = 127, 1024
PAD, BOS, EOS = 256, 257, 258
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: d = json.loads(l); done.add((d['tag'], d['kind'], d['key']))
        except Exception: pass
def emit(r):
    open(OUT, 'a').write(json.dumps(r, ensure_ascii=False) + '\n'); print(json.dumps(r, ensure_ascii=False)[:240], flush=True)
eng, cfg = load_checkpoint(CK, torch.float32); G = FixedGenerator(eng, warm=W)
words_ok = set(open('/workspace/words_alpha.txt').read().split())
val = {s: np.fromfile(f'{VB}/{s}_val.i16', dtype=np.int16).astype(np.int64) for s in SRC}
rng = np.random.default_rng(0); VW = []
for s in SRC:
    for wi, st in enumerate(rng.integers(0, len(val[s]) - (W + L + 1), 8)):
        VW.append((f'{s}:{wi}', s, val[s][st:st + W + L].tolist()))
isl = lambda t: (65 <= t <= 90) or (97 <= t <= 122)

def run_chunks(ids):
    """cold grid from ids[0]; returns logits (len, V) and the snapshot before each chunk."""
    G.cold(); snaps, out = [], []
    for j in range(0, len(ids), SEQ):
        snaps.append(G.get()); out.append(G.chunk(ids[j:j + SEQ]))
    return torch.cat(out), snaps

def greedy_from(snaps, seq, maxn, stop_nonletter=True, sample=None, g=None):
    """continue seq (list) from the committed snapshot of its last grid chunk; returns ids, per-step logits."""
    seq = list(seq); com = ((len(seq) - 1) // SEQ) * SEQ; G.put(snaps[com // SEQ]); outs, lgs = [], []
    for _ in range(maxn):
        s = G.get(); lg = G.chunk(seq[com:])[-1].clone(); G.put(s); lgs.append(lg.clone())
        lg[PAD] = -1e9; lg[BOS] = -1e9
        if sample is None:
            nid = int(lg.argmax())
        else:
            T, pmin = sample; p = torch.softmax(lg / T, -1); p[p < pmin * p.max()] = 0
            nid = int(torch.multinomial(p / p.sum(), 1, generator=g))
        if stop_nonletter and not isl(nid): break
        outs.append(nid); seq.append(nid)
        if len(seq) - com > SEQ:
            G.chunk(seq[com:com + SEQ]); com += SEQ
    return outs, lgs

for key, src, ctx in VW:
    if (TAG, 'tf', key) not in done:
        t = time.time(); lg, snaps = run_chunks(ctx); lg = lg[:-1]; tgt = torch.tensor(ctx[1:])
        sc = slice(W - 1, None); top5 = lg.topk(5, -1).indices
        nll = F.cross_entropy(lg, tgt, reduction='none')
        r7 = slice(7 * SEQ - 1, 8 * SEQ - 1)          # rows predicting ids 896..1023 (= chunk 7 targets)
        def trunc(start_chunk):
            G.cold(); o = []
            for j in range(start_chunk * SEQ, 8 * SEQ, SEQ): o.append(G.chunk(ctx[j:j + SEQ]))
            o = torch.cat(o)                            # rows for ids start..1023
            rows = o[(7 - start_chunk) * SEQ - 1:(8 - start_chunk) * SEQ - 1] if start_chunk < 7 else None
            return rows
        # chunk-7 NLL with reset at chunk 6: rows 127..254 of the 2-chunk pass = predictions for ids 7*128 .. 8*128-1
        o6 = trunc(6); n6 = F.cross_entropy(o6, torch.tensor(ctx[7 * SEQ:8 * SEQ]), reduction='none')
        G.cold(); o7 = G.chunk(ctx[7 * SEQ:8 * SEQ])  # reset at chunk 7: rows 0..126 predict ids 7*128+1..
        n7 = F.cross_entropy(o7[:-1], torch.tensor(ctx[7 * SEQ + 1:8 * SEQ]), reduction='none')
        nfull = nll[7 * SEQ - 1:8 * SEQ - 1]           # predictions for ids 896..1023
        emit(dict(tag=TAG, kind='tf', key=key, src=src, top1=float((lg[sc].argmax(-1) == tgt[sc]).float().mean()),
                  top5=float((top5[sc] == tgt[sc, None]).any(-1).float().mean()), nll=float(nll[sc].mean()),
                  lr_full=float(nfull.mean()), lr_reset6=float(n6.mean()),
                  lr_full_from1=float(nfull[1:].mean()), lr_reset7_from1=float(n7.mean()), secs=round(time.time() - t, 1)))
    if src == 'pg19' or (TAG, 'compl', key) in done:
        continue
    t = time.time(); lg, snaps = run_chunks(ctx)
    pos = [i for i in range(W, len(ctx) - 12) if isl(ctx[i]) and not isl(ctx[i - 1])]
    cands = []
    for p in pos:
        e = p
        while e < len(ctx) and isl(ctx[e]): e += 1
        w = bytes(ctx[p:e]).decode('ascii', 'replace')
        if e < len(ctx) and len(w) >= 5 and w.lower() in words_ok: cands.append((p, e, w))
    pick = [cands[i] for i in np.random.default_rng(1).permutation(len(cands))[:NCOMP]]
    res = []
    for p, e, w in pick:
        for k in (1, 2, 3):
            gen, lgs = greedy_from(snaps, ctx[:p + k], 20)
            comp = bytes(ctx[p:p + k] + gen).decode('ascii', 'replace')
            res.append(dict(word=w, k=k, completion=comp, exact=comp == w, tf_top1_next=bool(int(lg[p + k - 1].argmax()) == ctx[p + k]),
                            first_step_vs_tf_maxabs=float((lgs[0] - lg[p + k - 1]).abs().max())))
    emit(dict(tag=TAG, kind='compl', key=key, src=src, items=res, secs=round(time.time() - t, 1)))

# ---- free generation
for src in ('fineweb', 'wiki', 'tinystories'):
    a = val[src]; b = np.flatnonzero(a == BOS); lens = np.diff(np.append(b, len(a)))
    for k in [int(x) for x in np.flatnonzero(lens > 400)[[3, 7]]]:
        ctx = a[b[k] - W:b[k]].tolist() + a[b[k]:b[k] + 256].tolist(); true = a[b[k] + 256:b[k] + 320].tolist()
        for dec, sample in (('greedy', None), ('minp_T0.7_p0.1', (0.7, 0.1))):
            key = f'{dec}:{src}_doc{k}'
            if (TAG, 'free', key) in done: continue
            t = time.time(); _, snaps = run_chunks(ctx)
            gen, lgs = greedy_from(snaps, ctx, 64, stop_nonletter=False, sample=sample, g=torch.Generator().manual_seed(0))
            tf, _ = run_chunks(ctx + gen)
            cons = max(float((lgs[i] - tf[len(ctx) - 1 + i]).abs().max()) for i in range(len(gen)))
            lp = torch.log_softmax(tf[len(ctx) - 1:len(ctx) - 1 + len(gen)], -1)
            gen_lp = float(lp[torch.arange(len(gen)), torch.tensor(gen)].mean())
            ttf, _ = run_chunks(ctx + true); lpt = torch.log_softmax(ttf[len(ctx) - 1:len(ctx) - 1 + len(true)], -1)
            true_lp = float(lpt[torch.arange(len(true)), torch.tensor(true)].mean())
            def rate(ids):
                ws = re.findall(r'[A-Za-z]{2,}', bytes(i for i in ids if i < 256).decode('utf-8', 'replace'))
                return (sum(x.lower() in words_ok for x in ws) / len(ws)) if ws else None, len(ws)
            rg, ng = rate(gen); rt, nt = rate(true)
            emit(dict(tag=TAG, kind='free', key=key, decoder=dec, src=src, generated=show(gen), true=show(true),
                      real_word_rate=rg, n_words=ng, true_real_word_rate=rt, true_n_words=nt,
                      gen_vs_tf_logits_maxabs=cons, mean_logp_generated=gen_lp, mean_logp_true=true_lp,
                      secs=round(time.time() - t, 1)))
print('ALLDONE', TAG, flush=True)
