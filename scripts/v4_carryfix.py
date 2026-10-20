"""Long-carry diagnosis + inference-only carry policies (no weight change; trainer untouched; attn_S/attn_z NOT bounded
by norm at inference). Resumable per record into OUT.

Carry = per block (attn_S, attn_z): a plain running SUM over every past id of phi(k) v^T and phi(k) (phi = elu+1),
read by y_t = phi(q_t) S_t / phi(q_t) z_t. Policies change what the next chunk READS:
  base        : unchanged (full sum since the cold start)
  decayG      : S,z *= G at every chunk boundary (exponential forgetting, joint so the S/z ratio is kept)
  winK        : committed state = cold replay of only the last K committed chunks (sliding window, exact recompute)
  resetN      : cold restart every N committed chunks (state = replay of the chunks since the last multiple of N)
Phases:
  diag : base, 8 non-pg19 windows: per chunk index, per block ||S||_F, ||z||, and the share of the attention denominator
         phi(q).z (and of ||numerator||) that comes from the carry rather than from the current chunk.
  tf   : NLL by chunk index (0..8) on the 24 non-pg19 v4_eval windows (rng 0, 127+1024 ids) per policy.
  gen  : greedy and min-p (T0.7, p0.1, seed 0) 40 ids, warm = 64 real corpus ids before a BOS: 3 CONT (256-id prefix),
         3 LONG CONT (1024-id prefix, generation in grid chunk 8 -> the long carry matters), 3 prompts.
"""
import os, sys, json, time, re, difflib
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fractus.fixed_generation import load_checkpoint, FixedGenerator, SEQ, show

torch.set_num_threads(int(os.environ.get('THREADS', '8')))
CK, TAG = os.environ['CKPT'], os.environ['TAG']
OUT = os.environ.get('OUT', '/workspace/scratch/carry/carry.jsonl')
PHASES = os.environ.get('PHASES', 'diag,tf,gen').split(',')
TFPOL = os.environ.get('TFPOL', 'base,decay0.95,decay0.9,decay0.8,decay0.7,decay0.5,win2,win3,reset2,reset4').split(',')
GENPOL = os.environ.get('GENPOL', 'base,decay0.8,win2').split(',')
VB = '/workspace/v4build'; PAD, BOS, EOS = 256, 257, 258
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: d = json.loads(l); done.add((d['tag'], d['kind'], d['key']))
        except Exception: pass
def emit(r):
    open(OUT, 'a').write(json.dumps(r, ensure_ascii=False) + '\n'); print(json.dumps(r, ensure_ascii=False)[:260], flush=True)

eng, cfg = load_checkpoint(CK, torch.float32); G = FixedGenerator(eng, warm=64)
words_ok = set(open('/workspace/words_alpha.txt').read().split())
POL = ['base', 1.0]
def parse(p):
    if p == 'base': return ('base', 0)
    for k in ('decay', 'win', 'reset'):
        if p.startswith(k): return (k, float(p[len(k):]))
    raise ValueError(p)

# read-side transform: the chunk reads G*carry (decay); everything else is replay logic below
for blk in eng.blocks:
    orig = blk._build_carry
    def bc(B, blk=blk, orig=orig):
        if POL[0] == 'decay':
            blk.attn_S = blk.attn_S * POL[1]; blk.attn_z = blk.attn_z * POL[1]
        return orig(B)
    blk._build_carry = bc

# diagnostics hook on the attention kernel
REC = None
for bi, blk in enumerate(eng.blocks):
    a = blk.attn; f0 = a._linear_attention_causal_vectorized
    def hook(q, k, v, carry=None, bi=bi, f0=f0, **kw):
        out = f0(q, k, v, carry=carry, **kw)
        if REC is not None and carry is not None:
            S0, z0 = carry
            zc = torch.cumsum(k, 1)                               # in-chunk z_t
            dc, d0 = (q * zc).sum(-1), (q * z0[:, None]).sum(-1)    # (BH, L)
            num0 = torch.einsum('blp,bpq->blq', q, S0).norm(dim=-1)
            numc = (out[0] * (dc + d0)[..., None]).sub(torch.einsum('blp,bpq->blq', q, S0)).norm(dim=-1)
            REC.setdefault(bi, []).append(dict(S=float(S0.flatten(1).norm(dim=-1).mean()), z=float(z0.norm(dim=-1).mean()),
                den_share=float((d0 / (d0 + dc)).mean()), num_share=float((num0 / (num0 + numc + 1e-20)).mean()),
                den_share_last=float((d0 / (d0 + dc))[:, -1].mean())))
        return out
    a._linear_attention_causal_vectorized = hook

def commit_state(chunks, pol):
    """state after committing `chunks` (list of id lists) under policy pol (base/decay: sequential)."""
    k, p = pol; G.cold()
    if k in ('base', 'decay'):
        for c in chunks: G.chunk(c)
        return G.get()
    if k == 'win': use = chunks[-int(p):] if chunks else []
    else: m = len(chunks) % int(p); use = chunks[len(chunks) - m:] if m else []
    for c in use: G.chunk(c)
    return G.get()

def tf_nll(ids, pol):
    """per-id NLL rows (len-1) with policy; incremental for base/decay, replay for win/reset."""
    k, p = pol; POL[:] = [k, p]; chunks = [ids[j:j + SEQ] for j in range(0, len(ids), SEQ)]; out = []
    if k in ('base', 'decay'):
        G.cold(); out = [G.chunk(c) for c in chunks]
    else:
        for i, c in enumerate(chunks):
            G.put(commit_state(chunks[:i], pol)) if i else G.cold(); out.append(G.chunk(c))
    POL[:] = ['base', 0]
    lg = torch.cat(out)[:-1]
    return F.cross_entropy(lg, torch.tensor(ids[1:]), reduction='none')

val = {s: np.fromfile(f'{VB}/{s}_val.i16', dtype=np.int16).astype(np.int64) for s in ['fineweb', 'wiki', 'tinystories', 'pg19']}
rng = np.random.default_rng(0); VW = []
for s in ['fineweb', 'wiki', 'tinystories', 'pg19']:
    for wi, st in enumerate(rng.integers(0, len(val[s]) - 1152, 8)):
        if s != 'pg19': VW.append((f'{s}:{wi}', val[s][st:st + 1151].tolist()))
bychunk = lambda nll: [float(nll[max(0, 128 * c - 1):128 * c + 127].mean()) for c in range(9)]

if 'diag' in PHASES and (TAG, 'diag', 'base') not in done:
    agg = {}
    for key, ctx in VW[::3]:
        G.cold()
        for c in range(9):
            REC = {}; G.chunk(ctx[c * SEQ:(c + 1) * SEQ])
            for bi, lst in REC.items():
                for f, v in lst[0].items(): agg.setdefault((bi, f), [[] for _ in range(9)])[c].append(v)
        REC = None
    res = {}
    for (bi, f), cols in agg.items(): res.setdefault(f'block{bi}', {})[f] = [float(np.mean(x)) for x in cols]
    emit(dict(tag=TAG, kind='diag', key='base', n_windows=len(VW[::3]), per_block=res))
REC = None

if 'tf' in PHASES:
    for pn in TFPOL:
        if (TAG, 'tf', pn) in done: continue
        sub = int(os.environ.get('TFSUB', 1)); t = time.time(); rows = [bychunk(tf_nll(ctx, parse(pn))) for key, ctx in VW[::sub]]
        m = np.mean(rows, 0)
        emit(dict(tag=TAG, kind='tf', key=pn, nll_by_chunk=m.tolist(), mean_c0_8=float(m.mean()), mean_c3_8=float(m[3:].mean()),
                  n_windows=len(rows), tfsub=sub, per_window=rows, secs=round(time.time() - t, 1)))

if 'gen' in PHASES:
    fw = val['fineweb']; bos = np.flatnonzero(fw == 257); WARM = fw[bos[1] - 64:bos[1]].tolist()
    cases = []
    for src in ('fineweb', 'wiki', 'tinystories'):
        a = val[src]; b = np.flatnonzero(a == 257); lens = np.diff(np.append(b, len(a)))
        k = int(np.flatnonzero(lens > 400)[3]); cases.append((f'cont_{src}_doc{k}', a[b[k] - 64:b[k] + 256].tolist(), a[b[k] + 256:b[k] + 296].tolist()))
    for src in ('fineweb', 'wiki', 'tinystories'):
        a = val[src]; b = np.flatnonzero(a == 257); lens = np.diff(np.append(b, len(a)))
        k = int(np.flatnonzero(lens > 1200)[1]); cases.append((f'long_{src}_doc{k}', a[b[k] - 64:b[k] + 1024].tolist(), a[b[k] + 1024:b[k] + 1064].tolist()))
    cases += [(p, WARM + G.encode_prompt(p), None) for p in ['The capital of France is', 'Once upon a time', 'Hello']]
    def gen(ctx, pol, n=40, sample=None):
        k, p = pol; POL[:] = [k, p]; g = torch.Generator().manual_seed(0)
        seq = list(ctx); com = ((len(seq) - 1) // SEQ) * SEQ
        chunks = [seq[j:j + SEQ] for j in range(0, com, SEQ)]; snap = commit_state(chunks, pol); out = []
        for _ in range(n):
            G.put(snap); lg = G.chunk(seq[com:])[-1].clone(); lg[PAD] = -1e9; lg[BOS] = -1e9
            if sample is None: nid = int(lg.argmax())
            else:
                T, pm = sample; pr = torch.softmax(lg / T, -1); pr[pr < pm * pr.max()] = 0
                nid = int(torch.multinomial(pr / pr.sum(), 1, generator=g))
            out.append(nid); seq.append(nid)
            if nid == EOS: break
            if len(seq) - com > SEQ:
                chunks.append(seq[com:com + SEQ]); com += SEQ
                if k in ('base', 'decay'): G.put(snap); G.chunk(chunks[-1]); snap = G.get()
                else: snap = commit_state(chunks, pol)
        POL[:] = ['base', 0]; return out
    for pn in GENPOL:
        for dn, smp in (('greedy', None), ('minp_T0.7_p0.1', (0.7, 0.1))):
            for label, ctx, true in cases:
                if dn != 'greedy' and not label.startswith(os.environ.get('MINP_CASES', 'long_')): continue
                key = f'{pn}:{dn}:{label}'
                if (TAG, 'gen', key) in done: continue
                t = time.time(); o = gen(ctx, parse(pn), sample=smp)
                txt = bytes(i for i in o if i < 256).decode('utf-8', 'replace'); ws = re.findall(r'[A-Za-z]{2,}', txt)
                r = dict(tag=TAG, kind='gen', key=key, policy=pn, decoder=dn, case=label, unique40=len(set(o[:40])), generated=show(o),
                         real_word_rate=(sum(w.lower() in words_ok for w in ws) / len(ws)) if ws else None, secs=round(time.time() - t, 1))
                if true is not None:
                    r.update(exact=sum(a == b for a, b in zip(o, true)), seqmatch=difflib.SequenceMatcher(None, o, true).ratio(), true=show(true))
                emit(r)
print('ALLDONE', TAG, flush=True)
