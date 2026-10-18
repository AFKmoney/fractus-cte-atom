"""Fixed generation + variant evaluation (docs/DEEP_DIVE_GENERATION.md). Inference only; trainer untouched.

Uses fractus/fixed_generation.py: fixed 128-id grid, committed carry, open chunk re-forwarded,
trainer-identical load, optional warm start (corpus doc tail before BOS), optional final norm.
Variants:  a = fixes only (cold BOS)   b = fixes + warm start   c = fixes + warm start + final LayerNorm
Usage: CKPT=<.pt or bf16raw dir> OUT=gen_fixed.jsonl python3 scripts/gen_fixed.py
Env: DOCS (0,5000,10000)  N (40)  WARM (64)  STAGES (warm,norm,gen)  PRIMER (corpus .i16)
"""
import os, sys, json, time, difflib
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fractus.fixed_generation import load_checkpoint, FixedGenerator, WARM_TAIL, SEQ, show
torch.set_num_threads(int(os.environ.get('THREADS', '7')))
CK = os.environ.get('CKPT', '/workspace/ckpt_v3_85000/fractus_atom_1b_gpu0.pt')
OUT = os.environ.get('OUT', '/workspace/gen_fixed.jsonl')
N = int(os.environ.get('N', '40')); W = int(os.environ.get('WARM', '64'))
STAGES = os.environ.get('STAGES', 'warm,norm,gen').split(',')
CORPUS = os.environ.get('PRIMER', '/workspace/lmsys/ref/atom_corpus/reasoning__syllogisms.i16')
DOCS = [int(x) for x in os.environ.get('DOCS', '0,5000,10000').split(',') if x]
PROMPTS = os.environ['PROMPTS'].split('|') if os.environ.get('PROMPTS') else ['The capital of France is', 'Once upon a time', 'Hello']

def emit(r):
    print(json.dumps(r, ensure_ascii=False), flush=True); open(OUT, 'a').write(json.dumps(r, ensure_ascii=False) + '\n')

t0 = time.time(); eng, cfg = load_checkpoint(CK); G = FixedGenerator(eng, final_norm=None, warm=W)
emit({'kind': 'meta', 'ckpt': CK, 'ids_processed': cfg.get('ids_processed'), 'load_s': round(time.time() - t0, 1), 'strict_load': True})
a = np.fromfile(CORPUS, dtype=np.int16).astype(np.int64); bos = np.where(a == 257)[0]
docs = {d: a[int(bos[d]):int(bos[d]) + 257 + N].tolist() for d in DOCS}

def tf(ctx_pre, doc, carry_pre=None):
    """TF logits of doc given ids before it. carry_pre: ids committed as earlier chunk(s) WITH carry kept (not reset)."""
    G.cold(); rows = []
    if carry_pre:
        for j in range(0, len(carry_pre), SEQ): G.chunk(carry_pre[j:j + SEQ])
    seq = ctx_pre + doc
    for j in range(0, len(seq), SEQ): rows.append(G.chunk(seq[j:j + SEQ]))
    return torch.cat(rows, 0)[len(ctx_pre):]

def m(lg, doc, i, j):
    tg = torch.tensor(doc[i + 1:j + 1]); l = lg[i:j]
    return round(float(F.cross_entropy(l, tg)), 3), round(float((l.argmax(-1) == tg).float().mean()), 3)

WARMS = {'W0_coldBOS': ([], None), 'W8': (WARM_TAIL[-8:], None), 'W32': (WARM_TAIL[-32:], None), 'W64': (WARM_TAIL[-64:], None),
         'W127': (WARM_TAIL[-127:], None),
         'Wctx_hotcarry_BOSpos0': ([], WARM_TAIL[-128:]),                 # previous chunk committed, carry kept, BOS starts the next chunk
         'Wctx_hotcarry_BOSpos64': (WARM_TAIL[-64:], WARM_TAIL[-192:-64])}  # previous chunk with carry kept + 64-id tail before BOS
if 'warm' in STAGES:
    for name, (pre, cp) in WARMS.items():
        r = {'kind': 'tf_warm', 'variant': name}
        for d, doc in docs.items():
            lg = tf(pre, doc, cp)
            r[f'doc{d}'] = {'first64': m(lg, doc, 0, 64), 'ids64_256': m(lg, doc, 64, 256), 'cont40': m(lg, doc, 256, 256 + N)}
        r['mean_first64_nll'] = round(float(np.mean([r[f'doc{d}']['first64'][0] for d in docs])), 3)
        r['mean_cont40'] = [round(float(np.mean([r[f'doc{d}']['cont40'][k] for d in docs])), 3) for k in (0, 1)]
        emit(r)
if 'norm' in STAGES:
    for fn in [None, 'rms', 'ln']:
        G.final_norm = fn; r = {'kind': 'tf_norm', 'final_norm': fn, 'warm': W}
        for d, doc in docs.items():
            lg = tf(WARM_TAIL[-W:] if W else [], doc)
            r[f'doc{d}'] = {'first64': m(lg, doc, 0, 64), 'cont40': m(lg, doc, 256, 256 + N),
                            'maxlogit_mean': round(float(lg.max(-1).values.mean()), 2)}
            if fn is None: G._ref_argmax = lg.argmax(-1)
            else: r[f'doc{d}']['argmax_agree_vs_off'] = 'see doc0' if d != DOCS[0] else None
        r['mean_cont40'] = [round(float(np.mean([r[f'doc{d}']['cont40'][k] for d in docs])), 3) for k in (0, 1)]
        emit(r)
    G.final_norm = None
VARIANTS = {'a_fixes_only': (0, None), 'b_fixes_warm': (W, None), 'c_fixes_warm_ln': (W, 'ln')}
if 'gen' in STAGES:
    for vname, (w, fn) in VARIANTS.items():
        G.final_norm = fn
        for d, doc in docs.items():
            primer, true = doc[:257], doc[257:]
            t = time.time(); out = G.generate(G.context(primer, warm=w), max_new=N, stop_at_eos=False)
            em = sum(int(x == y) for x, y in zip(out, true)); px = next((i for i, (x, y) in enumerate(zip(out, true)) if x != y), len(out))
            emit({'kind': 'gen_cont', 'variant': vname, 'doc': d, 'unique40': len(set(out)), 'exact_pos': em, 'exact_prefix': px,
                  'seqmatch': round(difflib.SequenceMatcher(a=out, b=true, autojunk=False).ratio(), 3),
                  'true': show(true), 'generated': show(out), 'ids': out, 'secs': round(time.time() - t, 1)})
        for p in PROMPTS:
            for temp, k in [(0.0, 1), (0.7, 20)]:
                t = time.time(); out = G.generate(G.context(G.encode_prompt(p), warm=w), max_new=N, temperature=temp, top_k=k)
                emit({'kind': 'gen_prompt', 'variant': vname, 'prompt': p, 'temp': temp, 'top_k': k, 'n': len(out),
                      'unique40': len(set(out[-40:])), 'generated': show(out), 'ids': out, 'secs': round(time.time() - t, 1)})
    G.final_norm = None
emit({'kind': 'done', 'total_s': round(time.time() - t0, 1)})
