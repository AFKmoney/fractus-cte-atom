"""Generation hunt (docs/DEEP_DIVE_GENERATION.md, part 2). Inference only, trainer untouched.
Variants on the fixed path (fractus/fixed_generation.py): warm length, tokenizer-driven boundary markers
(the model emits bytes only; the Atomizer inserts markers by its own rule, exactly as build_atom_corpus does),
scale-free nucleus sampling with a presence penalty, plus an explicit parity check of the re-forward path.
"""
import os, sys, json, time, difflib
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fractus.fixed_generation import load_checkpoint, FixedGenerator, WARM_TAIL, SEQ, show
from fractus.atom_tokenizer import AtomFractusTokenizer
torch.set_num_threads(int(os.environ.get('THREADS', '7')))
CK = os.environ.get('CKPT', '/workspace/ckpt_v3_85000/fractus_atom_1b_gpu0.pt')
OUT = os.environ.get('OUT', '/workspace/gen_hunt.jsonl'); N = 40
eng, cfg = load_checkpoint(CK); G = FixedGenerator(eng); tok = AtomFractusTokenizer()
a = np.fromfile('/workspace/lmsys/ref/atom_corpus/reasoning__syllogisms.i16', dtype=np.int16).astype(np.int64); bos = np.where(a == 257)[0]
DOCS = {d: a[int(bos[d]):int(bos[d]) + 257 + N].tolist() for d in (0, 5000, 10000)}
PROMPTS = ['The capital of France is', 'Once upon a time', 'Hello']
def emit(r):
    print(json.dumps(r, ensure_ascii=False), flush=True); open(OUT, 'a').write(json.dumps(r, ensure_ascii=False) + '\n')

# ---- parity: per-step re-forward of the open chunk == one teacher-forced pass on the same grid
if os.environ.get('PARITY', '1') == '1':
    doc = DOCS[0]; ctx = WARM_TAIL[-64:] + doc; ref = G.tf_logits(ctx)
    seq = ctx[:64 + 257]; G.cold(); com = 0; rows = []
    while len(seq) - com > SEQ: G.chunk(seq[com:com + SEQ]); com += SEQ
    for n in doc[257:]:
        s = G.get(); rows.append(G.chunk(seq[com:])[-1]); G.put(s); seq.append(n)
        if len(seq) - com > SEQ: G.chunk(seq[com:com + SEQ]); com += SEQ
    g = torch.stack(rows); r = ref[64 + 256:64 + 256 + N]
    emit({'kind': 'parity_reforward_vs_tf', 'maxabs': float((g - r).abs().max()), 'top1_agree': float((g.argmax(-1) == r.argmax(-1)).float().mean()),
          'note': 'thought_state injection at the same 128-grid boundaries in both'})

def pick(lg, hist, mode, g):
    lg = lg.clone(); lg[256] = -1e9; lg[257] = -1e9
    if mode.get('bytes_only'): lg[259:266] = -1e9          # markers come from the tokenizer, not the model
    if mode.get('sample'):
        x = lg / lg[lg > -1e8].std().clamp_min(1e-6)          # scale-free: logits in units of their std
        for i in set(hist[-32:]):
            if i < 256: x[i] -= mode.get('presence', 1.0)
        x = x / mode.get('T', 0.8)
        p = torch.softmax(x, -1); sp, si = p.sort(descending=True); keep = sp.cumsum(0) - sp < mode.get('top_p', 0.9)
        sp = sp * keep; return int(si[torch.multinomial(sp / sp.sum(), 1, generator=g)])
    return int(lg.argmax())

def run(prompt_ids, warm, mode, seed=0):
    """prompt_ids: BOS + ... (as encoded). tokenizer-driven markers: the byte stream is re-encoded by the
    Atomizer (encode_open) after every new byte and the context rebuilt; committed chunks are recomputed
    only if the encoder changed ids inside them (a look-ahead cut)."""
    g = torch.Generator().manual_seed(seed); pre = WARM_TAIL[-warm:] if warm else []
    tm = mode.get('tok_markers'); body = list(prompt_ids)
    pbytes = bytes(i for i in prompt_ids if i < 256)
    if tm:   # the prompt re-encoded by the same rule so the stream is consistent
        body = tok.encode_open(pbytes.decode('utf-8', 'replace'))[0] if prompt_ids[0] == 257 else body
    gen_bytes = []; out = []; ctx = pre + list(body)
    G.cold(); committed = []; com = 0
    while len(ctx) - com > SEQ: G.chunk(ctx[com:com + SEQ]); com += SEQ
    committed = ctx[:com]; snap = G.get(); hist = list(ctx)
    for _ in range(N):
        G.put(snap); lg = G.chunk(ctx[com:])[-1]
        nid = pick(lg, hist, mode, g); out.append(nid); hist.append(nid)
        if tm and nid < 256:
            gen_bytes.append(nid)
            new = pre + enc_open_bytes(pbytes + bytes(gen_bytes))
        else:
            new = ctx + [nid]
        if new[:com] != committed:      # look-ahead cut landed inside committed chunks: recompute from scratch
            G.cold(); com = 0
            while len(new) - com > SEQ: G.chunk(new[com:com + SEQ]); com += SEQ
            committed = new[:com]; snap = G.get()
        else:
            G.put(snap)
            while len(new) - com > SEQ: G.chunk(new[com:com + SEQ]); com += SEQ
            committed = new[:com]; snap = G.get()
        ctx = new
        if nid == 258: break
    gen_ids = ctx[len(pre) + len(body):] if tm else out
    return out, gen_ids

def enc_open_bytes(raw):
    at = tok.atomizer; packets = at.encode_bytes(raw, reset=True, flush=False); ids = [257]
    for p in packets:
        ids.extend(p.payload); c = at._BOUNDARY_CODES.get(p.boundary, 0)
        if c: ids.append(258 + c)
    ids.extend(bytes(at._buffer)); return ids

VARIANTS = {
  'b_W64_greedy': (64, {}),
  'W127_greedy': (127, {}),
  'W64_tokmarkers_greedy': (64, {'bytes_only': 1, 'tok_markers': 1}),
  'W127_tokmarkers_greedy': (127, {'bytes_only': 1, 'tok_markers': 1}),
  'W64_bytes_only_no_markers': (64, {'bytes_only': 1}),
  'W64_sample_T0.8_p0.9_pres1': (64, {'sample': 1}),
  'W64_tokmarkers_sample': (64, {'bytes_only': 1, 'tok_markers': 1, 'sample': 1}),
}
sel = os.environ.get('VARIANTS'); VARIANTS = {k: v for k, v in VARIANTS.items() if not sel or k in sel.split(',')}
for vn, (w, mode) in VARIANTS.items():
    for d, doc in DOCS.items():
        primer, true = doc[:257], doc[257:]; t = time.time()
        out, gen_ids = run(primer, w, mode)
        tb = bytes(i for i in true if i < 256); gb = bytes(i for i in gen_ids if i < 256)
        em = sum(int(x == y) for x, y in zip(gen_ids, true))
        emb = sum(int(x == y) for x, y in zip(gb, tb))
        emit({'kind': 'cont', 'variant': vn, 'doc': d, 'unique40': len(set(gen_ids[:N])), 'exact_pos_ids': em, 'exact_pos_bytes': emb,
              'seqmatch_ids': round(difflib.SequenceMatcher(a=gen_ids, b=true, autojunk=False).ratio(), 3),
              'seqmatch_bytes': round(difflib.SequenceMatcher(a=gb, b=tb, autojunk=False).ratio(), 3),
              'true': show(true), 'generated': show(gen_ids), 'secs': round(time.time() - t, 1)})
    for p in PROMPTS:
        out, gen_ids = run(tok.encode_open(p)[0], w, mode)
        emit({'kind': 'prompt', 'variant': vn, 'prompt': p, 'unique40': len(set(gen_ids[-40:])), 'generated': show(gen_ids)})
emit({'kind': 'done'})
