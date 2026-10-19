"""Copy head on the real v4 checkpoint (box, CPU, fp32; trainer untouched). Phase 1 + body caching.

  equiv : gate=0 equivalence. Fixed-grid TF logits, tick_chunk_train_ce, greedy generation: old vs +copy head.
          Then a random non-zero gate: incremental open-chunk decoding (snapshot incl. K/V window) == TF rows.
  val   : body hidden states h_body of the 32 v4_eval val windows (127+1024 ids, same rng) -> cache/val/*.pt
  train : body hidden states of NTRAIN random 512-id windows of atom_corpus_v4 train_006 -> cache/train/*.pt
The copy head sits at the readout and the body never sees it, so the head can then be trained on cached h_body
exactly (scripts/copyhead_1b_train.py). Resumable: existing outputs are skipped.
"""
import os, sys, json, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fractus.fixed_generation import load_checkpoint, FixedGenerator, SEQ
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.nn.copy_head import attach_copy_head, make_copy_generator, _body

torch.set_num_threads(int(os.environ.get('THREADS', '7')))
CK = os.environ.get('CKPT', '/workspace/ckpt_v4_good/fractus_atom_1b_gpu0.pt')
D = os.environ.get('DIR', '/workspace/scratch/copyhead')
PH = os.environ.get('PHASES', 'equiv,val,train').split(',')
NTRAIN, TB, TL = int(os.environ.get('NTRAIN', 256)), int(os.environ.get('TB', 8)), 512
VB = '/workspace/v4build'
def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)

eng, cfg = load_checkpoint(CK, torch.float32)
G = FixedGenerator(eng, warm=127)
val = {s: np.fromfile(f'{VB}/{s}_val.i16', dtype=np.int16).astype(np.int64) for s in ['fineweb', 'wiki', 'tinystories', 'pg19']}
rng = np.random.default_rng(0); VW = []
for s, a in val.items():
    for wi, st in enumerate(rng.integers(0, len(a) - (127 + 1024 + 1), 8)):
        VW.append((f'{s}:{wi}', a[st:st + 127 + 1024].tolist()))

if 'equiv' in PH and not os.path.exists(f'{D}/equiv.json'):
    res = {}
    ctx = VW[0][1]
    L0 = G.tf_logits(ctx)
    tgt = torch.tensor(ctx[1:])
    res['nll_base_fineweb0'] = float(F.cross_entropy(L0[:-1][126:], tgt[126:]))
    # tick_chunk_train_ce (trainer path) before attach
    def ce_path():
        eng.reset_thought(1); out = []
        with torch.no_grad():
            for j in range(0, 1024, SEQ):
                x = torch.tensor([ctx[j:j + SEQ]]); y = torch.tensor([ctx[j + 1:j + SEQ + 1]])
                out.append(float(eng.tick_chunk_train_ce(x, y)[0]))
        return out
    ce0 = ce_path()
    gctx = ctx[:300]
    g0 = G.generate(gctx, max_new=24)
    head = attach_copy_head(eng, n_heads=8, d_head=64, mem=128)
    CG = make_copy_generator(FixedGenerator)(eng, warm=127)
    L1 = CG.tf_logits(ctx)
    res['tf_logits_maxabs_gate0'] = float((L1 - L0).abs().max())
    res['tf_logits_bit_equal_gate0'] = bool(torch.equal(L1, L0))
    ce1 = ce_path()
    res['train_ce_path_maxabs_gate0'] = max(abs(a - b) for a, b in zip(ce0, ce1))
    g1 = CG.generate(gctx, max_new=24)
    res['greedy_ids_equal_gate0'] = g0 == g1
    # non-zero gate: function changes, and incremental decoding with the window cache == TF
    torch.manual_seed(0)
    with torch.no_grad():
        head.gate.normal_(0, 0.5)
    L2 = CG.tf_logits(ctx[:300])
    res['tf_logits_maxabs_gate_rand_vs_base'] = float((L2 - L0[:300]).abs().max())
    seq = ctx[:300]; CG.cold(); CG.chunk(seq[0:128]); CG.chunk(seq[128:256]); com = 256
    diffs = []
    for p in (256, 270, 299):
        s = CG.get(); lg = CG.chunk(seq[com:p + 1])[-1]; CG.put(s)
        diffs.append(float((lg - L2[p]).abs().max()))
    res['incremental_vs_tf_maxabs_gate_rand'] = max(diffs)
    res['logit_scale_absmax'] = float(L0.abs().max())
    with torch.no_grad():
        head.gate.zero_()
    json.dump(res, open(f'{D}/equiv.json', 'w'), indent=1); log('equiv', res)

if 'val' in PH:
    for key, ctx in VW:
        f = f'{D}/cache/val/{key.replace(":", "_")}.pt'
        if os.path.exists(f): continue
        t = time.time(); eng.reset_thought(1); hs = []
        with torch.inference_mode():
            for j in range(0, len(ctx), SEQ):
                h, _ = _body(eng, torch.tensor([ctx[j:j + SEQ]])); hs.append(h[0])
        torch.save(dict(key=key, ids=torch.tensor(ctx), h=torch.cat(hs).clone()), f + '.tmp'); os.replace(f + '.tmp', f)
        log('val', key, round(time.time() - t, 1))

if 'train' in PH:
    tr = np.memmap(f'{VB}/atom_corpus_v4/train_006.i16', dtype=np.int16, mode='r')
    starts = np.random.default_rng(1).integers(0, len(tr) - TL - 2, NTRAIN)
    for b0 in range(0, NTRAIN, TB):
        f = f'{D}/cache/train/b{b0:05d}.pt'
        if os.path.exists(f): continue
        t = time.time()
        ids = torch.from_numpy(np.stack([np.asarray(tr[s:s + TL + 1], dtype=np.int64) for s in starts[b0:b0 + TB]]))
        eng.reset_thought(ids.shape[0]); hs = []
        with torch.inference_mode():
            for j in range(0, TL, SEQ):
                h, _ = _body(eng, ids[:, j:j + SEQ]); hs.append(h)
        torch.save(dict(ids=ids, h=torch.cat(hs, 1).clone()), f + '.tmp'); os.replace(f + '.tmp', f)
        log('train', b0, round(time.time() - t, 1))
log('done')
