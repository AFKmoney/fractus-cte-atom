"""Copy head, test 1 (native path): gate=0 equivalence on the real v4 checkpoint, fp32, CPU.
Also measures the run-to-run noise floor of the unchanged engine (same input twice), so a non-zero diff can be
attributed. Checks: fixed-grid TF logits, greedy + beam-4 ids through scripts/gen_best.Decoder (batched snapshots),
and with a random non-zero gate: open-chunk incremental decoding with the K/V window == TF rows."""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractus.fixed_generation import load_checkpoint, FixedGenerator
from fractus.nn.copy_head import attach_copy_head
from gen_best import Decoder
torch.set_num_threads(int(os.environ.get('THREADS', '2')))
OUT = os.environ.get('OUT', '/workspace/scratch/copyhead/equiv_native.json')
eng, cfg = load_checkpoint('/workspace/ckpt_v4_good/fractus_atom_1b_gpu0.pt', torch.float32)
G = FixedGenerator(eng, warm=127); dec = Decoder(G)
a = np.fromfile('/workspace/v4build/wiki_val.i16', dtype=np.int16).astype(np.int64)
ctx = a[5000:5000 + 300].tolist()
r = {}
A = G.tf_logits(ctx); A2 = G.tf_logits(ctx)
r['noise_floor_same_engine_maxabs'] = float((A - A2).abs().max()); r['noise_floor_bit_equal'] = bool(torch.equal(A, A2))
g0 = dec.greedy(ctx[:200], n=16); b0 = dec.beam(ctx[:200], n=12, W=4)
head = attach_copy_head(eng, n_heads=8, d_head=64, mem=128)
B1 = G.tf_logits(ctx)
r['gate0_maxabs'] = float((B1 - A).abs().max()); r['gate0_bit_equal'] = bool(torch.equal(B1, A))
r['greedy_equal'] = dec.greedy(ctx[:200], n=16) == g0; r['beam4_equal'] = dec.beam(ctx[:200], n=12, W=4) == b0
torch.manual_seed(0)
with torch.no_grad(): head.gate.normal_(0, 0.5)
C = G.tf_logits(ctx); r['gate_rand_changes_maxabs'] = float((C - A).abs().max())
seq = ctx; G.cold(); G.chunk(seq[:128]); G.chunk(seq[128:256]); d = []
for p in (256, 263, 299):
    s = G.get(); lg = G.chunk(seq[256:p + 1])[-1]; G.put(s); d.append(float((lg - C[p]).abs().max()))
r['gate_rand_incremental_vs_tf_maxabs'] = max(d)
# batched decoder rows with the window (stack/row) vs single
x = dec.start(seq[:270]); lg1, _, _ = dec.last([x]); lg2, _, _ = dec.last([x, x])
r['gate_rand_batched_vs_single_maxabs'] = float((lg2[1] - lg1[0]).abs().max())
r['gate_rand_decoder_vs_tf_maxabs'] = float((lg1[0][:256] - C[269][:256]).abs().max())
json.dump(r, open(OUT, 'w'), indent=1); print(r)
