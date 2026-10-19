"""Copy head plumbing check on a small random engine (fast): for ngram 1 and 3, gate 0 bit-equal to the plain
engine; random gate: incremental open-chunk decoding (FixedGenerator get/put with the window) == TF rows, batched
gen_best.Decoder rows == single row, and reset_rows clears only the masked rows."""
import os, sys, json
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.fixed_generation import FixedGenerator
from fractus.nn.copy_head import attach_copy_head
from gen_best import Decoder
torch.set_num_threads(2); res = {}
for ng in (1, 3):
    torch.manual_seed(0)
    eng = ContinuousThoughtEngine(vocab_size=266, d_model=64, n_heads=4, d_head=16, n_levels=2, n_oscillators=4,
                                  coupling_rank=2, n_experts=4, top_k=2, expert_d_ff=64, siren_rank=8, n_layers=2).eval()
    G = FixedGenerator(eng, warm=0); dec = Decoder(G)
    ctx = torch.randint(0, 256, (300,)).tolist()
    A = G.tf_logits(ctx)
    head = attach_copy_head(eng, n_heads=2, d_head=16, mem=128, ngram=ng)
    B0 = G.tf_logits(ctx); r = {'gate0_bit_equal': bool(torch.equal(B0, A)), 'gate0_maxabs': float((B0 - A).abs().max())}
    with torch.no_grad(): head.gate.normal_(0, 1.0)
    C = G.tf_logits(ctx); r['gate_rand_changes'] = float((C - A).abs().max())
    G.cold(); G.chunk(ctx[:128]); G.chunk(ctx[128:256]); d = []
    for p in (256, 257, 258, 280, 299):
        s = G.get(); d.append(float((G.chunk(ctx[256:p + 1])[-1] - C[p]).abs().max())); G.put(s)
    r['incremental_vs_tf_maxabs'] = max(d)
    x = dec.start(ctx[:270]); l1, _, _ = dec.last([x]); l2, _, _ = dec.last([x, x])
    r['batched_vs_single_maxabs'] = float((l2[1] - l1[0]).abs().max())
    r['decoder_vs_tf_maxabs'] = float((l1[0][:256] - C[269][:256]).abs().max())
    head.reset_state(3); head.attend(torch.randn(3, 128, 64))
    ck = head.cm.clone(); head.reset_rows(torch.tensor([False, True, False]))
    r['reset_rows_ok'] = bool(head.cm[1].sum() == 0 and torch.equal(head.cm[0], ck[0]) and head.lv[1].sum() == 0)
    res[f'ngram{ng}'] = r
print(json.dumps(res, indent=1)); json.dump(res, open('/workspace/scratch/copyhead/equiv_tiny.json', 'w'), indent=1)
