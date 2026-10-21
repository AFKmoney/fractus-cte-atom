"""The incremental stream path of FixedGenerator computes the same logits as the re-forward path and as the
teacher-forced fixed grid (docs/GEN_FINAL.md). Tiny random engine, CPU, fp32. Uses n_levels=2 so the per-level
open-chunk state (mean over levels only at a chunk end) is exercised. 4 experts = dense MoE dispatch: the capacity
dispatch (n_experts > 2*top_k) can drop (token, expert) pairs in a full 128-id chunk that a 1-id step never drops,
so it is not bit-comparable by construction (0 drops measured on the real 1B checkpoint, docs/GEN_FINAL.md)."""
import torch

from fractus.atom_tokenizer import VOCAB_SIZE
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.fixed_generation import SEQ, FixedGenerator, pick_next


def _gen():
    torch.manual_seed(0)
    eng = ContinuousThoughtEngine(vocab_size=VOCAB_SIZE, d_model=64, n_layers=2, n_heads=4, d_head=16, n_levels=2,
                                  n_oscillators=4, coupling_rank=2, n_experts=4, top_k=2, expert_d_ff=64,
                                  siren_rank=8).eval()
    return FixedGenerator(eng, warm=0)


def test_stream_equals_tf_and_reforward():
    G = _gen()
    ids = torch.randint(0, 256, (2 * SEQ + 40,)).tolist()
    tf = G.tf_logits(ids)
    n0 = 20
    rows = [G.stream_start(ids[:n0])] + [G.stream_push(t) for t in ids[n0:-1]]
    s = torch.stack(rows)
    assert (s - tf[n0 - 1:-1]).abs().max() < 1e-3
    for cw in (0, 2):
        out_a = G.generate(ids[:150], max_new=130, carry_window=cw, stop_at_eos=False)
        out_b = G.generate_stream(ids[:150], max_new=130, carry_window=cw, stop_at_eos=False)
        assert out_a == out_b


def test_pick_next_masks_and_bans():
    lg = torch.zeros(VOCAB_SIZE)
    lg[257] = 50.0   # BOS never sampled
    lg[258] = 40.0   # EOS banned while min_new not reached
    lg[65] = 10.0
    assert pick_next(lg, [], ban_eos=True) == 65
    assert pick_next(lg, []) == 258
    # no-repeat 2-gram: after "A" the id "A" (65) was already seen following "A"
    lg2 = torch.zeros(VOCAB_SIZE); lg2[65] = 10.0; lg2[66] = 9.0
    assert pick_next(lg2, [65, 65], no_repeat_ngram=2) == 66
