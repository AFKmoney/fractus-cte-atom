"""Grow lock for the Atom fork: body grows, vocab 266 does not."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from fractus.atom_tokenizer import VOCAB_SIZE, AtomFractusTokenizer
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.grow import grow_atom, grow_summary


def test_grow_body_keeps_vocab_and_trains():
    tok = AtomFractusTokenizer()
    ids = torch.tensor(tok.encode("Fractus grandit. Le vocabulaire ne grandit pas."), dtype=torch.long)
    assert int(ids.max()) < VOCAB_SIZE

    eng = ContinuousThoughtEngine(
        vocab_size=VOCAB_SIZE, d_model=32, n_heads=4, d_head=8, n_levels=1,
        n_oscillators=4, coupling_rank=2, n_experts=2, top_k=1,
        expert_d_ff=32, siren_rank=4, n_layers=1,
    )
    before = sum(p.numel() for p in eng.parameters())
    grown = grow_atom(eng, dict(
        d_model=48, n_layers=2, n_experts=4, n_oscillators=6, coupling_rank=2,
        n_heads=4, d_head=12, siren_rank=4, expert_d_ff=32, top_k=2,
    ))
    after = sum(p.numel() for p in grown.parameters())
    assert grown.vocab_size == VOCAB_SIZE
    assert grown.output_head.weight.data_ptr() == grown.observe.weight.data_ptr()
    assert len(grown.blocks) == 2
    assert grown.blocks[0].moe.n_experts == 4
    assert after > before
    # Old expert phase survived the copy.
    assert torch.allclose(grown.blocks[0].moe.expert_phases[:2], eng.blocks[0].moe.expert_phases[:2])

    chunk = ids[:16].unsqueeze(0)
    target = ids[1:17]
    grown.train()
    logits, lb = grown.tick_chunk_train(chunk)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), target) + 0.01 * lb
    loss.backward()
    assert logits.shape[-1] == VOCAB_SIZE

    new_idx = grown.blocks[0].moe.add_expert(dominant_idx=0)
    assert new_idx == 4
    assert grown.blocks[0].moe.n_experts == 5
    grown.reset_thought(1)
    logits2, _ = grown.tick_chunk_train(chunk)
    assert logits2.shape[-1] == VOCAB_SIZE
    print(grow_summary(eng, grown))


if __name__ == "__main__":
    test_grow_body_keeps_vocab_and_trains()
    print("grow ok")
