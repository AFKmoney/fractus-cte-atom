#!/usr/bin/env python3
"""Measure Kuramoto order r against n_steps and dt.

The block builds KuramotoLayer with n_steps=1, dt=0.1. _rk4_integrate then
takes a single step. The class default of 4 is never used by the model.
Train and generate are aligned. The integration is just short.

This does not change weights. It only changes how far the phase moves
before the MoE reads theta. r is the mean order parameter |mean exp(i theta)|.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from fractus.atom_tokenizer import VOCAB_SIZE, AtomFractusTokenizer
from fractus.continuous_engine import ContinuousThoughtEngine


def order_r(theta: torch.Tensor) -> float:
    z = torch.exp(1j * theta.to(torch.float32))
    return float(z.mean(dim=-1).abs().mean())


def main() -> None:
    tok = AtomFractusTokenizer()
    ids, feat = tok.encode_with_features("Fractus pense en continu. Le couplage doit bouger.")
    ids = torch.tensor(ids[:32]).unsqueeze(0)
    feat = feat[:32].unsqueeze(0)
    print("n_steps dt     r_after   phase_span_deg")
    for n_steps in (1, 2, 4):
        for dt in (0.1, 0.2, 0.5):
            torch.manual_seed(0)
            eng = ContinuousThoughtEngine(
                vocab_size=VOCAB_SIZE, d_model=64, n_heads=4, d_head=16, n_levels=1,
                n_oscillators=16, coupling_rank=4, n_experts=4, top_k=2,
                expert_d_ff=64, siren_rank=8, n_layers=2,
            )
            for blk in eng.blocks:
                blk.kuramoto.n_steps = n_steps
                blk.kuramoto.dt = dt
            eng.eval()
            with torch.no_grad():
                eng.reset_thought(1)
                eng.tick_chunk_train(ids, feat=feat)
                rs = [order_r(blk.kuramoto_phases) for blk in eng.blocks]
                span = float(
                    (eng.blocks[0].kuramoto_phases.max() - eng.blocks[0].kuramoto_phases.min())
                    * 180 / math.pi
                )
            print(f"{n_steps:7d} {dt:<6} {sum(rs)/len(rs):.4f}    {span:.1f}")


if __name__ == "__main__":
    main()
