"""End-to-end engine training benchmark (CPU): tok/s for tick_chunk_train + backward.

Small config (fits CPU comfortably) — used to measure RELATIVE end-to-end gains
of each optimization. Attention-core micro-bench at true 1B shapes lives in
bench_attention.py.

Usage: py benchmarks/bench_engine.py [--steps 20]
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fractus.continuous_engine import ContinuousThoughtEngine

CONFIG = dict(
    vocab_size=50257,
    d_model=128,
    n_heads=2,
    d_head=64,
    n_levels=2,
    n_oscillators=8,
    coupling_rank=4,
    n_experts=8,
    top_k=2,
    expert_d_ff=128,
    siren_rank=32,
    n_layers=2,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(0)
    eng = ContinuousThoughtEngine(**CONFIG)
    eng.reset_thought(args.batch)
    opt = torch.optim.SGD(eng.parameters(), lr=1e-3, momentum=0.9)

    B, C = args.batch, args.seq
    toks = torch.randint(0, 50257, (B * (C + 1) * (args.steps + 1),))

    t0 = time.perf_counter()
    n_tok = 0
    for s in range(args.steps):
        chunk = toks[s * B * C : (s + 1) * B * C].view(B, C)
        target = toks[s * B * C + 1 : (s + 1) * B * C + 1].view(B, C)
        import torch.nn.functional as F

        logits, lb = eng.tick_chunk_train(chunk)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        loss = ce + 0.02 * lb
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        n_tok += B * C
    dt = time.perf_counter() - t0
    print(f"config={CONFIG['d_model']}d x{CONFIG['n_layers']}blocks E{CONFIG['n_experts']} "
          f"B={B} SEQ={C}: {n_tok/dt:.0f} tok/s ({dt:.2f}s / {args.steps} steps)")


if __name__ == "__main__":
    main()
