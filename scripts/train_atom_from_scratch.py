#!/usr/bin/env python
"""Train a Fractus CTE from zero on Atomizer ids.

Source architecture: HF thefinalboss/fractus-cte (ContinuousThoughtEngine).
This script never loads an x8 / GPT-2 checkpoint. vocab 50257 weights cannot
map onto vocab 266. START_TOKEN is 0 because the run is new — that exception
does not apply to the existing x8 resume.

Usage (smoke, CPU):
    python scripts/train_atom_from_scratch.py --scale smoke --steps 30

Usage (1B config, fresh, on a pod — do not point this at x8run):
    python scripts/train_atom_from_scratch.py --scale 1b --corpus data/atom_corpus.i16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from fractus.atom_tokenizer import VOCAB_SIZE, AtomFractusTokenizer
from fractus.continuous_engine import ContinuousThoughtEngine

SCALE = {
    "smoke": dict(
        d_model=64, n_heads=4, d_head=16, n_levels=1,
        n_oscillators=4, coupling_rank=2,
        n_experts=4, top_k=2, expert_d_ff=64, siren_rank=8,
        n_layers=2,
    ),
    "1b": dict(
        d_model=1280, n_heads=20, d_head=64, n_levels=2,
        n_oscillators=16, coupling_rank=8,
        n_experts=128, top_k=2, expert_d_ff=2048, siren_rank=64,
        n_layers=16,
    ),
}

SMOKE_TEXT = (
    "Fractus pense en continu. L'Atomizer coupe le flux en spans d'octets, "
    "pas en BPE. Bonjour Philippe. 2+2=4. def tick(): return h\n"
) * 8


def load_ids(path: str | None, tok: AtomFractusTokenizer) -> torch.Tensor:
    if not path:
        return torch.tensor(tok.encode(SMOKE_TEXT), dtype=torch.long)
    if path.endswith(".i16"):
        arr = np.fromfile(path, dtype=np.int16)
    elif path.endswith(".npy"):
        arr = np.load(path, mmap_mode="r")
    else:
        raise SystemExit(f"unsupported corpus: {path}")
    if arr.size and int(arr.max()) >= VOCAB_SIZE:
        raise SystemExit("corpus id >= vocab 266 — this is not an Atom stream")
    return torch.as_tensor(np.asarray(arr, dtype=np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=sorted(SCALE), default="smoke")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="checkpoints/atom_scratch")
    ap.add_argument("--max-span-bytes", type=int, default=32)
    ap.add_argument("--pack-mode", default="linguistic")
    args = ap.parse_args()

    tok = AtomFractusTokenizer(
        max_span_bytes=args.max_span_bytes, pack_mode=args.pack_mode
    )
    ids = load_ids(args.corpus, tok)
    if ids.numel() < args.seq_len + 1:
        raise SystemExit("corpus shorter than seq-len+1")

    cfg = dict(SCALE[args.scale])
    engine = ContinuousThoughtEngine(vocab_size=VOCAB_SIZE, **cfg)
    n_params = sum(p.numel() for p in engine.parameters())
    opt = torch.optim.AdamW(engine.parameters(), lr=args.lr, weight_decay=0.01)
    engine.train()

    losses = []
    t0 = time.time()
    n = ids.numel()
    for step in range(args.steps):
        start = (step * args.seq_len) % (n - args.seq_len - 1)
        chunk = ids[start:start + args.seq_len].unsqueeze(0)
        target = ids[start + 1:start + 1 + args.seq_len]
        engine.reset_thought(batch_size=1)
        logits, lb = engine.tick_chunk_train(chunk)
        ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), target)
        loss = ce + 0.01 * lb
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
        opt.step()
        losses.append(float(ce.detach()))
        if step % max(1, args.steps // 5) == 0 or step == args.steps - 1:
            print(
                f"step {step} ce {float(ce.detach()):.4f} lb {float(lb.detach()):.4f}",
                flush=True,
            )

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, f"fractus_atom_{args.scale}.pt")
    torch.save(
        {
            "model_state": engine.state_dict(),
            "config": {**cfg, "vocab_size": VOCAB_SIZE, "scale": args.scale},
            "tokenizer": {
                "version": tok.VERSION,
                "vocab_size": VOCAB_SIZE,
                "max_span_bytes": args.max_span_bytes,
                "pack_mode": args.pack_mode,
            },
            "steps": args.steps,
            "start_token": 0,
            "parent": "hf:thefinalboss/fractus-cte",
            "note": "from scratch. not an x8 resume.",
        },
        ckpt,
    )
    summary = {
        "ckpt": ckpt,
        "params": n_params,
        "vocab_size": VOCAB_SIZE,
        "steps": args.steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "seconds": round(time.time() - t0, 2),
        "tokens_seen": args.steps * args.seq_len,
    }
    with open(os.path.join(args.out, "scratch_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
