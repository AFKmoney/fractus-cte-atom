"""CPU smoke test of the v2 training loop pieces, end to end.

Exercises exactly what fast4gpu_boost_v2.py does per step — synthetic int32
memmap shard -> per-chunk fetch -> tick_chunk_train_ce -> backward -> SGD ->
scheduled sampling via sample_tokens_chunked — under BOTH attention kernels,
asserting they produce equivalent losses. No GPU required.

Open-heart value: if this passes, swapping the live pod's code cannot change
the training signal beyond documented float32 rounding.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.nn import attention as attn_mod
from fractus.nn.ce import sample_tokens_chunked

VOCAB = 1000
CFG = dict(vocab_size=VOCAB, d_model=64, n_heads=1, d_head=64,
           n_levels=2, n_oscillators=8, coupling_rank=4,
           n_experts=8, top_k=2, expert_d_ff=64, siren_rank=16,
           n_layers=2)


def _make_shard(tmp: Path, n_tokens: int = 4096, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    p = tmp / "shard_gpu0.npy"
    mm = np.lib.format.open_memmap(p, mode="w+", dtype=np.int32,
                                   shape=(n_tokens,))
    mm[:] = rng.integers(0, VOCAB, size=n_tokens)
    mm.flush()
    return p


def _fetch(mm, start, count):
    view = np.asarray(mm[start : start + count])
    return torch.from_numpy(view).to(torch.int64)


def _run_loop(shard_path, impl, n_steps=6):
    prev_impl = attn_mod._ACTIVE_IMPL
    attn_mod.set_attention_impl(impl)
    try:
        torch.manual_seed(123)
        eng = ContinuousThoughtEngine(**CFG)
        eng.reset_thought(batch_size=2)
        opt = torch.optim.SGD(eng.parameters(), lr=7e-4, momentum=0.9)

        mm = np.load(str(shard_path), mmap_mode="r")
        B, SEQ = 2, 16
        step = B * SEQ
        losses = []
        for i, start in enumerate(range(0, n_steps * step, step)):
            block = _fetch(mm, start, step + 1)
            chunk = block[:step].view(B, SEQ).long()
            target = block[1:].view(B, SEQ)

            ce_tf, lb, h = eng.tick_chunk_train_ce(chunk, target,
                                                   ce_chunk=128,
                                                   return_hidden=True)
            loss = ce_tf + 0.02 * lb
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if i % 2 == 0:
                # scheduled-sampling branch: exercise sampling mechanics only
                with torch.no_grad():
                    samp = sample_tokens_chunked(
                        h.reshape(-1, h.shape[-1]).detach(),
                        eng.output_head.weight, temperature=0.9, ce_chunk=128)
                assert samp.shape == (B * SEQ,)
                assert samp.min() >= 0 and samp.max() < VOCAB

            assert torch.isfinite(loss), f"non-finite loss at step {i}"
            losses.append(float(ce_tf.detach()))
        return losses
    finally:
        attn_mod.set_attention_impl(prev_impl)


def test_v2_loop_runs_and_kernels_agree(tmp_path):
    shard = _make_shard(tmp_path)
    l_cs = _run_loop(shard, "cumsum")
    l_ch = _run_loop(shard, "chunked")
    # same data, same init (same seed): losses must match within fp32 rounding
    for a, b in zip(l_cs, l_ch):
        rel = abs(a - b) / max(abs(a), 1e-9)
        assert rel < 5e-4, f"losses diverge: cumsum={a:.6f} chunked={b:.6f} ({rel:.2e})"


def test_fetch_matches_legacy_whole_shard_load(tmp_path):
    """The v2 per-chunk int32 fetch must give the exact tokens v1 loaded."""
    shard = _make_shard(tmp_path, n_tokens=512, seed=42)
    legacy = torch.from_numpy(np.load(str(shard), mmap_mode="r")).to(torch.int64)
    mm = np.load(str(shard), mmap_mode="r")
    for start in (0, 37, 256):
        got = _fetch(mm, start, 33)
        want = legacy[start : start + 33]
        assert torch.equal(got, want)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
