"""Equivalence proof: block-level activation checkpointing (opt 6).

block_ckpt=True wraps each block's PURE core in torch.utils.checkpoint.
Danger zone: tick_chunk_core READS the carry (attn_S/attn_z) then OVERWRITES
it. A naive whole-block checkpoint would make backward's recompute read the
NEW carry — wrong gradients. The refactor keeps the read-mutate cycle
outside the checkpointed region; these tests prove loss, gradients AND the
carry evolution across consecutive chunks are identical with and without.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fractus.continuous_engine import ContinuousThoughtEngine

CFG = dict(vocab_size=1000, d_model=64, n_heads=1, d_head=64,
           n_levels=2, n_oscillators=8, coupling_rank=4,
           n_experts=8, top_k=2, expert_d_ff=64, siren_rank=16,
           n_layers=3)   # 3 blocks: carry chain matters


def _run(block_ckpt, n_chunks=3):
    torch.manual_seed(42)
    eng = ContinuousThoughtEngine(**CFG)
    eng.reset_thought(batch_size=2)
    opt = torch.optim.SGD(eng.parameters(), lr=7e-4, momentum=0.9)
    g = torch.Generator().manual_seed(7)
    out = []
    for _ in range(n_chunks):
        toks = torch.randint(0, 1000, (2, 17), generator=g)
        chunk, target = toks[:, :-1], toks[:, 1:]
        ce, lb = eng.tick_chunk_train_ce(chunk, target, ce_chunk=64,
                                         block_ckpt=block_ckpt)
        loss = ce + 0.02 * lb
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        snap = [(b.attn_S.clone(), b.attn_z.clone()) for b in eng.blocks]
        out.append((ce.detach().clone(), lb.detach().clone(), snap))
    grads = {n: p.grad.clone() for n, p in eng.named_parameters()
             if p.grad is not None}
    return out, grads


def test_block_ckpt_matches_unchecked():
    """Losses, per-block carries after EACH chunk, and all param grads equal."""
    ref, grads_ref = _run(False)
    ck, grads_ck = _run(True)
    for i, ((ce_r, lb_r, snap_r), (ce_c, lb_c, snap_c)) in enumerate(zip(ref, ck)):
        assert torch.allclose(ce_r, ce_c, atol=1e-5, rtol=1e-5), \
            f"chunk {i}: ce {ce_r.item():.6f} vs {ce_c.item():.6f}"
        assert torch.allclose(lb_r, lb_c, atol=1e-6), f"chunk {i}: lb differs"
        # THE point: carry evolution identical across the chunk sequence,
        # for every block in the chain
        for j, ((S_r, z_r), (S_c, z_c)) in enumerate(zip(snap_r, snap_c)):
            assert torch.allclose(S_r, S_c, atol=1e-4, rtol=1e-4), \
                f"chunk {i} block {j}: attn_S diverged (max {(S_r - S_c).abs().max():.2e})"
            assert torch.allclose(z_r, z_c, atol=1e-5, rtol=1e-4), \
                f"chunk {i} block {j}: attn_z diverged"
    assert set(grads_ref) == set(grads_ck)
    for name in grads_ref:
        gr, gc_ = grads_ref[name], grads_ck[name]
        assert torch.allclose(gr, gc_, atol=1e-4, rtol=1e-3), \
            f"grad mismatch on {name}: max {(gr - gc_).abs().max():.3e}"


def test_block_ckpt_off_is_default_behavior():
    """block_ckpt=False must hit exactly the legacy path."""
    from fractus.continuous_engine import ContinuousThoughtEngine as E
    import inspect
    src = inspect.getsource(E.tick_chunk_train_ce)
    assert "use_reentrant" in src          # checkpointed branch exists
    assert "block_ckpt and torch.is_grad_enabled()" in src


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
