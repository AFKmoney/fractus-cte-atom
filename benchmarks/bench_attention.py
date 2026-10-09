"""Micro-benchmark of the linear-attention core at true 1B shapes.

1B config per block: n_heads=20, d_head=64, n_levels=2, chunk C=128.
The engine flattens (B, nlev, H) into G = B*nlev*H groups of (C, dH).

Measures forward-only and fwd+bwd wall time + RSS delta, per implementation.
Each (impl, B) cell runs in its own subprocess: the einsum reference
allocates O(G*C*dH^2) and can hit the native commit limit at larger B —
a segfault in one cell must not kill the whole table.

Usage: py benchmarks/bench_attention.py [--impl auto] [--iters 10]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_utils import RssTracker, timed

from fractus.nn.attention import FractalLinearAttention


def make_inputs(G, C, dH, seed=0):
    torch.manual_seed(seed)
    q = torch.rand(G, C, dH) + 0.5   # elu+1 features are positive; keep magnitudes realistic
    k = torch.rand(G, C, dH) + 0.5
    v = torch.randn(G, C, dH)
    return q, k, v


def run_impl(attn, fn, G, C, dH, carry, iters):
    q, k, v = make_inputs(G, C, dH)
    S0 = torch.randn(G, dH, dH) * 0.01 if carry else None
    z0 = torch.randn(G, dH) * 0.01 if carry else None

    # forward only
    def fwd():
        with torch.no_grad():
            fn(attn, q, k, v, carry=(S0, z0) if carry else None)

    # fwd + bwd (grad wrt inputs)
    def fwdbwd():
        qg = q.clone().requires_grad_(True)
        kg = k.clone().requires_grad_(True)
        vg = v.clone().requires_grad_(True)
        y = fn(attn, qg, kg, vg, carry=(S0, z0) if carry else None)
        loss = (y[0] if isinstance(y, tuple) else y).sum()
        loss.backward()

    t_fwd = timed(fwd, iters=iters)
    t_fwdbwd = timed(fwdbwd, iters=iters)

    with RssTracker() as rss:
        fwdbwd()
    rss_delta = rss.delta   # computed by __exit__, must be read after the block

    return t_fwd, t_fwdbwd, rss_delta


def _impl_table():
    from fractus.nn import attention as A
    impls = {}
    if hasattr(A.FractalLinearAttention, "_linear_attention_causal_einsum"):
        impls["einsum"] = A.FractalLinearAttention._linear_attention_causal_einsum
    for name in ("cumsum", "chunked"):
        if hasattr(A.FractalLinearAttention, f"_linear_attention_causal_{name}"):
            impls[name] = getattr(A.FractalLinearAttention, f"_linear_attention_causal_{name}")
    if not impls:
        # pristine repo state: only the vectorized body exists -> treat it as einsum ref
        impls["einsum"] = A.FractalLinearAttention._linear_attention_causal_vectorized
    return impls


def run_row(name, B, iters):
    """Single cell, executed inside the spawned subprocess."""
    fn = _impl_table()[name]
    attn = FractalLinearAttention(d_model=1280, n_heads=20, d_head=64, n_levels=2)
    dH, C = 64, 128
    G = B * 2 * 20
    t_fwd, t_fb, rss = run_impl(attn, fn, G, C, dH, carry=True, iters=iters)
    print(f"{name:<10} {B:>2} {G:>9} {t_fwd*1e3:>9.1f} {t_fb*1e3:>11.1f} {rss/1e6:>9.0f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="auto", choices=["auto", "einsum", "cumsum", "chunked"])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--batch-sizes", default="2,4,8")
    ap.add_argument("--row", nargs=2, metavar=("IMPL", "B"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.row:
        run_row(args.row[0], int(args.row[1]), args.iters)
        return

    impls = _impl_table()
    chosen = list(impls.items()) if args.impl == "auto" else [(args.impl, impls[args.impl])]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    me = str(Path(__file__).resolve())
    print(f"{'impl':<10} {'B':>2} {'G':>9} {'fwd ms':>9} {'fwd+bwd ms':>11} {'RSSdMB':>9}")
    for name, _ in chosen:
        for B in batch_sizes:
            r = subprocess.run([sys.executable, "-u", me, "--row", name, str(B),
                                "--iters", str(args.iters)],
                               capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                print(r.stdout.strip().splitlines()[-1], flush=True)
            else:
                why = "segfault/crash" if r.returncode else "no output"
                print(f"{name:<10} {B:>2} {B * 2 * 20:>9} {'-':>9} {'-':>11} "
                      f"{'-':>9}  <- {why} (rc={r.returncode})", flush=True)


if __name__ == "__main__":
    main()
