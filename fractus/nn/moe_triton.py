"""Triton grouped low-rank expert forward.

One launch per expert that has tokens. Tokens for that expert are gathered,
run through the low-rank Siren (scale * (h @ V) @ U.T + b, GELU, again),
and scattered back. No Python loop over tokens, no giant einsum.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _lr_expert_kernel(
    h_ptr, V_ptr, U_ptr, b_ptr, scale,
    out_ptr,
    N, D, F, R,
    stride_h_n, stride_h_d,
    stride_V_d, stride_V_r,
    stride_U_f, stride_U_r,
    stride_out_n, stride_out_f,
    BLOCK_N: tl.constexpr, BLOCK_F: tl.constexpr, BLOCK_R: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_f = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
    mask_n = offs_n < N
    mask_f = offs_f < F

    acc = tl.zeros((BLOCK_N, BLOCK_F), dtype=tl.float32)
    for r0 in range(0, R, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R
        # hV[n, r] = h[n, d] @ V[d, r]
        hV = tl.zeros((BLOCK_N, BLOCK_R), dtype=tl.float32)
        for d0 in range(0, D, 64):
            offs_d = d0 + tl.arange(0, 64)
            mask_d = offs_d < D
            h = tl.load(
                h_ptr + offs_n[:, None] * stride_h_n + offs_d[None, :] * stride_h_d,
                mask=mask_n[:, None] & mask_d[None, :], other=0.0,
            )
            V = tl.load(
                V_ptr + offs_d[:, None] * stride_V_d + offs_r[None, :] * stride_V_r,
                mask=mask_d[:, None] & mask_r[None, :], other=0.0,
            )
            hV += tl.dot(h, V)
        U = tl.load(
            U_ptr + offs_f[:, None] * stride_U_f + offs_r[None, :] * stride_U_r,
            mask=mask_f[:, None] & mask_r[None, :], other=0.0,
        )
        acc += tl.dot(hV, tl.trans(U))

    b = tl.load(b_ptr + offs_f, mask=mask_f, other=0.0)
    acc = acc * scale + b[None, :]
    # GELU
    acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(
        out_ptr + offs_n[:, None] * stride_out_n + offs_f[None, :] * stride_out_f,
        acc, mask=mask_n[:, None] & mask_f[None, :],
    )


def grouped_lowrank_forward(h, topk_idx, U1, V1, b1, s1, U2, V2, b2, s2):
    """h: (B, L, D), topk_idx: (B, L, K). Returns (B, L, K, D)."""
    B, L, D = h.shape
    K = topk_idx.shape[-1]
    N = B * L
    F = U1.shape[1]
    R = U1.shape[2]
    h_flat = h.reshape(N, D)
    out = h.new_zeros(N, K, D)
    for k in range(K):
        idx = topk_idx[:, :, k].reshape(-1)
        for e in idx.unique().tolist():
            rows = (idx == e).nonzero(as_tuple=False).squeeze(-1)
            n = rows.numel()
            if n == 0:
                continue
            he = h_flat.index_select(0, rows).contiguous()
            h1 = he.new_empty(n, F)
            grid = (triton.cdiv(n, 32), triton.cdiv(F, 64))
            _lr_expert_kernel[grid](
                he, V1[e].contiguous(), U1[e].contiguous(), b1[e].contiguous(),
                float(s1[e]),
                h1, n, D, F, R,
                he.stride(0), he.stride(1),
                V1[e].stride(0), V1[e].stride(1),
                U1[e].stride(0), U1[e].stride(1),
                h1.stride(0), h1.stride(1),
                BLOCK_N=32, BLOCK_F=64, BLOCK_R=16,
            )
            ye = h1.new_empty(n, D)
            grid2 = (triton.cdiv(n, 32), triton.cdiv(D, 64))
            _lr_expert_kernel[grid2](
                h1, V2[e].contiguous(), U2[e].contiguous(), b2[e].contiguous(),
                float(s2[e]),
                ye, n, F, D, R,
                h1.stride(0), h1.stride(1),
                V2[e].stride(0), V2[e].stride(1),
                U2[e].stride(0), U2[e].stride(1),
                ye.stride(0), ye.stride(1),
                BLOCK_N=32, BLOCK_F=64, BLOCK_R=16,
            )
            out[rows, k] = ye
    return out.reshape(B, L, K, D)
