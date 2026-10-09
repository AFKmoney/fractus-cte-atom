"""Chunked cross-entropy over the vocabulary (memory-flat, Liger-style).

Why (opt 3, 2026-08-22): tick_chunk_train materializes logits of shape
(B, C, vocab) — at the 1B config that is 4·128·50257 ≈ 25.7M elements, held in
bf16 plus an fp32 copy inside F.cross_entropy and another for its softmax
backward. That is several hundred MB of transient activation per forward,
doubled with scheduled sampling — a large share of the VRAM pressure that
forces B<=4 and blocks torch.compile on the pod.

Math: identical to F.cross_entropy on the full logits —

    loss = (1/N) · Σ_rows CE(row)

computed `ce_chunk` rows at a time. Each chunk is wrapped in
torch.utils.checkpoint so its logits are NOT retained for backward: they are
recomputed once during the backward pass. Peak activation becomes ONE chunk
(ce_chunk · vocab · 4 bytes) instead of all N = B·C rows — e.g. ce_chunk=2048
→ ~0.41 GB transient regardless of batch size, vs ~3.3 GB retained at B=16
dense. Cost: one extra head matmul per chunk in backward (+~25% head FLOPs).
NOTE at legacy B=4: N=512 rows < any useful ce_chunk, so chunking engages
only when batch grows — which is exactly the regime we are unlocking.

Open-heart: parameter shapes untouched; purely a runtime computation change.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_ce(hi: torch.Tensor, weight: torch.Tensor, ti: torch.Tensor) -> torch.Tensor:
    logits_i = F.linear(hi, weight)
    return F.cross_entropy(logits_i.float(), ti, reduction="sum")


def chunked_cross_entropy(
    h: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ce_chunk: int = 8192,
) -> torch.Tensor:
    """Mean next-token CE without retaining full-vocab logits for backward.

    h       : (N, d_model)   hidden states (flattened B*C rows)
    weight  : (vocab, d_model) tied output/embedding weight
    targets : (N,)           target token ids
    ce_chunk: rows per chunk — peak transient = ce_chunk · vocab · 4 bytes

    Returns scalar mean CE. Grad flows to h and weight exactly as the dense
    path (values match within float32 accumulation-order rounding).
    """
    assert h.dim() == 2 and targets.dim() == 1 and h.shape[0] == targets.shape[0]
    n = h.shape[0]
    if n == 0:
        return h.new_zeros((), dtype=torch.float32)
    total = None
    for i in range(0, n, ce_chunk):
        hi = h[i : i + ce_chunk]
        ti = targets[i : i + ce_chunk]
        # Single-chunk + non-reentrant checkpoint collides with a second
        # CE pass (scheduled sampling). Direct CE is cheap at N<=ce_chunk.
        if torch.is_grad_enabled() and n > ce_chunk:
            part = checkpoint(_chunk_ce, hi, weight, ti, use_reentrant=False)
        else:
            part = _chunk_ce(hi, weight, ti)
        total = part if total is None else total + part
    return total / n


@torch.no_grad()
def sample_tokens_chunked(
    h: torch.Tensor,
    weight: torch.Tensor,
    temperature: float = 0.9,
    ce_chunk: int = 8192,
) -> torch.Tensor:
    """Per-position sample from softmax(logits / T) without full-vocab logits.

    Used by scheduled sampling: statistically identical to one multinomial
    over the whole flattened batch (independent rows either way), but drawn
    chunk-wise so peak memory stays at ce_chunk·vocab. NOTE: the RNG stream is
    consumed per chunk, so samples are NOT seed-identical to the dense
    multinomial — the DISTRIBUTION is identical.
    """
    n = h.shape[0]
    out = torch.empty(n, dtype=torch.long, device=h.device)
    for i in range(0, n, ce_chunk):
        hi = h[i : i + ce_chunk].to(weight.dtype)
        logits = F.linear(hi, weight).float()
        probs = torch.softmax(logits / temperature, dim=-1)
        out[i : i + ce_chunk] = torch.multinomial(probs, 1).view(-1)
    return out
