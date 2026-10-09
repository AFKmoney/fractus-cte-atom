"""AR-facing training extras: scheduled-sampling mix, SS_PROB ramp,
anti-repeat log-prob (penalize echoing the last input token).

Used by scripts/fast4gpu_boost_v4.py. Shapes-compatible with the v3 CE path.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def ss_prob_at(
    tokens_done: int,
    start: float = 0.2,
    end: float = 0.5,
    ramp_tokens: int = 50_000_000,
) -> float:
    """Linear ramp start → end over ramp_tokens. Clamped to [start, end]."""
    if ramp_tokens <= 0:
        return float(end)
    t = min(1.0, max(0.0, tokens_done / float(ramp_tokens)))
    return float(start + (end - start) * t)


def mix_scheduled_sampling(
    chunk: torch.Tensor,
    sampled: torch.Tensor,
    ss_prob: float,
) -> torch.Tensor:
    """Replace a fraction of *previous* tokens with model samples.

    Position 0 is never replaced (need a real start). `sampled[:, t]` is the
    model's draw at t; we feed it as input at t+1 (standard SS).
    """
    B, C = chunk.shape
    use_ss = torch.rand(B, C, device=chunk.device) < ss_prob
    use_ss[:, 0] = False
    prev = torch.cat([chunk[:, :1], sampled[:, :-1]], dim=1)
    return torch.where(use_ss, prev, chunk)


def _chunk_repeat_logit(hi: torch.Tensor, weight: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    # RAW logit of the echo class, not logprob. Tied-head + residual makes
    # p(echo)≈1 at init so log p saturates at 0 and d(log p)/dθ ≈ 0.
    logits = F.linear(hi, weight).float()
    return logits.gather(-1, pi.view(-1, 1)).sum()


def chunked_repeat_logprob(
    h: torch.Tensor,
    weight: torch.Tensor,
    prev_ids: torch.Tensor,
    ce_chunk: int = 8192,
) -> torch.Tensor:
    """Mean logit of class prev_ids[i] at row i.

    Name kept for call-sites. Add λ · this to the loss to suppress
    last-token echo. Memory-flat like chunked CE.
    """
    assert h.dim() == 2 and prev_ids.dim() == 1 and h.shape[0] == prev_ids.shape[0]
    n = h.shape[0]
    if n == 0:
        return h.new_zeros((), dtype=torch.float32)
    total = None
    for i in range(0, n, ce_chunk):
        hi = h[i : i + ce_chunk]
        pi = prev_ids[i : i + ce_chunk]
        if torch.is_grad_enabled() and n > ce_chunk:
            part = checkpoint(_chunk_repeat_logit, hi, weight, pi, use_reentrant=False)
        else:
            part = _chunk_repeat_logit(hi, weight, pi)
        total = part if total is None else total + part
    return total / n


def echo_frac(generated: list[int], last_prompt_id: int) -> float:
    """Fraction of greedy tokens equal to the last prompt token."""
    if not generated:
        return 0.0
    return sum(1 for t in generated if t == last_prompt_id) / len(generated)
