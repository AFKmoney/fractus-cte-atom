"""One v4 training step: TF CE + Switch-LB + anti-repeat + optional SS.

Separated from the 1B launcher so tests can run it on a tiny CPU engine.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from fractus.nn.ce import sample_tokens_chunked
from fractus.train.ar_loss import chunked_repeat_logprob, mix_scheduled_sampling


def v4_forward_losses(
    eng,
    chunk: torch.Tensor,
    target: torch.Tensor,
    *,
    lb_coef: float = 0.02,
    repeat_coef: float = 0.1,
    ce_chunk: int = 2048,
    block_ckpt: bool = False,
):
    """Teacher-forced losses. Returns (total, extras_dict). extras has tensors
    with grad: ce, lb, repeat, h.
    """
    out = eng.tick_chunk_train_ce(
        chunk, target, ce_chunk=ce_chunk, return_hidden=True, block_ckpt=block_ckpt,
    )
    ce, lb, h = out
    weight = eng.output_head.weight
    repeat = chunked_repeat_logprob(
        h.reshape(-1, h.shape[-1]),
        weight,
        chunk.reshape(-1),
        ce_chunk=ce_chunk,
    )
    total = ce + lb_coef * lb + repeat_coef * repeat
    return total, {"ce": ce, "lb": lb, "repeat": repeat, "h": h}


def v4_ss_pass(
    eng,
    chunk: torch.Tensor,
    target: torch.Tensor,
    h: torch.Tensor,
    *,
    ss_prob: float,
    lb_coef: float = 0.02,
    ce_chunk: int = 2048,
    block_ckpt: bool = False,
):
    """Scheduled-sampling pass. Samples from detached TF hidden, mixes into
    inputs, CE again. Returns (loss, ce_ss).
    """
    with torch.no_grad():
        samp = sample_tokens_chunked(
            h.reshape(-1, h.shape[-1]).detach(),
            eng.output_head.weight,
            temperature=0.9,
            ce_chunk=ce_chunk,
        ).view_as(chunk)
    mixed = mix_scheduled_sampling(chunk, samp, ss_prob)
    # Dense CE — B*SEQ*vocab is tiny at B<=4. Avoids nn.ce checkpoint
    # metadata collision on the second pass.
    logits, lb2 = eng.tick_chunk_train(mixed)
    ce_ss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        target.reshape(-1),
    )
    return 0.5 * ce_ss + lb_coef * lb2, ce_ss


def should_ss(ss_rate: float) -> bool:
    return random.random() < ss_rate


def snapshot_carry(eng):
    snap = {
        'thought': eng.thought_state.detach().clone(),
        'blocks': [],
    }
    for blk in eng.blocks:
        item = {}
        if hasattr(blk, 'attn_S'):
            item['attn_S'] = blk.attn_S.detach().clone()
        if hasattr(blk, 'attn_z'):
            item['attn_z'] = blk.attn_z.detach().clone()
        if hasattr(blk, 'kuramoto_phases'):
            item['phases'] = blk.kuramoto_phases.detach().clone()
        snap['blocks'].append(item)
    return snap


def restore_carry(eng, snap):
    eng.thought_state = snap['thought'].clone()
    for blk, item in zip(eng.blocks, snap['blocks']):
        if 'attn_S' in item:
            blk.attn_S = item['attn_S'].clone()
        if 'attn_z' in item:
            blk.attn_z = item['attn_z'].clone()
        if 'phases' in item:
            blk.kuramoto_phases = item['phases'].clone()
