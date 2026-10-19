"""Copy head: a small exact causal SOFTMAX attention added at the readout of Fractus CTE-Atom.

Why (docs/COPY_HEAD.md): every CTEBlock mixes positions only through LINEAR attention
(elu_plus_one feature maps, carry S/z). The v4 copy probe shows ~zero copy gain: a word repeated in
the same 128-id chunk costs 2.86 nats/id vs 2.96 at its first mention. Linear attention compresses
the past into a dH x dH state per head and has no sharp lookup, so exact recall is hard for it.
This module adds one sharp lookup over a short exact window and nothing else.

Design
  * Placement: READOUT. h_body = output of the last CTEBlock (unchanged recurrence: thought_state and
    every block carry still come from h_body). logits = output_head(h_body + gate * out).
    The body never sees the head, so the head can be fine-tuned on cached body states, and a
    zero gate gives back the old function bit for bit.
  * Window: the current 128-id chunk (causal) + the cached K/V of the last `mem` ids of the
    previous chunk(s). One fixed-size window per row. No growth with context.
  * Induction-friendly keys: key at position j is built from x_{j-1} (the previous position's
    normed state), value from x_j. Query at t = x_t. A match "x_{j-1} ~ x_t" returns x_j, which is the
    continuation the first time this context was seen. One layer can copy with this shift.
    (shift_keys=False gives a plain causal head.)
  * ngram > 1: query from x_t .. x_{t-g+1}, key from x_{j-1} .. x_{j-g} (one projection of the concatenation),
    so a match needs the last g positions to agree (single-position keys are ambiguous on letters).
  * Gate: per-channel vector, init 0 => output exactly h_body (h + 0*y == h bitwise for finite y).
  * State per row (B): K/V cache (B, n, nH, dH), validity mask (B, n), last normed state (B, 1, D)
    for the shifted key of the next chunk's position 0. All detached between chunks (as the carry).

Attach: attach_copy_head(eng, ...) adds eng.copy_head (state_dict keys copy_head.*) and wraps
eng.tick_chunk_train_ce / eng.tick_chunk_train so the trainer paths use it. Old checkpoints load with
load_with_copy_head (strict on every old key, only copy_head.* may be missing).
"""
from __future__ import annotations

import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


class CopyHead(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, d_head: int = 64, mem: int = 128,
                 shift_keys: bool = True, ngram: int = 1):
        super().__init__()
        self.ngram = ngram
        self.d_model, self.n_heads, self.d_head, self.mem = d_model, n_heads, d_head, mem
        self.shift_keys = shift_keys
        self.diag_only = False   # ablation: attend only to the own position (= a plain linear readout correction)
        # ghost > 0: add (o - o.detach()) * ghost. Value exactly 0 (the forward is unchanged), but q/k/v/o get the
        # gradient they would get with gate = ghost. With gate = 0 and a sparse copy signal they otherwise get none.
        self.ghost = 0.0
        self.norm = nn.LayerNorm(d_model)
        inner = n_heads * d_head
        self.w_q = nn.Linear(d_model * ngram, inner, bias=False)
        self.w_k = nn.Linear(d_model * ngram, inner, bias=False)
        self.w_v = nn.Linear(d_model, inner, bias=False)
        self.w_o = nn.Linear(inner, d_model, bias=False)
        self.gate = nn.Parameter(torch.zeros(d_model))
        for w in (self.w_q, self.w_k, self.w_v):
            nn.init.normal_(w.weight, std=1.0 / math.sqrt(w.in_features))
        nn.init.normal_(self.w_o.weight, std=1.0 / math.sqrt(inner))
        self.reset_state(1)

    # ---------------- state
    def reset_state(self, B: int, device=None, dtype=None):
        p = self.w_q.weight
        device = device or p.device
        dtype = dtype or p.dtype
        self.ck = torch.zeros(B, 0, self.n_heads, self.d_head, device=device, dtype=dtype)
        self.cv = torch.zeros(B, 0, self.n_heads, self.d_head, device=device, dtype=dtype)
        self.cm = torch.zeros(B, 0, dtype=torch.bool, device=device)
        self.lx = torch.zeros(B, self.ngram, self.d_model, device=device, dtype=dtype)   # last ngram normed states
        self.lv = torch.zeros(B, self.ngram, dtype=torch.bool, device=device)            # lx valid?

    def reset_rows(self, mask: torch.Tensor):
        """Forget the window of the rows in mask (document boundary), like trainer reset_rows."""
        keep = ~mask.to(self.cm.device).view(-1, 1)
        self.cm = self.cm & keep
        self.lv = self.lv & keep
        self.lx = self.lx * keep.view(-1, 1, 1).to(self.lx.dtype)

    def get_state(self):
        return (self.ck.clone(), self.cv.clone(), self.cm.clone(), self.lx.clone(), self.lv.clone())

    def put_state(self, s):
        self.ck, self.cv, self.cm, self.lx, self.lv = (t.clone() for t in s)

    # ---------------- math
    def attend(self, h: torch.Tensor, update: bool = True) -> torch.Tensor:
        """h: (B, C, D) body output of one chunk. Returns h + gate * y. Updates the window if update."""
        B, C, D = h.shape
        if self.ck.shape[0] != B:
            self.reset_state(B, h.device, h.dtype)
        x = self.norm(h)
        nH, dH = self.n_heads, self.d_head
        g = self.ngram
        hist = torch.cat([self.lx.to(x.dtype), x], 1)                                     # (B, g+C, D)
        hv = torch.cat([self.lv, torch.ones(B, C, dtype=torch.bool, device=h.device)], 1)
        xq = torch.cat([hist[:, g - i:g - i + C] for i in range(g)], -1)                  # x_t .. x_{t-g+1}
        if self.shift_keys:
            xk = torch.cat([hist[:, g - 1 - i:g - 1 - i + C] for i in range(g)], -1)      # x_{j-1} .. x_{j-g}
            kvalid = hv[:, g - 1:g - 1 + C]
        else:
            xk = xq
            kvalid = torch.ones(B, C, dtype=torch.bool, device=h.device)
        q = self.w_q(xq).view(B, C, nH, dH)
        k = self.w_k(xk).view(B, C, nH, dH)
        v = self.w_v(x).view(B, C, nH, dH)
        K = torch.cat([self.ck.to(k.dtype), k], 1)            # (B, n+C, nH, dH)
        V = torch.cat([self.cv.to(v.dtype), v], 1)
        n = self.ck.shape[1]
        causal = torch.ones(C, C, dtype=torch.bool, device=h.device).tril()
        allow = torch.cat([self.cm.view(B, 1, n).expand(B, C, n),
                           causal.view(1, C, C).expand(B, C, C) & kvalid.view(B, 1, C)], 2)  # (B, C, n+C)
        if self.diag_only:
            allow = torch.zeros_like(allow); allow[:, :, n:] = torch.eye(C, dtype=torch.bool, device=h.device)
        att = torch.einsum('bqhd,bkhd->bhqk', q, K) / math.sqrt(dH)
        att = att.masked_fill(~allow.view(B, 1, C, n + C), float('-inf'))
        w = torch.softmax(att.float(), -1)
        w = torch.nan_to_num(w, nan=0.0).to(V.dtype)        # rows with nothing allowed -> 0
        y = torch.einsum('bhqk,bkhd->bqhd', w, V).reshape(B, C, nH * dH)
        o = self.w_o(y)
        out = h + self.gate.to(h.dtype) * o
        if self.ghost and torch.is_grad_enabled():
            out = out + (o - o.detach()) * self.ghost
        if update:
            m = self.mem
            self.ck = K[:, -m:].detach() if m > 0 else K[:, :0].detach()
            self.cv = V[:, -m:].detach() if m > 0 else V[:, :0].detach()
            self.cm = torch.cat([self.cm, kvalid], 1)[:, -m:] if m > 0 else self.cm[:, :0]
            self.lx = hist[:, -g:].detach()
            self.lv = hv[:, -g:]
        return out

    def extra_repr(self):
        return (f'd_model={self.d_model}, n_heads={self.n_heads}, d_head={self.d_head}, mem={self.mem}, '
                f'shift_keys={self.shift_keys}, ngram={self.ngram}')


# ------------------------------------------------------------------ engine integration
def _body(eng, observations, feat=None, block_ckpt=False):
    """Exactly the body of tick_chunk_train_ce (carry read/mutate outside checkpoint)."""
    obs_vecs = eng.observe(observations)
    if feat is not None:
        obs_vecs = obs_vecs + eng.atom_feat(feat)
    h = obs_vecs.clone()
    h[:, 0, :] = h[:, 0, :] + eng.thought_state[:, 0, :]
    B = observations.shape[0]
    total_lb = torch.tensor(0.0, device=h.device)
    if block_ckpt and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint
        for blk in eng.blocks:
            S, z = blk._build_carry(B)
            h, lb, S_f, z_f, theta = checkpoint(blk._tick_chunk_core_pure, h, S, z, use_reentrant=False)
            blk._store_carry(B, S_f, z_f)
            blk.kuramoto_phases = theta.detach()
            total_lb = total_lb + lb
    else:
        for blk in eng.blocks:
            h, lb = blk.tick_chunk_core(h)
            total_lb = total_lb + lb
    eng.last_lb_loss = total_lb
    eng.thought_state = h[:, -1:, :].detach()          # body state, as before
    return h, total_lb


def attach_copy_head(eng, n_heads=4, d_head=64, mem=128, shift_keys=True, ngram=1):
    """Add eng.copy_head and route tick_chunk_train_ce / tick_chunk_train through it."""
    p = next(eng.parameters())
    eng.copy_head = CopyHead(eng.d_model, n_heads, d_head, mem, shift_keys, ngram).to(device=p.device, dtype=p.dtype)

    def tick_chunk_train_ce(self, observations, targets, ce_chunk=8192, return_hidden=False,
                            block_ckpt=False, feat=None):
        from .ce import chunked_cross_entropy
        h, lb = _body(self, observations, feat, block_ckpt)
        h2 = self.copy_head.attend(h)
        ce = chunked_cross_entropy(h2.reshape(-1, self.d_model), self.output_head.weight,
                                   targets.reshape(-1), ce_chunk=ce_chunk)
        return (ce, lb, h2) if return_hidden else (ce, lb)

    def tick_chunk_train(self, observations, feat=None):
        h, lb = _body(self, observations, feat)
        logits = self.output_head(self.copy_head.attend(h))
        C = observations.shape[1]                      # same hit counter as the original
        self._tick_count = getattr(self, "_tick_count", 0) + int(C)
        n_exp = self.blocks[0].moe.n_experts
        if not hasattr(self, "_expert_hits") or self._expert_hits.numel() != n_exp:
            self._expert_hits = torch.zeros(n_exp)
        with torch.no_grad():
            gates = self.blocks[0].moe._compute_gates(self.blocks[0].kuramoto_phases[:, 0:1, :])
            topk_idx = gates.topk(self.blocks[0].moe.top_k, dim=-1).indices
        self._expert_hits += torch.bincount(topk_idx.reshape(-1), minlength=n_exp).detach().cpu()
        return logits, lb

    orig_reset = eng.reset_thought                      # bound method of the class

    def reset_thought(self, batch_size=1):
        orig_reset(batch_size)
        self.copy_head.reset_state(batch_size)

    eng.tick_chunk_train_ce = types.MethodType(tick_chunk_train_ce, eng)
    eng.tick_chunk_train = types.MethodType(tick_chunk_train, eng)
    eng.reset_thought = types.MethodType(reset_thought, eng)
    return eng.copy_head


def load_with_copy_head(eng, sd: dict):
    """Load an old (no copy head) or new checkpoint. Every non-copy_head key must match (strict);
    copy_head.* may be missing (fresh init, gate=0) but nothing may be unexpected."""
    missing, unexpected = eng.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing if not k.startswith('copy_head.')]
    if bad_missing or unexpected:
        raise RuntimeError(f'load_with_copy_head: missing={bad_missing[:8]} unexpected={list(unexpected)[:8]}')
    return [k for k in missing if k.startswith('copy_head.')]


# ------------------------------------------------------------------ generation (fixed 128 grid)
# fractus.fixed_generation.FixedGenerator and scripts/gen_best.Decoder handle eng.copy_head natively:
# the snapshot gains a third element (the K/V window, validity mask and last normed state), cold() resets it,
# chunk() applies the head before the tied head; thought_state and the block carries stay body-only.
def make_copy_generator(base_cls):
    """Back-compat alias (the base class is copy-head aware)."""
    return base_cls
