"""Inference path that computes the TRAINING function of fast4gpu_atom_v3 (docs/DEEP_DIVE_GENERATION.md).

Inference only. The trainer does not import this file.

What it does, and why (measured on v3 step_10000 / step_85000):
  * Fixed 128-id chunk grid. Committed chunks keep their carry (attn_S, attn_z, thought_state).
    The open chunk is re-forwarded from the committed carry on every step and then thrown away.
    gen_block mode A and the old Space re-added the window to the carry on every step
    (double counting): top-1 0.20-0.25 instead of 0.85 teacher-forced on the same positions.
  * Warm start (WARM_TAIL): the end of a real corpus document (... <B6><EOS>) is placed before BOS
    in the first chunk, from zero carry. That is what a lane looks like in training: reset_rows zeroes
    the carry of the chunk that contains BOS, and BOS sits at a random in-chunk position after the
    previous document's tail. A cold BOS at position 0 is a state training almost never produces.
  * Load like the trainer resume: strict=True after resetting the batch-sized carry buffers,
    gate temperature from the config, Kuramoto routing fix NOT re-applied (omega x4 would compound).
  * Optional final norm before the tied head (final_norm='ln' | 'rms' | None). The weights were trained
    WITHOUT one. 'ln' (no affine) changes the function. 'rms' is a per-position positive rescale of h:
    greedy argmax is unchanged and only sampling temperature is affected.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .atom_tokenizer import VOCAB_SIZE, AtomFractusTokenizer
from .continuous_engine import ContinuousThoughtEngine

SEQ = 128
PAD, BOS, EOS = 256, 257, 258
CFG_KEYS = ('d_model', 'n_heads', 'd_head', 'n_levels', 'n_oscillators', 'coupling_rank',
            'n_experts', 'top_k', 'expert_d_ff', 'siren_rank', 'n_layers')
# Last 192 ids of reasoning__syllogisms doc 0 (ends "... mental models." <B6><EOS>), real corpus ids.
WARM_TAIL = [116, 32, 109, 105, 115, 116, 97, 107, 101, 115, 46, 10, 261, 10, 261, 84, 104, 105, 115, 32, 99, 111, 109, 112, 97, 99, 116, 32, 116, 114, 101, 97, 116, 109, 101, 110, 116, 32, 105, 115, 32, 259, 97, 32, 115, 116, 97, 114, 116, 105, 110, 103, 32, 112, 111, 105, 110, 116, 59, 32, 100, 101, 112, 116, 104, 32, 259, 99, 111, 109, 101, 115, 32, 102, 114, 111, 109, 32, 100, 101, 108, 105, 98, 101, 114, 97, 116, 101, 32, 112, 114, 97, 99, 116, 105, 99, 101, 32, 259, 111, 110, 32, 114, 101, 97, 108, 32, 99, 97, 115, 101, 115, 44, 32, 104, 111, 110, 101, 115, 116, 32, 114, 101, 118, 105, 101, 119, 32, 259, 111, 102, 32, 111, 117, 116, 99, 111, 109, 101, 115, 44, 32, 97, 110, 100, 32, 105, 116, 101, 114, 97, 116, 105, 111, 110, 32, 259, 111, 110, 32, 116, 104, 101, 32, 117, 110, 100, 101, 114, 108, 121, 105, 110, 103, 32, 109, 101, 110, 116, 97, 108, 32, 259, 109, 111, 100, 101, 108, 115, 46, 264, 258]


def build_from_state(sd: dict, cfg: dict, dtype=torch.float32) -> ContinuousThoughtEngine:
    """Rebuild the engine exactly like fast4gpu_atom_v3.build_engine on resume."""
    sd = {(k[10:] if k.startswith('_orig_mod.') else k): v for k, v in sd.items()}
    with torch.device('meta'):
        eng = ContinuousThoughtEngine(vocab_size=VOCAB_SIZE, **{k: cfg[k] for k in CFG_KEYS})
    msd = eng.state_dict()
    for k in [k for k in sd if k in msd and sd[k].shape != msd[k].shape]:
        sd[k] = torch.zeros(msd[k].shape)          # batch-sized carry buffers; reset_thought(1) anyway
    out = {}
    for k, v in sd.items():
        if v.is_floating_point():
            # siren_* are the unused grow copies (not on the forward path): keep them as stored
            v = v if '.siren_' in k else v.to(dtype)
        out[k] = v
    missing, unexpected = eng.load_state_dict(out, strict=True, assign=True)
    for blk in eng.blocks:
        blk.moe.temperature = float(cfg.get('gate_temp', 2.5))
    eng.eval()
    return eng


def load_checkpoint(path: str, dtype=torch.float32) -> tuple:
    """.pt/.bin ({'model_state','config'}) via mmap, or a bf16raw dir from scripts/ckpt_to_bf16raw.py."""
    if os.path.isdir(path):
        ix = json.load(open(os.path.join(path, 'index.json')))
        mm = np.memmap(os.path.join(path, 'weights.bf16'), dtype=np.uint8, mode='c')
        sd = {}
        for k, m in ix['tensors'].items():
            a = mm[m['off']:m['off'] + m['nbytes']]
            t = (torch.from_numpy(a.view(np.int16)).view(torch.bfloat16) if m['dtype'] == 'bfloat16'
                 else torch.from_numpy(a).view(getattr(torch, m['dtype'])))
            sd[k] = t.view(m['shape'])
        cfg = ix['config']
    else:
        ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        sd, cfg = ck['model_state'], ck['config']
    return build_from_state(sd, cfg, dtype), cfg


class FixedGenerator:
    def __init__(self, eng: ContinuousThoughtEngine, final_norm: str | None = None, warm: int = 64):
        self.eng = eng
        self.tok = AtomFractusTokenizer()
        self.final_norm = final_norm
        self.warm = warm
        self.dtype = next(p for n, p in eng.named_parameters() if 'observe' in n).dtype
        self.h_ref = None   # reference norm for 'rms' (set by calibrate or first call)

    # ---- carry
    def cold(self):
        e = self.eng
        e.reset_thought(1)
        e.thought_state = e.thought_state.to(self.dtype)
        for b in e.blocks:
            b.attn_S, b.attn_z, b.kuramoto_phases = (b.attn_S.to(self.dtype), b.attn_z.to(self.dtype),
                                                     b.kuramoto_phases.to(self.dtype))

    def get(self):
        e = self.eng
        return (e.thought_state.clone(), [(b.attn_S.clone(), b.attn_z.clone(), b.kuramoto_phases.clone()) for b in e.blocks])

    def put(self, s):
        e = self.eng
        e.thought_state = s[0].clone()
        for b, (x, z, k) in zip(e.blocks, s[1]):
            b.attn_S, b.attn_z, b.kuramoto_phases = x.clone(), z.clone(), k.clone()

    # ---- one chunk = tick_chunk_train math (no counters), with the optional final norm
    @torch.inference_mode()
    def chunk(self, ids) -> torch.Tensor:
        e = self.eng
        x = torch.tensor([list(ids)])
        h = e.observe(x).clone()
        h[:, 0, :] = h[:, 0, :] + e.thought_state[:, 0, :]
        for blk in e.blocks:
            h, _ = blk.tick_chunk_core(h)
        e.thought_state = h[:, -1:, :].detach()
        return self.head(h[0])

    def head(self, h: torch.Tensor) -> torch.Tensor:
        h = h.float()
        if self.final_norm == 'ln':
            h = F.layer_norm(h, h.shape[-1:])
        elif self.final_norm == 'rms':
            n = h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            ref = self.h_ref if self.h_ref is not None else float(n.median())
            h = h * (ref / n)
        return h @ self.eng.output_head.weight.float().t()

    # ---- full teacher-forced logits on the fixed grid (for TF loss / parity)
    def tf_logits(self, ids) -> torch.Tensor:
        self.cold()
        return torch.cat([self.chunk(ids[j:j + SEQ]) for j in range(0, len(ids), SEQ)], 0)

    def context(self, prompt_ids, warm: int | None = None):
        w = self.warm if warm is None else warm
        return (WARM_TAIL[-w:] if w > 0 else []) + list(prompt_ids)

    def encode_prompt(self, text: str):
        ids, _ = self.tok.encode_open(text)    # BOS + bytes (+ markers), no closing <B6><EOS>
        return list(ids)

    def generate(self, ctx, max_new=40, temperature=0.0, top_k=1, seed=0, stop_at_eos=True):
        """ctx: full id context (warm tail + BOS + prompt). Grid starts at ctx[0], zero carry."""
        g = torch.Generator().manual_seed(seed)
        seq = list(ctx)
        self.cold()
        com = 0
        while len(seq) - com > SEQ:
            self.chunk(seq[com:com + SEQ]); com += SEQ
        out = []
        for _ in range(max_new):
            s = self.get(); lg = self.chunk(seq[com:])[-1].clone(); self.put(s)
            lg[PAD] = -1e9; lg[BOS] = -1e9
            if temperature <= 0:
                nid = int(lg.argmax())
            else:
                v, i = torch.topk(lg / temperature, max(1, int(top_k)))
                nid = int(i[torch.multinomial(torch.softmax(v, -1), 1, generator=g)])
            out.append(nid); seq.append(nid)
            if stop_at_eos and nid == EOS:
                break
            if len(seq) - com > SEQ:
                self.chunk(seq[com:com + SEQ]); com += SEQ
        return out


def show(ids) -> str:
    """Decode with boundary markers shown (<B1>.., <EOS>)."""
    s, buf = [], []
    for i in ids:
        i = int(i)
        if i >= 256:
            s.append(bytes(buf).decode('utf-8', 'replace')); buf = []
            s.append({PAD: '<PAD>', BOS: '<BOS>', EOS: '<EOS>'}.get(i, f'<B{i - 258}>'))
        else:
            buf.append(i)
    s.append(bytes(buf).decode('utf-8', 'replace'))
    return ''.join(s)
