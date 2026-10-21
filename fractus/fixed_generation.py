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
    ch = {k[len('copy_head.'):]: v for k, v in out.items() if k.startswith('copy_head.')}
    out = {k: v for k, v in out.items() if not k.startswith('copy_head.')}
    missing, unexpected = eng.load_state_dict(out, strict=True, assign=True)
    if ch:   # v5 copy-head checkpoint (docs/COPY_HEAD.md): readout head, strict on its own keys
        from .nn.copy_head import attach_copy_head
        c = cfg.get('copy_head', {})
        attach_copy_head(eng, n_heads=int(c.get('n_heads', 8)), d_head=int(c.get('d_head', 64)),
                         mem=int(c.get('mem', 128)), shift_keys=bool(c.get('shift_keys', True)),
                         ngram=int(c.get('ngram', 1)))
        eng.copy_head.load_state_dict({k: v.to(dtype) for k, v in ch.items()}, strict=True)
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
    def __init__(self, eng: ContinuousThoughtEngine, final_norm: str | None = None, warm: int = 64,
                 carry_decay: float = 1.0):
        self.eng = eng
        self.set_carry_decay(carry_decay)
        self.tok = AtomFractusTokenizer()
        self.final_norm = final_norm
        self.warm = warm
        self.dtype = next(p for n, p in eng.named_parameters() if 'observe' in n).dtype
        self.h_ref = None   # reference norm for 'rms' (set by calibrate or first call)

    def set_carry_decay(self, gamma: float = 1.0):
        """Inference-only: each chunk reads gamma * (attn_S, attn_z) (fractus/carry_policy.py, docs/LONG_CARRY.md).
        1.0 (default) = the trained function."""
        from .carry_policy import set_carry_decay
        set_carry_decay(self.eng, gamma)

    # ---- carry
    # Engines with a copy head (fractus/nn/copy_head.py) carry one more piece of state: the K/V window.
    # The snapshot is then (thought_state, blocks, copy_head_state); without a head it is unchanged.
    def cold(self):
        e = self.eng
        e.reset_thought(1)
        e.thought_state = e.thought_state.to(self.dtype)
        for b in e.blocks:
            b.attn_S, b.attn_z, b.kuramoto_phases = (b.attn_S.to(self.dtype), b.attn_z.to(self.dtype),
                                                     b.kuramoto_phases.to(self.dtype))
        if getattr(e, 'copy_head', None) is not None:
            e.copy_head.reset_state(1, dtype=self.dtype)

    def get(self):
        e = self.eng
        s = (e.thought_state.clone(), [(b.attn_S.clone(), b.attn_z.clone(), b.kuramoto_phases.clone()) for b in e.blocks])
        if getattr(e, 'copy_head', None) is not None:
            s = s + (e.copy_head.get_state(),)
        return s

    def put(self, s):
        e = self.eng
        e.thought_state = s[0].clone()
        for b, (x, z, k) in zip(e.blocks, s[1]):
            b.attn_S, b.attn_z, b.kuramoto_phases = x.clone(), z.clone(), k.clone()
        if len(s) > 2:
            e.copy_head.put_state(s[2])

    def readout(self, h):
        """Body hidden (B, L, D) -> what the tied head reads (copy head applied, window updated)."""
        ch = getattr(self.eng, 'copy_head', None)
        return h if ch is None else ch.attend(h)

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
        return self.head(self.readout(h)[0])

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

    def _commit(self, seq, com, carry_window):
        """commit seq[com:com+SEQ]. carry_window K>0: the committed state is a cold replay of only the last K
        committed chunks (sliding window, docs/LONG_CARRY.md); 0 = full carry (trained function)."""
        if carry_window and carry_window > 0:
            self.cold()
            for j in range(max(0, com + SEQ - carry_window * SEQ), com + SEQ, SEQ):
                self.chunk(seq[j:j + SEQ])
        else:
            self.chunk(seq[com:com + SEQ])
        return com + SEQ

    def generate(self, ctx, max_new=40, temperature=0.0, top_k=1, seed=0, stop_at_eos=True, carry_window=0):
        """ctx: full id context (warm tail + BOS + prompt). Grid starts at ctx[0], zero carry."""
        g = torch.Generator().manual_seed(seed)
        seq = list(ctx)
        self.cold()
        com = 0
        while len(seq) - com > SEQ:
            com = self._commit(seq, com, carry_window)
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
                com = self._commit(seq, com, carry_window)
        return out

    # ---- exact incremental stepping (docs/GEN_FINAL.md)
    # Same function as generate(): fixed 128-id grid, committed carry, thought_state added at grid position 0 only.
    # Instead of re-forwarding the whole open chunk for every new id, the open chunk keeps its PER-LEVEL linear
    # attention state (the kernel state before the mean over levels that _store_carry does at a chunk end), so each
    # new id costs one 1-position forward. Every op of the block is per position (attention is causal, Kuramoto
    # and MoE act per position), so the logits equal the re-forward ones up to float rounding.
    @torch.inference_mode()
    def _core(self, ids, flats, first: bool):
        e = self.eng
        h = e.observe(torch.tensor([list(ids)])).clone()
        if first:
            h[:, 0, :] = h[:, 0, :] + e.thought_state[:, 0, :]
        new = []
        for blk, (S, z) in zip(e.blocks, flats):
            h, _, S2, z2, _th = blk._tick_chunk_core_pure(h, S, z)
            new.append((S2, z2))
        return h, new

    @torch.inference_mode()
    def _open_flats(self):
        # _build_carry applies the optional inference carry decay once per chunk, exactly like chunk() does.
        return [tuple(t.clone() for t in blk._build_carry(1)) for blk in self.eng.blocks]

    @torch.inference_mode()
    def stream_start(self, ctx, carry_window: int = 0) -> torch.Tensor:
        """Commit every full chunk of ctx, forward the open part once; returns next-id logits (V,)."""
        if getattr(self.eng, 'copy_head', None) is not None:
            raise NotImplementedError('stream path: engines with a copy head use generate()')
        self.cold()
        self.seq, self.com, self.cw = list(ctx), 0, carry_window
        while len(self.seq) - self.com > SEQ:
            self.com = self._commit(self.seq, self.com, carry_window)
        h, self.flats = self._core(self.seq[self.com:], self._open_flats(), True)
        self.h_last = h[:, -1:, :]
        return self.head(h[0, -1:])[0]

    @torch.inference_mode()
    def stream_push(self, nid: int) -> torch.Tensor:
        """Append one id; returns next-id logits (V,)."""
        e = self.eng
        self.seq.append(int(nid))
        if len(self.seq) - self.com > SEQ:          # the open chunk is full: commit it, nid opens the next one
            if self.cw and self.cw > 0:
                self.com = self._commit(self.seq, self.com, self.cw)
            else:
                for blk, (S, z) in zip(e.blocks, self.flats):
                    blk._store_carry(1, S, z)
                e.thought_state = self.h_last.detach()
                self.com += SEQ
            h, self.flats = self._core([nid], self._open_flats(), True)
        else:
            h, self.flats = self._core([nid], self.flats, False)
        self.h_last = h[:, -1:, :]
        return self.head(h[0, -1:])[0]

    def generate_stream(self, ctx, max_new=40, *, temperature=0.0, top_k=0, top_p=1.0, min_p=0.0,
                        rep_penalty=1.0, rep_window=64, no_repeat_ngram=0, min_new=0, stop_at_eos=True,
                        carry_window=0, seed=0):
        """Decode with the incremental path. Only PAD and BOS are masked; EOS is masked only while len(out) < min_new.
        Boundary markers (<B1>..) are ordinary ids: nothing is forced and generation does not stop on them."""
        g = torch.Generator().manual_seed(seed)
        lg = self.stream_start(ctx, carry_window)
        out = []
        for _ in range(max_new):
            nid = pick_next(lg, out, g, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p,
                            rep_penalty=rep_penalty, rep_window=rep_window, no_repeat_ngram=no_repeat_ngram,
                            ban_eos=len(out) < min_new)
            out.append(nid)
            if stop_at_eos and nid == EOS:
                break
            lg = self.stream_push(nid)
        return out


def pick_next(logits, gen, g=None, *, temperature=0.0, top_k=0, top_p=1.0, min_p=0.0, rep_penalty=1.0,
              rep_window=64, no_repeat_ngram=0, ban_eos=False) -> int:
    """Logits processing for one id. Order: masks -> repetition penalty -> n-gram ban -> T -> top-k -> top-p -> min-p.
    gen = ids generated so far (the prompt is not penalized). Greedy when temperature <= 0."""
    z = logits.float().clone()
    z[PAD] = -1e9; z[BOS] = -1e9
    if ban_eos:
        z[EOS] = -1e9
    if rep_penalty != 1.0 and gen:
        ids = torch.tensor(sorted(set(gen[-rep_window:])))
        v = z[ids]
        z[ids] = torch.where(v > 0, v / rep_penalty, v * rep_penalty)
    n = int(no_repeat_ngram)
    if n > 0 and len(gen) >= n - 1:
        pre = tuple(gen[len(gen) - (n - 1):]) if n > 1 else ()
        for i in range(len(gen) - n + 1):
            if tuple(gen[i:i + n - 1]) == pre:
                z[gen[i + n - 1]] = -1e9
    if temperature <= 0:
        return int(z.argmax())
    z = z / temperature
    if top_k and top_k > 0:
        kth = torch.topk(z, min(int(top_k), z.numel())).values[-1]
        z[z < kth] = -float('inf')
    p = torch.softmax(z, -1)
    if top_p < 1.0:
        sp, si = p.sort(descending=True)
        drop = (sp.cumsum(0) - sp) > top_p
        p[si[drop]] = 0.0
    if min_p > 0:
        p = p * (p >= min_p * p.max())
    return int(torch.multinomial(p / p.sum(), 1, generator=g))


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
