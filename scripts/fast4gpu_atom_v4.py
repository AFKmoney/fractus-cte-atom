#!/usr/bin/env python3
"""Fractus CTE-Atom trainer — the 1B loop of fast4gpu_boost_v4, made Atom-safe.

v4 (2026-10-18), copy of fast4gpu_atom_v3.py for the real-language corpus atom_corpus_v4
(docs/CORPUS_V4.md). Training math, SS (incl. SS_TAIL_IDS), carry, LR schedule, guards and
checkpoints are unchanged. Env-gated additions (defaults = v3 behavior):
  - CORPUS may be a directory: every train_*.i16 in it is read as one stream, concatenated in
    name order (no copy, one memmap per shard). A single .i16 / .npy still works as before.
  - CORPUS_SWITCH=1: resuming weights + optimizer (CKPT_IN / OPT_IN) on a NEW corpus. The lane
    offset then legitimately starts at START_TOKEN=0 (allowed only with this flag; it is not a
    rewind of the old corpus). IDS_BASE (ids already trained before the switch) is written to the
    checkpoint config and manifest as ids_total = IDS_BASE + ids on the new corpus.
  - VAL_CORPUS (.i16) + VAL_EVERY (steps, default 2000, 0 = off) + VAL_CHUNKS (default 64):
    periodic held-out loss. B lanes of the val file, VAL_CHUNKS consecutive SEQ-id chunks per
    lane from a zero carry (BOS resets per row, as in training), teacher forced, no_grad. The
    training carry is snapshotted before and restored after, so training is not perturbed
    (cost ~ VAL_CHUNKS forward passes). Each val lane starts on a BOS. Logged as
    "VAL ce=... nats/id (... bits/id) on N held-out ids @ ids" and appended to
    OUT_DIR/val_log.jsonl. Also run once before step 1 (VAL_AT_START=1).

v3 (2026-10-18), copy of fast4gpu_atom_v2.py with one env-gated addition (default = v2 behavior):
  - SS_TAIL_IDS (default 0 = off, whole chunk as in v2). When > 0, the scheduled-sampling
    pass may replace an input with the model's own draw only at the last SS_TAIL_IDS input
    positions of each chunk (row of SEQ ids). Positions 0 .. SEQ-SS_TAIL_IDS-1 keep the ground
    truth inputs. Inside the tail, each position is still replaced with probability ss_p
    (same SS_PROB_* ramp), the draw is the same one as v2 (sampled at T=0.9 from the
    detached TF hidden at the previous position), and the loss is unchanged
    (0.5 * CE over all SEQ targets + lb). SS_RATE (fraction of steps with an SS pass),
    carry, LR schedule, guards and checkpoints are identical to v2.
    Extra log field on SS steps: ss_tail = CE of the SS pass on the last SS_TAIL_IDS targets.

v2 (2026-10-17), copy of fast4gpu_atom.py with three env-gated additions (defaults = old behavior):
  - SS_RAMP_ORIGIN_IDS: the SS prob ramp counts ids from this absolute position
    (default 0 = old behavior: ramp on absolute ids, already finished on a resume).
  - CARRY_DECAY (default 1.0 = off): attn_S and attn_z (the carried linear-attention
    state) are multiplied by this factor after each chunk, detached / no_grad.
    CARRY_CLAMP (default 0 = off): clamp the norm of each (row, head) of attn_S / attn_z
    to at most this value after each chunk. thought_state and kuramoto phases untouched.
  - EPOCH_WRAP=1 (default 0): when a lane is exhausted, start the next pass of the corpus.
    START_TOKEN / start_token_next stay monotonic; the lane offset read is
    off % period, period = ((lane_len - 1) // SEQ) * SEQ. All carries reset at a wrap.

Why a new file: boost_v4 cannot train this fork.
  - It loads an x8 checkpoint by default and slices the 50257 head to 266 rows
    (BPE rows read as bytes), with strict=False.
  - It refuses anything that is not a .npy int32 shard, so atom_corpus.i16 is rejected.
  - It sets the gate temperature but not omega x4 (partial Kuramoto fix).
  - It calls eng.set_p0_routing() and eng.routing_stats(), which do not exist on this
    engine, and reads probe keys (gate_go, mean_unique, echo_frac, head) that
    unique40_probe no longer returns. It stops before the first step.
  - Its tok/s counts B*SEQ once per step while SS_RATE=1.0 runs two forward+backward.

What this file does instead:
  - No CKPT_IN  -> fresh engine, vocab 266, full apply_kuramoto_routing_fix, start at id 0.
  - CKPT_IN set -> Atom checkpoints only: head and embed must have 266 rows, strict load,
    no slicing, Kuramoto fix NOT re-applied (omega x4 would compound), START_TOKEN required
    and must not be 0.
  - Corpus: .i16 (raw int16, as written by build_atom_corpus.py) or .npy. Every id is
    checked to be in 0..265 before step 1.
  - Multi-GPU: N_GPU independent processes, each on a contiguous slice of the corpus.
  - Throughput is reported in Atom ids, split: TF ids/s, SS ids/s, probe time, wall.

It never opens a pod and never downloads anything.

CPU smoke (small body):
  SCALE=smoke CORPUS=data/atom_corpus.i16 BATCH=1 SEQ=32 MAX_STEPS=20 \\
    python -u scripts/fast4gpu_atom.py

One GPU benchmark (1B shape, fresh):
  CUDA_VISIBLE_DEVICES=0 GPU_ID=0 N_GPU=1 CORPUS=data/atom_corpus.i16 \\
    BATCH=4 SEQ=128 FRACTUS_ATTN_IMPL=chunked BLOCK_CKPT=1 MAX_STEPS=200 \\
    python -u scripts/fast4gpu_atom.py
"""
from __future__ import annotations

import json
import shutil
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("FRACTUS_ATTN_IMPL", "chunked")
from fractus.atom_tokenizer import VOCAB_SIZE  # noqa: E402
from fractus.continuous_engine import ContinuousThoughtEngine  # noqa: E402
from fractus.generate_aligned import unique40_probe  # noqa: E402
from fractus.kuramoto_fix import apply_kuramoto_routing_fix  # noqa: E402
from fractus.nn.ce import sample_tokens_chunked  # noqa: E402
from fractus.train.ar_loss import ss_prob_at  # noqa: E402
from fractus.train.v4_step import (  # noqa: E402
    restore_carry, should_ss, snapshot_carry, v4_forward_losses, v4_ss_pass,
)

assert VOCAB_SIZE == 266, VOCAB_SIZE


def mix_ss_tail(chunk: torch.Tensor, sampled: torch.Tensor, ss_prob: float, tail: int) -> torch.Tensor:
    """v2 mix_scheduled_sampling restricted to the last `tail` input positions.
    sampled[:, t] is the model draw at t, fed as input at t+1. Position 0 is never replaced.
    tail <= 0 -> whole chunk (identical to fractus.train.ar_loss.mix_scheduled_sampling)."""
    B, C = chunk.shape
    use_ss = torch.rand(B, C, device=chunk.device) < ss_prob
    use_ss[:, 0] = False
    if 0 < tail < C:
        use_ss[:, :C - tail] = False
    prev = torch.cat([chunk[:, :1], sampled[:, :-1]], dim=1)
    return torch.where(use_ss, prev, chunk)


def ss_pass_tail(eng, chunk, target, h, *, ss_prob, tail, lb_coef=0.02, ce_chunk=2048):
    """Same as fractus.train.v4_step.v4_ss_pass, with the mix limited to the tail.
    Returns (loss, ce_ss over all SEQ targets, ce over the last `tail` targets)."""
    with torch.no_grad():
        samp = sample_tokens_chunked(
            h.reshape(-1, h.shape[-1]).detach(), eng.output_head.weight,
            temperature=0.9, ce_chunk=ce_chunk,
        ).view_as(chunk)
    mixed = mix_ss_tail(chunk, samp, ss_prob, tail)
    logits, lb2 = eng.tick_chunk_train(mixed)
    ce_ss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), target.reshape(-1))
    with torch.no_grad():
        t = tail if 0 < tail < chunk.shape[1] else chunk.shape[1]
        ce_tail = torch.nn.functional.cross_entropy(
            logits[:, -t:].reshape(-1, logits.size(-1)).float(), target[:, -t:].reshape(-1))
    return 0.5 * ce_ss + lb_coef * lb2, ce_ss, ce_tail

SCALES = {
    "smoke": dict(
        d_model=64, n_heads=4, d_head=16, n_levels=1,
        n_oscillators=4, coupling_rank=2,
        n_experts=4, top_k=2, expert_d_ff=64, siren_rank=8,
        n_layers=2,
    ),
    "1b": dict(
        d_model=1280, n_heads=20, d_head=64, n_levels=2,
        n_oscillators=16, coupling_rank=8,
        n_experts=128, top_k=2, expert_d_ff=2048, siren_rank=64,
        n_layers=16,
    ),
}
CFG_KEYS = tuple(SCALES["1b"].keys())


# ---------------------------------------------------------------- guards

class ConcatIds:
    """Read-only concatenation of several 1-D int16 memmaps. Supports len() and [a:b] slices
    (step 1), which is all the lane fetch uses. Slices that cross a shard boundary are joined."""

    def __init__(self, arrays):
        self.arrays = arrays
        self.starts = np.concatenate([[0], np.cumsum([a.shape[0] for a in arrays])]).astype(np.int64)
        self.shape = (int(self.starts[-1]),)
        self.dtype = arrays[0].dtype

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        out = np.concatenate([np.asarray(x) for x in self.arrays])
        return out if dtype is None else out.astype(dtype)

    def __getitem__(self, sl):
        if not isinstance(sl, slice) or sl.step not in (None, 1):
            raise TypeError("ConcatIds supports [a:b] slices only")
        a, b, _ = sl.indices(self.shape[0])
        if a == 0 and b == self.shape[0]:
            return self
        if b - a > (1 << 26):   # big slice (per-GPU shard): lazy view, never load it into RAM
            return _ConcatView(self, a, b)
        if b <= a:
            return np.zeros(0, dtype=self.dtype)
        i = int(np.searchsorted(self.starts, a, side="right") - 1)
        parts = []
        while a < b:
            s0, s1 = int(self.starts[i]), int(self.starts[i + 1])
            e = min(b, s1)
            parts.append(np.asarray(self.arrays[i][a - s0:e - s0]))
            a = e
            i += 1
        return parts[0] if len(parts) == 1 else np.concatenate(parts)


class _ConcatView:
    """Lazy [lo:hi] window of a ConcatIds (used when N_GPU > 1 splits the stream)."""

    def __init__(self, base, lo, hi):
        self.base, self.lo, self.shape, self.dtype = base, lo, (hi - lo,), base.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, sl):
        a, b, _ = sl.indices(self.shape[0])
        return self.base[self.lo + a:self.lo + b]


def open_corpus(path: str):
    """Memmap an Atom stream. Refuses anything that is not ids in 0..265."""
    p = str(path)
    if os.path.isdir(p):
        files = sorted(Path(p).glob("train_*.i16"))
        if not files:
            raise SystemExit(f"corpus dir {p}: no train_*.i16 shards")
        arr = ConcatIds([np.memmap(f, dtype=np.int16, mode="r") for f in files])
        print(f"corpus dir {p}: {len(files)} shards " + ", ".join(f.name for f in files), flush=True)
    elif p.endswith(".i16"):
        arr = np.memmap(p, dtype=np.int16, mode="r")
    elif p.endswith(".npy"):
        arr = np.load(p, mmap_mode="r")
        if arr.ndim != 1 or arr.dtype.kind not in "iu":
            raise SystemExit(f"corpus {p}: need a 1-D integer array, got {arr.dtype} {arr.shape}")
    else:
        raise SystemExit(f"corpus {p}: need .i16 or .npy")
    n = int(arr.shape[0])
    if n == 0:
        raise SystemExit(f"corpus {p}: empty")
    step = 1 << 24
    lo, hi = 1 << 30, -1
    for i in range(0, n, step):
        part = np.asarray(arr[i:i + step])
        lo, hi = min(lo, int(part.min())), max(hi, int(part.max()))
    if lo < 0 or hi >= VOCAB_SIZE:
        raise SystemExit(
            f"corpus {p}: ids span {lo}..{hi}, Atom ids must be 0..{VOCAB_SIZE - 1}. "
            "This is not an Atom stream (GPT-2 shards are refused)."
        )
    return arr, n


def check_atom_state(sd: dict, where: str) -> None:
    """Refuse any state dict whose embed or head is not exactly 266 rows."""
    for key in ("observe.weight", "output_head.weight"):
        if key not in sd:
            raise SystemExit(f"{where}: missing {key}, not a ContinuousThoughtEngine checkpoint")
        rows = int(sd[key].shape[0])
        if rows != VOCAB_SIZE:
            raise SystemExit(
                f"{where}: {key} has {rows} rows, need {VOCAB_SIZE}. "
                "x8 / GPT-2 weights do not map onto Atom ids. Not loading, not slicing."
            )


def build_engine(scale: str, ckpt_in: str | None, log):
    """Return (engine, cfg, birth) where birth records how the weights came to be."""
    if not ckpt_in:
        cfg = dict(SCALES[scale])
        eng = ContinuousThoughtEngine(vocab_size=VOCAB_SIZE, **cfg)
        fix = apply_kuramoto_routing_fix(eng, log=log)
        return eng, cfg, {"fresh": True, "kuramoto_fix": fix, "parent": "hf:thefinalboss/fractus-cte"}

    ck = torch.load(ckpt_in, map_location="cpu", weights_only=False)
    sd = ck.get("model_state", ck)
    sd = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    check_atom_state(sd, ckpt_in)
    ck_cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
    if int(ck_cfg.get("vocab_size", VOCAB_SIZE)) != VOCAB_SIZE:
        raise SystemExit(f"{ckpt_in}: config vocab_size={ck_cfg.get('vocab_size')}, need {VOCAB_SIZE}")
    missing = [k for k in CFG_KEYS if k not in ck_cfg]
    if missing:
        raise SystemExit(f"{ckpt_in}: config lacks {missing}; cannot rebuild the body exactly")
    cfg = {k: ck_cfg[k] for k in CFG_KEYS}
    eng = ContinuousThoughtEngine(vocab_size=VOCAB_SIZE, **cfg)
    # carry-state buffers are batch-sized (reset to zero at step 1 anyway by reset_thought(B))
    _msd = eng.state_dict()
    _bt = ("thought_state", "attn_S", "attn_z", "kuramoto_phases")
    for k in [k for k in sd if k in _msd and sd[k].shape != _msd[k].shape and k.split(".")[-1] in _bt]:
        sd[k] = _msd[k]
    eng.load_state_dict(sd, strict=True)
    # The routing fix was applied once, at birth. Re-applying omega x4 would compound.
    if os.environ.get("KURAMOTO_FIX_ON_RESUME", "0") == "1":
        fix = apply_kuramoto_routing_fix(eng, log=log)
    else:
        fix = ck_cfg.get("kuramoto_fix", "applied at birth")
        gate_temp = float(ck_cfg.get("gate_temp", os.environ.get("GATE_TEMP", "2.5")))
        with torch.no_grad():
            for blk in eng.blocks:
                if hasattr(blk, "moe") and hasattr(blk.moe, "temperature"):
                    blk.moe.temperature = gate_temp
    return eng, cfg, {"fresh": False, "resumed_from": str(ckpt_in), "kuramoto_fix": fix,
                      "parent": ck_cfg.get("parent", "hf:thefinalboss/fractus-cte")}


def resolve_start(ckpt_in: str | None) -> int:
    raw = os.environ.get("START_TOKEN")
    if not ckpt_in:
        return int(raw or 0)
    if raw is None:
        raise SystemExit("resume: set START_TOKEN from the manifest (start_token_next)")
    if int(raw) == 0 and os.environ.get("CORPUS_SWITCH", "0") != "1":
        raise SystemExit("resume: START_TOKEN=0 is legal only for a fresh model. Never rewind.")
    return int(raw)


def routing_snapshot(eng) -> dict:
    hits = getattr(eng, "_expert_hits", None)
    out = {"lb": float(getattr(eng, "last_lb_loss", torch.tensor(float("nan"))))}
    if hits is not None and hits.numel() and float(hits.sum()) > 0:
        frac = hits / hits.sum()
        out.update(alive=int((hits > 0).sum()), n=int(hits.numel()), max_frac=float(frac.max()))
    return out


# ---------------------------------------------------------------- main

def main() -> None:
    gpu = int(os.environ.get("GPU_ID", "0"))
    n_gpu = max(1, int(os.environ.get("N_GPU", "1")))
    scale = os.environ.get("SCALE", "1b")
    if scale not in SCALES:
        raise SystemExit(f"SCALE must be one of {sorted(SCALES)}")
    lb_coef = float(os.environ.get("LB_COEF", "0.02"))
    lr = float(os.environ.get("LR", "7e-4"))
    opt_name = os.environ.get("OPT", "sgd")
    ema_beta = float(os.environ.get("EMA_BETA", "0.98"))
    ss_rate = float(os.environ.get("SS_RATE", "1.0"))
    ss_p0 = float(os.environ.get("SS_PROB_START", "0.2"))
    ss_p1 = float(os.environ.get("SS_PROB_END", "0.5"))
    ss_ramp = int(os.environ.get("SS_RAMP_TOKENS", "50000000"))
    ss_origin = int(os.environ.get("SS_RAMP_ORIGIN_IDS", "0"))
    ss_tail = int(os.environ.get("SS_TAIL_IDS", "0"))
    carry_decay = float(os.environ.get("CARRY_DECAY", "1.0"))
    carry_clamp = float(os.environ.get("CARRY_CLAMP", "0"))
    epoch_wrap = os.environ.get("EPOCH_WRAP", "0") == "1"
    repeat_coef = float(os.environ.get("REPEAT_COEF", "0.1"))
    probe_every = int(os.environ.get("PROBE_EVERY", "200"))
    log_every = int(os.environ.get("LOG_EVERY", "40"))
    save_every = int(os.environ.get("SAVE_EVERY", "4000"))
    max_steps = int(os.environ.get("MAX_STEPS", "0"))
    B = int(os.environ.get("BATCH", "4"))
    SEQ = int(os.environ.get("SEQ", "128"))
    ce_chunk = int(os.environ.get("CE_CHUNK", "2048"))
    block_ckpt = os.environ.get("BLOCK_CKPT", "0") == "1"
    use_compile = os.environ.get("COMPILE", "0") == "1"
    sync_timing = os.environ.get("SYNC_TIMING", "1") == "1"
    corpus_path = os.environ.get("CORPUS", str(ROOT / "data" / "atom_corpus.i16"))
    ckpt_in = os.environ.get("CKPT_IN") or None
    out_dir = Path(os.environ.get("OUT_DIR", str(ROOT / "checkpoints" / "atom")))
    ckpt_out = Path(os.environ.get("CKPT_OUT", str(out_dir / f"fractus_atom_{scale}_gpu{gpu}.pt")))
    manifest_out = ckpt_out.parent / f"RESUME_MANIFEST_atom_gpu{gpu}.json"

    log = lambda *a: print(f"GPU {gpu}:", *a, flush=True)  # noqa: E731

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = torch.device("cuda:0")
        autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
        sync = torch.cuda.synchronize
    else:
        from contextlib import nullcontext
        device = torch.device("cpu")
        autocast = nullcontext
        sync = lambda: None  # noqa: E731
    if not sync_timing:
        sync = lambda: None  # noqa: E731

    torch.manual_seed(42 + gpu)

    corpus, n_total = open_corpus(corpus_path)
    lo = gpu * n_total // n_gpu
    hi = (gpu + 1) * n_total // n_gpu
    shard = corpus[lo:hi]
    shard_len = hi - lo
    step_ids = B * SEQ
    # B lanes. Row b always reads lane b, so the carry of row b at step n+1
    # continues exactly the text row b saw at step n. The old layout
    # (one contiguous block viewed as (B, SEQ)) handed row b a carry from text
    # (B-1)*SEQ ids earlier, not its own predecessor, whenever B > 1.
    lane_len = shard_len // B
    if lane_len < SEQ + 2:
        raise SystemExit(f"corpus slice {lo}..{hi} too short for {B} lanes of {SEQ + 1} ids")
    lane_starts = np.arange(B, dtype=np.int64) * lane_len
    period = ((lane_len - 1) // SEQ) * SEQ   # lane offsets 0, SEQ, ..., period - SEQ

    # START_TOKEN / start_token_next is the offset inside each lane.
    start = resolve_start(ckpt_in)
    start = (start // SEQ) * SEQ

    eng, cfg, birth = build_engine(scale, ckpt_in, log)
    n_params = sum(p.numel() for p in eng.parameters())
    log(f"ATOM scale={scale} params={n_params:,} vocab={VOCAB_SIZE} B={B} SEQ={SEQ} "
        f"attn={os.environ['FRACTUS_ATTN_IMPL']} block_ckpt={block_ckpt} ss_rate={ss_rate} "
        f"opt={opt_name} lr={lr} fresh={birth['fresh']}")
    log(f"corpus {corpus_path} total={n_total:,} slice={lo:,}..{hi:,} "
        f"lanes={B}x{lane_len:,} lane_offset={start:,}")
    log(f"v2: ss_prob {ss_p0}->{ss_p1} over {ss_ramp:,} ids from {ss_origin:,} ss_rate={ss_rate} | "
        f"carry_decay={carry_decay} carry_clamp={carry_clamp} | epoch_wrap={epoch_wrap} "
        f"period={period:,} epoch={start // period} lane_off={start % period:,}")
    log(f"v3: ss_tail_ids={ss_tail} (SS replacement only at input positions "
        f"{max(1, SEQ - ss_tail) if 0 < ss_tail < SEQ else 1}..{SEQ - 1} of each {SEQ}-id chunk)")
    ids_base = int(os.environ.get("IDS_BASE", "0"))
    corpus_switch = os.environ.get("CORPUS_SWITCH", "0") == "1"
    log(f"v4: corpus_switch={corpus_switch} ids_base={ids_base:,} (ids trained before this corpus)")

    # --- v4: held-out validation shard
    val_path = os.environ.get("VAL_CORPUS") or None
    val_every = int(os.environ.get("VAL_EVERY", "2000"))
    val_chunks = int(os.environ.get("VAL_CHUNKS", "64"))
    val_arr = None
    if val_path and val_every > 0:
        val_arr, n_val = open_corpus(val_path)
        v_lane = n_val // B
        v_chunks = min(val_chunks, (v_lane - 1) // SEQ)
        if v_chunks < 1:
            raise SystemExit(f"val corpus {val_path}: too short for {B} lanes of {SEQ + 1} ids")
        # each val lane starts on a BOS (zero carry at a doc start, as in training; a zero carry
        # in the middle of a doc is a state training never produces and inflates the loss)
        _vb = np.flatnonzero(np.asarray(val_arr[:n_val]) == 257)
        v_starts = np.asarray([_vb[min(len(_vb) - 1, np.searchsorted(_vb, b_ * v_lane))] for b_ in range(B)], dtype=np.int64)
        v_chunks = min(v_chunks, int((n_val - 1 - v_starts.max()) // SEQ))
        log(f"v4: val {val_path} total={n_val:,} lanes={B}x{v_lane:,} chunks/lane={v_chunks} "
            f"-> {B * SEQ * v_chunks:,} ids per eval, every {val_every} steps")

    eng = eng.to(device)
    eng.reset_thought(B)
    if use_compile:
        # Compile the pure core of each block, not the engine. The engine
        # compile was bypassed: every pass called payload(), which returned
        # eng._orig_mod. The pure core has fixed shapes, so Inductor can fuse
        # gelu, bias, scales, Kuramoto and LayerNorm. Works with BLOCK_CKPT
        # because the checkpoint calls this same method.
        try:
            for blk in eng.blocks:
                blk._tick_chunk_core_pure = torch.compile(
                    blk._tick_chunk_core_pure, dynamic=False)
            log("torch.compile ON (_tick_chunk_core_pure per block)")
        except Exception as exc:  # pragma: no cover
            log(f"compile skip: {exc}")
    payload = lambda: eng  # noqa: E731

    if opt_name == "adamw":
        opt = torch.optim.AdamW(eng.parameters(), lr=lr, weight_decay=0.01)
    else:
        opt = torch.optim.SGD(eng.parameters(), lr=lr, momentum=0.9)

    # --- pod patch: linear LR warmup + optimizer-state save/resume ---
    warmup_steps = int(os.environ.get("WARMUP_STEPS", "0"))
    opt_path = ckpt_out.with_suffix(".opt")
    opt_in = os.environ.get("OPT_IN") or (str(Path(ckpt_in).with_suffix(".opt")) if ckpt_in else None)
    if ckpt_in and opt_in and os.path.exists(opt_in):
        opt.load_state_dict(torch.load(opt_in, map_location="cpu", weights_only=False))
        log(f"optimizer state loaded from {opt_in}")
    elif ckpt_in:
        log("WARNING: no optimizer state found, optimizer restarts from zero moments")

    _nf = {"count": 0}

    def _log_nonfinite() -> None:
        _nf["count"] += 1
        c = _nf["count"]
        if c <= 20 or c % 1000 == 0:
            bad = [nm for nm, p in eng.named_parameters()
                   if p.grad is not None and not bool(torch.isfinite(p.grad).all())][:5]
            log(f"non-finite grad norm at step {n}: optimizer step skipped "
                f"(skip #{c}) first bad params: {bad}")

    lr_sched = os.environ.get("LR_SCHED", "const")
    lr_min = float(os.environ.get("LR_MIN", "3e-5"))
    grad_clip = float(os.environ.get("GRAD_CLIP", "1.0"))
    log(f"lr schedule={lr_sched} lr_max={lr} lr_min={lr_min} over {max_steps} steps "
        f"warmup={warmup_steps} grad_clip={grad_clip}")
    _lr_now = {"lr": lr}

    def _apply_lr() -> None:
        # Overrides whatever lr a loaded optimizer state carried, every step.
        f = min(1.0, (n + 1) / warmup_steps) if warmup_steps > 0 else 1.0
        cur = lr
        _off = int(os.environ.get("LR_STEP_OFFSET", "0"))
        _tot = int(os.environ.get("LR_TOTAL_STEPS", "0")) or max_steps
        if lr_sched == "cosine" and _tot > 0:
            t = min(1.0, (n + _off) / _tot)
            cur = lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * t))
        cur *= f
        _lr_now["lr"] = cur
        for g in opt.param_groups:
            g["lr"] = cur

    # --- pod patch: spike guard, rollback request, rotating/good checkpoints ---
    gn_skip_mult = float(os.environ.get("GN_SKIP_MULT", "10"))
    gn_skip_floor = float(os.environ.get("GN_SKIP_FLOOR", "300"))
    gn_grace = int(os.environ.get("GN_GRACE", "200"))          # accepted steps before the EMA is trusted
    gn_grace_abs = float(os.environ.get("GN_GRACE_ABS", "3000"))  # threshold during the grace period
    max_consec_skips = int(os.environ.get("MAX_CONSEC_SKIPS", "50"))
    bad_ema = float(os.environ.get("ROLLBACK_EMA_TF", "4.0"))
    bad_ema_steps = int(os.environ.get("ROLLBACK_EMA_STEPS", "1000"))
    ckpt_keep = int(os.environ.get("CKPT_KEEP", "2"))
    good_ema_max = float(os.environ.get("GOOD_EMA_MAX", "3.0"))
    good_quiet = int(os.environ.get("GOOD_QUIET_STEPS", "2000"))
    min_free = float(os.environ.get("MIN_FREE_GB", "17")) * 1e9
    _gs = {"tf": {"ema": 0.0, "cnt": 0}, "ss": {"ema": 0.0, "cnt": 0},
           "skips": 0, "consec": 0, "skip_now": False, "last_bad": -10**9, "bad_ema_run": 0}

    def _guard(gn_t, kind: str) -> bool:
        """True = take the optimizer step. False = skip it (non-finite or spike)."""
        g = float(gn_t)
        if not math.isfinite(g):
            _log_nonfinite()
            _gs["skip_now"] = True; _gs["last_bad"] = n
            return False
        st = _gs[kind]
        thr = max(gn_skip_floor, gn_skip_mult * st["ema"]) if st["cnt"] >= gn_grace else gn_grace_abs
        if g > thr:
            _gs["skips"] += 1; _gs["skip_now"] = True; _gs["last_bad"] = n
            if _gs["skips"] <= 50 or _gs["skips"] % 100 == 0:
                log(f"GN SPIKE {kind} gn={g:.1f} > thr={thr:.1f} (ema={st['ema']:.1f}) at step {n}: "
                    f"step skipped (skip #{_gs['skips']}, consecutive steps with a skip before this one: {_gs['consec']})")
            return False
        st["ema"] = g if st["cnt"] == 0 else 0.99 * st["ema"] + 0.01 * g
        st["cnt"] += 1
        return True

    def _request_rollback(reason: str, off_now: int) -> None:
        info = {"reason": reason, "step_in_run": n, "start_token_next": off_now,
                "ids": off_now * B, "lr_at_stop": _lr_now["lr"],
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (out_dir / "ROLLBACK.json").write_text(json.dumps(info, indent=1))
        log(f"ROLLBACK REQUESTED: {reason}; exiting with code 42 (no checkpoint written) {info}")
        sys.exit(42)

    def _rotate_and_mark(off_now: int) -> None:
        """Hard-link the fresh files into step_<n>/, keep the last ckpt_keep step dirs
        (only the newest keeps its .opt), and refresh good/ if the window was healthy."""
        sd_ = out_dir / f"step_{n:08d}"
        if sd_.exists():
            shutil.rmtree(sd_)
        sd_.mkdir()
        for src in (ckpt_out, opt_path, manifest_out):
            if src.exists():
                os.link(src, sd_ / src.name)
        steps = sorted(d for d in out_dir.glob("step_*") if d.is_dir())
        for d in steps[:-ckpt_keep]:
            shutil.rmtree(d, ignore_errors=True)
        for d in steps[-ckpt_keep:-1]:
            (d / opt_path.name).unlink(missing_ok=True)
        quiet = n - _gs["last_bad"]
        ok = (ema_tf is not None and ema_tf < good_ema_max and quiet >= good_quiet
              and n >= good_quiet)
        log(f"ckpt step_{n:08d}: ema_tf={ema_tf if ema_tf is None else round(ema_tf, 3)} "
            f"steps_since_bad={quiet} -> {'GOOD' if ok else 'not good'}")
        if ok:
            gtmp = out_dir / "good.tmp"
            shutil.rmtree(gtmp, ignore_errors=True)
            gtmp.mkdir()
            for f in sd_.iterdir():
                os.link(f, gtmp / f.name)
            (gtmp / "GOOD.json").write_text(json.dumps({
                "step_in_run": n, "ids_processed": off_now * B, "start_token_next": off_now,
                "ema_tf": ema_tf, "gn_ema_tf": _gs["tf"]["ema"], "skips_total": _gs["skips"],
                "lr": _lr_now["lr"], "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                indent=1))
            shutil.rmtree(out_dir / "good", ignore_errors=True)
            os.replace(gtmp, out_dir / "good")

    def _ensure_space() -> None:
        """Before writing a new checkpoint (~15.5 GB + tmp), drop older step dirs if needed."""
        if shutil.disk_usage(out_dir).free >= min_free:
            return
        for d in sorted(d for d in out_dir.glob("step_*") if d.is_dir())[:-1]:
            shutil.rmtree(d, ignore_errors=True)
            log(f"low disk: removed {d.name}")
            if shutil.disk_usage(out_dir).free >= min_free:
                return

    def fetch(off: int) -> torch.Tensor:
        """(B, SEQ+1): row b = lane b, ids [off, off+SEQ]."""
        rows = [np.asarray(shard[s + off:s + off + SEQ + 1], dtype=np.int64) for s in lane_starts]
        return torch.from_numpy(np.stack(rows)).to(device, non_blocking=True)

    def reset_rows(mask: torch.Tensor) -> None:
        """Zero the carry of the rows in mask only. Other rows keep their document."""
        e = payload()
        keep = (~mask).to(e.thought_state.dtype)
        e.thought_state = e.thought_state * keep.view(-1, 1, 1)
        for blk in e.blocks:
            blk.attn_S = blk.attn_S * keep.view(-1, 1, 1, 1).to(blk.attn_S.dtype)
            blk.attn_z = blk.attn_z * keep.view(-1, 1, 1).to(blk.attn_z.dtype)
            blk.kuramoto_phases = blk.kuramoto_phases * keep.view(-1, 1, 1).to(blk.kuramoto_phases.dtype)

    @torch.no_grad()
    def bound_carry() -> None:
        """v2: bound attn_S / attn_z between chunks (detached, like the carry itself)."""
        if carry_decay == 1.0 and carry_clamp <= 0:
            return
        for blk in payload().blocks:
            S = blk.attn_S.detach()
            z = blk.attn_z.detach()
            if carry_decay != 1.0:
                S = S * carry_decay
                z = z * carry_decay
            if carry_clamp > 0:
                nS = S.float().flatten(2).norm(dim=-1).clamp_min(1e-12)       # (B, nH)
                S = S * (carry_clamp / nS).clamp(max=1.0).to(S.dtype)[..., None, None]
                nz = z.float().norm(dim=-1).clamp_min(1e-12)                  # (B, nH)
                z = z * (carry_clamp / nz).clamp(max=1.0).to(z.dtype)[..., None]
            blk.attn_S = S
            blk.attn_z = z

    ckpt_out.parent.mkdir(parents=True, exist_ok=True)

    def save(off: int) -> None:
        ids_done = off * B
        _ensure_space()
        if os.environ.get("SAVE_OPT", "1") == "1":
            otmp = opt_path.with_suffix(".opt.tmp")
            torch.save(opt.state_dict(), otmp)
            os.replace(otmp, opt_path)
        tmp = ckpt_out.with_suffix(ckpt_out.suffix + ".tmp")
        torch.save({
            "model_state": payload().state_dict(),
            "config": {
                **cfg, "vocab_size": VOCAB_SIZE, "atom": True, "scale": scale,
                "gpu": gpu, "n_gpu": n_gpu, "corpus": str(corpus_path),
                "corpus_slice": [lo, hi], "ids_processed": ids_done,
                "layout": "lanes", "lane_len": lane_len,
                "start_token_next": off, "batch": B, "seq": SEQ, "lr": lr, "opt": opt_name,
                "gate_temp": float(os.environ.get("GATE_TEMP", "2.5")),
                "kuramoto_fix": birth["kuramoto_fix"], "parent": birth["parent"],
                "fresh_start_token": 0 if birth["fresh"] else None,
                "trainer": "fast4gpu_atom", "trainer_file": "fast4gpu_atom_v4", "ss_tail_ids": ss_tail,
                "corpus_switch": corpus_switch, "ids_base": ids_base, "ids_total": ids_base + ids_done,
                "val_corpus": val_path,
            },
        }, tmp)
        os.replace(tmp, ckpt_out)
        mtmp = manifest_out.with_suffix(".json.tmp")
        mtmp.write_text(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trainer": "fast4gpu_atom", "gpu": gpu, "n_gpu": n_gpu, "scale": scale,
            "vocab_size": VOCAB_SIZE, "ids_processed": ids_done,
            "layout": "lanes", "batch": B, "lane_len": lane_len,
            "start_token_next": off, "ids_base": ids_base, "ids_total": ids_base + ids_done,
            "note": "start_token_next is the offset inside each lane; resume with the same BATCH",
            "corpus": str(corpus_path), "corpus_slice": [lo, hi], "ckpt": ckpt_out.name,
        }, indent=1))
        os.replace(mtmp, manifest_out)
        log(f"saved -> {ckpt_out} @ lane offset {off:,} ({ids_done:,} ids)")
        _rotate_and_mark(off)

    def probe(ids_done: int) -> None:
        snap = snapshot_carry(payload())
        p = unique40_probe(payload(), max_new=40, mode="prefix")
        payload().reset_thought(B)
        restore_carry(payload(), snap)
        mean_u = sum(r["unique"] for r in p["rows"]) / max(1, len(p["rows"]))
        log(f"UNIQUE@40 PREFIX mean_unique={mean_u:.1f} @ {ids_done:,} ids | routing {routing_snapshot(payload())}")
        for r in p["rows"]:
            log(f"  {r['prompt']!r}: unique={r['unique']}/{r['n']} -> {r['text']!r}")

    def val_eval(ids_done: int, step: int) -> None:
        """Held-out teacher-forced CE. Training carry is saved and restored around it."""
        if val_arr is None:
            return
        t0 = time.time()
        e = payload()
        snap = snapshot_carry(e)
        e.reset_thought(B)
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for c in range(v_chunks):
                rows = [np.asarray(val_arr[s + c * SEQ:s + c * SEQ + SEQ + 1], dtype=np.int64) for s in v_starts]
                blk_ = torch.from_numpy(np.stack(rows)).to(device, non_blocking=True)
                ch, tg = blk_[:, :SEQ], blk_[:, 1:]
                bos = (ch == 257).any(dim=1)
                if bool(bos.any()):
                    reset_rows(bos)
                with autocast():
                    ce_v, _lb = e.tick_chunk_train_ce(ch, tg, ce_chunk=ce_chunk, block_ckpt=False)
                tot += float(ce_v); cnt += 1
        e.reset_thought(B)
        restore_carry(e, snap)
        ce_m = tot / max(cnt, 1)
        n_ids = cnt * B * SEQ
        log(f"VAL ce={ce_m:.4f} nats/id ({ce_m / math.log(2):.4f} bits/id) on {n_ids:,} held-out ids "
            f"@ {ids_done:,} ids (total {ids_base + ids_done:,}) step {step} [{time.time() - t0:.1f}s]")
        with open(out_dir / "val_log.jsonl", "a") as vf:
            vf.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "step": step,
                                 "ids": ids_done, "ids_total": ids_base + ids_done, "val_ce": ce_m,
                                 "val_ids": n_ids, "secs": round(time.time() - t0, 1)}) + "\n")

    ema_tf = ema_ss = None
    t_tf = t_ss = t_probe = 0.0
    ids_tf = ids_ss = 0
    n = 0
    off = start
    t_wall = time.time()
    last_off = lane_len - SEQ - 1
    out_dir.mkdir(parents=True, exist_ok=True)
    if val_arr is not None and os.environ.get("VAL_AT_START", "1") == "1":
        val_eval(off * B, 0)
        t_wall = time.time()

    while epoch_wrap or off <= last_off:
        if epoch_wrap:
            if off > start and off % period == 0:
                payload().reset_thought(B)   # new pass: lane b no longer continues its text
                log(f"EPOCH WRAP at start_token {off:,} ({off * B:,} ids) -> epoch {off // period}")
            block = fetch(off % period)
        else:
            block = fetch(off)
        chunk = block[:, :SEQ]
        target = block[:, 1:]
        off += SEQ
        # Reset at each BOS so the model sees an empty thought at document
        # boundaries. Continuity stays inside a document. Without this, a long
        # run almost never sees an empty state, and generation (which starts
        # from zero) is a regime the model never trained on. Measured: 81%
        # accuracy with the training carry, 3% after reset, on a memorized phrase.
        # Per row: a BOS in lane b resets lane b only, not the other documents.
        bos_rows = (chunk == 257).any(dim=1)
        if bool(bos_rows.any()):
            reset_rows(bos_rows)
        pos = off * B
        # Snapshot only when the SS pass will run. Cloning the carry every
        # step costs ~105 MB per row (16 blocks of 1280x1280), which is
        # ~6.7 GB at B=64, for a restore that never happens when SS is off.
        ss_prob = ss_prob_at(pos - ss_origin, ss_p0, ss_p1, ss_ramp)
        do_ss = ss_rate > 0 and should_ss(ss_rate)
        carry = snapshot_carry(payload()) if do_ss else None

        sync(); t0 = time.time()
        with autocast():
            loss, ex = v4_forward_losses(
                payload(), chunk, target,
                lb_coef=lb_coef, repeat_coef=repeat_coef,
                ce_chunk=ce_chunk, block_ckpt=block_ckpt,
            )
        loss.backward()
        _gn = torch.nn.utils.clip_grad_norm_(eng.parameters(), grad_clip)
        gn_tf = _gn
        _apply_lr()
        if _guard(_gn, "tf"):
            opt.step()
        opt.zero_grad(set_to_none=True)
        sync(); t_tf += time.time() - t0
        ids_tf += step_ids
        tf_v = float(ex["ce"].detach())
        ema_tf = tf_v if ema_tf is None else ema_beta * ema_tf + (1 - ema_beta) * tf_v

        ss_v = ss_tail_v = None
        if do_ss:
            restore_carry(payload(), carry)
            sync(); t0 = time.time()
            with autocast():
                loss2, ce_ss, ce_tail = ss_pass_tail(
                    payload(), chunk, target, ex["h"].detach(),
                    ss_prob=ss_prob, tail=ss_tail, lb_coef=lb_coef, ce_chunk=ce_chunk,
                )
            loss2.backward()
            _gn = torch.nn.utils.clip_grad_norm_(eng.parameters(), grad_clip)
            _apply_lr()
            if _guard(_gn, "ss"):
                opt.step()
            opt.zero_grad(set_to_none=True)
            sync(); t_ss += time.time() - t0
            ids_ss += step_ids
            ss_v = float(ce_ss.detach())
            ss_tail_v = float(ce_tail)
            ema_ss = ss_v if ema_ss is None else ema_beta * ema_ss + (1 - ema_beta) * ss_v

        bound_carry()
        n += 1
        _gs["consec"] = _gs["consec"] + 1 if _gs["skip_now"] else 0
        _gs["skip_now"] = False
        _gs["bad_ema_run"] = _gs["bad_ema_run"] + 1 if (ema_tf is not None and ema_tf > bad_ema) else 0
        if _gs["consec"] > max_consec_skips:
            _request_rollback(f"{_gs['consec']} consecutive skipped steps", off)
        if _gs["bad_ema_run"] >= bad_ema_steps:
            _request_rollback(f"ema_tf > {bad_ema} for {bad_ema_steps} steps", off)
        if n % log_every == 0:
            wall = max(time.time() - t_wall, 1e-9)
            mem = ""
            if device.type == "cuda":
                mem = f" mem_peak={torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
            ss_s = f" ss={ss_v:.3f} ema_ss={ema_ss:.3f} ss_p={ss_prob:.3f} ss_tail={ss_tail_v:.3f}" if ss_v is not None else ""
            log(
                f"{pos:>12,} tf={tf_v:.3f} ema_tf={ema_tf:.3f}{ss_s} "
                f"gn={float(gn_tf):.3f} gn_ema={_gs['tf']['ema']:.1f} skips={_gs['skips']} lr={_lr_now['lr']:.3e} "
                f"rep={float(ex['repeat'].detach()):.3f} lb={float(ex['lb'].detach()):.3f} | "
                f"data {ids_tf / wall:.0f} ids/s wall | "
                f"TF {ids_tf / max(t_tf, 1e-9):.0f} ids/s ({t_tf / wall:.0%}) "
                f"SS {ids_ss / max(t_ss, 1e-9):.0f} ids/s ({t_ss / wall:.0%}) "
                f"probe {t_probe / wall:.0%}{mem}"
            )
        if probe_every > 0 and n % probe_every == 0:
            t0 = time.time(); probe(pos); t_probe += time.time() - t0
        if val_arr is not None and n % val_every == 0:
            t0 = time.time(); val_eval(pos, n); t_probe += time.time() - t0
        if save_every > 0 and n % save_every == (gpu * 500) % save_every:
            save(off)
        if max_steps and n >= max_steps:
            break

    wall = max(time.time() - t_wall, 1e-9)
    save(off)
    summary = {
        "steps": n, "ids_tf": ids_tf, "ids_ss": ids_ss, "wall_s": round(wall, 2),
        "data_ids_per_s_wall": round(ids_tf / wall, 1),
        "tf_ids_per_s": round(ids_tf / max(t_tf, 1e-9), 1),
        "ss_ids_per_s": round(ids_ss / max(t_ss, 1e-9), 1) if ids_ss else None,
        "share": {"tf": round(t_tf / wall, 3), "ss": round(t_ss / wall, 3),
                  "probe": round(t_probe / wall, 3)},
        "ema_tf": ema_tf, "params": n_params, "scale": scale, "batch": B, "seq": SEQ,
        "block_ckpt": block_ckpt, "compile": use_compile, "device": str(device),
    }
    if device.type == "cuda":
        summary["mem_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    log("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
