#!/usr/bin/env python3
"""Fractus-1B boost trainer v2 — optimized, open-heart compatible.

Drop-in evolution of scripts/fast4gpu_boost.py. Training SEMANTICS are
preserved (same loss = CE + LB_COEF*lb, same SS schedule, same SGD recipe,
same checkpoint format and resume offsets) — only the computation changes:

  1. Attention kernel: cumsum (default) or memory-flat 'chunked'
     (FRACTUS_ATTN_IMPL=chunked). Both proven equal to the einsum reference
     by tests/test_attention_equivalence.py.
  2. Memory-flat CE: tick_chunk_train_ce + chunked_cross_entropy
     (CE_CHUNK rows/chunk; 0 = legacy dense logits path). Loss identical to
     dense within fp32 rounding.
  3. Data pipeline: int32 memmap sliced per chunk — NO whole-shard int64
     upcast (saves ~3.4 GB RAM per process at phase-2 shard sizes). Embedding
     accepts int32 indices directly.
  4. Optional gradient accumulation (ACCUM) to decouple effective batch from
     VRAM. Default ACCUM=1 = exactly the legacy per-step update.

Env:
  GPU_ID, BATCH=4, SEQ=128, LR=7e-4, SS_RATE=0.25, SS_PROB=0.2,
  LB_COEF=0.02, GATE_TEMP=2.5, EMA_BETA=0.98
  CKPT_IN, CKPT_OUT, START_TOKEN, SHARD (.npy int32 memmap)
  FRACTUS_ATTN_IMPL=cumsum|chunked    attention kernel
  CE_CHUNK=2048                       rows per CE chunk (0 = dense legacy);
                                      caps transient logits at ~0.41 GB regardless of batch
  ACCUM=1                             optimizer step every N batches
  COMPILE=0                           1 = torch.compile engine (needs free VRAM)

Usage (one process per GPU):
  CUDA_VISIBLE_DEVICES=0 GPU_ID=0 python -u scripts/fast4gpu_boost_v2.py
"""
from __future__ import annotations

import os
import sys
import time
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

os.environ.setdefault("FRACTUS_ATTN_IMPL", os.environ.get("FRACTUS_ATTN_IMPL", "cumsum"))
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.nn.ce import sample_tokens_chunked

GPU = int(os.environ.get("GPU_ID", "0"))
LB_COEF = float(os.environ.get("LB_COEF", "0.02"))
GATE_TEMP = float(os.environ.get("GATE_TEMP", "2.5"))
LR = float(os.environ.get("LR", "7e-4"))
EMA_BETA = float(os.environ.get("EMA_BETA", "0.98"))
SS_PROB = float(os.environ.get("SS_PROB", "0.2"))
SS_RATE = float(os.environ.get("SS_RATE", "0.25"))
B = int(os.environ.get("BATCH", "4"))
SEQ = int(os.environ.get("SEQ", "128"))
# 2048 caps transient logits at ~0.41 GB (2048 x 50257 x fp32) at ANY batch
# size; 16384 would allow a 3.3 GB transient once N = B*SEQ exceeds it.
CE_CHUNK = int(os.environ.get("CE_CHUNK", "2048"))
ACCUM = max(1, int(os.environ.get("ACCUM", "1")))
# 1 = per-block activation checkpointing inside tick_chunk_train_ce (exact
# math, recompute in backward) — fits the full 1B config in ~16 GB VRAM.
BLOCK_CKPT = os.environ.get("BLOCK_CKPT", "0") == "1"
USE_COMPILE = os.environ.get("COMPILE", "0") == "1"
ATTN_IMPL = os.environ.get("FRACTUS_ATTN_IMPL", "cumsum")

TARGET = dict(
    d_model=1280,
    n_heads=20,
    d_head=64,
    n_levels=2,
    n_oscillators=16,
    coupling_rank=8,
    n_experts=128,
    top_k=2,
    expert_d_ff=2048,
    siren_rank=64,
    n_layers=16,
)

torch.manual_seed(42 + GPU)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0")
autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

default_merged = ROOT / "checkpoints" / "FRACTUS_1B_STAGE2_MERGED.pt"
default_gpu = ROOT / "checkpoints" / f"fractus_1b_gpu{GPU}.pt"
CKPT_IN = Path(os.environ.get("CKPT_IN", str(default_gpu if default_gpu.exists() else default_merged)))
CKPT_OUT = Path(os.environ.get("CKPT_OUT", str(default_gpu)))
SHARD = Path(os.environ.get("SHARD", str(ROOT / "data" / f"shard_gpu{GPU}.npy")))

print(f"GPU {GPU}: BOOSTv2 B={B} SEQ={SEQ} LR={LR} SS_RATE={SS_RATE} "
      f"attn={ATTN_IMPL} ce_chunk={CE_CHUNK} accum={ACCUM}", flush=True)
print(f"GPU {GPU}: load {CKPT_IN}", flush=True)

ck = torch.load(CKPT_IN, map_location="cpu", weights_only=False)
sd = ck.get("model_state", ck)
clean = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}

eng = ContinuousThoughtEngine(vocab_size=266, **TARGET)
own = eng.state_dict()
loaded = 0
for k, v in clean.items():
    if k in own and own[k].shape == v.shape:
        own[k] = v
        loaded += 1
    elif (
        k in own
        and v.dim() >= 1
        and own[k].dim() >= 1
        and v.shape[0] > own[k].shape[0]
        and v.shape[1:] == own[k].shape[1:]
    ):
        own[k] = v[: own[k].shape[0]].contiguous()
        loaded += 1
eng.load_state_dict(own, strict=False)
print(f"GPU {GPU}: loaded_tensors={loaded}", flush=True)

with torch.no_grad():
    for blk in eng.blocks:
        if hasattr(blk, "moe") and hasattr(blk.moe, "temperature"):
            blk.moe.temperature = GATE_TEMP

eng = eng.to(device)
eng.reset_thought(B)

if USE_COMPILE:
    try:
        eng = torch.compile(eng)
        print(f"GPU {GPU}: torch.compile ON", flush=True)
    except Exception as e:
        print(f"GPU {GPU}: compile skip: {e}", flush=True)
else:
    print(f"GPU {GPU}: compile disabled (set COMPILE=1 once VRAM allows)", flush=True)

opt = torch.optim.SGD(eng.parameters(), lr=LR, momentum=0.9)


# --- data pipeline: int32 memmap, zero whole-shard copies -------------------
if not SHARD.exists():
    alt = Path(str(SHARD).replace(".pt", ".npy")) if str(SHARD).endswith(".pt") else None
    if alt is None or not alt.exists():
        raise FileNotFoundError(f"Shard not found: {SHARD}")
    SHARD = alt

import numpy as np

if str(SHARD).endswith(".npy"):
    shard_mm = np.load(str(SHARD), mmap_mode="r")   # int32 on disk
    shard_len = int(shard_mm.shape[0])
    print(f"GPU {GPU}: memmap shard {SHARD} len={shard_len:,} dtype={shard_mm.dtype}",
          flush=True)
else:
    raise FileNotFoundError(
        f"v2 trainer expects .npy int32 shards, got {SHARD}. "
        f"For legacy .pt shards use fast4gpu_boost.py or re-shard via shard_corpus.py.")

step_tokens = B * SEQ


def fetch(start: int, count: int) -> torch.Tensor:
    """Slice [start, start+count) from the int32 memmap -> CUDA long tensor.

    The numpy slice is a contiguous view into the page cache; the copy is one
    small per-chunk buffer, never the whole shard.
    """
    view = np.asarray(shard_mm[start : start + count])       # zero-copy view
    return torch.from_numpy(view).to(torch.int64, non_blocking=True).to(device)


start_token = int(os.environ.get("START_TOKEN", "0"))
start_token = (start_token // step_tokens) * step_tokens
print(f"GPU {GPU}: RESUME start_token={start_token} step={step_tokens} shard_len={shard_len:,}",
      flush=True)

t0 = time.time()
ema_tf = None
ema_ss = None
n = 0
tok_sess = 0
pending_backward = False

CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)


def save_ckpt(tokens_done: int):
    payload_eng = eng._orig_mod if hasattr(eng, "_orig_mod") else eng
    # atomic write: a concurrent HF sync must never read a half-written file
    tmp = CKPT_OUT.with_suffix(CKPT_OUT.suffix + ".tmp")
    torch.save(
        {
            "model_state": payload_eng.state_dict(),
            "config": {
                **TARGET,
                "gpu": GPU,
                "boost": True,
                "boost_v2": True,
                "batch": B,
                "lr": LR,
                "ss_rate": SS_RATE,
                "tokens_processed": tokens_done,
            },
        },
        tmp,
    )
    os.replace(tmp, CKPT_OUT)
    print(f"GPU {GPU}: saved [boostv2] -> {CKPT_OUT}", flush=True)


for start in range(start_token, shard_len - step_tokens - SEQ - 1, step_tokens):
    block = fetch(start, step_tokens + 1)
    chunk = block[:step_tokens].view(B, SEQ).long()
    target = block[1:].view(B, SEQ)

    # ---- teacher-forced pass ---------------------------------------------
    with autocast():
        if CE_CHUNK > 0:
            ce_tf, lb, h = eng.tick_chunk_train_ce(chunk, target,
                                                   ce_chunk=CE_CHUNK,
                                                   return_hidden=True,
                                                   block_ckpt=BLOCK_CKPT)
        else:
            out = eng.tick_chunk_train(chunk)
            logits, lb = out if isinstance(out, tuple) else (out, eng.last_lb_loss)
            ce_tf = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                    target.reshape(-1))
    loss = ce_tf + LB_COEF * lb

    # ---- scheduled sampling pass (same schedule & semantics as v1) --------
    ss_fired = False
    ce_ss_v = None
    if random.random() < SS_RATE:
        with torch.no_grad():
            if CE_CHUNK > 0:
                samp = sample_tokens_chunked(
                    h.reshape(-1, h.shape[-1]).detach(),
                    (eng._orig_mod if hasattr(eng, "_orig_mod") else eng).output_head.weight,
                    temperature=0.9, ce_chunk=CE_CHUNK,
                ).view(B, SEQ)
            else:
                samp = torch.multinomial(
                    torch.softmax(logits.detach().float().reshape(-1, logits.size(-1)) / 0.9, dim=-1),
                    1,
                ).view(B, SEQ)
        mixed = chunk.clone()
        use_ss = torch.rand(B, SEQ, device=device) < SS_PROB
        use_ss[:, 0] = False
        prev = torch.cat([chunk[:, :1], samp[:, :-1]], dim=1)
        mixed = torch.where(use_ss, prev, mixed)
        ss_fired = True

    if ACCUM == 1:
        # EXACT legacy v1 semantics: TF step, then (if fired) a separate SS step.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if ss_fired:
            with autocast():
                if CE_CHUNK > 0:
                    ce_ss, lb2 = eng.tick_chunk_train_ce(mixed, target, ce_chunk=CE_CHUNK,
                                                         block_ckpt=BLOCK_CKPT)
                else:
                    out2 = eng.tick_chunk_train(mixed)
                    logits2, lb2 = out2 if isinstance(out2, tuple) else (out2, eng.last_lb_loss)
                    ce_ss = F.cross_entropy(logits2.reshape(-1, logits2.size(-1)),
                                            target.reshape(-1))
                loss2 = 0.5 * ce_ss + LB_COEF * lb2
            loss2.backward()
            torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            ce_ss_v = float(ce_ss.item())
            ema_ss = ce_ss_v if ema_ss is None else EMA_BETA * ema_ss + (1 - EMA_BETA) * ce_ss_v
    else:
        # ACCUM>1 (documented deviation): grads from TF (and SS, if fired)
        # accumulate; one clip+step every ACCUM batches.
        (loss / ACCUM).backward()
        if ss_fired:
            with autocast():
                if CE_CHUNK > 0:
                    ce_ss, lb2 = eng.tick_chunk_train_ce(mixed, target, ce_chunk=CE_CHUNK,
                                                         block_ckpt=BLOCK_CKPT)
                else:
                    out2 = eng.tick_chunk_train(mixed)
                    logits2, lb2 = out2 if isinstance(out2, tuple) else (out2, eng.last_lb_loss)
                    ce_ss = F.cross_entropy(logits2.reshape(-1, logits2.size(-1)),
                                            target.reshape(-1))
                loss2 = 0.5 * ce_ss + LB_COEF * lb2
            (loss2 / ACCUM).backward()
            ce_ss_v = float(ce_ss.item())
            ema_ss = ce_ss_v if ema_ss is None else EMA_BETA * ema_ss + (1 - EMA_BETA) * ce_ss_v
        pending_backward = True

    tf_v = float(ce_tf.detach().item())
    lb_v = float(lb.detach().item()) if torch.is_tensor(lb) else float(lb)
    ema_tf = tf_v if ema_tf is None else EMA_BETA * ema_tf + (1 - EMA_BETA) * tf_v

    n += 1
    tok_sess += step_tokens

    if ACCUM > 1 and n % ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        pending_backward = False

    if n % 40 == 0:
        tps = tok_sess / max(time.time() - t0, 1e-6)
        extra = f" ss={ce_ss_v:.3f} ema_ss={ema_ss:.3f}" if ce_ss_v is not None else ""
        try:
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            mem_s = f"mem={mem_gb:.1f}GB"
        except Exception:
            mem_s = ""
        print(
            f"GPU {GPU}: {start + step_tokens:>12,} tf={tf_v:.3f} ema_tf={ema_tf:.3f}{extra} "
            f"lb={lb_v:.3f} {tps:.0f} tok/s {mem_s} [boostv2]",
            flush=True,
        )

    if n % 800 == 0:
        save_ckpt(start + step_tokens)

if pending_backward:
    torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)

save_ckpt(start_token + n * step_tokens)
print(f"GPU {GPU}: DONE", flush=True)
