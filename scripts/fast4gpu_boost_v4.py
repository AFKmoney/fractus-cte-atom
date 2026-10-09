#!/usr/bin/env python3
"""Fractus-1B boost trainer v4 — AR gap.

v3 kernels/data/ckpt semantics, PLUS the losses that actually target free-run:

  - P0 routing on (phase carry + per-token MoE + switch LB)
  - SS_RATE default 1.0 (every step), SS_PROB ramps 0.2 → 0.5
  - anti-repeat: λ · mean log p(class = last input token)
  - unique@40 greedy probe (no ban) on a timer — THIS is the go/no-go, not ema_tf

Env (v3 plus):
  SS_RATE=1.0  SS_PROB_START=0.2  SS_PROB_END=0.5  SS_RAMP_TOKENS=50000000
  REPEAT_COEF=0.1  PROBE_EVERY=200
  P0=1  (set 0 to ablate routing surgery)

One GPU:
  CUDA_VISIBLE_DEVICES=$i GPU_ID=$i START_TOKEN=<manifest> \\
    BATCH=8 CE_CHUNK=2048 FRACTUS_ATTN_IMPL=chunked BLOCK_CKPT=1 \\
    python -u scripts/fast4gpu_boost_v4.py

Smoke ONE gpu 1–2h and read unique@40 before touching the other seven.
"""
from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

os.environ.setdefault("FRACTUS_ATTN_IMPL", os.environ.get("FRACTUS_ATTN_IMPL", "cumsum"))
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.generate_aligned import unique40_probe
from fractus.train.ar_loss import ss_prob_at
from fractus.train.v4_step import v4_forward_losses, v4_ss_pass, should_ss, snapshot_carry, restore_carry

GPU = int(os.environ.get("GPU_ID", "0"))
LB_COEF = float(os.environ.get("LB_COEF", "0.02"))
GATE_TEMP = float(os.environ.get("GATE_TEMP", "2.5"))
LR = float(os.environ.get("LR", "7e-4"))
EMA_BETA = float(os.environ.get("EMA_BETA", "0.98"))
SS_RATE = float(os.environ.get("SS_RATE", "1.0"))
SS_PROB_START = float(os.environ.get("SS_PROB_START", "0.2"))
SS_PROB_END = float(os.environ.get("SS_PROB_END", "0.5"))
SS_RAMP_TOKENS = int(os.environ.get("SS_RAMP_TOKENS", "50000000"))
REPEAT_COEF = float(os.environ.get("REPEAT_COEF", "0.1"))
PROBE_EVERY = int(os.environ.get("PROBE_EVERY", "200"))
P0 = os.environ.get("P0", "1") == "1"
B = int(os.environ.get("BATCH", "4"))
SEQ = int(os.environ.get("SEQ", "128"))
CE_CHUNK = int(os.environ.get("CE_CHUNK", "2048"))
ACCUM = max(1, int(os.environ.get("ACCUM", "1")))
BLOCK_CKPT = os.environ.get("BLOCK_CKPT", "0") == "1"
USE_COMPILE = os.environ.get("COMPILE", "0") == "1"
ATTN_IMPL = os.environ.get("FRACTUS_ATTN_IMPL", "cumsum")

TARGET = dict(
    d_model=1280, n_heads=20, d_head=64, n_levels=2,
    n_oscillators=16, coupling_rank=8, n_experts=128, top_k=2,
    expert_d_ff=2048, siren_rank=64, n_layers=16,
)

torch.manual_seed(42 + GPU)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
else:
    device = torch.device("cpu")
    from contextlib import nullcontext
    autocast = nullcontext

default_merged = ROOT / "checkpoints" / "FRACTUS_1B_STAGE2_MERGED.pt"
default_gpu = ROOT / "checkpoints" / f"fractus_1b_gpu{GPU}.pt"
CKPT_IN = Path(os.environ.get("CKPT_IN", str(default_gpu if default_gpu.exists() else default_merged)))
CKPT_OUT = Path(os.environ.get("CKPT_OUT", str(default_gpu)))
SHARD = Path(os.environ.get("SHARD", str(ROOT / "data" / f"shard_gpu{GPU}.npy")))
MANIFEST_OUT = Path(os.environ.get(
    "MANIFEST_OUT", str(CKPT_OUT.parent / f"RESUME_MANIFEST_gpu{GPU}.json")))

print(
    f"GPU {GPU}: BOOSTv4 B={B} SEQ={SEQ} LR={LR} SS_RATE={SS_RATE} "
    f"ss_prob={SS_PROB_START}->{SS_PROB_END} repeat={REPEAT_COEF} P0={P0} "
    f"attn={ATTN_IMPL} ce_chunk={CE_CHUNK} block_ckpt={BLOCK_CKPT}",
    flush=True,
)
print(f"GPU {GPU}: load {CKPT_IN}", flush=True)

ck = torch.load(CKPT_IN, map_location="cpu", weights_only=False)
sd = ck.get("model_state", ck)
clean = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}

eng = ContinuousThoughtEngine(vocab_size=50257, **TARGET)
own = eng.state_dict()
loaded = 0
for k, v in clean.items():
    if k in own and own[k].shape == v.shape:
        own[k] = v
        loaded += 1
    elif (
        k in own and v.dim() >= 1 and own[k].dim() >= 1
        and v.shape[0] > own[k].shape[0] and v.shape[1:] == own[k].shape[1:]
    ):
        own[k] = v[: own[k].shape[0]].contiguous()
        loaded += 1
eng.load_state_dict(own, strict=False)
print(f"GPU {GPU}: loaded_tensors={loaded}", flush=True)

eng.set_p0_routing(
    P0, lb_mode=("switch_topk" if P0 else "soft_var"),
)
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

payload = lambda: (eng._orig_mod if hasattr(eng, "_orig_mod") else eng)

opt = torch.optim.SGD(eng.parameters(), lr=LR, momentum=0.9)

if not SHARD.exists():
    raise FileNotFoundError(f"Shard not found: {SHARD}")

import numpy as np
if not str(SHARD).endswith(".npy"):
    raise FileNotFoundError(f"v4 expects .npy int32 shards, got {SHARD}")
shard_mm = np.load(str(SHARD), mmap_mode="r")
shard_len = int(shard_mm.shape[0])
print(f"GPU {GPU}: memmap shard {SHARD} len={shard_len:,} dtype={shard_mm.dtype}", flush=True)

step_tokens = B * SEQ


def fetch(start: int, count: int) -> torch.Tensor:
    view = np.asarray(shard_mm[start : start + count])
    return torch.from_numpy(view).to(torch.int64, non_blocking=True).to(device)


start_token = int(os.environ.get("START_TOKEN", "0"))
start_token = (start_token // step_tokens) * step_tokens
print(f"GPU {GPU}: RESUME start_token={start_token} step={step_tokens} shard_len={shard_len:,}",
      flush=True)

t0 = time.time()
ema_tf = ema_ss = ema_rep = None
n = 0
tok_sess = 0
pending_backward = False
CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)


def save_ckpt(tokens_done: int):
    tmp = CKPT_OUT.with_suffix(CKPT_OUT.suffix + ".tmp")
    torch.save(
        {
            "model_state": payload().state_dict(),
            "config": {
                **TARGET, "gpu": GPU, "boost_v4": True, "batch": B, "lr": LR,
                "ss_rate": SS_RATE, "repeat_coef": REPEAT_COEF, "p0": P0,
                "tokens_processed": tokens_done,
            },
        },
        tmp,
    )
    os.replace(tmp, CKPT_OUT)
    mtmp = MANIFEST_OUT.with_suffix(MANIFEST_OUT.suffix + ".tmp")
    mtmp.write_text(json.dumps({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": GPU, "trainer": "fast4gpu_boost_v4",
        "attn_impl": ATTN_IMPL, "p0": P0, "batch": B,
        "tokens_processed": tokens_done, "start_token_next": tokens_done,
        "shard": str(SHARD), "shard_len": shard_len, "ckpt": CKPT_OUT.name,
    }, indent=1))
    os.replace(mtmp, MANIFEST_OUT)
    print(f"GPU {GPU}: saved [boostv4] -> {CKPT_OUT} @ {tokens_done:,} tok", flush=True)


def run_probe(tokens_done: int):
    eng_p = payload()
    thought = eng_p.thought_state
    carries = [(b.attn_S.clone(), b.attn_z.clone(), b.kuramoto_phases.clone())
               for b in eng_p.blocks]
    B_live = thought.shape[0]
    probe = unique40_probe(eng_p, max_new=40, mode="prefix")
    probe_c = unique40_probe(eng_p, max_new=40, mode="carry")
    # restore live train state
    eng_p.thought_state = thought
    for b, (S, z, ph) in zip(eng_p.blocks, carries):
        b.attn_S, b.attn_z, b.kuramoto_phases = S, z, ph
    eng_p.reset_thought(B_live)  # batch may have been set to 1
    # reset_thought zeros — put live carries back
    eng_p.thought_state = thought
    for b, (S, z, ph) in zip(eng_p.blocks, carries):
        b.attn_S, b.attn_z, b.kuramoto_phases = S, z, ph
    gate = "GO" if probe["gate_go"] else "NO-GO"
    print(
        f"GPU {GPU}: UNIQUE@40 PREFIX {gate} mean_u={probe['mean_unique']:.1f} "
        f"echo={probe['mean_echo_frac']:.2f} | "
        f"CARRY u={probe_c['mean_unique']:.1f} echo={probe_c['mean_echo_frac']:.2f} "
        f"@ {tokens_done:,} tok",
        flush=True,
    )
    for r in probe["rows"]:
        print(
            f"  {r['prompt']!r}: unique={r['unique']} echo={r['echo_frac']:.2f} head={r['head']}",
            flush=True,
        )
    rs = eng_p.routing_stats()
    print(
        f"  routing alive={rs.get('alive_experts')} "
        f"H={rs.get('dispatch_entropy', float('nan')):.3f} "
        f"max_frac={rs.get('max_frac', float('nan')):.3f}",
        flush=True,
    )
    return probe


for start in range(start_token, shard_len - step_tokens - SEQ - 1, step_tokens):
    block = fetch(start, step_tokens + 1)
    chunk = block[:step_tokens].view(B, SEQ).long()
    target = block[1:].view(B, SEQ)
    tokens_now = start + step_tokens
    ss_prob = ss_prob_at(tokens_now, SS_PROB_START, SS_PROB_END, SS_RAMP_TOKENS)
    carry_snap = snapshot_carry(eng)

    with autocast():
        loss, extras = v4_forward_losses(
            payload(), chunk, target,
            lb_coef=LB_COEF, repeat_coef=REPEAT_COEF,
            ce_chunk=CE_CHUNK, block_ckpt=BLOCK_CKPT,
        )
    ce_tf, lb, rep, h = extras["ce"], extras["lb"], extras["repeat"], extras["h"]

    ss_fired = False
    ce_ss_v = None
    if should_ss(SS_RATE):
        ss_fired = True

    if ACCUM == 1:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if ss_fired:
            restore_carry(eng, carry_snap)
            with autocast():
                loss2, ce_ss = v4_ss_pass(
                    payload(), chunk, target, h.detach(),
                    ss_prob=ss_prob, lb_coef=LB_COEF,
                    ce_chunk=CE_CHUNK, block_ckpt=False,
                )
            loss2.backward()
            torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            ce_ss_v = float(ce_ss.item())
            ema_ss = ce_ss_v if ema_ss is None else EMA_BETA * ema_ss + (1 - EMA_BETA) * ce_ss_v
    else:
        (loss / ACCUM).backward()
        if ss_fired:
            restore_carry(eng, carry_snap)
            with autocast():
                loss2, ce_ss = v4_ss_pass(
                    payload(), chunk, target, h.detach(),
                    ss_prob=ss_prob, lb_coef=LB_COEF,
                    ce_chunk=CE_CHUNK, block_ckpt=False,
                )
            (loss2 / ACCUM).backward()
            ce_ss_v = float(ce_ss.item())
            ema_ss = ce_ss_v if ema_ss is None else EMA_BETA * ema_ss + (1 - EMA_BETA) * ce_ss_v
        pending_backward = True

    tf_v = float(ce_tf.detach().item())
    lb_v = float(lb.detach().item()) if torch.is_tensor(lb) else float(lb)
    rp_v = float(rep.detach().item())
    ema_tf = tf_v if ema_tf is None else EMA_BETA * ema_tf + (1 - EMA_BETA) * tf_v
    ema_rep = rp_v if ema_rep is None else EMA_BETA * ema_rep + (1 - EMA_BETA) * rp_v

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
            mem_s = f" mem={torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
        except Exception:
            mem_s = ""
        print(
            f"GPU {GPU}: {tokens_now:>12,} tf={tf_v:.3f} ema_tf={ema_tf:.3f}{extra} "
            f"rep={rp_v:.3f} ema_rep={ema_rep:.3f} lb={lb_v:.3f} "
            f"ssp={ss_prob:.2f} {tps:.0f} tok/s{mem_s} [boostv4]",
            flush=True,
        )

    if PROBE_EVERY > 0 and n % PROBE_EVERY == 0:
        run_probe(tokens_now)

    # All 8 GPUs save, staggered so they never write 8x4.3G at once.
    if n % 4000 == (GPU * 500) % 4000:
        save_ckpt(tokens_now)

if pending_backward:
    torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)

save_ckpt(start_token + n * step_tokens)
print(f"GPU {GPU}: DONE", flush=True)
