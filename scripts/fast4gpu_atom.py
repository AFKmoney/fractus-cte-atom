#!/usr/bin/env python3
"""Fractus CTE-Atom trainer — the 1B loop of fast4gpu_boost_v4, made Atom-safe.

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
from fractus.train.ar_loss import ss_prob_at  # noqa: E402
from fractus.train.v4_step import (  # noqa: E402
    restore_carry, should_ss, snapshot_carry, v4_forward_losses, v4_ss_pass,
)

assert VOCAB_SIZE == 266, VOCAB_SIZE

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

def open_corpus(path: str):
    """Memmap an Atom stream. Refuses anything that is not ids in 0..265."""
    p = str(path)
    if p.endswith(".i16"):
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
    if int(raw) == 0:
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

    def _apply_lr() -> None:
        f = min(1.0, (n + 1) / warmup_steps) if warmup_steps > 0 else 1.0
        for g in opt.param_groups:
            g["lr"] = lr * f

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

    ckpt_out.parent.mkdir(parents=True, exist_ok=True)

    def save(off: int) -> None:
        ids_done = off * B
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
                "trainer": "fast4gpu_atom",
            },
        }, tmp)
        os.replace(tmp, ckpt_out)
        mtmp = manifest_out.with_suffix(".json.tmp")
        mtmp.write_text(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trainer": "fast4gpu_atom", "gpu": gpu, "n_gpu": n_gpu, "scale": scale,
            "vocab_size": VOCAB_SIZE, "ids_processed": ids_done,
            "layout": "lanes", "batch": B, "lane_len": lane_len,
            "start_token_next": off,
            "note": "start_token_next is the offset inside each lane; resume with the same BATCH",
            "corpus": str(corpus_path), "corpus_slice": [lo, hi], "ckpt": ckpt_out.name,
        }, indent=1))
        os.replace(mtmp, manifest_out)
        log(f"saved -> {ckpt_out} @ lane offset {off:,} ({ids_done:,} ids)")

    def probe(ids_done: int) -> None:
        snap = snapshot_carry(payload())
        p = unique40_probe(payload(), max_new=40, mode="prefix")
        payload().reset_thought(B)
        restore_carry(payload(), snap)
        mean_u = sum(r["unique"] for r in p["rows"]) / max(1, len(p["rows"]))
        log(f"UNIQUE@40 PREFIX mean_unique={mean_u:.1f} @ {ids_done:,} ids | routing {routing_snapshot(payload())}")
        for r in p["rows"]:
            log(f"  {r['prompt']!r}: unique={r['unique']}/{r['n']} -> {r['text']!r}")

    ema_tf = ema_ss = None
    t_tf = t_ss = t_probe = 0.0
    ids_tf = ids_ss = 0
    n = 0
    off = start
    t_wall = time.time()
    last_off = lane_len - SEQ - 1

    while off <= last_off:
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
        ss_prob = ss_prob_at(pos, ss_p0, ss_p1, ss_ramp)
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
        _gn = torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
        _apply_lr()
        if bool(torch.isfinite(_gn)):
            opt.step()
        else:
            log(f"non-finite grad norm at step {n}: optimizer step skipped")
        opt.zero_grad(set_to_none=True)
        sync(); t_tf += time.time() - t0
        ids_tf += step_ids
        tf_v = float(ex["ce"].detach())
        ema_tf = tf_v if ema_tf is None else ema_beta * ema_tf + (1 - ema_beta) * tf_v

        ss_v = None
        if do_ss:
            restore_carry(payload(), carry)
            sync(); t0 = time.time()
            with autocast():
                loss2, ce_ss = v4_ss_pass(
                    payload(), chunk, target, ex["h"].detach(),
                    ss_prob=ss_prob, lb_coef=lb_coef, ce_chunk=ce_chunk, block_ckpt=False,
                )
            loss2.backward()
            _gn = torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
            _apply_lr()
            if bool(torch.isfinite(_gn)):
                opt.step()
            else:
                log(f"non-finite grad norm at step {n}: optimizer step skipped")
            opt.zero_grad(set_to_none=True)
            sync(); t_ss += time.time() - t0
            ids_ss += step_ids
            ss_v = float(ce_ss.detach())
            ema_ss = ss_v if ema_ss is None else ema_beta * ema_ss + (1 - ema_beta) * ss_v

        n += 1
        if n % log_every == 0:
            wall = max(time.time() - t_wall, 1e-9)
            mem = ""
            if device.type == "cuda":
                mem = f" mem_peak={torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
            ss_s = f" ss={ss_v:.3f} ema_ss={ema_ss:.3f}" if ss_v is not None else ""
            log(
                f"{pos:>12,} tf={tf_v:.3f} ema_tf={ema_tf:.3f}{ss_s} "
                f"rep={float(ex['repeat'].detach()):.3f} lb={float(ex['lb'].detach()):.3f} | "
                f"data {ids_tf / wall:.0f} ids/s wall | "
                f"TF {ids_tf / max(t_tf, 1e-9):.0f} ids/s ({t_tf / wall:.0%}) "
                f"SS {ids_ss / max(t_ss, 1e-9):.0f} ids/s ({t_ss / wall:.0%}) "
                f"probe {t_probe / wall:.0%}{mem}"
            )
        if probe_every > 0 and n % probe_every == 0:
            t0 = time.time(); probe(pos); t_probe += time.time() - t0
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
