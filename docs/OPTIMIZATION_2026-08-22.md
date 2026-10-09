# Fractus-1B training optimization — 2026-08-22

**Cardinal rule: open-heart surgery.**
Every change to the code must be **mathematically equivalent to the
reference implementation**, proven by numerical equivalence tests, without
changing the parameter shapes in any way. The brain (`.pt`) does not move;
only the body (the code) is operated on. Token-exact resume via the manifests
stays valid — a pod can switch to this code **mid-run**, without throwing away
any digested tokens.

*(English edition 2026-08-26; original written in French during the
2026-08-22/23 session. All numbers unchanged.)*

---

## 1. Diagnosis: why ~1000 tok/s/GPU when a 5090 can do far more

1B config per block: 20 heads × d_head=64, 2 levels, chunk C=128.
The engine flattens `(B, niv, H)` into `G = B·niv·H = B·40` groups.

### Bottleneck #1 — a cumsum disguised as a matmul (`attention.py`, old code)

```python
S = torch.einsum("tj,bjpq->btpq", mask_tril, outer)   # Σ_{j≤t} outer[j]
```

This is a causal cumsum written as a multiplication by a lower-triangular
mask:

| Form | Operations | At C=128, D=64 |
|---|---|---|
| masked (old) | O(C²·D²) MACs + backward O(C²·D²) | ≈ 67M MACs/group |
| cumsum (new) | O(C·D²) additions | ≈ 0.5M adds/group → **~128× less** |

Backpropagation benefits identically: the derivative of a cumsum is a
reverse cumsum, not another giant masked matmul.

### Bottleneck #2 — materializing the `(G, C, D, D)` tensors

`outer` and `S` are `G·128·64·64` elements each. At B=4: G=160 → **84M
elements (~168 MB in bf16)** each, held for the backward pass,
per block × 16 blocks. This is THE VRAM pressure that forces `BATCH=2-4`
and forbids `torch.compile` ("compile disabled for VRAM" comment in
fast4gpu_boost.py).

*Chunked* form (block=64): only `(G, 64, 64)` (scores) + the running state
`(G, D, D)` are materialized → **~32× fewer dominant activations**.

### Bottleneck #3 — the tied 50257 head

Logits `(B, C, 50257)` fp32 (~103 MB at B=4) + softmax backward are
materialized twice with SS. The head ≈ **half of active FLOPs**
(64.3M of ~192M MACs/token). Chunked checkpointed CE flattens logits memory
(`ce_chunk·vocab` transient), without touching FLOPs.

### Bottleneck #4 — data pipeline

`np.load(mmap).to(torch.int64)` copied **the whole shard into RAM**:
430M tokens × 8 bytes = **3.4 GB/process × 8 processes**. Per-chunk
int32→long slicing removes that (the page cache does the rest).

### GPU utilization estimate

Active FLOPs/token ≈ 1.15 GFLOP (fwd+bwd) → at 1000 tok/s ≈ 1.15 TFLOPS
on a 5090 (~200 TFLOPS peak bf16): **<1% utilization**. The model is bound
by memory / small batches / Python overhead, not by compute. The entire
optimization space is there.

---

## 2. The optimizations (all proven equivalent)

### Opt 1 — cumsum kernel (`fractus/nn/attention.py`)
- `_linear_attention_causal_einsum`: **reference kept** (ground truth).
- `_linear_attention_causal_cumsum`: production default.
- `_linear_attention_causal_vectorized`: dispatcher unchanged for callers.
- Proofs: `tests/test_attention_equivalence.py::test_forward_matches_einsum_*`,
  `test_gradients_match_with_carry`, `test_both_match_looped_reference`.

### Opt 2 — memory-flat chunked form (`_linear_attention_causal_chunked`)
Intra-block via a masked `(block×block)` score matrix + inter-block via the
running state `(S_run, z_run)`; inclusive update AFTER reading (= exact
semantics S_t including token t). Selection: `set_attention_impl('chunked')`
or env `FRACTUS_ATTN_IMPL=chunked`. Ragged → exact cumsum fallback.
- Pod recommendation: **chunked on GPU** (VRAM → compile + large batch);
  cumsum remains perfect on CPU.

### Opt 3 — chunked checkpointed CE (`fractus/nn/ce.py`)
`chunked_cross_entropy(h, W, targets, ce_chunk)`: identical mean loss,
identical grads (fp32 summation-order tolerance), flat memory peaks thanks to
per-chunk `torch.utils.checkpoint` (recompute in backward).
New engine entry point: `tick_chunk_train_ce(obs, targets[, return_hidden])`.
`sample_tokens_chunked(...)` replaces the dense multinomial for SS
(identical distribution; RNG stream consumed per-chunk → draws are not
bit-identical, statistically equivalent).

### Opt 4 — SS semantics preserved + optional accumulation
`scripts/fast4gpu_boost_v2.py` replicates v1 EXACTLY at `ACCUM=1`
(default): TF step then separate SS step. `ACCUM>1` = new capability,
documented deviation (TF+SS grads accumulated, one clip+step every N
batches).

### Opt 5 — zero-copy data pipeline
int32 memmap slice per chunk → transfer → `.long()` on GPU.
A single `(B·SEQ+1)` fetch provides both chunk and target. RAM saved:
~27 GB on an 8×5090 pod.

### Opt 6 — block activation checkpointing (`BLOCK_CKPT`)
Without it, the full 1B config does NOT fit in 16 GB GPUs (OOM even with the
chunked kernel — MoE low-rank activations dominate). Naive whole-block
checkpointing is FORBIDDEN: `tick_chunk_core` reads-and-mutates the carry
inside the block, so naive recomputation produces wrong gradients. Refactor:
pure core `_tick_chunk_core_pure(h, carry_S_flat, carry_z_flat) -> (h, lb,
S_final, z_final, theta)` with the carry read-mutate cycle kept OUTSIDE the
checkpoint region (`_build_carry` / `_store_carry`). Engine method gains
`block_ckpt=`; trainer env `BLOCK_CKPT=1`. Proven equivalent by
`tests/test_block_ckpt.py` (losses, ALL param grads, carry evolution across
consecutive chunks).

---

## 3. How to deploy on a pod (open heart)

```bash
# 1. Back up state (nothing more to do: HF = source of truth)
#    The gpu*.pt checkpoints + RESUME_MANIFEST stay valid as-is.

# 2. Replace the body:
#    fractus/nn/attention.py, fractus/nn/ce.py (new),
#    fractus/continuous_engine.py (method added, nothing removed),
#    scripts/fast4gpu_boost_v2.py (new).

# 3. Relaunch every GPU with the SAME offsets as before the stop:
CUDA_VISIBLE_DEVICES=$i GPU_ID=$i \
  START_TOKEN=$(python -c "import json;print(json.load(open('checkpoints/RESUME_MANIFEST_8GPU.json'))['gpus']['$i']['start_token_next'])") \
  BATCH=8 CE_CHUNK=2048 FRACTUS_ATTN_IMPL=chunked BLOCK_CKPT=1 COMPILE=1 \
  python -u scripts/fast4gpu_boost_v2.py
```

Recommended escalation order (validate at every step):
1. Code swap, same settings as v1 (`BATCH=4 CE_CHUNK=0`) → check ema_tf
   continues its curve exactly (real-condition equivalence).
2. `CE_CHUNK=2048` → identical loss, VRAM ↓.
3. `FRACTUS_ATTN_IMPL=chunked` → VRAM ↓↓ (measured §4: ×15–25 vs reference,
   flat memory where the reference explodes).
4. `COMPILE=1` then raise `BATCH` (8, 16…) — watch tok/s and VRAM.
5. Optional `ACCUM` if a larger effective batch is wanted without OOM.

Non-regression criteria (see docs/TRUSTED_LOSS.md): ema_tf keeps descending
without NaN, lb stable ~14, generation probes unchanged in behavior.

For fully automated pods, `scripts/pod_deploy_x8.sh` performs the whole
flow (env → assets from HF → smoke benchmark → launch → verify) detached,
with stage markers so re-runs are cheap.

---

## 4. Measured results

### CPU local (torch 2.9.1+cpu, 2026-08-23)

Attention micro-bench at real 1B shapes (G=B·40 groups, C=128, dH=64),
carry active; median over 8 iterations. Each cell runs in its own
subprocess: a native crash in one cell does not take down the table.

| Impl | B | G | fwd ms | fwd+bwd ms | RSS Δ MB |
|---|---|---|---|---|---|
| einsum (ref) | 2 | 80  | 267.4 | 572.1 | +3 |
| einsum (ref) | 4 | 160 | 584.7 | 1126.6 | 0 |
| einsum (ref) | 8 | 320 | — | — | native crash |
| cumsum       | 2 | 80  | 322.4 | 799.1 | −8 |
| cumsum       | 4 | 160 | — | — | native crash |
| chunked      | 2 | 80  | **12.3** | **31.9** | 0 |
| chunked      | 4 | 160 | **23.0** | **72.4** | 0 |
| chunked      | 8 | 320 | **40.5** | **150.0** | −10 |

Honest reading of these numbers:

- **Chunked vs einsum at equal B**: ×21.7 (fwd) and ×17.9 (fwd+bwd) at B=2;
  ×25.4 / ×15.6 at B=4.
- **Memory**: einsum and cumsum materialize the (G, C, dH, dH) tensor
  (~0.34 GB at G=160, ~1.3 GB at G=320, multiplied by backward buffers).
  On this commit-memory-limited machine they segfault (rc=3221225477) from
  G≥160–320; chunked is memory-flat ((G, block²+dH²)) and crosses every
  tested size. This is exactly the property that unlocks BATCH≥8 +
  torch.compile on pods.
- **Cumsum on CPU is slower than einsum**: the masked einsum drops into very
  optimized BLAS bmm, while the element-at-a-time scan is bandwidth-bound.
  The O(C²·dH²)→O(C·dH²) FLOP reduction only pays off on GPU/compile —
  which is why the repo default stays `cumsum` (mathematically proven,
  simple semantics) and the pod recommendation jumps straight to `chunked`.
- Absolute CPU timings do not transfer to CUDA; what transfers: the relative
  kernel ordering, the memory profile, and the measured fact that chunked
  scales where the reference explodes. GPU bench to be run on pods (same
  script).

End-to-end engine CPU (bench_engine.py, d=128, 2 blocks, E8, B=8, SEQ=128,
20 steps):

| Kernel | tok/s |
|---|---|
| cumsum (default) | 703 |
| chunked | 764 (+8.7 %) |

### Real GPU validation (6× RTX 5060 Ti 16 GB, torch 2.11+cu128, 2026-08-24)

Full 1B config (d1280 ×16 blocks, E128, bf16 autocast), synthetic smoke then
complete trainer v2 on real checkpoints + phase-2 shards:

| Config | tok/s/GPU | VRAM peak |
|---|---|---|
| cumsum + BLOCK_CKPT, B=2 | 338 | 14.3 GB |
| chunked + BLOCK_CKPT, B=2 | 431 | 14.0 GB |
| **chunked + BLOCK_CKPT, B=3** | **483** | **14.8 GB** |
| chunked + BLOCK_CKPT, B=4 | 501 | 15.6 GB |

- **Without BLOCK_CKPT, the 1B config does NOT fit in 16 GB** (OOM from the
  MoE forward on, even with the chunked kernel): block checkpointing is what
  opens up 16 GB cards. Naive version forbidden (carry read-mutate) →
  pure-core refactor `_tick_chunk_core_pure`, proven equivalent by
  `tests/test_block_ckpt.py` (losses, grads, carry evolution across
  consecutive chunks). Suite: 46/46.
- On GPU, **chunked beats cumsum by ×1.27** at equal B: the FLOP reduction
  finally pays off (unlike CPU, see above).
- Complete trainer v2 (SS branch included): **~340–382 tok/s/GPU sustained**
  at B=3, stable mem 14.8 GB, `loaded_tensors=424` on each GPU (real weights
  fully loaded), lb≈14.03, tf descending.

**ETA for the phase-2 pass on that x6 5060 Ti box** (3,433M tokens remaining,
aggregate ~2,170 tok/s): **≈ 17–19.5 days** (~18 days midpoint).

### Production x8 launch (8× RTX 5090 32 GB, torch 2.11+cu128, 2026-08-26)

Deployed with `scripts/pod_deploy_x8.sh` (one-shot, detached, stage-marked):
env setup → HF asset pull (code + manifests + 8 checkpoints + 8 shards) →
smoke → launch → verify.

Resume offsets honored exactly: gpu0–5 from `checkpoints/X6_MANIFEST.json`
(1,249,152 … 1,397,760 — positions reached by the short-lived x6 validation
run), gpu6 = 655,360 / gpu7 = 768,000 from the production
`RESUME_MANIFEST_8GPU.json`.

| Measurement | Value |
|---|---|
| Smoke GPU0, B=8 | **1,527 tok/s/GPU**, peak 19.85 GB |
| Sustained, all 8 GPUs, B=8 | **1,554–1,604 tok/s/GPU** (~12,600 aggregate) |
| Steady-state VRAM | ~18.9 GB / 32 GB |
| Health | lb = 14.028 stable on all 8; tf descending; SS branch firing |
| Config | B=8 SEQ=128 CE_CHUNK=2048 chunked BLOCK_CKPT=1 expandable_segments |

Reading:

- On 5090s the optimized stack sustains **~×4.2 the per-GPU throughput**
  measured on 5060 Ti at its B=3 sweet spot (~365 tok/s in-trainer there vs
  ~1,570 here), and roughly ×1.4–1.7 the pre-optimization 5090 baseline
  estimate (900–1100 tok/s/GPU).
- Remaining phase-2 volume ≈ 3,420M tokens ÷ 12,600 tok/s ≈
  **≈ 3.1 days** projected (vs ~18 days on the x6 box, vs 4.5–5.5 days
  baseline estimate for unoptimized v1 on 8×5090).
- Safety net: local atomic saves every 800 iterations + hourly HF sync to
  `checkpoints/x8run/fractus_1b_gpu{i}.pt` + `checkpoints/X8_MANIFEST.json`.

## 5. Tests

```bash
py -m pytest tests/ -q        # 46 passed = repo (28) + equivalences (14) + smoke v2 (2) + block-ckpt (2)
py benchmarks/bench_attention.py --iters 8
py benchmarks/bench_engine.py --steps 20
```

Status: **46/46 passing** (local torch 2.9 CPU and pod torch 2.11+cu128,
2026-08-24). The two engine proofs
(`test_engine_tick_chunk_train_ce_matches_train`,
`test_engine_end_to_end_chunk_equivalence`) require explicit weight cloning
(`load_state_dict`) between compared instances — two constructions under one
seed have DIFFERENT weights, trap documented in the docstrings.

---

*Document produced during the 2026-08-22/23 optimization session, extended
2026-08-24 (GPU validation) and 2026-08-26 (English edition + production x8).
Rule: never merge into fractus-cte unless section 4 is filled.*
