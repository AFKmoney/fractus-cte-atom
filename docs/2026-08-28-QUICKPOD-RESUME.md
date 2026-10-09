# QuickPod resume — 2026-08-28

Exact-token resume of Fractus-1B boost_v4 after the RunPod 8×5090 died.

## What happened

- RunPod SSH (`66.222.154.19:35370`) refused mid-pass.
- Last good HF sync: `checkpoints/x8run/` at **2026-08-28T14:40:54Z**.
- GPU0 `tokens_processed = 79,922,176` / shard `429,896,462` (~18.6% of this Phase-2 pass).
- New machine: QuickPod 8× RTX 5090, empty disk.

## What we did (do not skip these)

1. Snapshot `thefinalboss/fractus-cte` code + `scripts/fast4gpu_boost_v4.py`.
2. Download 8× `checkpoints/x8run/fractus_1b_gpu{i}.pt` + 8× `RESUME_gpu{i}.json`.
3. Download Phase-2 memmaps from `thefinalboss/fractus-datasets`  
   `tokenized/phase2/shard_phase2_gpu{i}.npy` → symlink `data/shard_gpu{i}.npy`.
4. Stock image torch **2.2.1+cu121 has no kernels for Blackwell 5090**  
   (`CUDA error: no kernel image is available`). Upgrade: **torch 2.11.0+cu128**.
5. `set_p0_routing` missing on the published `continuous_engine.py` — added a flag setter so v4 can boot. Weights already carry the earlier routing surgery.
6. Live `unique@40` probe crashed the first launch (`generate_greedy_prefix` not in that generate_aligned). **PROBE_EVERY=0**. Probes stay offline.
7. Relaunch one process per GPU:

```
CUDA_VISIBLE_DEVICES=$i GPU_ID=$i START_TOKEN=<RESUME.start_token_next> \
  BATCH=4 SEQ=128 CE_CHUNK=2048 FRACTUS_ATTN_IMPL=chunked BLOCK_CKPT=1 \
  SS_RATE=1.0 SS_PROB_START=0.2 SS_PROB_END=0.5 P0=1 COMPILE=0 PROBE_EVERY=0 \
  CKPT_IN=checkpoints/x8run/fractus_1b_gpu$i.pt \
  CKPT_OUT=checkpoints/x8run/fractus_1b_gpu$i.pt \
  SHARD=data/shard_gpu$i.npy \
  python -u scripts/fast4gpu_boost_v4.py
```

Hourly HF push of `checkpoints/x8run/*`.

## Resume offsets used 2026-08-28

| GPU | start_token_next |
|-----|------------------|
| 0 | 79,922,176 |
| 1 | 78,443,520 |
| 2 | 76,420,096 |
| 3 | 77,746,176 |
| 4 | 76,160,000 |
| 5 | 77,310,976 |
| 6 | 77,793,280 |
| 7 | 76,677,120 |

Shard length ≈ 429.9M int32 tokens / GPU.

## Honest metrics after resume (first minutes)

- Throughput ~840–940 tok/s/GPU, climbing.
- `ema_tf` on GPU0/1 settled toward mid-teens after the first inflated steps (do **not** read the first 40-step EMA as the run).
- Speech gate remains **unique@40 greedy PREFIX** (no ban, no temperature). Last offline probe before the pod death: 3 / 12 / 1 / 6 on the four standard prompts. That is **NO-GO**.
- Finishing this Phase-2 pass from ~80M @ ~900 tok/s ≈ **4.1–4.5 days** wall if 8 GPUs stay up.

## What is NOT true

- Low teacher-force CE ≠ the model speaks.
- Banned / temperature decode is not the metric.
- This pass is not “4.2B tokens from zero”. Phase-1 weights were already in the seed. This is the remainder of the 430M/GPU Phase-2 shards.
