# How Fractus Is Trained

**Updated:** 2026-08-26 (production x8 run live — measured rates replace estimates; adds optimized v2/v3 path + time-to-finish math)  
**Author:** Philippe-Antoine Robert  
**Model repo:** https://huggingface.co/thefinalboss/fractus-cte  
**Dataset:** https://huggingface.co/datasets/thefinalboss/fractus-datasets

This note explains the actual training procedure for Fractus-1B (Continuous Thought Engine): multi-GPU runs, corpus handling, mid-train surgery, and recovery after host failure.

---

## 1. What Fractus is (training-relevant)

Fractus is not a standard decoder-only transformer trained only with next-token CE on a frozen residual stream.

| Component | Role in training |
|-----------|------------------|
| Continuous thought state | Carries state across ticks |
| Kuramoto oscillators | Phase dynamics for temporal structure / routing |
| Phase-routed MoE | Sparse experts selected via phase/gates |
| Dense CE path (tick_chunk_train) | Teacher-forced sequence loss aligned with gen path |
| Scheduled sampling (SS) | Mix of ground-truth and model predictions as inputs |
| Load-balance loss | Keeps experts alive |
| Gate temperature | Controls routing softness |

Weights live in .pt checkpoints. Code lives in fractus/. Both are required to run.

---

## 2. Corpus (source of truth = HF dataset)

**Source of truth:** thefinalboss/fractus-datasets

A local full_corpus.pt (~4.23B) was historically a concatenation artifact built from this dataset. If that file is missing after a host crash, rebuild from the same HF dataset. The dataset is not lost.

### Dataset layout

- neuro_paradigms_1b/ — neuroscience to architecture paradigms (jsonl.gz)
- cognitive_skills/ — skill / coding / reasoning JSONL
- neuro_code_math/ — math + code + applied neuroscience
- data/training_corpus.pt and datasets/*.pt — already-tokenized streams
- literature / esoteric / repos / identity subsets

### Tokenized streams used in practice

| Stream | Approx size | Location |
|--------|-------------|----------|
| Phase-1 tokenized .pt union | ~1.52B tokens | dataset data/ + datasets/*.pt |
| Phase-2 full raw tokenize | 3.44B tokens | tokenized/phase2/shard_phase2_gpu0-7.npy |

Phase-2: stream all relevant JSONL/JSONL.GZ, GPT-2 BPE encode, write 8 equal int32 numpy memmap shards.

Anti re-ingest: phase-1 and phase-2 are separate streams. After phase-1 progress, switch to phase-2 rather than restarting the same ordered stream from token 0 when the goal is new data.

---

## 3. Multi-GPU training layout

Target: 8x RTX 5090 (recovery). Earlier: 4x.

| Setting | Typical value |
|---------|----------------|
| Processes | 1 Python process per GPU |
| CUDA_VISIBLE_DEVICES | equals GPU_ID |
| Batch | 2 or 3 (B=4 can OOM with SS) |
| Sequence length | 128 |
| LR | 7e-4 (SGD momentum 0.9) |
| SS_RATE | 0.25 |
| SS_PROB | 0.2 |
| LB_COEF | 0.02 |
| Gate temperature | 2.5 |
| TF32 + cudnn.benchmark | on |
| torch.compile | often off (VRAM) |
| Large shards | .npy memmap |

Script: scripts/fast4gpu_boost.py

Each GPU reads only its shard and writes checkpoints/fractus_1b_gpu{i}.pt

### Launch example



Repeat for GPUs 1-7. Never launch multiple workers without unique CUDA_VISIBLE_DEVICES.

---

## 4. Loss signals

| Signal | Meaning |
|--------|---------|
| tf / ema_tf | Teacher-forced dense CE |
| ss / ema_ss | Loss under scheduled-sampling inputs |
| lb | Load-balance term |

Do not equate low TF loss with coherent free-run text. TF can be strong while greedy AR still mono-token collapses until SS + decode path close the train/gen gap.

See docs/LOSS_VS_GEN.md, docs/TRUSTED_LOSS.md, docs/GEN_PROBE_*.

---

## 5. Checkpointing and merge

- Per-GPU: fractus_1b_gpu0.pt ... gpu7.pt
- Mean-merge floating tensors across GPUs -> unified brain (e.g. FRACTUS_1B_PHASE2_LIVE_MERGED.pt)
- Hourly HF sync of 8 individuals + merged (Xet for binaries)
- RESUME_MANIFEST_8GPU.json stores per-GPU token offsets

Checkpoints can be merged, reloaded, continued. Mid-train edits are possible when careful.

---

## 6. Mid-train operability

Distinctive vs typical LLM pretrain:

- Probe experts/phases offline without destroying live checkpoint state
- Decode-path surgery (align tick_chunk vs single-step, anti-collapse)
- Merge parallel trained shards into one model
- Continue after host migration from HF weights

See COMPOSABILITY_AND_SURGERY.md, OPERABILITY_MIDTRAIN.md, DECODE_SURGERY.md.

---

## 7. Recovery playbook (host death)

1. Treat HF as source of truth
2. New pod + torch matching GPU arch (5090 needs recent CUDA builds)
3. Code + dataset from HF
4. Restore weights from checkpoints/fractus_1b_gpu*.pt or merged
5. Restore shards from tokenized/phase2/*.npy or rebuild
6. Resume START_TOKEN from RESUME_MANIFEST_8GPU.json
7. Re-enable hourly Xet upload

Phase switch record: PHASE_SWITCH.json

---

## 8. Current production recipe (phase 2)

1. Dataset fully present from HF
2. Phase-2 tokenization done: 3,439,171,703 tokens -> 8x npy shards (~430M/GPU)
3. Train 8-way BATCH=2, SS on, memmap shards
4. Weights continued from phase-1 (not random init)
5. Checkpoints + phase-2 shards + manifests on HF

Throughput ~900-1100 tok/s/GPU at B=2. One full phase-2 pass ~4-5 days wall-clock.

### Time-to-finish math (phase 2, computed 2026-08-24)

- Shard size (exact, from PHASE_SWITCH.json): **429,896,462 tokens/GPU**
  (×8 GPUs = 3,439,171,696 tokens). The training loop is a **single pass,
  no wrap** (`range(start_token, shard_len - step_tokens - SEQ - 1, ...)`).
- Progress: phase-2 began 2026-08-17T23:26 UTC; last checkpoint sync
  2026-08-18T04:05 (~0.7M tok/GPU); paused since ~2026-08-20 for the
  optimization work → **<1% consumed; ≈425–430M tokens/GPU remain**.
- Wall-clock formula: `days = remaining / (tok_s_per_gpu * 86,400)` —
  all 8 GPUs run in parallel, so per-GPU rate is what matters. Read live
  tok/s from the pod stdout each step.

| Rate (tok/s/GPU) | Time to finish one pass |
|---|---|
| 900–1100 (baseline, measured) | **4.5 – 5.5 days** |
| ~1300 (post-opt, ×1.3) | ~3.5 days |
| ~2000 (×2) | ~2.3 days |
| **~1570 (MEASURED production x8, 2026-08-26)** | **~3.1 days** |

**Production status 2026-08-26:** phase-2 resumed on 8×RTX 5090 with the
optimized stack (v3 lineage: chunked kernel + BLOCK_CKPT + CE_CHUNK=2048,
B=8). Sustained **1,554–1,604 tok/s/GPU ≈ 12,600 aggregate**, VRAM
18.9 GB/32 GB, lb = 14.028 stable. Resume honored gpu0–5 at the x6 positions
(1.25–1.40M), gpu6=655,360 / gpu7=768,000. Hourly safety sync to
`checkpoints/x8run/` + `checkpoints/X8_MANIFEST.json`. Projected finish:
**≈3.1 days from launch**. Deployment automated via
`scripts/pod_deploy_x8.sh`; full measurements in
`OPTIMIZATION_2026-08-22.md` §4.

---

## 9. What done is not

Finishing tokens is not a finished model.

Progress criteria:

1. Stable multi-GPU run
2. TF loss trending down without NaNs
3. SS loss not exploding vs TF
4. Gen probes: rising uniqueness / less mono-token lock
5. Checkpoints recoverable purely from HF

---

## 10. One-sentence summary

Fractus is trained as eight parallel continuous-thought engines on sharded token streams from the HF neuroscience-grounded dataset, optimized with dense teacher-forced CE plus scheduled sampling, checkpointed per GPU, mean-merged and uploaded hourly, and designed so training can be paused, surgically modified, merged, and resumed without treating the run as a single disposable monolith.

---

Machine notes from the live 8x5090 recovery run. Update when the recipe changes.


---

## 11. Optimized v2 training path (fractus-opt, 2026-08-22/23)

A drop-in optimization pass exists in **github.com/AFKmoney/fractus-opt**
(full rationale: `docs/OPTIMIZATION_2026-08-22.md` there). Open-heart
guarantees: every production-path change is proven equivalent to the kept
reference (forward + gradients + carried states; 44/44 tests), checkpoints
and resume offsets unchanged.

What changes (all behind env flags, v1 semantics at defaults):

| Change | Flag | Effect |
|---|---|---|
| Attention kernel cumsum / chunked | `FRACTUS_ATTN_IMPL` | O(C·dH²) instead of O(C²·dH²); chunked is memory-flat ((G, block²+dH²) not (G,C,dH,dH)) → unlocks BATCH≥8 + torch.compile. Measured ×15–25 vs reference at 1B shapes; reference crashes on memory where chunked scales. |
| Memory-flat CE head | `CE_CHUNK` (2048 default) | full-vocab logits never retained for backward (~0.41 GB transient cap at any batch). Loss identical within fp32 rounding. |
| Zero-copy int32 data fetch | always in v2 | no whole-shard int64 upcast → ~27 GB RAM saved per pod. |
| Gradient accumulation | `ACCUM=1` default | ACCUM>1 is a documented deviation from per-step updates. |

Trainer: `scripts/fast4gpu_boost_v2.py` — same loss, same SS schedule, same
SGD recipe, same checkpoint format, same `START_TOKEN` resume semantics as
`fast4gpu_boost.py`. Deployment order and non-regression criteria:
`OPTIMIZATION_2026-08-22.md` §3.

---

*Machine notes from the live 8x5090 recovery run. Update when the recipe changes.*


## CPU mini-Fractus and merges

See **docs/CPU_MINI_MERGE_AND_DIMENSIONS.md** — shape rules, what CPU does and does not advance, mini→1B is not free mean-merge.
