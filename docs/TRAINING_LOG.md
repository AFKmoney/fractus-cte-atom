# Training log, Oct 2026 (ATOM 1B, AdamW restart)

All entries are from 2026-10-11 unless noted. Only facts recorded during the run are listed. Times are not given.

## Chronology

1. **First run: SGD + repeat term.** Trained with SGD and the `REPEAT_COEF` term. The run diverged: the repeat log-prob plunged and CE degraded (seen at `REPEAT_COEF=0.1`). The term is unbounded.
2. **Restart with `REPEAT_COEF=0`.** The run continued without the repeat term.
3. **Pod expired.** The Vast.ai pod stopped at the contract `end_date`, after about 4.5 hours.
4. **New pod, resume.** A new pod was rented and training resumed from the last checkpoint.
5. **Patches to `scripts/fast4gpu_atom.py`** (applied on the pod; see [`pod_patches.diff`](pod_patches.diff)):
   1. Strict checkpoint load fix: batch-dependent carry buffers (`thought_state`, `attn_S`, `attn_z`, `kuramoto_phases`) are replaced by the new engine's buffers on load.
   2. Added the missing `ss_prob = ss_prob_at(pos, ss_p0, ss_p1, ss_ramp)` line.
   3. Linear LR warmup (`WARMUP_STEPS`).
   4. Optimizer state saved to `<ckpt>.opt` (written before the `.pt` and the manifest).
   5. Optimizer state reloaded on resume, with a warning if missing.

   The pod's script also skips the optimizer step when the gradient norm is non-finite (logged). This guard is not part of the five patch notes.
6. **LR / optimizer tests** (table below). SGD stayed on a plateau whatever the LR. AdamW at 3e-4 trained; AdamW at 1e-3 diverged.
7. **Switch to AdamW.** Current run: `OPT=adamw LR=3e-4 WARMUP_STEPS=800 BATCH=16 SEQ=128 COMPILE=1 PROBE_EVERY=0 SS_RATE=0.25 REPEAT_COEF=0 SAVE_EVERY=10000 ...`

## LR / optimizer tests

| Optimizer | LR | Result |
|---|---|---|
| SGD | 7e-4 | plateau, tf ≈ 3.2 |
| SGD | 3e-3 | plateau, tf ≈ 3.2 |
| SGD | 2e-4 | plateau, tf ≈ 3.2 |
| SGD (61M-id run) | n/a | stayed at tf ≈ 3.3 |
| AdamW | 3e-4 | tf ≈ 2.24 at 1.64M ids, from scratch |
| AdamW | 1e-3 | diverges |

tf = teacher-forced loss. SGD tf ≈ 3.2-3.3 regardless of LR.

## Throughput benchmarks

1x RTX 5090, B=16, SEQ=128.

| Setting | ids/s (wall) |
|---|---|
| Non-compiled | 3.65k |
| `COMPILE=1`, `SS_RATE=1` | ~8.9k |
| `COMPILE=1`, `SS_RATE=0.25` | 13.5k |
| `COMPILE=1`, `SS_RATE=0` | ~16k |
| `COMPILE=1`, AdamW, `SS_RATE=0.25` | ~11.4k |

## Operational notes

- `COMPILE=1` requires `PROBE_EVERY=0`. The probe crashes with `index is on cpu` in `unique40_probe`.
- Resume: set `CKPT_IN` and `START_TOKEN` (= `start_token_next` from the manifest), keep the `.opt` next to the `.pt`, and use `WARMUP_STEPS=0` if the `.opt` is loaded.
- The Hugging Face Hub silently ignores `.pt` files. Checkpoints are stored as `.bin` and optimizer state as `.opt.bin`.
- Hugging Face checkpoint locations (`thefinalboss/fractus-cte-atom`): `checkpoints/atom/` is the current AdamW run; `checkpoints/atom_sgd_61M/` is the archived SGD run.
- Vast.ai: rentals have an `end_date`. Pick offers whose duration covers the whole run.

## 2026-10-12: NaN gradient in `elu_plus_one`

- **Symptom.** The first full-corpus AdamW run (`LR=3e-4`, `WARMUP_STEPS=800`) logged `non-finite grad norm` from step 1769 (~3.6M ids). Within a few hundred steps, nearly every TF and SS step was skipped (about 1.25 skip messages per step). The weights stayed frozen at tf ≈ 2.3 (ema_tf ≈ 2.36) up to ~82M ids. An earlier AdamW run with the same seed crashed at the same point with a device-side assert in `sample_tokens_chunked` (NaN probabilities), because it had no skip guard and the weights became NaN.
- **Cause.** `fractus/nn/stats.py` `elu_plus_one` computed `exp(x)` for all x in the discarded branch of `torch.where`. For x > ~88.7, `exp(x) = inf`, and the backward gives 0 * inf = NaN while the forward stays finite. AdamW pushed the block-0 key pre-activation to 88.9.
- **Reproduction (CPU, frozen checkpoint, B=2, SEQ=128).** Max pre-activation 73.1 gave a finite gradient (norm 15.6). Max 88.9 gave a NaN gradient in `observe.weight`, `blocks.0.attn.w_qkv`, `blocks.0.attn.b_qkv` and `blocks.0.norm_attn`. With the fix, the same inputs gave finite gradients (norms 8.5 to 21.7).
- **Ruled out.** Unbounded carry: documents are ~3.7k ids with a BOS in every lane, so each row resets every ~29 chunks. The repeat term: the SS pass, which has no repeat term, also failed.
- **Fix.** `neg = alpha * (torch.exp(x.clamp(max=0.0)) - 1.0) + 1.0`. Same forward values, finite gradient. Skips now log the first 5 parameter names with non-finite gradients.
- **Restart.** The frozen checkpoints were deleted and the run restarted from scratch with the fix (`MAX_STEPS=2153780`, one pass over every lane, `SAVE_EVERY=5000`).
- **Validation.** Fresh run, 2000 steps, same config: 0 non-finite messages; tf 2.158, ema_tf 2.134 at 4.10M ids (step 2000), past the old failure step.
- **Attention safe division: tested, not kept.** `num / torch.where(safe, den, 1)` in `fractus/nn/attention.py` was forward bit-identical (fp32 and bf16) but TF throughput fell from 12.7k to 9.4k ids/s at step 400. Reverted; it is not needed for the NaN fix.

## 2026-10-12/13: divergences without NaN, guarded restart

- **Run with LR 3e-4 constant (fresh, full corpus).** Mean tf per 100M ids: 2.07 (0-100M), 2.42-2.61 (100-700M). Between 710.2M and 710.7M ids, tf went from 3.03 to 36.65; ema_tf then stayed at ~12-50 (mean tf 13.9 over 700-780M ids). 0 non-finite gradients. Corpus check at the jump position (all 16 lanes): no ids outside 0-265, no zeros, runs of at most 2-3 identical ids, id histogram close to 10M ids earlier. Stopped at 778.2M ids.
- **Resume from 409.6M ids with cosine LR (3e-4 to 3e-5 over 1,953,780 steps), clip 1.0.** Pre-clip grad norm bucket means between 700M and 780M ids: 40.5-60.2, no upward trend. At 788.9M ids gn = 583 (tf 30.3), at 789.7M ids gn = 3,220 (tf 78.9), LR 2.94e-4. Later gn values reached 963,532. Stopped at 849.9M ids. An automatic 10 h push had replaced `checkpoints/atom/` with the diverged 829.4M checkpoint; it was restored to the 409.6M checkpoint.
- **Clipping.** `clip_grad_norm_(..., 1.0)` was active in both runs (it already existed in the script); it did not prevent the divergences.
- **Guarded restart** from the 409.6M checkpoint: LR 1e-4 cosine to 1e-5, 500-step warmup, spike guard (skip when gn > max(300, 10 x EMA)), rollback request after 50 consecutive skipped steps or ema_tf > 4.0 for 1,000 steps (runner restarts from the last good checkpoint at half LR, max 3 times), rotating step checkpoints with a `good/` copy (ema_tf < 3.0, no skip in the last 2,000 steps), and a push loop that uploads only `good/`. Healthy checkpoint archived in `checkpoints/atom_409M_good/`.


## 2026-10-13/17: guarded run, rollbacks, LR offset fix, resume at 2.485B ids

- **Rollbacks.** The guarded run rolled back twice to the last `good/` checkpoint at half LR: on Oct 13 and on Oct 14 (00:57 UTC, 51 consecutive skipped steps at 877.8M ids; LR 5e-5 to 2.5e-5, min 2.5e-6).
- **LR offset fix.** A runner restart reset the cosine schedule to step 0. `scripts/fast4gpu_atom.py` now reads `LR_STEP_OFFSET` and `LR_TOTAL_STEPS` and computes `t = (n + LR_STEP_OFFSET) / LR_TOTAL_STEPS`, so a resume continues the schedule instead of restarting it. The runner (`scripts/atom_runner.py`) stores both values in its state file.
- **Resume.** On Oct 15 (17:22 UTC) the run resumed from the `good/` checkpoint at 2.485B ids with `WARMUP_STEPS=0`, `LR_STEP_OFFSET=785000`, `LR_TOTAL_STEPS=1725158`, 940,158 steps left. Guards (spike skip, rollback request, `good/` checkpoints, push of `good/` only) unchanged.
- **Status (Oct 16).** ~3.61B ids, ~11.3k ids/s wall (TF 82%, SS 17%), LR ~5.2e-6, mem peak 29.3 GB. Expected to finish around Oct 17.

## 2026-10-17: v2 trainer (fast4gpu_atom_v2), resume from atom_final

- **Final v1 run.** Finished Oct 17 16:48 UTC at step 940,158: 4,410,941,440 ids (one full pass over every lane), ema_tf 0.332, 3,199 skipped steps in total, 11,279 ids/s wall (TF 82%, SS 17%). Pushed to HF `checkpoints/atom_final/` (commit `d9baf70`).
- **New file `scripts/fast4gpu_atom_v2.py`** (copy of `fast4gpu_atom.py`, which is unchanged). Every addition is env-gated, and the defaults keep the old behavior:
  - `SS_RAMP_ORIGIN_IDS`: the SS probability ramp counts ids from this position. In v1 the ramp ran on absolute ids, so on a resume it was already finished. There is no other cap on the SS probability: `ss_prob_at` clamps to [start, end], and the `0.5 *` in `v4_ss_pass` is a loss weight.
  - `CARRY_DECAY` / `CARRY_CLAMP`: scale or norm-clamp `attn_S` and `attn_z` after each chunk (detached). `thought_state` and the Kuramoto phases are not touched.
  - `EPOCH_WRAP=1`: `start_token_next` 275,683,840 is exactly the end of each lane (period = ((lane_len-1)//SEQ)*SEQ). The lane offset becomes `off % period`; `START_TOKEN`, `ids_processed` and the manifests stay monotonic; all carries reset at a wrap.
  - The log line also prints `ss_p`.
- **New runner and pusher.** `scripts/run_v2.py` (copy of the run8 runner: `/workspace/ckpt_v2`, `run_v2_state.json`, `train_v2.log`, same rollback logic), `scripts/push_good_atom_v2.py` (push_good v2 on `ckpt_v2`, pushes `good/` every 10 h to `checkpoints/atom_v2/` only), `scripts/launch_v2.sh`.
- **LR=0 carry-decay check.** good/ checkpoint, 60 steps, no weight change, TF loss:

  | attn_S/attn_z decay per chunk | TF loss |
  |---|---|
  | 1.0 | 0.5-1.2 |
  | 0.99 | 0.8-1.3 |
  | 0.97 | 1.0-2.2 |
  | 0.9 | 2.2-3.3 |
  | 0.5 | 6.6-9.8 |

  Uniform is ln 266 = 5.58. The model depends on the absolute size of the carry.
- **Aborted run 1** (decay 0.5, SS_RATE 1.0, SS 0.5->0.85, LR 1e-4 warmup 500): tf 13.7 at step 200, ema_tf 6.7 at step 800, rollback request at step 1000 (ema_tf > 4 for 1000 steps). Stopped.
- **Aborted run 2** (decay 0.97, same otherwise): ema_tf 8.3 / 11.9 / 11.3 / 9.4 at steps 200/400/600/800, 30 skips. Stopped.
- **Isolation tests.** 300 steps each from good/, no runner, scratch dirs. tf / ema_tf / ss at steps 50, 100, 200, 300 (ema starts high because the first chunk starts from an empty carry; "-" = no SS pass on that logged step):

  | test | decay | SS_RATE | LR (warmup 500) | step 50 | step 100 | step 200 | step 300 | skips | ids/s |
  |---|---|---|---|---|---|---|---|---|---|
  | T4 | 1.0 | 0.25 | 2.5e-5 | 0.44 / 16.7 / - | 0.69 / 6.46 / 1.84 | 0.65 / 1.36 / - | 0.63 / 0.78 / - | 1 | 11,585 |
  | T1 | 1.0 | 1.0 | 1e-4 | 2.89 / 17.7 / 4.12 | 3.90 / 8.79 / 5.02 | 8.78 / 8.37 / 9.89 | 7.39 / 9.46 / 9.08 | 9 | 7,478 |
  | T2 | 1.0 | 0.25 | 1e-4 | 0.46 / 16.7 / - | 0.73 / 6.56 / 1.54 | 2.78 / 2.08 / - | 2.56 / 2.70 / - | 1 | 11,464 |
  | T3 | 0.97 | 1.0 | 2.5e-5 | 2.09 / 17.6 / 3.32 | 2.05 / 7.74 / 3.55 | 4.06 / 3.91 / 5.38 | 4.34 / 4.06 / 6.17 | 3 | 7,435 |

  SS on every step damages TF even at LR 1e-5 (T1, T3). LR 1e-4 also degrades TF on its own (T2). Carry decay 0.97 is not the main factor. 0 non-finite gradients in all tests.
- **Launched run (Oct 17 21:45 UTC) = T4 config.** `CARRY_DECAY=1.0 SS_RATE=0.25 SS_PROB_START=0.5 SS_PROB_END=0.85 SS_RAMP_TOKENS=100000000 SS_RAMP_ORIGIN_IDS=4410941440 EPOCH_WRAP=1`, AdamW LR 2.5e-5 cosine to 2.5e-6, warmup 500, `LR_TOTAL_STEPS=MAX_STEPS=2153780` (one pass), weights and optimizer from `ckpt_full/good/` (step_00940158), `START_TOKEN=275683840`, guards and rollback unchanged, `OUT_DIR=/workspace/ckpt_v2`.
