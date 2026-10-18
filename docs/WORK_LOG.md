# Work log: Fractus CTE-Atom 1B (Oct 2026)

Fractus is an ACI (artificial cognitive intelligence), not an LLM. This log records every training and operations decision, why it was taken, and what it did, with numbers. It is chronological, and times are US Eastern (ET, UTC-4). Where a source only gives a date, no time is shown. Sources: `docs/TRAINING_LOG.md`, `docs/GENERATION_SURGERY.md`, and the pod logs (`runner.log`, `push.log`, `ROLLBACK_*.json`, `manual_push.log`, `final_push.log`, `train*.log`). New decisions are appended at the end.

Hardware: one Vast.ai RTX 5090 pod (instance 55477318 for the Oct 12-17 runs, $0.525/h plus storage). Model: ContinuousThoughtEngine "1b" scale, 1,864,309,426 parameters, vocab 266 (Atom ids). Corpus: `atom_corpus.i16`, 4,410,941,912 ids, 16 lanes of 275,683,869 ids. B=16, SEQ=128 (2,048 ids/step).

## Oct 11

| Decision | Why | Effect |
|---|---|---|
| First run: SGD + `REPEAT_COEF` anti-repeat term | Initial v4 recipe | Diverged: the repeat log-prob plunged and CE degraded at `REPEAT_COEF=0.1` (the term is unbounded). |
| Restart with `REPEAT_COEF=0` | Remove the unbounded term | Run continued. |
| Pod expired, new pod rented, resume from last checkpoint | Vast contract `end_date` reached after ~4.5 h | Lesson: pick offers whose duration covers the whole run. The new pod was set up on Oct 11, about 21:30 ET. |
| Pod patches to `fast4gpu_atom.py`: strict load with batch-sized carry buffers replaced, missing `ss_prob` line, LR warmup, `.opt` save/reload | Resume failed or was incomplete without them | Resume works with the optimizer state. |
| LR / optimizer sweep | SGD stuck | SGD plateaued at tf ~3.2 for LR 2e-4 / 7e-4 / 3e-3. AdamW 3e-4 reached tf 2.24 at 1.64M ids. AdamW 1e-3 diverged. **Switched to AdamW 3e-4, warmup 800.** |
| Throughput benchmarks (COMPILE=1) | Choose SS_RATE | SS_RATE=1: ~8.9k ids/s; 0.25: 13.5k; 0: ~16k; AdamW + 0.25: ~11.4k. **Kept SS_RATE=0.25.** |

## Oct 12

| Decision | Why | Effect |
|---|---|---|
| Fix `elu_plus_one` (`exp(x.clamp(max=0))` in the unused branch) | From step 1769 (~3.6M ids), non-finite grad norms skipped almost every step and froze the weights at tf ~2.3. Cause: `exp(x)=inf` for x>88.7 in the discarded `torch.where` branch, and 0*inf=NaN in backward. | Reproduced on CPU (pre-activation 88.9 gives NaN, 73.1 is finite). After the fix: 0 non-finite in 2000 steps, tf 2.158 at 4.10M ids. Frozen checkpoints deleted, restart from scratch with `MAX_STEPS=2153780`. |
| Attention safe division tested, not kept | Possible second NaN source | Forward bit-identical, but TF throughput fell from 12.7k to 9.4k ids/s. Reverted. |

## Oct 12-13: divergences without NaN

| Decision | Why | Effect |
|---|---|---|
| Run at LR 3e-4 constant | First clean full-corpus run | Mean tf 2.07 (0-100M), 2.42-2.61 (100-700M). **Divergence between 710.2M and 710.7M ids**: tf 3.03 to 36.65, ema_tf 12-50 afterwards, 0 non-finite. The corpus at that position was checked and found clean. Stopped at 778.2M. |
| Resume from 409.6M ids, cosine 3e-4 to 3e-5 over 1,953,780 steps, clip 1.0 | Lower LR over time | **Diverged at ~789M ids**: gn 583 (tf 30.3) at 788.9M, gn 3,220 (tf 78.9) at 789.7M, later up to 963,532. Stopped at 849.9M. Clip 1.0 did not prevent either divergence. |
| Restore HF `checkpoints/atom/` | The 10 h auto-push had uploaded the diverged 829.4M checkpoint | Restored to the 409.6M checkpoint (HF commit `8d8507d`). Healthy copy archived in `checkpoints/atom_409M_good/`. |
| **Guarded restart** (Oct 13 09:22 ET, `run8.py` + `launch8.sh`) from 409.6M (start_token 25,600,000): LR 1e-4 cosine to 1e-5, warmup 500, spike guard (skip if gn > max(300, 10 x EMA), 3000 during a 200-step grace), rollback request (exit 42) after 50 consecutive skipped steps or ema_tf > 4.0 for 1000 steps, runner restarts from `good/` at half LR (max 3), rotating step checkpoints (keep 2, only the newest keeps `.opt`) plus `good/` (hard links), push of `good/` only | Make divergence recoverable automatically and never push a diverged checkpoint | Ran from 409.6M to 871.9M ids before the first rollback. |
| Consecutive-skip counter fix (`patch7b`) | The counter was incremented per guard call (TF and SS) and reset by the other pass's accepted step, so it did not count steps | Now counts steps that had at least one skip. The rollback criterion means "50 consecutive bad steps". |

## Oct 13 (evening ET): two rollbacks, relaxed good criterion

| Decision | Why | Effect |
|---|---|---|
| **Rollback 1**, automatic (Oct 13 20:48 ET) | 51 consecutive skipped steps at step 225,711, 871,856,128 ids, LR 9.71e-5 | Restart from `good/` at LR 5e-5 (min 5e-6). Data position kept moving forward (start_token 54,491,008). |
| **Rollback 2**, automatic (Oct 13 20:57 ET) | 51 consecutive skipped steps after only 2,911 steps, 877,817,856 ids, LR 5e-5 | Restart at LR 2.5e-5 (min 2.5e-6), start_token 54,863,616, 1,725,158 steps left. This LR then stayed for the rest of v1. |
| **Relaxed good criterion**, `push_good.py` v2 (Oct 13 18:29 ET) | The trainer's strict `good/` (no skip in the last 2000 steps) almost never updated, because isolated spike skips happen every few hundred steps | A step checkpoint becomes `good/` if ema_tf < 3.0, < 20 skips in the last 2000 steps, no ROLLBACK.json. 181 promotions in `push.log`; 10 h pushes to `checkpoints/atom` + `atom_latest` at 2.84B, 3.25B, 3.66B, 4.07B ids. |
| Rotation bug fix | (Details not recoverable from the pod logs; to be filled in.) Current behavior: keep the last 2 `step_*` dirs, only the newest keeps `.opt`, `good/` is a hard-linked copy, so pruning never deletes `good/` files; `_ensure_space` drops old step dirs below 17 GB free. | Rotation and `good/` ran without loss for the rest of v1. |

## Oct 15: credit stop, LR offset fix

| Decision | Why | Effect |
|---|---|---|
| Manual push of `step_00735000` (Oct 15 10:20 ET) | Secure a recent good checkpoint before the run stopped | 2,383,097,856 ids, ema_tf 0.561 pushed to `checkpoints/atom` (`a29f2e7`) and `atom_latest` (`ccbeb87`). |
| Credit stop | Vast credit ran out and the trainer stopped (exact stop time not in the logs) | Training paused at the last `good/` (2,485,497,856 ids). |
| Relaunch (Oct 15 13:18 ET) caught with stale runner state | The state file still held start_token 54,863,616 and a cosine restarting at step 0 | Stopped after ~4 min. |
| **LR offset fix** + resume (Oct 15 13:22 ET) | A runner restart reset cosine to step 0 | Trainer computes `t = (n + LR_STEP_OFFSET) / LR_TOTAL_STEPS`; runner stores both. Resumed at start_token 155,343,616 (2.485B ids), `WARMUP_STEPS=0`, `LR_STEP_OFFSET=785000`, `LR_TOTAL_STEPS=1725158`, 940,158 steps left. |
| Oct 16 status | - | ~3.61B ids, ~11.3k ids/s wall (TF 82%, SS 17%), LR ~5.2e-6, mem peak 29.3 GB. |

## Oct 15-17: generation surgery (inference only, no weight change)

Checkpoint `atom_latest` step_00770000 (2.455B ids, ema_tf 0.484), CPU. Full tables are in `docs/GENERATION_SURGERY.md`.

| Decision | Why | Effect |
|---|---|---|
| Block modes A (tick the last C ids each step) vs B (carry committed at chunk boundaries, regenerate the pending block) | Mirror the training forward | Mode B at C=128 reproduces training CE exactly. Free-running output is morpheme salad in every config: unique@40 0.35-0.53; ~20 correct ids on in-corpus text, then drift. Mode A is worse (0.25-0.40). |
| Start states warm / cold BOS / 256-id primer | Find the right regime | In-corpus TF after one chunk 0.1-2.5. Novel prompts TF 6-17 nats/id regardless of start. Cold first chunk 55-200 nats/id. |
| Measure carry norms vs TF reference | Suspected attn_S inflation | In TF, attn_S grows ~linearly (~2.7e9 Frobenius per chunk, 7.4e10 after 27 chunks), attn_z ~1.5e7 per chunk. No decay. Generation is **not** inflated: Mode A S at or below the reference, z under (S/z ~2x reference with primer, ~20x from BOS). thought_state 2-7x inflated in Mode A. Mode B carry equals TF. |
| x0.5 carry (half), clamp to reference norm | Test carry fixes | Mode A x0.5: greedy unique@40 from ~0.28 to ~0.46, still not words. Clamp: no change. In Mode B the fixes never trigger within 80 ids. Noise between reruns is about ±0.15. |
| Inference-side SS (20%) | Test exposure bias at inference | unique@40 0.35-0.525, no quality change. |
| Conclusion | - | The failure is exposure bias in the model (next-id prediction on its own samples), not the inference driving. A training-side fix (more SS) is needed. |

## Oct 17: end of v1, atom_final

| Decision | Why | Effect |
|---|---|---|
| Run to the end of the pass | - | Finished Oct 17 12:48 ET, step 940,158, **4,410,941,440 ids, ema_tf 0.332**, gn_ema 25.1, 3,199 skips in total, 11,279 ids/s wall. |
| Final push to `checkpoints/atom_final/` (13:08-13:10 ET) | Archive the end-of-pass checkpoint separately from the rolling `atom/` | HF commit `d9baf70`. GOOD.json ids 4,410,941,440; .bin/.opt.bin sizes match `good/`. |

## Oct 17: v2 trainer and its first attempts

| Decision | Why | Effect |
|---|---|---|
| New `scripts/fast4gpu_atom_v2.py` (old trainer untouched, `.bak20261017`) with env-gated `SS_RAMP_ORIGIN_IDS`, `CARRY_DECAY`/`CARRY_CLAMP`, `EPOCH_WRAP`. New `run_v2.py`, `push_good_atom_v2.py` (to `checkpoints/atom_v2/`), `launch_v2.sh`, ckpt dir `/workspace/ckpt_v2` | Stronger SS (to 0.85) against exposure bias; bounded carry; a second pass (`start_token_next` 275,683,840 is exactly the end of each lane, so v1 would run 0 steps) | GPU smoke test (tiny model, 300 steps): no NaN, SS every step, ss_p 0.5 to 0.85, 3 epoch wraps. |
| SS settings found in v1 | Asked why SS was weak | v1 `SS_RATE=0.25` (share ~17%). The ramp on absolute ids was already at its end (0.5) on every resume. No hard cap elsewhere. |
| **Attempt 1** (13:12 ET): decay 0.5, SS_RATE 1.0, SS 0.5 to 0.85 over 100M ids, LR 1e-4 cosine to 1e-5, warmup 500 | User spec | tf 13.7 at step 200, ema_tf 6.7 at step 800, 25 skips; automatic rollback at step 1000 (13:17 ET). Stopped. 7,491 ids/s. |
| LR=0 decay check (60 steps, no weight change) | Separate the decay effect from training | TF loss: decay 1.0 0.5-1.2; 0.99 0.8-1.3; 0.97 1.0-2.2; 0.9 2.2-3.3; 0.5 6.6-9.8 (uniform = 5.58). x0.5 breaks the trained model by itself. |
| **Attempt 2** (15:36 ET): decay 0.97, otherwise the same | User choice after the table | ema_tf 8.3 / 11.9 / 11.3 / 9.4 at steps 200/400/600/800, 30 skips. Stopped 15:41 ET before the rollback. |
| Isolation tests T1-T4 (17:33 ET, 300 steps each, scratch only) | Find the factor | At step 300: T4 (decay 1.0, SS 0.25, LR 2.5e-5) tf 0.63 / ema 0.78. T1 (1.0, SS 1.0, 1e-4) 7.39 / 9.46. T2 (1.0, SS 0.25, 1e-4) 2.56 / 2.70. T3 (0.97, SS 1.0, 2.5e-5) 4.34 / 4.06. **SS on every step is the main cause** (T1 is at tf 2.9 at step 50 with LR 1e-5). LR 1e-4 is a second cause. Decay 0.97 is not the main factor. 0 NaN. |
| **Launch v2 = T4 config** (17:45 ET) | Only stable configuration | `CARRY_DECAY=1.0 SS_RATE=0.25 SS_PROB 0.5 to 0.85 over 100M ids from 4,410,941,440 EPOCH_WRAP=1`, LR 2.5e-5 cosine to 2.5e-6, warmup 500, 2,153,780 steps (one pass), from `ckpt_full/good/` at start_token 275,683,840. First 2,000 steps: tf 0.65-1.85, ema_tf 1.36 / 1.47 / 1.16 / 1.10 / 1.12 at steps 200 / 600 / 1400 / 1800 / 2000, ss 1.8-4.6 at ss_p 0.50-0.51, 14 skipped steps, 0 non-finite, 11,384 ids/s wall (TF 82%, SS 18%). Stable, but above v1's final 0.332: the LR went back up 10x from 2.5e-6 and the data restarts at the lane heads. |

## Costs and credit
- Vast credit: $9.61 at 13:14 ET, $8.31 at 15:41 ET, $7.28 at 17:40 ET on Oct 17. At $0.525/h that is about 14 h.
- One v2 pass at ~11.3k ids/s is ~4.5 days, about $57 + storage. **The credit must be topped up** before then, or the run stops as it did on Oct 15.

## Oct 17: v2 generation tests

| Decision | Why | Effect |
|---|---|---|
| Generation test on v2 step_00035000 (19:45 ET; 4,482,621,440 ids, ema_tf 0.848, ss_p ~0.77). Weights only, scp read-only from the pod (md5 match), CPU bf16 via `scripts/ckpt_to_bf16raw.py` + memmap in `scripts/gen_block.py`; BOS + `encode_open`, 40 greedy ids, carry warm across blocks, reset only at BOS | Check whether stronger SS (ss_p 0.5 to 0.85) changes free-running output | unique@40 (France / Once upon a time / Hello): **A:128 14/16/15, mean 15.0/40 (0.375)**; B:128 14/16/16, mean 15.3/40; A:32 14/18/18, mean 16.7/40. Prompt TF from BOS (B) 71.5 / 119.0 / 170.0 nats/id. Word-like fragments ("from a", "a bo"), not coherent; same range as v1 step_770000 (0.30-0.425). 1m59s for 9 runs. Raw: `/workspace/gen_v2_35000.jsonl`, details in GENERATION_SURGERY.md. |
| Generation test at ss_p 0.85: v2 step_00050000 (first good ckpt after ss_p reached 0.85; 4,513,341,440 ids, ema_tf 0.934, promoted 20:15 ET). Same protocol | SS ramp end reached; user rule: if A:128 mean unique@40 < 20/40, raise SS_RATE | **A:128 18/17/14, mean 16.3/40 (0.408)**; A:32 16/13/19, mean 16.0/40; B:128 16/15/16, mean 15.7/40. Prompt TF (B) 46.8 / 121.5 / 128.0 nats/id. Still fragments, e.g. A:128 France `'tning me, akenwedalle <B1>wovans pa and pan'`. 16.3 < 20 -> rule applies. Raw: `/workspace/gen_v2_50000.jsonl`. |
| **SS_RATE 0.25 -> 0.35** (user rule "<20/40 -> SS on 10% more of the steps"); nothing else changed | Exposure bias remains the failure mode; more SS steps is the planned lever | Stopped runner/pusher/trainer (pids 10676/10677/10679) at 20:32 ET right after step_00055000 was promoted (4,523,581,440 ids, ema_tf 0.901). Moved `ckpt_v2/step_00050000` and `step_00055000` to `/workspace/ckpt_v2_pre_ss035/` (the step counter restarts at 0 and the trainer rotates step dirs by name). `run_v2.py` edited only `SS_RATE` (backup `run_v2.py.bak_ss025_20261018_0033`); `run_v2_state.json` set to ckpt_in `ckpt_v2_pre_ss035/step_00055000`, start_token 282,723,840, steps_left 2,098,780, lr_step_offset 55,000 (lr_total_steps 2,153,780, LR 2.5e-5 -> 2.5e-6, warmup 500 unchanged). Relaunched with `launch_v2.sh` at 20:33 ET (pids 12183/12184/12186). Optimizer state loaded; ids continue from 4,523,581,440; lr 9.99e-6 at step 200 (warmup), 2.496e-5 from step 600; ss_p 0.850; 0 non-finite. Throughput 11,376 -> ~10,700 ids/s, SS share of wall time 17% -> 22-23%. ema_tf 0.84-1.05 before vs 0.98-1.39 in the first 20 min after; 31 skips in the first ~6,800 steps (v1/v2 before: ~20 per 5,000). |
| Retest after SS_RATE 0.35: new-run step_00005000 (4,533,821,440 ids, 10.24M ids at SS_RATE 0.35, ema_tf 1.115, promoted 20:49 ET). Same protocol | Check the effect of the bump | **A:128 12/15/15, mean 14.0/40 (0.35)**; A:32 18/14/15, mean 15.7/40; B:128 13/15/16, mean 14.7/40. Prompt TF (B) 48.8 / 102.5 / 141.0. No improvement yet (within the ±0.15 run-to-run noise noted earlier); 10M ids is a short exposure. Raw: `/workspace/gen_v2_ss035_5000.jsonl`. |
| CONT-mode test (in-corpus continuation) on v2 new-run step_00005000, `reasoning__syllogisms.i16` docs 0/5000/10000: primer = BOS + 256 doc ids, 40 greedy ids vs the true next 40 (`scripts/gen_cont_syllog.py`) | Separate exposure bias from missing knowledge on an in-distribution document | TF on the true continuation 0.88 nats/id (C512), top-1 71%; free-running exact matches B:512 5.7/40, B:128 3.3, A:128 2.0. Correct for 1-6 ids, then derails. unique@40 B:512 19.0, B:128 17.3, A:128 16.3 (true text 19.7). Primer TF 4.3-4.5 nats/id, dominated by 20-128-nat spikes after BOS and at template slots. 8-14 nats at C=128 chunk starts. 6.8 min. Raw: `/workspace/gen_cont_syllog.jsonl`. |
| **SS_RATE 0.35 -> 0.50** (user ladder after 14.0/40 at 0.35); nothing else changed | Retest at 0.35 still < 20/40 | Stopped at 21:07 ET right after the 0.35-run step_00010000 was promoted (4,544,061,440 ids, ema_tf 1.294). That dir moved to `/workspace/ckpt_v2_pre_ss050/` as the only backup (sizes, inode = good/, manifest ids and zip CRC verified); then `ckpt_v2_pre_ss035/` and the older `ckpt_v2/step_00005000` deleted. State: start_token 284,003,840, lr_step_offset 65,000, steps_left 2,088,780. Relaunched 21:07 ET; optimizer loaded, ids continue, lr 2.495e-5 after warmup, ss_p 0.850, 0 non-finite. 10,640 -> ~9,750 ids/s (SS share 23% -> 29-30%); ema_tf 1.35-1.76 (0.35 run: 1.0-1.4); 24 skips in ~8,000 steps. Pod disk 41 GB free after cleanup. |
| Retest at SS_RATE 0.50: new-run step_00005000 (4,554,301,440 ids, 10.24M ids at 0.50, ema_tf 1.591, promoted 21:25 ET). Short prompts + CONT test (`gen_cont_syllog.py`, syllogisms docs 0/5000/10000, BOS + 256 doc ids, 40 greedy ids) | Ladder step | Short prompts unique@40: **A:128 14/13/17, mean 14.7/40**; A:32 14/17/17, mean 16.0; B:128 13/13/15, mean 13.7. CONT: **B:512 unique 15/15/19, mean 16.3/40; exact position matches 3/5/5, mean 4.3/40; matching prefix 0/4/0**; B:128 unique 14/17/19 (16.7), exact 3/5/3 (3.7), prefix 0/4/0. TF on the true continuation (C512) 2.72/1.26/1.35 nats/id, top-1 0.43/0.60/0.63, worse than the 0.35 baseline (0.88 nats/id, 71%; B:512 unique 19.0, exact 5.7). CONT B:512 16.3 < 20 -> next rung. Raw: `/workspace/gen_v2_ss050_5000.jsonl`, `/workspace/gen_cont_v2_ss050_5000.jsonl`. |
| **SS_RATE 0.50 -> 0.75** (rule: CONT B:512 unique@40 mean < 20) | Ladder step | Stopped at 21:44 ET right after the 0.50-run step_00010000 was promoted (4,564,541,440 ids, ema_tf 1.564); moved to `/workspace/ckpt_v2_pre_ss075/` (verified as above), `ckpt_v2_pre_ss050/` and `ckpt_v2/step_00005000` deleted. State: start_token 285,283,840, lr_step_offset 75,000, steps_left 2,078,780. Relaunched 21:44 ET; optimizer loaded, ids continue, lr 2.493e-5 after warmup, ss_p 0.850, 0 non-finite. ~9,720 -> ~8,330 ids/s (SS share 30% -> 39-40%); first 800 steps ema_tf 1.8-2.5. Pod disk 41 GB free. |
