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
