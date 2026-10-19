# Copy head for Fractus CTE-Atom (prototype, box CPU only, 2026-10-19)

Status: **prototype, not trained on the pod, trainer untouched.** All runs on the box CPU, fp32, from a read-only
copy of v4 good `step_25000` (77,246,464 v4 ids, md5 9235ea05…). Raw results (every jsonl/log row, tagged by source file): `docs/COPY_HEAD_results.jsonl`.
**Short answer: the copy head does not help v4 yet.**
- Trained alone on cached v4 states, it lowers val NLL by ~0.05 nats/id. A control with the same parameters and **no
  lookup** (diagonal-only attention = a linear readout correction) gets the same gain or more. The repeat-vs-first-
  mention gap does not open.
- The plumbing is exact. A gate of 0 is bit-equal to v4. Incremental decoding with the K/V window matches
  teacher-forced rows to ≤1e-4 on logits of magnitude ~236.
- The head **can** copy. On pure repeats it reaches 99.9 % top-1 with 3-gram keys (47 % with 1-gram keys). With live
  init (gate=1) it learns an exact cross-chunk copy of 16 random letters inside random-letter filler: 96 % same-chunk,
  96 % cross-chunk.
- With the gate at exactly 0 it **never** learned that task (≤5 % at 1,500 steps). A "ghost" straight-through
  gradient did not fix this.
- Mixed with real text, no arm learned the synthetic copy within the CPU budget (800-3,000 steps). That includes a tiny
  Fractus from scratch, the head alone, and gate 1. In the tiny Fractus, the 3-gram live head improved text NLL a lot
  (2.21 vs 2.35 nats/id at step 600, 2.165 vs 2.329 at 800).

## 1. Why

Fractus mixes positions only through **linear** attention (elu+1 feature maps, carried S/z state; the softmax is only
over levels). The v4 copy probe (DEEP_DIVE_GENERATION.md §10) shows almost no copying. Rest-of-word NLL is 2.96 nats/id
at a word's first mention, 2.86 when the word was already seen in the same 128-id chunk, and 2.93 when it was seen in
an earlier chunk. Hypothesis: linear attention keeps a dH×dH summary per head and cannot look up one exact past
position. A small exact softmax window should add that lookup.

## 2. Design (`fractus/nn/copy_head.py`)

- **Placement: readout.** `h_body` is the output of block 16, unchanged. `thought_state` and every block carry still
  come from `h_body`. The logits become `output_head(h_body + gate ⊙ W_o·attn(h_body))`. The body never sees the head.
  That gives two properties:
  1. A zero gate gives back v4 exactly.
  2. The head can be trained exactly on cached body states. That is what made the 1B test cheap on CPU.
- **Window.** Exact causal softmax over the current 128-id chunk plus the cached K/V of the previous `mem`=128 ids,
  with a validity mask. The window has a fixed size and never grows.
- **Shifted (induction) keys.** The key at j is built from x_{j-1..j-g} and the value from x_j. The query at t is built
  from x_t..x_{t-g+1}, where x = LN(h_body). A match "the context before j looks like now" returns what followed it.
  `ngram`=g (1 or 3) sets how many positions must agree.
- **Gate.** A per-channel vector, initialized to 0, so the head is exactly unchanged at init (§4.1).
  `GATE_INIT`/`COPY_GATE_INIT` overrides it (§4.3).
- **State per row.** K/V cache, mask, and the last g normed states. Everything is detached between chunks, like the
  carry. `reset_rows(mask)` clears rows at a BOS, as the trainer does.
- **Generation.** `FixedGenerator` handles `eng.copy_head` natively. The snapshot gains a third element (the window),
  `cold()` resets it, and `chunk()` applies the head. `gen_best.Decoder.stack/row` no longer hard-code `range(3)`.
  Checkpoints with `copy_head.*` keys load through `load_checkpoint`, which reads `config["copy_head"]`. Checkpoints
  without the head behave as before.
- **Size.** 8 heads × 64: 2.63 M params at ngram 1, 5.25 M at ngram 3 (+0.14–0.28 % of 1.89 B).

## 3. Scripts

| file | what |
|---|---|
| `fractus/nn/copy_head.py` | `CopyHead`, `attach_copy_head`, `load_with_copy_head` (strict on old keys) |
| `scripts/copyhead_1b_prep.py` | gate-0 equivalence on v4; caches `h_body` for 32 val windows + 256×512 train ids (train_006) |
| `scripts/copyhead_1b_train.py` | head-only fine-tune on cached states; val NLL per source + copy probe |
| `scripts/copyhead_equiv.py`, `copyhead_equiv_tiny.py` | equivalence and incremental/batched decoding checks |
| `scripts/copyhead_tiny.py` | tiny Fractus from scratch, base vs head, synthetic copy / recall + TinyStories |
| `scripts/copyhead_mech_check.py` | the head alone (embedding → head → linear) on the same copy rows |
| `scripts/fast4gpu_atom_v5_copyhead.py` | **draft** v5 trainer (v4 + head). NOT launched |

## 4. Results

### 4.1 Equivalence (gate = 0, fp32)
| check (v4 step_25000, fp32, CPU) | result |
|---|---|
| fixed-grid TF logits, 300 ids wiki val: engine vs engine (noise floor) | bit-equal (max \|Δ\| 0) |
| same, + copy head gate 0 (native `FixedGenerator`) | **bit-equal** (max \|Δ\| 0) |
| greedy 16 ids / beam-4 12 ids via `gen_best.Decoder` (batched snapshots), gate 0 | identical ids |
| trainer path `tick_chunk_train_ce`, 8 chunks of val doc, gate 0 | identical CE (max \|Δ\| 0.0) |
| gate random N(0, 0.5): logits change by | 0.93 (so the head is live) |
| gate random: open-chunk incremental decode (snapshot incl. window) vs TF rows | max \|Δ\| 1.05e-4 (logits up to 236) |
| gate random: batched decoder row vs single row / decoder vs TF | 2.9e-5 / 2.3e-5 |
| small random engine, ngram 1 and 3: all of the above + `reset_rows` | bit-equal at gate 0 in 10 of 10 reruns at normal load, incremental ≤3.6e-7, reset ok |

Two runs out of ~15 were not bit-equal at gate 0, and both ran while the box CPU was heavily oversubscribed (load
13-17). One was the first 1B harness, with a subclass generator: max \|Δ\| 1.5e-4 at logit scale 236, about 6e-7
relative. The other was the small engine: 3.9e-7. A same-engine rerun then gave 0.0, and every rerun at normal load
was bit-equal. `h + 0·y` is exact for finite y, so I attribute these runs to the threaded CPU kernels. I did not
prove it.

### 4.2 Head-only fine-tune on the real v4 checkpoint (test 3)
Setup:
- The body is frozen. Only the head and its gate are trained. AdamW lr 1e-3, clip 1.0, batches of 8 × 512 ids (4 grid
  chunks, window carried across chunks), 600 steps.
- Body states are cached exactly: 29 train batches = 119k ids of `atom_corpus_v4/train_006`, plus 3 held-out batches.
- Eval: the same 32 val windows as `scripts/v4_eval.py` (127 warm + 1,024 scored ids, 4 sources). The gate-0 rows
  reproduce the §10 v4 numbers, for example fineweb mean 2.743 and copy first 2.957 / same 2.842 / earlier 2.930.
- "val" = mean over the 30 windows < 6 nats. Two pg19 windows sit at 13-15 nats in every model and are excluded.
- Copy probe = rest-of-word NLL (bytes 2..n of words of 5 or more letters): first mention (9,791 bytes), repeat in the
  same chunk (440), repeat seen only in an earlier chunk (1,947). gain = first − repeat.

| arm (600 steps) | params | held-out train NLL | val | fineweb | wiki | tinystories | first | same (gain) | earlier (gain) |
|---|---|---|---|---|---|---|---|---|---|
| v4 (gate 0, step 0) | — | 2.936 | 2.683 | 2.743 | 2.830 | 2.507 | 2.957 | 2.842 (0.114) | 2.930 (0.027) |
| head, 1-gram keys | 2.63 M | 2.807 | 2.634 | 2.702 | 2.775 | 2.455 | 2.888 | 2.797 (0.091) | 2.836 (0.052) |
| head, 3-gram keys | 5.25 M | 2.809 | 2.638 | 2.710 | 2.771 | 2.465 | 2.883 | 2.799 (0.083) | 2.830 (0.053) |
| head, 3-gram, gate init 1 | 5.25 M | 2.850 | 2.702 | 2.756 | 2.858 | 2.534 | 2.905 | 2.770 (0.135) | 2.843 (0.062) |
| **control: no lookup** (diag-only, 1-gram) | 2.63 M | **2.753** | **2.630** | 2.704 | 2.756 | 2.464 | 2.874 | 2.752 (0.123) | 2.834 (0.041) |

Reading:
- Every arm gains ~0.05 nats/id, and the control with no lookup gains as much or more. So the gain is a readout
  correction of `h_body`, not copying.
- The same-chunk gap stays at 0.08-0.13 and the earlier-chunk gap at 0.04-0.06, against 0.11 / 0.03 at init. The
  repeat bytes improve about as much as the first-mention bytes. The +0.02-0.03 on the earlier-chunk gap is within
  the noise of 1,947 bytes. The longer 1-gram run (same seed, stopped at 300 steps) followed the same curve.
- Gate init 1 on the 1B makes things worse (val 2.70). This is honest evidence that, at this data scale, the head adds
  noise more than lookups.
- Limits: 119k training ids and 600 steps. The head was trained on a frozen body whose states were never asked to be
  matchable. Whether exact recall would emerge with joint training at GPU scale is **not** answered here.

### 4.3 Can the head learn to copy at all? (tests 2 + mechanism)
Tiny Fractus, same engine code: d_model 128, 4 layers, 4 × 32 heads, 2 levels, 8 experts, 1.6 M params. Trained
from scratch on CPU, AdamW 2e-3, batch of 16 rows × 256 ids = 2 grid chunks. Rows:
- copy_same: 16 random letters `|S|`, and again `|S` 40-60 ids later in the same chunk;
- copy_cross: the first copy in chunk 1, the second in chunk 2 (130-190 ids later);
- ar: 16 pairs `KQ=ab;`, then 4 queries;
- TinyStories text.

Score = top-1 on the 16 letters of the second copy and on the 2 value letters per query. Chance ≈ 0.04.

| run | steps | copy_same | copy_cross | AR | TinyStories val NLL |
|---|---|---|---|---|---|
| tiny base, mix 6/4/6 | 800 | 0.022 | 0.036 | 0.016 | 2.242 |
| tiny + head 1-gram, gate 0, mix 6/4/6 | 800 | 0.020 | 0.021 | 0.018 | 2.250 |
| tiny base, mix 6/6/4 (synth-heavy) | 600 / 800 | 0.020 / 0.012 | 0.020 / 0.015 | 0.014 / 0.023 | 2.348 / 2.329 |
| tiny + head 1-gram, gate 0, synth-heavy (stopped) | 650 | 0.023 | 0.028 | 0.012 | 2.297 |
| tiny + head 3-gram, gate 0, synth-heavy (stopped) | 250 | 0.014 | 0.019 | 0.039 | 2.467 |
| **tiny + head 3-gram, gate 1, synth-heavy** | 600 / 800 | 0.018 / 0.026 | 0.026 / 0.020 | 0.043 / 0.033 | **2.210 / 2.165** |

Neither arm copies within the budget. With a live 3-gram head, text NLL improves by 0.14-0.16 nats at the same step (600 and 800).

Mechanism checks: the head alone, embedding (std 0.1) → CopyHead (2 × 32) → linear readout, same rows.

| setting | result |
|---|---|
| pure repeats (period 32, every position copyable), 1-gram keys | 47 % top-1. Single-letter keys are ambiguous, so this is the expected ceiling |
| pure repeats, **3-gram keys** | **99.9 %** at step 300 |
| copy rows in TinyStories text, 1-gram / 3-gram, gate 0, 3,000 steps | 2.7 % / 2.4 % (chance) |
| copy rows in random-letter filler, 3-gram, gate 0, 1,500 steps | ≤5 % (gate decays 0.07 → 0.03) |
| same, 3-gram, gate 0 + ghost gradient (value-0 straight-through, as if gate=1) | ≤5 % |
| same, 3-gram, **gate init 1** | **94 % / 94 % at 750, 96 % / 96 % at 1,000-1,250** (same / cross chunk), then collapse to chance at 1,500 (lr 3e-3, no clip, no schedule) |

So the window, mask, cross-chunk cache and shifted keys work: 96 % exact recall across a chunk boundary. But under a
sparse copy signal, a zero gate starves q/k of gradient and the lookup never forms. Learning shows a phase transition,
from chance at 500 steps to 94 % at 750.

## 5. GPU cost estimate (not measured on GPU)
Active multiply-accumulates per id, from the checkpoint shapes (the MoE uses 2 of 128 experts with rank-64 factors):
- body ≈ 16 × (w_qkv 4.9 M + w_out 1.6 M + 2 experts 0.85 M + linear-attention state ≈ 0.3 M) ≈ **123 M**;
- tied head: 0.34 M;
- copy head 8 × 64, 1-gram: projections 2.6 M + attention (≤128 + 128 keys) × 512 × 2 ≈ 0.2 M = **2.9 M (2.3 %)**;
- copy head, 3-gram: **5.4 M (4.4 %)**.

In the trainer the body is block-checkpointed (forward twice + backward) and the head is not, so the head's share of
step FLOPs is ~1.7 % (1-gram) or ~3.3 % (3-gram). The CPU timings agree in magnitude:
- body forward of 8 × 512 ids: ~65 s on 4 threads;
- head forward + backward + readout of the same batch: ~1 s.

Estimate on the pod (5090, 10.5k ids/s wall, BATCH 16, SEQ 128):
- **−2 to −4 % ids/s** (≈ 10.1-10.3k);
- plus ~15 uncompiled small kernels per chunk (≲0.5 ms of a ~150 ms TF step);
- **memory +<0.1 GB**: attention maps 16 × 8 × 128 × 256 fp32 ≈ 17 MB, 2.6-5.3 M params + AdamW ≈ 30-60 MB. Peak is
  29.3 / 32.6 GB now.

On the 1.1 B plan this is a few hours of extra wall time at most. These are estimates, not measurements.

Draft trainer `scripts/fast4gpu_atom_v5_copyhead.py`: a new file, generated from v4 by targeted edits, with v4
otherwise byte-for-byte.
- It strict-loads v4 weights; only `copy_head.*` may be missing.
- The v4 `.opt` loads into group 0, and the head is group 1 with its own peak `COPY_LR`.
- `reset_rows`, `snapshot_carry` and `restore_carry` include the window, so the SS pass and val never see the TF
  chunk's K/V.
- It adds `COPY_ONLY=1` (freeze the body), `COPY_NGRAM` and `COPY_GATE_INIT`.

CPU smoke tests at SCALE=smoke:
- v4 → v5 step-0 VAL 3.5202 = v4 VAL 3.5202 (exact at gate 0);
- v5 → v5 resume: VAL identical, and both optimizer groups reload;
- 20 steps COPY_ONLY: 3.52 → 3.45;
- control (v4 continuing): 3.52 → 3.59.

**Not launched.**

## 6. v4 val curve and extrapolation to the end of the plan (1.1 B v4 ids)
Pod `ckpt_v4/val_log.jsonl`, read-only ssh at 04:28 ET: 41 VAL points, 262,144 held-out ids each, every 2,500 steps.
- Since the rollback (31 M v4 ids), VAL went 2.879 → **2.572 nats/id at 200.1 M v4 ids** (step 85,000). The trainer
  ran at 10.5k ids/s wall, lr 1.09e-5 (cosine, 537,109 steps, plan 1.1 B v4 ids).
- Slope: −0.127 nats per 100 M ids over 31-200 M; −0.074 over 100-200 M; −0.082 over 150-200 M. The curve is
  flattening.

Extrapolation to 1.1 B v4 ids:

| fit (31-200 M) | VAL at 0.5 B | VAL at 1.1 B |
|---|---|---|
| linear (do not trust: no saturation) | — | 1.40 (1.84-1.91 from the later windows) |
| log-linear | — | 2.35 (−0.09 per doubling) |
| power law a + b·(n/1e8)^−c, best fit a = 2.447, b = 0.206, c = 0.57, rmse 0.016 | 2.53 | **2.50** |

Expect **~2.35-2.50 nats/id** (3.4-3.6 bits/id) at the end of the plan. The cosine LR decay may buy a little more at
the end. The power law says most of the remaining gain is already in.

## 7. Recommendation
1. **Do not add the head to the running v4 run.** On v4 it gives no copy gain beyond a linear readout correction.
2. If exact recall is still wanted, the recipe these tests support is:
   - 3-gram shifted keys;
   - a **live gate** (gate ≈ 1), not 0;
   - first pre-train the head with the body frozen (`COPY_ONLY=1`) on data with dense repetition, so the lookup's
     phase transition happens;
   - then train jointly with lr warmup and clipping (the CPU run collapsed after the transition without them).

   This gives up "exactly v4 at step 0", but the body weights are untouched and the head can be removed by zeroing the
   gate. It would need a GPU A/B against v4 on the copy probe before use.
3. The bigger lever for "does it speak" is still data and steps (§6): ~0.1-0.2 nats/id more is expected from the
   current run.
