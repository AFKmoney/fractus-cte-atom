# Brief for Claude — Fractus CTE-Atom

Read this before touching the repo. The owner is Philippe-Antoine. The operator he trusts on this run is the Grok session that built the fork. You are being handed a finished adaptation, not a blank project.

## What this is

Fractus CTE-Atom is a fork of `thefinalboss/fractus-cte`. The brain is the Continuous Thought Engine. The tokenizer is the Atomizer from `AFKmoney/atom-ai`. Vocab is 266. It trains from zero.

It is not a transformer. It is not ATOM's toroidal field. It is not a resume of the x8 run.

GitHub: `AFKmoney/fractus-cte-atom`.
Hugging Face: `thefinalboss/fractus-cte-atom`.
Parent engine: `thefinalboss/fractus-cte`.

## How to see it

Three properties are the product.

1. It thinks continuously. A tick advances residual thought `h`. Attention carry `(S, z)` and Kuramoto phase stay live. Do not turn a tick into a one-shot prompt-to-completion function.
2. It trains without a finish line. A checkpoint is a seed. This fork starts at token 0 only because the head is new. After that, never rewind.
3. It opens quickly. The head is 266 ids, about 0.34M tied parameters at `d_model=1280`. Do not put a 50k BPE head back.

The alphabet is closed on purpose. 0–255 are UTF-8 bytes, 256–258 are PAD/BOS/EOS, 259–265 are Atomizer boundary markers. Growing the vocab orphans the tokenizer. Grow the body: width, depth, experts, oscillators, rank. `grow_atom` locks vocab at 266.

## What is already plugged

Do not re-implement these. Read them.

- `fractus/tokenizer.py` — `FractusTokenizer` is the Atomizer. `gpt2_compatible()` returns Atom.
- `fractus/atom_tokenizer.py` — closed encode for documents, `encode_open` for generation prompts.
- `fractus/continuous_engine.py` — CTE. Default vocab 266. Embed init `N(0, 0.02)`. `atom_feat` is the 320-d packet channel, zero-init.
- `fractus/nn/attention.py` — `FractalLinearAttention`, on the train forward, with carry.
- `fractus/nn/moe.py` — phase-routed MoE. `LazyStructuredSirenLinear` is the expert module.
- `fractus/nn/lazy_siren.py` — the low-rank body. Not a dead file.
- `fractus/grow.py` — `grow_atom` copies stacked factors and syncs the Siren modules.
- `fractus/kuramoto_fix.py` — gate temp 2.5, omega ×4. Applied by the scratch trainer.
- `fractus/train/v4_step.py` — the train step: chunked CE, load-balance, anti-repeat, scheduled sampling.
- `fractus/generate_aligned.py` — speech gate. Greedy PREFIX, no ban, no temperature, open prefix.
- `fractus/atom_session.py` — one constructor for engine, routing fix, memory, modes, RAG, grow.
- `scripts/train_atom_from_scratch.py` — CPU smoke and the from-zero entry. B=1, resets thought every step. Not the 1B throughput reference.
- `scripts/fast4gpu_atom.py` — the Atom 1B trainer. Fresh start with the full Kuramoto fix, or a strict resume of a 266-row checkpoint only. Reads `.i16` and `.npy`, refuses any id outside 0..265, never slices a 50257 head. `START_TOKEN=0` only when there is no `CKPT_IN`. Reports Atom ids/s split into TF, SS, and probe.
- `scripts/pod_atom_from_scratch.sh` — does not open a pod. Refuses unless `ATOM_POD=1` on a box that already has the GPU.
- `data/atom_corpus.i16` — 1,059,743 Atom ids. A starter. Not the phase-2 stream.

## What you must not do

- Do not load, merge, or resume an x8 / GPT-2 checkpoint. Those weights are vocab 50257. They do not map onto 266.
- Do not set `START_TOKEN=0` on any existing run. It is legal only for this new model.
- Do not ingest phase-2 shards or `training_corpus.pt`. They are GPT-2 ids.
- Do not put GPT-2 BPE back, and do not call the Atomizer a "small vocab we should grow."
- Do not replace fractal attention, Kuramoto, the Siren experts, or the tied head with a normal transformer block.
- Do not treat a falling teacher-forced CE as speech. The gate is greedy PREFIX unique@40, no ban, no temperature.
- Do not ban the previous id in generation. These ids are bytes. A ban forces `ntntnt` and forbids double letters, spaces, and UTF-8 runs.
- Do not close a generation prompt with EOS. Use `encode_open`. A closed prompt makes the model think the text is finished.
- Do not open a pod, spend money, or launch a 1B run unless Philippe says so in the current thread.
- Do not delete the parent x8 artifacts, and do not rewrite `fractus/` into a simpler model.
- Do not describe Fractus as "won't speak" as a verdict. Speech is an empirical gate on the current run.

## What you may do

- Read the code and the README.
- Run CPU smokes: tokenizer round-trip, init CE near `ln(266) ≈ 5.58`, a few `v4` steps, `unique40_probe`.
- Build a larger Atom corpus from raw text with `scripts/build_atom_corpus.py`. Never from GPT-2 `.pt` or `.npy` shards.
- Fix a measured bug, with the measurement in the same change.
- Extend data. The 1.06M starter is not enough for a 1B to speak. Active-param Chinchilla is about 2.4B Atom ids.

## How speech actually failed, and what fixed the path

Two bugs, both measured.

1. `generate_aligned._pick` banned the previous id on every step. On BPE that hides a mono-token lock. On bytes it makes speech impossible. The ban is off. The gate is greedy with no ban.
2. Generation encoded the prompt as a finished document: closing boundary plus EOS. Training had learned the continuation without that EOS. `encode_open` emits the committed packets plus the unflushed tail, no EOS. `encode_open(prompt)` must match the prefix of the closed encode of the longer text.

After those two fixes, a small model that had memorized `abcdef Fractus 12345. ` continued `abcdef ` into `Fractus 12345.`. That is the path working. It is not a claim about a 1B.

## Numbers worth keeping

- Vocab 266. Init CE 5.592, echo 0.0, after embed `std=0.02`.
- 1B shape: ~0.99B total, ~119.6M active/token, top-2 of 128.
- Data target on active params: ~2.4B Atom ids. Have 1.06M.
- Parent speech gate, still the gate here: greedy PREFIX unique@40. On the x8 run, CE 5–8 coexisted with unique@40 around 3.3. NO-GO.

## Throughput

The parent number is tokens. This fork counts Atom ids. Do not mix them.

- The ~1,000/s figure is from the x8 run: GPT-2 BPE tokens, vocab 50257, `fast4gpu_boost_v4.py`, B=4, seq 128, chunked attention, RTX 5090. About 990 tok/s/GPU on the 27–28 Aug SS night, ~670 at B=2, ~840–940 on the QuickPod resume. B=5 was refused, OOM risk. That is a token measurement. It is not an Atom measurement.
- An Atom id is a byte or a boundary marker. One BPE token is several bytes. 2.4B Atom ids is less text than 2.4B BPE tokens. The training loop still counts positions, and a position is now an Atom id.
- No 5090 has been timed on this fork. Do not write "1,000 Atom ids/s" as if it had been measured. The body (attention, Kuramoto, MoE) costs per position either way. The head is the part that changed: 266 instead of 50257, so CE and the tied embed are cheaper. That can make a step faster. It does not make the old token rate the new id rate.
- If, and only if, a later benchmark shows ~1,000 Atom ids/s/GPU, then 2.4B ids is ~28 days on 1 GPU and ~7 on 4. Until that benchmark exists, those days are a transposition, not a plan.
- 1,000 positions/s on ~120M active params is ~0.7 TFLOPS, under 1% of a 5090 FP32 peak. Same fact as "the step is the bottleneck." A half-day run is a peak-FLOP ceiling (6 × 2.4B × 120M ≈ 1.7×10¹⁸ ops). No measurement on this body supports it.
- Before optimizing a kernel, measure where memory goes (activations per component, state kept for backward, MoE buffers) and how many Atom ids each expert sees per step. Top-2 of 128 at B=4 means tiny matmuls. If the benchmark confirms that, the lever is freeing memory to raise the batch, not a faster kernel.
- CPU smoke of `fast4gpu_atom.py`, 2026-10-09: scale smoke, 106,624 params, B=1, seq 32, 20 steps, corpus `atom_corpus.i16`. Wall **1,378 Atom ids/s**, TF **1,903 ids/s**, SS **2,090 ids/s**. Device CPU. Not a 1B rate. Guards passed: GPT-2 ids refused, 50257 head refused and not sliced, resume does not re-apply omega ×4, `START_TOKEN=0` refused on resume.
- Time on one GPU is `ids to see / ids per second`. Nothing else. For the 1B shape that is ~2.4B Atom ids. These rows are arithmetic, not measurements:

  | ids/s | days on 1 GPU |
  |---|---|
  | 1,000 | ~28 |
  | 5,000 | ~6 |
  | 15,000 | ~2 |
  | 30,000 | ~1 |

  15,000 ids/s on ~120M active params is ~10 TFLOPS (6 FLOPs × params × ids/s). A small fraction of a 5090 peak, so not absurd for the card. It is still unmeasured on this body. Do not pick a row and call it the plan.
- Two levers, both unmeasured on a 5090: step speed, and model size. A body with 4× fewer active params needs ~4× fewer ids (Chinchilla on active params) and each step is cheaper. Wall time can fall like the square of the size. That is a reason to prove speech on a small body before paying for the 1B. It is not a measured speedup.
- An Atom id is a byte or a boundary. The same text is about 3–4× more positions than BPE. The 266 head is cheaper. The extra positions have to be paid for by a faster step.
- First things to time on one GPU, before any kernel rewrite: the SS pass (it doubles the work when `SS_RATE=1`), the per-step `.item()` expert-hit loop in `tick_chunk_train` (one sync per expert), `torch.compile`, and the memory that capped the parent at B=4. The number that matters is Atom ids/s from `fast4gpu_atom.py` on one 5090.

## Kuramoto step count

Train and generate use the same integrator. That part is clean. The block builds `KuramotoLayer(..., n_steps=1, dt=0.1)`. `_rk4_integrate` only loops when `n_steps > 1`, so the model takes one RK4 step. The class default of 4, and the unrolled branches for 2–4, are dead at runtime. They are only hit by the equivalence test.

Measured on a fresh smoke body, one forward, no weight change (`scripts/measure_kuramoto_r.py`). r tracks total time `T = n_steps × dt`, not the step count, while RK4 still holds:

| pair | r | theta MAE |
|---|---|---|
| 1×0.2 vs 2×0.1 | 0.022 / 0.022 | 0.01° |
| 1×0.4 vs 4×0.1 | 0.046 / 0.046 | 0.04° |
| 1×2.0 vs 4×0.5 | 0.128 / 0.242 | 22° |

At T=0.4, one step matches four. At dt=2.0, RK4 is no longer the same trajectory. So a larger T can be one step, until the single-step error shows up.

The model does not read r. It reads MoE gates. On this fresh smoke (8 experts, top-2), alive experts stayed 2/8 and normalized dispatch entropy stayed ~0.51 at every T above. r moved. The gate did not. Do not treat a higher r as more living experts until a small train measures both. Do not change the default `n_steps=1, dt=0.1` until that train exists. Fixing T before the first real run is a setting. Changing it after is retuning a clock the weights were trained on.

## Chunk routing

Fixed. `_tick_chunk_core_pure` now passes `phases_in = theta`. Each position is gated by its own phase. Generation already did this (`theta[:, 0:1]`). The old expand of `theta[:, -1]` is gone.

Measured on a smoke body (64-d, 8 experts, top-2, Kuramoto fix on):

- Last-theta routing: after 400 steps the leak is real. Changing only the last id moved position-0 logits by **0.028**. Theta spread between pos 0 and pos last was 0.024.
- Per-token routing: the same probe stays at **0.000** at every step from 0 to 400. Last-position logits still move (the token changed). Earlier positions do not.
- Throughput on the same CPU smoke, 40 steps after warmup: last-theta **2,257 ids/s**, per-token **2,088 ids/s**. About 7% slower on this tiny body. Not a 1B number.
- Alive experts stayed 2/8 either way on this short run. The switch removes the leak. It does not by itself wake more experts.

Do not put the expand back. It is a causal leak once phases depend on position, and it disagrees with generation.

## Expert hits

The `_expert_hits` counter in `tick_chunk_train` is not the real count. It calls `_compute_gates` on `kuramoto_phases[:, 0:1, :]` of block 0 only. One position, one block. `tick_chunk_train_ce` (the teacher-forced path) does not count hits at all. With top-2 and B=1, that counter is structurally about 2.

The real count is `topk_idx` inside the MoE forward, over every position and every block. Measured on a smoke body, per-token routing, all positions:

| setup | θ̄ std (rad) | spacing 2π/E | ratio | alive / block |
|---|---|---|---|---|
| E=8, fresh | 0.007–0.017 | 0.785 | 0.01–0.02 | 2/8 |
| E=8, 200 steps | 0.005–0.023 | 0.785 | 0.01–0.03 | 2/8 |
| E=128, fresh | 0.004–0.006 | 0.049 | 0.08–0.12 | 2–3/128 |
| E=128, 200 steps | 0.005–0.012 | 0.049 | 0.10–0.25 | 2–3/128 |

Across both blocks, 4–5 experts alive in total. The counter was not the whole story. The real routing is also narrow.

θ̄ is the only thing the gate reads (`atan2` of the summed sin/cos). Its std across positions is far below the expert spacing. Every token lands on the same two experts. Raising r does not fix this: r is how synchronized the oscillators are, not how far θ̄ moves from token to token. At 128 experts the spacing is 2.8°, and the ratio is still under 1.

Fixed, then recalibrated. The first `phase_proj` used weight std 1. On a LayerNorm'd h that makes the phase std `√d_model` (8 rad at d=64, 36 rad at d=1280). After `remainder(2π)` that is a random hash: θ̄ std saturates at π/√3 ≈ 1.81, the uniform-on-a-circle value. Neighboring tokens land on unrelated experts.

The weight std is now `c / √d_model`, so the phase has std ~c rad before the modulo. Default `c=1` (`PHASE_PROJ_C`). Measured on the smoke, d=64:

| c | phase std before modulo | E=8 alive, fresh → 100 | E=128 alive, fresh → 100 | neigh / spacing (E=128) |
|---|---|---|---|---|
| 0.5 | 0.55 | 3 → 3–4 | 15–21 → 5–6 | ~1 |
| 1.0 | 1.10 | 4–5 → 4 | 28–31 → 9–10 | ~2–3 |
| 2.0 | 2.19 | 6–7 → 6 | 40–46 → 10–14 | ~2–5 |

`c=1` is the default. The phase std stays near c after 100 steps, so the optimizer is not just shrinking `w`. The alive count still falls at E=128: the model concentrates. That is not the old flat-phase collapse (2 alive), and it is not the random hash (45 alive, θ̄ std stuck at 1.81). The causal leak stays at 0.

## GPU benchmark protocol

The CPU smoke (2026-10-09, all fixes in) was **1,180 Atom ids/s wall**, TF **1,878**, SS **1,968**, B=1, seq 32, 106,754 params. `SS_RATE=0.5`. `ids_ss / ids_tf = 768/1280 = 0.60`. If SS ran every step, the wall could not exceed `1 / (1/1878 + 1/1968) ≈ 960` ids/s. The 1,180 matches SS on about 60% of steps. Always report the ratio. A wall number without it is ambiguous.

On the GPU box, in this order. Do not open a pod from here.

1. Guards first. `pytest` and the leak probe on the card. Position 0 must stay at 0 when only the last id changes.
2. Fix T before timing. Kuramoto cost depends on `n_steps`. If the target is one step at a larger dt, benchmark that, not the default `n_steps=1, dt=0.1`.
3. A small matrix on the 1B shape, from zero, about 100 steps each. The 1.06M corpus is enough to measure rate. Vary:
   - `BATCH` 4, 8, 16, until OOM
   - `BLOCK_CKPT` 0/1
   - `SS_RATE` 0 and 1, to isolate the SS cost
   - `COMPILE` 0/1
4. For each run record TF ids/s, SS ids/s, `ids_ss/ids_tf`, `mem_peak`, and probe share. Set `PROBE_EVERY=0` for a pure benchmark.

Days for 2.4B ids on one card = 2.4e9 / measured ids/s. If SS is much slower than TF on GPU, the first suspect is the `.item()` loop in `tick_chunk_train` (one sync per expert, 128 at the 1B shape). That counter only sees block 0, position 0. It is not the real hit count.

## If you are unsure

Stop. Say what you measured. Do not invent a last session, do not launch training from memory, and do not "simplify" the body to make a demo look like a transformer.
