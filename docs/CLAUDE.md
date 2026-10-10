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
- `scripts/fast4gpu_boost_v4.py` — parent 1B loop. Not Atom-ready: slices a larger head to 266, refuses `.i16`, sets gate temp only.
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
- CPU smoke, measured 2026-10-09 on this fork: **1,729 Atom ids/s**. Config: 106,624 params, 2 blocks, `d_model=64`, B=1, seq 32, `v4_forward_losses` with backward, 20 steps after warmup. Last CE 5.05. This is the small body, not the 1B. Do not quote 1,729 as a 1B rate.
- `fast4gpu_boost_v4.py` cannot run that benchmark yet.
  - Default checkpoint is the parent (`fractus_1b_gpu{N}.pt` or `FRACTUS_1B_STAGE2_MERGED.pt`). It builds vocab 266, copies matching tensors, and slices any row whose first dim is larger. That cuts a 50257 head down to 266. It is not a from-scratch Atom init.
  - It accepts only `.npy` shards. `data/atom_corpus.i16` is refused.
  - Kuramoto: `moe.temperature = GATE_TEMP` only. It does not call `apply_kuramoto_routing_fix`, so omega ×4 is missing.
  - `START_TOKEN` defaults to 0. On this script that is a parent-resume footgun, not the legal fresh-model zero.

## If you are unsure

Stop. Say what you measured. Do not invent a last session, do not launch training from memory, and do not "simplify" the body to make a demo look like a transformer.
