---
license: mit
language:
  - en
  - fr
tags:
  - continuous-thought-engine
  - atomizer
  - byte-span
  - mixture-of-experts
  - kuramoto
  - from-scratch
library_name: pytorch
pipeline_tag: text-generation
---

# Fractus CTE-Atom

A fork of the Hugging Face Fractus Continuous Thought Engine. The only change to the brain is the tokenizer: GPT-2 BPE is gone, the Atomizer is the id stream. This is a new model, trained from zero. It is not a resume of the x8 run.

Engine source: [`thefinalboss/fractus-cte`](https://huggingface.co/thefinalboss/fractus-cte).
GitHub: [`AFKmoney/fractus-cte-atom`](https://github.com/AFKmoney/fractus-cte-atom).
Atomizer source: [`AFKmoney/atom-ai`](https://github.com/AFKmoney/atom-ai), vendored as `fractus/atomizer.py`.

No x8 checkpoint is in this repo. Do not load one.

## Philosophy

Three properties, and they are the product.

**It thinks continuously.** A tick advances a residual thought `h` through the block stack. Attention carry `(S, z)` and Kuramoto phase stay live across chunks. Output is a next-id distribution on that state. The Atomizer does not change this. It only changes what an id is.

**It trains without a finish line.** The checkpoint is a seed. This fork starts at token 0 because the head is new. After that, training only moves forward. Infinite means unbounded continuation, not a promise that a run never crashes.

**It opens quickly.** The parent paid about 128M parameters for a 50257-id head before a block ran. This head is 266 ids (`266 × 1280 ≈ 0.34M` tied). No merge table. No Hugging Face tokenizer runtime.

## GPU, measured 2026-10-09

One RTX 5090, 1B shape, chunked attention, `torch.compile`, seq 128, B=2:

- Trainable parameters: **985,074,354** (Siren copies frozen: 879,235,072)
- TF: **119 Atom ids/s**, mean step 2.155 s, `mem_peak` 23.0 GB
- B=4: OOM on the per-token `index_select` of the expert factors

The GPU does about 200 ms of real work per step. The wall is about 2 s. The card is idle. The next lever is a capacity dispatch that does not copy weights per token. Do not start the 2.4B-id run at 119 ids/s.

## Status, 2026-10-09

Fixes in the body, measured:

- Per-token MoE routing. The chunk is gated by each position's own θ, not the last position's. Causal leak on position 0 is **0** after the switch. It was 0.028 after 400 steps with the old expand.
- Learned phase encode. `_encode_from_hidden` uses `phase_proj` (`Linear(d_model, 1)`, weight std `c/√d_model`, default `c=1`). The old mean of a LayerNorm'd h was ~0, so every token hit the same experts. At `c=1`, phase std before the modulo is ~1.1 rad. Alive experts: 4–5/8 and 28–31/128 fresh, settling to 4/8 and 9–10/128 after 100 steps. Not the old collapse (2 alive), not a random hash.
- Speech path works. A small body trained on one repeated phrase with `reset_thought` every step, greedy PREFIX, no ban: prompt `abcdef `, output `Fractus 12345. abcdef Fracef FraceuFra`, `unique=19/40`. Without the reset the model memorizes the carry: **81% accuracy with the training carry, 3% after `reset_thought`**. Generation starts from zero, so it collapsed.
- The trainer cuts the corpus into B lanes. Row b always reads lane b, so its carry continues its own text. The old contiguous layout was wrong for B > 1. BOS reset is per row: a BOS in one lane does not wipe the others. `start_token_next` is the offset inside each lane; resume with the same `BATCH`.
- `fast4gpu_atom.py` is the 1B trainer. Fresh start, vocab 266, full Kuramoto fix, refuses a 50257 head, refuses ids outside 0–265, refuses `START_TOKEN=0` on resume.
- Corpus is 1,059,743 Atom ids. Enough to measure throughput. Not enough to speak. Chinchilla target on ~120M active params is ~2.4B ids.
- No GPU rate has been measured. The protocol is in `docs/CLAUDE.md`.

## What was done

The parent body is intact and on the Atom path.

- `FractusTokenizer` is the Atomizer. Vocab is 266. `gpt2_compatible()` returns Atom on purpose, so old call sites cannot silently build a 50257 head.
- `LazyStructuredSirenLinear` is the MoE expert, not a comment. Two modules per expert, on the dense and sparse forwards. After `grow_atom`, those modules are copied from the stacked factors, so the body that computes is the body that grew.
- Fractal linear attention is on the train forward: QKV, level offsets, ELU feature map, causal linear attention, softmax over `level_logits`, carry `(S, z)`.
- Kuramoto routing fix (gate temperature 2.5, omega ×4) runs at the start of `scripts/train_atom_from_scratch.py`.
- The train loop is `v4_forward_losses`: chunked CE, load-balance, anti-repeat, optional scheduled sampling. `--grow-at N` adds a layer and rebuilds the optimizer.
- Packet features (320-d) enter through `atom_feat`, zero-init, placed on the boundary id. A run that does not pass features is unchanged.
- Embedding init is `N(0, 0.02)`. Step-0 next-token CE measured **5.592**, echo fraction **0.0**. Target is `ln(266) ≈ 5.583`.
- Speech path: greedy PREFIX does not ban the previous id. A prompt is an open prefix (`encode_open`): no EOS, no closing boundary on the tail. A closed encode made the model think the text was finished.
- A memorized short phrase spoke. Prompt `abcdef `, output `Fractus 12345. abcdef Fracef FraceuFra`, `unique=19/40`. Teacher-forced accuracy on that phrase was 1 with `reset_thought` every step. This proves the path can emit a continuation it learned. It is not a 1B speech claim.
- Starter corpus: `data/atom_corpus.i16`, 1,059,743 Atom ids, two French files. Not the phase-2 3.44B stream.
- `scripts/pod_atom_from_scratch.sh` refuses to run unless `ATOM_POD=1` is set on a machine that already has the GPU. It does not open a pod.
- `scripts/train_1b_gpu.py` refuses a checkpoint whose vocab is not 266.

## Atomization

ATOM does not build a vocabulary. The Atomizer cuts a raw UTF-8 byte stream into spans on structural boundaries: whitespace, punctuation, newline, a switch between letters and digits, a max length of 32 bytes, or the end of the stream. Each span is a packet: the bytes, the boundary that closed it, and a 320-d feature vector. Two spans with the same letters can still differ, because the features include a rolling context sketch. The packet is an event, not a permanent vocab row.

Fractus cannot eat a packet. It ticks one integer. So this fork turns each packet into ids the existing head can embed and predict:

1. The payload is emitted as raw bytes, ids 0–255. No merge. `é` is the UTF-8 bytes of `é`, not a French token.
2. After the payload, one boundary id (259–265) records why the span closed.
3. A training document is wrapped with BOS (257) and EOS (258). A generation prompt is not. `encode_open` leaves the tail unflushed and does not write EOS, so the prompt is a prefix of the longer text the model trained on.

Decode throws away every id ≥ 256 and UTF-8-decodes what remains. Round-trip on the bytes is exact. There is no merge table and no GPT-2 file.

The 320-d features are not ids. They enter through `atom_feat`, a linear map into `d_model`, zero at init, added on the boundary position. A run that does not pass features is unchanged.

CPU smoke, measured 2026-10-09 on this fork with all fixes in (`fast4gpu_atom.py`, B=1, seq 32, 106,754 params, `SS_RATE=0.5`):

Compare TF ids/s, not wall. The B=1 smoke was wall 1,180 / TF 1,878 with `ids_ss/ids_tf = 0.60`. The B=2 smoke was wall 2,244 / TF 2,298 with SS off. The wall doubled because SS did not run. The real batch effect is TF 1,878 → 2,298, +22%. Keep `SS_RATE` fixed and record the ratio, or wall numbers are not comparable.

These are CPU numbers on the small body. They are not a 1B rate. The ~1,000/s parent figure is BPE tokens on a 5090, not Atom ids.

## The id stream

| ids | meaning |
|---|---|
| 0–255 | one raw UTF-8 byte |
| 256 | PAD |
| 257 | BOS |
| 258 | EOS |
| 259–265 | boundary marker after a span |

Boundary ids are `258 + Atomizer._BOUNDARY_CODES[name]`: whitespace 259, punctuation 260, newline 261, class_transition 262, max_span 263, eos-flush 264, generated 265.

Closed encode (training documents): `[BOS] + (payload bytes + boundary id) × packets + [EOS]`.
Open encode (generation prompts): `[BOS] + committed packets + tail bytes`. No EOS. No boundary on the unflushed tail.

Decode drops every id `≥ 256` and UTF-8-decodes the rest. Round-trip on the bytes is exact.

The alphabet is closed. It does not grow. Every UTF-8 string already fits in 0–255. What grows is the body: `d_model`, depth, experts, oscillators, rank. `grow_atom` refuses a vocab change.

## Body

Not a transformer. Residual thought `h`, then per block: fractal linear attention, Kuramoto RK4, phase-routed MoE (128 experts, top-2, low-rank Siren). Tied embed and head. Confidence and salience heads. Persistent memory, cognitive modes, and a knowledge-base retriever exist and are constructed together in `fractus/atom_session.py`.

1B shape: `d_model=1280`, 16 blocks, 20 heads, 128 experts, top-2, `expert_d_ff=2048`, `siren_rank=64`, `n_levels=2`. Counted without allocating the weights: about **0.99B** total, **119.6M** active per token. The old 1.05B figure included the 50257 head. Do not quote it for this fork.

Chinchilla on active params: `20 × 119.6M ≈ 2.4B` Atom ids, not BPE tokens. An id is a byte or a boundary. 2.4B ids is less text than 2.4B GPT-2 tokens. Top-4 would roughly double the active expert cost and the data target. 2 of 128 is the parent router, not a new choice. The old ~1,000/s figure is BPE tokens on the x8 run. It has not been remeasured in Atom ids.

## Train

```bash
pip install torch numpy
python scripts/build_atom_corpus.py --src data/raw --out data/atom_corpus
python scripts/train_atom_from_scratch.py --scale smoke --steps 40
python scripts/train_atom_from_scratch.py --scale 1b --corpus data/atom_corpus.i16
```

`scripts/fast4gpu_atom.py` is the Atom 1B trainer. Fresh engine, full Kuramoto fix, or a strict resume of a 266-row checkpoint. It reads `.i16` and `.npy` and refuses any id outside 0..265. It does not slice a parent head. `scripts/fast4gpu_boost_v4.py` is the parent loop and is not Atom-safe. A CPU smoke of the Atom trainer (106,624 params, B=1, seq 32, 20 steps) ran at **1,378 Atom ids/s** wall. That is not a 1B rate.

`START_TOKEN=0` is legal only on this new model.

## What to measure

Teacher-forced CE falling is not speech. The parent run already showed that: CE around 5–8, unique@40 still about 3.3.

1. **Tokenizer lock.** `decode(encode(text)) == text` on French, English, code, accents, newlines. Every train id in `0..265`. Boundary histogram must not be 100% `max_span`.
2. **Init CE, frozen weights.** Near `ln(266) ≈ 5.58`. Echo fraction near 0. A step-0 CE of tens means the logit scale is blown (the old `N(0,1)` embed did this: echo 1.0, CE ~58).
3. **Teacher-forced CE.** An optimisation trace. Not the gate.
4. **Speech gate.** Greedy PREFIX, no ban, no temperature, open prefix. Count unique ids over 40 generated steps, and unique bytes after dropping ids `≥ 256`. Do not declare speech off a falling CE.
5. **Prefix alignment.** `encode_open(prompt)` must equal the prefix of `encode(prompt + continuation)` up to the prompt bytes. If it does not, generation is off the manifold the model trained on.
6. **Load balance.** `last_lb_loss` and per-expert hits. Top-2 of 128 with a dead gate is the parent failure mode.

## What is not done

- No 1B run. No pod was opened.
- The starter corpus is 1.06M ids. That is about 0.04% of the 2.4B active-param target. A 1B on it will memorize the two files.
- The speech proof is a memorized 25-id phrase, not free language.
- `unique40_probe` now generates. A low unique count is still NO-GO.

## Small-model checks already run

| check | result |
|---|---|
| Init CE / echo | 5.592 / 0.0 |
| 80-step smoke, no reset | CE 5.56 → ~2.6, thought norm stayed live |
| Carry across two ticks | moved, and did not return to the previous state |
| Memorized phrase, open prefix, no ban | `abcdef ` → `Fractus 12345.` |
| Grow then Siren vs factors | identical |
| Pod script without `ATOM_POD=1` | exit 2 |
