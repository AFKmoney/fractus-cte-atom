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
- A memorized short phrase spoke. Prompt `abcdef `, output `Fractus 12345.`. Teacher-forced accuracy on that phrase was 1. This proves the path can emit a continuation it learned. It is not a 1B speech claim.
- Starter corpus: `data/atom_corpus.i16`, 1,059,743 Atom ids, two French files. Not the phase-2 3.44B stream.
- `scripts/pod_atom_from_scratch.sh` refuses to run unless `ATOM_POD=1` is set on a machine that already has the GPU. It does not open a pod.
- `scripts/train_1b_gpu.py` refuses a checkpoint whose vocab is not 266.

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

Chinchilla on active params: `20 × 119.6M ≈ 2.4B` Atom ids. Top-4 would roughly double the active expert cost and the data target. 2 of 128 is the parent router, not a new choice. 4B Atom ids train faster than 4B BPE tokens, because the head is 266 instead of 50257. They are less text, because an id is a byte.

## Train

```bash
pip install torch numpy
python scripts/build_atom_corpus.py --src data/raw --out data/atom_corpus
python scripts/train_atom_from_scratch.py --scale smoke --steps 40
python scripts/train_atom_from_scratch.py --scale 1b --corpus data/atom_corpus.i16
```

`--scale smoke` is the CPU check. `--scale 1b` is the parent shape with vocab 266. The checkpoint records `start_token: 0` and `parent: hf:thefinalboss/fractus-cte`. Phase-2 shards and `training_corpus.pt` are GPT-2 ids. The loader rejects any id `≥ 266`.

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
