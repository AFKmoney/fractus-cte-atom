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

A fork of the Hugging Face Fractus Continuous Thought Engine, with ATOM's Atomizer in place of GPT-2 BPE. This repository is a **new model**, trained from zero. It is not a resume of the x8 run.

Engine source: [`thefinalboss/fractus-cte`](https://huggingface.co/thefinalboss/fractus-cte).
GitHub: [`AFKmoney/fractus-cte-atom`](https://github.com/AFKmoney/fractus-cte-atom).
ATOM Atomizer source: [`AFKmoney/atom-ai`](https://github.com/AFKmoney/atom-ai) `src/io/atomizer.py`, vendored here as `fractus/atomizer.py`.

No x8 checkpoint is in this repo. Do not load one.

## What this fork is

Fractus CTE is not a transformer. It keeps a residual thought state `h`, advances it tick by tick through a stack of blocks (linear attention carry `(S, z)`, Kuramoto phase clock, phase-routed MoE), and emits a next-id distribution from a head tied to the input embedding.

The Hugging Face engine is the one that was actually trained. Its public tokenizer is GPT-2 byte-level BPE, vocab **50257**. That choice dominates the parameter count: the tied embed/head is about `2 × 50257 × 1280` parameters, on the order of 128M, before the blocks.

ATOM does not have a vocabulary. `Atomizer` (`atomizer-v1-byte-span`) cuts a raw UTF-8 byte stream into reversible spans on structural boundaries (whitespace, punctuation, newline, alpha/digit transition, max span, end of stream). Each span is an `AtomPacket`: payload bytes, boundary name, phase, and a 320-d feature vector. Two equal surface spans can still differ in features, because features include a rolling context sketch. The packet is an event, not a permanent vocab row.

Fractus cannot consume that packet directly. `observe` is `nn.Embedding(vocab, d)` and `output_head` is tied to it, so every tick must be one integer in `[0, vocab)`. This fork adapts the Atomizer **to that contract**. It does not turn Fractus into the toroidal field, and it does not turn ATOM into a transformer.

## The id stream

`fractus/atom_tokenizer.py` (`atom-fractus-v1`):

| ids | meaning |
|---|---|
| 0–255 | one raw UTF-8 byte from the packet payload |
| 256 | PAD |
| 257 | BOS |
| 258 | EOS |
| 259–265 | boundary marker after each span |

Boundary ids are `258 + Atomizer._BOUNDARY_CODES[name]`:

| id | boundary |
|---|---|
| 259 | whitespace |
| 260 | punctuation |
| 261 | newline |
| 262 | class_transition |
| 263 | max_span |
| 264 | eos (flush of a pending span) |
| 265 | generated (not used on the train path) |

Default pack: `linguistic`, `max_span_bytes=32`. Encode is `[BOS] + (payload bytes + boundary id) × packets + [EOS]`. Decode drops every id `≥ 256` and UTF-8-decodes the surviving bytes. Round-trip is exact on the bytes, including French and code, as long as the Atomizer does not split a code point (it does not, except the explicit byte-tick mode `max_span_bytes=1`).

There is no learned merge table and no GPT-2 file.

The 320-d packet features are **not** in the id stream. They stay on the packet. Putting them into the tick would be a second input channel (`tick_vec`), which this fork does not do. The boundary id is the part of the Atomizer decision the CE head can see.

## What was deliberately not done

- The x8 run was not resumed, remerged, or rewritten. `START_TOKEN=0` is legal **only** on this new model.
- Phase-2 shards (`shard_phase2_gpu*.npy`) and `training_corpus.pt` were not reused. Those files are GPT-2 ids. An id `≥ 266` is rejected by the scratch loader.
- No 1B training run was launched. A CPU smoke (`--scale smoke`, 68,656 params, a few steps) only checked that the graph runs. Its cross-entropy sits far above `ln(266) ≈ 5.58` because the tiny net is randomly initialised and untrained. That number is not a speech result.
- Old train scripts under `scripts/train_1b_*.py` still mention vocab 50257. They belong to the parent engine. The scratch entry point is `scripts/train_atom_from_scratch.py`.

## Train from zero

Retokenize raw text only (`.txt`, `.jsonl`, `.jsonl.gz`):

```bash
pip install torch numpy
python scripts/build_atom_corpus.py --src data/raw --out data/atom_corpus
python scripts/train_atom_from_scratch.py --scale 1b --corpus data/atom_corpus.i16
```

`--scale smoke` is the CPU path check. `--scale 1b` is the parent 1B shape (`d_model=1280`, 16 blocks, 20 heads, 128 experts, top-2, `expert_d_ff=2048`) with `vocab_size=266`. The checkpoint records `tokenizer.version`, `start_token: 0`, and `parent: hf:thefinalboss/fractus-cte`.

Because the head shrank from 50257 to 266, a fresh 1B is no longer ~1.05B parameters. The tied embed/head drops from ~128M to `266 × 1280 ≈ 0.34M`. The block stack is unchanged. Count parameters from the constructed module. Do not quote the old 1.05B figure for this fork.

Chinchilla on the parent used **active** parameters (~119M/token), not `20 × 1.05B`. Recompute it after the first real init: active params × 20 is the sparse-MoE token target. The old 2.4–3.7B token band was for the GPT-2 head. It is not a target to copy blindly, and it is not a reason to train on the old shards.

## What to measure

Teacher-forced CE falling is not speech. The parent run already showed that. Measure the following, in this order.

1. **Tokenizer lock, before any GPU spend.** Round-trip on French, English, code, and a string with accents and newlines: `decode(encode(text)) == text`. Every train id must sit in `0..265`. Boundary histogram should not be 100% `max_span` (that means the packer is just cutting at 32 bytes and the linguistic mode is unused). Report bytes per packet and packets per 1,000 characters against a GPT-2 encode of the same sample, so the fertility change is a number.

2. **Init CE, step 0, frozen weights.** On a random batch, mean CE should be near `ln(266) ≈ 5.58` nats if the head is uninformed. A step-0 CE of tens means the logit scale is blown up (init, residual, or a shape bug), not that the data is hard. Log CE and load-balance separately. Do not add them into one number and call it CE.

3. **Teacher-forced CE on the Atom stream.** Useful only as an optimisation trace. It is not the speech gate.

4. **Speech gate.** Greedy PREFIX, no ban, no temperature. Generate from a short prompt, decode with the Atom tokenizer (drop ids `≥ 256`), and count unique characters over the first 40 generated **bytes** (the parent gate was unique@40 on BPE tokens; on this fork the unit is the byte, so report both unique bytes @40 and unique packets @40). The parent NO-GO was unique@32 in the 12–27 range and not English. Do not declare speech off a falling CE.

5. **Boundary fidelity under generation.** The fraction of generated ids that are boundary markers, and whether a boundary id is ever emitted in the middle of a UTF-8 code point. A decode that needs `errors="replace"` on the model's own output is a fail even if CE is low.

6. **Expert load balance.** `last_lb_loss` and per-expert hit counts. Top-2 of 128 with a dead gate is the same failure mode as the parent. The Kuramoto routing fix (`apply_kuramoto_routing_fix`, gate temperature 2.5, omega scale ×4) is a parent-run load fix. It is not applied by the scratch script. Decide explicitly before a long run; do not inherit it by accident.

7. **Parameter and token accounting.** Print `sum(p.numel())` and an active-param estimate (embed + norms + attention + the top-2 experts, not all 128). Quote the token target off that count. Keep a `start_token` in the checkpoint. On this fork it starts at 0 and then only moves forward.

## Layout

| path | role |
|---|---|
| `fractus/continuous_engine.py` | HF CTE stack, unchanged contract |
| `fractus/atomizer.py` | vendored ATOM byte-span atomizer |
| `fractus/atom_tokenizer.py` | packet → id stream, vocab 266 |
| `fractus/tokenizer.py` | parent GPT-2 wrapper, unused by the scratch path |
| `scripts/build_atom_corpus.py` | raw text → int16 memmap |
| `scripts/train_atom_from_scratch.py` | fresh init only |
| `docs/` | parent training notes, historical, not a resume protocol for this fork |

## License

MIT. Engine and notes from Fractus CTE. Atomizer from atom-ai.
