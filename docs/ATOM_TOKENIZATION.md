# Atom tokenization on Fractus CTE

Source of the engine: Hugging Face `thefinalboss/fractus-cte` (not the stale GitHub snapshot, not the x8 weights).

The x8 run stays where it is. This tree is a fork for a **new** model. Its head is vocab 266. A 50257 checkpoint does not load into it.

## Why a new vocab

Fractus ticks one integer. `observe` is `nn.Embedding(vocab, d)` and `output_head` is tied to that table. ATOM's Atomizer does not assign ids: it emits reversible byte spans plus a boundary (`whitespace`, `punctuation`, `newline`, `class_transition`, `max_span`, `eos`).

Adaptation, not a second architecture:

| id | meaning |
|---|---|
| 0–255 | raw UTF-8 byte from the packet payload |
| 256 | PAD |
| 257 | BOS |
| 258 | EOS |
| 259–265 | Atomizer boundary marker after each span |

Decode drops ids ≥ 256 and rebuilds the string from the bytes. No BPE, no GPT-2, no learned merge table.

## Train from zero

1. Retokenize **raw** text. Do not reuse `training_corpus.pt` or `shard_phase2_*.npy` (those are GPT-2 ids).
2. `python scripts/train_atom_from_scratch.py --scale 1b --corpus data/atom_corpus.i16`
3. `START_TOKEN=0` is correct here only. It is not a resume of x8.

Smoke (CPU, tiny width) is `--scale smoke`. That is a path check, not the 1B run.
