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
- `scripts/train_atom_from_scratch.py` — the trainer. Refuses ids ≥ 266.
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

## If you are unsure

Stop. Say what you measured. Do not invent a last session, do not launch training from memory, and do not "simplify" the body to make a demo look like a transformer.
