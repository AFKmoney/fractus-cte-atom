# v4 final checkpoint: inference audit, fast exact decoder, decoding sweep (Oct 21, 2026)

Checkpoint: HF `checkpoints/atom_v4/step_final/fractus_atom_1b_gpu0.bin` (7,560,913,511 bytes, 1,864,649,906 params,
6,412,060,672 total ids, pod VAL 2.4483). Box CPU, fp32, memory-mapped. Inference code only; trainer untouched; no
bound on attn_S/attn_z. Data and scripts: `docs/gen_final_data/`.

## 1. Load check (teacher-forced, fixed 128-id grid, B=1)
Same 16 val windows as `scripts/v4_capacity_check.py` (127 context + 1,024 scored ids each):
median NLL **2.444** nats/id (all 16), mean 2.388 on the 12 non-pg19 windows, top-1 0.31, top-5 0.68.
step_285000 on the same windows: 2.447 mean. The strict load reproduces the trainer's function.

## 2. Step-by-step vs teacher-forced logits
- Re-forward path (`FixedGenerator.generate`: committed carry, open chunk re-forwarded): max |dlogit| <= 1.4e-4 at
  18 positions including 127/128/255/256; argmax identical everywhere.
- Tokenizer: `encode` = corpus ids on 117 val docs; `encode_open` = corpus prefix on 116 cut points.
- `atom_feat` weights are exactly 0 and the trainer never passes `feat`, so the feature path is a no-op.
- MoE capacity drops at B=1: 0 of 4,096 (token, expert) pairs on a 128-id val chunk.

## 3. Bugs fixed
1. `fractus/generate_aligned.py::generate_window(carry=True)` (used by the `fractus-cte-atom-demo` Space) fed every
   new id as its own 1-id chunk: thought_state was added at every id (training: once per 128-id chunk) and the
   attention state was averaged over levels at every id; prompt as one chunk of any length; cold BOS at position 0.
   Teacher-forced through that path: NLL **2,290 / 10,559 / 3,143** vs TF 2.796 / 2.696 / 1.755 (wiki / fineweb /
   tinystories, 448 ids each), argmax agreement 0.2-2.5 %. Now routed to the fixed path (64-id warm start, open prompt,
   exact incremental decoder). `carry=False` (used by the trainer's probe) unchanged.
2. Space `thefinalboss/Fractus-Space` loaded `checkpoints/atom_v3`; now `checkpoints/atom_v4/step_final`.
   Demo Space `fractus-cte-atom-demo` loaded `checkpoints/atom_latest`; now `atom_v4/step_final`.

## 4. Fast exact decoder: `FixedGenerator.generate_stream` (+ `stream_start`, `stream_push`, `pick_next`)
Every block op is per position (causal linear attention, Kuramoto, MoE), so the open chunk can be extended one id at
a time if it keeps its per-level attention state; the mean over levels (`_store_carry`) and the thought_state
injection happen only at a 128-id grid boundary, as in training. carry_window K re-uses the cold replay of the
last K chunks. Cost 0.2-0.25 s/id on 7 CPU threads (re-forward: up to ~1.3 s/id).
- Tiny engine (`tests/test_fixed_generation_stream.py`): logits = TF within 1e-3, generated ids = `generate()` for
  carry_window 0 and 2.
- 1B final, 448 ids x 3 docs: NLL 2.7967 / 2.6986 / 1.7627 vs TF 2.7961 / 2.6957 / 1.7555; argmax agreement 1.00 /
  0.95 / 0.96. The residual is discrete MoE top-k flips on float near-ties (von Mises gate on one angle), which then
  persist through the carry; not a logic difference.
- `pick_next`: masks PAD and BOS only (EOS only while len < min_new); repetition penalty (CTRL-style on the last
  rep_window generated ids), no-repeat n-gram, temperature, top-k, top-p, min-p. Boundary markers are ordinary ids;
  nothing is forced and generation does not stop on them.

## 5. Decoding sweep (CONT: 64-id real doc tail + BOS + 256-id doc prefix, 80 new ids, 12 docs = 4 fineweb, 4 wiki,
4 tinystories; sampling = 2 seeds). exact@40 = ids equal to the true continuation at the same position; real-word =
share of generated [A-Za-z]{2,} words in words_alpha; distinct-word = distinct / total generated words.

| config | n | unique@40 | exact@40 | real-word@40 | real-word@80 | real-word>=3 | distinct-word | 
| greedy_cw0 | 12 | 9.5 | 4.42 | 0.88 | 0.85 | 0.85 | 0.34 |
| oldspace_T0.7_k40_cold | 12 | 4.8 | 1.08 | 0.00 | 0.00 | 0.00 | 0.83 |
| greedy_cw2 | 12 | 9.5 | 4.42 | 0.88 | 0.89 | 0.88 | 0.34 |
| greedy_ng4_cw2 | 12 | 12.0 | 3.50 | 0.76 | 0.72 | 0.63 | 0.96 |
| T0.7_k40_cw0 | 24 | 17.5 | 3.25 | 0.41 | 0.46 | 0.37 | 0.97 |
| T0.7_p0.9_ng4_cw2 | 24 | 16.2 | 2.92 | 0.56 | 0.54 | 0.44 | 0.99 |
| minp0.1_T0.7_ng4_cw2 | 24 | 15.3 | 2.79 | 0.63 | 0.55 | 0.45 | 0.99 |
| T0.8_p0.9_rep1.15_cw2 | 24 | 38.6 | 1.17 | 0.20 | 0.21 | 0.11 | 0.99 |
| T0.6_k20_rep1.1_ng6_cw2 | 24 | 35.9 | 1.88 | 0.39 | 0.38 | 0.27 | 0.98 |
| T0.5_p0.8_ng8_cw2 | 24 | 14.2 | 3.29 | 0.70 | 0.70 | 0.62 | 0.78 |

`oldspace` = the old demo path (generate_window carry=True, T0.7 top-k 40, cold).

- carry_window 2 = full carry on greedy here (prefix only 320 ids); it helps beyond ~3 chunks (docs/LONG_CARRY.md).
- Repetition penalty on bytes destroys words (real-word 0.20 at 1.15): letters must repeat. Do not use it.
- **Default deterministic: greedy + no-repeat 4-gram + carry_window 2** (unique 12.0, real-word 0.76, distinct 0.96).
- **Default sampling: T 0.5, top-p 0.8, no-repeat 8-gram, carry_window 2** (real-word 0.70, distinct 0.78,
  exact 3.3/40), best real-word rate among the samplers.

### Samples (unedited, seed 0)
- greedy (before), fineweb_doc3: `then then the the <B1>and the the the pe the <B1>the pe the the the thee <B1>...`
- greedy + no-repeat 4 + cw2, wiki_doc18: `he the an than <B1>and this thous te f thune <B1>are tof tore forer th n`
  (true: `ed by the Milwaukee <B1>Brewers in the 33rd round <B1>of the`)
- T0.5 p0.8 ng8 cw2, tinystories_doc5: ` Tim was the <B1>band s tothe was to sto <B1>the barit t the the <B1>bis tand.`
  (true: ` "Sam, the battery <B1>is tight in the car. It <B1>will not fall out."`)
- T0.5 p0.8 ng8 cw2, fineweb_doc3: `stere wentof cerengend <B1>tor then thene the tere <B1>the the peneas td`

## Verdict
The inference path computes the trained function (load NLL 2.44, step-by-step = TF). The remaining gap to coherent
text is in the weights at 2.44 nats/id (~3.5 bits per id); decoding only trades loops for non-words. No inference
mismatch left on the main path.
