# Long carry: why carry from beyond ~256 ids hurts v4, and inference-only fixes

Checkpoint: box copy of v4 step_285000 (609,726,464 v4 ids, `/workspace/ckpt_v4_285k`). Inference only, fp32 fixed
generation path, no weight change, no norm bound on attn_S/attn_z. Trainer and pod untouched.
Script: `scripts/v4_carryfix.py` (summary: `scripts/v4_carryfix_summary.py`). Data: `docs/long_carry_data/`.

## Problem
NLL on grid chunk 7 is 2.498 with the full carry and 2.343 with the carry reset one chunk earlier (`docs/V4_610M_EVAL.md`).
step_25000 (77M ids) still gained from the full carry (2.665 vs 2.715), so this regression appeared with more training.

## Mechanism (fractus/continuous_engine.py, fractus/nn/attention.py)
- Per block, the carry is `(attn_S, attn_z)` = running SUMS of phi(k) v^T and phi(k), phi = elu+1, mean over levels
  (`_store_carry`). There is no decay. Every chunk adds the full S0/z0 to all its positions, and the BOS reset is the only
  forgetting.
- Read: y_t = phi(q_t) S_t / phi(q_t) z_t. phi > 0, so this is a positive weighted average over EVERY id since BOS with
  no recency. The weight of the current chunk shrinks like 1/(number of ids seen).
- Measured (baseline, 8 windows): ||z|| grows linearly (block 15: 2.3e6 per chunk). The carry's share of the
  denominator matches c*128/(c*128+t) almost exactly:

### Carry diagnosis (baseline, 8 windows): block 15 / mean over blocks 1-15

| chunk | ||z|| block15 | ||S|| block15 | carry share of denominator (mean blocks 1-15) | at last position |
|---|---|---|---|---|
| 0 | 0 | 0 | 0.000 | 0.000 |
| 1 | 2.32e+06 | 3.45e+08 | 0.649 | 0.446 |
| 2 | 5.74e+06 | 8.25e+08 | 0.789 | 0.635 |
| 3 | 9.32e+06 | 1.34e+09 | 0.849 | 0.729 |
| 4 | 1.3e+07 | 1.9e+09 | 0.884 | 0.786 |
| 5 | 1.66e+07 | 2.47e+09 | 0.904 | 0.822 |
| 6 | 2.03e+07 | 3.03e+09 | 0.919 | 0.848 |
| 7 | 2.4e+07 | 3.62e+09 | 0.931 | 0.868 |
| 8 | 2.78e+07 | 4.22e+09 | 0.939 | 0.884 |

  At chunk 8, ~94 % of every attention read is the carried average. The current 128 ids, which carry most of the
  next-byte information, are diluted. Block 0 gives NaN for the denominator share (its |S| ~ 1e10-1e11, overflow in the
  probe only); its numerator share follows the same curve.
- Training does see long carries (lanes run continuously until a BOS), so the weights are not OOD here. But the model
  cannot shut out the old average, and the gain from the past falls below the cost of the dilution.
- Other recurrent state: `thought_state` (last hidden, added to position 0 of the next chunk) is a 1-chunk state.
  `kuramoto_phases` is stored but recomputed from h every chunk (`_encode_from_hidden`), so it does not accumulate.

## Inference-only fixes (TF NLL by chunk position)
- decayG: S, z *= G at each chunk boundary (joint, so the S/z ratio is kept).
- winK: committed state = cold replay of the last K chunks only (exact sliding window).
- resetN: cold every N chunks.

### TF NLL by grid chunk (12 non-pg19 windows = every 2nd of the 24; nats/id)

| policy | c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | mean c3-8 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 2.713 | 2.238 | 2.405 | 2.374 | 2.379 | 2.394 | 2.421 | 2.526 | 2.535 | 2.4382 |
| decay0.8 | 2.713 | 2.226 | 2.395 | 2.357 | 2.370 | 2.345 | 2.396 | 2.466 | 2.477 | 2.4018 |
| decay0.5 | 2.713 | 2.224 | 2.416 | 2.382 | 2.397 | 2.402 | 2.417 | 2.496 | 2.534 | 2.4381 |
| decay0.9 | 2.713 | 2.227 | 2.397 | 2.371 | 2.371 | 2.366 | 2.405 | 2.498 | 2.502 | 2.4186 |
| win2 | 2.713 | 2.238 | 2.405 | 2.357 | 2.329 | 2.313 | 2.339 | 2.398 | 2.437 | 2.3625 |
| decay0.95 | 2.713 | 2.238 | 2.398 | 2.368 | 2.376 | 2.379 | 2.411 | 2.523 | 2.512 | 2.4281 |
| decay0.7 | 2.713 | 2.223 | 2.388 | 2.357 | 2.371 | 2.356 | 2.389 | 2.455 | 2.482 | 2.4016 |
| reset2 | 2.713 | 2.238 | 2.549 | 2.291 | 2.401 | 2.276 | 2.382 | 2.397 | 2.782 | 2.4216 |

- Best is **win2**: -0.076 nats/id on chunks 3-8 (2.4382 -> 2.3625) and -0.128 on chunk 7 (2.526 -> 2.398).
- decay 0.7-0.8 gives -0.037. decay 0.5 and 0.95 give ~0.
- reset2 swings 2.276 / 2.782 depending on the reset position.
- Window cost: 2 extra chunk forwards per 128 generated ids.

## Generation (warm = 64 real corpus ids before BOS; 40 ids; CONT = 256-id prefix; LONG = 1,024-id prefix, so generation
runs in grid chunk 8; min-p only on LONG)



| policy | decoder | group | n | unique@40 | exact /40 | seqmatch | real-word rate |
|---|---|---|---|---|---|---|---|
| base | greedy | cont | 3 | 10.33 | 6.00 | 0.19 | 0.85 |
| base | greedy | long | 3 | 8.00 | 3.33 | 0.27 | 0.81 |
| base | greedy | prompt | 3 | 15.33 | - | - | 0.76 |
| base | minp_T0.7_p0.1 | long | 3 | 14.67 | 3.67 | 0.33 | 0.55 |
| decay0.8 | greedy | cont | 3 | 9.67 | 1.33 | 0.17 | 0.73 |
| decay0.8 | greedy | long | 3 | 7.00 | 4.00 | 0.28 | 1.00 |
| decay0.8 | greedy | prompt | 3 | 15.33 | - | - | 0.76 |
| decay0.8 | minp_T0.7_p0.1 | long | 3 | 15.67 | 1.67 | 0.29 | 0.50 |
| win2 | greedy | cont | 3 | 10.33 | 6.00 | 0.19 | 0.85 |
| win2 | greedy | long | 3 | 10.33 | 4.00 | 0.23 | 0.88 |
| win2 | greedy | prompt | 3 | 15.33 | - | - | 0.76 |
| win2 | minp_T0.7_p0.1 | long | 3 | 15.33 | 2.00 | 0.32 | 0.63 |

### Raw outputs (unedited, markers shown)

- **base / greedy / cont_fineweb_doc3** (u10, exact 5): `thend the the <B1>the the the Sof the <B1>the `  
  true: `Who Won!
<B3>is for all students. All <B1>stud`
- **base / greedy / cont_wiki_doc3** (u9, exact 6): `n the the <B1>the the the the the <B1>the Sof `  
  true: `s are forest <B1>and shrubland, but it has `
- **base / greedy / cont_tinystories_doc3** (u12, exact 7): `the bind the the <B1>the the cound the the `  
  true: `g, green thumb! <B1>It was so big, Bob had `
- **base / greedy / long_fineweb_doc1** (u8, exact 6): ` The The The <B1>ise ise te te the te <B1>the `  
  true: `it is likely <B1>that there has been a sig `
- **base / greedy / long_wiki_doc3** (u10, exact 2): `e <B1>ande an an ond ond ond oure <B1>an ofofo`  
  true: `ator Cay, the Bahamas <B1>Chliabothrus stri`
- **base / greedy / long_tinystories_doc22** (u6, exact 2): `e the the <B1>the the the the the he <B1>the w`  
  true: `an away.<B2>
<B3>Sara and Tom hugged their <B1>mo`
- **base / greedy / The capital of France is** (u13, exact -): `tion <B1>and and the whe <B1>sof the the she s`
- **base / greedy / Once upon a time** (u19, exact -): `, there <B1>was a little girl named <B1>Sucare`
- **base / greedy / Hello** (u14, exact -): `ur fore and <B1>ande the ale sof the <B1>sof t`
- **base / minp_T0.7_p0.1 / long_fineweb_doc1** (u14, exact 3): ` Thee In The cereace <B1>tator rero tane is`  
  true: `it is likely <B1>that there has been a sig `
- **base / minp_T0.7_p0.1 / long_wiki_doc3** (u14, exact 3): `anere <B1>oniof ond ane Suator d an tandest`  
  true: `ator Cay, the Bahamas <B1>Chliabothrus stri`
- **base / minp_T0.7_p0.1 / long_tinystories_doc22** (u16, exact 5): `anelo whenom. The <B1>he the her. Thed be t`  
  true: `an away.<B2>
<B3>Sara and Tom hugged their <B1>mo`
- **decay0.8 / greedy / cont_fineweb_doc3** (u10, exact 1): `thend the <B1>the the the Sof Sthe <B1>the Sth`  
  true: `Who Won!
<B3>is for all students. All <B1>stud`
- **decay0.8 / greedy / cont_wiki_doc3** (u8, exact 3): `n t the <B1>the the the the <B1>the the the So`  
  true: `s are forest <B1>and shrubland, but it has `
- **decay0.8 / greedy / cont_tinystories_doc3** (u11, exact 0): `t the the the <B1>cound the the sthe sto <B1>s`  
  true: `g, green thumb! <B1>It was so big, Bob had `
- **decay0.8 / greedy / long_fineweb_doc1** (u8, exact 7): ` The <B1>The The there pe <B1>the the the the `  
  true: `it is likely <B1>that there has been a sig `
- **decay0.8 / greedy / long_wiki_doc3** (u8, exact 3): `o o on of <B1>the of the the the <B1>the the t`  
  true: `ator Cay, the Bahamas <B1>Chliabothrus stri`
- **decay0.8 / greedy / long_tinystories_doc22** (u5, exact 2): `e the <B1>the the the the the the <B1>the the `  
  true: `an away.<B2>
<B3>Sara and Tom hugged their <B1>mo`
- **decay0.8 / greedy / The capital of France is** (u13, exact -): `tion <B1>and and the whe <B1>sof the the she s`
- **decay0.8 / greedy / Once upon a time** (u19, exact -): `, there <B1>was a little girl named <B1>Sucare`
- **decay0.8 / greedy / Hello** (u14, exact -): `ur fore and <B1>ande the ale sof the <B1>sof t`
- **decay0.8 / minp_T0.7_p0.1 / long_fineweb_doc1** (u16, exact 1): ` The Alle <B1>cle teale Sucto <B1>towo tre tit`  
  true: `it is likely <B1>that there has been a sig `
- **decay0.8 / minp_T0.7_p0.1 / long_wiki_doc3** (u14, exact 2): ` in thel <B1>ofin he in Sulelind wo hene <B1>t`  
  true: `ator Cay, the Bahamas <B1>Chliabothrus stri`
- **decay0.8 / minp_T0.7_p0.1 / long_tinystories_doc22** (u17, exact 2): `anere <B1>tono he beay. She. Hey wa <B1>d wast`  
  true: `an away.<B2>
<B3>Sara and Tom hugged their <B1>mo`
- **win2 / greedy / cont_fineweb_doc3** (u10, exact 5): `thend the the <B1>the the the Sof the <B1>the `  
  true: `Who Won!
<B3>is for all students. All <B1>stud`
- **win2 / greedy / cont_wiki_doc3** (u9, exact 6): `n the the <B1>the the the the the <B1>the Sof `  
  true: `s are forest <B1>and shrubland, but it has `
- **win2 / greedy / cont_tinystories_doc3** (u12, exact 7): `the bind the the <B1>the the cound the the `  
  true: `g, green thumb! <B1>It was so big, Bob had `
- **win2 / greedy / long_fineweb_doc1** (u10, exact 6): ` There <B1>The tome the the the <B1>the toure `  
  true: `it is likely <B1>that there has been a sig `
- **win2 / greedy / long_wiki_doc3** (u12, exact 2): `of 1999999999999.<B2> An 19999999999, <B1>s 19`  
  true: `ator Cay, the Bahamas <B1>Chliabothrus stri`
- **win2 / greedy / long_tinystories_doc22** (u9, exact 4): `ethe the <B1>the wand the the the wathe <B1>th`  
  true: `an away.<B2>
<B3>Sara and Tom hugged their <B1>mo`
- **win2 / greedy / The capital of France is** (u13, exact -): `tion <B1>and and the whe <B1>sof the the she s`
- **win2 / greedy / Once upon a time** (u19, exact -): `, there <B1>was a little girl named <B1>Sucare`
- **win2 / greedy / Hello** (u14, exact -): `ur fore and <B1>ande the ale sof the <B1>sof t`
- **win2 / minp_T0.7_p0.1 / long_fineweb_doc1** (u14, exact 2): ` Thee <B1>Spromerere The toto <B1>re to sane t`  
  true: `it is likely <B1>that there has been a sig `
- **win2 / minp_T0.7_p0.1 / long_wiki_doc3** (u18, exact 1): `aneme <B1>Sonof Sortof Matherd Sche <B1>188 it`  
  true: `ator Cay, the Bahamas <B1>Chliabothrus stri`
- **win2 / minp_T0.7_p0.1 / long_tinystories_doc22** (u14, exact 3): `ay. They <B1>hey bimave hey. They wat <B1>we t`  
  true: `an away.<B2>
<B3>Sara and Tom hugged their <B1>mo`

Reading:
- Prompts and 256-id CONT fit within 1-2 chunks: win2 is identical to baseline there (by construction), and decay 0.8 is
  identical on prompts.
- decay 0.8 hurts short CONT (exact 6.00 -> 1.33) because it also forgets the warm/prompt chunk.
- On LONG contexts win2 gives greedy unique@40 8.0 -> 10.3, exact 3.33 -> 4.00, real-word rate 0.81 -> 0.88. min-p:
  exact 3.67 -> 2.00, real-word 0.55 -> 0.63.
- Generation is still not speech in any policy (the/wand/1999 loops). The carry fix improves prediction (TF), not
  free-running output.
- Equivalence: `FixedGenerator.generate(carry_window=2)` reproduces the experiment's win2 greedy output exactly, and
  `carry_window=0` reproduces the baseline (`docs/long_carry_data/eqwin.json`).

## What changed (defaults keep the trained function)
- `fractus/carry_policy.py`: `set_carry_decay(eng, g)` (default 1.0 = off). Every decoder.
- `fractus/fixed_generation.py`: `FixedGenerator(..., carry_decay=1.0)`, `generate(..., carry_window=0)`.
- Space `thefinalboss/Fractus-Space`: sliders "carry decay" (default 1.0) and "carry window" (default 0, greedy/top-k
  only).

## Trainer-side variant (written, NOT launched)
`scripts/draft_train_carry_decay.py` runs the unmodified v4 trainer in-process:
- `CARRY_MODE=fixed CARRY_GAMMA=g`: the existing `CARRY_DECAY` knob (`bound_carry()` scales S, z once per chunk).
- `CARRY_MODE=learned`: per-block, per-head gamma = sigmoid(carry_logit), init logit(g). It is applied where the chunk
  reads the carry, so gamma gets gradient. Old checkpoints load with the new keys filled from the init.

CPU smoke test: gradients reach carry_logit and old state dicts load. A learned decay lets training choose the forgetting
rate (e.g. g ~ 0.6-0.8, ~3-5 chunks effective memory) instead of the uniform average. It needs a GPU A/B before any
use, and the inference side must apply the same gamma.
