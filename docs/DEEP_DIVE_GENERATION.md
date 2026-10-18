# Deep dive: why Fractus CTE-Atom learns (TF loss ~0.5) but does not talk

Date: 2026-10-18, ~09:00-11:00 ET. Fractus is an ACI (fractal linear attention with a carried state, Kuramoto phase routing, phase-routed MoE), not a transformer.
Checkpoints:
- v3 step_00010000 (4,656,701,440 ids, md5 88f7b917...)
- v3 step_00085000 (4,810,301,440 ids, ema_tf 0.571, md5 212d5927...). Copied read-only from the pod, md5 verified.

All runs are on box CPU. The trainer, runner and pusher on the pod were not touched.
Code:
- `fractus/fixed_generation.py`
- `scripts/gen_fixed.py`
- `scripts/gen_hunt.py`
- `scripts/deep_dive/*.py` (harness, exp1-exp6)

Raw data: `docs/deep_dive_data/`.

## 1. Short answer

**It is not one hidden inference bug.** The definitive test is greedy decoding with the exact training forward (fixed 128-id grid, committed carry, fp32). It gives the same gibberish as the gen scripts:
- doc 0: exact 5/40
- doc 5000: 1/40
- doc 10000: 1/40

Three things combine:

1. **The generation code had real bugs.** They are all fixed now (section 3).
   - gen_block mode A and the HF Space double-counted the carry.
   - The Space ran an old copy of the model code without `phase_proj`, with a `strict=False` load. It re-applied the Kuramoto fix (omega x4), appended `<B6><EOS>` to every prompt and stopped at the first boundary marker.
2. **Cold start is a state training never produces.** In training, BOS sits at a random chunk position, after the tail of the previous document (`reset_rows` zeroes the carry of the whole chunk). Generation put BOS at position 0 with nothing before it.
   - NLL on the first 32 ids after a cold BOS: **46-52 nats/id**.
   - Same ids with the exact training alignment: **~5**.
   - Short prompts live entirely in this regime. The warm start fixes most of it.
3. **The model itself: template memorizer, extreme overconfidence, chaotic forward.** This is the part the inference fixes cannot remove.
   - The TF top-1 of 0.85 on the CONT window is a template artifact. That window is the fixed sentence "Here is a treatment of **X** from a Y angle, formatted as a Z, written for a W". Deeper in the body top-1 is 0.0-0.3 in places, 0.70 on average over syllogism doc heads.
   - Out-of-corpus English: top-1 **0.13**, NLL **10.5 nats/id**, which is worse than uniform (ln 266 = 5.58).
   - Wrong predictions cost 5-150 nats. |logit| reaches 1709, and there is no final norm before the tied head.
   - A 1e-4 relative perturbation of the embeddings grows about 170x through the 16 blocks and flips 6.4% of top-1.
   - Temperature 0.7 / top-k 20 gives nearly the same text as greedy, because the distributions are one-hot.

## 2. Parity: training forward vs generation paths (step_10000, fp32, teacher-forced through each path)

| path (CONT docs 0/5000/10000, 40 ids) | max abs logit diff vs one-chunk training forward | top-1 agree | TF top-1 |
|---|---|---|---|
| B:512 (re-forward of the whole sequence) | **0.0 / 0.0 / 0.0** | 1.00 | 0.85 / 0.90 / 0.85 |
| B:128 (commit C/2, re-forward pending) | 21 / 18 / 18 (median 9-11) | 0.85-0.93 | 0.93 / 0.85 / 0.83 |
| A:128 (old sliding window on committed carry) | 57 / 71 / 67 (median 35-43) | 0.20-0.23 | **0.23 / 0.23 / 0.20** |
| fixed grid, open chunk re-forwarded (gen_fixed) vs one TF pass on the same grid | 0.0 | 1.00 | |

Why B:128 differs from the one-chunk forward: `thought_state` (the last hidden state of the previous chunk, norm ~2000) is added to h[:,0] of every chunk. Moving the chunk boundaries therefore changes the function. Training uses a 128 grid with arbitrary alignment to the text; gen_fixed uses the same grid.

Ruled out by measurement:
- **Causality / off-by-one.** Target = next id. In fp32, positions 0..257 are identical whether the forward is 297 or 258 ids long (max diff 1e-3, top-1 agreement 1.000).
- **MoE capacity drops (cf=4).** Kept fraction is 1.000 at B=1 and at B=16, with 104-128 experts used.
- **Batch row placement.** Only bf16 noise.
- **Pod vs repo model code.** Identical (stats.py differs only in CRLF).
- **Tokenization.** Re-encoding decoded docs reproduces 20/20 docs exactly; the prompt encoding (`encode_open`) is the same rule.
- **A "0.97 bound on attn_S/attn_z".** It does not exist anywhere. The pod trainer env has CARRY_DECAY=1.0, CARRY_CLAMP=0, REPEAT_COEF=0, FRACTUS_ATTN_IMPL=chunked, COMPILE=1.

bf16 vs fp32: in pure bf16 (what gen_block / gen_cont used) the same position differs by up to 30-100 logits between mathematically equal paths (top-1 agreement 0.80-0.95). In fp32 the difference is 1e-3. Greedy CONT exact on step_10000, cold:
- bf16: 2 / 3 / 1
- fp32: 5 / 1 / 1

## 3. Cold start vs training alignment (step_10000 bf16, TF NLL nats/id; first 32 ids after BOS)

| ids of the previous doc before BOS (k) | doc 0 | doc 5000 | doc 10000 |
|---|---|---|---|
| 0 (generation) | 52.4 | 7.9 | 46.4 |
| 8 | 35.6 | 2.9 | 24.1 |
| 32 | 15.0 | 1.9 | 10.9 |
| 64 | 8.4 | 1.4 | 5.7 |
| exact training alignment (111 / 119 / 118) | 5.1 | 1.6 | 5.3 |

On step_85000 (fp32, fixed path, first 64 ids, mean of the 3 docs):

| warm variant | NLL first 64 ids | CONT TF NLL | CONT top-1 |
|---|---|---|---|
| W0 (cold BOS) | 40.3 | 1.07 | 0.78 |
| W8 | 27.0 | 0.73 | 0.84 |
| W32 | 10.8 | 0.88 | 0.82 |
| W64 | 6.8 | 0.71 | 0.81 |
| **W127** | **3.4** | 1.03 | 0.78 |
| previous chunk with hot carry, BOS at chunk position 0 | 4.0 | 1.03 | 0.78 |
| previous chunk with hot carry plus 64-id tail before BOS (not training semantics) | 7.1 | 2.03 | 0.54 |

## 4. Template vs generalization (step_10000 fp32, warm 64)

Per-class breakdown (first 768 ids of 16 syllogism docs and 8 negotiation docs, plus held-out text I wrote):

| set | NLL | top-1 | notes |
|---|---|---|---|
| syllogism doc heads | 1.98 | 0.70 | 99.6% of 16-grams occur in >=5 other docs (the corpus is extremely templated) |
| negotiation, 16-grams also present in syllogism docs (46%) | 0.52 | 0.88 | |
| negotiation, novel 16-grams (54%) | 2.97 | 0.51 | |
| corpus-style header, new topic (photosynthesis) | 6.83 | 0.31 | |
| plain English paragraph | **10.51** | **0.13** | max-prob still 0.82: confidently wrong |

By id class in the syllogism docs:

| class | NLL | top-1 |
|---|---|---|
| in-word bytes | 1.93 | 0.72 |
| word-initial | 2.72 | 0.63 |
| boundary markers | 1.80 | 0.58 |
| doubled letters | 3.11 | 0.57 |

When wrong, the median NLL is 5.3 and p99 is 28.8.

Corruption sensitivity: one wrong input id, averaged over 5 docs x 4 positions, top-1 on the following targets:

| window after the corrupted id | top-1 clean | corrupted with the model's own top-2 | corrupted with a random letter |
|---|---|---|---|
| 1-4 | 0.61 | 0.41 | 0.45 |
| 5-8 | 0.65 | 0.56 | 0.65 |
| 9-16 | 0.63 | 0.61 | 0.62 |
| 33-64 | 0.60 | 0.58 | 0.60 |

Errors do not compound much: the model recovers within about 8 ids. The problem is not exposure bias; base accuracy outside templates is too low.

## 5. Fixes (inference only)

- `fractus/fixed_generation.py`:
  - trainer-identical load (`strict=True` after resetting the batch-sized carry buffers, gate temperature from the config, no second Kuramoto fix);
  - fixed 128 grid with committed carry, open chunk re-forwarded;
  - `WARM_TAIL` (real corpus doc tail ending `<B6><EOS>`);
  - `encode_open`;
  - stop only at EOS;
  - optional `final_norm` ('rms' = per-position rescale, argmax-preserving; 'ln' = LayerNorm without affine).
- `scripts/gen_fixed.py`: variant evaluation.
- `scripts/gen_block.py` and `scripts/gen_cont_syllog.py`: mode A is now a pure sliding window from zero carry (`A_LEGACY=1` restores the old one).
- HF Space `thefinalboss/Fractus-Space` (copy in `space/Fractus-Space/`):
  - model code synced with this repo;
  - the new loader and generator, loading `checkpoints/atom_v3/fractus_atom_1b_gpu0.bin`;
  - UI controls for warm length, temperature / top-k and final norm.

Final norm (step_85000, warm 64, CONT TF):

| final norm | NLL | top-1 | mean max logit |
|---|---|---|---|
| off | 0.71 | 0.81 | 12.8-22.6 |
| rms | 0.75 | 0.81 (argmax unchanged) | |
| ln | **3.77** | 0.81 | 0.3-0.5 |

LN without affine collapses the logits to near-uniform, which destroys sampling. A final norm only makes sense if it is trained.

## 6. Retest of the variants, step_85000 fp32 (CONT docs 0/5000/10000, 40 greedy ids)

| variant | unique@40 (mean) | exact-position (mean) | sequence-match ratio |
|---|---|---|---|
| a fixes only (cold BOS) | 17.7 | 3.7 | 0.375 |
| b fixes + warm 64 | 15.7 | 2.3 | 0.408 |
| c fixes + warm 64 + LN | 16.7 | 2.3 | 0.433 |

Raw CONT outputs:

| variant | doc | generated | true |
|---|---|---|---|
| a | 0 | ` re <B1>written a profeprancting <B1>whaas ncer` | ` step-<B2>by-step procedure, written <B1>for a` |
| a | 5000 | `thetetrun and atheound <B1>abores asigheve ` | `ndations <B1>angle, formatted as a Q&<B2>A dia` |
| a | 10000 | `crical appfofesticel angle, astrma w-<B2>by` | `oretical foundations <B1>angle, formatted a` |
| b | 0 | ` cenario <B1>walkthrough, writter <B1>ache wr ` | (same as above) |
| b | 5000 | `thean athe-tr <B1>athe de che coretive numu` | |
| b | 10000 | `oracatical found applicatication <B1>aplion` | |
| c | 10000 | `oracatical found applicatication <B1>angle,` | |

Short prompts, greedy:

| variant | prompt | output |
|---|---|---|
| a | The capital of France is | `ovewew <B1>t aulythe om akiutyre ouof alin ` |
| b | The capital of France is | `toretive.<B2> a commun ohimas and <B1>corectio` |
| a | Once upon a time | `raction <B1>mind, makes the ste abstracctic` |
| b | Once upon a time | `aing <B1>ulexaith ference depstin <B1>deractas` |
| a | Hello | `ds.<B6>r montai hicton to a <B1>lthe mo cabt a` |
| b | Hello | ` a oreisthit ericth l eveluston <B1>orith l` |

The full list, including T=0.7 / k=20, is in `docs/deep_dive_data/gen_fixed_85000.jsonl`.

## 7. Recommendation

- **Inference.** Always use the fixed path (`fractus/fixed_generation.py`): fp32, warm start (W64-W127), never mode A. This removes every measured train/inference mismatch. It does not make the model talk.
- **Training** (not done here; trainer untouched). The bottleneck is what the model has learned:
  - a synthetic corpus where 99.6% of 16-grams are shared templates;
  - logit scale in the hundreds to thousands, with no final norm;
  - no generalization off-template (worse than uniform on plain English).

  Candidate changes, in order:
  1. a trained final norm, or a logit cap / z-loss;
  2. real, diverse text in the corpus;
  3. doc-aligned BOS resets with occasional cold starts, so the cold state is seen;
  4. a validation set of held-out non-template text, tracked alongside ema_tf (ema_tf ~0.5 hides the problem).


## 8. Hunt for the missing piece (step_85000, fp32, inference only; `scripts/gen_hunt.py`, raw data in `docs/deep_dive_data/gen_hunt_85000.jsonl`)

Ideas checked:

- **(2) Appending positions as in training vs re-forwarding the open chunk.** Fixed-grid re-forward vs one TF pass: max abs logit diff 2e-4, top-1 agreement 1.000. They are identical.
- **(3) thought_state injection at chunk boundaries.** It is applied at the same 128-grid boundaries in both paths (the parity above). A different grid (B:128 at C/2) changes logits by up to ~21 (section 2), so the grid must match training. gen_fixed already matches it.
- **(5) bf16 vs fp32.** The difference is real but second-order: bf16 changes 5-20% of top-1 through the chaotic forward. Greedy CONT exact on step_10000: bf16 2/3/1 vs fp32 5/1/1. Use fp32.
- **(1) Template vs non-template TF.** See section 4. Template 16-grams reach top-1 0.88; novel 16-grams 0.51; plain English 0.13. This is the dominant effect.

Variants, ranked by CONT exact (mean over docs 0/5000/10000, 40 ids):

| variant | unique@40 | exact /40 | seqmatch | prompt unique@40 |
|---|---|---|---|---|
| **W127 greedy** | **19.0** | **4.33** | **0.550** | 18.0 |
| W64 bytes only, markers never emitted (exact counted on ids, so marker positions count as misses) | 14.7 | 3.67 | 0.442 | 17.3 |
| W127 + tokenizer-driven markers (model's markers masked; encoder rule re-inserts them) | 16.0 | 3.00 | 0.439 | 17.7 |
| W64 greedy (previous default) | 15.7 | 2.33 | 0.408 | 17.0 |
| W64 + tokenizer-driven markers | 17.3 | 1.67 | 0.488 | 17.3 |
| W64 scale-free sampling (logits/std, T 0.8, top-p 0.9, presence 1.0 over last 32) | 35.3 | 0.67 | 0.142 | 27.7 (random bytes) |
| W64 tokenizer markers + scale-free sampling | 31.7 | 0.67 | 0.113 | 37.0 (random bytes) |

The ideas, ranked by effect:
1. **Warm length 127.** This is the training alignment where BOS sits at the last position of a chunk: +2.0 exact and +0.14 seqmatch vs W64. It is the only clear gain.
2. **Tokenizer-driven markers and bytes-only decoding.** Mixed, within noise (±1 exact). The model's own marker placement is not the problem.
3. **bf16 → fp32.** Small; use fp32.
4. **Re-forward vs append, and injection boundaries.** No effect: already exact.
5. **Repetition-aware scale-free sampling.** Harmful. Dividing by the logit std flattens the near one-hot distributions into random bytes. With the native scale, T 0.7 / k 20 is nearly greedy (section 6).

Raw outputs of the best variant (W127 greedy):

| doc / prompt | generated | true |
|---|---|---|
| doc 0 | ` e <B1>princiores writtten <B1>a aprofeson.\n<B3>*` | ` step-<B2>by-step procedure, written <B1>for a` |
| doc 5000 | `nandath-the-the-skil angl angle,<B2> format` | `ndations <B1>angle, formatted as a Q&<B2>A dia` |
| doc 10000 | `atorical devefoundatiopm angle, as a bef` | `oretical foundations <B1>angle, formatted a` |
| The capital of France is | `the <B1>udoreasign. a a ceprising <B1>the nale` | |
| Once upon a time | `astion. acth <B1>gits a commo ara <B1>pandese-` | |
| Hello | `d. amanisth de design thinking. <B1>from Wh` | |

W127 + tokenizer markers on "Once upon a time" produced `astion. <B1>a ste step-by-step procedure <B1>fro`: a corpus template phrase, not an answer to the prompt.

**Conclusion.** No remaining train/inference mismatch was found. The fixed path reproduces the training forward bit for bit (fp32), and the best alignment (W127) is now the Space default (Space commit 9f38825). What remains is the model:
- it reproduces corpus template fragments ("angle, formatted", "step-by-step procedure", "design thinking");
- it misspells off-template words;
- it ignores prompts outside the corpus distribution.

The missing piece is on the training/data side (section 7), not in the inference body.

## 9. Decoding strategies (step 5,025,341,440 ids, HF `checkpoints/atom_v3` @ 87c4d701; fp32, fixed path, warm 127; `scripts/gen_best.py`)

Run 2026-10-18, ~18:15-19:55 ET. There are 10 cases:
- CONT syllogism docs 0/5000/10000 (40 ids after the first 257);
- 3 short prompts;
- 3 corpus-template prompts ("Explain X from a practical application angle ... Here is a treatment of **X** ... **Definition.** ") for X = photosynthesis, the capital of France, a short story;
- an in-corpus control topic (negotiation).

Real-word rate is the share of generated alphabetic words (2+ letters) found in a 370k English word list. It is lenient.

Batched hypotheses (beam rows with stacked carries) agree with the single-row forward on argmax: True. The max abs logit difference is 5.1, the usual fp32/MoE chaos.

| decoder | CONT unique@40 | CONT exact /40 | CONT seqmatch | prompt real-word | template real-word |
|---|---|---|---|---|---|
| minp_T0.7_p0.1_ng4 | 17.7 | 5.33 | 0.575 | 0.20 | 0.57 |
| beam4 | 16.3 | 3.67 | 0.442 | 0.41 | 0.59 |
| beam8 | 17.0 | 4.33 | 0.333 | 0.36 | 0.68 |
| greedy | 16.0 | 4.00 | 0.408 | 0.36 | 0.51 |
| minp_T2_p0.05_ng4 | 18.0 | 4.67 | 0.433 | 0.47 | 0.30 |
| lookahead_K4_D8 | 17.7 | 4.00 | 0.367 | 0.36 | 0.51 |
| contrastive_k4_a0.3 | 17.7 | 4.00 | 0.367 | 0.36 | 0.51 |
| minp_T5_p0.05_ng4 | 19.0 | 3.00 | 0.258 | 0.10 | 0.14 |
| typical0.9_T5 | 21.0 | 3.33 | 0.317 | 0.07 | 0.00 |
| minp_T20_p0.05_ng4 | 30.7 | 1.33 | 0.133 | 0.16 | 0.23 |

Ranking: no decoder clearly wins. The top two:
- **min-p on raw logits (T 0.7, p_min 0.1, no-repeat 4-gram)** is best on CONT (exact 5.3, seqmatch 0.58). This is one seed on 3 docs, so it is noisy.
- **beam 8** (length-normalized) gives the most real words on prompts and templates (0.52 / 0.68).

The other decoders:
- Word look-ahead (K 4, D 8) and contrastive search (k 4, α 0.3) reproduce greedy almost exactly. The top-1 id dominates by tens to hundreds of nats, so alternatives never win.
- Temperatures of 5-20 on raw logits, and typical sampling, produce byte noise.

Raw outputs, min-p T0.7:

| case | generated | true |
|---|---|---|
| doc0 | ` <B1>worked thrke emple <B1>(citinntiten fors ` | ` step-<B2>by-step procedure, written <B1>for a` |
| doc5000 | `ndandationsengle,<B2> formatted as a a befo` | `ndations <B1>angle, formatted as a Q&<B2>A dia` |
| doc10000 | `aroricatical de-ticnion-fomasengl <B1>ated ` | `oretical foundations <B1>angle, formatted a` |
| The capital of France is | `omincel <B1>adelonte rsiom ancresiong <B1>al a` | |
| Once upon a time | `ation real/<B2> decisin siontomerme <B1>anat a` | |
| Hello | `w d mplyincit mprmizatit <B1>masinitingnto ` | |
| photosynthesis | `Problempriticeng <B1>on is the prace <B1>of is` | |
| the capital of France | `Pronal <B1>mprehey.* <B1>mobatining is the pra` | |
| a short story | `Problelemmatin <B1>is the practice <B1>of ieat` | |

Raw outputs, beam 8:

| case | generated | true |
|---|---|---|
| doc0 | ` <B1>compare-and-contrast table description` | ` step-<B2>by-step procedure, written <B1>for a` |
| doc5000 | `ndatication-sen** <B1>ath-the-skill angle,<B2>` | `ndations <B1>angle, formatted as a Q&<B2>A dia` |
| doc10000 | `orocicatical frication-fed <B1>alimition-fo` | `oretical foundations <B1>angle, formatted a` |
| The capital of France is | `omprming <B1>a debuging/error-cor-correctio` | |
| Once upon a time | `ning <B1>awith its cos clommmistakes tining` | |
| Hello | `w d mplyinicit mplizit masit mitanionit ` | |
| photosynthesis | `Problem <B1>mithenking is the practice <B1>of ` | |
| the capital of France | `Probley <B1>mpraxtis is the practice <B1>of ic` | |
| a short story | `Problelem <B1>frating is the practice <B1>ofyi` | |

The template format is the only lever that changes the character of the output. With it, every decoder writes the corpus definition scaffold: "Problem thinking is the practice of ...", "... framing is the practice of ...". It does not write the requested topic: photosynthesis becomes "Problem <B1>mithenking", a short story becomes "Problelem framing".

The in-corpus control shows the same failure. On the verbatim corpus prompt for negotiation, encoded identically to the training ids, greedy writes `Analog fthomprmice ...` instead of "Negotiation is the practice of ...".

### 9b. The missing piece: the model does not copy from its own context (`scripts/copy_probe.py`)

Every corpus doc names its topic X three times: in the question line, in "Here is a treatment of **X**", and in "**Definition.** X ...". The 2nd and 3rd mentions are pure in-context copies, 50-300 ids after the first.

TF on 16 training docs (negotiation and syllogisms; fp32, fixed path, real 127-id previous-doc tail):

| span | NLL (nats/id) | top-1 | first id NLL | first id top-1 |
|---|---|---|---|---|
| whole doc (first 512 ids) | 2.77 | 0.62 | | |
| "Here is a treatment of **X**" (copy) | 7.87 | 0.10 | 13.4 | 0.00 |
| "**Definition.** X" (copy) | 7.24 | 0.12 | 18.6 | 0.00 |

Visual check (negotiation doc 467, teacher-forced argmax):
```
true: Here is a treatment of *<B2>*negotiation** from a <B1>comparative
pred: Hereris a arer rent of *<B2>*ocx rrcl cn** from f caompararive
```

The model cannot reproduce a word it read about 60 ids earlier, even inside a shard where every doc has the same topic. This holds while it predicts template continuations ("perspective (contra", "** from") well.

Low TF loss comes from scaffold positions. Every slot that must be filled from context (topic names, entities, prompt words) is guessed from the global prior and comes out as letter soup. No decoder can fix this, because the right token is not among the candidates: first-id NLL at copy slots is 12-31 nats.

This is the strongest candidate for the "something we haven't seen". The carried field state (linear attention plus carry plus thought_state) does not provide token-level retrieval of recent content. A corpus where every doc is a recombined template never forced it to learn retrieval: the topic is predictable from the shard.

Suggested next measurements (inference only):
- an induction-head style probe on random byte strings repeated twice in one chunk vs across chunks;
- whether copying works within the open 128-id chunk but fails across the chunk boundary, i.e. through the carry.

**Does it speak?** No. With every decoder, the best output is template scaffolding with misspelled slot words. Beam and min-p change a few bytes, not the outcome.
