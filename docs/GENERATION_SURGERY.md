# Generation surgery: block-mode inference with carried state (Fractus CTE-Atom)

Checkpoint: HF `thefinalboss/fractus-cte-atom` `checkpoints/atom_latest` (step_00770000, 2,454,777,856 ids, ema_tf 0.484), CPU fp32, mmap load, eval, no weight changes. Script: `scripts/gen_block.py`. Raw results: `/workspace/gb1.jsonl`, `gb2.jsonl`, `gb3.jsonl` (verbatim outputs + ids).

## How training drives the model (mirrored)
`scripts/fast4gpu_atom.py`: SEQ=128 per chunk, B=16 lanes, `tick_chunk_train_ce` per chunk; carry
(`thought_state`, per-block `attn_S`, `attn_z`, `kuramoto_phases`; 49 batch-sized keys in the ckpt) persists
across chunks and is zeroed per lane only when the chunk contains BOS (257). `thought_state` is added only at
position 0 of the next chunk. Inference uses `tick_chunk` (same per-block core; verified identical CE to
`tick_chunk_train_ce`, and `FRACTUS_ATTN_IMPL=chunked` vs `cumsum` gives identical numbers).

## Modes
- **A (block tick)**: each step feeds the last C ids as one chunk and keeps the resulting carry.
- **B (regen block)**: carry committed at chunk boundaries; each step restores the committed carry, re-runs the pending block (C/2..C ids), takes the last-position id; at C ids the first C/2 are committed.
- Start states: `warm` = lane-0 carry stored in the ckpt (no BOS); `cold` = BOS reset; `primer` = BOS + 256 ids of an in-corpus document head (reasoning__syllogisms) committed in C-sized chunks, then the prompt.

## Teacher-forced diagnostics (CE nats/id, chunk 128)
- lanes.i16 (older eval slice): warm 6.9, 6.2, 4.5, 5.0, 5.5, 5.8; cold 100, 29, 9.0, 7.4, 7.3, 5.3.
- In-corpus document (syllogisms), 30 chunks: cold-from-BOS 22.6, 1.47, 0.69, 4.65, 2.2, 0.39, 0.18 ... mostly 0.1-2.5; lane-0 warm is worse (4.0, 3.2, 2.6 ...). So for in-corpus text, BOS reset + ≥1 chunk of document is the correct regime; the stored lane-0 carry does not help unrelated text.
- Short novel prompts (prompt TF): warm 8.9-17.7; primer 6.2-14; the in-corpus continuation tail after the primer: **0.97**.

## Generation results (N=120 unless noted, unique@40, ids/s on loaded 8-core CPU)
| config | unique@40 | ids/s | note |
|---|---|---|---|
| B:128 greedy warm | 0.35-0.40 | 0.75 | pseudo-word salad |
| B:32 T0.8 k40 rp1.1 primer | 0.40-0.525 | 1.1 | salad, some primer-domain words |
| A:128 T0.8 k40 rp1.1 primer | 0.40 | 0.66 | worse: re-fed overlap corrupts carry |
| B:128 greedy primer, in-corpus continuation | 0.375 | 0.92 | ~20 correct ids then drift |
| B:128 greedy primer, "The capital of France is" | 0.45 | 1.28 | salad |

Verbatim samples:
- B:128 greedy primer, continue in-corpus doc: `'ioner looking to to deepenging.\n\n**Derition.** Systesidece ice w oing precting e ececesing watedefr acesands ofr th'`
- B:128 greedy warm, "The capital of France is": `'rompureriplespaces- ticaliticationgr tical dingrs teses tonator thecad diuprcaprescesioratioration d he pan en ner'`
- B:32 T0.8 primer, "The capital of France is": `'leterepepatioog acior atizineriting atier uatier loong tinemer anerditing\nederlying tengcuertton. Det s athef dis pl'`
- A:128 T0.8 primer, "Bonjour, je m'appelle": `'rid ctsd ctnndstete-ctitbityctyycathtastitharabisryrystherathithere are r  re r io frathe a t ta p as aun th l as das'`

## What works / what does not
- Works: Mode B at C=128 reproduces the training forward exactly; the carry mechanics are correct (in-corpus TF 0.1-2 after one chunk; continuation TF 0.97).
- Does not work: free-running generation in any mode/config. Even on in-corpus text with low TF loss, greedy output holds for ~20 ids and then decays into morpheme salad (unique@40 0.35-0.53). Mode A is strictly worse than B (re-feeding overlapping context writes duplicates into attn_S/attn_z).
- Prompts outside the corpus distribution have TF 6-17 nats/id regardless of start state; a 256-id primer does not fix that.

## Reason for remaining failure
Inference driving is not the bottleneck: Mode B is the training forward. The model's low ema_tf (0.48) is local next-byte prediction on in-corpus text, which degrades as soon as the input is its own samples (exposure bias), and it does not generalize to short novel prompts. Fixing this needs a training-side change (e.g. scheduled sampling / SS_RATE>0, more diverse data), not an inference change.

## Addendum (gb3)
- A:128 greedy primer, in-corpus continuation: unique@40 0.275, 1.0 ids/s: `'io.\n**** at bat cte The pllyindsstnduly ing ly- prng ting plionthecusthe p p ts the tas thass sss bs bs athorathon'`
- A:128 greedy primer, France: unique@40 0.325.
- B:128 T0.8 k40 rp1.1 primer, continuation: unique@40 0.475, 1.21 ids/s: `'iong tionery.\n\n*Drefion.** Slesdedatiice ongiceasing ing wea expe eds or heds ped atefctey wtit nnef acthelt men,f'`
- B:128 T0.8 k40 rp1.1 primer, France: unique@40 0.50, 1.4 ids/s.
Confirms B > A. Even the best B config degrades into salad.

## Addendum 2 (gb4: bf16 weights, N=80, 3 prompts; fp32 mmap thrashed the page cache, so bf16 was used: ~2.2-3.1 ids/s)
| config | unique@40 (Bonjour / France / Il était) | ids/s | prompt TF |
|---|---|---|---|
| A:32:0:1:1.0:primer | 0.35 / 0.375 / 0.25 | 2.83 | [] / [] / [] |
| A:32:0.8:40:1.1:primer | 0.55 / 0.275 / 0.45 | 2.77 | [] / [] / [] |
| B:128:0:1:1.0:cold | 0.375 / 0.3 / 0.425 | 2.44 | [200.0] / [57.75] / [113.5] |
| B:128:0.8:40:1.1:cold | 0.475 / 0.5 / 0.45 | 2.5 | [200.0] / [55.25] / [114.5] |
| B:128:0:1:1.0:primer | 0.425 / 0.375 / 0.5 | 2.51 | [11.438] / [7.031] / [14.188] |
| B:128:0.8:40:1.1:primer | 0.375 / 0.475 / 0.475 | 2.8 | [11.688] / [6.75] / [13.812] |

Verbatim (all in gb4.jsonl):
- A:32:0:1:1.0:primer, "Bonjour, je m'appelle": `'z aind  x thhherncctcte.cbpepepeeecelecenceelllllis.... un bbs v varnrwrwrwrwrw'`
- A:32:0:1:1.0:primer, 'The capital of France is': `'penedume/tt. sthDhe bs bisca ca in an n e telseestme,terotrort.tid tid a '`
- A:32:0:1:1.0:primer, 'Il était une fois': `' wlddeng/neeeeeneteatetataan.. d d p  p bn cn c b  Thaaaaaaaaaaraanaeaereeeeee'`
- A:32:0.8:40:1.1:primer, "Bonjour, je m'appelle": `'zid s a ctxathhhen. ct pupplplececicastcastcakisinisisec\x04-w,t�tcero&KRdiog'`
- A:32:0.8:40:1.1:primer, 'The capital of France is': `'lenwling outhods..s th t r   is a   cenameanestssststtttttthtant,,,,iniiiiii'`
- A:32:0.8:40:1.1:primer, 'Il était une fois': `'zg d pte binthe thon. rsts  aus ce te deateatin wind,  thepngppr- thorere Q'`
- B:128:0:1:1.0:cold, "Bonjour, je m'appelle": `'it nrgh sisssstt phio list lisest, listtraivillew proh ofppoati atorlp tat'`
- B:128:0:1:1.0:cold, 'The capital of France is': `'onditon.wf atstsA is ssonatin antpe tinervessnd oneatrtnvevsov soend sest tep'`
- B:128:0:1:1.0:cold, 'Il était une fois': `'t deltisatiora poin iorg to ocaevenerete an dapt che vore commiston dereta('`
- B:128:0.8:40:1.1:cold, "Bonjour, je m'appelle": `'it rgnr bhoossaplatiplin. ts pling troFaisis-(v�ute a a-th ambkes deteasap-pi'`
- B:128:0.8:40:1.1:cold, 'The capital of France is': `'wom and isabteg hhit clovise, form aungerthano y.\nIpoan ler yas blingepoh.* a'`
- B:128:0.8:40:1.1:cold, 'Il était une fois': `'g lhe cstin usting deraing ite te walpyine reath manaingo ferom a pe.hPronun'`
- B:128:0:1:1.0:primer, "Bonjour, je m'appelle": `'zer catinit aorion ated wr iprenepring acortiooeineing tithetingeraly ong\n\n'`
- B:128:0:1:1.0:primer, 'The capital of France is': `'speplenentize foripurs atipliooonene denerstatindeng. Setitine we terlengetle'`
- B:128:0:1:1.0:primer, 'Il était une fois': `' ze apriprineng tlortzioo atinener tineratinering tlyintoo�� uryiaceiu. The i'`
- B:128:0.8:40:1.1:primer, "Bonjour, je m'appelle": `'zizKt as for acin ing cule ateormepkit, writen wredipengate tepiatiatingyione'`
- B:128:0.8:40:1.1:primer, 'The capital of France is': `"suponedepriplyizes  aces ong ineringriplesatotothe atine' ceintinendernt ur"`
- B:128:0.8:40:1.1:primer, 'Il était une fois': `'ierprong szen apptot ze ippacinerinesti tienenderace Ding tintiwederlyinouqu'`

Findings: Mode A at block 32 collapses into letter runs (unique@40 0.25-0.55). Plain BOS reset gives prompt TF 55-200 per id (the first chunk after a reset is far out of distribution) and output similar to primer runs. Primer and plain-BOS starts have similar unique@40 (0.3-0.5). No config produces coherent text; the conclusion above holds.

## Carry surgery round (bf16, N=80; scripts/gen_surgery2.py; raw: /workspace/gs2.jsonl, gs3.jsonl, gs2_ref.json)

### 1. Carry norms: generation vs teacher-forced (TF) reference
The reference is TF over one in-corpus document starting at BOS, in 128-id chunks (27 clean chunks; chunk 28 contains BOS and resets). In the reference, attn_S Frobenius (mean over 16 blocks) grows about linearly, ~2.7e9 per chunk (3.7e9 after 1 chunk, 2.5e10 after 10, 7.4e10 after 27), and attn_z grows ~1.5e7 per chunk. Nothing decays (as in DISCOVERY_LOG §7.2). Past 27 chunks the reference is extrapolated linearly. thought_state L2 in TF stays at ~600-1300.

| config (prompt: Bonjour) | gen pos | carry ticks | S_fro mean/max | ref S_fro mean/max | z_l2 mean/max | ref z mean/max | thought L2 (ref) | phases L2 mean/max |
|---|---|---|---|---|---|---|---|---|
| B:128:0:1:1.0:cold:none:0 | 0 | 0 | 0/0 | 3.73e+09/4.72e+10 | 0/0 | 1.27e+07/1.5e+08 | 0 (1317) | 0.0/0.0 |
| B:128:0:1:1.0:cold:none:0 | 10 | 0 | 0/0 | 3.73e+09/4.72e+10 | 0/0 | 1.27e+07/1.5e+08 | 0 (1317) | 0.0/0.0 |
| B:128:0:1:1.0:cold:none:0 | 20 | 0 | 0/0 | 3.73e+09/4.72e+10 | 0/0 | 1.27e+07/1.5e+08 | 0 (1317) | 0.0/0.0 |
| B:128:0:1:1.0:cold:none:0 | 40 | 0 | 0/0 | 3.73e+09/4.72e+10 | 0/0 | 1.27e+07/1.5e+08 | 0 (1317) | 0.0/0.0 |
| B:128:0:1:1.0:cold:none:0 | 79 | 0 | 0/0 | 3.73e+09/4.72e+10 | 0/0 | 1.27e+07/1.5e+08 | 0 (1317) | 0.0/0.0 |
| B:128:0:1:1.0:cold:none:0 | 80 | 0 | 0/0 | 3.73e+09/4.72e+10 | 0/0 | 1.27e+07/1.5e+08 | 0 (1317) | 0.0/0.0 |
| B:128:0:1:1.0:primer:none:0 | 0 | 2 | 5.88e+09/7.18e+10 | 5.87e+09/7.18e+10 | 2.68e+07/2.99e+08 | 2.68e+07/2.99e+08 | 827 (831) | 163.719/166.563 |
| B:128:0:1:1.0:primer:none:0 | 10 | 2 | 5.88e+09/7.18e+10 | 5.87e+09/7.18e+10 | 2.68e+07/2.99e+08 | 2.68e+07/2.99e+08 | 827 (831) | 163.719/166.563 |
| B:128:0:1:1.0:primer:none:0 | 20 | 2 | 5.88e+09/7.18e+10 | 5.87e+09/7.18e+10 | 2.68e+07/2.99e+08 | 2.68e+07/2.99e+08 | 827 (831) | 163.719/166.563 |
| B:128:0:1:1.0:primer:none:0 | 40 | 2 | 5.88e+09/7.18e+10 | 5.87e+09/7.18e+10 | 2.68e+07/2.99e+08 | 2.68e+07/2.99e+08 | 827 (831) | 163.719/166.563 |
| B:128:0:1:1.0:primer:none:0 | 79 | 2 | 5.88e+09/7.18e+10 | 5.87e+09/7.18e+10 | 2.68e+07/2.99e+08 | 2.68e+07/2.99e+08 | 827 (831) | 163.719/166.563 |
| B:128:0:1:1.0:primer:none:0 | 80 | 2 | 5.88e+09/7.18e+10 | 5.87e+09/7.18e+10 | 2.68e+07/2.99e+08 | 2.68e+07/2.99e+08 | 827 (831) | 163.719/166.563 |
| A:128:0:1:1.0:cold:none:0 | 0 | 0 | 0/0 | 3.73e+09/4.72e+10 | 0/0 | 1.27e+07/1.5e+08 | 0 (1317) | 0.0/0.0 |
| A:128:0:1:1.0:cold:none:0 | 10 | 10 | 1.21e+09/1.63e+10 | 2.52e+10/2.81e+11 | 2.61e+06/3.62e+07 | 1.46e+08/1.49e+09 | 4909 (914) | 79.967/82.565 |
| A:128:0:1:1.0:cold:none:0 | 20 | 20 | 1.55e+09/2.14e+10 | 5.47e+10/5.95e+11 | 3.48e+06/4.79e+07 | 2.98e+08/2.97e+09 | 3829 (704) | 92.137/95.562 |
| A:128:0:1:1.0:cold:none:0 | 40 | 40 | 1.89e+09/2.53e+10 | 1.09e+11/1.18e+12 | 5.26e+06/7.13e+07 | 5.98e+08/5.93e+09 | 2265 (643) | 112.559/116.117 |
| A:128:0:1:1.0:cold:none:0 | 79 | 79 | 4.02e+09/5.53e+10 | 2.16e+11/2.34e+12 | 8.72e+06/1.17e+08 | 1.18e+09/1.17e+10 | 2159 (643) | 144.061/147.072 |
| A:128:0:1:1.0:cold:none:0 | 80 | 80 | 4.05e+09/5.57e+10 | 2.19e+11/2.37e+12 | 8.81e+06/1.18e+08 | 1.2e+09/1.19e+10 | 2267 (643) | 144.578/147.298 |
| A:128:0:1:1.0:primer:none:0 | 0 | 2 | 5.88e+09/7.18e+10 | 5.87e+09/7.18e+10 | 2.68e+07/2.99e+08 | 2.68e+07/2.99e+08 | 814 (831) | 163.785/166.67 |
| A:128:0:1:1.0:primer:none:0 | 10 | 12 | 2.62e+10/3.46e+11 | 3.17e+10/3.56e+11 | 5.54e+07/6.33e+08 | 1.76e+08/1.79e+09 | 2587 (660) | 84.248/92.778 |
| A:128:0:1:1.0:primer:none:0 | 20 | 22 | 4.73e+10/6.19e+11 | 6.39e+10/7.07e+11 | 9.12e+07/1.08e+09 | 3.28e+08/3.27e+09 | 2077 (748) | 99.638/112.516 |
| A:128:0:1:1.0:primer:none:0 | 40 | 42 | 8.49e+10/1.09e+12 | 1.15e+11/1.24e+12 | 1.9e+08/2.34e+09 | 6.28e+08/6.23e+09 | 1527 (643) | 115.808/131.591 |
| A:128:0:1:1.0:primer:none:0 | 79 | 81 | 1.77e+11/2.31e+12 | 2.22e+11/2.4e+12 | 4.97e+08/6.12e+09 | 1.21e+09/1.2e+10 | 1281 (643) | 143.602/156.769 |
| A:128:0:1:1.0:primer:none:0 | 80 | 82 | 1.81e+11/2.36e+12 | 2.24e+11/2.43e+12 | 5.06e+08/6.24e+09 | 1.23e+09/1.22e+10 | 1286 (643) | 144.371/158.854 |

Findings:
- **No attn_S inflation in generation.** In Mode A (one carry tick per generated id), S_fro after 80 ticks is at or below the TF reference for the same number of chunks: primer start 1.8-2.2e11 vs ref 2.2e11; plain BOS 3-4e9 vs ref 2.2e11. In Mode A, attn_z *undershoots*: 5e8 vs ref 1.2e9 (primer), 9e6 vs 1.2e9 (BOS). So the S/z ratio in generation is ~2x the reference (primer) and ~20x (plain BOS).
- **thought_state is inflated in Mode A**: 1300-4900 vs 640-910 in TF (2-7x).
- **In Mode B the carry is frozen during 80 ids.** With block 128 and a short prompt, the pending block never reaches 128 ids, so there is no commit (ticks stay at 0 for BOS, 2 for primer). Its carry equals the TF carry exactly. The failure in Mode B therefore does not come from the carry.

### 2-3. Partial carry reset and inference-side scheduled sampling (unique@40: Bonjour / France / Il était; ids/s mean)

| config | unique@40 | ids/s |
|---|---|---|
| B:128:0:1:1.0:cold:none:0 | 0.4 / 0.35 / 0.4 | 2.27 |
| B:128:0:1:1.0:primer:none:0 | 0.4 / 0.275 / 0.375 | 2.23 |
| B:128:0:1:1.0:primer:none:0 (rerun) | 0.35 / 0.5 / 0.35 | 2.12 |
| A:128:0:1:1.0:cold:none:0 | 0.425 / 0.475 / 0.425 | 2.31 |
| A:128:0:1:1.0:cold:none:0 (rerun) | 0.4 / 0.275 / 0.425 | 2.39 |
| A:128:0:1:1.0:primer:none:0 | 0.275 / 0.3 / 0.225 | 2.24 |
| A:128:0:1:1.0:primer:none:0 (rerun) | 0.3 / 0.3 / 0.325 | 2.38 |
| A:128:0:1:1.0:primer:half:0 | 0.5 / 0.475 / 0.4 | 2.11 |
| A:128:0.8:40:1.1:primer:half:0 | 0.45 / 0.425 / 0.275 | 2.5 |
| A:128:0:1:1.0:primer:clamp:0 | 0.325 / 0.275 / 0.275 | 2.48 |
| A:128:0.8:40:1.1:primer:clamp:0 | 0.225 / 0.325 / 0.225 | 2.57 |
| B:128:0:1:1.0:primer:half:0 | 0.425 / 0.4 / 0.375 | 2.61 |
| B:128:0.8:40:1.1:primer:half:0 | 0.425 / 0.55 / 0.425 | 2.51 |
| B:128:0:1:1.0:primer:clamp:0 | 0.35 / 0.4 / 0.375 | 2.53 |
| B:128:0.8:40:1.1:primer:clamp:0 | 0.375 / 0.5 / 0.5 | 2.68 |
| B:128:0:1:1.0:primer:none:0.2 | 0.525 / 0.4 / 0.35 | 0.97 |
| B:128:0.8:40:1.1:primer:none:0.2 | 0.4 / 0.475 / 0.425 | 2.64 |
| B:128:0.8:40:1.1:primer:none:0 | 0.45 / 0.5 / 0.525 | 2.61 |
| A:128:0.8:40:1.1:primer:none:0 | 0.25 / 0.3 / 0.325 | 2.53 |

Notes:
- In Mode B the half/clamp fixes never trigger within 80 ids, because no block boundary is crossed. B half/clamp rows are therefore effectively baseline reruns. Identical baseline configs run twice gave different greedy outputs (bf16 CPU nondeterminism under load), e.g. B:128 greedy primer France 0.275 vs 0.50. Unique@40 differences of about ±0.15 are noise.
- Mode A x0.5 (half) raises greedy unique@40 from 0.275/0.30/0.225-0.325 to 0.50/0.475/0.40. The output is still not words. Clamping to the reference norm changes nothing (0.275-0.325), since the generated S/z are already at or below the reference.
- Inference SS (20% of generated positions in the fed block replaced by prompt ids, Mode B): unique@40 0.35-0.525, no change in quality. The ids/s values of 0.13/0.40 come from box contention.
- The B:128 T0.8 primer baseline and A:128 T0.8 primer baseline in this round were not completed (run stalled >7 min and was killed); see gb4 for B:128 T0.8 primer (0.375/0.475/0.475).

Verbatim (Mode A half, greedy, primer):
- "Bonjour, je m'appelle": `'caisss tn znusentudetderbr inte fo vntevee. f weroe sexpkalerssir m ting t to'`
- 'The capital of France is': `' plewtkcon dinglevewt cewt frend ceriviliung ohe the uakit pl ce. tanoly is d'`
- 'Il était une fois': `' pzesedevencesiwrg alyesyes fesingaer undexait he t the coranclio  ghe re un'`
Verbatim (Mode B T0.8 primer + 20% inference SS):
- "Bonjour, je m'appelle": `'zericzatorr a ce aens weipe  ile- upt fplicacucacapitiniongation.  aainiroert'`
- 'The capital of France is': `"speronediprbcteotipeselyinanondezi ule\neLitinivie aloduatog mr'deing teeder "`
- 'Il était une fois': `' wrg wle tgernt prtrone acndhoncteedinrzipeizaeni ing tedinpttertye racn-nton'`

Verdict: none of the carry fixes or inference SS makes this checkpoint generate coherent text. The carry is not inflated relative to TF. In Mode B it is identical to TF, so the remaining failure comes from the model's own next-id predictions on its sampled inputs and on out-of-corpus prompts, not from the inference driving.

Trainer check (scripts/fast4gpu_atom.py): `SS_RATE` exists, default 1.0 (`SS_PROB_START` 0.2 -> `SS_PROB_END` 0.5 over `SS_RAMP_TOKENS` 5e7). The current run logs an SS pass on only some steps (`ss=`/`ema_ss` ~1.17 present on a subset of log lines; SS takes ~17% of wall time), so the run's effective SS_RATE is <1.0. Per-lane carry reset at BOS already exists (`reset_rows` on any chunk containing 257, applied to thought_state, attn_S, attn_z, kuramoto_phases).
