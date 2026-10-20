# v4 step_285000 (609.7M v4 ids) vs step_25000 (77.2M): eval + externalization test

Box copy of the pod's newest v4 good checkpoint (`/workspace/ckpt_v4/good`, step_in_run 285000, 609,726,464 v4 ids,
ids_total 5,921,787,904, md5 badf2843..., pod VAL 2.4887). Inference only, fp32 fixed generation path. Trainer and pod
untouched. Same protocol as the step_25000 eval: `scripts/v4_eval.py`, plus `scripts/v4_extern.py`,
`scripts/v4_capacity_check.py` and `scripts/v4_poscurve.py`. Results: `docs/v4_eval_data/`.

**Verdict: teacher-forced prediction improved by 0.2-0.3 nats/id, but the model has not moved toward speaking.** Greedy
still loops ("the the the", "wand wand"). The real-word rate of prompt outputs fell (greedy 0.90 -> 0.56). Exact word
completion from a true 1-3 byte prefix is 0/144, 0/144, 2/144.

### Val NLL per source (8 windows x 1024 scored ids each, 127-id context)

| model | source | NLL nats/id (mean of windows) | median window NLL | top-1 | bits/byte |
|---|---|---|---|---|---|
| v4_610M | fineweb | 2.561 | 2.558 | 0.275 | 3.840 |
| v4_610M | wiki | 2.562 | 2.491 | 0.273 | 3.872 |
| v4_610M | tinystories | 2.197 | 2.177 | 0.355 | 3.305 |
| v4_610M | pg19 | 5.304 | 2.505 | 0.236 | 8.014 |
| v4_77M | fineweb | 2.741 | 2.694 | 0.253 | 4.111 |
| v4_77M | wiki | 2.831 | 2.670 | 0.246 | 4.278 |
| v4_77M | tinystories | 2.508 | 2.443 | 0.289 | 3.773 |
| v4_77M | pg19 | 5.502 | 2.695 | 0.207 | 8.313 |

### Copy probe on real text (words of >=5 letters in the scored region; previous occurrence in the same 128-id grid chunk vs only in an earlier chunk)

| model | class | words | NLL/id all bytes | top-1 all | NLL first byte | top-1 first byte | NLL bytes 2..n | top-1 bytes 2..n | median distance (ids) |
|---|---|---|---|---|---|---|---|---|---|
| v4_610M | first | 1631 | 2.870 | 0.148 | 3.790 | 0.044 | 2.717 | 0.166 | - |
| v4_610M | repeat_same_chunk | 80 | 2.713 | 0.173 | 3.376 | 0.138 | 2.592 | 0.180 | 37 |
| v4_610M | repeat_earlier_chunk | 334 | 2.801 | 0.148 | 3.579 | 0.045 | 2.668 | 0.165 | 205 |
| v4_77M | first | 1631 | 3.131 | 0.116 | 4.180 | 0.044 | 2.956 | 0.128 | - |
| v4_77M | repeat_same_chunk | 80 | 3.014 | 0.123 | 3.884 | 0.025 | 2.856 | 0.141 | 37 |
| v4_77M | repeat_earlier_chunk | 334 | 3.048 | 0.103 | 3.757 | 0.036 | 2.926 | 0.114 | 205 |

### Generation (fp32 fixed path, warm = real 127-id val tail; CONT = 256-id val prefix, 40 ids)

| model | decoder | CONT unique@40 | CONT exact /40 | CONT seqmatch | CONT real-word | prompt unique@40 | prompt real-word |
|---|---|---|---|---|---|---|---|
| v4_610M | greedy | 8.3 | 5.33 | 0.217 | 0.80 | 10.7 | 0.56 |
| v4_610M | beam8 | 7.3 | 3.67 | 0.242 | 0.82 | 12.0 | 0.64 |
| v4_610M | minp_T0.7_p0.1_ng4 | 17.7 | 4.67 | 0.150 | 0.60 | 15.7 | 0.33 |
| v4_77M | greedy | 8.0 | 5.33 | 0.275 | 0.96 | 9.3 | 0.90 |
| v4_77M | beam8 | 10.7 | 4.67 | 0.183 | 0.63 | 11.3 | 0.87 |
| v4_77M | minp_T0.7_p0.1_ng4 | 14.0 | 4.33 | 0.225 | 0.40 | 13.0 | 0.64 |

### Raw outputs (unedited, markers shown)

**v4_610M / greedy**

| case | generated | true |
|---|---|---|
| fineweb_doc3 | `tof tof therenthe <B1>the the the the the <B1>` | `Who Won!⏎<B3>is for all students. All <B1>stud` |
| wiki_doc3 | `r  te the the <B1>the the the the the the <B1>` | `s are forest <B1>and shrubland, but it has ` |
| tinystories_doc3 | `thhe was the the <B1>the the wastoy the to ` | `g, green thumb! <B1>It was so big, Bob had ` |
| The capital of France is | `therered <B1>wand warerer wather <B1>wathe wan` |  |
| Once upon a time | `dered <B1>wand wand wand wand thererered <B1>a` |  |
| Hello | `furered <B1>wand wand wather athere werered` |  |

**v4_610M / beam8**

| case | generated | true |
|---|---|---|
| fineweb_doc3 | `fffof therenthen <B1>thend the the the the ` | `Who Won!⏎<B3>is for all students. All <B1>stud` |
| wiki_doc3 | `n  the the the the <B1>the the the the the ` | `s are forest <B1>and shrubland, but it has ` |
| tinystories_doc3 | `thhe the they they <B1>they they they they ` | `g, green thumb! <B1>It was so big, Bob had ` |
| The capital of France is | `thered <B1>waringerengingengeng.⏎<B3>⏎<B3>⏎<B3>⏎<B3>⏎<B3>R` |  |
| Once upon a time | `and wand <B1>and wand wathe the therere <B1>wa` |  |
| Hello | `wand <B1>wand wand watherered wathered were` |  |

**v4_610M / minp_T0.7_p0.1_ng4**

| case | generated | true |
|---|---|---|
| fineweb_doc3 | `stemed (200), batand <B1>theld dero than it` | `Who Won!⏎<B3>is for all students. All <B1>stud` |
| wiki_doc3 | `s s t wan of <B1>the che theld frin than <B1>t` | `s are forest <B1>and shrubland, but it has ` |
| tinystories_doc3 | `s ere wherory. <B1>toun Sullld hey wid we t` | `g, green thumb! <B1>It was so big, Bob had ` |
| The capital of France is | `tinge <B1>Songerenteand Suctor Stho <B1>angeis` |  |
| Once upon a time | `dended <B1>wiofingher and ator d wo than <B1>t` |  |
| Hello | `fulle wanghed <B1>beand waterd d wo tono it` |  |

**v4_77M / greedy**

| case | generated | true |
|---|---|---|
| fineweb_doc3 | `th Seres teres the \| \| \|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|⏎` | `Who Won!⏎<B3>is for all students. All <B1>stud` |
| wiki_doc3 | `s  an an an ale alin <B1>aliand an ale ale ` | `s are forest <B1>and shrubland, but it has ` |
| tinystories_doc3 | `n  a and a a and and <B1>ande ande and and ` | `g, green thumb! <B1>It was so big, Bob had ` |
| The capital of France is | ` <B1>ande and Mind are and and <B1>are ande an` |  |
| Once upon a time | ` <B1>and and and bere and are <B1>ande and and` |  |
| Hello | `ur the <B1>whe whe whe are are and <B1>and and` |  |

**v4_77M / beam8**

| case | generated | true |
|---|---|---|
| fineweb_doc3 | `thhed \| \|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|⏎<B3>\|⏎<B3>\|\|⏎<B3>\|\|\|\|⏎` | `Who Won!⏎<B3>is for all students. All <B1>stud` |
| wiki_doc3 | `s  He He ale ale ale ande ande <B1>the are ` | `s are forest <B1>and shrubland, but it has ` |
| tinystories_doc3 | `s  the the the <B1>Minde he he are are and ` | `g, green thumb! <B1>It was so big, Bob had ` |
| The capital of France is | ` and <B1>the thererere there <B1>where and are` |  |
| Once upon a time | ` the <B1>the where whe ware <B1>there and the ` |  |
| Hello | `und th!<B2>⏎<B3>Mind Mind and Mine Min Mind Mi` |  |

**v4_77M / minp_T0.7_p0.1_ng4**

| case | generated | true |
|---|---|---|
| fineweb_doc3 | `Jeere Suntof Seteane Stato <B1>Sche tandest` | `Who Won!⏎<B3>is for all students. All <B1>stud` |
| wiki_doc3 | `s e t al <B1>ofhe he ar atato arero hand it` | `s are forest <B1>and shrubland, but it has ` |
| tinystories_doc3 | `s  le al ale berea s athe <B1>arere tande t` | `g, green thumb! <B1>It was so big, Bob had ` |
| The capital of France is | ` <B1>whe wheromer be <B1>and ande bero tend s ` |  |
| Once upon a time | ` <B1>whe wherome bre ar and be <B1>the thanes ` |  |
| Hello | ` nd ther <B1>omerere an ande d hero than <B1>t` |  |


### (a) Teacher-forced accuracy on val (32 windows, 1,024 scored ids each)

| model | source | top-1 | top-5 | NLL |
|---|---|---|---|---|
| v4_610M | fineweb | 0.275 | 0.643 | 2.558 (median) |
| v4_610M | wiki | 0.273 | 0.642 | 2.491 (median) |
| v4_610M | tinystories | 0.355 | 0.726 | 2.177 (median) |
| v4_610M | pg19 | 0.236 | 0.527 | 2.505 (median) |
| v4_610M | all | 0.285 | 0.634 | 2.482 (median) |
| v4_77M | fineweb | 0.253 | 0.614 | 2.694 (median) |
| v4_77M | wiki | 0.246 | 0.614 | 2.670 (median) |
| v4_77M | tinystories | 0.289 | 0.662 | 2.443 (median) |
| v4_77M | pg19 | 0.207 | 0.499 | 2.695 (median) |
| v4_77M | all | 0.249 | 0.597 | 2.639 (median) |

### (a) Word completion (greedy from the true prefix; real words >=5 letters; 6 per window, 24 windows)

| model | given bytes | words | exact word | TF top-1 on next byte | first-step logits vs TF max abs diff |
|---|---|---|---|---|---|
| v4_610M | 1 | 144 | 0.000 | 0.194 | 1.58e+00 |
| v4_610M | 2 | 144 | 0.000 | 0.118 | 4.42e+00 |
| v4_610M | 3 | 144 | 0.014 | 0.118 | 4.47e+00 |
| v4_77M | 1 | 144 | 0.000 | 0.167 | 1.35e-02 |
| v4_77M | 2 | 144 | 0.000 | 0.090 | 1.11e-02 |
| v4_77M | 3 | 144 | 0.000 | 0.090 | 5.90e+00 |

### (b) Free generation (6 val contexts x 64 ids)

| model | decoder | real-word rate gen | words gen | real-word rate true | mean log p(generated) | mean log p(true) | gen vs TF logits max abs diff |
|---|---|---|---|---|---|---|---|
| v4_610M | greedy | 0.824 | 74 | 0.970 | -1.292 | -2.494 | 1.72e-02 |
| v4_610M | minp_T0.7_p0.1 | 0.516 | 64 | 0.970 | -1.709 | -2.494 | 1.72e-02 |
| v4_77M | greedy | 0.919 | 74 | 0.970 | -1.504 | -2.615 | 1.60e-04 |
| v4_77M | minp_T0.7_p0.1 | 0.515 | 68 | 0.970 | -1.973 | -2.615 | 1.14e-04 |

### Long range: NLL on grid chunk 7 (ids 897..1023 of each window)

| model | full carry (7 chunks before) | carry reset 1 chunk before (128-255 ids context) | full, same ids | reset at chunk (0-127 ids, in-chunk only) |
|---|---|---|---|---|
| v4_610M | 2.498 | 2.343 | 2.498 | 2.618 |
| v4_77M | 2.665 | 2.715 | 2.665 | 3.716 |

(pg19 excluded from the long-range rows: two windows sit at 13-15 nats in every model.)

### Raw free generations (unedited, markers shown)

- **v4_610M / greedy:fineweb_doc3**: `tof tof therenthe <B1>the the the the the <B1>the the fof the the the `  
  true: `Who Won!
<B3>is for all students. All <B1>students must have the
<B3>to a`
- **v4_610M / minp_T0.7_p0.1:fineweb_doc3**: `stemed (200), batand <B1>theld dero than ithe <B1>froritede t f thelle`  
  true: `Who Won!
<B3>is for all students. All <B1>students must have the
<B3>to a`
- **v4_610M / greedy:fineweb_doc7**: `rekes the the the the <B1>tonde the the the the <B1>the the the the th`  
  true: `nty wealthy countries <B1>on their abilities to process <B1>written an`
- **v4_610M / minp_T0.7_p0.1:fineweb_doc7**: `stere thent tont ake <B1>therk thin tond <B1>the the tito prman the <B1>p`  
  true: `nty wealthy countries <B1>on their abilities to process <B1>written an`
- **v4_610M / greedy:wiki_doc3**: `r  te the the <B1>the the the the the the <B1>the the the Son Sof <B1>the`  
  true: `s are forest <B1>and shrubland, but it has <B1>also been found in gard`
- **v4_610M / minp_T0.7_p0.1:wiki_doc3**: `s s t wan of <B1>the che the herero than sthe <B1>broris pind therind `  
  true: `s are forest <B1>and shrubland, but it has <B1>also been found in gard`
- **v4_610M / greedy:wiki_doc7**: `e t <B1>(190) (1) (190) 190193)<B2> 19993, 19993) ther <B1>1999998) the t`  
  true: `a at <B1>Anton Malatinský Stadium <B1>on 24 April 2022, replacing <B1>Ma`
- **v4_610M / minp_T0.7_p0.1:wiki_doc7**: `ans <B1>wal tof 2019200 <B1>tator Frs Phendeston terotio <B1>int Pathe 19`  
  true: `a at <B1>Anton Malatinský Stadium <B1>on 24 April 2022, replacing <B1>Ma`
- **v4_610M / greedy:tinystories_doc3**: `thhe was the the <B1>the the wastoy the to <B1>the the the ware tound `  
  true: `g, green thumb! <B1>It was so big, Bob had never <B1>seen one before. `
- **v4_610M / minp_T0.7_p0.1:tinystories_doc3**: `s ere wherory. <B1>toun Sullld hey wid we tous <B1>witho s anto the wa`  
  true: `g, green thumb! <B1>It was so big, Bob had never <B1>seen one before. `
- **v4_610M / greedy:tinystories_doc7**: `lly and and <B1>and and the touthe the <B1>and they the are the <B1>and t`  
  true: `y closed the door <B1>and they played all day.<B2>
<B3>When they came bac`
- **v4_610M / minp_T0.7_p0.1:tinystories_doc7**: ` Tllow he he <B1>bitoke he tom hero he. t <B1>washerd the want stoy <B1>a`  
  true: `y closed the door <B1>and they played all day.<B2>
<B3>When they came bac`
- **v4_77M / greedy:fineweb_doc3**: `th Seres teres the | | ||||||||||||||||
<B3>|
<B3>|
<B3><B3><B3>|||||
<B3>|||
<B3>||
`  
  true: `Who Won!
<B3>is for all students. All <B1>students must have the
<B3>to a`
- **v4_77M / minp_T0.7_p0.1:fineweb_doc3**: `Jeere Suntof Seteane Sucton <B1>che tend sthes Brotit pinctinere <B1>a`  
  true: `Who Won!
<B3>is for all students. All <B1>students must have the
<B3>to a`
- **v4_77M / greedy:fineweb_doc7**: `reare <B1>arerere arere are an <B1>are are arere are are <B1>an are are a`  
  true: `nty wealthy countries <B1>on their abilities to process <B1>written an`
- **v4_77M / minp_T0.7_p0.1:fineweb_doc7**: `suere <B1>Sanofin Orale Suctor Acho Rune ithus <B1>Sorit iin Aus Relle`  
  true: `nty wealthy countries <B1>on their abilities to process <B1>written an`
- **v4_77M / greedy:wiki_doc3**: `s  an an an ale alin <B1>aliand an ale ale alind alin <B1>ale alind al`  
  true: `s are forest <B1>and shrubland, but it has <B1>also been found in gard`
- **v4_77M / minp_T0.7_p0.1:wiki_doc3**: `s e t al <B1>ofhe he ar atato arero hand ithe and <B1>it iliman the al`  
  true: `s are forest <B1>and shrubland, but it has <B1>also been found in gard`
- **v4_77M / greedy:wiki_doc7**: `e d <B1>an an an an an an an <B1>and an arere are <B1>and and an are arer`  
  true: `a at <B1>Anton Malatinský Stadium <B1>on 24 April 2022, replacing <B1>Ma`
- **v4_77M / minp_T0.7_p0.1:wiki_doc7**: `ans t <B1>Son <B1>conthale Suctor Shan is atithe ind <B1>it pin tin arale`  
  true: `a at <B1>Anton Malatinský Stadium <B1>on 24 April 2022, replacing <B1>Ma`
- **v4_77M / greedy:tinystories_doc3**: `n  a and a a and and <B1>ande ande and and and and <B1>are are are and`  
  true: `g, green thumb! <B1>It was so big, Bob had never <B1>seen one before. `
- **v4_77M / minp_T0.7_p0.1:tinystories_doc3**: `s  le al ale berea s athe <B1>arere tande the bare <B1>t ilind ande le`  
  true: `g, green thumb! <B1>It was so big, Bob had never <B1>seen one before. `
- **v4_77M / greedy:tinystories_doc7**: `!""<B6><EOS>f and and and and and <B1>and and and and and <B1>and and ande an`  
  true: `y closed the door <B1>and they played all day.<B2>
<B3>When they came bac`
- **v4_77M / minp_T0.7_p0.1:tinystories_doc7**: `!""<B6><EOS>wal The Lit a he the d he <B1>itand irer barorit <B1>ind aned. "
`  
  true: `y closed the door <B1>and they played all day.<B2>
<B3>When they came bac`

### Word completions, raw (first 3 windows of each source, k=2)

- **v4_610M**: Extract→`Expe`; Retailer→`Rerind`; largest→`lan`; history→`hin`; download→`do`; guessed→`gure`; allows→`alethe`; sizes→`sin`; baseline→`ban`; creativity→`cre`; following→`forererere`; Another→`An`; During→`Dutered`; Calif→`Can`; Laboratory→`Lan`; quite→`qurered`; built→`bur`; indeed→`inded`; criminal→`cre`; legislature→`le`; previously→`pre`; establish→`es`; determine→`dererereres`; factual→`fanderered`; memory→`merofofof`; provides→`pred`; drivers→`drererered`; primitives→`pro`; device→`de`; hardware→`hand`; Augusta→`Aur`; Cemetery→`Ce`; cadets→`can`; Joanna→`Jon`; January→`Jan`; While→`Whe`; there→`the`; asked→`as`; strong→`sthe`; robot→`ro`; could→`co`; beautiful→`beand`; friends→`fre`; played→`plout`; setting→`se`; touched→`to`; laugh→`lan`; after→`af`; things→`theand`; aisle→`aithe`; swing→`swand`; named→`nand`; loved→`lo`; smiled→`sme`
- **v4_77M**: Extract→`Exalin`; Retailer→`Re`; largest→`lallllin`; history→`hin`; download→`do`; guessed→`gure`; allows→`aliin`; sizes→`sin`; baseline→`bate`; creativity→`cre`; following→`fon`; Another→`An`; During→`Du`; Calif→`Can`; Laboratory→`Land`; quite→`quthe`; built→`bus`; indeed→`inded`; criminal→`cre`; legislature→`le`; previously→`pre`; establish→`es`; determine→`de`; factual→`fand`; memory→`me`; provides→`pre`; drivers→`dr`; primitives→`pr`; device→`de`; hardware→`han`; Augusta→`Aure`; Cemetery→`Ce`; cadets→`can`; Joanna→`Jon`; January→`Jand`; While→`Whe`; there→`the`; asked→`as`; strong→`sthe`; robot→`ro`; could→`co`; beautiful→`be`; friends→`fre`; played→`ple`; setting→`se`; touched→`to`; laugh→`la`; after→`af`; things→`the`; aisle→`ain`; swing→`swhe`; named→`nare`; loved→`lo`; smiled→`sm`

### Capacity drops (B=1 inference vs B=16 training batch, 16 windows)

| model | dropped (token, expert) pairs B=1 | B=16 | median NLL B=1 / B=16 | mean NLL (ok windows) B=1 / B=16 |
|---|---|---|---|---|
| v4_610M | 0.021 % | 0 | 2.4638 / 2.4591 | 2.4472 / 2.4469 |
| v4_77M | 0.086 % | 0 | 2.6470 / 2.6319 | 2.6840 / 2.6761 |

B=1 inference computes effectively the training function.

### NLL by grid chunk from a cold start (24 non-pg19 windows)

| model | c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 |
|---|---|---|---|---|---|---|---|---|---|
| v4_610M | 2.488 | 2.365 | 2.444 | 2.368 | 2.404 | 2.462 | 2.466 | 2.498 | 2.516 |
| v4_77M | 4.011 | 2.709 | 2.747 | 2.718 | 2.631 | 2.685 | 2.677 | 2.665 | 2.716 |

610M gets worse after chunk 3 as the carry grows. Mechanism and fix: `docs/LONG_CARRY.md`.

### Notes
- Word completion "first-step logits vs TF" differs in 8/432 items by up to 4.5 logits (others <= 3e-5). This is a
  discrete MoE top-k routing flip on float near-ties between the open-chunk and full-chunk passes. Capacity drops
  (0.02 %) do not explain it.
- pg19 means are dominated by two windows at 13-15 nats/id in every model, so medians are given.
