# Corpus v4: real-language Atom corpus

Date: 2026-10-18, ~18:15-19:40 ET. The build ran on the box. The pod only received the finished shards (scp, bandwidth-limited). The v3 trainer was not touched.

**Why.** The v1-v3 corpus is 99.83% slot-filled templates, with only 243,575 distinct 16-grams in 4.41B ids (`docs/CORPUS_TEMPLATE_AUDIT.md`). The model has never seen real language. v4 replaces it with real text. A 100M-id subsample of the old corpus is kept so the skills vocabulary is not dropped all at once.

## Where it is

- Pod: `/workspace/atom_corpus_v4/` holds `train_000.i16` .. `train_006.i16`, `val.i16`, `manifest.json` and `SHA256SUMS`. All 8 sums were checked OK on the pod.
- HF: `thefinalboss/fractus-cte-atom`, folder `atom_corpus_v4/` (commit `150fef55`, then the manifest commit).
- Box (build workspace, not in git): `/workspace/v4build/`.
- Copies in git: `docs/corpus_v4_data/` (`manifest.json`, `SHA256SUMS`, `verify.txt`, `cpu_val_baseline_v3_step85000.json`).

## Encoding (identical to what the trainer has always read)

- Tokenizer `atom-fractus-v1` (`fractus/atom_tokenizer.py`): Atomizer, `pack_mode=linguistic`, `max_span_bytes=32`.
- Ids: 0-255 raw UTF-8 bytes, 256 PAD (never used), 257 BOS, 258 EOS, 259-265 boundary markers (259 whitespace, 260 punctuation, 261 newline, 262 class_transition, 263 max_span, 264 `<B6>` end-of-stream).
- A document is `BOS, payload bytes with a boundary id after each span, EOS`. Documents are concatenated with no padding and no other separator. This is the same layout as `atom_corpus.i16`: every EOS is followed by the next BOS.
- Files are raw little-endian int16 (`.i16`). The trainer memmaps them as before.
- The final `<B6>` (264) appears only when the last span is still pending at the end of the text. If the last byte already closed a span (for example a final "." after a long span), the reference tokenizer emits 260 and no 264. That is why 87.7% of v4 docs end in `264 258`. The old corpus always ended in `264 258` only because its templates did.
- `scripts/atom_fast_ids.py` produces the ids without the 320-dim torch features (4.7 MB/s per core against 0.23 MB/s for the reference). It was checked id for id against `AtomFractusTokenizer.encode`:
  - 3,009 synthetic edge-case strings (UTF-8 multibyte, emoji, CRLF, long words, digits/alpha transitions);
  - real cleaned docs: 150 FineWeb-Edu, 150 Wikipedia, 150 TinyStories, 40 PG-19 (first 20k chars each);
  - 2 old-corpus docs decoded and re-encoded.

  All were identical (`docs/corpus_v4_data/verify.txt`).

## Sources and mix

| source | HF dataset (what was read) | train ids | share | docs | val docs / ids / text bytes |
|---|---|---|---|---|---|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` sample-10BT, 1 of 14 parquet files (`011_00000`), rows in seeded random order | 700,003,079 | 49.99% | 143,737 | 411 / 2,081,341 / 2.00 MB |
| Wikipedia en | `wikimedia/wikipedia` 20231101.en, 2 of 41 files (`00034`, `00026`) | 250,000,472 | 17.85% | 92,985 | 622 / 1,573,099 / 1.50 MB |
| TinyStoriesV2 | `roneneldan/TinyStories` `TinyStoriesV2-GPT4-train.txt`, streamed, read only up to the target | 200,000,552 | 14.28% | 238,401 | 621 / 521,939 / 0.50 MB |
| PG-19 | `emozilla/pg19` (deepmind/pg19 as parquet), 2 of 23 files (`00018`, `00014`) | 150,085,320 | 10.72% | 517 books | 6 books / 1,422,461 / 1.36 MB |
| old corpus | `atom_corpus/*.i16`, ~0.91M ids of whole docs from each of the 110 shards (seeded offset, HTTP range reads) | 100,206,033 | 7.16% | 27,623 | none |
| **total** | | **1,400,295,456** | | **503,263** | **1,660 / 5,598,840 / 5.37 MB** |

- About 1.03-1.05 ids per text byte. The linguistic packer makes long spans, so boundary ids are about 4% of all ids.
- Parquet sources were read as a few whole files (`hf_hub_download`), never the full dump: 1 + 2 + 2 files, about 3.9 GB of the ~47 GB of these datasets. Streaming the large PG-19 rows over HTTP kept getting cut ("peer closed connection"), and the downloads resume. The files are cached in `/workspace/v4build/dl/` on the box and can be removed once the user approves.

## Cleaning

- CRLF/CR become LF. NUL is removed. Trailing spaces at line ends are stripped. Three or more newlines become two. Leading and trailing whitespace is stripped.
- PG-19 is re-flowed: the hard ~70-char line wraps inside a paragraph become spaces, and blank-line paragraph breaks are kept. Books are capped at ~400k chars, cut at a paragraph boundary.
- Docs are dropped when they are too short: fewer than 300 chars (FineWeb, Wikipedia), 150 (TinyStories) or 2,000 (PG-19). They are also dropped when they contain U+FFFD (bad UTF-8 upstream). Drops: FineWeb 146, Wikipedia 11,149 (stubs), TinyStories 13, PG-19 2.
- Exact dedupe uses a blake2b-64 hash of the cleaned text, within each source and then globally, with val hashes first so no val doc also sits in train. Exact duplicates removed: FineWeb 51, Wikipedia 3, others 0, cross-source 0.

## Interleaving and shards

- All 503,263 train docs were shuffled at document level across sources (numpy `default_rng(0)` permutation). They were then written in that order into 7 shards of about 200M ids each, cut at doc boundaries (the last shard absorbs the remainder).
- The trainer reads `train_000..006` concatenated in name order as one stream (`CORPUS=/workspace/atom_corpus_v4`). With B=16, each lane is a contiguous 87.5M-id stretch of the shuffled stream, so every lane sees all sources mixed.
- Val (`val.i16`, 5,598,840 ids, 1,660 docs, 5.37 MB of text) holds the first held-out docs of the four real sources, shuffled. It has no old-corpus docs and is not TinyStories-only.

## Diversity: distinct 16-grams

16-grams of ids that do not cross a doc start were counted with a 64-bit hash, on 100M-id samples. This is the same definition as the audit.

| sample (100,000,000 ids) | distinct 16-grams |
|---|---|
| v4 (first 100M ids of `train_000`, all sources mixed) | **75,355,484** |
| old corpus (the 100M-id subsample from all 110 shards) | 182,388 |
| old corpus, all 4,410,941,912 ids (audit) | 243,575 |

On equal-size samples v4 has **413x** more distinct 16-grams than the old corpus, and 309x more than the whole 4.41B-id old corpus. Per real source, on the first 10M ids: FineWeb 9,245,800, PG-19 9,358,679, Wikipedia 8,489,230, TinyStories 6,724,692.

## Starting point: v3 on real text (CPU check)

`scripts/corpus_v4/cpu_val_baseline.py` ran v3 step_00085000 (4,810,301,440 ids, fp32, chunked attention, the training forward). It used 4 lanes that each start on a BOS, with a per-row carry reset at BOS, over the first 8x128 ids:

| fineweb | wiki | tinystories | pg19 | old templated |
|---|---|---|---|---|
| 19.8 | 23.6 | 24.7 | 425 | 2.85 nats/id |

Uniform over 266 ids would be 5.58 nats/id. On real text the v3 model is confidently wrong, and the loss grows along long docs (PG-19). Consequences for the v4 run:

- The first steps will have very high CE and large gradient norms. `GRAD_CLIP=1.0` still bounds each update.
- With v3's guards, the switch itself would trip the rollback (ema_tf > 4.0 for 1,000 steps), the grace-period skip threshold, and the good/ criterion. `run_v4.py` relaxes them, with every value overridable by env: rollback at ema_tf > 30 for 3,000 steps, a 2,000-step GN grace with an absolute threshold of 1e6, and good/ at ema_tf < 4.5.
- Watch the first VAL lines and ema_tf during the first ~30 minutes.

This sample is small (4x1,024 ids per source), so it shows the order of magnitude only.

## Trainer v4 (prepared, not launched)

- `scripts/fast4gpu_atom_v4.py`: a copy of v3 with the same math, SS, carry, LR, guards and checkpoints. Env-gated additions:
  - a directory `CORPUS` (multi-shard memmap, no copy);
  - `CORPUS_SWITCH=1`, which allows `START_TOKEN=0` on a new corpus while resuming weights and optimizer;
  - `IDS_BASE`, which sets `ids_total` in the checkpoint and manifest;
  - periodic held-out loss: `VAL_CORPUS`, `VAL_EVERY`, `VAL_CHUNKS`. B lanes start on a BOS with a zero carry, teacher-forced, no_grad, and the training carry is saved and restored around the eval. It logs `VAL ce=` lines and `OUT_DIR/val_log.jsonl`, and runs once before step 1.
- `scripts/run_v4.py` (installed on the pod as `/workspace/run_v4.py`):
  - refuses to start while the v3 trainer is alive;
  - hard-links the current `/workspace/ckpt_v3/good/` (weights + AdamW state) into `/workspace/ckpt_v4_from_v3/`;
  - `IDS_BASE` = its ids_processed;
  - B=16, SEQ=128, SS_RATE 0.35, SS_TAIL_IDS 8, ss_p 0.85 (v3's effective value, since its ramp had finished), EPOCH_WRAP 1;
  - LR 2.35e-5 (v3's LR at its last good checkpoint), cosine to 2.5e-6 over the planned steps, warmup 500;
  - VAL every 2,500 steps on 262,144 ids (~1-2% of wall time);
  - output in `/workspace/ckpt_v4`, `train_v4.log`, `runner_v4.log`;
  - `PLAN_IDS` sets the length of the run (default: one pass, 1,400,295,456 ids).
- `scripts/push_good_atom_v4.py` and `scripts/launch_v4.sh` (also on the pod): the v3 pusher and launcher with v4 paths, pushing to `checkpoints/atom_v4`.
- Launch: stop runner_v3, trainer_v3 and push_good_v3 first, then `bash /workspace/launch_v4.sh`.
- Disk: the pod had 32 GB free with the v4 corpus on it. Each v4 save writes ~15.4 GB, and v4's own rotation only removes v4 step dirs. v3's `ckpt_v3/` stays as it is.

## Time and cost at ~10.6k ids/s, $0.525/h

| plan | ids | hours | cost |
|---|---|---|---|
| one full pass | 1,400,295,456 | 36.7 h | $19.27 |
| `PLAN_IDS=1100000000` | 1,100,000,000 | 28.8 h | $15.13 |
| `PLAN_IDS=1250000000` | 1,250,000,000 | 32.8 h | $17.20 |

Add ~1-2% for VAL. Vast credit was $18.08 at 19:34 ET (Oct 18), and it falls by $0.525/h while v3 keeps running. A full pass does not fit. `PLAN_IDS` around 1.1B leaves margin for storage, the switch-over and evaluation.

## Effect on v3 during the build

Nothing ran on the pod CPU except a bandwidth-limited scp and one niced `sha256sum`. Instantaneous v3 rate (position delta over 200-400 s; the log granularity is 409,600 ids):

| measurement | ids/s |
|---|---|
| before the build | 10,240 |
| during the build | 10,214 |
| during the copy | 10,922 |
| after | 10,240 |

The cumulative rate printed by the trainer stayed at 10,645. There was no measurable effect.

## Reproduce

```
python scripts/build_corpus_v4.py verify
python scripts/build_corpus_v4.py fetch --source fineweb --target-ids 700000000 --val-bytes 2000000 --workers 4
python scripts/build_corpus_v4.py fetch --source wiki --target-ids 250000000 --val-bytes 1500000 --workers 4
python scripts/build_corpus_v4.py fetch --source tinystories --target-ids 200000000 --val-bytes 500000 --workers 2
python scripts/build_corpus_v4.py fetch --source pg19 --target-ids 150000000 --val-bytes 1000000 --workers 3
python scripts/build_corpus_v4.py old --target-ids 100000000
python scripts/build_corpus_v4.py mix --shard-ids 200000000
python scripts/build_corpus_v4.py stats --sample-ids 100000000
```

All stages use seed 0 and `nice -n 19`. Each fetch stage writes `<source>.done.json` and is skipped when run again.
