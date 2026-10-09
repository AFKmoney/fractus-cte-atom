# X8 Unified Checkpoint — Sandbox Live Operation ("Ciel Ouvert")

**Date:** 2026-09-02 (probe 22:56–23:12 UTC, operation 23:23–23:40 UTC)
**Host:** Arena sandbox — CPU fp32, 2 threads, ~1.25 tok/s (PREFIX) / ~16 tok/s (carry)
**Target:** `checkpoints/x8run/FRACTUS_1B_X8_MERGED.pt` — ts 2026-08-29T04:53:37Z,
8 QuickPod brains, `mean_float_params`, trainer `fast4gpu_boost_v4`,
103.3M phase-2 tokens/GPU (mean). d_model 1280, 16 blocks, **n_heads 20**
(manifest — README's "16" is stale), 128 experts, top-k 2.
**Runtime base:** `apply_kuramoto_routing_fix` applied at load (ω_std 0.0815→0.2993,
gate temp 1.0→2.5) — the QuickPod load procedure, reproduced.

> **TL;DR (FR)** — Le dernier unifié est NO-GO au speech gate (18.2/18.8 unique@40
> greedy, < 20), cohérent avec les derniers nombres GPU. L'opération ciel ouvert a
> ensuite fait *parler* le même cerveau en état continu : greedy pur = lock total
> (2.14/48) → chirurgie de décode (la leur) + fuite (S,z) 0.99 = fluide, zéro
> répétition (32.86 / 28.57). L'effondrement vit dans le régime greedy, pas dans
> la distribution (top-150 déjà riche). L'état continu sans oubli se rétrécit
> (chevauchement de vocabulaire 91 % puis 100 % sur 3 segments). Kuramoto : r = 0.003
> partout, le bruit de phase de `decode_surgery` est un no-op. Détails + repro
> ci-dessous.

---

## 1. The gate probe (README protocol, reproduced exactly)

unique@40, greedy, PREFIX (per-step reset + window-64 re-encode), previous-token
mask, no ban, no temperature, `no_grad`. Two passes on the same loaded engine:
**RAW** (checkpoint as published: `moe.temperature = 1.0` code default) and
**FIX** (+ open-heart load fix: gate temp 2.5, ω×4, noise 0 for determinism).

| Config | unique@40 (5 prompts) | mean | Gate (≥20, no lock) |
|---|---|---|---|
| RAW | 17, 19, 22, 19, 14 | **18.2** | **NO-GO** |
| FIX | 17, 16, 23, 28, 10 | **18.8** | **NO-GO** |

- Rust prompt: RAW 20/40, FIX 25/40 — no code structure in either.
- Confidence saturates 0.987/0.989 within 6 ticks on incoherent output.
- Carry-True (native continuous mode, 3 prompts × 32): collapses to hard
  2-token cycles (unique 4/3/2 of 32) — identical in RAW and FIX.
- Verdict consistent with last GPU probes (unique@32: 16/14/27/12 ≈ 17 mean):
  the 8-brain merge preserved the level, did not unlock the gate.
- **The merge did not regress anything; it is exactly where the README says it is.**

Full detail: `docs/X8_UNIFIED_PROBE.json`.

## 2. The operation: making it speak in continuous state

Battery: 7 prompts × 48 tokens, continuous (S,z) state (per-tick, no reset).
Metrics: unique, max_run (mono-token chain), alt2 (period-2 cycle), top5_share,
topic_hits (prompt keywords found), r (Kuramoto order, block 0), confidence.

| Step | Intervention | unique/48 (mean) | score | max_run |
|---|---|---|---|---|
| S0 | greedy carry (baseline) | **2.14** | −3.01 | 46.4 |
| S1 | `generate_with_surgery` (their live mode, verbatim) | **32.86** | **35.44** | 1.0 |
| S2a | S1 minus random escapes | 27.71 | 30.73 | 1.0 |
| S2b | S2a @ T 0.8 | 27.57 | 30.58 | 1.0 |
| S2c | S2a, ban 20→8 | **14.0** | 14.39 | 1.0 |
| S2d | residual leak 0.95 (vs 0.9) | 26.14 | 29.16 | 1.0 |
| S3a | S2a + **(S,z) leak λ=0.99** | 28.57 | **31.64** | 1.0 |
| S3b | (S,z) leak λ=0.98 | 28.29 | 31.33 | 1.0 |
| S3c | (S,z) leak λ=0.95 | 28.43 | 31.58 | 1.0 |
| S4 | S3b + PersistentMemory anchoring (recall/8, gain 0.3) | 28.14 | 31.18 | 1.0 |
| S5 | S3b + "meaningful" escapes (top-k off-ban, /8) | 28.0 | 31.04 | 1.0 |

score = mean_unique + 30·topic_hits − 10·top5_share. Best config: **S3a**.
Sample (S3a, "Once upon a time"):

```
time Catch whoDM ringsNA caregivers qualifier characteristic Git Prince 555 1924
stability 274ftimeDF wrongcy demoralPU fools Catch 1918DM disengpersonal caregivers…
```

Fluid, lock-free, continuous — but salad, not language (topic_hits capped at
~0.30 under every decode intervention: meaning is a training product, not a
decode product).

**The recipe that lifts S0 → S1/S3a on identical weights:**
residual leak 0.9 + 0.08 noise · ban window 20 · frequency penalty 1.2 ·
top-k 150 @ T 1.15 · MoE gate temp ≥ 3.0 · (S,z) leak 0.99.

## 3. The three deep findings

### 3.1 The collapse lives in greedy, not in the distribution
S0 (argmax): 48 consecutive identical tokens. S1 (top-150 sampling): 32.86
unique of 48, zero repeats. **The attractor sits in the top-1 landscape; the
top-150 support is already rich** — partly by architecture (von Mises soft
gates spread mass by design), partly by training (anti-copy surgery, LB).
Consequence: the unique@40 greedy gate measures the model's most pathological
regime. Recommended second birth metric: **"gate échantillonné"** —
unique@48, top-k 150, T 1.15, ban 20, continuous state. Today: 18.2 greedy
(NO-GO) vs 32.86 sampled (GO).

### 3.2 The continuous state, as built, is a memory of degradation
192-token long run (S3a): unique 36/192 on the surface, but the text is a
macro-cycle — a ~12-token motif repeated 3× (ban 20 breaks <20 loops, not 20+
motifs). Multi-prompt in ONE continuous state (3 segments × 48, no reset):
vocabulary overlap with prior segments **91 %** (segment 2) then **100 %**
(segment 3) — the longer the state lives, the more the vocabulary shrinks.
The (S,z) accumulator (`attn_S += outer.detach()`, unbounded, no decay) is a
shrink machine, not a memory. The leak (λ 0.99, half-life ~69 ticks) makes
continuity sustainable; **active forgetting** is the missing organ.
Continuity should be measured by vocabulary stability (overlap < 0.7 AND
unique ≥ 20 over 3+ segments), not by absence of reset.

### 3.3 Kuramoto is a spectator
r ≈ **0.0028** in every configuration, from collapsed greedy to fluid sampling.
The oscillators never synchronize: no regime, no dynamic cognitive mode.
Furthermore, **verified in code**: the phase noise in `decode_surgery.py` is a
**no-op** on the `tick_single` path — the MoE reads a fresh θ (derived from the
hidden state every tick), not the mutated `blk.kuramoto_phases`; only the
expert-hit counter reads that buffer. Everything that works in S1 comes from
the residual leak, the ban, the frequency penalty and sampling — not from the
dynamics. Fix: mutate the θ the MoE actually reads (or remove the illusion).

## 4. What did not work (honestly)

- **PersistentMemory** (S4): recall top-2 every 8 steps, gain 0.3 into the
  residual state → no measurable move (unique, topic, nothing). At 103M
  tokens/GPU the engine's own embeddings don't carry enough signal yet.
- **RAG + Rust ladder** (S7): retrieval works (correct top-3 snippets),
  injection works — **code_score 0.0** in all three variants (plain / RAG /
  code preseed). The only Rust inheritance: one "push" token (from
  "v.push(4)"). The "ingest a book, know Rust forever" thesis remains a
  training-stage protocol.
- **Random escapes** (theirs): +5.2 diversity, but 12.5 % of the text is
  random draws — visible noise, not speech.
- **Meaningful escapes** (S5): ≈ random escapes, minus their noise.

## 5. Prescriptions (ordered)

1. **Bounded memory + active forgetting on (S,z)** — the leak is the start;
   next step is selective decay (importance-weighted) or normalization of S/z
   so it codes a *mean*, not a sum. Control metric: vocabulary stability.
2. **Gate v2 = three birth metrics** — (a) greedy unique@40 (formal,
   unchanged); (b) sampled unique@48 top-150 @ T1.15 ban 20 (the body);
   (c) continuous-state stability (overlap < 0.7, unique ≥ 20 over 3
   segments). Both (a) and (b) must pass; today only (b) does.
3. **r must exceed ~0.3** — via training (LB not detached, Kuramoto under
   gradient), not via decode surgery (measured: zero effect on decode).
4. **Content comes from training** — topic_hits caps at ~0.30 under every
   decode intervention; the salad is the exact reflection of 103M tokens/GPU.

## 6. Scaling protocol (the "ça scale" test)

The body (decode surgery + bounded state) is level-agnostic: it operates on
dynamics, which are scale-invariant. Proof protocol — re-run
`scripts/probe_ciel_ouvert.py` at each checkpoint (430M, 3.4B, …):
plot (a) greedy unique@40, (b) sampled unique@48, (c) segment overlap vs
tokens/GPU. The body's value should *shrink* as the brain grows (each
intervention losing necessity is the scaling curve). Body = O(1) code;
brain = O(tokens). Floor raised (fluid babbling from early checkpoints),
ceiling unchanged (meaning scales with training).

## 7. Push manifest (verified 2026-09-02)

| File | Size | SHA-256 (first 16) | Content |
|---|---|---|---|
| `docs/X8_CIEL_OUVERT.md` | — | (this file) | this journal |
| `docs/X8_UNIFIED_PROBE.json` | 21,991 B | `8ddb271c06d2a204` | gate probe results (RAW/FIX, ids, texts, confidences) |
| `docs/OPERATION_CIEL_OUVERT_RESULTS.json` | 58,758 B | `c9933ee7885ffc7d` | full operation results (S0–S7) |
| `scripts/probe_ciel_ouvert.py` | 20,843 B | `65b97830e693ec05` | reproducible harness (`python scripts/probe_ciel_ouvert.py <ckpt> [--device cuda]`) |
| `docs/MASTER_RUN_LOG.md` | — | (appended) | timeline entry §3 |
| `docs/DISCOVERY_LOG.md` | — | (appended) | findings 3.x |

Verification performed before commit:
- both JSONs parse; all keys present; key figures cross-checked against the
  run log (18.2/18.8 NO-GO; 32.86/28.57; r 0.0028; overlap 0.909/1.0;
  code_score 0.0×3) — an early hand-rounded table (32.4/14.3) was corrected
  from the JSON, which is ground truth.
- checkpoint re-loaded and re-verified: 4,872,291,305 bytes, manifest fields
  match (n_heads 20, n_sources 8, ts 2026-08-29T04:53:37Z).
- harness compiles clean; identical logic to the run that produced the JSONs.
- environment note: fp32 is the only working inference dtype (state (S,z) is
  `torch.zeros` → fp32 by construction; mixed bf16/fp32 crashes in
  `_linear_attention_causal_cumsum`). QuickPod fp32 cost is nil (4.2 GB in 32 GB).


**Push record:** pushed to HF `main` 2026-09-03 as a single git commit (6 files) on top of HF `main` @ 813f471. HF main had diverged from the GH mirror base (2026-08-18) —
the two modified docs were re-merged on top of the current HF versions
(MASTER_RUN_LOG entry added as §11 after the 2026-08-28 decode-surgery-I
entry; DISCOVERY_LOG findings added as §7 after the 2026-08-22/23 optimization
section). The four new files are byte-identical to GH patch commit 29878e2.

## 8. Notes on the codebase (for the next training cycle)

1. **`kuramoto_fix` is called nowhere in the published code.** The public
   checkpoint loads at `moe.temperature = 1.0` (code default) — the RAW
   config. QuickPod applies the fix at load (the FIX config). The README
   should say: apply `apply_kuramoto_routing_fix` at load — one line,
   otherwise everyone probes the wrong runtime.
2. **Phase-noise no-op** in `decode_surgery.py` (finding 3.3): one-line fix
   (apply noise to the θ the MoE reads) or delete the dead code.
3. **README n_heads = 16** → manifest says 20 (20×64=1280). One number to fix.
4. `memory.py` docstring promises `engine.inject_memory(...)` — the engine has
   no such method (wiring is manual). Either implement it or fix the docstring.

## 9. Re-run verification on the HF repo (2026-09-03)

The full operation was re-run from a byte-verified snapshot of HF `main`
@ 5326f88 (92/92 files match the git tree sha1s; HF git does not support
partial clones from this sandbox, so the tree was materialized via the raw
API and each file verified against its blob hash), using the HF checkpoint
re-downloaded from the same path, same harness, seed 42.

- Results: `docs/X8_RERUN_ON_HF_REPO_2026-09-03.json`. Every S0–S7 metric is
  within ±1.0 of the 2026-09-02 run: S0 (greedy collapse) identical at
  2.14 unique/48, S3c identical at 28.43, cross-overlap trajectory
  0.0 → 0.952 → 1.0 (original 0.0 → 0.909 → 1.0), identical RAG retrieval,
  same conclusions.
- Two harness bugs, introduced when converting the 2026-09-02 harness into
  `scripts/probe_ciel_ouvert.py` (absent from the original run), were caught by
  this re-run and fixed: a leftover `_kf` module alias (NameError at load) and
  the repo root missing from `sys.path`. The fixed harness is the pushed one.
- Not bit-identical: the sandbox re-provisioned its torch build between the two
  runs, so fp32 argmax ties resolve differently and sampled sequences drift by
  ~1 token on average. Code, checkpoint, prompts, and seed are byte-verified
  identical across runs.
- Best-config selection flipped from S3a (leak 0.99) to S3c (leak 0.95):
  scores 31.64 vs 31.58 — a tie. The (S,z) leak recipe (0.95–0.99) stands.
