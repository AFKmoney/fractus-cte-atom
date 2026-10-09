# Fractus Discovery Log — Bugs, Optimizations, Emergent Features

**Updated:** 2026-09-02 (adds sandbox live-operation discoveries on the X8 unified checkpoint)

This document records what we found *while* running Fractus-1B — things that go beyond the original design notes. Empirical, not marketing.

---

## 1. Critical training bugs found (and fixed)

### 1.1 Kuramoto was not learning during training

**Symptom:** `omega` stayed near init (±0.05), phase order parameter r ≈ 0.01–0.03, soft dynamics.

**Root cause:** In `CTEBlock.tick_chunk_core`, Kuramoto integration ran under `torch.no_grad()`:



Comment in code even said "clock, not learned". So CE loss never reached `omega` / coupling.

**Fix:** Remove `no_grad` around Kuramoto; keep phase *state* detached for carry, keep differentiable `theta` for MoE routing so parameters receive gradients.

**Status:** Fixed in `fractus/continuous_engine.py` (surgery 2026-08-16).

### 1.2 Load-balance loss was computed then thrown away

**Symptom:** Offline probe showed ~70% experts dead (90+/128), only 2–3 experts active per block.

**Root causes:**
1. `tick_chunk_train` did `total_lb + lb.detach()` — no gradient through LB
2. `fast4gpu.py` optimized **only** cross-entropy — never added `lb_loss` to the loss

**Fix:**
- Keep LB in the graph
- `tick_chunk_train` returns `(logits, lb_loss)`
- Surgery trainer: `loss = CE + 0.02 * lb`

**Status:** Fixed; live `lb≈14` on all GPUs after resume.

### 1.3 Probe false alarm: "all phases are zero"

**Symptom:** A probe script reported all Kuramoto phases at 0.0.

**Root cause:** Script called `reset_thought()` before measuring — which zeros state buffers.

**Reality in checkpoints:** phases nonzero, std ≈ 1.8 across all 16 blocks × 4 GPUs.

**Lesson:** Never diagnose dynamical state after an intentional reset.

---

## 2. Optimizations discovered in production

| Optimization | What we learned |
|--------------|-----------------|
| 4-GPU independent shards + mean-merge | Works; unified `.pt` generates; different lexical attractors than single shard |
| Resume from exact token offset | Manifest-driven `start_token` preserves progress across pod reboot |
| Gate temperature ↑ (1.0 → 2.5) | Softens von Mises routing; more experts can enter the top-k mix |
| Omega scale ×4 on resume | Restores phase-rate diversity without wiping weights |
| `tick_vec` multimodal path | Vision patches can drive CTE without touching token embedding |
| CPU eyes prototype (CIFAR) | Small CTE+PatchEmbed learns real images offline while 1B trains on GPU |

---

## 3. Features that emerged beyond the original plan

### 3.1 Operable / open-heart model
Weights (`.pt`) + body (`fractus/` code) are separable. We can:
- merge brains
- change routing temperature
- inject LB pressure
- add vision front-end
without a full retrain from zero.

### 3.2 Infinite-ish checkpoint fusion (same architecture)
Compatible checkpoints can be mean-merged and trained again:

```
train → merge → train → merge → ...
```

Constraint: same shapes (d_model, layers, experts, etc.). Divergent merges can soup skills; shard-merge is the proven path.

### 3.3 Mid-training generation behavior
At loss ~33→25, generation pipeline works but outputs **word-level repetition collapse** (real tokens stuck in loops: "Colorado", "Population", "Fate", "thinks"…). Documents that Fractus emits lexical tokens before coherent sentences.

### 3.4 Parallel modality track
Text 1B digestion and vision prototype can run in parallel (GPU text + CPU eyes) without stopping the main run.

### 3.5 Routing pathology as first-class debug target
Expert-hit histograms + phase order parameter `r` are necessary metrics. Loss alone hides "model learns with 3 experts".

---

## 4. Live metrics after routing surgery (resume)

Resume offsets preserved from pre-crash run (~176–187M tokens/GPU).

| GPU | Resume start | Snapshot tokens | CE loss | lb |
|-----|--------------|-----------------|---------|-----|
| 0 | 176,281,600 | 176,614,400 | 36.5 | 14.026 |
| 1 | 186,137,600 | 186,444,800 | 12.4 | 14.024 |
| 2 | 186,854,400 | 187,212,800 | 34.7 | 14.026 |
| 3 | 183,833,600 | 184,166,400 | 41.0 | 14.027 |

Post-surgery CE can spike briefly then fall (routing distribution shift). GPU1 recovered fastest into the teens.

---

## 5. What this means for the Fractus thesis

Fractus is not only "another 1B trained on shards". The run forced discovery of:

1. **Silent non-learning** of the phase clock under `no_grad`
2. **Silent expert death** without LB in the loss
3. **Composable checkpoints** as a workflow
4. **Operability** (surgery without discarding digestion)
5. **A body that speaks before the brain converges** — decode-level surgery on the
   same weights turns 2.14 unique/48 (greedy carry) into 32.86 unique/48 fluid,
   repeat-free speech. The collapse is a sampling-regime attractor, not a weights
   defect: the thesis holds above the training gate (see §7).

The architecture does more than the first README described — because production training exposed the dynamical bottlenecks.

---

## 6. Discoveries from the optimization session (2026-08-22/23)

### 6.1 The "attention matmul" was a disguised cumsum

The production kernel computed causal sums via
`einsum("tj,bjpq->btpq", tril_mask, outer)` — an O(C²·dH²) masked contraction
that IS mathematically a prefix sum. Replaced by `torch.cumsum(outer, dim=1)`
(O(C·dH²)), proven bit-close to the kept einsum reference (forward, gradients,
carried S/z). Lesson: profile semantics, not just FLOPs — the mask+matmul form
hid a linear-scan structure for months.

### 6.2 Discrete routing makes strict equivalence claims wrong at block level

topk over von Mises gates is DISCRETE: a float32 rounding difference near a
gate tie flips expert choice for isolated tokens (measure-zero boundary),
and depth amplifies it. Honest framing adopted: continuous stages are proven
bit-close; full-block equality is asserted statistically (median at rounding
scale, mismatch fraction bounded). Tests encode exactly this.

### 6.3 Two constructions under one seed have DIFFERENT weights

`manual_seed(s); A = Model(); B = Model()` gives two different networks — the
RNG stream continues across constructions. Any A/B equivalence test must clone
weights (`load_state_dict`) instead of re-seeding. This masqueraded as a
"routing divergence" for half a session before being identified.

### 6.4 CPU commit-limit segfaults are data, not noise

At 1B shapes the reference kernels materialize (G, C, dH, dH) and hit the
native commit limit on a RAM-constrained host (segfault, not MemoryError).
The benchmark harness now runs each cell in its own subprocess so one crash
documents itself without killing the table — and the crash boundary itself
measures the memory win of the chunked kernel.

---

## 7. Sandbox live operation on the X8 unified checkpoint (2026-09-02)

External CPU sandbox (fp32) loaded `FRACTUS_1B_X8_MERGED.pt` and ran the full
README gate protocol + a live speech operation on the continuous state.
Empirical findings:

### 7.1 The generation collapse is a greedy-regime attractor, not a distribution property

**Symptom:** greedy carry (native continuous mode) = 48 consecutive identical
tokens (2.14 unique/48). Same weights, top-150 sampling @ T 1.15 with ban 20:
32.86 unique/48, zero repeats, no 2-cycles.

**Root cause:** von Mises soft gates (architecture) + anti-copy/LB training
spread mass across the top-150 support; the argmax landscape still holds short
repetition attractors. The gate's greedy unique@40 measures the most
pathological regime of the model.

**Status:** Measured. Action: add a second birth metric — sampled
unique@48 (top-k 150, T 1.15, ban 20, continuous state). Gate v2 = greedy
(formal) + sampled (the "body") + state stability. See
docs/X8_CIEL_OUVERT.md §5.

### 7.2 The continuous (S,z) state is a memory of degradation, not a memory

**Symptom:** in one unreset thought stream, vocabulary overlap with earlier
segments: 91 % (segment 2), 100 % (segment 3). 192-token run = a ~12-token
motif macro-cycle (ban 20 breaks <20 loops, not 20+ motifs).

**Root cause:** `attn_S += outer.detach()` is unbounded, no decay — the readout
`q·S / q·z` becomes a function of the whole history and the effective
vocabulary shrinks with state age. A multiplicative leak on (S,z)
(λ 0.99, half-life ~69 ticks) makes continuity sustainable but does not invert
the shrinkage.

**Status:** Measured. Action: active forgetting (selective/importance-weighted
decay, or normalize S/z to code a mean). Continuity metric: overlap < 0.7 AND
unique ≥ 20 over 3+ segments.

### 7.3 Decode-surgery phase noise is a no-op on the tick path

**Symptom:** r ≈ 0.0028 in every configuration; perturbing
`blk.kuramoto_phases` never changes outputs.

**Root cause:** `CTEBlock.tick_single` computes a fresh θ from the hidden state
each tick and feeds it to the MoE; the stored `kuramoto_phases` buffer is read
only by the expert-hit counter. `decode_surgery.generate_with_surgery` mutates
the dead buffer.

**Status:** Verified in code. Action: apply phase noise to the θ the MoE
actually reads (one line) or delete the dead code. Kuramoto order must be
raised in training (LB under gradient), not in decode.

### 7.4 Publish gap: the load fix is not wired into the public path

**Symptom:** a fresh load of the public checkpoint runs at
`moe.temperature = 1.0` (code default) — the RAW config — while QuickPod runs
at 2.5 after `apply_kuramoto_routing_fix`. Nothing in the public code calls
the fix; `memory.py` docstring promises an `engine.inject_memory()` that does
not exist.

**Status:** Measured (probe RAW = 1.0 vs FIX = 2.5). Action: README line —
"apply `apply_kuramoto_routing_fix` at load"; implement or de-document
`inject_memory`.

---

---

*Keep this log updated when new probes, merges, or surgeries land.*
