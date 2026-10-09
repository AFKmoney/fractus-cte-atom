# Small Fractus on CPU, merges, and dimension rules

**Updated:** 2026-08-20 07:12 UTC  
**Repo:** thefinalboss/fractus-cte

This note states **what is allowed** and **what is not magic** when training many small Fractus models on CPU and merging them — especially versus the frozen **Fractus-1B** phase-2 run.

---

## 1. Two tracks (do not confuse them)

| Track | Hardware | What it advances |
|-------|----------|------------------|
| **Fractus-1B phase2** | 8× GPU (e.g. RTX 5090) | Token counter on the ~3.44B / ~4B corpus, 1B weights, production brain |
| **Mini-Fractus lab** | CPU (or 1 small GPU) | Ideas, routing experiments, decode, merge *practice*, diversity of seeds |

**CPU ≠ 8×5090.**  
Training many small models on CPU **does not** move the 1B phase-2 progress (~4% of the pass at freeze). It does **not** replace the multi-day 1B digestion.

Both tracks are valid. Only one advances the **1B corpus counter**.

---

## 2. Mean-merge: when it is clean

Fractus checkpoints can be **mean-merged** when they share the **same tensor layout**.

### Clean merge (supported pattern)

- Same architecture hyperparameters (see §3).
- Same parameter names and shapes in `state_dict` / `model_state`.
- Average floating-point tensors across N runs (optionally keep non-float buffers from one parent).
- Used in production for **8× GPU** → `FRACTUS_1B_*_MERGED.pt`.

```text
merge(gpu0, gpu1, …, gpu7)  →  one 1B brain    ✅ same shapes
merge(mini_a, mini_b, mini_c) → one mini brain ✅ if identical mini config
```

### Not automatic

```text
merge(mini_64d, fractus_1b_1280d)  →  ❌ shape mismatch
```

A **badly sized** mini-Fractus **cannot** be mean-merged into the 1B. There is no free reshape that preserves a trained dynamical system.

---

## 3. Dimension contract (1B reference)

Production **Fractus-1B** (phase2 freeze / boost) uses approximately:

| Hyperparameter | 1B value |
|----------------|----------|
| `d_model` | 1280 |
| `n_layers` | 16 |
| `n_heads` | 20 |
| `d_head` | 64 |
| `n_oscillators` | 16 |
| `coupling_rank` | 8 |
| `n_experts` | 128 |
| `top_k` | 2 |
| `expert_d_ff` | 2048 |
| `siren_rank` | 64 |
| `vocab_size` | 50257 (GPT-2 BPE) |

**Rule:** any checkpoint you intend to mean-merge with **this** 1B must match these shapes **exactly**.

Mini configs for CPU lab **must document their own table**. Example lab tier (illustrative only — choose what fits RAM/CPU):

| Hyperparameter | Example mini | Mergeable into 1B? |
|----------------|--------------|--------------------|
| `d_model` | 256 or 512 | No |
| `n_layers` | 4–8 | No (unless equal 16) |
| `n_experts` | 32 | No (unless equal 128) |
| same as 1B table | full 1B | Yes, but not “small” on CPU |

**Same-arch mini swarm:** all minis share **one** fixed mini table → merge **among themselves** ✅  
**Mini → 1B:** not by mean-merge ❌

---

## 4. Mini → 1B: what would be required (not implemented as magic)

If you ever want knowledge or structure from minis **into** the 1B, you need an explicit scheme, for example:

| Scheme | Idea | Status |
|--------|------|--------|
| **Mean-merge** | Average aligned tensors | Only identical shapes |
| **Interpolation / resize** | Pad or project layers | Research; can destroy dynamics |
| **Distillation** | 1B trains to match mini outputs on data | Possible; costs compute and a loss design |
| **Expert transplant** | Copy MoE expert slices if widths match | Only if expert ranks/dims match |
| **Vorax / pools** | Do **not** merge weights; write knowledge via organs | Preferred for *facts* without touching 1B shapes |

**Default recommendation while waiting for GPU pay cycle:**

1. Keep **1B freeze** as the production brain (HF).  
2. On CPU, train **same-config minis**, merge **minis with minis**, probe routing / decode / Kuramoto fixes.  
3. Put durable knowledge into **Fractus-Vorax** (hash → pools), not into a fantasy mini→1B average.  
4. On next 8×GPU run: resume 1B from `FROZEN_RESUME_MANIFEST` + Kuramoto prep (`docs/KURAMOTO_BOTTLENECK_AND_FIX.md`).

---

## 5. Practical CPU lab recipe

```text
1. Fix one MINI_CONFIG (write the table in the run README).
2. Train N seeds or N data shards on CPU (short seq, small batch).
3. Mean-merge the N minis → mini_merged.pt
4. Eval routing entropy, gen probes, Kuramoto r / dead experts on the mini.
5. Document results; do not claim phase2 1B progress.
```

Optional: apply the same **Kuramoto routing fix family** (wider ω, gate temperature, LB in loss) on minis to validate the fix cheaply before the 1B resume — still not a substitute for applying it on the real 1B freeze at resume time.

---

## 6. One-page rules

1. **CPU mini training ≠ 1B phase2 progress.**  
2. **Mean-merge only for identical architecture shapes.**  
3. **Mini → 1B mean-merge is invalid** without a separate, explicit transfer method.  
4. **Document every config table** next to every checkpoint family.  
5. **Production path** remains: freeze 1B → Kuramoto prep → resume phase2 on multi-GPU.  
6. **Knowledge path** without shape wars: Vorax ingestion / organs.

---

## Related docs

- `docs/HOW_FRACTUS_IS_TRAINED.md` — multi-GPU phase2 recipe  
- `docs/KURAMOTO_BOTTLENECK_AND_FIX.md` — routing fix before next 1B resume  
- `docs/NEXT_TRAINING_CHECKLIST.md` — pod checklist  
- `docs/FROZEN_STATE.md` / `checkpoints/FROZEN_RESUME_MANIFEST.json` — resume offsets  
- Fractus-Vorax — sealed 1B + write-side knowledge (no requirement to merge mini weights into 1B)
