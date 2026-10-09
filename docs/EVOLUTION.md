# Fractus — evolution log (timestamped)

Canonical chronology of the **system**, not the mythology.  
Sister repos: `thefinalboss/fractus-cte`, `fractus-p0`, `fractus-vorax`, `fractus-datasets`.  
Last updated: **2026-08-28 03:50 UTC**.

Dates are UTC calendar days as recorded in docs, manifests, and HF `lastModified`. Token counts are **manifest snapshots**, not live GPU clocks.

---

## 2026-04 — seed (outside this repo)

- Preprint work on K3 elliptic CM / automorphy (separate mathematical line).
- Fractus as an idea: thought is continuous, accumulating, oscillatory — not one forward pass.

## 2026-07 — architectures in parallel

- **Fractus CTE** specified as multi-block continuous engine (linear attention + Kuramoto + PhaseRoutedMoE).
- **Palimpseste-max**: hypervector append-only memory, CPU, released on HF.
- **Fractus-Vorax** (late July / mid-August design): sealed CTE brain + organs that *write* knowledge (`.kn`) without training the 1B again.
- eQchip-Q / SOLARIS: other tracks, not this codebase.

## 2026-08-12 — training doctrine

- `docs/2026-08-12-fractus-chinchilla.md`: Chinchilla-style epoch stacking is **rejected** for this MoE + HV setup. One real pass + Vorax ingest.

## 2026-08-13 → 2026-08-15 — first long 4× GPU burn

- 4× RTX 5090, ~1k tok/s/GPU class.
- Loss from chaotic high values down through the 30s.
- Discoveries (logged, not marketing):
  - generation collapse to single-token attractors
  - train graph ≠ length-1 generate graph
  - mid-train surgery is possible without throwing the `.pt`
  - mean-merge of same-shape shards produces a usable brain
- Hugging Face Space attempt: torch pin / `audioop` / Gradio stack failures on Python 3.13.

## 2026-08-16 — outage

- Pod / filesystem incident. Resume from last HF `.pt`, not from token 0.
- Migration toward 8× GPU recipe. Docs: `BOOST_AND_OUTAGE.md`, `BOOST_B4_RECOVERY.md`.

## 2026-08-17 → 2026-08-18 — phase 2 + freeze + Vorax

- Phase-2 shards from `thefinalboss/fractus-datasets` (tokenized npy memmap).
- Anti re-ingest via manifest / `PHASE_SWITCH`.
- **Freeze snapshot:** ~15–17M tokens/GPU on ~430M-token phase-2 shards (~4% of that pass).
- Vorax repo stands up: compiler → traces / Hebbian / spawn organs, `:core` a sealed `.pt`.
- Decision: finish the pass later on next GPU budget; do not pretend 4% = trained.

## 2026-08-19 — Kuramoto bottleneck named

- Agent probe: degenerate circular mean, post-RK4 routing arc ~**25°/360°**, \(r \approx 0.01\) even on a fresh path; ~70% dead experts on probe windows.
- Not “wait for more tokens.” Prep: `GATE_TEMP`, `OMEGA_SCALE`, real LB in the loss, Kuramoto **in** autograd.
- Docs: `KURAMOTO_BOTTLENECK_AND_FIX.md`, `CPU_MINI_MERGE_AND_DIMENSIONS.md` (CPU minis ≠ 1B progress; mini→1B mean-merge is invalid).

## 2026-08-21 — wrong cluster

- QuickPod heterogeneous RTX (3090 / A4000 / 5060 Ti / A2000 12GB). 1B B=2 does not fit 12GB.
- Cluster deleted. Marketplace: **no 8×** offers at probe time. Credit stays on QuickPod; wait for identical 24GB+.

## 2026-08-22 → 2026-08-26 — open-heart kernels + x8

- Mission: faster phase-2 **without breaking living checkpoints**.
- Linear attention: masked einsum \(O(C^2)\) → cumsum / **chunked**.
- `chunked_cross_entropy`, `BLOCK_CKPT` (1B in ~16GB class), atomic ckpt writes, int32 memmap.
- Tests: attention/CE equivalence suite.
- White paper v2 + `arxiv/main.tex` landed on the repo.
- **X8_MANIFEST** (pushed): `status: x8_running`, start tokens ~**34.5–35.8M**/GPU.
- README of 26 Aug claimed ~1550–1600 tok/s/GPU aggregate ~12.6k on 8×5090 — treat as a run log, re-measure on the next pod.

## 2026-08-27 — P0 / v4 / DiffusionBlocks / lineage made explicit

- Package [fractus-p0](https://huggingface.co/thefinalboss/fractus-p0):
  - **P0 routing:** atan2 encode from \(h\) (not LayerNorm-mean≈0), phase carry into RK4, per-token MoE phases, Switch LB. **Not** open-heart-equivalent (dynamics change, shapes do not).
  - **v4 trainer:** SS ramp, anti-repeat λ=0.1, PREFIX unique@40 as language gate. CARRY length-1 is a separate hole.
  - CPU mini: PREFIX counts `0 1 2 → 4 5 6…`; CARRY locks `4 4 4…`.
  - **Fractus-native DiffusionBlocks** (Sakana/UTokyo ICLR 2026 mapped, not copied): one CTEBlock trained per step, noise band on residual, CE + MSE denoise, 0 new params. Mini 6/6 tests; `/σ²` denoise detonated last block — removed.
- Explicit statement: Fractus is a **linear-attention RNN**, not a transformer. Open items listed in the root README (decay on \(S\), DDP vs merge, MI probe on \(\bar\theta\), matched PPL, eval split).

---

## What “evolution” is not

- Not “AGI shipped.”
- Not “4.2B tokens digested” — a few percent of phase-2 plus earlier phase-1 shards.
- Not “mean-merge = 8× data efficiency.”
- Not “Kuramoto is solved” — P0 changes the drive; the MI test is still due.

## Next (operational)

1. Identical 24GB+ GPUs, x4 acceptable.
2. Resume from X8 / freeze manifest — never token 0 by accident.
3. Body = P0 + v4. Gate = PREFIX unique@40.
4. Optional: \(\gamma\) decay on \(S\); MI(\(\bar\theta\), token) vs tick; DDP if the fabric allows; matched transformer PPL on a held-out **source split**.
5. Vorax for knowledge writes. Do not re-pretrain the brain for facts.


## 2026-08-27 evening → 2026-08-28 03:50 UTC — SS on 8×5090 (this night)

Full log: `docs/2026-08-28-NIGHT-SS-X8.md`.

- Named the gap in production: CE teacher-force ≠ generation. unique@40 PREFIX NO-GO ~3.3 while tf on GPU0 was already ~5–8.
- Scheduled sampling **enabled and surviving** after checkpoint-metadata crash on the second backward.
- Carry snapshot/restore; SS dense CE; no CE-checkpoint when N≤ce_chunk.
- Volume 80→200 GB after disk-full trainer death.
- All 8 live checkpoints + hourly HF push to `thefinalboss/fractus-cte/checkpoints/x8run/`.
- B=4 + SS_RATE=1.0 live: ~27 GB, ~990 tok/s/GPU, ~40M tokens/GPU on 430M phase-2 shards.
- Speech not claimed. SS is the training-side fix; PREFIX unique@40 is the gate.

## 2026-08-28 16:35 PDT — Decode surgery I

Space output `Retail×32` proved the length-1 carry decode is an attractor, not a sentence.

`fractus/generate_aligned.py` now decodes with a 64-token causal window and masks the previous token. Trainers not stopped.

unique@32 after: 16 / 14 / 27 / 12. Short cycles left. Gate still PREFIX unique@40.

`docs/2026-08-28-DECODE-WINDOW.md`

