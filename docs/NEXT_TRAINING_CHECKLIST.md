# Next training checklist (post-freeze + Kuramoto fix)

1. New pod, torch matching GPUs (5090 → recent CUDA build).
2. Pull `thefinalboss/fractus-cte` + phase2 shards from `fractus-datasets` `tokenized/phase2/*.npy`.
3. Download freeze ckpts `fractus_1b_gpu0..7.pt` (or from FROZEN merge then split only if needed — prefer per-GPU freeze).
4. **Apply Kuramoto fix before long run:**
   ```bash
   export GATE_TEMP=2.5 OMEGA_SCALE=4.0 OMEGA_NOISE=0.01 LB_COEF=0.05
   python scripts/prep_kuramoto_fix_resume.py
   ```
5. Resume with `START_TOKEN` from `checkpoints/FROZEN_RESUME_MANIFEST.json` per GPU (not 0).
6. Train: B=2, SEQ=128, LR=7e-4, SS_RATE=0.25, memmap phase2 shards, `loss = ce + LB_COEF * lb`.
7. Smoke: log omega std, expert load entropy; confirm TF still drops.
8. Hourly HF push of 8 ckpts + merge.

## Optimized relaunch (fractus-opt, 2026-08-24)

After the Kuramoto fix, swap in the proven-equivalent optimized path
(repo **AFKmoney/fractus-opt**, guide: its `docs/OPTIMIZATION_2026-08-22.md` §3):

1. Replace `fractus/nn/attention.py`, add `fractus/nn/ce.py`,
   replace `fractus/continuous_engine.py`, use `scripts/fast4gpu_boost_v2.py`.
2. First relaunch with v1 settings (`BATCH=4 CE_CHUNK=0`) → ema_tf must
   continue its curve exactly (real-conditions equivalence check).
3. Escalate: `CE_CHUNK=2048` → `FRACTUS_ATTN_IMPL=chunked` → `COMPILE=1`
   + raise `BATCH`. Validate tok/s + VRAM at each step; read live tok/s
   from stdout to update the time-to-finish table in
   `HOW_FRACTUS_IS_TRAINED.md` §8.
4. Budget: one full phase-2 pass ≈ 425–430M tok/GPU remaining →
   **4.5–5.5 days at baseline rate** (see HOW_FRACTUS_IS_TRAINED.md §8).

See `docs/KURAMOTO_BOTTLENECK_AND_FIX.md` and `docs/HOW_FRACTUS_IS_TRAINED.md`.
