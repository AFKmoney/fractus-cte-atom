# Kuramoto bottleneck and fix (next training)

**Updated:** 2026-08-20 04:47 UTC  
**Status:** DIAGNOSED + PREP READY (apply on pod before resume)

## What was measured

Agent probe (2026-08-19) on Kuramoto tensors (mmap, low RAM):

1. **Degenerate circular mean** of the phase fan — atan2(0,0)-style collapse on a fresh model path.
2. **Post-RK4 routing arc ~25° / 360°** — with 128 experts spaced ~2.8°, only a thin slice of the expert ring is reachable → structurally dead experts.
3. **Order parameter r ≈ 0.0101** on fresh dynamics — same order as prod logs (0.01–0.03).
4. **Not only probe no_grad** — soft dynamics / narrow omega band are in the oscillator setup itself.

Matches earlier mid-train observations: phases stuck, omega std ~0.02–0.03, high dead-expert rate on probe windows.

## Will more tokens alone fix it?

**Unlikely as the sole plan.**

Finishing phase2 without opening the routing mostly trains the already-open expert subset. Dead experts stay dark.

Training helps **after** the route can move: wider omega, softer gates, real load-balance in the loss, Kuramoto not under no_grad in tick_chunk_train.

## Fix recipe (next pod)

### A. On checkpoint load (existing freeze weights)

```bash
export GATE_TEMP=2.5
export OMEGA_SCALE=4.0
export OMEGA_NOISE=0.01
export OMEGA_CLAMP=0.5
export LB_COEF=0.05

python scripts/prep_kuramoto_fix_resume.py
# backups: checkpoints/fractus_1b_gpu{i}_pre_kuramoto_fix.pt
# writes:  checkpoints/fractus_1b_gpu{i}.pt + KURAMOTO_FIX_APPLIED.json
```

`fractus/kuramoto_fix.py` → `apply_kuramoto_routing_fix(engine)`:
- sets MoE temperature
- omega ← clamp(omega * scale + noise)

### B. During train (must stay true)

| Requirement | Why |
|-------------|-----|
| loss = ce + LB_COEF * lb | LB not detached |
| Kuramoto in autograd path | omega can keep adapting |
| GATE_TEMP applied each load | runtime attr, not always in ckpt |
| START_TOKEN from FROZEN_RESUME_MANIFEST.json | do not restart phase2 at 0 |

### C. New models only

KuramotoLayer default omega init widened: uniform(-0.35, 0.35) was (-0.05, 0.05).

## Success checks (after fix + short smoke)

1. omega.std() per block clearly above pre-fix (logged in KURAMOTO_FIX_APPLIED.json).
2. Offline probe without resetting live phases: higher expert-usage entropy, fewer zero-load experts.
3. Routing arc / phase diversity up vs ~25° regime.
4. TF loss still trends down (fix must not destroy the freeze).

## Relation to freeze

Freeze remains valid: ~15–17M tokens/GPU on phase2, weights on HF.  
Fix = open-heart on load, then continue digestion. Not a full reinit.

## Files

| Path | Role |
|------|------|
| docs/KURAMOTO_BOTTLENECK_AND_FIX.md | this note |
| fractus/kuramoto_fix.py | apply helper |
| scripts/prep_kuramoto_fix_resume.py | offline prep all GPU ckpts |
| fractus/nn/phase_ode.py | wider omega init for new builds |
| scripts/fast4gpu_surgery.py | historical mid-train surgery (same family) |

## One line

Dead experts are a routing-arc problem; scale omega + soften gates + real LB, then finish the pass — do not wait for tokens alone.
