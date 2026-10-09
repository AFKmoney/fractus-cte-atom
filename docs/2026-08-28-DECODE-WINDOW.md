# 2026-08-28 — Decode surgery I: kill the length-1 carry lock

Weights untouched. Trainers kept running (boost_v4, QuickPod 8×5090).

## Symptom (Space + greedy tick_chunk)

```
Retail × 32
unique@32 = 1
lock = 26702×32
ckpt tokens = 84,018,176
```

Same family as Amnesty / relation / cutting: one prompt → one attractor → copy forever.

## Cause

`generate_chunk` warmed the prompt as one `tick_chunk`, then advanced with **length-1 chunks + thought/phase carry**.

A single generated token cannot pull Kuramoto + thought_state out of the basin that just emitted it. The same 2 experts reopen. Same logit wins.

Teacher-force CE does not see this loop. SS on the trainer is the train-side fix; this surgery is the decode-side fix so we stop measuring a plumbing bug.

## Surgery (code only)

File: `fractus/generate_aligned.py`

1. Default decode = **causal window 64**: reset thought + attn carry, `tick_chunk(last 64 ids)`, take last logit.
2. **Immediate anti-copy**: `logits[prev] = -inf`. Not a frequency ban. Forbids the period-1 fixed point only.
3. `generate_chunk` / `generate_greedy_prefix` / `generate_greedy_ids` all go through that path. Length-1 carry is banned.
4. Space `thefinalboss/Fractus-Space` loads the same module. One shared 1B at boot.

Trainers were not restarted. `PROBE_EVERY=0` still. REPEAT_COEF unchanged (0.1).

## Probe after surgery

GPU0 `x8run` at **88,114,176** tokens. Greedy PREFIX window 64, max_new=32. No frequency ban.

| Prompt | unique@32 before | unique@32 after | notes |
|---|---|---|---|
| Hello, my name is | 1 | **16** | salad, then short cycle `Bone Seasons` |
| Once upon a time | 1 | **14** | `verifying morals` ×4 |
| The meaning of life is | 1 | **27** | highest diversity, not English |
| Fractus is | 2 | **12** | collapses to `Sandwich` |

Mono-token lock is broken. Short cycles (2–4 tokens) remain. Speech gate still NO-GO.

## What this is not

- Not a claim that Fractus speaks.
- Not a weight wipe.
- Not temperature / top-k theater as the metric. Gate stays unique@40 greedy PREFIX.
