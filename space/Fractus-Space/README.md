---
title: Fractus
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.19.2
python_version: "3.10"
app_file: app.py
pinned: true
license: mit
short_description: Fractus CTE-Atom 1B, generation on the training function
suggested_hardware: cpu-upgrade
---

# Fractus CTE-Atom 1B: fixed generation path

Fractus is an ACI (a field / flow architecture: fractal linear attention with a carried state,
Kuramoto phase routing, phase-routed MoE). This Space loads
`checkpoints/atom_v3/fractus_atom_1b_gpu0.bin` from `thefinalboss/fractus-cte-atom`.

Fixed on 2026-10-18 (see `docs/DEEP_DIVE_GENERATION.md` in GitHub `AFKmoney/fractus-cte-atom`):

- model code synced with the training repo. The previous copy had no Kuramoto `phase_proj`, so a
  `strict=False` load silently dropped trained weights;
- strict load, exactly like the trainer resume (no second Kuramoto routing fix);
- open prompt encoding (no `<B6><EOS>` after the prompt);
- fixed 128-id chunk grid, committed carry, open chunk re-forwarded each step (no double counting);
- warm start: the tail of a real corpus document before BOS, as BOS appears in training; default 127 ids (best in the 2026-10-18 hunt: CONT unique@40 19.0, exact 4.3/40 vs 2.3 at 64)
- generation stops only at EOS or max new ids;
- optional final norm before the tied head (off by default).
