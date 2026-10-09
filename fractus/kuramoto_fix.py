"""Kuramoto / MoE routing fix applied on checkpoint load (next training).

Problem (measured 2026-08-19):
  - Circular mean of phase fan can be degenerate (atan2(0,0))
  - Post-RK4 routing arc ~25°/360° with 128 experts @ ~2.8° spacing → many dead experts
  - Order parameter r ≈ 0.01–0.03 matches prod logs; also seen on a FRESH model
  - Not explained by probe no_grad alone

Fix (open-heart, preserves trained directions):
  1. GATE_TEMP > 1 softens von Mises gates (more experts get mass)
  2. OMEGA_SCALE amplifies |omega| (keep sign), optional noise, clamp
  3. Training must include LB_COEF * load_balance_loss (not detached)
  4. Kuramoto path must NOT be under torch.no_grad during train

Env:
  GATE_TEMP   default 2.5
  OMEGA_SCALE default 4.0
  OMEGA_NOISE default 0.01
  OMEGA_CLAMP default 0.5
"""
from __future__ import annotations

import os
from typing import Any


def apply_kuramoto_routing_fix(engine: Any, log=print) -> dict:
    gate_temp = float(os.environ.get("GATE_TEMP", "2.5"))
    omega_scale = float(os.environ.get("OMEGA_SCALE", "4.0"))
    omega_noise = float(os.environ.get("OMEGA_NOISE", "0.01"))
    omega_clamp = float(os.environ.get("OMEGA_CLAMP", "0.5"))

    import torch

    stats = {"gate_temp": gate_temp, "omega_scale": omega_scale, "blocks": 0, "omega_std_before": None, "omega_std_after": None}
    with torch.no_grad():
        for blk in engine.blocks:
            if hasattr(blk, "moe") and hasattr(blk.moe, "temperature"):
                blk.moe.temperature = gate_temp
            if hasattr(blk, "kuramoto") and hasattr(blk.kuramoto, "omega"):
                om = blk.kuramoto.omega
                if stats["omega_std_before"] is None:
                    stats["omega_std_before"] = float(om.std().item())
                om.mul_(omega_scale)
                if omega_noise > 0:
                    om.add_(torch.randn_like(om) * omega_noise)
                om.clamp_(-omega_clamp, omega_clamp)
                stats["omega_std_after"] = float(om.std().item())
                stats["blocks"] += 1
    log(
        f"[kuramoto_fix] blocks={stats['blocks']} temp={gate_temp} "
        f"omega_scale={omega_scale} std {stats['omega_std_before']:.4f} -> {stats['omega_std_after']:.4f}"
    )
    return stats
