#!/usr/bin/env python3
"""Prepare Fractus-1B checkpoints for NEXT training after Kuramoto bottleneck fix.

Usage (on pod, after download freeze ckpts):
  python scripts/prep_kuramoto_fix_resume.py
  # writes checkpoints/fractus_1b_gpu{i}_pre_kuramoto_fix.pt backups
  # overwrites fractus_1b_gpu{i}.pt with fixed omega/temp metadata in config

Does NOT run training. Safe offline CPU.
"""
import os, sys, json, copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.kuramoto_fix import apply_kuramoto_routing_fix

TARGET = dict(
    d_model=1280, n_heads=20, d_head=64, n_levels=2, n_oscillators=16,
    coupling_rank=8, n_experts=128, top_k=2, expert_d_ff=2048, siren_rank=64, n_layers=16,
)

CKPT_DIR = Path(os.environ.get("CKPT_DIR", "checkpoints"))
N_GPU = int(os.environ.get("N_GPU", "8"))


def load_engine(path: Path) -> tuple:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state", ck)
    clean = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    eng = ContinuousThoughtEngine(vocab_size=266, **TARGET)
    own = eng.state_dict()
    loaded = 0
    for k, v in clean.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded += 1
    eng.load_state_dict(own, strict=False)
    return eng, ck, loaded


def main():
    report = {"gpus": {}, "env": {
        "GATE_TEMP": os.environ.get("GATE_TEMP", "2.5"),
        "OMEGA_SCALE": os.environ.get("OMEGA_SCALE", "4.0"),
        "OMEGA_NOISE": os.environ.get("OMEGA_NOISE", "0.01"),
        "LB_COEF": os.environ.get("LB_COEF", "0.05"),  # slightly stronger default for next run
    }}
    for i in range(N_GPU):
        src = CKPT_DIR / f"fractus_1b_gpu{i}.pt"
        if not src.exists():
            print(f"skip missing {src}")
            continue
        bak = CKPT_DIR / f"fractus_1b_gpu{i}_pre_kuramoto_fix.pt"
        if not bak.exists():
            bak.write_bytes(src.read_bytes())
            print(f"backup {bak}")
        eng, ck, loaded = load_engine(src)
        stats = apply_kuramoto_routing_fix(eng)
        out = {
            "model_state": eng.state_dict(),
            "config": {
                **(ck.get("config") if isinstance(ck, dict) else {}),
                **TARGET,
                "kuramoto_fix": True,
                "gate_temp": stats["gate_temp"],
                "omega_scale": stats["omega_scale"],
                "omega_std_before": stats["omega_std_before"],
                "omega_std_after": stats["omega_std_after"],
            },
        }
        # preserve non-weight keys if present
        if isinstance(ck, dict):
            for k in ("thought_state", "optimizer", "tokens_processed"):
                if k in ck:
                    out[k] = ck[k]
        torch.save(out, src)
        report["gpus"][str(i)] = {"loaded": loaded, **{k: stats[k] for k in stats if k != "blocks"}}
        print(f"GPU {i}: fixed and saved {src}")
    man = CKPT_DIR / "KURAMOTO_FIX_APPLIED.json"
    man.write_text(json.dumps(report, indent=2))
    print("report", man)


if __name__ == "__main__":
    main()
