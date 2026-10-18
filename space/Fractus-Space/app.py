"""Fractus CTE-Atom: generation that computes the training function.

Fixed on 2026-10-18 (docs/DEEP_DIVE_GENERATION.md in AFKmoney/fractus-cte-atom):
  * model code synced with the training repo. The old copy had no Kuramoto phase_proj and used
    full-matrix attn_S, so strict=False silently dropped trained weights;
  * load exactly like the trainer resume: strict=True, gate temperature from the config,
    Kuramoto routing fix NOT re-applied (the old Space multiplied omega by 4 again);
  * prompts are encoded open (no closing <B6><EOS>);
  * fixed 128-id chunk grid. The committed carry is kept. The open chunk is re-forwarded from the
    committed carry on every step (the old loop re-added it to attn_S/attn_z every step);
  * optional warm start: the tail of a real corpus document (... <B6><EOS>) before BOS, as in training;
  * generation does not stop at boundary markers, only at EOS or max new ids;
  * optional final norm before the tied head (off by default: the weights were trained without one).
"""
from __future__ import annotations

import os
import threading
import time

import torch
import gradio as gr

from fractus.fixed_generation import FixedGenerator, load_checkpoint, show

torch.set_num_threads(max(1, os.cpu_count() or 2))
CKPT_REPO = os.environ.get("CKPT_REPO", "thefinalboss/fractus-cte-atom")
CKPT_FILE = os.environ.get("CKPT_FILE", "checkpoints/atom_v3/fractus_atom_1b_gpu0.bin")

_lock = threading.Lock()
GEN = None
META = {"file": CKPT_FILE, "ids": None, "ready": False, "error": "", "load_s": None}


def preload():
    global GEN
    from huggingface_hub import hf_hub_download
    t = time.time()
    print("[Fractus Atom] downloading", CKPT_REPO, CKPT_FILE, flush=True)
    path = hf_hub_download(CKPT_REPO, CKPT_FILE, repo_type="model")
    eng, cfg = load_checkpoint(path, dtype=torch.float32)   # strict=True inside; raises on any missing/unexpected key
    GEN = FixedGenerator(eng, final_norm=None, warm=127)
    META.update(ids=cfg.get("ids_processed"), ready=True, load_s=round(time.time() - t, 1))
    print("[Fractus Atom] READY (strict load) ids_processed=%s in %ss" % (META["ids"], META["load_s"]), flush=True)


def infer(prompt, max_new, warm, temperature, top_k, final_norm):
    prompt = (prompt or "").strip()
    if not prompt:
        return "(empty prompt)"
    if GEN is None:
        return "model not loaded: %s" % META.get("error")
    with _lock:
        try:
            t = time.time()
            GEN.final_norm = None if final_norm == "off" else final_norm
            w = int(warm)
            ctx = GEN.context(GEN.encode_prompt(prompt[:400]), warm=w)
            out = GEN.generate(ctx, max_new=int(max_new), temperature=float(temperature), top_k=int(top_k))
            txt = GEN.tok.decode(out)
            return "%s\n\n---\nwith markers: %s\nunique ids=%d/%d | warm=%d | T=%.2f top_k=%d | final_norm=%s | %.1fs" % (
                txt if txt.strip() else "(only markers/whitespace)", show(out), len(set(out)), len(out), w,
                float(temperature), int(top_k), final_norm, time.time() - t)
        except Exception as e:  # noqa: BLE001
            META["error"] = "%s: %s" % (type(e).__name__, e)
            return "error: %s" % META["error"]


print("[Fractus Atom] process boot, preloading", flush=True)
try:
    preload()
except Exception as e:  # noqa: BLE001
    META["error"] = "%s: %s" % (type(e).__name__, e)
    print("[Fractus Atom] PRELOAD FAILED", META["error"], flush=True)

STATUS = ("READY Atom 1B | %s | ids_processed=%s | strict load" % (META["file"], META["ids"])
          if META["ready"] else "NOT READY: %s" % META.get("error"))

demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Textbox(lines=3, label="Prompt", placeholder="The capital of France is"),
        gr.Slider(8, 64, value=24, step=8, label="max new ids"),
        gr.Slider(0, 127, value=127, step=1, label="warm start: ids of a real corpus doc tail before BOS (0 = cold BOS)"),
        gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="temperature (0 = greedy)"),
        gr.Slider(1, 50, value=20, step=1, label="top-k (used when temperature > 0)"),
        gr.Radio(["off", "rms", "ln"], value="off", label="final norm before the tied head (trained without: off)"),
    ],
    outputs=gr.Textbox(lines=12, label="output"),
    title="Fractus CTE-Atom: fixed generation path",
    description=STATUS + " | fixed 128-id grid, committed carry, open chunk re-forwarded, warm start. "
                "Raw byte-level output of an ACI field model; CPU, ~2-5 s per id.",
    examples=[
        ["The capital of France is", 24, 127, 0.0, 20, "off"],
        ["Once upon a time", 24, 127, 0.0, 20, "off"],
        ["Hello", 24, 127, 0.0, 20, "off"],
    ],
    cache_examples=False,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.queue()
    kw = {}
    if not os.environ.get("SPACE_ID"):
        kw["server_name"] = "0.0.0.0"
        kw["server_port"] = int(os.environ.get("PORT", "7860"))
    demo.launch(**kw)
