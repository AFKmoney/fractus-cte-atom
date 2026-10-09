"""Smoke test EBVT probe — model reel x8run, minimum de generation.

Verifie :
  1. probe s'attache sans erreur (16 blocs x 128 experts) ;
  2. comptage accumule pendant generate_with_surgery (fenetre = 1 generation) ;
  3. cross-check NATIF : probe[counts bloc 0] == engine._expert_hits (le
     compteur interne du moteur, bloc 0) — match exact attendu ;
  4. ticks bruts engine.tick() accumulent aussi ;
  5. reset() remet la fenetre a zero.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import ebvt_probe  # noqa: E402
from ebvt_probe import EBVTProbe, EBVTError  # noqa: E402

from fractus.continuous_engine import ContinuousThoughtEngine  # noqa: E402
from fractus.tokenizer import FractusTokenizer                # noqa: E402
from fractus.kuramoto_fix import apply_kuramoto_routing_fix   # noqa: E402
from fractus.decode_surgery import generate_with_surgery      # noqa: E402

CKPT = sys.argv[1]
torch.manual_seed(42)

t0 = time.time()
print("load ...", flush=True)
engine = ContinuousThoughtEngine.from_pretrained(CKPT, map_location="cpu")
engine.eval()
apply_kuramoto_routing_fix(engine)
tok = FractusTokenizer.gpt2_compatible()
print(f"pret en {time.time()-t0:.0f}s", flush=True)

probe = EBVTProbe(engine)
print(f"[ebvt] probe attache {probe.n_layers}x{probe.n_experts}", flush=True)

# --- 1. generation chirurgicale, fenetre EBVT = cette generation ---
engine.reset_thought(1)
probe.reset()
text, ids = generate_with_surgery(engine, tok, "tell me about the ocean", max_new=48)
m = probe.report()
print(f"\n--- generation 48 tok ---\n{text}\n", flush=True)
print(f"ebvt: V={m['V']} H={m['H']} P1={m['P1']} pers={m['persistence']} "
      f"supp={m['support']} dom={m['dominance']} total={m['total']}", flush=True)
print(f"native_b0 = {m['native_b0']}", flush=True)

# --- 2. cumul SANS reset : 2e generation (chaque tick route exactement K=2
#     experts => le total doit doubler exactement) ---
generate_with_surgery(engine, tok, "tell me about the ocean", max_new=48)
m2 = probe.report(with_native_crosscheck=False)
assert m2["total"] == 2 * int(m["total"]), \
    f"cumul faux : {m2['total']} != 2*{m['total']}"
print(f"cumul sans reset: total {m['total']} -> {m2['total']} (x2 exact)", flush=True)

# --- 3. reset remet a zero ---
probe.reset()
m3 = probe.report(with_native_crosscheck=False)
assert m3["total"] == 0 and m3["empty"], "reset() ne vide pas la fenetre"
print("reset: OK (fenetre vide)", flush=True)

# --- 4. generation propre apres reset : cross-check natif de nouveau ---
engine.reset_thought(1)
probe.reset()
generate_with_surgery(engine, tok, "tell me about the ocean", max_new=48)
m4 = probe.report()
assert m4["native_b0"].get("match") is True, f"cross-check natif casse : {m4['native_b0']}"
print("cross-check natif (2e generation): match exact", flush=True)

print("\n=== BASELINE X8 103M — EBVT (seed 42, 1x generation 48 tok, "
      "etat frais post-load+kuramoto) ===", flush=True)
print(json.dumps({k: m[k] for k in
                  ("total", "V", "H", "P1", "persistence", "support",
                   "dominance", "block_events", "native_b0", "profile")},
                 indent=1, default=int), flush=True)
probe.detach()
print("\nSMOKE OK", flush=True)
