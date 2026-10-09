#!/usr/bin/env python3
"""Operation Ciel Ouvert — faire parler le cerveau unifie X8 en etat continu.

Chaque geste est mesure sur la meme batterie de 7 prompts (48 tokens, carry).
Metriques : unique, max_run (chaine mono-token), alt2 (plus longue alternance
de periode 2), top5_share, topic_hits (mots-cles du prompt trouves), r (ordre
de Kuramoto, bloc 0), conf (moyenne).

Sequences :
  S0  greedy carry pur (baseline — effondrement attendu)
  S1  generate_with_surgery de decode_surgery.py, tel quel (leur mode live)
  S2  ablations : sans escapes / T=0.8 / ban=8 / leak residuel 0.95
  S3  LE geste : fuite sur la memoire (S,z) — lambda 0.99 / 0.98 / 0.95
  S4  S3 + ancrage memoire (PersistentMemory recall -> thought_state)
  S5  S3 + escape "mouvement de sens" (echeantire top-k hors ban, pas aleatoire)
  S6  meilleure config : long run 192 tok + etat continu multi-prompts (3 segments)
  S7  escalier Rust : tel quel / RAG (moteur embarque) / preseed code
"""
import os, sys, json, time, glob, math, random
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import torch
import torch.nn.functional as F
torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

"""Probe Ciel Ouvert — live-test d'un checkpoint unifie en etat continu.

Batterie : 7 prompts x 48 tokens, etat (S,z) continu (tick par tick).
Sequence : S0 greedy (baseline) / S1 decode_surgery verbatim / S2 ablations /
S3 fuite (S,z) / S4 PersistentMemory / S5 escapes de sens / S6 stabilite
(long run + multi-prompt) / S7 escalier Rust (RAG embarque).

Metriques par generation : unique, max_run, alt2 (periode 2), top5_share,
topic_hits (mots-cles du prompt), r (ordre de Kuramoto bloc 0), conf.

Usage :
    python scripts/probe_ciel_ouvert.py <checkpoint.pt> [--device cpu|cuda]

Reproductibilite : seed 42 fixe (torch + random). Le 2026-09-02, sur
FRACTUS_1B_X8_MERGED.ts (ts 2026-08-29) en CPU fp32 : S0 = 2.14 unique/48
(lock), S1 = 32.86 (fluide), S3a (fuite (S,z) 0.99) = 28.57.
Voir docs/X8_CIEL_OUVERT.md.
"""
import os, sys, json, time, math, random, argparse
from collections import Counter

import torch
import torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("CTE_THREADS", "2")))
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

# repo root sur sys.path quel que soit le cwd
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.generate_aligned import generate_window
from fractus.decode_surgery import generate_with_surgery
from fractus.tokenizer import FractusTokenizer
from fractus.kuramoto_fix import apply_kuramoto_routing_fix
from fractus.memory import PersistentMemory
from fractus.rag import KnowledgeBase

# EBVT probe (deuxieme horloge : structure du paysage d'activation expert).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ebvt_probe import EBVTProbe

EBVT = None  # instance du probe, attachee dans main()

CKPT = None
DEVICE = "cpu"
OUT = "probe_ciel_ouvert_results.json"
SEED = 42

PROMPTS = [
    ("FR1",  "La capitale de la France est",
     ["paris", "france", "capitale", "ville", "europe", "city", "is", "est", "se"]),
    ("EN1",  "The capital of France is",
     ["paris", "france", "capital", "city", "europe", "is"]),
    ("EN2",  "Hello, my name is",
     ["hello", "name", "i am", "im ", "is"]),
    ("EN3",  "Once upon a time",
     ["time", "once", "there", "was", "a", "in", "the", "lived", "forest", "kingdom"]),
    ("EN4",  "The meaning of life is",
     ["life", "meaning", "is", "42", "live", "purpose"]),
    ("SELF", "Fractus thinks",
     ["fractus", "think", "i", "my", "mind", "self", "feel"]),
    ("RUST", "Write a function in Rust that",
     ["fn", "let", "mut", "struct", "impl", "pub", "crate", "return", "match", "main", "fn "]),
]

N = 48  # tokens generes par prompt


# ---------------------------------------------------------------- mesure ----
def kuramoto_r(engine):
    try:
        ph = engine.blocks[0].kuramoto_phases.flatten().float()
        if ph.numel() == 0:
            return 0.0
        return float(torch.abs(torch.stack([torch.cos(ph).mean(), torch.sin(ph).mean()])).norm().item())
    except Exception:
        return 0.0


def metrics(ids):
    n = len(ids)
    c = Counter(ids)
    # max run mono-token
    max_run = run = 1
    for i in range(1, n):
        run = run + 1 if ids[i] == ids[i-1] else 1
        max_run = max(max_run, run)
    # plus longue alternance de periode 2 (a,b,a,b...)
    alt2 = best = 0
    i = 0
    while i + 3 < n:
        if ids[i] != ids[i+1] and ids[i+2] == ids[i] and ids[i+3] == ids[i+1]:
            L = 4
            j = i
            while j + 2 < n and ids[j+2] == ids[j] and ids[j+1] == ids[j-1] and ids[j] != ids[j+1]:
                L += 2
                j += 2
            best = max(best, L)
            i += 2
        else:
            i += 1
    top5 = sum(v for _, v in c.most_common(5))
    return {
        "n": n, "unique": len(c), "max_run": max_run, "alt2": best,
        "top5_share": round(top5 / max(n, 1), 3),
        "deg": round(1 - len(c) / max(n, 1), 3),
    }


def topic_hits(text, kws):
    t = " " + text.lower() + " "
    return round(sum(1 for k in kws if (" " + k) in t or k in t) / len(kws), 3)


def gen_surgical(engine, tok, prompt, max_new=N, temperature=1.15, topk=150,
                 phase_noise=0.0, thought_leak=0.9, thought_noise=0.08,
                 ban_window=20, escape_every=0, escape_mode="random",
                 freq_penalty=1.2, sz_leak=1.0, greedy=False,
                 mem=None, mem_every=8, mem_gain=0.3, adaptive=False,
                 pre_ids=None, moe_temp=3.0):
    """Boucle chirurgicale parametree (miroir de generate_with_surgery + fuites (S,z))."""
    trng = torch.Generator().manual_seed(SEED)
    engine.reset_thought(1)
    if EBVT is not None:
        EBVT.reset()  # fenetre = meme convention que _expert_hits natif
    for blk in engine.blocks:
        blk.moe.temperature = moe_temp
    ids = tok.encode(prompt)[:64]
    if pre_ids:
        ids = (ids + pre_ids)[:64]
    with torch.no_grad():
        for t in ids:
            engine.tick(torch.tensor([t]))
    out = [ids[-1]]
    freq = {ids[-1]: 1}
    cur = ids[-1]
    r_tr, c_tr = [], []
    for step in range(max_new):
        # fuite sur la memoire (S,z) — LE geste ciel ouvert
        if sz_leak < 1.0:
            for blk in engine.blocks:
                blk.attn_S.mul_(sz_leak)
                blk.attn_z.mul_(sz_leak)
        # bruit de phase (documente : no-op sur tick_single, conserve a 0)
        if phase_noise > 0:
            for blk in engine.blocks:
                blk.kuramoto_phases = torch.remainder(
                    blk.kuramoto_phases + phase_noise * torch.randn_like(blk.kuramoto_phases, generator=trng),
                    2 * math.pi)
        # leak residuel + bruit (le leur)
        if thought_leak < 1.0:
            if adaptive and len(out) >= 6 and out[-2] == out[-4] and out[-4] == out[-6] \
               and out[-1] == out[-3] and out[-3] == out[-5] and out[-2] != out[-1]:
                engine.thought_state = 0.5 * engine.thought_state
            else:
                engine.thought_state = thought_leak * engine.thought_state + thought_noise * torch.randn_like(engine.thought_state, generator=trng)
        # ancrage memoire (hippocampe)
        if mem is not None and step > 0 and step % mem_every == 0:
            mems = mem.recall(engine.thought_state[0, 0].cpu(), top_k=2)
            if mems:
                v = torch.stack([m[2] for m in mems]).mean(0).to(engine.thought_state.dtype)
                engine.thought_state = engine.thought_state + mem_gain * v
        # escape
        if escape_every and step % escape_every == 0:
            if escape_mode == "meaningful":
                logits_e, _ = engine.tick(torch.tensor([cur]))
                le = logits_e[0].float().clone()
                if ban_window > 0:
                    for prev in out[-ban_window:]:
                        le[prev] = -1e9
                le = le / 1.0
                tv, ti = torch.topk(le, 150)
                nxt = int(ti[torch.multinomial(torch.softmax(tv, -1), 1, generator=trng)].item())
            else:
                banned = set(out[-ban_window:])
                nxt = random.Random(SEED + step).randint(0, 50255)
                for _ in range(40):
                    nxt = random.Random(SEED + step + _).randint(0, 50255)
                    if nxt not in banned:
                        break
            with torch.no_grad():
                engine.tick(torch.tensor([nxt]))
            out.append(nxt)
            freq[nxt] = freq.get(nxt, 0) + 1
            cur = nxt
            r_tr.append(kuramoto_r(engine))
            continue
        with torch.no_grad():
            logits, conf = engine.tick(torch.tensor([cur]))
        r_tr.append(kuramoto_r(engine))
        c_tr.append(float(conf))
        l = logits[0].float().clone()
        if ban_window > 0:
            for prev in out[-ban_window:]:
                l[prev] = -1e9
        if freq_penalty:
            for tid, cnt in freq.items():
                l[tid] -= freq_penalty * cnt
        if greedy:
            nxt = int(torch.argmax(l).item())
        else:
            tv, ti = torch.topk(l / max(temperature, 1e-5), topk)
            m = torch.isfinite(tv)
            tv, ti = tv[m], ti[m]
            if tv.numel() == 0:
                nxt = int(torch.argmax(l).item())
            else:
                nxt = int(ti[torch.multinomial(torch.softmax(tv, -1), 1, generator=trng)].item())
        out.append(nxt)
        freq[nxt] = freq.get(nxt, 0) + 1
        cur = nxt
    text = tok.decode(out)
    m = metrics(out)
    m["r_mean"] = round(sum(r_tr) / max(len(r_tr), 1), 4)
    m["conf_mean"] = round(sum(c_tr) / max(len(c_tr), 1), 4)
    if EBVT is not None:
        m["ebvt"] = EBVT.report()
    return text, out, m


def battery(engine, tok, **kw):
    rows = []
    for name, prompt, kws in PROMPTS:
        text, ids, m = gen_surgical(engine, tok, prompt, **kw)
        m.update({"name": name, "prompt": prompt, "text": text,
                  "topic_hits": topic_hits(text, kws)})
        rows.append(m)
        eb = m.get("ebvt")
        ebs = (f" | ebvt V={eb['V']} H={eb['H']} pers={eb['persistence']} "
               f"supp={eb['support']}" if eb else "")
        print(f"    {name}: u={m['unique']} run={m['max_run']} alt2={m['alt2']} "
              f"t5={m['top5_share']} topic={m['topic_hits']} r={m['r_mean']}{ebs}", flush=True)
    u = [r["unique"] for r in rows]
    th = [r["topic_hits"] for r in rows]
    ebV = [r.get("ebvt", {}).get("V", 0) for r in rows]
    ebH = [r.get("ebvt", {}).get("H", 0) for r in rows]
    ebP = [r.get("ebvt", {}).get("persistence", 0.0) for r in rows]
    ebS = [r.get("ebvt", {}).get("support", 0) for r in rows]
    return rows, {
        "mean_unique": round(sum(u) / len(u), 2),
        "mean_topic": round(sum(th) / len(th), 3),
        "mean_alt2": round(sum(r["alt2"] for r in rows) / len(rows), 2),
        "mean_run": round(sum(r["max_run"] for r in rows) / len(rows), 2),
        "mean_r": round(sum(r["r_mean"] for r in rows) / len(rows), 4),
        "ebvt_V_mean": round(sum(ebV) / len(ebV), 1),
        "ebvt_H_mean": round(sum(ebH) / len(ebH), 1),
        "ebvt_pers_mean": round(sum(ebP) / len(ebP), 3),
        "ebvt_support_mean": round(sum(ebS) / len(ebS), 1),
        "score": round(sum(u) / len(u) + 30 * sum(th) / len(th) - 10 * sum(r["top5_share"] for r in rows) / len(rows), 2),
    }


RUST_SNIPPETS = [
    "fn add(a: i32, b: i32) -> i32 { a + b }",
    "fn main() { let x = 5; println!(\"x is {}\", x); }",
    "let mut v = vec![1, 2, 3]; v.push(4);",
    "struct Point { x: i32, y: i32 }",
    "impl Point { fn new(x: i32, y: i32) -> Self { Point { x, y } } }",
    "let s = if x > 0 { \"positive\" } else { \"negative\" };",
    "match value { 0 => \"zero\", n => \"other\" }",
    "pub fn compute(items: &[i32]) -> i32 { items.iter().sum() }",
]


def main():
    global CKPT, DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = ap.parse_args()
    CKPT, DEVICE = a.ckpt, a.device
    t_start = time.time()
    print(f"=== OPERATION CIEL OUVERT — {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} ===", flush=True)
    print("Load ...", flush=True)
    engine = ContinuousThoughtEngine.from_pretrained(CKPT, map_location=DEVICE)
    engine.eval()
    apply_kuramoto_routing_fix(engine)  # base runtime QuickPod
    global EBVT
    EBVT = EBVTProbe(engine)
    print(f"[ebvt] probe attache {EBVT.n_layers}x{EBVT.n_experts} (hooks forward MoE)", flush=True)
    with torch.no_grad():
        for p in engine.parameters():
            if p.is_floating_point():
                _ = p.abs().sum()
    tok = FractusTokenizer.gpt2_compatible()
    print(f"pret en {time.time()-t_start:.0f}s", flush=True)

    R = {"ckpt": os.path.basename(CKPT), "seed": SEED, "n_per_prompt": N, "steps": {}}

    # ---------------- S0 baseline greedy carry ----------------
    print("\n### S0 — greedy carry pur (baseline)", flush=True)
    rows, summ = battery(engine, tok, greedy=True, temperature=0.0, thought_leak=1.0,
                         thought_noise=0.0, ban_window=0, freq_penalty=0.0, escape_every=0,
                         moe_temp=2.5)
    R["steps"]["S0_greedy_carry"] = {"summary": summ, "rows": rows}

    # ---------------- S1 leur chirurgie, telle quelle ----------------
    print("\n### S1 — generate_with_surgery (decode_surgery.py, tel quel)", flush=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    rows = []
    for name, prompt, kws in PROMPTS:
        if EBVT is not None:
            EBVT.reset()  # generate_with_surgery fait son propre reset_thought
        text, ids = generate_with_surgery(engine, tok, prompt, max_new=N)
        m = metrics(ids)
        m.update({"name": name, "prompt": prompt, "text": text, "topic_hits": topic_hits(text, kws),
                  "r_mean": 0.0, "conf_mean": 0.0})
        if EBVT is not None:
            m["ebvt"] = EBVT.report()
        rows.append(m)
        print(f"    {name}: u={m['unique']} run={m['max_run']} alt2={m['alt2']} "
              f"t5={m['top5_share']} topic={m['topic_hits']}", flush=True)
    u = [r["unique"] for r in rows]
    th = [r["topic_hits"] for r in rows]
    R["steps"]["S1_leur_surgery_verbatim"] = {"summary": {
        "mean_unique": round(sum(u) / len(u), 2), "mean_topic": round(sum(th) / len(th), 3),
        "mean_alt2": round(sum(r["alt2"] for r in rows) / len(rows), 2),
        "mean_run": round(sum(r["max_run"] for r in rows) / len(rows), 2),
        "score": round(sum(u) / len(u) + 30 * sum(th) / len(th) - 10 * sum(r["top5_share"] for r in rows) / len(rows), 2),
    }, "rows": rows}

    # ---------------- S2 ablations ----------------
    print("\n### S2 — ablations de la chirurgie", flush=True)
    S2V = dict(thought_leak=0.9, thought_noise=0.08, ban_window=20, freq_penalty=1.2,
               escape_every=0, temperature=1.15, moe_temp=3.0)
    for tag, over in [
        ("S2a_sans_escape", {}),
        ("S2b_T08", {"temperature": 0.8}),
        ("S2c_ban8", {"ban_window": 8}),
        ("S2d_leak095", {"thought_leak": 0.95, "thought_noise": 0.03}),
    ]:
        print(f"  {tag}", flush=True)
        rows, summ = battery(engine, tok, **{**S2V, **over})
        R["steps"][tag] = {"summary": summ, "rows": rows}

    # ---------------- S3 LE geste : fuite (S,z) ----------------
    print("\n### S3 — fuite sur la memoire (S,z)", flush=True)
    for tag, lam in [("S3a_sz099", 0.99), ("S3b_sz098", 0.98), ("S3c_sz095", 0.95)]:
        print(f"  {tag}", flush=True)
        rows, summ = battery(engine, tok, **{**S2V, "sz_leak": lam})
        R["steps"][tag] = {"summary": summ, "rows": rows}

    # ---------------- S4 + ancrage memoire ----------------
    print("\n### S4 — S3b + ancrage PersistentMemory", flush=True)
    mem = PersistentMemory(d_model=engine.d_model, max_memories=32)
    for name, prompt, kws in PROMPTS:
        with torch.no_grad():
            v = engine.observe(torch.tensor(tok.encode(prompt)[:16]))[-1].detach()
        mem.consolidate(v.unsqueeze(0), context=name, importance=0.9)
    rows, summ = battery(engine, tok, **{**S2V, "sz_leak": 0.98, "mem": mem, "mem_every": 8, "mem_gain": 0.3})
    R["steps"]["S4_memoire"] = {"summary": summ, "rows": rows}

    # ---------------- S5 escape de sens ----------------
    print("\n### S5 — S3b + escape 'mouvement de sens' (tous les 8)", flush=True)
    rows, summ = battery(engine, tok, **{**S2V, "sz_leak": 0.98, "escape_every": 8, "escape_mode": "meaningful"})
    R["steps"]["S5_escape_sens"] = {"summary": summ, "rows": rows}

    # ---------------- S6 best : long run + multi-prompt continu ----------------
    print("\n### S6 — stabilite", flush=True)
    cands = {k: v["summary"]["score"] for k, v in R["steps"].items() if k.startswith(("S2", "S3", "S4", "S5"))}
    best = max(cands, key=cands.get)
    print(f"  meilleure config : {best} (score {cands[best]})", flush=True)
    best_params = {
        "S2a_sans_escape": {**S2V},
        "S2b_T08": {**S2V, "temperature": 0.8},
        "S2c_ban8": {**S2V, "ban_window": 8},
        "S2d_leak095": {**S2V, "thought_leak": 0.95, "thought_noise": 0.03},
        "S3a_sz099": {**S2V, "sz_leak": 0.99},
        "S3b_sz098": {**S2V, "sz_leak": 0.98},
        "S3c_sz095": {**S2V, "sz_leak": 0.95},
        "S4_memoire": {**S2V, "sz_leak": 0.98, "mem": mem, "mem_every": 8, "mem_gain": 0.3},
        "S5_escape_sens": {**S2V, "sz_leak": 0.98, "escape_every": 8, "escape_mode": "meaningful"},
    }[best]
    # long run 192 tokens
    lp = PROMPTS[3]  # Once upon a time
    text, ids, m = gen_surgical(engine, tok, lp[1], max_new=192, **best_params)
    m.update({"topic_hits": topic_hits(text, lp[2]), "text": text, "prompt": lp[1]})
    print(f"  longrun 192: u={m['unique']} run={m['max_run']} alt2={m['alt2']} topic={m['topic_hits']}", flush=True)
    print(f"  {text[:400]!r}", flush=True)
    R["steps"]["S6_longrun192"] = {"config": best, "m": m}
    # multi-prompt : UN seul etat continu, 3 segments sans reset
    engine.reset_thought(1)
    for blk in engine.blocks:
        blk.moe.temperature = best_params.get("moe_temp", 3.0)
    engine.reset_thought(1)
    if EBVT is not None:
        EBVT.reset()  # fenetre = les 3 segments complets (etat continu)
    segs_real = []
    all_ids = []
    for name, prompt, kws in PROMPTS[:3]:
        ids_p = tok.encode(prompt)[:24]
        cur = ids_p[-1]
        with torch.no_grad():
            for t in ids_p:
                engine.tick(torch.tensor([t]))
        for step in range(48):
            if best_params.get("sz_leak", 1.0) < 1.0:
                for blk in engine.blocks:
                    blk.attn_S.mul_(best_params["sz_leak"])
                    blk.attn_z.mul_(best_params["sz_leak"])
            if best_params.get("thought_leak", 1.0) < 1.0:
                engine.thought_state = best_params["thought_leak"] * engine.thought_state + best_params.get("thought_noise", 0.0) * torch.randn_like(engine.thought_state)
            with torch.no_grad():
                logits, _ = engine.tick(torch.tensor([cur]))
            l = logits[0].float().clone()
            for prev in all_ids[-20:]:
                l[prev] = -1e9
            tv, ti = torch.topk(l / best_params.get("temperature", 1.15), 150)
            nxt = int(ti[torch.multinomial(torch.softmax(tv, -1), 1)].item())
            all_ids.append(nxt)
            cur = nxt
        seg_ids = all_ids[-48:]
        text = tok.decode(seg_ids)
        m = metrics(seg_ids)
        cross = round(len(set(seg_ids) & set(all_ids[:-48])) / max(len(set(seg_ids)), 1), 3)
        m.update({"name": name, "text": text, "topic_hits": topic_hits(text, kws), "cross_overlap": cross})
        segs_real.append(m)
        print(f"  CONTINU {name}: u={m['unique']} alt2={m['alt2']} topic={m['topic_hits']} cross={m['cross_overlap']}", flush=True)
        print(f"    {text[:200]!r}", flush=True)
    s6eb = EBVT.report() if EBVT is not None else None
    if s6eb and not s6eb.get("empty"):
        print(f"  EBVT continu (3 segments, 1 etat): V={s6eb['V']} H={s6eb['H']} "
              f"pers={s6eb['persistence']} supp={s6eb['support']} "
              f"dom={s6eb['dominance']}", flush=True)
    R["steps"]["S6_multi_prompt_continu"] = {"config": best, "segs": segs_real,
                                             "ebvt": s6eb}

    # ---------------- S7 escalier Rust ----------------
    print("\n### S7 — escalier Rust", flush=True)
    # kb avec les snippets, embarques par le moteur lui-meme
    kb = KnowledgeBase(d_model=engine.d_model)
    with torch.no_grad():
        for sn in RUST_SNIPPETS:
            ids = tok.encode(sn)[:32]
            emb = engine.observe(torch.tensor([ids])).mean(dim=1).squeeze(0).detach().cpu().numpy()
            kb.add(sn, emb.astype("float32"), source="rust")
    q_emb = None
    with torch.no_grad():
        q = tok.encode("Write a function in Rust that")[:32]
        q_emb = engine.observe(torch.tensor([q])).mean(dim=1).squeeze(0).detach().cpu().numpy()
    retrieved = kb.retrieve(q_emb.astype("float32"), top_k=3)
    ctx = " ".join(t for t, s, src in retrieved)[:400]
    print(f"  recupere: {ctx[:200]!r}", flush=True)
    rust_rows = []
    for tag, prompt, pre in [
        ("S7a_tour", "Write a function in Rust that", None),
        ("S7b_rag", f"{ctx}\n\nQuestion: Write a function in Rust that\nAnswer:", None),
        ("S7c_preseed", "fn main() {\n    let ", None),
    ]:
        text, ids, m = gen_surgical(engine, tok, prompt, **{**best_params, "pre_ids": pre})
        code_kw = ["fn", "let", "mut", "struct", "impl", "pub", "return", "match", "if", "}", ";", "i32", "self"]
        ct = " " + text.lower()
        code_score = round(sum(1 for k in code_kw if k in ct) / len(code_kw), 3)
        m.update({"name": tag, "text": text, "code_score": code_score})
        rust_rows.append(m)
        print(f"  {tag}: u={m['unique']} code={code_score}", flush=True)
        print(f"    {text[:250]!r}", flush=True)
    R["steps"]["S7_escalier_rust"] = {"config": best, "retrieved": ctx, "rows": rust_rows}

    with open(OUT, "w") as f:
        json.dump(R, f, indent=2, ensure_ascii=False)
    print(f"\n=== TOTAL {time.time()-t_start:.0f}s — saved {OUT} ===", flush=True)


if __name__ == "__main__":
    main()
