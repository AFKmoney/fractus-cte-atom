"""Runner v4 (fast4gpu_atom_v4, ckpt_v4): continue the v3 model on the real-language corpus
atom_corpus_v4 (docs/CORPUS_V4.md). PREPARED, NOT LAUNCHED. Launch only after v3 is stopped.

First launch:
  - refuses to start while the v3 trainer (/workspace/trainer_v3.pid) is alive (one GPU, one run);
  - takes the latest v3 GOOD checkpoint (/workspace/ckpt_v3/good: weights + AdamW state), hard-links
    it into /workspace/ckpt_v4_from_v3/ (no copy, no extra disk) so v3's rotation cannot remove it,
    and records its ids_processed as IDS_BASE;
  - starts at lane offset 0 of the new corpus (CORPUS_SWITCH=1; not a rewind: the corpus is new).
Same as run_v3.py otherwise: SS_RATE 0.35, SS_TAIL_IDS 8, ss_p 0.85 (v3's effective value: its
ramp had already finished), B=16, SEQ=128, cosine LR, rollback to good/ with half LR on exit 42.
Adds a held-out val loss every VAL_EVERY steps (VAL lines in train_v4.log, ckpt_v4/val_log.jsonl).

Env knobs (first launch only, then run_v4_state.json rules):
  PLAN_IDS   ids to train (default: one pass of atom_corpus_v4 = manifest train_ids)
  LR_V4      peak LR after warmup (default 2.35e-5 = v3's LR at its last good checkpoint)
"""
import json, os, subprocess, sys, time
from pathlib import Path
CORPUS = Path("/workspace/atom_corpus_v4")
OUT = Path("/workspace/ckpt_v4")
SRC_GOOD = Path("/workspace/ckpt_v3/good")
FROZEN = Path("/workspace/ckpt_v4_from_v3")
STATE = Path("/workspace/run_v4_state.json")
LOG = open("/workspace/runner_v4.log", "a", buffering=1)
def log(*a): print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), *a, file=LOG, flush=True)
IDS_PER_STEP = 16 * 128
BASE = dict(CUDA_VISIBLE_DEVICES="0", GPU_ID="0", N_GPU="1", CORPUS=str(CORPUS), CORPUS_SWITCH="1",
            VAL_CORPUS=str(CORPUS / "val.i16"), VAL_EVERY=os.environ.get("VAL_EVERY", "2500"),
            VAL_CHUNKS=os.environ.get("VAL_CHUNKS", "128"),
            BATCH="16", SEQ="128", FRACTUS_ATTN_IMPL="chunked", BLOCK_CKPT="1", COMPILE="1",
            PROBE_EVERY="0", SS_RATE="0.35", SS_PROB_START="0.85", SS_PROB_END="0.85", SS_RAMP_TOKENS="100000000",
            SS_RAMP_ORIGIN_IDS="0", CARRY_DECAY="1.0", CARRY_CLAMP="0", EPOCH_WRAP="1", REPEAT_COEF="0",
            LOG_EVERY="200", SAVE_EVERY="5000", SS_TAIL_IDS=os.environ.get("SS_TAIL_IDS", "8"),
            OUT_DIR=str(OUT), OPT="adamw", LR_SCHED="cosine", GRAD_CLIP="1.0",
            # Real text starts at a far higher CE than the templated v3 corpus (~0.6). CPU check of v3
            # step_85000 on v4 held-out docs (scripts/corpus_v4/cpu_val_baseline.py, first 1,024 ids
            # of 4 docs per source): fineweb 19.8, wiki 23.6, tinystories 24.7, pg19 425 nats/id
            # (old templated corpus 2.85 on the same harness). The v3 guards would fire on the switch
            # itself: rollback if ema_tf > 4.0 for 1000 steps, good if ema_tf < 3.0, and grad-norm
            # skip above 3000 during the 200-step grace. For v4: rollback only on a real collapse,
            # a 2000-step grace with a very high absolute skip threshold (GRAD_CLIP=1.0 still bounds
            # every update), good/ once ema_tf < 4.5. Override via env after the first VAL lines.
            ROLLBACK_EMA_TF=os.environ.get("ROLLBACK_EMA_TF", "30.0"),
            ROLLBACK_EMA_STEPS=os.environ.get("ROLLBACK_EMA_STEPS", "3000"),
            GN_GRACE=os.environ.get("GN_GRACE", "2000"),
            GN_GRACE_ABS=os.environ.get("GN_GRACE_ABS", "1000000"),
            GOOD_EMA_MAX=os.environ.get("GOOD_EMA_MAX", "4.5"),
            HF_HUB_ENABLE_HF_TRANSFER="1")
MAX_ROLLBACKS = int(os.environ.get("MAX_ROLLBACKS", "3"))

def alive(pidfile):
    try:
        pid = int(Path(pidfile).read_text().strip()); os.kill(pid, 0)
        return Path(f"/proc/{pid}/cmdline").read_bytes().find(b"fast4gpu_atom_v3") >= 0
    except Exception:
        return False

if STATE.exists():
    st = json.loads(STATE.read_text())
else:
    if alive("/workspace/trainer_v3.pid"):
        log("v3 trainer still running: refusing to start v4 (stop v3 first)"); sys.exit(3)
    man = json.loads((CORPUS / "manifest.json").read_text())
    good = json.loads((SRC_GOOD / "GOOD.json").read_text())
    FROZEN.mkdir(parents=True, exist_ok=True)
    for f in ("fractus_atom_1b_gpu0.pt", "fractus_atom_1b_gpu0.opt", "RESUME_MANIFEST_atom_gpu0.json", "GOOD.json"):
        if not (FROZEN / f).exists():
            os.link(SRC_GOOD / f, FROZEN / f)
    plan = int(os.environ.get("PLAN_IDS", man["train_ids"]))
    steps = max(1, plan // IDS_PER_STEP)
    st = {"lr": float(os.environ.get("LR_V4", "2.35e-5")), "lr_min": 2.5e-6, "warmup": 500,
          "ckpt_in": str(FROZEN / "fractus_atom_1b_gpu0.pt"), "opt_in": str(FROZEN / "fractus_atom_1b_gpu0.opt"),
          "start_token": 0, "steps_left": steps, "rollbacks": 0, "lr_step_offset": 0, "lr_total_steps": steps,
          "ids_base": int(good["ids_processed"]), "from_v3": good, "plan_ids": plan}
    log("first launch: v3 good", good, "plan_ids", plan, "steps", steps)
OUT.mkdir(parents=True, exist_ok=True)
while True:
    STATE.write_text(json.dumps(st, indent=1))
    env = dict(os.environ, **BASE, LR=str(st["lr"]), LR_MIN=str(st["lr_min"]),
               WARMUP_STEPS=str(st["warmup"]), CKPT_IN=st["ckpt_in"], IDS_BASE=str(st["ids_base"]),
               START_TOKEN=str(st["start_token"]), MAX_STEPS=str(st["steps_left"]),
               LR_STEP_OFFSET=str(st.get("lr_step_offset", 0)), LR_TOTAL_STEPS=str(st.get("lr_total_steps", 0)))
    env.pop("OPT_IN", None)
    if st.get("opt_in"): env["OPT_IN"] = st["opt_in"]
    log("launch", json.dumps(st))
    with open("/workspace/train_v4.log", "a") as f:
        p = subprocess.Popen([sys.executable, "-u", "scripts/fast4gpu_atom_v4.py"], cwd="/workspace/repo",
                             env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        Path("/workspace/trainer_v4.pid").write_text(str(p.pid))
        rc = p.wait()
    log("trainer exit code", rc)
    if rc != 42:
        log("stop (not a rollback request)"); break
    rb = json.loads((OUT / "ROLLBACK.json").read_text())
    (OUT / "ROLLBACK.json").rename(OUT / f"ROLLBACK_{st['rollbacks'] + 1}.json")
    st["rollbacks"] += 1
    if st["rollbacks"] > MAX_ROLLBACKS:
        log("too many rollbacks, stopping cleanly", rb); break
    good = OUT / "good" / "fractus_atom_1b_gpu0.pt"
    if good.exists():
        st["ckpt_in"] = str(good); st["opt_in"] = None   # good/ carries its own .opt
    st["steps_left"] = max(1, st["steps_left"] - int(rb["step_in_run"]))
    if st.get("lr_total_steps"): st["lr_step_offset"] = st.get("lr_step_offset", 0) + int(rb["step_in_run"])
    st["start_token"] = int(rb["start_token_next"])   # data keeps moving forward
    st["lr"] /= 2; st["lr_min"] /= 2
    log("rollback", rb, "->", st["ckpt_in"], "lr", st["lr"])
