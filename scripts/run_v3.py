"""Runner v3 (fast4gpu_atom_v3, ckpt_v3, SS_TAIL_IDS=8): scheduled sampling only on the last 8 input ids
of each 128-id chunk. Same as run_v2.py otherwise: restarts the trainer from the last GOOD checkpoint
with half the LR when the trainer exits with code 42 (rollback request)."""
import json, os, subprocess, sys, time
from pathlib import Path
OUT = Path("/workspace/ckpt_v3")
STATE = Path("/workspace/run_v3_state.json")
LOG = open("/workspace/runner_v3.log", "a", buffering=1)
def log(*a): print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), *a, file=LOG, flush=True)
BASE = dict(CUDA_VISIBLE_DEVICES="0", GPU_ID="0", N_GPU="1", CORPUS="/workspace/atom_corpus.i16",
            BATCH="16", SEQ="128", FRACTUS_ATTN_IMPL="chunked", BLOCK_CKPT="1", COMPILE="1",
            PROBE_EVERY="0", SS_RATE="0.35", SS_PROB_START="0.5", SS_PROB_END="0.85", SS_RAMP_TOKENS="100000000", SS_RAMP_ORIGIN_IDS="4410941440", CARRY_DECAY="1.0", CARRY_CLAMP="0", EPOCH_WRAP="1", REPEAT_COEF="0", LOG_EVERY="200", SAVE_EVERY="5000",
            SS_TAIL_IDS=os.environ.get("SS_TAIL_IDS", "8"),
            OUT_DIR=str(OUT), OPT="adamw", LR_SCHED="cosine", GRAD_CLIP="1.0",
            HF_HUB_ENABLE_HF_TRANSFER="1")
MAX_ROLLBACKS = int(os.environ.get("MAX_ROLLBACKS", "3"))
# Initial state (v2 step_00060000 = 4,636,221,440 ids; v2 segment step N <-> absolute LR step 50,000 + N).
# opt_in: step_60000's .opt no longer exists (rotated); the v2 step_75000 AdamW state (the newest .opt available) is used for the first launch only.
st = json.loads(STATE.read_text()) if STATE.exists() else {
    "lr": 2.5e-5, "lr_min": 2.5e-6, "warmup": 500,
    "ckpt_in": "/workspace/ckpt_v3_resume60k/fractus_atom_1b_gpu0.pt",
    "opt_in": "/workspace/ckpt_v3_resume60k/opt_from_v2_step75000.opt",
    "start_token": 289763840, "steps_left": 2043780, "rollbacks": 0,
    "lr_step_offset": 110000, "lr_total_steps": 2153780}
OUT.mkdir(parents=True, exist_ok=True)
while True:
    STATE.write_text(json.dumps(st, indent=1))
    env = dict(os.environ, **BASE, LR=str(st["lr"]), LR_MIN=str(st["lr_min"]),
               WARMUP_STEPS=str(st["warmup"]), CKPT_IN=st["ckpt_in"],
               START_TOKEN=str(st["start_token"]), MAX_STEPS=str(st["steps_left"]),
               LR_STEP_OFFSET=str(st.get("lr_step_offset", 0)), LR_TOTAL_STEPS=str(st.get("lr_total_steps", 0)))
    env.pop("OPT_IN", None)
    if st.get("opt_in"): env["OPT_IN"] = st["opt_in"]
    log("launch", json.dumps(st))
    with open("/workspace/train_v3.log", "a") as f:
        p = subprocess.Popen([sys.executable, "-u", "scripts/fast4gpu_atom_v3.py"], cwd="/workspace/repo",
                             env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        Path("/workspace/trainer_v3.pid").write_text(str(p.pid))
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
