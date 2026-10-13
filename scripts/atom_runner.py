"""Pod runner (copied from the training pod; paths are pod paths). Runner for the Atom resume: restarts the trainer from the last GOOD checkpoint
with half the LR when the trainer exits with code 42 (rollback request)."""
import json, os, subprocess, sys, time
from pathlib import Path
OUT = Path("/workspace/ckpt_full")
STATE = Path("/workspace/run8_state.json")
LOG = open("/workspace/runner.log", "a", buffering=1)
def log(*a): print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), *a, file=LOG, flush=True)
BASE = dict(CUDA_VISIBLE_DEVICES="0", GPU_ID="0", N_GPU="1", CORPUS="/workspace/atom_corpus.i16",
            BATCH="16", SEQ="128", FRACTUS_ATTN_IMPL="chunked", BLOCK_CKPT="1", COMPILE="1",
            PROBE_EVERY="0", SS_RATE="0.25", REPEAT_COEF="0", LOG_EVERY="200", SAVE_EVERY="5000",
            OUT_DIR=str(OUT), OPT="adamw", LR_SCHED="cosine", GRAD_CLIP="1.0",
            HF_HUB_ENABLE_HF_TRANSFER="1")
MAX_ROLLBACKS = int(os.environ.get("MAX_ROLLBACKS", "3"))
if STATE.exists():
    st = json.loads(STATE.read_text())
else:
    st = {"lr": 1e-4, "lr_min": 1e-5, "warmup": 500,
          "ckpt_in": "/workspace/ckpt_resume409/fractus_atom_1b_gpu0.pt",
          "start_token": 25600000, "steps_left": 1953780, "rollbacks": 0}
while True:
    STATE.write_text(json.dumps(st, indent=1))
    env = dict(os.environ, **BASE, LR=str(st["lr"]), LR_MIN=str(st["lr_min"]),
               WARMUP_STEPS=str(st["warmup"]), CKPT_IN=st["ckpt_in"],
               START_TOKEN=str(st["start_token"]), MAX_STEPS=str(st["steps_left"]))
    log("launch", json.dumps(st))
    with open("/workspace/train.log", "a") as f:
        p = subprocess.Popen([sys.executable, "-u", "scripts/fast4gpu_atom.py"], cwd="/workspace/repo",
                             env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        Path("/workspace/trainer.pid").write_text(str(p.pid))
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
        st["ckpt_in"] = str(good)
    st["steps_left"] = max(1, st["steps_left"] - int(rb["step_in_run"]))
    st["start_token"] = int(rb["start_token_next"])   # data keeps moving forward
    st["lr"] /= 2; st["lr_min"] /= 2
    log("rollback", rb, "->", st["ckpt_in"], "lr", st["lr"])
