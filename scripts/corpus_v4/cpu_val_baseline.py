"""CPU baseline: teacher-forced CE of a v3 checkpoint on the start of each v4 source's held-out
docs and on the old templated corpus. Same forward as training (tick_chunk_train_ce, carry kept
across 128-id chunks, fresh carry at the first chunk). Read-only, box only.

python scripts/corpus_v4/cpu_val_baseline.py CKPT WORK_DIR [chunks]
"""
import json, sys, time
from pathlib import Path
import os
os.environ.setdefault("FRACTUS_ATTN_IMPL", "chunked")   # same as the trainer; must be set before the import
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fractus.fixed_generation import load_checkpoint

ck, work = sys.argv[1], Path(sys.argv[2])
chunks = int(sys.argv[3]) if len(sys.argv) > 3 else 8
torch.set_num_threads(4)
eng, cfg = load_checkpoint(ck, torch.float32)
SEQ, B = 128, 4
res = {"ckpt": ck, "ids_processed": cfg.get("ids_processed"), "chunks": chunks, "lanes": B}
srcs = [(s, work / f"{s}_val.i16") for s in ("fineweb", "wiki", "pg19", "tinystories")] + [("old_templated", work / "old.i16")]
for name, p in srcs:
    a = np.fromfile(p, dtype=np.int16)
    lane = len(a) // B
    bos = np.flatnonzero(a == 257)
    st0 = [int(bos[min(len(bos) - 1, np.searchsorted(bos, b * lane))]) for b in range(B)]   # lanes start on a BOS
    eng.reset_thought(B)
    tot = 0.0; t0 = time.time()
    with torch.no_grad():
        for c in range(chunks):
            rows = np.stack([a[st0[b] + c * SEQ: st0[b] + c * SEQ + SEQ + 1] for b in range(B)]).astype(np.int64)
            x = torch.from_numpy(rows)
            m = (x[:, :SEQ] == 257).any(dim=1)          # per-row carry reset at BOS, as the trainer does
            if bool(m.any()):
                k = (~m).float()
                eng.thought_state = eng.thought_state * k.view(-1, 1, 1)
                for blk in eng.blocks:
                    blk.attn_S = blk.attn_S * k.view(-1, 1, 1, 1)
                    blk.attn_z = blk.attn_z * k.view(-1, 1, 1)
                    blk.kuramoto_phases = blk.kuramoto_phases * k.view(-1, 1, 1)
            ce, _ = eng.tick_chunk_train_ce(x[:, :SEQ], x[:, 1:], ce_chunk=2048)
            tot += float(ce)
    res[name] = round(tot / chunks, 4)
    print(name, res[name], f"{time.time() - t0:.0f}s", flush=True)
print(json.dumps(res))
