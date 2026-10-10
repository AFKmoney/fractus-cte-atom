import sys, time, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
sys.path.insert(0, "/workspace/fractus-cte-atom-main")
from fractus.atom_tokenizer import AtomFractusTokenizer

OUT = Path("/workspace/atom_corpus")
OUT.mkdir(exist_ok=True)

def convert_one(path):
    tok = AtomFractusTokenizer()
    out_path = OUT / (Path(path).stem + ".i16")
    if out_path.exists() and out_path.stat().st_size > 0:
        return Path(path).name, 0, 0, True
    ids = []
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = "\n".join(m.get("content","") for m in obj.get("messages",[]) if isinstance(m, dict))
            if not text.strip():
                continue
            enc, _ = tok.encode_with_features(text)
            ids.extend(enc)
            n += 1
    if ids:
        arr = np.asarray(ids, dtype=np.int16)
        arr.tofile(out_path)
    return Path(path).name, n, len(ids), False

files = sorted(Path("/workspace/fractus-datasets").rglob("*.jsonl"))
print(f"files {len(files)}", flush=True)
total = 0
skipped = 0
t0 = time.time()
with ProcessPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(convert_one, str(p)): p for p in files}
    for i, fut in enumerate(as_completed(futs)):
        name, n, s, was_skip = fut.result()
        if was_skip:
            skipped += 1
        else:
            total += s
        if i % 5 == 0:
            print(f"{i}/{len(files)} {name} docs={n} ids={s} new={total:,} skipped={skipped} {total/max(time.time()-t0,1):.0f}/s", flush=True)
print(f"DONE new={total:,} skipped={skipped}", flush=True)
