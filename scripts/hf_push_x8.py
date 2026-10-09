#!/usr/bin/env python3
import os, time, json
from pathlib import Path
from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]
REPO = 'thefinalboss/fractus-cte'
CKPT_DIR = Path('/workspace/fractus-p0/checkpoints')
STATE = Path('/workspace/hf_push_state.json')
api = HfApi(token=TOKEN)

def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {}

def save_state(s):
    STATE.write_text(json.dumps(s))

def once():
    st = load_state()
    uploaded = []
    for i in range(8):
        p = CKPT_DIR / f'fractus_1b_gpu{i}_live.pt'
        if not p.exists() or p.stat().st_size < 1_000_000_000:
            continue
        key = f'gpu{i}'
        mtime = p.stat().st_mtime
        size = p.stat().st_size
        prev = st.get(key) or {}
        if prev.get('mtime') == mtime and prev.get('size') == size:
            continue
        dest = f'checkpoints/x8run/fractus_1b_gpu{i}.pt'
        print(f'UPLOAD {p} -> {REPO}/{dest} {size/1e9:.2f}G', flush=True)
        api.upload_file(path_or_fileobj=str(p), path_in_repo=dest, repo_id=REPO, repo_type='model')
        man = CKPT_DIR / f'RESUME_gpu{i}.json'
        if man.exists():
            api.upload_file(path_or_fileobj=str(man), path_in_repo=f'checkpoints/x8run/RESUME_gpu{i}.json', repo_id=REPO, repo_type='model')
        st[key] = {'mtime': mtime, 'size': size, 't': time.time()}
        save_state(st)
        uploaded.append(i)
    print('uploaded', uploaded, flush=True)
    return uploaded

if __name__ == '__main__':
    while True:
        try:
            once()
        except Exception as e:
            print('PUSH_ERR', e, flush=True)
        time.sleep(3600)
