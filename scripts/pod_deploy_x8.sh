#!/bin/sh
# Fractus x8 one-shot deploy: env -> assets -> smoke -> launch -> verify.
# Designed to run DETACHED on a fresh vast.ai 8x5090 pod (POSIX sh, no bashisms):
#   tr -d '\r' < pod_deploy_x8.sh > /root/deploy.sh && nohup sh /root/deploy.sh > /root/deploy.log 2>&1 &
# Progress: tail -f /root/deploy.log   (marker files /root/.st_* make re-runs cheap)
set -e
# prerequisite: HF token already written to /root/.hf_token (kept OUT of this file)
[ -s /root/.hf_token ] || { echo "ERROR: write HF token to /root/.hf_token first"; exit 1; }
chmod 600 /root/.hf_token
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ ! -f /root/.st_env ]; then
  echo "=== [1/5] pip: huggingface_hub + hf_transfer"
  pip install -q -U huggingface_hub hf_transfer
  touch /root/.st_env
fi

if [ ! -f /root/.st_torch ]; then
  echo "=== [2/5] torch cu128 (sm_120 needs >=2.7) — several GB, be patient"
  pip install -q -U torch --index-url https://download.pytorch.org/whl/cu128
  python3 -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"
  touch /root/.st_torch
fi

if [ ! -f /root/.st_assets ]; then
  echo "=== [3/5] assets: code + manifests + 8 ckpts (~36GB) + 8 shards (~14GB)"
  python3 - << 'PYEOF'
import os, json, re, shutil
from huggingface_hub import snapshot_download, hf_hub_download, HfApi
tok = open('/root/.hf_token').read().strip()
REPO, DREPO = 'thefinalboss/fractus-cte', 'thefinalboss/fractus-datasets'
api = HfApi(token=tok)

print('-- code', flush=True)
snapshot_download(REPO, repo_type='model', local_dir='/root/fractus-opt', token=tok,
                  allow_patterns=['fractus/*', 'scripts/*', 'tests/*', 'benchmarks/*',
                                  'docs/*', 'README.md'])

print('-- manifests', flush=True)
for f in ('checkpoints/X6_MANIFEST.json', 'checkpoints/RESUME_MANIFEST_8GPU.json'):
    p = hf_hub_download(REPO, f, repo_type='model', token=tok)
    shutil.copy(p, '/root/' + os.path.basename(f))
x6 = json.load(open('/root/X6_MANIFEST.json'))['gpus']
r8 = json.load(open('/root/RESUME_MANIFEST_8GPU.json'))

print('-- checkpoints x8', flush=True)
os.makedirs('/root/ckpts', exist_ok=True)
offs = {}
for i in range(8):
    key = f'gpu_{i}'
    if i < 6 and key in x6 and x6[key].get('start_token') is not None:
        src, offs[i] = f'checkpoints/x6run/', int(x6[key]['start_token'])
    elif key in r8:
        src, offs[i] = 'checkpoints/', int(r8[key]['start_token'])
    elif str(i) in r8:
        src, offs[i] = 'checkpoints/', int(r8[str(i)]['start_token'])
    else:
        raise RuntimeError(f'no offset for gpu {i}')
    p = hf_hub_download(REPO, src + f'fractus_1b_gpu{i}.pt', repo_type='model', token=tok)
    shutil.copy(p, f'/root/ckpts/fractus_1b_gpu{i}.pt')
    print('  ckpt', i, 'resume_at', offs[i], flush=True)

print('-- shards x8', flush=True)
os.makedirs('/root/shards', exist_ok=True)
npys = [f.path for f in api.list_repo_tree(DREPO, repo_type='dataset', recursive=True)
        if f.path.endswith('.npy')]
phase2 = [p for p in npys if 'phase2' in p.lower()] or npys
by_idx = {}
for p in phase2:
    m = re.search(r'gpu[_\-]?(\d+)', os.path.basename(p))
    if m is not None:
        by_idx.setdefault(int(m.group(1)), p)
ordered = [by_idx.get(i) for i in range(8)]
if any(v is None for v in ordered):
    ordered = sorted(phase2)[:8]
assert len(ordered) == 8, f'expected 8 shards, got {len(ordered)}'
shard_path = {}
for i, p in enumerate(ordered):
    q = hf_hub_download(DREPO, p, repo_type='dataset', token=tok)
    dst = f'/root/shards/shard_gpu{i}.npy'
    if not os.path.exists(dst):
        shutil.copy(q, dst)
    shard_path[i] = dst
    print('  shard', i, '<-', os.path.basename(p), flush=True)

json.dump({'offsets': {str(k): v for k, v in offs.items()},
           'shards': {str(k): v for k, v in shard_path.items()}},
          open('/root/assets.json', 'w'), indent=1)

# generate launch_x8.sh with literal values (no shell indirection)
L = ['#!/bin/sh', 'mkdir -p /root/logs /root/ckpts_run']
for i in range(8):
    L.append(
        f"CUDA_VISIBLE_DEVICES={i} GPU_ID={i} START_TOKEN={offs[i]} BATCH=__BATCH__ "
        f"SEQ=128 CE_CHUNK=2048 FRACTUS_ATTN_IMPL=chunked BLOCK_CKPT=1 "
        f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"CKPT_IN=/root/ckpts/fractus_1b_gpu{i}.pt "
        f"CKPT_OUT=/root/ckpts_run/fractus_1b_gpu{i}.pt "
        f"SHARD={shard_path[i]} "
        f"python -u /root/fractus-opt/scripts/fast4gpu_boost_v2.py >> /root/logs/g{i}.log 2>&1 &")
open('/root/launch_x8.sh', 'w').write('\n'.join(L) + '\n')

sync = '''#!/usr/bin/env python3
"""Hourly HF safety-sync: parses token positions from logs, uploads ckpt_run
files (never while .tmp exists) + X8_MANIFEST.json."""
import re, json, time
from huggingface_hub import HfApi
tok = open('/root/.hf_token').read().strip()
api, REPO = HfApi(token=tok), 'thefinalboss/fractus-cte'
while True:
    time.sleep(3600)
    try:
        man = {'status': 'x8_running', 'gpus': {}}
        for i in range(8):
            pos = None
            try:
                txt = open(f'/root/logs/g{i}.log').read()
                for m in re.finditer(r'GPU \\d:\\s+([\\d,]+) tf=', txt):
                    pos = int(m.group(1).replace(',', ''))
            except OSError:
                pass
            man['gpus'][f'gpu_{i}'] = {'start_token': pos}
            ck, tmp = f'/root/ckpts_run/fractus_1b_gpu{i}.pt', f'/root/ckpts_run/fractus_1b_gpu{i}.pt.tmp'
            if pos is not None and not os.path.exists(tmp):
                api.upload_file(path_or_fileobj=ck,
                                path_in_repo=f'checkpoints/x8run/fractus_1b_gpu{i}.pt',
                                repo_id=REPO, repo_type='model',
                                commit_message='x8 hourly sync gpu%d @ %s' % (i, pos))
                print('synced', i, pos, flush=True)
        api.upload_file(path_or_fileobj=json.dumps(man, indent=1).encode(),
                        path_in_repo='checkpoints/X8_MANIFEST.json',
                        repo_id=REPO, repo_type='model',
                        commit_message='x8 manifest')
    except Exception as e:
        print('sync error (retry next hour):', e, flush=True)
'''
sync = sync.replace('import re, json, time', 'import re, json, time, os')
open('/root/sync_hf.py', 'w').write(sync)

smoke = '''#!/usr/bin/env python3
"""Real-shape throughput probe on GPU0: tick_chunk_train_ce under bf16 autocast."""
import os, sys, time
import torch
sys.path.insert(0, '/root/fractus-opt')
from fractus.continuous_engine import ContinuousThoughtEngine
B = int(os.environ.get('B', '8'))
CFG = dict(vocab_size=50257, d_model=1280, n_heads=20, d_head=64, n_levels=2,
           n_oscillators=16, coupling_rank=8, n_experts=128, top_k=2,
           expert_d_ff=2048, siren_rank=64, n_layers=16)
torch.manual_seed(0)
eng = ContinuousThoughtEngine(**CFG).cuda()
eng.reset_thought(batch_size=B)
opt = torch.optim.SGD(eng.parameters(), lr=7e-4, momentum=0.9)
toks = torch.randint(0, 50257, (B, 129), device='cuda')
chunk, target = toks[:, :-1], toks[:, 1:]
times = []
ok = True
for i in range(24):
    try:
        t0 = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ce, lb = eng.tick_chunk_train_ce(chunk, target, ce_chunk=2048, block_ckpt=True)
            loss = ce + 0.02 * lb
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        if i >= 6:
            times.append(time.perf_counter() - t0)
    except torch.cuda.OutOfMemoryError:
        ok = False
        break
if not ok or len(times) < 3:
    print(f'B={B} RESULT FAIL oom')
    raise SystemExit(1)
med = sorted(times)[len(times) // 2]
mem = torch.cuda.max_memory_allocated() / 1e9
print(f'B={B} RESULT tok_s_per_gpu={B * 128 / med:.0f} peakVRAM={mem:.2f}GB')
'''
open('/root/gpu_smoke.py', 'w').write(smoke)
print(json.dumps(json.load(open('/root/assets.json')), indent=1), flush=True)
PYEOF
  touch /root/.st_assets
fi

echo "=== [4/5] smoke GPU0: B=8 then B=4 fallback"
B_CHOSEN=""
for b in 8 4; do
  if B=$b python3 /root/gpu_smoke.py | tee /tmp/smoke_b$b.txt | grep -q "RESULT"; then
    B_CHOSEN=$b
    grep "RESULT" /tmp/smoke_b$b.txt
    break
  else
    echo "B=$b failed (see /tmp/smoke_b$b.txt)"
  fi
done
[ -z "$B_CHOSEN" ] && { echo "NO_BATCH_WORKS"; exit 1; }
sed -i "s/BATCH=__BATCH__/BATCH=$B_CHOSEN/" /root/launch_x8.sh

echo "=== [5/5] LAUNCH x8 (B=$B_CHOSEN) + hourly HF sync"
rm -f /root/logs/g*.log
mkdir -p /root/logs
nohup sh /root/launch_x8.sh > /root/launch.log 2>&1 &
nohup python3 -u /root/sync_hf.py >> /root/logs/sync.log 2>&1 &
sleep 150
echo "--- processes:"; pgrep -fc 'fast4gpu_boost_v2' || true
echo "--- GPUs:"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
for i in 0 1 2 3 4 5 6 7; do echo "g$i: $(grep -E 'boostv2|Error|error' /root/logs/g$i.log | tail -n 1)"; done
touch /root/.deploy_done
echo "=== DEPLOY_COMPLETE"
