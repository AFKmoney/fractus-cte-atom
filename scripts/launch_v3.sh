cd /workspace
mkdir -p /workspace/ckpt_v3
nohup python -u /workspace/run_v3.py > /workspace/run_v3.out 2>&1 </dev/null &
echo $! > /workspace/run_v3.pid
nohup python -u /workspace/push_good_atom_v3.py > /workspace/push_v3.log 2>&1 </dev/null &
echo $! > /workspace/push_v3.pid
echo launched
