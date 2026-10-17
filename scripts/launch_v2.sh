cd /workspace
mkdir -p /workspace/ckpt_v2
nohup python -u /workspace/run_v2.py > /workspace/run_v2.out 2>&1 </dev/null &
echo $! > /workspace/train_v2.pid
nohup python -u /workspace/push_good_atom_v2.py > /workspace/push_v2.log 2>&1 </dev/null &
echo $! > /workspace/push_v2.pid
echo launched
