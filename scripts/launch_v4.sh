# v4 launcher (PREPARED, NOT RUN). Stop v3 first (runner_v3 + trainer_v3 + push_good_v3), then:
cd /workspace
mkdir -p /workspace/ckpt_v4
nohup python -u /workspace/run_v4.py > /workspace/run_v4.out 2>&1 </dev/null &
echo $! > /workspace/run_v4.pid
RELAX_EMA_MAX=4.5 nohup python -u /workspace/push_good_atom_v4.py > /workspace/push_v4.log 2>&1 </dev/null &
echo $! > /workspace/push_v4.pid
echo launched
