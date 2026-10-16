cd /workspace
mkdir -p /workspace/ckpt_full
nohup python -u /workspace/run8.py > /workspace/run8.out 2>&1 </dev/null &
echo $! > /workspace/train.pid
nohup python -u /workspace/push_good.py > /workspace/push.log 2>&1 </dev/null &
echo $! > /workspace/push.pid
echo launched
