import os, sys, time, shutil, json, traceback
for l in open('/workspace/hf.env'):
    k,_,v=l.strip().partition('=')
    if k=='HF_TOKEN': os.environ['HF_TOKEN']=v
from huggingface_hub import HfApi
REPO='thefinalboss/fractus-cte-atom'; G='/workspace/ckpt_full/good'
INTERVAL=int(os.environ.get('PUSH_INTERVAL','36000'))
api=HfApi(); last=None; last_t=0.0
def push(info):
    st='/workspace/push_stage_good'; shutil.rmtree(st,ignore_errors=True); os.makedirs(st)
    names={'fractus_atom_1b_gpu0.pt':'fractus_atom_1b_gpu0.bin','fractus_atom_1b_gpu0.opt':'fractus_atom_1b_gpu0.opt.bin',
           'RESUME_MANIFEST_atom_gpu0.json':'RESUME_MANIFEST_atom_gpu0.json','GOOD.json':'GOOD.json'}
    for a,b in names.items():
        if os.path.exists(f'{G}/{a}'): os.link(f'{G}/{a}',f'{st}/{b}')
    os.system(f'tail -n 300 /workspace/train.log > {st}/train_tail.log')
    api.upload_folder(folder_path=st,repo_id=REPO,repo_type='model',path_in_repo='checkpoints/atom',
        commit_message=f"atom GOOD ckpt {info['ids_processed']} ids (ema_tf {info['ema_tf']:.3f})")
    shutil.rmtree(st,ignore_errors=True)
    print('pushed GOOD',info,time.ctime(),flush=True)
while True:
    try:
        if os.path.exists(G+'/GOOD.json'):
            info=json.load(open(G+'/GOOD.json'))
            key=(info['ids_processed'],info['ts'])
            if key!=last and (last is None or time.time()-last_t>=INTERVAL):
                push(info); last=key; last_t=time.time()
    except Exception: traceback.print_exc()
    time.sleep(60)
