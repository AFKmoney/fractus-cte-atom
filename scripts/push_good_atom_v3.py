# push_good v3 (atom_v3 run: ckpt_v3 -> checkpoints/atom_v3, SS_TAIL_IDS=8): relaxed "good" watcher + pusher (does not touch the trainer)
import os, re, time, shutil, json, traceback, glob
for l in open('/workspace/hf.env'):
    k,_,v=l.strip().partition('=')
    if k=='HF_TOKEN': os.environ['HF_TOKEN']=v
from huggingface_hub import HfApi
REPO='thefinalboss/fractus-cte-atom'; OUT='/workspace/ckpt_v3'; G=OUT+'/good'
LOG='/workspace/train_v3.log'; STATE='/workspace/push_good_v3_state.json'
INTERVAL=int(os.environ.get('PUSH_INTERVAL','36000'))
EMA_MAX=float(os.environ.get('RELAX_EMA_MAX','3.0'))
WIN=int(os.environ.get('RELAX_WINDOW_STEPS','2000')); MAX_SKIPS=int(os.environ.get('RELAX_MAX_SKIPS','20'))
IDS_PER_STEP=2048
api=HfApi()
st={'last_key':None,'last_t':0.0}
if os.path.exists(STATE): st.update(json.load(open(STATE)))
def save_state(): json.dump(st,open(STATE,'w'))
RE_LINE=re.compile(r'^GPU 0:\s+([\d,]+) tf=.*?gn_ema=([\d.]+) skips=(\d+)')
RE_SAVED=re.compile(r'saved -> .*\(([\d,]+) ids\)')
RE_CK=re.compile(r'ckpt (step_\d+): ema_tf=([\d.]+)')
def scan():
    lines=open(LOG,errors='replace').read().splitlines()
    m=max(i for i,l in enumerate(lines) if 'lr schedule=cosine' in l)  # current trainer segment only
    pts=[]; cks=[]; last_ids=None
    for l in lines[m:]:
        a=RE_LINE.match(l)
        if a: pts.append((int(a[1].replace(',','')),float(a[2]),int(a[3]))); continue
        a=RE_SAVED.search(l)
        if a: last_ids=int(a[1].replace(',','')); continue
        a=RE_CK.search(l)
        if a and last_ids: cks.append((a[1],float(a[2]),last_ids))
    out=[]
    for name,ema,ids in cks:
        before=[p for p in pts if p[0]<=ids]; start=[p for p in pts if p[0]<=ids-WIN*IDS_PER_STEP]
        if not before or not start: continue
        out.append(dict(step_dir=name,ids_processed=ids,ema_tf=ema,gn_ema_tf=before[-1][1],
            skips_total=before[-1][2],skips_window=before[-1][2]-start[-1][2]))
    return out
def complete(d,ids):
    try:
        man=json.load(open(d+'/RESUME_MANIFEST_atom_gpu0.json'))
        return man['ids_processed']==ids and os.path.exists(d+'/fractus_atom_1b_gpu0.pt') and os.path.exists(d+'/fractus_atom_1b_gpu0.opt')
    except Exception: return False
def promote(c):
    d=f"{OUT}/{c['step_dir']}"; tmp=OUT+'/good.relaxed.tmp'; shutil.rmtree(tmp,ignore_errors=True); os.makedirs(tmp)
    for f in ('fractus_atom_1b_gpu0.pt','fractus_atom_1b_gpu0.opt','RESUME_MANIFEST_atom_gpu0.json'): os.link(f'{d}/{f}',f'{tmp}/{f}')
    if not complete(tmp,c['ids_processed']): shutil.rmtree(tmp,ignore_errors=True); return False
    man=json.load(open(tmp+'/RESUME_MANIFEST_atom_gpu0.json'))
    info=dict(c,start_token_next=man['start_token_next'],ts=man['ts'],criterion=f'relaxed: ema_tf<{EMA_MAX}, skips in last {WIN} steps<{MAX_SKIPS}, no ROLLBACK.json')
    json.dump(info,open(tmp+'/GOOD.json','w'),indent=1)
    old=OUT+'/good.old'; shutil.rmtree(old,ignore_errors=True)
    if os.path.exists(G): os.replace(G,old)
    os.replace(tmp,G); shutil.rmtree(old,ignore_errors=True)
    print('promoted to good/',info,time.ctime(),flush=True); return True
def push(info):
    s='/workspace/push_stage_good_v3'; shutil.rmtree(s,ignore_errors=True); os.makedirs(s)
    names={'fractus_atom_1b_gpu0.pt':'fractus_atom_1b_gpu0.bin','fractus_atom_1b_gpu0.opt':'fractus_atom_1b_gpu0.opt.bin',
           'RESUME_MANIFEST_atom_gpu0.json':'RESUME_MANIFEST_atom_gpu0.json','GOOD.json':'GOOD.json'}
    for a,b in names.items():
        if os.path.exists(f'{G}/{a}'): os.link(f'{G}/{a}',f'{s}/{b}')
    os.system(f'tail -n 300 {LOG} > {s}/train_tail.log')
    for p in ('checkpoints/atom_v3',):
        api.upload_folder(folder_path=s,repo_id=REPO,path_in_repo=p,
            commit_message=f"atom good ckpt {info['ids_processed']} ids (ema_tf {info['ema_tf']:.3f}) -> {p}")
    shutil.rmtree(s,ignore_errors=True); print('pushed good',info,time.ctime(),flush=True)
while True:
    try:
        if not glob.glob(OUT+'/ROLLBACK*.json') or True:
            cur=json.load(open(G+'/GOOD.json')) if os.path.exists(G+'/GOOD.json') else {'ids_processed':0}
            if not os.path.exists(OUT+'/ROLLBACK.json'):
                for c in sorted(scan(),key=lambda c:-c['ids_processed']):
                    if c['ids_processed']<=cur['ids_processed']: break
                    if c['ema_tf']<EMA_MAX and c['skips_window']<MAX_SKIPS and complete(f"{OUT}/{c['step_dir']}",c['ids_processed']):
                        promote(c); break
            if os.path.exists(G+'/GOOD.json'):
                info=json.load(open(G+'/GOOD.json')); key=[info['ids_processed'],info['ts']]
                if key!=st['last_key'] and time.time()-st['last_t']>=INTERVAL:
                    push(info); st['last_key']=key; st['last_t']=time.time(); save_state()
    except Exception: traceback.print_exc()
    time.sleep(60)
