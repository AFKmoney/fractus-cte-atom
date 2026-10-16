# manual one-shot push mirroring push_good.py v2 push() (own stage dir, real copy snapshot)
import os, json, shutil, time, subprocess
for l in open("/workspace/hf.env"):
    k,_,v=l.strip().partition("=")
    if k=="HF_TOKEN": os.environ["HF_TOKEN"]=v
from huggingface_hub import HfApi
REPO="thefinalboss/fractus-cte-atom"; G="/workspace/ckpt_full/good"; LOG="/workspace/train.log"
s="/workspace/manual_push_stage"; shutil.rmtree(s,ignore_errors=True); os.makedirs(s)
names={"fractus_atom_1b_gpu0.pt":"fractus_atom_1b_gpu0.bin","fractus_atom_1b_gpu0.opt":"fractus_atom_1b_gpu0.opt.bin",
       "RESUME_MANIFEST_atom_gpu0.json":"RESUME_MANIFEST_atom_gpu0.json","GOOD.json":"GOOD.json"}
info=json.load(open(G+"/GOOD.json"))
for a,b in names.items(): shutil.copy2(f"{G}/{a}",f"{s}/{b}")
man=json.load(open(s+"/RESUME_MANIFEST_atom_gpu0.json")); g2=json.load(open(s+"/GOOD.json"))
assert man["ids_processed"]==g2["ids_processed"]==info["ids_processed"], "snapshot inconsistent"
assert os.path.getsize(s+"/fractus_atom_1b_gpu0.bin")==os.path.getsize(G+"/fractus_atom_1b_gpu0.pt")
os.system(f"tail -n 300 {LOG} > {s}/train_tail.log")
print("snapshot ok",info,time.ctime(),flush=True)
api=HfApi()
for p in ("checkpoints/atom","checkpoints/atom_latest"):
    r=api.upload_folder(folder_path=s,repo_id=REPO,path_in_repo=p,
        commit_message="atom good ckpt %d ids (ema_tf %.3f) -> %s [manual]"%(info["ids_processed"],info["ema_tf"],p))
    print("committed",p,r.oid if hasattr(r,"oid") else r,time.ctime(),flush=True)
print("DONE",time.ctime(),flush=True)
