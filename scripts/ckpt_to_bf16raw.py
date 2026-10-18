"""Stream an fp32 Fractus checkpoint (.pt/.bin, {'model_state','config'}) into a raw bf16 blob + JSON index,
one tensor at a time (low anon RSS). gen_block.py loads it via np.memmap (file-backed pages, no OOM risk).
Usage: python3 scripts/ckpt_to_bf16raw.py IN.pt OUT_DIR"""
import sys,os,json,torch,numpy as np
src,dst=sys.argv[1],sys.argv[2]; os.makedirs(dst,exist_ok=True)
ck=torch.load(src,map_location='cpu',weights_only=False,mmap=True)
sd=ck['model_state']; idx={'config':ck['config'],'tensors':{}}; off=0
with open(os.path.join(dst,'weights.bf16'),'wb') as f:
    for k,t in sd.items():
        if t.is_floating_point(): b=t.detach().to(torch.bfloat16).contiguous().view(torch.int16); dt='bfloat16'
        else: b=t.detach().contiguous(); dt=str(t.dtype).split('.')[-1]
        a=b.numpy(); f.write(a.tobytes()); n=a.nbytes
        idx['tensors'][k]={'off':off,'nbytes':n,'shape':list(t.shape),'dtype':dt}; off+=n; del b,a
json.dump(idx,open(os.path.join(dst,'index.json'),'w'))
print('wrote',off,'bytes',len(sd),'tensors')
