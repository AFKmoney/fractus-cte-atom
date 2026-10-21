"""Step 1+2: TF NLL of the final ckpt on held-out val + stepwise-vs-TF logit parity."""
import sys,time,json,torch,numpy as np,torch.nn.functional as F
sys.path.insert(0,'/workspace/repo_v4')
from fractus.fixed_generation import load_checkpoint, FixedGenerator, SEQ
torch.set_num_threads(7)
CK='/workspace/ckpt_v4_final/checkpoints/atom_v4/step_final/fractus_atom_1b_gpu0.bin'
eng,cfg=load_checkpoint(CK,torch.float32); G=FixedGenerator(eng)
out=open('verify.jsonl','a')
def emit(r): out.write(json.dumps(r)+'\n'); out.flush(); print(json.dumps(r)[:400],flush=True)
VB='/workspace/v4build'; rng=np.random.default_rng(0); W=[]
for s in ['fineweb','wiki','tinystories','pg19']:
    a=np.fromfile(f'{VB}/{s}_val.i16',dtype=np.int16).astype(np.int64)
    for st in rng.integers(0,len(a)-1152,8): W.append((s,a[st:st+1151].tolist()))
sel=[0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]
# parity first (cheap, the key test)
for wi in [8,16]:
    s,ctx=W[wi]; ctx=ctx[:300]
    lg=G.tf_logits(ctx)
    for p in [5,100,127,128,129,200,255,256,270]:
        seq=ctx[:p+1]; G.cold(); com=0
        while len(seq)-com>SEQ: com=G._commit(seq,com,0)
        st=G.get(); l2=G.chunk(seq[com:])[-1]; G.put(st)
        d=(l2-lg[p]).abs().max().item()
        emit(dict(kind='parity',win=wi,pos=p,maxabs=d,argmax_eq=int(l2.argmax())==int(lg[p].argmax())))
for wi in sel:
    s,ctx=W[wi]; t=time.time()
    lg=G.tf_logits(ctx)[:-1]; tg=torch.tensor(ctx[1:])
    nll=F.cross_entropy(lg,tg,reduction='none')[126:]; top=(lg.argmax(-1)==tg).float()[126:]
    t5=(lg.topk(5,-1).indices==tg[:,None]).any(-1).float()[126:]
    emit(dict(kind='tf',win=wi,src=s,nll=nll.mean().item(),top1=top.mean().item(),top5=t5.mean().item(),secs=time.time()-t))
print('DONE',flush=True)
