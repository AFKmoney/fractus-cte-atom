"""Decoding sweep on CONT corpus docs (final v4 ckpt). Resumable jsonl."""
import sys,os,json,time,re,torch,numpy as np
sys.path.insert(0,'/workspace/repo_v4')
from fractus.fixed_generation import load_checkpoint, FixedGenerator, show, pick_next
torch.set_num_threads(7)
OUT=os.environ.get('OUT','tune.jsonl'); NNEW=int(os.environ.get('NNEW',80))
eng,cfg=load_checkpoint('/workspace/ckpt_v4_final/checkpoints/atom_v4/step_final/fractus_atom_1b_gpu0.bin',torch.float32); G=FixedGenerator(eng)
WORDS=set(open('/workspace/words_alpha.txt').read().split())
done=set()
if os.path.exists(OUT):
    for l in open(OUT): d=json.loads(l); done.add((d['cfg'],d['doc'],d.get('seed',0)))
docs=[]
for s in ['fineweb','wiki','tinystories']:
    a=np.fromfile(f'/workspace/v4build/{s}_val.i16',dtype=np.int16).astype(int); b=np.flatnonzero(a==257)
    lens=np.diff(np.append(b,len(a))); ks=np.flatnonzero(lens>700)[[3,6,9,12]]
    for k in ks: docs.append((f'{s}_doc{k}',a[b[k]-64:b[k]+256].tolist(),a[b[k]+256:b[k]+256+NNEW].tolist()))
def txt(ids): return bytes(i for i in ids if i<256).decode('utf-8','replace')
def rw(t,m=2):
    ws=re.findall(r'[A-Za-z]{%d,}'%m,t); return (sum(w.lower() in WORDS for w in ws)/len(ws)) if ws else 0.0
def old_space(ctx,n,T=0.7,k=40,seed=0):
    from fractus.generate_aligned import _reset
    g=torch.Generator().manual_seed(seed); out=[]
    with torch.inference_mode():
        _reset(eng); lg=eng.tick_chunk(torch.tensor([ctx]))[0,-1]
        for _ in range(n):
            v,i=torch.topk(lg.float()/T,k); nid=int(i[torch.multinomial(torch.softmax(v,-1),1,generator=g)]); out.append(nid)
            lg=eng.tick_chunk(torch.tensor([[nid]]))[0,-1]
    return out
CFGS=json.loads(os.environ['CFGS'])
for name,kw in CFGS:
    seeds=kw.pop('seeds',[0])
    for sd in seeds:
        for dn,ctx,true in docs:
            if (name,dn,sd) in done: continue
            t=time.time()
            if name.startswith('oldspace'): gen=old_space(ctx[64:] if kw.get('cold') else ctx,NNEW,seed=sd)
            else: gen=G.generate_stream(ctx,NNEW,seed=sd,stop_at_eos=False,**kw)
            r=dict(cfg=name,doc=dn,seed=sd,u40=len(set(gen[:40])),exact40=sum(p==q for p,q in zip(gen[:40],true)),
                   rw40=rw(txt(gen[:40])),rw=rw(txt(gen)),rw3=rw(txt(gen),3),n_eos=gen.count(258),gen=show(gen),true=show(true),secs=round(time.time()-t,1))
            open(OUT,'a').write(json.dumps(r,ensure_ascii=False)+'\n'); print(json.dumps(r,ensure_ascii=False)[:250],flush=True)
print('ALLDONE',flush=True)
