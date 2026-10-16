"""Carry-norm measurement + partial carry reset + inference-side scheduled sampling.
PHASE=ref|gen ; CONFIGS=mode:C:T:k:rp:state:fix:ss  fix in none|half|clamp ; ss = fraction (0..1)
Appends one JSON line per prompt to OUT."""
import sys,os,json,time,random
sys.argv=['x']; sys.path.insert(0,os.path.dirname(__file__))
from gen_block import *
import numpy as np
OUT=os.environ.get('OUT','/workspace/gs2.jsonl'); REF='/workspace/gs2_ref.json'
def norms():
    S=[float(b.attn_S.float().norm()) for b in eng.blocks]; Smx=[float(b.attn_S.float().abs().max()) for b in eng.blocks]
    z=[float(b.attn_z.float().norm()) for b in eng.blocks]; zmx=[float(b.attn_z.float().abs().max()) for b in eng.blocks]
    ph=[float(b.kuramoto_phases.float().norm()) for b in eng.blocks]
    th=float(eng.thought_state.float().norm())
    m=lambda v:(round(sum(v)/len(v),3),round(max(v),3))
    return {'S_fro':m(S),'S_max':m(Smx),'z_l2':m(z),'z_max':m(zmx),'ph_l2':m(ph),'thought_l2':round(th,3),'_S':S,'_z':z}
def emit(r):
    with open(OUT,'a') as f: f.write(json.dumps(r,ensure_ascii=False)+'\n')
if os.environ.get('PHASE')=='ref':
    d=np.fromfile(PRIMER_FILE,dtype=np.int16,count=200000).astype(np.int64).tolist(); i=d.index(257); d=d[i:]
    cold(); ref=[]
    with torch.inference_mode():
        for j in range(int(os.environ.get('NREF','40'))):
            x=d[j*128:j*128+129]
            if 257 in x[1:]: print('BOS at chunk',j)
            lg=tick(x[:-1]); n=norms(); n['tf']=round(float(F.cross_entropy(lg.float(),torch.tensor(x[1:]))),3); n['chunks']=j+1; ref.append(n)
            print(j+1,n['tf'],n['S_fro'],n['z_l2'],n['thought_l2'],flush=True)
    json.dump(ref,open(REF,'w')); sys.exit()
ref=json.load(open(REF))
ref=ref[:27]   # chunk 28 contains BOS (reset); keep one clean document
def refat(t):
    if t<=len(ref): return ref[t-1]
    r=ref[-1]; f=t/len(ref)   # S,z grow ~linearly with chunks in TF; extrapolate
    return {'S_fro':[x*f for x in r['S_fro']],'z_l2':[x*f for x in r['z_l2']],'thought_l2':r['thought_l2'],'_S':[x*f for x in r['_S']],'_z':[x*f for x in r['_z']]}
TICKS=[0]
def tk(ids):
    TICKS[0]+=1; return tick(ids)
def fixcarry(fix):
    if fix=='half':
        for b in eng.blocks: b.attn_S=b.attn_S*0.5; b.attn_z=b.attn_z*0.5
    elif fix=='clamp':
        r=refat(TICKS[0])
        for i,b in enumerate(eng.blocks):
            for name,key in (('attn_S','_S'),('attn_z','_z')):
                t=getattr(b,name); n=float(t.float().norm()); lim=r[key][i]
                if n>lim>0: setattr(b,name,t*(lim/n))
def run(prompt,mode,C,T,k,rp,state,fix,ss,N,rng):
    TICKS[0]=0; cold()
    with torch.inference_mode():
        if state=='primer':
            P=primer_ids()
            for j in range(0,len(P),C): tk(P[j:j+C])
        if state=='primer': ids=list(tok.encode_open("\n\n"+prompt)[0])[1:]
        else: ids=list(tok.encode_open(prompt)[0])
        pool=[x for x in ids if x<256] or ids
        hist=list(ids); out=[]; log={}
        def feed(block,start):  # inference SS: replace ss fraction of generated positions by prompt ids
            if ss<=0: return block
            b=list(block); g0=len(hist)-len(out)
            for j in range(len(b)):
                if start+j>=g0 and rng.random()<ss: b[j]=rng.choice(pool)
            return b
        if mode=='B':
            pend=list(ids); base=0
            while len(pend)>C: tk(pend[:C//2]); fixcarry(fix); pend=pend[C//2:]; base+=C//2
        for p in range(N):
            if p in (0,10,20,40,80) or p==N-1:
                n=norms(); n.pop('_S'); n.pop('_z'); rr=refat(max(TICKS[0],1)); log[p]={'ticks':TICKS[0],**n,'ref_S_fro':rr['S_fro'],'ref_z_l2':rr['z_l2'],'ref_thought':rr['thought_l2']}
            if mode=='B':
                s=get(); lg=tick(feed(pend,len(hist)-len(pend)))[-1]; put(s)
            else:
                ctx=hist[-C:]; lg=tk(feed(ctx,len(hist)-len(ctx)))[-1]; fixcarry(fix)
            n_=pick(lg,hist,T,k,rp); out.append(n_); hist.append(n_)
            if mode=='B':
                pend.append(n_)
                if len(pend)>=C: tk(pend[:C//2]); fixcarry(fix); pend=pend[C//2:]
        n=norms(); n.pop('_S'); n.pop('_z'); rr=refat(max(TICKS[0],1)); log[N]={'ticks':TICKS[0],**n,'ref_S_fro':rr['S_fro'],'ref_z_l2':rr['z_l2'],'ref_thought':rr['thought_l2']}
    return out,log
N=int(os.environ.get('N','80'))
for c in os.environ['CONFIGS'].split(','):
    mode,C,T,k,rp,state,fix,ss=c.split(':'); C=int(C);T=float(T);k=int(k);rp=float(rp);ss=float(ss)
    for p in PROMPTS:
        torch.manual_seed(0); rng=random.Random(0); t=time.time()
        out,log=run(p,mode,C,T,k,rp,state,fix,ss,N,rng); dt=time.time()-t; last=out[-40:]
        emit({'cfg':c,'prompt':p,'output':tok.decode(out),'ids':out,'ids_s':round(N/dt,2),'unique40':round(len(set(last))/len(last),3),'carry':log})
        print(c,p,round(len(set(last))/len(last),3),flush=True)
