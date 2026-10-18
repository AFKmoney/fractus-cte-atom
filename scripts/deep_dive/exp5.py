# Exp5: MoE capacity drops. Training: one (16,128) chunk -> A=4096 assignments, C=128 slots/expert, arrival order row-major.
# Generation: (1,L) chunk -> C=max(8, 4*2L/128 rounded up to 8). Measure the kept fraction, and the effect of removing the cap.
from ddlib import *
a,b=corpus(SYL); K=64
def keepstats(fn):
    CAP['on']=True; CAP['rec']=[]; out=fn(); CAP['on']=False
    rec=CAP['rec']; return out,rec
d=0; s=int(b[1000]); ids=a[s:s+640].tolist(); pre=a[s-K:s].tolist()
# (1) B=1 chunks of 128 (gen/eval harness)
_,rec=keepstats(lambda: fwd_chunks(pre+ids,128,train=True))
kb=[r['keep'][0] for r in rec]   # per block per chunk (L,)
k1=torch.stack([r['keep'].mean() for r in rec]).view(-1,NB)
log('B=1 C=128 chunks: Ccap',rec[0]['C'],'kept frac per chunk (mean over blocks):',k1.mean(1).round(decimals=3).tolist())
log('  per block (mean over chunks):',k1.mean(0).round(decimals=2).tolist())
last=torch.stack([r['keep'][0,-1] for r in rec]); log('  kept frac at the LAST position of a chunk (the one generation reads):',round(float(last.mean()),3))
pos=torch.stack([r['keep'][0] for r in rec if r['L']==128]).mean(0); log('  kept by position quartile:',[round(float(pos[i:i+32].mean()),3) for i in range(0,128,32)])
log('  experts used per block-chunk (of 128):',np.mean([r['n_exp_used'] for r in rec]))
# (2) training-like (16,128): 16 different docs, same chunk index; our doc in row 0 and in row 15
docs=[int(b[x]) for x in range(2000,2016)]
batch=[a[o+256:o+384].tolist() for o in docs]
cold(16); CAP['on']=True; CAP['rec']=[]; lgB=tick(batch,train=True); CAP['on']=False
kk=torch.stack([r['keep'] for r in CAP['rec']])  # (NB,16,128)
log('B=16 training chunk: Ccap',CAP['rec'][0]['C'],'kept frac overall %.3f'%float(kk.mean()),'by row:',kk.mean((0,2)).round(decimals=2).tolist())
log('  experts used per block:',np.mean([r['n_exp_used'] for r in CAP['rec']]))
# same 16 rows each as B=1 (gen regime)
k1s=[]
for r_ in batch:
    cold(1); CAP['on']=True; CAP['rec']=[]; tick(r_,train=True); CAP['on']=False; k1s.append(torch.stack([r['keep'] for r in CAP['rec']]).mean())
log('  same rows forwarded one by one (B=1): kept %.3f'%float(torch.stack(k1s).mean()))
# (3) effect on TF of removing the cap (exact top-2, no drop), B=1 chunks
tg=torch.tensor(ids[1:])
lgc=fwd_chunks(pre+ids,128,train=True)[K:]; CAP['nodrop']=True; lgn=fwd_chunks(pre+ids,128,train=True)[K:]; CAP['nodrop']=False
for nm,l in [('capacity(default)',lgc),('nodrop',lgn)]:
    nll,ok=stats(l[:-1],ids[1:]); log(nm,'nll %.3f top1 %.3f'%(float(nll[16:].mean()),float(ok[16:].float().mean())))
log('maxabs logit diff cap vs nodrop: %.1f'%float((lgc-lgn).abs().max()), 'top1 agree %.3f'%float((lgc.argmax(-1)==lgn.argmax(-1)).float().mean()))
# (4) how row placement inside a training batch changes the logits of one doc chunk
r0=[batch[0]]+batch[1:]; r15=batch[1:]+[batch[0]]
cold(16); l0=tick(r0,train=True)[0]; cold(16); l15=tick(r15,train=True)[15]; cold(1); l1=tick([batch[0]],train=True)[0]
tg2=torch.tensor(batch[0][1:])
for nm,l in [('row0_of_16',l0),('row15_of_16',l15),('alone_B1',l1)]:
    nll,ok=stats(l[:-1],batch[0][1:]); log(nm,'(cold chunk, mid-doc ids 256..384) nll %.3f top1 %.3f'%(float(nll.mean()),float(ok.float().mean())))
log('logit maxabs row0 vs row15 %.1f, row0 vs B1 %.1f'%(float((l0-l15).abs().max()),float((l0-l1).abs().max())))
