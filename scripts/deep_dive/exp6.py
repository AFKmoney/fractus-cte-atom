# Exp6: numerics. Is the forward causal and stable? Same position, different total length / dispatch path / tiny perturbation.
from ddlib import *
a,b=corpus(SYL); s=int(b[0]); doc=a[s:s+297].tolist()
t=time.time(); full=fwd_single(doc); log(NUM,'fwd297 secs',round(time.time()-t,1))
short=fwd_single(doc[:258])
d=(full[:258]-short).abs().max(-1).values
log(NUM,'single chunk len297 vs len258, positions 0..257: maxabs %.4g median %.4g top1 agree %.3f'%(float(d.max()),float(d.median()),float((full[:258].argmax(-1)==short.argmax(-1)).float().mean())))
CAP['nodrop']=True; nd=fwd_single(doc); CAP['nodrop']=False
d=(full-nd).abs().max(-1).values
log(NUM,'capacity vs grouped (no-drop) dispatch: maxabs %.4g median %.4g top1 agree %.3f'%(float(d.max()),float(d.median()),float((full.argmax(-1)==nd.argmax(-1)).float().mean())))
# tiny perturbation of the embedding table (relative 1e-4) -> logit change
W=eng.observe.weight; W0=W.data.clone(); torch.manual_seed(0)
W.data.add_(torch.randn_like(W0)*W0.float().std().to(W0.dtype)*1e-4)
pt=fwd_single(doc); W.data.copy_(W0)
d=(full-pt).abs().max(-1).values
log(NUM,'embedding perturbed by 1e-4 rel: maxabs %.4g median %.4g top1 agree %.3f'%(float(d.max()),float(d.median()),float((full.argmax(-1)==pt.argmax(-1)).float().mean())))
nll,ok=stats(full[:-1],doc[1:]); log(NUM,'doc0 cont TF nll %.3f top1 %.3f | primer nll %.3f'%(float(nll[256:].mean()),float(ok[256:].float().mean()),float(nll[:256].mean())))
log(NUM,'logit scale: |logit| max %.1f, mean max-logit %.1f, h norm last block mean %.1f'%(float(full.abs().max()),float(full.max(-1).values.mean()),0))
# per-block growth of a perturbation: hook block outputs
outs={}
def mk(i):
    def hk(m,inp,out): outs.setdefault(i,[]).append(out[0].detach().float().clone())
    return hk
hs=[blk.register_forward_hook(lambda m,i_,o,k=i: outs.setdefault(k,[]).append(o[0].detach().float().clone())) for i,blk in enumerate(eng.blocks)]
# blocks are called via tick_chunk_core (method, not forward) -> patch instead
for h in hs: h.remove()
orig=type(eng.blocks[0]).tick_chunk_core
REC=[]
def patched(self,h):
    o=orig(self,h); REC.append(o[0].detach().float().clone()); return o
type(eng.blocks[0]).tick_chunk_core=patched
REC.clear(); fwd_single(doc); A=list(REC)
W.data.add_(torch.randn_like(W0)*W0.float().std().to(W0.dtype)*1e-4); REC.clear(); fwd_single(doc); Bp=list(REC); W.data.copy_(W0)
type(eng.blocks[0]).tick_chunk_core=orig
log(NUM,'per-block relative diff of the residual h after a 1e-4 embedding perturbation:',[round(float((x-y).norm()/x.norm()),4) for x,y in zip(A,Bp)])
log(NUM,'per-block residual norm (mean over positions):',[round(float(x[0].norm(dim=-1).mean()),1) for x in A])
