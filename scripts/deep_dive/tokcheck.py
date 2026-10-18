# Tokenization checks: corpus == tok.encode(decoded text)? marker look-ahead distribution; encode_open of prompts.
import sys,json,numpy as np
sys.path.insert(0,'/workspace/repo')
from fractus.atom_tokenizer import AtomFractusTokenizer
from fractus.atomizer import Atomizer
tok=AtomFractusTokenizer()
a=np.fromfile('/workspace/lmsys/ref/atom_corpus/reasoning__syllogisms.i16',dtype=np.int16).astype(np.int64)
bos=np.where(a==257)[0]; eos=np.where(a==258)[0]
match=0; n=0
for d in range(0,11400,570):
    ids=a[bos[d]:eos[d]+1].tolist()
    txt=tok.decode(ids)
    re=tok.encode(txt)
    n+=1; match+= (re==ids)
print('re-encode exact match docs',match,'/',n)
# look-ahead of each marker: bytes consumed after the marker position when the atomizer decided it
at=Atomizer(max_span_bytes=32,pack_mode='linguistic')
la=[]; kinds={}
orig_fp=at._flush_prefix; orig_pend=at._flush_pending
def fp(cut,boundary):
    la.append(len(at._buffer)-cut); kinds.setdefault(('prefix',boundary),[]).append(len(at._buffer)-cut); return orig_fp(cut,boundary)
def fpe(boundary='eos'):
    if at._buffer: la.append(0); kinds.setdefault(('pending',boundary),[]).append(0)
    return orig_pend(boundary)
at._flush_prefix=fp; at._flush_pending=fpe
for d in range(0,11400,570):
    txt=tok.decode(a[bos[d]:eos[d]+1].tolist()); at.encode(txt)
la=np.array(la)
print('markers',len(la),'with lookahead>0: %.3f'%(la>0).mean(),'mean lookahead|>0 %.2f'%la[la>0].mean(),'max',la.max())
print({str(k):(len(v),round(float(np.mean(v)),2)) for k,v in kinds.items()})
# packet length stats in corpus (bytes between markers)
seg=[];c=0
for i in a[:2_000_000]:
    if i>=256: seg.append(c); c=0
    else: c+=1
seg=np.array(seg); print('bytes between markers: mean %.1f median %d max %d frac>=24 %.3f'%(seg.mean(),np.median(seg),seg.max(),(seg>=24).mean()))
u,cn=np.unique(a,return_counts=True); fr=dict(zip(u.tolist(),cn.tolist())); N=len(a)
print('id share: space %.3f markers %.3f'%(fr.get(32,0)/N, sum(v for k,v in fr.items() if k>=259)/N))
for p in ['The capital of France is','Once upon a time','Hello']:
    ids,_=tok.encode_open(p); print(repr(p),'encode_open',ids, 'encode(closed)',tok.encode(p))
