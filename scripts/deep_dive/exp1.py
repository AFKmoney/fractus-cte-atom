# Exp1: cold-start (generation-style) vs exact training alignment (previous-doc tail in the BOS chunk).
from ddlib import *
G=1833208861; LANE=275683869
a,bos=corpus(); prevfile=json.load(open('prev_file_tail256.json'))
def prev_ids(d,k):
    s=int(bos[d])
    if k==0: return []
    if s>=k: return a[s-k:s].tolist()
    return (prevfile+a[:s].tolist())[-k:]
def regions(nll,ok):
    R={}
    for name,(i,j) in {'p0_32':(0,32),'p32_64':(32,64),'p64_128':(64,128),'p128_256':(128,256),'cont256_296':(256,296)}.items():
        R[name]=(round(float(nll[i:j].mean()),3),round(float(ok[i:j].float().mean()),3))
    return R
out={}
for d in [int(x) for x in os.environ.get('DOCS','0,5000,10000').split(',')]:
    s=int(bos[d]); doc=a[s:s+297].tolist()
    p=(G-6*LANE+s)%128
    res={}
    for k in sorted(set([0,1,8,32,64,p,127])):
        seq=prev_ids(d,k)+doc
        lg=fwd_chunks(seq,128,train=True)[k:]
        nll,ok=stats(lg[:-1],doc[1:])
        res[f'k{k}'+('_TRAIN_EXACT' if k==p else '')]=regions(nll,ok)
        log(d,k,'exact' if k==p else '',res[list(res)[-1]], 'pos0 nll',round(float(nll[0]),2))
    lg=fwd_single(doc); nll,ok=stats(lg[:-1],doc[1:]); res['cold_single_chunk']=regions(nll,ok); log(d,'single',res['cold_single_chunk'])
    out[d]={'bos_pos_in_train_chunk':p,'res':res}
json.dump(out,open(f'exp1_{NUM}.json','w'),indent=1)
