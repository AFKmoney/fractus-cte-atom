# Exp2: parity (training forward vs every generation path, teacher-forced) + greedy from the training forward.
from ddlib import *
G=1833208861; LANE=275683869
a,bos=corpus(); prevfile=json.load(open('prev_file_tail256.json'))
def prev_ids(d,k):
    s=int(bos[d])
    if k==0: return []
    return a[s-k:s].tolist() if s>=k else (prevfile+a[:s].tolist())[-k:]
def tickg(ids):            # gen_block.tick semantics
    if 257 in ids: cold()
    return tick(ids)[0]
def prime(ids,C,keep):
    pend=list(ids)
    while len(pend)-keep>=1 and len(pend)>C:
        c=min(C,len(pend)-keep); tickg(pend[:c]); pend=pend[c:]
    return pend
def genpath_tf(primer,true,mode,C):
    """run gen_cont_syllog.gen() control flow, but append the TRUE id each step. returns (40,V) logits."""
    cold(); rows=[]
    if mode=='B':
        pend=prime(primer,C,C//2)
        for n in true:
            s=get(); rows.append(tickg(pend)[-1]); put(s)
            pend.append(n)
            if len(pend)>=C: tickg(pend[:C//2]); pend=pend[C//2:]
    else:
        prime(primer,C,0); ctx=list(primer)
        for n in true:
            rows.append(tickg(ctx[-C:])[-1]); ctx.append(n)
    return torch.stack(rows)
def train_greedy(pre,primer,N=40,C=128):
    """greedy with the training forward: fixed chunk grid (chunks of C from index 0 of pre+primer),
    zero carry at start, committed chunks keep their carry, the open chunk is re-forwarded every step."""
    seq=pre+primer; cold(); com=0; out=[]
    while len(seq)-com>C: tick(seq[com:com+C],train=True); com+=C
    for _ in range(N):
        s=get(); lg=tick(seq[com:],train=True)[0,-1]; put(s)
        lg=lg.clone(); lg[256]=-1e9; lg[257]=-1e9          # same masking as gen_block.pick
        n=int(lg.argmax()); out.append(n); seq.append(n)
        if len(seq)-com==C+0 and False: pass
        if len(seq)-com>C: tick(seq[com:com+C],train=True); com+=C
    return out
def ed(x,y):
    import difflib; return round(difflib.SequenceMatcher(a=x,b=y,autojunk=False).ratio(),3)
def gstats(out,true):
    em=sum(int(x==y) for x,y in zip(out,true)); pre=next((i for i,(x,y) in enumerate(zip(out,true)) if x!=y),len(out))
    return {'exact_pos':em,'exact_prefix':pre,'unique40':len(set(out)),'seqmatch_ratio':ed(out,true),'text':rep_ids(out)}
R={}
for d in [int(x) for x in os.environ.get('DOCS','0,5000,10000').split(',')]:
    s=int(bos[d]); doc=a[s:s+297].tolist(); primer,true=doc[:257],doc[257:]
    p=(G-6*LANE+s)%128; r={'true':rep_ids(true),'bos_pos':p}
    # reference: training forward (tick_chunk_train), cold, chunk grid from BOS
    ref=fwd_chunks(doc,128,train=True)[256:296]
    refs={'train_cold_C128':ref,'single_chunk_cold':fwd_single(doc)[256:296]}
    pre=prev_ids(d,p); refs['train_exact_align']=fwd_chunks(pre+doc,128,train=True)[p+256:p+296]
    tt=torch.tensor(true)
    for k,v in refs.items():
        nll,ok=stats(v,true); r[f'tf_{k}']={'nll':round(float(nll.mean()),3),'top1':round(float(ok.float().mean()),3),
            'err_pos':[i for i in range(40) if not ok[i]],'argmax_at_err':rep_ids([int(v[i].argmax()) for i in range(40) if not ok[i]])}
        log(d,k,r[f'tf_{k}'])
    # parity: gen paths teacher-forced
    for mode,C in [('B',512),('B',128),('A',128)]:
        g=genpath_tf(primer,true,mode,C)
        for k in ['single_chunk_cold','train_cold_C128']:
            v=refs[k]; diff=(g-v).abs().max(-1).values
            r[f'parity_{mode}{C}_vs_{k}']={'maxabs':round(float(diff.max()),3),'median_maxabs':round(float(diff.median()),3),
               'top1_agree':round(float((g.argmax(-1)==v.argmax(-1)).float().mean()),3)}
        nll,ok=stats(g,true); r[f'tf_genpath_{mode}{C}']={'nll':round(float(nll.mean()),3),'top1':round(float(ok.float().mean()),3)}
        log(d,mode,C,{k:v for k,v in r.items() if f'{mode}{C}' in k})
    # greedy with the training forward
    for name,pp in [('cold',[]),('train_exact_align',pre)]:
        t=time.time(); out=train_greedy(pp,primer); r[f'greedy_train_{name}']=gstats(out,true); r[f'greedy_train_{name}']['ids']=out
        log(d,'greedy',name,r[f'greedy_train_{name}'],round(time.time()-t))
    # sanity: full re-forward from scratch at step 10 equals the incremental logits
    R[d]=r; json.dump(R,open(f'exp2_{NUM}.json','w'),indent=1)
