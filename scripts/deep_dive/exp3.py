# Exp3: top-1 by id class, entropy/confidence, template share, corruption sensitivity, out-of-corpus text.
from ddlib import *
SYLa,SYLb=corpus(SYL); NEGa,NEGb=corpus(NEG)
L=int(os.environ.get('LEN','768')); K=64
ALNUM=set(range(48,58))|set(range(65,91))|set(range(97,123))
def cls(x,y):
    if y>=259: return 'marker'
    if y==x: return 'doubled(y==x)'
    if y==32: return 'space'
    if y in ALNUM and (x in (32,10) or x>=256): return 'word_initial'
    if y in ALNUM and x in ALNUM: return 'in_word'
    if y==10: return 'newline'
    return 'other'
def doc_ids(arr,b,d,n): s=int(b[d]); return arr[s:s+n].tolist()
def warm_fwd(arr,b,d,n):
    s=int(b[d]); pre=arr[s-K:s].tolist() if s>=K else []; seq=pre+arr[s:s+n].tolist()
    return fwd_chunks(seq,128,train=True)[len(pre):], seq[len(pre):]
# n-gram template index from other syllogism docs
def hashes(x,n=16):
    x=np.asarray(x,dtype=np.uint64); h=np.zeros(len(x)-n+1,dtype=np.uint64)
    for i in range(n): h=h*np.uint64(1000003)+x[i:len(x)-n+1+i]
    return h
EVAL_SYL=list(range(0,11400,600))[:16]; EVAL_NEG=list(range(0,11000,1100))[:8]
pool=[d for d in range(0,11400,3) if d not in EVAL_SYL][:3000]
H=np.concatenate([hashes(SYLa[SYLb[d]:SYLb[d+1]]) for d in pool]); Hu,Hc=np.unique(H,return_counts=True); del H
log('template index: 16-grams from',len(pool),'other syllogism docs, unique',len(Hu))
rows=[]
def analyse(lg,ids,src,d):
    p=torch.log_softmax(lg[:-1],-1); tg=torch.tensor(ids[1:])
    nll=-p.gather(1,tg[:,None])[:,0]; ok=(lg[:-1].argmax(-1)==tg); ent=-(p.exp()*p).sum(-1); mp=p.exp().max(-1).values
    hh=hashes(ids); cnt=np.zeros(len(ids)-1,dtype=np.int64)
    if src=='syl' or src=='neg' or True:
        idx=np.searchsorted(Hu,hh); idx=np.clip(idx,0,len(Hu)-1); c=np.where(Hu[idx]==hh,Hc[idx],0)
        cnt[15-1:]=c  # n-gram ids[t-14..t+1] ends at target t+1 -> row t
    mx=lg[:-1].max(-1).values; lstd=lg[:-1].std(-1); inlog=lg[:-1].gather(1,torch.tensor(ids[:-1])[:,None])[:,0]
    for t in range(len(ids)-1):
        rows.append(dict(src=src,d=d,t=t,c=cls(ids[t],ids[t+1]),nll=float(nll[t]),ok=bool(ok[t]),ent=float(ent[t]),mp=float(mp[t]),tmpl=int(cnt[t]),
                         maxlogit=float(mx[t]),lstd=float(lstd[t]),inlogit=float(inlog[t])))
T0=time.time()
for d in EVAL_SYL:
    lg,ids=warm_fwd(SYLa,SYLb,d,L); analyse(lg,ids,'syl',d)
log('syl done',round(time.time()-T0))
for d in EVAL_NEG:
    lg,ids=warm_fwd(NEGa,NEGb,d,L); analyse(lg,ids,'neg',d)
log('neg done',round(time.time()-T0))
OOC={
 'plain_english': "The river had been rising for three days before anyone in the village thought to move the grain. By the time the mayor rang the bell, the water was already in the lower streets, carrying chairs, baskets and a surprised goat past the bakery. Old Marta, who had seen two floods before, simply climbed to her attic with a loaf of bread and a book, and waited for the sun.",
 'corpus_style_novel_topic': "Explain photosynthesis from a historical angle for a curious beginner. Provide a short timeline.\nHere is a treatment of **photosynthesis** from a historical angle, formatted as a short timeline, written for a curious beginner. In 1779 Jan Ingenhousz showed that plants release oxygen only in light. Later work by Calvin and Benson traced the path of carbon through the cycle that now carries their names.",
}
for name,txt in OOC.items():
    ids=tok.encode(txt)[:-2]   # drop closing <B6><EOS>
    pre=SYLa[SYLb[1]-K:SYLb[1]].tolist(); lg=fwd_chunks(pre+ids,128,train=True)[K:]
    analyse(lg,ids,name,0)
import pandas as pd
df=pd.DataFrame(rows); df.to_pickle('exp3_rows.pkl')
def summ(g): return pd.Series(dict(n=len(g),share=0,nll=g.nll.mean(),top1=g.ok.mean(),entropy=g.ent.mean(),maxprob=g.mp.mean(),nll_when_wrong=g.nll[~g.ok].mean()))
for src in df.src.unique():
    s=df[(df.src==src)&(df.t>=16)]
    t=s.groupby('c').apply(summ); t['share']=t.n/len(s); log('\n==',src,'all: nll %.3f top1 %.3f'%(s.nll.mean(),s.ok.mean())); log(t.round(3).to_string())
s=df[(df.src=='syl')&(df.t>=16)]
s=s.assign(tm=np.where(s.tmpl>=5,'tmpl>=5docs',np.where(s.tmpl>0,'tmpl1-4','novel16gram')))
t=s.groupby('tm').apply(summ); t['share']=t.n/len(s); log('\n== syl by 16-gram template frequency'); log(t.round(3).to_string())
s2=df[(df.src=='neg')&(df.t>=16)]; s2=s2.assign(tm=np.where(s2.tmpl>=5,'tmpl>=5docs',np.where(s2.tmpl>0,'tmpl1-4','novel16gram')))
t=s2.groupby('tm').apply(summ); t['share']=t.n/len(s2); log('\n== neg (index built from syl docs) by template'); log(t.round(3).to_string())
log('\n== logit scale: max logit mean %.1f p99 %.1f, logit std %.1f; logit of the current input id (rep term): mean %.1f'%(df.maxlogit.mean(),df.maxlogit.quantile(.99),df.lstd.mean(),df.inlogit.mean()))
log('nll quantiles when wrong:',df[~df.ok].nll.quantile([.5,.9,.99]).round(2).tolist())
