# Exp4: corruption sensitivity. Replace one TRUE input id by (a) the model's own top-2 (a plausible error) or (b) a random letter,
# keep everything else teacher-forced, measure top-1 on the following targets vs clean.
from ddlib import *
a,b=corpus(SYL); K=64; L=768; rng=np.random.default_rng(0)
W=[(1,4),(5,8),(9,16),(17,32),(33,64)]
res={'top2':{w:[] for w in W},'rand':{w:[] for w in W}}; nres={'top2':{w:[] for w in W},'rand':{w:[] for w in W}}
for d in [0,5000,10000,2500,7500]:
    s=int(b[d]); pre=a[s-K:s].tolist() if s>=K else json.load(open('prev_file_tail256.json'))[-K:]
    ids=a[s:s+L].tolist(); lg=fwd_chunks(pre+ids,128,train=True)[K:]; tg=torch.tensor(ids[1:])
    ok0=(lg[:-1].argmax(-1)==tg).float(); nll0=F.cross_entropy(lg[:-1],tg,reduction='none')
    for c in [300,400,500,600]:
        top2=int(lg[c-1].topk(2).indices[1]) if int(lg[c-1].argmax())==ids[c] else int(lg[c-1].argmax())
        for kind,new in [('top2',top2),('rand',int(rng.choice([x for x in range(97,123) if x!=ids[c]])))]:
            x=list(ids); x[c]=new
            lg2=fwd_chunks(pre+x,128,train=True)[K:]
            ok2=(lg2[:-1].argmax(-1)==tg).float(); nll2=F.cross_entropy(lg2[:-1],tg,reduction='none')
            for (i,j) in W:
                sl=slice(c+i-1,c+j)   # row c is the prediction made right after the corrupted input
                res[kind][(i,j)].append((float(ok0[sl].mean()),float(ok2[sl].mean())))
                nres[kind][(i,j)].append((float(nll0[sl].mean()),float(nll2[sl].mean())))
            log(d,c,kind,repr(chr(ids[c]) if ids[c]<256 else ids[c]),'->',repr(chr(new) if new<256 else new),
                ' '.join(f'{i}-{j}: {res[kind][(i,j)][-1][0]:.2f}->{res[kind][(i,j)][-1][1]:.2f}' for i,j in W))
out={}
for kind in res:
    out[kind]={f'{i}-{j}':{'top1_clean':round(np.mean([u for u,v in res[kind][(i,j)]]),3),'top1_corrupt':round(np.mean([v for u,v in res[kind][(i,j)]]),3),
       'nll_clean':round(np.mean([u for u,v in nres[kind][(i,j)]]),3),'nll_corrupt':round(np.mean([v for u,v in nres[kind][(i,j)]]),3)} for i,j in W}
log(json.dumps(out,indent=1)); json.dump(out,open('exp4.json','w'),indent=1)
