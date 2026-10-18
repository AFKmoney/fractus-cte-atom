import numpy as np, os, json, sys, collections
D='atom_corpus'; fs=sorted(x for x in os.listdir(D) if x.endswith('.i16'))
NS=int(os.environ.get('NS',2000)); K=16; P=np.uint64(1000003)
rng=np.random.default_rng(0)
def dec(ids):
    return bytes(int(i) for i in ids if 0<=i<256).decode('utf-8','replace')
def grams(arr, starts, ends):
    # concatenate docs, hash 16-grams not crossing docs
    segs=[arr[s:e].astype(np.uint64) for s,e in zip(starts,ends)]
    lens=np.array([len(x) for x in segs]); A=np.concatenate(segs); did=np.repeat(np.arange(len(segs)),lens)
    n=len(A)-K+1; h=np.zeros(n,np.uint64)
    with np.errstate(over='ignore'):
        for k in range(K): h=h*P+A[k:k+n]
    ok=did[:n]==did[K-1:K-1+n]
    return h[ok], did[:n][ok]
tot=sum(os.path.getsize(f'{D}/{f}')//2 for f in fs)
res=[]; xs_h=[]; xs_s=[]
for si,f in enumerate(fs):
    a=np.memmap(f'{D}/{f}',dtype=np.int16,mode='r')
    b=np.flatnonzero(a==257); e=np.append(b[1:],a.size)
    nd=len(b)
    # prefix dup on all docs: first 64 ids after BOS
    L=64; okp=(e-b)>L
    pref=np.stack([np.asarray(a[s+1:s+1+L]) for s in b[okp]]).astype(np.int64)
    ph=np.zeros(len(pref),np.uint64)
    with np.errstate(over='ignore'):
        for k in range(L): ph=ph*P+pref[:,k].astype(np.uint64)
    u,inv,cnt=np.unique(ph,return_inverse=True,return_counts=True)
    pdup=float((cnt[inv]>1).mean()); nuniq_pref=len(u)
    # 32-id prefix too
    ph32=np.zeros(len(pref),np.uint64)
    with np.errstate(over='ignore'):
        for k in range(32): ph32=ph32*P+pref[:,k].astype(np.uint64)
    u32,inv32,c32=np.unique(ph32,return_inverse=True,return_counts=True); pdup32=float((c32[inv32]>1).mean())
    # 16-gram doc frequency in a sample
    sel=np.sort(rng.choice(nd,min(NS,nd),replace=False))
    h,did=grams(a,b[sel],e[sel])
    pair=np.unique(np.stack([h,did.astype(np.uint64)],1),axis=0)
    uh,df=np.unique(pair[:,0],return_counts=True)
    gdf=df[np.searchsorted(uh,h)]
    t5=float((gdf>=6).mean()); t1=float((gdf>=2).mean()); uniqr=len(uh)/len(h)
    perdoc=np.bincount(did,weights=(gdf>=6),minlength=len(sel))/np.maximum(1,np.bincount(did,minlength=len(sel)))
    # cross-shard sample: 100 docs
    s2=np.sort(rng.choice(nd,min(100,nd),replace=False)); h2,_=grams(a,b[s2],e[s2]); h2=np.unique(h2)
    xs_h.append(h2); xs_s.append(np.full(len(h2),si,np.int32))
    r=rng.integers(nd); mid=b[r]+(e[r]-b[r])//2
    res.append(dict(shard=f[:-4],ids=int(a.size),share=a.size/tot,docs=int(nd),mean_doc_ids=float(a.size/nd),
        tmpl16_ge5other=t5,tmpl16_ge1other=t1,uniq16_ratio=uniqr,doc_tmpl_p10=float(np.percentile(perdoc,10)),doc_tmpl_p90=float(np.percentile(perdoc,90)),
        prefix64_dup=pdup,prefix64_unique=int(nuniq_pref),prefix32_dup=pdup32,
        snip_head=dec(a[b[r]+1:b[r]+260]),snip_mid=dec(a[mid:mid+200])))
    print(si,f,round(t5,3),round(pdup,3),nd,flush=True)
# cross-shard: share of each shard's 16-gram types that occur in >=2 other shards
H=np.concatenate(xs_h); S=np.concatenate(xs_s)
uh,inv,c=np.unique(H,return_inverse=True,return_counts=True)
for si,r in enumerate(res):
    m=S==si; r['xshard16_ge2other']=float((c[inv[m]]>=3).mean())
json.dump(res,open('audit.json','w'),indent=1)
print('done')
