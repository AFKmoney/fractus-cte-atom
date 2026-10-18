import numpy as np, os, json
P=np.uint64(1000003); fs=sorted(x for x in os.listdir('atom_corpus') if x.endswith('.i16')); out={}; allu=[]
for f in fs:
    a=np.memmap(f'atom_corpus/{f}',dtype=np.int16,mode='r'); us=[]
    for st in range(0,a.size-15,8_000_000):
        x=np.asarray(a[st:min(a.size,st+8_000_015)]).astype(np.uint64); n=len(x)-15; h=np.zeros(n,np.uint64)
        with np.errstate(over='ignore'):
            for k in range(16): h=h*P+x[k:k+n]
        us.append(np.unique(h))
    u=np.unique(np.concatenate(us)); out[f[:-4]]=int(len(u)); allu.append(u); print(f,a.size,len(u),flush=True)
g=np.unique(np.concatenate(allu)); out['_corpus_distinct16']=int(len(g)); out['_sum_shard_distinct16']=int(sum(len(u) for u in allu))
json.dump(out,open('distinct16.json','w'),indent=1); print('corpus distinct',len(g))
