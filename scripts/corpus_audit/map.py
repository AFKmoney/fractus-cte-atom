import mmap,os,json
F=open('full/atom_corpus/atom_corpus.i16','rb'); M=mmap.mmap(F.fileno(),0,access=mmap.ACCESS_READ)
fs=sorted(x for x in os.listdir('atom_corpus') if x.endswith('.i16')); off=0; out=[]
for f in fs:
    S=open('atom_corpus/'+f,'rb').read(); n=len(S)
    pos=M.find(S[:4096], off) if M[off:off+4096]!=S[:4096] else off
    if pos<0: pos=M.find(S[:4096])
    if pos<0: out.append((f,n,None,None)); print(f,'NOT FOUND'); continue
    # length of match
    eq = M[pos:pos+n]==S
    if eq: m=n
    else:
        lo,hi=0,n
        while lo<hi:
            mid=(lo+hi+1)//2
            if M[pos:pos+mid]==S[:mid]: lo=mid
            else: hi=mid-1
        m=lo
    out.append((f,n,pos,m)); 
    if pos!=off or m!=n: print(f,n,'pos',pos,'expected',off,'match',m)
    off=pos+m
print('end',off,M.size()); json.dump(out,open('map.json','w'))
