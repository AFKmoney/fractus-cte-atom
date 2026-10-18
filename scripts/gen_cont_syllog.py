"""CONT-mode generation test: primer = BOS + first 256 ids of an in-corpus document,
generate 40 greedy ids, compare with the true next 40 ids. Uses the gen_block.py
loader/primitives (bf16 memmap, warm-carry chunk ticks, reset only at BOS).
Usage: CKPT=<bf16raw dir> DTYPE=bfloat16 DOCS=0,5000,10000 CFGS=B:512,B:128,A:128 OUT=... python3 scripts/gen_cont_syllog.py
"""
import os,sys,json,time,numpy as np,torch,torch.nn.functional as F
sys.path.insert(0,os.path.dirname(__file__))
import gen_block as G
CORPUS=os.environ.get('PRIMER','/workspace/lmsys/ref/atom_corpus/reasoning__syllogisms.i16')
OUT=os.environ.get('OUT','/workspace/gen_cont_syllog.jsonl')
NP,NC=256,40
a=np.fromfile(CORPUS,dtype=np.int16); bos=np.where(a==257)[0]
tok=G.tok
def rep_ids(ids):  # decode with boundary markers shown
    s=[]; buf=[]
    for i in ids:
        if 259<=i<=265: s.append(tok.decode(buf)); buf=[]; s.append(f'<B{i-258}>')
        elif i in (256,257,258): s.append(tok.decode(buf)); buf=[]; s.append({256:'<PAD?256>',257:'<BOS>',258:'<EOS>'}[i])
        else: buf.append(i)
    s.append(tok.decode(buf)); return ''.join(s)
def tf_loss(seq,C):
    """teacher-forced CE, cold at BOS, chunks of C from BOS (training-style). returns per-target nll list"""
    G.cold(); nll=[]
    with torch.inference_mode():
        for j in range(0,len(seq)-1,C):
            ch=seq[j:j+C]; tg=seq[j+1:j+C+1]
            lg=G.tick(ch)[:len(tg)].float()
            nll+=F.cross_entropy(lg,torch.tensor(tg),reduction='none').tolist()
            if j+C>=len(seq)-1: break
    return nll,lg
def gen(ids,mode,C,N=NC):
    G.cold(); out=[]; hist=list(ids)
    with torch.inference_mode():
        if mode=='B':
            pend,tf=G.prime(ids,C,C//2)
            for _ in range(N):
                s=G.get(); lg=G.tick(pend)[-1]; G.put(s)
                n=G.pick(lg,hist,0.0,1,1.0); out.append(n); hist.append(n); pend.append(n)
                if len(pend)>=C: G.tick(pend[:C//2]); pend=pend[C//2:]
        else:
            pend,tf=G.prime(ids,C,0); ctx=list(hist)
            for _ in range(N):   # 2026-10-18: pure sliding window (old code re-added the window to the carry each step)
                if G.A_LEGACY: lg=G.tick(ctx[-C:])[-1]
                else: s=G.get(); G.cold(); lg=G.eng.tick_chunk(torch.tensor([ctx[-C:]]))[0][-1]; G.put(s)
                n=G.pick(lg,hist,0.0,1,1.0); out.append(n); hist.append(n); ctx.append(n)
    return out,tf
T0=time.time()
for d in [int(x) for x in os.environ.get('DOCS','0').split(',')]:
    s=int(bos[d]); seq=a[s:s+1+NP+NC].astype(np.int64).tolist(); assert seq[0]==257 and 257 not in seq[1:]
    primer,true=seq[:1+NP],seq[1+NP:]
    t=time.time(); nll128,_=tf_loss(seq,128); nll512,lg512=tf_loss(seq,512)
    # top-1 TF accuracy on continuation (C=128 chunks)
    rec={'kind':'tf','doc_index':d,'corpus_offset':s,'primer_len':len(primer),
         'primer_tf_C128':round(float(np.mean(nll128[:NP])),4),'cont_tf_C128':round(float(np.mean(nll128[NP:])),4),
         'primer_tf_C512':round(float(np.mean(nll512[:NP])),4),'cont_tf_C512':round(float(np.mean(nll512[NP:])),4),
         'cont_nll_C128':[round(x,3) for x in nll128[NP:]],'primer_nll_C128':[round(x,3) for x in nll128[:NP]],
         'tf_C128_chunkstart_pos':[round(nll128[j],3) for j in range(0,len(nll128),128)],
         'primer_tf_C128_excl_first16':round(float(np.mean(nll128[16:NP])),4),
         'top1_acc_cont_C512':round(float((lg512[-NC:].argmax(-1)==torch.tensor(true)).float().mean()),3),
         'primer_tail':tok.decode(primer[1:])[-120:],'primer_tail_markers':rep_ids(primer[1:])[-160:],
         'true_cont':tok.decode(true),'true_cont_markers':rep_ids(true),'true_ids':true,'true_unique40':len(set(true)),'secs':round(time.time()-t,1)}
    print(json.dumps(rec,ensure_ascii=False),flush=True); open(OUT,'a').write(json.dumps(rec,ensure_ascii=False)+'\n')
    for c in [x for x in os.environ.get('CFGS','B:512,B:128,A:128').split(',') if x]:
        mode,C=c.split(':'); C=int(C); torch.manual_seed(0); t=time.time()
        out,tf=gen(primer,mode,C); dt=time.time()-t
        em=sum(int(x==y) for x,y in zip(out,true))
        pre=next((i for i,(x,y) in enumerate(zip(out,true)) if x!=y),len(out))
        r={'kind':'gen','doc_index':d,'cfg':c,'greedy':True,'unique40_count':len(set(out)),'unique40':round(len(set(out))/40,3),
           'exact_match_pos':em,'exact_prefix':pre,'generated':tok.decode(out),'generated_markers':rep_ids(out),'ids':out,
           'harness_prompt_tf':tf,'secs':round(dt,1),'ids_s':round(NC/dt,2)}
        print(json.dumps(r,ensure_ascii=False),flush=True); open(OUT,'a').write(json.dumps(r,ensure_ascii=False)+'\n')
print(json.dumps({'total_secs':round(time.time()-T0,1)}),flush=True)
