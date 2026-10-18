"""Block-mode generation for Fractus CTE-Atom with warm carried state.
Mirrors fast4gpu_atom.py: chunk forward via the same per-block core, carry
(thought_state, attn_S, attn_z, kuramoto_phases) passed between chunks, reset only at BOS.
Modes:
  A  'block tick'  : every step feed the last C ids as one chunk, keep the resulting carry.
  B  'regen block' : carry is committed at chunk boundaries; every step restores the
                     committed carry and re-runs the whole pending block (C/2..C ids),
                     takes the next id from the last position. When the pending block
                     reaches C ids, its first C/2 ids are committed (carry advanced).
  2026-10-18 fix (docs/DEEP_DIVE_GENERATION.md): mode A used to call tick(ctx[-C:]) on the committed
  carry every step, adding the same ids to attn_S/attn_z again on each step (teacher-forced through it:
  top-1 0.20-0.25 vs 0.85). Mode A is now a pure sliding window from zero carry (A_LEGACY=1 = old).
  For the training function (fixed grid + committed carry + warm start) use scripts/gen_fixed.py.
Usage: CKPT=... CONFIGS=... python3 scripts/gen_block.py > out.jsonl
"""
import sys,os,time,json,gc,resource,torch,torch.nn.functional as F
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from fractus.atom_tokenizer import VOCAB_SIZE,AtomFractusTokenizer
from fractus.continuous_engine import ContinuousThoughtEngine
torch.set_num_threads(int(os.environ.get('THREADS','6')))
CK=os.environ.get('CKPT','/workspace/ckpt_new/checkpoints/atom_latest/fractus_atom_1b_gpu0.bin')
def load_raw_bf16(d):   # dir from scripts/ckpt_to_bf16raw.py: file-backed copy-on-write memmap (evictable pages)
    import numpy as np
    ix=json.load(open(os.path.join(d,'index.json'))); mm=np.memmap(os.path.join(d,'weights.bf16'),dtype=np.uint8,mode='c'); sd={}
    for k,m in ix['tensors'].items():
        a=mm[m['off']:m['off']+m['nbytes']]
        sd[k]=(torch.from_numpy(a.view(np.int16)).view(torch.bfloat16) if m['dtype']=='bfloat16' else torch.from_numpy(a).view(getattr(torch,m['dtype']))).view(m['shape'])
    return {'config':ix['config'],'model_state':sd}
ck=load_raw_bf16(CK) if os.path.isdir(CK) else torch.load(CK,map_location='cpu',weights_only=False,mmap=True)
cfg=ck['config']; sd=ck['model_state']
keys=['d_model','n_heads','d_head','n_levels','n_oscillators','coupling_rank','n_experts','top_k','expert_d_ff','siren_rank','n_layers']
with torch.device('meta'): eng=ContinuousThoughtEngine(vocab_size=VOCAB_SIZE,**{k:cfg[k] for k in keys})
msd=eng.state_dict(); bad=[k for k in sd if sd[k].shape!=msd[k].shape]   # batch-sized stream-state keys
LANE=int(os.environ.get('LANE','0'))
A_LEGACY=os.environ.get('A_LEGACY','0')=='1'
saved={k:sd[k][LANE:LANE+1].float().clone() for k in bad}
for k in bad: sd[k]=torch.zeros(msd[k].shape)
DT=getattr(torch,os.environ.get('DTYPE','float32'))
if DT!=torch.float32:   # materialize one tensor at a time as bf16 (half the RSS, no mmap thrash)
    for k in list(sd): sd[k]=sd[k].to(DT) if sd[k].is_floating_point() else sd[k]
    saved={k:v.to(DT) for k,v in saved.items()}
eng.load_state_dict(sd,strict=True,assign=True); del sd,ck; gc.collect()
for b in eng.blocks: b.moe.temperature=float(cfg['gate_temp'])
eng.eval(); SEQ=int(cfg.get('seq',128)); NB=len(eng.blocks)
print(json.dumps({'ckpt':CK,'ids_processed':cfg.get('ids_processed'),'seq':SEQ,'n_state_keys':len(bad)}),flush=True)
tok=AtomFractusTokenizer()
def get():
    return (eng.thought_state.clone(),[(b.attn_S.clone(),b.attn_z.clone(),b.kuramoto_phases.clone()) for b in eng.blocks])
def put(s):
    eng.thought_state=s[0].clone()
    for b,(a,z,k) in zip(eng.blocks,s[1]): b.attn_S,b.attn_z,b.kuramoto_phases=a.clone(),z.clone(),k.clone()
def warm():
    put((saved['thought_state'],[(saved[f'blocks.{i}.attn_S'],saved[f'blocks.{i}.attn_z'],saved[f'blocks.{i}.kuramoto_phases']) for i in range(NB)]))
def cold():
    eng.reset_thought(1)
    if DT!=torch.float32:
        eng.thought_state=eng.thought_state.to(DT)
        for b in eng.blocks: b.attn_S,b.attn_z,b.kuramoto_phases=b.attn_S.to(DT),b.attn_z.to(DT),b.kuramoto_phases.to(DT)
def tick(ids):
    if 257 in ids: cold()          # training: reset at any chunk containing BOS
    return eng.tick_chunk(torch.tensor([ids]))[0]
def pick(lg,hist,T,k,rp):
    lg=lg.float().clone(); lg[256]=-1e9; lg[257]=-1e9
    if rp!=1.0:
        for i in set(hist[-64:]): lg[i]=lg[i]/rp if lg[i]>0 else lg[i]*rp
    if T<=0: return int(lg.argmax())
    v,i=torch.topk(lg/T,k); return int(i[torch.multinomial(torch.softmax(v,-1),1)])
def prime(ids,C,keep):
    """commit prompt in chunks of C, leave >=keep ids pending; returns pending + TF losses"""
    tf=[]; pend=list(ids)
    while len(pend)-keep>=1 and len(pend)>C:
        c=min(C,len(pend)-keep); ch=pend[:c]
        lg=tick(ch)
        if len(pend)>c: tf.append(round(float(F.cross_entropy(lg,torch.tensor(pend[1:c+1]))),3))
        pend=pend[c:]
    return pend,tf
PRIMER_FILE=os.environ.get('PRIMER','/workspace/lmsys/ref/atom_corpus/reasoning__syllogisms.i16')
def primer_ids(n=256):
    import numpy as np
    p=np.fromfile(PRIMER_FILE,dtype=np.int16,count=4096).astype(np.int64).tolist()
    i=p.index(257); return p[i:i+n]
def gen(prompt,mode,C,T,k,rp,N,state):
    warm() if state=='warm' else cold()
    if state=='primer':   # BOS-led in-distribution document head, committed in C-sized chunks
        P=primer_ids(); 
        with torch.inference_mode():
            for j in range(0,len(P),C): tick(P[j:j+C])
    if prompt=='<CONT>':   # continue the primer document itself (in-distribution check)
        P=primer_ids(256+int(os.environ.get('CONT_TAIL','64'))); ids=P[256:]
    else:
        ids,_=tok.encode_open(prompt if state=='cold' else "\n\n"+prompt)
        ids=list(ids) if state=='cold' else list(ids)[1:]   # cold keeps BOS (resets), warm drops it
    out=[]; hist=list(ids)
    with torch.inference_mode():
        if mode=='B':
            pend,tf=prime(ids,C,C//2)
            s=get(); lg=tick(pend); 
            if len(pend)>1: tf.append(round(float(F.cross_entropy(lg[:-1],torch.tensor(pend[1:]))),3))
            put(s)
            for _ in range(N):
                s=get(); lg=tick(pend)[-1]; put(s)
                n=pick(lg,hist,T,k,rp); out.append(n); hist.append(n); pend.append(n)
                if len(pend)>=C:
                    tick(pend[:C//2]); pend=pend[C//2:]
        else:  # A
            pend,tf=prime(ids,C,0); tf=tf  # prompt fully committed except tail
            ctx=list(hist)
            for _ in range(N):
                if A_LEGACY: lg=tick(ctx[-C:])[-1]   # old: re-adds the window to the committed carry every step (double counting)
                else:                                  # fixed: pure sliding window, zero carry, nothing committed
                    s=get(); cold(); lg=eng.tick_chunk(torch.tensor([ctx[-C:]]))[0][-1]; put(s)
                n=pick(lg,hist,T,k,rp); out.append(n); hist.append(n); ctx.append(n)
    return out,tf
PROMPTS=os.environ['PROMPTS'].split('|') if os.environ.get('PROMPTS') else ["Bonjour, je m'appelle","The capital of France is","Q: What is 2+2?\nA:","Il était une fois"]
if __name__=='__main__':
    N=int(os.environ.get('N','120'))
    for c in os.environ.get('CONFIGS','B:128:0:1:1.0:warm').split(','):
        mode,C,T,k,rp,state=c.split(':'); C=int(C);T=float(T);k=int(k);rp=float(rp)
        for p in PROMPTS:
            torch.manual_seed(0); t=time.time(); out,tf=gen(p,mode,C,T,k,rp,N,state); dt=time.time()-t
            last=out[-40:]
            print(json.dumps({'cfg':c,'prompt':p,'output':tok.decode(out),'ids':out,'ids_s':round(N/dt,2),
              'unique40':round(len(set(last))/len(last),3),'unique40_count':len(set(last)),'n_last':len(last),'secs':round(dt,1),'prompt_tf':tf,'rss_gb':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6,2)},ensure_ascii=False),flush=True)
