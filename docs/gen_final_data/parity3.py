import sys,time,json,torch,numpy as np,torch.nn.functional as F
sys.path.insert(0,'/workspace/repo_v4')
from fractus.fixed_generation import load_checkpoint, FixedGenerator, SEQ
from fractus.generate_aligned import _reset
torch.set_num_threads(7)
eng,cfg=load_checkpoint('/workspace/ckpt_v4_final/checkpoints/atom_v4/step_final/fractus_atom_1b_gpu0.bin',torch.float32); G=FixedGenerator(eng)
N=64
for s,k in [('wiki',5),('fineweb',7),('tinystories',9)]:
    a=np.fromfile(f'/workspace/v4build/{s}_val.i16',dtype=np.int16).astype(int); b=np.flatnonzero(a==257)
    ctx=a[b[k]-64:b[k]+448].tolist(); tg=torch.tensor(ctx[N:])
    T=G.tf_logits(ctx)[N-1:-1]
    t=time.time(); rows=[G.stream_start(ctx[:N])]
    for nid in ctx[N:-1]: rows.append(G.stream_push(nid))
    S=torch.stack(rows); ts=(time.time()-t)/len(rows)
    with torch.inference_mode():
        _reset(eng); out=[eng.tick_chunk(torch.tensor([ctx[:N]]))[0,-1]]
        for nid in ctx[N:-1]: out.append(eng.tick_chunk(torch.tensor([[nid]]))[0,-1])
    O=torch.stack(out).float()
    r=dict(src=s,n=len(tg),nll_tf=F.cross_entropy(T,tg).item(),nll_stream=F.cross_entropy(S,tg).item(),nll_oldspace=F.cross_entropy(O,tg).item(),
      argmax_agree_stream=(S.argmax(-1)==T.argmax(-1)).float().mean().item(),argmax_agree_old=(O.argmax(-1)==T.argmax(-1)).float().mean().item(),
      median_maxabs_stream=(S-T).abs().max(-1).values.median().item(),sec_per_id_stream=ts)
    print(json.dumps(r),flush=True)
