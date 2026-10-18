"""In-context copy probe (inference only): can the model copy the topic name it was just given?
In every corpus doc the topic X appears in the question line, again in 'Here is a treatment of **X**', and in
'**Definition.** X ...'. The 2nd/3rd mentions are pure in-context copies. TF NLL / top-1 at those spans vs the
first mention (not copyable) and vs all positions. fp32, fixed path, real 127-id tail of the previous doc."""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fractus.fixed_generation import load_checkpoint, FixedGenerator
torch.set_num_threads(int(os.environ.get('THREADS', os.cpu_count())))
CK = os.environ.get('CKPT', '/workspace/ckpt_v3_latest/checkpoints/atom_v3/fractus_atom_1b_gpu0.bin')
eng, _ = load_checkpoint(CK); G = FixedGenerator(eng, warm=127)
def spans(ids, needle_before, stop):
    """id-index spans of the text following each occurrence of needle_before up to stop (bytes only)."""
    pos = [i for i, t in enumerate(ids) if t < 256]; b = bytes(ids[i] for i in pos); out = []; s = 0
    while True:
        k = b.find(needle_before, s)
        if k < 0: break
        st = k + len(needle_before); en = b.find(stop, st)
        out.append((pos[st], pos[en - 1] + 1)); s = en
    return out
rows = []
rng = np.random.default_rng(0)
for shard in ('speaking__negotiation', 'reasoning__syllogisms'):
    a = np.fromfile(f'/workspace/lmsys/ref/atom_corpus/{shard}.i16', dtype=np.int16).astype(int); bos = np.flatnonzero(a == 257)
    for r in sorted(rng.choice(np.arange(1, len(bos) - 1), int(os.environ.get('NDOC', 8)), replace=False)):
        doc = a[bos[r]:bos[r] + 512].tolist(); tail = a[bos[r] - 127:bos[r]].tolist(); ctx = tail + doc
        lg = G.tf_logits(ctx)[:-1]; tgt = torch.tensor(ctx[1:]); nll = F.cross_entropy(lg, tgt, reduction='none'); t1 = (lg.argmax(-1) == tgt).float()
        def stat(sp):   # sp in doc coordinates; target index j predicts ctx[j+1] -> doc index j+1-127
            idx = [127 + i - 1 for i in range(*sp)]
            return float(nll[idx].mean()), float(t1[idx].mean()), float(nll[idx[0]]), float(t1[idx[0]])
        q = spans(doc, b'Here is a treatment of **', b'**')
        d = spans(doc, b'**Definition.** ', b' ')
        line = bytes(t for t in doc if t < 256).split(b'\n')[0]
        rec = dict(shard=shard, doc=int(r), all_nll=float(nll[127:].mean()), all_top1=float(t1[127:].mean()))
        if q: rec['treatment_copy'] = stat(q[0])
        if d: rec['definition_copy'] = stat(d[0])
        rows.append(rec); print(json.dumps(rec), flush=True)
json.dump(rows, open(os.environ.get('OUT', '/workspace/copy_probe.json'), 'w'), indent=1)
