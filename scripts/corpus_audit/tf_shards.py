"""TF loss / top-1 of the latest v3 checkpoint (fp32, fixed generation path) on docs of the largest
non-syllogism shards + held-out plain English. Inference only."""
import sys, os, json, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/repo')
from fractus.fixed_generation import load_checkpoint, FixedGenerator
torch.set_num_threads(os.cpu_count())
CK = os.environ.get('CKPT', '/workspace/ckpt_v3_85000/fractus_atom_1b_gpu0.pt')
D = '/workspace/corpus_audit/atom_corpus'
SHARDS = os.environ.get('SHARDS', 'understanding__inferential_comprehension,coding__algorithmic_thinking,'
                        'reasoning__counterargument_construction,speaking__executive_communication').split(',')
NDOC, LEN, W = 3, int(os.environ.get('LEN', 512)), 127
eng, meta = load_checkpoint(CK, torch.float32)
G = FixedGenerator(eng, warm=W)
def score(ctx, start):
    lg = G.tf_logits(ctx)[:-1].float(); tgt = torch.tensor(ctx[1:])
    nll = F.cross_entropy(lg, tgt, reduction='none')[start - 1:]; t1 = (lg.argmax(-1) == tgt).float()[start - 1:]
    return float(nll.mean()), float(t1.mean()), float(nll[:64].mean()), float(t1[:64].mean())
out = []
rng = np.random.default_rng(1)
for s in SHARDS:
    a = np.memmap(f'{D}/{s}.i16', dtype=np.int16, mode='r'); b = np.flatnonzero(a == 257)
    for r in sorted(rng.choice(np.arange(1, len(b) - 1), NDOC, replace=False)):
        doc = a[b[r]:min(b[r + 1], b[r] + LEN)].astype(int).tolist(); tail = a[b[r] - W:b[r]].astype(int).tolist()
        t = time.time(); nll, t1, n64, t64 = score(tail + doc, W + 1)
        out.append(dict(set=s, doc=int(r), n=len(doc), nll=nll, top1=t1, nll_first64=n64, top1_first64=t64, secs=time.time() - t)); print(json.dumps(out[-1]), flush=True)
TXT = {'plain_english': "The river had been rising for three days before anyone in the village thought to move the grain. By the time the mayor rang the bell, the water was already in the lower streets, carrying chairs, baskets and a surprised goat past the bakery. Old Marta, who had seen two floods before, simply climbed to her attic with a loaf of bread and a book, and waited for the sun.",
       'plain_english_2': "My grandfather kept bees behind the shed. Every August he would put on a white veil, light a small smoker made from an old coffee tin, and lift the frames one by one while we watched from the kitchen window. He never wore gloves. He said the bees knew him, and in thirty years I only saw him stung twice."}
for k, v in TXT.items():
    ctx = G.context(G.encode_prompt(v), W); nll, t1, n64, t64 = score(ctx, W + 1)
    out.append(dict(set=k, n=len(ctx) - W, nll=nll, top1=t1, nll_first64=n64, top1_first64=t64)); print(json.dumps(out[-1]), flush=True)
json.dump(dict(ckpt=CK, meta={k: str(v) for k, v in (meta or {}).items() if k in ('step', 'tokens_seen')}, rows=out), open('/workspace/corpus_audit/tf_shards.json', 'w'), indent=1)
print('done')
