"""Deep-dive harness: one loader, several forward paths, instrumentation. Box CPU, bf16 memmap."""
import os, sys, json, gc, time
import numpy as np, torch, torch.nn.functional as F
REPO = '/workspace/repo'
sys.path.insert(0, REPO)
from fractus.atom_tokenizer import VOCAB_SIZE, AtomFractusTokenizer
from fractus.continuous_engine import ContinuousThoughtEngine
import fractus.nn.attention as ATT
import fractus.nn.moe as MOE
torch.set_num_threads(int(os.environ.get('THREADS', '7')))
CK = os.environ.get('CKPT', '/workspace/ckpt_v2_test/v3_step10000/bf16raw')
NUM = os.environ.get('NUM', 'bf16')   # bf16 = exactly what gen_block/gen_cont do (DTYPE=bfloat16); amp = fp32 small params + autocast bf16 (training-like)
tok = AtomFractusTokenizer()

def _load():
    ix = json.load(open(os.path.join(CK, 'index.json')))
    mm = np.memmap(os.path.join(CK, 'weights.bf16'), dtype=np.uint8, mode='c'); sd = {}
    for k, m in ix['tensors'].items():
        a = mm[m['off']:m['off'] + m['nbytes']]
        t = (torch.from_numpy(a.view(np.int16)).view(torch.bfloat16) if m['dtype'] == 'bfloat16'
             else torch.from_numpy(a).view(getattr(torch, m['dtype']))).view(m['shape'])
        sd[k] = t
    return ix['config'], sd

cfg, sd = _load()
KEYS = ['d_model','n_heads','d_head','n_levels','n_oscillators','coupling_rank','n_experts','top_k','expert_d_ff','siren_rank','n_layers']
with torch.device('meta'):
    eng = ContinuousThoughtEngine(vocab_size=VOCAB_SIZE, **{k: cfg[k] for k in KEYS})
msd = eng.state_dict()
bad = [k for k in sd if sd[k].shape != msd[k].shape]
SAVED = {k: sd[k][0:1].float().clone() for k in bad}   # lane-0 training carry (not used unless asked)
for k in bad: sd[k] = torch.zeros(msd[k].shape)
if NUM == 'fp32':   # everything the forward uses in fp32 (the unused siren_* grow copies stay bf16 memmap)
    for k in list(sd):
        if sd[k].is_floating_point(): sd[k] = sd[k].to(torch.bfloat16) if '.siren_' in k else sd[k].float()
elif NUM == 'amp':
    for k in list(sd):
        if sd[k].is_floating_point() and sd[k].numel() < 2_000_000: sd[k] = sd[k].float()
else:
    for k in list(sd):
        if sd[k].is_floating_point(): sd[k] = sd[k].to(torch.bfloat16)
eng.load_state_dict(sd, strict=True, assign=True); del sd; gc.collect()
for b in eng.blocks: b.moe.temperature = float(cfg['gate_temp'])
eng.eval(); NB = len(eng.blocks)
SDT = torch.float32 if NUM in ('amp','fp32') else torch.bfloat16

def ctx():
    if NUM == 'amp':
        return torch.autocast('cpu', dtype=torch.bfloat16)
    import contextlib; return contextlib.nullcontext()

def cold(B=1):
    eng.reset_thought(B)
    eng.thought_state = eng.thought_state.to(SDT)
    for b in eng.blocks:
        b.attn_S, b.attn_z, b.kuramoto_phases = b.attn_S.to(SDT), b.attn_z.to(SDT), b.kuramoto_phases.to(SDT)

def get():
    return (eng.thought_state.clone(), [(b.attn_S.clone(), b.attn_z.clone(), b.kuramoto_phases.clone()) for b in eng.blocks])

def put(s):
    eng.thought_state = s[0].clone()
    for b, (a, z, k) in zip(eng.blocks, s[1]): b.attn_S, b.attn_z, b.kuramoto_phases = a.clone(), z.clone(), k.clone()

def tick(ids, train=False):
    """one chunk, B=1 or B rows; ids: list or 2-D list. returns float logits (B,C,V)"""
    x = torch.tensor(ids if isinstance(ids[0], list) else [ids])
    with torch.inference_mode(), ctx():
        lg = eng.tick_chunk_train(x)[0] if train else eng.tick_chunk(x)
    return lg.float()

def fwd_single(ids):
    """whole sequence as ONE chunk from zero carry (== gen_block B:512 re-forward)."""
    cold(); return tick(ids)[0]

def fwd_chunks(ids, C=128, start_pad=0, train=False):
    """training-like: zero carry, chunks of C starting at index 0 (start_pad ids of the
    previous document precede BOS in the first chunk, as in a lane). Returns logits for all ids."""
    cold(); out = []
    for j in range(0, len(ids), C):
        out.append(tick(ids[j:j + C], train=train)[0])
    return torch.cat(out, 0)

def set_impl(name): ATT.set_attention_impl(name)

# ---- MoE capacity instrumentation
CAP = {'on': False, 'rec': []}
_orig_cap = MOE.PhaseRoutedMoE._capacity_expert_forward
def _cap_hook(self, h, topk_idx, cf=4.0):
    if CAP['on']:
        B, L, D = h.shape; K = topk_idx.shape[-1]; N = B * L; E = self.n_experts
        e = topk_idx.reshape(-1); A = e.shape[0]
        oh = F.one_hot(e, E).float(); pos = (oh.cumsum(0) * oh).sum(1) - 1
        C = max(8, int((cf * A / E + 7) // 8 * 8))
        keep = (pos < C).float().view(B, L, K)
        CAP['rec'].append({'B': B, 'L': L, 'C': C, 'keep': keep.mean(-1).cpu(),
                           'n_exp_used': int((oh.sum(0) > 0).sum())})
    if CAP.get('nodrop'):
        return self._grouped_expert_forward(h, topk_idx)   # exact top-k, no capacity drop
    return _orig_cap(self, h, topk_idx, cf)
MOE.PhaseRoutedMoE._capacity_expert_forward = _cap_hook

def rep_ids(ids):
    s = []; buf = []
    for i in ids:
        i = int(i)
        if i >= 256:
            s.append(bytes(buf).decode('utf-8', 'replace')); buf = []
            s.append({256: '<PAD>', 257: '<BOS>', 258: '<EOS>'}.get(i, f'<B{i-258}>'))
        else: buf.append(i)
    s.append(bytes(buf).decode('utf-8', 'replace')); return ''.join(s)

SYL = '/workspace/lmsys/ref/atom_corpus/reasoning__syllogisms.i16'
NEG = '/workspace/lmsys/ref/atom_corpus/speaking__negotiation.i16'
_corp = {}
def corpus(p=SYL):
    if p not in _corp:
        a = np.fromfile(p, dtype=np.int16).astype(np.int64); _corp[p] = (a, np.where(a == 257)[0])
    return _corp[p]

def stats(lg, tgt):
    tgt = torch.tensor(tgt)
    nll = F.cross_entropy(lg, tgt, reduction='none')
    return nll, (lg.argmax(-1) == tgt)

def log(*a):
    print(*a, flush=True)
