"""Mechanism check, pure repeats: rows of 32 random letters repeated (every position after the first period is
copyable). embedding -> CopyHead(2x32, ngram=argv[2]) -> linear, gate init argv[1]. Prints step, loss, top-1, |gate|.
Usage: python scripts/copyhead_mech_repeat.py <gate_init> <ngram>   (docs/COPY_HEAD.md section 4.3)"""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fractus.nn.copy_head import CopyHead
torch.set_num_threads(2); torch.manual_seed(0)
L=128; D=64
def batch(rng,B=32):
    S=rng.integers(97,123,(B,32)); return torch.tensor(np.concatenate([S,S,S,S,S],1)[:, :L+1])  # period-32 repeats
emb=nn.Embedding(266,D); nn.init.normal_(emb.weight,std=0.1); head=CopyHead(D,2,32,128,ngram=int(sys.argv[2])); out=nn.Linear(D,266)
with torch.no_grad(): head.gate.fill_(float(sys.argv[1]))
opt=torch.optim.AdamW(list(emb.parameters())+list(head.parameters())+list(out.parameters()),lr=3e-3)
rng=np.random.default_rng(0)
for s in range(1,401):
    ids=batch(rng); head.reset_state(32)
    lg=out(head.attend(emb(ids[:,:L]))); l=F.cross_entropy(lg[:,32:].reshape(-1,266),ids[:,33:L+1].reshape(-1))
    opt.zero_grad(); l.backward(); opt.step()
    if s%100==0: print(s, float(l), float((lg[:,32:].argmax(-1)==ids[:,33:L+1]).float().mean()), float(head.gate.abs().mean()))
