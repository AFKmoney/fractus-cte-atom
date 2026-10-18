"""Decoding strategies on the fixed generation path (fractus/fixed_generation.py). Inference only; trainer untouched.

Hypotheses carry their own committed state (fixed 128 grid, committed carry, open chunk re-forwarded), so beams,
look-ahead rollouts and contrastive candidates are batched rows with stacked carries.

Decoders: greedy | beam (width W, length-normalized log-prob) | word look-ahead (top-K first ids at each word start,
greedy rollout of D ids, pick by summed log-prob, commit the word) | min-p sampling on RAW logits (T, p_min, no-repeat
n-gram) | typical sampling | contrastive search (k, alpha).

Usage (eval):   CKPT=... python scripts/gen_best.py            -> OUT jsonl
Library:        from scripts.gen_best import Decoder
"""
import os, sys, json, time, difflib, re
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fractus.fixed_generation import load_checkpoint, FixedGenerator, WARM_TAIL, SEQ, show

PAD, BOS, EOS = 256, 257, 258
BOUND = set(range(259, 266)) | {10}


class Hyp:
    __slots__ = ('seq', 'com', 'snap', 'score', 'gen', 'hid')
    def __init__(s, seq, com, snap, score=0.0, gen=None, hid=None):
        s.seq, s.com, s.snap, s.score, s.gen, s.hid = seq, com, snap, score, gen or [], hid


class Decoder:
    def __init__(self, G: FixedGenerator):
        self.G, self.e = G, G.eng

    # ---- batched state helpers
    @staticmethod
    def stack(snaps):
        return (torch.cat([s[0] for s in snaps]),
                [tuple(torch.cat([s[1][j][k] for s in snaps]) for k in range(3)) for j in range(len(snaps[0][1]))])

    @staticmethod
    def row(state, i):
        return (state[0][i:i + 1].clone(), [tuple(t[i:i + 1].clone() for t in b) for b in state[1]])

    @torch.inference_mode()
    def _forward(self, rows, snaps):
        """rows: equal-length id lists; returns (logits B,L,V fp32, hidden B,L,D, new batched state)."""
        G, e = self.G, self.e
        G.put(self.stack(snaps))
        x = torch.tensor(rows)
        h = e.observe(x).clone()
        h[:, 0, :] = h[:, 0, :] + e.thought_state[:, 0, :]
        for blk in e.blocks:
            h, _ = blk.tick_chunk_core(h)
        e.thought_state = h[:, -1:, :].detach()
        return G.head(h.reshape(-1, h.shape[-1])).reshape(h.shape[0], h.shape[1], -1), h, G.get()

    def start(self, ctx):
        G = self.G; G.cold(); com = 0
        while len(ctx) - com > SEQ:
            G.chunk(ctx[com:com + SEQ]); com += SEQ
        return Hyp(list(ctx), com, G.get())

    def last(self, hyps):
        """next-id logits (B,V) and last hidden (B,D) for hyps of equal length."""
        lg, h, _ = self._forward([x.seq[x.com:] for x in hyps], [x.snap for x in hyps])
        out = lg[:, -1].clone(); out[:, PAD] = -1e9; out[:, BOS] = -1e9
        return out, h[:, -1], h

    def extend(self, hyps):
        """commit full chunks for hyps whose open part exceeds SEQ (batched)."""
        need = [x for x in hyps if len(x.seq) - x.com > SEQ]
        if need:
            _, _, st = self._forward([x.seq[x.com:x.com + SEQ] for x in need], [x.snap for x in need])
            for i, x in enumerate(need):
                x.snap = self.row(st, i); x.com += SEQ
        return hyps

    def child(self, x, nid, add):
        return Hyp(x.seq + [nid], x.com, x.snap, x.score + add, x.gen + [nid], x.hid)

    # ---- decoders
    def greedy(self, ctx, n=40, **_):
        x = self.start(ctx)
        for _ in range(n):
            lg, _, _ = self.last([x]); nid = int(lg[0].argmax())
            x = self.extend([self.child(x, nid, 0.0)])[0]
            if nid == EOS: break
        return x.gen

    def beam(self, ctx, n=40, W=4, alpha=1.0, **_):
        live = [self.start(ctx)]; done = []
        for _ in range(n):
            lg, _, _ = self.last(live); lp = F.log_softmax(lg, -1)
            cand = []
            for i, x in enumerate(live):
                v, ids = lp[i].topk(W)
                cand += [(x.score + float(a), i, int(b)) for a, b in zip(v, ids)]
            cand.sort(key=lambda c: -c[0]); nxt = []
            for sc, i, nid in cand:
                h = Hyp(live[i].seq + [nid], live[i].com, live[i].snap, sc, live[i].gen + [nid])
                (done if nid == EOS else nxt).append(h)
                if len(nxt) == W: break
            live = self.extend(nxt)
            if not live: break
        pool = done + live
        best = max(pool, key=lambda h: h.score / max(1, len(h.gen)) ** alpha)
        return best.gen

    def lookahead(self, ctx, n=40, K=4, D=8, **_):
        x = self.start(ctx)
        while len(x.gen) < n:
            word_start = (not x.gen and (x.seq[-1] in BOUND or x.seq[-1] == BOS or x.seq[-1] == 32)) or (x.gen and x.gen[-1] in BOUND)
            lg, _, _ = self.last([x]); lp = F.log_softmax(lg[0], -1)
            if not word_start:
                nid = int(lp.argmax()); x = self.extend([self.child(x, nid, float(lp[nid]))])[0]
                if nid == EOS: break
                continue
            v, ids = lp.topk(K)
            hs = self.extend([self.child(x, int(b), float(a)) for a, b in zip(v, ids)])
            for _ in range(D - 1):
                lg2, _, _ = self.last(hs); lp2 = F.log_softmax(lg2, -1); m = lp2.argmax(-1)
                hs = self.extend([self.child(h, int(m[i]), float(lp2[i, m[i]])) for i, h in enumerate(hs)])
            best = max(hs, key=lambda h: h.score)
            roll = best.gen[len(x.gen):]
            cut = next((j + 1 for j, t in enumerate(roll) if j > 0 and (t in BOUND or t == EOS)), len(roll))
            for t in roll[:cut]:
                x = self.extend([self.child(x, t, 0.0)])[0]
                if t == EOS or len(x.gen) >= n: break
            if x.gen and x.gen[-1] == EOS: break
        return x.gen[:n]

    @staticmethod
    def _ban_ngrams(gen, nsz):
        if nsz <= 0 or len(gen) < nsz - 1: return []
        pre = tuple(gen[-(nsz - 1):]); return [gen[i + nsz - 1] for i in range(len(gen) - nsz + 1) if tuple(gen[i:i + nsz - 1]) == pre]

    def sample(self, ctx, n=40, T=1.0, min_p=0.05, typical=0.0, ngram=4, seed=0, **_):
        g = torch.Generator().manual_seed(seed); x = self.start(ctx)
        for _ in range(n):
            lg, _, _ = self.last([x]); z = lg[0] / T
            for b in self._ban_ngrams(x.gen, ngram): z[b] = -1e9
            p = torch.softmax(z, -1)
            if typical > 0:
                lp = torch.log(p.clamp_min(1e-30)); H = -(p * lp).sum(); dev = (-lp - H).abs()
                o = dev.argsort(); keep = torch.zeros_like(p, dtype=torch.bool)
                cm = p[o].cumsum(0); k = int((cm < typical).sum()) + 1; keep[o[:k]] = True; p = p * keep
            if min_p > 0:
                p = p * (p >= min_p * p.max())
            nid = int(torch.multinomial(p / p.sum(), 1, generator=g))
            x = self.extend([self.child(x, nid, 0.0)])[0]
            if nid == EOS: break
        return x.gen

    def contrastive(self, ctx, n=40, k=4, alpha=0.6, **_):
        x = self.start(ctx)
        lg, hl, hall = self.last([x]); H = F.normalize(hall[0].float(), dim=-1)
        for _ in range(n):
            p = torch.softmax(lg[0], -1); v, ids = p.topk(k)
            hs = [self.child(x, int(b), 0.0) for b in ids]
            # candidate hidden = last hidden after appending the candidate (batched); also gives next logits
            hs = self.extend(hs)
            lgc, hc, _ = self.last(hs)
            sim = (F.normalize(hc.float(), dim=-1) @ H.t()).max(-1).values
            score = (1 - alpha) * v - alpha * sim
            j = int(score.argmax()); x = hs[j]; lg = lgc[j:j + 1]
            H = torch.cat([H, F.normalize(hc[j:j + 1].float(), dim=-1)])
            if int(ids[j]) == EOS: break
        return x.gen


# ------------------------------------------------------------------ evaluation
VARIANTS = [
    ('greedy', 'greedy', {}),
    ('minp_T0.7_p0.1_ng4', 'sample', dict(T=0.7, min_p=0.1)),
    ('minp_T2_p0.05_ng4', 'sample', dict(T=2, min_p=0.05)),
    ('minp_T5_p0.05_ng4', 'sample', dict(T=5, min_p=0.05)),
    ('minp_T10_p0.05_ng4', 'sample', dict(T=10, min_p=0.05)),
    ('minp_T20_p0.05_ng4', 'sample', dict(T=20, min_p=0.05)),
    ('typical0.9_T5', 'sample', dict(T=5, min_p=0.0, typical=0.9)),
    ('typical0.9_T10', 'sample', dict(T=10, min_p=0.0, typical=0.9)),
    ('beam4', 'beam', dict(W=4)),
    ('beam8', 'beam', dict(W=8)),
    ('lookahead_K4_D8', 'lookahead', dict(K=4, D=8)),
    ('contrastive_k4_a0.3', 'contrastive', dict(k=4, alpha=0.3)),
    ('contrastive_k4_a0.6', 'contrastive', dict(k=4, alpha=0.6)),
]

def template_prompt(topic):
    return (f"Explain {topic} from a practical application angle for an intermediate practitioner looking to deepen understanding. "
            f"Provide a step-by-step procedure.\nHere is a treatment of **{topic}** from a practical application angle, formatted as a "
            f"step-by-step procedure, written for an intermediate practitioner looking to deepen understanding.\n\n**Definition.** ")

def main():
    torch.set_num_threads(int(os.environ.get('THREADS', os.cpu_count())))
    CK = os.environ.get('CKPT', '/workspace/ckpt_v3_latest/checkpoints/atom_v3/fractus_atom_1b_gpu0.bin')
    OUT = os.environ.get('OUT', '/workspace/gen_best.jsonl'); N = 40; W = int(os.environ.get('WARM', 127))
    only = os.environ.get('ONLY', '')
    eng, cfg = load_checkpoint(CK); G = FixedGenerator(eng, warm=W); dec = Decoder(G)
    words = set(open('/workspace/words_alpha.txt').read().split()) if os.path.exists('/workspace/words_alpha.txt') else set()
    a = np.fromfile('/workspace/lmsys/ref/atom_corpus/reasoning__syllogisms.i16', dtype=np.int16).astype(np.int64)
    bos = np.where(a == 257)[0]
    cases = [('cont', f'doc{d}', WARM_TAIL[-W:] + a[int(bos[d]):int(bos[d]) + 257].tolist(), a[int(bos[d]) + 257:int(bos[d]) + 257 + N].tolist()) for d in (0, 5000, 10000)]
    cases += [('prompt', p, G.context(G.encode_prompt(p), W), None) for p in ['The capital of France is', 'Once upon a time', 'Hello']]
    cases += [('template', t, G.context(G.encode_prompt(template_prompt(t)), W), None) for t in ['photosynthesis', 'the capital of France', 'a short story', 'negotiation']]
    # batched-forward parity: beam rows vs single row
    if os.environ.get('PARITY', '1') == '1':
      x = dec.start(cases[0][2]); l1, _, _ = dec.last([x]); l4, _, _ = dec.last([x, x, x, x])
      rec = dict(kind='parity_batch', maxabs=float((l4 - l1).abs().max()), argmax_same=bool((l4.argmax(-1) == l1.argmax(-1)).all()))
      print(json.dumps(rec), flush=True); open(OUT, 'a').write(json.dumps(rec) + '\n')
    V = {v[0]: v for v in VARIANTS}
    order = [V[o] for o in only.split(',')] if only else VARIANTS
    skip = set(filter(None, os.environ.get('SKIP_CASES', '').split(',')))
    for name, fn, kw in order:
        for kind, label, ctx, true in cases:
            if label in skip: continue
            t = time.time(); gen = getattr(dec, fn)(ctx, n=N, **kw)
            txt = bytes(i for i in gen if i < 256).decode('utf-8', 'replace')
            ws = re.findall(r'[A-Za-z]{2,}', txt)
            r = dict(kind=kind, variant=name, case=label, unique40=len(set(gen[:N])), generated=show(gen),
                     real_word_rate=(sum(w.lower() in words for w in ws) / len(ws)) if ws and words else None, secs=round(time.time() - t, 1))
            if true is not None:
                r.update(exact=sum(p == q for p, q in zip(gen, true)), seqmatch=difflib.SequenceMatcher(None, gen, true).ratio(), true=show(true))
            print(json.dumps(r, ensure_ascii=False), flush=True); open(OUT, 'a').write(json.dumps(r, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()
