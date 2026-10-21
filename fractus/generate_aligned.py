"""Train-aligned generation for Fractus CTE. No frequency ban / scramble."""
from __future__ import annotations
from typing import List, Optional
import torch

def _reset(engine):
    engine.eval()
    if hasattr(engine, "reset_thought"):
        engine.reset_thought(1)
    for blk in getattr(engine, "blocks", []):
        if hasattr(blk, "attn_S"):
            blk.attn_S.zero_()
        if hasattr(blk, "attn_z"):
            blk.attn_z.zero_()

def _pick(logits: torch.Tensor, prev: Optional[int], temperature: float, top_k: int, ban_prev: bool = False) -> int:
    l = logits.float().reshape(-1).clone()
    # Atom ids are bytes. Banning the previous id forbids double letters,
    # spaces and UTF-8 runs, and forces ntntnt. The speech gate is greedy
    # PREFIX with no ban. ban_prev stays available for a cycle probe.
    if ban_prev and prev is not None and 0 <= prev < l.numel():
        l[prev] = -1e9
    if temperature <= 1e-5:
        return int(l.argmax().item())
    l = l / max(temperature, 1e-5)
    k = min(max(1, top_k), l.numel())
    topv, topi = torch.topk(l, k)
    return int(topi[torch.multinomial(torch.softmax(topv, -1), 1)].item())

@torch.no_grad()
def generate_window(
    engine,
    tokenizer,
    prompt: str,
    max_new: int = 40,
    temperature: float = 0.0,
    top_k: int = 40,
    window: int = 128,
    force_second: bool = False,
    carry: bool = False,
) -> tuple[str, List[int]]:
    ids = tokenizer.encode(prompt)[:window] or [0]
    feat = None
    opener = getattr(tokenizer, "encode_open", None) or getattr(getattr(tokenizer, "_atom", None), "encode_open", None)
    if opener is not None:
        ids, feat = opener(prompt)
        ids = ids[:window] or [0]
        feat = feat[: len(ids)]
    elif hasattr(tokenizer, "encode_with_features"):
        ids, feat = tokenizer.encode_with_features(prompt)
        ids = ids[:window] or [0]
        feat = feat[: len(ids)]
    out: List[int] = []
    prev = ids[-1]
    if carry:
        # 2026-10-21 (docs/GEN_FINAL.md): the old loop fed every new id as its own 1-id chunk, so thought_state was
        # added at every id and the attention state was averaged over levels at every id (TF NLL 2,290-10,559 vs
        # 1.8-2.8 on v4 final). Now the fixed 128-id grid, exact incremental path, warm start, open prompt.
        from .fixed_generation import FixedGenerator
        G = getattr(engine, "_fixed_gen", None)
        if G is None:
            G = FixedGenerator(engine, warm=64)
            engine._fixed_gen = G
        ctx = G.context(G.encode_prompt(prompt))
        out = G.generate_stream(ctx, max_new=max_new, temperature=temperature, top_k=top_k)
        return tokenizer.decode(out), out
    if carry:
        _reset(engine)
        ctx_feat = feat.unsqueeze(0) if feat is not None else None
        logits = engine.tick_chunk(torch.tensor([ids], dtype=torch.long), feat=ctx_feat)
        cur = logits[0, -1]
        for i in range(max_new):
            if force_second and i == 0:
                l = cur.float().clone()
                l[int(l.argmax())] = -1e9
                nxt = _pick(l, prev=prev, temperature=temperature, top_k=top_k, ban_prev=False)
            else:
                nxt = _pick(cur, prev=prev, temperature=temperature, top_k=top_k, ban_prev=False)
            out.append(nxt)
            ids.append(nxt)
            logits = engine.tick_chunk(torch.tensor([[nxt]], dtype=torch.long), feat=None)
            cur = logits[0, -1]
            prev = nxt
    else:
        for i in range(max_new):
            _reset(engine)
            ctx = ids[-window:]
            ctx_feat = None
            if feat is not None:
                ctx_feat = feat[-len(ctx):].unsqueeze(0)
                if ctx_feat.shape[1] != len(ctx):
                    ctx_feat = None
            logits = engine.tick_chunk(torch.tensor([ctx], dtype=torch.long), feat=ctx_feat)
            cur = logits[0, -1]
            if force_second and i == 0:
                l = cur.float().clone()
                l[int(l.argmax())] = -1e9
                nxt = _pick(l, prev=prev, temperature=temperature, top_k=top_k, ban_prev=False)
            else:
                nxt = _pick(cur, prev=prev, temperature=temperature, top_k=top_k, ban_prev=False)
            out.append(nxt)
            ids.append(nxt)
            if feat is not None:
                feat = torch.cat([feat, torch.zeros(1, feat.shape[-1])], dim=0)
            prev = nxt
    return tokenizer.decode(out), out

@torch.no_grad()
def generate_greedy_prefix(engine, tokenizer, prompt: str, max_new: int = 40, window: int = 128):
    return generate_window(engine, tokenizer, prompt, max_new=max_new, temperature=0.0, top_k=1, window=window, carry=False)

@torch.no_grad()
def generate_greedy_ids(engine, tokenizer, prompt: str, max_new: int = 40, window: int = 128):
    return generate_greedy_prefix(engine, tokenizer, prompt, max_new=max_new, window=window)

@torch.no_grad()
def generate_chunk(engine, tokenizer, prompt: str, max_new: int = 40, **kw):
    return generate_window(engine, tokenizer, prompt, max_new=max_new, window=128, **kw)

@torch.no_grad()
def unique40_probe(engine, max_new=40, mode="prefix", prompts=None, tokenizer=None):
    """Greedy PREFIX probe. Counts unique ids over the generated span.

    The previous body returned the arguments and generated nothing. The gate
    is this function. tokenizer is required on the Atom fork: ids are bytes,
    so unique ids and unique bytes are the same count.
    """
    if tokenizer is None:
        from .tokenizer import FractusTokenizer
        tokenizer = FractusTokenizer()
    prompts = prompts or ["Bonjour", "The ", "def "]
    rows = []
    for prompt in prompts:
        if mode == "prefix":
            text, ids = generate_greedy_prefix(engine, tokenizer, prompt, max_new=max_new)
        else:
            text, ids = generate_window(engine, tokenizer, prompt, max_new=max_new, temperature=0.0, top_k=1)
        rows.append({
            "prompt": prompt,
            "text": text,
            "unique": len(set(ids)),
            "n": len(ids),
        })
    return {"mode": mode, "max_new": max_new, "rows": rows}

# space-sync 2026-08-29b force_second carry
