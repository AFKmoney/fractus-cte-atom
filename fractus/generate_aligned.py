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

def _pick(logits: torch.Tensor, prev: Optional[int], temperature: float, top_k: int) -> int:
    l = logits.float().reshape(-1).clone()
    if prev is not None and 0 <= prev < l.numel():
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
    out: List[int] = []
    prev = ids[-1]
    if carry:
        _reset(engine)
        logits = engine.tick_chunk(torch.tensor([ids], dtype=torch.long))
        cur = logits[0, -1]
        for i in range(max_new):
            if force_second and i == 0:
                l = cur.float().clone()
                l[int(l.argmax())] = -1e9
                nxt = _pick(l, prev=prev, temperature=temperature, top_k=top_k)
            else:
                nxt = _pick(cur, prev=prev, temperature=temperature, top_k=top_k)
            out.append(nxt)
            ids.append(nxt)
            logits = engine.tick_chunk(torch.tensor([[nxt]], dtype=torch.long))
            cur = logits[0, -1]
            prev = nxt
    else:
        for i in range(max_new):
            _reset(engine)
            ctx = ids[-window:]
            logits = engine.tick_chunk(torch.tensor([ctx], dtype=torch.long))
            cur = logits[0, -1]
            if force_second and i == 0:
                l = cur.float().clone()
                l[int(l.argmax())] = -1e9
                nxt = _pick(l, prev=prev, temperature=temperature, top_k=top_k)
            else:
                nxt = _pick(cur, prev=prev, temperature=temperature, top_k=top_k)
            out.append(nxt)
            ids.append(nxt)
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
