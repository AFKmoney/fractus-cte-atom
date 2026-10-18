"""Ids-only re-implementation of AtomFractusTokenizer.encode (pack_mode=linguistic,
max_span_bytes=32), without the 320-dim torch features that the training ids never use.

The segmentation rules are copied line for line from fractus/atomizer.py
(Atomizer.encode_bytes, linguistic branch). build_corpus_v4.py --verify checks that the
output is identical, id for id, to AtomFractusTokenizer(max_span_bytes=32,
pack_mode="linguistic").encode(text) on real documents before any shard is written.

Id map (fractus/atom_tokenizer.py): 0..255 bytes, 256 PAD, 257 BOS, 258 EOS,
259 whitespace, 260 punctuation, 261 newline, 262 class_transition, 263 max_span,
264 eos boundary, 265 generated.
"""
from __future__ import annotations

import string

BOS, EOS = 257, 258
B_WS, B_PUNCT, B_NL, B_CT, B_MAX, B_EOS = 259, 260, 261, 262, 263, 264

_PUNCT = frozenset(bytes(string.punctuation, "ascii"))
_WS = frozenset((9, 10, 13, 32))
# class codes: 0 space, 1 digit, 2 alpha, 3 control, 4 punctuation, 5 continuation, 6 other
_CLS = []
for _v in range(256):
    if _v in (9, 10, 13, 32):
        _CLS.append(0)
    elif 48 <= _v <= 57:
        _CLS.append(1)
    elif 65 <= _v <= 90 or 97 <= _v <= 122:
        _CLS.append(2)
    elif _v < 32:
        _CLS.append(3)
    elif _v in _PUNCT:
        _CLS.append(4)
    elif 0x80 <= _v <= 0xBF:
        _CLS.append(5)
    else:
        _CLS.append(6)
_CLS = tuple(_CLS)


def _ends_at_utf8_boundary(buf) -> bool:
    if not buf:
        return True
    last = buf[-1]
    if last < 0x80:
        return True
    if 0x80 <= last <= 0xBF:
        cc = 0
        for v in reversed(buf):
            if not (0x80 <= v <= 0xBF):
                break
            cc += 1
        if cc >= len(buf):
            return False
        lead = buf[-cc - 1]
        if 0xC2 <= lead <= 0xDF:
            exp = 1
        elif 0xE0 <= lead <= 0xEF:
            exp = 2
        elif 0xF0 <= lead <= 0xF4:
            exp = 3
        else:
            return True
        return cc >= exp
    return not (0xC2 <= last <= 0xF4)


def encode_ids(text: str, max_span: int = 32) -> list:
    """Same ids as AtomFractusTokenizer(max_span, 'linguistic').encode(text)."""
    raw = text.encode("utf-8")
    soft = max(2, (max_span * 3) // 4)
    out = [BOS]
    ext = out.extend
    app = out.append
    buf = bytearray()
    last_soft = 0
    PUNCT, CLS = _PUNCT, _CLS
    for v in raw:
        buf.append(v)
        n = len(buf)
        # --- Atomizer._boundary_reason
        if v == 10 or v == 13:
            ext(buf); app(B_NL); buf = bytearray(); last_soft = 0
            continue
        if v in PUNCT:
            reason = B_PUNCT
        elif (v == 9 or v == 32) and any(b not in (9, 10, 13, 32) for b in buf[:-1]):
            reason = B_WS
        elif n >= 2 and not (0x80 <= v <= 0xBF):
            pc, cc = CLS[buf[-2]], CLS[v]
            reason = B_CT if (pc != cc and pc in (1, 2) and cc in (1, 2)) else 0
        else:
            reason = 0
        if reason:
            last_soft = n
            if n >= soft:
                ext(buf); app(reason); buf = bytearray(); last_soft = 0
                continue
        if n >= max_span and _ends_at_utf8_boundary(buf):
            # --- Atomizer._best_linguistic_cut
            cut = None
            if n >= 2:
                if 2 <= last_soft < n and _ends_at_utf8_boundary(buf[:last_soft]):
                    cut = last_soft
                else:
                    for i in range(n - 2, 0, -1):
                        b = buf[i]
                        if b in (9, 10, 13, 32) or b in PUNCT:
                            c = i + 1
                            if c >= 2 and _ends_at_utf8_boundary(buf[:c]):
                                cut = c
                                break
            if cut is not None and 2 <= cut < n:
                cb = buf[cut - 1]
                if cb in (9, 10, 13, 32):
                    lab = B_WS
                elif cb in PUNCT:
                    lab = B_PUNCT
                else:
                    lab = B_MAX
                ext(buf[:cut]); app(lab); buf = buf[cut:]; last_soft = 0
            else:
                ext(buf); app(B_MAX); buf = bytearray(); last_soft = 0
    if buf:
        ext(buf); app(B_EOS)
    app(EOS)
    return out
