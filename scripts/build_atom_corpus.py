#!/usr/bin/env python
"""Retokenize raw text with the Atomizer adapted to Fractus.

Does not read GPT-2 ``.pt`` corpora. Those ids are vocab 50257 and are
invalid for this head. Input is jsonl / txt / jsonl.gz. Output is an int16
memmap plus a small json sidecar (vocab, span, pack mode, token count).

Usage:
    python scripts/build_atom_corpus.py --src data/raw --out data/atom_corpus
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fractus.atom_tokenizer import VOCAB_SIZE, AtomFractusTokenizer


def extract_text(entry: dict) -> str:
    for key in ("text", "content", "output", "completion", "response"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val
    msgs = entry.get("messages") or entry.get("conversations")
    if isinstance(msgs, list):
        parts = []
        for msg in msgs:
            if isinstance(msg, dict):
                parts.append(str(msg.get("content") or msg.get("value") or ""))
            elif isinstance(msg, str):
                parts.append(msg)
        return "\n".join(p for p in parts if p)
    return ""


def iter_texts(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    if path.endswith(".txt"):
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield handle.read()
        return
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                yield line
                continue
            if isinstance(entry, str):
                yield entry
            elif isinstance(entry, dict):
                text = extract_text(entry)
                if text:
                    yield text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="file or directory of txt/jsonl")
    ap.add_argument("--out", required=True, help="output prefix (writes .i16 and .json)")
    ap.add_argument("--max-span-bytes", type=int, default=32)
    ap.add_argument("--pack-mode", default="linguistic")
    ap.add_argument("--min-chars", type=int, default=8)
    args = ap.parse_args()

    paths = []
    if os.path.isdir(args.src):
        for root, _, files in os.walk(args.src):
            for name in files:
                if name.endswith((".jsonl", ".jsonl.gz", ".txt", ".txt.gz")):
                    paths.append(os.path.join(root, name))
    else:
        paths.append(args.src)
    if not paths:
        raise SystemExit(f"no raw text under {args.src}")

    tok = AtomFractusTokenizer(max_span_bytes=args.max_span_bytes, pack_mode=args.pack_mode)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    bin_path = args.out if args.out.endswith(".i16") else args.out + ".i16"
    ids = []
    docs = 0
    for path in paths:
        for text in iter_texts(path):
            if len(text) < args.min_chars:
                continue
            ids.extend(tok.encode(text))
            docs += 1
    arr = np.asarray(ids, dtype=np.int16)
    arr.tofile(bin_path)
    meta = {
        "tokenizer": AtomFractusTokenizer.VERSION,
        "vocab_size": VOCAB_SIZE,
        "max_span_bytes": args.max_span_bytes,
        "pack_mode": args.pack_mode,
        "docs": docs,
        "tokens": int(arr.size),
        "bin": os.path.basename(bin_path),
        "dtype": "int16",
        "note": "fresh Atom id stream. Do not mix with GPT-2 50257 shards.",
    }
    with open(os.path.splitext(bin_path)[0] + ".json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
