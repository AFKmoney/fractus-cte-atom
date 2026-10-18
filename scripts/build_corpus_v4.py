#!/usr/bin/env python3
"""Build the v4 real-language Atom corpus for Fractus CTE-Atom.

Why: the v1-v3 corpus (atom_corpus/, 4.41B ids) is 99.83% slot-filled templates and holds
only 243,575 distinct 16-grams (docs/CORPUS_TEMPLATE_AUDIT.md). v4 replaces it with real
text, interleaved at document level, plus a 100M-id subsample of the old corpus.

Encoding is the one the trainer has always read (fractus/atom_tokenizer.py, Atomizer
linguistic pack mode, max_span_bytes=32): each document is
    BOS(257)  payload bytes + boundary ids (259..264) ...  <B6>(264)  EOS(258)
and documents are concatenated with no padding or extra separator. Files are raw
little-endian int16 (.i16), exactly what open_corpus() in fast4gpu_atom_v*.py memmaps.
The ids come from scripts/atom_fast_ids.py, an ids-only copy of the Atomizer rules;
`verify` checks it id-for-id against AtomFractusTokenizer before anything is used.

Stages (all resumable, all write under --work):
  verify                       fast encoder == reference tokenizer on real docs of every source
  fetch --source S --target-ids N --val-bytes B
                               stream S from the HF Hub (no full dump), clean, exact-dedupe,
                               encode; first B bytes of kept docs go to the held-out val set
  old --target-ids N           ~N ids of whole docs sampled from all 110 old shards (HTTP range reads)
  mix --out DIR                global exact dedupe, document-level shuffle across sources,
                               train_XXX.i16 shards + val.i16 + manifest.json
  stats --out DIR              distinct 16-gram counts, v4 sample vs old-corpus sample

Sources: fineweb (HuggingFaceFW/fineweb-edu sample-10BT, 1 of 14 parquet files), wiki
(wikimedia/wikipedia 20231101.en, 2 of 41 files), pg19 (emozilla/pg19 = deepmind/pg19 as parquet,
2 of 23 files), tinystories (roneneldan/TinyStories TinyStoriesV2-GPT4-train.txt, streamed and
read only as far as needed). Rows are taken in seeded random order inside the picked files.

Runs on the box at nice 19 with few workers; it never touches the training pod.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from atom_fast_ids import BOS, EOS, encode_ids  # noqa: E402

HF_REPO = "thefinalboss/fractus-cte-atom"
MIN_CHARS = {"fineweb": 300, "wiki": 300, "tinystories": 150, "pg19": 2000}
PG19_MAX_CHARS = 400_000


# ------------------------------------------------------------------ cleaning

_MULTI_NL = re.compile(r"\n{3,}")
_TRAIL_WS = re.compile(r"[ \t]+\n")


def clean_basic(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    t = _TRAIL_WS.sub("\n", t)
    t = _MULTI_NL.sub("\n\n", t)
    return t.strip()


def clean_pg19(t: str) -> str:
    """PG-19 is hard-wrapped at ~70 chars. Re-flow single newlines inside a paragraph,
    keep blank-line paragraph breaks, cap very long books at a paragraph boundary."""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    paras = [p for p in re.split(r"\n\s*\n", t)]
    out, n = [], 0
    for p in paras:
        p = " ".join(x.strip() for x in p.split("\n") if x.strip())
        if not p:
            continue
        out.append(p)
        n += len(p) + 2
        if n >= PG19_MAX_CHARS:
            break
    return "\n\n".join(out)


def keep(src: str, t: str) -> bool:
    if len(t) < MIN_CHARS[src]:
        return False
    if "\ufffd" in t:          # mojibake / invalid UTF-8 upstream
        return False
    return True


# ------------------------------------------------------------------ sources

PARQUET = {   # repo, file prefix, files to fetch (seeded pick), shard size ~= 1/N of the dataset
    "fineweb": ("HuggingFaceFW/fineweb-edu", "sample/10BT/", 1),
    "wiki": ("wikimedia/wikipedia", "20231101.en/", 2),
    "pg19": ("emozilla/pg19", "data/train-", 2),
}


def iter_parquet(src: str, seed: int, dl_dir: str):
    """A few seeded parquet files of the dataset (never the full dump), rows in seeded random
    order (row groups shuffled, rows shuffled inside each group). hf_hub_download is used
    instead of HTTP streaming because long streamed reads of the big PG-19 rows kept being
    cut by the server ("peer closed connection"); downloads resume and retry."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    repo, prefix, k = PARQUET[src]
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.startswith(prefix) and f.endswith(".parquet"))
    rng = np.random.default_rng(seed)
    pick = [files[i] for i in rng.choice(len(files), size=min(k, len(files)), replace=False)]
    print(f"{src}: parquet files {pick}", flush=True)
    # interleave the picked files row group by row group
    def fetch(f):
        for attempt in range(10):
            try:
                return hf_hub_download(repo, f, repo_type="dataset", local_dir=dl_dir)
            except Exception as exc:   # flaky CDN: retry, the download resumes
                print(f"{src}: download {f} failed ({type(exc).__name__}: {str(exc)[:120]}), retry {attempt + 1}", flush=True)
                time.sleep(10 * (attempt + 1))
        raise SystemExit(f"{src}: could not download {f}")
    pfs = [pq.ParquetFile(fetch(f)) for f in pick]
    groups = [(i, g) for i, pf in enumerate(pfs) for g in range(pf.metadata.num_row_groups)]
    for j in rng.permutation(len(groups)):
        i, g = groups[j]
        col = pfs[i].read_row_group(g, columns=["text"]).column(0).to_pylist()
        for r in rng.permutation(len(col)):
            t = col[r]
            if isinstance(t, str):
                yield t


def iter_source(src: str, seed: int = 0, dl_dir: str = "/workspace/v4build/dl"):
    """Yield raw text strings (TinyStories streamed line by line; others: iter_parquet)."""
    if src == "tinystories":
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        path = "datasets/roneneldan/TinyStories/TinyStoriesV2-GPT4-train.txt"
        buf = []
        with fs.open(path, "rb", block_size=8 << 20) as fh:
            for raw in fh:
                line = raw.decode("utf-8", errors="replace")
                if line.strip() == "<|endoftext|>":
                    if buf:
                        yield "".join(buf)
                    buf = []
                else:
                    buf.append(line)
        if buf:
            yield "".join(buf)
        return
    if src not in PARQUET:
        raise SystemExit(f"unknown source {src}")
    yield from iter_parquet(src, seed, dl_dir)


def _work(args):
    src, t = args
    t = clean_pg19(t) if src == "pg19" else clean_basic(t)
    if not keep(src, t):
        return None
    h = int.from_bytes(hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest(), "little")
    ids = np.asarray(encode_ids(t), dtype=np.int16)
    return h, len(t.encode("utf-8")), ids.tobytes()


def _init_worker():
    try:
        os.nice(19)
    except OSError:
        pass


# ------------------------------------------------------------------ stages

def stage_verify(args):
    from fractus.atom_tokenizer import AtomFractusTokenizer
    tok = AtomFractusTokenizer(max_span_bytes=32, pack_mode="linguistic")
    rep = {}
    for src in args.sources.split(","):
        n = ok = nbytes = 0
        for t in iter_source(src, args.seed, str(Path(args.work) / 'dl')):
            t = clean_pg19(t) if src == "pg19" else clean_basic(t)
            if not keep(src, t):
                continue
            t = t[:20_000]                 # the reference encoder is slow (torch features)
            ok += int(tok.encode(t) == encode_ids(t))
            n += 1
            nbytes += len(t.encode())
            if n >= args.verify_docs:
                break
        rep[src] = {"docs": n, "identical": ok, "bytes": nbytes}
        print(src, rep[src], flush=True)
        if ok != n:
            raise SystemExit(f"fast encoder differs from AtomFractusTokenizer on {src}")
    (Path(args.work) / "verify.json").write_text(json.dumps(rep, indent=1))


def stage_fetch(args):
    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    src = args.source
    done = work / f"{src}.done.json"
    if done.exists():
        print(f"{src}: already done {done.read_text()}"); return
    f_ids = open(work / f"{src}.i16", "wb")
    f_val = open(work / f"{src}_val.i16", "wb")
    lens, hashes, vlens, vhashes = [], [], [], []
    seen = set()
    st = {"seen": 0, "short_or_bad": 0, "dup": 0, "kept": 0, "ids": 0, "bytes": 0,
          "val_docs": 0, "val_ids": 0, "val_bytes": 0}
    t0 = time.time()
    gen = ((src, t) for t in iter_source(src, args.seed, str(work / 'dl')))
    with Pool(args.workers, initializer=_init_worker) as pool:
        for res in pool.imap(_work, gen, chunksize=8 if src == "pg19" else 64):
            st["seen"] += 1
            if res is None:
                st["short_or_bad"] += 1; continue
            h, nb, raw = res
            if h in seen:
                st["dup"] += 1; continue
            seen.add(h)
            n_ids = len(raw) // 2
            if st["val_bytes"] < args.val_bytes:
                f_val.write(raw); vlens.append(n_ids); vhashes.append(h)
                st["val_docs"] += 1; st["val_ids"] += n_ids; st["val_bytes"] += nb
                continue
            f_ids.write(raw); lens.append(n_ids); hashes.append(h)
            st["kept"] += 1; st["ids"] += n_ids; st["bytes"] += nb
            if st["kept"] % 20000 == 0:
                el = time.time() - t0
                print(f"{src}: {st['ids']:,} ids {st['kept']:,} docs {st['ids'] / el:,.0f} ids/s {st}", flush=True)
            if st["ids"] >= args.target_ids:
                break
        pool.terminate()
    f_ids.close(); f_val.close()
    np.asarray(lens, np.int64).tofile(work / f"{src}.len")
    np.asarray(hashes, np.uint64).tofile(work / f"{src}.h64")
    np.asarray(vlens, np.int64).tofile(work / f"{src}_val.len")
    np.asarray(vhashes, np.uint64).tofile(work / f"{src}_val.h64")
    st["secs"] = round(time.time() - t0, 1)
    done.write_text(json.dumps(st, indent=1))
    print(f"{src}: DONE {st}", flush=True)


def stage_old(args):
    """Whole docs from every old per-skill shard, at a seeded random offset (HTTP range reads)."""
    import requests
    from huggingface_hub import HfApi, hf_hub_url
    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    done = work / "old.done.json"
    if done.exists():
        print("old: already done"); return
    info = HfApi().model_info(HF_REPO, files_metadata=True)
    shards = sorted((s.rfilename, s.size) for s in info.siblings
                    if s.rfilename.startswith("atom_corpus/") and s.rfilename.endswith(".i16")
                    and s.rfilename != "atom_corpus/atom_corpus.i16")
    per = args.target_ids // len(shards)
    rng = np.random.default_rng(args.seed)
    f_ids = open(work / "old.i16", "wb")
    lens, hashes, seen = [], [], set()
    st = {"shards": len(shards), "docs": 0, "ids": 0, "dup": 0, "per_shard_target": per}
    for name, size in shards:
        n_ids = size // 2
        want = min(n_ids, per + 40_000)       # margin for the partial docs at both ends
        start = int(rng.integers(0, max(1, n_ids - want)))
        exp = want * 2
        for attempt in range(10):
            try:
                r = requests.get(hf_hub_url(HF_REPO, name), timeout=300,
                                 headers={"Range": f"bytes={start * 2}-{(start + want) * 2 - 1}"})
                r.raise_for_status()
                if len(r.content) != exp:
                    raise IOError(f"short read {len(r.content)} != {exp}")
                break
            except Exception as exc:   # flaky CDN: retry the same range
                print(f"old: {name} range read failed ({type(exc).__name__}), retry {attempt + 1}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit(f"old: could not read {name}")
        a = np.frombuffer(r.content, dtype=np.int16)
        bos = np.flatnonzero(a == BOS)
        got = 0
        for s, e in zip(bos[:-1], bos[1:]):
            d = a[s:e]
            if d[-1] != EOS:
                continue
            h = int.from_bytes(hashlib.blake2b(d.tobytes(), digest_size=8).digest(), "little")
            if h in seen:
                st["dup"] += 1; continue
            seen.add(h)
            f_ids.write(d.tobytes()); lens.append(len(d)); hashes.append(h)
            got += len(d)
            if got >= per:
                break
        st["docs"] = len(lens); st["ids"] += got
        print(f"old: {name} +{got:,} ids, total {st['ids']:,}", flush=True)
    f_ids.close()
    np.asarray(lens, np.int64).tofile(work / "old.len")
    np.asarray(hashes, np.uint64).tofile(work / "old.h64")
    done.write_text(json.dumps(st, indent=1))
    print("old: DONE", st)


def _write_docs(out_path, order, srcs, mm, offs, lens):
    """Write docs (src_index, doc_index) in order; returns ids written."""
    n = 0
    with open(out_path, "wb") as f:
        for si, di in order:
            o, l = offs[si][di], lens[si][di]
            f.write(np.asarray(mm[si][o:o + l]).tobytes())
            n += int(l)
    return n


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stage_mix(args):
    work, out = Path(args.work), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    srcs = [s for s in ("fineweb", "wiki", "tinystories", "pg19", "old") if (work / f"{s}.len").exists()]
    mm, offs, lens, hs = [], [], [], []
    for s in srcs:
        mm.append(np.memmap(work / f"{s}.i16", dtype=np.int16, mode="r"))
        L = np.fromfile(work / f"{s}.len", dtype=np.int64)
        lens.append(L); offs.append(np.concatenate([[0], np.cumsum(L)[:-1]]))
        hs.append(np.fromfile(work / f"{s}.h64", dtype=np.uint64))
        assert int(L.sum()) == mm[-1].shape[0], (s, int(L.sum()), mm[-1].shape)
    # global exact dedupe (val hashes first, so no val doc also sits in train)
    seen = set()
    vsrc = [s for s in srcs if (work / f"{s}_val.len").exists()]
    for s in vsrc:
        seen.update(int(x) for x in np.fromfile(work / f"{s}_val.h64", dtype=np.uint64))
    table, per_src = [], {}
    for si, s in enumerate(srcs):
        kept = dup = 0; ids = 0
        for di, h in enumerate(hs[si].tolist()):
            if h in seen:
                dup += 1; continue
            seen.add(h); table.append((si, di)); kept += 1; ids += int(lens[si][di])
        per_src[s] = {"docs": kept, "ids": ids, "cross_source_dups_dropped": dup,
                      "fetch": json.loads((work / f"{s}.done.json").read_text())}
    table = np.asarray(table, dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    table = table[rng.permutation(len(table))]
    # train shards at doc boundaries
    shards, cur, cur_ids, k = [], [], 0, 0
    def flush():
        nonlocal cur, cur_ids, k
        if not cur:
            return
        p = out / f"train_{k:03d}.i16"
        n = _write_docs(p, cur, srcs, mm, offs, lens)
        shards.append({"file": p.name, "ids": n, "docs": len(cur), "bytes": n * 2})
        print(f"mix: {p.name} {n:,} ids {len(cur):,} docs", flush=True)
        cur, cur_ids, k = [], 0, k + 1
    total_ids = int(sum(int(lens[si][di]) for si, di in table.tolist()))
    n_sh = max(1, round(total_ids / args.shard_ids))   # no tiny tail shard: the last one absorbs it
    for si, di in table.tolist():
        cur.append((si, di)); cur_ids += int(lens[si][di])
        if cur_ids >= args.shard_ids and k < n_sh - 1:
            flush()
    flush()
    # val shard: all held-out docs, shuffled across sources
    vparts, vmeta = [], {}
    for s in vsrc:
        a = np.fromfile(work / f"{s}_val.i16", dtype=np.int16)
        L = np.fromfile(work / f"{s}_val.len", dtype=np.int64)
        o = np.concatenate([[0], np.cumsum(L)[:-1]])
        vparts += [a[o[i]:o[i] + L[i]] for i in range(len(L))]
        vmeta[s] = {"docs": int(len(L)), "ids": int(L.sum()),
                    "text_bytes": json.loads((work / f"{s}.done.json").read_text())["val_bytes"]}
    vorder = np.random.default_rng(args.seed + 1).permutation(len(vparts))
    val = np.concatenate([vparts[i] for i in vorder]).astype(np.int16)
    val.tofile(out / "val.i16")
    for sh in shards:
        sh["sha256"] = _sha256(out / sh["file"])
    total = sum(sh["ids"] for sh in shards)
    man = {
        "name": "atom_corpus_v4", "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "builder": "scripts/build_corpus_v4.py", "tokenizer": "atom-fractus-v1",
        "atomizer": {"pack_mode": "linguistic", "max_span_bytes": 32}, "vocab_size": 266,
        "dtype": "int16 little-endian raw (.i16)",
        "doc_format": "BOS(257) payload+boundary ids ... <B6>(264) EOS(258), docs concatenated, no padding",
        "train_order": "train_000.i16, train_001.i16, ... concatenated in name order",
        "shuffle_seed": args.seed, "train_ids": total, "train_docs": int(len(table)),
        "shards": shards, "per_source": per_src,
        "share": {s: round(per_src[s]["ids"] / total, 4) for s in per_src},
        "val": {"file": "val.i16", "ids": int(val.size), "docs": len(vparts), "per_source": vmeta,
                "text_bytes": sum(v["text_bytes"] for v in vmeta.values()),
                "sha256": _sha256(out / "val.i16")},
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: man[k] for k in ("train_ids", "train_docs", "share")}, indent=1))


def _distinct16(a: np.ndarray) -> int:
    """Distinct 16-grams (64-bit polynomial hash) that do not cross a doc start (BOS at
    positions 1..15 of the window), as in docs/CORPUS_TEMPLATE_AUDIT.md. Memory-lean."""
    n = a.size - 15
    if n <= 0:
        return 0
    P = np.uint64(1099511628211)
    h = np.zeros(n, dtype=np.uint64)
    tmp = np.empty(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(16):
            np.multiply(h, P, out=h)
            tmp[:] = a[j:j + n]
            tmp += np.uint64(1)
            h += tmp
    del tmp
    c = np.zeros(a.size + 1, dtype=np.int32)
    np.cumsum(a == BOS, out=c[1:])
    ok = (c[16:16 + n] - c[1:1 + n]) == 0
    h = h[ok]
    del c, ok
    h.sort()
    return int(1 + np.count_nonzero(h[1:] != h[:-1])) if h.size else 0


def stage_stats(args):
    out, work = Path(args.out), Path(args.work)
    man = json.loads((out / "manifest.json").read_text())
    N = args.sample_ids
    # v4: contiguous N ids from the start of train_000 (already doc-shuffled across sources)
    v4 = np.fromfile(out / man["shards"][0]["file"], dtype=np.int16, count=N)
    # old: N ids of the old-corpus subsample (whole docs from all 110 shards)
    old = np.fromfile(work / "old.i16", dtype=np.int16, count=N)
    res = {"sample_ids": N}
    res["v4_distinct16"] = _distinct16(v4)
    res["old_distinct16"] = _distinct16(old)
    res["old_full_corpus_distinct16_reference"] = 243_575
    res["old_full_corpus_ids_reference"] = 4_410_941_912
    for s in ("fineweb", "wiki", "tinystories", "pg19"):
        p = work / f"{s}.i16"
        if p.exists():
            res[f"{s}_distinct16_first_{N // 10:,}"] = _distinct16(np.fromfile(p, dtype=np.int16, count=N // 10))
    res["ratio_v4_over_old"] = round(res["v4_distinct16"] / max(1, res["old_distinct16"]), 1)
    man["distinct16"] = res
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["verify", "fetch", "old", "mix", "stats"])
    ap.add_argument("--work", default="/workspace/v4build")
    ap.add_argument("--out", default="/workspace/v4build/atom_corpus_v4")
    ap.add_argument("--source")
    ap.add_argument("--sources", default="fineweb,wiki,tinystories,pg19")
    ap.add_argument("--target-ids", type=int, default=0)
    ap.add_argument("--val-bytes", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-ids", type=int, default=200_000_000)
    ap.add_argument("--verify-docs", type=int, default=200)
    ap.add_argument("--sample-ids", type=int, default=100_000_000)
    args = ap.parse_args()
    try:
        os.nice(19)
    except OSError:
        pass
    {"verify": stage_verify, "fetch": stage_fetch, "old": stage_old,
     "mix": stage_mix, "stats": stage_stats}[args.stage](args)


if __name__ == "__main__":
    main()
