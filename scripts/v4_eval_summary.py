import json, sys, collections, numpy as np
P = sys.argv[1] if len(sys.argv) > 1 else '/workspace/scratch/dec/v4_eval.jsonl'
R = [json.loads(l) for l in open(P) if l.strip()]
tags = sorted({r['tag'] for r in R}, key=lambda t: (t != 'v3_330000', t))
out = []
out.append('### Val NLL per source (8 windows x 1024 scored ids each, 127-id context)\n')
out.append('| model | source | NLL nats/id (mean of windows) | median window NLL | top-1 | bits/byte |'); out.append('|---|---|---|---|---|---|')
for t in tags:
    for s in ['fineweb', 'wiki', 'tinystories', 'pg19']:
        w = [r for r in R if r['tag'] == t and r['kind'] == 'valwin' and r['src'] == s]
        if not w: continue
        nll = np.mean([x['nll'] for x in w]); tot = sum(x['nll'] * x['ids'] for x in w) / np.log(2); by = sum(x['bytes'] for x in w)
        out.append(f"| {t} | {s} | {nll:.3f} | {np.median([x['nll'] for x in w]):.3f} | {np.mean([x['top1'] for x in w]):.3f} | {tot / by:.3f} |")
out.append('\n### Copy probe on real text (words of >=5 letters in the scored region; previous occurrence in the same 128-id grid chunk vs only in an earlier chunk)\n')
out.append('| model | class | words | NLL/id all bytes | top-1 all | NLL first byte | top-1 first byte | NLL bytes 2..n | top-1 bytes 2..n | median distance (ids) |')
out.append('|---|---|---|---|---|---|---|---|---|---|')
for t in tags:
    acc = collections.defaultdict(lambda: collections.Counter()); D = collections.defaultdict(list)
    for r in R:
        if r['tag'] == t and r['kind'] == 'valwin':
            for c, e in r['words'].items():
                for k, v in e.items():
                    if k == 'dists': D[c] += v
                    else: acc[c][k] += v
    for c in ['first', 'repeat_same_chunk', 'repeat_earlier_chunk']:
        e = acc[c]
        if not e['n']: continue
        md = int(np.median(D[c])) if D[c] else '-'
        out.append(f"| {t} | {c} | {e['n_first']} | {e['nll'] / e['n']:.3f} | {e['top1'] / e['n']:.3f} | {e['nll_first'] / e['n_first']:.3f} | {e['top1_first'] / e['n_first']:.3f} | "
                   f"{e['nll_rest'] / max(1, e['n_rest']):.3f} | {e['top1_rest'] / max(1, e['n_rest']):.3f} | {md} |")
out.append('\n### Generation (fp32 fixed path, warm = real 127-id val tail; CONT = 256-id val prefix, 40 ids)\n')
out.append('| model | decoder | CONT unique@40 | CONT exact /40 | CONT seqmatch | CONT real-word | prompt unique@40 | prompt real-word |')
out.append('|---|---|---|---|---|---|---|---|')
for t in tags:
    for d in ['greedy', 'beam8', 'minp_T0.7_p0.1_ng4']:
        g = [r for r in R if r['tag'] == t and r['kind'] == 'gen' and r['decoder'] == d]
        if not g: continue
        c = [x for x in g if 'exact' in x]; p = [x for x in g if 'exact' not in x]
        f = lambda L, k: np.mean([x[k] or 0 for x in L]) if L else float('nan')
        out.append(f"| {t} | {d} | {f(c,'unique40'):.1f} | {f(c,'exact'):.2f} | {f(c,'seqmatch'):.3f} | {f(c,'real_word_rate'):.2f} | {f(p,'unique40'):.1f} | {f(p,'real_word_rate'):.2f} |")
out.append('\n### Raw outputs (unedited, markers shown)\n')
for t in tags:
    for d in ['greedy', 'beam8', 'minp_T0.7_p0.1_ng4']:
        g = [r for r in R if r['tag'] == t and r['kind'] == 'gen' and r['decoder'] == d]
        if not g: continue
        out.append(f'**{t} / {d}**\n'); out.append('| case | generated | true |'); out.append('|---|---|---|')
        for x in g:
            esc = lambda s: s.replace('|', '\\|').replace('\n', '⏎').replace('`', "'")
            out.append(f"| {x['case']} | `{esc(x['generated'])}` | " + (f"`{esc(x['true'])}` |" if 'true' in x else ' |'))
        out.append('')
print('\n'.join(out))
