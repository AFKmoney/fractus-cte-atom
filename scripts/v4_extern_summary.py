import json, sys, collections, numpy as np
P = sys.argv[1] if len(sys.argv) > 1 else '/workspace/scratch/extern/extern.jsonl'
R = [json.loads(l) for l in open(P) if l.strip()]
tags = sorted({r['tag'] for r in R})
o = ['### (a) Teacher-forced accuracy on val (32 windows, 1,024 scored ids each)\n',
     '| model | source | top-1 | top-5 | NLL |', '|---|---|---|---|---|']
for t in tags:
    for s in ['fineweb', 'wiki', 'tinystories', 'pg19', 'all']:
        w = [r for r in R if r['tag'] == t and r['kind'] == 'tf' and (s == 'all' or r['src'] == s)]
        if w: o.append(f"| {t} | {s} | {np.mean([x['top1'] for x in w]):.3f} | {np.mean([x['top5'] for x in w]):.3f} | {np.median([x['nll'] for x in w]):.3f} (median) |")
o += ['\n### (a) Word completion (greedy from the true prefix; real words >=5 letters; 6 per window, 24 windows)\n',
      '| model | given bytes | words | exact word | TF top-1 on next byte | first-step logits vs TF max abs diff |', '|---|---|---|---|---|---|']
for t in tags:
    it = [i for r in R if r['tag'] == t and r['kind'] == 'compl' for i in r['items']]
    for k in (1, 2, 3):
        x = [i for i in it if i['k'] == k]
        if x: o.append(f"| {t} | {k} | {len(x)} | {np.mean([i['exact'] for i in x]):.3f} | {np.mean([i['tf_top1_next'] for i in x]):.3f} | {max(i['first_step_vs_tf_maxabs'] for i in x):.2e} |")
o += ['\n### (b) Free generation (6 val contexts x 64 ids)\n',
      '| model | decoder | real-word rate gen | words gen | real-word rate true | mean log p(generated) | mean log p(true) | gen vs TF logits max abs diff |', '|---|---|---|---|---|---|---|---|']
for t in tags:
    for d in ['greedy', 'minp_T0.7_p0.1']:
        x = [r for r in R if r['tag'] == t and r['kind'] == 'free' and r['decoder'] == d]
        if not x: continue
        ng = sum(r['n_words'] for r in x); rg = sum((r['real_word_rate'] or 0) * r['n_words'] for r in x) / max(ng, 1)
        nt = sum(r['true_n_words'] for r in x); rt = sum((r['true_real_word_rate'] or 0) * r['true_n_words'] for r in x) / max(nt, 1)
        o.append(f"| {t} | {d} | {rg:.3f} | {ng} | {rt:.3f} | {np.mean([r['mean_logp_generated'] for r in x]):.3f} | {np.mean([r['mean_logp_true'] for r in x]):.3f} | {max(r['gen_vs_tf_logits_maxabs'] for r in x):.2e} |")
o += ['\n### Long range: NLL on grid chunk 7 (ids 897..1023 of each window)\n',
      '| model | full carry (7 chunks before) | carry reset 1 chunk before (128-255 ids context) | full, same ids | reset at chunk (0-127 ids, in-chunk only) |', '|---|---|---|---|---|']
for t in tags:
    w = [r for r in R if r['tag'] == t and r['kind'] == 'tf' and r['src'] != 'pg19']
    if w: o.append(f"| {t} | {np.mean([r['lr_full'] for r in w]):.3f} | {np.mean([r['lr_reset6'] for r in w]):.3f} | {np.mean([r['lr_full_from1'] for r in w]):.3f} | {np.mean([r['lr_reset7_from1'] for r in w]):.3f} |")
o.append('\n(pg19 excluded from the long-range rows: two windows sit at 13-15 nats in every model.)')
o.append('\n### Raw free generations (unedited, markers shown)\n')
for t in tags:
    for r in [r for r in R if r['tag'] == t and r['kind'] == 'free']:
        o.append(f"- **{t} / {r['key']}**: `{r['generated']}`  \n  true: `{r['true']}`")
o.append('\n### Word completions, raw (first 3 windows of each source, k=2)\n')
for t in tags:
    it = [(r['key'], i) for r in R if r['tag'] == t and r['kind'] == 'compl' and int(r['key'].split(':')[1]) < 3 for i in r['items'] if i['k'] == 2]
    o.append(f"- **{t}**: " + '; '.join(f"{i['word']}→`{i['completion']}`" for _, i in it))
print('\n'.join(o))
