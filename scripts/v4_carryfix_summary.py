"""Markdown tables for scripts/v4_carryfix.py results (tf by chunk, diag, gen + raw outputs)."""
import json, sys, numpy as np
P = sys.argv[1] if len(sys.argv) > 1 else '/workspace/scratch/carry/carry.jsonl'
R = [json.loads(l) for l in open(P)]
tf = [r for r in R if r['kind'] == 'tf']; gen = [r for r in R if r['kind'] == 'gen']; dg = [r for r in R if r['kind'] == 'diag']
print('### TF NLL by grid chunk (12 non-pg19 windows = every 2nd of the 24; nats/id)\n')
print('| policy | ' + ' | '.join(f'c{i}' for i in range(9)) + ' | mean c3-8 |'); print('|---' * 11 + '|')
for r in tf:
    pw = np.array(r['per_window']); pw = pw[::2] if len(pw) == 24 else pw; m = pw.mean(0)
    print(f"| {r['key']} | " + ' | '.join('%.3f' % x for x in m) + ' | %.4f |' % m[3:].mean())
if dg:
    d = dg[0]['per_block']; print('\n### Carry diagnosis (baseline, 8 windows): block 15 / mean over blocks 1-15\n')
    print('| chunk | ||z|| block15 | ||S|| block15 | carry share of denominator (mean blocks 1-15) | at last position |'); print('|---|---|---|---|---|')
    for c in range(9):
        sh = np.mean([d[f'block{b}']['den_share'][c] for b in range(1, 16)]); sl = np.mean([d[f'block{b}']['den_share_last'][c] for b in range(1, 16)])
        print(f"| {c} | {d['block15']['z'][c]:.3g} | {d['block15']['S'][c]:.3g} | {sh:.3f} | {sl:.3f} |")
print('\n### Generation (40 ids, warm 64)\n')
print('| policy | decoder | group | n | unique@40 | exact /40 | seqmatch | real-word rate |'); print('|---|---|---|---|---|---|---|---|')
groups = {}
for r in gen:
    g = 'cont' if r['case'].startswith('cont_') else 'long' if r['case'].startswith('long_') else 'prompt'
    groups.setdefault((r['policy'], r['decoder'], g), []).append(r)
f = lambda xs: '%.2f' % np.mean(xs) if xs else '-'
for (p, dn, g), rs in groups.items():
    print(f"| {p} | {dn} | {g} | {len(rs)} | {f([r['unique40'] for r in rs])} | {f([r['exact'] for r in rs if 'exact' in r])} | "
          f"{f([r['seqmatch'] for r in rs if 'seqmatch' in r])} | {f([r['real_word_rate'] for r in rs if r['real_word_rate'] is not None])} |")
print('\n### Raw outputs (unedited, markers shown)\n')
for r in gen:
    print(f"- **{r['policy']} / {r['decoder']} / {r['case']}** (u{r['unique40']}, exact {r.get('exact','-')}): `{r['generated']}`" + (f"  \n  true: `{r['true']}`" if 'true' in r else ''))
