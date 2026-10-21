import json,re,collections,statistics as st,sys
R=collections.defaultdict(list)
for l in open(sys.argv[1] if len(sys.argv)>1 else 'tune.jsonl'):
    d=json.loads(l); R[d['cfg']].append(d)
def distinct(g):
    t=re.sub(r'<[^>]+>',' ',g); ws=re.findall(r'[A-Za-z]{2,}',t); return len(set(w.lower() for w in ws))/len(ws) if ws else 0
print('| config | n | unique@40 | exact@40 | real-word@40 | real-word@80 | real-word>=3 | distinct-word | ')
for k,v in R.items():
    m=lambda f: st.mean(f(x) for x in v)
    print(f"| {k} | {len(v)} | {m(lambda x:x['u40']):.1f} | {m(lambda x:x['exact40']):.2f} | {m(lambda x:x['rw40']):.2f} | {m(lambda x:x['rw']):.2f} | {m(lambda x:x['rw3']):.2f} | {m(lambda x:distinct(x['gen'])):.2f} |")
