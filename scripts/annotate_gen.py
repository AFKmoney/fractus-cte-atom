"""Add prompt_ids / repr decode (boundary markers shown as <Bk>) + run metadata to a gen_block.py jsonl, print a summary.
Usage: python3 scripts/annotate_gen.py FILE.jsonl '{"step_dir":...}'"""
import json,sys;sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__file__),'..'))
from fractus.atom_tokenizer import AtomFractusTokenizer as T
t=T(); names={257:'<BOS>',258:'<EOS>',256:'<PAD>'}
for k in range(1,8): names.setdefault(258+k,f'<B{k}>')
fn=sys.argv[1]; extra=json.loads(sys.argv[2]) if len(sys.argv)>2 else {}
L=[json.loads(l) for l in open(fn)]; L[0].update(extra); out=[L[0]]; by={}
for d in L[1:]:
    d['prompt_ids']=list(t.encode_open(d['prompt'])[0]); d['decoded_repr']=repr(t.decode(d['ids']))
    d['ids_render_repr']=repr(''.join(names[i] if i>=256 else bytes([i]).decode('latin-1') for i in d['ids']))
    out.append(d); by.setdefault(d['cfg'],[]).append(d['unique40_count'])
open(fn,'w').write(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
for d in out[1:]: print(d['cfg'],'|',d['prompt'],'|',f"{d['unique40_count']}/40",'| tf',d['prompt_tf'],'|',d['secs'],'s\n   ',d['ids_render_repr'],'\n   ids',d['ids'])
for c,v in by.items(): print(c,v,'mean',round(sum(v)/len(v),2),'/40 =',round(sum(v)/len(v)/40,3))
