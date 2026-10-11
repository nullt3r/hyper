# usage (host): python3 spec_render.py (reads ~/hyper-requests, writes ~/spec/segments.bin + .json), then
# build/tokdump model.gguf < segments.bin > ids.txt, then python3 spec_sim.py (in ~/spec)
# Render Codex sessions (largest request per prompt_cache_key) roughly as the Qwen3.8 chat template does; mark the
# model-generated assistant turns. Output: segments.bin (NUL-separated) + segments.json ([session, gen, kind])
import json, glob, os
D = os.path.expanduser('~/hyper-requests')
sess = {}
for f in sorted(glob.glob(D + '/*-responses.json')):
    d = json.load(open(f)); k = d.get('prompt_cache_key') or f
    n = lambda r: len(r['input']) if isinstance(r.get('input'), list) else 1
    if k not in sess or n(d) > n(sess[k]): sess[k] = d
def txt(c):
    if isinstance(c, str): return c
    return ''.join(x.get('text', '') for x in (c or []) if isinstance(x, dict))
segs = []
for si, d in enumerate(sess.values()):
    segs.append((si, 0, '', '<|im_start|>system\n' + (d.get('instructions') or '') + '\n\n# Tools\n' + json.dumps(d.get('tools') or []) + '<|im_end|>\n'))
    items = d['input']; i = 0
    if isinstance(items, str): items = [{'type': 'message', 'role': 'user', 'content': items}]
    items = [x for x in items if isinstance(x, dict)]
    def is_asst(x): return x.get('type') in ('reasoning', 'function_call') or (x.get('type') == 'message' and x.get('role') == 'assistant')
    while i < len(items):
        it = items[i]; t = it.get('type', 'message')
        if is_asst(it):
            reasoning = body = calls = ''
            while i < len(items) and is_asst(items[i]):
                x = items[i]; tt = x['type']
                if tt == 'reasoning': reasoning += txt(x.get('content')) or txt(x.get('summary'))
                elif tt == 'message': body += txt(x.get('content'))
                else:
                    try: args = json.loads(x.get('arguments') or '{}')
                    except Exception: args = {'input': x.get('arguments') or ''}
                    if not isinstance(args, dict): args = {'input': json.dumps(args)}
                    calls += '<tool_call>\n<function=' + x.get('name', '') + '>\n' + ''.join(
                        '<parameter=%s>\n%s\n</parameter>\n' % (k, v if isinstance(v, str) else json.dumps(v)) for k, v in args.items()) + '</function>\n</tool_call>'
                i += 1
            segs.append((si, 0, '', '<|im_start|>assistant\n'))
            if reasoning: segs.append((si, 1, 'think', '<think>\n' + reasoning + '\n</think>\n\n'))
            if body: segs.append((si, 1, 'text', body + ('\n\n' if calls else '')))
            if calls: segs.append((si, 1, 'call', calls))
            segs.append((si, 1, 'end', '<|im_end|>'))
            segs.append((si, 0, '', '\n'))
        elif t == 'function_call_output':
            o = it.get('output'); o = o if isinstance(o, str) else json.dumps(o)
            segs.append((si, 0, '', '<|im_start|>user\n<tool_response>\n' + o + '\n</tool_response><|im_end|>\n')); i += 1
        elif t == 'message':
            role = {'developer': 'system'}.get(it.get('role'), it.get('role'))
            segs.append((si, 0, '', '<|im_start|>%s\n%s<|im_end|>\n' % (role, txt(it.get('content'))))); i += 1
        else: i += 1
open(os.path.expanduser('~/spec/segments.bin'), 'w').write('\0'.join(s[3].replace('\0', '') for s in segs))
json.dump([s[:3] for s in segs], open(os.path.expanduser('~/spec/segments.json'), 'w'))
print(len(sess), 'sessions', len(segs), 'segments', sum(len(s[3]) for s in segs if s[1]), 'generated chars')
