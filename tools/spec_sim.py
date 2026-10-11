# Offline simulation of speculative decoding policies on real generated tokens (exact: a draft token counts only if it
# equals the token the model actually produced). Cost model from Flash-Next replays (262k ctx, T 1):
#   step verifying R rows = A + B*R ms; MTP step additionally C ms for the draft chain; MTP: 1.88 accepted / step
import json, random, sys, collections
A_, B_, C_ = 5.9, 4.4, 1.2
segs = json.load(open('segments.json')); ids = [list(map(int, l.split())) for l in open('ids.txt')]
sessions = collections.defaultdict(list)
for (si, gen, kind), t in zip(segs, ids): sessions[si].append((gen, kind, t))
NS = (32, 16, 8, 6, 4, 3, 2)
def run(policy, Dmax=8, minlen=4, q=None, seed=1):
    rng = random.Random(seed)
    tot_t = 0.0; tot_tok = 0; steps = 0; ng_steps = 0; ng_tok = 0; by = collections.Counter(); byt = collections.Counter()
    for si, parts in sessions.items():
        stream = []; spans = []
        for gen, kind, t in parts:
            if gen: spans.append((len(stream), len(stream) + len(t), kind))
            stream += t
        idx = {n: {} for n in NS}; done = 0   # positions indexed so far (keys ending at j < done)
        def index_upto(p):
            nonlocal done
            for j in range(done, p):
                for n in NS:
                    if j - n + 1 >= 0: idx[n][hash(tuple(stream[j - n + 1:j + 1]))] = j
            done = max(done, p)
        for s, e, kind in spans:
            p = s
            while p < e:
                index_upto(p - 1)   # keys ending before p-1: an earlier occurrence of the current suffix
                best = None
                if policy != 'mtp_only':
                    for n in NS:
                        if p - n < 0: continue
                        j = idx[n].get(hash(tuple(stream[p - n:p])))
                        if j is not None and stream[j - n + 1:j + 1] == stream[p - n:p] and j + 1 < p:
                            m = n   # actual match length (extend backwards)
                            while m < 256 and j - m >= 0 and p - m - 1 >= 0 and stream[j - m] == stream[p - m - 1]: m += 1
                            best = (j, m); break
                # what an n-gram draft of length D would get
                def ng_accept(D):
                    j, m = best; d = stream[j + 1:min(j + 1 + D, p)]; a = 0
                    while a < len(d) and p + a < e and d[a] == stream[p + a]: a += 1
                    return a, len(d)
                # MTP model: copy-able positions (n-gram would accept 3) -> MTP accepts 3; elsewhere a chain with
                # per-token acceptance q (calibrated so the overall mean is the measured 1.88)
                def mtp_accept():
                    if best is not None and ng_accept(3)[0] >= 3: return 3
                    a = 0
                    while a < 3 and rng.random() < q: a += 1
                    return a
                if policy == 'plain':
                    a, cost = 0, A_ + B_
                elif policy == 'mtp_only' or best is None or best[1] < minlen:
                    a, cost = mtp_accept(), A_ + B_ * 3.12 + C_
                    if policy == 'ngram_only': a, cost = 0, A_ + B_
                else:
                    D = Dmax
                    a, dl = ng_accept(D)
                    if dl == 0: a, cost = mtp_accept(), A_ + B_ * 3.12 + C_
                    else: cost = A_ + B_ * (1 + dl); ng_steps += 1; ng_tok += a + 1
                k = min(a + 1, e - p)
                p += k; tot_tok += k; tot_t += cost; steps += 1; by[kind] += k; byt[kind] += cost
    return dict(tps=1000 * tot_tok / tot_t, tok_step=tot_tok / steps, ng_share=ng_tok / max(1, tot_tok), steps=steps,
                kinds={k: round(1000 * by[k] / byt[k], 1) for k in by})
# calibrate q: MTP-only mean accepted / step = 1.88
lo, hi = 0.0, 1.0
for _ in range(14):
    q = (lo + hi) / 2
    r = run('mtp_only', q=q)
    if r['tok_step'] - 1 < 1.88: lo = q
    else: hi = q
print('calibrated q %.3f' % q, file=sys.stderr)
base = run('mtp_only', q=q)
print('plain     ', run('plain'))
print('mtp_only  ', base)
for Dmax in (3, 5, 8, 12, 16, 24):
    for minlen in (3, 6, 12):
        r = run('hybrid', Dmax=Dmax, minlen=minlen, q=q)
        print('hybrid D%-2d m%-2d' % (Dmax, minlen), 'speedup %.3f' % (r['tps'] / base['tps']), r)
