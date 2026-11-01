#!/usr/bin/env python3
# Offline expert-placement simulator for GLM (engine5) on routing traces (HYPER5_ROUTE_TRACE).
# Replays prompt / decode rows through a placement policy and reports CPU experts per decode job and an estimated
# decode time (CPU job time per distinct CPU experts from cpumoebench + the GPU part of a layer + swap cost).
# usage: placesim.py trace.bin stats.bin [--gpu 71] [--policy cur|static|oracle|window] [--every 16 --decay 0.92 ...]
import argparse, struct, sys
import numpy as np

def read_trace(path):
    ev = []
    with open(path, 'rb') as f:
        while True:
            h = f.read(16)
            if len(h) < 16: break
            typ, rows, nm, K = struct.unpack('<4i', h)
            a = np.frombuffer(f.read(2 * rows * nm * K), dtype=np.int16).reshape(nm, rows, K)
            ev.append((typ, a))
    return ev

def read_stats(path, moe_layers):
    with open(path, 'rb') as f:
        nl, w = struct.unpack('<2i', f.read(8))
        s = np.frombuffer(f.read(8 * nl * w), dtype=np.uint64).reshape(nl, w).astype(np.float64)
    return s[moe_layers]

# cpumoebench (Maya-L, 30 threads, GAP 260 us): job time by distinct CPU experts (us)
T_JOB = [0, 221, 346, 502, 665, 829, 990, 1150, 1310, 1470, 1630, 1790, 1950, 2110, 2270, 2430, 2590]

def tjob(n): return T_JOB[n] if n < len(T_JOB) else T_JOB[-1] + 160 * (n - len(T_JOB) + 1)

def simulate(ev, prior, G, a):
    nm, E = prior.shape
    owner = np.zeros((nm, E), bool)                     # True: on a GPU
    for l in range(nm): owner[l, np.argsort(-prior[l], kind='stable')[:G]] = True
    score = prior / np.maximum(prior.sum(1, keepdims=True), 1) * 256.0 * 8
    cnt = np.zeros((nm, E))
    dec = [ev_ for ev_ in ev if ev_[0] == 2]
    # oracle: placement fixed to the trace's own decode routing (best static choice in hindsight)
    if a.policy == 'oracle':
        tot = np.zeros((nm, E))
        for _, x in dec:
            for l in range(nm): np.add.at(tot[l], x[l].ravel(), 1)
        owner[:] = False
        for l in range(nm): owner[l, np.argsort(-tot[l], kind='stable')[:G]] = True
    hist = np.zeros(64, np.int64); tcpu = 0.0; swaps = 0; steps = 0; prompt_routed = False; jobs = 0
    future = None
    if a.policy == 'window':   # lookahead: counts of the next W decode steps (an upper bound for a predictive policy)
        future = [x for _, x in dec]
    di = 0
    sw_prompt = 0
    def rebalance(budget):
        nonlocal swaps, sw_prompt
        cand = []
        for l in range(nm):
            cpu = np.where(~owner[l])[0]; gpu = np.where(owner[l])[0]
            sc = score[l]
            ci = cpu[np.argsort(-sc[cpu], kind='stable')]; gi = gpu[np.argsort(sc[gpu], kind='stable')]
            for i in range(min(len(ci), len(gi))):
                si, so = sc[ci[i]], sc[gi[i]]
                if si < so * a.ratio + a.min: break
                cand.append((si - so, l, ci[i], gi[i]))
        cand.sort(key=lambda c: -c[0])
        for _, l, ein, eout in cand[:budget]:
            owner[l, ein] = True; owner[l, eout] = False
        swaps += min(len(cand), budget)
        if budget >= 1 << 20: sw_prompt += min(len(cand), budget)
    for typ, x in ev:
        if typ == 1:
            if a.policy in ('cur', 'window'):
                for l in range(nm): np.add.at(score[l], x[l].ravel(), a.pw)
                prompt_routed = True
            continue
        # decode step (x: [nm][rows][K])
        if a.policy == 'cur':
            if prompt_routed or (steps + 1) % a.every == 0:
                score = score * a.decay + cnt; cnt[:] = 0
                rebalance(1 << 20 if prompt_routed else a.budget)
                if prompt_routed: steps = 0
                prompt_routed = False
            else: pass
            steps += 1
        elif a.policy == 'window' and di % a.every == 0:
            fut = np.zeros((nm, E))
            for y in future[di:di + a.window]:
                for l in range(nm): np.add.at(fut[l], y[l].ravel(), 1)
            score = fut; rebalance(a.budget)
        di += 1
        for l in range(nm):
            ids = np.unique(x[l].ravel())
            ids = ids[ids >= 0]
            n_cpu = int((~owner[l, ids]).sum())
            hist[min(n_cpu, 63)] += 1; tcpu += tjob(n_cpu); jobs += 1
            np.add.at(cnt[l], x[l].ravel(), 1)
    n_dec = len(dec)
    t_tok = (tcpu + jobs * a.gap) / max(n_dec, 1) + swaps * a.swap_us / max(n_dec, 1)
    mean = (hist * np.arange(64)).sum() / max(hist.sum(), 1)
    return dict(sw_prompt=sw_prompt, steps=n_dec, cpu_per_job=mean, swaps=swaps, est_tps=1e6 / t_tok if n_dec else 0, hist=hist[:9] / max(hist.sum(), 1))

ap = argparse.ArgumentParser()
ap.add_argument('trace'); ap.add_argument('stats')
ap.add_argument('--gpu', type=int, default=71, help='GPU experts per layer (all GPUs together)')
ap.add_argument('--policy', default='cur', choices=['cur', 'static', 'oracle', 'window'])
ap.add_argument('--every', type=int, default=16); ap.add_argument('--decay', type=float, default=0.92)
ap.add_argument('--budget', type=int, default=64); ap.add_argument('--ratio', type=float, default=1.25)
ap.add_argument('--min', type=float, default=6.0); ap.add_argument('--pw', type=float, default=0.25)
ap.add_argument('--window', type=int, default=64)
ap.add_argument('--gap', type=float, default=420.0, help='GPU part of a layer, us (calibrated: the trace run measured 27.2 t/s at 2.50 CPU experts/job)'); ap.add_argument('--swap_us', type=float, default=300.0)
ap.add_argument('--first_moe', type=int, default=3)
a = ap.parse_args()
ev = read_trace(a.trace)
nm = ev[0][1].shape[0]
prior = read_stats(a.stats, list(range(a.first_moe, a.first_moe + nm)))
r = simulate(ev, prior, a.gpu, a)
print("%-7s steps %5d  CPU experts/job %.2f  swaps %6d (after prompts %d)  est %.1f t/s  jobs with 0..8 CPU experts: %s" % (
    a.policy, r['steps'], r['cpu_per_job'], r['swaps'], r['sw_prompt'], r['est_tps'], ' '.join('%.2f' % v for v in r['hist'])))
