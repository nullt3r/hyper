#!/usr/bin/env python3
"""Replay dumped server requests (HYPER_DUMP_REQUEST) against a running hyper-server, in order.

usage: replay.py URL DUMP_DIR [--session KEY | --largest] [--first N] [--count N] [--max-tokens N] [--temp T] [--out FILE]

Each request is sent non-streaming with the given output cap and temperature (default: greedy). Prints, per request,
the wall time and a hash of the output text, and writes the outputs to --out (JSON) so two runs can be compared.
Speed and draft acceptance per request are in the server log ("eval time", "draft acceptance").
"""
import argparse, glob, hashlib, json, os, sys, time, urllib.request
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("dir")
    ap.add_argument("--session")
    ap.add_argument("--largest", action="store_true", help="the session (prompt_cache_key) with the most requests")
    ap.add_argument("--first", type=int, default=0, help="skip this many requests of the session")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--out")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*-responses.json")))
    reqs = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        reqs.append((f, d))
    if a.largest or a.session:
        key = a.session or Counter(d.get("prompt_cache_key") for _, d in reqs).most_common(1)[0][0]
        reqs = [(f, d) for f, d in reqs if d.get("prompt_cache_key") == key]
        print(f"session {key}: {len(reqs)} requests", file=sys.stderr)
    reqs = reqs[a.first:a.first + a.count]

    results = []
    for f, d in reqs:
        d = dict(d)
        d["stream"] = False
        d["max_output_tokens"] = a.max_tokens
        d["temperature"] = a.temp
        body = json.dumps(d).encode()
        t0 = time.time()
        req = urllib.request.Request(a.url.rstrip("/") + "/v1/responses", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            resp = json.load(r)
        dt = time.time() - t0
        text = json.dumps(resp.get("output", []), sort_keys=True)
        h = hashlib.sha1(text.encode()).hexdigest()[:12]
        usage = resp.get("usage", {})
        print(f"{os.path.basename(f)}  {dt:7.2f} s  out {usage.get('output_tokens', '?'):>5}  hash {h}", flush=True)
        results.append({"file": os.path.basename(f), "seconds": dt, "usage": usage, "hash": h, "output": resp.get("output", [])})
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(results, fh)


if __name__ == "__main__":
    main()
