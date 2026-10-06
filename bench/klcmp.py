# KL(ref_a || ref_b) and top-1 agreement between two reference dumps over the common tokens
import sys, numpy as np
def load(p):
    with open(p, "rb") as f:
        first = 0
        n = int(np.frombuffer(f.read(4), np.int32)[0])
        if n == 0x32464552:   # v2: logits from `first` on
            n, nv, first = (int(v) for v in np.frombuffer(f.read(12), np.int32))
        else:
            nv = int(np.frombuffer(f.read(4), np.int32)[0])
        toks = np.frombuffer(f.read(4 * n), np.int32)
        lg = np.frombuffer(f.read(), np.float32).reshape(n - first, nv)
    return toks, lg
ta, a = load(sys.argv[1]); tb, b = load(sys.argv[2])
assert (ta == tb).all()
n = min(len(a), len(b))
kl = []; top = 0
for i in range(n):
    la = a[i].astype(np.float64); lb = b[i].astype(np.float64)
    la -= la.max(); lb -= lb.max()
    pa = np.exp(la); za = pa.sum(); pa /= za
    lpa = la - np.log(za); lpb = lb - np.log(np.exp(lb).sum())
    kl.append(float((pa * (lpa - lpb)).sum())); top += a[i].argmax() == b[i].argmax()
kl = np.array(kl)
print(f"n={n} top1 {100*top/n:.2f}%  KL mean {kl.mean():.6f} max {kl.max():.5f}  first64 {kl[:64].mean():.6f}")
print("first 24 KL:", " ".join(f"{v:.3f}" for v in kl[:24]))
