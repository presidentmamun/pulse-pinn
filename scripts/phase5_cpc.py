"""Phase 5: CPC-PINN (periodic embedding + Psi potential, mass exact) vs the Phase 4 baselines.
Same 10 pilot cases x 4 levels and same noise seeds; CPC fits run batched (pulsens/pinn_batch.py).
Appends to reports/phase5/cpc.jsonl, then prints CPC medians next to the pilot medians (B0..B3).
Usage: python phase5_cpc.py [iters=8000] [K=10] [kind=CPC] [harmonics=8] [hold_col=-1] [d_floor=0, or e.g. 0.8ab = floor only on levels a,b] [label=kind]"""
import os, sys, json, glob, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from pulsens.pinn import load_case
from pulsens.pinn_batch import fit_batch

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
iters = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
kind = sys.argv[3] if len(sys.argv) > 3 else "CPC"
NH = int(sys.argv[4]) if len(sys.argv) > 4 else 8
HOLD = int(sys.argv[5]) if len(sys.argv) > 5 else -1
FS = sys.argv[6] if len(sys.argv) > 6 else "0"
FLOOR = float(FS.rstrip("abcd")); FLEV = FS[len(FS.rstrip("abcd")):] or "abcd"
label = sys.argv[7] if len(sys.argv) > 7 else kind
OUT = os.path.join(R, "reports", "phase5"); os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "cpc.jsonl")
files = sorted(glob.glob(os.path.join(R, "data", "twin_l1", "case_*.npz")))[:K]
ds, lv = [], []
for p in files:
    for level in "abcd":
        ds.append(load_case(p, level, np.random.default_rng(int(os.path.basename(p)[5:8]) * 10 + "abcd".index(level)))); lv.append(level)
t0 = time.time()
outs = fit_batch(ds, lv, kind, iters=iters, log_every=1000, K=NH, hold_col=None if HOLD < 0 else HOLD, d_floor=[FLOOR if l in FLEV else 0.0 for l in lv])
wall = time.time() - t0
with open(LOG, "a") as fh:
    for d, l, o in zip(ds, lv, outs):
        o.update(case=d["case"], level=l, model=label, iters=iters, harmonics=NH, hold_col=HOLD, d_floor=FLOOR if l in FLEV else 0.0); fh.write(json.dumps(o) + "\n")
print(f"{len(ds)} {label} fits, {iters} iters, {wall / 60:.1f} min ({wall / len(ds):.1f} s per fit)")
cases = {d["case"] for d in ds}
rows = [json.loads(s) for s in open(os.path.join(R, "reports", "phase4", "pilot.jsonl"))]
rows = [r for r in rows if r["case"] in cases] + [dict(o) for o in outs]
print("level model  rmse  |sbp|  |dbp|  shape_r  |beta|")
for l in "abcd":
    for m in ("B0", "B1", "B2", "B3", label):
        v = [r for r in rows if r["level"] == l and r["model"] == m]
        if not v:
            continue
        med = lambda k, a=True: np.median([abs(r[k]) if a else r[k] for r in v]) if k in v[0] else np.nan
        print(f"  {l}   {m:4s} {med('rmse_mmHg'):5.2f} {med('sbp_err'):6.2f} {med('dbp_err'):6.2f} {med('shape_corr', False):8.4f} {med('beta_rel_err'):6.3f}")
