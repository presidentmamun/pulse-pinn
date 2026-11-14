"""Check the batched trainer against the single-fit pilot: same 10 cases x 4 levels, one model.
Usage: python check_batch.py [model=B2] [iters=8000]"""
import os, sys, json, glob, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from pulsens.pinn import load_case
from pulsens.pinn_batch import fit_batch

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
kind = sys.argv[1] if len(sys.argv) > 1 else "B2"
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
files = sorted(glob.glob(os.path.join(R, "data", "twin_l1", "case_*.npz")))[:10]
ds, lv = [], []
for p in files:
    for level in "abcd":
        ds.append(load_case(p, level, np.random.default_rng(int(os.path.basename(p)[5:8]) * 10 + "abcd".index(level)))); lv.append(level)
t0 = time.time()
outs = fit_batch(ds, lv, kind, iters=iters, log_every=1000)
wall = time.time() - t0
pilot = {}
for line in open(os.path.join(R, "reports", "phase4", "pilot.jsonl")):
    r = json.loads(line)
    if r["model"] == kind:
        pilot[(r["case"], r["level"])] = r
print(f"{len(ds)} fits in {wall / 60:.1f} min ({wall / len(ds):.1f} s per fit)")
rows = []
for d, l, o in zip(ds, lv, outs):
    p = pilot.get((d["case"], l))
    rows.append((l, o["rmse_mmHg"], p["rmse_mmHg"] if p else np.nan, abs(o["beta_rel_err"]), abs(p["beta_rel_err"]) if p else np.nan))
for l in "abcd":
    v = np.array([r[1:] for r in rows if r[0] == l])
    print(f"level {l}: rmse batch {np.median(v[:, 0]):.2f} vs pilot {np.median(v[:, 1]):.2f} | |beta| batch {np.median(v[:, 2]):.3f} vs pilot {np.median(v[:, 3]):.3f}")
