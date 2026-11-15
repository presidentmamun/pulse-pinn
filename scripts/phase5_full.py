"""Phase 5 full run: all twin_l1 cases x 4 levels for one model and one seed, in GPU chunks of E fits.
Resumable: fits already in reports/phase5/full.jsonl (same model, seed, case, level) are skipped.
Seed s changes both the diameter-noise draw and the network init (s=0 reproduces the pilot noise).
Usage: python phase5_full.py label kind harmonics hold_col d_floor seed [E=40] [iters=8000] [max_cases=all]
  e.g. CPCv2 CPC 4 6 0.8ab 0   |   B3 B3 8 -1 0 0"""
import os, sys, json, glob, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from pulsens.pinn import load_case
from pulsens.pinn_batch import fit_batch

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
label, kind, NH, HOLD, FS, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6])
E = int(sys.argv[7]) if len(sys.argv) > 7 else 40
iters = int(sys.argv[8]) if len(sys.argv) > 8 else 8000
MAXC = int(sys.argv[9]) if len(sys.argv) > 9 else None
FLOOR = float(FS.rstrip("abcd")); FLEV = FS[len(FS.rstrip("abcd")):] or "abcd"
OUT = os.path.join(R, "reports", "phase5"); os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "full.jsonl" if iters == 8000 else f"full_test{iters}.jsonl")
done = set()
if os.path.exists(LOG):
    for s in open(LOG):
        r = json.loads(s)
        if r["model"] == label and r["seed"] == SEED:
            done.add((r["case"], r["level"]))
files = sorted(glob.glob(os.path.join(R, "data", "twin_l1", "case_*.npz")))[:MAXC]
todo = []
for p in files:
    c = int(os.path.basename(p)[5:8])
    for level in "abcd":
        todo.append((p, c, level))
jobs = [(p, c, l) for p, c, l in todo]
t_all = time.time(); n = 0
for i in range(0, len(jobs), E):
    chunk = jobs[i:i + E]
    ds, lv = [], []
    for p, c, l in chunk:
        d = load_case(p, l, np.random.default_rng(c * 10 + "abcd".index(l) + 100000 * SEED))
        if (d["case"], l) in done:
            continue
        ds.append(d); lv.append(l)
    if not ds:
        continue
    t0 = time.time()
    outs = fit_batch(ds, lv, kind, iters=iters, log_every=1000, seeds=[SEED] * len(ds), K=NH, hold_col=None if HOLD < 0 else HOLD,
                     d_floor=[FLOOR if l in FLEV else 0.0 for l in lv])
    with open(LOG, "a") as fh:
        for d, l, o in zip(ds, lv, outs):
            o.update(case=d["case"], level=l, model=label, seed=SEED, iters=iters, harmonics=NH, hold_col=HOLD,
                     d_floor=FLOOR if l in FLEV else 0.0); fh.write(json.dumps(o) + "\n")
    n += len(ds); w = time.time() - t0
    print(f"{label} seed {SEED}: chunk {i // E + 1}, {len(ds)} fits in {w / 60:.1f} min ({w / len(ds):.1f} s per fit); {n} done, {(time.time() - t_all) / 60:.0f} min total", flush=True)
print(f"finished {label} seed {SEED}: {n} new fits")
