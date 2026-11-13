"""Phase 4 pilot: B0 + B1/B2/B3 PINNs x data levels a-d on the first K accepted twin cases.
Appends one JSON line per fit to reports/phase4/pilot.jsonl (resumable: finished fits are skipped).
Usage: python phase4_pilot.py [K=10] [iters=8000] [models=B1,B2,B3] [levels=a,b,c,d]
"""
import os, sys, json, glob, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from pulsens.pinn import load_case, fit, b0_cuff_scaling

R = os.environ.get("PULSE_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(R, "reports", "phase4"); os.makedirs(OUT, exist_ok=True)
K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
MODELS = (sys.argv[3] if len(sys.argv) > 3 else "B1,B2,B3").split(",")
LEVELS = (sys.argv[4] if len(sys.argv) > 4 else "a,b,c,d").split(",")
LOG = os.path.join(OUT, "pilot.jsonl")

done = set()
if os.path.exists(LOG):
    for line in open(LOG):
        r = json.loads(line); done.add((r["case"], r["level"], r["model"], r["iters"]))
files = sorted(glob.glob(os.path.join(R, "data", "twin_l1", "case_*.npz")))[:K]
t0 = time.time()
for path in files:
    for level in LEVELS:
        d = load_case(path, level, np.random.default_rng(int(os.path.basename(path)[5:8]) * 10 + "abcd".index(level)))
        jobs = (["B0"] if level in ("c", "d") else []) + MODELS
        for model in jobs:
            key = (d["case"], level, model, ITERS)
            if key in done:
                continue
            res = b0_cuff_scaling(d) if model == "B0" else fit(d, level, model, iters=ITERS)
            res.update(case=d["case"], level=level, model=model, iters=ITERS)
            open(LOG, "a").write(json.dumps(res) + "\n")
            print(f"[{(time.time() - t0) / 60:5.1f} min] case {d['case']:03d} {level} {model}: rmse {res['rmse_mmHg']:.1f} mmHg "
                  f"sbp {res['sbp_err']:+.1f} dbp {res['dbp_err']:+.1f} r {res['shape_corr']:.3f}"
                  + (f" beta {res['beta_rel_err']:+.2f}" if "beta_rel_err" in res else "") + f" ({res.get('secs', 0)} s)", flush=True)
print("PILOT DONE", round((time.time() - t0) / 60, 1), "min")
