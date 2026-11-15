"""Phase 5 level-d bias test: the distant cuff over-reads SBP (amplification, +5..10 mmHg in the twin).
Variant 'dc': subtract a fixed population-mean amplification (default 7.5 mmHg) from the cuff SBP before the fit.
DBP is used as is. Same noise/bias draws as phase5_full.py (same rng seeds), so results pair with level d.
Note: 7.5 is the mean of the twin's bias range, i.e. an optimistic prior (a literature value would be used on real data).
Usage: python phase5_dcorr.py label kind harmonics hold_col seed [E=40] [shift_mmHg=7.5] [max_cases=all]
Writes reports/phase5/dcorr.jsonl (resumable)."""
import os, sys, json, glob, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from pulsens.pinn import load_case, MMHG
from pulsens.pinn_batch import fit_batch

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
label, kind, NH, HOLD, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
E = int(sys.argv[6]) if len(sys.argv) > 6 else 40
SHIFT = float(sys.argv[7]) if len(sys.argv) > 7 else 7.5
MAXC = int(sys.argv[8]) if len(sys.argv) > 8 else None
iters = 8000
LOG = os.path.join(R, "reports", "phase5", "dcorr.jsonl" if MAXC is None else "dcorr_test.jsonl")
done = set()
if os.path.exists(LOG):
    for s in open(LOG):
        try:
            r = json.loads(s)
        except ValueError:
            continue
        if r["model"] == label and r["seed"] == SEED and r["shift"] == SHIFT:
            done.add(r["case"])
files = sorted(glob.glob(os.path.join(R, "data", "twin_l1", "case_*.npz")))[:MAXC]
t_all = time.time(); n = 0
for i in range(0, len(files), E):
    ds = []
    for p in files[i:i + E]:
        c = int(os.path.basename(p)[5:8])
        d = load_case(p, "d", np.random.default_rng(c * 10 + 3 + 100000 * SEED))
        if d["case"] in done:
            continue
        s, dd = d["cuff"]
        d["cuff_raw"] = (s, dd); d["cuff"] = (s - SHIFT * MMHG, dd)
        ds.append(d)
    if not ds:
        continue
    t0 = time.time()
    outs = fit_batch(ds, ["d"] * len(ds), kind, iters=iters, log_every=1000, seeds=[SEED] * len(ds), K=NH,
                     hold_col=None if HOLD < 0 else HOLD, d_floor=[0.0] * len(ds))
    with open(LOG, "a") as fh:
        for d, o in zip(ds, outs):
            o.update(case=d["case"], level="dc", model=label, seed=SEED, iters=iters, harmonics=NH, hold_col=HOLD,
                     shift=SHIFT, cuff_bias_sbp=d["cuff_raw"][0] / MMHG - d["SBP"])
            fh.write(json.dumps(o) + "\n")
    n += len(ds); w = time.time() - t0
    print(f"{label} dc seed {SEED}: {len(ds)} fits in {w / 60:.1f} min ({w / len(ds):.1f} s per fit); {n} done, {(time.time() - t_all) / 60:.0f} min total", flush=True)
print(f"finished {label} dc seed {SEED}: {n} new fits")
