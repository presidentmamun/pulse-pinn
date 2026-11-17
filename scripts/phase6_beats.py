"""Phase 6b: why does the PINN under-read PWV on real data? Two checks on the 8 real subjects.

1. Beat statistics of the real D(t): period CV, pulse-amplitude CV, slow baseline drift (as % of the pulse amplitude).
   These numbers also drive the realistic-noise twin (phase6_twin_realnoise.py).
2. Two cleaner ways to fit the same data (B3, level c, assumed cuff 120/80, seed 0):
   - "ens": ensemble-averaged beat (drift removed, all beats folded with their own period, 40 phase bins), one fit per subject;
   - "beat": every complete beat fitted on its own (its own period), median over beats.
Writes reports/phase6/beats.json (stats) and appends fits to reports/phase6/real.jsonl (model "B3ens" / "B3beat").
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from pulsens.pinn import LX, MMHG
from pulsens.pinn_batch import fit_batch
sys.argv = sys.argv[:1]                      # phase6_real reads argv; use its defaults
import importlib.util
spec = importlib.util.spec_from_file_location("p6", os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase6_real.py"))
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(R, "reports", "phase6")
GOOD = ["s01", "s02", "s04", "s05", "s06", "s07", "s08", "s10"]
CUFF = (120 * MMHG, 80 * MMHG)
MODE = os.environ.get("P6MODE", "all")      # "norm": only the taper-normalised ensemble fits


def load_build():
    src = open(spec.origin, encoding="utf-8").read()
    src = src[:src.index("done = set()")]           # only the helper functions, not the run
    g = {"__file__": spec.origin}; exec(compile(src, spec.origin, "exec"), g)
    return g["build"], g["period"]


build, period = load_build()


def beats_of(t, Dm, Tc):
    """Beat feet = diastolic minima, one per period window."""
    k = int(round(Tc / (t[1] - t[0])))
    s = np.convolve(Dm, np.ones(3) / 3, "same")
    feet, i = [], int(np.argmin(s[:k]))
    while i < len(s):
        feet.append(i)
        lo, hi = i + int(0.7 * k), min(len(s), i + int(1.3 * k) + 1)
        if lo >= len(s) - 1 or hi - lo < 2:
            break
        i = lo + int(np.argmin(s[lo:hi]))
    return feet


stats, jobs = {}, []
for sid in GOOD:
    d0, info = build(sid)
    z = np.load(os.path.join(R, "reports", "phase1", f"{sid}_long_ts.npz"))
    t = z["t"]
    nC = 9
    Dsel = (np.sqrt(np.asarray(d0["A"]).reshape(len(t), nC) * 4 / np.pi)) * 1e3        # mm, frames x 9
    Dm = Dsel.mean(1)
    feet = beats_of(t, Dm, d0["Tc"])
    per = np.diff(t[feet])
    amps = [Dm[a:b].max() - Dm[a:b].min() for a, b in zip(feet[:-1], feet[1:])]
    base = Dm[feet]
    drift = (base.max() - base.min()) / np.mean(amps) * 100 if len(base) > 1 else 0.0
    stats[sid] = dict(n_beats=len(per), period_cv_pct=round(float(np.std(per) / np.mean(per) * 100), 1),
                      amp_cv_pct=round(float(np.std(amps) / np.mean(amps) * 100), 1), drift_pct_of_pulse=round(float(drift), 1),
                      feet_s=[round(float(t[f]), 3) for f in feet])
    # drift removal: linear interpolation of the feet baseline, applied to every column
    bl = np.interp(t, t[feet], base) - base.mean()
    Dc = Dsel - bl[:, None]
    # ensemble beat: each beat on its own 0..1 phase, 40 bins
    ph, rows = [], []
    for a, b in zip(feet[:-1], feet[1:]):
        idx = np.arange(a, b)
        ph.append((t[idx] - t[a]) / (t[b] - t[a])); rows.append(Dc[idx])
    ph = np.concatenate(ph); rows = np.concatenate(rows)
    nb = 40; bins = np.minimum((ph * nb).astype(int), nb - 1)
    De = np.stack([rows[bins == k].mean(0) if np.any(bins == k) else np.full(nC, np.nan) for k in range(nb)])
    ok = ~np.isnan(De[:, 0]); tc = (np.arange(nb) + .5) / nb
    De = np.stack([np.interp(tc, tc[ok], De[ok, j], period=1.0) for j in range(nC)], 1)
    xi_cols = np.unique(np.asarray(d0["xi"]))
    def case(tau, D, Tc, tag):
        A = np.pi * (D * 1e-3) ** 2 / 4
        T, X = np.meshgrid(tau, xi_cols, indexing="ij")
        return dict(Tc=float(Tc), tau=T.ravel(), xi=X.ravel(), A=A.ravel(), A_s=float(A.mean()), A_min=float(A.min()),
                    eval_tau=np.linspace(0, 1, 101)[:-1], eval_p=np.full(100, 100 * MMHG), beta_true=1.0, c0_true=1.0,
                    SBP=0.0, DBP=0.0, case=int(sid[1:]), D_mid=D[:, 4] * 1e-3, tau_mid=tau, cuff=CUFF, tag=tag)
    if MODE != "norm":
        jobs.append((sid, "B3ens", 0, case(tc, De, float(np.mean(per)), "ens")))
    # same ensemble beat, every site scaled to the common mean diameter (removes static taper / tracking offsets)
    Dn = De * (De.mean() / De.mean(0))[None, :]
    jobs.append((sid, "B3ensnorm", 0, case(tc, Dn, float(np.mean(per)), "ensnorm")))
    if MODE == "norm":
        continue
    for k, (a, b) in enumerate(zip(feet[:-1], feet[1:])):
        idx = np.arange(a, b + 1)
        tph = (t[idx] - t[a]) / (t[b] - t[a])          # resample every beat to the same 40 phases (fit_batch needs equal sizes)
        Db = np.stack([np.interp(tc, tph, Dc[idx, j]) for j in range(nC)], 1)
        jobs.append((sid, "B3beat", k, case(tc, Db, t[b] - t[a], f"beat{k}")))
    print(sid, json.dumps({k: v for k, v in stats[sid].items() if k != "feet_s"}), flush=True)

if MODE != "norm":
    json.dump(stats, open(os.path.join(OUT, "beats.json"), "w"), indent=1)
print(f"{len(jobs)} fits", flush=True)
t0 = time.time()
for i in range(0, len(jobs), 40):
    ch = jobs[i:i + 40]
    outs = fit_batch([j[3] for j in ch], ["c"] * len(ch), "B3", iters=8000, log_every=4000, seeds=[0] * len(ch), K=8, return_pred=True)
    with open(os.path.join(OUT, "real.jsonl"), "a") as fh:
        for (sid, model, k, d), o in zip(ch, outs):
            p = np.array(o["p_pred_mmHg"])
            fh.write(json.dumps(dict(subject=sid, level="c", model=model, seed=0, beat=k, iters=8000, cuff_assumed=[120.0, 80.0],
                                     beta=o["beta"], c0=o["c0"], A0=o["A0"], SBP=round(float(p.max()), 1), DBP=round(float(p.min()), 1),
                                     MAP=round(float(p.mean()), 1), PP=round(float(p.max() - p.min()), 1), loss=o["loss"],
                                     loss_data=o["loss_data"], p_pred_mmHg=o["p_pred_mmHg"])) + "\n")
    print(f"chunk {i // 40 + 1}: {len(ch)} fits, {(time.time() - t0) / 60:.1f} min", flush=True)
print("beats done")
