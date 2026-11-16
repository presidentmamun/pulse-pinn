"""Phase 6: the twin-validated PINN applied to real carotid ultrasound (zea-carotid-2023, longitudinal views).

Input: reports/phase1/sNN_long_ts.npz from track_long.py (30 Hz, 150 frames, near/far wall per image column).
Each subject becomes a case laid out like a twin case: 9 columns 2.5 mm apart over the central 20 mm, D(t) per column.
Levels: a (diameter only) and c (diameter + an ASSUMED population cuff of 120/80 mmHg, labelled as an assumption,
because the dataset has no pressure). No ground truth exists, so the outputs are the estimates themselves
(beta, c0 = local PWV, pressure waveform) plus plausibility checks.

Usage: python scripts/phase6_real.py [model=B3|CPCv2] [seeds=0,1,2] [subjects=all] [iters=8000] [cuff=120/80]
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from pulsens.pinn import LX, MMHG
from pulsens.pinn_batch import fit_batch

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P1 = os.path.join(R, "reports", "phase1")
OUT = os.path.join(R, "reports", "phase6"); os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "real.jsonl")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "B3"
SEEDS = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2").split(",")]
GOOD = ["s01", "s02", "s04", "s05", "s06", "s07", "s08", "s10"]          # s03, s09 excluded in Phase 1 (tilted vessel)
SUBJ = (sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] != "all" else GOOD)
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
CUFF = tuple(float(v) for v in (sys.argv[5] if len(sys.argv) > 5 else "120/80").split("/"))   # ASSUMED cuff, mmHg
# model label -> (kind, harmonics, hold-out column, data-loss floor for levels a/b) — same settings as the twin runs
SPEC = {"B3": ("B3", 8, None, 0.0), "CPCv2": ("CPC", 4, 6, 0.8)}
KIND, NH, HOLD, FLOOR = SPEC[MODEL]


def period(t, D, hr_hint):
    """Cardiac period from the autocorrelation of the detrended mean diameter (parabolic sub-frame refinement)."""
    x = D - np.polyval(np.polyfit(t, D, 2), t)
    x = (x - x.mean()) / (x.std() + 1e-12)
    dt = t[1] - t[0]
    ac = np.correlate(x, x, "full")[len(x) - 1:] / np.arange(len(x), 0, -1)
    lo, hi = int(60 / 140 / dt), int(60 / 40 / dt)
    k = lo + int(np.argmax(ac[lo:hi]))
    if 0 < k < len(ac) - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    T = k * dt
    if hr_hint and abs(60 / T - hr_hint) > 15:                          # trust the QC'd Phase 1 HR if they disagree
        T = 60.0 / hr_hint
    return float(T), float(ac[int(round(k))])


def build(sid):
    z = np.load(os.path.join(P1, f"{sid}_long_ts.npz"))
    js = json.load(open(os.path.join(P1, f"{sid}_long.json")))
    t, xs = z["t"], z["xs_cols"]
    Dc = z["far"] - z["near"]                                           # mm, frames x columns
    x0 = float(np.median(xs))
    targets = x0 + np.arange(-10, 10.01, 2.5)
    cols = [int(np.argmin(np.abs(xs - v))) for v in targets]
    Dsel = np.stack([np.nanmedian(Dc[:, max(0, c - 2):c + 3], 1) for c in cols], 1)   # 5-column median per site
    for j in range(Dsel.shape[1]):                                      # fill rare gaps
        bad = ~np.isfinite(Dsel[:, j])
        if bad.any():
            Dsel[bad, j] = np.interp(t[bad], t[~bad], Dsel[~bad, j])
    Tc, acpk = period(t, Dsel.mean(1), js.get("HR"))
    tau = (t % Tc) / Tc
    xi = (xs[cols] - x0) / (LX * 1e3)
    A = np.pi * (Dsel * 1e-3) ** 2 / 4
    T, X = np.meshgrid(tau, xi, indexing="ij")
    d = dict(Tc=Tc, tau=T.ravel(), xi=X.ravel(), A=A.ravel(), A_s=float(A.mean()), A_min=float(A.min()),
             eval_tau=np.linspace(0, 1, 101)[:-1], eval_p=np.full(100, 100 * MMHG), beta_true=1.0, c0_true=1.0,
             SBP=0.0, DBP=0.0, case=int(sid[1:]), D_mid=Dsel[:, 4] * 1e-3, tau_mid=tau)
    info = dict(subject=sid, HR_fit=round(60 / Tc, 1), HR_phase1=js.get("HR"), autocorr=round(acpk, 2),
                D_mean_mm=round(float(Dsel.mean()), 3), distension_pct=round(float((np.percentile(Dsel.mean(1), 95) / np.percentile(Dsel.mean(1), 5) - 1) * 100), 1),
                span_mm=round(float(xs[cols][-1] - xs[cols][0]), 1))
    return d, info


done = set()
if os.path.exists(LOG):
    for s in open(LOG):
        r = json.loads(s); done.add((r["subject"], r["level"], r["model"], r["seed"], tuple(r["cuff_assumed"] or CUFF)))
jobs = []
for sid in SUBJ:
    d0, info = build(sid)
    print(json.dumps(info), flush=True)
    for lv in "ac":
        for sd in SEEDS:
            if (sid, lv, MODEL, sd, CUFF) in done or (lv == "a" and CUFF != (120.0, 80.0)):   # level a ignores the cuff
                continue
            d = dict(d0)
            if lv == "c":
                d["cuff"] = (CUFF[0] * MMHG, CUFF[1] * MMHG)
            jobs.append((sid, lv, sd, d, info))
print(f"{len(jobs)} fits to run", flush=True)
t0 = time.time()
for i in range(0, len(jobs), 40):
    ch = jobs[i:i + 40]
    outs = fit_batch([j[3] for j in ch], [j[1] for j in ch], KIND, iters=ITERS, log_every=2000, seeds=[j[2] for j in ch], K=NH,
                     hold_col=HOLD, d_floor=[FLOOR if j[1] in "ab" else 0.0 for j in ch], return_pred=True)
    with open(LOG, "a") as fh:
        for (sid, lv, sd, d, info), o in zip(ch, outs):
            p = np.array(o["p_pred_mmHg"])
            row = dict(subject=sid, level=lv, model=MODEL, seed=sd, iters=ITERS, cuff_assumed=list(CUFF) if lv == "c" else None,
                       beta=o["beta"], c0=o["c0"], A0=o["A0"], SBP=round(float(p.max()), 1), DBP=round(float(p.min()), 1),
                       MAP=round(float(p.mean()), 1), PP=round(float(p.max() - p.min()), 1), loss=o["loss"], loss_data=o["loss_data"],
                       p_pred_mmHg=o["p_pred_mmHg"], **{k: v for k, v in info.items() if k != "subject"})
            fh.write(json.dumps(row) + "\n")
    print(f"chunk {i // 40 + 1}: {len(ch)} fits, {(time.time() - t0) / 60:.1f} min", flush=True)
print("phase6 done")
