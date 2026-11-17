"""Phase 6c: does real-data-like imperfection make the PINN under-read stiffness on the twin, where the truth is known?

Real carotid D(t) (phase6_beats.py) shows period CV ~13%, pulse-amplitude CV ~12% and a slow baseline drift of ~70% of
the pulse amplitude; the twin has none of this. Here the twin diameters are rebuilt with those imperfections and fitted with
B3 at level c (true cuff), first 40 cases, seed 0:
  clean   - the twin as it is (template replayed, same noise)
  drift   - + slow drift (0.25 Hz, peak-to-peak 70% of the pulse amplitude)
  beats   - + beat-to-beat period (CV 13%) and amplitude (CV 12%) variation
  both    - drift + beats
  fixed   - both, then the real-data preprocessing (drift removal at the feet + ensemble-averaged beat)
Output: reports/phase6/twin_realnoise.jsonl with beta_rel_err and c0_rel_err per case and condition.
"""
import os, sys, json, glob, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from pulsens.pinn import LX, MMHG
from pulsens.pinn_batch import fit_batch

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(R, "reports", "phase6", "twin_realnoise.jsonl")
NC = int(sys.argv[1]) if len(sys.argv) > 1 else 40
CONDS = (sys.argv[2] if len(sys.argv) > 2 else "clean,drift,beats,both,fixed,taper,taperfix").split(",")
TAPER = 0.025                               # static per-site diameter offsets, SD 2.5% (real data: ~3%)
PCV, ACV, DRIFT, FDRIFT = 0.13, 0.12, 0.70, 0.25


def template(z):
    """Noise-free periodic template per column, as a function of phase (200 bins)."""
    pr = json.loads(str(z["params"])); Tc = 60.0 / pr["HR"]
    ph = (z["scan_t"] % Tc) / Tc; Dt = z["scan_D_true"]
    g = np.linspace(0, 1, 201)[:-1]
    order = np.argsort(ph)
    return Tc, g, np.stack([np.interp(g, ph[order], Dt[order, j], period=1.0) for j in range(Dt.shape[1])], 1)


def synth(z, cond, rng):
    Tc, g, T = template(z)
    t = z["scan_t"] - z["scan_t"][0]
    noise = z["scan_D"] - z["scan_D_true"]
    if cond == "clean":
        return z["scan_D"].copy(), t, Tc
    if cond in ("taper", "taperfix"):          # only static site-to-site offsets, nothing else
        off = 1 + TAPER * rng.standard_normal(z["scan_D"].shape[1])
        return z["scan_D"] * off[None, :], t, Tc
    beats = cond in ("beats", "both", "fixed"); drift = cond in ("drift", "both", "fixed")
    # beat schedule
    per = Tc * (1 + (PCV * rng.standard_normal(20) if beats else np.zeros(20))); per = np.clip(per, 0.6 * Tc, 1.4 * Tc)
    amp = 1 + (ACV * rng.standard_normal(20) if beats else np.zeros(20))
    edges = np.concatenate([[-rng.uniform(0, Tc)], -rng.uniform(0, Tc) + np.cumsum(per)])
    k = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(per) - 1)
    phase = (t - edges[k]) / per[k]
    mean = T.mean(0)
    D = np.stack([mean[j] + amp[k] * (np.interp(phase, g, T[:, j], period=1.0) - mean[j]) for j in range(T.shape[1])], 1)
    if drift:
        pulse = (T.max(0) - T.min(0)).mean()
        D += (DRIFT * pulse / 2) * np.sin(2 * np.pi * FDRIFT * t + rng.uniform(0, 2 * np.pi))[:, None]
    return D + noise, t, float(np.mean(per))


def case_from(z, D, t, Tc, cond):
    pr, m = json.loads(str(z["params"])), json.loads(str(z["metrics"]))
    xi = (z["scan_x"] - 0.075) / LX
    if cond == "fixed":     # the real-data preprocessing: drift removal at the beat feet + ensemble beat
        Dm = D.mean(1); k = int(round(Tc / (t[1] - t[0]))); s = np.convolve(Dm, np.ones(3) / 3, "same")
        feet, i = [], int(np.argmin(s[:k]))
        while i < len(s):
            feet.append(i); lo, hi = i + int(.7 * k), min(len(s), i + int(1.3 * k) + 1)
            if lo >= len(s) - 1 or hi - lo < 2: break
            i = lo + int(np.argmin(s[lo:hi]))
        base = Dm[feet]; D = D - (np.interp(t, t[feet], base) - base.mean())[:, None]
        ph, rows = [], []
        for a, b in zip(feet[:-1], feet[1:]):
            idx = np.arange(a, b); ph.append((t[idx] - t[a]) / (t[b] - t[a])); rows.append(D[idx])
        ph = np.concatenate(ph); rows = np.concatenate(rows); nb = 40; bins = np.minimum((ph * nb).astype(int), nb - 1)
        De = np.stack([rows[bins == q].mean(0) if np.any(bins == q) else np.full(D.shape[1], np.nan) for q in range(nb)])
        ok = ~np.isnan(De[:, 0]); tau = (np.arange(nb) + .5) / nb
        D = np.stack([np.interp(tau, tau[ok], De[ok, j], period=1.0) for j in range(D.shape[1])], 1)
        Tc = float(np.mean(np.diff(t[feet])))
    else:
        tau = (t % Tc) / Tc
    if cond == "taperfix":                      # the fix: every site scaled to the common mean diameter
        D = D * (D.mean() / D.mean(0))[None, :]
    A = np.pi * (D * 1e-3) ** 2 / 4
    T, X = np.meshgrid(tau, xi, indexing="ij")
    mid = int(np.argmin(np.abs(z["x"] - 0.075))); p_mid = z["p"][:, mid]
    return dict(Tc=Tc, tau=T.ravel(), xi=X.ravel(), A=A.ravel(), A_s=float(A.mean()), A_min=float(A.min()),
                eval_tau=z["t"] / (60.0 / pr["HR"]), eval_p=p_mid, beta_true=m["beta"], c0_true=m["c0"], SBP=m["SBP"], DBP=m["DBP"],
                case=pr["case"], D_mid=D[:, 4] * 1e-3, tau_mid=tau, cuff=(float(p_mid.max()), float(p_mid.min())))


files = sorted(glob.glob(os.path.join(R, "data", "twin_l1", "case_*.npz")))[:NC]
done = set()
if os.path.exists(OUT):
    for s in open(OUT):
        r = json.loads(s); done.add((r["case"], r["cond"]))
jobs = []
for p in files:
    z = np.load(p); c = json.loads(str(z["params"]))["case"]
    for cond in CONDS:
        if (c, cond) in done: continue
        rng = np.random.default_rng(1000 + c)            # same imperfections for 'both' and 'fixed'
        D, t, Tc = synth(z, cond, rng)
        jobs.append((c, cond, case_from(z, D, t, Tc, cond)))
print(f"{len(jobs)} fits", flush=True)
t0 = time.time()
jobs.sort(key=lambda j: len(j[2]["tau"]))        # fit_batch needs the same number of data points within a chunk
chunks, cur = [], []
for j in jobs:
    if cur and (len(cur) == 40 or len(cur[0][2]["tau"]) != len(j[2]["tau"])):
        chunks.append(cur); cur = []
    cur.append(j)
if cur: chunks.append(cur)
for i, ch in enumerate(chunks):
    outs = fit_batch([j[2] for j in ch], ["c"] * len(ch), "B3", iters=8000, log_every=4000, seeds=[0] * len(ch), K=8)
    with open(OUT, "a") as fh:
        for (c, cond, d), o in zip(ch, outs):
            o.update(case=c, cond=cond, model="B3", level="c", seed=0); fh.write(json.dumps(o) + "\n")
    print(f"chunk {i + 1}: {len(ch)} fits, {(time.time() - t0) / 60:.1f} min", flush=True)
print("realnoise done")
