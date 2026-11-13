"""Carotid segment case for the L1 solver: synthetic inflow + RCR outlet, and summary metrics.

Parameters (all SI unless the name says otherwise):
    D0_mm    reference diameter at p_ref
    dist_ref fractional diameter change per 40 mmHg (sets beta: beta = 40 mmHg / (dist_ref * sqrt(A0)))
    RT       total peripheral resistance (Pa s/m^3); R1 = Z0, R2 = RT - R1
    C        Windkessel compliance (m^3/Pa)
    HR       heart rate (bpm)
    Qmean    mean inflow (m^3/s); waveform shape is fixed, scaled to this mean
"""
import numpy as np
from .l1 import Vessel, Solver

MMHG = 133.322


def inflow(HR, Qmean):
    Tc = 60.0 / HR
    tau = np.linspace(0, Tc, 4000, endpoint=False)
    Ts = 0.3 * Tc

    def shape(t):
        t = t % Tc
        s = 14.0 * np.sin(np.pi * t / Ts) ** 2 if t < Ts else 0.0
        return 3.3 + s + 1.0 * np.sin(2 * np.pi * t / Tc)
    scale = Qmean / np.mean([shape(t) for t in tau])
    return lambda t: scale * shape(t), Tc


def beta_from(D0_mm, dist_ref):
    A0 = np.pi * (D0_mm * 1e-3) ** 2 / 4
    return A0, 40 * MMHG / (dist_ref * np.sqrt(A0))


def run_case(D0_mm=6.5, dist_ref=0.10, RT=None, C=None, HR=60.0, Qmean=6e-6, L=0.15, N=151,
             p_ref_mmHg=80.0, n_cycles=8, record=False):
    A0, beta = beta_from(D0_mm, dist_ref)
    v = Vessel(L, A0, beta, p_ref=p_ref_mmHg * MMHG)
    q_in, Tc = inflow(HR, Qmean)
    RT = RT if RT is not None else 93 * MMHG / Qmean
    R1 = v.Z0
    R2 = RT - R1
    C = C if C is not None else 0.8 / R2
    s = Solver(v, N, q_in, dict(R1=R1, C=C, R2=R2, p_out=0.0, pc0=90 * MMHG), cfl=0.5)
    r = s.run(n_cycles * Tc, probes=(0.0, L / 2, L))
    t = r["t"]
    last = t > (n_cycles - 1) * Tc
    prev = (t > (n_cycles - 2) * Tc) & ~last
    pm = r["p"][:, 1] / MMHG
    pl, pp = pm[last], np.interp(t[last] - Tc, t[prev], pm[prev])
    D = np.sqrt(4 * r["A"][last, 1] / np.pi) * 1e3
    out = dict(SBP=float(pl.max()), DBP=float(pl.min()), MAP=float(pl.mean()), PP=float(pl.max() - pl.min()),
               D_min_mm=float(D.min()), D_max_mm=float(D.max()), distension_pct=float((D.max() - D.min()) / D.min() * 100),
               cycle_change=float(np.max(np.abs(pl - pp)) / max(pl.max() - pl.min(), 1e-9)),
               beta=float(beta), Z0=float(R1), R1=float(R1), R2=float(R2), RT=float(RT), C=float(C), c0=float(v.c0),
               HR=HR, Qmean=Qmean, D0_mm=D0_mm, dist_ref=dist_ref)
    if record:
        out["_trace"] = dict(t=t[last] - t[last][0], p_in=r["p"][last, 0] / MMHG, p_mid=pl, p_out=r["p"][last, 2] / MMHG,
                             D_mid=D, Q_in=r["Q"][last, 0] * 1e6, Q_mid=r["Q"][last, 1] * 1e6)
    return out
