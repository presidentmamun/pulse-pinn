"""Phase 2 / L1 verification gates. Writes reports/phase2/l1_verify.json and l1_verify.png.
Tests:
  1 MMS convergence (manufactured solution, observed order ~2)
  2 wave speed vs theory c0 = sqrt(beta sqrt(A0) / (2 rho))
  3 reflection coefficient at a resistive outlet vs linear theory (R - Z0)/(R + Z0)
  4 mass conservation: d(volume) vs integral of (Q_in - Q_out)
  5 physiological carotid case: periodic state, pressure range
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pulsens.l1 import Vessel, Solver
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = os.environ.get("PULSE_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(R, "reports", "phase2"); os.makedirs(OUT, exist_ok=True)
MMHG = 133.322
res, fig_data = {}, {}

# carotid-like reference (from Phase 1: D ~6.5 mm, distension ~10% for PP ~40 mmHg)
D0 = 6.5e-3; A0 = np.pi * D0 ** 2 / 4
beta = 40 * MMHG / (0.10 * np.sqrt(A0))

# ---------- 1 MMS convergence ----------
L, T = 0.15, 0.2
def exact(x, t):
    A = A0 * (1 + 0.05 * np.sin(2 * np.pi * x / L) * np.cos(2 * np.pi * t / T))
    Q = 5e-6 * (1 + 0.3 * np.cos(2 * np.pi * x / L) * np.sin(2 * np.pi * t / T))
    return A, Q
vm = Vessel(L, A0, beta)
def mms_source(x, t):  # S = dU/dt + dF/dx - S_phys, by central differences
    ht, hx = 1e-7 * T, 1e-7 * L
    Ap, Qp = exact(x, t + ht); Am, Qm = exact(x, t - ht)
    A, Q = exact(x, t)
    F0p, F1p = vm.flux(*exact(x + hx, t)); F0m, F1m = vm.flux(*exact(x - hx, t))
    s0, s1 = vm.source(A, Q)
    return ((Ap - Am) / (2 * ht) + (F0p - F0m) / (2 * hx) - s0,
            (Qp - Qm) / (2 * ht) + (F1p - F1m) / (2 * hx) - s1)
errs, Ns = [], [50, 100, 200, 400]
for N in Ns:
    x = np.linspace(0, L, N); Ai, Qi = exact(x, 0.0)
    s = Solver(vm, N, None, dict(R1=1.0), cfl=0.4, A_init=Ai, Q_init=Qi, mms_source=mms_source, exact=exact)
    dt0 = 0.4 * (L / (N - 1)) / (vm.c(A0 * 1.06) + 1.0)
    s.run(T, dt=dt0)
    Ae, _ = exact(s.x, s.t)
    errs.append(float(np.sqrt(np.mean((s.A - Ae) ** 2)) / A0))
orders = [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]
res["mms"] = dict(N=Ns, rel_L2_err_A=errs, observed_order=orders, pass_=bool(orders[-1] > 1.8))

# ---------- 2 + 3 wave speed and reflection (inviscid, small pulse, long tube) ----------
Lw = 2.0; vw = Vessel(Lw, A0, beta, mu=0.0)
q_amp, tp, sig = 2e-6, 0.05, 0.01
pulse = lambda t: q_amp * np.exp(-((t - tp) / sig) ** 2)
def wave_run(Rfac):
    s = Solver(vw, 801, pulse, dict(R1=Rfac * vw.Z0, C=0.0, p_out=vw.p_ref), cfl=0.5)
    return s.run(1.0, probes=(0.5, 1.5))
r1 = wave_run(1.0)
t, p = r1["t"], r1["p"] - vw.p_ref
i1, i2 = np.argmax(p[:, 0]), np.argmax(p[:, 1])
c_meas = 1.0 / (t[i2] - t[i1])
res["wave_speed"] = dict(c_theory=float(vw.c0), c_measured=float(c_meas), rel_err=float(abs(c_meas - vw.c0) / vw.c0),
                         pass_=bool(abs(c_meas - vw.c0) / vw.c0 < 0.02))
refl = {}
for Rf in (1.0, 3.0, 1 / 3):
    r = wave_run(Rf); pp = r["p"][:, 1] - vw.p_ref; tt = r["t"]
    t_inc = 1.5 / vw.c0 + tp; t_ref = (2 * Lw - 1.5) / vw.c0 + tp
    inc = pp[np.abs(tt - t_inc) < 0.03]; rf = pp[np.abs(tt - t_ref) < 0.03]
    G_meas = float((rf.max() if abs(rf.max()) > abs(rf.min()) else rf.min()) / inc.max())
    refl[f"R={Rf:.2f}Z0"] = dict(theory=float((Rf - 1) / (Rf + 1)), measured=G_meas)
    if Rf == 3.0: fig_data["refl"] = (tt, pp)
res["reflection"] = dict(cases=refl, pass_=bool(all(abs(v["measured"] - v["theory"]) < 0.03 for v in refl.values())))
fig_data["wave"] = (t, p)

# ---------- 4 + 5 physiological carotid with RCR outlet ----------
HR = 60.0; Tc = 60.0 / HR
def q_in(t):  # synthetic carotid inflow: mean ~6 mL/s, systolic peak ~20 mL/s
    tau = t % Tc; Ts = 0.3 * Tc
    sys_ = 14e-6 * np.sin(np.pi * tau / Ts) ** 2 if tau < Ts else 0.0
    return 3.3e-6 + sys_ + 1.0e-6 * np.sin(2 * np.pi * tau / Tc)
vp = Vessel(0.15, A0, beta, p_ref=80 * MMHG)
Qm = np.mean([q_in(tt) for tt in np.linspace(0, Tc, 2000)])
RT = 93 * MMHG / Qm; R1 = vp.Z0; R2 = RT - R1; C = 0.8 / R2   # tau = R2*C = 0.8 s
s = Solver(vp, 151, q_in, dict(R1=R1, C=C, R2=R2, p_out=0.0, pc0=85 * MMHG), cfl=0.5)
r = s.run(10 * Tc, probes=(0.0, 0.075, 0.15))
t = r["t"]; pm = r["p"][:, 1] / MMHG
last = t > 9 * Tc; prev = (t > 8 * Tc) & (t <= 9 * Tc)
p_last, p_prev = pm[last], np.interp(t[last] - Tc, t[prev], pm[prev])
cyc = float(np.max(np.abs(p_last - p_prev)) / (p_last.max() - p_last.min()))
dV = r["vol"][-1] - r["vol"][0]
flux = float(np.trapezoid(r["qin"] - r["qout"], t))
SV = float(np.trapezoid(r["qin"][last], t[last]))
Dm = np.sqrt(4 * r["A"][last, 1] / np.pi)
res["mass"] = dict(dVolume=float(dV), integral_Qin_minus_Qout=flux, err_rel_to_stroke_volume=float(abs(dV - flux) / SV),
                   pass_=bool(abs(dV - flux) / SV < 1e-3))
res["physiological"] = dict(HR=HR, Z0=float(vp.Z0), R1=float(R1), R2=float(R2), C=float(C), c0=float(vp.c0),
                            SBP=float(p_last.max()), DBP=float(p_last.min()), MAP=float(p_last.mean()),
                            D_min_mm=float(Dm.min() * 1e3), D_max_mm=float(Dm.max() * 1e3),
                            distension_pct=float((Dm.max() - Dm.min()) / Dm.min() * 100),
                            cycle_to_cycle_change=cyc, pass_=bool(cyc < 0.01))
res["all_pass"] = bool(all(v["pass_"] for k, v in res.items() if isinstance(v, dict) and "pass_" in v))

json.dump(res, open(os.path.join(OUT, "l1_verify.json"), "w"), indent=1)
print(json.dumps(res, indent=1))

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].loglog(Ns, errs, "o-"); ax[0, 0].loglog(Ns, errs[0] * (Ns[0] / np.array(Ns)) ** 2, "k--", label="order 2")
ax[0, 0].set_title(f"MMS: observed order {orders[-1]:.2f}"); ax[0, 0].set_xlabel("N"); ax[0, 0].legend()
tw, pw = fig_data["wave"]; ax[0, 1].plot(tw, pw[:, 0], label="x=0.5 m"); ax[0, 1].plot(tw, pw[:, 1], label="x=1.5 m")
tr, pr = fig_data["refl"]; ax[0, 1].plot(tr, pr, ":", label="x=1.5 m, R=3Z0")
ax[0, 1].set_title(f"wave speed {c_meas:.3f} vs {vw.c0:.3f} m/s"); ax[0, 1].legend(); ax[0, 1].set_xlabel("s")
ax[1, 0].plot(t, r["p"][:, 0] / MMHG, label="inlet"); ax[1, 0].plot(t, pm, label="mid"); ax[1, 0].plot(t, r["p"][:, 2] / MMHG, label="outlet")
ax[1, 0].set_title("carotid pressure (mmHg)"); ax[1, 0].legend(); ax[1, 0].set_xlabel("s")
ax[1, 1].plot(t[last] - 9 * Tc, Dm * 1e3); ax[1, 1].set_title(f"mid diameter (mm), distension {res['physiological']['distension_pct']:.1f}%")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "l1_verify.png"), dpi=80)
