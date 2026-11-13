"""Phase 2 / L1 calibration: tune C (pulse pressure), RT (mean pressure) and wall stiffness (distension)
so the carotid L1 case matches healthy-adult targets from Phase 1 + literature.

Targets (mid-segment): PP 40 mmHg, MAP 93 mmHg (about 120/80), diameter distension 10% (Phase 1 median of good subjects).
Writes reports/phase2/l1_calibration.json and .png.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from pulsens.carotid import run_case, MMHG
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = os.environ.get("PULSE_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(R, "reports", "phase2"); os.makedirs(OUT, exist_ok=True)
T_PP, T_MAP, T_DIST = 40.0, 93.0, 10.0
Qmean = 6e-6

RT = 93 * MMHG / Qmean
C, dist_ref = None, 0.10
hist = []
for it in range(12):
    t0 = time.time()
    m = run_case(dist_ref=dist_ref, RT=RT, C=C, Qmean=Qmean)
    m["iter"], m["secs"] = it, round(time.time() - t0, 1)
    hist.append(m)
    print(f"it {it}: {m['SBP']:.1f}/{m['DBP']:.1f} MAP {m['MAP']:.1f} PP {m['PP']:.1f} dist {m['distension_pct']:.2f}% "
          f"C {m['C']:.3e} RT {m['RT']:.3e} dist_ref {dist_ref:.4f} cyc {m['cycle_change']:.4f} ({m['secs']} s)", flush=True)
    if abs(m["PP"] - T_PP) < 0.5 and abs(m["MAP"] - T_MAP) < 0.5 and abs(m["distension_pct"] - T_DIST) < 0.15:
        break
    C = m["C"] * (m["PP"] / T_PP)
    RT = m["RT"] * (T_MAP / m["MAP"])
    dist_ref = dist_ref * (T_DIST / m["distension_pct"])

final = run_case(dist_ref=dist_ref, RT=RT, C=C, Qmean=Qmean, n_cycles=10, record=True)
tr = final.pop("_trace")
final["tau_R2C_s"] = final["R2"] * final["C"]
res = dict(targets=dict(PP=T_PP, MAP=T_MAP, distension_pct=T_DIST), iterations=hist, calibrated=final,
           converged=bool(abs(final["PP"] - T_PP) < 1 and abs(final["MAP"] - T_MAP) < 1 and abs(final["distension_pct"] - T_DIST) < 0.3
                          and final["cycle_change"] < 0.01))
json.dump(res, open(os.path.join(OUT, "l1_calibration.json"), "w"), indent=1)
print(json.dumps(final, indent=1)); print("converged", res["converged"])

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].plot(tr["t"], tr["p_in"], label="inlet"); ax[0].plot(tr["t"], tr["p_mid"], label="mid"); ax[0].plot(tr["t"], tr["p_out"], label="outlet")
ax[0].set_title(f"pressure {final['SBP']:.0f}/{final['DBP']:.0f} mmHg"); ax[0].set_xlabel("s"); ax[0].legend()
ax[1].plot(tr["t"], tr["D_mid"]); ax[1].set_title(f"mid diameter, distension {final['distension_pct']:.1f}%"); ax[1].set_xlabel("s"); ax[1].set_ylabel("mm")
ax[2].plot(tr["t"], tr["Q_in"], label="inlet"); ax[2].plot(tr["t"], tr["Q_mid"], label="mid"); ax[2].set_title("flow (mL/s)"); ax[2].legend(); ax[2].set_xlabel("s")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "l1_calibration.png"), dpi=80)
