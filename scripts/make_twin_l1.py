"""Phase 2 / L1 twin dataset: N carotid cases over a healthy-to-stiff parameter range (Latin hypercube).

Per case (data/twin_l1/case_XXX.npz):
  params, metrics
  full last-cycle fields on a uniform grid: t (F), x (151), A, Q, p (F x 151)  [float32]
  synthetic scan: 5 s at 30 Hz, diameter at 9 columns over a 2 cm window around mid-vessel,
                  Gaussian noise sigma_D (0.03-0.05 mm, as measured in Phase 1), random start phase.
                  The pressure is stored for evaluation only, never as a training input.
Summary: reports/phase2/twin_l1_summary.json/.png
Usage: python make_twin_l1.py [N] [workers]
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from multiprocessing import Pool

R = os.environ.get("PULSE_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = os.path.join(R, "data", "twin_l1")
REP = os.path.join(R, "reports", "phase2")
MMHG = 133.322
# Independent LHS over model parameters, then ACCEPT only cases whose outputs are physiological.
# (Sampling output targets and calibrating per case fails: some PP targets are physically unreachable,
#  because the first wave alone gives PP ~ Z0 * flow pulse.) See NOTES 2026-11-13.
RANGES = dict(dist_ref=(0.04, 0.20), D0_mm=(5.5, 8.5), HR=(50.0, 100.0), Qmean_mLs=(4.0, 8.0), tau_s=(0.2, 0.7), MAP_mmHg=(80.0, 110.0))
ACCEPT = dict(PP=(25.0, 60.0), distension_pct=(3.0, 14.0))
FR, SCAN_S, SCAN_HZ = 400, 5.0, 30.0


def lhs(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for k, (lo, hi) in RANGES.items():
        u = (rng.permutation(n) + rng.random(n)) / n
        out.append(lo + u * (hi - lo))
    return [dict(zip(RANGES, row), case=i, seed=int(rng.integers(1 << 30))) for i, row in enumerate(np.array(out).T)]


def one(pr):
    from pulsens.carotid import run_case, beta_from
    from pulsens.l1 import Vessel
    t0 = time.time()
    Qm = pr["Qmean_mLs"] * 1e-6
    RT = pr["MAP_mmHg"] * MMHG / Qm
    A0, beta = beta_from(pr["D0_mm"], pr["dist_ref"])
    C = pr["tau_s"] / (RT - Vessel(0.15, A0, beta).Z0)
    n_cyc = int(np.clip(np.ceil(7 * pr["tau_s"] * pr["HR"] / 60) + 5, 8, 20))
    m = run_case(D0_mm=pr["D0_mm"], dist_ref=pr["dist_ref"], RT=RT, C=C, HR=pr["HR"], Qmean=Qm,
                 n_cycles=n_cyc, pc0_mmHg=pr["MAP_mmHg"], field_frames=FR)
    f = m.pop("_field")
    m["tau_s"] = float(m["R2"] * m["C"])
    m["accepted"] = bool(m["cycle_change"] < 0.01 and all(lo <= m[k] <= hi for k, (lo, hi) in ACCEPT.items()))
    if not m["accepted"]:
        print(f"case {pr['case']:03d} rejected PP {m['PP']:.1f} dist {m['distension_pct']:.1f}% cyc {m['cycle_change']:.4f}", flush=True)
        return dict(params=pr, metrics=m)
    # synthetic scan
    rng = np.random.default_rng(pr["seed"])
    Tc = 60.0 / pr["HR"]
    ts = np.arange(int(SCAN_S * SCAN_HZ)) / SCAN_HZ + rng.random() * Tc
    cols = np.linspace(0.065, 0.085, 9)
    ix = [int(round(c / (f["x"][1] - f["x"][0]))) for c in cols]
    D = np.sqrt(4 * f["A"][:, ix] / np.pi) * 1e3                      # mm, F x 9
    D_true = np.stack([np.interp(ts, f["t"], D[:, j], period=Tc) for j in range(len(ix))], axis=1)
    sig = rng.uniform(0.03, 0.05)
    D_scan = D_true + rng.normal(0, sig, D_true.shape)
    p_scan = np.stack([np.interp(ts, f["t"], f["p"][:, j] / MMHG, period=Tc) for j in ix], axis=1)
    m.update(n_cycles=n_cyc, sigma_D_mm=float(sig), secs=round(time.time() - t0, 1), tau_s=float(m["R2"] * m["C"]), MAP_target=pr["MAP_mmHg"])
    np.savez_compressed(os.path.join(DATA, f"case_{pr['case']:03d}.npz"),
                        t=f["t"].astype(np.float32), x=f["x"].astype(np.float32),
                        A=f["A"].astype(np.float32), Q=f["Q"].astype(np.float32), p=f["p"].astype(np.float32),
                        scan_t=ts.astype(np.float32), scan_x=cols.astype(np.float32), scan_D=D_scan.astype(np.float32),
                        scan_D_true=D_true.astype(np.float32), scan_p_eval_only=p_scan.astype(np.float32),
                        params=json.dumps(pr), metrics=json.dumps(m))
    print(f"case {pr['case']:03d} {m['SBP']:.0f}/{m['DBP']:.0f} dist {m['distension_pct']:.1f}% cyc {m['cycle_change']:.4f} {m['secs']} s", flush=True)
    return dict(params=pr, metrics=m)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(DATA, exist_ok=True); os.makedirs(REP, exist_ok=True)
    t0 = time.time()
    with Pool(W) as pool:
        rows = pool.map(one, lhs(N), chunksize=1)
    acc = [r for r in rows if r["metrics"]["accepted"]]
    ms = [r["metrics"] for r in acc]
    key = ["SBP", "DBP", "MAP", "PP", "distension_pct", "c0", "tau_s", "cycle_change"]
    stats = {k: dict(min=float(min(m[k] for m in ms)), median=float(np.median([m[k] for m in ms])), max=float(max(m[k] for m in ms))) for k in key}
    summ = dict(N_tried=N, N_accepted=len(acc), workers=W, minutes=round((time.time() - t0) / 60, 1), ranges=RANGES, accept=ACCEPT,
                stats_accepted=stats, cases=[dict(case=r["params"]["case"], accepted=r["metrics"]["accepted"], **{k: r["metrics"][k] for k in key}) for r in rows])
    json.dump(summ, open(os.path.join(REP, "twin_l1_summary.json"), "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    rej = [r["metrics"] for r in rows if not r["metrics"]["accepted"]]
    ax[0].scatter([m["distension_pct"] for m in rej], [m["PP"] for m in rej], c="#cccccc", s=10, label="rejected")
    sc = ax[0].scatter([m["distension_pct"] for m in ms], [m["PP"] for m in ms], c=[m["c0"] for m in ms], s=14, label="accepted")
    ax[0].set_xlabel("distension %"); ax[0].set_ylabel("PP mmHg"); ax[0].set_title("cases (colour = wave speed)"); ax[0].legend(); fig.colorbar(sc, ax=ax[0])
    ax[1].hist([m["SBP"] for m in ms], 20, alpha=.6, label="SBP"); ax[1].hist([m["DBP"] for m in ms], 20, alpha=.6, label="DBP"); ax[1].legend(); ax[1].set_title("mid pressure, accepted (mmHg)")
    for r in acc[:: max(1, len(acc) // 6)]:
        z = np.load(os.path.join(DATA, f"case_{r['params']['case']:03d}.npz"))
        ax[2].plot(z["scan_t"] - z["scan_t"][0], z["scan_D"][:, 4], ".-", ms=2, lw=.6)
    ax[2].set_title("synthetic 30 Hz scans (mid column)"); ax[2].set_xlabel("s"); ax[2].set_ylabel("D mm")
    fig.tight_layout(); fig.savefig(os.path.join(REP, "twin_l1_summary.png"), dpi=80)
    print("DONE", json.dumps(dict(minutes=summ["minutes"], tried=N, accepted=len(acc), stats=stats), indent=1))
