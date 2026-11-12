"""Plane-wave DAS beamforming of zea carotid RF -> B-mode stack, M-mode, wall/area tracking.
usage: python carotid_bf.py <file.hdf5> <frames: e.g. 0 or 0-149> [workers]"""
import sys, os, json, time, numpy as np, h5py, hdf5plugin
from scipy.signal import hilbert
R = os.environ.get("PULSE_ROOT", r"F:\my-project\pulse-pinn"); REP = os.path.join(R, "reports"); OUT = os.path.join(R, "data", "derived")
os.makedirs(OUT, exist_ok=True)
FN = sys.argv[1]; FR = sys.argv[2]; NW = int(sys.argv[3]) if len(sys.argv) > 3 else 6
frames = list(range(int(FR.split("-")[0]), int(FR.split("-")[1]) + 1)) if "-" in FR else [int(FR)]
XS = np.arange(-19e-3, 19e-3, 0.15e-3); ZS = np.arange(1e-3, 39e-3, 0.075e-3)

def find(f, name):
    hits = []
    f.visit(lambda n: hits.append(n) if n.split("/")[-1] == name else None)
    return hits[0] if hits else None

def params(f):
    g = lambda n: np.asarray(f[find(f, n)][()])
    P = {k: g(k) for k in ["t0_delays", "polar_angles", "focus_distances", "initial_times", "time_to_next_transmit", "probe_geometry"]}
    for k in ["sampling_frequency", "center_frequency", "sound_speed"]:
        P[k] = float(np.ravel(g(k))[0])
    P["raw"] = find(f, "raw_data")
    return P

def beamform(f, P, fi):
    pw = np.where(P["focus_distances"] == 0)[0]
    rf = np.asarray(f[P["raw"]][fi, pw, :, :, 0], dtype=np.float32)      # (ntx, nax, nel)
    iq = hilbert(rf, axis=1).astype(np.complex64)
    c, fs = P["sound_speed"], P["sampling_frequency"]; ex = P["probe_geometry"][:, 0]
    X, Z = np.meshgrid(XS, ZS); ens = np.zeros((len(pw),) + X.shape, np.complex64)
    rx = np.sqrt((X[..., None] - ex) ** 2 + Z[..., None] ** 2) / c                 # (nz,nx,nel)
    apo = (np.abs(X[..., None] - ex) <= Z[..., None] / (2 * 1.5)).astype(np.float32)  # F#1.5
    nax = rf.shape[1]
    for j, t in enumerate(pw):
        tx = np.min(P["t0_delays"][t][None, None, :] + rx, axis=2)              # Huygens envelope
        idx = (tx[..., None] + rx - P["initial_times"][t]) * fs
        i0 = np.clip(idx.astype(np.int32), 0, nax - 2); w = idx - i0
        ok = (idx >= 0) & (idx < nax - 1)
        el = np.arange(rf.shape[2])[None, None, :]
        s = iq[j][i0, el] * (1 - w) + iq[j][i0 + 1, el] * w
        ens[j] = np.sum(s * apo * ok, axis=2)
    img = ens.sum(0)
    # Doppler across the 21-PW ensemble (PRF = 1/pw interval); crude wall filter = remove ensemble mean
    order = np.argsort(np.arange(len(pw)))  # acquisition order
    e = ens[order] - ens.mean(0, keepdims=True)
    r1 = np.sum(np.conj(e[:-1]) * e[1:], axis=0)
    prf = 1.0 / float(np.median(P["time_to_next_transmit"][fi, pw[:-1]]))
    vax = (c * prf / (4 * np.pi * P["center_frequency"])) * np.angle(r1)
    pdop = np.mean(np.abs(e) ** 2, axis=0)
    return img, vax.astype(np.float32), pdop.astype(np.float32)

def work(fi):
    with h5py.File(FN, "r") as f:
        P = params(f); return fi, beamform(f, P, fi)

if __name__ == "__main__":
    t0 = time.time(); base = os.path.basename(FN).replace(".hdf5", "")
    with h5py.File(FN, "r") as f:
        P = params(f)
    ttn = P["time_to_next_transmit"]; per = ttn.sum(axis=1)
    pw = np.where(P["focus_distances"] == 0)[0]
    info = dict(file=base, frame_period_ms=round(float(np.median(per)) * 1e3, 3), frame_rate_hz=round(1 / float(np.median(per)), 2),
                duration_s=round(float(per.sum()), 3), n_pw=int(len(pw)), pw_angles_deg=[round(float(a), 2) for a in np.degrees(P["polar_angles"][pw])],
                pw_interval_us=round(float(np.median(ttn[0, pw])) * 1e6, 1), fs=P["sampling_frequency"], fc=P["center_frequency"], c=P["sound_speed"])
    print(json.dumps(info), flush=True)
    if NW > 1 and len(frames) > 1:
        from multiprocessing import Pool
        with Pool(NW) as pool: res = dict(pool.map(work, frames))
    else:
        res = dict(work(fi) for fi in frames)
    stack = np.stack([np.abs(res[fi][0]) for fi in frames]).astype(np.float32)
    vax = np.stack([res[fi][1] for fi in frames]); pdop = np.stack([res[fi][2] for fi in frames])
    db = 20 * np.log10(stack / stack.max() + 1e-6)
    np.savez_compressed(os.path.join(OUT, f"{base}_bmode_{FR}.npz"), db=db.astype(np.float16), vax=vax.astype(np.float16), pdop=pdop, xs=XS, zs=ZS, frames=np.array(frames), info=json.dumps(info))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2 if len(frames) > 1 else 1, figsize=(12 if len(frames) > 1 else 6, 5), squeeze=False)
    ax[0, 0].imshow(db[0], cmap="gray", vmin=-60, vmax=0, extent=[XS[0]*1e3, XS[-1]*1e3, ZS[-1]*1e3, ZS[0]*1e3]); ax[0, 0].set_title(f"{base} frame {frames[0]}"); ax[0, 0].set_xlabel("x mm"); ax[0, 0].set_ylabel("z mm")
    if len(frames) > 1:
        cx = len(XS) // 2; mm = db[:, :, cx - 10:cx + 10].mean(axis=2).T
        tt = np.cumsum(per)[frames] - per[0]
        ax[0, 1].imshow(mm, cmap="gray", vmin=-60, vmax=0, aspect="auto", extent=[tt[0], tt[-1], ZS[-1]*1e3, ZS[0]*1e3]); ax[0, 1].set_title("M-mode (centre, 3 mm wide)"); ax[0, 1].set_xlabel("s")
    fig.tight_layout(); fig.savefig(os.path.join(REP, f"{base}_bf_{FR}.png"), dpi=90)
    print("done", base, FR, round(time.time() - t0, 1), "s", flush=True)
