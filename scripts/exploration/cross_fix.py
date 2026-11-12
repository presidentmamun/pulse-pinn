import numpy as np, os, json
from scipy import ndimage as nd
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.environ.get("PULSE_ROOT", r"F:\my-project\pulse-pinn"); D = os.path.join(R, "data", "derived"); REP = os.path.join(R, "reports"); FR = 30.0
C = np.load(os.path.join(D, "5_cross_2cm_L_0000_bmode_0-149.npz")); cx, cz = C["xs"] * 1e3, C["zs"] * 1e3
db = nd.gaussian_filter(C["db"].astype(np.float32), (0, 3, 3)); m = db.mean(0)
# seed: darkest 6 mm disc in z 10-20, x -10..5
disc = nd.uniform_filter(m, (int(5 / (cz[1]-cz[0])), int(5 / (cx[1]-cx[0]))))
zi = np.where((cz > 10) & (cz < 20))[0]; xi = np.where((cx > -10) & (cx < 5))[0]
import sys; x0, z0 = float(sys.argv[1]), float(sys.argv[2])
th = np.linspace(0, 2 * np.pi, 72, endpoint=False); rr = np.arange(2.0, 5.5, 0.05)
A = np.zeros(150); cents = []; edges = []
for f in range(150):
    img = db[f]; rad = []
    for t_ in th:
        zz = z0 + rr * np.sin(t_); xx = x0 + rr * np.cos(t_)
        iz = (zz - cz[0]) / (cz[1] - cz[0]); ix = (xx - cx[0]) / (cx[1] - cx[0])
        pr = nd.map_coordinates(img, [iz, ix], order=1); g = np.gradient(pr)
        rad.append(rr[np.argmax(g)])
    rad = nd.median_filter(np.array(rad), 5, mode="wrap")
    xs_ = x0 + rad * np.cos(th); zs_ = z0 + rad * np.sin(th)
    A[f] = 0.5 * abs(np.dot(xs_, np.roll(zs_, 1)) - np.dot(zs_, np.roll(xs_, 1))); edges.append((xs_, zs_))
As = savgol_filter(A, 7, 2); Deq = 2 * np.sqrt(As / np.pi)
ts = np.load(os.path.join(D, "carotid_timeseries.npz")); Ds = ts["Ds"]; t = ts["t"]
def hr(x):
    x = x - x.mean(); ac = np.correlate(x, x, "full")[len(x)-1:]; lo = int(FR * 0.4); k = lo + np.argmax(ac[lo:int(FR * 1.6)]); return 60 * FR / k
r = np.corrcoef(Ds, Deq)[0, 1]
lag = int(np.argmax(np.correlate(Ds - Ds.mean(), Deq - Deq.mean(), "full")) - 149)
S = dict(seed_mm=[round(float(x0), 1), round(float(z0), 1)], A_mm2=[round(float(As.min()), 1), round(float(As.max()), 1)], Deq_mm=[round(float(Deq.min()), 2), round(float(Deq.max()), 2)],
         area_change_pct=round(float((As.max() - As.min()) / As.min() * 100), 1), HR_from_A=round(hr(As), 1), corr_long_vs_cross=round(float(r), 2), lag_frames=lag)
print(json.dumps(S)); json.dump(S, open(os.path.join(REP, "carotid_cross_summary.json"), "w"))
np.savez(os.path.join(D, "carotid_cross_ts.npz"), t=t, A=A, As=As, Deq=Deq)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].imshow(C["db"][0].astype(float), cmap="gray", vmin=-60, vmax=0, extent=[cx[0], cx[-1], cz[-1], cz[0]]); ax[0].plot(*edges[0], "y-"); ax[0].plot(*edges[75], "c-"); ax[0].set_title("cross: lumen frame 0 (y), 75 (c)")
ax[1].plot(t, Ds, "r", label="D long (mm)"); ax[1].plot(t, Deq, "b", label="D eq. from cross area (mm)"); ax[1].legend(); ax[1].set_xlabel("s")
fig.tight_layout(); fig.savefig(os.path.join(REP, "carotid_cross_fix.png"), dpi=80)
