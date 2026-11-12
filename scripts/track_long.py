"""Longitudinal carotid wall tracking -> D(t), distension, HR, QC figure.
usage: python track_long.py <bmode npz> <subject label>"""
import sys, os, json, numpy as np
from scipy import ndimage as nd
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.environ.get("PULSE_ROOT", r"F:\my-project\pulse-pinn"); REP = os.path.join(R, "reports", "phase1"); os.makedirs(REP, exist_ok=True)
L = np.load(sys.argv[1]); lab = sys.argv[2]; FR = 30.0
xs, zs = L["xs"] * 1e3, L["zs"] * 1e3; dz = zs[1] - zs[0]; n = L["db"].shape[0]; t = np.arange(n) / FR
db = nd.gaussian_filter(L["db"].astype(np.float32), (0, 2, 2)); mean = db.mean(0)
cols = np.where((xs > -12) & (xs < 12))[0]; zr = (zs > 5) & (zs < 30)
prof = nd.gaussian_filter1d(mean, 12, axis=0)
# lumen = dark band with bright walls both above and below (within 2-6 mm)
nz = len(zs); a1, a2 = int(2 / dz), int(6 / dz)
def score(col):
    sc = np.full(nz, -np.inf)
    for i in np.where(zr)[0]:
        if i - a2 < 0 or i + a2 >= nz: continue
        up = col[i - a2:i - a1].max(); dn = col[i + a1:i + a2].max(); sc[i] = min(up, dn) - col[i]
    return sc
zc = np.array([zs[np.argmax(score(prof[:, c]))] for c in cols])
zc = nd.median_filter(zc, 15)
p = np.polyfit(xs[cols], zc, 1); zcf = np.polyval(p, xs[cols]); ang = float(np.degrees(np.arctan(p[0])))
near = np.full((n, len(cols)), np.nan); far = near.copy(); w = int(8 / dz)
for k, c in enumerate(cols):
    i0 = np.searchsorted(zs, zcf[k]); up = slice(max(0, i0 - w), i0); dn = slice(i0, min(len(zs), i0 + w))
    g = np.gradient(db[:, :, c], axis=1)
    near[:, k] = zs[up][np.argmin(g[:, up], axis=1)]; far[:, k] = zs[dn][np.argmax(g[:, dn], axis=1)]
dperp = (far - near) * np.cos(np.radians(ang)); Dt = np.nanmedian(dperp, axis=1); Ds = savgol_filter(Dt, 7, 2)
resid = float(np.std(Dt - Ds)); spread = float(np.nanmedian(np.nanstd(dperp, axis=1)))
x = Ds - Ds.mean(); ac = np.correlate(x, x, "full")[n - 1:]; lo = int(FR * 0.4); kk = lo + int(np.argmax(ac[lo:int(FR * 1.6)])); hr = 60 * FR / kk
S = dict(subject=lab, file=os.path.basename(sys.argv[1]), vessel_angle_deg=round(ang, 1), lumen_centre_mm=round(float(zc.mean()), 1),
         D_min=round(float(Ds.min()), 2), D_max=round(float(Ds.max()), 2), D_mean=round(float(Ds.mean()), 2),
         distension_pct=round(float((Ds.max() - Ds.min()) / Ds.min() * 100), 1), HR=round(hr, 1),
         frame_noise_mm=round(resid, 3), column_spread_mm=round(spread, 3), autocorr_peak=round(float(ac[kk] / ac[0]), 2))
json.dump(S, open(os.path.join(REP, f"{lab}_long.json"), "w"), indent=1); print(json.dumps(S))
np.savez(os.path.join(REP, f"{lab}_long_ts.npz"), t=t, D=Dt, Ds=Ds, near=near, far=far, xs_cols=xs[cols])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5)); ex = [xs[0], xs[-1], zs[-1], zs[0]]
ax[0].imshow(L["db"][0].astype(float), cmap="gray", vmin=-60, vmax=0, extent=ex); ax[0].plot(xs[cols], near[0], "c.", ms=1); ax[0].plot(xs[cols], far[0], "y.", ms=1); ax[0].set_title(f"{lab} frame 0 (angle {ang:.1f})")
ax[1].plot(t, Dt, ".", color="0.6"); ax[1].plot(t, Ds, "r"); ax[1].set_title(f"D(t) mm, HR {hr:.0f}, dist {S['distension_pct']}%"); ax[1].set_xlabel("s")
fig.tight_layout(); fig.savefig(os.path.join(REP, f"{lab}_long.png"), dpi=80)
