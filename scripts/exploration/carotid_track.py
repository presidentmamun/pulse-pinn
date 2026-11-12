"""Wall/diameter + Doppler velocity (longitudinal) and lumen area (cross) from beamformed npz."""
import numpy as np, os, json
from scipy import ndimage as nd
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.environ.get("PULSE_ROOT", r"F:\my-project\pulse-pinn"); D = os.path.join(R, "data", "derived"); REP = os.path.join(R, "reports")
FR = 30.0
L = np.load(os.path.join(D, "5_long_2cm_L_0000_bmode_0-149.npz")); C = np.load(os.path.join(D, "5_cross_2cm_L_0000_bmode_0-149.npz"))
xs, zs = L["xs"] * 1e3, L["zs"] * 1e3; dz = zs[1] - zs[0]; t = np.arange(150) / FR
db = nd.gaussian_filter(L["db"].astype(np.float32), (0, 2, 2)); vax = L["vax"].astype(np.float32); pd = L["pdop"]
mean = db.mean(0)
# lumen centre line: darkest smoothed band in z 7..24 mm, per column
cols = np.where((xs > -12) & (xs < 12))[0]; zr = (zs > 7) & (zs < 24)
prof = nd.gaussian_filter1d(mean, 12, axis=0)
zc = np.array([zs[zr][np.argmin(prof[zr, c])] for c in cols])
p = np.polyfit(xs[cols], zc, 1); zc_fit = np.polyval(p, xs[cols]); ang = np.degrees(np.arctan(p[0]))
near = np.full((150, len(cols)), np.nan); far = near.copy()
for k, c in enumerate(cols):
    i0 = np.searchsorted(zs, zc_fit[k])
    for f in range(150):
        g = np.gradient(db[f, :, c])
        up = slice(max(0, i0 - int(8 / dz)), i0); dn = slice(i0, min(len(zs), i0 + int(8 / dz)))
        near[f, k] = zs[up][np.argmin(g[up])]   # intensity falls entering lumen from above
        far[f, k] = zs[dn][np.argmax(g[dn])]    # rises leaving lumen below
dperp = (far - near) * np.cos(np.radians(ang))
Dt = np.nanmedian(dperp, axis=1); Ds = savgol_filter(Dt, 7, 2)
# Doppler: axial velocity in the central 50 % of the lumen
vm = np.zeros(150); pw = np.zeros(150)
for f in range(150):
    vals = []; pv = []
    for k, c in enumerate(cols):
        a, b = near[f, k], far[f, k]; m = (a + b) / 2; h = (b - a) * 0.25
        sel = (zs > m - h) & (zs < m + h); vals.append(vax[f, sel, c]); pv.append(pd[f, sel, c])
    vals = np.concatenate(vals); vm[f] = np.median(vals); pw[f] = np.median(np.concatenate(pv))
vax_frac_alias = float(np.mean(np.abs(vax[:, :, cols]) > 0.5))
s = np.sin(np.radians(abs(ang))) if abs(ang) > 1 else np.nan
vvessel = vm / s
# cross section: lumen area via region growing from darkest blob
cdb = nd.gaussian_filter(C["db"].astype(np.float32), (0, 3, 3)); cm = cdb.mean(0); cx, cz = C["xs"] * 1e3, C["zs"] * 1e3
roi = np.ix_((cz > 8) & (cz < 24), (cx > -14) & (cx < 8))
sub = nd.gaussian_filter(cm, 15)[roi]; iz, ix = np.unravel_index(np.argmin(sub), sub.shape)
seed = (np.where((cz > 8) & (cz < 24))[0][iz], np.where((cx > -14) & (cx < 8))[0][ix])
pa = (cx[1] - cx[0]) * (cz[1] - cz[0]); A = np.zeros(150); masks = []
for f in range(150):
    img = cdb[f]; ring = img[max(0, seed[0]-140):seed[0]+140, max(0, seed[1]-70):seed[1]+70]
    thr = np.percentile(ring, 45)
    lab, _ = nd.label(img < thr); m = nd.binary_fill_holes(lab == lab[seed]); m = nd.binary_opening(m, iterations=2)
    lab2, _ = nd.label(m); m = lab2 == lab2[seed] if lab2[seed] else m
    A[f] = m.sum() * pa; masks.append(m)
As = savgol_filter(A, 7, 2); Deq = 2 * np.sqrt(As / np.pi)
def hr(x):
    x = x - x.mean(); ac = np.correlate(x, x, "full")[len(x)-1:]; lo = int(FR * 0.4); k = lo + np.argmax(ac[lo:int(FR * 1.6)]); return 60 * FR / k
S = dict(vessel_angle_deg=round(float(ang), 1), D_mm=dict(min=round(float(Ds.min()), 2), max=round(float(Ds.max()), 2), mean=round(float(Ds.mean()), 2)),
         distension_pct=round(float((Ds.max() - Ds.min()) / Ds.min() * 100), 1), HR_from_D=round(hr(Ds), 1),
         A_mm2=dict(min=round(float(As.min()), 1), max=round(float(As.max()), 1)), Deq_cross_mm=dict(min=round(float(Deq.min()), 2), max=round(float(Deq.max()), 2)),
         area_change_pct=round(float((As.max() - As.min()) / As.min() * 100), 1), HR_from_A=round(hr(As), 1),
         v_axial_m_s=dict(min=round(float(vm.min()), 3), max=round(float(vm.max()), 3)), v_vessel_m_s=dict(min=round(float(np.nanmin(vvessel)), 2), max=round(float(np.nanmax(vvessel)), 2)),
         HR_from_v=round(hr(vm), 1), frac_pixels_near_nyquist=round(vax_frac_alias, 3), nyquist_axial_m_s=0.557)
json.dump(S, open(os.path.join(REP, "carotid_summary.json"), "w"), indent=1); print(json.dumps(S, indent=1))
np.savez(os.path.join(D, "carotid_timeseries.npz"), t=t, D=Dt, Ds=Ds, A=A, As=As, v_axial=vm, v_vessel=vvessel, angle=ang)
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ex = [xs[0], xs[-1], zs[-1], zs[0]]
ax[0, 0].imshow(L["db"][0].astype(float), cmap="gray", vmin=-60, vmax=0, extent=ex); ax[0, 0].plot(xs[cols], near[0], "c.", ms=1); ax[0, 0].plot(xs[cols], far[0], "y.", ms=1); ax[0, 0].set_title(f"long frame 0, walls (angle {ang:.1f} deg)")
im = ax[0, 1].imshow(np.where(pd[30] > np.percentile(pd[30], 70), vax[30], np.nan), cmap="RdBu_r", vmin=-0.56, vmax=0.56, extent=ex); ax[0, 1].set_title("axial velocity frame 30 (m/s), masked by power"); plt.colorbar(im, ax=ax[0, 1])
ax[0, 2].imshow(C["db"][0].astype(float), cmap="gray", vmin=-60, vmax=0, extent=[cx[0], cx[-1], cz[-1], cz[0]]); ax[0, 2].contour(cx, cz, masks[0], [0.5], colors="y"); ax[0, 2].set_title("cross frame 0, lumen")
ax[1, 0].plot(t, Dt, ".", color="0.6"); ax[1, 0].plot(t, Ds, "r"); ax[1, 0].plot(t, Deq, "b--", label="from cross area"); ax[1, 0].legend(); ax[1, 0].set_title("diameter mm"); ax[1, 0].set_xlabel("s")
ax[1, 1].plot(t, A, ".", color="0.6"); ax[1, 1].plot(t, As, "b"); ax[1, 1].set_title("lumen area mm^2 (cross)"); ax[1, 1].set_xlabel("s")
ax[1, 2].plot(t, vm, "g"); ax[1, 2].axhline(0.557, ls=":"); ax[1, 2].axhline(-0.557, ls=":"); ax[1, 2].set_title("median axial velocity in lumen (m/s)"); ax[1, 2].set_xlabel("s")
fig.tight_layout(); fig.savefig(os.path.join(REP, "carotid_tracking.png"), dpi=80); print("ok")
