"""Recompute MIMIC ABP beat statistics with MAP = mean over each beat (fixes the median-based MAP)."""
import os, glob, json, numpy as np
from scipy.signal import find_peaks, butter, filtfilt
R = os.environ.get("PULSE_ROOT", r"F:\my-project\pulse-pinn"); out = []
for fn in sorted(glob.glob(os.path.join(R, "data", "mimic", "*.npz"))):
    z = np.load(fn); fs = float(z["fs"]); x = z["ABP"].astype(float); x = np.nan_to_num(x, nan=np.nanmedian(x))
    b, a = butter(3, 15 / (fs / 2)); xf = filtfilt(b, a, x)
    pk, _ = find_peaks(xf, distance=int(0.33 * fs), prominence=15)
    tr = np.array([int(np.argmin(xf[pk[i]:pk[i + 1]]) + pk[i]) for i in range(len(pk) - 1)])
    maps = [x[tr[i]:tr[i + 1]].mean() for i in range(len(tr) - 1)]
    s = dict(record=os.path.basename(fn)[:-4], SBP=round(float(np.median(xf[pk])), 1), DBP=round(float(np.median(xf[tr])), 1), MAP=round(float(np.median(maps)), 1),
             PP=round(float(np.median(xf[pk[1:]] - xf[tr])), 1), HR=round(float(60 * fs / np.median(np.diff(pk))), 1))
    out.append(s); print(json.dumps(s))
os.makedirs(os.path.join(R, "reports", "phase1"), exist_ok=True)
json.dump(out, open(os.path.join(R, "reports", "phase1", "mimic_stats.json"), "w"), indent=1)
