"""Find MIMIC-IV waveform records with ABP, stream a 10-min window each, beat stats + mean beat shape."""
import wfdb, numpy as np, os, re, urllib.request, json, traceback
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.environ.get("PULSE_ROOT", r"F:\my-project\pulse-pinn"); D = os.path.join(R, "data", "mimic"); REP = os.path.join(R, "reports")
DB = "mimic4wdb/0.1.0"; BASE = "https://physionet.org/files/" + DB + "/"
H = {"User-Agent": "pulse-pinn"}
get = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=60).read().decode("utf-8", "ignore")
files = json.loads(get("https://huggingface.co/api/datasets/lucky9-cyou/MIMIC-IV-Waveform"))["siblings"]
masters = []
for x in files:
    m = re.match(r"files/mimic4wdb/0\.1\.0/(waves/p\d+/p\d+/(\d+))/(\d+)\.hea$", x["rfilename"])
    if m and m.group(2) == m.group(3): masters.append((m.group(1), m.group(2)))
print("master records:", len(masters))
found, summary, names_seen = [], [], set()
ABPN = ("ABP", "ART", "Art", "AOBP")
for d, n in masters[:80]:
    if len(found) >= 6: break
    try:
        pn = DB + "/" + d
        h = wfdb.rdheader(n, pn_dir=pn)
        segs = [s for s in (getattr(h, "seg_name", None) or []) if s and s != "~"]
        best = None
        for s in segs:
            sh = wfdb.rdheader(s, pn_dir=pn)
            names_seen.update(sh.sig_name or [])
            if s.endswith("layout"): 
                if not any(k in (sh.sig_name or []) for k in ABPN): break
                continue
            k = next((k for k in ABPN if k in (sh.sig_name or [])), None)
            if k and sh.sig_len > sh.fs * 600 and (best is None or sh.sig_len > best[3]):
                best = (d, s, sh.fs, sh.sig_len, k)
        if best: found.append(best); print("found", best)
    except Exception as e:
        print("skip", d, e)
print("signal names seen:", sorted(names_seen))
print("records with ABP:", len(found))
for d, s, fs, n, key in found:
    try:
        start = int(n // 2 - fs * 300)
        rec = wfdb.rdrecord(s, pn_dir=DB + "/" + d, sampfrom=max(0, start), sampto=max(0, start) + int(fs * 600))
        sig = {nm: rec.p_signal[:, i] for i, nm in enumerate(rec.sig_name)}
        np.savez_compressed(os.path.join(D, s + ".npz"), fs=fs, **{k: v.astype(np.float32) for k, v in sig.items()})
        abp = sig[key]; ok = np.isfinite(abp)
        x = np.nan_to_num(abp, nan=np.nanmedian(abp))
        b, a = butter(3, 15 / (fs / 2)); xf = filtfilt(b, a, x)
        pk, _ = find_peaks(xf, distance=int(0.33 * fs), prominence=15)
        tr = [int(np.argmin(xf[pk[i]:pk[i + 1]]) + pk[i]) for i in range(len(pk) - 1)]
        sbp, dbp = xf[pk], xf[tr]; hr = 60 * fs / np.diff(pk)
        beats = []
        for i in range(len(tr) - 1):
            seg = xf[tr[i]:tr[i + 1]]
            if 0.4 * fs < len(seg) < 1.5 * fs: beats.append(np.interp(np.linspace(0, 1, 200), np.linspace(0, 1, len(seg)), seg))
        beats = np.array(beats)
        st = dict(record=f"{d}/{s}", fs=fs, signals=list(rec.sig_name), valid_frac=round(float(ok.mean()), 3),
                  n_beats=int(len(pk)), SBP=round(float(np.median(sbp)), 1), DBP=round(float(np.median(dbp)), 1),
                  MAP=round(float(np.median(xf)), 1), HR=round(float(np.median(hr)), 1),
                  SBP_iqr=[round(float(q), 1) for q in np.percentile(sbp, [25, 75])])
        summary.append(st); print(json.dumps(st))
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
        t = np.arange(int(fs * 8)) / fs; ax[0].plot(t, abp[:len(t)]); ax[0].set_title(s + " ABP 8 s"); ax[0].set_xlabel("s"); ax[0].set_ylabel("mmHg")
        if len(beats):
            ax[1].plot(np.linspace(0, 1, 200), beats.T, color="0.8", lw=0.3); ax[1].plot(np.linspace(0, 1, 200), beats.mean(0), "r")
        ax[1].set_title(f"{len(beats)} beats normalised"); fig.tight_layout(); fig.savefig(os.path.join(REP, f"mimic_{s}.png"), dpi=90); plt.close(fig)
    except Exception:
        traceback.print_exc()
json.dump(summary, open(os.path.join(REP, "mimic_summary.json"), "w"), indent=1)
print("mimic done", len(summary))
