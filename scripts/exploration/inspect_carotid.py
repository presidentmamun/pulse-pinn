"""Stage 1: structure + scan parameters of the zea carotid HDF5 files (lazy reads only)."""
import h5py, numpy as np, os, json, glob
R = os.environ.get("PULSE_ROOT", r"F:\my-project\pulse-pinn")
out = open(os.path.join(R, "reports", "carotid_structure.txt"), "w", encoding="utf-8")
def p(*a): print(*a, file=out, flush=True)
for fn in sorted(glob.glob(os.path.join(R, "data", "carotid", "*.hdf5"))):
    p("=" * 70, os.path.basename(fn), round(os.path.getsize(fn) / 1e9, 2), "GB")
    with h5py.File(fn, "r") as f:
        for k, v in f.attrs.items(): p("  @", k, "=", str(v)[:300])
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                s = f"  {name}  shape={obj.shape} dtype={obj.dtype}"
                if obj.size <= 64:
                    s += "  value=" + np.array2string(np.asarray(obj[()]).ravel()[:64], precision=6, max_line_width=400)
                elif obj.ndim <= 2 and obj.size <= 200000:
                    a = np.asarray(obj[()], dtype=float); s += f"  min={a.min():.6g} max={a.max():.6g} mean={a.mean():.6g}"
                p(s)
            else:
                p(f"  [{name}]")
            for ak, av in obj.attrs.items(): p("     @", ak, "=", str(av)[:300])
        f.visititems(visit)
        # one small raw sample: stats of first frame, first transmit
        for key in ("data/raw_data", "data/image", "data/beamformed_data", "data/image_sc"):
            if key in f:
                d = f[key]; idx = tuple([0] * min(2, d.ndim - 2))
                a = np.asarray(d[idx], dtype=float)
                p(f"  SAMPLE {key}{idx}: shape={a.shape} min={a.min():.4g} max={a.max():.4g} std={a.std():.4g}")
out.close()
print("carotid report written")
