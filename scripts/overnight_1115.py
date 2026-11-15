"""Overnight chain 2026-11-15 -> 16. One GPU job at a time, hard stop 08:30 JST (all jobs are resumable).
0 wait for B3 seed 2; 1 dcorr smoke test (3 cases); 2 CPCv2 seed 1; 3 B3 dcorr seeds 0-2 (only if smoke test ok)."""
import os, sys, time, json, subprocess, datetime as dt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P5 = os.path.join(R, "reports", "phase5")
ST = os.path.join(P5, "overnight_status.txt")
DEAD = dt.datetime(2026, 11, 16, 8, 30)
PY = sys.executable

def st(msg):
    with open(ST, "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now():%m-%d %H:%M:%S} {msg}\n")

def run(args, log):
    left = (DEAD - dt.datetime.now()).total_seconds()
    if left < 120:
        st(f"SKIP (deadline) {' '.join(args)}"); return None
    st(f"START {' '.join(args)} -> {log}")
    with open(os.path.join(P5, log), "a", encoding="utf-8") as fh:
        p = subprocess.Popen([PY, "-u"] + args, cwd=R, stdout=fh, stderr=subprocess.STDOUT)
        try:
            rc = p.wait(timeout=left)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait(); st(f"KILLED at deadline: {' '.join(args)}"); return "deadline"
    st(f"END rc={rc} {' '.join(args)}"); return rc

st("chain started")
while "finished B3 seed 2" not in open(os.path.join(P5, "full_seed2_b3.log"), encoding="utf-8", errors="ignore").read():
    if dt.datetime.now() > DEAD:
        st("deadline while waiting for B3 seed 2"); sys.exit()
    time.sleep(60)
st("B3 seed 2 finished")

ok = False
rc = run(["scripts/phase5_dcorr.py", "B3dc", "B3", "8", "-1", "0", "3", "7.5", "3"], "dcorr_smoke.log")
try:
    rows = [json.loads(s) for s in open(os.path.join(P5, "dcorr_test.jsonl"))]
    ok = rc == 0 and len(rows) >= 3 and all(r["rmse_mmHg"] == r["rmse_mmHg"] and r["rmse_mmHg"] < 30 for r in rows)
    st(f"smoke test rows={len(rows)} rmse={[round(r['rmse_mmHg'], 2) for r in rows]} ok={ok}")
except Exception as e:
    st(f"smoke test failed: {e!r}")

if run(["scripts/phase5_full.py", "CPCv2", "CPC", "4", "6", "0.8ab", "1", "40"], "full_seed1_cpcv2.log") == "deadline":
    sys.exit()
if ok:
    for s in (0, 1, 2):
        if run(["scripts/phase5_dcorr.py", "B3dc", "B3", "8", "-1", str(s), "40", "7.5"], f"dcorr_seed{s}_b3.log") == "deadline":
            sys.exit()
else:
    st("dcorr runs skipped (smoke test not ok)")
st("chain done")
