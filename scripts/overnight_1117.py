"""Night of 2026-11-16/17: real data (8 subjects, B3 + CPCv2, levels a/c, 3 seeds), cuff-assumption sensitivity,
then the cuff-bias correction for CPCv2 on the twin (seeds 0-2). One GPU job at a time; every step is resumable."""
import os, sys, subprocess, datetime as dt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ST = os.path.join(R, "reports", "phase6", "overnight_status.txt")
def st(m):
    with open(ST, "a", encoding="utf-8") as f: f.write(f"{dt.datetime.now():%m-%d %H:%M:%S} {m}\n")
steps = [
    (["scripts/phase6_real.py", "B3", "0,1,2", "all", "8000", "120/80"], "phase6_b3.log"),
    (["scripts/phase6_real.py", "CPCv2", "0,1,2", "all", "8000", "120/80"], "phase6_cpcv2.log"),
    (["scripts/phase6_real.py", "B3", "0,1,2", "all", "8000", "110/70"], "phase6_b3_110.log"),
    (["scripts/phase6_real.py", "B3", "0,1,2", "all", "8000", "130/85"], "phase6_b3_130.log"),
] + [(["scripts/phase5_dcorr.py", "CPCv2dc", "CPC", "4", "6", str(s), "40", "7.5"], f"dcorr_seed{s}_cpcv2.log") for s in (0, 1, 2)]
st("chain started")
for args, log in steps:
    st("START " + " ".join(args) + " -> " + log)
    d = os.path.join(R, "reports", "phase6" if "phase6" in args[0] else "phase5")
    with open(os.path.join(d, log), "a", encoding="utf-8") as fh:
        rc = subprocess.call([sys.executable, "-u"] + args, cwd=R, stdout=fh, stderr=subprocess.STDOUT)
    st(f"END rc={rc} " + " ".join(args))
    if rc != 0:
        st("step failed, continuing with the next one")
st("chain done")
