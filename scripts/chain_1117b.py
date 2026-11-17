"""2026-11-17 day chain: beat-by-beat / ensemble fits on real data, then the realistic-noise twin. Resumable steps."""
import os, sys, subprocess, datetime as dt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ST = os.path.join(R, "reports", "phase6", "chain_1117b_status.txt")
def st(m):
    with open(ST, "a", encoding="utf-8") as f: f.write(f"{dt.datetime.now():%m-%d %H:%M:%S} {m}\n")
st("chain started")
for args, log in ((["scripts/phase6_beats.py"], "beats.log"), (["scripts/phase6_twin_realnoise.py", "40"], "realnoise.log")):
    st("START " + " ".join(args))
    with open(os.path.join(R, "reports", "phase6", log), "w", encoding="utf-8") as fh:
        rc = subprocess.call([sys.executable, "-u"] + args, cwd=R, stdout=fh, stderr=subprocess.STDOUT)
    st(f"END rc={rc} " + " ".join(args))
st("chain done")
