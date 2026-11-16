"""Shift sensitivity sweep, seeds 1-2 (2026-11-16 evening): B3dc at level d, shifts 2.5, 5 and 10 mmHg.
Shift 0 is level d itself (same rng draws), and 7.5 is already done for seeds 0-2.
Each case's true cuff bias is 5-10 mmHg, so each run also tests a prior that is wrong by a case-specific amount.
One GPU job at a time; every run is resumable."""
import os, sys, subprocess, datetime as dt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P5 = os.path.join(R, "reports", "phase5")
ST = os.path.join(P5, "shift_sweep_s12_status.txt")


def st(msg):
    with open(ST, "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now():%m-%d %H:%M:%S} {msg}\n")


st("sweep started")
for seed, shift in [(s, x) for s in ("1", "2") for x in ("2.5", "5", "10")]:
    args = ["scripts/phase5_dcorr.py", "B3dc", "B3", "8", "-1", seed, "40", shift]
    log = f"shift_{shift}_s{seed}_b3.log"
    st(f"START seed={seed} shift={shift} -> {log}")
    with open(os.path.join(P5, log), "a", encoding="utf-8") as fh:
        rc = subprocess.call([sys.executable, "-u"] + args, cwd=R, stdout=fh, stderr=subprocess.STDOUT)
    st(f"END rc={rc} seed={seed} shift={shift}")
    if rc != 0:
        st("sweep stopped after an error"); sys.exit(1)
st("sweep done")
