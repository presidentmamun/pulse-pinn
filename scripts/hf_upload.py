"""Upload the L1 twin benchmark + PINN results to a Hugging Face dataset repo.
Usage: python hf_upload.py [repo_id=presidentmamun/pulse-pinn-twin-l1] [--public]
Creates the repo PRIVATE unless --public is given (visibility of an existing repo is not changed).
Re-run any time: the results file and case table are rebuilt, unchanged files are skipped by the Hub."""
import os, sys, json, glob, shutil, tempfile
import numpy as np
from huggingface_hub import HfApi

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
REPO = args[0] if args else "presidentmamun/pulse-pinn-twin-l1"
PUBLIC = "--public" in sys.argv

stage = tempfile.mkdtemp(prefix="hf_pulse_")
try:
    shutil.copy(os.path.join(R, "hf", "README.md"), stage)
    os.makedirs(os.path.join(stage, "twin_l1")); os.makedirs(os.path.join(stage, "cases")); os.makedirs(os.path.join(stage, "results"))
    rows = []
    for p in sorted(glob.glob(os.path.join(R, "data", "twin_l1", "case_*.npz"))):
        shutil.copy(p, os.path.join(stage, "twin_l1"))
        z = np.load(p)
        pr, me = json.loads(str(z["params"])), json.loads(str(z["metrics"]))
        rows.append({"case": pr["case"], **{f"param_{k}": v for k, v in pr.items() if k not in ("case", "seed")},
                     **{k: me[k] for k in ("SBP", "DBP", "MAP", "PP", "D_min_mm", "D_max_mm", "distension_pct",
                                           "beta", "c0", "Z0", "C", "sigma_D_mm")}})
    with open(os.path.join(stage, "cases", "cases.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_res = 0  # copy only complete lines: the file may be appended to by a running job
    with open(os.path.join(stage, "results", "phase5_full.jsonl"), "w") as fo:
        for name in ("full.jsonl", "dcorr.jsonl"):
            src = os.path.join(R, "reports", "phase5", name)
            if not os.path.exists(src):
                continue
            with open(src) as fi:
                for s in fi:
                    try:
                        json.loads(s)
                    except ValueError:
                        continue
                    fo.write(s if s.endswith("\n") else s + "\n"); n_res += 1

    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", private=not PUBLIC, exist_ok=True)
    api.upload_folder(folder_path=stage, repo_id=REPO, repo_type="dataset",
                      commit_message=f"Update: {len(rows)} cases, {n_res} PINN fits")
    info = api.dataset_info(REPO)
    print(f"uploaded {len(rows)} cases, {n_res} fits -> https://huggingface.co/datasets/{REPO} (private={info.private})")
finally:
    shutil.rmtree(stage, ignore_errors=True)
