@echo off
set R=F:\my-project\pulse-pinn
cd /d %R%
if not exist logs mkdir logs
if not exist data\carotid mkdir data\carotid
if not exist data\mimic mkdir data\mimic
if not exist reports mkdir reports
echo [%date% %time%] start > logs\run.log
if not exist venv\Scripts\python.exe C:\Python314\python.exe -m venv venv >> logs\run.log 2>&1
venv\Scripts\python.exe -m pip install -q --upgrade pip >> logs\run.log 2>&1
venv\Scripts\python.exe -m pip install -q numpy scipy h5py matplotlib wfdb >> logs\run.log 2>&1
echo [%time%] deps done >> logs\run.log
set B=https://huggingface.co/datasets/zeahub/zea-carotid-2023/resolve/main
for %%f in (README.md LICENSE reconstruct.py pipeline.yaml data/5_long_2cm_L_0000.hdf5 data/5_cross_2cm_L_0000.hdf5) do (
  echo [%time%] get %%f >> logs\run.log
  curl -sS -L -C - --retry 5 -A pulse-pinn -o data\carotid\%%~nxf %B%/%%f >> logs\run.log 2>&1
)
echo [%time%] carotid download done >> logs\run.log
venv\Scripts\python.exe scripts\inspect_carotid.py >> logs\run.log 2>&1
echo [%time%] carotid inspect done >> logs\run.log
venv\Scripts\python.exe scripts\mimic_abp.py >> logs\run.log 2>&1
echo [%date% %time%] ALL DONE >> logs\run.log
