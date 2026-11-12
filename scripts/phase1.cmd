@echo off
rem Phase 1: download one longitudinal view per subject, beamform, track. Runs BelowNormal, 4 workers.
set R=F:\my-project\pulse-pinn
cd /d %R%
set OMP_NUM_THREADS=1
set B=https://huggingface.co/datasets/zeahub/zea-carotid-2023/resolve/main/data
echo [%date% %time%] phase1 start > logs\phase1.log
start "" /belownormal /wait /b venv\Scripts\python.exe scripts\mimic_map.py >> logs\phase1.log 2>&1
start "" /belownormal /wait /b venv\Scripts\python.exe scripts\track_long.py data\derived\5_long_2cm_L_0000_bmode_0-149.npz s05 >> logs\phase1.log 2>&1
for %%s in (2:2_long_2cm_0000 3:3_long_2cm_L_0000 4:4_long_2cm_L_0000 6:6_long_2cm_L_0000 7:7_long_2cm_L_0000 8:8_long_2cm_L_0000 9:9_long_2cm_L_0000 10:10_long_2cm_L_0000 1:long_2cm_v1_patient1_L_0000) do call :one %%s
echo [%date% %time%] PHASE1 DONE >> logs\phase1.log
goto :eof
:one
for /f "tokens=1,2 delims=:" %%a in ("%1") do (set SID=%%a& set F=%%b)
set SL=0%SID%
set SL=s%SL:~-2%
echo [%time%] %SL% download %F% >> logs\phase1.log
curl -sS -L -C - --retry 10 -A pulse-pinn -o data\carotid\%F%.hdf5 %B%/%F%.hdf5 >> logs\phase1.log 2>&1
echo [%time%] %SL% beamform >> logs\phase1.log
start "" /belownormal /wait /b venv\Scripts\python.exe scripts\carotid_bf.py data\carotid\%F%.hdf5 0-149 4 >> logs\phase1.log 2>&1
start "" /belownormal /wait /b venv\Scripts\python.exe scripts\track_long.py data\derived\%F%_bmode_0-149.npz %SL% >> logs\phase1.log 2>&1
echo [%time%] %SL% done >> logs\phase1.log
goto :eof
