@echo off
cd /d F:\my-project\pulse-pinn
echo [%date% %time%] bf start > logs\bf.log
venv\Scripts\python.exe scripts\carotid_bf.py data\carotid\5_long_2cm_L_0000.hdf5 0-149 4 >> logs\bf.log 2>&1
venv\Scripts\python.exe scripts\carotid_bf.py data\carotid\5_cross_2cm_L_0000.hdf5 0-149 4 >> logs\bf.log 2>&1
echo BF ALL DONE >> logs\bf.log
