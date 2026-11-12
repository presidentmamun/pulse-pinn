# Run log

## 2026-11-13: Phase 0 (data exploration)
- Searched Hugging Face for blood-flow data. No public dataset has wall motion, velocity *and* pressure together.
- **zea-carotid-2023, subject 5** (longitudinal + cross views):
  - Raw RF, 150 frames at 30 Hz.
  - Each frame has 128 focused lines and 21 plane waves (±18°, 88.4 µs apart).
  - Our plane-wave DAS beamforming gives clean images.
  - Longitudinal D(t): 5.85–6.69 mm, about 14% distension, HR about 75.
  - Cross-section lumen area: 42–54 mm². It is noisier and needs a manual seed. It is a separate recording, so it is not beat-paired with the longitudinal one.
  - Axial Doppler: not usable (vessel about 4° from perpendicular to the beam).
- **MIMIC-IV waveform:**
  - 6 ICU records with ABP, 10 min each, 62.5 Hz.
  - SBP 94–153, DBP 41–67, beat-mean MAP 63–95, HR 63–104.
- The commercial CFD (Ansys Fluent 18.1) has no licence, so we build our own solver.

## 2026-11-13: Phase 1 started
- Pipeline `scripts/phase1.cmd`: one longitudinal view per subject, then beamforming, then tracking, then a QC figure.

## 2026-11-13: Phase 0b literature (first pass)
- Closest prior work: the Columbia (Konofagou) group's PINN + Pulse Wave Imaging, which gives carotid compliance and pressure, including a human feasibility study. It uses high-frame-rate data and linearized 1D equations.
- Kissas 2020 (4D-flow MRI, 1D PINN).
- Womersley Fourier features (APS DFD 2021).
- Gap kept: identifiability, conventional 30 Hz scans, nonlinear/axisymmetric NS-PINN, CPC-PINN combination. See PLAN §2.
