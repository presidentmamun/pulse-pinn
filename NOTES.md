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

## 2026-11-13: Phase 1 first pass (8 of 10 subjects)
- Beamforming ~32 min per subject on the home PC (4 workers, BelowNormal).
- Longitudinal tracking results (D = diameter, dist = (Dmax-Dmin)/Dmin):

| Subj | Angle | D mean (mm) | Dist | HR | Frame noise (mm) | QC |
|---|---|---|---|---|---|---|
| s02 | -3.0 | 6.87 | 13.6% | 56 | 0.070 | good |
| s04 | 0.0 | 7.45 | 13.2% | 60 | 0.084 | good |
| s05 | -4.3 | 6.25 | 14.3% | 75 | 0.042 | good |
| s07 | 1.9 | 6.02 | 9.4% | 55 | 0.052 | good |
| s10 | 1.7 | 6.99 | 12.7% | 50 | 0.040 | good |
| s01 | -1.3 | 7.48 | 35.0% | 67 | 0.258 | wall jumps between layers |
| s08 | -2.1 | 7.60 | 34.3% | 58 | 0.192 | wall jumps between layers |
| s03 | -13.7 | 6.49 | 33.2% | 72 | 0.075 | tilted vessel, baseline drift |

- 5 clean subjects give physiological distension of 9–14%. The three ~34% values are tracking artefacts, not physiology.
- Fix needed: temporally constrained wall tracking (search near the previous frame's wall, not per frame).
- s06 and s09: downloads were cut off (curl error 18). Rerun with resume (`scripts/phase1b.cmd`).

## 2026-11-13: Tracker v2 (`scripts/track_long.py`)
- Old tracker: per-frame search within +-8 mm, so it jumped between tissue layers in s01/s08.
- v2:
  - Wall seeds come from the temporal-mean image: the strongest edge 1.5-5.5 mm from the lumen centre (lumen-intima band).
  - Per frame, the search is sequential: +-0.5 mm of the previous frame's wall, clamped to +-1.2 mm of the seed.
  - Sub-pixel parabola refinement.
  - New output `distension_p5_95_pct`, which is robust to outliers.
- Results (10 subjects; 8 usable):

| Subj | D mean (mm) | Dist p5-95 | HR | Noise (mm) | Column spread (mm) | QC |
|---|---|---|---|---|---|---|
| s01 | 6.34 | 11.0% | 69 | 0.034 | 0.49 | good (was a layer jump, now clean, dicrotic notch visible) |
| s02 | 6.61 | 10.6% | 56 | 0.029 | 0.83 | good |
| s04 | 7.34 | 8.0% | 62 | 0.050 | 1.03 | fair |
| s05 | 6.24 | 11.9% | 75 | 0.036 | 0.37 | good |
| s07 | 5.89 | 10.0% | 55 | 0.059 | 0.41 | good |
| s08 | 8.13 | 4.7% | 60 | 0.033 | 1.04 | fair (was a layer jump) |
| s10 | 6.91 | 10.1% | 50 | 0.031 | 0.29 | good |
| s03 | 6.90 | 18.2% | - | 0.046 | 1.53 | exclude: vessel tilted 14 deg, probe drift |
| s06 | 7.18 | 8.0% | 60 | 0.026 | 0.34 | good (re-run after truncated download) |
| s09 | 7.04 | 7.1% | 150? | 0.053 | 1.84 | exclude: vessel tilted 27 deg, HR implausible, weak periodicity (autocorr 0.34) |

- Distension (p5-95) is 5-12%, consistent with healthy adult carotids.

## 2026-11-13: Phase 2 / L1 solver (`pulsens/l1.py`) verified
- Model: nonlinear 1D (A, Q), tube law p = p_ref + beta (sqrt A - sqrt A0), Poiseuille friction.
- Scheme: two-step Lax-Wendroff. Characteristic BCs: inlet flow, outlet RCR Windkessel.
- Verification (`scripts/verify_l1.py`, home PC, under 2 min). All gates pass:
  - MMS: observed order 2.00 (N = 50 to 400).
  - Wave speed: 5.078 vs theory 5.015 m/s (+1.2%). The excess is the expected nonlinear u + c effect.
  - Reflection at a resistive outlet: R = Z0 gives 0.006 (theory 0); 3 Z0 gives 0.522 (0.5); Z0/3 gives -0.509 (-0.5).
  - Mass conservation: error 2.8e-6 of stroke volume.
  - Carotid RCR case is periodic: cycle-to-cycle change 0.02%. 107/80 mmHg, distension 6.8%.
- Next: calibrate RCR + beta to Phase 1 / MIMIC ranges (PP ~ 40 mmHg, distension 8-12%), then generate the L1 twin dataset.

## 2026-11-13: Phase 2 / L1 calibrated (`scripts/calibrate_l1.py`, `pulsens/carotid.py`)
- Targets (mid-segment): PP 40 mmHg, MAP 93 mmHg, diameter distension 10% (Phase 1 median of good subjects).
- Fixed-point iteration on C (pulse pressure), RT (mean pressure) and wall stiffness (distension). It converged in 12 iterations, about 11 s per run on the home PC.
- Result: 113/74 mmHg, PP 39.8 mmHg, MAP 93.0, distension 10.2% (D 6.40 to 7.05 mm), cycle-to-cycle change 2e-5.
- Calibrated values:
  - beta = 9.19e6 Pa/m (dist_ref 0.101)
  - c0 = 5.0 m/s
  - R1 = Z0 = 1.60e8 Pa s/m^3
  - R2 = 1.90e9
  - C = 1.96e-10 m^3/Pa
  - tau = R2*C = 0.37 s
- Caveats:
  - The tau is shorter than whole-body values (about 1-1.5 s). Here C stands for the cerebral bed fed by one carotid, so this is acceptable, but tau should be varied in the twin dataset.
  - There is a small pressure amplification from inlet to outlet (about 2 mmHg), caused by outlet reflection.
