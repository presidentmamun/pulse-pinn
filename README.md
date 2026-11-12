# Pulse-PINN

**Can a physics-informed neural network recover blood pressure and arterial stiffness from a cheap carotid ultrasound clip?**

The arm cuff measures neither central pressure nor vessel stiffness, and a catheter is invasive. Ultrasound sees the artery wall move well, but sees blood velocity poorly and pressure not at all. This project asks a plain question: from ultrasound-type data, which of the hidden quantities can a PINN really recover? Those quantities are pressure p(x,t) and wall stiffness β.

> Status: **early research, work in progress.** Phase 0 (data analysis) is done and Phase 1 (all subjects) is running. See [PLAN.md](PLAN.md) for the full research plan and [NOTES.md](NOTES.md) for the run log.

## Approach (short)
1. **Real data**
   - In-vivo carotid ultrasound: raw RF from 10 subjects, beamformed with our own plane-wave delay-and-sum code.
   - Real ICU arterial pressure waveforms, used for physiological ranges.
2. **Own 3D Navier–Stokes solver ("PulseNS")**
   - GPU projection method with an immersed moving wall and reduced fluid–structure coupling.
   - Verified against manufactured solutions, Poiseuille and Womersley flow.
   - It generates "twin" data that mimics the scanner: 30 Hz frames, real noise levels, and a known ground-truth pressure and β.
3. **PINNs at three physics-fidelity levels:** 1D reduced model, axisymmetric NS, full 3D NS.
4. **Identifiability study:** how accurately p and β come back from
   - (a) wall motion only
   - (b) wall motion + velocity
   - (c) the same plus one cuff reading
5. **New architecture candidate (CPC-PINN)**
   - A periodic cardiac time embedding.
   - Mass conservation built into the network by a potential or stream function.
   - A hard tube-law constraint.
   - It is compared with standard PINN baselines in an ablation.
6. **Clinically meaningful reporting:**
   - mmHg errors against cuff-validation bars
   - stiff-vs-healthy classification
   - repeatability
   - PWV and distensibility indices

This is a methods and feasibility study on public data and simulation. **It is not a clinical validation.**

## First result: carotid wall motion from raw RF (subject 5)
- Diameter: 5.85–6.69 mm.
- Distension: about 14%.
- Heart rate: about 75 bpm.
- The recording is 5 s at 30 Hz.
- Axial Doppler velocity is not usable in this view, because the vessel lies about 4° from perpendicular to the beam.

## Repository layout
```
PLAN.md                 research plan (what / why / how, feasibility, phases)
NOTES.md                run log
scripts/
  carotid_bf.py         plane-wave DAS beamforming + Doppler ensemble from zea HDF5 RF
  track_long.py         longitudinal wall tracking -> D(t), distension, HR, QC figure
  mimic_abp.py          find MIMIC-IV waveform records with ABP, beat statistics
  mimic_map.py          beat-mean MAP / SBP / DBP / PP
  phase1.cmd            batch pipeline (download -> beamform -> track) for all subjects
  exploration/          first-pass exploration scripts kept for provenance
```
Set `PULSE_ROOT` to the working folder that holds `data/`, `reports/` and `logs/`.

## Data (not included in this repo)
- **TU/e carotid 2023** ([zeahub/zea-carotid-2023](https://huggingface.co/datasets/zeahub/zea-carotid-2023)): CC BY 4.0, by van Nierop et al., TU/e. We derive beamformed images and wall-motion signals from the raw RF.
- **MIMIC-IV Waveform Database** v0.1.0 (PhysioNet): cite Moody et al. and PhysioNet (Goldberger et al. 2000), and follow the PhysioNet licence terms.

## Requirements
Python ≥ 3.10. Install with `pip install -r requirements.txt`.

## AI assistance
Code, simulations and analysis in this project are developed with AI assistance (Anthropic Claude). The author reviews them and is responsible for all claims.

## Licence
Code: MIT (see `LICENSE`). Data remain under their original licences.

Author: Khairuzzaman Mamun, PhD
