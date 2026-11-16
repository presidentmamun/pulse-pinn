---
license: cc-by-4.0
pretty_name: Pulse-PINN carotid twin benchmark (L1)
tags:
  - physics-informed-neural-networks
  - pinn
  - hemodynamics
  - carotid
  - blood-pressure
  - arterial-stiffness
  - inverse-problems
  - synthetic
task_categories:
  - tabular-regression
size_categories:
  - n<1K
configs:
  - config_name: results
    data_files:
      - split: train
        path: results/phase5_full.jsonl
  - config_name: cases
    data_files:
      - split: train
        path: cases/cases.jsonl
---

# Pulse-PINN carotid twin benchmark (L1)

Synthetic "digital twin" carotid artery cases for a simple inverse problem: estimate the pressure waveform and the wall stiffness from a diameter-only ultrasound scan at about 30 Hz.
The dataset also holds the per-fit results of the physics-informed neural networks (PINNs) tested on it.

Code: https://github.com/presidentmamun/pulse-pinn

## What is inside

| Path | Content |
|---|---|
| `twin_l1/case_XXX.npz` | 106 accepted cases from the 1D carotid model (our own solver) |
| `cases/cases.jsonl` | One row per case: the sampled parameters and the true outputs (SBP, DBP, PP, distension, beta, wave speed c0, ...) |
| `results/phase5_full.jsonl` | One row per PINN fit: model, seed, data level, errors |

### Twin cases (`twin_l1/*.npz`)
The cases come from a Latin hypercube over 200 parameter sets. Only cases with physiological outputs are kept (PP 25-60 mmHg, distension 3-14%), which leaves 106.

Each file holds:
- full last-cycle fields on a uniform grid:
  - `t` (400)
  - `x` (151)
  - `A`, `Q` and `p` (400 x 151, SI units)
- a synthetic scan of 5 s at 30 Hz, at 9 columns over a 2 cm window:
  - `scan_D`: noisy diameter. The noise sigma is 0.03-0.05 mm, matched to real carotid scans.
  - `scan_D_true`
  - `scan_p_eval_only`: pressure, for scoring only. Never use it as a training input.
- `params` and `metrics`, as JSON strings.

Sampled parameter ranges:
- reference distension 0.04-0.20
- D0 5.5-8.5 mm
- HR 50-100 bpm
- mean flow 4-8 mL/s
- tau 0.2-0.7 s
- MAP 80-110 mmHg

### Data levels
- **a**: diameter only
- **b**: a + mean velocity at the middle column
- **c**: a + an ideal cuff SBP/DBP at the same site
- **d**: a + a biased cuff at a distant site (SBP +5 to +10, DBP ±2 mmHg)
- **dc** (model `B3dc`): level d with the cuff SBP shifted by a fixed prior (column `shift`: 7.5 mmHg for seeds 0-2; 2.5, 5 and 10 mmHg for seed 0 as a sensitivity sweep) and the true per-case bias in `cuff_bias_sbp`

### Models in `results`
- **B2**: Fourier-feature MLP
- **B3**: modified MLP (Wang, Teng and Perdikaris 2021)
- **CPCv2**: our cardiac-periodic conservative PINN

Each fit uses 8000 Adam iterations. Results columns:
- `rmse_mmHg`, `sbp_err` and `dbp_err` (mmHg)
- `shape_corr`
- `beta_rel_err` and `c0_rel_err`
- `secs`, plus the run settings

## Status
Work in progress. B2 and B3 are complete for seeds 0-2, CPCv2 for seeds 0-1 (106 cases x 4 levels each), B3dc (shift 7.5) for seeds 0-2, and the B3dc shift sweep (2.5, 5, 10) for seed 0. The numbers are not final and are not yet peer reviewed.

## Licence and citation
CC BY 4.0. The cases are fully synthetic and were made with our own solver. They contain no patient data.
Author: Khairuzzaman Mamun, PhD. Please cite the GitHub repository until a paper is available.

Code, simulations and analysis were developed with AI assistance (Anthropic Claude). The author reviews them and is responsible for all claims.
