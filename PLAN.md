# Pulse-PINN research plan

**Status:** plan v1, 2026-11-13. Nothing in phases 2–7 has started.
**Author:** Khairuzzaman Mamun. Code, simulations and analysis are developed with AI assistance (Anthropic Claude) and reviewed by the author.

---

## 1. The question (what)

> How much of the hidden haemodynamics can a physics-informed neural network recover from the kind of data a cheap bedside carotid ultrasound gives? The targets are pressure in the vessel and wall stiffness β. The data is wall motion, and perhaps velocity.

There are three sub-questions:
1. **Identifiability.** Which data level makes pressure and β recoverable, and how accurately?
   - (a) wall motion A(t) only
   - (b) A + velocity
   - (c) A + velocity + one cuff pressure
2. **Method.** Which PINN architecture solves this robustly? Can a problem-specific architecture beat the generic ones?
3. **Reality check.** Applied to real in-vivo carotid data from 10 subjects, does the method give physiologically plausible pressure and β?

## 2. Why this question (why)

- **Clinical value.** Central and carotid pressure, and arterial stiffness, predict cardiovascular risk. The arm cuff measures neither. A catheter does, but it is invasive.
- **The gap.**
  - The closest prior work, Kissas et al. 2020, used 4D-flow MRI, which is expensive.
  - Ultrasound is cheap and everywhere, but it gives wall motion well and velocity poorly.
  - No study we know of asks plainly what is and is not identifiable from ultrasound-type data.
  - A literature check is still needed before writing.
- **Honesty as a contribution.** Showing that β is *not* identifiable from A alone, and what minimal extra measurement fixes that, is a useful result in itself.

## 2b. Real-world (physical and medical) benefit, and how it shapes the plan

The study should be designed backwards from what a doctor or patient would actually gain.

### Who benefits, and how
| Use case | Today | What our method could add | What the output must achieve to matter |
|---|---|---|---|
| **Arterial stiffness screening** (cardiovascular risk, vascular ageing) | Pulse-wave velocity (PWV) with special devices; cIMT ultrasound | β, or local PWV, from a routine carotid ultrasound clip, with no extra device | Separate healthy from stiff. Error clearly smaller than the healthy-vs-stiff difference (stiff ≈ 2–3× β). Repeatable within a subject |
| **Central or carotid blood pressure** | Arm cuff (peripheral only), or invasive catheter | Carotid pressure waveform estimate, needing at most one cuff reading | Systolic/diastolic error within about ±5 mmHg mean and 8 mmHg SD. This is the usual cuff-validation target (AAMI/ISO 81060-2), used here as a reference bar, not a claim |
| **Hypertension follow-up** (treatment effect on vessels) | Cuff BP only | A change in stiffness over months from the same cheap scan | Repeatability good enough to detect a realistic change |
| **Low-resource or rural clinics** (e.g. Bangladesh) | No PWV devices, no MRI | Any portable ultrasound plus software | Robust to operator, probe and noise; runs on a laptop |
| **ICU / perioperative** (longer term) | Arterial line (invasive) | Non-invasive trend monitoring | Out of scope for paper 1; mention as future |

### Physical insight (science benefit)
- It gives a clear answer to *what information wall motion actually carries* about pressure and stiffness. That is the identifiability map, useful for anyone designing ultrasound or wearable sensors.
- It tests whether 1D blood-flow assumptions hold in the carotid against 3D Navier–Stokes, a long-standing modelling question.
- It shows which minimal extra measurement (velocity? one cuff?) turns an unsolvable problem into a solvable one. Device makers can act on that directly.

### Changes to the plan because of this
1. **Clinically meaningful metrics** in Phases 4–6, beyond RMSE:
   - systolic, diastolic and pulse-pressure error in mmHg, against the ±5/8 mmHg bar
   - β and local PWV error, compared with the healthy-vs-stiff gap
   - classification: does the method correctly flag stiff vs healthy across the twin cases (sensitivity/specificity)?
2. **Repeatability test:**
   - Same twin case with different noise or beat windows.
   - In real data, the left vs right side and different recordings of the same subject (the dataset has several per subject). The coefficient of variation is reported.
3. **Stiffness sweep realism:** the β range in the twin set must cover both "young healthy" and "elderly/hypertensive" vessels. Use literature carotid distensibility values to set it.
4. **Derived clinical indices** reported alongside β, because doctors know these, not β:
   - local PWV
   - distensibility coefficient DC = ΔA/(A·ΔP)
   - Peterson modulus
5. **Practical constraints:** report runtime on an ordinary laptop, and how sensitive the result is to probe angle and noise, which reflects operator variability.
6. **Claims discipline:**
   - This is a feasibility/methods study on public data and simulation, not a clinical validation.
   - The paper must say plainly what is still needed for clinical use: a paired cuff/catheter + ultrasound study, more subjects, regulatory steps.
   - We do not claim diagnostic accuracy.
7. **Possible collaboration path:**
   - Clinical validation needs paired data we do not have.
   - Contacting a group with paired ultrasound + pressure data (e.g. TU/e, the dataset authors, or the Sellier/Canterbury group) is a natural next step after the preprint.

## 3. What we already know (data analysis, 2026-11-13)

### Real carotid ultrasound: `zeahub/zea-carotid-2023`
- **Source and licence:** Hugging Face, CC BY 4.0, 10 subjects. Subject 5 has been downloaded (`5_long_2cm_L`, `5_cross_2cm_L`, about 5 GB each).
- **Format:**
  - Raw RF, 150 frames at **30 Hz**, so 5 s, about 6 beats.
  - Each frame has 128 focused lines and 21 plane waves (−18° to +18°, 88.4 µs apart, PRF about 11.3 kHz).
  - fc 7.8 MHz, fs 31.25 MHz.
- **Beamforming:** done with our own plane-wave DAS (`scripts/carotid_bf.py`). The images are good.
- **Longitudinal diameter D(t): good.**
  - 5.85–6.69 mm, about 14% swing, HR about 75–80.
  - Clear systolic upstroke and diastolic decay.
  - The 14% may include tracking noise and needs checking.
- **Cross-section area: moderate.**
  - 42–54 mm², equivalent diameter 7.3–8.3 mm, HR 78.
  - Noisier, and needed a manual seed.
  - It is a separate recording, so it cannot be paired beat by beat with the longitudinal one.
- **Velocity: not usable yet.**
  - The vessel is about 4° from perpendicular to the beam, so axial Doppler sees only about 7% of the flow, plus noise.
  - Vector Doppler from the ±18° steered plane waves is possible, but it is a research task of its own (optional, Phase 6b).
- **Pressure:** none.
- **Spatial span:** about 24 mm of vessel. A pulse wave crosses that in about 5 ms, far below one 33 ms frame. So the real data is effectively **A(t) at one location**, with no resolved wave travel.

### Real ICU arterial pressure: MIMIC-IV Waveform (PhysioNet `mimic4wdb/0.1.0`)
- 6 records, with 10 minutes of ABP each.
- Sampled at **62.5 Hz**, which is coarse: fine features such as the dicrotic notch are blurred.
- SBP 94–153, DBP 41–67, HR 63–104.
- **Use:** realistic pressure ranges and waveform shapes for the simulator's boundary conditions. It is *not* paired with the ultrasound.
- The MAP estimate so far is crude and needs recomputing as the beat-mean.

### Tools
- A commercial solver (Ansys Fluent 18.1) was available but **without a licence**, so it is not used.
- That means our own solver is the main path, and an open-source solver is the cross-check.

## 4. Feasibility: can Claude do the whole study?

Short answer: **yes, with clear risks and gates.**

| Item | Feasible? | Risk | Mitigation |
|---|---|---|---|
| Own 3D incompressible NS solver (GPU, PyTorch) | Yes | **High.** Subtle numerical bugs could invalidate everything | Verification ladder (§5.2) with hard pass criteria before any data is generated; cross-check with OpenFOAM |
| Wall–fluid coupling (moving wall) | Yes, as reduced FSI | Medium. Full 3D FSI is too heavy | Ring-wise thin-wall tube law coupled to local pressure (§5.1). β is then an exact ground-truth parameter |
| Compute | Yes | Medium | Per case about 50k–800k cells and 10⁴–10⁵ steps: minutes to about 2 h on one consumer GPU. Verification and 1D/axisymmetric work also runs on CPU |
| Twin data generation (50–200 cases) | Yes | Low | Scripted sweeps, logs, detached jobs |
| PINN training and ablations | Yes | Medium. PINNs can fail to converge | Staged difficulty, established tricks first, several seeds |
| Real data processing (10 subjects) | Yes | Medium. Automatic tracking failed once on a cross view | Use longitudinal views (robust). Run a QC image for every subject. Use manual seeds where needed |
| Paper writing | Yes, as a draft | Medium | the author must understand and check every claim. Disclose AI assistance per journal policy (Claude cannot be an author) |
| Literature review | Partly | Medium. A novelty claim must be checked | Do a proper search in Phase 0b before claiming novelty |

**Hard limits we accept:**
1. **No in-vivo pressure ground truth.** The real-data result can only be checked for plausibility.
2. **No usable in-vivo velocity** unless Phase 6b succeeds.
3. 30 Hz frame rate.
4. Coarse 62.5 Hz ABP.

**Conditions under which the plan is NOT feasible, and we stop or redesign:**
- The solver fails the verification gates after reasonable effort.
- PINN identifiability results are trivially negative at every data level.

## 5. How: methods

### 5.1 Our own 3D Navier–Stokes solver ("PulseNS")
- **Equations:**
  - Incompressible NS: ρ(∂u/∂t + u·∇u) = −∇p + μ∇²u, with ∇·u = 0.
  - Newtonian first (ρ = 1060 kg/m³, μ = 3.5 mPa·s). Carreau-Yasuda later as an option.
- **Discretisation:**
  - Cartesian staggered (MAC) grid on GPU.
  - Projection method (Chorin / incremental pressure correction).
  - 2nd-order space. Adams–Bashforth or Crank–Nicolson in time.
  - Pressure Poisson equation solved by conjugate gradient with a multigrid or FFT preconditioner.
- **Geometry:**
  - Straight tube first, D₀ about 6 mm, L about 40–60 mm.
  - Later options: a mild taper, and the carotid bifurcation.
- **Wall:**
  - Immersed boundary (direct forcing or Brinkman penalisation), so the moving wall needs no remeshing.
- **Reduced FSI:**
  - At each axial station, wall radius follows the local cross-section mean pressure through the tube law p − p_ext = β(√A − √A₀).
  - Optional wall inertia and viscosity terms.
  - This gives real wave propagation, and β is a known ground truth.
- **Boundary conditions:**
  - Inlet: a literature carotid flow waveform, scaled by HR and amplitude, with a Womersley profile.
  - Outlet: a 3-element Windkessel (R₁, C, R₂), tuned so the pressure range matches MIMIC (systolic about 100–150, diastolic about 40–80 mmHg).
- **Outputs, saved every 1 ms then downsampled:**
  - 3D fields.
  - Cross-section-averaged A(x,t), Q(x,t), ū(x,t), p̄(x,t).
  - The wall position.

### 5.2 Verification ladder (gates, all must pass before Phase 3)
1. Method of Manufactured Solutions: observed convergence order about 2.
2. Steady Poiseuille: velocity profile error < 1%.
3. Womersley pulsatile flow in a rigid tube (analytic): error < 2% in amplitude and phase.
4. Mass conservation: inflow minus outflow balanced by dA/dt to < 0.1%.
5. Wave speed in the compliant tube matches theory, c = √(β√A₀ / 2ρA₀) (thin-wall tube-law form), within 5%.
6. Grid and time-step convergence on the final case: < 2% change in key outputs.
7. Optional cross-check with OpenFOAM (free, via WSL) on the rigid Womersley case, and if possible a prescribed-moving-wall case.

### 5.3 Twin dataset ("the simulator acts like the scanner")
- **Parameter sweep, Latin hypercube, 50–200 cases:**
  - β (healthy to stiff, about 3×)
  - D₀ 5–8 mm
  - HR 55–110
  - inflow amplitude
  - Windkessel R and C
- **Ultrasound-like sampler:**
  - 30 Hz frames and a 24–38 mm window.
  - The D(t) noise model is fitted to the real tracking residuals.
  - Optional velocity "measurements" with a realistic angle and noise.
- **Ground truth kept hidden from the PINN:** p(x,t), β, and the true u.

### 5.4 PINN formulation
- The real data is A at one location over time, so the main PINN works on the **1D reduced model** in (x,t). That model is mass, momentum (with a profile and friction closure) and the tube law, over the simulated segment.
- The 3D solver supplies the truth, and a test of the 1D assumptions themselves.
- **Data levels (the identifiability experiment):**
  - (a) A(x₀,t)
  - (b) A + ū
  - (c) A + ū + one cuff pressure (systolic and diastolic only, as a real cuff gives)
  - (d) as (c) but with the cuff at a distant site, which is a realistic option
- **Unknowns:** p(x,t), β (scalar, or β(x)), Windkessel parameters, inlet flow (when not given).
- **Metrics:**
  - pressure RMSE and systolic/diastolic error in mmHg
  - β relative error
  - calibration of uncertainty
  - runtime

### 5.4b Physics fidelity levels inside the PINN (added 2026-11-13, on review)
| Level | PINN physics | Inputs → outputs | Role |
|---|---|---|---|
| L1 | 1D reduced model (cross-section averaged NS + profile/friction closure + tube law) | (x,t) → A, Q (via Ψ), p from the tube law | Fast; clinic-style; needs the closure assumption |
| **L2 (main candidate)** | **Axisymmetric incompressible NS** (r,z,t), no closure | (r,z,t) → u_r, u_z, p (continuity exact via the Stokes stream function ψ: u_z = (1/r)∂ψ/∂r, u_r = −(1/r)∂ψ/∂z) | The wall boundary R(t) comes straight from ultrasound D(t)/2 (no-slip on the moving wall). The tube law at the wall gives absolute p and β. The Womersley profile and wall shear come out of NS itself |
| L3 | Full 3D NS | (x,y,z,t) → u, v, w, p | Upper bound, on twin data only (where velocity fields can be given); too costly and under-determined for real data |

- **Why:** "How much physics fidelity is needed?" becomes a result in its own right.
- **CPC ideas carry over to L2:**
  - periodic harmonic time embedding
  - hard tube law at the wall
  - continuity by stream function
- The twin data from the 3D solver is the truth for all three levels.

### 5.5 Training architecture: baselines (all re-implemented, same budget)
1. Vanilla MLP PINN (tanh).
2. Fourier-feature PINN (Tancik / Wang).
3. Modified MLP with two gating side networks (Wang, Teng & Perdikaris 2021). These are the "side networks" mentioned in the blog post.
4. Adaptive activation slopes (Jagtap et al.).
5. Adaptive loss weighting: gradient-norm / learning-rate annealing, and NTK-based.
6. Causal training for time-dependent problems (Wang et al. 2022).
7. PirateNet (residual adaptive network, 2024).
8. Optional: KAN-based PINN (PIKAN), XPINN domain decomposition.

### 5.6 New architecture ideas (candidates to develop and ablate)
**A. "Cardiac-periodic conservative PINN" (CPC-PINN), the main candidate**
- *Periodic embedding:*
  - The time input is replaced by harmonics sin(kωt) and cos(kωt), k = 1..K.
  - ω comes from the data's HR and is fine-tuned as a trainable parameter.
  - Exact periodicity is then built in, with no periodic penalty term.
- *Mass conservation by construction:*
  - The network outputs a potential Ψ(x,t), with A = A₀ + ∂Ψ/∂x and Q = −∂Ψ/∂t.
  - ∂A/∂t + ∂Q/∂x = 0 then holds exactly, leaving only the momentum residual.
- *Tube law as a hard constraint:*
  - p = p_ext + β(√A − √A₀) is computed analytically, not learned.
  - β is a trainable scalar, or a small field.
- **Effect:** two of the three physics laws become exact. The loss has only data and momentum, which should give easier optimisation and better identifiability.

**B. Learned friction and profile closure from 3D**
- The 1D model needs a guess for the velocity-profile and friction term.
- Train a tiny network closure f(Re, Womersley α, A, Q) on the 3D twin data. Use it inside the 1D PINN.
- This tests whether "physics + learned closure" beats the textbook closure.

**C. Uncertainty-aware PINN**
- Deep ensembles or a Bayesian last layer, giving credible intervals for β and p.
- An identifiability failure then shows up as wide or flat posteriors, a strong way to present the result.

**D. Pretrained or meta-learned PINN (extension; possibly paper 2)**
- Pretrain on many twin cases, a neural-operator option (DeepONet/FNO).
- Then fine-tune with physics loss on each real subject: seconds instead of minutes.

**Ablation plan:**
- Add components A1 (periodic), A2 (Ψ-mass), A3 (hard tube law), B, C one at a time against the baselines.
- Use 5 seeds and the same compute budget.
- Report accuracy, convergence speed and failure rate.

### 5.7 Real-data application
- Download the longitudinal views of all 10 subjects (about 5 GB per file).
- Run the same beamforming and tracking pipeline, with a QC image per subject.
- Fit the PINN at each data level that is realistic for real data: (a), and (c) with a population-typical cuff value labelled as an assumption.
- Report plausibility: pressure ranges, β ranges compared with the literature, and consistency across beats.
- Optional Phase 6b: try vector Doppler from the steered plane waves to get velocity, enabling level (b) on real data.

## 6. Phases, gates and rough time

| Phase | Work | Gate to pass | Rough time |
|---|---|---|---|
| 0 | Data download and first analysis | — | **Done** (2026-11-13) |
| 0b | Literature search: PINN haemodynamics, ultrasound pressure estimation, identifiability | Novelty statement holds | 1 session |
| 1 | Full data characterisation (10 subjects), D(t) noise model, MIMIC beat-mean MAP | QC images OK | 1–2 sessions |
| 2 | PulseNS solver, verification ladder | All §5.2 gates pass | 3–6 sessions (biggest risk) |
| 3 | Twin dataset generation | Physiological ranges match MIMIC and the carotid data | 1–2 sessions + GPU time |
| 4 | Baseline PINNs + identifiability (levels a–d) | Clear, reproducible result table | 2–3 sessions |
| 5 | New architecture (A–C) + ablation | An improvement over baselines, or an honest null result | 2–4 sessions |
| 6 | Real data (10 subjects) | Plausible outputs, QC documented | 1–2 sessions |
| 7 | Figures, paper draft, arXiv | author review | 2–3 sessions |

**Target venues** (decide after the results):
- Journal of Biomechanics
- Biomechanics and Modeling in Mechanobiology
- Journal of the Royal Society Interface
- Computer Methods in Applied Mechanics and Engineering (if the architecture result is strong)

Post a preprint on arXiv first.

## 7. Working rules for this project
- Everything runs by script, as detached jobs with log files; check rarely (saves tokens).
- Every result must be reproducible from scripts in this folder, with a fixed seed and a config file per run.
- Keep `NOTES.md` (a run log) updated after every phase. This plan is updated when decisions change.
- Each phase's design is reviewed by the author before its code is written.
- Attribution and licences:
  - zea-carotid is CC BY 4.0: cite it and mark modifications.
  - MIMIC-IV waveform: cite PhysioNet and follow its licence terms.

