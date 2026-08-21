# Literature scan: long-rollout stability for neural PDE surrogates

Run: 2026-06-04. Scope: techniques for stabilising autoregressive rollouts
of stiff dispersive PDE surrogates (esp. water-wave / Zakharov-like).

## Top-3 to try next (ranked by expected gain × implementation speed)

### 1. K-step unrolled training with WIG (with-gradient) + curriculum, k = 4–8

- **What:** Replace the 1-step pushforward with multi-step unrolling where
  gradients flow through *all* k steps, ramping k via curriculum (1 → 2 → 4 → 8).
- **Reference:** List, Chen, Bali, Thuerey, *"How Temporal Unrolling Supports
  Neural Physics Simulators"* (arXiv:2402.12971, 2024).
- **Why for us:** The TUM group's empirical finding is that m ≤ 4 doesn't
  stabilise their flows and m ≥ 8 does, and that **stop-gradient (our current
  1-step pushforward) has a bias scaling like m²** vs. true gradient — exactly
  the "coherent linear-in-t bias" we are seeing. PDE-Refiner's paper notes
  pushforward "cannot learn to include low-amplitude information" because no
  gradient passes through the predicted input.
- **Sketch:** Currently `loss = MSE(model(stop_grad(model(u))), target)`.
  Change to `for i in range(k): u = model(u); loss += w_i * MSE(u, target_i)`
  with no stop_grad. Memory ~linear in k; at width=128/modes=64, k=4–8 fits
  on a single H100 with batch_size halved. Curriculum: k=1 for ~30% epochs,
  double every 20%.
- **Risk:** Gradient explosion on chaotic Benjamin-Feir IC. Mitigate with
  gradient clipping + tanh-style saturation on h (already have h_clip_max=5.0).
  Fallback: NOG (gradients only through the last step of an unrolled rollout).
- **Time:** 2–3 h.

### 2. PDE-Refiner (diffusion-style refinement of residual)

- **What:** Instead of predicting next state in one shot, predict it in K
  denoising-style refinement steps with an exponentially-decaying noise
  schedule. Same model called K times conditioning on noise level.
- **Reference:** Lippe et al., *PDE-Refiner*, NeurIPS 2023 (arXiv:2308.05732).
  Follow-up: Yu et al., *PDESpectralRefiner* (arXiv:2506.10711, 2025) with
  non-uniform spectral band weighting.
- **Why for us:** Lippe's diagnosis is that standard MSE training neglects
  low-amplitude high-k modes — and those modes drive modulational instability
  in Zakharov. If our bias on Tanaka solitons is from misfit at exactly the
  wavenumbers that don't dominate L2 (sidebands, near-Nyquist tails), this
  directly attacks the cause. KS results showed +30% rollout horizon.
- **Sketch:** Add a noise-level input channel c = log(σ_k). Train: sample
  k ∈ {0..K}, add noise σ_k·ε to the residual target, train to denoise.
  Rollout: K iterative passes per timestep. K=3–4 works in the paper.
- **Risk:** Paper reports gain is **smaller on FNO than U-Net** because
  FNO doesn't model high-freq noise as cleanly. With spectral DNO this is a
  concern, but modes=64 still gives more spectral room than vanilla FNO.
  Inference cost goes up Kx; budget 3–4× rollout time.
- **Time:** 2–3 h to wire up + retraining.

### 3. Predict tendency (∂u/∂t) + RK4 at inference, not next-state

- **What:** Reframe the DNO target as the temporal derivative, then integrate
  with a real RK4/AB-2 stepper at rollout time.
- **Reference:** Bryutkin & Karniadakis, *"Predicting Change, Not States: An
  Alternate Framework for Neural PDE Surrogates"* (arXiv:2412.13074, 2024).
- **Why for us:** Zakharov is stiff and dispersive — predicting state in one
  shot at dt=0.01 forces the network to internalise both fast dispersive
  phase rotation *and* slow nonlinear envelope. Splitting via an external
  integrator lets the network learn just the (smoother) tendency. On
  chaotic KS: FNO correlation time 140 → 184 steps. On NS: rollout L2
  0.715 → 0.100. "Error remains stable or decreases" with horizon — exactly
  the failure mode we are fighting.
- **Sketch:** Training target becomes (u_{t+dt} - u_t)/dt (or higher-order
  finite diff). Loss unchanged. Inference: replace `u = model(u)` with an
  RK4 wrapper calling `model(u)` 4× per step. No new params, no curriculum.
- **Risk:** 4× rollout cost. Choice of finite-diff scheme matters
  (Richardson or 4th-order). Best paired with #1 above.
- **Time:** 2 h — smallest code diff of the three.

## Other promising directions (one-liners)

- **PhysicsCorrect** (arXiv:2507.02227, 2024) — training-free; projects
  rollout state onto PDE-residual manifold via cached Jacobian pseudoinverse.
  Cheap rescue for an already-trained checkpoint; Zakharov has cheap
  spectral residual eval.
- **SGNO / spectral truncation on the residual** (arXiv:2602.18801, 2026) —
  smooth mask on high-k feedback during rollout; ~10 lines of code, may
  stop high-k blow-up on random-sea ICs.
- **Mass/energy hard projection** (Hu et al., arXiv:2505.24579, 2025) —
  post-step projection onto conserved scalars; complements the in-progress
  H-loss task and removes drift without retraining.
- **JAWS spatially-adaptive Jacobian regularisation** (arXiv:2603.05538, 2026)
  — softer than spectral norm penalty, adapts to local stiffness; relevant
  where Stokes crests have sharp gradients.
- **NOG variant of #1** — unroll forward, gradient only on last step; same
  memory as 1-step pushforward, captures most of WIG's benefit. Try if k=4
  WIG OOMs.

## Skip list

- **Hard-symplectic / strict Hamiltonian architectures.** Equations are
  damped/forced at the discretisation level; canonical (η, ξ) structure
  isn't easily preserved through a Fourier modes layer; benchmarks are
  N-body / pendulum, not stiff dispersive PDEs. A *soft* H-conservation
  loss (in-progress) is the right level; don't go architectural.
- **Global Lipschitz / spectral-norm constraints on the operator.** Useful
  for diffusive flows; for dispersive problems, the true operator's
  L2-induced norm is exactly 1 (unitary phase rotation), so the constraint
  either does nothing useful or kills modulational dynamics we actually want.
- **Latent / flow-matching generative rollout** (arXiv:2503.22600 etc.).
  Heavy implementation, paper-deadline-hostile, and reported gains are
  mostly on visually-textured 2D NS — not our problem mode (coherent
  linear-in-t bias).

## Recommended order for our deadline

Run #1 first (k-step WIG + curriculum). Highest leverage, directly attacks
the failure mode pushforward leaves on the table. If clean win, stop.
If Tanaka drift persists, layer in #3 (tendency target). Save #2
(PDE-Refiner) for if #1+#3 don't close the gap.

## Sources

- [PDE-Refiner (arXiv:2308.05732)](https://arxiv.org/abs/2308.05732)
- [PDE-Refiner project page](https://phlippe.github.io/PDERefiner/)
- [PDESpectralRefiner (arXiv:2506.10711)](https://arxiv.org/abs/2506.10711)
- [How Temporal Unrolling Supports Neural Physics Simulators (arXiv:2402.12971)](https://arxiv.org/abs/2402.12971)
- [Thuerey Group blog post on unconditionally stable autoregressive operators](https://ge.in.tum.de/2024/08/05/how-to-train-unconditionally-stable-autoregressive-neural-operators/)
- [PhysicsCorrect (arXiv:2507.02227)](https://arxiv.org/abs/2507.02227)
- [Predicting Change, Not States (arXiv:2412.13074)](https://arxiv.org/abs/2412.13074)
- [SGNO (arXiv:2602.18801)](https://arxiv.org/abs/2602.18801)
- [JAWS (arXiv:2603.05538)](https://arxiv.org/abs/2603.05538)
- [Adaptive Correction for Conservation Laws (arXiv:2505.24579)](https://arxiv.org/abs/2505.24579)
- [Message Passing Neural PDE Solvers (arXiv:2202.03376)](https://arxiv.org/abs/2202.03376)
