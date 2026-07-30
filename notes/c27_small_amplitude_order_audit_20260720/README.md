# C27 small-amplitude Craig--Sulem order audit

This audit tests the local surface-amplitude order of the accepted C27 final
checkpoint. It is an operator test, not a rollout test and not a substitute
for a separately trained no-$G_1$ ablation.

## Protocol

For each of the ten registered evaluation families, eight frame-zero truth
states were selected by sorting the 32 cases by `(depth, case_id)`, dividing
that ordering into eight equal-count strata, and taking each stratum midpoint.
This gives 80 fixed states $(\eta_j,\xi_j,h_j)$. For

$$
\varepsilon\in\{1,1/2,1/4,1/8,1/16,1/32\},
$$

only the surface was scaled: $\eta_j\mapsto\varepsilon\eta_j$; $\xi_j$ and
$h_j$ were held fixed. Define

$$
q_j^{(6)}(\varepsilon)=G^{(6)}(\varepsilon\eta_j;h_j)\xi_j,
\qquad q_{0,j}=G_0(h_j)\xi_j,
\qquad q_{1,j}=G_1(\eta_j;h_j)\xi_j,
$$

where $G^{(6)}$ uses float64 and pad-eight products. The trained-model Taylor
remainder is evaluated as

$$
r_{\theta,j}(\varepsilon)
=G_\theta(\varepsilon\eta_j;h_j)\xi_j
-G_\theta(0;h_j)\xi_j
-\varepsilon D_\eta G_\theta(0;h_j)[\eta_j]\xi_j.
$$

Every norm below is projected to $|k|\leq128$ and divided by
$\|q_{0,j}\|_{L^2}$. A log--log slope is fitted separately for every state on
$\varepsilon\in\{1/4,1/8,1/16,1/32\}$. The reported point estimate is the
median of the 80 statewise slopes. The confidence interval is a 5,000-sample
bootstrap that resamples within each family stratum.

## Results

| quantity | expected order | median slope | stratified 95% bootstrap CI |
|---|---:|---:|---:|
| $\|q^{(6)}-q_0\|/\|q_0\|$ | 1 | 1.000164 | [0.999997, 1.000550] |
| $\|q^{(6)}-q_0-\varepsilon q_1\|/\|q_0\|$ | 2 | 2.000026 | [1.999994, 2.000097] |
| $\|r_\theta(\varepsilon)\|/\|q_0\|$ | 2 | 1.966587 | [1.943305, 1.991166] |

At full scale, the pooled median $G_0$ truncation is $1.5860\times10^{-2}$,
whereas the pooled median $G_0+G_1$ truncation is
$1.6745\times10^{-3}$, a factor of about 9.47 smaller. The reference
$G_0+G_1$ remainder has essentially exact quadratic scaling in every family;
family median slopes range from 1.9937 to 2.0003.

The trained C27 remainder is also nearly quadratic. Its pooled estimate is
slightly below two because it is formed by subtracting float32 deployed-model
fields; the weakest finite-depth Stokes remainders approach that subtraction
floor at the smallest amplitudes. The structural statement
$R_\theta(0)=D_\eta R_\theta(0)=0$ remains exact by construction. The separate
JVP check confirms that the deployed first derivative agrees with the padded
analytic $G_1$: its $G_1$-relative error has median $1.04\times10^{-6}$ and
95th percentile $1.63\times10^{-5}$. The largest relative value, 0.0165, is a
low-signal deep-Stokes state with $\|G_1\|=4.67\times10^{-7}$; its absolute
projected RMS error is only $7.73\times10^{-9}$, or $9.96\times10^{-7}$ of
$\|G_0\xi\|$.

Thus the experiment directly supports the intended local hierarchy:

$$
G^{(6)}(\varepsilon\eta)\xi-G_0\xi=O(\varepsilon),
\qquad
G^{(6)}(\varepsilon\eta)\xi-G_0\xi-\varepsilon G_1(\eta)\xi
=O(\varepsilon^2),
$$

and C27 does not reintroduce an order-zero or order-one learned correction.
It does not, by itself, estimate the causal rollout benefit of $G_1$; that
question belongs to the matched trained no-$G_1$ control.

The complete run took 16.21 seconds on JAX's CPU backend at nice level 10,
restricted to logical CPUs 16--31. No GPU was visible to the process.
