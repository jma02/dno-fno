# Highest Tanaka-v2 sign-count case

Date: 2026-07-23

## Question

The largest value of the historical sign-transition count in
`data/tanaka_2_adaptive_g0.npz` is \(162\). This note determines whether that
value is caused by an error in the stored \(G(\eta)\xi\) label, instability of
the order-six Craig--Sulem evaluation, growth of unresolved modes during the
rollout, or an artifact already present in the initial condition.

The answer is:

1. the order-six label is correct for the state stored in the archive;
2. no high-frequency energy surge occurs at the offending frame;
3. two counterpropagating waves nearly cancel in the low modes of
   \(q=G(\eta)\xi\), exposing a small high-frequency floor;
4. that floor is seeded at \(t=0\) by the piecewise-linear interpolation used
   to place the smooth Tanaka profiles on the periodic grid;
5. \(G_0\) and the derivative used by the sign diagnostic magnify the small
   state-space artifact, making it visually prominent in \(q_x\).

Thus the oscillation is most visible in \(G(\eta)\xi\), but it is not created
there.

## 1. Exact row

The row is in batch 0:

| quantity | value |
| --- | ---: |
| case ID | 31 |
| local saved sample | 76 |
| dense rollout index | 954 |
| time | \(76.32\) |
| depth | \(0.26861435\) |
| historical sign count | \(162\) |

It is the bottom row of
[`v2_old_signrule_examples_20260723.png`](v2_old_signrule_examples_20260723.png).

The initial condition is the superposition of two nearly equal solitary waves
traveling in opposite directions:

| \(a/h\) | center | direction |
| ---: | ---: | ---: |
| \(0.05548354\) | \(3.55484969\) | \(-1\) |
| \(0.05353141\) | \(5.03270393\) | \(+1\) |

Their speeds are approximately \(0.532324\) and \(0.531822\). On a domain of
length \(2\pi\), their encounter period is therefore

\[
T_{\mathrm{enc}}
  = \frac{2\pi}{0.532324+0.531822}
  = 5.9044.
\]

The large sign counts recur at approximately this period. They are tied to
the head-on encounters, not to monotone deterioration of the numerical
solution.

## 2. What the historical count measures

For grid values

\[
q_j=[G(\eta)\xi](x_j),
\]

the historical cleaner formed the cyclic forward difference

\[
(D_+q)_j=\frac{q_{j+1}-q_j}{\Delta x},
\]

discarded differences smaller than \(3\%\) of
\(\max_j |(D_+q)_j|\), and counted cyclic changes of sign among the remaining
values.

For a Fourier mode \(e^{ikx}\), the forward difference has multiplier

\[
\frac{e^{ik\Delta x}-1}{\Delta x}=ik+O(k^2\Delta x).
\]

At this depth and for the modes at issue,

\[
\widehat{G_0\xi}_k
  = |k|\tanh(h|k|)\widehat{\xi}_k
  \simeq |k|\widehat{\xi}_k.
\]

Consequently, the quantity inspected by the sign counter scales approximately
as

\[
\widehat{D_+G_0\xi}_k \simeq ik|k|\widehat{\xi}_k.
\]

The plotted diagnostic therefore amplifies a small mode of \(\xi\) by roughly
\(k^2\). This explains why an almost invisible state-space tail can look
large in a plot of \(q_x\).

## 3. Frame-by-frame evidence: cancellation, not growth

At \(t=76.32\),

\[
\operatorname{RMS}(q)=1.3216\times 10^{-4},
\qquad
\operatorname{RMS}(\xi)=3.5389\times 10^{-4}.
\]

Both are the global minima among the 200 saved frames of this trajectory:
they are respectively \(3.22\%\) and \(2.78\%\) of their trajectory medians.
The sign count is largest precisely when the dominant low-frequency
contributions nearly cancel.

The absolute high-frequency content does not increase at this frame:

| modes \(|k|>32\) | value at \(t=76.32\) | trajectory median | ratio |
| --- | ---: | ---: | ---: |
| \(\eta\) discrete \(L^2\) | \(4.2242\times10^{-7}\) | \(4.2883\times10^{-7}\) | \(0.985\) |
| \(\xi\) discrete \(L^2\) | \(4.7410\times10^{-8}\) | \(4.6990\times10^{-8}\) | \(1.009\) |
| \(q\) discrete \(L^2\) | \(4.3011\times10^{-6}\) | \(4.2643\times10^{-6}\) | \(1.009\) |

Across the trajectory, the sign count has correlation \(0.878\) with
\(-\log_{10}\operatorname{RMS}(q)\), but correlation only \(0.054\) with the
absolute high-band amplitude. The linear Hamiltonian in modes \(33\) through
\(128\) has coefficient of variation \(0.532\%\), and its value at the
offending frame is \(0.9995\) times its median. The maximum full Hamiltonian
drift is \(4.81\times10^{-9}\).

These observations rule out a spectral cascade. A small tail is transported
almost conservatively; cancellation only changes its size relative to the
low modes.

## 4. Direct audit of \(G(\eta)\xi\)

Recomputing the label on CPU from the stored float32 state, using float64,
Craig--Sulem order six, padding factor eight, and the production
\(|k|\leq128\) projection, gives

\[
\frac{\|q_{\mathrm{recomputed}}-q_{\mathrm{stored}}\|_2}
     {\|q_{\mathrm{stored}}\|_2}
=2.80\times10^{-6}.
\]

The recomputed and stored labels both have sign count \(162\). Padding factors
\(2,4,8,16\) agree to approximately \(10^{-15}\) in float64. Embedding the
same band-limited state on \(N=2048\), evaluating there, and restricting back
to \(N=1024\) changes the result by \(3.30\times10^{-14}\) relatively.

The Craig--Sulem decomposition is:

| contribution | discrete \(L^2\) norm |
| --- | ---: |
| \(G_0\xi\) | \(1.23016\times10^{-4}\) |
| \(G_1(\eta)\xi\) | \(1.00408\times10^{-5}\) |
| \(G_2(\eta)\xi\) | \(2.31363\times10^{-7}\) |
| \(G_3(\eta)\xi\) | \(3.93205\times10^{-9}\) |
| \(G_4(\eta)\xi\) | \(2.82914\times10^{-10}\) |
| \(G_5(\eta)\xi\) | \(2.35876\times10^{-11}\) |
| \(G_6(\eta)\xi\) | \(1.39360\times10^{-12}\) |

\(G_0\) alone gives sign count \(168\); adding \(G_1\) changes it to \(162\).
Above mode 32, \(G_0\xi\) and the complete order-six result have correlation
\(0.9999933\). At mode 104,

\[
|\widehat{\xi}_{104}|=8.42579\times10^{-9},\qquad
104|\widehat{\xi}_{104}|=8.76283\times10^{-7},
\]

while the stored value is

\[
|\widehat q_{104}|=8.76230\times10^{-7}.
\]

Modes above 32 contain only \(0.106\%\) of the energy of \(q\), but they contain
\(77.93\%\) of the energy of \(q_x\). Hence the visual roughness is almost
entirely the elementary \(|D|\) amplification of a small tail of \(\xi\), not
an order-six failure.

## 5. Where the tail enters

The v2 constructor solves for a smooth Tanaka profile at 257 collocation
points. It then places each profile with `jnp.interp`, which is
piecewise-linear, evaluates that interpolant on an \(8N\) grid, and truncates
its Fourier series to the \(N\)-grid modes.

Oversampling prevents aliasing above the target Nyquist frequency, but it does
not remove the corners in the derivative of the piecewise-linear interpolant.
It instead resolves their Fourier tail accurately inside the retained band.

For the exact case-31 specification at \(t=0\), changing only the interpolation
of the same Tanaka knots gives:

| construction from 257 knots | \(\|\widehat\eta_{80:128}\|_2\) | \(\|\widehat\xi_{80:128}\|_2\) |
| --- | ---: | ---: |
| current linear interpolation | \(3.51994\times10^{-7}\) | \(7.31912\times10^{-9}\) |
| shape-preserving cubic Hermite | \(1.10690\times10^{-8}\) | \(2.08159\times10^{-10}\) |
| natural cubic spline | \(3.98966\times10^{-9}\) | \(7.45165\times10^{-11}\) |

Thus smooth interpolation reduces the initial high-band seed by factors of
approximately \(32\)--\(98\), while changing the full low-frequency state by
only about \(1.6\times10^{-4}\) relatively.

Merely increasing the fine placement grid from \(N=1024\) to \(2048\) or
\(4096\) does not reduce the floor: all three grids converge to the same
piecewise-linear function. Increasing the Tanaka solve from 257 to 513
collocation points helps, but smooth interpolation is the decisive change:

| collocation points and interpolation | \(\|\widehat\eta_{80:128}\|_2\) | \(\|\widehat\xi_{80:128}\|_2\) |
| --- | ---: | ---: |
| 257, linear | \(3.51994\times10^{-7}\) | \(7.31912\times10^{-9}\) |
| 513, linear | \(3.02012\times10^{-8}\) | \(4.56598\times10^{-10}\) |
| 513, shape-preserving cubic Hermite | \(3.97761\times10^{-9}\) | \(7.38490\times10^{-11}\) |
| 513, natural cubic spline | \(3.92111\times10^{-9}\) | \(7.32194\times10^{-11}\) |

The current \(N,2N,4N\) fixed-band audit does not expose this particular defect
when all three constructions use the same 257-knot linear interpolant. It
correctly verifies convergence to the declared piecewise-linear construction,
not that the construction faithfully preserves the smoothness of the
continuous Tanaka profile. This is a limitation of that audit, rather than a
contradiction of its earlier results.

## 6. What is ruled out

The evidence rules out the following explanations for this case:

- order-six Craig--Sulem truncation;
- insufficient convolution padding;
- the \(N=1024\) DNO evaluation grid;
- float32 storage of an otherwise clean label;
- a time-integrator instability;
- growth or transfer of energy into high modes at the collision;
- the previously identified unprojected saved-\(q\) evaluation bug.

Lowering the label cutoff is not a satisfactory correction. Projection to
\(|k|\leq64\) reduces the sign count to \(8\), but removes \(3.19\%\) of the
label in \(L^2\) at the cancellation frame. It suppresses the symptom and can
also erase physical modes in other families.

## 7. Recommended correction

The smallest principled generator change is:

1. replace the piecewise-linear placement of the Tanaka profile by cubic
   Hermite reconstruction using the native slopes
   \(\eta_x=\tan\theta\);
2. initially retain the existing 257 nonnegative Tanaka collocation nodes,
   which give 513 full nodes after reflection;
3. retain the existing three-copy periodic construction and fixed-band
   projection;
4. validate the change on a depth-and-steepness-stratified panel before
   regenerating the Tanaka corpus.

The exact paired \(T=200\) case-31 test has now been completed. Tangent-Hermite
placement changes the count at \(t=76.32\) from \(162\) to \(4\), never exceeds
\(6\) over the dense trajectory, and preserves Hamiltonian drift at
\(2.39\times10^{-12}\). The complete follow-up is
[`tanaka_case31_tangent_resampling_trial_20260723.md`](tanaka_case31_tangent_resampling_trial_20260723.md).
Increasing the collocation count is therefore optional convergence insurance,
not the operative correction.

The old sign count should remain a historical diagnostic, not a universal
rejection rule. This row illustrates why: it combines a genuine, removable
construction floor with a physically valid cancellation that makes any
relative oscillation count unusually sensitive.
