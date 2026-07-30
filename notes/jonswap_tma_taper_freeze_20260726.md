# JONSWAP/TMA taper decision

## Decision

Keep and freeze the cosine-squared density window

\[
W(k)=
\begin{cases}
1,&0\leq k\leq96,\\
\cos^2\!\left(\dfrac{\pi}{2}\dfrac{k-96}{32}\right),
  &96<k<128,\\
0,&k\geq128.
\end{cases}
\]

This window is part of the initial-condition definition. It is not a
realization-dependent filter and is not an acceptance rule.

## Why 96 is a support-based choice

The declared random-sea support satisfies \(k_p\leq24\). For \(r\geq1\),

\[
\frac{\omega(rk_p,h)}{\omega(k_p,h)}
=
\left[
r\frac{\tanh(rk_ph)}{\tanh(k_ph)}
\right]^{1/2}
\geq\sqrt r.
\]

Therefore \(K_0=96=4\max k_p\) leaves the spectrum untapered at least through
\(2\omega_p\) for every allowed case. The interval \(96<k<128\) then gives
32 Fourier cells on the \(2\pi\)-periodic domain over which the density and
its first derivative decay to zero. These two elementary facts explain the
numbers without inspecting a realized wave.

## Endpoint comparison

The CPU-only audit sampled 32 deterministic cases from each of the 27
population cells, for 864 cases. It reconstructed fields on 512 points and
performed no time integration. The comparison used a practically untapered
\(k\leq128\) spectrum as a reference. The current window retains almost all
of that spectral mass:

| depth stratum | median retained mass | minimum retained mass | median spectral total variation | 95th percentile |
| --- | ---: | ---: | ---: | ---: |
| shallow | 99.226% | 98.469% | 0.767% | 1.466% |
| finite | 99.913% | 99.660% | 0.0869% | 0.264% |
| deep | 99.914% | 99.653% | 0.0861% | 0.257% |

Earlier tapers reduce slope, but they also change the sampled surface. The
following entries are medians. The first number is ensemble RMS slope divided
by the value under the current \(96\)--\(128\) window; the second is the
realized relative \(L^2\) change in \(\eta\).

| taper | shallow | finite | deep |
| --- | ---: | ---: | ---: |
| \(48\)--\(64\) | 0.801 / 0.282 | 0.861 / 0.090 | 0.861 / 0.092 |
| \(64\)--\(96\) | 0.905 / 0.140 | 0.934 / 0.045 | 0.934 / 0.045 |
| \(80\)--\(112\) | 0.957 / 0.071 | 0.970 / 0.023 | 0.970 / 0.023 |
| \(96\)--\(128\) | 1.000 / 0.000 | 1.000 / 0.000 | 1.000 / 0.000 |

Thus moving the endpoint mainly smooths the fields by deleting prescribed
tail energy. Even the moderate \(80\)--\(112\) alternative changes a typical
shallow elevation by about 7% while reducing its ensemble RMS slope by only
about 4%. There is no numerical failure that requires this change.

## Independent numerical evidence

- The 27-case constructor pilot has maximum fixed-band \(N=512\) versus
  \(N=1024\) DNO discrepancy \(4.11\times10^{-4}\).
- Its short paired-GL2 check has maximum defect \(3.85\times10^{-8}\).
- The shallow, finite, and deep full-horizon cases all reach the last saved
  time not exceeding 16 peak periods. Their paired defects are
  \(4.52\times10^{-6}\), \(5.63\times10^{-6}\), and
  \(4.75\times10^{-6}\), respectively, and every implicit stage is solved.
- Direct \(N=512,1024,2048,4096\) reconstructions agree at shared points.
  The visible fine scales are therefore the resolved spectral tail, not
  spatial aliasing.

The machine-readable endpoint audit is
[`jonswap_tma_taper_freeze_20260726.json`](jonswap_tma_taper_freeze_20260726.json).
The calculation is reproduced by

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/audit_jonswap_tma_taper.py --cases-per-cell 32
```

## Production consequence

The code now names the window as `cosine_squared_density_v1`, fixes the paper
defaults \(K_0=96\), \(K=128\), and 16-point cell quadrature, and records the
window name in every durable proposal. A different taper is a new generator
revision, not a silent configuration change.
