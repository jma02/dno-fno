# Neural DNO optimization report

Status as of October 21, 2026. Training and the accuracy-search goal are paused at the user's request.

The substantial verified gain is execution speed: the trained hero now runs about **3–4% faster than the optimized M3 solver on Ada**, with its weights unchanged. It remains **21–23% slower than M2**. Accuracy is the remaining obstacle: on our standard cohort, the hero beats M2's median error in all four families and M3's in Tanaka and Benjamin–Feir, but not in Stokes or JONSWAP/TMA. The latest training experiments have not replaced this model.

## Current speed and accuracy

The accepted model is the epoch-37 primary weight-EMA checkpoint of the hero run. Its optimized evaluation is saved in [contextreal_eval](../outputs/hero_speed_20261019/contextreal_eval/). The original one-epoch hero is a separate checkpoint, not the trained hero.

### Full single rollout runtime

Seconds on the local RTX 6000 Ada, batch size one. Each entry is one synchronized full-trajectory measurement after a full warm-up; compilation, graph capture and host transfers are excluded. The workload uses 1,024 spatial points, retained modes through 128, internal timestep 0.01, four Gauss–Legendre fixed-point iterations and 251 saved frames. Stokes/JONSWAP run to T=20; Tanaka/BF to T=200.

| Family | M2 | Trained hero | M3 |
| --- | ---: | ---: | ---: |
| Stokes | 0.777 | 0.942 | 0.982 |
| Tanaka | 7.637 | 9.403 | 9.759 |
| Benjamin–Feir | 7.639 | 9.428 | 9.753 |
| JONSWAP/TMA | 0.772 | 0.943 | 0.984 |

These are **single-rollout timings**, not batch-32 runtimes divided by 32. Sources: [hero timing](../outputs/hero_speed_20261019/contextreal_final_rollouts.json), [classical timing](../outputs/hero_speed_20261019/layout_final_rollouts.json), and [Figure 12d documentation](../paper-plots/README.md#12d-one-epoch-full-data).

### Median terminal surface error

Percent relative L2 error against M6, using the same **32 cases per family**. These accuracy statistics are not the errors of the individual timed cases.

| Family | Original one-epoch hero | M2 | Trained hero | M3 |
| --- | ---: | ---: | ---: | ---: |
| Stokes | 0.001750 | 0.000470 | 0.000411 | 0.000144 |
| Tanaka | 0.038876 | 0.189258 | 0.010116 | 0.012373 |
| Benjamin–Feir | 0.065160 | 0.402603 | 0.021914 | 0.226740 |
| JONSWAP/TMA | 0.029712 | 0.019172 | 0.006764 | 0.005882 |

Stokes needs approximately **2.84× lower error**, and JONSWAP about **13% lower error**, to match M3's medians. The hero's 95th-percentile errors are below M3's in all four families, although its Tanaka tail is worse than the original one-epoch control. Longer training did not improve every case.

An existing alternate cohort, with different simulation IDs, supports the BF advantage but narrows the Tanaka claim: typical Tanaka accuracy is roughly tied with M3, while Stokes/JONSWAP still lag. It is an already-inspected development cohort, not a new untouched holdout.

Sources: [hero summaries](../outputs/hero_speed_20261019/contextreal_eval/all_summaries.json), [M2 summaries](../outputs/momentum_classical_M2_none_20261010/all_summaries.json), [M3 summaries](../outputs/momentum_classical_M3_none_20261010/all_summaries.json), and the alternate-cohort comparison in the [October 20 log](../experiments/experiments-2026-10-20.md).

## What made inference faster

On October 19, the freshly measured pre-graph hero took about 1.57 seconds for the short trajectories and 15.42 seconds for the long ones. The accepted path now takes about 0.94 and 9.4 seconds: roughly **39–40% less runtime**, without changing trained weights, timestep, precision or iteration count.

The accepted execution changes were:

- Graph-compatible FFT execution and unrolling the four integration iterations, reducing repeated dispatch overhead.
- Reusing stage-input spectra and the first surface-network activation instead of recomputing them.
- Packing projections and transforms, changing field/projection layouts, and using branch-major product transforms to avoid unnecessary copies and launches.
- Packing saved-state input transforms, so diagnostics do not undo the integration speedup.
- Using cuFFTDx for both FP64 Bernoulli transforms in the shared solver work.
- Removing redundant DC/Nyquist clearing from the context inverse transform, whose input already has the required real endpoints.

Shared solver improvements were also applied to the classical comparisons. We did not obtain the apparent win by retaining old, slower M3 timings. Padding/backend choices were checked against the reference; M6 retains padding where an unpadded full rollout failed the accuracy check.

The final endpoint-clear removal saved another 2.18–4.68% across families and preserved every saved eta, xi and DNO value in the 128-case evaluation bit-for-bit. Earlier execution changes were evaluated with their own rollout results rather than assuming numerical identity.

A further context-projection/addition fusion measured a small 0.93–1.74% gain, but **is not accepted**: one evaluation encountered an unexplained CUDA illegal-address failure. Successful later evaluations do not resolve that failure. The published local coordinates still use the accepted October 19 path.

Sources: [October 19 execution experiments](../experiments/experiments-2026-10-19.md), [October 20 fusion checks](../experiments/experiments-2026-10-20.md), and the [complete-array audit](../outputs/hero_speed_20261019/contextreal_eval_audit.json).

## CPU offload and the encoder decoder

Computing G0+G1 on the CPU while the GPU computes the correction did not help. A warmed single Tanaka forward took **550 microseconds with CPU overlap versus 142 microseconds GPU-only**, about 3.86× longer. Host dispatch and dependency waits outweighed the overlap; this was a forward-call comparison, not a CPU-offloaded full-rollout timing.

The M3 encoder–decoder was optimized too: its recorded short/long rollout costs fell from **4.270/41.864 seconds to 2.571/25.273 seconds**. These are measurements of that candidate's earlier execution path, not a new run through the current hero backend. Its GPU profile explains the remaining expense:

| Component | Summed GPU kernel time | Share |
| --- | ---: | ---: |
| Classical terms and associated transforms | 123.4 microseconds | 72.8% |
| Encoder and head | 38.0 microseconds | 22.4% |
| Decoder and join | 8.0 microseconds | 4.7% |

Kernel-time sums are not end-to-end latency. The cheaper M1 encoder–decoder also failed to replace the control: it was slower and substantially less accurate. Neither candidate earned promotion.

Plots: [GPU versus CPU offload profile](../paper-plots/figures/gpu-vs-cpu-offload-profile.png) and [M3 encoder–decoder profile](../paper-plots/figures/m3-encoder-decoder-profile.png). Measurements and decisions are in the [October 19](../experiments/experiments-2026-10-19.md) and [October 20](../experiments/experiments-2026-10-20.md) logs.

## Training speed and the actual epoch budget

The EMA trainer now includes the execution improvements that were initially missing from its copy of the original trainer:

- Packed branch projections/FFTs and metric reads every 16 updates reduced compute-block time by **41.1%** in the controlled two-Ada, batch-1,024 benchmark. Metric buffering alone preserved optimizer/EMA states exactly.
- Evaluating the validation Hadamard diagnostic every 16 batches reduced warmed validation-compute time by **26.1%**. Ordinary validation still covers the complete split; the sampled Hadamard diagnostic is not the full-split mean.
- Keeping the nested gate explicitly FP32 removed accidental FP64 promotion of branch FFTs, saving **6.56% per update** in the tested full-loss fixture.

These measurements describe different experiments and should not be multiplied together into a claimed total speedup. Inference-only custom FFT improvements are not automatically backward-pass improvements.

The current full-data recipe has **11,796,720 training rows and 737,295 batch-16 updates per epoch**. Recent epochs take approximately **2 hours 10 minutes to 2 hours 22 minutes**, including validation. The original one-epoch hero itself took about 2 hours 8 minutes including validation/export.

The earlier roughly four-minute estimate belonged to a large-batch, two-GPU workload. Batch 16 creates 64× as many optimizer updates as global batch 1,024, and the current recipe also performs scheduled Hadamard and detached rollout-teacher work. That old estimate does not describe this experiment.

Sources: [September 28 training optimizations](../experiments/experiments-2026-09-28.md), [October 12 batch/gate measurements](../experiments/experiments-2026-10-12.md), and [current launch record](../experiments/experiments-2026-10-21.md).

## Accuracy experiments and what they taught us

The accepted recipe combines RMS-normalized surface features, Fourier context, a nested learned branch, gradient EMA, weight EMA, and detached rollout supervision alongside the existing physics losses. These are ingredients of the selected model, not a claim that every ingredient has independently established the same benefit. Rollout-generated targets use M6 at visited states without backpropagation through the trajectory.

Recent experiments used fresh initialization, one full-data epoch and primary weight-EMA rollouts against the existing one-epoch control. None of the following produced a broadly better replacement:

| Recent change | Paired overall terminal error change | Main finding |
| --- | ---: | --- |
| Full-spectrum DNO targets on rollout states | +6.5% | More direct supervision did not improve overall rollouts. |
| Retained-band RHS supervision | −1.9% | Small pooled gain, severe Tanaka tail regression. |
| Reflection-consistent product features | −0.1% | Tanaka improved, JONSWAP tails worsened. |
| Separate even/odd feature streams | +8.0% | Restoring geometric flexibility did not yield a broad gain. |
| Modal residual weighting | +29.5% | Reweighting Fourier errors hurt overall accuracy. |
| Mixed quadratic/cubic features | +12.0% | Extra leading-order flexibility was insufficient. |
| Coherent error across detached integration stages | +4.9% | Some medians improved, but Tanaka outliers worsened sharply. |
| Coupling neighboring learned branches | +2.8% | A favorable local direction probe did not survive full training. |

Here, overall change is the geometric mean of the 128 **paired candidate/control error ratios**, minus one; negative is better. It is not a ratio of family medians. The primary hero remains unchanged.

Reflection averaging of a frozen RHS-supervised one-epoch model did produce a real accuracy gain. A separate timing test of that shared-geometry construction with hero37 weights still added approximately **35% runtime**. The cheaper trained symmetry variants were not equivalent to that averaging operation and did not retain a convincing overall benefit at the desired cost.

The strongest remaining mechanisms are:

1. **Stokes has persistent learned speed bias.** Translating the predicted surface to align with the reference removes most of its squared error. The forcing is coherent in time; baseline arithmetic and training/deployment mismatch are too small to explain the dominant drift. Better momentum conservation can remove acceleration while leaving an incorrect constant speed.
2. **Some steep Tanaka cases have growing high-frequency profile errors.** Switching to M6 halfway through those trajectories retains most of the eventual error: the corrupted coupled state can continue amplifying the discrepancy without further neural forcing. This is different from pure phase drift.
3. **JONSWAP has both phase and profile error.** A universal phase-only penalty therefore does not address the whole problem.
4. **Poor optimization conditioning is now a concrete lead.** The features feeding the output heads are highly correlated; adding more multiplier directions or capacity has not consistently helped. This motivates changing the optimizer coordinates rather than adding inference work.

These conclusions and the completed-screen comparisons are recorded in the [October 20 accuracy log](../experiments/experiments-2026-10-20.md). They do not establish that capacity can never matter or that one new optimizer change will solve the remaining errors.

## Paused head conditioning experiment

The new experiment changes how Adam updates the ordinary surface-output heads. It calibrates a fixed, regularized whitening transform from fresh TRAIN features, applies it to head gradients before Adam and to updates afterward, and keeps the physical network parameters unchanged in format. Both EMAs, all losses and the inference computation remain intact. Training adds two small packed matrix multiplications; inference adds none.

The motivation is a TRAIN-only CPU probe:

- The 160-feature second-moment matrix had condition number about 5.39 million.
- A transformed first-Adam direction improved held-out data descent by 17.7% at matched response size.
- The number of held-out rows with a descending direction rose from 45/128 to 124/128.

That probe is **not a rollout result** and does not reproduce the full batch-16 training trajectory. The actual one-epoch test was paused before validation or any EMA rollout. Targeted FP32/FP64 coordinate-update, weight-decay, EMA and counter tests passed; the broad loss test suite hit its time cap without completing.

When the user chooses to resume, the useful next evidence is the completed primary-EMA rollout comparison against the existing one-epoch control—not a new control or another frozen direction probe. Promotion requires a broad improvement without repeating the severe Tanaka-tail tradeoffs. There is no reason to move figure coordinates before that result.

Sources: [conditioning probe](../outputs/hero_speed_20261020/postlift_whitening_train_probe.json) and [implementation, launch and pause record](../experiments/experiments-2026-10-21.md).

## Pause state

At 00:21 EDT, the run in `outputs/c27_head_whitening_scratch1_20261021` was suspended at the last displayed update **49,105/737,295**, about **6.7%**. The Python process and launcher were verified in stopped state. GPU utilization subsequently read zero on both devices; the suspended run retains approximately 19.4 GiB on GPU0 to preserve its in-memory state. No other jobs were signaled.

This is an **in-memory pause, not a saved mid-epoch checkpoint**. The training process is PID 3769011, its uv parent 3769008, launcher 3769002 and timeout supervisor 3769000. All four were stopped. The timeout supervisor retains its original wall-clock deadline, so it must not be blindly resumed after a long pause; verify live identities and handle that supervisor before continuing the child processes. No automatic resume is scheduled. The accuracy-search goal is also marked paused.
