| Date | Time | File / Variant | Motivation | What Tried / Evidence | Correctness | Timing | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-12 | 00:16 | Two-GPU FP64 Tanaka evaluation handoff | GPU 0 has computed most of the 32 reference trajectories; GPU 1 is idle. Run independent surrogate IC halves concurrently without losing reference work. | Add optional precomputed predictions to the existing evaluator; metrics and output code stay unchanged. Coordinator below: GPU 1 evaluates ICs 0:16 immediately; after full matching FP64 truth is saved, stop only the old evaluator and run ICs 16:32 on GPU 0. Concatenate in original order. Final epoch40, dt=0.8, tmax=200, 80 substeps, batch2, no soliton damping; NCCL_P2P_LEVEL=PHB. | PASS: 12 evaluator tests, including identical saved arrays and nonfinite metrics with ordinary/precomputed predictions; Ruff/Pyright clean. At00:21 both GPUs100%; GPU1 FP64 parameter/output/device probe passed. Full reference cache protocol/IDs/shape checked before old-job termination. Final rollout results PENDING. First launcher exited harmlessly on unavailable pidfd API; corrected with installed psutil. | 00:00 (5.706 s CPU tests); corrected GPU launch00:21; paused01:02. | PAUSED at user direction to prioritize only the two previous failures. Both original processes suspended with progress retained in memory; no automatic resumption scheduled. |
| 2026-09-12 | 01:04 | Final Tanaka-only-tangent model: failed-case finiteness test | The immediate question is whether retraining fixes the two previously problematic ICs, not performance across a new 32-case benchmark. No reference trajectories are required to test finiteness. | Pause broad-evaluation PIDs3953921/3987236 with identity-checked psutil handles. Launch simulation16471 on GPU0 and16624 on GPU1 in separate persistent sessions, using final epoch40 and existing surrogate_rollout_batched only. Dataset rows33784/64384, T200, save interval0.8, 80 substeps, internal dt0.01, four GL2 iterations, cutoff128; float64 model/integration, no adaptive stabilizer. Each job writes all predicted fields plus first nonfinite saved time. | COMPLETED; both fail. First saved NaN16471=164.8,16624=127.2, simultaneously in eta/xi/Gxi; no infinities. Independent archive audit20:32 verifies float64 shape251x1x1024, saved times0..200, and bit-identical initial eta/xi and depths versus the five-epoch fine-tune archive. That earlier fine-tune had16471 finite through200 and16624 firstNaN188.8; original FP64 model had16471 finite and16624 firstNaN164.0. No truth generation or accuracy metric. | Launched01:04; completed01:16. GPU0 00:12 (700.892 s); GPU1 00:11 (682.173 s). | REJECT AS A NaN FIX: worse on both matched cases. Only these two targeted evaluations completed; broad panel still paused, with no full-panel result or automatic continuation. No model/data/solver change from this test. |
| 2026-09-12 | 20:52 | Compiled static Stokes labels for balanced-family generation | Stokes states require no integration, but eager per-state construction and DNO evaluation add dispatch overhead when generating hundreds of thousands of independent states. | Cache JIT compilation of the existing constructor and DNO target, without changing formulas, float64 precision, order6, pad8, cutoff128 or numerical rejection checks. GPU1 evaluates32 states,8/group: eager cold12.753s, compiled cold1.895s, compiled warm0.283s. These are not matched warm/cold speedup measurements. | All32 states accepted and finite; maximum eager/compiled relative field difference4.87e-16. Seven focused CPU tests pass, including both Stokes branches; Ruff/Pyright clean. | 00:00 (GPU benchmark20:52:03–20:52:19; CPU tests6.856s). | KEEP: +7 production lines, no new helpers. Warm label throughput about113 states/s, excluding parameter sampling and storage. |

### Broad evaluation launch details (paused at 01:02)

Run: `outputs/c27_tanaka_tangent_paper_dataset_20260910_144736`.
Output: `eval_final_tanaka_n32_fp64` within that run.
The original evaluator is PID `3953921`; a `psutil.Process` handle verifies
process identity before termination, protecting against PID reuse.
Partial prediction halves are saved as `parallel_pred_gpu0.npz` and
`parallel_pred_gpu1.npz` for recovery; final metrics use all 32 ICs in their
original order. Cached-truth timing in the merged evaluator measures reuse,
not the original reference-generation cost recorded in the original console log.

Executed as an inline Python command in persistent tmux session
`c27_tanaka_eval_dual_gpu_20260912`, with `CUDA_VISIBLE_DEVICES=0,1`,
`NCCL_P2P_LEVEL=PHB`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
`UV_OFFLINE=1`, `UV_NO_SYNC=1`, `UV_NO_CACHE=1`, and
`MPLCONFIGDIR=/tmp/matplotlib-c27-dual-eval`.
Console: `outputs/c27_tanaka_tangent_paper_dataset_20260910_144736/eval_dual_gpu.console.log`.

Verified at 00:21: corrected coordinator PID `3987236` is running in tmux;
GPU 0 and GPU 1 both report 100% utilization. GPU 1 passed the FP64 parameter,
output, and device-placement checks and started IC chunk 0:2 of its first 16
simulations, including the previously problematic simulation `16471` later in
that half. GPU 0's original PID `3953921` continues reference generation;
its second-half prediction assignment will include simulation `16624`.
Final rollouts and metrics remain pending. The production evaluator change
adds only three lines (optional argument and condition); the rest of the
code diff is indentation and a regression test. Source/test commit: `4f4ebab`.

The first launch at 00:19 exited before starting either worker or signaling the
original job: this standalone Python build lacks `os.pidfd_open`. The corrected
launcher uses the already-installed `psutil` process handle instead. Original
reference generation was uninterrupted.

```python
import os
os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import psutil
import time
import numpy as np
import jax
import jax.numpy as jnp
from solver.evals import eval_suite as ev

jax.config.update("jax_enable_x64", True)
run_dir = Path("outputs/c27_tanaka_tangent_paper_dataset_20260910_144736").resolve()
dataset = Path("outputs/paper_dataset/arrays").resolve()
out_dir = run_dir / "eval_final_tanaka_n32_fp64"
out_dir.mkdir(exist_ok=True)
original_pid = 3953921
argv = Path(f"/proc/{original_pid}/cmdline").read_bytes().decode().split("\0")
assert "solver/evals/eval_suite.py" in argv
assert Path(argv[argv.index("--run_dir") + 1]).resolve() == run_dir
original_process = psutil.Process(original_pid)
devices = jax.devices("gpu")
assert len(devices) == 2
cfg = ev.FAMILY_CONFIGS["tanaka"]
ics, _, nx, length = ev._load_paper_dataset_ics(dataset, "tanaka", 32)
times = np.arange(0.0, cfg.tmax + 0.5 * cfg.dt, cfg.dt, dtype=np.float64)
protocol = ev._truth_protocol("tanaka", cfg, nx=nx, length=length, ics=ics)
cache_path = out_dir / "tanaka_truth_cache.npz"

def predict_half(device_index: int, selected: list[ev.IC]) -> ev.RolloutPayload:
    device = devices[device_index]
    with jax.default_device(device):
        loaded = ev.load_run(run_dir, checkpoint="final")
        loaded = loaded._replace(params=jax.device_put(loaded.params, device))
        assert all(p.dtype == jnp.float64 and p.devices() == {device}
                   for p in jax.tree_util.tree_leaves(loaded.params))
        predict = ev.build_predict_gxi_batched(loaded)
        probe = predict(
            jnp.asarray(np.stack([ic.eta for ic in selected[:2]])),
            jnp.asarray(np.stack([ic.xi for ic in selected[:2]])),
            jnp.asarray(np.log([ic.depth for ic in selected[:2]])),
        )
        probe.block_until_ready()
        assert probe.dtype == jnp.float64 and probe.devices() == {device}
        assert np.isfinite(np.asarray(probe)).all()
        print(f"GPU {device_index}: FP64 placement passed; simulation IDs "
              f"{[ic.simulation_id for ic in selected]}", flush=True)
        result = ev._rollout_ic_chunks(
            selected, 2,
            lambda chunk: ev.surrogate_rollout_batched(
                chunk, jnp.asarray(times), nx, length, cfg, predict,
            ),
            label=f"tanaka surrogate GPU {device_index}",
        )
        np.savez(
            out_dir / f"parallel_pred_gpu{device_index}.npz",
            **result, times=times,
            simulation_ids=np.asarray([ic.simulation_id for ic in selected]),
        )
        print(f"GPU {device_index}: completed {len(selected)} surrogate ICs", flush=True)
        return result

started = time.perf_counter()
with ThreadPoolExecutor(max_workers=2) as pool:
    first = pool.submit(predict_half, 1, ics[:16])
    print("GPU 1 starts ICs 0:16; preserving GPU 0 reference generation", flush=True)
    while not cache_path.is_file():
        if first.done():
            first.result()
        if not original_process.is_running():
            raise RuntimeError("Original evaluation exited before publishing reference cache")
        time.sleep(5)
    truth = ev._try_load_cached_truth(
        "tanaka", out_dir, expected_shape=(len(times), len(ics), nx),
        expected_simulation_ids=[ic.simulation_id for ic in ics],
        expected_protocol_json=protocol,
    )
    assert truth is not None, "Do not stop the original job without matching FP64 truth"
    del truth
    original_process.terminate()
    original_process.wait()
    print("All 32 references verified and preserved; GPU 0 starts ICs 16:32", flush=True)
    second = pool.submit(predict_half, 0, ics[16:])
    halves = [first.result(), second.result()]

pred: ev.RolloutPayload = {
    name: np.concatenate([half[name] for half in halves], axis=1)
    for name in ("eta", "xi", "gxi")
}
pred["wall_s"] = time.perf_counter() - started
with jax.default_device(devices[0]):
    loaded = ev.load_run(run_dir, checkpoint="final")
    predict = ev.build_predict_gxi_batched(loaded)
    summary = ev.run_family(
        "tanaka", cfg, loaded, predict, out_dir, dataset, 32, out_dir,
        {"run_dir": str(run_dir), "selection": "final",
         "path": str(run_dir / "final_ckpt"), "epoch": loaded.epoch},
        2, pred=pred,
    )
summary["surrogate_gpu_count"] = 2
ev._write_json(out_dir / "tanaka_summary.json", summary)
summaries = {"tanaka": summary}
ev._write_json(out_dir / "all_summaries.json", summaries)
ev._write_json(out_dir / "macro_summary.json", ev.compute_macro_summary(summaries))
print(f"Done. Results written to {out_dir}", flush=True)

```

### Targeted failed-case launch at 01:04

Two inline commands run the same code below with argument `16471` and
`CUDA_VISIBLE_DEVICES=0`, or argument `16624` and `CUDA_VISIBLE_DEVICES=1`.
Persistent sessions: `c27_failed_16471_20260912_gpu0` and
`c27_failed_16624_20260912_gpu1`. Environment otherwise matches the broad
launcher above; `MPLCONFIGDIR=/tmp/matplotlib-c27-failed`.
Outputs: `eval_final_failed_tanaka_fp64/simulation_<id>.npz` and `.json`
within the trained run; consoles are `eval_failed_<id>.console.log`.
PIDs at launch: GPU0 `3997936`, GPU1 `3997852`.

Completion audit at 20:32: both targeted processes exited after writing their
JSON and NPZ results around 01:16. All three predicted fields become NaN at
the times recorded above. Broad-evaluation processes remain suspended; the
remaining previously tested ICs have not been evaluated to completion.

```python
import os
os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import json
from pathlib import Path
import sys
import numpy as np
import jax
import jax.numpy as jnp
from solver.evals import eval_suite as ev

jax.config.update("jax_enable_x64", True)
simulation_id = int(sys.argv[1])
run_dir = Path("outputs/c27_tanaka_tangent_paper_dataset_20260910_144736").resolve()
dataset = Path("outputs/paper_dataset/arrays").resolve()
out_dir = run_dir / "eval_final_failed_tanaka_fp64"
out_dir.mkdir(exist_ok=True)
ics, _, nx, length = ev._load_paper_dataset_ics(dataset, "tanaka", 32)
ic = next(ic for ic in ics if ic.simulation_id == simulation_id)
cfg = ev.FAMILY_CONFIGS["tanaka"]
times = np.arange(0.0, cfg.tmax + 0.5 * cfg.dt, cfg.dt, dtype=np.float64)
loaded = ev.load_run(run_dir, checkpoint="final")
assert loaded.epoch == 40
assert all(p.dtype == jnp.float64 for p in jax.tree_util.tree_leaves(loaded.params))
predict = ev.build_predict_gxi_batched(loaded)
print(f"START simulation={simulation_id} GPU={os.environ['CUDA_VISIBLE_DEVICES']} "
      f"epoch={loaded.epoch} FP64 T={cfg.tmax} internal_dt={cfg.dt/cfg.substeps} "
      f"no extra stabilizer, no reference generation", flush=True)
pred = ev.surrogate_rollout_batched(
    [ic], jnp.asarray(times), nx, length, cfg, predict,
)
assert all(np.asarray(pred[name]).dtype == np.float64 for name in ("eta", "xi", "gxi"))
finite = np.logical_and.reduce([
    np.isfinite(np.asarray(pred[name])).all(axis=(1, 2))
    for name in ("eta", "xi", "gxi")
])
bad = np.flatnonzero(~finite)
np.savez_compressed(
    out_dir / f"simulation_{simulation_id}.npz",
    times=times, simulation_ids=np.asarray([simulation_id]),
    depths=np.asarray([ic.depth]),
    pred_eta=pred["eta"], pred_xi=pred["xi"], pred_gxi=pred["gxi"],
)
summary = {
    "simulation_id": simulation_id, "dataset_row": ic.meta["dataset_row"],
    "depth": ic.depth, "checkpoint": str(run_dir / "final_ckpt"), "epoch": loaded.epoch,
    "model_and_integration_dtype": "float64", "extra_stabilizer": False,
    "internal_dt": cfg.dt / cfg.substeps, "save_dt": cfg.dt, "tmax": cfg.tmax,
    "picard_iterations": ev.GL2_ITERATIONS, "cutoff": cfg.filter_fraction * nx / 2,
    "reference_generation": False, "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
    "all_saved_values_finite": bool(finite.all()),
    "first_nonfinite_saved_time": float(times[bad[0]]) if bad.size else None,
    "wall_s": float(pred["wall_s"]),
}
ev._write_json(out_dir / f"simulation_{simulation_id}.json", summary)
print(json.dumps(summary, indent=2, allow_nan=False), flush=True)

```
