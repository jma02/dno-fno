| Date | Time | File / Variant | Motivation | What Tried / Evidence | Correctness | Timing | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-12 | 00:16 | Two-GPU FP64 Tanaka evaluation handoff | GPU 0 has already computed most of the 32 reference trajectories; GPU 1 is idle. Compute independent surrogate IC halves concurrently while preserving the original reference work. | Add an optional precomputed prediction argument to the existing evaluator, leaving its metrics and output code unchanged. Launch the coordinator below: GPU 1 evaluates ICs 0:16 immediately; after the original process atomically publishes a matching full FP64 truth cache, stop only that original process and evaluate ICs 16:32 on GPU 0. Concatenate in original IC order, then call the existing evaluator with all predictions and cached truth. Same final epoch-40 checkpoint, dt=0.8, tmax=200, 80 substeps, batch=2, no soliton damping; NCCL_P2P_LEVEL=PHB. | Prelaunch PASS: 12 evaluator tests, including identical saved arrays and nonfinite-failure metrics with ordinary versus precomputed predictions; Ruff and Pyright clean. Runtime asserts verify model/input-output FP64 device placement and full reference cache protocol/IDs/shape before terminating the old job. Rollout results PENDING; numerical failures remain evaluation outcomes, not suppressed. | 00:00 (5.706 s focused CPU tests); GPU run pending launch verification and completion. | IN PROGRESS: preserve all original reference work; use both GPUs for the remaining surrogate evaluation. No training, model, dataset, or rollout-protocol changes. |

### Launch details

Run: `outputs/c27_tanaka_tangent_paper_dataset_20260910_144736`.
Output: `eval_final_tanaka_n32_fp64` within that run.
The original evaluator is PID `3953921`; the coordinator opens a PID file
descriptor before waiting, so its later signal cannot target a reused PID.
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

```python
import os
os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import select
import signal
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
original_fd = os.pidfd_open(original_pid)
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
        if select.select([original_fd], [], [], 0)[0]:
            raise RuntimeError("Original evaluation exited before publishing reference cache")
        time.sleep(5)
    truth = ev._try_load_cached_truth(
        "tanaka", out_dir, expected_shape=(len(times), len(ics), nx),
        expected_simulation_ids=[ic.simulation_id for ic in ics],
        expected_protocol_json=protocol,
    )
    assert truth is not None, "Do not stop the original job without matching FP64 truth"
    del truth
    signal.pidfd_send_signal(original_fd, signal.SIGTERM)
    select.select([original_fd], [], [])
    os.close(original_fd)
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
