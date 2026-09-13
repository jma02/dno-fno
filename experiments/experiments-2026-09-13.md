| Date | Time | File / Variant | Motivation | What Tried / Evidence | Correctness | Timing | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-13 | 01:53 | Balanced C27 Modal launch preparation | Local two-GPU training could not allocate memory alongside another user's job; user requested Modal instead. Reuse the same trainer and batch/update budget on two A100-80GB GPUs. | Confirmed Modal CLI1.4.2 and profile/workspace sciml-at-ud. Balanced arrays are not present at /outputs/paper_dataset_balanced_20260912/arrays on volume dno-fno-train-data. Prepared existing launcher with EPOCHS260, batch1024, seed0, same losses, GPU_SPEC=A100-80GB:2. Match image Python minor version3.11 and pin numerical packages to local versions; correct stale all-family translation label. Requested upload of11 NPY files totaling14.715GB plus scoped trainer/model/solver source. | Local Ruff/Pyright and shell syntax pass; independent launcher/path/hyperparameter audit passes. Remote image build and training NOT tested: approval system rejected upload before process creation, requiring explicit permission for disclosure to this specific Modal workspace/volume. No upload, remote training or local job pause occurred; our failed local trainer was already stopped. | 00:02 (inspection/preparation01:51–01:53; no GPU training). | User explicitly approved this destination and two-A100 training before01:59; see the following execution record. Initial rejection was not bypassed. |
| 2026-09-13 | 01:59 | Approved balanced dataset upload and Modal C27 run | User approved disclosure of the14.715GB balanced dataset and scoped training code to sciml-at-ud/dno-fno-train-data, plus training on two A100-80GB GPUs, after local GPU-memory failure. | Existing upload entrypoint completed11 NPY files; existing detached launcher submitted c27_balanced_tanaka_tangent_modal_20260913 with260 epochs, global batch1024, same architecture/losses and FP32 train/val. Training app ap-sDX1uTgQnbRpNWYkCJNgNX; function call fc-01M2CNTNT9DCNGZ57027DG328E. No smaller batch or new trainer. | COMPLETED PASS: physical devices are two A100-SXM4-80GB; replicated state/optimizer counters verified at0; all260 consecutive epoch records have finite metrics; best/final epoch260 has training loss.0003892254346, validation total.0003607055300 and validation relative-L2.0003308064795. Final/latest/best checkpoints and summary are saved on the volume; process returned0. Independent remote config SHA matches local balanced run after excluding only dataset path/run name. All pinned numerical packages match local versions; uploaded trainer sets NCCL_P2P_LEVEL=PHB before JAX. All11 uploaded filenames/sizes checked; remote row counts and normalization match. | Image/upload01:59–02:03; worker started02:04; first finite updates02:05; first epoch/validation/checkpoints verified02:09. Finished15:42; runtime13:38 (49074.695s trainer,49086s wrapper). Final log audited17:23. | TRAINING COMPLETE. No local jobs stopped or modified. Checkpoints/logs persist on approved volume; rollout/NaN tests have not run. Do not infer rollout stability from finite training metrics. This is the260-epoch matched-update-budget balanced-corpus experiment, not a40-epoch shortened run. |
| 2026-09-13 | 17:31 | Balanced C27 final checkpoint: full previous FP64 rollout panel | Equal family row counts may improve stability beyond Tanaka-only translation regularization; test the same held-out ICs, including both previous NaN failures, without changing the integrator or adding soliton damping. | Downloaded completed epoch260 checkpoint from approved Modal volume. Started two local GPU workers, 64 disjoint ICs each: known Tanaka failures16471/16624 first, then all remaining previously tested Stokes, JONSWAP, Tanaka and BF cases, 32 per family overall. Existing model-only rollout functions, singleton first case then batches of at most2; each simulation saved immediately. Reuse archived references only for explicitly labeled error comparisons, never regenerate truth. | RUNNING: both workers restored epoch260; GPU placement, all parameter/probe dtypes float64 and finite initial predictions passed. All128 IDs asserted equal to original archives; independent audit found balanced dataset initial eta/xi/depth identical to originals. Internal dt.01, GL2 four iterations, cutoff128; T200 for Tanaka/BF and20 for Stokes/JONSWAP. Reference fields were integrated in FP64 but archived as FP32; they are not a full-precision truth cache. No stability outcome yet. | Started17:31; startup verified17:32; completion pending. | Evaluation in progress; no production code changed. Both GPUs used; old suspended reference-generation jobs left suspended. Do not claim the balanced model fixes NaNs before results arrive. |

### Executed commands (upload01:59; training submitted02:03)

```sh
MODAL_PROFILE=sciml-at-ud modal run scripts/modal_train.py::upload_dataset \
  --dataset outputs/paper_dataset_balanced_20260912/arrays

MODAL_PROFILE=sciml-at-ud GPU_SPEC=A100-80GB:2 EPOCHS=260 \
  DATASET=/data/outputs/paper_dataset_balanced_20260912/arrays \
  RUN_NAME=c27_balanced_tanaka_tangent_modal_20260913 \
  bash scripts/launch_c27_paper_dataset_modal.sh
```

Numerical package pins: JAX/JAXlib0.9.2, Flax0.12.6, Optax0.2.5,
Orbax0.11.33, NumPy2.4.2, SciPy1.17.0. The existing trainer sets
`NCCL_P2P_LEVEL=PHB` before importing JAX. Core training/validation stays FP32;
existing FP64 analytic-G1/Hadamard computations stay unchanged. No local GPU
resources are needed for the proposed remote training.

### Remote run and verification

- Upload app: `ap-oghqmhf5mjGPvhtUzaaEXf` (completed).
- Training app: https://modal.com/apps/sciml-at-ud/main/ap-sDX1uTgQnbRpNWYkCJNgNX
- Function call: `fc-01M2CNTNT9DCNGZ57027DG328E`.
- Initial container: `ta-01M2CNV3AA7CNAEJ22ZRH0RDHR`.
- Run directory: `/data/outputs/c27_balanced_tanaka_tangent_modal_20260913`.
- Local submission log: `logs/c27_balanced_tanaka_tangent_modal_20260913.log`.
- Numerical image: `im-zHAQ6GlMKcYN1OvLt8cece`; source/environment commit `9719a83`.
- Canonical config SHA256, excluding dataset path and run name:
  `67345af255cf170d3a891ffb6931baa126e2c790baf8cae891c52024abc71742`,
  identical remotely and in the failed local balanced-run configuration.

Remote config/device/package checks used read-only `modal container exec`;
version inspection used package metadata without importing JAX. The local
entrypoint printed a final-log-fetch timeout on disconnect, but the detached
worker started normally and made finite updates; this was not a training timeout.
The existing wrapper commits the volume every300seconds and on exit; checkpoint
retries retain the same run name and260-epoch schedule. No automatic rollout
evaluation was added. Future rollout evaluation remains FP64 on the held-out ICs.

First-epoch audit at02:09: all logged scalars finite;924 optimizer updates
completed, including the post500-step warmup losses. Training loss
`.002071135133983284`, validation total loss `.0019658407188712563`, validation
relative-L2 loss `.0013552555099105095`. Both `latest_ckpt/` and `best_val_ckpt/`
exist in the run directory. These are startup checks, not final accuracy or
rollout-stability results; the260-epoch run continued independently of this client.

Completion audit at17:23: the run finished15:42, returncode0, all260 consecutive
epoch records finite. Trainer runtime49074.694540085seconds (about13h38m);
best epoch260, training loss`.0003892254346138585`, validation total
`.00036070553000659255`, validation relative-L2`.0003308064795305615`.
`final_ckpt/`, `latest_ckpt/`, `best_val_ckpt/`, `summary.json` and the complete
training log are saved on the Modal volume. No rollout tests or new jobs were
started during this read-only status check. The previously failing ICs still
need FP64 rollout evaluation; finite training alone does not establish a NaN fix.

### Balanced model rollout launch (17:31)

Downloaded the run recursively with `MODAL_PROFILE=sciml-at-ud modal volume get
dno-fno-train-data /outputs/c27_balanced_tanaka_tangent_modal_20260913 outputs/`.
Local checkpoint: `outputs/c27_balanced_tanaka_tangent_modal_20260913/final_ckpt`.
Persistent tmux sessions: `c27_balanced_eval_gpu0_20260913` and
`c27_balanced_eval_gpu1_20260913`. Consoles: `eval_gpu0.console.log` and
`eval_gpu1.console.log` under the downloaded run. Results:
`eval_final_n32_fp64/simulation_<id>.npz` and matching JSON, under that run.

Each worker runs the following inline code using `uv run python -u -c`, with
argument0/1 respectively and `CUDA_VISIBLE_DEVICES=0`/`1`, `JAX_PLATFORMS=cuda`,
`UV_OFFLINE=1 UV_NO_SYNC=1 UV_NO_CACHE=1 OMP_NUM_THREADS=4`, and
`MPLCONFIGDIR=/tmp/matplotlib-c27-balanced`. No new evaluator script was added.
The first case is a singleton to match previous failed-IC evaluations; remaining
cases use batches of at most2. Workers together cover all128 archived IDs once.
Startup audit at17:33: Python PIDs213458/213464 both actively use their assigned
GPUs (100% utilization); disjoint64-case selections and initial arrays passed
independent checks against the original dataset. Old suspended jobs untouched.
NaN checks use only fresh FP64 predictions. Relative eta errors explicitly use
the archived FP32 reference values, not newly generated FP64 reference files.

```python
import os
os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from solver.evals import eval_suite as ev

jax.config.update("jax_enable_x64", True)
gpu = int(sys.argv[1])
run_dir = Path("outputs/c27_balanced_tanaka_tangent_modal_20260913").resolve()
dataset = Path("outputs/paper_dataset_balanced_20260912/arrays").resolve()
archive_dir = Path("outputs/c27_paper_dataset_20260908_141423/eval_final_n32")
out_dir = run_dir / "eval_final_n32_fp64"
out_dir.mkdir(exist_ok=True)
loaded = ev.load_run(run_dir, checkpoint="final")
assert loaded.epoch == 260
assert len(jax.devices()) == 1 and jax.devices()[0].platform == "gpu"
assert all(p.dtype == jnp.float64 for p in jax.tree_util.tree_leaves(loaded.params))
predict = ev.build_predict_gxi_batched(loaded)
panels = {}
for family in ev.FAMILY_CONFIGS:
    ics, _, nx, length = ev._load_paper_dataset_ics(dataset, family, 32)
    with np.load(archive_dir / f"{family}_trajs.npz") as archive:
        assert [ic.simulation_id for ic in ics] == archive["simulation_ids"].tolist()
    panels[family] = ics

failed_ids = (16471, 16624)
jobs = [("tanaka", [ic for ic in panels["tanaka"] if ic.simulation_id == failed_ids[gpu]])]
for family in ("stokes", "jonswap_tma", "tanaka", "benjamin_feir"):
    selected = [ic for ic in panels[family] if ic.simulation_id not in failed_ids]
    selected = selected[gpu::2]
    jobs.extend((family, selected[start:start + 2]) for start in range(0, len(selected), 2))
assert sum(len(ics) for _, ics in jobs) == 64
print(f"GPU {gpu}: epoch260 FP64; queued64 exact previous ICs, known failure first; no reference generation or soliton damping", flush=True)

completed = 0
for family, ics in jobs:
    cfg = ev.FAMILY_CONFIGS[family]
    times = np.arange(0.0, cfg.tmax + 0.5 * cfg.dt, cfg.dt, dtype=np.float64)
    ids = [ic.simulation_id for ic in ics]
    probe = predict(
        jnp.asarray(np.stack([ic.eta for ic in ics])),
        jnp.asarray(np.stack([ic.xi for ic in ics])),
        jnp.asarray(np.log([ic.depth for ic in ics])),
    )
    assert probe.dtype == jnp.float64 and np.isfinite(np.asarray(probe)).all()
    print(f"START GPU{gpu} {family} IDs={ids} T={cfg.tmax} dt={cfg.dt/cfg.substeps}", flush=True)
    pred = ev.surrogate_rollout_batched(ics, jnp.asarray(times), nx, length, cfg, predict)
    assert all(np.asarray(pred[name]).dtype == np.float64 for name in ("eta", "xi", "gxi"))
    with np.load(archive_dir / f"{family}_trajs.npz") as archive:
        positions = [archive["simulation_ids"].tolist().index(sid) for sid in ids]
        reference_eta = archive["truth_eta"][:, positions].astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        rel_l2_eta = ev._rel_l2(np.asarray(pred["eta"]), reference_eta)
    for index, ic in enumerate(ics):
        fields = {name: np.asarray(pred[name])[:, index:index + 1] for name in ("eta", "xi", "gxi")}
        finite = np.logical_and.reduce([np.isfinite(values).all(axis=(1, 2)) for values in fields.values()])
        bad = np.flatnonzero(~finite)
        np.savez_compressed(
            out_dir / f"simulation_{ic.simulation_id}.npz",
            times=times, simulation_ids=np.asarray([ic.simulation_id]),
            depths=np.asarray([ic.depth]), initial_eta=ic.eta, initial_xi=ic.xi,
            **{f"pred_{name}": values for name, values in fields.items()},
            rel_l2_eta=rel_l2_eta[:, index:index + 1],
        )
        final_error = float(rel_l2_eta[-1, index])
        summary = {
            "family": family, "simulation_id": ic.simulation_id,
            "dataset_row": ic.meta["dataset_row"], "depth": ic.depth,
            "checkpoint": str(run_dir / "final_ckpt"), "epoch": loaded.epoch,
            "model_and_integration_dtype": "float64", "extra_stabilizer": False,
            "internal_dt": cfg.dt / cfg.substeps, "save_dt": cfg.dt, "tmax": cfg.tmax,
            "picard_iterations": ev.GL2_ITERATIONS, "cutoff": cfg.filter_fraction * nx / 2,
            "reference_generation": False, "gpu": gpu,
            "reference": str(archive_dir / f"{family}_trajs.npz"),
            "reference_storage_dtype": "float32",
            "all_saved_values_finite": bool(finite.all()),
            "first_nonfinite_saved_time": float(times[bad[0]]) if bad.size else None,
            "final_rel_l2_eta_vs_archived_fp32_reference": final_error if np.isfinite(final_error) else None,
            "chunk_wall_s": float(pred["wall_s"]), "chunk_size": len(ics),
        }
        ev._write_json(out_dir / f"simulation_{ic.simulation_id}.json", summary)
        completed += 1
        print(f"DONE {completed}/64 GPU{gpu}: " + json.dumps(summary, allow_nan=False), flush=True)
print(f"COMPLETE GPU{gpu}: all64 ICs saved", flush=True)
```
