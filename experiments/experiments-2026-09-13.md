| Date | Time | File / Variant | Motivation | What Tried / Evidence | Correctness | Timing | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-13 | 01:53 | Balanced C27 Modal launch preparation | Local two-GPU training could not allocate memory alongside another user's job; user requested Modal instead. Reuse the same trainer and batch/update budget on two A100-80GB GPUs. | Confirmed Modal CLI1.4.2 and profile/workspace sciml-at-ud. Balanced arrays are not present at /outputs/paper_dataset_balanced_20260912/arrays on volume dno-fno-train-data. Prepared existing launcher with EPOCHS260, batch1024, seed0, same losses, GPU_SPEC=A100-80GB:2. Match image Python minor version3.11 and pin numerical packages to local versions; correct stale all-family translation label. Requested upload of11 NPY files totaling14.715GB plus scoped trainer/model/solver source. | Local Ruff/Pyright and shell syntax pass; independent launcher/path/hyperparameter audit passes. Remote image build and training NOT tested: approval system rejected upload before process creation, requiring explicit permission for disclosure to this specific Modal workspace/volume. No upload, remote training or local job pause occurred; our failed local trainer was already stopped. | 00:02 (inspection/preparation01:51–01:53; no GPU training). | User explicitly approved this destination and two-A100 training before01:59; see the following execution record. Initial rejection was not bypassed. |
| 2026-09-13 | 01:59 | Approved balanced dataset upload and Modal C27 run | User approved disclosure of the14.715GB balanced dataset and scoped training code to sciml-at-ud/dno-fno-train-data, plus training on two A100-80GB GPUs, after local GPU-memory failure. | Existing upload entrypoint completed11 NPY files; existing detached launcher submitted c27_balanced_tanaka_tangent_modal_20260913 with260 epochs, global batch1024, same architecture/losses and FP32 train/val. Training app ap-sDX1uTgQnbRpNWYkCJNgNX; function call fc-01M2CNTNT9DCNGZ57027DG328E. No smaller batch or new trainer. | COMPLETED PASS: physical devices are two A100-SXM4-80GB; replicated state/optimizer counters verified at0; all260 consecutive epoch records have finite metrics; best/final epoch260 has training loss.0003892254346, validation total.0003607055300 and validation relative-L2.0003308064795. Final/latest/best checkpoints and summary are saved on the volume; process returned0. Independent remote config SHA matches local balanced run after excluding only dataset path/run name. All pinned numerical packages match local versions; uploaded trainer sets NCCL_P2P_LEVEL=PHB before JAX. All11 uploaded filenames/sizes checked; remote row counts and normalization match. | Image/upload01:59–02:03; worker started02:04; first finite updates02:05; first epoch/validation/checkpoints verified02:09. Finished15:42; runtime13:38 (49074.695s trainer,49086s wrapper). Final log audited17:23. | TRAINING COMPLETE. No local jobs stopped or modified. Checkpoints/logs persist on approved volume; rollout/NaN tests have not run. Do not infer rollout stability from finite training metrics. This is the260-epoch matched-update-budget balanced-corpus experiment, not a40-epoch shortened run. |
| 2026-09-13 | 17:31 | Balanced C27 final checkpoint: full previous FP64 rollout panel | Equal family row counts may improve stability beyond Tanaka-only translation regularization; test the same held-out ICs, including both previous NaN failures, without changing the integrator or adding soliton damping. | Downloaded completed epoch260 checkpoint from approved Modal volume. Two local workers cover128 exact previous ICs, known Tanaka failures first. Model-only FP64 rollouts reuse archived FP32 reference fields only for labeled error comparisons; no truth generation. | PARTIAL:70/128 complete before fine-tuning preparation; all32 Stokes and32 JONSWAP plus4 other Tanaka cases finite. Both known Tanaka failures persist:16471 first saved NaNs179.2;16624 at150.4. IDs and initial arrays match originals; internal dt.01, GL2 four iterations, cutoff128, no soliton damping. | Started17:31; both workers paused during18:35–18:37 fine-tuning preparation, preserving their current batches. Remaining58 cases not complete. | Balancing did not remove either targeted failure. Workers213458/213464 are paused for the requested local fine-tune; queued fine-tuned tests will resume them after the priority pair. Old reference-generation workers remain suspended. |
| 2026-09-13 | 18:38 | Old mixed-wave data preparation for balanced-model fine-tuning | The current dataset omits old shallow/mid/wide mixed wavetrains. User requested a short local fine-tune to test whether restoring these examples improves rollout performance without damping. | Copy six original shallow_steep shards into experiment-only NPY arrays; retain120 frames per simulation and original order-six labels. Seed0 split by whole simulation into80/10/10. Reuse existing fine-tuner with a dataset-path override and corrected output provenance, source commit0e74921 (+5 net lines). Preserve the balanced checkpoint and its normalization; do not add legacy input support to production loaders. | PASS:1,440,000 finite rows,12,000 distinct initial-state hashes; train1,152,000/9600 simulations, validation and test144,000/1200 each. Physical fields bit-identical float32; metadata only promoted to float64. IDs11/12/13 mean zero Tanaka selection. Loader,13 tangent tests, exact-fit gradient regression, Ruff and scoped Pyright pass. | 00:03 (18:38–18:41;183.4135s conversion). | Dataset ready. Original balanced rollout workers paused with70 results saved; source dataset/checkpoint unchanged. Real-checkpoint smoke and full fine-tune recorded separately below. |
| 2026-09-13 | 18:42 | Mixed-wave fine-tune: real two-GPU warm-start smoke | The new dataset changes input amplitudes and has no Tanaka rows; verify the actual checkpoint, zero-selection loss path, two-device batches and checkpoint saving before a full run. | Execute existing fine-tuner in memory with2,049 evenly spaced TRAIN rows and1,025 validation rows, global batch1024 and1epoch/LR2e-6. Three real updates include a replicated singleton tail; validation uses FP64. Restore the saved final checkpoint and compare normalization metadata with the balanced source. | PASS: three replicated optimizer/state updates, finite training/validation/parameters; train/val translation loss exactly0; source statistics unchanged; final epoch1 checkpoint reloads. Smoke validation L2.00273133 and composite.00381602 are subset results, not rollout evidence. Initial harness import failed before training; adding its script-directory import path fixed the harness, with no further production changes. | 00:01 (successful smoke18:42–18:43, about43s including import/checkpoint validation). | Proceed with the existing five-epoch trainer. Smoke checkpoint is not used as the full-run initialization. |
| 2026-09-13 | 18:44 | Balanced C27 fine-tuned on old mixed-wave families | Adding the missing mixed wavetrains may improve learned dynamics away from clean solitary-wave profiles. Test this intervention by warm-starting the balanced model while keeping its normalization and architecture fixed. | Launch c27_balanced_mixed_wave_finetune_20260913 from original balanced epoch260, source0e74921. Five epochs,1,125 updates/epoch, batch1024,two RTX6000Ada GPUs, fresh AdamW cosine LR2e-6,weight decay1e-4,no warmup. Train exclusively on1,152,000 old packet rows with simulation-disjoint144,000-row validation. L2+mode6+Hadamard.01/every16/global8 retained; Tanaka-only translation naturally0 because packet IDs11/12/13 are not Tanaka. | RUNNING: startup config/scales audited, PID251906; at18:46 both GPUs100% and120+ finite updates. FP32 training/FP64 validation. Queued all128 exact previous ICs for FP64 unguarded final-checkpoint rollouts, two failed Tanakas first; no reference generation. Source checkpoint and datasets unchanged. | Submitted18:44, trainer started18:45. Warm speed about2.6updates/s; final training/evaluation timings pending. | Evaluate as an ablation, not a proven fix. Fine-tuned workers resume original balanced panel PIDs213458/213464 after their first priority case; remaining old/new panels share GPUs. Coordinator also resumes these identities on exit. Older reference-generation jobs stay suspended. |

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

### Mixed-wave fine-tuning dataset prepared at 18:41

The inline CPU command below ran from 18:38:10 to 18:41:13 EDT
(183.414 seconds, 00:03). It prepared
`outputs/c27_mixed_wave_finetune_20260913_dataset` from the six existing
`shallow_steep{,_mid,_wide}_v1_shard{00,01}.npz` archives, without adding a
production loader, conversion script, or new simulation.

All 1,440,000 rows retain their original saved float32 eta, xi, and order-six
DNO labels exactly. The old rollout used float64, but its stored fields were
float32; promoting depth, time, and grid metadata to float64 preserves their
stored values and does not recover additional precision. No recentering,
rescaling, filtering, or relabeling was applied.

There are 12,000 simulations, each with all 120 retained frames. A seed-zero
permutation assigns whole simulations to 9,600 TRAIN / 1,200 validation /
1,200 TEST simulations, giving 1,152,000 / 144,000 / 144,000 rows. Every
physical field was checked for finiteness and exact source equality while
copying. All 12,000 initial (eta, xi, depth) byte hashes are distinct, including
across shards. IDs 11/12/13 retain the old shallow/mid/wide source meanings;
there are no family-2 rows, so a Tanaka-only translation mask is false.

TRAIN row counts for families 11/12/13 are 380,040 / 387,000 / 384,960;
validation counts are 51,600 / 47,160 / 45,240; TEST counts are
48,360 / 45,840 / 49,800. The dataset's `preparation.json` records source
metadata, original per-shard case IDs, assigned simulation-ID ranges, and
validation results. No simulation crosses a split boundary.

Executed with `UV_CACHE_DIR=/tmp/uv-cache uv run --offline --no-sync python -u -`
using this stdin code:

```python
import hashlib
import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import zipfile
import numpy as np

started = time.perf_counter()
started_local = datetime.now().astimezone().isoformat(timespec="seconds")
output = Path("outputs/c27_mixed_wave_finetune_20260913_dataset").resolve()
assert not output.exists(), output
sources = [
    (family_id, name, Path(f"data/{name}_v1_shard{shard:02d}.npz"))
    for family_id, name in ((11, "shallow_steep"), (12, "shallow_steep_mid"), (13, "shallow_steep_wide"))
    for shard in range(2)
]
batches = []
source_summaries = []
simulation_count = 0
row_count = 0
grid = None
for family_id, name, path in sources:
    with zipfile.ZipFile(path) as archive:
        meta = json.loads(archive.read("meta.json"))
    with np.load(path, allow_pickle=False) as archive:
        source_grid = archive["x"]
        if grid is None:
            grid = source_grid
        assert np.array_equal(grid, source_grid)
        assert meta["keep_samples"] == 120
        case_ids = []
        for key in sorted(key for key in archive.files if key.startswith("case_id_batch_")):
            tag = key.removeprefix("case_id_")
            row_cases = archive[key]
            cases, counts = np.unique(row_cases, return_counts=True)
            assert np.all(counts == 120)
            assert np.array_equal(row_cases, np.repeat(cases, counts))
            batches.append((path, tag, family_id, name, simulation_count, row_count, cases))
            case_ids.extend(map(int, cases))
            simulation_count += len(cases)
            row_count += row_cases.size
        assert len(case_ids) == len(set(case_ids)) == 2000
        source_summaries.append({
            "path": str(path), "bytes": path.stat().st_size,
            "family_id": family_id, "parameter_group_id": name,
            "metadata": meta, "source_case_ids": case_ids,
            "simulation_id_start": simulation_count - len(case_ids),
            "simulations": len(case_ids), "rows": len(case_ids) * 120,
        })
assert simulation_count == 12000 and row_count == 1440000 and grid is not None
order = np.random.default_rng(0).permutation(simulation_count)
splits = np.full(simulation_count, "train", dtype="U10")
splits[order[9600:10800]] = "validation"
splits[order[10800:]] = "test"
state_hashes = {}
print(f"Prepared {simulation_count} simulation identities / {row_count} rows; writing fields", flush=True)
with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
    directory = Path(temporary)
    arrays = {
        name: np.lib.format.open_memmap(
            directory / f"{name}.npy", mode="w+", dtype=dtype,
            shape=(row_count, grid.size) if name in {"eta", "xi", "gxi"} else (row_count,),
        )
        for name, dtype in {
            "eta": "float32", "xi": "float32", "gxi": "float32",
            "depth": "float64", "time": "float64", "dataset_split": "U10",
            "family_id": "int16", "simulation_id": "int64", "frame_index": "int32",
            "parameter_group_id": "U18",
        }.items()
    }
    for path, tag, family_id, name, first_simulation, first_row, cases in batches:
        with np.load(path, allow_pickle=False) as archive:
            size = len(cases) * 120
            rows = slice(first_row, first_row + size)
            initial_fields = {}
            for field in ("eta", "xi", "gxi", "depth", "time"):
                values = archive[f"{field}_{tag}"]
                assert values.shape == ((size, grid.size) if field in {"eta", "xi", "gxi"} else (size,))
                assert np.isfinite(values).all(), (path, tag, field)
                if field in {"eta", "xi", "gxi"}:
                    assert values.dtype == np.float32
                if field in {"eta", "xi", "depth"}:
                    initial_fields[field] = values[::120].copy()
                if field == "depth":
                    assert np.all(values > 0) and np.array_equal(values, np.repeat(values[::120], 120))
                if field == "time":
                    times = values.reshape(-1, 120)
                    assert np.all(times[:, 0] == 0) and np.all(np.diff(times, axis=1) > 0)
                    assert np.array_equal(times, np.broadcast_to(archive["subsample_times"], times.shape))
                arrays[field][rows] = values
                assert np.array_equal(arrays[field][rows], values), (path, tag, field)
            for local, case_id in enumerate(cases):
                digest = hashlib.sha256()
                for field in ("eta", "xi", "depth"):
                    digest.update(initial_fields[field][local].tobytes())
                key = digest.hexdigest()
                identity = (str(path), int(case_id))
                assert key not in state_hashes, (identity, state_hashes.get(key))
                state_hashes[key] = identity
            ids = np.repeat(np.arange(first_simulation, first_simulation + len(cases)), 120)
            arrays["simulation_id"][rows] = ids
            arrays["dataset_split"][rows] = splits[ids]
            arrays["family_id"][rows] = family_id
            arrays["parameter_group_id"][rows] = name
            arrays["frame_index"][rows] = np.tile(np.arange(120), len(cases))
        print(f"Copied and checked {path.name}/{tag}: {size} rows", flush=True)
    for values in arrays.values():
        values.flush()
    np.save(directory / "x.npy", grid.astype(np.float64))
    ids, starts, counts = np.unique(arrays["simulation_id"], return_index=True, return_counts=True)
    assert np.array_equal(ids, np.arange(12000)) and np.all(counts == 120)
    assert np.array_equal(arrays["dataset_split"], np.repeat(splits, 120))
    assert np.array_equal(arrays["frame_index"], np.tile(np.arange(120), 12000))
    assert len(state_hashes) == simulation_count
    assert np.array_equal(np.unique(arrays["family_id"]), [11, 12, 13])
    assert not np.any(arrays["family_id"] == 2)
    summary = {
        "started_local": started_local,
        "finished_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_seconds": time.perf_counter() - started,
        "dataset": str(output), "rows": row_count, "simulations": simulation_count,
        "frames_per_simulation": 120, "split_seed": 0,
        "split_counts": {
            split: {
                "simulations": int(np.count_nonzero(splits == split)),
                "rows": int(np.count_nonzero(arrays["dataset_split"] == split)),
                "rows_per_family": {
                    str(family): int(np.count_nonzero((arrays["dataset_split"] == split) & (arrays["family_id"] == family)))
                    for family in (11, 12, 13)
                },
            }
            for split in ("train", "validation", "test")
        },
        "physical_fields_byte_identical_to_source_float32": True,
        "metadata_values_preserved_promoted_to_float64": True,
        "all_fields_finite": True, "simulation_disjoint_splits": True,
        "unique_initial_state_hashes": len(state_hashes), "tanaka_family_rows": 0,
        "labels": "Original saved float32 order-six DNO labels; no relabeling or new simulation",
        "label_source_code": "2ec525f:solver/gen_data/generate_shallow_steep_dataset.py",
        "sources": source_summaries,
    }
    (directory / "preparation.json").write_text(json.dumps(summary, indent=2) + "\n")
    directory.rename(output)
print(json.dumps({key: value for key, value in summary.items() if key != "sources"}, indent=2), flush=True)
```

### Mixed-wave GPU smoke and queued fine-tune (18:42–18:45)

Source commit: `0e74921`. Successful smoke console:
`outputs/c27_balanced_mixed_wave_finetune_smoke_20260913.console.log`.
The following inline command was run with the same GPU environment as the full
launch below. The first attempt omitted the script import path and exited before
training; this is the corrected command that passed actual updates and reload.

```python
import os
os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import json
import runpy
import sys
from pathlib import Path
import jax
import numpy as np

source = Path("outputs/c27_balanced_tanaka_tangent_modal_20260913").resolve()
dataset = Path("outputs/c27_mixed_wave_finetune_20260913_dataset").resolve()
output = Path("outputs/c27_balanced_mixed_wave_finetune_smoke_20260913").resolve()
sys.path.insert(0, str(Path("train-jax-10m").resolve()))
program = runpy.run_path("train-jax-10m/finetune_tanaka.py", run_name="finetune_smoke")
globals_ = program["main"].__globals__
split = globals_["build_dataset_split_indices"]
globals_["build_dataset_split_indices"] = lambda arrays: tuple(
    indices[np.linspace(0, len(indices) - 1, min(limit, len(indices)), dtype=int)]
    for indices, limit in zip(split(arrays), (2049, 1025, 1), strict=True)
)
sys.argv = ["finetune_tanaka.py", "--run_dir", str(source), "--output_dir", str(output),
            "--dataset", str(dataset), "--epochs", "1", "--lr", "2e-6", "--batch_size", "1024"]
program["main"]()
with (output / "final_ckpt/metadata.json").open() as handle:
    metadata = json.load(handle)
with (source / "final_ckpt/metadata.json").open() as handle:
    source_metadata = json.load(handle)
assert metadata["stats"] == source_metadata["stats"]
assert metadata["epoch"] == 1
assert metadata["history"][0]["train_tangent_loss"] == 0
assert metadata["history"][0]["val_tangent_loss"] == 0
loaded = globals_["load_run"](output, checkpoint="final")
assert loaded.epoch == 1
assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(loaded.params))
print("SMOKE PASS: three real two-GPU updates including singleton tail; finite FP64 validation; original scales preserved; final checkpoint reloads; tangent zero.", flush=True)
```

Full run is persistent tmux session `c27_mixed_wave_finetune_20260913`.
Environment: `CUDA_VISIBLE_DEVICES=0,1 JAX_PLATFORMS=cuda UV_OFFLINE=1
UV_NO_SYNC=1 UV_NO_CACHE=1 OMP_NUM_THREADS=4`, with
`MPLCONFIGDIR=/tmp/matplotlib-c27-mixed-finetune`. It executes the following
inline Python using `uv run python -u -c`. Pipeline console:
`outputs/c27_balanced_mixed_wave_finetune_20260913.pipeline.log`.
The trainer has a separate `.console.log` beside it. Evaluation consoles/results
are under the new run. The embedded evaluation is the earlier full-panel code,
with the new checkpoint/epoch and baseline-worker resumption after each first case.
No external scheduler or new production evaluator was introduced.

```python
import os
os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import json
from pathlib import Path
import signal
import subprocess
import sys

source = Path("outputs/c27_balanced_tanaka_tangent_modal_20260913").resolve()
run_dir = Path("outputs/c27_balanced_mixed_wave_finetune_20260913").resolve()
dataset = Path("outputs/c27_mixed_wave_finetune_20260913_dataset").resolve()
evaluation_code = "import os\nos.environ.setdefault(\"NCCL_P2P_LEVEL\", \"PHB\")\nos.environ.setdefault(\"XLA_PYTHON_CLIENT_PREALLOCATE\", \"false\")\n\nimport json\nimport signal\nimport sys\nfrom pathlib import Path\nimport numpy as np\nimport jax\nimport jax.numpy as jnp\nfrom solver.evals import eval_suite as ev\n\njax.config.update(\"jax_enable_x64\", True)\ngpu = int(sys.argv[1])\nrun_dir = Path(\"outputs/c27_balanced_mixed_wave_finetune_20260913\").resolve()\ndataset = Path(\"outputs/paper_dataset_balanced_20260912/arrays\").resolve()\narchive_dir = Path(\"outputs/c27_paper_dataset_20260908_141423/eval_final_n32\")\nout_dir = run_dir / \"eval_final_n32_fp64\"\nout_dir.mkdir(exist_ok=True)\nloaded = ev.load_run(run_dir, checkpoint=\"final\")\nassert loaded.epoch == 5\nassert len(jax.devices()) == 1 and jax.devices()[0].platform == \"gpu\"\nassert all(p.dtype == jnp.float64 for p in jax.tree_util.tree_leaves(loaded.params))\npredict = ev.build_predict_gxi_batched(loaded)\npanels = {}\nfor family in ev.FAMILY_CONFIGS:\n    ics, _, nx, length = ev._load_paper_dataset_ics(dataset, family, 32)\n    with np.load(archive_dir / f\"{family}_trajs.npz\") as archive:\n        assert [ic.simulation_id for ic in ics] == archive[\"simulation_ids\"].tolist()\n    panels[family] = ics\n\nfailed_ids = (16471, 16624)\njobs = [(\"tanaka\", [ic for ic in panels[\"tanaka\"] if ic.simulation_id == failed_ids[gpu]])]\nfor family in (\"stokes\", \"jonswap_tma\", \"tanaka\", \"benjamin_feir\"):\n    selected = [ic for ic in panels[family] if ic.simulation_id not in failed_ids]\n    selected = selected[gpu::2]\n    jobs.extend((family, selected[start:start + 2]) for start in range(0, len(selected), 2))\nassert sum(len(ics) for _, ics in jobs) == 64\nprint(f\"GPU {gpu}: fine-tuned epoch5 FP64; queued64 exact previous ICs, known failure first; no reference generation or soliton damping\", flush=True)\n\ncompleted = 0\nfor family, ics in jobs:\n    cfg = ev.FAMILY_CONFIGS[family]\n    times = np.arange(0.0, cfg.tmax + 0.5 * cfg.dt, cfg.dt, dtype=np.float64)\n    ids = [ic.simulation_id for ic in ics]\n    probe = predict(\n        jnp.asarray(np.stack([ic.eta for ic in ics])),\n        jnp.asarray(np.stack([ic.xi for ic in ics])),\n        jnp.asarray(np.log([ic.depth for ic in ics])),\n    )\n    assert probe.dtype == jnp.float64 and np.isfinite(np.asarray(probe)).all()\n    print(f\"START GPU{gpu} {family} IDs={ids} T={cfg.tmax} dt={cfg.dt/cfg.substeps}\", flush=True)\n    pred = ev.surrogate_rollout_batched(ics, jnp.asarray(times), nx, length, cfg, predict)\n    assert all(np.asarray(pred[name]).dtype == np.float64 for name in (\"eta\", \"xi\", \"gxi\"))\n    with np.load(archive_dir / f\"{family}_trajs.npz\") as archive:\n        positions = [archive[\"simulation_ids\"].tolist().index(sid) for sid in ids]\n        reference_eta = archive[\"truth_eta\"][:, positions].astype(np.float64)\n    with np.errstate(over=\"ignore\", invalid=\"ignore\"):\n        rel_l2_eta = ev._rel_l2(np.asarray(pred[\"eta\"]), reference_eta)\n    for index, ic in enumerate(ics):\n        fields = {name: np.asarray(pred[name])[:, index:index + 1] for name in (\"eta\", \"xi\", \"gxi\")}\n        finite = np.logical_and.reduce([np.isfinite(values).all(axis=(1, 2)) for values in fields.values()])\n        bad = np.flatnonzero(~finite)\n        np.savez_compressed(\n            out_dir / f\"simulation_{ic.simulation_id}.npz\",\n            times=times, simulation_ids=np.asarray([ic.simulation_id]),\n            depths=np.asarray([ic.depth]), initial_eta=ic.eta, initial_xi=ic.xi,\n            **{f\"pred_{name}\": values for name, values in fields.items()},\n            rel_l2_eta=rel_l2_eta[:, index:index + 1],\n        )\n        final_error = float(rel_l2_eta[-1, index])\n        summary = {\n            \"family\": family, \"simulation_id\": ic.simulation_id,\n            \"dataset_row\": ic.meta[\"dataset_row\"], \"depth\": ic.depth,\n            \"checkpoint\": str(run_dir / \"final_ckpt\"), \"epoch\": loaded.epoch,\n            \"model_and_integration_dtype\": \"float64\", \"extra_stabilizer\": False,\n            \"internal_dt\": cfg.dt / cfg.substeps, \"save_dt\": cfg.dt, \"tmax\": cfg.tmax,\n            \"picard_iterations\": ev.GL2_ITERATIONS, \"cutoff\": cfg.filter_fraction * nx / 2,\n            \"reference_generation\": False, \"gpu\": gpu,\n            \"reference\": str(archive_dir / f\"{family}_trajs.npz\"),\n            \"reference_storage_dtype\": \"float32\",\n            \"all_saved_values_finite\": bool(finite.all()),\n            \"first_nonfinite_saved_time\": float(times[bad[0]]) if bad.size else None,\n            \"final_rel_l2_eta_vs_archived_fp32_reference\": final_error if np.isfinite(final_error) else None,\n            \"chunk_wall_s\": float(pred[\"wall_s\"]), \"chunk_size\": len(ics),\n        }\n        ev._write_json(out_dir / f\"simulation_{ic.simulation_id}.json\", summary)\n        completed += 1\n        print(f\"DONE {completed}/64 GPU{gpu}: \" + json.dumps(summary, allow_nan=False), flush=True)\n        if completed == 1:\n            baseline_pid = (213458, 213464)[gpu]\n            command = Path(f\"/proc/{baseline_pid}/cmdline\").read_bytes().decode()\n            assert \"outputs/c27_balanced_tanaka_tangent_modal_20260913\" in command\n            assert \"surrogate_rollout_batched\" in command\n            os.kill(baseline_pid, signal.SIGCONT)\n            print(f\"Resumed baseline panel PID{baseline_pid}; subsequent old/new panel jobs share GPU{gpu}\", flush=True)\nprint(f\"COMPLETE GPU{gpu}: all64 ICs saved\", flush=True)\n"
try:
    print("TRAIN START: five epochs, both GPUs, mixed waves only, unchanged source scales", flush=True)
    with Path(str(run_dir) + ".console.log").open("w") as handle:
        subprocess.run(
            [sys.executable, "-u", "train-jax-10m/finetune_tanaka.py",
             "--run_dir", str(source), "--output_dir", str(run_dir),
             "--dataset", str(dataset), "--epochs", "5", "--lr", "2e-6",
             "--batch_size", "1024", "--translation_tangent_weight", "10", "--seed", "0"],
            stdout=handle, stderr=subprocess.STDOUT, check=True,
        )
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["epochs_completed"] == 5
    original = json.loads((source / "final_ckpt/metadata.json").read_text())
    final = json.loads((run_dir / "final_ckpt/metadata.json").read_text())
    assert final["epoch"] == 5 and final["stats"] == original["stats"]
    print("TRAIN COMPLETE: queueing full128-IC FP64 panel; known failures first", flush=True)
    workers = []
    for gpu in (0, 1):
        with (run_dir / f"eval_gpu{gpu}.console.log").open("w") as handle:
            workers.append(subprocess.Popen(
                [sys.executable, "-u", "-c", evaluation_code, str(gpu)],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                stdout=handle, stderr=subprocess.STDOUT,
            ))
    returncodes = [worker.wait() for worker in workers]
    assert returncodes == [0, 0], returncodes
    print("COMPLETE: fine-tuned full128-IC panel saved", flush=True)
finally:
    for pid in (213458, 213464):
        command_path = Path(f"/proc/{pid}/cmdline")
        if command_path.exists():
            command = command_path.read_bytes().decode()
            if str(source.relative_to(Path.cwd())) in command and "surrogate_rollout_batched" in command:
                os.kill(pid, signal.SIGCONT)
                print(f"Baseline rollout worker {pid} resumed", flush=True)
```
