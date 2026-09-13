| Date | Time | File / Variant | Motivation | What Tried / Evidence | Correctness | Timing | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-13 | 01:53 | Balanced C27 Modal launch preparation | Local two-GPU training could not allocate memory alongside another user's job; user requested Modal instead. Reuse the same trainer and batch/update budget on two A100-80GB GPUs. | Confirmed Modal CLI1.4.2 and profile/workspace sciml-at-ud. Balanced arrays are not present at /outputs/paper_dataset_balanced_20260912/arrays on volume dno-fno-train-data. Prepared existing launcher with EPOCHS260, batch1024, seed0, same losses, GPU_SPEC=A100-80GB:2. Match image Python minor version3.11 and pin numerical packages to local versions; correct stale all-family translation label. Requested upload of11 NPY files totaling14.715GB plus scoped trainer/model/solver source. | Local Ruff/Pyright and shell syntax pass; independent launcher/path/hyperparameter audit passes. Remote image build and training NOT tested: approval system rejected upload before process creation, requiring explicit permission for disclosure to this specific Modal workspace/volume. No upload, remote training or local job pause occurred; our failed local trainer was already stopped. | 00:02 (inspection/preparation01:51–01:53; no GPU training). | User explicitly approved this destination and two-A100 training before01:59; see the following execution record. Initial rejection was not bypassed. |
| 2026-09-13 | 01:59 | Approved balanced dataset upload and Modal C27 run | User approved disclosure of the14.715GB balanced dataset and scoped training code to sciml-at-ud/dno-fno-train-data, plus training on two A100-80GB GPUs, after local GPU-memory failure. | Existing upload entrypoint completed11 NPY files; existing detached launcher submitted c27_balanced_tanaka_tangent_modal_20260913 with260 epochs, global batch1024, same architecture/losses and FP32 train/val. Training app ap-sDX1uTgQnbRpNWYkCJNgNX; function call fc-01M2CNTNT9DCNGZ57027DG328E. No smaller batch or new trainer. | COMPLETED PASS: physical devices are two A100-SXM4-80GB; replicated state/optimizer counters verified at0; all260 consecutive epoch records have finite metrics; best/final epoch260 has training loss.0003892254346, validation total.0003607055300 and validation relative-L2.0003308064795. Final/latest/best checkpoints and summary are saved on the volume; process returned0. Independent remote config SHA matches local balanced run after excluding only dataset path/run name. All pinned numerical packages match local versions; uploaded trainer sets NCCL_P2P_LEVEL=PHB before JAX. All11 uploaded filenames/sizes checked; remote row counts and normalization match. | Image/upload01:59–02:03; worker started02:04; first finite updates02:05; first epoch/validation/checkpoints verified02:09. Finished15:42; runtime13:38 (49074.695s trainer,49086s wrapper). Final log audited17:23. | TRAINING COMPLETE. No local jobs stopped or modified. Checkpoints/logs persist on approved volume; rollout/NaN tests have not run. Do not infer rollout stability from finite training metrics. This is the260-epoch matched-update-budget balanced-corpus experiment, not a40-epoch shortened run. |
| 2026-09-13 | 17:31 | Balanced C27 final checkpoint: full previous FP64 rollout panel | Equal family row counts may improve stability beyond Tanaka-only translation regularization; test the same held-out ICs, including both previous NaN failures, without changing the integrator or adding soliton damping. | Downloaded completed epoch260 checkpoint from approved Modal volume. Two local workers cover128 exact previous ICs, known Tanaka failures first. Model-only FP64 rollouts reuse archived FP32 reference fields only for labeled error comparisons; no truth generation. | PARTIAL:70/128 complete before fine-tuning preparation; all32 Stokes and32 JONSWAP plus4 other Tanaka cases finite. Both known Tanaka failures persist:16471 first saved NaNs179.2;16624 at150.4. IDs and initial arrays match originals; internal dt.01, GL2 four iterations, cutoff128, no soliton damping. | Started17:31; both workers paused during18:35–18:37 fine-tuning preparation, preserving their current batches. Remaining58 cases not complete. | Balancing did not remove either targeted failure. Workers213458/213464 are paused for the requested local fine-tune; queued fine-tuned tests will resume them after the priority pair. Old reference-generation workers remain suspended. |
| 2026-09-13 | 18:38 | Old mixed-wave data preparation for balanced-model fine-tuning | The current dataset omits old shallow/mid/wide mixed wavetrains. User requested a short local fine-tune to test whether restoring these examples improves rollout performance without damping. | Copy six original shallow_steep shards into experiment-only NPY arrays; retain120 frames per simulation and original order-six labels. Seed0 split by whole simulation into80/10/10. Reuse existing fine-tuner with a dataset-path override and corrected output provenance, source commit0e74921 (+5 net lines). Preserve the balanced checkpoint and its normalization; do not add legacy input support to production loaders. | PASS:1,440,000 finite rows,12,000 distinct initial-state hashes; train1,152,000/9600 simulations, validation and test144,000/1200 each. Physical fields bit-identical float32; metadata only promoted to float64. IDs11/12/13 mean zero Tanaka selection. Loader,13 tangent tests, exact-fit gradient regression, Ruff and scoped Pyright pass. | 00:03 (18:38–18:41;183.4135s conversion). | Dataset ready. Original balanced rollout workers paused with70 results saved; source dataset/checkpoint unchanged. Real-checkpoint smoke and full fine-tune recorded separately below. |
| 2026-09-13 | 18:42 | Mixed-wave fine-tune: real two-GPU warm-start smoke | The new dataset changes input amplitudes and has no Tanaka rows; verify the actual checkpoint, zero-selection loss path, two-device batches and checkpoint saving before a full run. | Execute existing fine-tuner in memory with2,049 evenly spaced TRAIN rows and1,025 validation rows, global batch1024 and1epoch/LR2e-6. Three real updates include a replicated singleton tail; validation uses FP64. Restore the saved final checkpoint and compare normalization metadata with the balanced source. | PASS: three replicated optimizer/state updates, finite training/validation/parameters; train/val translation loss exactly0; source statistics unchanged; final epoch1 checkpoint reloads. Smoke validation L2.00273133 and composite.00381602 are subset results, not rollout evidence. Initial harness import failed before training; adding its script-directory import path fixed the harness, with no further production changes. | 00:01 (successful smoke18:42–18:43, about43s including import/checkpoint validation). | Proceed with the existing five-epoch trainer. Smoke checkpoint is not used as the full-run initialization. |
| 2026-09-13 | 18:44 | Balanced C27 fine-tuned on old mixed-wave families | Adding the missing mixed wavetrains may improve learned dynamics away from clean solitary-wave profiles. Test this intervention by warm-starting the balanced model while keeping its normalization and architecture fixed. | Launch c27_balanced_mixed_wave_finetune_20260913 from original balanced epoch260, source0e74921. Five epochs,1,125 updates/epoch, batch1024,two RTX6000Ada GPUs, fresh AdamW cosine LR2e-6,weight decay1e-4,no warmup. Train exclusively on1,152,000 old packet rows with simulation-disjoint144,000-row validation. L2+mode6+Hadamard.01/every16/global8 retained; Tanaka-only translation naturally0 because packet IDs11/12/13 are not Tanaka. | RUNNING: startup config/scales audited, PID251906; at19:08 epochs1–2 complete with finite metrics and epoch3 underway. Epoch2 validation L2.00222811, composite.00268095, tangent0; both GPUs remain assigned. FP32 training/FP64 validation. Queued all128 exact previous ICs for FP64 unguarded final-checkpoint rollouts, two failed Tanakas first; no reference generation. Source checkpoint and datasets unchanged. | Submitted18:44, trainer started18:45. Warm speed about2.6updates/s; final training/evaluation timings pending. | Evaluate as an ablation, not a proven fix. Fine-tuned workers resume original balanced panel PIDs213458/213464 after their first priority case; remaining old/new panels share GPUs. Coordinator also resumes these identities on exit. Older reference-generation jobs stay suspended. |
| 2026-09-13 | 19:06 | Balanced NaN diagnosis: historical audit and same-state FP64 CPU probes | Earlier investigations found that accurate predictions on clean waves can coexist with unstable responses to developing distortions. Compare the surviving July C27 and balanced checkpoints on identical pre-failure states while both GPUs fine-tune. Also test mean/depth coordinates, raw output above cutoff and energy consistency instead of assuming missing data alone causes NaNs. | Actual-error feedback separates the models: for16624 at t116/t124, July instantaneous log-error-energy feedback rates are-.0974/-.1594, balanced+.1621/+.7394. At16471 t107.2/t115.2 both amplify, balanced more strongly. Initial mean/depth defects are nearly identical (~.036–.040%); raw Gxi above128 has negligible early effect. Shape-consistency defect worsens markedly for balanced16624 but not universally for16471; total-energy injection is not supported. Detailed methods, caveats, artifacts and executed code below. | PASS: CPU-only FP64; final checkpoints July40/balanced260, identical P128 states, no new rollouts or reference labels. Production-RHS agreement <=2.42e-15; canonical zero-mode control corrected before accepting results. Actual-error references are archived FP32 promoted to FP64. Feedback excludes model-reference forcing: local evidence, not a causal dataset result. | 00:01 combined successful numerical probes (energy7.10s, coordinate teacher2.09s/models1.60s, feedback2.88s); completed18:56–19:02. Earlier sandbox checkpoint restores timed out; host CPU restores succeeded without loader changes. Historical audit completed19:06. | No production change. Next discriminating experiment: short same-state checkpoint handoffs before error growth, comparing July, balanced and fine-tuned models without damping. Test stability around slightly distorted waves before broadening generation or increasing loss weights. Current five-epoch fine-tune and queued128-case panel continue unchanged. |

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

### NaN follow-up during mixed-wave fine-tuning (audited19:06)

Models compared:

- Original July C27: `outputs/c27_h1_to_l2_full_20260717_212550`, final epoch40,
  trained from scratch on v9 with L2 despite the directory name. Its later
  FP64, damping-off evaluations survived both exact ICs.
- Balanced September C27: `outputs/c27_balanced_tanaka_tangent_modal_20260913`,
  final epoch260. First saved NaNs:16471 at179.2,16624 at150.4.
- These probes do not use the fine-tuned checkpoint. Fine-tuning still runs on
  both GPUs; epoch1 completed with finite validation L2.00231120 and Hadamard.0176020.

#### Historical evidence and hypotheses not to recycle blindly

[July root-cause report](../notes/nan_root_cause_investigation_20260709.md)
found small errors on truth states alongside large disagreement between the
geometric xi evolution equation and the derivative of the learned Hamiltonian.
Short-wave growth preceded late aliasing and fixed-point failures. C16 improved
stability with several simultaneous changes, so it did not isolate a single
cause; today's model already includes its main architecture/RHS ingredients.

[C25 tail diagnosis](../notes/c25_tail_diagnosis_20260717.md) concerned finite,
mostly phase-drifting rollouts, unlike the present catastrophic failures.
The September9–10 logs already show FP64 delays rather than cures failure,
and initial positive DNO spectra do not guarantee stable coupled dynamics.
Unprojected high-order teacher outputs were not sufficiently converged for
arbitrary perturbed states. Some older hard-negative retraining worsened
rollouts; do not turn late corrupted states into labels without checking the teacher.

#### 1. Developing errors, measured on identical state pairs

For each saved balanced state, compare its eta/xi with the archived reference
at the same simulation/time. Both inputs are projected to modes at most128;
xi is mean-free. Reference storage is FP32, promoted for this FP64 computation.

For each model separately, subtract its evolution RHS at the reference state
from its RHS at the balanced state. Dot that difference with the actual state
error using the positive flat-water energy norm:
`E = mean(delta_eta^2 + delta_xi*G0(delta_xi))/2`.
The reported rate is the resulting `dE/dt / E`, i.e. instantaneous
log-error-energy feedback. It excludes the model's error at the reference state;
it is neither the full observed error derivative nor an eigenvalue.

| IC | Time | Surface relative error | July feedback rate | Balanced feedback rate |
| --- | --- | --- | --- | --- |
| 16471 | 107.2 | .0853% | +.0462 | +.0671 |
| 16471 | 115.2 | .1060% | +.0263 | +.0713 |
| 16624 | 116.0 | .0993% | -.0974 | +.1621 |
| 16624 | 124.0 | .3777% | -.1594 | +.7394 |

At16624/t124, the65–128 band has rates-.3799/+1.2052 (July/balanced);
the1–32 band has-.0811/+.5739. The mechanism is not exclusively high-frequency.
At16471 the high-band rate alone is not uniformly worse for balanced, despite
its larger total feedback at both tested times.

This is a concrete local difference in response to the *actual developing
error*, stronger evidence than random-direction tests at an initial clean wave.
It does not prove that the omitted mixed-wave families caused the difference,
or that switching checkpoints will reverse the full trajectory.

Artifacts:
`outputs/c27_balanced_tanaka_tangent_modal_20260913/diagnostics_error_feedback_20260913/summary.json`.
Eight evaluations of state pairs, two checkpoints, CPU-only:2.8805s at19:02.
Independent read-only audit confirmed matching simulation/frame indices, the
energy formula and band contributions summing to the total within6.4e-16.
The xi RHS constant does not affect this norm because G0 annihilates it.

#### 2. Energy and shape consistency: conditional support, not universal failure

The six balanced states are t0/107.2/115.2 for16471 and t0/116/124 for16624.
The last time in each group is the first saved state whose65–128 eta RMS exceeds
both10x the reference band RMS and1e-4 of reference total RMS. At those times
surface errors are only.106%/.378%, before the1% crossings at170.4/128.

Differentiate `H_theta = dx*sum(eta^2 + xi*G_theta(eta,depth)xi)/2`.
Compare its canonical RHS with the actual projected, dealiased production RHS.
Both canonical components must omit the zero mode: eta's mean is fixed and xi
has an arbitrary constant. An initial control retained the eta zero mode;
correcting that diagnostic projection removed a false control failure. The
reported production-flow energy rates were unchanged.

Relative xi-equation disagreement:

| IC / time | July | Balanced |
| --- | --- | --- |
| 16471 /0 | 7.98% | 6.31% |
| 16471 /107.2 | 41.83% | 36.36% |
| 16471 /115.2 | 61.86% | 55.06% |
| 16624 /0 | 13.61% | 10.46% |
| 16624 /116 | 123.04% | 133.38% |
| 16624 /124 | 89.85% | 187.46% |

Balanced is initially better and remains better on these16471 states.
Thus shape inconsistency alone does not distinguish both failures. At16624/t124,
normalized total-energy rates are negative: July-1.094e-4,
balanced-7.067e-4. The model can amplify trajectory errors without injecting
net total energy; do not report a simple energy-addition cause.

Controls: production-RHS agreement <=2.42e-15; reverse-gradient versus JVP
energy-rate agreement <=5.38e-16 after normalization; canonical energy-rate
residual <=1.73e-16; eta-equation relative defect <=7.59e-15; flat control passes.
Fourteen evaluations including flat controls:7.1008s, completed18:59.

Artifacts: `diagnostics_energy_20260913/{summary.json,states.npz,console.log}`
under the balanced run.

#### 3. Weakened alternatives

- **Output above128 entering nonlinear products:** a saved-field NumPy audit
  found the old surviving16471 rollout already has larger early tail effects.
  For balanced16624/t124, projecting Gxi before nonlinear products changes the
  total xi RHS by only.00148%, and shape defect187.4625% to187.4623%.
  Tail effects become large near the final finite frames (e.g.16624/t149.6),
  consistent with a late amplifier, not an established trigger. No filter change.
- **Positive eta mean / depth coordinates:** compare physically equivalent
  `(eta, h)` and `(eta-c, h+c)`, c=half/full mean.
  Full-mean prediction changes for16471 are July.03584% versus balanced.03650%;
  for16624,.03850% versus.03977%. Small and almost identical: deprioritize.
  Projected teacher orders6/8/10 converge; M10 full-shift discrepancies are
  1.79e-7/4.44e-7. CPU teacher2.0926s at18:56, models1.6036s at18:59.
  Artifacts: `diagnostics_mean_depth_20260913/{summary.json,teacher.json}`.

Initial sandbox model restores hung inside Orbax checkpoint finalization checks;
a bounded175s attempt timed out. Host execution with CPU-only JAX restored the
same checkpoints promptly. No checkpoint loader changes, GPU diagnostics, new
teacher trajectories, rollout damping or production refactor were needed.

#### Next experiment

Use a short common-state restart before ripple growth: feed the same saved
eta/xi into July, balanced and (once finished) fine-tuned models, holding FP64,
dt, iterations and cutoff fixed. Compare whether the error-feedback difference
persists over time, and separate response to an existing error from error
created on a clean reference state. This is a recommendation, not an additional
queued GPU job. The already queued full128-case panel is unchanged.

If the distinction survives that test, target training around controlled small
distortions of solitary waves, with converged projected labels/shape checks.
Neither equal family counts nor more copies of clean travelling waves guarantee
coverage of those directions.

#### Executed CPU diagnostic code

These are experiment-only commands, preserved here rather than adding production
utilities. All successful model probes used host execution with
`JAX_PLATFORMS=cpu`, empty `CUDA_VISIBLE_DEVICES`, bounded CPU affinity and FP64.
The mean/depth and error-feedback blocks set their CPU environment explicitly.

Energy/shape probe invocation (host CPU, repository working directory):

```sh
timeout 280s taskset -c 0-3 env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  UV_OFFLINE=1 UV_NO_SYNC=1 UV_NO_CACHE=1 \
  XLA_FLAGS='--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1' \
  .venv/bin/python -u /tmp/dno_energy_diagnostic_20260913.py \
  > outputs/c27_balanced_tanaka_tangent_modal_20260913/diagnostics_energy_20260913/console.log 2>&1
```

Energy/shape probe (the corrected final scratch program):

```python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path('/home/johnma/dno-fno')
RUN = ROOT / 'outputs/c27_balanced_tanaka_tangent_modal_20260913'
OLD = ROOT / 'outputs/c27_h1_to_l2_full_20260717_212550'
ARCHIVE = ROOT / 'outputs/c27_paper_dataset_20260908_141423/eval_final_n32/tanaka_trajs.npz'
OUT = RUN / 'diagnostics_energy_20260913'
OUT.mkdir(exist_ok=True)
started = perf_counter()
started_local = datetime.now().astimezone().isoformat()
nx = 1024
length = 2 * np.pi
dx = length / nx
k_np = np.fft.fftfreq(nx, 1 / nx)
retained = np.abs(k_np) <= 128
high = (np.abs(k_np) > 64) & retained
states: list[dict[str, object]] = []
eta_states: list[np.ndarray] = []
xi_states: list[np.ndarray] = []
depth_states: list[float] = []
onsets: list[dict[str, object]] = []
with np.load(ARCHIVE, allow_pickle=False) as archive:
    old_ids = archive['simulation_ids'].tolist()
    old_eta = archive['truth_eta']
    for sid in (16471, 16624):
        with np.load(RUN / f'eval_final_n32_fp64/simulation_{sid}.npz') as saved:
            times = saved['times']
            eta = saved['pred_eta'][:, 0]
            xi = saved['pred_xi'][:, 0]
            relative_error = saved['rel_l2_eta'][:, 0]
            depth = float(saved['depths'][0])
        reference = old_eta[:, old_ids.index(sid)].astype(np.float64)
        with np.errstate(invalid='ignore', over='ignore'):
            eta_fft = np.fft.fft(eta, axis=-1, norm='forward')
            ref_fft = np.fft.fft(reference, axis=-1, norm='forward')
            high_rms = np.sqrt(np.sum(np.abs(eta_fft[:, high]) ** 2, axis=-1))
            ref_high_rms = np.sqrt(np.sum(np.abs(ref_fft[:, high]) ** 2, axis=-1))
            ref_rms = np.sqrt(np.mean(reference ** 2, axis=-1))
        finite = np.isfinite(eta).all(axis=-1) & np.isfinite(xi).all(axis=-1)
        growth = np.flatnonzero(finite & (high_rms > 10 * ref_high_rms) & (high_rms > 1e-4 * ref_rms))
        if not growth.size:
            raise RuntimeError(f'No finite high-band onset for {sid}')
        onset = int(growth[0])
        pre = max(1, onset - 10)
        errors = {
            str(threshold): float(times[indices[0]]) if indices.size else None
            for threshold in (0.001, 0.01, 0.05)
            for indices in (np.flatnonzero(finite & (relative_error > threshold)),)
        }
        onsets.append({'simulation_id': sid, 'growth_time': float(times[onset]), 'eta_error_crossing_times': errors})
        for label, index in (('initial', 0), ('pre_growth', pre), ('growth_onset', onset)):
            projected_eta = np.fft.ifft(np.fft.fft(eta[index]) * retained).real
            projected_xi = np.fft.ifft(np.fft.fft(xi[index]) * retained).real
            projected_xi -= projected_xi.mean()
            states.append({
                'simulation_id': sid, 'selection': label, 'time': float(times[index]),
                'depth': depth, 'relative_eta_error': float(relative_error[index]),
                'eta_highband_rms': float(high_rms[index]),
                'reference_eta_highband_rms': float(ref_high_rms[index]),
                'projection_eta_rms_change': float(np.sqrt(np.mean((projected_eta - eta[index]) ** 2))),
                'projection_xi_rms_change': float(np.sqrt(np.mean((projected_xi - xi[index]) ** 2))),
            })
            eta_states.append(projected_eta)
            xi_states.append(projected_xi)
            depth_states.append(depth)
states.append({'simulation_id': None, 'selection': 'flat_control', 'time': 0.0, 'depth': depth_states[0]})
eta_states.append(np.zeros(nx))
xi_states.append(0.01 * np.sin(3 * np.arange(nx) * dx))
depth_states.append(depth_states[0])
np.savez_compressed(OUT / 'states.npz', eta=np.stack(eta_states), xi=np.stack(xi_states), depth=np.array(depth_states))
report: dict[str, object] = {
    'started_local': started_local, 'onset_definition': 'eta RMS in 64<abs(k)<=128 exceeds 10x archived reference band RMS and 1e-4 reference total RMS',
    'onsets': onsets, 'states': states, 'records': [],
    'state_protocol': 'Same saved balanced states for both models; project eta/xi to abs(k)<=128 and center xi, matching the retained phase space. Archived references used only for onset selection, not teacher labels.',
    'energy': 'H_theta = 0.5*dx*sum(eta**2 + xi*meanfree_model_Gxi)',
    'normalization': 'Energy rate divided by positive flat-wave E0=0.5*dx*sum(eta**2 + xi*G0(xi)); coupled defect uses sqrt(mean(delta_eta**2 + delta_xi*G0(delta_xi))).',
    'finite_difference_time_step': 1e-4,
}
(OUT / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
print(json.dumps({'onsets': onsets, 'selected_states': states}), flush=True)

import sys
sys.path.insert(0, str(ROOT))
import jax
import jax.numpy as jnp
from solver.evals import model_rollout as mr
from solver.solvers import time_integrator as ti
from solver.solvers.dno_series_jax import myfft, myifft

jax.config.update('jax_enable_x64', True)
assert jax.default_backend() == 'cpu'
k = jnp.asarray(k_np)
projector = jnp.asarray(retained)

def project(field: jax.Array) -> jax.Array:
    return jnp.fft.ifft(jnp.fft.fft(field, axis=-1) * projector, axis=-1).real

for model_name, run_dir in (('july_c27', OLD), ('balanced_c27', RUN)):
    model_started = perf_counter()
    loaded = mr.load_run(run_dir, checkpoint='final')
    assert all(x.dtype == jnp.float64 for x in jax.tree.leaves(loaded.params))
    predict = mr.build_predict_gxi_batched(loaded)

    def energy(eta: jax.Array, xi: jax.Array, log_depth: jax.Array) -> jax.Array:
        return 0.5 * dx * jnp.sum(eta * eta + xi * predict(eta, xi, log_depth))

    @jax.jit
    def diagnose(eta: jax.Array, xi: jax.Array, log_depth: jax.Array) -> dict[str, jax.Array]:
        h = jnp.exp(log_depth).reshape(-1, 1)
        g0 = jnp.abs(k)[None, :] * jnp.tanh(h * jnp.abs(k)[None, :])
        q = predict(eta, xi, log_depth)
        eta_x = ti.spectral_dx(eta, k)
        xi_x = ti.spectral_dx(xi, k)
        rhs_eta = project(q)
        rhs_xi = project(-eta + ti.dealiased_zakharov_xi_rhs(eta_x, xi_x, q))
        rhs_xi -= rhs_xi.mean(axis=-1, keepdims=True)
        H, (grad_eta, grad_xi) = jax.value_and_grad(energy, argnums=(0, 1))(eta, xi, log_depth)
        canonical_eta = project(grad_xi / dx)
        canonical_eta -= canonical_eta.mean(axis=-1, keepdims=True)
        canonical_xi = -project(grad_eta / dx)
        canonical_xi -= canonical_xi.mean(axis=-1, keepdims=True)
        defect_eta = rhs_eta - canonical_eta
        defect_xi = rhs_xi - canonical_xi
        Hdot = jnp.sum(grad_eta * rhs_eta + grad_xi * rhs_xi)
        filtered_rhs_xi = project(-eta + ti.dealiased_zakharov_xi_rhs(eta_x, xi_x, project(q)))
        filtered_rhs_xi -= filtered_rhs_xi.mean(axis=-1, keepdims=True)
        filtered_Hdot = jnp.sum(grad_eta * rhs_eta + grad_xi * filtered_rhs_xi)
        filtered_defect_xi = filtered_rhs_xi - canonical_xi
        _, Hdot_jvp = jax.jvp(energy, (eta, xi, log_depth), (rhs_eta, rhs_xi, jnp.zeros_like(log_depth)))
        eps = 1e-4
        Hdot_fd = (energy(eta + eps * rhs_eta, xi + eps * rhs_xi, log_depth) - energy(eta - eps * rhs_eta, xi - eps * rhs_xi, log_depth)) / (2 * eps)
        xi_g0 = jnp.fft.ifft(jnp.fft.fft(xi, axis=-1) * g0, axis=-1).real
        E0 = 0.5 * dx * jnp.sum(eta * eta + xi * xi_g0)
        rhs_xi_g0 = jnp.fft.ifft(jnp.fft.fft(rhs_xi, axis=-1) * g0, axis=-1).real
        defect_xi_g0 = jnp.fft.ifft(jnp.fft.fft(defect_xi, axis=-1) * g0, axis=-1).real
        coupled_defect = jnp.sqrt(jnp.maximum(jnp.mean(defect_eta**2 + defect_xi * defect_xi_g0), 0))
        coupled_rhs = jnp.sqrt(jnp.maximum(jnp.mean(rhs_eta**2 + rhs_xi * rhs_xi_g0), 0))
        params = ti.SolverParams(nx=nx, length=length, depth=h, gravity=1.0, dno_order=6, pad_factor=8, filter_fraction=0.25, k=k, g0=g0)
        spectral = ti.SpectralState(myfft(eta, nx), myfft(xi, nx))
        nonlinear = mr._rhs_nonlinear_if_surrogate(spectral, jnp.float64(0), params, lambda e, x: predict(e, x, log_depth))
        actual_eta = project(myifft(nonlinear.eta_hat) + xi_g0)
        actual_xi = project(myifft(nonlinear.xi_hat) - eta)
        actual_xi -= actual_xi.mean(axis=-1, keepdims=True)
        return {
            'H': H, 'flat_energy': E0, 'Hdot': Hdot, 'Hdot_over_flat_energy': Hdot / E0,
            'filtered_Gxi_Hdot_over_flat_energy': filtered_Hdot / E0,
            'filtered_Gxi_rhs_xi_relative_difference': jnp.linalg.norm(filtered_rhs_xi - rhs_xi) / jnp.maximum(jnp.linalg.norm(rhs_xi), 1e-30),
            'filtered_Gxi_xi_hamiltonian_defect_relative': jnp.linalg.norm(filtered_defect_xi) / jnp.maximum(jnp.linalg.norm(filtered_rhs_xi), 1e-30),
            'Hdot_jvp': Hdot_jvp, 'Hdot_fd': Hdot_fd,
            'jvp_absolute_disagreement_over_flat_energy': jnp.abs(Hdot_jvp - Hdot) / E0,
            'fd_absolute_disagreement_over_flat_energy': jnp.abs(Hdot_fd - Hdot) / E0,
            'canonical_flow_energy_rate_over_flat_energy': jnp.sum(grad_eta * canonical_eta + grad_xi * canonical_xi) / E0,
            'coupled_hamiltonian_defect_relative': coupled_defect / jnp.maximum(coupled_rhs, 1e-30),
            'eta_hamiltonian_defect_relative': jnp.linalg.norm(defect_eta) / jnp.maximum(jnp.linalg.norm(rhs_eta), 1e-30),
            'xi_hamiltonian_defect_relative': jnp.linalg.norm(defect_xi) / jnp.maximum(jnp.linalg.norm(rhs_xi), 1e-30),
            'actual_rhs_max_absolute_disagreement': jnp.maximum(jnp.max(jnp.abs(actual_eta - rhs_eta)), jnp.max(jnp.abs(actual_xi - rhs_xi))),
            'rms_rhs_eta': jnp.sqrt(jnp.mean(rhs_eta**2)), 'rms_rhs_xi': jnp.sqrt(jnp.mean(rhs_xi**2)),
        }

    for index, state_meta in enumerate(states):
        values = diagnose(jnp.asarray(eta_states[index][None]), jnp.asarray(xi_states[index][None]), jnp.asarray([np.log(depth_states[index])]))
        values = {key: float(value) for key, value in jax.device_get(values).items()}
        if not all(np.isfinite(value) for value in values.values()):
            raise FloatingPointError(values)
        if values['actual_rhs_max_absolute_disagreement'] > 1e-10 or values['jvp_absolute_disagreement_over_flat_energy'] > 1e-10 or abs(values['canonical_flow_energy_rate_over_flat_energy']) > 1e-10:
            raise AssertionError(values)
        record = {'model': model_name, 'checkpoint_epoch': loaded.epoch, 'state_index': index, **state_meta, **values}
        report['records'].append(record)
        report['elapsed_seconds'] = perf_counter() - started
        (OUT / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps(record, allow_nan=False), flush=True)
    print(f'MODEL_COMPLETE {model_name} seconds={perf_counter() - model_started:.3f}', flush=True)
    jax.clear_caches()

report['completed_local'] = datetime.now().astimezone().isoformat()
report['elapsed_seconds'] = perf_counter() - started
report['status'] = 'complete'
(OUT / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
print(f'COMPLETE {report["elapsed_seconds"]:.3f}s {OUT}', flush=True)
```

Mean/depth projected teacher:

```python
import os
os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", XLA_FLAGS="--xla_cpu_multi_thread_eigen=false")
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-4:])
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from solver.solvers.dno_series_jax import dno_series_eval
jax.config.update("jax_enable_x64", True)
start=time.monotonic()
root=Path("outputs/c27_balanced_tanaka_tangent_modal_20260913")
eta,xi,h=[],[],[]
mask=np.arange(513)<=128
for sim in (16471,16624):
    with np.load(root/"eval_final_n32_fp64"/f"simulation_{sim}.npz") as z:
        e,x,d=z["initial_eta"],z["initial_xi"],float(z["depths"][0])
    e=np.fft.irfft(np.fft.rfft(e)*mask,n=1024)
    x=np.fft.irfft(np.fft.rfft(x)*mask,n=1024);x-=x.mean()
    for fraction in (0.,.5,1.):
        c=float(e.mean())*fraction
        eta.append(e-c);xi.append(x);h.append(d+c)
eta,xi,h=map(jnp.asarray,(np.stack(eta),np.stack(xi),np.array(h)[:,None]))
k=jnp.asarray(2*np.pi*np.fft.fftfreq(1024,d=2*np.pi/1024))
teachers={}
result=dict(started_local=datetime.now().astimezone().isoformat(),state_order="16471 c=0,halfmean,mean;16624 c=0,halfmean,mean",pad_factor=8,output_cutoff=128,dtype="float64",models_not_evaluated_here=True)
for order in (6,8,10):
    stage=time.monotonic()
    value=np.asarray(dno_series_eval(eta,xi,k,h,order,pad_factor=8))
    value=np.fft.irfft(np.fft.rfft(value)*mask,n=1024)
    value-=value.mean(axis=-1,keepdims=True)
    teachers[order]=value
    result[str(order)]=dict(seconds=time.monotonic()-stage,invariance_relative_to_same_order_c0=[float(np.linalg.norm(value[i]-value[(i//3)*3])/np.linalg.norm(value[(i//3)*3])) for i in range(6)])
    print(order,result[str(order)],flush=True)
result["order_comparisons"]={f"{lo}_to_{hi}":[float(v) for v in np.linalg.norm(teachers[hi]-teachers[lo],axis=-1)/np.linalg.norm(teachers[hi],axis=-1)] for lo,hi in ((6,8),(8,10))}
result["all_finite"]=all(np.isfinite(v).all() for v in teachers.values())
result["elapsed_seconds"]=time.monotonic()-start
result["completed_local"]=datetime.now().astimezone().isoformat()
(root/"diagnostics_mean_depth_20260913"/"teacher.json").write_text(json.dumps(result,indent=2)+"\n")
print("FINAL",json.dumps(result),flush=True)
```

Mean/depth successful model probe:

```python
import os
os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1", XLA_FLAGS="--xla_cpu_multi_thread_eigen=false", MPLCONFIGDIR="/tmp/dno-mean-depth-matplotlib")
os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-4:])
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from solver.evals.model_rollout import load_run, build_predict_gxi_batched
jax.config.update("jax_enable_x64", True)
start=time.monotonic()
root=Path("outputs/c27_balanced_tanaka_tangent_modal_20260913")
destination=root/"diagnostics_mean_depth_20260913"
result=json.loads((destination/"summary.json").read_text())
result["host_model_attempt"]=dict(started_local=datetime.now().astimezone().isoformat(),cpu_affinity=sorted(os.sched_getaffinity(0)),jax_devices=[str(d) for d in jax.devices()],protocol="Same physical domain eta-c,h+c; xi unchanged and mean-free; P128 inputs and reported model outputs; six initial states, final checkpoints, float64 CPU.")
eta,xi,h=[],[],[]
states=[]
mask=np.arange(513)<=128
for sim in (16471,16624):
    with np.load(root/"eval_final_n32_fp64"/f"simulation_{sim}.npz") as z:
        e,x,d=z["initial_eta"],z["initial_xi"],float(z["depths"][0])
    e=np.fft.irfft(np.fft.rfft(e)*mask,n=1024)
    x=np.fft.irfft(np.fft.rfft(x)*mask,n=1024)
    x-=x.mean()
    for fraction in (0.,.5,1.):
        c=float(e.mean())*fraction
        eta.append(e-c);xi.append(x);h.append(d+c)
        states.append(dict(simulation_id=sim,mean_removed_fraction=fraction,c=c,depth=d+c,eta_mean=float((e-c).mean())))
eta,xi,h=map(jnp.asarray,(np.stack(eta),np.stack(xi),np.array(h)))
result["models"]={}
for name,run in (("july",Path("outputs/c27_h1_to_l2_full_20260717_212550")),("balanced",root)):
    stage=time.monotonic()
    loaded=load_run(run,checkpoint="final")
    print("LOADED",name,time.monotonic()-stage,flush=True)
    predict=build_predict_gxi_batched(loaded)
    raw=np.asarray(predict(eta,xi,jnp.log(h)))
    value=np.fft.irfft(np.fft.rfft(raw)*mask,n=1024)
    cases=[]
    for i in range(6):
        base=(i//3)*3
        cases.append(dict(**states[i],relative_invariance_defect=float(np.linalg.norm(value[i]-value[base])/np.linalg.norm(value[base])),absolute_rms_defect=float(np.sqrt(np.mean((value[i]-value[base])**2))),raw_high_band_relative_norm=float(np.linalg.norm(raw[i]-value[i])/np.linalg.norm(value[i]))))
    result["models"][name]=dict(run=str(run),epoch=loaded.epoch,seconds=time.monotonic()-stage,all_finite=bool(np.isfinite(raw).all()),cases=cases)
    (destination/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print("MODEL",name,json.dumps(result["models"][name]),flush=True)
result["status"]="complete"
result["host_model_attempt"]["elapsed_seconds"]=time.monotonic()-start
result["host_model_attempt"]["completed_local"]=datetime.now().astimezone().isoformat()
result["interpretation_limits"]=["Model coordinate-invariance defect is not itself a proof of rollout instability; compare the surviving July model.","Teacher finite-order coordinate dependence falls with order; the completed order10/P128 check is reused unchanged."]
(destination/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
print("FINAL",json.dumps(result["host_model_attempt"]),flush=True)
```

Actual-error feedback:

```python
import os
os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1", XLA_FLAGS="--xla_cpu_multi_thread_eigen=false", MPLCONFIGDIR="/tmp/dno-error-feedback-matplotlib")
os.environ.setdefault("NCCL_P2P_LEVEL","PHB")
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from solver.evals.model_rollout import load_run,build_predict_gxi_batched
from solver.solvers import time_integrator as ti
jax.config.update("jax_enable_x64",True)
start=time.monotonic()
root=Path("outputs/c27_balanced_tanaka_tangent_modal_20260913")
source=root/"diagnostics_energy_20260913"
destination=root/"diagnostics_error_feedback_20260913"
destination.mkdir(exist_ok=True)
metadata=json.loads((source/"summary.json").read_text())
indices=[1,2,4,5]
states=[metadata["states"][i] for i in indices]
with np.load(source/"states.npz") as z:
    pred_eta,pred_xi,depth=z["eta"][indices],z["xi"][indices],z["depth"][indices]
reference=Path("outputs/c27_paper_dataset_20260908_141423/eval_final_n32/tanaka_trajs.npz")
with np.load(reference) as z:
    ids=z["simulation_ids"]
    frame_indices=[int(round(s["time"]/.8)) for s in states]
    columns=[int(np.flatnonzero(ids==s["simulation_id"])[0]) for s in states]
    truth_eta=z["truth_eta"][frame_indices,columns].astype(np.float64)
    truth_xi=z["truth_xi"][frame_indices,columns].astype(np.float64)
eta=np.concatenate((pred_eta,truth_eta))
xi=np.concatenate((pred_xi,truth_xi))
k=2*np.pi*np.fft.fftfreq(1024,d=2*np.pi/1024)
mask=abs(k)<=128
eta=np.fft.ifft(np.fft.fft(eta)*mask).real
xi=np.fft.ifft(np.fft.fft(xi)*mask).real
xi-=xi.mean(axis=-1,keepdims=True)
delta_eta=eta[:4]-eta[4:]
delta_xi=xi[:4]-xi[4:]
g0=k[None,:]*np.tanh(depth[:,None]*k[None,:])
depth_all=np.concatenate((depth,depth))
eta_j,xi_j,k_j=map(jnp.asarray,(eta,xi,k))
eta_x=ti.spectral_dx(eta_j,k_j)
xi_x=ti.spectral_dx(xi_j,k_j)
result=dict(started_local=datetime.now().astimezone().isoformat(),cpu_affinity=sorted(os.sched_getaffinity(0)),jax_devices=[str(d) for d in jax.devices()],source_states=str(source/"states.npz"),reference=str(reference),reference_storage_dtype="float32 promoted to float64; no new reference generation",protocol="For each identical balanced-error state pair, evaluate both models at pred and archived-truth states. P128 eta/xi and centered xi. Full physical RHS=(P128 Gxi, -eta+P128 dealiased_Zakharov_xi_rhs), gravity1. Raw Gxi enters nonlinear products before final projection. No teacher needed for model-increment difference.",rate_definition="E=.5*mean(delta_eta^2+delta_xi*G0(delta_xi)); feedback_dE=mean(delta_eta*delta_Feta+delta_xi*G0(delta_Fxi)); rate=feedback_dE/E. This is instantaneous log-energy growth due to same-model error feedback, not total observed error derivative, which also includes model-reference forcing.",states=states,records=[])
for name,run in (("july",Path("outputs/c27_h1_to_l2_full_20260717_212550")),("balanced",root)):
    stage=time.monotonic()
    loaded=load_run(run,checkpoint="final")
    predict=build_predict_gxi_batched(loaded)
    gxi=predict(eta_j,xi_j,jnp.log(jnp.asarray(depth_all)))
    rhs_eta=np.asarray(ti.apply_lowpass(gxi,k_j,.25))
    rhs_xi=np.asarray(-eta_j+ti.apply_lowpass(ti.dealiased_zakharov_xi_rhs(eta_x,xi_x,gxi),k_j,.25))
    df_eta=rhs_eta[:4]-rhs_eta[4:]
    df_xi=rhs_xi[:4]-rhs_xi[4:]
    bands={}
    for label,lo,hi in (("all",0,128),("1_32",1,32),("33_64",33,64),("65_128",65,128)):
        band=(abs(k)>=lo)&(abs(k)<=hi)
        de,dx,dfe,dfx=[np.fft.ifft(np.fft.fft(v)*band).real for v in (delta_eta,delta_xi,df_eta,df_xi)]
        g0dx=np.fft.ifft(g0*np.fft.fft(dx)).real
        g0dfx=np.fft.ifft(g0*np.fft.fft(dfx)).real
        energy=.5*np.mean(de**2+dx*g0dx,axis=-1)
        feedback=np.mean(de*dfe+dx*g0dfx,axis=-1)
        bands[label]=(energy,feedback)
    for i,state in enumerate(states):
        record=dict(model=name,checkpoint_epoch=loaded.epoch,**state,bands={})
        for label,(energy,feedback) in bands.items():
            record["bands"][label]=dict(error_energy=float(energy[i]),feedback_dE=float(feedback[i]),feedback_log_energy_growth_rate=float(feedback[i]/energy[i]),fraction_of_total_error_energy=float(energy[i]/bands["all"][0][i]),contribution_to_total_growth_rate=float(feedback[i]/bands["all"][0][i]))
        result["records"].append(record)
    result[name+"_seconds"]=time.monotonic()-stage
    assert np.isfinite(rhs_eta).all() and np.isfinite(rhs_xi).all()
    print(name,json.dumps(result["records"][-4:]),flush=True)
result["all_rhs_finite"]=True
result["elapsed_seconds"]=time.monotonic()-start
result["completed_local"]=datetime.now().astimezone().isoformat()
(destination/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
print("DONE",result["elapsed_seconds"],result["completed_local"],flush=True)
```
