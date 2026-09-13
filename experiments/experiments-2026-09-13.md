| Date | Time | File / Variant | Motivation | What Tried / Evidence | Correctness | Timing | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-13 | 01:53 | Balanced C27 Modal launch preparation | Local two-GPU training could not allocate memory alongside another user's job; user requested Modal instead. Reuse the same trainer and batch/update budget on two A100-80GB GPUs. | Confirmed Modal CLI1.4.2 and profile/workspace sciml-at-ud. Balanced arrays are not present at /outputs/paper_dataset_balanced_20260912/arrays on volume dno-fno-train-data. Prepared existing launcher with EPOCHS260, batch1024, seed0, same losses, GPU_SPEC=A100-80GB:2. Match image Python minor version3.11 and pin numerical packages to local versions; correct stale all-family translation label. Requested upload of11 NPY files totaling14.715GB plus scoped trainer/model/solver source. | Local Ruff/Pyright and shell syntax pass; independent launcher/path/hyperparameter audit passes. Remote image build and training NOT tested: approval system rejected upload before process creation, requiring explicit permission for disclosure to this specific Modal workspace/volume. No upload, remote training or local job pause occurred; our failed local trainer was already stopped. | 00:02 (inspection/preparation01:51–01:53; no GPU training). | User explicitly approved this destination and two-A100 training before01:59; see the following execution record. Initial rejection was not bypassed. |
| 2026-09-13 | 01:59 | Approved balanced dataset upload and Modal C27 run | User approved disclosure of the14.715GB balanced dataset and scoped training code to sciml-at-ud/dno-fno-train-data, plus training on two A100-80GB GPUs, after local GPU-memory failure. | Existing upload entrypoint completed11 NPY files; existing detached launcher submitted c27_balanced_tanaka_tangent_modal_20260913 with260 epochs, global batch1024, same architecture/losses and FP32 train/val. Training app ap-sDX1uTgQnbRpNWYkCJNgNX; function call fc-01M2CNTNT9DCNGZ57027DG328E. No smaller batch or new trainer. | COMPLETED PASS: physical devices are two A100-SXM4-80GB; replicated state/optimizer counters verified at0; all260 consecutive epoch records have finite metrics; best/final epoch260 has training loss.0003892254346, validation total.0003607055300 and validation relative-L2.0003308064795. Final/latest/best checkpoints and summary are saved on the volume; process returned0. Independent remote config SHA matches local balanced run after excluding only dataset path/run name. All pinned numerical packages match local versions; uploaded trainer sets NCCL_P2P_LEVEL=PHB before JAX. All11 uploaded filenames/sizes checked; remote row counts and normalization match. | Image/upload01:59–02:03; worker started02:04; first finite updates02:05; first epoch/validation/checkpoints verified02:09. Finished15:42; runtime13:38 (49074.695s trainer,49086s wrapper). Final log audited17:23. | TRAINING COMPLETE. No local jobs stopped or modified. Checkpoints/logs persist on approved volume; rollout/NaN tests have not run. Do not infer rollout stability from finite training metrics. This is the260-epoch matched-update-budget balanced-corpus experiment, not a40-epoch shortened run. |
| 2026-09-13 | 17:31 | Balanced C27 final checkpoint: full previous FP64 rollout panel | Equal family row counts may improve stability beyond Tanaka-only translation regularization; test the same held-out ICs, including both previous NaN failures, without changing the integrator or adding soliton damping. | Downloaded completed epoch260 checkpoint from approved Modal volume. Two local workers cover128 exact previous ICs, known Tanaka failures first. Model-only FP64 rollouts reuse archived FP32 reference fields only for labeled error comparisons; no truth generation. | PARTIAL:70/128 complete before fine-tuning preparation; all32 Stokes and32 JONSWAP plus4 other Tanaka cases finite. Both known Tanaka failures persist:16471 first saved NaNs179.2;16624 at150.4. IDs and initial arrays match originals; internal dt.01, GL2 four iterations, cutoff128, no soliton damping. | Started17:31; both workers paused during18:35–18:37 fine-tuning preparation, preserving their current batches. Remaining58 cases not complete. | Balancing did not remove either targeted failure. Workers213458/213464 are paused for the requested local fine-tune; queued fine-tuned tests will resume them after the priority pair. Old reference-generation workers remain suspended. |
| 2026-09-13 | 18:38 | Old mixed-wave data preparation for balanced-model fine-tuning | The current dataset omits old shallow/mid/wide mixed wavetrains. User requested a short local fine-tune to test whether restoring these examples improves rollout performance without damping. | Copy six original shallow_steep shards into experiment-only NPY arrays; retain120 frames per simulation and original order-six labels. Seed0 split by whole simulation into80/10/10. Reuse existing fine-tuner with a dataset-path override and corrected output provenance, source commit0e74921 (+5 net lines). Preserve the balanced checkpoint and its normalization; do not add legacy input support to production loaders. | PASS:1,440,000 finite rows,12,000 distinct initial-state hashes; train1,152,000/9600 simulations, validation and test144,000/1200 each. Physical fields bit-identical float32; metadata only promoted to float64. IDs11/12/13 mean zero Tanaka selection. Loader,13 tangent tests, exact-fit gradient regression, Ruff and scoped Pyright pass. | 00:03 (18:38–18:41;183.4135s conversion). | Dataset ready. Original balanced rollout workers paused with70 results saved; source dataset/checkpoint unchanged. Real-checkpoint smoke and full fine-tune recorded separately below. |
| 2026-09-13 | 18:42 | Mixed-wave fine-tune: real two-GPU warm-start smoke | The new dataset changes input amplitudes and has no Tanaka rows; verify the actual checkpoint, zero-selection loss path, two-device batches and checkpoint saving before a full run. | Execute existing fine-tuner in memory with2,049 evenly spaced TRAIN rows and1,025 validation rows, global batch1024 and1epoch/LR2e-6. Three real updates include a replicated singleton tail; validation uses FP64. Restore the saved final checkpoint and compare normalization metadata with the balanced source. | PASS: three replicated optimizer/state updates, finite training/validation/parameters; train/val translation loss exactly0; source statistics unchanged; final epoch1 checkpoint reloads. Smoke validation L2.00273133 and composite.00381602 are subset results, not rollout evidence. Initial harness import failed before training; adding its script-directory import path fixed the harness, with no further production changes. | 00:01 (successful smoke18:42–18:43, about43s including import/checkpoint validation). | Proceed with the existing five-epoch trainer. Smoke checkpoint is not used as the full-run initialization. |
| 2026-09-13 | 18:44 | Balanced C27 fine-tuned on old mixed-wave families | Adding the missing mixed wavetrains may improve learned dynamics away from clean solitary-wave profiles. Test this intervention by warm-starting the balanced model while keeping its normalization and architecture fixed. | Launch c27_balanced_mixed_wave_finetune_20260913 from original balanced epoch260, source0e74921. Five epochs,1,125 updates/epoch, batch1024,two RTX6000Ada GPUs, fresh AdamW cosine LR2e-6,weight decay1e-4,no warmup. Train exclusively on1,152,000 old packet rows with simulation-disjoint144,000-row validation. L2+mode6+Hadamard.01/every16/global8 retained; Tanaka-only translation naturally0 because packet IDs11/12/13 are not Tanaka. | RUNNING: startup config/scales audited, PID251906; at19:30 epochs1–4 complete with finite metrics and epoch5 underway. Epoch4 validation L2.00217742, composite.00261087, tangent0; both GPUs remain assigned. FP32 training/FP64 validation. Queued all128 exact previous ICs for FP64 unguarded final-checkpoint rollouts, two failed Tanakas first; no reference generation. Source checkpoint and datasets unchanged. | Submitted18:44, trainer started18:45. Warm speed about2.6updates/s; final training/evaluation timings pending. | Evaluate as an ablation, not a proven fix. Fine-tuned workers resume original balanced panel PIDs213458/213464 after their first priority case; remaining old/new panels share GPUs. Coordinator also resumes these identities on exit. Older reference-generation jobs stay suspended. |
| 2026-09-13 | 19:06 | Balanced NaN diagnosis: historical audit and same-state FP64 CPU probes | Earlier investigations found that accurate predictions on clean waves can coexist with unstable responses to developing distortions. Compare the surviving July C27 and balanced checkpoints on identical pre-failure states while both GPUs fine-tune. Also test mean/depth coordinates, raw output above cutoff and energy consistency instead of assuming missing data alone causes NaNs. | Actual-error feedback separates the models: for16624 at t116/t124, July instantaneous log-error-energy feedback rates are-.0974/-.1594, balanced+.1621/+.7394. At16471 t107.2/t115.2 both amplify, balanced more strongly. Initial mean/depth defects are nearly identical (~.036–.040%); raw Gxi above128 has negligible early effect. Shape-consistency defect worsens markedly for balanced16624 but not universally for16471; total-energy injection is not supported. Detailed methods, caveats, artifacts and executed code below. | PASS: CPU-only FP64; final checkpoints July40/balanced260, identical P128 states, no new rollouts or reference labels. Production-RHS agreement <=2.42e-15; canonical zero-mode control corrected before accepting results. Actual-error references are archived FP32 promoted to FP64. Feedback excludes model-reference forcing: local evidence, not a causal dataset result. | 00:01 combined successful numerical probes (energy7.10s, coordinate teacher2.09s/models1.60s, feedback2.88s); completed18:56–19:02. Earlier sandbox checkpoint restores timed out; host CPU restores succeeded without loader changes. Historical audit completed19:06. | No production change. Next discriminating experiment: short same-state checkpoint handoffs before error growth, comparing July, balanced and fine-tuned models without damping. Test stability around slightly distorted waves before broadening generation or increasing loss weights. Current five-epoch fine-tune and queued128-case panel continue unchanged. |
| 2026-09-13 | 19:13 | Dataset-first follow-up: balanced training composition at failing depths | User correctly redirects the investigation toward the main changed variable: dataset coverage. Equal global family shares need not restore the old variety near the failing depths. Recompute the depth-conditioned counts after balancing rather than repeat obsolete pre-balance99% Tanaka figures. | Scan current TRAIN metadata at depths.20–.35:95,442 rows, comprising74,640 Tanaka (78.20%),12,354 Stokes,8,448 JONSWAP and0 BF. Original v9 at the same depths has738,216 rows:48.76% Tanaka plus246,663 mixed-wave,53,057 single-mode,42,017 Stokes and36,490 Gaussian-sea rows. Globally the old mixed-wave18.86%, single-mode6.55% and Gaussian6.04% populations are not explicitly present in balanced pretraining. Current packet-only fine-tune restores only the first population. | PASS: read-only NumPy metadata scan, no wave arrays, GPUs or training changes. Old counts reused from exact July TRAIN audit; current counts from balanced arrays. These are depth-conditioned exposures, not joint height/spectrum matches or proof of causality. Existing constructor/config audits independently cross-checked. | 00:00 (0.0309s metadata scan,19:13); surrounding source review19:11–19:13. | Prioritize dataset comparisons/add-back experiments. Do not launch the proposed common-state restart campaign. Keep current mixed-wave fine-tune and its queued evaluation unchanged; a null short fine-tune would not conclusively rule out the missing population. No production changes. |
| 2026-09-13 | 19:28 | Actual wave-shape and label coverage near failed Tanakas | Dataset differences are the main intervention to investigate. Equal family shares do not establish joint depth/height coverage; compare real TRAIN eta/xi/targets and distinguish missing wave types from mislabeled or numerically corrupted examples. | Sample4,096 depth-conditioned rows from each of9 populations (36,864 descriptors); then exhaust current TRAIN at h.20–.35. All20,974 rows with actual height/depth.35–.50 are Tanaka:20,969 steep single-crest and5 main two-crest,1,316 simulations. Stokes/JONSWAP maximum height/depth only.1061/.2055. Old packets supply comparable-height states with different low-mode shapes and surface-evolution directions.128 old/new label checks pass retained-order agreement; no labeling regression found. Full findings in notes/dataset_gap_diagnosis_20260913.md. | PASS: exact old TRAIN split and current simulation splits; float32 stored fields promoted for CPU calculations, no GPU/model calls. Exhaustive current count independently reviewed. Label M6-to8<1e-4 and M8-to10<1e-5 for128/128; worst M8-to10=3.56e-6. Selected packet statistics are not exhaustive population estimates; surface-translation residual is relative RMS, not physical energy. Coverage gap established, NaN causality unproven. | 00:12 investigation19:16–19:28. Feature scan92.00s; extra descriptors/artifact copy53.58s; teacher check4.54s; exhaustive current height/evolution scan.574s; extended surface fit.387s. Fine-tuning continues on both GPUs. | Prioritize adding strong mid-depth mixed-wave examples, already tested by the running fine-tune, rather than indiscriminately increasing Tanaka or random-sea counts. No source/dataset/training changes or additional jobs. Preserve report, actual executed diagnostic code and artifacts; evaluate queued128-IC panel before claiming a fix. |

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

### Dataset-first follow-up (19:13)

The user requested focusing on differences in the datasets, not additional
same-state restart diagnostics. No such restart job was launched; the current
fine-tune and its evaluation queue are unchanged.

The main architecture, active objectives (including Tanaka-only translation),
batch size and optimizer recipe now match July C27. Dataset-derived scales,
simulation-disjoint rather than row-random splits, and a nearly matched update
budget still differ. These are caveats, not a reason to defer the data audit.

Current balanced TRAIN rows at depths.20–.35, recomputed from metadata:

| Population | July TRAIN rows | Balanced TRAIN rows |
| --- | --- | --- |
| Tanaka | 359,989 | 74,640 |
| Stokes | 42,017 | 12,354 |
| BF | 0 | 0 |
| Gaussian random seas | 36,490 | 0 |
| JONSWAP/TMA | 0 | 8,448 |
| Single-mode | 53,057 | 0 |
| Mixed wavetrains | 246,663 | 0 |
| Total | 738,216 | 95,442 |
| Tanaka share | 48.76% | 78.20% |

This replaces the *pre-balancing*99.02% Tanaka statistic for the current model.
The smaller row total is partly intentional temporal subsampling; it is not
itself a count of missing independent wave shapes. The global balanced mixture
is25% per family but remains far from equally varied within this depth interval.
This comparison is depth-only, not a joint amplitude/spectrum comparison.

The confirmed omissions to investigate, in order of direct relevance here:

1. Old mixed wavetrains:1–5 components with randomized amplitudes, phases and
   directions.18.86% of July TRAIN; absent from balanced pretraining. Mid/wide
   variants overlap failing depths and wave heights. The current fine-tune tests
   their return:1,152,000 rows/9,600 TRAIN simulations,120 frames each.
2. Dedicated single-mode and Gaussian-sea populations:6.55% and6.04% of July
   TRAIN respectively. JONSWAP retains genuine random/bidirectional variation,
   but is not the same sampling distribution as the old Gaussian population.
3. Narrowed BF initial conditions: old depths.5–4 and independent sideband
   amplitudes/phases versus current depth5, equal sideband amplitudes and fixed
   relative phase. Neither BF population directly covers these failing depths,
   so this is a broader diversity difference, not the first depth-specific lead.

Current fine-tuning is100% old packets for five epochs, preserving the balanced
normalization and using a new simulation-level split. It does not recreate
July's mixture or train from scratch. Improvement would support the usefulness
of that missing data; no improvement would not by itself disprove it.
Further comparisons should examine wave shapes, spectra and eta/xi relationships
at matched depth and wave height, and teacher/preprocessing differences. Do not
equate spectral tails caused by interpolation error with useful physical diversity.

Existing evidence:
`outputs/c27_paper_dataset_20260908_141423/eval_final_n32/old_new_training_composition_audit_20260910.json`,
`dataset_richness_constructor_audit_20260910.json` beside it,
`outputs/paper_dataset_balanced_20260912/balance_audit.json`,
and `outputs/c27_mixed_wave_finetune_20260913_dataset/preparation.json`.

Executed CPU metadata scan (stdout result is the table above;0.0309s):

```python
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
import numpy as np
start = perf_counter()
root = Path("outputs/paper_dataset_balanced_20260912/arrays")
arrays = {name: np.load(root / f"{name}.npy", mmap_mode="r")
          for name in ("dataset_split", "family_id", "depth", "simulation_id")}
train = arrays["dataset_split"] == "train"
selected = train & (arrays["depth"] >= .2) & (arrays["depth"] <= .35)
names = {1: "stokes", 2: "tanaka", 3: "benjamin_feir", 4: "jonswap_tma"}
by_family = {name: {"train_rows": int(np.count_nonzero(selected & (arrays["family_id"] == family))),
                    "train_simulations": int(np.unique(arrays["simulation_id"][selected & (arrays["family_id"] == family)]).size)}
             for family, name in names.items()}
old = json.loads(Path("outputs/c27_paper_dataset_20260908_141423/eval_final_n32/old_new_training_composition_audit_20260910.json").read_text())["target_depth_train_rows"]["old"]
result = {"completed_local": datetime.now().astimezone().isoformat(), "dataset": str(root),
          "depth_interval_inclusive": [.2, .35], "balanced_by_family": by_family,
          "balanced_total": int(selected.sum()), "old_train_by_family": old,
          "balanced_tanaka_fraction": by_family["tanaka"]["train_rows"] / int(selected.sum()),
          "old_tanaka_fraction": old["tanaka"] / sum(old.values()),
          "elapsed_seconds": perf_counter() - start,
          "limitation": "Depth only: no joint matching of wave height, slope, spectrum, or eta-xi relationship. Old counts reused from exact TRAIN-split audit; balanced counts recomputed from current arrays. No full wave fields read."}
print(json.dumps(result, indent=2))
```

### Dataset wave-shape and label audit (19:16–19:28)

[Readable findings and next direction](../notes/dataset_gap_diagnosis_20260913.md).

Exact exhaustive count: at TRAIN depth0.20–0.35 and actual peak-to-trough
height/depth0.35–0.50,20,974 rows represent1,316 simulations, all Tanaka.
20,969 rows are steep single-crest and five are main two-crest. Median fraction
of instantaneous surface evolution not explained by a translation:.00012194.
This is a relative RMS residual, not an energy fraction or full-state rigid-motion
claim. All current Stokes and JONSWAP rows at these depths lie below height/depth
.1061 and.2055 respectively. The four-family corpus therefore has virtually no
non-solitary strong-wave examples here despite balanced global row shares.

The4,096-row-per-population descriptor audit includes old Tanaka/Stokes/
Gaussian/single-mode/packets, current Tanaka/Stokes/JONSWAP and the packet
fine-tune. Old packets have117 sampled height matches; fine-tune packets132;
current Tanaka1,146. Within the height interval, old/current Tanaka slope and
low-mode spectra overlap the failures. Missing packet states add different
shapes, not merely greater steepness or arbitrary near-cutoff noise.
Their median eta power in modes5–16 is7.22%, versus1.97% for current Tanaka.

An expanded4,549-wave label-only surface-evolution comparison confirms the
distinction. Height-matched old packets117/current Tanaka294/fine-tune packets132
have median unexplained surface-change fractions.10450/.00012075/.38264.
The exact current full-slice result above supersedes its sample median. Packet
samples are selected from the4,096-row audit and can contain correlated snapshots;
do not interpret the old-vs-fine-tune median difference as a controlled effect.
An initial573-wave probe (.667s,19:22) agreed qualitatively but is superseded
by the expanded comparison (.387s,19:26).

Label audit:16 selected TRAIN rows from each of8 old/current populations, with
up to8 height-matched per population, evaluated in FP64 at M1/M6/M8/M10,pad8,P128.
All128 pass retained-band order agreement. Median stored-versus-recomputedM6
Tanaka errors old/new1.208e-6/1.243e-6; learned-correction target beyondG0+G1
is1.384%/1.511% respectively. No evidence that new labels lost the nonlinear
correction. Historical static Stokes had unprojected targets, but sampled
out-of-band label norm is at most3.89e-6 relative. This is not independent
exact-solver certification or an exhaustive label audit.

Source review also found genuine differences that are less direct for these
failures: old Gaussian retains early unadjusted evolution, while current
JONSWAP retains only production after20 peak periods of adjustment. Both have
independent left/right phases and finite-depth significant heights.005–.03;
new JONSWAP is not uniformly spectrally narrower. Old single-mode data was not
pure G0: xi includes its free-surface factor and labels use fullM6. Its amplitude
bound nevertheless lies far below these failing heights. Steep Tanaka parameter
ranges are unchanged; main Tanaka permits weaker secondary crests now, while
direction quotas alter mixture weights. BF phases/amplitudes are more prescribed,
but old and new BF both miss these depths.

Artifacts under `outputs/c27_balanced_tanaka_tangent_modal_20260913/`:

- `diagnostics_dataset_coverage_20260913/summary.json`: provenance and actual
  feature/artifact-generation code.
- `coverage_descriptors.npz` in that directory:36,864 scalar-feature rows.
- `coverage_wave_samples.npz`:4,549 physical rows with source indices and
  an unenriched-depth-sample flag; `label_probe_samples.npz`:573 smaller probes.
- `exhaustive_balanced_height_slice.json`, `translation_fit_extended.json/npz`,
  and initial `translation_fit.json/npz` in the same directory.
- `diagnostics_dataset_labels_20260913/{summary.json,labels.npz,probe.py,console.log}`.

No production Python changes. Root scratch checks pass Ruff and Pyright.
The original simulation-level training splits/checkpoints are untouched.
Independent review confirms the projection and surface-residual calculation,
while cautioning against interpreting it as whole-state rigidity.

#### Executed feature scan

CPU NumPy, one BLAS thread, last four allowed CPUs. This maps uncompressed NPZ
members without loading the88GB archive into memory.

```python
import os
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
import json
import struct
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import NDArray

def archive_array(path: Path, name: str) -> np.memmap:
    with zipfile.ZipFile(path) as archive:
        info=archive.getinfo(name+".npy")
        assert info.compress_type==zipfile.ZIP_STORED
        with archive.open(info) as member:
            version=np.lib.format.read_magic(member)
            shape,fortran,dtype=(np.lib.format.read_array_header_1_0(member) if version==(1,0) else np.lib.format.read_array_header_2_0(member))
            header_bytes=member.tell()
    with path.open("rb") as stream:
        stream.seek(info.header_offset+26)
        name_bytes,extra_bytes=struct.unpack("<HH",stream.read(4))
    return np.memmap(path,mode="r",dtype=dtype,shape=shape,order="F" if fortran else "C",offset=info.header_offset+30+name_bytes+extra_bytes+header_bytes)

def features(eta: NDArray[np.floating[Any]], xi: NDArray[np.floating[Any]], depth: NDArray[np.floating[Any]]) -> dict[str, NDArray[np.float64]]:
    eta=np.asarray(eta,dtype=np.float64)
    xi=np.asarray(xi,dtype=np.float64)
    k=np.arange(513,dtype=np.float64)
    eh=np.fft.rfft(eta)
    xh=np.fft.rfft(xi)
    eh[:,129:]=0;xh[:,129:]=0
    slope=np.fft.irfft(1j*k*eh,n=1024)
    out={"waveheight_over_depth":np.ptp(eta,axis=-1)/depth,"max_abs_slope":np.max(abs(slope),axis=-1),"mean_eta_over_depth":eta.mean(axis=-1)/depth}
    for field,spectrum in (("eta",eh),("xi",xh)):
        power=abs(spectrum[:,1:129])**2
        total=power.sum(axis=-1)
        out[field+"_effective_mode_count"]=total**2/(power**2).sum(axis=-1)
        for lo,hi in ((1,4),(5,16),(17,32),(33,64),(65,128)):
            out[f"{field}_power_fraction_{lo}_{hi}"]=power[:,lo-1:hi].sum(axis=-1)/total
    w=np.sqrt(k[None,1:129]*np.tanh(depth[:,None]*k[None,1:129]))
    traveling_xi=-1j*eh[:,1:129]/w
    actual_xi=xh[:,1:129]
    dot=(traveling_xi.conj()*actual_xi).real.sum(axis=-1)
    norm_product=(abs(traveling_xi)**2).sum(axis=-1)*(abs(actual_xi)**2).sum(axis=-1)
    correlation=dot/np.sqrt(norm_product)
    out["best_linear_traveling_wave_relative_residual"]=np.sqrt(np.maximum(0,1-correlation**2))
    out["linear_traveling_wave_signed_correlation"]=correlation
    return out

def summarize(values: dict[str, NDArray[np.float64]], selected: NDArray[np.bool_]) -> dict[str, Any]:
    return {name:{"p05":float(q[0]),"p25":float(q[1]),"p50":float(q[2]),"p75":float(q[3]),"p95":float(q[4]),"max":float(np.max(value[selected]))} for name,value in values.items() for q in [np.quantile(value[selected],[.05,.25,.5,.75,.95])]} if selected.any() else {}

start=time.monotonic()
root=Path("outputs/c27_balanced_tanaka_tangent_modal_20260913")
destination=root/"diagnostics_dataset_coverage_20260913"
destination.mkdir(exist_ok=True)
old_path=Path("data/combined_dataset_v9.npz")
old={name:archive_array(old_path,name) for name in ("eta","xi","gxi","depth","source")}
n=len(old["depth"]);train_count=n-2*int(.1*n)
old_train=np.zeros(n,dtype=bool);old_train[np.random.default_rng(0).permutation(n)[:train_count]]=True
assert train_count==6107193
arrays=[]
for path in (Path("outputs/paper_dataset_balanced_20260912/arrays"),Path("outputs/c27_mixed_wave_finetune_20260913_dataset")):
    arrays.append({name:np.load(path/(name+".npy"),mmap_mode="r") for name in ("eta","xi","gxi","depth","family_id","dataset_split","simulation_id","parameter_group_id")})
balanced,finetune=arrays
populations=[
    ("old_tanaka",old,old_train,np.array([5,6,14]),"source",str(old_path)),
    ("old_stokes",old,old_train,np.array([3,4]),"source",str(old_path)),
    ("old_gaussian",old,old_train,np.array([0,1]),"source",str(old_path)),
    ("old_single_mode",old,old_train,np.array([2]),"source",str(old_path)),
    ("old_packets",old,old_train,np.array([11,12,13]),"source",str(old_path)),
    ("balanced_tanaka",balanced,balanced["dataset_split"]=="train",np.array([2]),"family_id","outputs/paper_dataset_balanced_20260912/arrays"),
    ("balanced_stokes",balanced,balanced["dataset_split"]=="train",np.array([1]),"family_id","outputs/paper_dataset_balanced_20260912/arrays"),
    ("balanced_jonswap",balanced,balanced["dataset_split"]=="train",np.array([4]),"family_id","outputs/paper_dataset_balanced_20260912/arrays"),
    ("finetune_packets",finetune,finetune["dataset_split"]=="train",np.array([11,12,13]),"family_id","outputs/c27_mixed_wave_finetune_20260913_dataset"),
]
result=dict(started_local=datetime.now().astimezone().isoformat(),depth_interval=[.2,.35],waveheight_matched_interval=[.35,.50],max_sampled_rows_per_population=4096,sampling="Evenly spaced indices of sorted eligible TRAIN rows; all eligible rows if <=4096. Summary matching is performed within this sample, NOT a full-population waveheight count.",fourier_convention="Positive integer modes1..128; powers exclude eta/xi mean; fractions normalized by total1..128 power. Max slope uses P128 eta. Effective count=(sum power)^2/sum power^2. Traveling diagnostic fits a real scalar to xi_hat=-i*eta_hat/sqrt(k*tanh(kh)); residual0 means that linear one-direction template fits, not proof of a nonlinear traveling wave.",limitations=["Old v9 uses row-level TRAIN split; balanced/fine-tune use simulation-level splits.","Sampled rows can be correlated snapshots of the same simulation; row counts are not independent sample counts.","Float32 stored physical fields are promoted for FFT diagnostics; no new physical precision.","Spectral content and height/depth do not jointly prove dynamical equivalence or a causal explanation of NaNs."],populations={},failure_initials=[])
probe_parts={name:[] for name in ("eta","xi","gxi","depth","population","source_row_index","source_id","parameter_group","waveheight_over_depth","height_matched")}
for label,data,train_mask,ids,source_key,path in populations:
    stage=time.monotonic()
    eligible=np.flatnonzero(train_mask&np.isin(data[source_key],ids)&(data["depth"]>=.2)&(data["depth"]<=.35))
    rows=eligible[np.linspace(0,len(eligible)-1,min(4096,len(eligible)),dtype=int)]
    eta=np.asarray(data["eta"][rows]);xi=np.asarray(data["xi"][rows]);depth=np.asarray(data["depth"][rows],dtype=np.float64)
    values=features(eta,xi,depth)
    matched=(values["waveheight_over_depth"]>=.35)&(values["waveheight_over_depth"]<=.50)
    entry=dict(dataset=path,source_ids=ids.tolist(),eligible_train_rows=len(eligible),sampled_rows=len(rows),sampled_height_matched_rows=int(matched.sum()),sampled_height_matched_fraction=float(matched.mean()),depth_only=summarize(values,np.ones(len(rows),dtype=bool)),height_matched=summarize(values,matched))
    if "simulation_id" in data:
        entry["eligible_train_simulations"]=len(np.unique(data["simulation_id"][eligible]))
        entry["sampled_train_simulations"]=len(np.unique(data["simulation_id"][rows]))
        entry["sampled_height_matched_simulations"]=len(np.unique(data["simulation_id"][rows[matched]]))
    match_positions=np.flatnonzero(matched)
    chosen_match=match_positions[np.linspace(0,len(match_positions)-1,min(32,len(match_positions)),dtype=int)] if len(match_positions) else np.array([],dtype=int)
    chosen=np.unique(np.concatenate((np.linspace(0,len(rows)-1,64-len(chosen_match),dtype=int),chosen_match)))
    for name in ("eta","xi"):
        probe_parts[name].append(np.asarray(data[name][rows[chosen]]))
    probe_parts["gxi"].append(np.asarray(data["gxi"][rows[chosen]]))
    probe_parts["depth"].append(depth[chosen])
    probe_parts["population"].append(np.full(len(chosen),label))
    probe_parts["source_row_index"].append(rows[chosen])
    probe_parts["source_id"].append(np.asarray(data[source_key][rows[chosen]],dtype=np.int32))
    groups=np.asarray(data["parameter_group_id"][rows[chosen]],dtype=str) if "parameter_group_id" in data else np.array([f"source_{int(s)}" for s in data[source_key][rows[chosen]]])
    probe_parts["parameter_group"].append(groups)
    probe_parts["waveheight_over_depth"].append(values["waveheight_over_depth"][chosen])
    probe_parts["height_matched"].append(matched[chosen])
    entry["label_probe_rows"]=len(chosen)
    entry["elapsed_seconds"]=time.monotonic()-stage
    result["populations"][label]=entry
    print(label,entry["eligible_train_rows"],entry["sampled_height_matched_rows"],"of",len(rows),"H/h median",entry["depth_only"]["waveheight_over_depth"]["p50"],"matched slope median",entry["height_matched"].get("max_abs_slope",{}).get("p50"),flush=True)
for sim in (16471,16624):
    with np.load(root/"eval_final_n32_fp64"/f"simulation_{sim}.npz") as z:
        values=features(z["initial_eta"][None,:],z["initial_xi"][None,:],z["depths"])
        result["failure_initials"].append(dict(simulation_id=sim,depth=float(z["depths"][0]),features={name:float(v[0]) for name,v in values.items()}))
probe={name:np.concatenate(parts) for name,parts in probe_parts.items()}
np.savez_compressed(destination/"label_probe_samples.npz",**probe)
result["label_probe"]=dict(path=str(destination/"label_probe_samples.npz"),rows=len(probe["depth"]),schema={name:{"shape":list(value.shape),"dtype":str(value.dtype)} for name,value in probe.items()},selection="Up to32 evenly spaced H/h-matched sampled rows plus32-or-more evenly spaced depth-only sampled rows; duplicates removed. Raw stored eta/xi/gxi copied unchanged.")
result["elapsed_seconds"]=time.monotonic()-start
result["completed_local"]=datetime.now().astimezone().isoformat()
(destination/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
print("DONE",result["elapsed_seconds"],result["completed_local"],flush=True)
```

#### Additional descriptor and representative-wave artifacts

```python
import os
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
import json
import struct
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import NDArray

def archive_array(path: Path, name: str) -> np.memmap:
    with zipfile.ZipFile(path) as archive:
        info=archive.getinfo(name+".npy")
        assert info.compress_type==zipfile.ZIP_STORED
        with archive.open(info) as member:
            version=np.lib.format.read_magic(member)
            shape,fortran,dtype=(np.lib.format.read_array_header_1_0(member) if version==(1,0) else np.lib.format.read_array_header_2_0(member))
            header_bytes=member.tell()
    with path.open("rb") as stream:
        stream.seek(info.header_offset+26)
        name_bytes,extra_bytes=struct.unpack("<HH",stream.read(4))
    return np.memmap(path,mode="r",dtype=dtype,shape=shape,order="F" if fortran else "C",offset=info.header_offset+30+name_bytes+extra_bytes+header_bytes)

def features(eta: NDArray[np.floating[Any]], xi: NDArray[np.floating[Any]], depth: NDArray[np.floating[Any]]) -> dict[str, NDArray[np.float64]]:
    eta=np.asarray(eta,dtype=np.float64)
    xi=np.asarray(xi,dtype=np.float64)
    k=np.arange(513,dtype=np.float64)
    eh=np.fft.rfft(eta)
    xh=np.fft.rfft(xi)
    eh[:,129:]=0;xh[:,129:]=0
    slope=np.fft.irfft(1j*k*eh,n=1024)
    out={"waveheight_over_depth":np.ptp(eta,axis=-1)/depth,"max_abs_slope":np.max(abs(slope),axis=-1),"mean_eta_over_depth":eta.mean(axis=-1)/depth}
    for field,spectrum in (("eta",eh),("xi",xh)):
        power=abs(spectrum[:,1:129])**2
        total=power.sum(axis=-1)
        out[field+"_effective_mode_count"]=total**2/(power**2).sum(axis=-1)
        for lo,hi in ((1,4),(5,16),(17,32),(33,64),(65,128)):
            out[f"{field}_power_fraction_{lo}_{hi}"]=power[:,lo-1:hi].sum(axis=-1)/total
    w=np.sqrt(k[None,1:129]*np.tanh(depth[:,None]*k[None,1:129]))
    traveling_xi=-1j*eh[:,1:129]/w
    actual_xi=xh[:,1:129]
    dot=(traveling_xi.conj()*actual_xi).real.sum(axis=-1)
    norm_product=(abs(traveling_xi)**2).sum(axis=-1)*(abs(actual_xi)**2).sum(axis=-1)
    correlation=dot/np.sqrt(norm_product)
    out["best_linear_traveling_wave_relative_residual"]=np.sqrt(np.maximum(0,1-correlation**2))
    out["linear_traveling_wave_signed_correlation"]=correlation
    return out

def summarize(values: dict[str, NDArray[np.float64]], selected: NDArray[np.bool_]) -> dict[str, Any]:
    return {name:{"p05":float(q[0]),"p25":float(q[1]),"p50":float(q[2]),"p75":float(q[3]),"p95":float(q[4]),"max":float(np.max(value[selected]))} for name,value in values.items() for q in [np.quantile(value[selected],[.05,.25,.5,.75,.95])]} if selected.any() else {}

start=time.monotonic()
root=Path("outputs/c27_balanced_tanaka_tangent_modal_20260913")
destination=root/"diagnostics_dataset_coverage_20260913"
destination.mkdir(exist_ok=True)
old_path=Path("data/combined_dataset_v9.npz")
old={name:archive_array(old_path,name) for name in ("eta","xi","gxi","depth","source")}
n=len(old["depth"]);train_count=n-2*int(.1*n)
old_train=np.zeros(n,dtype=bool);old_train[np.random.default_rng(0).permutation(n)[:train_count]]=True
assert train_count==6107193
arrays=[]
for path in (Path("outputs/paper_dataset_balanced_20260912/arrays"),Path("outputs/c27_mixed_wave_finetune_20260913_dataset")):
    arrays.append({name:np.load(path/(name+".npy"),mmap_mode="r") for name in ("eta","xi","gxi","depth","family_id","dataset_split","simulation_id","parameter_group_id")})
balanced,finetune=arrays
populations=[
    ("old_tanaka",old,old_train,np.array([5,6,14]),"source",str(old_path)),
    ("old_stokes",old,old_train,np.array([3,4]),"source",str(old_path)),
    ("old_gaussian",old,old_train,np.array([0,1]),"source",str(old_path)),
    ("old_single_mode",old,old_train,np.array([2]),"source",str(old_path)),
    ("old_packets",old,old_train,np.array([11,12,13]),"source",str(old_path)),
    ("balanced_tanaka",balanced,balanced["dataset_split"]=="train",np.array([2]),"family_id","outputs/paper_dataset_balanced_20260912/arrays"),
    ("balanced_stokes",balanced,balanced["dataset_split"]=="train",np.array([1]),"family_id","outputs/paper_dataset_balanced_20260912/arrays"),
    ("balanced_jonswap",balanced,balanced["dataset_split"]=="train",np.array([4]),"family_id","outputs/paper_dataset_balanced_20260912/arrays"),
    ("finetune_packets",finetune,finetune["dataset_split"]=="train",np.array([11,12,13]),"family_id","outputs/c27_mixed_wave_finetune_20260913_dataset"),
]

descriptor_parts={}
wave_parts={name:[] for name in ("eta","xi","depth","population","source_row_index","source_id","parameter_group","waveheight_over_depth","height_matched","depth_sample")}
gxi_sources=[]
for label,data,train_mask,ids,source_key,path in populations:
    eligible=np.flatnonzero(train_mask&np.isin(data[source_key],ids)&(data["depth"]>=.2)&(data["depth"]<=.35))
    rows=eligible[np.linspace(0,len(eligible)-1,min(4096,len(eligible)),dtype=int)]
    eta=np.asarray(data["eta"][rows]);xi=np.asarray(data["xi"][rows]);depth=np.asarray(data["depth"][rows],dtype=np.float64)
    values=features(eta,xi,depth)
    matched=(values["waveheight_over_depth"]>=.35)&(values["waveheight_over_depth"]<=.50)
    metadata={"depth":depth,"population":np.full(len(rows),label),"source_row_index":rows,"source_id":np.asarray(data[source_key][rows],dtype=np.int32),"parameter_group":np.asarray(data["parameter_group_id"][rows],dtype=str) if "parameter_group_id" in data else np.array([f"source_{int(s)}" for s in data[source_key][rows]])}
    for name,value in {**metadata,**values}.items():
        descriptor_parts.setdefault(name,[]).append(value)
    matched_positions=np.flatnonzero(matched)
    matched_chosen=matched_positions[np.linspace(0,len(matched_positions)-1,min(256,len(matched_positions)),dtype=int)] if len(matched_positions) else np.array([],dtype=int)
    base_chosen=np.linspace(0,len(rows)-1,512-len(matched_chosen),dtype=int)
    chosen=np.unique(np.concatenate((base_chosen,matched_chosen)))
    for name,value in {**metadata,"eta":eta,"xi":xi,"waveheight_over_depth":values["waveheight_over_depth"],"height_matched":matched}.items():
        wave_parts[name].append(value[chosen])
    wave_parts["depth_sample"].append(np.isin(chosen,base_chosen))
    gxi_sources.append((label,data,rows[chosen]))
    print("DESCRIPTORS",label,len(chosen),time.monotonic()-start,flush=True)
descriptors={name:np.concatenate(parts) for name,parts in descriptor_parts.items()}
np.savez_compressed(destination/"coverage_descriptors.npz",**descriptors)
print("DESCRIPTORS_READY",time.monotonic()-start,flush=True)
gxi_parts=[]
for label,data,rows in gxi_sources:
    gxi_parts.append(np.asarray(data["gxi"][rows]))
    print("LABELS",label,time.monotonic()-start,flush=True)
waves={name:np.concatenate(parts) for name,parts in wave_parts.items()}
waves["gxi"]=np.concatenate(gxi_parts)
np.savez_compressed(destination/"coverage_wave_samples.npz",**waves)
summary=json.loads((destination/"summary.json").read_text())
summary["coverage_descriptors"]={"path":str(destination/"coverage_descriptors.npz"),"rows":len(descriptors["depth"]),"sampling":"All original4096 deterministic depth-only rows/population; metadata and scalarfeatures, no spatial arrays."}
summary["coverage_wave_samples"]={"path":str(destination/"coverage_wave_samples.npz"),"rows":len(waves["depth"]),"selection":"Up to512/population: up to256 height-matched plus evenly spaced depth-only rows, duplicatesremoved. depth_sample=True identifies base depth-only sample; height_matched identifies H/h=.35–.50. Raw physicalfields unchanged.","rows_per_population":{label:int(np.sum(waves["population"]==label)) for label,*_ in populations}}
summary["additional_artifacts_elapsed_seconds"]=time.monotonic()-start
summary["additional_artifacts_completed_local"]=datetime.now().astimezone().isoformat()
(destination/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print("DONE",summary["additional_artifacts_elapsed_seconds"],summary["additional_artifacts_completed_local"],flush=True)
```

#### Label recomputation

Successful CPU invocation used `timeout 110s taskset -c 4-7`,
`JAX_PLATFORMS=cpu`, empty `CUDA_VISIBLE_DEVICES`, two CPU/BLAS threads,
`PYTHONPATH=/home/johnma/dno-fno`, and the following
`diagnostics_dataset_labels_20260913/probe.py`:

```python
"""Bounded CPU audit of saved TRAIN labels, without model evaluation or rollouts."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")

import jax
import jax.numpy as jnp
import numpy as np

from solver.solvers.dno_series_jax import dno_series_eval

jax.config.update("jax_enable_x64", True)
started = perf_counter()
out = Path(__file__).resolve().parent
source = out.parent / "diagnostics_dataset_coverage_20260913/label_probe_samples.npz"
with np.load(source, allow_pickle=False) as data:
    population = data["population"].astype(str)
    selected = []
    for name in np.unique(population):
        if name == "finetune_packets":
            continue
        available = np.flatnonzero(population == name)
        chosen = []
        for height_matched in (False, True):
            group = available[data["height_matched"][available] == height_matched]
            if len(group):
                chosen.extend(group[np.linspace(0, len(group) - 1, min(8, len(group)), dtype=int)].tolist())
        remaining = np.setdiff1d(available, chosen)
        if len(chosen) < 16 and len(remaining):
            chosen.extend(remaining[np.linspace(0, len(remaining) - 1, min(16 - len(chosen), len(remaining)), dtype=int)].tolist())
        selected.extend(chosen)
    indices = np.asarray(selected)
    arrays = {name: data[name][indices] for name in data.files if data[name].ndim and len(data[name]) == len(population)}

population = arrays["population"].astype(str)
depth = arrays["depth"].astype(np.float64).reshape(-1)
eta = arrays["eta"].astype(np.float64)
xi = arrays["xi"].astype(np.float64)
target = arrays["gxi"].astype(np.float64)
assert eta.shape == xi.shape == target.shape == (len(indices), 1024)
assert np.all((depth >= .2) & (depth <= .35))
k = np.fft.fftfreq(1024, 1 / 1024)
mask = np.abs(k) <= 128


def project(field: np.ndarray) -> np.ndarray:
    return np.fft.ifft(np.fft.fft(field, axis=-1) * mask, axis=-1).real


def norm(field: np.ndarray) -> np.ndarray:
    return np.linalg.norm(field, axis=-1)


projected_eta = project(eta)
projected_xi = project(xi)
projected_xi -= projected_xi.mean(axis=-1, keepdims=True)
projected_target = project(target)
k_jax = jnp.asarray(k)


@jax.jit
def teachers(eta: jax.Array, xi: jax.Array, depth: jax.Array) -> tuple[jax.Array, ...]:
    return tuple(dno_series_eval(eta, xi, k_jax, depth[:, None], order, pad_factor=8) for order in (1, 6, 8, 10))


report = {
    "started_local": datetime.now().astimezone().isoformat(),
    "source": str(source), "jax_devices": [str(device) for device in jax.devices()],
    "cpu_affinity": sorted(os.sched_getaffinity(0)),
    "selection": "At most16 rows/population, up to8 height-matched and8 non-height-matched; fill unavailable stratum slots from the other stratum. Evenly spaced within supplied samples; omit finetune_packets duplicate source. Actual TRAIN, h=.20-.35; enriched sample, not prevalence estimate.",
    "protocol": "FP64 evaluation of saved float32 fields promoted to64. P128 both inputs; center xi; evaluate existing DNO series M1/M6/M8/M10 at pad8; P128 outputs and remove output means. No model/checkpoint, no new rollout.",
    "convergence_screen": "finite outputs and ||M8-M6||/||M8||<1e-4 and ||M10-M8||/||M10||<1e-5. This is retained-band order agreement, not an independent exact-solver certification.",
    "caveat": "Stored labels were generally computed before float32 field/depth rounding. P128 input projection can also change historical unprojected target definitions. Unconverged rows are not evidence of wrong labels.",
    "records": [], "populations": {},
}
outputs = {order: [] for order in (1, 6, 8, 10)}
for start in range(0, len(indices), 16):
    stop = start + 16
    evaluated = teachers(jnp.asarray(projected_eta[start:stop]), jnp.asarray(projected_xi[start:stop]), jnp.asarray(depth[start:stop]))
    for order, values in zip(outputs, jax.device_get(evaluated)):
        values = project(np.asarray(values))
        values -= values.mean(axis=-1, keepdims=True)
        outputs[order].append(values)
outputs = {order: np.concatenate(values) for order, values in outputs.items()}
finite = np.logical_and.reduce([np.isfinite(values).all(axis=-1) for values in outputs.values()])
denominator = np.maximum(norm(projected_target), 1e-30)
metrics = {
    "order8_vs6_relative": norm(outputs[8] - outputs[6]) / np.maximum(norm(outputs[8]), 1e-30),
    "order10_vs8_relative": norm(outputs[10] - outputs[8]) / np.maximum(norm(outputs[10]), 1e-30),
    "label_vs_order6_relative": norm(projected_target - outputs[6]) / denominator,
    "label_vs_order10_relative": norm(projected_target - outputs[10]) / denominator,
    "label_above128_relative": norm(target - projected_target) / np.maximum(norm(target), 1e-30),
    "eta_above128_relative": norm(eta - projected_eta) / np.maximum(norm(eta), 1e-30),
    "xi_above128_relative": norm(xi - project(xi)) / np.maximum(norm(xi), 1e-30),
    "stored_correction_beyond_G0G1_relative": norm(projected_target - outputs[1]) / denominator,
    "order10_correction_beyond_G0G1_relative": norm(outputs[10] - outputs[1]) / np.maximum(norm(outputs[10]), 1e-30),
    "waveheight_over_depth": np.ptp(projected_eta, axis=-1) / depth,
    "maximum_slope": np.max(np.abs(np.fft.ifft(1j * k * np.fft.fft(projected_eta)).real), axis=-1),
    "eta_mean_over_depth": projected_eta.mean(axis=-1) / depth,
}
converged = finite & (metrics["order8_vs6_relative"] < 1e-4) & (metrics["order10_vs8_relative"] < 1e-5)
for i in range(len(indices)):
    record = {"sample_index": int(indices[i]), "population": str(population[i]), "depth": float(depth[i]), "finite_teacher": bool(finite[i]), "converged_teacher": bool(converged[i])}
    for key in ("source_row_index", "source_id", "parameter_group", "height_matched"):
        if key in arrays:
            record[key] = arrays[key][i].item()
    record.update({key: float(value[i]) if np.isfinite(value[i]) else None for key, value in metrics.items()})
    report["records"].append(record)
for name in np.unique(population):
    selected = population == name
    summary = {"rows": int(selected.sum()), "finite_teacher_rows": int((selected & finite).sum()), "converged_teacher_rows": int((selected & converged).sum())}
    for subset_name, subset in (("all_finite", selected & finite), ("converged", selected & converged)):
        summary[subset_name] = {
            key: dict(zip(("median", "p90", "max"), map(float, np.quantile(value[subset], (.5, .9, 1)))))
            for key, value in metrics.items()
        } if subset.any() else {}
    report["populations"][str(name)] = summary
np.savez_compressed(out / "labels.npz", sample_index=indices, population=population, depth=depth, eta=projected_eta, xi=projected_xi, stored_gxi=target, stored_gxi_P128=projected_target, **{f"teacher_M{order}": value for order, value in outputs.items()}, converged=converged)
report.update(completed_local=datetime.now().astimezone().isoformat(), elapsed_seconds=perf_counter() - started, actual_executed_code=Path(__file__).read_text())
(out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps({"elapsed_seconds": report["elapsed_seconds"], "populations": report["populations"]}), flush=True)
```

#### Initial small surface-translation fit

Run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/uv-cache
uv run --offline --no-sync /tmp/dno_dataset_translation_20260913.py`:

```python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

root = Path('/home/johnma/dno-fno')
directory = root / 'outputs/c27_balanced_tanaka_tangent_modal_20260913/diagnostics_dataset_coverage_20260913'
started = perf_counter()
with np.load(directory / 'label_probe_samples.npz') as saved:
    eta = saved['eta'].astype(np.float64)
    gxi = saved['gxi'].astype(np.float64)
    population = saved['population']
    row_indices = saved['source_row_index']
    heights = saved['waveheight_over_depth']
sample_count = len(eta)
with np.load(root / 'outputs/c27_paper_dataset_20260908_141423/eval_final_n32/tanaka_trajs.npz') as truth:
    indices = [int(np.flatnonzero(truth['simulation_ids'] == sid)[0]) for sid in (16471, 16624)]
    eta = np.concatenate((eta, truth['truth_eta'][0, indices].astype(np.float64)))
    gxi = np.concatenate((gxi, truth['truth_gxi'][0, indices].astype(np.float64)))
k = np.fft.fftfreq(eta.shape[-1], 1 / eta.shape[-1])
retained = (abs(k) <= 128) & (k != 0)
eta_x = np.fft.ifft(np.fft.fft(eta) * (1j * k * retained)).real
gxi = np.fft.ifft(np.fft.fft(gxi) * retained).real
coefficient = np.sum(eta_x * gxi, axis=-1) / np.sum(eta_x**2, axis=-1)
residual = gxi - coefficient[:, None] * eta_x
fraction = np.linalg.norm(residual, axis=-1) / np.linalg.norm(gxi, axis=-1)
assert np.isfinite(fraction).all()
populations: dict[str, object] = {}
result: dict[str, object] = {
    'completed_local': datetime.now().astimezone().isoformat(),
    'definition': 'min_c ||P128 Gxi - c*d_x(P128 eta)||_2 / ||P128 Gxi||_2; DC removed. Zero means the instantaneous surface evolution is a translation; one means none is explained by a translation.',
    'scope': 'Selected TRAIN examples at depths .20-.35. Probe selection deliberately includes high-wave-height rows, so pooled medians are not population estimates. Report matched H/h .35-.50 separately. No model predictions or new teacher calls. Failed IC targets are archived float32 truth, promoted to float64.',
    'populations': populations,
    'failed_initial_conditions': {str(sid): float(value) for sid, value in zip((16471, 16624), fraction[sample_count:])},
}
for name in np.unique(population):
    selections = {'selected_rows': population == name,
                  'height_matched': (population == name) & (heights >= .35) & (heights <= .50)}
    populations[str(name)] = {
        key: {'rows': int(np.count_nonzero(mask)),
              'quantiles_p10_p50_p90': np.quantile(fraction[:sample_count][mask], [.1, .5, .9]).tolist() if mask.any() else None,
              'source_row_indices': row_indices[mask].tolist()}
        for key, mask in selections.items()
    }
result['elapsed_seconds'] = perf_counter() - started
(directory / 'translation_fit.json').write_text(json.dumps(result, indent=2) + '\n')
np.savez_compressed(directory / 'translation_fit.npz', population=population, source_row_index=row_indices,
                    waveheight_over_depth=heights, translation_residual_fraction=fraction[:sample_count])
print(json.dumps(result, indent=2))
```

#### Exhaustive current depth/height coverage and surface-evolution fit

Same NumPy/uv environment as above, scratch program
`/tmp/dno_dataset_height_slice_20260913.py`:

```python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

root = Path('/home/johnma/dno-fno/outputs/paper_dataset_balanced_20260912/arrays')
out = Path('/home/johnma/dno-fno/outputs/c27_balanced_tanaka_tangent_modal_20260913/diagnostics_dataset_coverage_20260913')
started = perf_counter()
arrays = {name: np.load(root / f'{name}.npy', mmap_mode='r')
          for name in ('eta', 'gxi', 'depth', 'family_id', 'simulation_id', 'dataset_split', 'parameter_group_id')}
selected = (arrays['dataset_split'] == 'train') & (arrays['depth'] >= .2) & (arrays['depth'] <= .35)
k = np.fft.rfftfreq(1024, 1 / 1024)
retained = (k <= 128) & (k != 0)
populations: dict[str, object] = {}
for family, name in ((1, 'stokes'), (2, 'tanaka'), (3, 'benjamin_feir'), (4, 'jonswap_tma')):
    rows = np.flatnonzero(selected & (arrays['family_id'] == family))
    matched_rows: list[np.ndarray] = []
    residual_fractions: list[np.ndarray] = []
    maximum_height = 0.0
    for first in range(0, len(rows), 512):
        indices = rows[first:first + 512]
        eta = arrays['eta'][indices].astype(np.float64)
        height = np.ptp(eta, axis=-1) / arrays['depth'][indices]
        maximum_height = max(maximum_height, float(height.max()))
        matched = (height >= .35) & (height <= .5)
        matched_rows.append(indices[matched])
        if not matched.any():
            continue
        eta_x = np.fft.irfft(np.fft.rfft(eta[matched]) * (1j * k * retained), n=1024)
        gxi = np.fft.irfft(np.fft.rfft(arrays['gxi'][indices[matched]].astype(np.float64)) * retained, n=1024)
        coefficient = np.sum(eta_x * gxi, axis=-1) / np.sum(eta_x**2, axis=-1)
        residual = np.linalg.norm(gxi - coefficient[:, None] * eta_x, axis=-1) / np.linalg.norm(gxi, axis=-1)
        assert np.isfinite(residual).all()
        residual_fractions.append(residual)
    matched_indices = np.concatenate(matched_rows) if matched_rows else np.array([], dtype=np.int64)
    values = np.concatenate(residual_fractions) if residual_fractions else np.array([])
    groups, counts = np.unique(arrays['parameter_group_id'][matched_indices], return_counts=True)
    populations[name] = {
        'depth_matched_rows': len(rows), 'maximum_waveheight_over_depth': maximum_height if len(rows) else None,
        'depth_and_height_matched_rows': int(matched_indices.size),
        'depth_and_height_matched_simulations': int(np.unique(arrays['simulation_id'][matched_indices]).size),
        'groups': {str(g): int(c) for g, c in zip(groups, counts)},
        'translation_residual_quantiles_p0_p10_p50_p90_p100': np.quantile(values, [0, .1, .5, .9, 1]).tolist() if values.size else None,
    }
result = {
    'completed_local': datetime.now().astimezone().isoformat(), 'elapsed_seconds': perf_counter() - started,
    'dataset': str(root), 'depth_interval_inclusive': [.2, .35], 'waveheight_over_depth_interval_inclusive': [.35, .5],
    'scope': 'Exhaustive over current balanced TRAIN rows in the depth interval. Actual peak-to-trough height, not significant height or crest relative to zero. Translation residual uses projected stored labels, not model outputs. This tests nearby parameter coverage, not full functional distance to a failed state.',
    'populations': populations,
}
(out / 'exhaustive_balanced_height_slice.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
```

#### Expanded surface-evolution fit

Executed on CPU with the same one-thread NumPy/uv environment:

```python
from datetime import datetime
from pathlib import Path
from time import perf_counter
import json
import numpy as np
root=Path("outputs/c27_balanced_tanaka_tangent_modal_20260913/diagnostics_dataset_coverage_20260913")
started=perf_counter()
with np.load(root/"coverage_wave_samples.npz") as z:
    eta=z["eta"].astype(np.float64); q=z["gxi"].astype(np.float64)
    group=z["population"]; depth=z["depth"]; height=z["waveheight_over_depth"]
    base=z["depth_sample"]; row=z["source_row_index"]
k=np.fft.rfftfreq(1024,1/1024)
keep=(k>0)&(k<=128)
eta_x=np.fft.irfft(np.fft.rfft(eta)*(1j*k*keep),n=1024)
q=np.fft.irfft(np.fft.rfft(q)*keep,n=1024)
speed=np.sum(eta_x*q,axis=-1)/np.sum(eta_x**2,axis=-1)
residual=np.linalg.norm(q-speed[:,None]*eta_x,axis=-1)/np.linalg.norm(q,axis=-1)
slope=abs(eta_x).max(axis=-1)
assert np.isfinite(residual).all()
populations={}
for name in np.unique(group):
    selections={"unenriched_depth_sample":(group==name)&base,
                "height_matched":(group==name)&(height>=.35)&(height<=.50),
                "unenriched_height_matched":(group==name)&base&(height>=.35)&(height<=.50)}
    populations[str(name)]={key:{"rows":int(mask.sum()),"residual_p10_p50_p90":np.quantile(residual[mask],[.1,.5,.9]).tolist() if mask.any() else None}
                           for key,mask in selections.items()}
examples=[]
for name in ("old_packets","finetune_packets"):
    for sid,h,H,s in ((16471,.2803729881082513,.40193132779046686,.15394692404457022),(16624,.2680272405214529,.4375265626989194,.1743788262489858)):
        eligible=np.flatnonzero((group==name)&(height>=.35)&(height<=.5))
        distance=np.log(depth[eligible]/h)**2+np.log(height[eligible]/H)**2+np.log(slope[eligible]/s)**2
        index=eligible[np.argmin(distance)]
        examples.append({"population":name,"target_ic":sid,"row":int(row[index]),"depth":float(depth[index]),"height_over_depth":float(height[index]),"slope":float(slope[index]),"translation_residual_fraction":float(residual[index]),"limitation":"Nearest among this selected sample in log(depth,height/depth,slope), not exhaustive nearest profile."})
result={"completed_local":datetime.now().astimezone().isoformat(),"elapsed_seconds":perf_counter()-started,
        "definition":"min_c norm(P128 Gxi-c*eta_x)/norm(P128 Gxi); relative RMS of instantaneous surface evolution not explained by a translation. Not total energy or full eta/xi traveling-state test.",
        "sample_rows":len(group),"sampling":"At most512/population from4096 descriptor sample. Explicit unenriched depth_sample subset; height-matched selection deliberately enriched. No population-frequency estimates from enriched rows.",
        "populations":populations,"nearby_examples":examples}
(root/"translation_fit_extended.json").write_text(json.dumps(result,indent=2)+"\n")
np.savez_compressed(root/"translation_fit_extended.npz",population=group,source_row_index=row,depth=depth,height=height,base=base,residual=residual)
print(json.dumps(result,indent=2))
```

#### Direction-group exposure check

This read-only metadata addendum confirms that target-depth random-sea
exposure is.5975% of all July TRAIN rows versus.8944% for balanced JONSWAP,
not a reduction. Three balanced-direction JONSWAP groups contribute2,624
of the8,448 target-depth JONSWAP rows. The change from log-uniform to uniform
depth within a branch is not evidence of lower total sea exposure after
rebalancing the full dataset.

```python
from pathlib import Path
import json
import numpy as np
r=Path("outputs/paper_dataset_balanced_20260912/arrays")
a={n:np.load(r/f"{n}.npy",mmap_mode="r") for n in ("dataset_split","family_id","parameter_group_id","depth","simulation_id")}
pick=(a["dataset_split"]=="train")&(a["depth"]>=.2)&(a["depth"]<=.35)
out={}
for family in (2,4):
    selection=pick&(a["family_id"]==family)
    groups,counts=np.unique(a["parameter_group_id"][selection],return_counts=True)
    out[str(family)]={str(g):int(c) for g,c in zip(groups,counts)}
old_total=6107193
new_total=int(np.count_nonzero(a["dataset_split"]=="train"))
out["target_depth_random_sea_share_all_train"]={"july":36490/old_total,"balanced_jonswap":8448/new_total}
print(json.dumps(out,indent=2))
```
