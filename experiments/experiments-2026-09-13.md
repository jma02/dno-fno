| Date | Time | File / Variant | Motivation | What Tried / Evidence | Correctness | Timing | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-13 | 01:53 | Balanced C27 Modal launch preparation | Local two-GPU training could not allocate memory alongside another user's job; user requested Modal instead. Reuse the same trainer and batch/update budget on two A100-80GB GPUs. | Confirmed Modal CLI1.4.2 and profile/workspace sciml-at-ud. Balanced arrays are not present at /outputs/paper_dataset_balanced_20260912/arrays on volume dno-fno-train-data. Prepared existing launcher with EPOCHS260, batch1024, seed0, same losses, GPU_SPEC=A100-80GB:2. Match image Python minor version3.11 and pin numerical packages to local versions; correct stale all-family translation label. Requested upload of11 NPY files totaling14.715GB plus scoped trainer/model/solver source. | Local Ruff/Pyright and shell syntax pass; independent launcher/path/hyperparameter audit passes. Remote image build and training NOT tested: approval system rejected upload before process creation, requiring explicit permission for disclosure to this specific Modal workspace/volume. No upload, remote training or local job pause occurred; our failed local trainer was already stopped. | 00:02 (inspection/preparation01:51–01:53; no GPU training). | User explicitly approved this destination and two-A100 training before01:59; see the following execution record. Initial rejection was not bypassed. |
| 2026-09-13 | 01:59 | Approved balanced dataset upload and Modal C27 run | User approved disclosure of the14.715GB balanced dataset and scoped training code to sciml-at-ud/dno-fno-train-data, plus training on two A100-80GB GPUs, after local GPU-memory failure. | Existing upload entrypoint completed11 NPY files; existing detached launcher submitted c27_balanced_tanaka_tangent_modal_20260913 with260 epochs, global batch1024, same architecture/losses and FP32 train/val. Training app ap-sDX1uTgQnbRpNWYkCJNgNX; function call fc-01M2CNTNT9DCNGZ57027DG328E. No smaller batch or new trainer. | STARTUP PASS: physical devices are two A100-SXM4-80GB; replicated state/optimizer counters verified at0; first128 updates have finite losses (latestabout.00261). Independent remote config SHA matches local balanced run after excluding only dataset path/run name. All pinned numerical packages match local versions; uploaded trainer sets NCCL_P2P_LEVEL=PHB before JAX. All11 uploaded filenames/sizes checked; remote row counts and normalization match. | Image/upload01:59–02:03; worker started02:04; first finite updates verified02:05. Training ongoing. | RUNNING REMOTELY. No local jobs stopped or modified. Checkpoints/logs persist on approved volume; final training/rollout results pending. This is the260-epoch matched-update-budget balanced-corpus experiment, not a40-epoch shortened run. |

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
