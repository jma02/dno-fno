| Date | Time | File / Variant | Motivation | What Tried / Evidence | Correctness | Timing | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-13 | 01:53 | Balanced C27 Modal launch preparation | Local two-GPU training could not allocate memory alongside another user's job; user requested Modal instead. Reuse the same trainer and batch/update budget on two A100-80GB GPUs. | Confirmed Modal CLI1.4.2 and profile/workspace sciml-at-ud. Balanced arrays are not present at /outputs/paper_dataset_balanced_20260912/arrays on volume dno-fno-train-data. Prepared existing launcher with EPOCHS260, batch1024, seed0, same losses, GPU_SPEC=A100-80GB:2. Match image Python minor version3.11 and pin numerical packages to local versions; correct stale all-family translation label. Requested upload of11 NPY files totaling14.715GB plus scoped trainer/model/solver source. | Local Ruff/Pyright and shell syntax pass; independent launcher/path/hyperparameter audit passes. Remote image build and training NOT tested: approval system rejected upload before process creation, requiring explicit permission for disclosure to this specific Modal workspace/volume. No upload, remote training or local job pause occurred; our failed local trainer was already stopped. | 00:02 (inspection/preparation01:51–01:53; no GPU training). | User explicitly approved this destination and two-A100 training before01:59; see the following execution record. Initial rejection was not bypassed. |
| 2026-09-13 | 01:59 | Approved balanced dataset upload and Modal C27 run | User approved disclosure of the14.715GB balanced dataset and scoped training code to sciml-at-ud/dno-fno-train-data, plus training on two A100-80GB GPUs, after local GPU-memory failure. | Execute existing upload entrypoint with approved profile; app ap-oghqmhf5mjGPvhtUzaaEXf builds pinned numerical image and uploads11 NPY files. Intended fresh run c27_balanced_tanaka_tangent_modal_20260913, EPOCHS260, global batch1024, same architecture/losses, FP32 train/val, NCCL_P2P_LEVEL=PHB. | Upload/image build IN PROGRESS; training not yet submitted. Remote path/run-name absence confirmed before upload. No local jobs stopped or modified. | Started01:59; completion pending. | IN PROGRESS: verify upload sizes, launch using existing detached Modal recipe, then inspect real GPU count, resolved configuration, versions and first finite updates. |

### Approved commands (upload started01:59; training pending)

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
