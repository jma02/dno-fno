| Date | Time | File / Variant | Motivation | What Tried / Evidence | Correctness | Timing | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-13 | 01:53 | Balanced C27 Modal launch preparation | Local two-GPU training could not allocate memory alongside another user's job; user requested Modal instead. Reuse the same trainer and batch/update budget on two A100-80GB GPUs. | Confirmed Modal CLI1.4.2 and profile/workspace sciml-at-ud. Balanced arrays are not present at /outputs/paper_dataset_balanced_20260912/arrays on volume dno-fno-train-data. Prepared existing launcher with EPOCHS260, batch1024, seed0, same losses, GPU_SPEC=A100-80GB:2. Match image Python minor version3.11 and pin numerical packages to local versions; correct stale all-family translation label. Requested upload of11 NPY files totaling14.715GB plus scoped trainer/model/solver source. | Local Ruff/Pyright and shell syntax pass; independent launcher/path/hyperparameter audit passes. Remote image build and training NOT tested: approval system rejected upload before process creation, requiring explicit permission for disclosure to this specific Modal workspace/volume. No upload, remote training or local job pause occurred; our failed local trainer was already stopped. | 00:02 (inspection/preparation01:51–01:53; no GPU training). | AWAIT EXPLICIT DESTINATION APPROVAL. Do not bypass upload rejection. Once approved, upload to sciml-at-ud/dno-fno-train-data and use the existing launcher with a fresh run name; verify actual remote devices/configuration and first optimizer updates. |

### Prepared commands (upload blocked; neither command executed)

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
