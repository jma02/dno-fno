#!/bin/zsh
set -euo pipefail

script_path="${(%):-%N}"
repo_root="${script_path:A:h:h}"

detect_backend() {
  if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    echo "mps"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "cuda"
    return
  fi

  echo "cpu"
}

backend="${1:-auto}"
if [[ "$backend" == "auto" ]]; then
  backend="$(detect_backend)"
fi

case "$backend" in
  mps)
    export PYTORCH_ENABLE_MPS_FALLBACK="1"
    unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
    ;;
  cuda)
    unset PYTORCH_ENABLE_MPS_FALLBACK 2>/dev/null || true
    ;;
  cpu)
    unset PYTORCH_ENABLE_MPS_FALLBACK 2>/dev/null || true
    ;;
  *)
    echo "Unknown backend: $backend" >&2
    echo "Expected one of: auto, mps, cuda, cpu" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

cd "$repo_root"
uv sync --python 3.9

echo "Initialized dno-fno environment"
echo "  backend: $backend"
echo "  repo:    $repo_root"
echo
echo "Run training with:"
echo "  uv run --python 3.9 python train/1d_dno_fno.py --device $backend"
