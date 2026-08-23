#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-$ROOT/.venv/bin/python}
OUTPUT_BASE=${OUTPUT_BASE:-$ROOT/outputs/paper_dataset_literature_aligned_v1}

STOKES_ROOT=$ROOT/outputs/paper_dataset_cap4_revision2_20260728
TANAKA_ROOT=$ROOT/outputs/paper_dataset_revision3_literature_aligned_v1
BF_ROOT=$ROOT/outputs/paper_dataset_bf_revision4_jonswap_revision3_literature_aligned_v1
JONSWAP_ROOT=$ROOT/outputs/paper_dataset_jonswap_revision4_relative_band_v1

STOKES_TRAIN=(
    "$STOKES_ROOT/train/stokes/chunk_00000_02048/paper_dataset_stokes_train.summary.json"
    "$STOKES_ROOT/train/stokes/chunk_02048_02048/paper_dataset_stokes_train.summary.json"
    "$STOKES_ROOT/train/stokes/chunk_04096_04096/paper_dataset_stokes_train.summary.json"
    "$STOKES_ROOT/train/stokes/chunk_08192_08192/paper_dataset_stokes_train.summary.json"
)
TANAKA_TRAIN=(
    "$TANAKA_ROOT/train/tanaka/chunk_00000_02048/paper_dataset_tanaka_train.summary.json"
    "$TANAKA_ROOT/train/tanaka/chunk_02048_02048/paper_dataset_tanaka_train.summary.json"
    "$TANAKA_ROOT/train/tanaka/chunk_04096_04096/paper_dataset_tanaka_train.summary.json"
    "$TANAKA_ROOT/train/tanaka/chunk_08192_08192/paper_dataset_tanaka_train.summary.json"
)
BF_TRAIN=(
    "$BF_ROOT/train/benjamin_feir/chunk_00000_02048/paper_dataset_benjamin_feir_train.summary.json"
    "$BF_ROOT/train/benjamin_feir/chunk_02048_02048/paper_dataset_benjamin_feir_train.summary.json"
    "$BF_ROOT/train/benjamin_feir/chunk_04096_04096/paper_dataset_benjamin_feir_train.summary.json"
    "$BF_ROOT/train/benjamin_feir/chunk_08192_08192/paper_dataset_benjamin_feir_train.summary.json"
)
JONSWAP_TRAIN=(
    "$JONSWAP_ROOT/train/jonswap_tma/chunk_00000_02048/paper_dataset_jonswap_tma_train.summary.json"
    "$JONSWAP_ROOT/train/jonswap_tma/chunk_02048_02048/paper_dataset_jonswap_tma_train.summary.json"
    "$JONSWAP_ROOT/train/jonswap_tma/chunk_04096_04096/paper_dataset_jonswap_tma_train.summary.json"
    "$JONSWAP_ROOT/train/jonswap_tma/chunk_08192_04096/paper_dataset_jonswap_tma_train.summary.json"
    "$JONSWAP_ROOT/train/jonswap_tma/chunk_12288_02048/paper_dataset_jonswap_tma_train.summary.json"
    "$JONSWAP_ROOT/train/jonswap_tma/chunk_14336_02048/paper_dataset_jonswap_tma_train.summary.json"
)
FIXED_SPLITS=(
    "$STOKES_ROOT/validation/stokes/c01024/paper_dataset_stokes_validation.summary.json"
    "$TANAKA_ROOT/validation/tanaka/chunk_00000_01024/paper_dataset_tanaka_validation.summary.json"
    "$BF_ROOT/validation/benjamin_feir/chunk_00000_01024/paper_dataset_benjamin_feir_validation.summary.json"
    "$JONSWAP_ROOT/validation/jonswap_tma/chunk_00000_01024/paper_dataset_jonswap_tma_validation.summary.json"
    "$STOKES_ROOT/test/stokes/c01024/paper_dataset_stokes_test.summary.json"
    "$TANAKA_ROOT/test/tanaka/chunk_00000_01024/paper_dataset_tanaka_test.summary.json"
    "$BF_ROOT/test/benjamin_feir/chunk_00000_01024/paper_dataset_benjamin_feir_test.summary.json"
    "$JONSWAP_ROOT/test/jonswap_tma/chunk_00000_01024/paper_dataset_jonswap_tma_test.summary.json"
)

cd "$ROOT"
[[ -x "$PYTHON" ]] || { printf 'missing Python: %s\n' "$PYTHON" >&2; exit 1; }
mkdir -p "$OUTPUT_BASE/logs"

append_first() {
    local count=$1 name=$2 index
    local -n source=$name
    for ((index = 0; index < count; index++)); do
        COMMAND+=(--chunk-summary "${source[index]}")
    done
}

build_checkpoint() {
    local accepted=$1 standard_count=$2 jonswap_count=$3 tag
    printf -v tag '%05d' "$accepted"
    local output_root="$OUTPUT_BASE/combined/c${tag}_v01024_t01024"
    local name="paper_dataset_all_splits_c${tag}"
    local preflight="$OUTPUT_BASE/logs/${name}.preflight.json"
    local build_log="$OUTPUT_BASE/logs/${name}.build.json"

    COMMAND=(
        env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
        MPLCONFIGDIR=/tmp/mpl-paper-dataset-combined
        "$PYTHON" scripts/build_paper_dataset_view.py
    )
    append_first "$standard_count" STOKES_TRAIN
    append_first "$standard_count" TANAKA_TRAIN
    append_first "$standard_count" BF_TRAIN
    append_first "$jonswap_count" JONSWAP_TRAIN
    local summary
    for summary in "${FIXED_SPLITS[@]}"; do
        COMMAND+=(--chunk-summary "$summary")
    done
    COMMAND+=(--output-root "$output_root" --name "$name")

    "${COMMAND[@]}" --dry-run >"$preflight"
    "${COMMAND[@]}" --execute >"$build_log"
}

build_checkpoint 2048 1 1
build_checkpoint 4096 2 2
build_checkpoint 8192 3 3
build_checkpoint 16384 4 6

printf 'literature-aligned paper-dataset views complete: %s\n' "$OUTPUT_BASE"
