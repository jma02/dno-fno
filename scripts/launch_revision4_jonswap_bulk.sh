#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-$ROOT/.venv/bin/python}
OUTPUT_BASE=${OUTPUT_BASE:-$ROOT/outputs/paper_dataset_jonswap_revision4_relative_band_v1}
GPU_ZERO=${GPU_ZERO:-0}
GPU_ONE=${GPU_ONE:-1}
lane_zero=""
lane_one=""

# lane split count accepted_before stream relative_root label
# The last 8,192-simulation learning-curve increment is split into three disjoint
# execution shards so the two remaining GPU lanes have nearly equal work.
CHUNKS=(
    "0 train 2048 2048 1 train/jonswap_tma/chunk_02048_02048 train_02048_02048"
    "0 train 4096 4096 2 train/jonswap_tma/chunk_04096_04096 train_04096_04096"
    "1 validation 1024 0 100 validation/jonswap_tma/chunk_00000_01024 validation_00000_01024"
    "1 test 1024 0 200 test/jonswap_tma/chunk_00000_01024 test_00000_01024"
    "1 train 4096 8192 3 train/jonswap_tma/chunk_08192_04096 train_08192_04096"
    "0 train 2048 12288 4 train/jonswap_tma/chunk_12288_02048 train_12288_02048"
    "1 train 2048 14336 5 train/jonswap_tma/chunk_14336_02048 train_14336_02048"
)

cd "$ROOT"
if ((BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1))); then
    printf 'Bash 5.1 or newer is required\n' >&2
    exit 1
fi
[[ -x "$PYTHON" ]] || { printf 'missing Python: %s\n' "$PYTHON" >&2; exit 1; }
[[ "$GPU_ZERO" != "$GPU_ONE" ]] || { printf 'GPUs must be distinct\n' >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { printf 'nvidia-smi is required\n' >&2; exit 1; }
for gpu in "$GPU_ZERO" "$GPU_ONE"; do
    nvidia-smi -i "$gpu" --query-gpu=index --format=csv,noheader >/dev/null 2>&1 \
        || { printf 'unavailable GPU: %s\n' "$gpu" >&2; exit 1; }
done
mkdir -p "$OUTPUT_BASE/logs"

declare -a COMMAND PREFLIGHTS
LANE=""; SPLIT=""; COUNT=""; BEFORE=""; STREAM=""; LEAF=""; LABEL=""
CHUNK_ROOT=""; SUMMARY=""; PREFLIGHT=""; LOG=""; GPU=""

build_chunk() {
    read -r LANE SPLIT COUNT BEFORE STREAM LEAF LABEL <<<"$1"
    GPU=$GPU_ONE
    [[ "$LANE" == 0 ]] && GPU=$GPU_ZERO
    CHUNK_ROOT="$OUTPUT_BASE/$LEAF"
    SUMMARY="$CHUNK_ROOT/paper_dataset_jonswap_tma_${SPLIT}.summary.json"
    PREFLIGHT="$OUTPUT_BASE/logs/jonswap_${LABEL}.preflight.json"
    LOG="$OUTPUT_BASE/logs/jonswap_${LABEL}.log"
    COMMAND=(
        "$PYTHON" scripts/generate_paper_dataset_jonswap.py
        --solver-batch-size 8
        --family jonswap_tma --split "$SPLIT"
        --accepted-simulations "$COUNT" --accepted-simulations-before "$BEFORE"
        --stream-id "$STREAM" --first-attempt-index 0
        --batch-size 32
        --platform gpu --output-root "$CHUNK_ROOT"
    )
}

run_child() {
    local child=""
    stop_child() {
        local status=$1
        trap - INT TERM
        [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true
        [[ -z "$child" ]] || wait "$child" 2>/dev/null || true
        exit "$status"
    }
    trap 'stop_child 130' INT
    trap 'stop_child 143' TERM
    "$@" & child=$!
    local status=0
    wait "$child" || status=$?
    trap - INT TERM
    return "$status"
}

preflight_chunk() {
    build_chunk "$1"
    run_child env CUDA_VISIBLE_DEVICES="$GPU" \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        JAX_ENABLE_X64=true \
        MPLCONFIGDIR="/tmp/mpl-paper-jonswap-revision4-gpu${GPU}" \
        "${COMMAND[@]}" --dry-run >"$PREFLIGHT"
    PREFLIGHTS+=("$PREFLIGHT")
}

# Freeze every remaining plan before either lane resumes numerical work.
for row in "${CHUNKS[@]}"; do preflight_chunk "$row"; done
"$PYTHON" -c '
import json, sys
from pathlib import Path

plans = [json.loads(Path(path).read_text()) for path in sys.argv[1:]]
contracts = []
for plan in plans:
    run = plan.get("run_spec", {})
    config = run.get("configuration", {})
    numerical = plan.get("execution", {}).get("numerical", {})
    policy = config.get("jonswap_horizon_bucketing", {})
    if (plan.get("schema") != "paper_dataset_quota_preflight_v1"
            or plan.get("no_numerical_generation_performed") is not True
            or run.get("family_name") != "jonswap_tma"
            or run.get("revision_id") != 4 or run.get("batch_size") != 32
            or run.get("maximum_retries_per_parameter_group") != 32
            or config.get("execution_platform") != "gpu"
            or plan.get("execution") != config.get("trajectory_execution")
            or numerical.get("nx") != 2048
            or numerical.get("maximum_wavenumber") != 704.0
            or numerical.get("gl2_iteration_cap") != 5
            or policy.get("solver_batch_size") != 8):
        raise SystemExit("invalid revision-4 JONSWAP preflight")
    contracts.append(json.dumps({key: config.get(key) for key in (
        "trajectory_execution", "ordered_cell_ids")},
        sort_keys=True, separators=(",", ":")))
if len(plans) != 7 or len(set(contracts)) != 1:
    raise SystemExit("JONSWAP preflights do not share one numerical contract")
' "${PREFLIGHTS[@]}"

check_summary() {
    "$PYTHON" -c '
import json, sys
from pathlib import Path
p, s = (json.loads(Path(path).read_text()) for path in sys.argv[1:])
if s.get("status") != "complete" or s.get("run_spec") != p.get("run_spec"):
    raise SystemExit("completion differs from its preflight")
' "$PREFLIGHT" "$SUMMARY"
}

run_chunk() {
    build_chunk "$1"
    printf '\n[%s] starting %s on GPU %s\n' \
        "$(date --iso-8601=seconds)" "$CHUNK_ROOT" "$GPU" >>"$LOG"
    run_child env CUDA_VISIBLE_DEVICES="$GPU" \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        JAX_ENABLE_X64=true \
        MPLCONFIGDIR="/tmp/mpl-paper-jonswap-revision4-gpu${GPU}" \
        PYTHONUNBUFFERED=1 "${COMMAND[@]}" --execute >>"$LOG" 2>&1
    check_summary
}

run_lane() {
    local wanted=$1 row
    for row in "${CHUNKS[@]}"; do
        build_chunk "$row"
        [[ "$LANE" != "$wanted" ]] || run_chunk "$row"
    done
}

cleanup() {
    local status=$1 pid
    trap - EXIT INT TERM
    for pid in "$lane_zero" "$lane_one"; do
        [[ -z "$pid" ]] || kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "$lane_zero" "$lane_one"; do
        [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
    done
    exit "$status"
}
trap 'cleanup $?' EXIT
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

run_lane 0 & lane_zero=$!
run_lane 1 & lane_one=$!
finished=""; first_status=0
wait -n -p finished "$lane_zero" "$lane_one" || first_status=$?
if [[ "$finished" == "$lane_zero" ]]; then
    lane_zero=""; remaining=$lane_one
else
    lane_one=""; remaining=$lane_zero
fi
((first_status == 0)) || cleanup "$first_status"
remaining_status=0
wait "$remaining" || remaining_status=$?
[[ "$remaining" != "$lane_zero" ]] || lane_zero=""
[[ "$remaining" != "$lane_one" ]] || lane_one=""
((remaining_status == 0)) || cleanup "$remaining_status"

summaries=(
    "$OUTPUT_BASE/train/jonswap_tma/chunk_00000_02048/paper_dataset_jonswap_tma_train.summary.json"
)
for row in "${CHUNKS[@]}"; do build_chunk "$row"; summaries+=("$SUMMARY"); done
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PYTHON" -c '
import json, sys
from pathlib import Path
from scripts.build_paper_dataset_view import load_completed_chunk

chunks = tuple(load_completed_chunk(Path(path)) for path in sys.argv[1:])
expected = {
    "train": ((0, 2048, 0), (2048, 4096, 1), (4096, 8192, 2),
              (8192, 12288, 3), (12288, 14336, 4),
              (14336, 16384, 5)),
    "validation": ((0, 1024, 100),), "test": ((0, 1024, 200),),
}
observed = {}
for split, wanted in expected.items():
    selected = sorted((c for c in chunks if c.split.value == split),
                      key=lambda c: c.accepted_before)
    got = tuple((c.accepted_before, c.accepted_after, c.stream_id)
                for c in selected)
    if got != wanted or len({c.root for c in selected}) != len(selected):
        raise SystemExit(f"invalid JONSWAP {split} intervals: {got}")
    observed[split] = got
if len(chunks) != 8 or len({c.revision_id for c in chunks}) != 1:
    raise SystemExit("completed JONSWAP chunks do not share one revision")
print(json.dumps({"status": "passed", "family": "jonswap_tma",
    "accepted_simulations": {"train": 16384, "validation": 1024, "test": 1024},
    "intervals_with_stream_id": observed,
    "full_four_family_view_preflight_pending": True}, indent=2))
' "${summaries[@]}" >"$OUTPUT_BASE/jonswap_view_builder_input_check.json"

trap - EXIT INT TERM
printf 'revision-4 JONSWAP generation complete: %s\n' "$OUTPUT_BASE"
printf 'final gate: run the view-builder preflight with all four families\n'
