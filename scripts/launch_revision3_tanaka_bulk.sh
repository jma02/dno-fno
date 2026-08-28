#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-$ROOT/.venv/bin/python}
OUTPUT_BASE=${OUTPUT_BASE:-$ROOT/outputs/paper_dataset_revision3_literature_aligned_v1}
GPU_ZERO=${GPU_ZERO:-0}
GPU_ONE=${GPU_ONE:-1}
GPU_ADMISSION_SCRIPT=${GPU_ADMISSION_SCRIPT:-$ROOT/scripts/check_tanaka_gpu_admission.sh}
lane_zero=""
lane_one=""
GPU_ADMISSION_EXIT_AUTHORIZED=false
GUARDED_EXIT_STATUS=0

# lane split count accepted_before stream relative_root label
CHUNKS=(
    "1 train 2048 0 0 train/tanaka/chunk_00000_02048 train_00000_02048"
    "1 train 2048 2048 1 train/tanaka/chunk_02048_02048 train_02048_02048"
    "1 train 4096 4096 2 train/tanaka/chunk_04096_04096 train_04096_04096"
    "0 train 8192 8192 3 train/tanaka/chunk_08192_08192 train_08192_08192"
    "0 validation 1024 0 100 validation/tanaka/chunk_00000_01024 validation_00000_01024"
    "1 test 1024 0 200 test/tanaka/chunk_00000_01024 test_00000_01024"
)

cd "$ROOT"
if ((BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1))); then
    printf 'Bash 5.1 or newer is required\n' >&2
    exit 1
fi
[[ -x "$PYTHON" ]] || { printf 'missing Python: %s\n' "$PYTHON" >&2; exit 1; }
[[ -f "$GPU_ADMISSION_SCRIPT" ]] \
    || { printf 'missing GPU admission helper: %s\n' "$GPU_ADMISSION_SCRIPT" >&2; exit 1; }
# shellcheck source=scripts/check_tanaka_gpu_admission.sh
source "$GPU_ADMISSION_SCRIPT"

guard_reserved_exit_status() {
    local status=$1
    GUARDED_EXIT_STATUS=$status
    if [[ "$GPU_ADMISSION_EXIT_AUTHORIZED" != true \
            && ("$status" == "$TANAKA_GPU_TEMPORARILY_UNAVAILABLE_EXIT" \
                || "$status" == "$TANAKA_GPU_TELEMETRY_ERROR_EXIT") ]]; then
        printf 'reserved GPU-admission exit %s came from a non-admission command; remapping to ordinary failure\n' \
            "$status" >&2
        GUARDED_EXIT_STATUS=1
    fi
}

guard_early_exit() {
    local status=$1
    trap - EXIT
    set +e
    guard_reserved_exit_status "$status"
    exit "$GUARDED_EXIT_STATUS"
}

require_gpu_admission_or_exit() {
    local label=$1 status=0
    GPU_ADMISSION_EXIT_AUTHORIZED=false
    require_tanaka_gpu_admission \
        "$GPU_ZERO" "$GPU_ONE" tanaka_gpu_admission_default_log "$label" \
        || status=$?
    if ((status != 0)); then
        GPU_ADMISSION_EXIT_AUTHORIZED=true
        exit "$status"
    fi
}

trap 'guard_early_exit $?' EXIT
require_gpu_admission_or_exit pre-preflight
mkdir -p "$OUTPUT_BASE/logs"

declare -a COMMAND PREFLIGHTS
LANE=""; SPLIT=""; COUNT=""; BEFORE=""; STREAM=""; LEAF=""; LABEL=""
CHUNK_ROOT=""; SUMMARY=""; PREFLIGHT=""; LOG=""; GPU=""

build_chunk() {
    read -r LANE SPLIT COUNT BEFORE STREAM LEAF LABEL <<<"$1"
    GPU=$GPU_ONE
    [[ "$LANE" == 0 ]] && GPU=$GPU_ZERO
    CHUNK_ROOT="$OUTPUT_BASE/$LEAF"
    SUMMARY="$CHUNK_ROOT/paper_dataset_tanaka_${SPLIT}.summary.json"
    PREFLIGHT="$OUTPUT_BASE/logs/tanaka_${LABEL}.preflight.json"
    LOG="$OUTPUT_BASE/logs/tanaka_${LABEL}.log"
    COMMAND=(
        "$PYTHON" scripts/generate_paper_dataset.py
        --family tanaka --split "$SPLIT"
        --accepted-cases "$COUNT" --accepted-cases-before "$BEFORE"
        --stream-id "$STREAM" --first-attempt-index 0
        --batch-size 256 --maximum-attempts-per-accepted-case 4
        --platform gpu --output-root "$CHUNK_ROOT"
    )
}

# Keep the numerical Python process attached to its worker lane on signals.
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
        MPLCONFIGDIR="/tmp/mpl-paper-tanaka-revision3-gpu${GPU}" \
        "${COMMAND[@]}" --dry-run >"$PREFLIGHT"
    PREFLIGHTS+=("$PREFLIGHT")
}

# Freeze all six plans before either GPU starts numerical generation.  This is
# a fresh deterministic population; it does not reproduce old random cases.
for row in "${CHUNKS[@]}"; do preflight_chunk "$row"; done
"$PYTHON" -c '
import json, sys
from pathlib import Path

plans = [json.loads(Path(path).read_text()) for path in sys.argv[1:]]
contracts = []
for plan in plans:
    run = plan.get("run_spec", {})
    config = run.get("configuration", {})
    if (plan.get("schema") != "paper_dataset_quota_preflight_v1"
            or plan.get("no_numerical_generation_performed") is not True
            or run.get("family_name") != "tanaka"
            or run.get("revision_id") != 3 or run.get("batch_size") != 256
            or run.get("maximum_attempts_per_accepted_case") != 4
            or config.get("execution_platform") != "gpu"
            or plan.get("execution") != config.get("trajectory_execution")):
        raise SystemExit("invalid revision-3 Tanaka preflight")
    contracts.append(json.dumps({key: config.get(key) for key in (
        "trajectory_execution", "ordered_cell_ids")},
        sort_keys=True, separators=(",", ":")))
if len(plans) != 6 or len(set(contracts)) != 1:
    raise SystemExit("Tanaka preflights do not share one numerical contract")
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
        MPLCONFIGDIR="/tmp/mpl-paper-tanaka-revision3-gpu${GPU}" \
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
    local status=$1
    trap - EXIT INT TERM
    set +e
    guard_reserved_exit_status "$status"
    status=$GUARDED_EXIT_STATUS
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

# Preflights can take long enough for another user to claim a device after the
# initial check.  Recheck immediately before either numerical lane is forked.
require_gpu_admission_or_exit pre-lane-fork
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

summaries=()
for row in "${CHUNKS[@]}"; do build_chunk "$row"; summaries+=("$SUMMARY"); done
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PYTHON" -c '
import json, sys
from pathlib import Path
from scripts.build_paper_dataset_view import load_completed_chunk

chunks = tuple(load_completed_chunk(Path(path)) for path in sys.argv[1:])
expected = {
    "train": ((0, 2048, 0), (2048, 4096, 1),
              (4096, 8192, 2), (8192, 16384, 3)),
    "validation": ((0, 1024, 100),), "test": ((0, 1024, 200),),
}
observed = {}
for split, wanted in expected.items():
    selected = sorted((c for c in chunks if c.split.value == split),
                      key=lambda c: c.accepted_before)
    got = tuple((c.accepted_before, c.accepted_after, c.stream_id)
                for c in selected)
    if got != wanted or len({c.root for c in selected}) != len(selected):
        raise SystemExit(f"invalid Tanaka {split} intervals: {got}")
    observed[split] = got
if len(chunks) != 6 or len({c.revision_id for c in chunks}) != 1:
    raise SystemExit("completed Tanaka chunks do not share one revision")
print(json.dumps({"status": "passed", "family": "tanaka",
    "accepted_cases": {"train": 16384, "validation": 1024, "test": 1024},
    "intervals_with_stream_id": observed,
    "historical_random_specifications_reused": False,
    "full_four_family_view_preflight_pending": True}, indent=2))
' "${summaries[@]}" >"$OUTPUT_BASE/tanaka_view_builder_input_check.json"

trap - EXIT INT TERM
printf 'revision-3 Tanaka generation complete: %s\n' "$OUTPUT_BASE"
printf 'final gate: run the view-builder preflight with all four families\n'
