#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-$ROOT/.venv/bin/python}
BASH_BIN=${BASH_BIN:-bash}
TMUX_BIN=${TMUX_BIN:-tmux}
SLEEP_BIN=${SLEEP_BIN:-sleep}
TIMEOUT_BIN=${TIMEOUT_BIN:-timeout}
NVIDIA_SMI_BIN=${NVIDIA_SMI_BIN:-nvidia-smi}

WAIT_SESSION=${WAIT_SESSION:-dno_jonswap_r4}
WAIT_INTERVAL_SECONDS=${WAIT_INTERVAL_SECONDS:-60}
TANAKA_GPU_WAIT_INTERVAL_SECONDS=${TANAKA_GPU_WAIT_INTERVAL_SECONDS:-60}
RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS:-10}
MAX_RETRIES=${MAX_RETRIES:-3}
AUDIT_TIMEOUT_SECONDS=${AUDIT_TIMEOUT_SECONDS:-3600}
BUILD_TIMEOUT_SECONDS=${BUILD_TIMEOUT_SECONDS:-1800}
TIMEOUT_KILL_AFTER_SECONDS=${TIMEOUT_KILL_AFTER_SECONDS:-120}

JONSWAP_ROOT=${JONSWAP_ROOT:-$ROOT/outputs/paper_corpus_jonswap_revision4_relative_band_v1}
TANAKA_ROOT=${TANAKA_ROOT:-$ROOT/outputs/paper_corpus_revision3_literature_aligned_v1}
JONSWAP_COMPLETION_GATE=${JONSWAP_COMPLETION_GATE:-$JONSWAP_ROOT/jonswap_view_builder_input_check.json}
TANAKA_COMPLETION_GATE=${TANAKA_COMPLETION_GATE:-$TANAKA_ROOT/tanaka_view_builder_input_check.json}
JONSWAP_AUDIT_ARTIFACT=${JONSWAP_AUDIT_ARTIFACT:-$JONSWAP_ROOT/jonswap_tma_completion_audit.json}
TANAKA_AUDIT_ARTIFACT=${TANAKA_AUDIT_ARTIFACT:-$TANAKA_ROOT/tanaka_completion_audit.json}

JONSWAP_LAUNCH_SCRIPT=${JONSWAP_LAUNCH_SCRIPT:-$ROOT/scripts/launch_revision4_jonswap_bulk.sh}
TANAKA_LAUNCH_SCRIPT=${TANAKA_LAUNCH_SCRIPT:-$ROOT/scripts/launch_revision3_tanaka_bulk.sh}
GPU_ADMISSION_SCRIPT=${GPU_ADMISSION_SCRIPT:-$ROOT/scripts/check_tanaka_gpu_admission.sh}
GPU_ZERO=${GPU_ZERO:-0}
GPU_ONE=${GPU_ONE:-1}
JONSWAP_AUDIT_SCRIPT=${JONSWAP_AUDIT_SCRIPT:-$ROOT/scripts/audit_completed_jonswap_revision4.py}
TANAKA_AUDIT_SCRIPT=${TANAKA_AUDIT_SCRIPT:-$ROOT/scripts/audit_completed_tanaka_revision3.py}
BUILD_SCRIPT=${BUILD_SCRIPT:-$ROOT/scripts/build_literature_aligned_paper_corpus_views.sh}
SUPERVISOR_LOG=${SUPERVISOR_LOG:-$ROOT/outputs/paper_corpus_sequential_supervisor.log}

ACTIVE_CHILD=""

log() {
    local message timestamp
    timestamp=$(date --iso-8601=seconds)
    printf -v message '[%s] %s\n' "$timestamp" "$*"
    printf '%s' "$message" >&2
    printf '%s' "$message" >>"$SUPERVISOR_LOG"
}

die() {
    log "ERROR: $*"
    exit 1
}

normalize_nonnegative_integer() {
    local name=$1 value=$2 maximum=${3:-}
    [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a nonnegative integer"
    value=$((10#$value))
    [[ -z "$maximum" || "$value" -le "$maximum" ]] \
        || die "$name must be at most $maximum"
    printf '%s' "$value"
}

normalize_positive_integer() {
    local name=$1 value=$2 maximum=${3:-}
    value=$(normalize_nonnegative_integer "$name" "$value" "$maximum")
    ((value > 0)) || die "$name must be positive"
    printf '%s' "$value"
}

stop_active_child() {
    local status=$1
    trap - INT TERM
    if [[ -n "$ACTIVE_CHILD" ]]; then
        kill -TERM "$ACTIVE_CHILD" 2>/dev/null || true
        wait "$ACTIVE_CHILD" 2>/dev/null || true
    fi
    exit "$status"
}

run_supervised_child() {
    local status=0
    "$@" &
    ACTIVE_CHILD=$!
    wait "$ACTIVE_CHILD" || status=$?
    ACTIVE_CHILD=""
    return "$status"
}

run_bounded_child() {
    local timeout_seconds=$1
    shift
    run_supervised_child "$TIMEOUT_BIN" \
        --signal=TERM \
        --kill-after="${TIMEOUT_KILL_AFTER_SECONDS}s" \
        "${timeout_seconds}s" \
        "$@"
}

completion_gate_passes() {
    local path=$1 family=$2
    [[ -f "$path" ]] || return 1
    "$PYTHON" -c '
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
family = sys.argv[2]
try:
    record = json.loads(path.read_text(encoding="utf-8"))
except (OSError, UnicodeDecodeError, json.JSONDecodeError):
    raise SystemExit(1)
expected = {"train": 16384, "validation": 1024, "test": 1024}
if (record.get("status") != "passed"
        or record.get("family") != family
        or record.get("accepted_cases") != expected):
    raise SystemExit(1)
' "$path" "$family" >/dev/null 2>&1
}

wait_for_session() {
    if ! "$TMUX_BIN" has-session -t "$WAIT_SESSION" 2>/dev/null; then
        log "tmux session $WAIT_SESSION is absent; checking its completion gate"
        return
    fi
    log "waiting for existing tmux session $WAIT_SESSION"
    while "$TMUX_BIN" has-session -t "$WAIT_SESSION" 2>/dev/null; do
        "$SLEEP_BIN" "$WAIT_INTERVAL_SECONDS"
    done
    log "tmux session $WAIT_SESSION ended; checking its completion gate"
}

resume_until_gate() {
    local label=$1 family=$2 corpus_root=$3 gate=$4 launch_script=$5
    local temporary_status=${6:-} telemetry_status=${7:-}
    local launch_number launch_status max_launches
    max_launches=$((MAX_RETRIES + 1))

    if completion_gate_passes "$gate" "$family"; then
        log "$label lightweight completion gate already passes: $gate"
        return
    fi

    for ((launch_number = 1; launch_number <= max_launches; launch_number++)); do
        log "$label exact-resume launch $launch_number/$max_launches: $launch_script"
        launch_status=0
        run_supervised_child env \
            OUTPUT_BASE="$corpus_root" \
            PYTHON="$PYTHON" \
            "$BASH_BIN" "$launch_script" || launch_status=$?

        if completion_gate_passes "$gate" "$family"; then
            log "$label lightweight completion gate passes after launch $launch_number"
            return
        fi

        if [[ -n "$temporary_status" \
                && "$launch_status" == "$temporary_status" ]]; then
            log "$label launcher reported newly unavailable GPUs; waiting without consuming retry budget"
            wait_for_tanaka_gpu_admission
            launch_number=$((launch_number - 1))
            continue
        fi
        if [[ -n "$telemetry_status" \
                && "$launch_status" == "$telemetry_status" ]]; then
            die "$label launcher rejected invalid GPU telemetry"
        fi

        log "$label launch $launch_number exited $launch_status without a valid gate"
        if ((launch_number < max_launches)); then
            log "$label retrying exact resume in ${RETRY_DELAY_SECONDS}s"
            "$SLEEP_BIN" "$RETRY_DELAY_SECONDS"
        fi
    done

    die "$label exhausted $MAX_RETRIES retries without a valid completion gate"
}

wait_for_tanaka_gpu_admission() {
    local admission_status=0

    while :; do
        admission_status=0
        check_tanaka_gpu_admission \
            "$GPU_ZERO" "$GPU_ONE" log supervisor-pre-launch \
            || admission_status=$?
        case "$admission_status" in
            0) return ;;
            1)
                log "Tanaka GPU admission retrying in ${TANAKA_GPU_WAIT_INTERVAL_SECONDS}s; foreign processes are never signaled"
                "$SLEEP_BIN" "$TANAKA_GPU_WAIT_INTERVAL_SECONDS" \
                    || die "Tanaka GPU admission sleep command failed"
                ;;
            *) die "Tanaka GPU admission telemetry is unavailable or malformed" ;;
        esac
    done
}

run_completion_audit() {
    local label=$1 schema=$2 corpus_root=$3 artifact=$4 audit_script=$5
    local expected_rows=$6 audit_status=0

    log "$label CPU completion audit starting (timeout ${AUDIT_TIMEOUT_SECONDS}s; TERM then KILL after ${TIMEOUT_KILL_AFTER_SECONDS}s): $audit_script"
    run_bounded_child "$AUDIT_TIMEOUT_SECONDS" env \
        CUDA_VISIBLE_DEVICES="" \
        JAX_PLATFORMS=cpu \
        JAX_ENABLE_X64=true \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$PYTHON" "$audit_script" \
        --root "$corpus_root" --output "$artifact" || audit_status=$?
    if ((audit_status == 124 || audit_status == 137)); then
        die "$label CPU completion audit exceeded ${AUDIT_TIMEOUT_SECONDS}s"
    fi
    ((audit_status == 0)) \
        || die "$label CPU completion audit exited $audit_status"

    run_supervised_child "$PYTHON" -c '
import json
import sys
from pathlib import Path

artifact = Path(sys.argv[1]).resolve()
expected_schema = sys.argv[2]
expected_root = Path(sys.argv[3]).resolve()
expected_rows = int(sys.argv[4])
try:
    record = json.loads(artifact.read_text(encoding="utf-8"))
except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid audit artifact: {error}") from error
expected_splits = {"train": 16384, "validation": 1024, "test": 1024}
if (record.get("schema") != expected_schema
        or record.get("status") != "pass"
        or Path(record.get("corpus_root", "")).resolve() != expected_root
        or record.get("accepted") != 18432
        or record.get("accepted_by_split") != expected_splits
        or record.get("retained_rows") != expected_rows
        or record.get("attempted", 0) < record.get("accepted", 0)
        or record.get("rejected")
        != record.get("attempted") - record.get("accepted")):
    raise SystemExit("completion audit artifact does not satisfy the release gate")
' "$artifact" "$schema" "$corpus_root" "$expected_rows" \
        || die "$label completion audit artifact did not pass validation: $artifact"
    log "$label CPU completion audit passed: $artifact"
}

mkdir -p "$(dirname "$SUPERVISOR_LOG")"
WAIT_INTERVAL_SECONDS=$(normalize_nonnegative_integer \
    WAIT_INTERVAL_SECONDS "$WAIT_INTERVAL_SECONDS" 60)
TANAKA_GPU_WAIT_INTERVAL_SECONDS=$(normalize_nonnegative_integer \
    TANAKA_GPU_WAIT_INTERVAL_SECONDS "$TANAKA_GPU_WAIT_INTERVAL_SECONDS" 3600)
RETRY_DELAY_SECONDS=$(normalize_nonnegative_integer \
    RETRY_DELAY_SECONDS "$RETRY_DELAY_SECONDS" 60)
MAX_RETRIES=$(normalize_nonnegative_integer MAX_RETRIES "$MAX_RETRIES")
AUDIT_TIMEOUT_SECONDS=$(normalize_positive_integer \
    AUDIT_TIMEOUT_SECONDS "$AUDIT_TIMEOUT_SECONDS" 86400)
BUILD_TIMEOUT_SECONDS=$(normalize_positive_integer \
    BUILD_TIMEOUT_SECONDS "$BUILD_TIMEOUT_SECONDS" 86400)
TIMEOUT_KILL_AFTER_SECONDS=$(normalize_positive_integer \
    TIMEOUT_KILL_AFTER_SECONDS "$TIMEOUT_KILL_AFTER_SECONDS" 3600)

[[ -x "$PYTHON" ]] || die "missing Python executable: $PYTHON"
command -v "$BASH_BIN" >/dev/null 2>&1 || die "missing bash command: $BASH_BIN"
command -v "$TMUX_BIN" >/dev/null 2>&1 || die "missing tmux command: $TMUX_BIN"
command -v "$SLEEP_BIN" >/dev/null 2>&1 || die "missing sleep command: $SLEEP_BIN"
command -v "$TIMEOUT_BIN" >/dev/null 2>&1 \
    || die "missing timeout command: $TIMEOUT_BIN"
for required_script in \
    "$JONSWAP_LAUNCH_SCRIPT" "$TANAKA_LAUNCH_SCRIPT" \
    "$JONSWAP_AUDIT_SCRIPT" "$TANAKA_AUDIT_SCRIPT" "$BUILD_SCRIPT" \
    "$GPU_ADMISSION_SCRIPT"; do
    [[ -f "$required_script" ]] || die "missing required script: $required_script"
done
# shellcheck source=scripts/check_tanaka_gpu_admission.sh
source "$GPU_ADMISSION_SCRIPT"

trap 'stop_active_child 130' INT
trap 'stop_active_child 143' TERM

log "sequential corpus supervisor started (maximum retries per family: $MAX_RETRIES)"
wait_for_session
resume_until_gate \
    JONSWAP jonswap_tma "$JONSWAP_ROOT" \
    "$JONSWAP_COMPLETION_GATE" "$JONSWAP_LAUNCH_SCRIPT"
run_completion_audit \
    JONSWAP paper_corpus_jonswap_tma_revision4_completion_audit_v1 \
    "$JONSWAP_ROOT" "$JONSWAP_AUDIT_ARTIFACT" \
    "$JONSWAP_AUDIT_SCRIPT" 294912

if ! completion_gate_passes "$TANAKA_COMPLETION_GATE" tanaka; then
    wait_for_tanaka_gpu_admission
fi
resume_until_gate \
    Tanaka tanaka "$TANAKA_ROOT" \
    "$TANAKA_COMPLETION_GATE" "$TANAKA_LAUNCH_SCRIPT" \
    "$TANAKA_GPU_TEMPORARILY_UNAVAILABLE_EXIT" \
    "$TANAKA_GPU_TELEMETRY_ERROR_EXIT"
run_completion_audit \
    Tanaka paper_corpus_tanaka_revision3_completion_audit_v1 \
    "$TANAKA_ROOT" "$TANAKA_AUDIT_ARTIFACT" \
    "$TANAKA_AUDIT_SCRIPT" 3686400

log "both audited family gates pass; building final literature-aligned views (timeout ${BUILD_TIMEOUT_SECONDS}s; TERM then KILL after ${TIMEOUT_KILL_AFTER_SECONDS}s)"
build_status=0
run_bounded_child "$BUILD_TIMEOUT_SECONDS" \
    "$BASH_BIN" "$BUILD_SCRIPT" || build_status=$?
if ((build_status == 124 || build_status == 137)); then
    die "final view build exceeded ${BUILD_TIMEOUT_SECONDS}s"
fi
((build_status == 0)) || die "final view build exited $build_status"
log "final literature-aligned view build completed"
