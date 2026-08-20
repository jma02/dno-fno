#!/usr/bin/env bash

# Shared read-only admission check for the two batch-256 Tanaka GPU lanes.
# The measured peak was 35,025 MiB; the 40-GiB floor preserves 5,935 MiB.
TANAKA_GPU_MINIMUM_FREE_MIB=${TANAKA_GPU_MINIMUM_FREE_MIB:-40960}
TANAKA_GPU_MEASURED_PEAK_MIB=35025
NVIDIA_SMI_BIN=${NVIDIA_SMI_BIN:-nvidia-smi}
TANAKA_GPU_TEMPORARILY_UNAVAILABLE_EXIT=75
TANAKA_GPU_TELEMETRY_ERROR_EXIT=70

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'check_tanaka_gpu_admission.sh is a source-only helper\n' >&2
    exit "$TANAKA_GPU_TELEMETRY_ERROR_EXIT"
fi

tanaka_gpu_admission_default_log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" >&2
}

_tanaka_gpu_admission_error() {
    local logger=$1
    shift
    "$logger" "ERROR: Tanaka GPU admission telemetry invalid: $*"
    return 2
}

_tanaka_gpu_admission_compact() {
    local value=$1
    value=${value//[[:space:]]/}
    printf '%s' "$value"
}

_tanaka_gpu_admission_query() {
    local gpu=$1 logger=$2 telemetry compute_apps status=0
    local observed_index free_mib total_mib extra pid compact_pids="" commas

    telemetry=$("$NVIDIA_SMI_BIN" -i "$gpu" \
        --query-gpu=index,memory.free,memory.total \
        --format=csv,noheader,nounits 2>&1) || status=$?
    ((status == 0)) \
        || _tanaka_gpu_admission_error \
            "$logger" "physical GPU $gpu inventory query exited $status"
    ((status == 0)) || return 2
    [[ -n "$telemetry" && "$telemetry" != *$'\n'* ]] \
        || { _tanaka_gpu_admission_error \
            "$logger" "physical GPU $gpu inventory must be exactly one row"; return 2; }
    commas=${telemetry//[^,]/}
    [[ "$commas" == ",," ]] \
        || { _tanaka_gpu_admission_error \
            "$logger" "physical GPU $gpu inventory must contain exactly three fields"; return 2; }

    IFS=',' read -r observed_index free_mib total_mib extra <<<"$telemetry"
    [[ "$observed_index" =~ ^[[:space:]]*(0|[1-9][0-9]*)[[:space:]]*$ \
        && "$free_mib" =~ ^[[:space:]]*(0|[1-9][0-9]*)[[:space:]]*$ \
        && "$total_mib" =~ ^[[:space:]]*(0|[1-9][0-9]*)[[:space:]]*$ ]] \
        || { _tanaka_gpu_admission_error \
            "$logger" "physical GPU $gpu inventory contains malformed fields"; return 2; }
    observed_index=$(_tanaka_gpu_admission_compact "$observed_index")
    free_mib=$(_tanaka_gpu_admission_compact "$free_mib")
    total_mib=$(_tanaka_gpu_admission_compact "$total_mib")
    extra=$(_tanaka_gpu_admission_compact "$extra")
    [[ -z "$extra" && "$observed_index" =~ ^(0|[1-9][0-9]*)$ \
        && "$free_mib" =~ ^(0|[1-9][0-9]*)$ \
        && "$total_mib" =~ ^(0|[1-9][0-9]*)$ ]] \
        || { _tanaka_gpu_admission_error \
            "$logger" "physical GPU $gpu inventory contains malformed fields"; return 2; }
    [[ "$observed_index" == "$gpu" ]] \
        || { _tanaka_gpu_admission_error \
            "$logger" "requested physical GPU $gpu but nvidia-smi returned $observed_index"; return 2; }
    ((10#$free_mib <= 10#$total_mib)) \
        || { _tanaka_gpu_admission_error \
            "$logger" "physical GPU $gpu reports free memory above total memory"; return 2; }
    ((10#$total_mib >= TANAKA_GPU_MINIMUM_FREE_MIB)) \
        || { _tanaka_gpu_admission_error \
            "$logger" "physical GPU $gpu total memory is below the admission floor"; return 2; }

    status=0
    compute_apps=$("$NVIDIA_SMI_BIN" -i "$gpu" \
        --query-compute-apps=pid --format=csv,noheader,nounits 2>&1) \
        || status=$?
    ((status == 0)) \
        || _tanaka_gpu_admission_error \
            "$logger" "physical GPU $gpu compute-process query exited $status"
    ((status == 0)) || return 2
    if [[ -n "$compute_apps" ]]; then
        while IFS= read -r pid; do
            [[ "$pid" =~ ^[[:space:]]*[1-9][0-9]*[[:space:]]*$ ]] \
                || { _tanaka_gpu_admission_error \
                    "$logger" "physical GPU $gpu compute-process row is malformed"; return 2; }
            pid=$(_tanaka_gpu_admission_compact "$pid")
            [[ ",$compact_pids," == *",$pid,"* ]] \
                || compact_pids=${compact_pids:+$compact_pids,}$pid
        done <<<"$compute_apps"
    fi

    TANAKA_GPU_ADMISSION_FREE_MIB=$((10#$free_mib))
    TANAKA_GPU_ADMISSION_TOTAL_MIB=$((10#$total_mib))
    TANAKA_GPU_ADMISSION_PIDS=$compact_pids
}

# Return 0 when both GPUs are admissible, 1 when valid telemetry says to wait,
# and 2 when telemetry or the requested physical indices cannot be trusted.
check_tanaka_gpu_admission() {
    local gpu_zero=$1 gpu_one=$2
    local logger=${3:-tanaka_gpu_admission_default_log}
    local label=${4:-check}
    local gpu_zero_free gpu_zero_total gpu_zero_pids
    local gpu_one_free gpu_one_total gpu_one_pids
    local detail_zero detail_one admission_margin_mib

    [[ "$gpu_zero" =~ ^(0|[1-9][0-9]*)$ \
        && "$gpu_one" =~ ^(0|[1-9][0-9]*)$ \
        && "$gpu_zero" != "$gpu_one" ]] \
        || { _tanaka_gpu_admission_error \
            "$logger" "configured physical GPU indices must be distinct canonical decimals"; return 2; }
    [[ "$TANAKA_GPU_MINIMUM_FREE_MIB" =~ ^(0|[1-9][0-9]*)$ \
        && 10#$TANAKA_GPU_MINIMUM_FREE_MIB -ge 40960 ]] \
        || { _tanaka_gpu_admission_error \
            "$logger" "minimum free memory must be at least 40960 MiB"; return 2; }
    admission_margin_mib=$((${TANAKA_GPU_MINIMUM_FREE_MIB} - TANAKA_GPU_MEASURED_PEAK_MIB))
    command -v "$NVIDIA_SMI_BIN" >/dev/null 2>&1 \
        || { _tanaka_gpu_admission_error \
            "$logger" "nvidia-smi command is unavailable: $NVIDIA_SMI_BIN"; return 2; }

    _tanaka_gpu_admission_query "$gpu_zero" "$logger" || return $?
    gpu_zero_free=$TANAKA_GPU_ADMISSION_FREE_MIB
    gpu_zero_total=$TANAKA_GPU_ADMISSION_TOTAL_MIB
    gpu_zero_pids=$TANAKA_GPU_ADMISSION_PIDS
    _tanaka_gpu_admission_query "$gpu_one" "$logger" || return $?
    gpu_one_free=$TANAKA_GPU_ADMISSION_FREE_MIB
    gpu_one_total=$TANAKA_GPU_ADMISSION_TOTAL_MIB
    gpu_one_pids=$TANAKA_GPU_ADMISSION_PIDS

    detail_zero="gpu=$gpu_zero free=$gpu_zero_free/$gpu_zero_total MiB"
    detail_one="gpu=$gpu_one free=$gpu_one_free/$gpu_one_total MiB"
    [[ -z "$gpu_zero_pids" ]] \
        || detail_zero="$detail_zero compute_pids=$gpu_zero_pids"
    [[ -z "$gpu_one_pids" ]] \
        || detail_one="$detail_one compute_pids=$gpu_one_pids"
    if [[ -n "$gpu_zero_pids" || -n "$gpu_one_pids" \
            || "$gpu_zero_free" -lt "$TANAKA_GPU_MINIMUM_FREE_MIB" \
            || "$gpu_one_free" -lt "$TANAKA_GPU_MINIMUM_FREE_MIB" ]]; then
        "$logger" "Tanaka GPU admission WAIT ($label): $detail_zero; $detail_one; required_free=${TANAKA_GPU_MINIMUM_FREE_MIB} MiB"
        return 1
    fi

    "$logger" "Tanaka GPU admission PASS ($label): $detail_zero; $detail_one; required_free=${TANAKA_GPU_MINIMUM_FREE_MIB} MiB; measured_peak=${TANAKA_GPU_MEASURED_PEAK_MIB} MiB; margin=${admission_margin_mib} MiB"
}

# Translate the tri-state check into stable process exit statuses for the
# direct launcher and its supervising retry loop.
require_tanaka_gpu_admission() {
    local status=0
    check_tanaka_gpu_admission "$@" || status=$?
    case "$status" in
        0) return 0 ;;
        1) return "$TANAKA_GPU_TEMPORARILY_UNAVAILABLE_EXIT" ;;
        *) return "$TANAKA_GPU_TELEMETRY_ERROR_EXIT" ;;
    esac
}
