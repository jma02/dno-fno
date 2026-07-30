#!/usr/bin/env bash
# Wait for C25 to complete, preserve C26's Modal checkpoint, stop its paid app,
# download the run, and resume the remaining epochs on the two local GPUs.

set -euo pipefail
cd /home/johnma/dno-fno

export MODAL_PROFILE=sciml-at-ud

C25_RUN="c25_capacity125_full_20260716_022603"
C26_RUN="c26_no_g1_oeta2_full_modal_l40sx4_20260716_211238"
C26_APP="ap-paVW3shGxebK1SNPNaURVc"
VOLUME="dno-fno-train-data"
POLL_SECONDS=60

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

checkpoint_epoch() {
  jq -r '.epoch' "$1"
}

log "handoff monitor started; waiting for C25 epoch 40"
while pgrep -f "1d_dno_fno_jax.py.*--run_name ${C25_RUN}" >/dev/null; do
  if [[ -f "outputs/${C25_RUN}/latest_ckpt/metadata.json" ]]; then
    log "C25 latest committed epoch $(checkpoint_epoch "outputs/${C25_RUN}/latest_ckpt/metadata.json")"
  else
    log "C25 has no committed latest checkpoint yet"
  fi
  sleep "${POLL_SECONDS}"
done

C25_FINAL="outputs/${C25_RUN}/final_ckpt/metadata.json"
if [[ ! -f "${C25_FINAL}" ]] || [[ "$(checkpoint_epoch "${C25_FINAL}")" != "40" ]]; then
  log "C25 process ended without a committed epoch-40 final checkpoint; leaving Modal C26 running"
  exit 1
fi
log "C25 epoch-40 final checkpoint is committed"

while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | rg -q '[0-9]'; do
  log "waiting for local GPU processes to release their contexts"
  sleep 15
done

REMOTE_META="$(mktemp /tmp/c26-modal-metadata.XXXXXX.json)"
for attempt in {1..10}; do
  if modal volume get \
      "${VOLUME}" \
      "/outputs/${C26_RUN}/latest_ckpt/metadata.json" \
      "${REMOTE_META}" \
      --force >/dev/null 2>&1; then
    REMOTE_EPOCH="$(checkpoint_epoch "${REMOTE_META}")"
    if modal volume ls \
        "${VOLUME}" \
        "/outputs/${C26_RUN}/latest_ckpt/ckpt_${REMOTE_EPOCH}" \
        --json | jq -e 'length > 0' >/dev/null; then
      log "persistent Modal volume contains committed C26 epoch ${REMOTE_EPOCH}"
      break
    fi
  fi
  if [[ "${attempt}" == "10" ]]; then
    log "could not verify a complete C26 checkpoint on the persistent volume; leaving app running"
    exit 1
  fi
  sleep 30
done

APP_STATE="$(modal app list --json | jq -r --arg id "${C26_APP}" '.[] | select(."App ID" == $id) | .State')"
if [[ "${APP_STATE}" != "stopped" ]]; then
  log "stopping Modal app ${C26_APP} at the last committed epoch"
  modal app stop "${C26_APP}" --yes
fi

for _ in {1..20}; do
  APP_STATE="$(modal app list --json | jq -r --arg id "${C26_APP}" '.[] | select(."App ID" == $id) | .State')"
  [[ "${APP_STATE}" == "stopped" ]] && break
  sleep 5
done
if [[ "${APP_STATE}" != "stopped" ]]; then
  log "Modal app did not reach stopped state; refusing to launch a duplicate local trainer"
  exit 1
fi
log "Modal app is stopped"

modal volume get \
  "${VOLUME}" \
  "/outputs/${C26_RUN}" \
  "outputs" \
  --force

LOCAL_META="outputs/${C26_RUN}/latest_ckpt/metadata.json"
if [[ ! -f "${LOCAL_META}" ]]; then
  log "download did not produce ${LOCAL_META}"
  exit 1
fi
LOCAL_EPOCH="$(checkpoint_epoch "${LOCAL_META}")"
LOCAL_PAYLOAD="outputs/${C26_RUN}/latest_ckpt/ckpt_${LOCAL_EPOCH}"
if [[ ! -d "${LOCAL_PAYLOAD}" ]] || [[ ! -f "${LOCAL_PAYLOAD}/_CHECKPOINT_METADATA" ]]; then
  log "downloaded C26 checkpoint is incomplete at epoch ${LOCAL_EPOCH}"
  exit 1
fi
log "downloaded and structurally validated C26 epoch ${LOCAL_EPOCH}"

RUN_NAME="${C26_RUN}" scripts/launch_c26_no_g1_local_resume.sh
