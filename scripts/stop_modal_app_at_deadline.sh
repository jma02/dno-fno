#!/usr/bin/env bash
# Stop one Modal app at an absolute local-time deadline.

set -euo pipefail
cd /home/johnma/dno-fno

APP_ID="${1:?usage: $0 APP_ID DEADLINE}"
DEADLINE="${2:?usage: $0 APP_ID DEADLINE}"

PROFILE="$(uv run modal profile current)"
if [[ "$PROFILE" != "jma02" ]]; then
  echo "refusing to manage Modal app under profile '$PROFILE'" >&2
  exit 1
fi

deadline_epoch="$(date -d "$DEADLINE" +%s)"
while (( $(date +%s) < deadline_epoch )); do
  remaining=$((deadline_epoch - $(date +%s)))
  (( remaining < 60 )) && wait_seconds="$remaining" || wait_seconds=60
  sleep "$wait_seconds"
done

echo "$(date --iso-8601=seconds) stopping $APP_ID at budget deadline $DEADLINE"
uv run modal app stop "$APP_ID"
