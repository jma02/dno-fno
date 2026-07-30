#!/usr/bin/env bash

cd /home/johnma/dno-fno || exit 1
export PATH="/home/johnma/.local/bin:/usr/local/bin:/usr/bin:/bin"

exec bash scripts/handoff_c26_modal_to_local.sh \
  >> logs/c26_modal_to_local_handoff_20260717.log 2>&1
