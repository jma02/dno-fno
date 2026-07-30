#!/bin/bash
# Submitted to at(1): start the idle-GPU watcher for C21 evaluation.

cd /home/johnma/dno-fno || exit 1
export PATH="/home/johnma/.local/bin:/usr/local/bin:/usr/bin:/bin"

exec env \
  RUN_DIR=outputs/c21_tangent_w100_from_c20_20260714_171442 \
  QUEUE_LOG=/tmp/c21_tangent_w100_from_c20_20260714_171442.non_tanaka_queue.log \
  bash scripts/queue_c21_non_tanaka_eval_when_gpus_idle.sh
