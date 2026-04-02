#!/usr/bin/env zsh

set -u

SCRIPT_DIR=${0:A:h}
REPO_ROOT=${SCRIPT_DIR:h:h}
SOLITON_ROOT=${SOLITON_ROOT:-/home/johnma/dno_locl/soliton_data}
OUTPUT_DIR=${1:-${REPO_ROOT}/outputs/tanaka_ic_comparisons_all_q4}
PYTHON_BIN=${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}

mkdir -p "${OUTPUT_DIR}"

for path in "${SOLITON_ROOT}"/coll.anim_s* "${SOLITON_ROOT}"/coll.anim_h* "${SOLITON_ROOT}"/coll.anim_f*; do
  case_name=${path:t}
  printf '[%s] START %s\n' "$(/usr/bin/date '+%F %T')" "${case_name}"
  MPLCONFIGDIR=/tmp/matplotlib JAX_PLATFORM_NAME=cpu \
    "${PYTHON_BIN}" -u -m solver.tanaka_ICs.plot_ic_comparisons \
    --cases "${case_name}" \
    --output_dir "${OUTPUT_DIR}"
  exit_code=$?
  printf '[%s] END %s status=%s\n' "$(/usr/bin/date '+%F %T')" "${case_name}" "${exit_code}"
done
