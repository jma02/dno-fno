#!/usr/bin/env bash
set -euo pipefail

# Start only the read-only post-completion runner.  This launcher deliberately
# does not inspect, signal, restart, or otherwise interact with generation jobs.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-$ROOT/.venv/bin/python}
TMUX_BIN=${TMUX_BIN:-tmux}
SESSION=${SESSION:-dno_paper_dataset_postcompletion}
LAUNCH_LOG=${LAUNCH_LOG:-$ROOT/outputs/paper_dataset_postcompletion_launcher.log}
RUNNER=$ROOT/scripts/run_paper_dataset_postcompletion.py

[[ -x "$PYTHON" ]] || { printf 'missing Python: %s\n' "$PYTHON" >&2; exit 1; }
[[ -f "$RUNNER" ]] || { printf 'missing runner: %s\n' "$RUNNER" >&2; exit 1; }
command -v "$TMUX_BIN" >/dev/null 2>&1 \
    || { printf 'missing tmux command: %s\n' "$TMUX_BIN" >&2; exit 1; }
if "$TMUX_BIN" has-session -t "$SESSION" 2>/dev/null; then
    printf 'post-completion tmux session already exists: %s\n' "$SESSION" >&2
    exit 1
fi

mkdir -p "$(dirname "$LAUNCH_LOG")"
printf -v COMMAND 'cd %q && exec env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false %q %q' \
    "$ROOT" "$PYTHON" "$RUNNER"
for argument in "$@"; do
    printf -v QUOTED_ARGUMENT '%q' "$argument"
    COMMAND+=" $QUOTED_ARGUMENT"
done
printf -v QUOTED_LOG '%q' "$LAUNCH_LOG"
COMMAND+=" >>$QUOTED_LOG 2>&1"

"$TMUX_BIN" new-session -d -s "$SESSION" "$COMMAND"
printf 'launched %s; log: %s\n' "$SESSION" "$LAUNCH_LOG"
