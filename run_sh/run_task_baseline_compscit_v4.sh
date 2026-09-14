#!/bin/bash
# Baseline result phase (README): ../src/task_baseline_compscit_v4.py
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/src/task_baseline_compscit_v4.py"
if [[ ! -f "$PY" ]]; then
  echo "run_task_baseline_compscit_v4.sh: missing $PY" >&2
  echo "Add task_baseline_compscit_v4.py (e.g. from main PhysVEC) and wire arguments here." >&2
  exit 1
fi
exec python "$PY" "$@"
