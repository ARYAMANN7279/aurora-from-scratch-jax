#!/bin/bash
# Restored replacement for the missing scratch wrapper. No child shell reads ~/.bashrc.
set -euo pipefail
task_repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
exec "$task_repo/.venv/bin/python" "$task_repo/experiments/lab/gpu_guard.py" "$@"
