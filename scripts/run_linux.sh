#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/linux_env.sh"
cd "$ROOT"
if [[ $# -eq 0 ]]; then
    echo "Usage: bash scripts/run_linux.sh <run_experiment.py|run_batch.py|scripts/check_environment.py> [arguments]" >&2
    exit 2
fi
exec "$PYTHON" "$@"
