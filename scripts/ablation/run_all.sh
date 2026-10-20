#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/linux_env.sh"
cd "$ROOT"
"$PYTHON" run_batch.py --suite configs/suites/ablation.json "$@"
