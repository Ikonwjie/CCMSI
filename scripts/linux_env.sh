#!/usr/bin/env bash
# Source this file from the repository's launchers.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-1,2}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHON="${PYTHON:-python}"
if ! command -v -- "$PYTHON" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON; activate the environment from environment.yml or set PYTHON." >&2
    exit 1
fi
