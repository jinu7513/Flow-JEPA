#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch, not a login-node shell}"
: "${FLOWJEPA_REPO:?}" "${FLOWJEPA_PYTHON:?}" "${FLOWJEPA_DATA_ROOT:?}"
: "${FLOWJEPA_EVAL_OUTPUT:?}" "${FLOWJEPA_EVAL_CHECKPOINT:?}" "${FLOWJEPA_EVAL_ID:?}"
cd "$FLOWJEPA_REPO"
export LOCAL_DATASET_DIR="$FLOWJEPA_DATA_ROOT"
export STABLEWM_HOME="$FLOWJEPA_LOG_ROOT/$FLOWJEPA_EVAL_ID/worldmodel-cache"
export SDL_VIDEODRIVER=dummy PYTHONUTF8=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'Job=%s Host=%s Python=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$FLOWJEPA_PYTHON" "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
"$FLOWJEPA_PYTHON" -c 'import torch; assert torch.cuda.is_available(); p=torch.cuda.get_device_properties(0); print("Allocated CUDA GPU:",p.name,"VRAM GiB:",p.total_memory/2**30)'
source "$FLOWJEPA_REPO/scripts/circe_eval_args.sh"
flowjepa_build_eval_args
printf 'Evaluation command:'; printf ' %q' "$FLOWJEPA_PYTHON" "${evaluation_args[@]}"; printf '\n'
nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu \
    --format=csv --loop=30 > "$FLOWJEPA_LOG_ROOT/$FLOWJEPA_EVAL_ID/gpu-$SLURM_JOB_ID.csv" &
gpu_monitor=$!
trap 'kill "$gpu_monitor" 2>/dev/null || true' EXIT
srun "$FLOWJEPA_PYTHON" "${evaluation_args[@]}"
