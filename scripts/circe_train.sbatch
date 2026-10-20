#!/bin/bash
# Paths/resources are supplied explicitly by circe_submit.sh.
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch, not a login-node shell}"
: "${FLOWJEPA_REPO:?}" "${FLOWJEPA_PYTHON:?}" "${FLOWJEPA_EXPERIMENT_DIR:?}"
: "${FLOWJEPA_DATA_ROOT:?}" "${FLOWJEPA_RUN_ID:?}" "${FLOWJEPA_MODE:?}"
cd "$FLOWJEPA_REPO"
[[ "$FLOWJEPA_PYTHON" = /* && -x "$FLOWJEPA_PYTHON" ]]
if [[ "$FLOWJEPA_MODE" = resume ]]; then
    test -d "$FLOWJEPA_EXPERIMENT_DIR"
else
    # Atomic reservation prevents two new jobs from sharing an experiment.
    mkdir "$FLOWJEPA_EXPERIMENT_DIR"
fi
exec 9>"$FLOWJEPA_EXPERIMENT_DIR/.training.lock"
flock -n 9 || { echo 'Another job is using this Flow-JEPA experiment' >&2; exit 1; }
export STABLEWM_HOME="$FLOWJEPA_EXPERIMENT_DIR/worldmodel"
export LOCAL_DATASET_DIR="$FLOWJEPA_DATA_ROOT"
export SPT_CACHE_DIR="$FLOWJEPA_EXPERIMENT_DIR/trainer"
export WANDB_DIR="$FLOWJEPA_EXPERIMENT_DIR/wandb"
export SDL_VIDEODRIVER=dummy PYTHONUTF8=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$STABLEWM_HOME" "$SPT_CACHE_DIR" "$WANDB_DIR"
test -f "$LOCAL_DATASET_DIR/datasets/pusht_expert_train.h5"
printf 'Job=%s Host=%s Workdir=%s Python=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$PWD" "$FLOWJEPA_PYTHON" "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
export FLOWJEPA_PRECISION=${FLOWJEPA_PRECISION:-16-mixed}
"$FLOWJEPA_PYTHON" -c 'import os,torch; assert torch.cuda.is_available(), "Allocated GPU is unavailable to PyTorch"; assert os.environ["FLOWJEPA_PRECISION"] != "bf16-mixed" or torch.cuda.is_bf16_supported(), "BF16 unsupported on allocated GPU"'
"$FLOWJEPA_PYTHON" -c 'import sys,torch; p=torch.cuda.get_device_properties(0); print("Python:",sys.executable,"Torch:",torch.__version__,"CUDA GPU:",p.name,"VRAM GiB:",p.total_memory/2**30)'
"$FLOWJEPA_PYTHON" -m pip freeze > "$FLOWJEPA_EXPERIMENT_DIR/packages-$SLURM_JOB_ID.txt"
git rev-parse HEAD > "$FLOWJEPA_EXPERIMENT_DIR/git-$SLURM_JOB_ID.txt"
git diff > "$FLOWJEPA_EXPERIMENT_DIR/changes-$SLURM_JOB_ID.patch"
source "$FLOWJEPA_REPO/scripts/circe_args.sh"
flowjepa_build_train_args
printf 'Training command:'; printf ' %q' "$FLOWJEPA_PYTHON" "${training_args[@]}"; printf '\n'
nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu \
    --format=csv --loop=30 > "$FLOWJEPA_EXPERIMENT_DIR/gpu-$SLURM_JOB_ID.csv" &
gpu_monitor=$!
trap 'kill "$gpu_monitor" 2>/dev/null || true' EXIT
srun "$FLOWJEPA_PYTHON" "${training_args[@]}"
