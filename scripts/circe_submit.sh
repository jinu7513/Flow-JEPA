#!/bin/bash
# Default: preview. --check: sbatch --test-only. --submit: enqueue a job.
set -euo pipefail
mode=${1:?Usage: bash scripts/circe_submit.sh smoke|train|resume [--check|--submit]}
action=${2:---preview}
[[ "$mode" =~ ^(smoke|train|resume)$ ]]
[[ "$action" =~ ^--(preview|check|submit)$ ]]
for key in FLOWJEPA_REPO FLOWJEPA_PYTHON FLOWJEPA_DATA_ROOT FLOWJEPA_RUNS_ROOT FLOWJEPA_LOG_ROOT; do
    value=${!key:-}
    [[ "$value" = /* && "$value" != *[$'\n\r']* ]] || { echo "$key must be an absolute Linux path" >&2; exit 1; }
done
: "${FLOWJEPA_PARTITION:?Confirm your current partition first}"
: "${FLOWJEPA_QOS:?Confirm your current QoS first}"
: "${FLOWJEPA_RUN_ID:?Use a new identifier beginning with flowjepa-}"
[[ "$FLOWJEPA_RUN_ID" =~ ^flowjepa-[a-zA-Z0-9-]{1,48}$ ]]
[[ -d "$FLOWJEPA_REPO/.git" && -x "$FLOWJEPA_PYTHON" ]]
test -f "$FLOWJEPA_DATA_ROOT/datasets/pusht_expert_train.h5"
export FLOWJEPA_MODE=$mode
[[ "$mode" = smoke ]] && export FLOWJEPA_RUN_ID="$FLOWJEPA_RUN_ID-smoke"
export FLOWJEPA_EXPERIMENT_DIR="$FLOWJEPA_RUNS_ROOT/$FLOWJEPA_RUN_ID"
if [[ "$mode" = resume ]]; then
    : "${FLOWJEPA_RESUME:?Set an absolute last.ckpt path from this experiment}"
    FLOWJEPA_RESUME=$(realpath -e "$FLOWJEPA_RESUME")
    case "$FLOWJEPA_RESUME" in
        "$FLOWJEPA_EXPERIMENT_DIR"/trainer/*/last.ckpt) ;;
        *) echo 'Resume checkpoint must belong to this Flow-JEPA experiment' >&2; exit 1;;
    esac
    export FLOWJEPA_RESUME
elif [[ -e "$FLOWJEPA_EXPERIMENT_DIR" ]]; then
    echo 'Experiment exists. Choose a fresh run ID or use resume.' >&2; exit 1
fi
time_limit=${FLOWJEPA_TIME:-00:15:00}
if [[ "$mode" != smoke ]]; then
    : "${FLOWJEPA_TIME:?Choose a walltime accepted by your QoS}"
fi
log_dir="$FLOWJEPA_LOG_ROOT/$FLOWJEPA_RUN_ID"
cmd=(sbatch --parsable --nodes=1 --ntasks=1 --gres=gpu:1
    --cpus-per-task=8 --mem=64G "--time=$time_limit" --no-requeue
    "--partition=$FLOWJEPA_PARTITION" "--qos=$FLOWJEPA_QOS"
    "--job-name=$FLOWJEPA_RUN_ID" "--chdir=$FLOWJEPA_REPO"
    "--output=$log_dir/%x-%j.out" "--error=$log_dir/%x-%j.err" --open-mode=append --export=ALL)
[[ -n "${FLOWJEPA_ACCOUNT:-}" ]] && cmd+=("--account=$FLOWJEPA_ACCOUNT")
script="$FLOWJEPA_REPO/scripts/circe_train.sbatch"
source "$(dirname "${BASH_SOURCE[0]}")/circe_args.sh"
flowjepa_build_train_args
printf 'Mode=%s GPU=1 CPUs=8 RAM=64G Time=%s Batch=%s\n' "$mode" "$time_limit" "${FLOWJEPA_BATCH_SIZE:-8}" >&2
printf 'Absolute Python: %s\nExperiment: %s\nLogs: %s\n' "$FLOWJEPA_PYTHON" "$FLOWJEPA_EXPERIMENT_DIR" "$log_dir" >&2
printf 'Training command:' >&2; printf ' %q' "$FLOWJEPA_PYTHON" "${training_args[@]}" >&2; printf '\n' >&2
printf 'Submission:' >&2; printf ' %q' "${cmd[@]}" "$script" >&2; printf '\n' >&2
if [[ "$action" = --preview ]]; then exit 0; fi
mkdir -p "$log_dir" "$FLOWJEPA_RUNS_ROOT"
if [[ "$action" = --check ]]; then
    "${cmd[@]}" --test-only "$script"
else
    "${cmd[@]}" "$script"
fi
