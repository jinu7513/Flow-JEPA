#!/bin/bash
# One GPU job evaluates all three conditions sequentially. Default: preview.
set -euo pipefail
mode=${1:?Usage: bash scripts/circe_eval_submit.sh smoke|full [--check|--submit]}
action=${2:---preview}
[[ "$mode" =~ ^(smoke|full)$ && "$action" =~ ^--(preview|check|submit)$ ]]
for key in FLOWJEPA_REPO FLOWJEPA_PYTHON FLOWJEPA_DATA_ROOT FLOWJEPA_RUNS_ROOT FLOWJEPA_LOG_ROOT FLOWJEPA_EVAL_CHECKPOINT; do
    value=${!key:-}
    [[ "$value" = /* && "$value" != *[$'\n\r']* ]] || { echo "$key must be an absolute Linux path" >&2; exit 1; }
done
: "${FLOWJEPA_PARTITION:?Confirm current allocation}"
: "${FLOWJEPA_QOS:?Confirm current QoS}"
: "${FLOWJEPA_EVAL_ID:?Choose a fresh ID starting with flowjepa-eval-}"
[[ "$FLOWJEPA_EVAL_ID" =~ ^flowjepa-eval-[a-zA-Z0-9-]{1,48}$ ]]
[[ -x "$FLOWJEPA_PYTHON" && -f "$FLOWJEPA_REPO/eval_pusht_suite.py" ]]
[[ "$FLOWJEPA_EVAL_CHECKPOINT" = *.pt && -f "$FLOWJEPA_EVAL_CHECKPOINT" ]]
test -f "$(dirname "$FLOWJEPA_EVAL_CHECKPOINT")/config.json"
test -f "$FLOWJEPA_DATA_ROOT/datasets/pusht_expert_train.h5"
export FLOWJEPA_EVAL_MODE=$mode
[[ "$mode" = smoke ]] && export FLOWJEPA_EVAL_ID="$FLOWJEPA_EVAL_ID-smoke"
eval_root=${FLOWJEPA_EVAL_ROOT:-$FLOWJEPA_RUNS_ROOT/evaluations}
[[ "$eval_root" = /* ]]
export FLOWJEPA_EVAL_OUTPUT="$eval_root/$FLOWJEPA_EVAL_ID"
[[ ! -e "$FLOWJEPA_EVAL_OUTPUT" ]] || { echo 'Evaluation exists; choose a new evaluation ID' >&2; exit 1; }
time_limit=${FLOWJEPA_EVAL_TIME:-00:15:00}
[[ "$mode" = smoke ]] || : "${FLOWJEPA_EVAL_TIME:?Choose a confirmed full-evaluation walltime}"
log_dir="$FLOWJEPA_LOG_ROOT/$FLOWJEPA_EVAL_ID"
cmd=(sbatch --parsable --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8
    --mem=64G "--time=$time_limit" --no-requeue "--partition=$FLOWJEPA_PARTITION"
    "--qos=$FLOWJEPA_QOS" "--job-name=$FLOWJEPA_EVAL_ID" "--chdir=$FLOWJEPA_REPO"
    "--output=$log_dir/%x-%j.out" "--error=$log_dir/%x-%j.err" --open-mode=append --export=ALL)
[[ -n "${FLOWJEPA_ACCOUNT:-}" ]] && cmd+=("--account=$FLOWJEPA_ACCOUNT")
source "$(dirname "${BASH_SOURCE[0]}")/circe_eval_args.sh"
flowjepa_build_eval_args
printf 'Mode=%s: one GPU, 8 CPUs, 64G RAM, time=%s, conditions=default/red_t/salt_pepper\n' "$mode" "$time_limit" >&2
printf 'Evaluation command:' >&2; printf ' %q' "$FLOWJEPA_PYTHON" "${evaluation_args[@]}" >&2; printf '\n' >&2
printf 'Submission:' >&2; printf ' %q' "${cmd[@]}" "$FLOWJEPA_REPO/scripts/circe_eval.sbatch" >&2; printf '\n' >&2
[[ "$action" = --preview ]] && exit 0
mkdir -p "$log_dir" "$eval_root"
if [[ "$action" = --check ]]; then
    "${cmd[@]}" --test-only "$FLOWJEPA_REPO/scripts/circe_eval.sbatch"
else
    "${cmd[@]}" "$FLOWJEPA_REPO/scripts/circe_eval.sbatch"
fi
