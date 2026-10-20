#!/bin/bash
flowjepa_build_eval_args() {
    evaluation_args=("$FLOWJEPA_REPO/eval_pusht_suite.py"
        "policy=$FLOWJEPA_EVAL_CHECKPOINT" "cache_dir=$FLOWJEPA_DATA_ROOT"
        "output.dir=$FLOWJEPA_EVAL_OUTPUT"
        "seed=${FLOWJEPA_EVAL_SEED:-42}"
        "hydra.run.dir=$FLOWJEPA_LOG_ROOT/$FLOWJEPA_EVAL_ID/hydra/${SLURM_JOB_ID:-JOB_ID}")
    if [[ "$FLOWJEPA_EVAL_MODE" = smoke ]]; then
        evaluation_args+=(eval.num_eval=1 eval.eval_budget=5 plan_config.horizon=1
            plan_config.receding_horizon=1 solver.num_samples=4 solver.topk=2 solver.n_steps=1)
    fi
}
