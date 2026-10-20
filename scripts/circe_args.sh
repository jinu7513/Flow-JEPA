#!/bin/bash
# Shared by command preview and the GPU batch job to avoid argument drift.
flowjepa_build_train_args() {
    training_args=("$FLOWJEPA_REPO/train.py" --config-name=pusht_circe
        "trainer.precision=${FLOWJEPA_PRECISION:-16-mixed}"
        "loader.batch_size=${FLOWJEPA_BATCH_SIZE:-8}"
        "num_workers=${FLOWJEPA_WORKERS:-6}"
        "output_model_name=$FLOWJEPA_RUN_ID" "subdir=$FLOWJEPA_RUN_ID"
        "hydra.run.dir=$FLOWJEPA_EXPERIMENT_DIR/hydra/${SLURM_JOB_ID:-JOB_ID}"
        "checkpoint.every_n_steps=${FLOWJEPA_CHECKPOINT_STEPS:-100}")
    if [[ "$FLOWJEPA_MODE" = smoke ]]; then
        training_args+=(trainer.max_epochs=1 +trainer.limit_train_batches=2
            +trainer.limit_val_batches=2 checkpoint.every_n_steps=1 wandb.enabled=false)
    elif [[ "${FLOWJEPA_WANDB:-0}" = 1 ]]; then
        : "${WANDB_ENTITY:?Set your W&B entity; keep API keys out of scripts}"
        local resume_policy=never
        [[ "$FLOWJEPA_MODE" = resume ]] && resume_policy=must
        training_args+=(wandb.enabled=true "wandb.config.entity=$WANDB_ENTITY"
            "wandb.config.id=$FLOWJEPA_RUN_ID" "wandb.config.name=$FLOWJEPA_RUN_ID"
            "wandb.config.resume=$resume_policy")
    else
        training_args+=(wandb.enabled=false)
    fi
    if [[ "$FLOWJEPA_MODE" = resume ]]; then
        training_args+=("checkpoint.resume=${FLOWJEPA_RESUME:?Set a trusted full-state .ckpt}")
    fi
}
