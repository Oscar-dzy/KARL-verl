set -x
ENGINE=${1:-vllm}

export VLLM_ALLREDUCE_USE_SYMM_MEM=0 # for vllm0.11.0 with TP


HF_MODEL_PATH=${HF_MODEL_PATH:-"/path/to/your/sft_model_checkpoint"}

SAVE_PATH=/path/to/your/save_model

train_path='[ "./data/grpo/verl-grpo-aircraft.parquet", "./data/grpo/verl-grpo-bird.parquet", "./data/grpo/verl-grpo-car.parquet", "./data/grpo/verl-grpo-food.parquet", "./data/grpo/verl-grpo-reptilia.parquet" ]'
val_path="./data/grpo/val_dataset.parquet"  # not used in training, only for running the training process
reward_path=/path/to/your/verl/verl/utils/reward_score/kvg_reward-knowledge.py  # your custom reward function path in verl framework



python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_path" \
    data.val_files="$val_path" \
    data.train_batch_size=8 \
    data.max_prompt_length=4800 \
    data.max_response_length=1500 \
    data.filter_overlong_prompts=False \
    data.filter_overlong_prompts_workers=8 \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path=$reward_path \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_kvg_grpo' \
    trainer.experiment_name='qwen3_vl_8b_fsdp' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=514 \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.total_epochs=2 $@
