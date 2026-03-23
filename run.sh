export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export WANDB_API_KEY=xxx  # your wandb api key
# export LOG_PATH=/path/to/your/log/file.log  # when you set DEBUG_MODE=true, you can set the log path here

export DEBUG_MODE=false
export HYDRA_FULL_ERROR=1
# export WANDB_MODE=offline

# GRPO
bash ./examples/grpo_trainer/run_qwen3_vl-8b-FSDP-kvg-knowledge.sh


# Merge FSDP checkpoints
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /path/to/your/save_model/global_step_xxx/actor \
    --target_dir /path/to/your/save_model/global_step_xxx/actor/huggingface