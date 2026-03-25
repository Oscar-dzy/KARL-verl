# KARL-verl
This is the **verl** framework modified based on the KARL framework.



## Preparation

1. Environment Setup

   ```bash
   conda create -n verl-KARL python==3.12.0
   conda activate verl-KARL
   
   pip install -r requirements.txt
   pip install flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
   ```

2. Update the following variables in `./examples/grpo_trainer/run_qwen3_vl-8b-FSDP-kvg-knowledge.sh`:

   - Adjust `train_path` and `val_path` to point to the respective GRPO data paths.

   - Set `HF_MODEL_PATH` to the path of the CoT-SFT trained model.

   - Set `SAVE_PATH` to the output directory for the GRPO-trained model.

   The configuration should look like this:

   ```sh
   HF_MODEL_PATH=${HF_MODEL_PATH:-"/path/to/your/sft_model_checkpoint"}
   SAVE_PATH=/path/to/your/save_model
   
   train_path='[
       "/path/to/your/data_grpo/verl-grpo-aircraft.parquet",
       "/path/to/your/data_grpo/verl-grpo-bird.parquet",
       "/path/to/your/data_grpo/verl-grpo-car.parquet",
       "/path/to/your/data_grpo/verl-grpo-food.parquet",
       "/path/to/your/data_grpo/verl-grpo-reptilia.parquet"
   ]'
   
   val_path="/path/to/your/data_grpo/val_dataset.parquet"

3. Adjust the `--local_dir` and `--target_dir` variables in `./run.sh` to point to the corresponding paths where the model weights are saved.

4. Adjust `visual_knowledge_path ` variable (line 32) in `./verl/utils/reward_score/kvg_reward-knowledge.py `to point to the corresponding paths



## Train 

Once the above preparations are complete, simply run the following command to start training:

```bash
bash run.sh
```



## Reward Design

The knowledge-based reward function is implemented in `./verl/utils/reward_score/kvg_reward-knowledge.py`, which contains the KARL knowledge reward logic. You may also modify it based on your own needs.
