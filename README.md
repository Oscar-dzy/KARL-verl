# KARL-verl
This is the **verl** framework modified based on the KARL framework.



## Preparation

1. After downloading the GRPO data, modify the data paths in the script `./examples/grpo_trainer/run_qwen3_vl-8b-FSDP-kvg-knowledge.sh`, i.e., update the `train_path` and `val_path` variables as shown below:

   ```sh
   ...
   train_path='[ "./data/grpo/verl-grpo-aircraft.parquet", "./data/grpo/verl-grpo-bird.parquet", "./data/grpo/verl-grpo-car.parquet", "./data/grpo/verl-grpo-food.parquet", "./data/grpo/verl-grpo-reptilia.parquet" ]'
   val_path="./data/grpo/val_dataset.parquet"
   ...
   ```

2. Modify the variable named `HF_MODEL_PATH` to update the path of the model saved after training in the CoT-SFT stage in the script `./examples/grpo_trainer/run_qwen3_vl-8b-FSDP-kvg-knowledge.sh`





## Train 



```bash
bash run.sh
```

