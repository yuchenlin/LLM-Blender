 #!/bin/bash

# An example script to demonstrate fine-tuning a PairRM checkpoint to a toy
# dataset.

dataset="mini_preferences"
eval_dataset="mini_preferences"
backbone_type="deberta"
backbone_name="microsoft/deberta-v3-large"
n_gpu=1

learning_rate=1e-5
num_train_epochs=15
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=1
source_maxlength=1224
candidate_maxlength=412

LAUNCH_CMD="deepspeed --num_gpus ${n_gpu}"

train_data_path="./data/${dataset}/all_train.json"
dev_data_path="./data/${eval_dataset}/all_test_items.json"
test_data_path="./data/${eval_dataset}/all_test_items.json"

# Run training
${LAUNCH_CMD} \
python3 -m llm_blender.train_ranker \
    --ranker_type "pairranker" \
    --model_type ${backbone_type} \
    --model_name ${backbone_name} \
    --run_name "finetune_mini_preferences" \
    --train_data_path ${train_data_path} \
    --eval_data_path ${dev_data_path} \
    --test_data_path ${test_data_path} \
    --using_metrics "human_preference" \
    --learning_rate ${learning_rate} \
    --source_maxlength ${source_maxlength} \
    --candidate_maxlength ${candidate_maxlength} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --num_train_epochs ${num_train_epochs} \
    --do_train True \
    --do_eval True \
    --do_predict True \
    --load_checkpoint "hf:llm-blender/PairRM" \
    --num_pos 5 \
    --num_neg 5 \
    --loss_type "instructgpt" \
    --sub_sampling_mode "all_pair" \
    --overwrite_output_dir True \
    --deepspeed "./zero_configs/zero3.json"
