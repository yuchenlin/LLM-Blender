USE_TF=0

# deepspeed --master_port 29510 \
# 		./ds_train.py \
# 		--cache_dir /net/nfs/mosaic/yuchenl/cache/ \
#         --model_name_or_path google/flan-t5-xxl \
#         --output_dir model_ckpts/flan_xl_fusion \
#         --do_train \
# 		--save_total_limit=10 \
#         --train_file ../../data/fuse_gen/train/top5_bertscore.jsonl \
# 		--validation_file ../../data/fuse_gen/val/top5_bertscore.mini.jsonl \
# 		--predict_with_generate 0 \
#         --learning_rate 1e-4 \
# 		--adam_eps 1e-06 \
#         --overwrite_output_dir \
#         --max_source_length 1024 \
#         --max_target_length 128 \
#         --per_device_train_batch_size 1 \
#         --per_device_eval_batch_size 1 \
# 	--deepspeed zero_2_bf16.json \
# 	--gradient_accumulation_steps 8 \
# 	--num_train_epochs 5 \
# 	--logging_steps 1 \
# 	--load_best_model_at_end=True \
# 	--save_steps 300 \
# 	--seed 42 \
# 	--report_to wandb \
# 	--run_name flan_xxl_fusion

# # 		--do_eval \
# # 	--eval_steps 300 \
# # --load_best_model_at_end=True \
# # 	--save_strategy=steps \
# # 	--evaluation_strategy=epochs \

# # --metric_for_best_model eval_loss \
# # 	--greater_is_better=False \
# # --eval_steps 1200000 \



deepspeed --master_port 29510 \
    ./ds_train.py \
    --cache_dir /net/nfs/mosaic/yuchenl/cache/ \
    --model_name_or_path google/flan-t5-xxl \
    --output_dir model_ckpts/flan_xl_fusion \
    --do_train \
    --save_total_limit=10 \
    --train_file ../../data/fuse_gen/train/top5_bertscore.jsonl \
    --predict_with_generate 0 \
    --learning_rate 1e-4 \
    --adam_eps 1e-06 \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 128 \
    --per_device_train_batch_size 1 \
    --deepspeed zero_2_bf16.json \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 5 \
    --logging_steps 1 \
    --save_steps 1000 \
    --seed 42 \
    --report_to wandb \
    --run_name flan_xxl_fusion
