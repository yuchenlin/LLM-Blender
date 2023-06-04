USE_TF=0

CUDA_VISIBLE_DEVICES=0,1,2,3,4 deepspeed --master_port 29511 \
	./ds_train.py \
	--cache_dir /net/nfs/mosaic/yuchenl/cache/ \
    --model_name_or_path /net/nfs/mosaic/yuchenl/models/llm_blender/llm_blender_xl/checkpoint-3500/ \
    --output_dir /net/nfs/mosaic/yuchenl/models/llm_blender/fuser_xl_prbar_0529/  \
    --do_train \
	--do_eval \
	--save_total_limit=10 \
	--train_file "../../data/fuse_gen/train/top3_deberta-bartscore.clean.jsonl" \
 	--validation_file "../../data/fuse_gen/val/top3_deberta-bartscore-test.clean.jsonl" \
	--predict_with_generate 0 \
    --learning_rate 5e-5 \
	--adam_eps 1e-06 \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 128 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 16 \
	--metric_for_best_model eval_loss \
	--greater_is_better=False \
	--deepspeed zero_2_bf16.json \
	--gradient_accumulation_steps 4 \
	--num_train_epochs 15 \
	--logging_steps 1 \
	--load_best_model_at_end=True \
	--save_strategy=steps \
	--evaluation_strategy=steps \
	--save_steps 50 \
	--eval_steps 50 \
	--seed 42 \
	--report_to wandb \
	--run_name fuser_xl_prbar_0529

# cd /net/nfs/mosaic/yuchenl/models/llm_blender
# watch -n 600 'rm */*/global*/*_states.pt'

# python -c 'from transformers import AutoModel; \
# from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
# model = AutoModel.from_pretrained("google/flan-t5-xl"); \
# estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)'
