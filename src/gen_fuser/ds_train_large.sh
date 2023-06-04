USE_TF=0

CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --master_port 29513 \
	./ds_train.py \
	--cache_dir /net/nfs/mosaic/yuchenl/cache/ \
	--model_name_or_path google/flan-t5-large \
	--output_dir /net/nfs/mosaic/yuchenl/models/llm_blender/fuser_large_prbar_0527 \
	--do_train \
	--do_eval \
	--save_total_limit=10 \
	--train_file ../../data/fuse_gen/train/top3_deberta-bartscore.clean.jsonl \
	--validation_file ../../data/fuse_gen/val/top3_deberta-bartscore-test.mini.jsonl \
	--predict_with_generate 0 \
	--learning_rate 1e-4 \
	--adam_eps 1e-06 \
	--overwrite_output_dir \
	--max_source_length 1024 \
	--max_target_length 128 \
	--per_device_train_batch_size 16 \
	--per_device_eval_batch_size 32 \
	--metric_for_best_model eval_loss \
	--greater_is_better=False \
	--deepspeed zero_2_bf16.json \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 30 \
	--logging_steps 1 \
	--load_best_model_at_end=True \
	--save_strategy=steps \
	--evaluation_strategy=steps \
	--save_steps 500 \
	--eval_steps 500 \
	--seed 42 \
	--report_to wandb \
	--run_name fuser_large_prbar_0527
