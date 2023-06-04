USE_TF=0

# CUDA_VISIBLE_DEVICES=7 deepspeed --master_port 29515 \
CUDA_VISIBLE_DEVICES=0 python \
	./ds_train.py \
	--cache_dir /net/nfs/mosaic/yuchenl/cache/ \
    --model_name_or_path /net/nfs/mosaic/yuchenl/models/llm_blender/llm_blender_xl/checkpoint-3000/ \
    --output_dir /home/yuchenl/test/  \
	--do_eval \
 	--validation_file "../../data/fuse_gen/val/top3_deberta-bartscore-test.clean.jsonl" \
	--predict_with_generate 0 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --per_device_eval_batch_size 32
    
# # --train_file "../../data/fuse_gen/train/top3_deberta-bartscore.clean.jsonl" \
# cd /net/nfs/mosaic/yuchenl/models/llm_blender
# watch -n 600 'rm */*/global*/*_states.pt'

# python -c 'from transformers import AutoModel; \
# from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
# model = AutoModel.from_pretrained("google/flan-t5-xl"); \
# estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)'
