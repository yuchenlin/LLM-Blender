#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --job-name=fuse_infer_3b
#SBATCH --output ../../jobs/%j.out
#SBATCH --gres=gpu:a6000:1

model_path="yuchenlin/gen_fuser" # yuchenlin/gen_fuser_3500
model_name="gen_fuser_beam4"
cd ../../
mkdir -p data/mix_128/fuse_gen/predictions/test/${model_name}/

CUDA_VISIBLE_DEVICES=0 python src/fusion_module/fuse_infer.py \
    --model_path $model_path --model_name $model_name \
    --start_index 0 \
    --end_index 5000 \
    --data_path data/mix_128/fuse_gen/test/top3_deberta-bartscore.jsonl \
    --output_file data/mix_128/fuse_gen/predictions/test/${model_name}/top3_deberta-bartscore.output.jsonl \
    --beam_size 4

# CUDA_VISIBLE_DEVICES=1 python src/fusion_module/fuse_infer.py \
#     --start_index 1250 \
#     --end_index 2500 \
#     --data_path data/fuse_gen/test/top5_bertscore.jsonl \
#     --output_file data/fuse_gen/predictions/${model_name}/top5_bertscore.output.1250-2500.jsonl &

# CUDA_VISIBLE_DEVICES=2 python src/fusion_module/fuse_infer.py \
#     --start_index 2500 \
#     --end_index 3750 \
#     --data_path data/fuse_gen/test/top5_bertscore.jsonl \
#     --output_file data/fuse_gen/predictions/${model_name}/top5_bertscore.output.2500-3750.jsonl &

# CUDA_VISIBLE_DEVICES=3 python src/fusion_module/fuse_infer.py \
#     --start_index 3750 \
#     --end_index 5000 \
#     --data_path data/fuse_gen/test/top5_bertscore.jsonl \
#     --output_file data/fuse_gen/predictions/${model_name}/top5_bertscore.output.3750-5000.jsonl &