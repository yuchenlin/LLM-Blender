import argparse  
from model_utils import EncDecModelManager
import json 
from tqdm import tqdm 

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--model_type', default="seq2seq", type=str, help='seq2seq or clm')
    parser.add_argument('--model_path', default="yuchenlin/gen_fuser", type=str, help='model path')
    parser.add_argument('--model_name', default="gf_0529", type=str, help='model name')
    parser.add_argument('--model_cache_dir', default='none', type=str, help='model name')
    parser.add_argument('--data_path', default="data/fuse_gen/test/top5_bertscore.jsonl", type=str, help='data path')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--batch_size',default=32, type=int, help='batch size')
    parser.add_argument('--beam_size',default=1, type=int, help='beam size')
    parser.add_argument('--output_file',default="", type=str, help='')
    # parser.add_argument('--skip_existing_files', action="store_true", help='')
    parser.add_argument('--start_index', default=0, type=int, help='')
    parser.add_argument('--end_index', default=-1, type=int, help='')
    parser.add_argument('--num_outputs',default=1, type=int, help='number of the sampled generations')
    parser.add_argument('--max_output_tokens',default=128, type=int, help='number of the sampled generations')
    return parser.parse_args()

args = parse_args()
mm = EncDecModelManager(args.model_path, args.model_name, args.model_cache_dir)
mm.load_model()

data = []
with open(args.data_path) as f:
    for line in f.read().splitlines():
        data.append(json.loads(line))

input_texts = [d['input'] for d in data]
output_texts = []

if args.end_index < 0:
    end_index = len(input_texts)
else:
    end_index = min(args.end_index, len(input_texts))

for i in tqdm(range(args.start_index, end_index, args.batch_size), ncols=100):
    batch = input_texts[i:min(i+args.batch_size, end_index)] # fix the bug that might generate the tail examples
    decoded_outputs = mm.infer_generate(batch, args) 
    output_texts += decoded_outputs

with open(args.output_file, 'w') as f:
    for i, o in zip(input_texts[args.start_index:end_index], output_texts): # get the right input for each output
        f.write(json.dumps({'input':i, 'output':o, 'output_source': args.model_name})+"\n")

"""
model_path="yuchenlin/gen_fuser"
model_name="gen-fuser-3b"
mkdir -p data/fuse_gen/predictions/${model_name}/

CUDA_VISIBLE_DEVICES=0 python src/fusion_module/fuse_infer.py \
    --model_path $model_path --model_name $model_name \
    --start_index 0 \
    --end_index 625 \
    --data_path data/fuse_gen/val/top3_deberta-bartscore-test.clean.jsonl \
    --output_file data/fuse_gen/predictions/${model_name}/top5_bertscore.output.0-625.jsonl &

CUDA_VISIBLE_DEVICES=1 python src/fusion_module/fuse_infer.py \
    --model_path $model_path --model_name $model_name \
    --start_index 625 \
    --end_index 1250 \
    --data_path data/fuse_gen/val/top3_deberta-bartscore-test.clean.jsonl \
    --output_file data/fuse_gen/predictions/${model_name}/top5_bertscore.output.625-1250.jsonl &

CUDA_VISIBLE_DEVICES=2 python src/fusion_module/fuse_infer.py \
    --model_path $model_path --model_name $model_name \
    --start_index 1250 \
    --end_index 1875 \
    --data_path data/fuse_gen/val/top3_deberta-bartscore-test.clean.jsonl \
    --output_file data/fuse_gen/predictions/${model_name}/top5_bertscore.output.1250-1875.jsonl &

CUDA_VISIBLE_DEVICES=3 python src/fusion_module/fuse_infer.py \
    --model_path $model_path --model_name $model_name \
    --start_index 1875 \
    --end_index 2500 \
    --data_path data/fuse_gen/val/top3_deberta-bartscore-test.clean.jsonl \
    --output_file data/fuse_gen/predictions/${model_name}/top5_bertscore.output.1875-2500.jsonl &

CUDA_VISIBLE_DEVICES=4 python src/fusion_module/fuse_infer.py \
    --model_path $model_path --model_name $model_name \
    --start_index 2500 \
    --end_index 3125 \
    --data_path data/fuse_gen/val/top3_deberta-bartscore-test.clean.jsonl \
    --output_file data/fuse_gen/predictions/${model_name}/top5_bertscore.output.2500-3125.jsonl & 

CUDA_VISIBLE_DEVICES=5 python src/fusion_module/fuse_infer.py \
    --model_path $model_path --model_name $model_name \
    --start_index 3125 \
    --end_index 3750 \
    --data_path data/fuse_gen/val/top3_deberta-bartscore-test.clean.jsonl \
    --output_file data/fuse_gen/predictions/${model_name}/top5_bertscore.output.3125-3750.jsonl & 

CUDA_VISIBLE_DEVICES=6 python src/fusion_module/fuse_infer.py \
    --model_path $model_path --model_name $model_name \
    --start_index 3750 \
    --end_index 4375 \
    --data_path data/fuse_gen/val/top3_deberta-bartscore-test.clean.jsonl \
    --output_file data/fuse_gen/predictions/${model_name}/top5_bertscore.output.3750-4375.jsonl & 

CUDA_VISIBLE_DEVICES=7 python src/fusion_module/fuse_infer.py \
    --model_path $model_path --model_name $model_name \
    --start_index 4375 \
    --end_index 5000 \
    --data_path data/fuse_gen/val/top3_deberta-bartscore-test.clean.jsonl \
    --output_file data/fuse_gen/predictions/${model_name}/top5_bertscore.output.4375-5000.jsonl & 
"""

