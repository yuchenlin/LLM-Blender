#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=gpt_cmp_eval
#SBATCH --output ../../jobs/%j.out
#SBATCH --qos=normal
#SBATCH -n 1

data_fir="../../data"
dataset="mixinstruct"
set="test"
postfix=""
input_file="${data_fir}/${dataset}/${set}_data_prepared${postfix}.json"
output_file="${data_fir}/${dataset}/${set}_data_prepared${postfix}_cmp_eval.json"

# Azure openai use engine parameter
python gpt_eval.py \
    --input_file ${input_file} \
    --output_file ${output_file} \
    --engine "ChatGPT" \
    --api_key "$OPENAI_API_KEY" \
    --num_threads 5 \

# OpenAI API use model parameter
python gpt_eval.py \
    --input_file ${input_file} \
    --output_file ${output_file} \
    --model "gpt-3.5-turbo" \
    --api_key "$OPENAI_API_KEY" \
    --num_threads 5 \



