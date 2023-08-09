#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=generate_candidates
#SBATCH --output ../../jobs/%j.out
#SBATCH --gres=gpu:a6000:1
#SBATCH --qos=normal
#SBATCH -n 1


# <===================== Generation for mixed using multiple models =====================>
dataset="mixinstruct_v2"
set="val"
prompt_max_length=256
output_max_length=256

cmd="sbatch"

model="chavinlo/alpaca-13b"
${cmd} _generate_candidates.sh "$dataset" "$set" "$model" "$prompt_max_length" "$output_max_length"

model="eachadea/vicuna-13b-1.1"
${cmd} _generate_candidates.sh "$dataset" "$set" "$model" "$prompt_max_length" "$output_max_length"

model="databricks/dolly-v2-12b" 
${cmd} _generate_candidates.sh "$dataset" "$set" "$model" "$prompt_max_length" "$output_max_length"

model="stabilityai/stablelm-tuned-alpha-7b" 
${cmd} _generate_candidates.sh "$dataset" "$set" "$model" "$prompt_max_length" "$output_max_length" 

model="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
${cmd} _generate_candidates.sh "$dataset" "$set" "$model" "$prompt_max_length" "$output_max_length"

model="TheBloke/koala-13B-HF" 
${cmd} _generate_candidates.sh "$dataset" "$set" "$model" "$prompt_max_length" "$output_max_length"

model="project-baize/baize-v2-13b"
${cmd} _generate_candidates.sh "$dataset" "$set" "$model" "$prompt_max_length" "$output_max_length"

model="google/flan-t5-xxl"
${cmd} _generate_candidates.sh "$dataset" "$set" "$model" "$prompt_max_length" "$output_max_length"

model="THUDM/chatglm-6b"
${cmd} _generate_candidates.sh "$dataset" "$set" "$model" "$prompt_max_length" "$output_max_length"

model="fnlp/moss-moon-003-sft"
${cmd} _generate_candidates.sh "$dataset" "$set" "$model" "$prompt_max_length" "$output_max_length"

model="mosaicml/mpt-7b-chat"
${cmd} _generate_candidates.sh "$dataset" "$set" "$model" "$prompt_max_length" "$output_max_length"

model="TheBloke/guanaco-13B-HF"
${cmd} _generate_candidates.sh "$dataset" "$set" "$model" "$prompt_max_length" "$output_max_length"

model="NousResearch/Nous-Hermes-13b"
${cmd} _generate_candidates.sh "$dataset" "$set" "$model" "$prompt_max_length" "$output_max_length"

model="ehartford/WizardLM-13B-Uncensored"
${cmd} _generate_candidates.sh "$dataset" "$set" "$model" "$prompt_max_length" "$output_max_length"

model="jondurbin/airoboros-13b"
${cmd} _generate_candidates.sh "$dataset" "$set" "$model" "$prompt_max_length" "$output_max_length"