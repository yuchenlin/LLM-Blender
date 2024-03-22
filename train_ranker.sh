#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name=bash
#SBATCH --output ./jobs/train_ranker/%j.out
#SBATCH --gres=gpu:4
#SBATCH -p a100
#SBATCH --mem=200G
#SBATCH -c 10
#SBATCH --qos=a100_wenhuchen

# module load cuda-11.8
nvidia-smi
# <== MODIFY THE FOLLOWING PARAMETERS ==>
dataset="UnifiedFeedback"
eval_dataset="reward_bench"
backbone_type="phi" # "deberta" or "roberta"
backbone_name="microsoft/phi-2" # "microsoft/deberta-v3-large" or "roberta-large"
n_gpu=4
ranker="PairRanker" # "PairRanker" or "Summareranker" or "SimCLS"
candidate_model="" # separted by comma. Empty string for all models
candidate_decoding_method="" # separted by comma. Empty string for all methods
n_candidates=-1 # number of candidates to generate
learning_rate=1e-5
num_train_epochs=5 
max_grad_norm=10e10 # set a large value to disable gradient clipping
fp16=True # whether to use fp16

max_train_data_size=-1 # -1 means no limit
max_eval_data_size=-1 # -1 means no limit
max_predict_data_size=-1 # -1 means no limit
do_inference=False # whether do inference instead of training, i.e. do test
# for inference, sometimes you want to use a checkpoint trained on another dataset
# to do inference on a dataset, you can set the checkpoint_trained_dataset to the dataset
# by default, it is set to the dataset you are doing inference on
checkpoint_trained_dataset=""
run_name_postfix="" # add a postfix to the run_name
# LAUNCH_CMD="torchrun \
# --rdzv_backend=c10d \
# --rdzv_endpoint="localhost:${localhost}" \
# --nnodes 1 \
# --nproc_per_node ${n_gpu} "

LAUNCH_CMD="deepspeed --num_gpus ${n_gpu}"

# set the dataset specific parameters below
if [[ $dataset =~ "mixinstruct" ]]; then
    echo "Using mixinstruct general datasets"
    source_maxlength=128
    candidate_maxlength=128
    per_device_train_batch_size=4
    per_device_eval_batch_size=8
    gradient_accumulation_steps=16
    using_metrics="bartscore"

elif [[ $dataset =~ "self_instruct" ]]; then
    echo "Using self_instruct user oriented datasets"
    source_maxlength=128
    candidate_maxlength=128
    per_device_train_batch_size=4
    per_device_eval_batch_size=4
    gradient_accumulation_steps=16
    using_metrics="bartscore"

elif [[ $dataset =~ "open_instruct" ]]; then
    echo "Using open_instruct user oriented datasets"
    source_maxlength=192
    candidate_maxlength=416
    per_device_train_batch_size=4
    per_device_eval_batch_size=2
    gradient_accumulation_steps=16
    using_metrics="comb_rate"

elif [[ $dataset =~ "reward_model" ]]; then
    echo "Using reward_model user oriented datasets"
    source_maxlength=1224
    candidate_maxlength=412
    per_device_train_batch_size=2
    per_device_eval_batch_size=1
    gradient_accumulation_steps=8
    using_metrics="human_preference"

elif [[ $dataset =~ "unified_feedback" ]]; then
    echo "Using unified_feedback user oriented datasets"
    source_maxlength=1224
    candidate_maxlength=412
    per_device_train_batch_size=4
    per_device_eval_batch_size=1
    gradient_accumulation_steps=4
    using_metrics="human_preference"

elif [[ $dataset =~ "UnifiedFeedback" ]]; then
    echo "Using unified_feedback user oriented datasets"
    source_maxlength=1224
    candidate_maxlength=412
    per_device_train_batch_size=1
    per_device_eval_batch_size=1
    gradient_accumulation_steps=16
    using_metrics="human_preference"

elif [[ $dataset =~ "pairrm_2.7b" ]]; then
    echo "Using unified_feedback user oriented datasets"
    source_maxlength=1224
    candidate_maxlength=412
    per_device_train_batch_size=1
    per_device_eval_batch_size=1
    gradient_accumulation_steps=8
    using_metrics="human_preference"

else
    echo "Unknown dataset: ${dataset}"
    echo "Please set the dataset specific parameters in the script"
    exit 1
fi

# <== Less likely to modify the following parameters ==>
localhost=$RANDOM # random port number
train_data_path="./data/${dataset}/all_train.json"
dev_data_path="./data/${eval_dataset}/all_test_items.json"
test_data_path="./data/${eval_dataset}/all_test_items.json"
if [ ! -f $test_data_path ]; then
    test_data_path=$dev_data_path
fi

if [[ $ranker = "PairRanker" ]]; then
    echo "Using PairRanker"
    ranker_type="pairranker"
    if [ $do_inference = "True" ]; then
        inference_mode="bubble" # do full for inference for its better performance
        if [ $inference_mode = "full" ]; then
            run_name="test_${dataset}_${ranker}_full_comparison"
        elif [ $inference_mode = "bubble" ]; then
            run_name="test_${dataset}_${ranker}_bubble_comparison"
        fi
        do_train="False"
        do_eval="False"
        do_test="True"
        # load_checkpoint="./outputs/${ranker_type}/${backbone_name}/train_${checkpoint_trained_dataset}_${ranker}${run_name_postfix}/checkpoint-best"
        load_checkpoint="checkpoint-best"
    else
        inference_mode="bubble" # do bubble for inference for its faster speed
        run_name="train_${dataset}_${ranker}"
        do_train="True"
        do_eval="True"
        do_test="True"
        load_checkpoint="" # no need to load checkpoint for training
    fi

    run_name="${run_name}${run_name_postfix}"

    ${LAUNCH_CMD} \
    train_ranker.py \
        --ranker_type ${ranker_type} \
        --model_type ${backbone_type} \
        --model_name ${backbone_name} \
        --run_name ${run_name} \
        --train_data_path ${train_data_path} \
        --eval_data_path ${dev_data_path} \
        --test_data_path ${test_data_path} \
        --n_candidates ${n_candidates} \
        --candidate_model "${candidate_model}" \
        --candidate_decoding_method "${candidate_decoding_method}" \
        --using_metrics ${using_metrics} \
        --learning_rate ${learning_rate} \
        --source_maxlength ${source_maxlength} \
        --candidate_maxlength ${candidate_maxlength} \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --num_train_epochs ${num_train_epochs} \
        --do_train ${do_train} \
        --do_eval ${do_eval} \
        --do_predict ${do_test} \
        --inference_mode ${inference_mode} \
        --load_checkpoint "${load_checkpoint}" \
        --max_train_data_size ${max_train_data_size} \
        --max_eval_data_size ${max_eval_data_size} \
        --max_predict_data_size ${max_predict_data_size} \
        --max_grad_norm ${max_grad_norm} \
        --fp16 ${fp16} \
        --num_pos 5 \
        --num_neg 5 \
        --loss_type "instructgpt" \
        --sub_sampling_mode "all_pair" \
        --overwrite_output_dir True \
        --deepspeed "./zero_configs/zero3.json" \


elif [[ $ranker = "Summareranker" ]]; then
    echo "Using Summareranker"
    ranker_type="summareranker"
    if [ $do_inference = "True" ]; then
        run_name="debug_${dataset}_${ranker}"
        do_train="False"
        do_eval="False"
        do_test="True"
        load_checkpoint="./outputs/${ranker_type}/${backbone_name}/train_${checkpoint_trained_dataset}_${ranker}${run_name_postfix}/checkpoint-best"
    else
        run_name="train_${dataset}_${ranker}"
        do_train="True"
        do_eval="True"
        do_test="True"
        load_checkpoint="" # no need to load checkpoint for training
    fi

    run_name="${run_name}${run_name_postfix}"

    ${LAUNCH_CMD} \
    train_ranker.py \
        --ranker_type ${ranker_type} \
        --model_type ${backbone_type} \
        --model_name ${backbone_name} \
        --run_name ${run_name} \
        --train_data_path ${train_data_path} \
        --eval_data_path ${dev_data_path} \
        --test_data_path ${test_data_path} \
        --n_candidates ${n_candidates} \
        --candidate_model "${candidate_model}" \
        --candidate_decoding_method "${candidate_decoding_method}" \
        --using_metrics ${using_metrics} \
        --learning_rate ${learning_rate} \
        --source_maxlength ${source_maxlength} \
        --candidate_maxlength ${candidate_maxlength} \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --num_train_epochs ${num_train_epochs} \
        --do_train ${do_train} \
        --do_eval ${do_eval} \
        --do_predict ${do_test} \
        --load_checkpoint "${load_checkpoint}" \
        --max_train_data_size ${max_train_data_size} \
        --max_eval_data_size ${max_eval_data_size} \
        --max_predict_data_size ${max_predict_data_size} \
        --max_grad_norm ${max_grad_norm} \
        --fp16 ${fp16} \
        --num_pos 1 \
        --num_neg 1 \
        --loss_type "MoE_BCE" \
        --sub_sampling_mode "top_bottom" \
        --overwrite_output_dir True \

elif [[ $ranker = "SimCLS" ]]; then
    echo "Using SimCLS"
    ranker_type="dual"
    if [ $do_inference = "True" ]; then
        run_name="debug_${dataset}_${ranker}"
        do_train="False"
        do_eval="False"
        do_test="True"
        load_checkpoint="./outputs/${ranker_type}/${backbone_name}/train_${checkpoint_trained_dataset}_${ranker}${run_name_postfix}/checkpoint-best"
    else
        run_name="train_${dataset}_${ranker}"
        do_train="True"
        do_eval="True"
        do_test="True"
        load_checkpoint="" # no need to load checkpoint for training
    fi

    run_name="${run_name}${run_name_postfix}"

    ${LAUNCH_CMD} \
    train_ranker.py \
        --ranker_type ${ranker_type} \
        --model_type ${backbone_type} \
        --model_name ${backbone_name} \
        --run_name ${run_name} \
        --train_data_path ${train_data_path} \
        --eval_data_path ${dev_data_path} \
        --test_data_path ${test_data_path} \
        --n_candidates ${n_candidates} \
        --candidate_model "${candidate_model}" \
        --candidate_decoding_method "${candidate_decoding_method}" \
        --using_metrics ${using_metrics} \
        --learning_rate ${learning_rate} \
        --source_maxlength ${source_maxlength} \
        --candidate_maxlength ${candidate_maxlength} \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --num_train_epochs ${num_train_epochs} \
        --do_train ${do_train} \
        --do_eval ${do_eval} \
        --do_predict ${do_test} \
        --load_checkpoint "${load_checkpoint}" \
        --max_train_data_size ${max_train_data_size} \
        --max_eval_data_size ${max_eval_data_size} \
        --max_predict_data_size ${max_predict_data_size} \
        --max_grad_norm ${max_grad_norm} \
        --fp16 ${fp16} \
        --loss_type "simcls" \
        --sub_sampling_mode "uniform" \
        --sub_sampling_ratio 0.3 \
        --overwrite_output_dir True \

else
    echo "Unknown ranker: ${ranker}"
fi
