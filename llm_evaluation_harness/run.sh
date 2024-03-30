# ARC
lm_eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-chat-hf \
    --tasks arc_challenge \
    --device cuda:0 \
    --batch_size 64 \
    --output_path $HOME/output/arc_challenge/hf_meta-llama/Llama-2-7b-chat-hf

lm_eval \
    --model hf \
    --model_args pretrained=openlm-research/open_llama_7b_v2 \
    --tasks arc_challenge \
    --device cuda:0 \
    --batch_size 64 \
    --output_path $HOME/output/arc_challenge/hf_open_llama_7b_v2

lm_eval \
    --model hf \
    --model_args pretrained=mosaicml/mpt-7b-chat \
    --tasks arc_challenge \
    --device cuda:0 \
    --batch_size 64 \
    --output_path $HOME/output/arc_challenge/hf_mpt-7b-chat


lm_eval --model llm_blender \
    --tasks arc_challenge \
    --device cuda:0 \
    --batch_size 128 \
    --output_path $HOME/output/arc_challenge/hf_llm_blender

###

lm_eval \
    --model hf \
    --model_args pretrained=openlm-research/open_llama_3b_v2 \
    --tasks mmlu_flan_n_shot_generative_stem \
    --num_fewshot 5 \
    --device cuda:0 \
    --batch_size 64 \
    --output_path $HOME/output/mmlu_flan_n_shot_generative_stem/hf_open_llama_3b_v2