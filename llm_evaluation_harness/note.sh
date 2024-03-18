# sudo mount /dev/nvme0n1p1 /mnt/tspdisk/

sudo fdisk /dev/nvme0n1 
# n -> p -> 1 -> enter -> enter -> w
sudo mkfs.ext4 /dev/nvme0n1p1

mkdir $HOME/tspdisk/
sudo mount -t ext4 /dev/nvme0n1p1 $HOME/tspdisk/


# sudo mount -o uid=user_id,ro /dev/nvme0n1p1 $HOME/tspdisk/

sudo chmod 777 $HOME/tspdisk/

# sudo chown azureuser:azureuser $HOME/tspdisk/


echo 'export HF_HOME="/home/azureuser/tspdisk/hf_cache"' >> ~/.bashrc

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
/bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda

# init conda
$HOME/miniconda/bin/conda init bash
source ~/.bashrc    

conda create -n tsp python=3.11 -y
conda activate tsp

cd $HOME

git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout v0.4.2
pip install -e .

cd $HOME
git clone https://github.com/tony92151/LLM-Blender-harness.git
cd LLM-Blender-harness
git checkout add-llm-evaluation-harness-support
pip install git+https://github.com/yuchenlin/LLM-Blender.git einops


cd $HOME
touch lm-evaluation-harness/lm_eval/models/llm_blender.py
cat LLM-Blender-harness/llm_evaluation_harness/harness_custom.py > lm-evaluation-harness/lm_eval/models/llm_blender.py

# edit line 13 of lm-evaluation-harness/lm_eval/models/llm_blender.py
sed -i '13s/.*/sys.path.append('$HOME/LLM-Blender-harness')/' lm-evaluation-harness/lm_eval/models/llm_blender.py
echo "from . import llm_blender" >> lm-evaluation-harness/lm_eval/models/__init__.py


# install vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.3.3
pip install -e .

# transformers
pip install transformers==4.33.2

## Download and init the model
cd LLM-Blender-harness
python -m llm_evaluation_harness.tsp_pipe

python -m lm_eval \
    --model llm_blender \
    --tasks mmlu_abstract_algebra \
    --device cuda:0 \
    --batch_size 4 \
    --output_path output/LlmBlender \
    --limit 10 \
    --log_samples


lm_eval \
    --model llm_blender \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8 \
    --log_samples \
    --output_path output/gpt-j-6B


lm_eval --model llm_blender \
    --tasks openbookqa \
    --device cuda:0

lm_eval --model hf \
    --model_args pretrained=mistralai/Mistral-7B-Instruct-v0.1 \
    --tasks mmlu_flan_n_shot_generative_stem \
    --device cuda:0 \
    --num_fewshot 5
    --batch_size 8 \
    --output_path output/Mistral


lm_eval --model llm_blender \
    --tasks mmlu_flan_n_shot_generative \
    --num_fewshot 5 \
    --device cuda:0 \
    --batch_size 64 \
    --output_path $HOME/tspdisk/output/llm_blender