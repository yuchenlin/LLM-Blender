# LLM-ranker
<!-- 
<p align="center" width="50%">
<img src="./assets/intro.png" alt="LLM-Blender" style="width: 50%; min-width: 300px; display: block; margin: auto;">
</p> -->

An Innovative ensembling framework to attain consistently superior performance by leveraging the diverse strengths and weaknesses of multiple open-source large language models (LLMs). 

## Resources

1. [Project website](http://yuchenlin.xyz/LLM-Blender/)
2. [Checkpoints](https://huggingface.co/llm-blender) & [Dataset](https://huggingface.co/datasets/llm-blender/mix-instruct) on HuggingFace🤗
3. [Paper Link](https://arxiv.org/abs/2212.10555)

## Overview

We introduce LLM-Blender, an innovative ensembling framework to attain consistently superior performance by leveraging the diverse strengths and weaknesses of multiple open-source large language models (LLMs). LLM-Blender cut the weaknesses through ranking and integrate the strengths through fusing generation to enhance the capability of LLMs.

<!-- <p align="center" width="100%">
<img src="./assets/intro.png" alt="LLM-Blender" style="width: 50%; min-width: 300px; display: block; margin: auto;">
</p> -->

![LLM-BLender](./assets/llm_blender.png)
1. improve the quality of LLM generated candidates
2. serve as a reward model used for RLHF

## Installation

First follow install pytorch that fits your local GPU cuda version.
```bash
pip install -r requirements.txt
```

## Training

The training scipt is integrated into a single bash. `./scripts/train_ranker`
Please follow the guide in the script to train the ranker.

Here are some explanations for the script parameters:

**Changing the torchrun cmd**
```bash
TORCHRUN_CMD=<you torchrun cmd path>
```
Normally, it's just `torchrun` with proper conda env activated.

**Changing the dataset**

```bash
dataset="<your dataset>`
```

**Changing the ranker backbone**

```bash
backbone_type="deberta" # "deberta" or "roberta"
backbone_name="microsoft/deberta-v3-large" # "microsoft/deberta-v3-large" or "roberta-large"
```

**Changing the ranker type**
```bash
ranker="Pairranker" # "PairRanker" or "Summaranker" or "SimCLS"
```

**Filter the candidates used**
```bash
candidate_model="flan-t5-xxl" # or "alpaca-native"
candidate_decoding_method="top_p_sampling" 
n_candidates=15 # number of candidates to generate
using_metrics="rouge1,rouge2,rougeLsum,bleu" # metrics used to train the signal
```
**Do Training or Inference**
```bash
do_inference=False # training
do_inference=True # inference
```
When doing inference, you can change `inference_mode` to `bubble` or `full` to select difference pairwise inference model

**Limit the datasize used for training, dev and test**
```bash
max_train_data_size=-1 # -1 means no limit
max_eval_data_size=-1 # -1 means no limit
max_predict_data_size=-1 # -1 means no limit
```
**Do inference on dataset A with ranker training on dataset B**
```bash
dataset=<A>
checkpoint_trained_dataset=<B>
do_inference=True
```

## Note

When using flan-t5 as the backbone, set `fp16=False` in case of overflow.
