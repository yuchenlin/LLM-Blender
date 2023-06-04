# LLM-Reranker

This projects aims to develop a general reranker (reward model) based on LLM for 2 purposes:

1. improve the quality of LLM generated candidates
2. serve as a reward model used for RLHF

## Installation

First follow install pytorch that fits your local GPU cuda version.
```bash
pip install -r requirements.txt
```

## Training

The training scipt is integrated into a single bash. `./scripts/train_reranker`
Please follow the guide in the script to train the reranker.

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

**Changing the reranker backbone**

```bash
backbone_type="deberta" # "deberta" or "roberta"
backbone_name="microsoft/deberta-v3-large" # "microsoft/deberta-v3-large" or "roberta-large"
```

**Changing the reranker type**
```bash
reranker="PairReranker" # "PairReranker" or "SummaReranker" or "SimCLS"
```

**Filter the candidates used**
```bash
candidate_model="flan-t5-xxl" # or "alpaca-native"
candidate_decoding_method="top_p_sampling" 
n_candidates=15 # number of candidates to generate
using_metrics="rouge1,rouge2,rougeLsum,bleu" # or other metrics
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
**Do inference on dataset A with reranker training on dataset B**
```bash
dataset=<A>
checkpoint_trained_dataset=<B>
do_inference=True
```

## Note

When using flan-t5 as the backbone, set `fp16=False` in case of overflow.
