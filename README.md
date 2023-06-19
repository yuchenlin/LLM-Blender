# LLM-Blender: Ensembling LLMs with Pairwise Ranking & Generative Fusion [ACL2023]

<div style="width:40% float:center diaplay:inline">
     <img src=./docs/logo-ai2.svg width=35%/> &nbsp; &nbsp; <img src=./docs/logo-usc.png width=25%/>
</div>

<a target="_blank" href="https://arxiv.org/abs/2306.02561">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-black?style=flat&logo=arxiv">
</a><a target="_blank" href="https://github.com/yuchenlin/LLM-Blender">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-green?style=flat&logo=github">
</a><a target="_blank" href="https://huggingface.co/datasets/llm-blender/mix-instruct">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-red?style=flat"></a><a target="_blank" href="https://huggingface.co/llm-blender">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat">
</a><a target="_blank" href="https://twitter.com/billyuchenlin/status/1663603372220616704?s=20">
<img style="height:22pt" src="https://img.shields.io/badge/-Tweet-blue?style=flat&logo=twitter">
</a>
<br>

<span style="color:#183385; font-size: 14pt; font-family: Roboto, Helvetica, Arial, Heveltica Neue, sans-serif">
     <b>Authors:</b> <a class="name" target="_blank" href="https://jdf-prog.github.io/">Dongfu Jiang</a>, 
     <a class="name" target="_blank" href="http://ink-ron.usc.edu/xiangren/">Xiang Ren</a>,
     <a class="name" target="_blank" href="http://yuchenlin.xyz">Bill Yuchen Lin</a>&nbsp; @ 
     <a class="btna" target="_blank" href="https://mosaic.allenai.org">AI2-Mosaic</a> &nbsp; 
          <a class="btna" target="_blank" href="http://inklab.usc.edu/">USC-INK</a> &nbsp; 
     </span>

## Overview

![LLM-BLender](./docs/llm_blender.png)

<details><summary>Abstract</summary> 

- We introduce LLM-Blender, an innovative ensembling framework to attain consistently superior performance by leveraging the diverse strengths of multiple open-source large language models (LLMs). LLM-Blender cut the weaknesses through ranking and integrate the strengths through fusing generation to enhance the capability of LLMs.


- Our framework consists of two complementary modules: **PairRanker** and **GenFuser**, addressing the observation that optimal LLMs for different examples can significantly vary. **PairRanker** employs a specialized pairwise comparison method to distinguish subtle differences between candidate outputs. **GenFuser** aims to merge the top-ranked candidates from the aggregation of PairRanker's pairwise comparisons into an improved output by capitalizing on their strengths and mitigating their weaknesses.
- To facilitate large-scale evaluation, we introduce a benchmark dataset, [**MixInstruct**](#data_release), which is a mixture of multiple instruction datasets featuring oracle pairwise comparisons for testing purposes. Our **LLM-Blender** significantly surpasses the best LLMs and baseline ensembling methods across various metrics on **MixInstruct**, establishing a substantial performance gap.

</details>

## Usage

### Installation

```bash
git clone https://github.com/yuchenlin/LLM-Blender.git
cd LLM-Blender
pip install -e .
```
Then you are good to go through our LLM-Blender with `import llm_blender`.

### Rank and Fusion

- Please first download our DeBERTa-v3-large PairRanker checkpoint to your local folder: [***checkpoint link***](https://drive.google.com/file/d/1EpvFu_qYY0MaIu0BAAhK-sYKHVWtccWg/view?usp=sharing)

```python
import llm_blender
ranker_config = llm_blender.RankerConfig
ranker_config.ranker_type = "pairranker"
ranker_config.model_type = "deberta"
ranker_config.model_name = "microsoft/deberta-v3-large"
ranker_config.load_checkpoint = "<your checkpoint path>"
ranker_config.cache_dir = "./hf_models"
ranker_config.source_max_length = 128
ranker_config.candidate_max_length = 128
ranker_config.n_tasks = 1
fuser_config = llm_blender.GenFuserConfig
fuser_config.model_name = "llm-blender/gen_fuser_3b"
fuser_config.cache_dir = "./hf_models"
fuser_config.max_length = 512
fuser_config.candidate_max_length = 128
blender_config = llm_blender.BlenderConfig
blender_config.device = "cuda"
blender = llm_blender.Blender(blender_config, ranker_config, fuser_config)
```

- Then you can rank with the following function

```python
inputs = ["input1", "input2"]
candidates_texts = [["candidate1 for input1", "candidatefor input1"], ["candidate1 for input2", "candidate2 for input2"]]
ranks = blender.rank(inputs, candidates_texts, return_scores=False, batch_size=2)
# ranks is a list of ranks where ranks[i][j] represents the ranks of candidate-j for input-i
```

- You can fuse the top-ranked candidates with the following code

```python
from llm_blender.blender.blender_utils import get_topk_candidates_from_ranks
topk_candidates = get_topk_candidates_from_ranks(ranks, candidates_texts, top_k=3)
fuse_generations = blender.fuse(inputs, topk_candidates, batch_size=2)
# fuse_generations are the fused generations from our fine-tuned checkpoint
```

- You can also do the rank and fusion as a whole

```python
fuse_generations, ranks = blender.rank_and_fuse(inputs, candidates_texts, return_scores=False, batch_size=2, top_k=3)
```

- Using llm-blender to directly compare two candidates
```python
candidates_A = [cands[0] for cands in candidates]
candidates_B = [cands[1] for cands in candidates]
comparison_results = blender.compare(inputs, candidates_A, candidates_B)
# comparison_results is a list of bool, where element[i] denotes whether candidates_A[i] is better than candidates_B[i] for inputs[i]
```
- Check more details on our example jupyter notebook usage: [`blender_usage.ipynb`](./blender_usage.ipynb)



## Data Release

- To facilitate large-scale evaluation, we introduce a benchmark dataset, **MixInstruct**, which is a mixture of multiple instruction datasets featuring oracle pairwise comparisons for testing purposes. 
- MixInstruct is the first large-scale dataset consisting of responses from 11 popular open-source LLMs on the instruction-following dataset. Each split of train/val/test contains 100k/5k/5k examples. 
- MixInstruct instruct is collected from 4 famous instruction dataset: Alpaca-GPT4, Dolly-15k, GPT4All-LAION and ShareGPT. The ground-truth outputs comes from either ChatGPT, GPT-4 or human annotations.
- MixInstruct is evaluated by both auto-metrics including BLEURT, BARTScore, BERTScore, etc. and ChatGPT. We provide 4771 examples on test split that is evaluated by ChatGPT through pariwise comparison.
- Code to construct the dataset: [`get_mixinstruct.py`](./llm_blender/download_dataset/get_mixinstruct.py)
- HuggingFace 🤗 [Dataset link](https://huggingface.co/datasets/llm-blender/mix-instruct)

<div align="center"> <img src=./docs/intro.png width=70%/> </div>


## Training

See more details in [`train_ranker.sh`](./train_ranker.sh)

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




### Model checkpoints

- [PairRanker checkpoint](https://drive.google.com/file/d/1EpvFu_qYY0MaIu0BAAhK-sYKHVWtccWg/view?usp=sharing) fine-tuned on DeBERTa-v3-Large (304m)

- [GenFuser checkpoint](https://huggingface.co/llm-blender/gen_fuser_3b) fine-tuned on Flan-T5-XL (3b)
