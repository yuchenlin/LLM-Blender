
import os
import torch
import logging
import numpy as np
import safetensors
from pathlib import Path
from ..pair_ranker.model_util import (
    build_ranker,
    build_tokenizer,
    build_collator,
)
from ..pair_ranker.config import RankerConfig
from ..gen_fuser.config import GenFuserConfig
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)
from typing import List
from huggingface_hub import snapshot_download
from transformers.utils.hub import TRANSFORMERS_CACHE

def get_torch_dtype(dtype_str):
    """
        Get the torch dtype from a string
    """
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "int8":
        return torch.int8
    else:
        raise ValueError("Invalid dtype {}".format(dtype_str))

def load_other_ranker(ranker_config: RankerConfig):
    """Load Other Ranker (Reward Model) from config file
        Currently supporting: 
            - BERT series model, e.g. OpenAssistant/reward-model-deberta-v3-large-v2
    """
    model_name = ranker_config.model_name
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, cache_dir=ranker_config.cache_dir,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=ranker_config.cache_dir)
    collator = build_collator(
        "other",
        tokenizer,
        ranker_config.source_maxlength,
        ranker_config.candidate_maxlength,
    )
    return model, tokenizer, collator
    
    
def load_ranker(ranker_config: RankerConfig):
    """Load PairRanker model from config file"""
    tokenizer = build_tokenizer(ranker_config.model_name, cache_dir=ranker_config.cache_dir)
    collator = build_collator(ranker_config.ranker_type, tokenizer,
        ranker_config.source_maxlength, ranker_config.candidate_maxlength,
    )
    ranker = build_ranker(
        ranker_config.ranker_type,
        ranker_config.model_type,
        ranker_config.model_name,
        ranker_config.cache_dir,
        ranker_config,
        tokenizer,
    )
    if ranker_config.load_checkpoint is not None:
        # load checkpoint from local path
        load_checkpoint = Path(ranker_config.load_checkpoint)
        if load_checkpoint.name == "pytorch_model.bin":
            load_checkpoint = load_checkpoint.parent
        
        if (load_checkpoint/"pytorch_model.bin").exists():
            # pytorch_model.bin
            state_dict = torch.load(load_checkpoint/"pytorch_model.bin", map_location="cpu")
            load_result = ranker.load_state_dict(state_dict, strict=False)
            if load_result.missing_keys:
                logging.warning(f"Missing keys: {load_result.missing_keys}")
            else:
                logging.info(f"Successfully loaded checkpoint from '{load_checkpoint}'")
        elif (load_checkpoint/"model.safetensors").exists():
            # model.safetensors
            load_result = safetensors.torch.load_model(ranker, load_checkpoint/"model.safetensors")
            missing_keys, unexpected_keys = load_result
            if missing_keys:
                logging.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys: {unexpected_keys}")
            if not missing_keys and not unexpected_keys:
                logging.info(f"Successfully loaded checkpoint from '{load_checkpoint}'")
        else:
            raise ValueError(f"Cannot find pytorch_model.bin or model.safetensors in {load_checkpoint}")
        
    return ranker, tokenizer, collator

def get_topk_candidates_from_ranks(ranks:List[List[int]], candidates:List[List[str]], top_k:int):
    """Get top k candidates from a list of ranks"""
    ranks = np.array(ranks)
    sorted_idxs = np.argsort(ranks, axis=1)
    candidates = np.array(candidates)
    topk_candidates = candidates[np.arange(len(candidates))[:, None], sorted_idxs[:, :top_k]]
    return topk_candidates

def load_fuser(fuser_config: GenFuserConfig):
    model_name = fuser_config.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=fuser_config.cache_dir)
    if fuser_config.device == "cpu":
        fuser = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, cache_dir=fuser_config.cache_dir,
            device_map={"": "cpu"}, torch_dtype=get_torch_dtype(fuser_config.torch_dtype),
        )
    else:
        fuser = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, cache_dir=fuser_config.cache_dir,
            device_map="auto", torch_dtype=get_torch_dtype(fuser_config.torch_dtype),
            load_in_4bit=fuser_config.load_in_4bit, load_in_8bit=fuser_config.load_in_8bit,
        )
    return fuser, tokenizer

class RankerDataset(torch.utils.data.Dataset):
    def __init__(self, inputs:List[str], candidates:List[List[str]], instructions:List[str]=None, scores=None):
        self.instructions = instructions
        self.inputs = inputs
        self.candidates = candidates
        self.scores = scores

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        instruction = self.instructions[index] if self.instructions is not None else ""
        input_text = self.inputs[index]
        candidates = self.candidates[index]
        scores = self.scores[index] if self.scores is not None else None
        batch = {
            'index' : index,
            'source' : instruction + input_text,
            'candidates' : candidates,
            'scores' : scores,
        }
        batch = {k: v for k, v in batch.items() if v is not None}
        return batch

class GenFuserDataset(torch.utils.data.Dataset):
    def __init__(self, inputs:List[str], candidates:List[List[str]], tokenizer, max_length, candidate_maxlength, instructions:List[str]=None, outputs:List[str]=None):
        """
            data: list of dict
            tokenizer: tokenizer
            max_length: max length of the input sequence
            top_k: number of top k candidate to select
            select_key: selection metric for the top k candidates
        """
        self.instructions = instructions
        self.inputs = inputs
        self.candidates = candidates
        self.outputs = outputs
        assert len(self.inputs) == len(self.candidates), "Number of inputs and candidates must be the same"
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.candidate_maxlength = candidate_maxlength

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        instruction = self.instructions[index] if self.instructions is not None else ""
        input_text = self.inputs[index]
        candidates = self.candidates[index]
        output = self.outputs[index] if self.outputs is not None else None
        if self.candidate_maxlength is not None:
            for i in range(len(candidates)):
                ids = self.tokenizer.encode(candidates[i], add_special_tokens=False)
                if len(ids) > self.candidate_maxlength:
                    ids = ids[:self.candidate_maxlength]
                    candidates[i] = self.tokenizer.decode(ids)
                    candidates[i] += "..."

        # concatenate input and candidates
        instruction = "Instruction: " + instruction # replace "</s>" with "</s>"
        input = "Input: " + input_text
        candidates = "</s>".join([f"Candidate {i}: <extra_id_{i}>:" + c for i, c in enumerate(candidates)]) # extra id
        fuse_input = "</s>".join([instruction, input, candidates])
        fuse_input += "</s>Summarize candidates into a better one for the given instruction:"

        # tokenize
        fuse_input_ids = self.tokenizer(
            fuse_input,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False,)
        fuse_input_ids = {k: v.squeeze(0) for k, v in fuse_input_ids.items()}

        if output is not None:
            labels_ids = self.tokenizer.encode(output, return_tensors="pt", add_special_tokens=False,).squeeze(0)
        else:
            labels_ids = None

        batch = {
            **fuse_input_ids,
            "labels": labels_ids,
        }
        batch = {k: v for k, v in batch.items() if v is not None}
        return batch

