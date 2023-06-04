import torch
import os
import numpy as np
import torch.nn.functional as F
from .ranker import (
    SummaReranker,
    DualReranker,
    CrossCompareReranker,
)
from .collator import (
    DualCollator,
    SCRCollator,
    CrossCompareCollator,
)
from transformers import (
    RobertaModel,
    BertModel,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,

)
from transformers.models.roberta.modeling_roberta import RobertaModel


def build_pretrained_model(model_type, model_name, **kwargs):
    model = None
    if model_type.startswith("roberta"):
        print("\nUsing RoBERTa model")
        model = RobertaModel.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("bert"):
        print("\nUsing BERT model")
        model = BertModel.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("t5"):
        print("\nUsing T5 model")
        model = T5ForConditionalGeneration.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("bart"):
        print("\nUsing BART model")
        model = BartForConditionalGeneration.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("deberta"):
        print("\nUsing DeBERTa model")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("xlm-roberta"):
        print("\nUsing XLM-RoBERTa model")
        from transformers import XLMRobertaModel
        model = XLMRobertaModel.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("alpaca") or model_type.startswith("llama"):
        print("\nUsing Alpaca/Llama model")
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("flan-t5"):
        print("\nUsing Flan-T5 model")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("opt"):
        print("\nUsing OPT model")
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        raise ValueError("Model type not supported")
    
    if model_type.startswith("opt"):
        model.config.out_hidden_state_size = model.config.word_embed_proj_dim
    else:
        model.config.out_hidden_state_size = model.config.hidden_size
    return model

def build_tokenizer(model_name, **kwargs):
    """
        Build the tokenizer from the model name
    """
    if "alpaca" in model_name or "llama" in model_name:
        # padding left
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def build_ranker(ranker_type, model_type, model_name, cache_dir, config, tokenizer):
    ranker = None
    pretrained_model = build_pretrained_model(model_type, model_name, cache_dir=cache_dir)
    pretrained_model.resize_token_embeddings(len(tokenizer))
    if ranker_type == "summareranker":
        ranker = SummaReranker(pretrained_model, config, tokenizer)
    elif ranker_type == "dual":
        ranker = DualReranker(pretrained_model, config, tokenizer)
    elif ranker_type == "pairranker":
        ranker = CrossCompareReranker(pretrained_model, config, tokenizer)
    return ranker

def build_collator(
    model_type:str,
    tokenizer,
    source_max_length:int,
    candidate_max_length:int,
    source_prefix:str = None,
    candidate1_prefix:str = None,
    candidate2_prefix:str = None,
    ):
    if model_type == "summareranker":
        return SCRCollator(source_max_length, tokenizer, candidate_max_length, source_prefix, candidate1_prefix)
    elif model_type == "dual":
        return DualCollator(source_max_length, tokenizer, candidate_max_length, source_prefix, candidate1_prefix)
    elif model_type == "pairranker":
        return CrossCompareCollator(source_max_length, tokenizer, candidate_max_length, source_prefix, candidate1_prefix, candidate2_prefix)
    else:
        raise ValueError(f"model_type {model_type} not supported")


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
    