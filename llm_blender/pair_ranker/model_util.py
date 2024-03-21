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
    DebertaRMCollator,
    StarlingRMCollator,
    UltraRMCollator
)
from .other_rms.starling_rm import StarlingRM
from .other_rms.ultra_rm import UltraRM
from transformers import (
    RobertaModel,
    BertModel,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)
from transformers.utils import is_flash_attn_2_available
from transformers.models.roberta.modeling_roberta import RobertaModel


def build_pretrained_model(model_type, model_name, **kwargs):
    model = None
    if model_type.startswith("roberta"):
        model = RobertaModel.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("bert"):
        model = BertModel.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("t5"):
        model = T5ForConditionalGeneration.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("bart"):
        model = BartForConditionalGeneration.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("deberta-rm"):
        model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("deberta"):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("xlm-roberta"):
        from transformers import XLMRobertaModel
        model = XLMRobertaModel.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("alpaca") or model_type.startswith("llama"):
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("flan-t5"):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("opt"):
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("starling-rm"):
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", **kwargs)
    elif model_type.startswith("ultra-rm"):
        model = UltraRM.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("other"):
        model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    elif model_type.startswith("phi"):
        if is_flash_attn_2_available():
            kwargs["attn_implementation"] = "flash_attention_2"
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
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
    elif "starling-rm" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", **kwargs)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.truncation_side = "left"
    elif "phi" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        tokenizer.add_special_tokens({"sep_token": "<|sepoftext|>"})
        tokenizer.sep_token = "<|sepoftext|>"
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def build_ranker(ranker_type, model_type, model_name, cache_dir, config, tokenizer):
    ranker = None
    pretrained_model = build_pretrained_model(model_type, model_name, cache_dir=cache_dir)
    if ranker_type == "summareranker":
        pretrained_model.resize_token_embeddings(len(tokenizer))
        ranker = SummaReranker(pretrained_model, config, tokenizer)
    elif ranker_type == "dual":
        pretrained_model.resize_token_embeddings(len(tokenizer))
        ranker = DualReranker(pretrained_model, config, tokenizer)
    elif ranker_type == "pairranker":
        pretrained_model.resize_token_embeddings(len(tokenizer))
        ranker = CrossCompareReranker(pretrained_model, config, tokenizer)
    elif ranker_type == "deberta-rm":
        ranker = pretrained_model
    elif ranker_type == "starling-rm":
        ranker = StarlingRM(pretrained_model, config, tokenizer)
    elif ranker_type == "ultra-rm":
        ranker = pretrained_model
    else:
        raise ValueError(f"ranker_type {ranker_type} not supported")
    return ranker

def build_collator(
    ranker_type:str,
    tokenizer,
    source_maxlength:int,
    candidate_maxlength:int,
    source_prefix:str = None,
    candidate1_prefix:str = None,
    candidate2_prefix:str = None,
    ):
    if ranker_type == "summareranker":
        return SCRCollator(source_maxlength, tokenizer, candidate_maxlength, source_prefix, candidate1_prefix)
    elif ranker_type == "dual":
        return DualCollator(source_maxlength, tokenizer, candidate_maxlength, source_prefix, candidate1_prefix)
    elif ranker_type == "pairranker":
        return CrossCompareCollator(source_maxlength, tokenizer, candidate_maxlength, source_prefix, candidate1_prefix, candidate2_prefix)
    elif ranker_type == "deberta-rm":
        return DebertaRMCollator(source_maxlength, tokenizer, candidate_maxlength)
    elif ranker_type == "starling-rm":
        return StarlingRMCollator(source_maxlength, tokenizer, candidate_maxlength)
    elif ranker_type == "ultra-rm":
        return UltraRMCollator(source_maxlength, tokenizer, candidate_maxlength)
    else:
        raise ValueError(f"ranker_type {ranker_type} not supported")


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
    