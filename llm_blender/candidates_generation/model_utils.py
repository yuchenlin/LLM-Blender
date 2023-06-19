from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModel,  
)
import torch
from typing import List
decoder_only_models = ["alpaca", "llama", "vicuna", "dolly", "oasst", "stablelm", "koala", "baize", "moss", "opt", "mpt", "guanaco", "hermes", "wizardlm", "airoboros"]
non_conv_models = ["flan-t5"] # models that do not use fastchat conv template
def build_model(model_name, **kwargs):
    """
        Build the model from the model name
    """
    if any([x in model_name.lower() for x in decoder_only_models]):
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    elif "chatglm" in model_name.lower():
        model = AutoModel.from_pretrained(model_name, **kwargs)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
    return model

def build_tokenizer(model_name, **kwargs):
    """
        Build the tokenizer from the model name
    """
    if any([x in model_name.lower() for x in decoder_only_models]):
        # padding left
        if "baize" in model_name.lower():
            # Baize is a special case, they did not configure tokenizer_config.json and we use llama-7b tokenizer
            tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", padding_side="left", **kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

