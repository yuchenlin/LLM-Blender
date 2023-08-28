import gradio as gr
import torch
import llm_blender
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    StoppingCriteria, StoppingCriteriaList,
)
from accelerate import infer_auto_device_map
from typing import List

from model_utils import build_tokenizer, build_model, get_llm_prompt, get_stop_str_and_ids
BASE_LLM_NAMES = [
    "chavinlo/alpaca-native",
    "eachadea/vicuna-13b-1.1",
    "databricks/dolly-v2-12b",
    "stabilityai/stablelm-tuned-alpha-7b",
    "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    "TheBloke/koala-13B-HF",
    "project-baize/baize-v2-13b",
    "google/flan-t5-xxl",
    "THUDM/chatglm-6b",
    "fnlp/moss-moon-003-sft",
    "mosaicml/mpt-7b-chat",
]

BASE_LLM_MODELS = {
    name: None for name in BASE_LLM_NAMES
}
BASE_LLM_TOKENIZERS = {
    name: None for name in BASE_LLM_NAMES
}

class StopTokenIdsCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the generated number of tokens exceeds `max_new_tokens`. Keep in
    mind for decoder-only type of transformers, this will **not** include the initial prompted tokens. This is very
    close to `MaxLengthCriteria` but ignores the number of initial tokens.

    Args:
        stop_token_ids (`List[int]`):
    """

    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.stop_token_ids:
            return all(_input_ids[-1] in self.stop_token_ids for _input_ids in input_ids)
        return False
    
def llm_generate(
    base_llm_name:str, instruction:str, input:str, 
    max_new_tokens:int, top_p=1.0, temperature=0.7,
) -> str:
    if BASE_LLM_MODELS.get(base_llm_name, None) is None:
        BASE_LLM_MODELS[base_llm_name] = build_model(
            base_llm_name, device_map="auto", 
            load_in_8bit=True, trust_remote_code=True)
    if BASE_LLM_TOKENIZERS.get(base_llm_name, None) is None:
        BASE_LLM_TOKENIZERS[base_llm_name] = build_tokenizer(
            base_llm_name, trust_remote_code=True)
    base_llm = BASE_LLM_MODELS[base_llm_name]
    base_llm_tokenizer = BASE_LLM_TOKENIZERS[base_llm_name]
    llm_prompt = get_llm_prompt(base_llm_name, instruction, input)
    stop_str, stop_token_ids = get_stop_str_and_ids(base_llm_tokenizer)

    template_length = len(base_llm_tokenizer.encode(
        llm_prompt.replace(instruction, "").replace(input, "")))
    
    encoded_llm_prompt = base_llm_tokenizer(llm_prompt, 
        max_length=256 + template_length, 
        padding='max_length', truncation=True, return_tensors="pt")

    input_ids = encoded_llm_prompt["input_ids"].to(base_llm.device)
    attention_mask = encoded_llm_prompt["attention_mask"].to(base_llm.device)

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "num_return_sequences": 1,
    }
    if stop_token_ids:
        generate_kwargs['stopping_criteria'] = StoppingCriteriaList([
            StopTokenIdsCriteria(stop_token_ids),
        ])
    
    output_ids = base_llm.generate(**generate_kwargs)
    output_ids_wo_prompt = output_ids[0, input_ids.shape[1]:]
    decoded_output = base_llm_tokenizer.decode(output_ids_wo_prompt, skip_special_tokens=True)
    if stop_str:
        pos = decoded_output.find(stop_str)
        if pos != -1:
            decoded_output = decoded_output[:pos]
    return decoded_output

def llms_generate(
    base_llm_names, instruction, input, 
    max_new_tokens, top_p=1.0, temperature=0.7,
):  
    return {
        base_llm_name: llm_generate(
        base_llm_name, instruction, input, max_new_tokens, top_p, temperature)
        for base_llm_name in base_llm_names
    }