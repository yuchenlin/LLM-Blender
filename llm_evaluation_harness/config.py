supported_model = [
    "lmsys/vicuna-13b-v1.1",
    "google/flan-t5-xxl",
    "stabilityai/stablelm-tuned-alpha-7b",
    "TheBloke/koala-7B-HF",
    "databricks/dolly-v2-12b",
    "THUDM/chatglm3-6b",
    "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    "mosesjun0h/llama-7b-hf-baize-lora-bf16",
    "fnlp/moss-moon-003-sft",
    "mosaicml/mpt-7b-chat",
    "mosaicml/mpt-7b-instruct",
    "chavinlo/alpaca-native",
]

batch_size_map = {
    "lmsys/vicuna-13b-v1.1": 4,
    "google/flan-t5-xxl": 6,  # 11.3 B
    "stabilityai/stablelm-tuned-alpha-7b": 12,
    "TheBloke/koala-7B-HF": 12,
    "databricks/dolly-v2-12b": 6,
    "THUDM/chatglm3-6b": 32,
    "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5": 6,
    "mosesjun0h/llama-7b-hf-baize-lora-bf16": 12,
    "fnlp/moss-moon-003-sft": 6,
    "mosaicml/mpt-7b-chat": 12,
    "mosaicml/mpt-7b-instruct": 12,
    "chavinlo/alpaca-native": 12,
}
