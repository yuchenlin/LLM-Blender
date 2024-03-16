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
    "chavinlo/alpaca-native" "THUDM/chatglm-6b",
]

batch_size_map = {
    "lmsys/vicuna-13b-v1.1": 6,
    "google/flan-t5-xxl": 8,  # 11.3 B
    "stabilityai/stablelm-tuned-alpha-7b": 16,
    "TheBloke/koala-7B-HF": 16,
    "databricks/dolly-v2-12b": 8,
    "THUDM/chatglm3-6b": 16,
    "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5": 8,
    "mosesjun0h/llama-7b-hf-baize-lora-bf16": 16,
    "fnlp/moss-moon-003-sft": 6,
    "mosaicml/mpt-7b-chat": 16,
    "mosaicml/mpt-7b-instruct": 16,
    "chavinlo/alpaca-native" "THUDM/chatglm-6b": 16,
}
