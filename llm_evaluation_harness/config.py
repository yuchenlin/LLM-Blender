supported_model = [
    "chavinlo/alpaca-13b",
    "eachadea/vicuna-13b-1.1",
    "databricks/dolly-v2-12b",
    "stabilityai/stablelm-tuned-alpha-7b",
    "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",

    "TheBloke/koala-13B-HF",
    "project-baize/baize-v2-13b",
    "google/flan-t5-xxl",
    "THUDM/chatglm-6b", # x
    "fnlp/moss-moon-003-sft", # x

    "mosaicml/mpt-7b-chat",
    # "TheBloke/guanaco-13B-HF",
    # "NousResearch/Nous-Hermes-13b",
    # "ehartford/WizardLM-13B-Uncensored",
    # "jondurbin/airoboros-13b",
]

batch_size_map = {
    "chavinlo/alpaca-13b": 4,
    "eachadea/vicuna-13b-1.1": 4,
    "databricks/dolly-v2-12b": 4,
    "stabilityai/stablelm-tuned-alpha-7b": 12,
    "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5": 4,
    "TheBloke/koala-13B-HF": 4,
    "project-baize/baize-v2-13b": 4,
    "google/flan-t5-xxl": 4,  # 11.3 B
    "THUDM/chatglm-6b": 12,
    "fnlp/moss-moon-003-sft": 4,
    "mosaicml/mpt-7b-chat": 12,
    "TheBloke/guanaco-13B-HF": 4,
    "NousResearch/Nous-Hermes-13b": 4,
    "ehartford/WizardLM-13B-Uncensored": 4,
    "jondurbin/airoboros-13b": 4,
}
