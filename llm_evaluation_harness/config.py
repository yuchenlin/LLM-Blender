supported_model = [
    # for test
    "mistralai/Mistral-7B-Instruct-v0.1",
    "lmsys/vicuna-13b-v1.5",

    # These model is usded in the paper
    # "chavinlo/alpaca-13b",
    # "eachadea/vicuna-13b-1.1",
    # "databricks/dolly-v2-12b",
    # "stabilityai/stablelm-tuned-alpha-7b",
    # "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    # "TheBloke/koala-13B-HF",
    # "project-baize/baize-v2-13b",
    # "google/flan-t5-xxl",
    # "THUDM/chatglm-6b", # need transformer==4.33.2
    # "fnlp/moss-moon-003-sft", # need transformer==4.33.2
    # "mosaicml/mpt-7b-chat",

    # below model is not usded in the paper
    # "TheBloke/guanaco-13B-HF", 
    # "NousResearch/Nous-Hermes-13b",
    # "ehartford/WizardLM-13B-Uncensored",
    # "jondurbin/airoboros-13b",
]

batch_size_map = {
    "mistralai/Mistral-7B-Instruct-v0.1":8,
    "lmsys/vicuna-13b-v1.5":8,
    
    "chavinlo/alpaca-13b": 6,
    "eachadea/vicuna-13b-1.1": 6,
    "databricks/dolly-v2-12b": 6,
    "stabilityai/stablelm-tuned-alpha-7b": 12,
    "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5": 6,
    "TheBloke/koala-13B-HF": 6,
    "project-baize/baize-v2-13b": 6,
    "google/flan-t5-xxl": 6,  # 11.3 B
    "THUDM/chatglm-6b": 12,
    "fnlp/moss-moon-003-sft": 6,
    "mosaicml/mpt-7b-chat": 12,
    "TheBloke/guanaco-13B-HF": 6,
    "NousResearch/Nous-Hermes-13b": 6,
    "ehartford/WizardLM-13B-Uncensored": 6,
    "jondurbin/airoboros-13b": 6,
}
