from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class RankerConfig:
    ranker_type:str = field(
        default=None,
        metadata={"help": "Ranker type, pairranker or reranker \
                  choices: summareranker, dual, pairranker, other;"},
    )
    model_type:str = field(default=None,
        metadata={"help": "Model type, deberta or roberta or other"}
    )
    model_name:str = field(default=None,
        metadata={"help": "Model name"}
    )
    cache_dir:str = field(default=None,
        metadata={"help": "Cache dir"}
    )
    load_checkpoint:str = field(default=None,
        metadata={"help": "Load checkpoint path"}
    )
    source_maxlength:int = field(default=None,
        metadata={"help": "Max length of the source sequence"}
    )
    candidate_maxlength:int = field(default=None,
        metadata={"help": "Max length of the candidate sequence"}
    )
    n_tasks:int = field(default=1,
        metadata={"help": "Number of tasks"}
    )
    num_pos:int = field(default=1,
        metadata={"help": "Number of positive examples used for training, used for top_bottom and all_pair sampling"}
    )
    num_neg:int = field(default=1,
        metadata={"help": "Number of negative examples used for training, used for top_bottom and all_pair sampling"}
    )
    sub_sampling_mode:str = field(default="all_pair",
        metadata={"help": "Sub sampling mode: top_bottom, all_pair, random, uniform"}
    )
    sub_sampling_ratio:float = field(default=0.5,
        metadata={"help": "Sub sampling ratio, used for random and uniform sampling"}
    )
    loss_type:str = field(default="instructgpt",
        metadata={"help": "Loss type: instructgpt, contrastive"}
    )
    reduce_type:str = field(default="linear",
        metadata={"help": "Reduce type: linear, max, mean"}
    )
    inference_mode:str = field(default="bubble",
        metadata={"help": "Inference mode: bubble, full"}
    )
    drop_out:float = field(default=0.05,
        metadata={"help": "Dropout rate"}
    )
    fp16:bool = field(default=True,
        metadata={"help": "Whether to use fp16"}
    )
    device:str = field(default=None,
        metadata={"help": "Device, cuda or cpu or mps"}
    )




                  
