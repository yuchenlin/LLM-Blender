from dataclasses import dataclass, field
@dataclass
class GenFuserConfig:
    model_name:str = field(default="llm-blender/gen_fuser_3b",
        metadata={"help": "Model name from huggingface.co/models"}
    )
    cache_dir:str = field(default=None,
        metadata={"help": "Cache dir"}
    )
    max_length:int = field(default=1024,
        metadata={"help": "Max length of the total sequence (source + top-k candidate)"}
    )
    candidate_maxlength:int = field(default=128,
        metadata={"help": "Max length of the candidate sequence"}
    )
    torch_dtype:str = field(default="bfloat16",
        metadata={"help": "torch dtype"})

                  
