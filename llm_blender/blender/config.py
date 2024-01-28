from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
@dataclass_json
@dataclass
class BlenderConfig:
    device:str = field(default="cuda",
        metadata={"help": "Device, cuda or cpu or mps"}
    )
    use_tqdm:bool = field(default=True,
        metadata={"help": "Use tqdm progress bar"}
    )
    