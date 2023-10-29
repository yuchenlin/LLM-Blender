from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
@dataclass_json
@dataclass
class BlenderConfig:
    device:str = field(default="cuda",
        metadata={"help": "Device, cuda or cpu"}
    )