from dataclasses import dataclass, field

@dataclass
class BlenderConfig:
    device:str = field(default="cuda",
        metadata={"help": "Device, cuda or cpu"}
    )