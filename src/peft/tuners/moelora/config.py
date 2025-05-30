from dataclasses import dataclass, field

from peft.tuners.lora import LoraConfig
from peft.utils.peft_types import PeftType


@dataclass
class MOELoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MMOELora`]
    """

    expert_num: int = field(default=4)

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.MOELORA
