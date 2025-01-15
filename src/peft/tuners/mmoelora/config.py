from dataclasses import dataclass, field

from peft.tuners.lora import LoraConfig
from peft.utils.peft_types import PeftType


@dataclass
class MMOELoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MMOELora`]
    """

    task_num: int = field(default=2, metadata={"help": "The number of tasks."})
    task_embedding_dim: int = field(default=64)
    expert_num: int = field(default=4)

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.MMOELORA
