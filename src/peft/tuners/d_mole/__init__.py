from .config import DMOLEConfig
from .model import DMOLEModel
from peft.utils import register_peft_method


__all__ = ["DMOLEModel", "DMOLEConfig"]

register_peft_method(name="dmole", config_cls=DMOLEConfig, model_cls=DMOLEModel, is_mixed_compatible=True, prefix="lora_")