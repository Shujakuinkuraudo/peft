from .config import MOELoraConfig
from .model import MOELoraModel
from peft.utils import register_peft_method


__all__ = ["MOELoraModel", "MOELoraConfig"]

register_peft_method(name="moelora", config_cls=MOELoraConfig, model_cls=MOELoraModel, is_mixed_compatible=True, prefix="lora_")