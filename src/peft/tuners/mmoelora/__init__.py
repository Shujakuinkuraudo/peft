from .config import MMOELoraConfig
from .model import MMOELoraModel
from peft.utils import register_peft_method


__all__ = ["MMOELoraModel", "MMOELoraConfig"]

register_peft_method(name="mmoelora", config_cls=MMOELoraConfig, model_cls=MMOELoraModel, is_mixed_compatible=True)