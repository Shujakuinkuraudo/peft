from .config import HiDeLLaVALoraConfig
from .model import HiDeLLaVALoraModel
from peft.utils import register_peft_method


__all__ = ["HiDeLLaVALoraModel", "HiDeLLaVALoraConfig"]

register_peft_method(name="hidellava", config_cls=HiDeLLaVALoraConfig, model_cls=HiDeLLaVALoraModel, is_mixed_compatible=True, prefix="hidellava_")