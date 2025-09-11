# here put the import lib

import importlib
import operator

from peft.tuners.lora import (
    LoraModel,
)
from .config import HiDeLLaVALoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.constants import (
    TRANSFORMERS_MODELS_TO_MMOELORA_TARGET_MODULES_MAPPING,
)
from torch import nn

from .layer import dispatch_default


class HiDeLLaVALoraModel(LoraModel):
    """
    Create HiDeLLaVA (HiDeLLaVA based LoRA) model from a pretrained transformers model.
    """

    prefix: str = "lora_"

    def __init__(self, model, config, adapter_name, **kwargs):
        super().__init__(model, config, adapter_name, **kwargs)
        # LoraModel.__init__(self, model, config, adapter_name, **kwargs)
        # self.add_adapter(adapter_name, self.peft_config[adapter_name])

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):

        new_module = dispatch_default(target, adapter_name, lora_config=lora_config, **kwargs)

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `torch.nn.Conv3d`, "
                "`transformers.pytorch_utils.Conv1D`."
            )

        return new_module

    def _create_and_replace(
        self,
        lora_config: HiDeLLaVALoraConfig,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "expert_num": lora_config.expert_num,
        }

        from .layer import HiDeMOELoraLayer

        if isinstance(target, HiDeMOELoraLayer):
            target.update_layer(
                adapter_name,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
            )
        elif isinstance(target, nn.Linear):
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            return

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_MMOELORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_MMOELORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config
