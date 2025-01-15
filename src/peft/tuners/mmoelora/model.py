# here put the import lib

import importlib
import operator

from peft.tuners.lora import (
    LoraModel,
)
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.constants import (
    TRANSFORMERS_MODELS_TO_MMOELORA_TARGET_MODULES_MAPPING,
)

from .layer import dispatch_default


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


class MMOELoraModel(LoraModel):
    """
    Create MMOELoRA (MMOE based LoRA) model from a pretrained transformers model.
    """

    def __init__(self, model, config, adapter_name, **kwargs):
        super().__init__(model, config, adapter_name, **kwargs)
        # LoraModel.__init__(self, model, config, adapter_name, **kwargs)
        # self.add_adapter(adapter_name, self.peft_config[adapter_name])

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = []

        if lora_config._custom_modules:
            # Experimental custom LoRA module support. Allows users to pass a custom mapping for unsupported layer
            # types by impelementing their own LoRA layers.
            def dynamic_dispatch_func(target, adapter_name, lora_config, **kwargs):
                new_module = None

                if isinstance(target, BaseTunerLayer):
                    target_base_layer = target.get_base_layer()
                else:
                    target_base_layer = target

                for key, custom_cls in lora_config._custom_modules.items():
                    if isinstance(target_base_layer, key):
                        new_module = custom_cls(target, adapter_name, **kwargs)
                        break

                return new_module

            dispatchers.append(dynamic_dispatch_func)

        # avoid eager bnb import
        # if is_bnb_available():
        #     from .bnb import dispatch_bnb_8bit

        #     dispatchers.append(dispatch_bnb_8bit)

        # if is_bnb_4bit_available():
        #     from .bnb import dispatch_bnb_4bit

        # dispatchers.append(dispatch_bnb_4bit)

        dispatchers.extend(
            [
                dispatch_default,
            ]
        )

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
            if new_module is not None:  # first match wins
                break

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
        lora_config,
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
            "task_num": lora_config.task_num,
            "task_embedding_dim": lora_config.task_embedding_dim,
            "expert_num": lora_config.expert_num,
        }

        # Regexp matching - Find key which matches current target_name in patterns provided
        # r_key = get_pattern_key(lora_config.rank_pattern.keys(), current_key)
        # alpha_key = get_pattern_key(lora_config.alpha_pattern.keys(), current_key)
        # r = lora_config.rank_pattern.get(r_key, lora_config.r)
        # alpha = lora_config.alpha_pattern.get(alpha_key, lora_config.lora_alpha)

        # kwargs = {
        #     "r": r,
        #     "lora_alpha": alpha,
        #     "lora_dropout": lora_config.lora_dropout,
        #     "fan_in_fan_out": lora_config.fan_in_fan_out,
        #     "init_lora_weights": lora_config.init_lora_weights,
        #     "use_rslora": lora_config.use_rslora,
        #     "use_dora": lora_config.use_dora,
        #     "ephemeral_gpu_offload": lora_config.runtime_config.ephemeral_gpu_offload,
        #     "lora_bias": lora_config.lora_bias,
        #     "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
        #     "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        # }
        # for torchao merging, we need the get_apply_tensor_subclass from the quantization config
        try:
            kwargs["get_apply_tensor_subclass"] = operator.attrgetter(
                "hf_quantizer.quantization_config.get_apply_tensor_subclass"
            )(self.model)
        except AttributeError:
            pass

        # quant_methods = ["gptq", "aqlm", "awq"]
        # for quant_method in quant_methods:
        #     quantization_config = get_quantization_config(self.model, method=quant_method)
        #     if quantization_config is not None:
        #         kwargs[f"{quant_method}_quantization_config"] = quantization_config

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        from .layer import MMOELoraLayer

        if isinstance(target, MMOELoraLayer) and not isinstance(target, AdaLoraLayer):
            target.update_layer(
                adapter_name,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
                lora_bias=lora_config.lora_bias,
            )
        else:
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_MMOELORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_MMOELORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config
