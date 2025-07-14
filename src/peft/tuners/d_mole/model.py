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
from torch import nn

from .layer import dispatch_default
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import DMOLEConfig


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


class DMOLEModel(LoraModel):
    """
    Create DMOLE (Dynamic Mixture of Experts LoRA) model from a pretrained transformers model.
    """

    prefix: str = "lora_"

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
        lora_config: "DMOLEConfig",
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
            "max_task_num": lora_config.max_task_num,
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

        from .layer import DMOLELayer

        if isinstance(target, DMOLELayer) and not isinstance(target, AdaLoraLayer):
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

    # def inject_adapter(
    #     self, model: nn.Module, adapter_name: str, autocast_adapter_dtype: bool = True, low_cpu_mem_usage: bool = False
    # ) -> None:
    #     r"""
    #     Creates adapter layers and replaces the target modules with the adapter layers. This method is called under the
    #     hood by `peft.mapping.get_peft_model` if a non-prompt tuning adapter class is passed.

    #     The corresponding PEFT config is directly retrieved from the `peft_config` attribute of the BaseTuner class.

    #     Args:
    #         model (`nn.Module`):
    #             The model to be tuned.
    #         adapter_name (`str`):
    #             The adapter name.
    #         autocast_adapter_dtype (`bool`, *optional*):
    #             Whether to autocast the adapter dtype. Defaults to `True`.
    #         low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
    #             Create empty adapter weights on meta device. Useful to speed up the loading process.

    #     """
    #     peft_config = self.peft_config[adapter_name]
    #     excluded_modules = []
    #     unmatched_modules = []
    #     # Note: If possible, all checks should be performed *at the start of this method*.
    #     # This way, we can raise early if something goes wrong, without leaving the model
    #     # in a bad (half-initialized) state.
    #     self._check_new_adapter_config(peft_config)

    #     _check_for_modules_to_save = getattr(peft_config, "modules_to_save", None) is not None

    #     model_config = self.get_model_config(model)

    #     peft_config = self._prepare_adapter_config(peft_config, model_config)

    #     self._prepare_model(peft_config, model)
    #     key_list = [key for key, _ in model.named_modules()]

    #     uses_dummy_target_modules = getattr(peft_config, "target_modules", None) == DUMMY_TARGET_MODULES
    #     if uses_dummy_target_modules:
    #         # dummy adapter, we allow not matching any module
    #         key_list = []

    #     # update peft_config.target_modules if required
    #     peft_config = _maybe_include_all_linear_layers(peft_config, model)

    #     # This is an optimization to reduce the number of entries in the target_modules list. The reason is that in some
    #     # circumstances, target_modules can contain hundreds of entries. Since each target module is checked against
    #     # each module of the net (which can be thousands), this can become quite expensive when many adapters are being
    #     # added. Often, the target_modules can be condensed in such a case, which speeds up the process.
    #     # A context in which this can happen is when diffusers loads non-PEFT LoRAs. As there is no meta info on
    #     # target_modules in that case, they are just inferred by listing all keys from the state_dict, which can be
    #     # quite a lot. See: https://github.com/huggingface/diffusers/issues/9297
    #     # As there is a small chance for undiscovered bugs, we apply this optimization only if the list of
    #     # target_modules is sufficiently big.
    #     # We also exclude IA³ from this optimization. This is because IA³ has both target_modules and
    #     # feedforward_modules, which are coupled (the latter must be a subset). It would be possible to change the logic
    #     # to keep both in sync, but it's not quite trivial and probably not worth the effort. See #2429.
    #     if (
    #         isinstance(peft_config.target_modules, (list, set))
    #         and (len(peft_config.target_modules) >= MIN_TARGET_MODULES_FOR_OPTIMIZATION)
    #         and (peft_config.peft_type != PeftType.IA3)
    #     ):
    #         names_no_target = [
    #             name
    #             for name in key_list
    #             if not any((name == suffix) or name.endswith("." + suffix) for suffix in peft_config.target_modules)
    #         ]
    #         new_target_modules = _find_minimal_target_modules(peft_config.target_modules, names_no_target)
    #         if len(new_target_modules) < len(peft_config.target_modules):
    #             peft_config.target_modules = new_target_modules

    #     for key in key_list:
    #         if not key:
    #             continue

    #         result = self._check_target_module_exists(peft_config, key)
    #         if isinstance(result, _ExcludedModule):
    #             excluded_modules.append(key)
    #         elif not result:
    #             unmatched_modules.append(key)
    #         else:
    #             self.targeted_module_names.append(key)
    #             parent, target, target_name = _get_submodules(model, key)
    #             ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
    #             with ctx():
    #                 self._create_and_replace(peft_config, adapter_name, target, target_name, parent, current_key=key)

    #     if not self.targeted_module_names and not uses_dummy_target_modules:
    #         if excluded_modules and not unmatched_modules:
    #             # All targeted modules were excluded
    #             raise ValueError(
    #                 "All modules were excluded. This is likely unintended. "
    #                 "Check your `target_modules`, `exclude_modules` and `modules_to_save` configuration."
    #             )
    #         elif not excluded_modules and unmatched_modules:
    #             # None of the targeted modules matched
    #             error_msg = (
    #                 f"Target modules {peft_config.target_modules} not found in the base model. "
    #                 f"Please check the target modules and try again."
    #             )
    #             if getattr(peft_config, "layers_to_transform", None) is not None:
    #                 error_msg += f" Note: You specified 'layers_to_transform': {peft_config.layers_to_transform}."
    #             if getattr(peft_config, "layers_pattern", None) is not None:
    #                 error_msg += f" You also specified 'layers_pattern': {peft_config.layers_pattern}."
    #             raise ValueError(error_msg)
    #         else:
    #             # Some modules did not match and some matched but were excluded
    #             error_msg = (
    #                 "No modules were targeted for adaptation. "
    #                 "This might be caused by a combination of mismatched target modules and excluded modules. "
    #                 "Please check your `target_modules` and `exclude_modules` configuration. You may also have "
    #                 "only targeted modules that are marked to be saved (`modules_to_save`)."
    #             )
    #             if getattr(peft_config, "layers_to_transform", None) is not None:
    #                 error_msg += f" Note: You specified 'layers_to_transform': {peft_config.layers_to_transform}."
    #             if getattr(peft_config, "layers_pattern", None) is not None:
    #                 error_msg += f" You also specified 'layers_pattern': {peft_config.layers_pattern}."
    #             raise ValueError(error_msg)

    #     elif hasattr(peft_config, "exclude_modules") and peft_config.exclude_modules and not excluded_modules:
    #         # exclude_modules was passed but was not used
    #         warnings.warn(
    #             f"You have passed exclude_modules={peft_config.exclude_modules} but no modules were excluded. "
    #             "Please check that exclude_modules was set correctly."
    #         )

    #     tied_target_modules = self._get_tied_target_modules(model=model)
    #     if tied_target_modules:
    #         warnings.warn(
    #             f"Model with `tie_word_embeddings=True` and the {tied_target_modules=} are part of the adapter. "
    #             "This can lead to complications, for example when merging the adapter "
    #             "or converting your model to formats other than safetensors. "
    #             "See for example https://github.com/huggingface/peft/issues/2018."
    #         )

    #     # It's important to set the adapter here (again), because otherwise it can happen that if a 2nd adapter is
    #     # added, and it targets different layer(s) than the first adapter (which is active), then those different
    #     # layers will be activated, which we don't want.
    #     self.set_adapter(self.active_adapters)
    #     self._mark_only_adapters_as_trainable(model)

    #     if self.peft_config[adapter_name].inference_mode:
    #         for n, p in model.named_parameters():
    #             if adapter_name in n:
    #                 p.requires_grad = False

    #     set_additional_trainable_modules(
    #         model=model,
    #         peft_config=peft_config,
    #         model_config=BaseTuner.get_model_config(self),
    #         adapter_name=adapter_name,
    #     )
