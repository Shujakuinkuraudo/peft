import warnings
from typing import Optional

import torch
from torch import nn

from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer

from .config import MOELoraConfig


class MOELoraLayer(LoraLayer):

    def __init__(
        self,
        expert_num: int,
        base_layer: nn.Linear,
    ):
        super().__init__(base_layer=base_layer)

        self.expert_num = expert_num
        self.lora_gate = nn.ModuleDict({})

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, **kwargs):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update({adapter_name: lora_dropout_layer})
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update({adapter_name: MOELinearA(self.in_features, r, self.expert_num)})
            self.lora_B.update({adapter_name: MOELinearB(r, self.out_features, self.expert_num)})
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.lora_gate.update({adapter_name: Gate(self.in_features, self.expert_num)})

        self._move_adapter_to_device_of_base_layer(adapter_name)

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(self.expert_num):
                nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)


class MOELoraLinear(nn.Module, MOELoraLayer):
    # Lora implemented in a dense layer
    # nn.Linear is the pretrained weights in LLM, MMOELoraLayer is the designed trainable Lora
    def __init__(
        self,
        base_layer: nn.Linear,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        super().__init__()
        MOELoraLayer.__init__(self, expert_num=kwargs.pop("expert_num", 2), base_layer=base_layer)
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.task_num = kwargs.pop("task_num", True)
        self.te_dim = kwargs.pop("task_embedding_dim", True)

        # nn.Linear.__init__(self, in_features, out_features, **kwargs)

        # init the Gate network
        # self.lora_task_embedding = nn.ModuleDict({})
        # self.lora_task_embedding.update(nn.ModuleDict({adapter_name: nn.Embedding(self.task_num + 1, self.te_dim)}))

        # Freezing the pre-trained weight matrix

        self.fan_in_fan_out = fan_in_fan_out

        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self._active_adapter = adapter_name

    def forward(self, x: torch.Tensor, *args, **kwargs):
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        # task_id = kwargs.pop(
        #     "task_id", torch.tensor([0] * len(x), dtype=torch.long).to(x.device)
        # )
        previous_dtype = x.dtype

        if self.disable_adapters:  # No adapter
            # if self.merged:
            #     self.unmerge(task_id)
            # result = self.base_layer(x, *args, **kwargs)
            # TODO: check this
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:  # general lora process
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                scaling = self.scaling[active_adapter]
                lora_B = self.lora_B[active_adapter]
                lora_A = self.lora_A[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                lora_gate = self.lora_gate[active_adapter]

                x = self._cast_input_dtype(x, lora_A.loraA[0].mlp.weight.dtype)

                expert_weight = lora_gate(x)

                for i in range(self.expert_num):
                    result += (
                        scaling * lora_B.loraB[i](lora_A.loraA[i](dropout(x))) * expert_weight[..., i].unsqueeze(-1)
                    )
            result = result.to(torch_result_dtype)

        result = result.to(previous_dtype)

        return result

    def __repr__(self):
        return "MOElora." + super().__repr__()


class MOELinearA(nn.Module):
    """MMOE based LoRA block"""

    def __init__(self, in_features: int, out_features: int, expert_num: int) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.in_features = in_features
        self.out_features = out_features

        assert self.out_features % self.expert_num == 0  # lora rank should be divided by expert number
        self.r = self.out_features // self.expert_num

        self.loraA = nn.ModuleList([Expert(self.in_features, self.r) for _ in range(self.expert_num)])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """input x is a vector, return output is a list"""
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraA[i](x))

        return outputs


class MOELinearB(nn.Module):
    """MMOE based LoRA block"""

    def __init__(self, in_features: int, out_features: int, expert_num: int) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.in_features = in_features
        self.out_features = out_features

        assert self.in_features % self.expert_num == 0
        self.r = self.in_features // self.expert_num

        self.loraB = nn.ModuleList([Expert(self.r, self.out_features) for _ in range(self.expert_num)])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """input x is a list, return output is also a list"""
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraB[i](x[i]))

        return outputs


class Expert(nn.Module):

    def __init__(self, in_features: int, out_features: int):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.mlp = nn.Linear(self.in_features, self.out_features, bias=False)

    def forward(self, x):
        # LoRA A or B block
        y = self.mlp(x)

        return y


class Gate(nn.Module):

    def __init__(self, input_size: int, expert_num: int):

        super().__init__()
        # 使用embedding来代替线性层
        self.GateL = nn.Linear(input_size, expert_num, bias=False)
        self.act = nn.Softmax(dim=1)  # 第0维为batch size

    def forward(self, x):

        original_dtype = x.dtype

        x = x.to(self.GateL.weight.dtype)  # Ensure input is in the same dtype as GateL
        y = self.GateL(x)
        y = self.act(y)
        y = y.to(original_dtype)

        return y


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: MOELoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    # if isinstance(target_base_layer, torch.nn.Embedding):
    #     embedding_kwargs = kwargs.copy()
    #     embedding_kwargs.pop("fan_in_fan_out", None)
    #     embedding_kwargs.update(lora_config.loftq_config)
    #     new_module = Embedding(target, adapter_name, **embedding_kwargs)
    # elif isinstance(target_base_layer, torch.nn.Conv2d):
    #     kwargs.update(lora_config.loftq_config)
    #     new_module = Conv2d(target, adapter_name, **kwargs)
    # elif isinstance(target_base_layer, torch.nn.Conv3d):
    #     kwargs.update(lora_config.loftq_config)
    #     new_module = Conv3d(target, adapter_name, **kwargs)
    if isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = MOELoraLinear(target_base_layer, adapter_name, **kwargs)
    # elif isinstance(target_base_layer, Conv1D):
    #     if not kwargs["fan_in_fan_out"]:
    #         warnings.warn(
    #             "fan_in_fan_out is set to False but the target module is `Conv1D`. "
    #             "Setting fan_in_fan_out to True."
    #         )
    #         kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
    #     kwargs.update(lora_config.loftq_config)
    #     new_module = Linear(
    #         target, adapter_name, is_target_conv_1d_layer=True, **kwargs
    #     )

    return new_module
