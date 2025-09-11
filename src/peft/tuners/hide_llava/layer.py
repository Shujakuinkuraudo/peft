import warnings
from typing import Optional

import torch
from torch import nn
from ...utils import transpose

from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer

from .config import HiDeLLaVALoraConfig


class HiDeMOELoraLayer(LoraLayer):

    def __init__(
        self,
        base_layer: nn.Module,
        in_features: int,
        out_features: int,
        expert_num: int,
        cur_task: int,
        training: bool,
        layer_num: int,
        expert_weight: list,
    ):

        super().__init__(base_layer=base_layer)
        self.expert_num = expert_num
        self.cur_task: int = cur_task
        self.training = training
        self.layer_num = layer_num
        self.expert_weight = expert_weight

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):  # type: ignore
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update({adapter_name: lora_dropout_layer})
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(
                {
                    adapter_name: HiDeMOELinearA(
                        self.in_features,
                        r,
                        self.expert_num,
                        self.cur_task,
                        self.training,
                        self.layer_num,
                        self.expert_weight,
                    )
                }
            )
            self.lora_B.update(
                {
                    adapter_name: HiDeMOELinearB(
                        r,
                        self.out_features,
                        self.expert_num,
                        self.cur_task,
                        self.training,
                        self.layer_num,
                        self.expert_weight,
                    )
                }
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):  # type: ignore
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(self.expert_num):
                nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)  # type: ignore
                nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)  # type: ignore


class HiDeMOELoraLinear(nn.Module, HiDeMOELoraLayer):
    # Lora implemented in a dense layer
    # nn.Linear is the pretrained weights in LLM, MMOELoraLayer is the designed trainable Lora
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        train_signal: bool = False,
        layer: int = 0,
        expert_weight: list = [],
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.expert_num = kwargs.pop("expert_num", True)
        self.te_dim = kwargs.pop("task_embedding_dim", True)
        self.cur_task = kwargs.pop("cur_task", True)

        super().__init__()

        HiDeMOELoraLayer.__init__(
            self,
            base_layer,
            in_features=in_features,
            out_features=out_features,
            expert_num=self.expert_num,
            cur_task=self.cur_task,
            training=train_signal,
            layer_num=layer,
            expert_weight=expert_weight,
        )

        self.layer = layer
        self.expert_weight = expert_weight
        self.training = train_signal

        # init the Gate network
        self.lora_router = nn.ModuleDict({})
        self.lora_router.update({adapter_name: nn.Linear(self.in_features, self.expert_num, bias=False)})

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self._active_adapter = adapter_name

    def merge(self):
        pass

    def unmerge(self):
        pass

    def forward(self, x: torch.Tensor, **kwargs):  # type: ignore
        previous_dtype = x.dtype

        if self.active_adapter not in self.lora_A.keys():  # No adapter, directly use linear
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:  # No adapter
            if self.r[self.active_adapter] > 0 and self.merged:  # merge the adapter to linear
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0:  # general lora process
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)

            if self.training:
                lora_a_output = self.lora_A[self.active_adapter].loraA[self.cur_task](
                    self.lora_dropout[self.active_adapter](x)
                )
                lora_b_output = self.lora_B[self.active_adapter].loraB[self.cur_task](lora_a_output)
                result += lora_b_output * self.scaling[self.active_adapter]
            else:
                if int(self.layer) != 31:
                    lora_a_output = self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                    lora_b_output = self.lora_B[self.active_adapter](lora_a_output)
                    result += lora_b_output * self.scaling[self.active_adapter]
                else:
                    for i in range(len(self.expert_weight)):
                        result += (  # lora process
                            self.lora_B[self.active_adapter].loraB[i](
                                self.lora_A[self.active_adapter].loraA[i](self.lora_dropout[self.active_adapter](x)),
                            )
                            * self.scaling[self.active_adapter]
                            * self.expert_weight[i]
                        )
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result


class HiDeMOELinearA(nn.Module):
    """MMOE based LoRA block"""

    def __init__(self, in_features, out_features, expert_num, cur_task, training, layer_num, weight) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.cur_task = cur_task
        self.in_features, self.out_features = in_features, out_features
        self.loraA = nn.ModuleList([])
        self.training = training
        self.layer = layer_num
        self.expert_weight = weight

        assert self.out_features % self.expert_num == 0  # lora rank should be divided by expert number
        self.r = self.out_features // self.expert_num

        for _ in range(self.expert_num):
            self.loraA.append(HiDeMOEExpert(self.in_features, self.r))

    def forward(self, x):
        """input x is a vector, return output is a list"""
        if self.training:
            assert 0 <= self.cur_task < self.expert_num, "Invalid current_task value"
            output = self.loraA[self.cur_task](x)
            return output
        else:
            merge_weight = 1.0
            if int(self.layer) != 31:
                # temp_mlp = nn.Linear(self.in_features, self.r, bias=False).to(x.device)

                fused_weight = torch.zeros((self.r, self.in_features), device=x.device)

                for i in range(self.cur_task + 1):
                    fused_weight += merge_weight * self.loraA[i].weight

                with torch.no_grad():
                    temp_mlp.weight.copy_(fused_weight)

                output = temp_mlp(x)

                return output
            else:
                outputs = []
                for i in range(self.expert_num):
                    outputs.append(self.loraA[i](x))

                return outputs


class HiDeMOELinearB(nn.Module):
    """MMOE based LoRA block"""

    def __init__(self, in_features, out_features, expert_num, cur_task, training, layer, weight) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.cur_task = cur_task
        self.in_features, self.out_features = in_features, out_features
        self.loraB = nn.ModuleList([])
        self.training = training
        self.layer = layer
        self.expert_weight = weight

        assert self.in_features % self.expert_num == 0
        self.r = self.in_features // self.expert_num

        for _ in range(self.expert_num):
            self.loraB.append(HiDeMOEExpert(self.r, self.out_features))

    def forward(self, x):
        """input x is a list, return output is also a list"""
        if self.training:
            assert 0 <= self.cur_task < self.expert_num, "Invalid current_task value"
            output = self.loraB[self.cur_task](x)
        else:
            merge_weight = 1.0
            if int(self.layer) != 31:
                temp_mlp = nn.Linear(self.r, self.out_features, bias=False).to(x.device)

                fused_weight = torch.zeros((self.out_features, self.r), device=x.device)

                for i in range(self.cur_task + 1):
                    fused_weight += merge_weight * self.loraB[i].weight

                with torch.no_grad():
                    temp_mlp.weight.copy_(fused_weight)

                output = temp_mlp(x)

                return output
            else:
                outputs = []
                for i in range(self.expert_num):
                    outputs.append(self.loraB[i](x[i]))

                return outputs


class HiDeMOEExpert(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()

        self.in_features, self.out_features = in_features, out_features
        self.mlp = nn.Linear(self.in_features, self.out_features, bias=False)
        self.weight = self.mlp.weight

    def forward(self, x):
        # LoRA A or B block
        y = self.mlp(x)

        return y


class HiDeMOEGate(nn.Module):

    def __init__(self, input_size, expert_num):

        super().__init__()
        # 使用embedding来代替线性层
        self.GateL = nn.Linear(input_size, expert_num, bias=False)
        self.act = nn.Softmax(dim=1)  # 第0维为batch size

    def forward(self, x):

        y = self.GateL(x)
        y = self.act(y)

        return y


class HiDeMOERouter(nn.Module):
    """
    Router using tokens choose top-1 experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.

    """

    def __init__(self, config: HiDeLLaVALoraConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        self.jitter_noise = config.router_jitter_noise
        self.ignore_padding_tokens = config.router_ignore_padding_tokens
        self.dtype = getattr(torch, config.router_dtype)

    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        self.input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dtype)

        if self.training and self.jitter_noise > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        # Shape: [num_groups, tokens_per_group, num_experts]
        self._cast_classifier()
        router_logits = self.classifier(hidden_states)

        # Apply Softmax and cast back to the original `dtype`
        router_probabilities = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        return router_probabilities, router_logits

    def _cast_classifier(self):
        if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
            self.classifier = self.classifier.to(self.dtype)

    def forward(self, hidden_states: torch.Tensor) -> Tuple:
        router_probs, router_logits = self._compute_router_probabilities(hidden_states)

        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)

        # Mask tokens outside expert capacity. Sum over each sequence
        token_priority = torch.cumsum(expert_index, dim=-2)
        # mask if the token routed to to the expert will overflow
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask

        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_index, router_probs, router_logits


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: HiDeLLaVALoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = HiDeMOELoraLinear(target_base_layer, adapter_name, **kwargs)
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
