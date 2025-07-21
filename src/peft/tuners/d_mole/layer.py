import warnings
from typing import Mapping, Optional

import torch
from torch import nn

from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer

from .config import DMOLEConfig


class DMOLELayer(LoraLayer):
    adapter_layer_names: tuple[str, ...] = (
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
        "lora_C",
        "lora_D",
        "lora_CD_gate",
    )

    other_param_names: tuple[str, ...] = (
        "r",
        "lora_alpha",
        "scaling",
        "lora_dropout",
        "lora_gate",
        "lora_taskid_to_loraid",
        "lora_taskid_list",
    )

    def __init__(
        self,
        expert_num: int,
        base_layer: nn.Linear,
    ):
        super().__init__(base_layer=base_layer)

        self.expert_num = expert_num
        self.lora_gate = nn.ModuleDict({})
        self.lora_taskid_to_loraid = {}
        self.lora_taskid_list = []

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

        self.lora_C = nn.ModuleDict(
            {
                adapter_name: nn.ModuleList(
                    [Expert(self.in_features, self.r[adapter_name]) for _ in range(self.task_num)]
                )
            }
        )
        self.lora_D = nn.ModuleDict(
            {
                adapter_name: nn.ModuleList(
                    [Expert(self.r[adapter_name], self.out_features) for _ in range(self.task_num)]
                )
            }
        )
        self.lora_CD_gate = nn.ModuleDict(
            {adapter_name: nn.ModuleList([Gate(self.in_features, 1) for _ in range(self.task_num)])}
        )

        self._move_adapter_to_device_of_base_layer(adapter_name)

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(self.expert_num):
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].loraA[i].mlp.weight, a=5**0.5)
                nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)


class DMOLE_AE_router(nn.Module):
    def __init__(self, input_size: int, max_task_num: int, hidden_size: int = 20):
        super().__init__()
        self.encoder = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(max_task_num)])
        self.decorder = nn.ModuleList([nn.Linear(hidden_size, input_size) for _ in range(max_task_num)])
        self.taskid_to_expertid = {}
        self.taskid_list = []

    def forward(self, x: torch.Tensor, task_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        # 输入task_ids说明处于训练阶段
        loss = None
        if task_ids is not None:
            task_id = int(task_ids[0].cpu().item())
            if self.taskid_to_expertid.get(task_id) is None:
                self.taskid_list.append(task_id)
                self.taskid_to_expertid[task_id] = len(self.taskid_list) - 1

            # x是一个batch，对batch中的每个sample求重建损失，之后求平均作为loss
            previous_dtype = x.dtype
            x = x.to(self.encoder[0].weight.dtype)

            x_rec = self.decorder[self.taskid_to_expertid[task_id]](self.encoder[self.taskid_to_expertid[task_id]](x))
            x_rec = x_rec.to(previous_dtype)

            loss = torch.mean((x - x_rec) ** 2)

        # 批次操作，对于每个sample，求其对应的重建损失，然后选择loss最小的task作为对应sample的id
        if len(self.taskid_list) == 0:
            raise ValueError("No task ids provided for inference. Please provide task ids during training.")
        if len(self.taskid_to_expertid) == 0:
            raise ValueError("No task ids provided for inference. Please provide task ids during training.")

        sample_loss_dist = torch.zeros(x.size(0), len(self.taskid_list), device=x.device)
        for i in range(len(self.taskid_list)):
            task_id = self.taskid_list[i]
            if self.taskid_to_expertid.get(task_id) is None:
                raise ValueError(f"Task ID {task_id} not found in taskid_to_expertid mapping.")

            previous_dtype = x.dtype
            x = x.to(self.encoder[0].weight.dtype)  # Ensure input is in

            x_rec = self.decorder[self.taskid_to_expertid[task_id]](self.encoder[self.taskid_to_expertid[task_id]](x))

            x_rec = x_rec.to(previous_dtype)

            sample_loss = torch.mean((x - x_rec) ** 2, dim=1)
            sample_loss_dist[:, i] = sample_loss

        sorted_losses, sorted_indices = torch.sort(sample_loss_dist, dim=1)
        # print(sorted_indices, flush=True)
        # tensor([[0, 1], [1, 0]])
        sorted_sample_task_ids = []
        for sample in range(sorted_indices.size(0)):
            sorted_task_ids = sorted_indices[sample, :].cpu().tolist()
            sorted_task_ids = [self.taskid_list[task_id] for task_id in sorted_task_ids]
            if task_ids is not None:
                sorted_task_ids.remove(task_ids[sample].cpu().item())
                sorted_task_ids.insert(0, task_ids[sample].cpu().item())
            sorted_sample_task_ids.append(sorted_task_ids)

        sorted_sample_task_ids = torch.LongTensor(sorted_sample_task_ids).to(x.device)

        self.latest_task_ids = sorted_sample_task_ids

        print("sorted_sample_task_ids", sorted_sample_task_ids, flush=True)

        return sorted_sample_task_ids, loss

    def get_extra_state(self):
        return {
            "taskid_to_expertid": self.taskid_to_expertid,
            "taskid_list": self.taskid_list,
        }

    def set_extra_state(self, state):
        self.taskid_to_expertid = state.get("taskid_to_expertid", {})
        self.taskid_list = state.get("taskid_list", [])

    def get_latest_task_ids(self):
        return self.latest_task_ids


class DMOLELinear(nn.Module, DMOLELayer):
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
        DMOLELayer.__init__(self, expert_num=kwargs.pop("expert_num", 2), base_layer=base_layer)
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.task_num = kwargs.pop("max_task_num", True)
        self.te_dim = kwargs.pop("task_embedding_dim", True)

        # Freezing the pre-trained weight matrix

        self.fan_in_fan_out = fan_in_fan_out

        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self._active_adapter = adapter_name

        self.activate = False

    def forward(self, x: torch.Tensor, *args, **kwargs):
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        task_ids = kwargs.pop("task_ids", None)
        if task_ids is not None:
            for sample in range(task_ids.size(0)):
                for task_id in task_ids[sample]:
                    if self.activate:
                        task_id = int(task_id.cpu().item())
                        if self.lora_taskid_to_loraid.get(task_id) is None:
                            self.lora_taskid_list.append(task_id)
                            self.lora_taskid_to_loraid[task_id] = len(self.lora_taskid_list) - 1
            self.activate = False

        previous_dtype = x.dtype

        if self.disable_adapters:  # No adapter
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
                if task_ids is not None:
                    for sample in range(x.size(0)):
                        count = 0
                        lora_result = torch.zeros_like(result[sample], dtype=torch_result_dtype)
                        for task_id in task_ids[sample]:
                            if count >= 2:
                                break
                            if task_id in self.lora_taskid_to_loraid:
                                lora_id = self.lora_taskid_to_loraid[task_id]
                                print(sample, task_id, lora_id, flush=True)
                                lora_result += (
                                    self.lora_CD_gate[active_adapter][lora_id](x[sample])
                                    * scaling
                                    * self.lora_D[active_adapter][lora_id](
                                        self.lora_C[active_adapter][lora_id](x[sample])
                                    )
                                )
                                print(lora_result)
                                count += 1
                        if count > 0:
                            result[sample] += lora_result / count

            result = result.to(torch_result_dtype)

        result = result.to(previous_dtype)

        return result

    def add_new(self, name):
        self.activate = True
        print(name, "added to DMOLELayer")

    def __repr__(self):
        return "DMOLE." + super().__repr__()


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
    lora_config: DMOLEConfig,
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
        new_module = DMOLELinear(target_base_layer, adapter_name, **kwargs)
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
