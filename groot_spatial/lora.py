"""Scoped Qwen attention LoRA, inspired by memWMLM's application adapter."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adapter import SpatialBackbone


@dataclass(frozen=True)
class LoRAConfig:
    layer_indices: tuple[int, ...] = (0, 1, 2, 3)
    projections: tuple[str, ...] = ("q_proj", "o_proj")
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0

    def __post_init__(self):
        if not self.layer_indices or len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("LoRA layer indices must be nonempty and unique")
        if any(i < 0 for i in self.layer_indices):
            raise ValueError("LoRA layer indices must be nonnegative")
        if not self.projections or len(set(self.projections)) != len(self.projections):
            raise ValueError("LoRA projections must be nonempty and unique")
        if not set(self.projections) <= {"q_proj", "k_proj", "v_proj", "o_proj"}:
            raise ValueError("unsupported attention projection")
        if self.rank <= 0 or self.alpha <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("invalid LoRA rank, alpha or dropout")


class LoRALinear(nn.Module):
    """Frozen projection plus zero-initialised low-rank residual.

    LeRobot puts frozen Qwen in eval mode inside its forward. The enclosing
    SpatialBackbone sets adapter_training independently, so optional LoRA
    dropout still follows the outer training/evaluation mode.
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_down = nn.Linear(base.in_features, config.rank, bias=False, device=base.weight.device)
        self.lora_up = nn.Linear(config.rank, base.out_features, bias=False, device=base.weight.device)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.scale = config.alpha / config.rank
        self.dropout = config.dropout
        self.enabled = True
        self.adapter_training = False

    def forward(self, inputs: Tensor) -> Tensor:
        result = self.base(inputs)
        if not self.enabled:
            return result
        x = F.dropout(inputs.to(self.lora_down.weight.dtype), p=self.dropout, training=self.adapter_training)
        delta = self.lora_up(self.lora_down(x)) * self.scale
        return result + delta.to(result.dtype)


def install_qwen_lora(adapter: SpatialBackbone, config: LoRAConfig | None = None) -> None:
    """Replace selected Qwen self-attention projections before optimizer creation."""
    if adapter.lora_config is not None:
        raise ValueError("LoRA is already installed")
    config = config or LoRAConfig()
    layers = adapter.base.language_model.layers
    targets = []
    # Validate the complete scope before changing anything.
    for index in config.layer_indices:
        if index >= len(layers):
            raise ValueError(f"LoRA layer {index} does not exist")
        attention = getattr(layers[index], "self_attn", None)
        for name in config.projections:
            projection = getattr(attention, name, None)
            if not isinstance(projection, nn.Linear):
                raise TypeError(f"layer {index} attention.{name} is not an unwrapped nn.Linear")
            targets.append((index, name, attention, projection))
    modules = [(index, name, attention, LoRALinear(projection, config))
               for index, name, attention, projection in targets]
    for index, name, attention, module in modules:
        setattr(attention, name, module)
    adapter._lora_modules = [(index, name, module) for index, name, _, module in modules]
    adapter.lora_config = asdict(config)


def lora_state_dict(adapter: SpatialBackbone) -> dict[str, Tensor]:
    return {
        f"{index}.{name}.{part}.weight": getattr(module, part).weight.detach().cpu().clone()
        for index, name, module in adapter._lora_modules for part in ("lora_down", "lora_up")
    }


def load_lora_state_dict(adapter: SpatialBackbone, state: dict[str, Tensor]) -> None:
    expected = lora_state_dict(adapter)
    if set(state) != set(expected):
        raise ValueError(f"LoRA checkpoint keys differ: missing={set(expected)-set(state)}, extra={set(state)-set(expected)}")
    with torch.no_grad():
        for index, name, module in adapter._lora_modules:
            for part in ("lora_down", "lora_up"):
                key = f"{index}.{name}.{part}.weight"
                target = getattr(module, part).weight
                if target.shape != state[key].shape:
                    raise ValueError(f"LoRA tensor shape mismatch for {key}")
                target.copy_(state[key])


def set_lora_enabled(adapter: SpatialBackbone, enabled: bool) -> None:
    for _, _, module in adapter._lora_modules:
        module.enabled = enabled
