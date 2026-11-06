"""Train exactly one embodiment's N1.7 state/action projection weights."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


INTERFACE_PATHS = (
    "state_encoder.layer1", "state_encoder.layer2",
    "action_encoder.W1", "action_encoder.W2", "action_encoder.W3",
    "action_decoder.layer1", "action_decoder.layer2",
)


class SingleEmbodimentLinear(nn.Module):
    """Original frozen bank plus a full trainable copy of one selected row.

    Other IDs are computed with the unchanged base bank. Keeping the selected
    row as a separate Parameter avoids optimizer state and AdamW decay for the
    other rows. Both paths retain the original per-example bmm semantics.
    """

    def __init__(self, base: nn.Module, embodiment_id: int):
        super().__init__()
        weight, bias = getattr(base, "W", None), getattr(base, "b", None)
        if not isinstance(weight, nn.Parameter) or not isinstance(bias, nn.Parameter):
            raise TypeError("expected CategorySpecificLinear with W and b parameters")
        if weight.ndim != 3 or bias.ndim != 2 or weight.shape[0] != bias.shape[0] or weight.shape[2] != bias.shape[1]:
            raise ValueError("unexpected category-specific linear shape")
        if not 0 <= embodiment_id < weight.shape[0]:
            raise ValueError(f"embodiment ID {embodiment_id} is outside the weight bank")
        self.base = base.requires_grad_(False)
        self.embodiment_id = embodiment_id
        # Follow the FP32 sidecar policy; cast to the base dtype on use.
        self.slot_W = nn.Parameter(weight[embodiment_id].detach().float().clone())
        self.slot_b = nn.Parameter(bias[embodiment_id].detach().float().clone())

    def forward(self, x: Tensor, cat_ids: Tensor) -> Tensor:
        selected_w = self.base.W[cat_ids]
        selected_b = self.base.b[cat_ids]
        active = cat_ids == self.embodiment_id
        selected_w = torch.where(active[:, None, None], self.slot_W.to(selected_w.dtype)[None], selected_w)
        selected_b = torch.where(active[:, None], self.slot_b.to(selected_b.dtype)[None], selected_b)
        return torch.bmm(x, selected_w) + selected_b.unsqueeze(1)


def install_action_interface(policy: nn.Module, embodiment_id: int) -> dict[str, SingleEmbodimentLinear]:
    """Replace the seven category-specific projections on an already loaded policy."""
    model = getattr(policy, "_groot_model", None)
    head = getattr(model, "action_head", None)
    if not isinstance(head, nn.Module):
        raise TypeError("expected LeRobot GR00T N1.7 action_head")
    targets: list[tuple[str, nn.Module, str, nn.Module]] = []
    for path in INTERFACE_PATHS:
        parent_name, child_name = path.split(".")
        parent = getattr(head, parent_name, None)
        original = getattr(parent, child_name, None)
        if not isinstance(parent, nn.Module) or not isinstance(original, nn.Module):
            raise TypeError(f"missing action head projection {path}")
        if isinstance(original, SingleEmbodimentLinear):
            raise ValueError("action interface is already installed")
        targets.append((path, parent, child_name, original))
    # Validate and allocate all wrappers before altering the action head.
    wrapped = [(path, parent, name, SingleEmbodimentLinear(original, embodiment_id))
               for path, parent, name, original in targets]
    for _, parent, name, module in wrapped:
        setattr(parent, name, module)
    return {path: module for path, _, _, module in wrapped}


def interface_state_dict(modules: dict[str, SingleEmbodimentLinear]) -> dict[str, Tensor]:
    return {f"{path}.{name}": getattr(module, name).detach().cpu().clone()
            for path, module in modules.items() for name in ("slot_W", "slot_b")}


def load_interface_state_dict(modules: dict[str, SingleEmbodimentLinear], state: dict[str, Tensor]) -> None:
    expected = {f"{path}.{name}": getattr(module, name) for path, module in modules.items()
                for name in ("slot_W", "slot_b")}
    if set(state) != set(expected):
        raise ValueError(f"interface checkpoint keys differ: missing={set(expected)-set(state)}, extra={set(state)-set(expected)}")
    for key, parameter in expected.items():
        if parameter.shape != state[key].shape:
            raise ValueError(f"interface tensor shape mismatch for {key}")
    with torch.no_grad():
        for key, parameter in expected.items():
            parameter.copy_(state[key])


def configure_processor_embodiment(processor: Any, tag: str, embodiment_id: int) -> None:
    """Persist an explicit new-embodiment mapping in a GR00T processor."""
    matches = [step for step in processor.steps if getattr(step, "embodiment_tag", None) == tag
               and hasattr(step, "embodiment_mapping")]
    if len(matches) != 1:
        raise ValueError(f"expected one GR00T pack step for embodiment {tag!r}, got {len(matches)}")
    step = matches[0]
    mapping = dict(step.embodiment_mapping)
    mapping[tag] = embodiment_id
    step.embodiment_mapping = mapping


def processor_embodiment(processor: Any, tag: str) -> int | None:
    matches = [step for step in processor.steps if getattr(step, "embodiment_tag", None) == tag
               and hasattr(step, "embodiment_mapping")]
    if len(matches) != 1:
        raise ValueError(f"expected one GR00T pack step for embodiment {tag!r}, got {len(matches)}")
    value = matches[0].embodiment_mapping.get(tag)
    return int(value) if value is not None else None
