"""Optional padding mask for LeRobot N1.7's VL self-attention projector.

The pinned LeRobot action head does not pass a mask into its VL transformer.
This changes only ``process_backbone_output`` on an already loaded action head;
the parameters, Qwen, DiT, action loss and checkpoint tensor names are untouched.
"""
from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import nn


def configure_vl_self_attention_mask(policy: nn.Module, enabled: bool) -> None:
    """Enable/disable mask for all blocks of the action head's VL transformer.

    In the checked LeRobot version blocks accept an additive attention bias
    [batch, 1, key_length]. Zero means valid and -inf means excluded. This is
    deliberately not a 0/1 float mask: Diffusers attention expects an additive
    bias, not a tokenizer-style mask. Padded queries may still produce outputs,
    but padded positions are excluded as keys in every self-attention layer.
    The standard N1.7 AlternateVLDiT separately masks its cross-attention keys;
    this function does not change the behavior of other action-head variants.
    """
    model = getattr(policy, "_groot_model", None)
    head = getattr(model, "action_head", None)
    if not enabled:
        if head is not None and hasattr(head, "_spatial_original_vl_processing"):
            head.process_backbone_output = head._spatial_original_vl_processing
            delattr(head, "_spatial_original_vl_processing")
        return
    if not isinstance(head, nn.Module):
        raise TypeError("expected LeRobot N1.7 action_head")
    if hasattr(head, "_spatial_original_vl_processing"):
        return
    transformer = getattr(head, "vl_self_attention", None)
    blocks = getattr(transformer, "transformer_blocks", None)
    if not isinstance(blocks, (nn.ModuleList, list, tuple)) or not blocks:
        raise ValueError("VL padding masking requires a nonempty VL self-attention transformer")
    if not callable(getattr(head, "process_backbone_output", None)) or not callable(getattr(head, "vlln", None)):
        raise TypeError("unsupported GR00T action head VL-processing interface")
    for block in blocks:
        if not hasattr(block, "pos_embed"):
            raise TypeError("unsupported VL transformer block layout; positional behavior is unknown")
        if block.pos_embed is not None:
            raise ValueError("VL padding masking does not support positional embeddings in VL self-attention")
        if "attention_mask" not in inspect.signature(block.forward).parameters:
            raise TypeError("VL transformer block does not support attention_mask")

    original = head.process_backbone_output

    def masked_processing(backbone_output: Any) -> Any:
        hidden = head.vlln(backbone_output["backbone_features"])
        mask = backbone_output.get("backbone_attention_mask")
        if not isinstance(mask, torch.Tensor) or hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
            raise ValueError("VL features [B,T,D] and backbone_attention_mask [B,T] are required")
        if mask.device != hidden.device:
            raise ValueError("VL features and padding mask must be on the same device")
        if not torch.all((mask == 0) | (mask == 1)).item():
            raise ValueError("backbone_attention_mask must contain only 0 or 1")
        valid = mask.bool()
        if not valid.any(dim=1).all().item():
            raise ValueError("every example must contain at least one unmasked VL token")
        attention_bias = torch.zeros(valid.shape, device=hidden.device, dtype=hidden.dtype)
        attention_bias = attention_bias.masked_fill(~valid, float("-inf"))[:, None, :]
        hidden = hidden.contiguous()
        for block in transformer.transformer_blocks:
            hidden = block(hidden, attention_mask=attention_bias)
        backbone_output["backbone_features"] = hidden
        return backbone_output

    head._spatial_original_vl_processing = original
    head.process_backbone_output = masked_processing
