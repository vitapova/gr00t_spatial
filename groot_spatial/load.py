"""Restore a trained sidecar into its exact LeRobot GR00T base policy."""
from pathlib import Path
from typing import Any
import warnings

import torch
from torch import nn

from . import (LoRAConfig, SpatialAdapterConfig, install_qwen_lora, install_spatial_adapter,
               load_lora_state_dict)
from .interface import install_action_interface, load_interface_state_dict, processor_embodiment
from .train import fingerprint_checkpoint, fingerprint_tree
from .vl_padding import configure_vl_self_attention_mask


def load_trained_sidecar(policy: nn.Module, sidecar_path: str | Path, base_checkpoint: str | Path,
                         *, mask_vl_self_attention: bool | None = None) -> Any:
    """Install weights onto an already loaded policy, verifying the base file hash.

    Load the policy and its processors from the same base_checkpoint before
    calling this function. This function does not load the base policy itself.
    """
    saved = torch.load(sidecar_path, map_location="cpu", weights_only=True)
    if saved.get("format_version") not in (1, 2):
        raise ValueError("unsupported sidecar checkpoint format")
    manifest = saved["manifest"]
    if fingerprint_checkpoint(Path(base_checkpoint)) != manifest["base_sha256"]:
        raise ValueError("sidecar checkpoint was trained on a different base checkpoint")
    spatial = dict(manifest["spatial"]["config"])
    mode = manifest.get("experiment", {"spatial_enabled": True, "lora_enabled": True,
                                       "train_interface": False, "embodiment_id": None})
    recorded_mask = mode.get("mask_vl_self_attention", False)
    if not isinstance(recorded_mask, bool):
        raise ValueError("invalid VL padding mask setting in checkpoint")
    if mask_vl_self_attention is not None and not isinstance(mask_vl_self_attention, bool):
        raise TypeError("mask_vl_self_attention must be bool or None")
    if mask_vl_self_attention is not None and mask_vl_self_attention != recorded_mask:
        warnings.warn("VL padding mask override differs from training checkpoint; "
                      "inference behavior may change", RuntimeWarning, stacklevel=2)
    effective_mask = recorded_mask if mask_vl_self_attention is None else mask_vl_self_attention
    spatial["layer_indices"] = tuple(spatial["layer_indices"])
    policy.requires_grad_(False)
    adapter = install_spatial_adapter(policy, SpatialAdapterConfig(**spatial))
    if adapter.adapter_metadata() != manifest["spatial"]:
        raise ValueError("spatial adapter metadata does not match the loaded policy")
    adapter.enabled = mode["spatial_enabled"]
    if mode["lora_enabled"]:
        lora = dict(manifest["lora"])
        lora["layer_indices"] = tuple(lora["layer_indices"])
        lora["projections"] = tuple(lora["projections"])
        install_qwen_lora(adapter, LoRAConfig(**lora))
    adapter.readers.load_state_dict(saved["spatial"], strict=True)
    load_lora_state_dict(adapter, saved["lora"])
    if mode["train_interface"]:
        if mode["embodiment_id"] is None:
            raise ValueError("interface checkpoint is missing an embodiment ID")
        modules = install_action_interface(policy, mode["embodiment_id"])
        load_interface_state_dict(modules, saved["interface"])
    elif saved.get("interface"):
        raise ValueError("checkpoint contains interface weights but interface training is disabled")
    configure_vl_self_attention_mask(policy, effective_mask)
    policy.eval()
    return adapter


def load_experiment(sidecar_path: str | Path, base_checkpoint: str | Path,
                    run_dir: str | Path, device: str = "cuda", *,
                    mask_vl_self_attention: bool | None = None) -> tuple[Any, Any, Any, Any]:
    """Load the base policy, both matching processors, and the trained sidecar.

    Returns (policy, preprocessor, postprocessor, adapter). This loads weights
    but deliberately does not connect to a robot or issue motor commands.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.groot.modeling_groot import GrootPolicy

    sidecar_path, base, run = Path(sidecar_path), Path(base_checkpoint).resolve(), Path(run_dir)
    saved = torch.load(sidecar_path, map_location="cpu", weights_only=True)
    if saved.get("format_version") not in (1, 2) or fingerprint_checkpoint(base) != saved["manifest"]["base_sha256"]:
        raise ValueError("unsupported sidecar format or mismatched base checkpoint")
    kind = saved["manifest"]["base_kind"]
    if saved["manifest"].get("processor_sha256") is not None:
        if fingerprint_tree(run, ("processors", "policy_config")) != saved["manifest"]["processor_sha256"]:
            raise ValueError("saved processors/policy config do not match this sidecar checkpoint")
        processor_path = run / "processors"
    else:
        processor_path = base
    if kind == "raw":
        if not (run / "policy_config" / "config.json").is_file():
            raise ValueError("raw-base run is missing its saved policy config")
        config = PreTrainedConfig.from_pretrained(str(run / "policy_config"))
        config.base_model_path = str(base)
        config.pretrained_path = None
    elif kind == "lerobot":
        config = PreTrainedConfig.from_pretrained(str(base))
    else:
        raise ValueError(f"unsupported base kind: {kind!r}")
    config.device = device
    config.use_bf16 = torch.device(device).type == "cuda"
    if kind == "raw":
        policy = GrootPolicy(config)
    else:
        policy = GrootPolicy.from_pretrained(str(base), config=config)
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=processor_path,
        preprocessor_overrides={"device_processor": {"device": device},
                                "groot_n1_7_vlm_encode_v1": {"device": device}},
    )
    mode = saved["manifest"].get("experiment", {})
    if mode.get("train_interface"):
        actual_id = processor_embodiment(preprocessor, config.embodiment_tag)
        if actual_id != mode.get("embodiment_id"):
            raise ValueError(f"processor embodiment ID {actual_id} does not match the trained interface")
    adapter = load_trained_sidecar(policy, sidecar_path, base, mask_vl_self_attention=mask_vl_self_attention)
    return policy, preprocessor, postprocessor, adapter
