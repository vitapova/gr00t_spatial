"""Add a spatial visual reader to a LeRobot GR00T N1.7 Qwen backbone.

The adapter is a sibling of the original backbone, not a replacement for the
Qwen image-token path, DeepStack, or the GR00T action head. It captures the
last vision block *before* the merger and injects gated residuals into selected
language decoder layer inputs. Only image inputs (not video) are supported.
"""

from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True)
class SpatialAdapterConfig:
    layer_indices: tuple[int, ...] = (0, 1)
    attention_dim: int = 512
    num_heads: int = 8
    max_images: int = 8
    gate_init: float = 0.01
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if not self.layer_indices or len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("layer_indices must be nonempty and unique")
        if any(i < 0 for i in self.layer_indices):
            raise ValueError("layer indices must be nonnegative")
        if self.attention_dim <= 0 or self.num_heads <= 0 or self.attention_dim % self.num_heads:
            raise ValueError("attention_dim must be positive and divisible by num_heads")
        if self.max_images <= 0 or not 0 < self.gate_init < 1 or not 0 <= self.dropout < 1:
            raise ValueError("invalid max_images, gate_init, or dropout")


def _unpack_image(patches: Tensor, height: int, width: int, merge: int) -> Tensor:
    """Convert Qwen's merge-block patch order to [height, width, channels]."""
    if height <= 0 or width <= 0 or height % merge or width % merge:
        raise ValueError("image grid must be positive and divisible by spatial_merge_size")
    if patches.ndim != 2 or patches.shape[0] != height * width:
        raise ValueError("vision patch count does not match image_grid_thw")
    return (
        patches.reshape(height // merge, width // merge, merge, merge, -1)
        .permute(0, 2, 1, 3, 4)
        .reshape(height, width, -1)
    )


def _coordinates(height: int, width: int, device: torch.device) -> Tensor:
    """Normalised row/column coordinates in row-major spatial order."""
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((yy, xx), dim=-1).reshape(-1, 2)


def _pack_workspace(
    patches: Tensor,
    grid_thw: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
    image_token_id: int,
    merge: int,
    max_images: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Map packed Qwen vision patches to padded per-example workspaces.

    A key is visible only from the last placeholder of its image onward.
    This conservative rule preserves autoregressive causality while allowing
    later text positions to attend to all patches of preceding images.
    """
    if patches.ndim != 2 or grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
        raise ValueError("expected packed vision [N,D] and image_grid_thw [images,3]")
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must both have shape [B,T]")
    if merge <= 0:
        raise ValueError("invalid spatial merge size")
    grids = grid_thw.detach().cpu().tolist()
    rows = []
    coords = []
    camera_ids = []
    visible_from = []
    image_index = 0
    patch_offset = 0
    for ids, valid in zip(input_ids, attention_mask):
        positions = ((ids == image_token_id) & valid.bool()).nonzero(as_tuple=True)[0]
        positions_list = positions.detach().cpu().tolist()
        cursor = 0
        image_in_sample = 0
        sample_rows, sample_coords, sample_cameras, sample_starts = [], [], [], []
        while cursor < len(positions_list):
            if image_index >= len(grids) or image_in_sample >= max_images:
                raise ValueError("image placeholders do not match grids or exceed max_images")
            t, h, w = (int(n) for n in grids[image_index])
            if t != 1 or h <= 0 or w <= 0 or h % merge or w % merge:
                raise ValueError("only single-frame image grids divisible by merge size are supported")
            placeholders = h * w // (merge * merge)
            span = positions_list[cursor : cursor + placeholders]
            if len(span) != placeholders or span != list(range(span[0], span[0] + placeholders)):
                raise ValueError("image placeholder positions do not match grid / merge layout")
            count = h * w
            frame = _unpack_image(patches[patch_offset : patch_offset + count], h, w, merge)
            sample_rows.append(frame.reshape(count, -1))
            sample_coords.append(_coordinates(h, w, patches.device))
            sample_cameras.append(torch.full((count,), image_in_sample, device=patches.device, dtype=torch.long))
            sample_starts.append(torch.full((count,), span[-1], device=patches.device, dtype=torch.long))
            patch_offset += count
            image_index += 1
            image_in_sample += 1
            cursor += placeholders
        if not sample_rows:
            raise ValueError("every example must have at least one image")
        rows.append(torch.cat(sample_rows))
        coords.append(torch.cat(sample_coords))
        camera_ids.append(torch.cat(sample_cameras))
        visible_from.append(torch.cat(sample_starts))
    if image_index != len(grids) or patch_offset != patches.shape[0]:
        raise ValueError("unused image grids or vision patches; refusing ambiguous batch")
    lengths = torch.tensor([row.shape[0] for row in rows], device=patches.device)
    values = pad_sequence(rows, batch_first=True)
    xy = pad_sequence(coords, batch_first=True)
    cameras = pad_sequence(camera_ids, batch_first=True)
    starts = pad_sequence(visible_from, batch_first=True)
    key_valid = torch.arange(values.shape[1], device=patches.device)[None, :] < lengths[:, None]
    return values, xy, cameras, starts, key_valid


class SpatialReader(nn.Module):
    """Position-aware, gated cross-attention from Qwen states to vision patches."""

    def __init__(self, qwen_dim: int, vision_dim: int, config: SpatialAdapterConfig):
        super().__init__()
        dim = config.attention_dim
        self.num_heads = config.num_heads
        self.head_dim = dim // config.num_heads
        self.dropout = config.dropout
        self.query_norm = nn.LayerNorm(qwen_dim)
        self.vision_norm = nn.LayerNorm(vision_dim)
        self.query = nn.Linear(qwen_dim, dim, bias=False)
        self.key = nn.Linear(vision_dim, dim, bias=False)
        self.value = nn.Linear(vision_dim, dim, bias=False)
        self.position = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.camera = nn.Embedding(config.max_images, dim)
        self.output = nn.Linear(dim, qwen_dim, bias=False)
        self.gate_logit = nn.Parameter(torch.tensor(math.log(config.gate_init / (1 - config.gate_init))))

    def forward(
        self, hidden: Tensor, workspace: tuple[Tensor, Tensor, Tensor, Tensor, Tensor], query_valid: Tensor
    ) -> Tensor:
        values, xy, cameras, starts, key_valid = workspace
        batch, seq, _ = hidden.shape
        if values.shape[0] != batch or query_valid.shape != (batch, seq):
            raise ValueError("workspace and query batch/sequence dimensions differ")
        dtype = self.query.weight.dtype
        q = self.query(self.query_norm(hidden.to(dtype)))
        visual = self.vision_norm(values.to(dtype))
        spatial = self.position(xy.to(dtype)) + self.camera(cameras)
        k = self.key(visual)
        v = self.value(visual)
        # Embedding weights stay FP32 under autocast; SDPA requires Q/K/V to
        # have the same dtype. Avoid promoting K/V back to FP32 when adding it.
        k = k + spatial.to(k.dtype)
        v = v + spatial.to(v.dtype)
        n = values.shape[1]
        q = q.reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, n, self.num_heads, self.head_dim).transpose(1, 2)
        positions = torch.arange(seq, device=hidden.device)[None, :, None]
        allowed = key_valid[:, None, :] & (positions >= starts[:, None, :]) & query_valid.bool()[:, :, None]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed[:, None], dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        out = self.output(out)
        # Explicitly zero queries without visible images (including padding).
        out = out * allowed.any(dim=-1, keepdim=True).to(out.dtype)
        return (torch.sigmoid(self.gate_logit) * out).to(hidden.dtype)


class SpatialBackbone(nn.Module):
    """Wrap the LeRobot N1.7 backbone. Not reentrant; checkpointing is rejected."""

    def __init__(self, backbone: nn.Module, config: SpatialAdapterConfig):
        super().__init__()
        if not all(hasattr(backbone, name) for name in ("model", "visual", "language_model", "prepare_input")):
            raise TypeError("expected LeRobot N1.7 Qwen3Backbone")
        if not hasattr(backbone.visual, "blocks") or not hasattr(backbone.language_model, "layers"):
            raise TypeError("unsupported Qwen3-VL vision/decoder layout")
        layers = backbone.language_model.layers
        if not backbone.visual.blocks or any(i >= len(layers) for i in config.layer_indices):
            raise ValueError("vision blocks or requested decoder layers do not exist")
        self.base = backbone
        self.config = config
        self.enabled = True
        # Modules are registered through the original Qwen layers; this list
        # is only an index and does not register them a second time.
        self._lora_modules = []
        self.lora_config = None
        self._forward_lock = threading.Lock()
        qconfig = backbone.model.config
        vision_dim = int(qconfig.vision_config.hidden_size)
        qwen_dim = int(qconfig.text_config.hidden_size)
        self.merge = int(qconfig.vision_config.spatial_merge_size)
        self.image_token_id = int(qconfig.image_token_id)
        self.readers = nn.ModuleDict({
            str(i): SpatialReader(qwen_dim, vision_dim, config) for i in config.layer_indices
        })
        # Match the device but keep FP32 trainable adapter weights by default.
        self.readers.to(device=next(backbone.parameters()).device)

    def prepare_input(self, batch: dict[str, Any]) -> Any:
        return self.base.prepare_input(batch)

    def adapter_state_dict(self) -> dict[str, Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.readers.state_dict().items()}

    def adapter_metadata(self) -> dict[str, Any]:
        return {"format_version": 1, "config": asdict(self.config), "merge": self.merge,
                "image_token_id": self.image_token_id}

    def forward(self, vl_input: Any) -> Any:
        for _, _, module in self._lora_modules:
            module.adapter_training = self.training
        if not self.enabled:
            return self.base(vl_input)
        if not self._forward_lock.acquire(blocking=False):
            raise RuntimeError("SpatialBackbone does not support concurrent or nested forwards")
        handles = []
        try:
            if any(vl_input.get(k) is not None for k in ("pixel_values_videos", "video_grid_thw")):
                raise ValueError("spatial adapter does not support video inputs")
            if torch.is_grad_enabled() and self.training and any(
                getattr(m, "gradient_checkpointing", False) and m.training for m in self.base.modules()
            ):
                raise RuntimeError("disable gradient checkpointing for the Qwen backbone before training")
            ids = vl_input["input_ids"]
            mask = vl_input["attention_mask"]
            grid = vl_input["image_grid_thw"]
            captured: dict[str, Any] = {}

            def capture(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
                if not isinstance(output, Tensor):
                    raise TypeError("expected a tensor from the last Qwen vision block")
                captured["workspace"] = _pack_workspace(
                    output, grid, ids, mask, self.image_token_id, self.merge, self.config.max_images
                )

            def inject(index: int):
                def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
                    if "workspace" not in captured:
                        raise RuntimeError("vision encoder did not produce a workspace before the decoder")
                    if args:
                        hidden = args[0]
                    else:
                        hidden = kwargs.get("hidden_states")
                    if not isinstance(hidden, Tensor):
                        raise TypeError("decoder hidden_states must be a tensor")
                    hidden = hidden + self.readers[str(index)](hidden, captured["workspace"], mask)
                    if args:
                        return (hidden, *args[1:]), kwargs
                    return args, {**kwargs, "hidden_states": hidden}
                return hook

            handles.append(self.base.visual.blocks[-1].register_forward_hook(capture))
            for index in self.config.layer_indices:
                handles.append(self.base.language_model.layers[index].register_forward_pre_hook(
                    inject(index), with_kwargs=True
                ))
            return self.base(vl_input)
        finally:
            for handle in handles:
                handle.remove()
            self._forward_lock.release()


def install_spatial_adapter(policy: nn.Module, config: SpatialAdapterConfig | None = None) -> SpatialBackbone:
    """Install into an already loaded LeRobot GrootPolicy; call before optimizer creation.

    This does not change base requires_grad flags. Freeze base parameters first
    when performing an adapter-only experiment.
    """
    model = getattr(policy, "_groot_model", None)
    if model is None or not hasattr(model, "backbone"):
        raise TypeError("expected a LeRobot GrootPolicy with _groot_model.backbone")
    if isinstance(model.backbone, SpatialBackbone):
        raise ValueError("a spatial adapter is already installed")
    adapter = SpatialBackbone(model.backbone, config or SpatialAdapterConfig())
    model.backbone = adapter
    return adapter
