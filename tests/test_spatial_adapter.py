import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from groot_spatial.adapter import (
    SpatialAdapterConfig,
    SpatialBackbone,
    SpatialReader,
    _pack_workspace,
    _unpack_image,
    install_spatial_adapter,
)


def test_unpack_merge_blocks():
    # Qwen order: blocks of 2x2, each block row-major, not global row-major.
    patches = torch.tensor([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15])[:, None]
    assert torch.equal(_unpack_image(patches, 4, 4, 2).flatten(), torch.arange(16))


def test_pack_multicamera_and_padding():
    # Sample zero has two 2x2 images; sample one has one 4x2 image.
    patches = torch.arange(16.0)[:, None]
    ids = torch.tensor([[0, 99, 4, 99, 5, 0], [0, 99, 99, 5, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]])
    grid = torch.tensor([[1, 2, 2], [1, 2, 2], [1, 4, 2]])
    values, xy, cameras, starts, valid = _pack_workspace(patches, grid, ids, mask, 99, 2, 4)
    assert values.shape == (2, 8, 1)
    assert torch.equal(values[0, :, 0], torch.arange(8.0))
    assert torch.equal(values[1, :, 0], torch.arange(8.0, 16.0))
    assert cameras[0].tolist() == [0] * 4 + [1] * 4
    assert starts[0].tolist() == [1] * 4 + [3] * 4
    assert starts[1].tolist() == [2] * 8
    assert valid.all() and torch.all((xy > 0) & (xy < 1))


def test_pack_rejects_wrong_grids_and_missing_images():
    with pytest.raises(ValueError):
        _pack_workspace(torch.ones(4, 3), torch.tensor([[1, 2, 2]]),
                        torch.tensor([[1, 2]]), torch.ones(1, 2), 99, 2, 4)
    with pytest.raises(ValueError):
        _pack_workspace(torch.ones(3, 3), torch.tensor([[1, 2, 2]]),
                        torch.tensor([[99]]), torch.ones(1, 1), 99, 2, 4)


def test_workspace_padding_not_visible():
    patches = torch.randn(20, 6)
    ids = torch.tensor([[99, 7, 0, 0, 0], [99, 99, 99, 99, 7]])
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
    workspace = _pack_workspace(patches, torch.tensor([[1, 2, 2], [1, 4, 4]]), ids, mask, 99, 2, 4)
    assert workspace[4][0].tolist() == [True] * 4 + [False] * 12
    reader = SpatialReader(8, 6, SpatialAdapterConfig(attention_dim=8, num_heads=2)).eval()
    hidden = torch.randn(2, 5, 8)
    original = reader(hidden, workspace, mask)
    vals = workspace[0].clone()
    vals[0, 4:] = torch.randn_like(vals[0, 4:]) * 100
    changed = reader(hidden, (vals, *workspace[1:]), mask)
    torch.testing.assert_close(original, changed)


def test_reader_causality_and_gradient():
    torch.manual_seed(2)
    config = SpatialAdapterConfig(layer_indices=(0,), attention_dim=8, num_heads=2)
    reader = SpatialReader(12, 6, config)
    patches = torch.randn(8, 6, requires_grad=True)
    ids = torch.tensor([[3, 99, 4, 99, 5, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    grid = torch.tensor([[1, 2, 2], [1, 2, 2]])
    ws = _pack_workspace(patches, grid, ids, mask, 99, 2, 4)
    hidden = torch.randn(1, 6, 12, requires_grad=True)
    out = reader(hidden, ws, mask)
    assert torch.equal(out[:, 0], torch.zeros_like(out[:, 0]))
    assert torch.equal(out[:, 5], torch.zeros_like(out[:, 5]))
    changed = patches.detach().clone()
    changed[4:] = torch.randn_like(changed[4:])
    ws_changed = _pack_workspace(changed, grid, ids, mask, 99, 2, 4)
    out_changed = reader(hidden, ws_changed, mask)
    torch.testing.assert_close(out[:, :3], out_changed[:, :3])
    assert not torch.allclose(out[:, 4], out_changed[:, 4])
    out.square().sum().backward()
    assert patches.grad is not None and patches.grad.abs().sum() > 0
    assert hidden.grad is not None and reader.gate_logit.grad is not None


def test_reader_cpu_bfloat16_autocast():
    reader = SpatialReader(12, 6, SpatialAdapterConfig(attention_dim=8, num_heads=2))
    ids = torch.tensor([[1, 99, 2]])
    mask = torch.ones_like(ids)
    workspace = _pack_workspace(torch.randn(4, 6), torch.tensor([[1, 2, 2]]), ids, mask, 99, 2, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = reader(torch.randn(1, 3, 12), workspace, mask)
        loss = out.square().sum()
    loss.backward()
    assert torch.isfinite(out).all() and reader.query.weight.grad is not None


class FakeVisionBlock(nn.Module):
    def forward(self, x):
        return x * 2


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 8)

    def forward(self, hidden_states):
        return hidden_states + self.layer(hidden_states).tanh()


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList([FakeVisionBlock()])
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([FakeDecoder(), FakeDecoder()])
        self.embedding = nn.Embedding(100, 8)
        self.model = SimpleNamespace(config=SimpleNamespace(
            image_token_id=99, vision_config=SimpleNamespace(hidden_size=4, spatial_merge_size=2),
            text_config=SimpleNamespace(hidden_size=8)))

    def prepare_input(self, x):
        return x

    def forward(self, inputs):
        # Mimics Qwen: vision is computed before decoder, DeepStack omitted.
        self.visual.blocks[-1](inputs["patches"])
        hidden = self.embedding(inputs["input_ids"])
        for layer in self.language_model.layers:
            hidden = layer(hidden)
        return hidden


def test_adapter_hooks_train_save_and_disable(tmp_path):
    torch.manual_seed(8)
    base = FakeBackbone().requires_grad_(False)
    wrapper = SpatialBackbone(base, SpatialAdapterConfig(attention_dim=8, num_heads=2))
    inputs = {"patches": torch.randn(4, 4), "input_ids": torch.tensor([[1, 99, 2, 0]]),
              "attention_mask": torch.tensor([[1, 1, 1, 0]]), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    expected = base(inputs)
    wrapper.enabled = False
    torch.testing.assert_close(wrapper(inputs), expected, rtol=0, atol=0)
    wrapper.enabled = True
    output = wrapper(inputs)
    assert not torch.allclose(output[:, 2], expected[:, 2], atol=1e-7)
    output.square().sum().backward()
    assert all(p.grad is None for p in base.parameters())
    assert all(p.grad is not None for p in wrapper.readers.parameters())
    checkpoint = tmp_path / "adapter.pt"
    torch.save({"metadata": wrapper.adapter_metadata(), "weights": wrapper.adapter_state_dict()}, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    clone = SpatialBackbone(copy.deepcopy(base), wrapper.config)
    assert saved["metadata"] == clone.adapter_metadata()
    clone.readers.load_state_dict(saved["weights"], strict=True)
    torch.testing.assert_close(wrapper(inputs), clone(inputs))
    assert len(base.visual.blocks[-1]._forward_hooks) == 0
    assert all(len(layer._forward_pre_hooks) == 0 for layer in base.language_model.layers)


def test_installer_and_guards():
    base = FakeBackbone()
    policy = nn.Module()
    policy._groot_model = nn.Module()
    policy._groot_model.backbone = base
    wrapper = install_spatial_adapter(policy, SpatialAdapterConfig(attention_dim=8, num_heads=2))
    assert policy._groot_model.backbone is wrapper
    with pytest.raises(ValueError, match="already installed"):
        install_spatial_adapter(policy)
    base.visual.gradient_checkpointing = True
    with pytest.raises(RuntimeError, match="checkpointing"):
        wrapper({})
    base.visual.gradient_checkpointing = False
    with pytest.raises(ValueError, match="video"):
        wrapper({"pixel_values_videos": torch.zeros(1)})


def test_real_qwen3_vl_tiny():
    transformers = pytest.importorskip("transformers")
    # Audio is irrelevant for Qwen3-VL and may be broken in local environments.
    transformers.utils.import_utils._librosa_available = False
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        image_token_id=99, video_token_id=98, vision_start_token_id=97, vision_end_token_id=96,
        text_config={"vocab_size": 128, "hidden_size": 16, "intermediate_size": 32,
                     "num_hidden_layers": 3, "num_attention_heads": 2, "num_key_value_heads": 2,
                     "head_dim": 8, "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
        vision_config={"depth": 2, "hidden_size": 8, "out_hidden_size": 16, "intermediate_size": 16,
                       "num_heads": 2, "patch_size": 2, "temporal_patch_size": 1,
                       "spatial_merge_size": 2, "num_position_embeddings": 16,
                       "deepstack_visual_indexes": [0]},
    )
    qwen = Qwen3VLForConditionalGeneration(config).eval()

    class QwenBackbone(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        @property
        def visual(self):
            return self.model.model.visual
        @property
        def language_model(self):
            return self.model.model.language_model
        def prepare_input(self, batch):
            return batch
        def forward(self, batch):
            return self.model(**batch, output_hidden_states=True).hidden_states[-1]

    base = QwenBackbone(qwen).requires_grad_(False)
    wrapper = SpatialBackbone(base, SpatialAdapterConfig(layer_indices=(0, 1), attention_dim=8, num_heads=2))
    batch = {"input_ids": torch.tensor([[10, 97, 99, 99, 99, 99, 96, 11]]),
             "attention_mask": torch.ones(1, 8, dtype=torch.long),
             "image_grid_thw": torch.tensor([[1, 4, 4]]),
             "pixel_values": torch.randn(16, 12)}
    if int(transformers.__version__.split(".")[0]) >= 5:
        batch["mm_token_type_ids"] = (batch["input_ids"] == 99).int()
    with torch.no_grad():
        reference = base(batch)
        wrapper.enabled = False
        torch.testing.assert_close(wrapper(batch), reference, rtol=0, atol=0)
    wrapper.enabled = True
    result = wrapper(batch)
    assert result.shape == reference.shape
    assert not torch.allclose(result[:, -1], reference[:, -1])
    result[:, -1].square().sum().backward()
    assert all(param.grad is None for param in base.parameters())
    assert wrapper.readers["0"].query.weight.grad.abs().sum() > 0
    assert wrapper.readers["1"].value.weight.grad.abs().sum() > 0

    # Two batch elements, with two cameras in the second element and padding
    # in the first. Changing sample 0's image must not affect sample 1.
    two_samples = {
        "input_ids": torch.tensor([[10, 97, 99, 99, 99, 99, 96, 11, 0, 0, 0, 0, 0, 0],
                                   [10, 97, 99, 99, 99, 99, 96, 97, 99, 99, 99, 99, 96, 11]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                        [1] * 14]),
        "image_grid_thw": torch.tensor([[1, 4, 4]] * 3),
        "pixel_values": torch.randn(48, 12),
    }
    if int(transformers.__version__.split(".")[0]) >= 5:
        two_samples["mm_token_type_ids"] = (two_samples["input_ids"] == 99).int()
    wrapper.eval()
    with torch.no_grad():
        output = wrapper(two_samples)
        modified = dict(two_samples)
        modified["pixel_values"] = two_samples["pixel_values"].clone()
        modified["pixel_values"][:16] = torch.randn(16, 12)
        output_changed = wrapper(modified)
    torch.testing.assert_close(output[1], output_changed[1])
    assert not torch.allclose(output[0, 7], output_changed[0, 7])
