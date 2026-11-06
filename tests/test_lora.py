import pytest
import torch
from torch import nn

from groot_spatial.lora import LoRAConfig, LoRALinear, install_qwen_lora, load_lora_state_dict, lora_state_dict
from groot_spatial.adapter import SpatialBackbone, SpatialAdapterConfig
from test_spatial_adapter import FakeBackbone


def test_lora_zero_init_gradient_and_autocast():
    torch.manual_seed(0)
    base = nn.Linear(5, 7)
    reference = base(torch.randn(3, 5))
    layer = LoRALinear(base, LoRAConfig(layer_indices=(0,), rank=2))
    x = torch.randn(3, 5)
    torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = layer(x)
        loss = out.square().mean()
    loss.backward()
    assert layer.lora_up.weight.grad.abs().sum() > 0
    assert layer.lora_down.weight.grad is not None
    assert all(p.grad is None for p in base.parameters())
    assert reference.shape == (3, 7)


def test_lora_validation_is_atomic():
    base = FakeBackbone()
    base.language_model.layers[0].self_attn = nn.Module()
    base.language_model.layers[0].self_attn.q_proj = nn.Linear(8, 8)
    wrapper = SpatialBackbone(base, SpatialAdapterConfig(attention_dim=8, num_heads=2))
    with pytest.raises(TypeError):
        install_qwen_lora(wrapper, LoRAConfig(layer_indices=(0,), projections=("q_proj", "o_proj")))
    assert type(base.language_model.layers[0].self_attn.q_proj) is nn.Linear


def test_lora_roundtrip_and_mode():
    base = FakeBackbone()
    for layer in base.language_model.layers:
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(8, 8)
        layer.self_attn.o_proj = nn.Linear(8, 8)
    wrapper = SpatialBackbone(base, SpatialAdapterConfig(attention_dim=8, num_heads=2))
    install_qwen_lora(wrapper, LoRAConfig(layer_indices=(0, 1), rank=2))
    saved = lora_state_dict(wrapper)
    with torch.no_grad():
        wrapper._lora_modules[0][2].lora_up.weight.fill_(1)
    load_lora_state_dict(wrapper, saved)
    assert all(torch.equal(saved[k], v) for k, v in lora_state_dict(wrapper).items())
    with pytest.raises(ValueError):
        load_lora_state_dict(wrapper, {})
