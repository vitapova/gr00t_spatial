from types import SimpleNamespace

import pytest
import torch
from torch import nn

from groot_spatial.interface import (SingleEmbodimentLinear, configure_processor_embodiment,
                                     install_action_interface, interface_state_dict,
                                     load_interface_state_dict, processor_embodiment)


class CategoryLinear(nn.Module):
    def __init__(self, cats=4, inp=3, out=5):
        super().__init__()
        self.W = nn.Parameter(torch.randn(cats, inp, out))
        self.b = nn.Parameter(torch.randn(cats, out))

    def forward(self, x, ids):
        return torch.bmm(x, self.W[ids]) + self.b[ids].unsqueeze(1)


def build_policy():
    policy = nn.Module()
    policy._groot_model = nn.Module()
    head = nn.Module()
    policy._groot_model.action_head = head
    for name, layers in (("state_encoder", ("layer1", "layer2")),
                         ("action_encoder", ("W1", "W2", "W3")),
                         ("action_decoder", ("layer1", "layer2"))):
        parent = nn.Module()
        setattr(head, name, parent)
        for layer in layers:
            setattr(parent, layer, CategoryLinear())
    return policy


def test_single_slot_parity_gradients_and_other_rows():
    torch.manual_seed(11)
    original = CategoryLinear()
    reference_w = original.W.detach().clone()
    reference_b = original.b.detach().clone()
    module = SingleEmbodimentLinear(original, 2)
    x = torch.randn(2, 4, 3)
    ids = torch.tensor([2, 1])
    expected = original(x, ids)
    torch.testing.assert_close(module(x, ids), expected, rtol=0, atol=0)
    optimizer = torch.optim.AdamW([module.slot_W, module.slot_b], lr=0.01, weight_decay=0.1)
    module(x, ids).square().sum().backward()
    assert module.slot_W.grad.abs().sum() > 0
    assert module.slot_b.grad.abs().sum() > 0
    assert original.W.grad is None and original.b.grad is None
    optimizer.step()
    assert torch.equal(original.W, reference_w) and torch.equal(original.b, reference_b)
    result = module(x, ids)
    assert not torch.equal(result[0], expected[0])
    torch.testing.assert_close(result[1], expected[1], rtol=0, atol=0)


def test_install_and_state_roundtrip_atomicity():
    policy = build_policy()
    policy.requires_grad_(False)
    modules = install_action_interface(policy, 2)
    assert len(modules) == 7 and len(interface_state_dict(modules)) == 14
    state = interface_state_dict(modules)
    with torch.no_grad():
        modules["action_decoder.layer2"].slot_b.add_(10)
    load_interface_state_dict(modules, state)
    assert all(torch.equal(state[k], v) for k, v in interface_state_dict(modules).items())
    with pytest.raises(ValueError, match="already installed"):
        install_action_interface(policy, 2)
    invalid = dict(state)
    invalid["action_decoder.layer2.slot_b"] = torch.zeros(1)
    before = interface_state_dict(modules)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_interface_state_dict(modules, invalid)
    assert all(torch.equal(before[k], v) for k, v in interface_state_dict(modules).items())
    clean = build_policy()
    with pytest.raises(ValueError, match="outside"):
        install_action_interface(clean, 10)
    assert isinstance(clean._groot_model.action_head.state_encoder.layer1, CategoryLinear)


def test_single_slot_bfloat16_autocast():
    base = CategoryLinear()
    module = SingleEmbodimentLinear(base, 2)
    x = torch.randn(2, 4, 3)
    ids = torch.tensor([2, 1])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = base(x, ids)
        result = module(x, ids)
        loss = result.square().mean()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    loss.backward()
    assert module.slot_W.grad is not None and torch.isfinite(module.slot_W.grad).all()


def test_processor_mapping():
    step = SimpleNamespace(embodiment_tag="new_embodiment", embodiment_mapping={"other": 0})
    proc = SimpleNamespace(steps=[step])
    assert processor_embodiment(proc, "new_embodiment") is None
    configure_processor_embodiment(proc, "new_embodiment", 3)
    assert processor_embodiment(proc, "new_embodiment") == 3
    assert step.embodiment_mapping["other"] == 0
