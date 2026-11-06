from types import SimpleNamespace

import pytest
import torch
from torch import nn

from demo_arm100 import predict_observation


class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self._groot_model = SimpleNamespace(config=SimpleNamespace(action_horizon=4, max_action_dim=8))
        self.resets = 0

    def reset(self):
        self.resets += 1

    def predict_action_chunk(self, inputs, *, noise):
        assert "action" not in inputs
        assert noise.shape == (1, 4, 8)
        return noise[:, :, :6]


class FakePreprocessor:
    def __init__(self):
        self.resets = 0
        self.last = None

    def reset(self):
        self.resets += 1

    def __call__(self, batch):
        self.last = batch
        assert set(batch) == {"observation.state", "observation.images.front", "observation.images.wrist", "task"}
        assert batch["observation.images.front"].shape == (1, 3, 4, 5)
        assert batch["observation.images.front"].dtype == torch.float32
        assert batch["observation.images.front"].max() == 1
        assert batch["task"] == ["Move the cube"]
        return {"state": batch["observation.state"]}


def sample():
    return {"observation.state": torch.arange(6), "observation.images.front": torch.full((3, 4, 5), 255, dtype=torch.uint8),
            "observation.images.wrist": torch.zeros((3, 4, 5)), "task": "Move the cube",
            "action": torch.full((4, 6), 1000.0)}


def test_predict_is_reproducible_and_does_not_leak_actions():
    policy, pre = FakePolicy(), FakePreprocessor()
    calls = []
    def post(action):
        calls.append(action)
        return action * 2 + 3
    first = predict_observation(policy, pre, post, sample(), seed=123)
    second = predict_observation(policy, pre, post, sample(), seed=123)
    third = predict_observation(policy, pre, post, sample(), seed=124)
    assert first.shape == (4, 6)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, third)
    assert policy.resets == pre.resets == 3 and len(calls) == 3
    assert first.device.type == "cpu"


def test_predict_validates_observation():
    policy, pre = FakePolicy(), FakePreprocessor()
    obs = sample()
    obs["observation.images.front"] = torch.full((3, 4, 5), 255.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        predict_observation(policy, pre, lambda x: x, obs)
    obs = sample()
    obs["observation.state"] = torch.zeros(5)
    with pytest.raises(ValueError, match="six finite"):
        predict_observation(policy, pre, lambda x: x, obs)
