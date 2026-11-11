import random
from dataclasses import asdict

import numpy as np
import pytest
import torch
from torch import nn

from groot_spatial import LoRAConfig, SpatialAdapterConfig, install_qwen_lora, install_spatial_adapter
from groot_spatial.load import load_trained_sidecar
from groot_spatial.train import (episode_split, fingerprint_checkpoint, restore_checkpoint,
                                 prepare_output_directory, read_episode_split, save_checkpoint, train_step,
                                 validate_output_directory)
from test_spatial_adapter import FakeBackbone


def make_policy():
    base = FakeBackbone()
    for layer in base.language_model.layers:
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(8, 8)
        layer.self_attn.o_proj = nn.Linear(8, 8)
    policy = nn.Module()
    policy._groot_model = nn.Module()
    policy._groot_model.backbone = base
    policy.requires_grad_(False)
    return policy


def test_episode_split():
    train, evaluate = episode_split(194, 0.1, 42)
    assert set(train).isdisjoint(evaluate)
    assert sorted(train + evaluate) == list(range(194))
    assert (train, evaluate) == episode_split(194, 0.1, 42)
    with pytest.raises(ValueError):
        episode_split(1, 0.1, 42)


def test_explicit_split(tmp_path):
    path = tmp_path / "split.json"
    path.write_text('{"train":[3,1], "eval":[2]}')
    assert read_episode_split(path, 5) == ([1, 3], [2])
    path.write_text('{"train":[1,2], "eval":[2]}')
    with pytest.raises(ValueError, match="disjoint"):
        read_episode_split(path, 5)


def test_force_overwrite_output_directory(tmp_path):
    dataset = tmp_path / "data"
    base = tmp_path / "base"
    output = tmp_path / "output"
    for path in (dataset, base, output):
        path.mkdir()
    sentinel = output / "old-checkpoint.pt"
    sentinel.write_text("previous run")
    with pytest.raises(ValueError, match="already exists"):
        validate_output_directory(output, dataset, base, None, False)
    assert sentinel.read_text() == "previous run"
    validated = validate_output_directory(output, dataset, base, None, True)
    prepare_output_directory(validated, True)
    assert output.is_dir() and not sentinel.exists()
    assert dataset.is_dir() and base.is_dir()


def test_overwrite_protects_sources_and_resume(tmp_path):
    dataset = tmp_path / "data"
    base = tmp_path / "base"
    output = tmp_path / "output"
    for path in (dataset, base, output):
        path.mkdir()
    checkpoint = output / "step.pt"
    checkpoint.write_text("checkpoint")
    for forbidden in (tmp_path, dataset, dataset / "nested", base, base / "nested", output):
        with pytest.raises(ValueError):
            validate_output_directory(forbidden, dataset, base, checkpoint, True)
    link = tmp_path / "link"
    link.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        validate_output_directory(link, dataset, base, None, True)
    assert checkpoint.read_text() == "checkpoint"


def test_checkpoint_restores_weights_optimizer_and_rng(tmp_path):
    torch.manual_seed(3)
    random.seed(3)
    np.random.seed(3)
    scfg = SpatialAdapterConfig(layer_indices=(0,), attention_dim=8, num_heads=2)
    lcfg = LoRAConfig(layer_indices=(0,), rank=2)
    policy = make_policy()
    adapter = install_spatial_adapter(policy, scfg)
    install_qwen_lora(adapter, lcfg)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=0.01)
    # Populate optimizer moments for both reader and LoRA parameters.
    loss = sum(p.square().sum() for p in policy.parameters() if p.requires_grad)
    loss.backward()
    optimizer.step()
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "model.safetensors").write_bytes(b"fake base identity")
    manifest = {"base_sha256": fingerprint_checkpoint(base_dir), "spatial": adapter.adapter_metadata(),
                "lora": asdict(lcfg)}
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, adapter=adapter, optimizer=optimizer, manifest=manifest,
                    step=5, epoch=2, cursor=7)
    expected_random = (random.random(), np.random.rand(), torch.rand(1))
    with pytest.raises(FileExistsError):
        save_checkpoint(checkpoint, adapter=adapter, optimizer=optimizer, manifest=manifest,
                        step=5, epoch=2, cursor=7)
    cloned_policy = make_policy()
    clone = install_spatial_adapter(cloned_policy, scfg)
    install_qwen_lora(clone, lcfg)
    clone_optimizer = torch.optim.AdamW([p for p in cloned_policy.parameters() if p.requires_grad], lr=0.01)
    assert restore_checkpoint(checkpoint, clone, clone_optimizer, manifest) == (5, 2, 7)
    assert random.random() == expected_random[0] and np.random.rand() == expected_random[1]
    torch.testing.assert_close(torch.rand(1), expected_random[2])
    for key, value in adapter.adapter_state_dict().items():
        torch.testing.assert_close(value, clone.adapter_state_dict()[key])
    assert len(clone_optimizer.state) == len(optimizer.state)
    restored_policy = make_policy()
    loaded = load_trained_sidecar(restored_policy, checkpoint, base_dir)
    for key, value in adapter.adapter_state_dict().items():
        torch.testing.assert_close(value, loaded.adapter_state_dict()[key])
    # Old sidecar-only checkpoints remain loadable for offline comparison.
    old = torch.load(checkpoint, weights_only=True)
    old["format_version"] = 1
    old.pop("interface")
    old_path = tmp_path / "old-v1.pt"
    torch.save(old, old_path)
    old_adapter = load_trained_sidecar(make_policy(), old_path, base_dir)
    assert old_adapter.enabled and old_adapter._lora_modules
    (base_dir / "model.safetensors").write_bytes(b"different base")
    with pytest.raises(ValueError, match="different base"):
        load_trained_sidecar(make_policy(), checkpoint, base_dir)


def test_train_step_updates_weights_and_rejects_nonfinite():
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor(1.0))
        def forward(self, batch):
            return (self.w * batch).square(), {}
    policy = Policy()
    optimizer = torch.optim.AdamW([{"params": list(policy.parameters()), "name": "interface"}], lr=0.01)
    before = policy.w.detach().clone()
    diagnostics = {}
    train_step(policy, torch.tensor(2.0), optimizer, 1.0, group_grad_norms=diagnostics)
    assert diagnostics["grad_norm_interface"] == pytest.approx(8.0)
    assert not torch.equal(before, policy.w)
    with pytest.raises(RuntimeError, match="non-finite"):
        train_step(policy, torch.tensor(float("nan")), optimizer, 1.0)


def test_checkpoint_vl_mask_inheritance_and_override(tmp_path):
    from test_vl_padding import FakeHead
    scfg = SpatialAdapterConfig(layer_indices=(0,), attention_dim=8, num_heads=2)
    lcfg = LoRAConfig(layer_indices=(0,), rank=2)
    policy = make_policy()
    policy._groot_model.action_head = FakeHead()
    adapter = install_spatial_adapter(policy, scfg)
    install_qwen_lora(adapter, lcfg)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=0.01)
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base")
    manifest = {"base_sha256": fingerprint_checkpoint(base), "spatial": adapter.adapter_metadata(),
                "lora": asdict(lcfg), "experiment": {"spatial_enabled": True, "lora_enabled": True,
                "train_interface": False, "embodiment_id": None, "mask_vl_self_attention": True}}
    checkpoint = tmp_path / "masked.pt"
    save_checkpoint(checkpoint, adapter=adapter, optimizer=optimizer, manifest=manifest,
                    step=1, epoch=0, cursor=1)

    def fresh():
        p = make_policy()
        p._groot_model.action_head = FakeHead()
        return p

    loaded = fresh()
    load_trained_sidecar(loaded, checkpoint, base)
    assert hasattr(loaded._groot_model.action_head, "_spatial_original_vl_processing")
    overridden = fresh()
    with pytest.warns(RuntimeWarning, match="differs from training"):
        load_trained_sidecar(overridden, checkpoint, base, mask_vl_self_attention=False)
    assert not hasattr(overridden._groot_model.action_head, "_spatial_original_vl_processing")

    # Checkpoints saved before the flag existed retain the original behavior;
    # forcing the new mode is possible but must warn about the changed model.
    legacy = torch.load(checkpoint, map_location="cpu", weights_only=True)
    legacy["manifest"]["experiment"].pop("mask_vl_self_attention")
    old_path = tmp_path / "unmasked.pt"
    torch.save(legacy, old_path)
    legacy_policy = fresh()
    load_trained_sidecar(legacy_policy, old_path, base)
    assert not hasattr(legacy_policy._groot_model.action_head, "_spatial_original_vl_processing")
    with pytest.warns(RuntimeWarning, match="differs from training"):
        load_trained_sidecar(fresh(), old_path, base, mask_vl_self_attention=True)
