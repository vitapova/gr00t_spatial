"""Optional CPU integration test: actual LeRobot N1.7 classes, tiny random weights."""
import pytest
import torch
from torch import nn
from types import SimpleNamespace
import os


def test_lerobot_n17_flow_matching_forward_and_backward(tmp_path, monkeypatch, capsys):
    transformers = pytest.importorskip("transformers")
    transformers.utils.import_utils._librosa_available = False
    pytest.importorskip("lerobot")
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from transformers.feature_extraction_utils import BatchFeature
    from lerobot.policies.groot.groot_n1_7 import Qwen3Backbone, GR00TN17, GR00TN17ActionHead, GR00TN17Config
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from groot_spatial import (SpatialAdapterConfig, install_spatial_adapter,
                               LoRAConfig, install_qwen_lora, set_lora_enabled)

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
    # No pretrained downloads: exercise the real LeRobot backbone methods with
    # a tiny Qwen, then the real LeRobot action head and flow-matching loss.
    base = Qwen3Backbone.__new__(Qwen3Backbone)
    nn.Module.__init__(base)
    base.model = Qwen3VLForConditionalGeneration(config)
    base.tune_llm = base.tune_visual = False
    base.requires_grad_(False)
    action_config = GR00TN17Config(
        backbone_embedding_dim=16, max_state_dim=6, max_action_dim=6, action_horizon=4,
        hidden_size=16, input_embedding_dim=16, state_history_length=1, max_num_embodiments=16,
        max_seq_len=32, use_alternate_vl_dit=False, use_vlln=True, state_dropout_prob=0.0,
        vl_self_attention_cfg=None,
        diffusion_model_cfg={"num_layers": 2, "num_attention_heads": 2, "attention_head_dim": 8,
                             "norm_type": "ada_norm", "dropout": 0.0, "final_dropout": False,
                             "output_dim": 16, "interleave_self_attention": False,
                             "positional_embeddings": None},
    )
    head = GR00TN17ActionHead(action_config).requires_grad_(False).eval()
    # Construct the real policy/model wrappers around the small components
    # without invoking from_pretrained or downloading NVIDIA weights.
    model = GR00TN17.__new__(GR00TN17)
    transformers.PreTrainedModel.__init__(model, action_config)
    model.backbone = base
    model.action_head = head
    policy = GrootPolicy.__new__(GrootPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(use_bf16=False)
    policy._groot_model = model
    wrapper = install_spatial_adapter(policy, SpatialAdapterConfig(layer_indices=(0, 1), attention_dim=8, num_heads=2))
    install_qwen_lora(wrapper, LoRAConfig(layer_indices=(0, 1, 2), rank=2))
    batch = BatchFeature({
        "input_ids": torch.tensor([[10, 97, 99, 99, 99, 99, 96, 11]]),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
        "pixel_values": torch.randn(16, 12),
    })
    action_input = BatchFeature({"state": torch.randn(1, 1, 6), "action": torch.randn(1, 4, 6),
                                 "action_mask": torch.ones(1, 4, 6), "embodiment_id": torch.zeros(1, dtype=torch.long)})
    wrapper.enabled = False
    baseline = wrapper(batch).backbone_features
    torch.testing.assert_close(baseline, base(batch).backbone_features, rtol=0, atol=0)
    wrapper.enabled = True
    loss, metrics = policy({**batch, **action_input})
    assert torch.isfinite(loss) and "loss" in metrics
    loss.backward()
    assert wrapper.readers["0"].value.weight.grad.abs().sum() > 0
    assert wrapper.readers["1"].query.weight.grad.abs().sum() > 0
    assert all(mod.lora_up.weight.grad is not None for _, _, mod in wrapper._lora_modules)
    assert any(mod.lora_up.weight.grad.abs().sum() > 0 for _, _, mod in wrapper._lora_modules)
    assert all(p.grad is None for p in base.parameters() if not p.requires_grad)
    assert all(p.grad is None for p in head.parameters())

    inference_inputs = {**batch, "state": action_input.state, "embodiment_id": action_input.embodiment_id}
    noise = torch.randn(1, 4, 6)
    model.eval()
    with torch.no_grad():
        wrapper.enabled = False
        set_lora_enabled(wrapper, False)
        original_actions = model.get_action(inference_inputs, noise=noise).action_pred
        wrapper.enabled = True
        set_lora_enabled(wrapper, True)
        modified_actions = model.get_action(inference_inputs, noise=noise).action_pred
    assert original_actions.shape == modified_actions.shape == (1, 4, 6)
    assert not torch.allclose(original_actions, modified_actions, atol=1e-7, rtol=1e-7)

    # If ARM100_TEST_ROOT is supplied, also feed real robot observations and
    # targets through LeRobot's real pack/normalization/transform pipeline.
    # Only its gated Qwen tokenizer is replaced with a tiny local test stub.
    root = os.environ.get("ARM100_TEST_ROOT")
    if root:
        from lerobot.configs import FeatureType
        from lerobot.datasets import LeRobotDataset
        from lerobot.utils.feature_utils import dataset_to_policy_features
        from lerobot.policies.groot.configuration_groot import GrootConfig
        from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors, GrootN17VLMEncodeStep
        from lerobot.utils.collate import lerobot_collate_fn
        from groot_spatial.train import _process_batch, train_step, main
        from torch.nn import functional as F

        data = LeRobotDataset("local/arm100", root=root, episodes=[0],
                              delta_timestamps={"action": [i / 30 for i in range(4)]},
                              video_backend="pyav", return_uint8=True)
        features = dataset_to_policy_features(data.meta.features)
        pcfg = GrootConfig(device="cpu", chunk_size=4, n_action_steps=4, max_state_dim=6, max_action_dim=6,
                           input_features={k: v for k, v in features.items() if v.type != FeatureType.ACTION},
                           output_features={k: v for k, v in features.items() if v.type == FeatureType.ACTION})
        proc, post = make_groot_pre_post_processors(pcfg, dataset_stats=data.meta.stats, dataset_meta=data.meta)

        class TinyProcessor:
            def apply_chat_template(self, conversation, **kwargs):
                return "tiny"
            def __call__(self, *, text, images, **kwargs):
                patches = []
                for image in images:
                    image = F.interpolate(image.float()[None], size=(8, 8), mode="bilinear", align_corners=False)[0] / 255
                    # [channels, block_h, intra_h, patch_h, block_w, intra_w, patch_w]
                    patches.append(image.reshape(3, 2, 2, 2, 2, 2, 2)
                                   .permute(1, 4, 2, 5, 0, 3, 6).reshape(16, 12))
                ids = torch.tensor([[10, 97, 99, 99, 99, 99, 96, 97, 99, 99, 99, 99, 96, 11]] * len(text))
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                        "image_grid_thw": torch.tensor([[1, 4, 4]] * len(images)),
                        "pixel_values": torch.cat(patches)}

        next(s for s in proc.steps if isinstance(s, GrootN17VLMEncodeStep))._proc = TinyProcessor()
        real_batch = lerobot_collate_fn([data[0]])
        processed = _process_batch(real_batch, data.meta.camera_keys, proc)
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-3)
        before = wrapper.readers["0"].value.weight.detach().clone()
        policy.train()
        train_loss, _ = train_step(policy, processed, optimizer, 1.0)
        assert torch.isfinite(torch.tensor(train_loss))
        assert not torch.equal(before, wrapper.readers["0"].value.weight)

        # Exercise the real training entry point and real ARM100 sampler. Only
        # checkpoint loading and the gated tokenizer are stubbed for this tiny model.
        for index, name, module in wrapper._lora_modules:
            setattr(base.language_model.layers[index].self_attn, name, module.base)
        model.backbone = base
        policy.config = pcfg
        checkpoint_dir = tmp_path / "base"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "config.json").write_text('{}')
        (checkpoint_dir / "model.safetensors").write_bytes(b"tiny-test")
        import lerobot.policies
        import importlib.metadata
        from lerobot.configs.policies import PreTrainedConfig
        monkeypatch.setattr(PreTrainedConfig, "from_pretrained", classmethod(lambda cls, *a, **kw: pcfg))
        monkeypatch.setattr(lerobot.policies, "make_policy", lambda **kwargs: policy)
        monkeypatch.setattr(lerobot.policies, "make_pre_post_processors", lambda **kwargs: (proc, post))
        version = importlib.metadata.version
        monkeypatch.setattr(importlib.metadata, "version", lambda name: "test" if name == "lerobot" else version(name))
        output = tmp_path / "output"
        main(["--dataset-root", root, "--base-checkpoint", str(checkpoint_dir), "--output-dir", str(output),
              "--device", "cpu", "--steps", "2", "--save-every", "2", "--eval-every", "1", "--eval-batches", "1",
              "--attention-dim", "8", "--heads", "2", "--spatial-layers", "0", "1",
              "--lora-layers", "0", "1", "2", "--lora-rank", "2"])
        checkpoint = torch.load(output / "step_00000002.pt", weights_only=True)
        assert checkpoint["step"] == 2 and checkpoint["spatial"] and checkpoint["lora"]

        trained_wrapper = model.backbone
        for index, name, module in trained_wrapper._lora_modules:
            setattr(base.language_model.layers[index].self_attn, name, module.base)
        model.backbone = base
        resumed_output = tmp_path / "resumed-output"
        main(["--dataset-root", root, "--base-checkpoint", str(checkpoint_dir), "--output-dir", str(resumed_output),
              "--resume", str(output / "step_00000002.pt"), "--device", "cpu", "--steps", "3",
              "--save-every", "2", "--eval-every", "1", "--eval-batches", "1", "--attention-dim", "8",
              "--heads", "2", "--spatial-layers", "0", "1", "--lora-layers", "0", "1", "2", "--lora-rank", "2"])
        assert torch.load(resumed_output / "step_00000003.pt", weights_only=True)["step"] == 3
        trained_wrapper = model.backbone
        for index, name, module in trained_wrapper._lora_modules:
            setattr(base.language_model.layers[index].self_attn, name, module.base)
        model.backbone = base
        from lerobot.policies.groot import configuration_groot
        monkeypatch.setattr(configuration_groot, "is_raw_groot_n1_7_checkpoint", lambda path: True)
        def fake_make_policy(cfg, ds_meta):
            cfg.input_features = pcfg.input_features
            cfg.output_features = pcfg.output_features
            policy.config = cfg
            return policy
        monkeypatch.setattr(lerobot.policies, "make_policy", fake_make_policy)
        raw_output = tmp_path / "raw-output"
        main(["--dataset-root", root, "--base-checkpoint", str(checkpoint_dir), "--output-dir", str(raw_output),
              "--device", "cpu", "--steps", "1", "--save-every", "1", "--eval-every", "0",
              "--chunk-size", "4", "--attention-dim", "8", "--heads", "2",
              "--spatial-layers", "0", "1", "--lora-layers", "0", "1", "2", "--lora-rank", "2"])
        assert (raw_output / "processors" / "policy_preprocessor.json").is_file()
        assert (raw_output / "processors" / "policy_postprocessor.json").is_file()
        assert (raw_output / "policy_config" / "config.json").is_file()

        trained_wrapper = model.backbone
        for index, name, module in trained_wrapper._lora_modules:
            setattr(base.language_model.layers[index].self_attn, name, module.base)
        model.backbone = base
        raw_resume = tmp_path / "raw-resumed"
        main(["--dataset-root", root, "--base-checkpoint", str(checkpoint_dir), "--output-dir", str(raw_resume),
              "--resume", str(raw_output / "step_00000001.pt"), "--device", "cpu", "--steps", "2",
              "--save-every", "1", "--eval-every", "0", "--chunk-size", "4", "--attention-dim", "8",
              "--heads", "2", "--spatial-layers", "0", "1", "--lora-layers", "0", "1", "2", "--lora-rank", "2"])
        assert (raw_resume / "step_00000002.pt").is_file()
        trained_wrapper = model.backbone
        for index, name, module in trained_wrapper._lora_modules:
            setattr(base.language_model.layers[index].self_attn, name, module.base)
        model.backbone = base
        from lerobot.policies.groot import modeling_groot
        from groot_spatial.load import load_experiment
        monkeypatch.setattr(modeling_groot, "GrootPolicy", lambda cfg: policy)
        loaded, _, _, adapter = load_experiment(raw_output / "step_00000001.pt", checkpoint_dir, raw_output, "cpu")
        assert loaded is policy and model.backbone is adapter and not policy.training
        (raw_output / "policy_config" / "config.json").write_text("{}")
        with pytest.raises(ValueError, match="processors/policy config"):
            load_experiment(raw_output / "step_00000001.pt", checkpoint_dir, raw_output, "cpu")

        # Train only the robot-specific interface, from real ARM100 frames.
        for index, name, module in adapter._lora_modules:
            setattr(base.language_model.layers[index].self_attn, name, module.base)
        model.backbone = base
        original_state_bank = head.state_encoder.layer1.W.detach().clone()
        only_interface = tmp_path / "interface-only"
        interface_args = ["--dataset-root", root, "--base-checkpoint", str(checkpoint_dir),
                          "--device", "cpu", "--save-every", "1", "--eval-every", "0", "--chunk-size", "4",
                          "--attention-dim", "8", "--heads", "2", "--spatial-layers", "0", "1",
                          "--lora-layers", "0", "1", "2", "--lora-rank", "2", "--train-interface"]
        main([*interface_args, "--output-dir", str(only_interface), "--steps", "1", "--no-spatial", "--no-lora"])
        interface_ckpt = torch.load(only_interface / "step_00000001.pt", weights_only=True)
        assert interface_ckpt["manifest"]["experiment"] == {
            "spatial_enabled": False, "lora_enabled": False, "train_interface": True, "embodiment_id": 10}
        assert len(interface_ckpt["interface"]) == 14 and not interface_ckpt["lora"]
        from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors_from_pretrained
        from groot_spatial.interface import processor_embodiment
        reloaded_proc, _ = make_groot_pre_post_processors_from_pretrained(
            pcfg, str(only_interface / "processors"), preprocessor_overrides={"device_processor": {"device": "cpu"}})
        assert processor_embodiment(reloaded_proc, "new_embodiment") == 10
        selected = head.state_encoder.layer1
        assert torch.equal(selected.base.W, original_state_bank)
        assert not torch.equal(head.action_decoder.layer2.slot_b, head.action_decoder.layer2.base.b[10])
        processed = _process_batch(real_batch, data.meta.camera_keys, proc, 10)
        model.eval()
        with torch.no_grad():
            predicted = model.get_action({k: v for k, v in processed.items() if k != "action"}, noise=noise).action_pred
        from groot_spatial.interface import INTERFACE_PATHS
        for path in INTERFACE_PATHS:
            parent_name, name = path.split(".")
            parent = getattr(head, parent_name)
            setattr(parent, name, getattr(parent, name).base)
        model.backbone = base
        loaded, loaded_pre, loaded_post, loaded_adapter = load_experiment(
            only_interface / "step_00000001.pt", checkpoint_dir, only_interface, "cpu")
        assert not loaded_adapter.enabled and not loaded_adapter._lora_modules
        with torch.no_grad():
            restored = model.get_action({k: v for k, v in processed.items() if k != "action"}, noise=noise).action_pred
        torch.testing.assert_close(restored, predicted, rtol=0, atol=0)

        # The offline demonstration uses a real ARM100 observation with the
        # same (stub-tokenizer) preprocessing and trained interface checkpoint.
        from demo_arm100 import predict_observation, main as demo_main
        import demo_arm100
        loaded._action_queue_steps = 4  # policy was assembled without __init__ in this tiny test
        observation = data[0]
        demo_actions = predict_observation(loaded, loaded_pre, loaded_post, observation, seed=17)
        assert demo_actions.shape == (4, 6) and torch.isfinite(demo_actions).all()
        torch.testing.assert_close(
            demo_actions, predict_observation(loaded, loaded_pre, loaded_post, observation, seed=17), rtol=0, atol=0)
        monkeypatch.setattr(demo_arm100, "load_experiment", lambda *args, **kwargs:
                            (loaded, loaded_pre, loaded_post, loaded_adapter))
        demo_main(["--base-checkpoint", str(checkpoint_dir), "--run-dir", str(only_interface),
                   "--checkpoint", str(only_interface / "step_00000001.pt"), "--dataset-root", root,
                   "--episode", "0", "--frame", "2", "--show-steps", "2", "--device", "cpu"])
        assert "Episode: 0" in capsys.readouterr().out
        demo_main(["--base-checkpoint", str(checkpoint_dir), "--run-dir", str(only_interface),
                   "--checkpoint", str(only_interface / "step_00000001.pt"), "--dataset-root", root,
                   "--show-steps", "1", "--device", "cpu"])
        assert "(validation)" in capsys.readouterr().out

        # The combined experiment has three independent optimizer groups and
        # restores their state and all three components on resume.
        for path in INTERFACE_PATHS:
            parent_name, name = path.split(".")
            parent = getattr(head, parent_name)
            setattr(parent, name, getattr(parent, name).base)
        model.backbone = base
        lora_only = tmp_path / "interface-lora"
        main([*interface_args, "--output-dir", str(lora_only), "--steps", "1", "--no-spatial"])
        lora_ckpt = torch.load(lora_only / "step_00000001.pt", weights_only=True)
        assert lora_ckpt["interface"] and lora_ckpt["lora"]
        assert not lora_ckpt["manifest"]["experiment"]["spatial_enabled"]
        trained = model.backbone
        for index, name, module in trained._lora_modules:
            setattr(base.language_model.layers[index].self_attn, name, module.base)
        model.backbone = base
        for path in INTERFACE_PATHS:
            parent_name, name = path.split(".")
            parent = getattr(head, parent_name)
            setattr(parent, name, getattr(parent, name).base)
        combined = tmp_path / "combined"
        main([*interface_args, "--output-dir", str(combined), "--steps", "1"])
        combined_ckpt = torch.load(combined / "step_00000001.pt", weights_only=True)
        assert combined_ckpt["interface"] and combined_ckpt["lora"] and combined_ckpt["spatial"]
        assert [g["name"] for g in combined_ckpt["optimizer"]["param_groups"]] == ["spatial", "lora", "interface"]
        trained = model.backbone
        for index, name, module in trained._lora_modules:
            setattr(base.language_model.layers[index].self_attn, name, module.base)
        model.backbone = base
        for path in INTERFACE_PATHS:
            parent_name, name = path.split(".")
            parent = getattr(head, parent_name)
            setattr(parent, name, getattr(parent, name).base)
        combined_resume = tmp_path / "combined-resume"
        main([*interface_args, "--output-dir", str(combined_resume), "--steps", "2",
              "--resume", str(combined / "step_00000001.pt")])
        assert (combined_resume / "step_00000002.pt").is_file()
