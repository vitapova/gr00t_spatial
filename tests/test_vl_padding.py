import pytest
import torch
from torch import nn

from groot_spatial.vl_padding import configure_vl_self_attention_mask


class FakeBlock(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, 2, batch_first=True, dropout=0.0)
        self.pos_embed = None

    def forward(self, hidden_states, attention_mask=None):
        mask = None if attention_mask is None else torch.isneginf(attention_mask[:, 0])
        attention, _ = self.attention(hidden_states, hidden_states, hidden_states,
                                      key_padding_mask=mask, need_weights=False)
        return hidden_states + attention


class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([FakeBlock(), FakeBlock()])

    def forward(self, hidden_states):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class FakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlln = nn.LayerNorm(8)
        self.vl_self_attention = FakeTransformer()

    def process_backbone_output(self, output):
        output["backbone_features"] = self.vl_self_attention(self.vlln(output["backbone_features"]))
        return output


def make_policy(head=None):
    policy = nn.Module()
    policy._groot_model = nn.Module()
    policy._groot_model.action_head = head or FakeHead()
    return policy


def run(head, x, mask):
    return head.process_backbone_output({"backbone_features": x, "backbone_attention_mask": mask})["backbone_features"]


def test_mask_invariance_gradients_and_disable():
    torch.manual_seed(1)
    policy = make_policy().eval()
    head = policy._groot_model.action_head
    x = torch.randn(1, 4, 8)
    valid = torch.ones(1, 4, dtype=torch.bool)
    native = run(head, x, valid)
    left_pad = torch.randn(1, 3, 8)
    padded = torch.cat([left_pad, x], dim=1).requires_grad_()
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
    native_padded = run(head, padded, mask)
    assert not torch.allclose(native, native_padded[:, 3:])

    configure_vl_self_attention_mask(policy, True)
    configure_vl_self_attention_mask(policy, True)  # idempotent
    masked_unpadded = run(head, x, valid)
    masked_padded = run(head, padded, mask)
    torch.testing.assert_close(masked_unpadded, native, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(masked_unpadded, masked_padded[:, 3:], rtol=1e-5, atol=1e-6)
    masked_padded[:, 3:].square().sum().backward()
    assert torch.count_nonzero(padded.grad[:, :3]) == 0
    assert torch.count_nonzero(padded.grad[:, 3:]) > 0
    configure_vl_self_attention_mask(policy, False)
    torch.testing.assert_close(run(head, x, valid), native, rtol=0, atol=0)


def test_mixed_batch_mask_and_validation():
    torch.manual_seed(2)
    policy = make_policy().eval()
    configure_vl_self_attention_mask(policy, True)
    head = policy._groot_model.action_head
    x = torch.randn(2, 6, 8)
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    mixed = run(head, x, mask)
    torch.testing.assert_close(mixed[0, 2:], run(head, x[0:1, 2:], mask[0:1, 2:])[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(mixed[1], run(head, x[1:2], mask[1:2])[0], rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="at least one"):
        run(head, x, torch.zeros(2, 6))
    with pytest.raises(ValueError, match="only 0 or 1"):
        run(head, x, torch.full((2, 6), 0.5))
    with pytest.raises(ValueError, match="required"):
        run(head, x, torch.ones(2, 5))


def test_rejects_unknown_transformer_or_positional_embeddings():
    policy = make_policy()
    policy._groot_model.action_head.vl_self_attention.transformer_blocks[0].pos_embed = nn.Identity()
    with pytest.raises(ValueError, match="positional embeddings"):
        configure_vl_self_attention_mask(policy, True)
    policy._groot_model.action_head.vl_self_attention = nn.Identity()
    with pytest.raises(ValueError, match="nonempty"):
        configure_vl_self_attention_mask(policy, True)


def test_actual_lerobot_vl_transformer():
    transformers = pytest.importorskip("transformers")
    transformers.utils.import_utils._librosa_available = False
    pytest.importorskip("lerobot")
    from lerobot.policies.groot.action_head.cross_attention_dit import SelfAttentionTransformer

    torch.manual_seed(5)
    head = FakeHead()
    head.vl_self_attention = SelfAttentionTransformer(num_attention_heads=2, attention_head_dim=4,
                                                      num_layers=2, dropout=0.0, final_dropout=False,
                                                      positional_embeddings=None)
    policy = make_policy(head).eval()
    x = torch.randn(1, 4, 8)
    pad = torch.randn(1, 3, 8)
    padded = torch.cat((pad, x), dim=1).requires_grad_()
    valid = torch.ones(1, 4, dtype=torch.bool)
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1]], dtype=torch.bool)
    native = run(head, x, valid)
    native_padded = run(head, padded, mask)
    assert not torch.allclose(native, native_padded[:, 3:])
    configure_vl_self_attention_mask(policy, True)
    masked = run(head, padded, mask)
    torch.testing.assert_close(masked[:, 3:], run(head, x, valid), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(run(head, x, valid), native, rtol=1e-5, atol=1e-6)
    masked[:, 3:].square().sum().backward()
    assert torch.count_nonzero(padded.grad[:, :3]) == 0
    assert torch.isfinite(padded.grad).all()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = run(head, padded.detach(), mask)
        separate = run(head, x, valid)
    torch.testing.assert_close(mixed[:, 3:], separate, rtol=1e-2, atol=1e-2)


def test_actual_lerobot_action_head_inference_and_training():
    transformers = pytest.importorskip("transformers")
    transformers.utils.import_utils._librosa_available = False
    pytest.importorskip("lerobot")
    from lerobot.policies.groot.groot_n1_7 import GR00TN17ActionHead, GR00TN17Config
    from transformers.feature_extraction_utils import BatchFeature

    torch.manual_seed(21)
    config = GR00TN17Config(
        backbone_embedding_dim=8, max_state_dim=6, max_action_dim=6, action_horizon=4,
        hidden_size=8, input_embedding_dim=8, state_history_length=1, max_num_embodiments=2,
        max_seq_len=32, use_alternate_vl_dit=True, use_vlln=True, state_dropout_prob=0.0,
        attend_text_every_n_blocks=2,
        vl_self_attention_cfg={"num_layers": 2, "num_attention_heads": 2, "attention_head_dim": 4,
                               "dropout": 0.0, "final_dropout": False, "positional_embeddings": None},
        diffusion_model_cfg={"num_layers": 4, "num_attention_heads": 2, "attention_head_dim": 4,
                             "norm_type": "ada_norm", "dropout": 0.0, "final_dropout": False,
                             "output_dim": 8, "interleave_self_attention": True, "positional_embeddings": None},
    )
    head = GR00TN17ActionHead(config).requires_grad_(False).eval()
    policy = make_policy(head).eval()
    features = torch.randn(1, 4, 8)
    left_pad = torch.randn(1, 3, 8)
    padded = torch.cat((left_pad, features), dim=1).requires_grad_()
    real_mask = torch.ones(1, 4, dtype=torch.bool)
    padded_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1]], dtype=torch.bool)
    real_image_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    padded_image_mask = torch.cat([torch.zeros(1, 3, dtype=torch.bool), real_image_mask], dim=1)
    action = BatchFeature({"state": torch.randn(1, 1, 6), "embodiment_id": torch.zeros(1, dtype=torch.long),
                           "action": torch.randn(1, 4, 6), "action_mask": torch.ones(1, 4, 6)})
    noise = torch.randn(1, 4, 6)

    def output(x, mask, image_mask):
        backbone = BatchFeature({"backbone_features": x, "backbone_attention_mask": mask, "image_mask": image_mask})
        inference_action = BatchFeature({"state": action.state, "embodiment_id": action.embodiment_id})
        return head.get_action(backbone, inference_action, noise=noise).action_pred

    with torch.no_grad():
        native = output(features, real_mask, real_image_mask)
        native_padded = output(padded, padded_mask, padded_image_mask)
    assert not torch.allclose(native, native_padded, rtol=1e-6, atol=1e-6)
    configure_vl_self_attention_mask(policy, True)
    with torch.no_grad():
        masked = output(features, real_mask, real_image_mask)
        masked_padded = output(padded, padded_mask, padded_image_mask)
    torch.testing.assert_close(masked, native, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(masked, masked_padded, rtol=1e-5, atol=1e-6)
    backbone = BatchFeature({"backbone_features": padded, "backbone_attention_mask": padded_mask,
                             "image_mask": padded_image_mask})
    head(backbone, action).loss.backward()
    assert torch.count_nonzero(padded.grad[:, :3]) == 0
    assert torch.count_nonzero(padded.grad[:, 3:]) > 0
    assert all(p.grad is None for p in head.parameters())


def test_qwen_and_action_head_single_vs_mixed_batch():
    transformers = pytest.importorskip("transformers")
    transformers.utils.import_utils._librosa_available = False
    pytest.importorskip("lerobot")
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from lerobot.policies.groot.groot_n1_7 import Qwen3Backbone, GR00TN17, GR00TN17ActionHead, GR00TN17Config
    from groot_spatial import SpatialAdapterConfig, install_spatial_adapter

    torch.manual_seed(23)
    qconfig = Qwen3VLConfig(
        image_token_id=99, video_token_id=98, vision_start_token_id=97, vision_end_token_id=96,
        text_config={"vocab_size": 128, "hidden_size": 16, "intermediate_size": 32,
                     "num_hidden_layers": 3, "num_attention_heads": 2, "num_key_value_heads": 2,
                     "head_dim": 8, "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
        vision_config={"depth": 2, "hidden_size": 8, "out_hidden_size": 16, "intermediate_size": 16,
                       "num_heads": 2, "patch_size": 2, "temporal_patch_size": 1,
                       "spatial_merge_size": 2, "num_position_embeddings": 16,
                       "deepstack_visual_indexes": [0]},
    )
    base = Qwen3Backbone.__new__(Qwen3Backbone)
    nn.Module.__init__(base)
    base.model = Qwen3VLForConditionalGeneration(qconfig)
    base.tune_llm = base.tune_visual = False
    base.requires_grad_(False)
    config = GR00TN17Config(
        backbone_embedding_dim=16, max_state_dim=6, max_action_dim=6, action_horizon=4,
        hidden_size=16, input_embedding_dim=16, state_history_length=1, max_num_embodiments=2,
        max_seq_len=32, use_alternate_vl_dit=True, use_vlln=True, state_dropout_prob=0.0,
        attend_text_every_n_blocks=2,
        vl_self_attention_cfg={"num_layers": 2, "num_attention_heads": 2, "attention_head_dim": 8,
                               "dropout": 0.0, "final_dropout": False, "positional_embeddings": None},
        diffusion_model_cfg={"num_layers": 4, "num_attention_heads": 2, "attention_head_dim": 8,
                             "norm_type": "ada_norm", "dropout": 0.0, "final_dropout": False,
                             "output_dim": 16, "interleave_self_attention": True, "positional_embeddings": None},
    )
    model = GR00TN17.__new__(GR00TN17)
    transformers.PreTrainedModel.__init__(model, config)
    model.backbone = base
    model.action_head = GR00TN17ActionHead(config).requires_grad_(False)
    policy = nn.Module()
    policy._groot_model = model
    adapter = install_spatial_adapter(policy, SpatialAdapterConfig(layer_indices=(0, 1), attention_dim=8, num_heads=2))
    policy.eval()
    image0, image1 = torch.randn(16, 12), torch.randn(16, 12)
    ids0 = torch.tensor([[10, 97, 99, 99, 99, 99, 96, 11]])
    ids1 = torch.tensor([[6, 7, 10, 97, 99, 99, 99, 99, 96, 11]])
    state = torch.randn(2, 1, 6)
    noise = torch.randn(2, 4, 6)
    single = {"input_ids": ids0, "attention_mask": torch.ones_like(ids0), "pixel_values": image0,
              "image_grid_thw": torch.tensor([[1, 4, 4]]), "state": state[:1],
              "embodiment_id": torch.zeros(1, dtype=torch.long)}
    mixed = {"input_ids": torch.cat((torch.cat((torch.zeros(1, 2, dtype=torch.long), ids0), 1), ids1), 0),
             "attention_mask": torch.tensor([[0, 0] + [1] * 8, [1] * 10]),
             "pixel_values": torch.cat((image0, image1)), "image_grid_thw": torch.tensor([[1, 4, 4]] * 2),
             "state": state, "embodiment_id": torch.zeros(2, dtype=torch.long)}
    with torch.no_grad():
        original_single = model.get_action(single, noise=noise[:1]).action_pred
        original_mixed = model.get_action(mixed, noise=noise).action_pred[:1]
    assert not torch.allclose(original_single, original_mixed, rtol=1e-6, atol=1e-6)
    configure_vl_self_attention_mask(policy, True)
    with torch.no_grad():
        masked_single = model.get_action(single, noise=noise[:1]).action_pred
        masked_mixed = model.get_action(mixed, noise=noise).action_pred[:1]
    torch.testing.assert_close(masked_single, original_single, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(masked_single, masked_mixed, rtol=1e-5, atol=1e-6)
    training_input = {**single, "action": torch.randn(1, 4, 6), "action_mask": torch.ones(1, 4, 6)}
    model(training_input).loss.backward()
    assert adapter.readers["0"].value.weight.grad.abs().sum() > 0
