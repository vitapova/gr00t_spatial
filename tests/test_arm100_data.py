"""Optional integration test against the real extracted ARM100 dataset."""
import os

import pytest
import torch


def test_real_arm100_action_windows():
    root = os.environ.get("ARM100_TEST_ROOT")
    if not root:
        pytest.skip("set ARM100_TEST_ROOT to the extracted ARM100 dataset")
    pytest.importorskip("lerobot")
    from lerobot.datasets import LeRobotDataset, EpisodeAwareSampler
    from lerobot.utils.collate import lerobot_collate_fn
    from torch.utils.data import DataLoader
    from groot_spatial.train import _process_batch

    data = LeRobotDataset("local/arm100", root=root, episodes=[0, 2],
                          delta_timestamps={"action": [i / 30 for i in range(4)]},
                          video_backend="pyav", return_uint8=True)
    sampler = EpisodeAwareSampler(data.meta.episodes["dataset_from_index"], data.meta.episodes["dataset_to_index"],
                                  episode_indices_to_use=data.episodes, drop_n_last_frames=3,
                                  shuffle=True, seed=42, absolute_to_relative_idx=data.absolute_to_relative_idx)
    loader = DataLoader(data, batch_size=2, sampler=sampler, num_workers=0, collate_fn=lerobot_collate_fn)
    batch = next(iter(loader))
    assert batch["action"].shape == (2, 4, 6)
    assert not batch["action_is_pad"].any()
    assert batch["observation.images.front"].shape == (2, 3, 480, 640)
    assert batch["observation.images.wrist"].shape == (2, 3, 480, 640)
    assert all(isinstance(task, str) and task for task in batch["task"])
    def fake_processor(raw):
        assert raw["observation.images.front"].dtype == torch.float32
        return {"input_ids": torch.ones(2, 1), "attention_mask": torch.ones(2, 1),
                "pixel_values": torch.ones(2, 1), "image_grid_thw": torch.ones(2, 3),
                "state": raw["observation.state"], "action": raw["action"],
                "action_mask": torch.ones_like(raw["action"])}
    result = _process_batch(batch, data.meta.camera_keys, fake_processor)
    torch.testing.assert_close(result["action"], batch["action"])
