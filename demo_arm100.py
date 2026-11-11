"""Offline ARM100 inference demo. Prints predictions; never controls a robot."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from groot_spatial.load import load_experiment
from groot_spatial.train import fingerprint_tree


CAMERA_KEYS = ("observation.images.front", "observation.images.wrist")
JOINT_NAMES = ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos",
               "wrist_roll.pos", "gripper.pos")


def predict_observation(policy: Any, preprocessor: Any, postprocessor: Any,
                        observation: dict[str, Any], *, seed: int = 42) -> torch.Tensor:
    """Predict one action chunk in dataset units from a single observation.

    The observation must contain two CHW RGB images, six state values and a
    task string. Image tensors may be uint8 [0,255] or floats in [0,1]. No
    target actions are passed to the preprocessor or policy. The returned CPU
    tensor is [horizon, 6] after the checkpoint's postprocessor.
    """
    task = observation.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    state = torch.as_tensor(observation["observation.state"])
    if state.shape != (6,) or not torch.isfinite(state).all():
        raise ValueError("observation.state must contain six finite values")
    batch: dict[str, Any] = {"observation.state": state.float().unsqueeze(0), "task": [task]}
    for key in CAMERA_KEYS:
        image = torch.as_tensor(observation[key])
        if image.ndim != 3 or image.shape[0] != 3 or image.shape[1] == 0 or image.shape[2] == 0:
            raise ValueError(f"{key} must have shape [3,H,W]")
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        elif image.is_floating_point():
            image = image.float()
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError(f"{key} float pixels must be finite and between 0 and 1")
        else:
            raise ValueError(f"{key} pixels must be uint8 or floating point")
        batch[key] = image.unsqueeze(0)

    policy.eval()
    # Clear action queues and any cached relative-action anchor before a new
    # observation; preprocessing must precede postprocessing to cache the state.
    policy.reset()
    preprocessor.reset()
    device = next(policy.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    config = policy._groot_model.config
    noise = torch.randn((1, int(config.action_horizon), int(config.max_action_dim)),
                        generator=generator, device=device, dtype=torch.float32)
    with torch.no_grad():
        inputs = preprocessor(batch)
        if inputs.get("action") is not None:
            raise ValueError("inference preprocessor unexpectedly supplied target actions")
        raw_chunk = policy.predict_action_chunk(inputs, noise=noise)
        chunk = postprocessor(raw_chunk)
    if not isinstance(chunk, torch.Tensor) or chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[2] != 6:
        raise ValueError("postprocessor must return action chunk [1,horizon,6]")
    if not torch.isfinite(chunk).all():
        raise ValueError("predicted action chunk contains non-finite values")
    return chunk[0].detach().float().cpu()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True, help="Original N1.7 checkpoint directory")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory containing processors")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained sidecar step_*.pt")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Extracted ARM100 dataset directory")
    parser.add_argument("--episode", type=int, default=None, help="Episode index; default: first validation episode")
    parser.add_argument("--frame", type=int, default=0, help="Frame index within the selected episode")
    parser.add_argument("--show-steps", type=int, default=5, help="Number of predicted actions to display")
    parser.add_argument("--seed", type=int, default=42, help="Action noise seed")
    parser.add_argument("--mask-vl-self-attention", action=argparse.BooleanOptionalAction, default=None,
                        help="Override checkpoint's VL self-attention padding mask setting")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--allow-dataset-mismatch", action="store_true",
                        help="Allow another dataset; train/validation membership will be reported as unknown")
    args = parser.parse_args(argv)
    if args.frame < 0 or args.show_steps <= 0 or (args.episode is not None and args.episode < 0):
        parser.error("episode/frame must be nonnegative and show-steps must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; choose a machine with a compatible NVIDIA GPU")
    root = args.dataset_root.expanduser().resolve()
    if not (root / "meta/info.json").is_file():
        parser.error("dataset-root must be an extracted LeRobot v3 dataset")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format_version") not in (1, 2):
        raise ValueError("unsupported sidecar checkpoint format")
    manifest = saved["manifest"]
    print("Checking dataset fingerprint...", flush=True)
    matches_dataset = fingerprint_tree(root, ("meta", "data", "videos")) == manifest.get("dataset_sha256")
    if not matches_dataset and not args.allow_dataset_mismatch:
        raise ValueError("dataset differs from the training run; use --allow-dataset-mismatch only intentionally")
    if args.episode is None:
        if not matches_dataset:
            parser.error("--episode is required for a different dataset")
        eval_episodes = manifest.get("eval_episodes", [])
        if not eval_episodes:
            parser.error("checkpoint has no recorded validation episodes; specify --episode")
        episode = int(eval_episodes[0])
    else:
        episode = args.episode

    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps

    dataset_id = manifest.get("dataset_id", "local/arm100")
    meta = LeRobotDatasetMetadata(dataset_id, root=root)
    if episode >= meta.total_episodes:
        parser.error(f"episode {episode} is outside dataset range 0..{meta.total_episodes - 1}")
    if (set(meta.camera_keys) != set(CAMERA_KEYS)
            or tuple(meta.features["action"]["shape"]) != (6,)
            or tuple(meta.features["observation.state"]["shape"]) != (6,)
            or tuple(meta.features["action"]["names"]) != JOINT_NAMES
            or tuple(meta.features["observation.state"]["names"]) != JOINT_NAMES):
        raise ValueError("dataset does not match the expected ARM100 cameras/action order")
    print("Loading GR00T N1.7 and trained sidecar...", flush=True)
    policy, preprocessor, postprocessor, _ = load_experiment(
        args.checkpoint, args.base_checkpoint, args.run_dir, device=str(device),
        mask_vl_self_attention=args.mask_vl_self_attention)
    recorded_mask = manifest.get("experiment", {}).get("mask_vl_self_attention", False)
    effective_mask = recorded_mask if args.mask_vl_self_attention is None else args.mask_vl_self_attention
    print(f"VL self-attention padding mask: {'on' if effective_mask else 'off'}", flush=True)
    delta = resolve_delta_timestamps(policy.config, meta)
    if not delta or "action" not in delta:
        raise ValueError("model config does not specify an action horizon")
    dataset = LeRobotDataset(dataset_id, root=root, episodes=[episode], delta_timestamps=delta,
                             video_backend=args.video_backend, return_uint8=True)
    if args.frame >= len(dataset):
        parser.error(f"frame {args.frame} is outside episode length {len(dataset)}")
    sample = dataset[args.frame]
    if int(sample["episode_index"]) != episode or int(sample["frame_index"]) != args.frame:
        raise RuntimeError("dataset index does not match the requested episode/frame")
    observation = {key: sample[key] for key in (*CAMERA_KEYS, "observation.state", "task")}
    prediction = predict_observation(policy, preprocessor, postprocessor, observation, seed=args.seed)
    target = torch.as_tensor(sample["action"]).float().cpu()
    if target.ndim != 2 or target.shape[-1] != 6:
        raise ValueError("dataset action target must have shape [horizon,6]")
    is_pad = torch.as_tensor(sample.get("action_is_pad", torch.zeros(target.shape[0], dtype=torch.bool))).bool()
    if is_pad.shape != (target.shape[0],):
        raise ValueError("dataset action padding mask does not match target horizon")
    n = min(prediction.shape[0], target.shape[0])
    valid = ~is_pad[:n]
    membership = ("validation" if episode in manifest.get("eval_episodes", []) else
                  "train" if episode in manifest.get("train_episodes", []) else "not in recorded split") if matches_dataset else "unknown"
    print(f"\nEpisode: {episode} ({membership}), frame: {args.frame}, FPS: {meta.fps}")
    print(f"Task: {sample['task']}")
    print("Order: " + ", ".join(JOINT_NAMES))
    print("Current state: " + str([round(float(x), 3) for x in torch.as_tensor(sample['observation.state'])]))
    print(f"Predicted horizon: {prediction.shape[0]}; comparable recorded steps: {int(valid.sum())}")
    for i in range(min(args.show_steps, prediction.shape[0])):
        pred = [round(float(x), 3) for x in prediction[i]]
        reference = [round(float(x), 3) for x in target[i]] if i < n and valid[i] else None
        print(f"t+{i:02d} predicted={pred}  recorded={reference}")
    if valid.any():
        absolute_error = (prediction[:n][valid] - target[:n][valid]).abs()
        print("Per-joint MAE (dataset units): " + str([round(float(x), 3) for x in absolute_error.mean(dim=0)]))
    print("Offline comparison only: no actions were sent to a robot.")


if __name__ == "__main__":
    main()
