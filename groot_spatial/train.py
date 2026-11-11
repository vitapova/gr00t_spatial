"""Single-device ARM100 spatial/LoRA and optional action-interface training."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from copy import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import (LoRAConfig, SpatialAdapterConfig, install_qwen_lora, install_spatial_adapter,
               load_lora_state_dict, lora_state_dict)
from .interface import (configure_processor_embodiment, install_action_interface,
                        interface_state_dict, load_interface_state_dict, processor_embodiment)
from .vl_padding import configure_vl_self_attention_mask


def episode_split(total: int, fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if total < 2 or not 0 < fraction < 1:
        raise ValueError("Need at least two episodes and eval fraction strictly between 0 and 1")
    episodes = list(range(total))
    random.Random(seed).shuffle(episodes)
    count = max(1, min(total - 1, round(total * fraction)))
    return sorted(episodes[count:]), sorted(episodes[:count])


def read_episode_split(path: Path, total: int) -> tuple[list[int], list[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"train", "eval"}:
        raise ValueError("split file must contain exactly 'train' and 'eval' lists")
    train, evaluate = data["train"], data["eval"]
    for name, values in (("train", train), ("eval", evaluate)):
        if (not isinstance(values, list) or not values or
                any(type(n) is not int or n < 0 or n >= total for n in values) or
                len(values) != len(set(values))):
            raise ValueError(f"invalid {name} episode list")
    if set(train) & set(evaluate):
        raise ValueError("train and eval episode lists must be disjoint")
    return sorted(train), sorted(evaluate)


def fingerprint_tree(root: Path, subdirs: tuple[str, ...]) -> str:
    """Fingerprint the bytes and relative paths, not file modification times."""
    digest = hashlib.sha256()
    files = sorted(p for d in subdirs for p in (root / d).rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"No input files found under {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def fingerprint_checkpoint(root: Path) -> str:
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise ValueError("checkpoint directory is empty")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def fingerprint_source() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_output_directory(output_arg: Path, dataset_root: Path, base: Path,
                              resume: Path | None, force: bool) -> Path:
    """Validate destructive output cleanup before starting an expensive run."""
    requested = output_arg.expanduser().absolute()
    if requested.is_symlink():
        raise ValueError("output-dir must not be a symbolic link")
    output = requested.resolve()
    protected = [dataset_root.resolve(), base.resolve(), Path.cwd().resolve(), Path.home().resolve()]
    if resume is not None:
        protected.append(resume.expanduser().resolve())
    for path in protected:
        if output == path or output in path.parents:
            raise ValueError(f"output-dir cannot contain or equal a protected path: {path}")
    if dataset_root.resolve() in output.parents or base.resolve() in output.parents:
        raise ValueError("output-dir must be outside the dataset and base checkpoint directories")
    if output.exists():
        if not output.is_dir():
            raise ValueError("output-dir exists but is not a directory")
        if not force:
            raise ValueError("output-dir already exists; use --force-overwrite to delete its contents")
    return output


def prepare_output_directory(output: Path, force: bool) -> None:
    """Create output, deleting only the explicitly selected directory if asked."""
    if output.is_symlink():
        raise ValueError("output-dir must not be a symbolic link")
    if output.exists() and force:
        if not output.is_dir():
            raise ValueError("output-dir exists but is not a directory")
        print(f"Removing existing output directory and all its contents: {output}", flush=True)
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)


def _rng_state() -> dict[str, Any]:
    np_state = np.random.get_state()
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(),
            "numpy": (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4])}


def _restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from saved checkpoint")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))


def save_checkpoint(path: Path, *, adapter: Any, optimizer: torch.optim.Optimizer,
                    manifest: dict[str, Any], step: int, epoch: int, cursor: int,
                    interface_modules: dict[str, Any] | None = None) -> None:
    # Exclusive creation avoids overwriting prior work, including on resume.
    payload = {"format_version": 2, "manifest": manifest, "step": step, "epoch": epoch, "cursor": cursor,
               "spatial": adapter.adapter_state_dict(), "lora": lora_state_dict(adapter),
               "interface": interface_state_dict(interface_modules or {}),
               "optimizer": optimizer.state_dict(), "rng": _rng_state()}
    with path.open("xb") as stream:
        torch.save(payload, stream)


def restore_checkpoint(path: Path, adapter: Any, optimizer: torch.optim.Optimizer,
                       manifest: dict[str, Any], interface_modules: dict[str, Any] | None = None) -> tuple[int, int, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2 or checkpoint.get("manifest") != manifest:
        raise ValueError("Checkpoint does not match model, dataset, split, adapter or training settings")
    adapter.readers.load_state_dict(checkpoint["spatial"], strict=True)
    load_lora_state_dict(adapter, checkpoint["lora"])
    load_interface_state_dict(interface_modules or {}, checkpoint["interface"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    _restore_rng(checkpoint["rng"])
    return int(checkpoint["step"]), int(checkpoint["epoch"]), int(checkpoint["cursor"])


def _normalise_images(batch: dict[str, Any], camera_keys: list[str]) -> dict[str, Any]:
    batch = dict(batch)
    for key in camera_keys:
        if key not in batch:
            raise ValueError(f"Missing required camera {key}")
        if batch[key].dtype == torch.uint8:
            batch[key] = batch[key].float().div(255.0)
    return batch


def _process_batch(batch: dict[str, Any], camera_keys: list[str], preprocessor: Any,
                   embodiment_id: int | None = None) -> Any:
    if batch["action"].ndim != 3 or batch["action"].shape[-1] != 6:
        raise ValueError("expected ARM100 action chunks [batch, horizon, 6]")
    if batch["observation.state"].shape[-1] != 6:
        raise ValueError("expected six-dimensional ARM100 state")
    if "task" not in batch:
        raise ValueError("dataset task instruction is missing")
    processed = preprocessor(_normalise_images(batch, camera_keys))
    for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "state", "action", "action_mask"):
        if key not in processed:
            raise ValueError(f"GR00T preprocessor did not produce {key}")
    if not processed["action_mask"].any():
        raise ValueError("all action targets are masked")
    if embodiment_id is not None:
        ids = processed.get("embodiment_id")
        if ids is None or not torch.all(ids == embodiment_id).item():
            raise ValueError(f"processed batch does not use trained embodiment ID {embodiment_id}")
    return processed


def train_step(policy: Any, batch: Any, optimizer: torch.optim.Optimizer, grad_clip: float,
               *, group_grad_norms: dict[str, float] | None = None) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    loss, _ = policy(batch)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite loss; refusing optimizer step")
    loss.backward()
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    for group in optimizer.param_groups:
        grads = [p.grad for p in group["params"] if p.grad is not None]
        if not grads or not any(torch.count_nonzero(g).item() for g in grads):
            raise RuntimeError(f"no nonzero gradients in {group.get('name', 'optimizer group')}; refusing optimizer step")
        if group_grad_norms is not None:
            # Report per-group norms before clipping; do not alter optimizer state.
            norms = torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in grads])
            group_grad_norms[f"grad_norm_{group.get('name', 'group')}"] = float(torch.linalg.vector_norm(norms))
    norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(norm)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Extracted LeRobot v3 dataset root")
    parser.add_argument("--dataset-id", default="local/arm100")
    parser.add_argument("--base-checkpoint", type=Path, required=True, help="Local raw N1.7 or saved LeRobot N1.7 checkpoint directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Training output directory")
    parser.add_argument("--force-overwrite", action="store_true",
                        help="Delete the entire existing output directory before starting")
    parser.add_argument("--resume", type=Path, help="Sidecar training checkpoint; use a new output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=40, help="Action horizon when starting from a raw N1.7 checkpoint")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--split-file", type=Path, help='JSON object with disjoint "train" and "eval" episode ID lists')
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--spatial-layers", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--lora-layers", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--attention-dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--train-interface", action="store_true",
                        help="Train the state/action encoder and decoder for one embodiment")
    parser.add_argument("--embodiment-id", type=int, default=None,
                        help="Selected slot when training the interface; raw checkpoints default to 10")
    parser.add_argument("--no-spatial", action="store_true", help="Disable spatial cross-attention")
    parser.add_argument("--no-lora", action="store_true", help="Disable Qwen LoRA")
    parser.add_argument("--mask-vl-self-attention", action="store_true",
                        help="Exclude padding keys from action head VL self-attention")
    args = parser.parse_args(argv)
    if min(args.steps, args.batch_size, args.chunk_size, args.eval_batches, args.save_every, args.log_every) <= 0:
        parser.error("steps, batch-size, eval-batches, save-every and log-every must be positive")
    if args.eval_every < 0 or args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        parser.error("invalid optimizer or evaluation settings")
    if args.embodiment_id is not None and not args.train_interface:
        parser.error("--embodiment-id requires --train-interface")
    if args.no_spatial and args.no_lora and not args.train_interface:
        parser.error("no trainable components selected")
    dataset_root = args.dataset_root.expanduser().resolve()
    base = args.base_checkpoint.expanduser().resolve()
    try:
        output = validate_output_directory(args.output_dir, dataset_root, base, args.resume, args.force_overwrite)
    except ValueError as error:
        parser.error(str(error))
    if not (dataset_root / "meta/info.json").is_file():
        parser.error("dataset-root must be an extracted LeRobot v3 dataset (meta/info.json missing)")
    if not (base / "config.json").is_file() or not list(base.glob("*.safetensors")):
        parser.error("base-checkpoint needs config.json and local safetensors weights")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; --device cpu is intended only for small tests")

    # Deliberately lazy: --help works without installing the full ML stack.
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets import EpisodeAwareSampler, LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.policies.groot.configuration_groot import GrootConfig, is_raw_groot_n1_7_checkpoint
    from lerobot.policies.groot.processor_groot import _set_groot_preprocessor_training
    from lerobot.utils.collate import lerobot_collate_fn

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    meta = LeRobotDatasetMetadata(args.dataset_id, root=dataset_root)
    expected_cameras = {"observation.images.front", "observation.images.wrist"}
    joint_names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos",
                   "wrist_roll.pos", "gripper.pos"]
    if (set(meta.camera_keys) != expected_cameras or tuple(meta.features["action"]["shape"]) != (6,)
            or tuple(meta.features["observation.state"]["shape"]) != (6,)
            or list(meta.features["action"]["names"]) != joint_names
            or list(meta.features["observation.state"]["names"]) != joint_names):
        raise ValueError("This script expects the two-camera six-action ARM100 dataset schema")
    raw_base = is_raw_groot_n1_7_checkpoint(base)
    if raw_base:
        index_file = base / "model.safetensors.index.json"
        if index_file.exists():
            weight_map = json.loads(index_file.read_text(encoding="utf-8")).get("weight_map", {})
            shards = set(weight_map.values())
            if not shards or any(Path(s).name != s or not (base / s).is_file() for s in shards):
                raise ValueError("raw checkpoint has missing or invalid safetensors shards")
        elif not (base / "model.safetensors").is_file():
            raise ValueError("raw checkpoint needs model.safetensors or a complete sharded safetensors index")
        config = GrootConfig(base_model_path=str(base), chunk_size=args.chunk_size, n_action_steps=args.chunk_size,
                             embodiment_tag="new_embodiment", use_relative_actions=False, device=str(device))
    else:
        if not (base / "model.safetensors").is_file():
            raise ValueError("saved LeRobot policy needs model.safetensors")
        config = PreTrainedConfig.from_pretrained(str(base))
        if not isinstance(config, GrootConfig):
            raise TypeError("base checkpoint must be a LeRobot GR00T N1.7 policy")
        checkpoint_cameras = {key for key in (config.input_features or {}) if key.startswith("observation.images.")}
        if checkpoint_cameras != expected_cameras:
            raise ValueError("checkpoint camera names differ from ARM100; explicit remapping is required")
        if not config.output_features or "action" not in config.output_features or tuple(config.output_features["action"].shape) != (6,):
            raise ValueError("checkpoint action dimensions differ from ARM100")
        config.pretrained_path = base
    config.device = str(device)
    config.use_bf16 = device.type == "cuda"
    train_eps, eval_eps = (read_episode_split(args.split_file, meta.total_episodes) if args.split_file
                           else episode_split(meta.total_episodes, args.eval_fraction, args.seed))
    timestamps = resolve_delta_timestamps(config, meta)
    if not timestamps or "action" not in timestamps or not timestamps["action"]:
        raise ValueError("policy does not specify an action chunk horizon")
    horizon = len(timestamps["action"])
    def make_dataset(episodes):
        return LeRobotDataset(args.dataset_id, root=dataset_root, episodes=episodes,
                              delta_timestamps=timestamps, video_backend=args.video_backend, return_uint8=True)
    train_data, eval_data = make_dataset(train_eps), make_dataset(eval_eps)
    def make_sampler(data, shuffle):
        return EpisodeAwareSampler(data.meta.episodes["dataset_from_index"], data.meta.episodes["dataset_to_index"],
                                   episode_indices_to_use=data.episodes, drop_n_last_frames=horizon - 1,
                                   shuffle=shuffle, seed=args.seed,
                                   absolute_to_relative_idx=data.absolute_to_relative_idx)
    sampler, eval_sampler = make_sampler(train_data, True), make_sampler(eval_data, False)
    loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler, num_workers=0,
                        collate_fn=lerobot_collate_fn, generator=torch.Generator().manual_seed(args.seed))
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, sampler=eval_sampler, num_workers=0,
                             collate_fn=lerobot_collate_fn, generator=torch.Generator().manual_seed(args.seed))
    print("Loading saved LeRobot policy and processors...", flush=True)
    policy = make_policy(cfg=config, ds_meta=meta)
    configure_vl_self_attention_mask(policy, args.mask_vl_self_attention)
    if raw_base:
        # Raw N1.7 needs dataset stats for this embodiment. LeRobot v3's
        # stats are global, so held-out loss is not a completely isolated test.
        train_proc, post_proc = make_pre_post_processors(policy_cfg=config, dataset_stats=meta.stats, dataset_meta=meta)
        eval_config = copy(config)
        eval_config._runtime_dataset_meta = None
        eval_proc, _ = make_pre_post_processors(policy_cfg=eval_config, dataset_stats=meta.stats)
        _set_groot_preprocessor_training(eval_proc, training=False)
    else:
        # Keep the fine-tuned checkpoint's processor stats and action semantics intact.
        overrides = {"device_processor": {"device": str(device)},
                     "groot_n1_7_vlm_encode_v1": {"device": str(device)}}
        train_proc, post_proc = make_pre_post_processors(policy_cfg=config, pretrained_path=base, dataset_meta=meta,
                                                         preprocessor_overrides=overrides)
        eval_proc, _ = make_pre_post_processors(policy_cfg=config, pretrained_path=base,
                                                preprocessor_overrides=overrides)
    selected_id = None
    if args.train_interface:
        tag = config.embodiment_tag
        current_id = processor_embodiment(train_proc, tag)
        selected_id = args.embodiment_id if args.embodiment_id is not None else (10 if raw_base else current_id)
        if selected_id is None:
            raise ValueError("saved policy processor lacks an embodiment mapping; specify a compatible checkpoint")
        if not raw_base and selected_id != current_id:
            raise ValueError("cannot change the embodiment ID of an existing LeRobot policy")
        if raw_base:
            for proc in (train_proc, eval_proc):
                pack = next(step for step in proc.steps if getattr(step, "embodiment_tag", None) == tag
                            and hasattr(step, "embodiment_mapping"))
                occupied = [name for name, value in pack.embodiment_mapping.items()
                            if int(value) == selected_id and name != tag]
                if occupied:
                    raise ValueError(f"embodiment ID {selected_id} is assigned to {occupied}; choose an unused ID")
        configure_processor_embodiment(train_proc, tag, selected_id)
        configure_processor_embodiment(eval_proc, tag, selected_id)
    policy.requires_grad_(False)
    spatial_cfg = SpatialAdapterConfig(layer_indices=tuple(args.spatial_layers), attention_dim=args.attention_dim,
                                       num_heads=args.heads)
    lora_cfg = LoRAConfig(layer_indices=tuple(args.lora_layers), rank=args.lora_rank,
                          alpha=args.lora_alpha, dropout=args.lora_dropout)
    adapter = install_spatial_adapter(policy, spatial_cfg)
    adapter.enabled = not args.no_spatial
    if args.no_spatial:
        adapter.readers.requires_grad_(False)
    if not args.no_lora:
        # Initializing optional LoRA must not advance training's noise RNG in
        # this ablation but not in an otherwise identical interface-only run.
        lora_rng = _rng_state()
        install_qwen_lora(adapter, lora_cfg)
        _restore_rng(lora_rng)
    interface_modules = install_action_interface(policy, selected_id) if args.train_interface else {}
    spatial_parameters = list(adapter.readers.parameters()) if not args.no_spatial else []
    lora_parameters = [p for _, _, module in adapter._lora_modules
                       for part in (module.lora_down, module.lora_up) for p in part.parameters()]
    trainable = [p for p in policy.parameters() if p.requires_grad]
    interface_parameters = [p for module in interface_modules.values() for p in (module.slot_W, module.slot_b)]
    if {id(p) for p in trainable} != {id(p) for p in spatial_parameters + lora_parameters + interface_parameters}:
        raise RuntimeError("unexpected trainable parameters outside the selected components")
    groups = [{"params": params, "name": name} for name, params in
              (("spatial", spatial_parameters), ("lora", lora_parameters), ("interface", interface_parameters))
              if params]
    optimizer = torch.optim.AdamW(groups,
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}; action horizon: {horizon}; "
          f"embodiment ID: {selected_id if selected_id is not None else 'processor default'}", flush=True)
    print(f"VL self-attention padding mask: {'on' if args.mask_vl_self_attention else 'off'}", flush=True)
    for group in groups:
        print(f"  {group['name']}: {sum(p.numel() for p in group['params']):,} parameters", flush=True)
    print("Hashing dataset and base checkpoint for reproducibility...", flush=True)
    import importlib.metadata
    manifest = {"dataset_sha256": fingerprint_tree(dataset_root, ("meta", "data", "videos")),
                "base_sha256": fingerprint_checkpoint(base), "dataset_id": args.dataset_id,
                "base_kind": "raw" if raw_base else "lerobot", "train_episodes": train_eps,
                "eval_episodes": eval_eps, "seed": args.seed,
                "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip, "video_backend": args.video_backend, "device": str(device),
                "eval_every": args.eval_every, "eval_batches": args.eval_batches,
                "spatial": adapter.adapter_metadata(), "lora": asdict(lora_cfg) if not args.no_lora else None,
                "experiment": {"spatial_enabled": not args.no_spatial, "lora_enabled": not args.no_lora,
                               "train_interface": args.train_interface, "embodiment_id": selected_id,
                               "mask_vl_self_attention": args.mask_vl_self_attention},
                "horizon": horizon,
                "torch": str(torch.__version__), "numpy": np.__version__, "python": sys.version,
                "lerobot": importlib.metadata.version("lerobot"),
                "transformers": importlib.metadata.version("transformers"),
                "adapter_source_sha256": fingerprint_source()}
    prepare_output_directory(output, args.force_overwrite)
    if raw_base or args.train_interface:
        # Preserve the dataset-derived stats and processing contract for inference.
        train_proc.save_pretrained(output / "processors")
        post_proc.save_pretrained(output / "processors")
        config.save_pretrained(output / "policy_config")
        manifest["processor_sha256"] = fingerprint_tree(output, ("processors", "policy_config"))
    step = epoch = cursor = 0
    if args.resume:
        step, epoch, cursor = restore_checkpoint(args.resume, adapter, optimizer, manifest, interface_modules)
        if step >= args.steps or not 0 <= cursor < len(sampler):
            raise ValueError("resume checkpoint step/cursor incompatible with requested run")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metrics_path = output / "metrics.jsonl"
    def log_metrics(values: dict[str, Any]) -> None:
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(values) + "\n")
        print(" ".join(f"{key}={value}" for key, value in values.items()), flush=True)
    policy.train()
    iterator = None
    while step < args.steps:
        if iterator is None:
            sampler.load_state_dict({"epoch": epoch, "start_index": cursor})
            iterator = iter(loader)
        try:
            batch = next(iterator)
        except StopIteration:
            raise RuntimeError("unexpected sampler exhaustion") from None
        cursor += int(batch["action"].shape[0])
        if cursor == len(sampler):
            cursor = 0
            epoch += 1
            iterator = None
        processed = _process_batch(batch, meta.camera_keys, train_proc, selected_id)
        should_log = (step + 1) % args.log_every == 0 or step == 0
        group_grad_norms = {} if should_log else None
        loss, norm = train_step(policy, processed, optimizer, args.grad_clip, group_grad_norms=group_grad_norms)
        step += 1
        if should_log:
            log_metrics({"step": step, "loss": loss, "grad_norm": norm, **(group_grad_norms or {})})
        if args.eval_every and step % args.eval_every == 0:
            policy.eval()
            # Keep evaluation flow-matching randomness from changing training RNG.
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                torch.manual_seed(args.seed + 12345)
                losses = []
                for i, ebatch in enumerate(eval_loader):
                    if i >= args.eval_batches:
                        break
                    loss_value, _ = policy(_process_batch(ebatch, meta.camera_keys, eval_proc, selected_id))
                    if not torch.isfinite(loss_value):
                        raise RuntimeError("non-finite evaluation loss")
                    losses.append(float(loss_value))
            policy.train()
            log_metrics({"step": step, "eval_flow_loss": sum(losses) / len(losses), "eval_batches": len(losses)})
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(output / f"step_{step:08d}.pt", adapter=adapter, optimizer=optimizer,
                            manifest=manifest, step=step, epoch=epoch, cursor=cursor,
                            interface_modules=interface_modules)


if __name__ == "__main__":
    main()
