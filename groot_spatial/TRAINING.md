# ARM100: spatial reader + Qwen LoRA

`python -m groot_spatial.train` performs single-device, offline training using
the real LeRobot dataset and GR00T N1.7 flow-matching objective. By default it
freezes the base weights and trains the spatial readers plus LoRA in Qwen
self-attention `q_proj` and `o_proj` (defaults: blocks 0–3, rank 8, alpha 16).
LoRA has a zero-initialised output projection, like memWMLM's application-side
adapter. The new `--train-interface` option additionally trains the selected
embodiment's full state encoder, action encoder and action decoder; the other
embodiments and the shared action DiT remain frozen. LoRA and spatial layers
are independently configurable.

**Status:** the training entry point has been smoke-tested with real ARM100
frames/actions, the actual LeRobot pack/normalization pipeline, a tiny random
Qwen/GR00T, and a stub tokenizer/checkpoint loader. It has **not** been run
with the NVIDIA or your robot checkpoint, the real Cosmos tokenizer, or a GPU.
It is not a validated robot policy. Do not deploy its output without further
offline checks and a supervised robot safety procedure.

## Setup

- Use a Linux NVIDIA CUDA machine with sufficient VRAM, Python and CUDA
  versions compatible with LeRobot. The code was checked against LeRobot
  commit `b65b05f7f909de72f02ef4ba7854ad40230aa722` / Transformers 5.22.1.
  Install the corresponding LeRobot source checkout with its `groot` and
  `training` extras, for example `pip install -e '.[groot,training]'` from
  that checkout. Make this project's directory available in `PYTHONPATH`.
- Obtain access to the gated `nvidia/Cosmos-Reason2-2B` model and authenticate
  with Hugging Face. GR00T and its Qwen processor may need local model assets
  or a network connection on first load.
- Extract `arm100/pusht_merged_03_v3.0.tar` into a **new empty directory** on
  the training machine. `--dataset-root` must point to the directory that
  directly contains `meta/`, `data/` and `videos/`, not the tar file.
- Provide a **local GR00T N1.7 checkpoint directory**, either the raw NVIDIA
  checkpoint (with config and safetensors) or a LeRobot-saved N1.7 policy's
  `pretrained_model/` directory, including its saved pre/post processors.
  N1.5 is not supported by the checked LeRobot version. The older fine-tuned
  checkpoint's version and availability have not yet been verified.

## Run

Example, from this project's directory:

```bash
python -m groot_spatial.train \
  --dataset-root /path/to/pusht_merged_03_v3.0 \
  --base-checkpoint /path/to/n1.7-checkpoint \
  --output-dir /path/to/new-run \
  --device cuda --batch-size 1 --steps 10000 --train-interface \
  --save-every 500 --eval-every 500
```

By default, `--output-dir` must not exist. Add `--force-overwrite` to delete
the existing output directory **and everything in it**, then start a new run.
The directory cannot contain the dataset, base checkpoint or resume checkpoint.
The run writes `manifest.json`, `metrics.jsonl`, and `step_XXXXXXXX.pt` checkpoints.
At each logging step `metrics.jsonl` includes pre-clipping gradient norms for
each enabled optimizer group (`grad_norm_interface`, `grad_norm_spatial`, and
`grad_norm_lora` as applicable). These help diagnose disconnected or weak
branches; they do not prove that a branch improves robot performance.
Checkpoint creation includes optimizer state, random states, sampler position,
reader, LoRA and (when enabled) interface weights. To resume, supply `--resume /path/to/step_XXXXXXXX.pt`
and a **new** `--output-dir`, repeating the same configuration arguments;
`--steps` can be increased. A mismatch in checkpoint/dataset contents,
episode split, hyperparameters, software versions or source files is rejected.
Resume checkpoints created with a different version of the training source are
rejected. Version-1 sidecar-only checkpoints can still be loaded for inference,
but cannot resume training with this new source/checkpoint format.
Resume is single-process, `num_workers=0`; bitwise GPU reproducibility is not
guaranteed across drivers/kernels. Checkpoints are written exclusively and an
interrupted write can leave an incomplete file: use the previous checkpoint.

The script detects whether the base is raw NVIDIA or a saved LeRobot policy:

- **Raw N1.7:** configure `new_embodiment`, use absolute actions, and choose
  the horizon with `--chunk-size` (default 40). LeRobot's dataset statistics
  provide normalization when checkpoint statistics are absent. The resolved
  policy config and both processors are saved in `policy_config/` and
  `processors/` for later loading. When `--train-interface` is enabled, the
  default slot is ID 10; `--embodiment-id N` selects another unused ID. The
  script validates the slot against the checkpoint's embodiment mapping,
  explicitly installs the mapping in both processors, checks each batch's ID,
  and persists it for inference. Without this flag, the previous processor
  behavior (including any fallback) is unchanged. A frozen raw action interface
  may not already have useful output mappings for SO100.
- **Saved LeRobot policy:** use its existing embodiment, action horizon,
  relative/absolute action conventions and processor statistics. Camera keys
  must be exactly `observation.images.front` and `observation.images.wrist`,
  and the action dimension must be six. No silent key renaming is performed.
  `--chunk-size` does not override a saved policy's horizon. With
  `--train-interface`, its existing explicit embodiment ID is used and cannot
  be changed to another slot.

This distinction matters for comparison: starting from the same **raw** base
as a conventional fine-tuning run tests an alternative adaptation. Starting
from a previously fine-tuned ARM100 policy tests an **additional improvement
on top of it**, not an alternative from the same starting point.

## Controlled ablations

Use the same base checkpoint, episode split, seed, horizon, steps and other
hyperparameters, but **separate new output directories**. With a raw checkpoint
all three use the same default ID 10 (or pass the same `--embodiment-id`):

```text
A. --train-interface --no-spatial --no-lora  # interface only
B. --train-interface --no-spatial            # interface + Qwen LoRA
C. --train-interface                         # interface + LoRA + spatial
```

These are flags added to the command in the Run section; omit any conflicting
flags already in that command. They start new experiments, not resumes of the
previous sidecar-only run. Only the selected embodiment row is optimized; it is
stored separately from the frozen 32-row banks, avoiding updates and AdamW
weight decay on unrelated rows. The encoder and decoder include nonlinear
layers and receive the normal GR00T inputs. The `--no-spatial` baseline skips
the cross-attention completely; `--no-lora` does not install LoRA layers.

## Splits and evaluation

By default, the script holds out a deterministic random 10% of *episodes*,
not frames. `--seed` and `--eval-fraction` control this. To match an existing
experiment exactly, supply `--split-file /path/split.json`, e.g.:

```json
{"train": [0, 1, 2], "eval": [3, 4]}
```

The two nonempty lists must be disjoint and contain valid episode indices;
they need not cover all episodes. All IDs are saved in the manifest. LeRobot
constructs consecutive action chunks and the sampler drops episode tails that
cannot provide a complete chunk. Images, state, task text and action targets all
come from the real dataset. Eval uses separate, non-augmenting preprocessing;
the reported `eval_flow_loss` is a diagnostic, not robot success rate.

**Leakage caveats:** if the existing fine-tuned checkpoint was trained on all
194 episodes, the held-out episodes are not unseen by its base weights. The
dataset's global normalization statistics also include held-out episodes when
used for a raw checkpoint. For a clean comparison, use the same train/held-out
split and normalization statistics for every method, ideally compute stats
from training episodes only, and document any prior exposure. These scripts do
not recompute training-only statistics.

## Reload for inference

Load the **same base policy and processors**, then install the sidecar. The
helper below reconstructs them for either base format:

```python
from groot_spatial.load import load_experiment

policy, preprocessor, postprocessor, adapter = load_experiment(
    sidecar_path="/path/to/new-run/step_00010000.pt",
    base_checkpoint="/path/to/n1.7-checkpoint",
    run_dir="/path/to/new-run",
    device="cuda",
)
# Use policy with these matching processors; this does not control a robot.
```

The loader verifies the base checkpoint and, for raw-base runs, the saved
processors and policy config against the recorded SHA-256. If loading a policy
yourself, the lower-level `load_trained_sidecar` helper cannot prove that the
supplied in-memory `policy` was actually loaded from the given path. The loader
has not been tested with a full NVIDIA checkpoint. Do not confuse the sidecar checkpoint with a standalone
LeRobot `model.safetensors`: the unmodified base and its processors are still
required. For baseline ablations disable both `adapter.enabled` and
`set_lora_enabled(adapter, False)`; re-enable both to evaluate the trained
variant. Fix noise/seeds when comparing stochastic action outputs.

## Limitations

- No multi-GPU, video inputs, Qwen activation checkpointing or `torch.compile`.
- Camera identity is image order, so preprocessing order must remain unchanged.
- The workspace describes the current observation; it is not temporal memory.
- Previously created root-level `train_arm100_spatial.py` and
  `gr00t_spatial_injection.py` are obsolete placeholders; do not use them.
