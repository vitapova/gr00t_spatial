# Spatial cross-attention for LeRobot GR00T N1.7

This is an experimental **adapter**, not a separate robot policy. For ARM100
training with scoped Qwen LoRA, see [TRAINING.md](TRAINING.md). It targets the LeRobot `groot` / N1.7 `Qwen3Backbone` layout, checked
against LeRobot commit `b65b05f7f909de72f02ef4ba7854ad40230aa722`.
The original vision-token insertion, Qwen DeepStack and GR00T action head /
flow-matching loss remain in the forward path.

The final vision block is intercepted before its patch merger. Qwen's
merge-block ordering is converted to a 2D grid for each image, then flattened
in row-major order. Each reader has its own learned coordinate and image-order
embeddings, Q/K/V projections, output projection, and near-zero sigmoid gate.
At each selected lower decoder block **input**, the reader output is added to
the existing hidden states as a residual. Layer indices are zero-based; the
default `(0, 1)` inserts before the first and second decoder blocks. The
camera embedding represents the *order in the input*, not an intrinsic camera
identity, so preprocessing must use the same camera order during training and
deployment. Qwen's pretrained visual tower already has positional information;
the additional coordinates provide an explicit 2D addressing signal.

To prevent leaking an image into earlier causal positions, its patches are
visible only at or after the final image placeholder of that image. Padding is
masked, and workspaces are kept separate across batch items. This is a
**per-observation feature workspace**: there is no temporal memory or 3D world
map. Video inputs, examples without images, concurrent forwards, and Qwen
gradient checkpointing during training are intentionally unsupported. Keep
Qwen gradient checkpointing disabled; don't use `torch.compile` with these
dynamic hooks. Output equivalence and gradients must be rechecked if changing
LeRobot / Transformers versions.

Separately, `--mask-vl-self-attention` optionally masks padding keys in the
action-head VL self-attention projector; this is independent of the spatial
reader and does not alter action padding or the flow-matching loss. Its setting
is stored with the training checkpoint and restored for inference.

## Adapter-only experiment: API example

This section describes the original reader-only API. For the new optional
robot-interface training and interface-only ablation, use
[TRAINING.md](TRAINING.md) rather than the reader-only example below.

Install the adapter *after* loading the base policy, but **before** creating
the optimizer, a distributed wrapper, or restoring the adapter weights. Use a
real, compatible N1.7 LeRobot policy and its corresponding processor / dataset
pipeline; this snippet is not a standalone training command:

```python
import torch
from groot_spatial import SpatialAdapterConfig, install_spatial_adapter

# policy = ...  # load a LeRobot GrootPolicy and the matching pre/post processors
# Move policy to its final device before installation, or move the complete policy afterwards.
policy.requires_grad_(False)  # adapter-only: freeze vision, Qwen AND action head
adapter = install_spatial_adapter(
    policy,
    SpatialAdapterConfig(layer_indices=(0, 1), attention_dim=512, num_heads=8),
)
optimizer = torch.optim.AdamW(adapter.readers.parameters(), lr=1e-4)

# In your existing LeRobot training loop, using real preprocessed batches:
# policy.train()
# loss, metrics = policy(batch)  # original GR00T flow-matching loss
# optimizer.zero_grad(set_to_none=True)
# loss.backward()
# optimizer.step()
```

Do not run Qwen under `torch.no_grad()` during adapter training: the frozen
Qwen and action-head operations must remain in the autograd graph so loss
gradients can reach the readers. The intended comparison can start from the
same previously fine-tuned ARM100 checkpoint, or from the base checkpoint; the
choice must be recorded because it changes the hypothesis being tested. A
fully frozen base checkpoint that was never adapted to this robot may not have
usable action mappings. The adapter does not itself train a new embodiment.

With **no LoRA installed**, disable the branch using `adapter.enabled = False`
for an exact original backbone comparison. When LoRA is installed, also call
`set_lora_enabled(adapter, False)` to disable it; `adapter.enabled` controls
only the spatial branch. For parity tests compare
backbone outputs in evaluation mode with identical inputs. The GR00T loss and
action generation are stochastic unless their random inputs are held fixed.

## Saving the sidecar

Wrapping changes the state-dict prefix of the original backbone. Do **not**
assume that a normal LeRobot `save_pretrained` / `from_pretrained` round trip
will reconstruct this custom architecture. Save the existing base policy using
its normal pipeline **before** installation and save the adapter separately:

```python
torch.save({
    "metadata": adapter.adapter_metadata(),
    "weights": adapter.adapter_state_dict(),
    "base_policy_identifier": "EXACT_BASE_CHECKPOINT_AND_REVISION",
}, "spatial_adapter.pt")
```

For inference/resume, load the exact same base policy and processors first,
install the adapter with the same `SpatialAdapterConfig`, then:

```python
saved = torch.load("spatial_adapter.pt", map_location="cpu", weights_only=True)
if saved["metadata"] != adapter.adapter_metadata():
    raise ValueError("adapter configuration does not match checkpoint")
adapter.readers.load_state_dict(saved["weights"], strict=True)
```

Save optimizer, scheduler, step, random states, dataset split/fingerprint,
LeRobot/Transformers versions and base checkpoint revision separately for an
actual resumable experiment. Verify the base checkpoint identity rather than
using the example placeholder. Loading a sidecar on a different base, camera
ordering, preprocessing, or embodiment is not guaranteed to be meaningful.

## Tests and deployment status

`python -m pytest tests -q` tests grid ordering,
batch/camera isolation, causality, gate/gradients, BF16 autocast, checkpoint
state and guards. When Transformers is available it also constructs a tiny
random **real Qwen3-VL** with DeepStack and tests its forward/backward; no model
weights are downloaded. An optional integration test exercises the real LeRobot
`GrootPolicy`, N1.7 backbone, original flow-matching action head and action
inference, assembled with tiny random weights. The Qwen tests were run against
Transformers 4.57.3 and 5.22.1; the LeRobot integration was run with 5.22.1, all
on CPU. None of these tests used the NVIDIA checkpoint, GPU, the ARM100 dataset,
or a robot. Before using it in an experiment, run an
end-to-end preprocessed ARM100 batch, check the loss and adapter gradients,
verify exact disabled-branch parity and checkpoint reload, and measure memory
and latency on the target GPU. Do not deploy unvalidated outputs to hardware.
