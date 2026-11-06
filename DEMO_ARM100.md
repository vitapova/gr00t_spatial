# Offline demo of a trained ARM100 policy

`demo_arm100.py` loads the original N1.7 checkpoint, the selected trained
sidecar, and its saved preprocessors/postprocessors. It reads one observation
from a LeRobot ARM100 episode, predicts an action chunk and displays predicted
and recorded actions side by side. **It never connects to or moves a robot.**

Run it in the same environment used for training, from the project directory:

```bash
python demo_arm100.py \
  --base-checkpoint /path/to/GR00T-N1.7-3B \
  --run-dir /path/to/training-run \
  --checkpoint /path/to/training-run/step_00010000.pt \
  --dataset-root /path/to/pusht_merged_03_v3.0 \
  --device cuda
```

Use an actual saved `step_XXXXXXXX.pt`, not the `model.safetensors` from the
base. For resumed training, `--run-dir` must point to the directory containing
that run's `processors/` and `policy_config/`. The loader verifies these files
and the base checkpoint against the sidecar. It can take time to hash the
checkpoint and the dataset. The process needs sufficient GPU memory, the
training environment's LeRobot version, and access to Cosmos-Reason2-2B.

By default the demo uses frame 0 of the **first validation episode recorded in
the sidecar**. To choose a specific frame and print more of the chunk:

```bash
python demo_arm100.py \
  --base-checkpoint /path/to/GR00T-N1.7-3B \
  --run-dir /path/to/training-run \
  --checkpoint /path/to/training-run/step_00010000.pt \
  --dataset-root /path/to/pusht_merged_03_v3.0 \
  --episode 12 --frame 100 --show-steps 10 --seed 42 --device cuda
```

`--frame` is zero-based **within** the episode. The dataset fingerprint must
match the training run. To intentionally use another compatible dataset, pass
`--allow-dataset-mismatch --episode N`; its train/validation membership is
reported as unknown. The decoder defaults to `--video-backend pyav`.

The six action components follow the dataset's joint order printed by the
script. Predictions are passed through the saved postprocessor into the
dataset's action units; recorded actions are used **only for comparison**, not
fed to the model. Padding beyond the end of an episode is not included in the
per-joint MAE. This is an open-loop snapshot, not a success-rate or safety
evaluation. Equal random seeds make repeated predictions reproducible within a
compatible execution environment; cross-device bitwise parity is not promised.

## Reuse on a single observation

```python
from groot_spatial.load import load_experiment
from demo_arm100 import predict_observation

policy, preprocessor, postprocessor, adapter = load_experiment(
    sidecar_path="/path/to/training-run/step_00010000.pt",
    base_checkpoint="/path/to/GR00T-N1.7-3B",
    run_dir="/path/to/training-run",
    device="cuda",
)

observation = {
    "observation.images.front": front_rgb_chw,
    "observation.images.wrist": wrist_rgb_chw,
    "observation.state": state_six_values,
    "task": "Grasp the green cube and put it on the orange cube",
}
actions = predict_observation(policy, preprocessor, postprocessor, observation, seed=42)
print(actions.shape)  # [predicted_horizon, 6]
```

Images must be RGB tensors shaped `[3,H,W]`, either `uint8` in `[0,255]` or
floating point in `[0,1]`. State must have six finite values in the same order
and units as the training dataset. This function returns data; it does not
enforce motor safety limits or execute commands. Validate calibration, action
format, limits, latency and emergency stop before any separately implemented
robot deployment.

The demo was tested with actual ARM100 frames/actions, the LeRobot processors
and a tiny randomly initialized Qwen/GR00T plus a tokenizer stub. Full N1.7
checkpoint loading and inference on a GPU have not been tested here.
