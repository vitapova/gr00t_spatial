# GR00T spatial cross-attention

Experimental LeRobot GR00T N1.7 adaptation for ARM100 / SO100 robot data. It
adds gated spatial cross-attention from early Qwen layers to the final vision
encoder features, while retaining the original image-token path and action
head. Optional Qwen LoRA and per-embodiment state/action-interface training can
be enabled or ablated independently.

## Start here

- [Training, ablations, checkpoints and setup](groot_spatial/TRAINING.md)
- [Offline inference demo](DEMO_ARM100.md)
- [Adapter design and limitations](groot_spatial/README.md)

Run commands from this repository root so Python can import `groot_spatial`:

```bash
python -m groot_spatial.train --help
python demo_arm100.py --help
```

The project targets LeRobot commit
`b65b05f7f909de72f02ef4ba7854ad40230aa722`. Installation requires a compatible
LeRobot environment with the `groot` and `training` extras and access to NVIDIA
model assets. See the setup document before running training.

Robot data, pretrained models and trained checkpoints are **not included**.
The offline demo does not connect to hardware or issue motor commands. Full
checkpoint/GPU and real-robot behavior must be verified on the target system;
the included CPU integration tests use a small randomly initialized model and
a tokenizer stub.

```bash
python -m pytest tests -q
```

The optional ARM100 integration test reads an extracted dataset when
`ARM100_TEST_ROOT` is set; without the optional dependencies or data it skips.
