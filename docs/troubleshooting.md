# Troubleshooting

Common problems when installing and running the pipeline, and what to do about
them. New to the terms used here? See [Concepts & glossary](concepts.md).

## Installation

### `uv: command not found`

`uv` isn't installed (or isn't on your `PATH`). Install it from the
[uv install guide](https://docs.astral.sh/uv/getting-started/installation/),
then open a new terminal so the updated `PATH` takes effect.

### Wrong Python version

The package needs Python 3.11 or 3.12. Check with `python --version`. If your
default Python is older or newer, point `uv venv` at a compatible interpreter,
e.g. `uv venv --python 3.12`.

### `ifcb-classify: command not found`

The virtual environment isn't active. Activate it first
(`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows), or use
the module form `python -m ifcb_classify ...`, which the guides use
interchangeably.

## GPU / CUDA

### PyTorch doesn't see my GPU

`torch.cuda.is_available()` returns `False` even though you have an NVIDIA GPU.
The usual cause is that the **CPU-only** build of PyTorch got installed (the
default from PyPI). Reinstall the CUDA build for your CUDA version, then
reinstall the package:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130  # match your CUDA version
uv pip install -e .
```

See [Installation → With CUDA](installation.md#with-cuda). Confirm the fix with
the check in [Verify the install](installation.md#verify-the-install).

### CUDA out of memory

Training aborts with `CUDA out of memory` (or the machine freezes). The batch or
the model is too big for your GPU's VRAM. In order of preference:

- **Lower `batch_size`** (e.g. `--batch-size 32`, then `16`). This is the first
  thing to try.
- **Use a smaller model** (e.g. `--model resnet18` or `convnext_tiny` instead of
  the default `resnet50`).
- **Shrink the input** with `--image-width`/`--image-height` if you raised them.

For YOLO chain training, pass a smaller `--batch` (or let `--batch -1`
auto-pick), or a smaller `--imgsz`. See
[Chain-counting annotation → Final training on GPU](guides/chain-counting-annotation.md#final-training-on-gpu).

### Training is slow even though it's on the GPU

A batch (or image size) that's slightly too big doesn't always fail with a clean
`CUDA out of memory`. Instead the driver can spill the overflow from dedicated
VRAM into the **shared GPU memory pool** (system RAM). Training keeps running on
the GPU, but every batch now shuttles data across the PCIe bus, so it slows to a
crawl — often far slower than it would be with a smaller batch that fits in
dedicated VRAM. This is easy to hit when training the YOLO chain detector at a
large `--imgsz` with a big `--batch`.

How to spot it: watch memory while training. In `nvidia-smi` (or, on Windows,
Task Manager → Performance → GPU) the **dedicated** VRAM is pinned at its limit
while **shared** GPU memory keeps climbing.

The fix is the same lever as
[CUDA out of memory](#cuda-out-of-memory): **lower `batch_size`** (or `--batch`
for `chains-train`), and/or reduce `--imgsz`, until the run fits inside dedicated
VRAM. Counterintuitively, a smaller batch that stays in VRAM trains much faster
than a larger one that spills.

## Training

### Training is extremely slow

You're almost certainly on CPU. Confirm with the
[install check](installation.md#verify-the-install); if it prints
`CUDA available: False`, every epoch runs on CPU. Either
[install the CUDA build](installation.md#with-cuda), or keep runs small for a
demo (fewer `--epochs`, a lighter `--model` like `convnext_tiny`). See
[Training → Hardware and training time](guides/training.md#hardware-and-training-time).

If the check prints `CUDA available: True` and it's still slow, the GPU may be
spilling VRAM into shared system memory — see
[Training is slow even though it's on the GPU](#training-is-slow-even-though-its-on-the-gpu).

### `FileNotFoundError` on the config or data directory

Paths in the guides are relative to the repository root. Run commands from the
directory you cloned into, and check that `--config`, `--data-dir`, and
`--input` point at paths that exist. Configs also expand date placeholders like
`{year}` at load time — see [Configuration](configuration.md#date-placeholders).

### Some classes score near-zero F1

Usually **too few images** in those classes, not a bug. Collect more, or exclude
tiny classes with `min_class_images` (keeping named ones via
`manual_include_classes`). See
[Training → Reading the results](guides/training.md#reading-the-results).

## Inference

### `--model` file not found / which checkpoint do I use?

Training writes a checkpoint named from the run settings (e.g.
`example-resnet50_..._best.pt`), **not** a fixed `model_best.pt`. Point `--model`
at whatever `*_best.pt` file landed in your `output_dir`. See
[Training → What training produces](guides/training.md#what-training-produces).

### Loading a checkpoint fails / weights-only error

Checkpoints saved outside this pipeline (raw state dicts) need unsafe pickle
loading and the class list supplied explicitly. Add `--allow-unsafe` and
`--classes`. Only do this for checkpoints you trust — see
[Inference → Legacy checkpoints](guides/inference.md#legacy-checkpoints).

### Inference wrote nothing / skipped files

By default `infer` skips bins whose output `.h5` already exists. Pass
`--overwrite` to regenerate them.
