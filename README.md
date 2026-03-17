# SMHI IFCB Classify Pipeline

[![Test](https://github.com/nodc-sweden/ifcb-pytorch-classify/actions/workflows/test.yml/badge.svg)](https://github.com/nodc-sweden/ifcb-pytorch-classify/actions/workflows/test.yml)
[![Lint](https://github.com/nodc-sweden/ifcb-pytorch-classify/actions/workflows/lint.yml/badge.svg)](https://github.com/nodc-sweden/ifcb-pytorch-classify/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/nodc-sweden/ifcb-pytorch-classify/graph/badge.svg)](https://codecov.io/gh/nodc-sweden/ifcb-pytorch-classify)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%E2%80%933.12-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Pipeline for training and running inference on IFCB (Imaging FlowCytobot) plankton images using PyTorch.

## Features

- **Training** — Fine-tune 40+ pretrained architectures (ResNet, EfficientNet, ConvNeXt, Vision Transformers, etc.) on class-folder organised image datasets
- **Inference** — Batch-classify raw IFCB bins (`.roi/.adc/.hdr`) via [pyifcb](https://github.com/joefutrelle/pyifcb) and output HDF5 files in IFCB Dashboard class_scores v3 format
- **Experiment tracking** — CSV (default), MLflow, or Weights & Biases
- **Device-flexible** — Auto-detects GPU for training, defaults to CPU for inference

## Installation

Requires Python 3.11–3.12 and PyTorch.

### CPU only

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### With CUDA

PyTorch from PyPI is CPU-only. To get CUDA support, install torch first from the [PyTorch wheel index](https://pytorch.org/get-started/locally/) for your CUDA version, then install the package:

```bash
uv venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  # adjust to your CUDA version
uv pip install -e .
```
### Optional extras

```bash
uv pip install -e ".[mlflow]"   # MLflow support
uv pip install -e ".[wandb]"    # Weights & Biases support
uv pip install -e ".[plots]"    # Interactive evaluation plots (plotly)
uv pip install -e ".[dev]"      # Development tools
```

## Usage

### Training

```bash
python -m ifcb_classify train --config configs/train_default.yaml
```

With CLI overrides:

```bash
python -m ifcb_classify train --config configs/train_default.yaml \
    --model convnext_tiny --lr 0.001 --epochs 30
```

Add `--plots` to generate evaluation plots after training:

```bash
python -m ifcb_classify train --config configs/train_default.yaml --plots
```

This produces static PNG plots (training curves, per-class F1, precision vs. recall scatter, class support distribution, top confused pairs) saved to `<output_dir>/plots/<run_name>/`. With the `plots` extra installed, interactive HTML plots are also generated (zoomable confusion matrix with row-normalized percentages, sortable per-class metrics table).

Training data should be organised in class folders:

```
training_data/V1/
  Asterionellopsis_glacialis/
  Dactyliosolen_fragilissimus/
  Dinophysis_acuminata/
  ...
```

### Inference

On a directory of raw IFCB bins (`.roi/.adc/.hdr`):

```bash
python -m ifcb_classify infer \
    --input /path/to/bins \
    --model output/model_best.pt \
    --output /path/to/class_scores
```

Or with a config file:

```bash
python -m ifcb_classify infer --config configs/infer_default.yaml
```

Output is one `{sample}_class.h5` file per bin, compatible with the IFCB Dashboard, [iRfcb](https://europeanifcbgroup.github.io/iRfcb/) and [ClassiPyR](https://europeanifcbgroup.github.io/ClassiPyR/).

### Dataset normalisation

Compute mean and std for normalised transforms:

```bash
python -m ifcb_classify normalise --data-dir training_data/V1
```

## Configuration

See `configs/train_default.yaml` and `configs/infer_default.yaml` for all available options. Key training parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `resnet50` | Model architecture (see `src/ifcb_classify/models/registry.py` for full list) |
| `transform` | `dataset_squarepad_augmented` | Image preprocessing pipeline |
| `lr` | `0.0001` | Learning rate |
| `batch_size` | `64` | Batch size |
| `epochs` | `20` | Number of training epochs |
| `checkpoint_metric` | `weighted_f1` | Metric used for best-model checkpointing |
| `tracker` | `csv` | Experiment tracker (`csv`, `mlflow`, `wandb`, `none`) |
| `plots` | `false` | Generate evaluation plots after training |

### Date placeholders

Path values in YAML configs support date placeholders that are expanded at load time (UTC). This is useful for continuous inference pipelines where input/output directories are organised by date.

| Placeholder | Example value | Description |
|-------------|---------------|-------------|
| `{year}` | `2026` | Four-digit year |
| `{month}` | `03` | Zero-padded month |
| `{day}` | `14` | Zero-padded day |
| `{date}` | `20260314` | Combined `YYYYMMDD` |

Example `infer.yaml`:

```yaml
input_path: /ifcb/data/{year}
output_dir: /ifcb/output/{year}
```

## Project structure

```
src/ifcb_classify/
  cli.py                 # CLI argument parsing
  config.py              # YAML config loading
  train.py               # Training loop
  infer.py               # Inference pipeline
  normalise.py           # Dataset mean/std computation
  metrics.py             # Evaluation metrics (F1, AUROC, etc.)
  plots.py               # Evaluation plots (static + interactive)
  checkpoint.py          # Best-model saving
  hdf5_output.py         # IFCB Dashboard v3 HDF5 writer
  models/
    factory.py           # Model instantiation
    registry.py          # 40+ architecture definitions
  data/
    datasets.py          # ImageFolder datasets with transforms
    transforms.py        # SquarePad, FullPad, ReflectPad
    ifcb_bin.py          # pyifcb wrapper for raw IFCB bins
  tracking/
    csv_tracker.py       # CSV logging
    mlflow_tracker.py    # MLflow integration
    wandb_tracker.py     # W&B integration
```

## Testing

```bash
python -m pytest tests/ -v
```

## License

See [LICENSE](LICENSE).
