# SMHI IFCB Classify Pipeline

Production pipeline for training and running inference on IFCB (Imaging FlowCytobot) plankton images using PyTorch.

## Features

- **Training** — Fine-tune 40+ pretrained architectures (ResNet, EfficientNet, ConvNeXt, Vision Transformers, etc.) on class-folder organised image datasets
- **Inference** — Batch-classify raw IFCB bins (`.roi/.adc/.hdr`) via [pyifcb](https://github.com/joefutrelle/pyifcb) and output HDF5 files in IFCB Dashboard class_scores v3 format
- **Experiment tracking** — CSV (default), MLflow, or Weights & Biases
- **Config-driven** — YAML configs with CLI overrides for all parameters
- **Device-flexible** — Auto-detects GPU for training, defaults to CPU for inference

## Installation

Requires Python 3.11+ and PyTorch. Install PyTorch first following https://pytorch.org/get-started/locally/, then:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

For experiment tracking extras:

```bash
uv pip install -e ".[mlflow]"   # MLflow support
uv pip install -e ".[wandb]"    # Weights & Biases support
```

For development:

```bash
uv pip install -e ".[dev]"
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

Training data should be organised in class folders:

```
training_data/V1/
  Asterionellopsis/
  Chaetoceros/
  Dinophysis/
  ...
```

### Inference

On a directory of raw IFCB bins:

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

Output is one `{sample}_class.h5` file per bin, compatible with the IFCB Dashboard.

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
| `transform` | `dataset_squarepad` | Image preprocessing pipeline |
| `lr` | `0.0001` | Learning rate |
| `batch_size` | `64` | Batch size |
| `epochs` | `20` | Number of training epochs |
| `checkpoint_metric` | `weighted_f1` | Metric used for best-model checkpointing |
| `tracker` | `csv` | Experiment tracker (`csv`, `mlflow`, `wandb`, `none`) |

## Project structure

```
src/ifcb_classify/
  cli.py                 # CLI argument parsing
  config.py              # YAML config loading
  train.py               # Training loop
  infer.py               # Inference pipeline
  normalise.py           # Dataset mean/std computation
  metrics.py             # Evaluation metrics (F1, AUROC, etc.)
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
