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

Requires Python 3.11–3.12, PyTorch, and [uv](https://docs.astral.sh/uv/getting-started/installation/).

### CPU only

**Linux/macOS:**
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

**Windows:**
```powershell
uv venv
.venv\Scripts\activate
uv pip install -e .
```

### With CUDA

PyTorch from PyPI is CPU-only. To get CUDA support, install torch first from the [PyTorch wheel index](https://pytorch.org/get-started/locally/) for your CUDA version, then install the package:

**Linux/macOS:**
```bash
uv venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130  # adjust to your CUDA version
uv pip install -e .
```

**Windows:**
```powershell
uv venv
.venv\Scripts\activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130  # adjust to your CUDA version
uv pip install -e .
```
### Optional extras

```bash
uv pip install -e ".[mlflow]"   # MLflow support
uv pip install -e ".[wandb]"    # Weights & Biases support
uv pip install -e ".[chains]"   # YOLO chain counting (see below)
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

This produces static PNG plots (training curves, per-class F1, precision vs. recall scatter, class support distribution, top confused pairs) saved to `<output_dir>/plots/<run_name>/`.  Interactive HTML plots are also generated (zoomable confusion matrix with row-normalized percentages, sortable per-class metrics table).

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

Legacy checkpoints (raw state dicts saved outside this pipeline) require unsafe pickle loading. Add `--allow-unsafe` to permit this:

```bash
python -m ifcb_classify infer \
    --input /path/to/bins \
    --model /path/to/legacy_model.pt \
    --classes /path/to/classes.txt \
    --allow-unsafe
```

Or with a config file:

```bash
python -m ifcb_classify infer --config configs/infer_default.yaml
```

Output is one `{sample}_class.h5` file per bin, compatible with the IFCB Dashboard, [iRfcb](https://europeanifcbgroup.github.io/iRfcb/) and [ClassiPyR](https://europeanifcbgroup.github.io/ClassiPyR/).

### Chain counting (optional)

Some plankton form chains of many cells in a single ROI (e.g. *Skeletonema*).
For these taxa, classification alone tells you *what* the ROI is but not *how
many* cells it contains. The optional chain-counting feature trains a small
[YOLO](https://docs.ultralytics.com/) object detector **per taxon** that counts
individual cells, and (during inference) stores the count alongside the
classification result.

This approach follows Groves et al. (2026), who demonstrated automatic
enumeration of marine diatom chains with YOLO:

> Groves, G. J. J., Arthur, G., Bresnan, E., Whyte, C., Arce, P., & Davidson, K.
> (2026). Automatic enumeration of chains of marine diatoms using "You Only Look
> Once"—a machine learning approach. *Journal of Plankton Research*, 48(2),
> fbaf064. https://doi.org/10.1093/plankt/fbaf064

Requires the `chains` extra:

```bash
uv pip install -e ".[chains]"
```

#### Training a detector for any chain-forming taxon

Train one detector per taxon you want to count. This works for any chain-forming
species — bring your own annotated data.

```bash
python -m ifcb_classify chains-train --config configs/chains_train_default.yaml
```

With CLI overrides (e.g. a larger model on a GPU):

```bash
python -m ifcb_classify chains-train \
    --class-name Skeletonema --data /path/to/datasets/skeletonema \
    --model yolo11x.pt --epochs 200 --device 0
```

The best checkpoint is written to `<project>/<name>/weights/best.pt`.

**Dataset layout** — object detection needs *bounding boxes* around individual
cells (the class-folder data used for classification has none), so ROIs must be
annotated first (e.g. with [Label Studio](https://labelstud.io/),
[CVAT](https://www.cvat.ai/), or [Roboflow](https://roboflow.com/)). Export in
YOLO format:

```
datasets/skeletonema/
  data.yaml                 # names + train/val image dirs
  images/train/*.png        labels/train/*.txt   # one .txt of boxes per image
  images/val/*.png          labels/val/*.txt
```

Each label `.txt` holds one line per cell: `class_id cx cy w h` (normalised
0–1). With a single taxon per detector, `class_id` is always `0`. A `data.yaml`:

```yaml
path: /abs/path/to/datasets/skeletonema
train: images/train
val: images/val
names:
  0: skeletonema
```

`--data` accepts either a `data.yaml` file or a directory containing one
(`data.local.yaml` is preferred over `data.yaml` when both exist).

**Compute** — `yolo11n.pt` (nano) trains in ~hours on CPU and is a good starting
point; use a larger model (`yolo11x.pt`) on a GPU (`--device 0`) for best
accuracy. CUDA requires a CUDA build of PyTorch (see [With CUDA](#with-cuda)).

See `configs/chains_train_default.yaml` for all options.

#### Counting during inference

Add a `chain_counting` block to your inference config to count cells while
classifying. Only ROIs whose **thresholded `class_name`** matches a configured
key are counted; all other ROIs get `chain_count = -1`.

```yaml
chain_counting:
  enabled: true
  conf: 0.25            # default; per-model override allowed
  iou: 0.30             # default; per-model override allowed
  models:
    Skeletonema_marinoi:
      weights: /models/chains/chains_skeletonema_yolo11n/weights/best.pt
      iou: 0.30
    # Several labels may share one detector (e.g. species + genus-level class):
    # Thalassiosira_spp: { weights: /models/chains/thalassiosira_best.pt }
```

> **Keys must match the classifier's output labels exactly.** A detector is a
> single-class "cell vs. not-cell" model, so one detector typically serves all
> species of a genus plus the genus-level class — map each label to the same
> weights.

```bash
python -m ifcb_classify infer --config configs/infer_with_chains.yaml
python -m ifcb_classify infer --config configs/infer_with_chains.yaml --no-count  # disable
```

The output `_class.h5` gains a `chain_count` dataset (int32, one per ROI; `-1`
where not counted) and a `chain_counter_models` JSON attribute recording the
weights/IoU/conf used. Existing consumers ignore the extra dataset. See
`configs/infer_with_chains.yaml` for a full example.

#### Counting on already-classified bins

If you already have `_class.h5` files and only want to add (or refresh) counts —
e.g. after training a new detector — use `chains-count` instead of re-running
`infer`. It reuses the stored `class_name` to decide which ROIs to count, so it
**skips the classifier entirely** and only runs the detector on the matching
ROIs, reading their pixels from the raw bins:

```bash
# Reuse the same inference config (input_path = raw bins, output_dir = the
# directory of existing *_class.h5 files, plus the chain_counting block):
python -m ifcb_classify chains-count --config configs/infer_with_chains.yaml

# Or point at the two directories directly:
python -m ifcb_classify chains-count \
    --input /path/to/raw/bins \
    --output output/class_scores \
    --config configs/infer_with_chains.yaml   # still needed for the detector block
```

Each file's `chain_count` dataset is written in place. Files that already carry
counts are skipped unless you pass `--overwrite`. The raw bins are still required
(the `.h5` stores scores, not pixels), but the expensive ResNet pass is avoided.

#### Validating count accuracy

`chains-eval` compares a detector's predicted counts against manual counts and
sweeps the NMS IoU so you can pick the best value per taxon. Provide a directory
of test images and a CSV with a filename column and an integer count column
(`file_name,cell_count`):

```bash
python -m ifcb_classify chains-eval \
    --weights output/chains/chains_skeletonema_yolo11n/weights/best.pt \
    --images /path/to/test_images \
    --counts-csv /path/to/test_image_counts.csv \
    --ious 0.3,0.5,0.7
```

It reports MAE, mean bias, exact-match and within-±1 accuracy, and total counts
per IoU. Add `--output results.csv` for per-image predictions.

**Checking one detector across species** — to verify that a single genus-level
detector generalises (rather than training per species), run the *same*
`--weights` against each species' test set and compare the metrics. Train a
dedicated detector only if a particular species shows high error. See
`configs/chains_eval_default.yaml`.

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
  chains/                # Optional YOLO chain counting (requires [chains] extra)
    config.py            # ChainTrainConfig + ChainCountingConfig + ChainEvalConfig
    train.py             # Per-taxon YOLO detector training
    counter.py           # Per-taxon cell counting at inference time
    eval.py              # Count-accuracy validation + IoU sweep
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
