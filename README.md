# SMHI IFCB Classify Pipeline

[![Test](https://github.com/nodc-sweden/ifcb-pytorch-classify/actions/workflows/test.yml/badge.svg)](https://github.com/nodc-sweden/ifcb-pytorch-classify/actions/workflows/test.yml)
[![Lint](https://github.com/nodc-sweden/ifcb-pytorch-classify/actions/workflows/lint.yml/badge.svg)](https://github.com/nodc-sweden/ifcb-pytorch-classify/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/nodc-sweden/ifcb-pytorch-classify/graph/badge.svg)](https://codecov.io/gh/nodc-sweden/ifcb-pytorch-classify)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%E2%80%933.12-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docs](https://github.com/nodc-sweden/ifcb-pytorch-classify/actions/workflows/docs.yml/badge.svg)](https://nodc-sweden.github.io/ifcb-pytorch-classify/)

Pipeline for training and running inference on IFCB (Imaging FlowCytobot) plankton images using PyTorch.

📖 **Full documentation: <https://nodc-sweden.github.io/ifcb-pytorch-classify/>**

## Capabilities

- **Training** — Fine-tune 40+ pretrained architectures (ResNet, EfficientNet, ConvNeXt, Vision Transformers, etc.) on class-folder organised image datasets, with optional evaluation plots (static PNG + interactive HTML)
- **Inference** — Batch-classify raw IFCB bins (`.roi/.adc/.hdr`) via [pyifcb](https://github.com/joefutrelle/pyifcb) into HDF5 files in IFCB Dashboard class_scores v3 format, with per-class decision thresholds; output works with the IFCB Dashboard, [iRfcb](https://europeanifcbgroup.github.io/iRfcb/) and [ClassiPyR](https://europeanifcbgroup.github.io/ClassiPyR/)
- **Chain / cell counting** — Optionally train per-taxon [YOLO](https://docs.ultralytics.com/) detectors that count individual cells in colony ROIs and store the count alongside each classification
- **Experiment tracking** — CSV (default), MLflow, or Weights & Biases
- **Built for pipelines** — Date-placeholder paths for date-organised continuous inference, and automatic device selection (GPU for training, CPU by default for inference)

## Quick start

Requires Python 3.11–3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
uv pip install -e .
```

Train a classifier, then classify a directory of raw IFCB bins:

```bash
python -m ifcb_classify train --config configs/train_default.yaml

python -m ifcb_classify infer \
    --input /path/to/bins \
    --model output/model_best.pt \
    --output /path/to/class_scores
```

See [Getting started](docs/getting-started.md) for the full walkthrough, and
[Installation](docs/installation.md) for CUDA and optional extras.

## Documentation

| Topic | Description |
|---|---|
| [Getting started](docs/getting-started.md) | Install, train, and run inference end to end |
| [Installation](docs/installation.md) | CPU, CUDA, and optional extras |
| [Training](docs/guides/training.md) | Training options, plots, normalisation |
| [Inference](docs/guides/inference.md) | Classifying raw bins, output format |
| [Chain counting](docs/guides/chain-counting.md) | Per-taxon YOLO cell counting |
| [Annotation & training](docs/guides/chain-counting-annotation.md) | Labelling workflow for chain detectors |
| [Configuration](docs/configuration.md) | Config parameters and date placeholders |

The rendered site (with the auto-generated API reference) is at
<https://nodc-sweden.github.io/ifcb-pytorch-classify/>.

## Testing

```bash
python -m pytest tests/ -v
```

## Citation

If you use this software, please cite it. GitHub's **"Cite this repository"**
button (top right) generates APA/BibTeX from [`CITATION.cff`](CITATION.cff), or
use:

```bibtex
@software{torstensson_ifcb_classify,
  author  = {Torstensson, Anders},
  title   = {ifcb-classify: a PyTorch pipeline for IFCB plankton image classification},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/nodc-sweden/ifcb-pytorch-classify}
}
```

## License

See [LICENSE](LICENSE).
