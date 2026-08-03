# SMHI IFCB Classify Pipeline

[![Test](https://github.com/nodc-sweden/ifcb-pytorch-classify/actions/workflows/test.yml/badge.svg)](https://github.com/nodc-sweden/ifcb-pytorch-classify/actions/workflows/test.yml)
[![Lint](https://github.com/nodc-sweden/ifcb-pytorch-classify/actions/workflows/lint.yml/badge.svg)](https://github.com/nodc-sweden/ifcb-pytorch-classify/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/nodc-sweden/ifcb-pytorch-classify/graph/badge.svg)](https://codecov.io/gh/nodc-sweden/ifcb-pytorch-classify)
[![Python 3.11–3.14](https://img.shields.io/badge/python-3.11%E2%80%933.14-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21775252.svg)](https://doi.org/10.5281/zenodo.21775252)

Pipeline for training and running inference on IFCB (Imaging FlowCytobot) plankton images using PyTorch.

**Full documentation: <https://nodc-sweden.github.io/ifcb-pytorch-classify/>**

## Capabilities

- **Training**: fine-tune 52 pretrained architectures (ResNet, EfficientNet, ConvNeXt, Vision Transformers, etc.) on class-folder organised image datasets, with optional evaluation plots (static PNG + interactive HTML). Run `ifcb-classify list-models` to see them all
- **Inference**: batch-classify raw IFCB bins (`.roi/.adc/.hdr`) via [ifcbkit](https://github.com/WHOIGit/ifcbkit) into IFCB Dashboard class_scores v3 HDF5 by default, or into `csv`, `mat` and `csv-labels` via `--format`, with per-class decision thresholds; output works with the IFCB Dashboard, [iRfcb](https://europeanifcbgroup.github.io/iRfcb/) and [ClassiPyR](https://europeanifcbgroup.github.io/ClassiPyR/)
- **Chain / cell counting**: optionally train per-taxon [YOLO](https://docs.ultralytics.com/) detectors that count individual cells in colony ROIs and store the count alongside each classification
- **Experiment tracking**: CSV (default), MLflow, or Weights & Biases
- **Built for pipelines**: date-placeholder paths for date-organised continuous inference, and automatic device selection (CUDA, then Apple MPS, then CPU) for both training and inference, overridable per run with `infer --device`

## Quick start

Requires Python 3.11 to 3.14 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
uv pip install -e .
```

Train a classifier on the bundled [example dataset](example_data/), then classify
a directory of raw IFCB bins. The trained checkpoint is named from the run
settings (e.g. `example-resnet50_..._best.pt`), so point `--model` at whatever
`*_best.pt` file lands in `output/`:

```bash
python -m ifcb_classify train --config configs/train_default.yaml \
    --data-dir example_data/plankton --dataset-version example

python -m ifcb_classify infer \
    --input example_data/bins \
    --model output/example-resnet50_dataset_squarepad_augmented_b64_lr0.0001_e20_best.pt \
    --output output/class_scores
```

(`example_data/bins/` is a bundled sample bin, so this runs as-is; point
`--input` at your own bins for real data.)

See [Getting started](docs/getting-started.md) for the full walkthrough, and
[Installation](docs/installation.md) for CUDA and optional extras.

## Documentation

| Topic | Description |
|---|---|
| [Getting started](docs/getting-started.md) | Install, train, and run inference end to end |
| [Installation](docs/installation.md) | CPU, CUDA, and optional extras |
| [Concepts & glossary](docs/concepts.md) | ML and IFCB terms explained for beginners |
| [Training](docs/guides/training.md) | Training options, plots, normalisation |
| [Inference](docs/guides/inference.md) | Classifying raw bins, output format |
| [Chain counting](docs/guides/chain-counting.md) | Per-taxon YOLO cell counting |
| [Chain-counting annotation](docs/guides/chain-counting-annotation.md) | Labelling workflow for chain detectors |
| [Configuration](docs/configuration.md) | Config parameters and date placeholders |
| [Troubleshooting](docs/troubleshooting.md) | Fixes for common install/run errors |

The rendered site (with the auto-generated API reference) is at
<https://nodc-sweden.github.io/ifcb-pytorch-classify/>.

## Testing

```bash
python -m pytest tests/ -v
```

## Citation

If you use this software, please cite it. GitHub's "Cite this repository" button
(top right) generates APA/BibTeX from [`CITATION.cff`](CITATION.cff), or use:

```bibtex
@software{torstensson_ifcb_classify,
  author    = {Torstensson, Anders},
  title     = {ifcb-classify: a PyTorch pipeline for IFCB plankton image classification},
  year      = {2026},
  version   = {0.3.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21775252},
  url       = {https://doi.org/10.5281/zenodo.21775252}
}
```

The DOI above is the concept DOI, which always resolves to the newest release.
Zenodo also mints one DOI per version, listed on the
[record page](https://doi.org/10.5281/zenodo.21775252), if you need to cite the
exact version you ran.

## License

This project is MIT licensed. See [LICENSE](LICENSE).

The optional `chains` extra is different. It installs
[Ultralytics](https://github.com/ultralytics/ultralytics) YOLO, which is
AGPL-3.0, so anything you build on top of the chain-counting features inherits
that licence rather than this one. Ultralytics sells a commercial licence if
AGPL-3.0 does not suit you. Nothing in the default install pulls it in: training,
inference and all output formats work without the extra.
